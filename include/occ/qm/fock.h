#pragma once
#include <occ/core/logger.h>
#include <occ/qm/ints.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/mo.h>

namespace occ::ints {
using occ::qm::SpinorbitalKind;
using occ::qm::MolecularOrbitals;

class FockBuilder {
  public:
    FockBuilder(size_t max_nprim, size_t max_l,
                double precision = std::numeric_limits<double>::epsilon(),
                int derivative_order = 0) {
        occ::timing::start(occ::timing::category::engine_construct);
        m_coulomb_engines.reserve(occ::parallel::nthreads);
        m_coulomb_engines.emplace_back(Operator::coulomb, max_nprim, max_l,
                                       derivative_order);
        m_coulomb_engines[0].set_precision(
            precision); // shellset-dependent precision
        for (size_t i = 1; i != occ::parallel::nthreads; ++i) {
            m_coulomb_engines.push_back(m_coulomb_engines[0]);
        }
        occ::timing::stop(occ::timing::category::engine_construct);
    }

    template <SpinorbitalKind kind, typename Func>
    void two_electron_integral_helper(
        Func &func, const BasisSet &obs, const shellpair_list_t &shellpair_list,
        const shellpair_data_t &shellpair_data, const Mat &D,
        double precision = std::numeric_limits<double>::epsilon(),
        const Mat &Schwarz = Mat()) const {
        occ::timing::start(occ::timing::category::fock);
        const auto n = obs.nbf();
        const auto nshells = obs.size();
        using occ::parallel::nthreads;
        Mat D_shblk_norm;

        if constexpr (kind == SpinorbitalKind::Restricted) {
            D_shblk_norm = compute_shellblock_norm(obs, D);
        }
        if constexpr (kind == SpinorbitalKind::Unrestricted) {
            assert((D.rows() == 2 * n) && (D.cols() == n) &&
                   "Unrestricted density matrix must be 2 nbf x nbf");
            D_shblk_norm = compute_shellblock_norm(obs, occ::qm::block::a(D));
            D_shblk_norm =
                D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, occ::qm::block::b(D)));
        } else if constexpr (kind == SpinorbitalKind::General) {
            assert((D.rows() == 2 * n) && (D.cols() == n) &&
                   "General density matrix must be 2 nbf x 2 nbf");
            D_shblk_norm = compute_shellblock_norm(obs, occ::qm::block::aa(D));
            D_shblk_norm = D_shblk_norm.cwiseMax(
                compute_shellblock_norm(obs, occ::qm::block::ab(D)));
            D_shblk_norm = D_shblk_norm.cwiseMax(
                compute_shellblock_norm(obs, occ::qm::block::ba(D)));
            D_shblk_norm = D_shblk_norm.cwiseMax(
                compute_shellblock_norm(obs, occ::qm::block::bb(D)));
        }
        const auto do_schwarz_screen =
            Schwarz.cols() != 0 && Schwarz.rows() != 0;
        double max_coeff = D_shblk_norm.maxCoeff();

        auto fock_precision = precision;
        // engine precision controls primitive truncation, assume worst-case
        // scenario (all primitive combinations add up constructively)
        auto max_nprim = obs.max_nprim();
        auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
        auto engine_precision =
            std::min(fock_precision / max_coeff,
                     std::numeric_limits<double>::epsilon()) /
            max_nprim4;
        assert(
            engine_precision > max_engine_precision &&
            "using precomputed shell pair data limits the max engine precision"
            " ... make max_engine_precision smaller and recompile");

        // construct the 2-electron repulsion integrals engine pool
        auto &engines = m_coulomb_engines;
        for (int i = 0; i < nthreads; ++i)
            engines[i].set_precision(
                engine_precision); // shellset-dependent precision
        // control will likely break
        // positive definiteness
        // stick with this simple recipe
        std::atomic<size_t> num_ints_computed{0};

        auto shell2bf = obs.shell2bf();

        auto lambda = [&](int thread_id) {
            auto &engine = engines[thread_id];
            const auto &buf = engine.results();
            // loop over permutationally-unique set of shells
            for (size_t s1 = 0, s1234 = 0; s1 != nshells; ++s1) {
                auto bf1_first =
                    shell2bf[s1]; // first basis function in this shell
                const auto &shell1 = obs[s1];
                auto n1 =
                    shell1.size(); // number of basis functions in this shell

                auto sp12_iter = shellpair_data.at(s1).begin();

                for (const auto &s2 : shellpair_list.at(s1)) {
                    auto bf2_first = shell2bf[s2];
                    const auto &shell2 = obs[s2];
                    auto n2 = shell2.size();

                    const auto *sp12 = sp12_iter->get();
                    ++sp12_iter;

                    const auto Dnorm12 =
                        do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

                    for (auto s3 = 0; s3 <= s1; ++s3) {
                        auto bf3_first = shell2bf[s3];
                        const auto &shell3 = obs[s3];
                        auto n3 = shell3.size();

                        const auto Dnorm123 =
                            do_schwarz_screen
                                ? std::max(
                                      D_shblk_norm(s1, s3),
                                      std::max(D_shblk_norm(s2, s3), Dnorm12))
                                : 0.;

                        auto sp34_iter = shellpair_data.at(s3).begin();

                        const auto s4_max = (s1 == s3) ? s2 : s3;
                        for (const auto &s4 : shellpair_list.at(s3)) {
                            if (s4 > s4_max)
                                break; // for each s3, s4 are stored in
                                       // monotonically increasing order

                            // must update the iter even if going to skip s4
                            const auto *sp34 = sp34_iter->get();
                            ++sp34_iter;

                            if ((s1234++) % nthreads != thread_id)
                                continue;

                            const auto Dnorm1234 =
                                do_schwarz_screen
                                    ? std::max(D_shblk_norm(s1, s4),
                                               std::max(D_shblk_norm(s2, s4),
                                                        std::max(D_shblk_norm(
                                                                     s3, s4),
                                                                 Dnorm123)))
                                    : 0.0;

                            if (do_schwarz_screen &&
                                Dnorm1234 * Schwarz(s1, s2) * Schwarz(s3, s4) <
                                    fock_precision)
                                continue;

                            auto bf4_first = shell2bf[s4];
                            const auto &shell4 = obs[s4];
                            auto n4 = shell4.size();

                            num_ints_computed += n1 * n2 * n3 * n4;

                            // compute the permutational degeneracy (i.e. # of
                            // equivalents) of the given shell set
                            auto s12_deg = (s1 == s2) ? 1 : 2;
                            auto s34_deg = (s3 == s4) ? 1 : 2;
                            auto s12_34_deg =
                                (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
                            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                            engine
                                .compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                    shell1, shell2, shell3, shell4, sp12, sp34);
                            const double *buf_1234 = buf[0];

                            if (buf_1234 == nullptr)
                                continue; // if all integrals screened out, skip
                                          // to next quartet

                            func(thread_id, bf1_first, n1, bf2_first, n2,
                                 bf3_first, n3, bf4_first, n4, buf_1234,
                                 s1234_deg);
                        }
                    }
                }
            }
        }; // end of lambda

        occ::timing::start(occ::timing::category::ints2e);
        occ::parallel::parallel_do(lambda);
        occ::timing::stop(occ::timing::category::ints2e);
    }

    template <SpinorbitalKind kind>
    Mat compute_fock(const BasisSet &obs,
                     const shellpair_list_t &shellpair_list,
                     const shellpair_data_t &shellpair_data, const occ::qm::MolecularOrbitals &mo,
                     double precision = std::numeric_limits<double>::epsilon(),
                     const Mat &Schwarz = Mat()) const {
        occ::timing::start(occ::timing::category::fock);
        using occ::parallel::nthreads;
        const auto n = obs.nbf();
        const auto &D = mo.D;
        std::vector<Mat> G(nthreads, Mat::Zero(D.rows(), D.cols()));

        auto lambda = [&](int thread_id, int bf1_first, int n1, int bf2_first,
                          int n2, int bf3_first, int n3, int bf4_first, int n4,
                          const double *buffer, double scale) {
            auto &g = G[thread_id];
            for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (auto f3 = 0; f3 != n3; ++f3) {
                        const auto bf3 = f3 + bf3_first;
                        for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                            const auto bf4 = f4 + bf4_first;
                            const auto value = buffer[f1234] * scale;
                            if constexpr (kind == SpinorbitalKind::Restricted) {
                                g(bf1, bf2) += D(bf3, bf4) * value;
                                g(bf3, bf4) += D(bf1, bf2) * value;
                                g(bf1, bf3) -= 0.25 * D(bf2, bf4) * value;
                                g(bf2, bf4) -= 0.25 * D(bf1, bf3) * value;
                                g(bf1, bf4) -= 0.25 * D(bf2, bf3) * value;
                                g(bf2, bf3) -= 0.25 * D(bf1, bf4) * value;
                            } else if constexpr (kind == SpinorbitalKind::
                                                             Unrestricted) {
                                auto ga = occ::qm::block::a(g);
                                auto gb = occ::qm::block::b(g);
                                const auto Da = occ::qm::block::a(D);
                                const auto Db = occ::qm::block::b(D);
                                ga(bf1, bf2) +=
                                    (Da(bf3, bf4) + Db(bf3, bf4)) * value;
                                ga(bf3, bf4) +=
                                    (Da(bf1, bf2) + Db(bf1, bf2)) * value;
                                gb(bf1, bf2) +=
                                    (Da(bf3, bf4) + Db(bf3, bf4)) * value;
                                gb(bf3, bf4) +=
                                    (Da(bf1, bf2) + Db(bf1, bf2)) * value;

                                ga(bf1, bf3) -= 0.5 * Da(bf2, bf4) * value;
                                ga(bf2, bf4) -= 0.5 * Da(bf1, bf3) * value;
                                ga(bf1, bf4) -= 0.5 * Da(bf2, bf3) * value;
                                ga(bf2, bf3) -= 0.5 * Da(bf1, bf4) * value;

                                gb(bf1, bf3) -= 0.5 * Db(bf2, bf4) * value;
                                gb(bf2, bf4) -= 0.5 * Db(bf1, bf3) * value;
                                gb(bf1, bf4) -= 0.5 * Db(bf2, bf3) * value;
                                gb(bf2, bf3) -= 0.5 * Db(bf1, bf4) * value;
                            } else if constexpr (kind ==
                                                 SpinorbitalKind::General) {
                                g(bf1, bf2) += 2 * D(bf3, bf4) * value;
                                g(bf3, bf4) += 2 * D(bf1, bf2) * value;
                                g(n + bf1, n + bf2) +=
                                    2 * D(n + bf3, n + bf4) * value;
                                g(n + bf3, n + bf4) +=
                                    2 * D(n + bf1, n + bf2) * value;

                                g(bf1, bf3) -= 0.5 * D(bf2, bf4) * value;
                                g(bf2, bf4) -= 0.5 * D(bf1, bf3) * value;
                                g(bf1, bf4) -= 0.5 * D(bf2, bf3) * value;
                                g(bf2, bf3) -= 0.5 * D(bf1, bf4) * value;

                                g(n + bf1, n + bf3) -=
                                    0.5 * D(n + bf2, n + bf4) * value;
                                g(n + bf2, n + bf4) -=
                                    0.5 * D(n + bf1, n + bf3) * value;
                                g(n + bf1, n + bf4) -=
                                    0.5 * D(n + bf2, n + bf3) * value;
                                g(n + bf2, n + bf3) -=
                                    0.5 * D(n + bf1, n + bf4) * value;

                                g(n + bf1, bf3) -=
                                    0.5 * D(n + bf2, bf4) * value;
                                g(n + bf2, bf4) -=
                                    0.5 * D(n + bf1, bf3) * value;
                                g(n + bf1, bf4) -=
                                    0.5 * D(n + bf2, bf3) * value;
                                g(n + bf2, bf3) -=
                                    0.5 * D(n + bf1, bf4) * value;
                                g(bf1, n + bf3) -=
                                    0.5 * D(n + bf2, bf4) * value;
                                g(bf2, n + bf4) -=
                                    0.5 * D(n + bf1, bf3) * value;
                                g(bf1, n + bf4) -=
                                    0.5 * D(n + bf2, bf3) * value;
                                g(bf2, n + bf3) -=
                                    0.5 * D(n + bf1, bf4) * value;

                                g(n + bf1, bf3) -=
                                    0.5 * D(bf2, n + bf4) * value;
                                g(n + bf2, bf4) -=
                                    0.5 * D(bf1, n + bf3) * value;
                                g(n + bf1, bf4) -=
                                    0.5 * D(bf2, n + bf3) * value;
                                g(n + bf2, bf3) -=
                                    0.5 * D(bf1, n + bf4) * value;
                                g(bf1, n + bf3) -=
                                    0.5 * D(bf2, n + bf4) * value;
                                g(bf2, n + bf4) -=
                                    0.5 * D(bf1, n + bf3) * value;
                                g(bf1, n + bf4) -=
                                    0.5 * D(bf2, n + bf3) * value;
                                g(bf2, n + bf3) -=
                                    0.5 * D(bf1, n + bf4) * value;
                            }
                        }
                    }
                }
            }
        }; // end of lambda

        two_electron_integral_helper<kind>(
            lambda, obs, shellpair_list, shellpair_data, D, precision, Schwarz);

        // accumulate contributions from all threads
        for (auto i = 1; i < nthreads; ++i) {
            G[0].noalias() += G[i];
        }
        // symmetrize the result and return
        Mat GG(G[0].rows(), G[0].cols());
        if constexpr (kind == SpinorbitalKind::Restricted ||
                      kind == SpinorbitalKind::General)
            GG = 0.5 * (G[0] + G[0].transpose());
        else if constexpr (kind == SpinorbitalKind::Unrestricted) {
            occ::qm::block::a(GG) = 0.5 * (occ::qm::block::a(G[0]) + occ::qm::block::a(G[0]).transpose());
            occ::qm::block::b(GG) = 0.5 * (occ::qm::block::b(G[0]) + occ::qm::block::b(G[0]).transpose());
        }
        occ::timing::stop(occ::timing::category::fock);
        return GG;
    }

    template <SpinorbitalKind kind>
    std::pair<Mat, Mat>
    compute_JK(const BasisSet &obs, const shellpair_list_t &shellpair_list,
               const shellpair_data_t &shellpair_data, const occ::qm::MolecularOrbitals &mo,
               double precision = std::numeric_limits<double>::epsilon(),
               const Mat &Schwarz = Mat()) const {
        occ::timing::start(occ::timing::category::jkmat);
        const auto n = obs.nbf();
        const auto nshells = obs.size();
        using occ::parallel::nthreads;
        const auto &D = mo.D;
        std::vector<Mat> J(nthreads, Mat::Zero(D.rows(), D.cols()));
        std::vector<Mat> K(nthreads, Mat::Zero(D.rows(), D.cols()));

        auto lambda = [&](int thread_id, int bf1_first, int n1, int bf2_first,
                          int n2, int bf3_first, int n3, int bf4_first, int n4,
                          const double *buffer, double scale) {
            auto &j = J[thread_id];
            auto &k = K[thread_id];
            auto ja = occ::qm::block::a(j);
            auto jb = occ::qm::block::b(j);
            auto ka = occ::qm::block::a(k);
            auto kb = occ::qm::block::b(k);
            const auto Da = occ::qm::block::a(D);
            const auto Db = occ::qm::block::b(D);
            for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (auto f3 = 0; f3 != n3; ++f3) {
                        const auto bf3 = f3 + bf3_first;
                        for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                            const auto bf4 = f4 + bf4_first;
                            const auto value = buffer[f1234] * scale;

                            if constexpr (kind == SpinorbitalKind::Restricted) {
                                j(bf1, bf2) += D(bf3, bf4) * value;
                                j(bf3, bf4) += D(bf1, bf2) * value;
                                k(bf1, bf3) += 0.25 * D(bf2, bf4) * value;
                                k(bf2, bf4) += 0.25 * D(bf1, bf3) * value;
                                k(bf1, bf4) += 0.25 * D(bf2, bf3) * value;
                                k(bf2, bf3) += 0.25 * D(bf1, bf4) * value;
                            } else if constexpr (kind == SpinorbitalKind::
                                                             Unrestricted) {

                                ja(bf1, bf2) +=
                                    (Da(bf3, bf4) + Db(bf3, bf4)) * value;
                                ja(bf3, bf4) +=
                                    (Da(bf1, bf2) + Db(bf1, bf2)) * value;

                                jb(bf1, bf2) +=
                                    (Da(bf3, bf4) + Db(bf3, bf4)) * value;
                                jb(bf3, bf4) +=
                                    (Da(bf1, bf2) + Db(bf1, bf2)) * value;

                                ka(bf1, bf3) += 0.5 * Da(bf2, bf4) * value;
                                ka(bf2, bf4) += 0.5 * Da(bf1, bf3) * value;
                                ka(bf1, bf4) += 0.5 * Da(bf2, bf3) * value;
                                ka(bf2, bf3) += 0.5 * Da(bf1, bf4) * value;

                                kb(bf1, bf3) += 0.5 * Db(bf2, bf4) * value;
                                kb(bf2, bf4) += 0.5 * Db(bf1, bf3) * value;
                                kb(bf1, bf4) += 0.5 * Db(bf2, bf3) * value;
                                kb(bf2, bf3) += 0.5 * Db(bf1, bf4) * value;
                            } else if constexpr (kind ==
                                                 SpinorbitalKind::General) {
                                j(bf1, bf2) += 2 * D(bf3, bf4) * value;
                                j(bf3, bf4) += 2 * D(bf1, bf2) * value;
                                j(n + bf1, n + bf2) +=
                                    2 * D(n + bf3, n + bf4) * value;
                                j(n + bf3, n + bf4) +=
                                    2 * D(n + bf1, n + bf2) * value;

                                k(bf1, bf3) += 0.5 * D(bf2, bf4) * value;
                                k(bf2, bf4) += 0.5 * D(bf1, bf3) * value;
                                k(bf1, bf4) += 0.5 * D(bf2, bf3) * value;
                                k(bf2, bf3) += 0.5 * D(bf1, bf4) * value;

                                k(n + bf1, n + bf3) +=
                                    0.5 * D(n + bf2, n + bf4) * value;
                                k(n + bf2, n + bf4) +=
                                    0.5 * D(n + bf1, n + bf3) * value;
                                k(n + bf1, n + bf4) +=
                                    0.5 * D(n + bf2, n + bf3) * value;
                                k(n + bf2, n + bf3) +=
                                    0.5 * D(n + bf1, n + bf4) * value;

                                k(n + bf1, bf3) +=
                                    0.5 * D(n + bf2, bf4) * value;
                                k(n + bf2, bf4) +=
                                    0.5 * D(n + bf1, bf3) * value;
                                k(n + bf1, bf4) +=
                                    0.5 * D(n + bf2, bf3) * value;
                                k(n + bf2, bf3) +=
                                    0.5 * D(n + bf1, bf4) * value;
                                k(bf1, n + bf3) +=
                                    0.5 * D(n + bf2, bf4) * value;
                                k(bf2, n + bf4) +=
                                    0.5 * D(n + bf1, bf3) * value;
                                k(bf1, n + bf4) +=
                                    0.5 * D(n + bf2, bf3) * value;
                                k(bf2, n + bf3) +=
                                    0.5 * D(n + bf1, bf4) * value;

                                k(n + bf1, bf3) +=
                                    0.5 * D(bf2, n + bf4) * value;
                                k(n + bf2, bf4) +=
                                    0.5 * D(bf1, n + bf3) * value;
                                k(n + bf1, bf4) +=
                                    0.5 * D(bf2, n + bf3) * value;
                                k(n + bf2, bf3) +=
                                    0.5 * D(bf1, n + bf4) * value;
                                k(bf1, n + bf3) +=
                                    0.5 * D(bf2, n + bf4) * value;
                                k(bf2, n + bf4) +=
                                    0.5 * D(bf1, n + bf3) * value;
                                k(bf1, n + bf4) +=
                                    0.5 * D(bf2, n + bf3) * value;
                                k(bf2, n + bf3) +=
                                    0.5 * D(bf1, n + bf4) * value;
                            }
                        }
                    }
                }
            }
        }; // end of lambda
        two_electron_integral_helper<kind>(
            lambda, obs, shellpair_list, shellpair_data, D, precision, Schwarz);

        // accumulate contributions from all threads
        for (auto i = 1; i < nthreads; ++i) {
            J[0] += J[i];
            K[0] += K[i];
        }
        // symmetrize the result and return
        Mat JJ(J[0].rows(), J[0].cols()), KK(K[0].rows(), K[0].cols());
        if constexpr (kind == SpinorbitalKind::Restricted ||
                      kind == SpinorbitalKind::General) {
            JJ = 0.5 * (J[0] + J[0].transpose());
            KK = 0.5 * (K[0] + K[0].transpose());
        } else if constexpr (kind == SpinorbitalKind::Unrestricted) {
            occ::qm::block::a(JJ) = 0.5 * (occ::qm::block::a(J[0]) + occ::qm::block::a(J[0]).transpose());
            occ::qm::block::b(JJ) = 0.5 * (occ::qm::block::b(J[0]) + occ::qm::block::b(J[0]).transpose());
            occ::qm::block::a(KK) = 0.5 * (occ::qm::block::a(K[0]) + occ::qm::block::a(K[0]).transpose());
            occ::qm::block::b(KK) = 0.5 * (occ::qm::block::b(K[0]) + occ::qm::block::b(K[0]).transpose());
        }
        occ::timing::stop(occ::timing::category::jkmat);
        return {JJ, KK};
    }

    template <SpinorbitalKind kind>
    Mat compute_J(const BasisSet &obs, const shellpair_list_t &shellpair_list,
                  const shellpair_data_t &shellpair_data, const MolecularOrbitals &mo,
                  double precision = std::numeric_limits<double>::epsilon(),
                  const Mat &Schwarz = Mat()) const {
        occ::timing::start(occ::timing::category::jmat);
        const auto n = obs.nbf();
        using occ::parallel::nthreads;
        const auto &D = mo.D;
        std::vector<Mat> J(nthreads, Mat::Zero(D.rows(), D.cols()));

        auto shell2bf = obs.shell2bf();

        auto lambda = [&](int thread_id, int bf1_first, int n1, int bf2_first,
                          int n2, int bf3_first, int n3, int bf4_first, int n4,
                          const double *buffer, double scale) {
            auto &j = J[thread_id];
            auto ja = occ::qm::block::a(j);
            auto jb = occ::qm::block::b(j);
            const auto Da = occ::qm::block::a(D);
            const auto Db = occ::qm::block::b(D);
            for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (auto f3 = 0; f3 != n3; ++f3) {
                        const auto bf3 = f3 + bf3_first;
                        for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                            const auto bf4 = f4 + bf4_first;
                            const auto value = buffer[f1234] * scale;

                            if constexpr (kind == SpinorbitalKind::Restricted) {
                                j(bf1, bf2) += D(bf3, bf4) * value;
                                j(bf3, bf4) += D(bf1, bf2) * value;
                            } else if constexpr (kind == SpinorbitalKind::
                                                             Unrestricted) {

                                ja(bf1, bf2) +=
                                    (Da(bf3, bf4) + Db(bf3, bf4)) * value;
                                ja(bf3, bf4) +=
                                    (Da(bf1, bf2) + Db(bf1, bf2)) * value;

                                jb(bf1, bf2) +=
                                    (Da(bf3, bf4) + Db(bf3, bf4)) * value;
                                jb(bf3, bf4) +=
                                    (Da(bf1, bf2) + Db(bf1, bf2)) * value;
                            } else if constexpr (kind ==
                                                 SpinorbitalKind::General) {
                                j(bf1, bf2) += 2 * D(bf3, bf4) * value;
                                j(bf3, bf4) += 2 * D(bf1, bf2) * value;

                                j(n + bf1, n + bf2) +=
                                    2 * D(n + bf3, n + bf4) * value;
                                j(n + bf3, n + bf4) +=
                                    2 * D(n + bf1, n + bf2) * value;
                            }
                        }
                    }
                }
            }
        };

        two_electron_integral_helper<kind>(
            lambda, obs, shellpair_list, shellpair_data, D, precision, Schwarz);

        // accumulate contributions from all threads
        for (auto i = 1; i < nthreads; ++i) {
            J[0] += J[i];
        }
        // symmetrize the result and return
        Mat JJ(J[0].rows(), J[0].cols());
        if constexpr (kind == SpinorbitalKind::Restricted ||
                      kind == SpinorbitalKind::General) {
            JJ = 0.5 * (J[0] + J[0].transpose());
        } else if constexpr (kind == SpinorbitalKind::Unrestricted) {
            occ::qm::block::a(JJ) = 0.5 * (occ::qm::block::a(J[0]) + occ::qm::block::a(J[0]).transpose());
            occ::qm::block::b(JJ) = 0.5 * (occ::qm::block::b(J[0]) + occ::qm::block::b(J[0]).transpose());
        }
        occ::timing::stop(occ::timing::category::jmat);
        return JJ;
    }

  private:
    mutable std::vector<libint2::Engine> m_coulomb_engines;
    std::vector<Mat> m_J, m_K, m_fock;
};

} // namespace occ::ints
