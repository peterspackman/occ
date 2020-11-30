#pragma once
#include <tonto/core/logger.h>
#include <tonto/qm/ints.h>
#include <tonto/qm/spinorbital.h>

namespace tonto::ints {
using tonto::MatRM;
using tonto::qm::SpinorbitalKind;

template<SpinorbitalKind kind>
MatRM compute_fock(
    const BasisSet &obs, const shellpair_list_t &shellpair_list,
    const shellpair_data_t &shellpair_data, const MatRM &D,
    double precision = std::numeric_limits<double>::epsilon(),
    const MatRM &Schwarz = MatRM())
{
    tonto::timing::start(tonto::timing::category::ints2e);
    tonto::timing::start(tonto::timing::category::fock);
    const auto n = obs.nbf();
    const auto nshells = obs.size();
    using tonto::parallel::nthreads;
    std::vector<MatRM> G(nthreads, MatRM::Zero(D.rows(), D.cols()));
    MatRM D_shblk_norm;

    if constexpr(kind == SpinorbitalKind::Restricted) {
        D_shblk_norm = compute_shellblock_norm(obs, D);
    }
    if constexpr(kind == SpinorbitalKind::Unrestricted) {
        assert((D.rows() == 2 * n) && (D.cols() ==n) && "Unrestricted density matrix must be 2 nbf x nbf");
        D_shblk_norm = compute_shellblock_norm(obs, D.alpha());
        size_t beta_rows = D.beta().rows();
        size_t beta_cols = D.beta().cols();
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.beta()));
    }
    else if constexpr(kind == SpinorbitalKind::General) {
        assert((D.rows() == 2 * n) && (D.cols() ==n) && "General density matrix must be 2 nbf x 2 nbf");
        D_shblk_norm = compute_shellblock_norm(obs, D.alpha_alpha());
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.alpha_beta()));
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.beta_alpha()));
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.beta_beta()));
    }
    const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
    double max_coeff = D_shblk_norm.maxCoeff();

    auto fock_precision = precision;
    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)
    auto max_nprim = obs.max_nprim();
    auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
    auto engine_precision = std::min(
        fock_precision / max_coeff,
        std::numeric_limits<double>::epsilon()) / max_nprim4;
    assert(engine_precision > max_engine_precision &&
           "using precomputed shell pair data limits the max engine precision"
           " ... make max_engine_precision smaller and recompile");

    // construct the 2-electron repulsion integrals engine pool
    using libint2::Engine;
    std::vector<Engine> engines(nthreads);
    engines[0] = Engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
    engines[0].set_precision(engine_precision); // shellset-dependent precision
    // control will likely break
    // positive definiteness
    // stick with this simple recipe
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }
    std::atomic<size_t> num_ints_computed{0};

    auto shell2bf = obs.shell2bf();

    auto lambda = [&](int thread_id) {
        auto &engine = engines[thread_id];
        auto &g = G[thread_id];
        const auto &buf = engine.results();
        // loop over permutationally-unique set of shells
        for (size_t s1 = 0, s1234 = 0; s1 != nshells; ++s1) {
            auto bf1_first = shell2bf[s1]; // first basis function in this shell
            auto n1 = obs[s1].size();      // number of basis functions in this shell

            auto sp12_iter = shellpair_data.at(s1).begin();

            for (const auto &s2 : shellpair_list.at(s1)) {
                auto bf2_first = shell2bf[s2];
                auto n2 = obs[s2].size();

                const auto *sp12 = sp12_iter->get();
                ++sp12_iter;

                const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

                for (auto s3 = 0; s3 <= s1; ++s3) {
                    auto bf3_first = shell2bf[s3];
                    auto n3 = obs[s3].size();

                    const auto Dnorm123 =
                            do_schwarz_screen
                            ? std::max(D_shblk_norm(s1, s3),
                                       std::max(D_shblk_norm(s2, s3), Dnorm12))
                            : 0.;

                    auto sp34_iter = shellpair_data.at(s3).begin();

                    const auto s4_max = (s1 == s3) ? s2 : s3;
                    for (const auto &s4 : shellpair_list.at(s3)) {
                        if (s4 > s4_max)
                            break; // for each s3, s4 are stored in monotonically increasing order

                        // must update the iter even if going to skip s4
                        const auto *sp34 = sp34_iter->get();
                        ++sp34_iter;

                        if ((s1234++) % nthreads != thread_id)
                            continue;

                        const auto Dnorm1234 =
                                do_schwarz_screen
                                ? std::max(D_shblk_norm(s1, s4),
                                    std::max(D_shblk_norm(s2, s4),
                                      std::max(D_shblk_norm(s3, s4), Dnorm123)))
                                : 0.0;

                        if (do_schwarz_screen && Dnorm1234 * Schwarz(s1, s2) * Schwarz(s3, s4) < fock_precision)
                            continue;

                        auto bf4_first = shell2bf[s4];
                        auto n4 = obs[s4].size();

                        num_ints_computed += n1 * n2 * n3 * n4;

                        // compute the permutational degeneracy (i.e. # of equivalents) of
                        // the given shell set
                        auto s12_deg = (s1 == s2) ? 1 : 2;
                        auto s34_deg = (s3 == s4) ? 1 : 2;
                        auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
                        auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                        engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                    obs[s1], obs[s2], obs[s3], obs[s4], sp12, sp34);
                        const auto *buf_1234 = buf[0];

                        if (buf_1234 == nullptr) continue; // if all integrals screened out, skip to next quartet

                        for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                            const auto bf1 = f1 + bf1_first;
                            for (auto f2 = 0; f2 != n2; ++f2) {
                                const auto bf2 = f2 + bf2_first;
                                for (auto f3 = 0; f3 != n3; ++f3) {
                                    const auto bf3 = f3 + bf3_first;
                                    for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                                        const auto bf4 = f4 + bf4_first;

                                        const auto value = buf_1234[f1234];

                                        const auto value_scal_by_deg = value * s1234_deg;
                                        if constexpr(kind == SpinorbitalKind::Restricted) {
                                            g(bf1, bf2) += D(bf3, bf4) * value_scal_by_deg;
                                            g(bf3, bf4) += D(bf1, bf2) * value_scal_by_deg;
                                            g(bf1, bf3) -= 0.25 * D(bf2, bf4) * value_scal_by_deg;
                                            g(bf2, bf4) -= 0.25 * D(bf1, bf3) * value_scal_by_deg;
                                            g(bf1, bf4) -= 0.25 * D(bf2, bf3) * value_scal_by_deg;
                                            g(bf2, bf3) -= 0.25 * D(bf1, bf4) * value_scal_by_deg;
                                        }
                                        else if constexpr(kind == SpinorbitalKind::Unrestricted) {
                                            auto ga = g.alpha();
                                            auto gb = g.beta();
                                            const auto Da = D.alpha();
                                            const auto Db = D.beta();
                                            ga(bf1, bf2) +=
                                                (Da(bf3, bf4) + Db(bf3, bf4)) * value_scal_by_deg;
                                            ga(bf3, bf4) +=
                                                (Da(bf1, bf2) + Db(bf1, bf2)) * value_scal_by_deg;

                                            gb(bf1, bf2) +=
                                                (Da(bf3, bf4) + Db(bf3, bf4)) * value_scal_by_deg;
                                            gb(bf3, bf4) +=
                                                (Da(bf1, bf2) + Db(bf1, bf2)) * value_scal_by_deg;

                                            ga(bf1, bf3) -= 0.5 * Da(bf2, bf4) * value_scal_by_deg;
                                            ga(bf2, bf4) -= 0.5 * Da(bf1, bf3) * value_scal_by_deg;
                                            ga(bf1, bf4) -= 0.5 * Da(bf2, bf3) * value_scal_by_deg;
                                            ga(bf2, bf3) -= 0.5 * Da(bf1, bf4) * value_scal_by_deg;

                                            gb(bf1, bf3) -= 0.5 * Db(bf2, bf4) * value_scal_by_deg;
                                            gb(bf2, bf4) -= 0.5 * Db(bf1, bf3) * value_scal_by_deg;
                                            gb(bf1, bf4) -= 0.5 * Db(bf2, bf3) * value_scal_by_deg;
                                            gb(bf2, bf3) -= 0.5 * Db(bf1, bf4) * value_scal_by_deg;
                                        }
                                        else if constexpr(kind == SpinorbitalKind::General) {
                                            g(bf1, bf2) += 2 * D(bf3, bf4) * value_scal_by_deg;
                                            g(bf3, bf4) += 2 * D(bf1, bf2) * value_scal_by_deg;
                                            g(n + bf1, n + bf2) += 2 * D(n + bf3, n + bf4) * value_scal_by_deg;
                                            g(n + bf3, n + bf4) += 2 * D(n + bf1, n + bf2) * value_scal_by_deg;

                                            g(bf1, bf3) -= 0.5 * D(bf2, bf4) * value_scal_by_deg;
                                            g(bf2, bf4) -= 0.5 * D(bf1, bf3) * value_scal_by_deg;
                                            g(bf1, bf4) -= 0.5 * D(bf2, bf3) * value_scal_by_deg;
                                            g(bf2, bf3) -= 0.5 * D(bf1, bf4) * value_scal_by_deg;

                                            g(n + bf1, n + bf3) -= 0.5 * D(n + bf2, n + bf4) * value_scal_by_deg;
                                            g(n + bf2, n + bf4) -= 0.5 * D(n + bf1, n + bf3) * value_scal_by_deg;
                                            g(n + bf1, n + bf4) -= 0.5 * D(n + bf2, n + bf3) * value_scal_by_deg;
                                            g(n + bf2, n + bf3) -= 0.5 * D(n + bf1, n + bf4) * value_scal_by_deg;

                                            g(n + bf1, bf3) -= 0.5 * D(n + bf2, bf4) * value_scal_by_deg;
                                            g(n + bf2, bf4) -= 0.5 * D(n + bf1, bf3) * value_scal_by_deg;
                                            g(n + bf1, bf4) -= 0.5 * D(n + bf2, bf3) * value_scal_by_deg;
                                            g(n + bf2, bf3) -= 0.5 * D(n + bf1, bf4) * value_scal_by_deg;
                                            g(bf1, n + bf3) -= 0.5 * D(n + bf2, bf4) * value_scal_by_deg;
                                            g(bf2, n + bf4) -= 0.5 * D(n + bf1, bf3) * value_scal_by_deg;
                                            g(bf1, n + bf4) -= 0.5 * D(n + bf2, bf3) * value_scal_by_deg;
                                            g(bf2, n + bf3) -= 0.5 * D(n + bf1, bf4) * value_scal_by_deg;

                                            g(n + bf1, bf3) -= 0.5 * D(bf2, n + bf4) * value_scal_by_deg;
                                            g(n + bf2, bf4) -= 0.5 * D(bf1, n + bf3) * value_scal_by_deg;
                                            g(n + bf1, bf4) -= 0.5 * D(bf2, n + bf3) * value_scal_by_deg;
                                            g(n + bf2, bf3) -= 0.5 * D(bf1, n + bf4) * value_scal_by_deg;
                                            g(bf1, n + bf3) -= 0.5 * D(bf2, n + bf4) * value_scal_by_deg;
                                            g(bf2, n + bf4) -= 0.5 * D(bf1, n + bf3) * value_scal_by_deg;
                                            g(bf1, n + bf4) -= 0.5 * D(bf2, n + bf3) * value_scal_by_deg;
                                            g(bf2, n + bf3) -= 0.5 * D(bf1, n + bf4) * value_scal_by_deg;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }; // end of lambda

    tonto::parallel::parallel_do(lambda);

    // accumulate contributions from all threads
    for (auto i = 1; i < nthreads; ++i) {
        G[0] += G[i];
    }
    // symmetrize the result and return
    MatRM GG(G[0].rows(), G[0].cols());
    if constexpr(kind == SpinorbitalKind::Restricted || kind == SpinorbitalKind::General)
        GG = 0.5 * (G[0] + G[0].transpose());
    else if constexpr(kind == SpinorbitalKind::Unrestricted) {
        GG.alpha() = 0.5 * (G[0].alpha() + G[0].alpha().transpose());
        GG.beta() = 0.5 * (G[0].beta() + G[0].beta().transpose());
    }
    tonto::timing::stop(tonto::timing::category::ints2e);
    tonto::timing::stop(tonto::timing::category::fock);
    return GG;
}


template<SpinorbitalKind kind>
std::pair<MatRM, MatRM> compute_JK(
    const BasisSet &obs, const shellpair_list_t &shellpair_list,
    const shellpair_data_t &shellpair_data, const MatRM &D,
    double precision = std::numeric_limits<double>::epsilon(),
    const MatRM &Schwarz = MatRM())
{
    tonto::timing::start(tonto::timing::category::ints2e);
    const auto n = obs.nbf();
    const auto nshells = obs.size();
    using tonto::parallel::nthreads;
    std::vector<MatRM> J(nthreads, MatRM::Zero(D.rows(), D.cols()));
    std::vector<MatRM> K(nthreads, MatRM::Zero(D.rows(), D.cols()));
    MatRM D_shblk_norm = compute_shellblock_norm(obs, D.block(0, 0, n, n));

    if constexpr(kind == SpinorbitalKind::Unrestricted) {
        assert((D.rows() == 2 * n) && (D.cols() ==n) && "Unrestricted density matrix must be 2 nbf x nbf");
        size_t beta_rows = D.beta().rows();
        size_t beta_cols = D.beta().cols();
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.beta()));
    }
    else if constexpr(kind == SpinorbitalKind::General) {
        assert((D.rows() == 2 * n) && (D.cols() ==n) && "General density matrix must be 2 nbf x 2 nbf");
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.alpha_beta()));
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.beta_alpha()));
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.beta_beta()));
    }
    const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
    double max_coeff = D_shblk_norm.maxCoeff();

    auto fock_precision = precision;
    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)
    auto max_nprim = obs.max_nprim();
    auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
    auto engine_precision = std::min(
        fock_precision / max_coeff,
        std::numeric_limits<double>::epsilon()) / max_nprim4;
    assert(engine_precision > max_engine_precision &&
           "using precomputed shell pair data limits the max engine precision"
           " ... make max_engine_precision smaller and recompile");

    // construct the 2-electron repulsion integrals engine pool
    using libint2::Engine;
    std::vector<Engine> engines(nthreads);
    engines[0] = Engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
    engines[0].set_precision(engine_precision); // shellset-dependent precision
    // control will likely break
    // positive definiteness
    // stick with this simple recipe
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }
    std::atomic<size_t> num_ints_computed{0};

    auto shell2bf = obs.shell2bf();

    auto lambda = [&](int thread_id) {
        auto &engine = engines[thread_id];
        auto &j = J[thread_id];
        auto &k = K[thread_id];
        auto ja = j.alpha();
        auto jb = j.beta();
        auto ka = k.alpha();
        auto kb = k.beta();
        const auto Da = D.alpha();
        const auto Db = D.beta();
        const auto &buf = engine.results();
        // loop over permutationally-unique set of shells
        for (size_t s1 = 0, s1234 = 0; s1 != nshells; ++s1) {
            auto bf1_first = shell2bf[s1]; // first basis function in this shell
            auto n1 = obs[s1].size();      // number of basis functions in this shell

            auto sp12_iter = shellpair_data.at(s1).begin();

            for (const auto &s2 : shellpair_list.at(s1)) {
                auto bf2_first = shell2bf[s2];
                auto n2 = obs[s2].size();

                const auto *sp12 = sp12_iter->get();
                ++sp12_iter;

                const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

                for (auto s3 = 0; s3 <= s1; ++s3) {
                    auto bf3_first = shell2bf[s3];
                    auto n3 = obs[s3].size();

                    const auto Dnorm123 =
                            do_schwarz_screen
                            ? std::max(D_shblk_norm(s1, s3),
                                       std::max(D_shblk_norm(s2, s3), Dnorm12))
                            : 0.;

                    auto sp34_iter = shellpair_data.at(s3).begin();

                    const auto s4_max = (s1 == s3) ? s2 : s3;
                    for (const auto &s4 : shellpair_list.at(s3)) {
                        if (s4 > s4_max)
                            break; // for each s3, s4 are stored in monotonically increasing order

                        // must update the iter even if going to skip s4
                        const auto *sp34 = sp34_iter->get();
                        ++sp34_iter;

                        if ((s1234++) % nthreads != thread_id)
                            continue;

                        const auto Dnorm1234 =
                                do_schwarz_screen
                                ? std::max(D_shblk_norm(s1, s4),
                                    std::max(D_shblk_norm(s2, s4),
                                      std::max(D_shblk_norm(s3, s4), Dnorm123)))
                                : 0.0;

                        if (do_schwarz_screen && Dnorm1234 * Schwarz(s1, s2) * Schwarz(s3, s4) < fock_precision)
                            continue;

                        auto bf4_first = shell2bf[s4];
                        auto n4 = obs[s4].size();

                        num_ints_computed += n1 * n2 * n3 * n4;

                        // compute the permutational degeneracy (i.e. # of equivalents) of
                        // the given shell set
                        auto s12_deg = (s1 == s2) ? 1 : 2;
                        auto s34_deg = (s3 == s4) ? 1 : 2;
                        auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
                        auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                        engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                    obs[s1], obs[s2], obs[s3], obs[s4], sp12, sp34);
                        const auto *buf_1234 = buf[0];

                        if (buf_1234 == nullptr) continue; // if all integrals screened out, skip to next quartet

                        for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                            const auto bf1 = f1 + bf1_first;
                            for (auto f2 = 0; f2 != n2; ++f2) {
                                const auto bf2 = f2 + bf2_first;
                                for (auto f3 = 0; f3 != n3; ++f3) {
                                    const auto bf3 = f3 + bf3_first;
                                    for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                                        const auto bf4 = f4 + bf4_first;

                                        const auto value = buf_1234[f1234];

                                        const auto value_scal_by_deg = value * s1234_deg;
                                        if constexpr(kind == SpinorbitalKind::Restricted) {
                                            j(bf1, bf2) += D(bf3, bf4) * value_scal_by_deg;
                                            j(bf3, bf4) += D(bf1, bf2) * value_scal_by_deg;
                                            k(bf1, bf3) += 0.25 * D(bf2, bf4) * value_scal_by_deg;
                                            k(bf2, bf4) += 0.25 * D(bf1, bf3) * value_scal_by_deg;
                                            k(bf1, bf4) += 0.25 * D(bf2, bf3) * value_scal_by_deg;
                                            k(bf2, bf3) += 0.25 * D(bf1, bf4) * value_scal_by_deg;
                                        }
                                        else if constexpr(kind == SpinorbitalKind::Unrestricted) {

                                            ja(bf1, bf2) +=
                                                (Da(bf3, bf4) + Db(bf3, bf4)) * value_scal_by_deg;
                                            ja(bf3, bf4) +=
                                                (Da(bf1, bf2) + Db(bf1, bf2)) * value_scal_by_deg;

                                            jb(bf1, bf2) +=
                                                (Da(bf3, bf4) + Db(bf3, bf4)) * value_scal_by_deg;
                                            jb(bf3, bf4) +=
                                                (Da(bf1, bf2) + Db(bf1, bf2)) * value_scal_by_deg;

                                            ka(bf1, bf3) += 0.5 * Da(bf2, bf4) * value_scal_by_deg;
                                            ka(bf2, bf4) += 0.5 * Da(bf1, bf3) * value_scal_by_deg;
                                            ka(bf1, bf4) += 0.5 * Da(bf2, bf3) * value_scal_by_deg;
                                            ka(bf2, bf3) += 0.5 * Da(bf1, bf4) * value_scal_by_deg;

                                            kb(bf1, bf3) += 0.5 * Db(bf2, bf4) * value_scal_by_deg;
                                            kb(bf2, bf4) += 0.5 * Db(bf1, bf3) * value_scal_by_deg;
                                            kb(bf1, bf4) += 0.5 * Db(bf2, bf3) * value_scal_by_deg;
                                            kb(bf2, bf3) += 0.5 * Db(bf1, bf4) * value_scal_by_deg;
                                        }
                                        else if constexpr(kind == SpinorbitalKind::General) {
                                            j(bf1, bf2) += 2 * D(bf3, bf4) * value_scal_by_deg;
                                            j(bf3, bf4) += 2 * D(bf1, bf2) * value_scal_by_deg;
                                            j(n + bf1, n + bf2) += 2 * D(n + bf3, n + bf4) * value_scal_by_deg;
                                            j(n + bf3, n + bf4) += 2 * D(n + bf1, n + bf2) * value_scal_by_deg;

                                            k(bf1, bf3) += 0.5 * D(bf2, bf4) * value_scal_by_deg;
                                            k(bf2, bf4) += 0.5 * D(bf1, bf3) * value_scal_by_deg;
                                            k(bf1, bf4) += 0.5 * D(bf2, bf3) * value_scal_by_deg;
                                            k(bf2, bf3) += 0.5 * D(bf1, bf4) * value_scal_by_deg;

                                            k(n + bf1, n + bf3) += 0.5 * D(n + bf2, n + bf4) * value_scal_by_deg;
                                            k(n + bf2, n + bf4) += 0.5 * D(n + bf1, n + bf3) * value_scal_by_deg;
                                            k(n + bf1, n + bf4) += 0.5 * D(n + bf2, n + bf3) * value_scal_by_deg;
                                            k(n + bf2, n + bf3) += 0.5 * D(n + bf1, n + bf4) * value_scal_by_deg;

                                            k(n + bf1, bf3) += 0.5 * D(n + bf2, bf4) * value_scal_by_deg;
                                            k(n + bf2, bf4) += 0.5 * D(n + bf1, bf3) * value_scal_by_deg;
                                            k(n + bf1, bf4) += 0.5 * D(n + bf2, bf3) * value_scal_by_deg;
                                            k(n + bf2, bf3) += 0.5 * D(n + bf1, bf4) * value_scal_by_deg;
                                            k(bf1, n + bf3) += 0.5 * D(n + bf2, bf4) * value_scal_by_deg;
                                            k(bf2, n + bf4) += 0.5 * D(n + bf1, bf3) * value_scal_by_deg;
                                            k(bf1, n + bf4) += 0.5 * D(n + bf2, bf3) * value_scal_by_deg;
                                            k(bf2, n + bf3) += 0.5 * D(n + bf1, bf4) * value_scal_by_deg;

                                            k(n + bf1, bf3) += 0.5 * D(bf2, n + bf4) * value_scal_by_deg;
                                            k(n + bf2, bf4) += 0.5 * D(bf1, n + bf3) * value_scal_by_deg;
                                            k(n + bf1, bf4) += 0.5 * D(bf2, n + bf3) * value_scal_by_deg;
                                            k(n + bf2, bf3) += 0.5 * D(bf1, n + bf4) * value_scal_by_deg;
                                            k(bf1, n + bf3) += 0.5 * D(bf2, n + bf4) * value_scal_by_deg;
                                            k(bf2, n + bf4) += 0.5 * D(bf1, n + bf3) * value_scal_by_deg;
                                            k(bf1, n + bf4) += 0.5 * D(bf2, n + bf3) * value_scal_by_deg;
                                            k(bf2, n + bf3) += 0.5 * D(bf1, n + bf4) * value_scal_by_deg;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }; // end of lambda

    tonto::parallel::parallel_do(lambda);

    // accumulate contributions from all threads
    for (auto i = 1; i < nthreads; ++i) {
        J[0] += J[i];
        K[0] += K[i];
    }
    // symmetrize the result and return
    MatRM JJ(J[0].rows(), J[0].cols()), KK(K[0].rows(), K[0].cols());
    if constexpr(kind == SpinorbitalKind::Restricted || kind == SpinorbitalKind::General) {
        JJ = 0.5 * (J[0] + J[0].transpose());
        KK = 0.5 * (K[0] + K[0].transpose());
    }
    else if constexpr(kind == SpinorbitalKind::Unrestricted) {
        JJ.alpha() = 0.5 * (J[0].alpha() + J[0].alpha().transpose());
        JJ.beta() = 0.5 * (J[0].beta() + J[0].beta().transpose());
        KK.alpha() = 0.5 * (K[0].alpha() + K[0].alpha().transpose());
        KK.beta() = 0.5 * (K[0].beta() + K[0].beta().transpose());
    }
    tonto::timing::stop(tonto::timing::category::ints2e);
    return {JJ, KK};
}

template<SpinorbitalKind kind>
MatRM compute_J(
    const BasisSet &obs, const shellpair_list_t &shellpair_list,
    const shellpair_data_t &shellpair_data, const MatRM &D,
    double precision = std::numeric_limits<double>::epsilon(),
    const MatRM &Schwarz = MatRM())
{
    tonto::timing::start(tonto::timing::category::ints2e);
    const auto n = obs.nbf();
    const auto nshells = obs.size();
    using tonto::parallel::nthreads;
    std::vector<MatRM> J(nthreads, MatRM::Zero(D.rows(), D.cols()));
    MatRM D_shblk_norm = compute_shellblock_norm(obs, D.block(0, 0, n, n));

    if constexpr(kind == SpinorbitalKind::Unrestricted) {
        assert((D.rows() == 2 * n) && (D.cols() ==n) && "Unrestricted density matrix must be 2 nbf x nbf");
        size_t beta_rows = D.beta().rows();
        size_t beta_cols = D.beta().cols();
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.beta()));
    }
    else if constexpr(kind == SpinorbitalKind::General) {
        assert((D.rows() == 2 * n) && (D.cols() ==n) && "General density matrix must be 2 nbf x 2 nbf");
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.alpha_beta()));
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.beta_alpha()));
        D_shblk_norm = D_shblk_norm.cwiseMax(compute_shellblock_norm(obs, D.beta_beta()));
    }
    const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
    double max_coeff = D_shblk_norm.maxCoeff();

    auto fock_precision = precision;
    // engine precision controls primitive truncation, assume worst-case scenario
    // (all primitive combinations add up constructively)
    auto max_nprim = obs.max_nprim();
    auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
    auto engine_precision = std::min(
        fock_precision / max_coeff,
        std::numeric_limits<double>::epsilon()) / max_nprim4;
    assert(engine_precision > max_engine_precision &&
           "using precomputed shell pair data limits the max engine precision"
           " ... make max_engine_precision smaller and recompile");

    // construct the 2-electron repulsion integrals engine pool
    using libint2::Engine;
    std::vector<Engine> engines(nthreads);
    engines[0] = Engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
    engines[0].set_precision(engine_precision); // shellset-dependent precision
    // control will likely break
    // positive definiteness
    // stick with this simple recipe
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }
    std::atomic<size_t> num_ints_computed{0};

    auto shell2bf = obs.shell2bf();

    auto lambda = [&](int thread_id) {
        auto &engine = engines[thread_id];
        auto &j = J[thread_id];
        auto ja = j.alpha();
        auto jb = j.beta();
        const auto Da = D.alpha();
        const auto Db = D.beta();
        const auto &buf = engine.results();
        // loop over permutationally-unique set of shells
        for (size_t s1 = 0, s1234 = 0; s1 != nshells; ++s1) {
            auto bf1_first = shell2bf[s1]; // first basis function in this shell
            auto n1 = obs[s1].size();      // number of basis functions in this shell

            auto sp12_iter = shellpair_data.at(s1).begin();

            for (const auto &s2 : shellpair_list.at(s1)) {
                auto bf2_first = shell2bf[s2];
                auto n2 = obs[s2].size();

                const auto *sp12 = sp12_iter->get();
                ++sp12_iter;

                const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

                for (auto s3 = 0; s3 <= s1; ++s3) {
                    auto bf3_first = shell2bf[s3];
                    auto n3 = obs[s3].size();

                    const auto Dnorm123 =
                            do_schwarz_screen
                            ? std::max(D_shblk_norm(s1, s3),
                                       std::max(D_shblk_norm(s2, s3), Dnorm12))
                            : 0.;

                    auto sp34_iter = shellpair_data.at(s3).begin();

                    const auto s4_max = (s1 == s3) ? s2 : s3;
                    for (const auto &s4 : shellpair_list.at(s3)) {
                        if (s4 > s4_max)
                            break; // for each s3, s4 are stored in monotonically increasing order

                        // must update the iter even if going to skip s4
                        const auto *sp34 = sp34_iter->get();
                        ++sp34_iter;

                        if ((s1234++) % nthreads != thread_id)
                            continue;

                        const auto Dnorm1234 =
                                do_schwarz_screen
                                ? std::max(D_shblk_norm(s1, s4),
                                    std::max(D_shblk_norm(s2, s4),
                                      std::max(D_shblk_norm(s3, s4), Dnorm123)))
                                : 0.0;

                        if (do_schwarz_screen && Dnorm1234 * Schwarz(s1, s2) * Schwarz(s3, s4) < fock_precision)
                            continue;

                        auto bf4_first = shell2bf[s4];
                        auto n4 = obs[s4].size();

                        num_ints_computed += n1 * n2 * n3 * n4;

                        // compute the permutational degeneracy (i.e. # of equivalents) of
                        // the given shell set
                        auto s12_deg = (s1 == s2) ? 1 : 2;
                        auto s34_deg = (s3 == s4) ? 1 : 2;
                        auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
                        auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

                        engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                    obs[s1], obs[s2], obs[s3], obs[s4], sp12, sp34);
                        const auto *buf_1234 = buf[0];

                        if (buf_1234 == nullptr) continue; // if all integrals screened out, skip to next quartet

                        for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                            const auto bf1 = f1 + bf1_first;
                            for (auto f2 = 0; f2 != n2; ++f2) {
                                const auto bf2 = f2 + bf2_first;
                                for (auto f3 = 0; f3 != n3; ++f3) {
                                    const auto bf3 = f3 + bf3_first;
                                    for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                                        const auto bf4 = f4 + bf4_first;

                                        const auto value = buf_1234[f1234];

                                        const auto value_scal_by_deg = value * s1234_deg;
                                        if constexpr(kind == SpinorbitalKind::Restricted) {
                                            j(bf1, bf2) += D(bf3, bf4) * value_scal_by_deg;
                                            j(bf3, bf4) += D(bf1, bf2) * value_scal_by_deg;
                                        }
                                        else if constexpr(kind == SpinorbitalKind::Unrestricted) {

                                            ja(bf1, bf2) +=
                                                (Da(bf3, bf4) + Db(bf3, bf4)) * value_scal_by_deg;
                                            ja(bf3, bf4) +=
                                                (Da(bf1, bf2) + Db(bf1, bf2)) * value_scal_by_deg;

                                            jb(bf1, bf2) +=
                                                (Da(bf3, bf4) + Db(bf3, bf4)) * value_scal_by_deg;
                                            jb(bf3, bf4) +=
                                                (Da(bf1, bf2) + Db(bf1, bf2)) * value_scal_by_deg;
                                        }
                                        else if constexpr(kind == SpinorbitalKind::General) {
                                            j(bf1, bf2) += 2 * D(bf3, bf4) * value_scal_by_deg;
                                            j(bf3, bf4) += 2 * D(bf1, bf2) * value_scal_by_deg;
                                            j(n + bf1, n + bf2) += 2 * D(n + bf3, n + bf4) * value_scal_by_deg;
                                            j(n + bf3, n + bf4) += 2 * D(n + bf1, n + bf2) * value_scal_by_deg;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }; // end of lambda

    tonto::parallel::parallel_do(lambda);

    // accumulate contributions from all threads
    for (auto i = 1; i < nthreads; ++i) {
        J[0] += J[i];
    }
    // symmetrize the result and return
    MatRM JJ(J[0].rows(), J[0].cols());
    if constexpr(kind == SpinorbitalKind::Restricted || kind == SpinorbitalKind::General) {
        JJ = 0.5 * (J[0] + J[0].transpose());
    }
    else if constexpr(kind == SpinorbitalKind::Unrestricted) {
        JJ.alpha() = 0.5 * (J[0].alpha() + J[0].alpha().transpose());
        JJ.beta() = 0.5 * (J[0].beta() + J[0].beta().transpose());
    }
    tonto::timing::stop(tonto::timing::category::ints2e);
    return JJ;
}
}
