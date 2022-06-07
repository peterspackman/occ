#pragma once
#include <array>
#include <occ/core/atom.h>
#include <occ/core/multipole.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/expectation.h>
#include <occ/qm/mo.h>
#include <occ/qm/occshell.h>
#include <vector>

namespace occ::qm {

namespace impl {
void j_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0, int bf1,
               int bf2, int bf3, double value) noexcept;
void jk_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, Eigen::Ref<Mat> K,
                int bf0, int bf1, int bf2, int bf3, double value) noexcept;
void fock_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0, int bf1,
                  int bf2, int bf3, double value) noexcept;
void j_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0, int bf1,
               int bf2, int bf3, double value) noexcept;
void jk_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, Eigen::Ref<Mat> K,
                int bf0, int bf1, int bf2, int bf3, double value) noexcept;
void fock_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0, int bf1,
                  int bf2, int bf3, double value) noexcept;
void j_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0, int bf1,
               int bf2, int bf3, double value) noexcept;
void jk_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, Eigen::Ref<Mat> K,
                int bf0, int bf1, int bf2, int bf3, double value) noexcept;
void fock_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0, int bf1,
                  int bf2, int bf3, double value) noexcept;

template <occ::qm::SpinorbitalKind sk>
void delegate_fock(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0, int bf1,
                   int bf2, int bf3, double value) {
    if constexpr (sk == SpinorbitalKind::Restricted) {
        impl::fock_inner_r(D, F, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
        impl::fock_inner_u(D, F, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::General) {
        impl::fock_inner_g(D, F, bf0, bf1, bf2, bf3, value);
    }
}

template <occ::qm::SpinorbitalKind sk>
void delegate_jk(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, Eigen::Ref<Mat> K,
                 int bf0, int bf1, int bf2, int bf3, double value) {
    if constexpr (sk == SpinorbitalKind::Restricted) {
        impl::jk_inner_r(D, J, K, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
        impl::jk_inner_u(D, J, K, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::General) {
        impl::jk_inner_g(D, J, K, bf0, bf1, bf2, bf3, value);
    }
}

template <occ::qm::SpinorbitalKind sk>
void delegate_j(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0, int bf1,
                int bf2, int bf3, double value) {
    if constexpr (sk == SpinorbitalKind::Restricted) {
        impl::j_inner_r(D, J, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
        impl::j_inner_u(D, J, bf0, bf1, bf2, bf3, value);
    } else if constexpr (sk == SpinorbitalKind::General) {
        impl::j_inner_g(D, J, bf0, bf1, bf2, bf3, value);
    }
}

} // namespace impl

class IntegralEngine {
  public:
    template <size_t num_centers> struct IntegralResult {
        int thread{0};
        std::array<int, num_centers> shell{0};
        std::array<int, num_centers> bf{0};
        std::array<int, num_centers> dims{0};
        const double *buffer{nullptr};
    };

    using ShellList = std::vector<OccShell>;
    using AtomList = std::vector<occ::core::Atom>;
    using ShellPairList = std::vector<std::vector<size_t>>;
    using IntEnv = cint::IntegralEnvironment;
    using ShellKind = OccShell::Kind;
    using Op = cint::Operator;

    IntegralEngine(const AtomList &at, const ShellList &sh)
        : m_aobasis(at, sh), m_env(at, sh) {

        if (is_spherical()) {
            compute_shellpairs<ShellKind::Spherical>();
        } else {
            compute_shellpairs<ShellKind::Cartesian>();
        }
    }

    inline auto nbf() const noexcept { return m_aobasis.nbf(); }
    inline auto nbf_aux() const noexcept { return m_auxbasis.nbf(); }
    inline auto nsh() const noexcept { return m_aobasis.nsh(); }
    inline auto nsh_aux() const noexcept { return m_auxbasis.nsh(); }
    inline const AOBasis &aobasis() const { return m_aobasis; }
    inline const AOBasis &auxbasis() const { return m_auxbasis; }

    inline const auto &first_bf() const noexcept {
        return m_aobasis.first_bf();
    }
    inline const auto &first_bf_aux() const noexcept {
        return m_auxbasis.first_bf();
    }
    inline const auto &shellpairs() const noexcept { return m_shellpairs; }
    inline const auto &shells() const noexcept { return m_aobasis.shells(); }

    inline void set_auxiliary_basis(const ShellList &bs, bool dummy = false) {
        if (!dummy) {
            clear_auxiliary_basis();
            m_auxbasis = AOBasis(m_aobasis.atoms(), bs);
            ShellList combined = m_aobasis.shells();
            combined.insert(combined.end(), m_auxbasis.shells().begin(),
                            m_auxbasis.shells().end());
            m_env = IntEnv(m_aobasis.atoms(), combined);
        } else {
            AtomList dummy_atoms;
            dummy_atoms.reserve(bs.size());
            for (const auto &shell : bs) {
                dummy_atoms.push_back(
                    {0, shell.origin(0), shell.origin(1), shell.origin(2)});
            }
            set_dummy_basis(dummy_atoms, bs);
        }
    }

    inline void set_dummy_basis(const AtomList &dummy_atoms,
                                const ShellList &bs) {
        clear_auxiliary_basis();
        m_auxbasis = AOBasis(dummy_atoms, bs);
        AtomList combined_sites = m_aobasis.atoms();
        combined_sites.insert(combined_sites.end(), dummy_atoms.begin(),
                              dummy_atoms.end());
        ShellList combined = m_aobasis.shells();
        combined.insert(combined.end(), m_auxbasis.shells().begin(),
                        m_auxbasis.shells().end());
        m_env = IntEnv(combined_sites, combined);
    }

    inline void clear_auxiliary_basis() {
        m_auxbasis = AOBasis();
        m_env = IntEnv(m_aobasis.atoms(), m_aobasis.shells());
    }

    inline bool have_auxiliary_basis() const noexcept {
        return m_auxbasis.nsh() > 0;
    }

    template <Op op, ShellKind kind, typename Lambda>
    void evaluate_two_center(Lambda &f, int thread_id = 0) const noexcept {
        occ::qm::cint::Optimizer opt(m_env, op, 2);
        auto nthreads = occ::parallel::get_num_threads();
        auto bufsize = buffer_size_1e(op);

        auto buffer = std::make_unique<double[]>(bufsize);
        const auto &first_bf = m_aobasis.first_bf();
        for (int p = 0, pq = 0; p < m_aobasis.size(); p++) {
            int bf1 = first_bf[p];
            const auto &sh1 = m_aobasis[p];
            for (const int &q : m_shellpairs.at(p)) {
                if (pq++ % nthreads != thread_id)
                    continue;
                int bf2 = first_bf[q];
                const auto &sh2 = m_aobasis[q];
                std::array<int, 2> idxs{p, q};
                IntegralResult<2> args{
                    thread_id,
                    idxs,
                    {bf1, bf2},
                    m_env.two_center_helper<op, kind>(idxs, opt.optimizer_ptr(),
                                                      buffer.get(), nullptr),
                    buffer.get()};
                if (args.dims[0] > -1)
                    f(args);
            }
        }
    }

    template <Op op, ShellKind kind, typename Lambda>
    void evaluate_four_center(Lambda &f, const Mat &Dnorm = Mat(),
                              const Mat &Schwarz = Mat(),
                              int thread_id = 0) const noexcept {
        auto nthreads = occ::parallel::get_num_threads();
        occ::qm::cint::Optimizer opt(m_env, Op::coulomb, 4);
        auto buffer = std::make_unique<double[]>(buffer_size_2e());
        std::array<int, 4> shell_idx;
        std::array<int, 4> bf;

        const auto &first_bf = m_aobasis.first_bf();
        const auto do_schwarz_screen =
            Schwarz.cols() != 0 && Schwarz.rows() != 0;
        for (int p = 0, pqrs = 0; p < m_aobasis.size(); p++) {
            const auto &sh1 = m_aobasis[p];
            bf[0] = first_bf[p];
            const auto &plist = m_shellpairs.at(p);
            for (const int &q : plist) {
                bf[1] = first_bf[q];
                const auto &sh2 = m_aobasis[q];
                const auto DnormPQ = do_schwarz_screen ? Dnorm(p, q) : 0.;
                for (int r = 0; r <= p; r++) {
                    const auto &sh3 = m_aobasis[r];
                    bf[2] = first_bf[r];
                    const auto s_max = (p == r) ? q : r;
                    const auto DnormPQR =
                        do_schwarz_screen
                            ? std::max(Dnorm(p, r),
                                       std::max(Dnorm(q, r), DnormPQ))
                            : 0.;

                    for (const int s : m_shellpairs.at(r)) {
                        if (s > s_max)
                            break;
                        if (pqrs++ % nthreads != thread_id)
                            continue;
                        const auto DnormPQRS =
                            do_schwarz_screen
                                ? std::max(
                                      Dnorm(p, s),
                                      std::max(Dnorm(q, s),
                                               std::max(Dnorm(r, s), DnormPQR)))
                                : 0.0;
                        if (do_schwarz_screen &&
                            DnormPQRS * Schwarz(p, q) * Schwarz(r, s) <
                                m_precision)
                            continue;

                        bf[3] = first_bf[s];
                        const auto &sh4 = m_aobasis[s];
                        shell_idx = {p, q, r, s};

                        IntegralResult<4> args{
                            thread_id, shell_idx, bf,
                            m_env.four_center_helper<Op::coulomb, kind>(
                                shell_idx, opt.optimizer_ptr(), buffer.get(),
                                nullptr),
                            buffer.get()};
                        if (args.dims[0] > -1)
                            f(args);
                    }
                }
            }
        }
    }

    template <ShellKind kind, typename Lambda>
    void evaluate_three_center_aux(Lambda &f,
                                   int thread_id = 0) const noexcept {
        auto nthreads = occ::parallel::get_num_threads();
        occ::qm::cint::Optimizer opt(m_env, Op::coulomb, 3);
        auto buffer = std::make_unique<double[]>(buffer_size_3e());
        IntegralResult<3> args;
        args.thread = thread_id;
        args.buffer = buffer.get();
        std::array<int, 3> shell_idx;
        const auto &first_bf_ao = m_aobasis.first_bf();
        const auto &first_bf_aux = m_auxbasis.first_bf();
        for (int auxP = 0; auxP < m_auxbasis.size(); auxP++) {
            if (auxP % nthreads != thread_id)
                continue;
            const auto &shauxP = m_auxbasis[auxP];
            args.bf[2] = first_bf_aux[auxP];
            args.shell[2] = auxP;
            for (int p = 0; p < m_aobasis.size(); p++) {
                args.bf[0] = first_bf_ao[p];
                args.shell[0] = p;
                const auto &shp = m_aobasis[p];
                const auto &plist = m_shellpairs.at(p);
                if ((shp.extent > 0.0) &&
                    (shp.origin - shauxP.origin).norm() > shp.extent) {
                    continue;
                }
                for (const int &q : plist) {
                    args.bf[1] = first_bf_ao[q];
                    args.shell[1] = q;
                    const auto &shq = m_aobasis[q];
                    shell_idx = {p, q,
                                 auxP + static_cast<int>(m_aobasis.size())};
                    if ((shq.extent > 0.0) &&
                        (shq.origin - shauxP.origin).norm() > shq.extent) {
                        continue;
                    }
                    args.dims = m_env.three_center_helper<Op::coulomb, kind>(
                        shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr);
                    if (args.dims[0] > -1) {
                        f(args);
                    }
                }
            }
        }
    }

    template <Op op, ShellKind kind = ShellKind::Cartesian>
    Mat one_electron_operator() const noexcept {
        auto nthreads = occ::parallel::get_num_threads();
        const auto nbf = m_aobasis.nbf();
        Mat result = Mat::Zero(nbf, nbf);
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(nbf, nbf));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }
        auto f = [&results](const IntegralResult<2> &args) {
            auto &result = results[args.thread];
            Eigen::Map<const occ::Mat> tmp(args.buffer, args.dims[0],
                                           args.dims[1]);
            result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) =
                tmp;
            if (args.shell[0] != args.shell[1]) {
                result.block(args.bf[1], args.bf[0], args.dims[1],
                             args.dims[0]) = tmp.transpose();
            }
        };

        auto lambda = [&](int thread_id) {
            evaluate_two_center<op, kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; ++i) {
            results[0].noalias() += results[i];
        }
        return results[0];
    }

    template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
    Mat compute_shellblock_norm(const Mat &matrix) const noexcept {
        occ::timing::start(occ::timing::category::ints1e);
        const auto nsh = m_aobasis.size();
        const auto &first_bf = m_aobasis.first_bf();
        Mat result(nsh, nsh);

        for (size_t s1 = 0; s1 < nsh; ++s1) {
            const auto &s1_first = first_bf[s1];
            const auto &s1_size = m_aobasis[s1].size();
            for (size_t s2 = 0; s2 < nsh; ++s2) {
                const auto &s2_first = first_bf[s2];
                const auto &s2_size = m_aobasis[s2].size();

                if constexpr (sk == SpinorbitalKind::Restricted) {
                    result(s1, s2) =
                        matrix.block(s1_first, s2_first, s1_size, s2_size)
                            .lpNorm<Eigen::Infinity>();
                } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
                    const auto alpha =
                        occ::qm::block::a(matrix)
                            .block(s1_first, s2_first, s1_size, s2_size)
                            .lpNorm<Eigen::Infinity>();
                    const auto beta =
                        occ::qm::block::b(matrix)
                            .block(s1_first, s2_first, s1_size, s2_size)
                            .lpNorm<Eigen::Infinity>();
                    result(s1, s2) = std::max(alpha, beta);
                } else if constexpr (sk == SpinorbitalKind::General) {
                    const auto aa =
                        occ::qm::block::aa(matrix)
                            .block(s1_first, s2_first, s1_size, s2_size)
                            .lpNorm<Eigen::Infinity>();
                    const auto bb =
                        occ::qm::block::bb(matrix)
                            .block(s1_first, s2_first, s1_size, s2_size)
                            .lpNorm<Eigen::Infinity>();
                    const auto ab =
                        occ::qm::block::ab(matrix)
                            .block(s1_first, s2_first, s1_size, s2_size)
                            .lpNorm<Eigen::Infinity>();
                    const auto ba =
                        occ::qm::block::ba(matrix)
                            .block(s1_first, s2_first, s1_size, s2_size)
                            .lpNorm<Eigen::Infinity>();
                    result(s1, s2) =
                        std::max(aa, std::max(ab, std::max(ba, bb)));
                }
            }
        }
        occ::timing::stop(occ::timing::category::ints1e);
        return result;
    }

    template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
    Mat fock_operator(const MolecularOrbitals &mo,
                      const Mat &Schwarz = Mat()) const noexcept {
        auto nthreads = occ::parallel::get_num_threads();
        constexpr Op op = Op::coulomb;
        std::vector<Mat> Fmats;
        Fmats.emplace_back(Mat::Zero(mo.D.rows(), mo.D.cols()));
        for (size_t i = 1; i < nthreads; i++) {
            Fmats.push_back(Fmats[0]);
        }
        Mat Dnorm = compute_shellblock_norm<sk, kind>(mo.D);

        const auto &D = mo.D;
        auto f = [&D, &Fmats](const IntegralResult<4> &args) {
            auto &F = Fmats[args.thread];
            auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
            auto pr_qs_degree = (args.shell[0] == args.shell[2])
                                    ? (args.shell[1] == args.shell[3] ? 1 : 2)
                                    : 2;
            auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
            auto scale = pq_degree * rs_degree * pr_qs_degree;

            for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
                const auto bf3 = f3 + args.bf[3];
                for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
                    const auto bf2 = f2 + args.bf[2];
                    for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
                        const auto bf1 = f1 + args.bf[1];
                        for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
                            const auto bf0 = f0 + args.bf[0];
                            const auto value = args.buffer[f0123] * scale;
                            impl::delegate_fock<sk>(D, F, bf0, bf1, bf2, bf3,
                                                    value);
                        }
                    }
                }
            }
        };
        auto lambda = [&](int thread_id) {
            evaluate_four_center<op, kind>(f, Dnorm, Schwarz, thread_id);
        };
        occ::timing::start(occ::timing::category::fock);
        occ::parallel::parallel_do(lambda);
        occ::timing::stop(occ::timing::category::fock);

        Mat F = Mat::Zero(Fmats[0].rows(), Fmats[0].cols());

        for (const auto &part : Fmats) {
            if constexpr (sk == SpinorbitalKind::Restricted) {
                F.noalias() += (part + part.transpose());
            } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
                auto Fa = occ::qm::block::a(part);
                auto Fb = occ::qm::block::b(part);
                occ::qm::block::a(F).noalias() += (Fa + Fa.transpose());
                occ::qm::block::b(F).noalias() += (Fb + Fb.transpose());
            } else if constexpr (sk == SpinorbitalKind::General) {
                auto Faa = occ::qm::block::aa(part);
                auto Fab = occ::qm::block::ab(part);
                auto Fba = occ::qm::block::ba(part);
                auto Fbb = occ::qm::block::bb(part);
                occ::qm::block::aa(F).noalias() += (Faa + Faa.transpose());
                occ::qm::block::ab(F).noalias() += (Fab + Fab.transpose());
                occ::qm::block::ba(F).noalias() += (Fba + Fba.transpose());
                occ::qm::block::bb(F).noalias() += (Fbb + Fbb.transpose());
            }
        }
        F *= 0.5;

        return F;
    }

    template <ShellKind sk = ShellKind::Cartesian>
    Mat fock_operator_mixed_basis(const Mat &D, const AOBasis &D_bs,
                                  bool is_shell_diagonal) {
        set_auxiliary_basis(D_bs.shells(), false);
        constexpr Op op = Op::coulomb;
        occ::timing::start(occ::timing::category::ints2e);
        auto nthreads = occ::parallel::get_num_threads();

        const int nbf = m_aobasis.nbf();
        const int nsh = m_aobasis.size();
        const int nbf_aux = m_auxbasis.nbf();
        const int nsh_aux = m_auxbasis.size();
        assert(D.cols() == D.rows() && D.cols() == n_D);

        std::vector<Mat> G(nthreads, Mat::Zero(nbf, nbf));

        // construct the 2-electron repulsion integrals engine
        auto shell2bf = m_aobasis.first_bf();
        auto shell2bf_D = m_auxbasis.first_bf();

        auto lambda = [&](int thread_id) {
            auto &g = G[thread_id];
            occ::qm::cint::Optimizer opt(m_env, Op::coulomb, 4);
            auto buffer = std::make_unique<double[]>(buffer_size_2e());

            std::array<int, 4> idxs;
            // loop over permutationally-unique set of shells
            for (int s1 = 0, s1234 = 0; s1 != nsh; ++s1) {
                int bf1_first =
                    shell2bf[s1]; // first basis function in this shell
                int n1 = m_aobasis[s1]
                             .size(); // number of basis functions in this shell

                for (int s2 = 0; s2 <= s1; ++s2) {
                    int bf2_first = shell2bf[s2];
                    int n2 = m_aobasis[s2].size();

                    for (int s3 = 0; s3 < nsh_aux; ++s3) {
                        int bf3_first = shell2bf_D[s3];
                        int n3 = D_bs[s3].size();

                        int s4_begin = is_shell_diagonal ? s3 : 0;
                        int s4_fence = is_shell_diagonal ? s3 + 1 : nsh_aux;

                        for (int s4 = s4_begin; s4 != s4_fence; ++s4, ++s1234) {
                            if (s1234 % nthreads != thread_id)
                                continue;

                            int bf4_first = shell2bf_D[s4];
                            int n4 = D_bs[s4].size();

                            // compute the permutational degeneracy (i.e. # of
                            // equivalents) of the given shell set
                            double s12_deg = (s1 == s2) ? 1.0 : 2.0;

                            std::array<int, 4> dims;
                            if (s3 >= s4) {
                                double s34_deg = (s3 == s4) ? 1.0 : 2.0;
                                double s1234_deg = s12_deg * s34_deg;
                                // auto s1234_deg = s12_deg;
                                std::array<int, 4> idxs{s1, s2, s3 + nsh,
                                                        s4 + nsh};
                                dims = m_env.four_center_helper<op, sk>(
                                    idxs, opt.optimizer_ptr(), buffer.get(),
                                    nullptr);

                                if (dims[0] >= 0) {
                                    const auto *buf_1234 = buffer.get();
                                    for (auto f4 = 0, f1234 = 0; f4 != n4;
                                         ++f4) {
                                        const auto bf4 = f4 + bf4_first;
                                        for (auto f3 = 0; f3 != n3; ++f3) {
                                            const auto bf3 = f3 + bf3_first;
                                            for (auto f2 = 0; f2 != n2; ++f2) {
                                                const auto bf2 = f2 + bf2_first;
                                                for (auto f1 = 0; f1 != n1;
                                                     ++f1, ++f1234) {
                                                    const auto bf1 =
                                                        f1 + bf1_first;

                                                    const auto value =
                                                        buf_1234[f1234];
                                                    const auto
                                                        value_scal_by_deg =
                                                            value * s1234_deg;
                                                    g(bf1, bf2) +=
                                                        2.0 * D(bf3, bf4) *
                                                        value_scal_by_deg;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            std::array<int, 4> idxs{s1, s3 + nsh, s2, s4 + nsh};
                            dims = m_env.four_center_helper<op, sk>(
                                idxs, opt.optimizer_ptr(), buffer.get(),
                                nullptr);
                            if (dims[0] < 0)
                                continue;

                            const auto *buf_1324 = buffer.get();

                            for (auto f4 = 0, f1324 = 0; f4 != n4; ++f4) {
                                const auto bf4 = f4 + bf4_first;
                                for (auto f2 = 0; f2 != n2; ++f2) {
                                    const auto bf2 = f2 + bf2_first;
                                    for (auto f3 = 0; f3 != n3; ++f3) {
                                        const auto bf3 = f3 + bf3_first;
                                        for (auto f1 = 0; f1 != n1;
                                             ++f1, ++f1324) {
                                            const auto bf1 = f1 + bf1_first;

                                            const auto value = buf_1324[f1324];
                                            const auto value_scal_by_deg =
                                                value * s12_deg;
                                            g(bf1, bf2) -=
                                                D(bf3, bf4) * value_scal_by_deg;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }; // thread lambda

        occ::parallel::parallel_do(lambda);

        // accumulate contributions from all threads
        for (size_t i = 1; i != nthreads; ++i) {
            G[0] += G[i];
        }
        occ::timing::stop(occ::timing::category::ints2e);

        clear_auxiliary_basis();
        // symmetrize the result and return
        return 0.5 * (G[0] + G[0].transpose());
    }

    template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
    Mat coulomb(const MolecularOrbitals &mo,
                const Mat &Schwarz = Mat()) const noexcept {
        auto nthreads = occ::parallel::get_num_threads();
        constexpr Op op = Op::coulomb;
        std::vector<Mat> Jmats;
        Jmats.emplace_back(Mat::Zero(mo.D.rows(), mo.D.cols()));
        for (size_t i = 1; i < nthreads; i++) {
            Jmats.push_back(Jmats[0]);
        }

        Mat Dnorm = compute_shellblock_norm<sk, kind>(mo.D);

        const auto &D = mo.D;
        auto f = [&D, &Jmats](const IntegralResult<4> &args) {
            auto &J = Jmats[args.thread];
            auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
            auto pr_qs_degree = (args.shell[0] == args.shell[2])
                                    ? (args.shell[1] == args.shell[3] ? 1 : 2)
                                    : 2;
            auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
            auto scale = pq_degree * rs_degree * pr_qs_degree;

            for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
                const auto bf3 = f3 + args.bf[3];
                for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
                    const auto bf2 = f2 + args.bf[2];
                    for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
                        const auto bf1 = f1 + args.bf[1];
                        for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
                            const auto bf0 = f0 + args.bf[0];
                            const auto value = args.buffer[f0123] * scale;
                            impl::delegate_j<sk>(D, J, bf0, bf1, bf2, bf3,
                                                 value);
                        }
                    }
                }
            }
        };
        auto lambda = [&](int thread_id) {
            evaluate_four_center<op, kind>(f, Dnorm, Schwarz, thread_id);
        };
        occ::timing::start(occ::timing::category::fock);
        occ::parallel::parallel_do(lambda);
        occ::timing::stop(occ::timing::category::fock);

        Mat J = Mat::Zero(Jmats[0].rows(), Jmats[0].cols());

        for (size_t i = 0; i < nthreads; i++) {
            if constexpr (sk == SpinorbitalKind::Restricted) {
                J.noalias() += (Jmats[i] + Jmats[i].transpose());
            } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
                {
                    auto Ja = occ::qm::block::a(Jmats[i]);
                    auto Jb = occ::qm::block::b(Jmats[i]);
                    occ::qm::block::a(J).noalias() += (Ja + Ja.transpose());
                    occ::qm::block::b(J).noalias() += (Jb + Jb.transpose());
                }
            } else if constexpr (sk == SpinorbitalKind::General) {
                {
                    auto Jaa = occ::qm::block::aa(Jmats[i]);
                    auto Jab = occ::qm::block::ab(Jmats[i]);
                    auto Jba = occ::qm::block::ba(Jmats[i]);
                    auto Jbb = occ::qm::block::bb(Jmats[i]);
                    occ::qm::block::aa(J).noalias() += (Jaa + Jaa.transpose());
                    occ::qm::block::ab(J).noalias() += (Jab + Jab.transpose());
                    occ::qm::block::ba(J).noalias() += (Jba + Jba.transpose());
                    occ::qm::block::bb(J).noalias() += (Jbb + Jbb.transpose());
                }
            }
        }
        J *= 0.5;
        return J;
    }

    template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
    std::pair<Mat, Mat>
    coulomb_and_exchange(const MolecularOrbitals &mo,
                         const Mat &Schwarz = Mat()) const noexcept {
        auto nthreads = occ::parallel::get_num_threads();
        constexpr Op op = Op::coulomb;
        std::vector<Mat> Jmats;
        std::vector<Mat> Kmats;
        Jmats.emplace_back(Mat::Zero(mo.D.rows(), mo.D.cols()));
        Kmats.emplace_back(Mat::Zero(mo.D.rows(), mo.D.cols()));
        for (size_t i = 1; i < nthreads; i++) {
            Jmats.push_back(Jmats[0]);
            Kmats.push_back(Kmats[0]);
        }

        Mat Dnorm = compute_shellblock_norm<sk, kind>(mo.D);

        const auto &D = mo.D;
        auto f = [&D, &Jmats, &Kmats](const IntegralResult<4> &args) {
            auto &J = Jmats[args.thread];
            auto &K = Kmats[args.thread];
            auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
            auto pr_qs_degree = (args.shell[0] == args.shell[2])
                                    ? (args.shell[1] == args.shell[3] ? 1 : 2)
                                    : 2;
            auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
            auto scale = pq_degree * rs_degree * pr_qs_degree;

            for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
                const auto bf3 = f3 + args.bf[3];
                for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
                    const auto bf2 = f2 + args.bf[2];
                    for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
                        const auto bf1 = f1 + args.bf[1];
                        for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
                            const auto bf0 = f0 + args.bf[0];
                            const auto value = args.buffer[f0123] * scale;
                            impl::delegate_jk<sk>(D, J, K, bf0, bf1, bf2, bf3,
                                                  value);
                        }
                    }
                }
            }
        };
        auto lambda = [&](int thread_id) {
            evaluate_four_center<op, kind>(f, Dnorm, Schwarz, thread_id);
        };
        occ::timing::start(occ::timing::category::fock);
        occ::parallel::parallel_do(lambda);
        occ::timing::stop(occ::timing::category::fock);

        std::pair<Mat, Mat> JK(Mat::Zero(Jmats[0].rows(), Jmats[0].cols()),
                               Mat::Zero(Kmats[0].rows(), Kmats[0].cols()));

        Mat &J = JK.first;
        Mat &K = JK.second;

        for (size_t i = 0; i < nthreads; i++) {
            if constexpr (sk == SpinorbitalKind::Restricted) {
                J.noalias() += (Jmats[i] + Jmats[i].transpose());
                K.noalias() += (Kmats[i] + Kmats[i].transpose());
            } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
                {
                    auto Ja = occ::qm::block::a(Jmats[i]);
                    auto Jb = occ::qm::block::b(Jmats[i]);
                    occ::qm::block::a(J).noalias() += (Ja + Ja.transpose());
                    occ::qm::block::b(J).noalias() += (Jb + Jb.transpose());
                }
                {
                    auto Ka = occ::qm::block::a(Kmats[i]);
                    auto Kb = occ::qm::block::b(Kmats[i]);
                    occ::qm::block::a(J).noalias() += (Ka + Ka.transpose());
                    occ::qm::block::b(J).noalias() += (Kb + Kb.transpose());
                }
            } else if constexpr (sk == SpinorbitalKind::General) {
                {
                    auto Jaa = occ::qm::block::aa(Jmats[i]);
                    auto Jab = occ::qm::block::ab(Jmats[i]);
                    auto Jba = occ::qm::block::ba(Jmats[i]);
                    auto Jbb = occ::qm::block::bb(Jmats[i]);
                    occ::qm::block::aa(J).noalias() += (Jaa + Jaa.transpose());
                    occ::qm::block::ab(J).noalias() += (Jab + Jab.transpose());
                    occ::qm::block::ba(J).noalias() += (Jba + Jba.transpose());
                    occ::qm::block::bb(J).noalias() += (Jbb + Jbb.transpose());
                }
                {
                    auto Kaa = occ::qm::block::aa(Kmats[i]);
                    auto Kab = occ::qm::block::ab(Kmats[i]);
                    auto Kba = occ::qm::block::ba(Kmats[i]);
                    auto Kbb = occ::qm::block::bb(Kmats[i]);
                    occ::qm::block::aa(J).noalias() += (Kaa + Kaa.transpose());
                    occ::qm::block::ab(J).noalias() += (Kab + Kab.transpose());
                    occ::qm::block::ba(J).noalias() += (Kba + Kba.transpose());
                    occ::qm::block::bb(J).noalias() += (Kbb + Kbb.transpose());
                }
            }
        }
        J *= 0.5;
        K *= 0.5;
        return JK;
    }

    template <ShellKind kind = ShellKind::Cartesian>
    Mat
    point_charge_potential(const std::vector<occ::core::PointCharge> &charges) {
        const auto nbf = m_aobasis.nbf();
        ShellList dummy_shells;
        dummy_shells.reserve(charges.size());
        for (size_t i = 0; i < charges.size(); i++) {
            dummy_shells.push_back(OccShell(charges[i]));
        }
        set_auxiliary_basis(dummy_shells, true);
        auto nthreads = occ::parallel::get_num_threads();
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(nbf, nbf));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }

        size_t nsh = m_aobasis.size();
        auto f = [nsh, &results](const IntegralResult<3> &args) {
            auto &result = results[args.thread];
            Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
            result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) +=
                tmp;
            if (args.shell[0] != args.shell[1]) {
                result.block(args.bf[1], args.bf[0], args.dims[1],
                             args.dims[0]) += tmp.transpose();
            }
        };

        auto lambda = [&](int thread_id) {
            evaluate_three_center_aux<kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; i++) {
            results[0] += results[i];
        }
        return results[0];
    }

    template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
    Vec electric_potential(const MolecularOrbitals &mo, const Mat3N &points) {
        ShellList dummy_shells;
        dummy_shells.reserve(points.cols());
        for (size_t i = 0; i < points.cols(); i++) {
            dummy_shells.push_back(
                OccShell({1.0, {points(0, i), points(1, i), points(2, i)}}));
        }
        set_auxiliary_basis(dummy_shells, true);
        auto nthreads = occ::parallel::get_num_threads();
        std::vector<Vec> results;
        results.emplace_back(Vec::Zero(points.cols()));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }
        const auto &D = mo.D;
        auto f = [&D, &results](const IntegralResult<3> &args) {
            auto &v = results[args.thread];
            auto scale = (args.shell[0] == args.shell[1]) ? 1 : 2;
            Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
            if constexpr (sk == SpinorbitalKind::Restricted) {
                v(args.shell[2]) +=
                    (D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                         .array() *
                     tmp.array())
                        .sum() *
                    scale;
            } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
                const auto alpha = occ::qm::block::a(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                const auto beta = occ::qm::block::a(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                v(args.shell[2]) +=
                    ((alpha.array() + beta.array()) * tmp.array()).sum() *
                    scale;
            } else if constexpr (sk == SpinorbitalKind::General) {
                const auto aa = occ::qm::block::aa(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                const auto ab = occ::qm::block::ab(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                const auto ba = occ::qm::block::ba(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                const auto bb = occ::qm::block::bb(D).block(
                    args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
                v(args.shell[2]) +=
                    ((aa.array() + ab.array() + ba.array() + bb.array()) *
                     tmp.array())
                        .sum() *
                    scale;
            }
        };

        auto lambda = [&](int thread_id) {
            evaluate_three_center_aux<kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; i++) {
            results[0] += results[i];
        }
        return 2 * results[0];
    }

    template <ShellKind kind>
    inline void compute_shellpairs(double threshold = 1e-12) {
        constexpr auto op = Op::overlap;
        const auto nsh = m_aobasis.size();
        m_shellpairs.resize(nsh);
        auto buffer = std::make_unique<double[]>(buffer_size_1e());
        for (int p = 0; p < nsh; p++) {
            auto &plist = m_shellpairs[p];
            const auto &sh1 = m_aobasis[p];
            for (int q = 0; q <= p; q++) {
                if (m_aobasis.shells_share_origin(p, q)) {
                    plist.push_back(q);
                    continue;
                }
                const auto &sh2 = m_aobasis[q];
                std::array<int, 2> idxs{p, q};
                std::array<int, 2> dims = m_env.two_center_helper<op, kind>(
                    idxs, nullptr, buffer.get(), nullptr);
                Eigen::Map<const occ::Mat> tmp(buffer.get(), dims[0], dims[1]);
                if (tmp.norm() >= threshold) {
                    plist.push_back(q);
                }
            }
        }
    }

    template <int order, SpinorbitalKind sk,
              ShellKind kind = ShellKind::Cartesian>
    auto multipole(const MolecularOrbitals &mo,
                   const Vec3 &origin = {0, 0, 0}) const {

        static_assert(sk == SpinorbitalKind::Restricted,
                      "Unrestricted and General cases not implemented for "
                      "multipoles yet");
        constexpr std::array<Op, 4> ops{Op::overlap, Op::dipole, Op::quadrupole,
                                        Op::hexadecapole};
        constexpr Op op = ops[order];

        auto nthreads = occ::parallel::get_num_threads();
        size_t num_components =
            occ::core::num_unique_multipole_components(order);
        m_env.set_common_origin({origin.x(), origin.y(), origin.z()});
        std::vector<Vec> results;
        results.push_back(Vec::Zero(num_components));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }
        const auto &D = mo.D;
        /*
         * For symmetric matrices
         * the of a matrix product tr(D @ O) is equal to
         * the sum of the elementwise product with the transpose:
         * tr(D @ O) == sum(D * O^T)
         * since expectation is -2 tr(D @ O) we factor that into the
         * inner loop
         */
        auto f = [&D, &results,
                  &num_components](const IntegralResult<2> &args) {
            auto &result = results[args.thread];
            size_t offset = 0;
            double scale = (args.shell[0] != args.shell[1]) ? 2.0 : 1.0;
            for (size_t n = 0; n < num_components; n++) {
                Eigen::Map<const occ::Mat> tmp(args.buffer + offset,
                                               args.dims[0], args.dims[1]);
                result(n) += scale * (D.block(args.bf[0], args.bf[1],
                                              args.dims[0], args.dims[1])
                                          .array() *
                                      tmp.array())
                                         .sum();
                offset += tmp.size();
            }
        };

        auto lambda = [&](int thread_id) {
            evaluate_two_center<op, kind>(f, thread_id);
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; ++i) {
            results[0].noalias() += results[i];
        }

        results[0] *= -2;
        return results[0];
    }

    template <ShellKind kind = ShellKind::Cartesian>
    Mat schwarz() const noexcept {
        auto nthreads = occ::parallel::get_num_threads();
        constexpr Op op = Op::coulomb;
        constexpr bool use_euclidean_norm{false};
        const auto nsh = m_aobasis.size();
        const auto &first_bf = m_aobasis.first_bf();
        std::vector<Mat> results;
        results.emplace_back(Mat::Zero(nsh, nsh));
        for (size_t i = 1; i < nthreads; i++) {
            results.push_back(results[0]);
        }

        auto f = [&results](const IntegralResult<4> &args) {
            auto &result = results[args.thread];
            auto N = args.dims[0] * args.dims[1];
            Eigen::Map<const occ::Mat> tmp(args.buffer, N, N);
            double sq_norm =
                use_euclidean_norm ? tmp.norm() : tmp.lpNorm<Eigen::Infinity>();
            double norm = std::sqrt(sq_norm);
            result(args.shell[0], args.shell[1]) = norm;
            result(args.shell[1], args.shell[0]) = norm;
        };

        auto lambda = [&](int thread_id) {
            auto buffer = std::make_unique<double[]>(buffer_size_2e());
            for (int p = 0, pq = 0; p < nsh; p++) {
                int bf1 = first_bf[p];
                const auto &sh1 = m_aobasis[p];
                for (const int &q : m_shellpairs.at(p)) {
                    if (pq++ % nthreads != thread_id)
                        continue;
                    int bf2 = first_bf[q];
                    const auto &sh2 = m_aobasis[q];
                    std::array<int, 4> idxs{p, q, p, q};
                    IntegralResult<4> args{
                        thread_id,
                        idxs,
                        {bf1, bf2, bf1, bf2},
                        m_env.four_center_helper<op, kind>(
                            idxs, nullptr, buffer.get(), nullptr),
                        buffer.get()};
                    if (args.dims[0] > -1)
                        f(args);
                }
            }
        };
        occ::parallel::parallel_do(lambda);

        for (auto i = 1; i < nthreads; ++i) {
            results[0].noalias() += results[i];
        }

        return results[0];
    }

    inline bool is_spherical() const noexcept {
        return m_aobasis.kind() == OccShell::Kind::Spherical;
    }

  private:
    double m_precision{1e-12};
    AOBasis m_aobasis, m_auxbasis;
    ShellPairList m_shellpairs;
    // TODO remove mutable
    mutable IntEnv m_env;

    inline size_t buffer_size_1e(const Op op = Op::overlap) const {
        auto bufsize = m_aobasis.max_shell_size() * m_aobasis.max_shell_size();
        switch (op) {
        case Op::dipole:
            bufsize *= occ::core::num_unique_multipole_components(1);
            break;
        case Op::quadrupole:
            bufsize *= occ::core::num_unique_multipole_components(2);
            break;
        case Op::octapole:
            bufsize *= occ::core::num_unique_multipole_components(3);
            break;
        case Op::hexadecapole:
            bufsize *= occ::core::num_unique_multipole_components(4);
            break;
        default:
            break;
        }
        return bufsize;
    }

    inline size_t buffer_size_3e() const {
        return m_auxbasis.max_shell_size() * buffer_size_1e();
    }

    inline size_t buffer_size_2e() const {
        return buffer_size_1e() * buffer_size_1e();
    }
};

} // namespace occ::qm
