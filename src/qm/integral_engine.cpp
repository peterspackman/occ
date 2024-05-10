#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm {

namespace impl {

template <SpinorbitalKind sk>
void accumulate_operator_symmetric(const Mat &source, Mat &dest) {
    if constexpr (sk == SpinorbitalKind::Restricted) {
        dest.noalias() += (source + source.transpose());
    } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
        auto source_a = occ::qm::block::a(source);
        auto source_b = occ::qm::block::b(source);
        occ::qm::block::a(dest).noalias() += (source_a + source_a.transpose());
        occ::qm::block::b(dest).noalias() += (source_b + source_b.transpose());
    } else if constexpr (sk == SpinorbitalKind::General) {
        auto source_aa = occ::qm::block::aa(source);
        auto source_ab = occ::qm::block::ab(source);
        auto source_ba = occ::qm::block::ba(source);
        auto source_bb = occ::qm::block::bb(source);
        occ::qm::block::aa(dest).noalias() +=
            (source_aa + source_aa.transpose());
        occ::qm::block::ab(dest).noalias() +=
            (source_ab + source_ab.transpose());
        occ::qm::block::ba(dest).noalias() +=
            (source_ba + source_ba.transpose());
        occ::qm::block::bb(dest).noalias() +=
            (source_bb + source_bb.transpose());
    }
}

inline void fock_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0,
                         int bf1, int bf2, int bf3, double value) {
    F(bf0, bf1) += D(bf2, bf3) * value;
    F(bf2, bf3) += D(bf0, bf1) * value;
    // K
    F(bf0, bf2) -= 0.25 * D(bf1, bf3) * value;
    F(bf1, bf3) -= 0.25 * D(bf0, bf2) * value;
    F(bf0, bf3) -= 0.25 * D(bf1, bf2) * value;
    F(bf1, bf2) -= 0.25 * D(bf0, bf3) * value;
}

inline void j_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                      int bf1, int bf2, int bf3, double value) {
    J(bf0, bf1) += D(bf2, bf3) * value;
    J(bf2, bf3) += D(bf0, bf1) * value;
}

inline void jk_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J,
                       Eigen::Ref<Mat> K, int bf0, int bf1, int bf2, int bf3,
                       double value) {
    J(bf0, bf1) += D(bf2, bf3) * value;
    J(bf2, bf3) += D(bf0, bf1) * value;
    // K
    K(bf0, bf2) += 0.25 * D(bf1, bf3) * value;
    K(bf1, bf3) += 0.25 * D(bf0, bf2) * value;
    K(bf0, bf3) += 0.25 * D(bf1, bf2) * value;
    K(bf1, bf2) += 0.25 * D(bf0, bf3) * value;
}

inline void fock_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0,
                         int bf1, int bf2, int bf3, double value) {
    auto Fa = occ::qm::block::a(F);
    auto Fb = occ::qm::block::b(F);
    const auto Da = occ::qm::block::a(D);
    const auto Db = occ::qm::block::b(D);
    Fa(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
    Fa(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
    Fb(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
    Fb(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;

    Fa(bf0, bf2) -= 0.5 * Da(bf1, bf3) * value;
    Fa(bf1, bf3) -= 0.5 * Da(bf0, bf2) * value;
    Fa(bf0, bf3) -= 0.5 * Da(bf1, bf2) * value;
    Fa(bf1, bf2) -= 0.5 * Da(bf0, bf3) * value;

    Fb(bf0, bf2) -= 0.5 * Db(bf1, bf3) * value;
    Fb(bf1, bf3) -= 0.5 * Db(bf0, bf2) * value;
    Fb(bf0, bf3) -= 0.5 * Db(bf1, bf2) * value;
    Fb(bf1, bf2) -= 0.5 * Db(bf0, bf3) * value;
}

inline void j_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                      int bf1, int bf2, int bf3, double value) {
    auto Ja = occ::qm::block::a(J);
    auto Jb = occ::qm::block::b(J);
    const auto Da = occ::qm::block::a(D);
    const auto Db = occ::qm::block::b(D);
    Ja(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
    Ja(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
    Jb(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
    Jb(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
}

inline void jk_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J,
                       Eigen::Ref<Mat> K, int bf0, int bf1, int bf2, int bf3,
                       double value) {
    auto Ja = occ::qm::block::a(J);
    auto Jb = occ::qm::block::b(J);
    auto Ka = occ::qm::block::a(K);
    auto Kb = occ::qm::block::b(K);
    const auto Da = occ::qm::block::a(D);
    const auto Db = occ::qm::block::b(D);
    Ja(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
    Ja(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
    Jb(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
    Jb(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;

    Ka(bf0, bf2) += 0.5 * Da(bf1, bf3) * value;
    Ka(bf1, bf3) += 0.5 * Da(bf0, bf2) * value;
    Ka(bf0, bf3) += 0.5 * Da(bf1, bf2) * value;
    Ka(bf1, bf2) += 0.5 * Da(bf0, bf3) * value;

    Kb(bf0, bf2) += 0.5 * Db(bf1, bf3) * value;
    Kb(bf1, bf3) += 0.5 * Db(bf0, bf2) * value;
    Kb(bf0, bf3) += 0.5 * Db(bf1, bf2) * value;
    Kb(bf1, bf2) += 0.5 * Db(bf0, bf3) * value;
}

inline void fock_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0,
                         int bf1, int bf2, int bf3, double value) {
    auto Faa = occ::qm::block::aa(F);
    auto Fab = occ::qm::block::ab(F);
    auto Fba = occ::qm::block::ba(F);
    auto Fbb = occ::qm::block::bb(F);
    const auto Daa = occ::qm::block::aa(D);
    const auto Dab = occ::qm::block::ab(D);
    const auto Dba = occ::qm::block::ba(D);
    const auto Dbb = occ::qm::block::bb(D);

    // J aa
    Faa(bf0, bf1) += 2 * Daa(bf2, bf3) * value;
    Faa(bf2, bf3) += 2 * Daa(bf0, bf1) * value;
    // J bb
    Fbb(bf0, bf1) += 2 * Dbb(bf2, bf3) * value;
    Fbb(bf2, bf3) += 2 * Dbb(bf0, bf1) * value;

    // K aa
    Faa(bf0, bf2) -= 0.5 * Daa(bf1, bf3) * value;
    Faa(bf1, bf3) -= 0.5 * Daa(bf0, bf2) * value;
    Faa(bf0, bf3) -= 0.5 * Daa(bf1, bf2) * value;
    Faa(bf1, bf2) -= 0.5 * Daa(bf0, bf3) * value;

    // K bb
    Fbb(bf0, bf2) -= 0.5 * Dbb(bf1, bf3) * value;
    Fbb(bf1, bf3) -= 0.5 * Dbb(bf0, bf2) * value;
    Fbb(bf0, bf3) -= 0.5 * Dbb(bf1, bf2) * value;
    Fbb(bf1, bf2) -= 0.5 * Dbb(bf0, bf3) * value;

    // Kab, Kba
    Fab(bf0, bf2) -= 0.5 * (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
    Fab(bf1, bf3) -= 0.5 * (Dab(bf0, bf2) + Dba(bf0, bf2)) * value;
    Fab(bf0, bf3) -= 0.5 * (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
    Fab(bf1, bf2) -= 0.5 * (Dab(bf0, bf3) + Dba(bf0, bf3)) * value;
    Fba(bf0, bf2) -= 0.5 * (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
    Fba(bf1, bf3) -= 0.5 * (Dab(bf0, bf2) + Dba(bf0, bf2)) * value;
    Fba(bf0, bf3) -= 0.5 * (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
    Fba(bf1, bf2) -= 0.5 * (Dab(bf0, bf3) + Dba(bf0, bf3)) * value;
}

inline void j_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                      int bf1, int bf2, int bf3, double value) {
    auto Jaa = occ::qm::block::aa(J);
    auto Jbb = occ::qm::block::bb(J);
    const auto Daa = occ::qm::block::aa(D);
    const auto Dbb = occ::qm::block::bb(D);

    // J aa
    Jaa(bf0, bf1) += 2 * Daa(bf2, bf3) * value;
    Jaa(bf2, bf3) += 2 * Daa(bf0, bf1) * value;
    // J bb
    Jbb(bf0, bf1) += 2 * Dbb(bf2, bf3) * value;
    Jbb(bf2, bf3) += 2 * Dbb(bf0, bf1) * value;
}

inline void jk_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J,
                       Eigen::Ref<Mat> K, int bf0, int bf1, int bf2, int bf3,
                       double value) {
    auto Jaa = occ::qm::block::aa(J);
    auto Jbb = occ::qm::block::bb(J);
    auto Kaa = occ::qm::block::aa(K);
    auto Kab = occ::qm::block::ab(K);
    auto Kba = occ::qm::block::ba(K);
    auto Kbb = occ::qm::block::bb(K);

    const auto Daa = occ::qm::block::aa(D);
    const auto Dab = occ::qm::block::ab(D);
    const auto Dba = occ::qm::block::ba(D);
    const auto Dbb = occ::qm::block::bb(D);

    // J aa
    Jaa(bf0, bf1) += 2 * Daa(bf2, bf3) * value;
    Jaa(bf2, bf3) += 2 * Daa(bf0, bf1) * value;
    // J bb
    Jbb(bf0, bf1) += 2 * Dbb(bf2, bf3) * value;
    Jbb(bf2, bf3) += 2 * Dbb(bf0, bf1) * value;

    // K aa
    Kaa(bf0, bf2) += 0.5 * Daa(bf1, bf3) * value;
    Kaa(bf1, bf3) += 0.5 * Daa(bf0, bf2) * value;
    Kaa(bf0, bf3) += 0.5 * Daa(bf1, bf2) * value;
    Kaa(bf1, bf2) += 0.5 * Daa(bf0, bf3) * value;

    // K bb
    Kbb(bf0, bf2) += 0.5 * Dbb(bf1, bf3) * value;
    Kbb(bf1, bf3) += 0.5 * Dbb(bf0, bf2) * value;
    Kbb(bf0, bf3) += 0.5 * Dbb(bf1, bf2) * value;
    Kbb(bf1, bf2) += 0.5 * Dbb(bf0, bf3) * value;

    // Kab, Kba
    Kab(bf0, bf2) += 0.5 * (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
    Kab(bf1, bf3) += 0.5 * (Dab(bf0, bf2) + Dba(bf0, bf2)) * value;
    Kab(bf0, bf3) += 0.5 * (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
    Kab(bf1, bf2) += 0.5 * (Dab(bf0, bf3) + Dba(bf0, bf3)) * value;
    Kba(bf0, bf2) += 0.5 * (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
    Kba(bf1, bf3) += 0.5 * (Dab(bf0, bf2) + Dba(bf0, bf2)) * value;
    Kba(bf0, bf3) += 0.5 * (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
    Kba(bf1, bf2) += 0.5 * (Dab(bf0, bf3) + Dba(bf0, bf3)) * value;
}

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

using ShellList = std::vector<Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;

template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center(Lambda &f, cint::IntegralEnvironment &env,
                         const AOBasis &basis, int thread_id = 0) {
    using Result = IntegralEngine::IntegralResult<2>;
    occ::qm::cint::Optimizer opt(env, op, 2);
    auto nthreads = occ::parallel::get_num_threads();
    auto bufsize = env.buffer_size_1e(op);
    const auto nsh = basis.size();

    auto buffer = std::make_unique<double[]>(bufsize);
    const auto &first_bf = basis.first_bf();
    for (int p = 0, pq = 0; p < nsh; p++) {
        int bf1 = first_bf[p];
        for (int q = 0; q <= p; q++) {
            if (pq++ % nthreads != thread_id)
                continue;
            int bf2 = first_bf[q];
            std::array<int, 2> idxs{p, q};
            Result args{thread_id,
                        idxs,
                        {bf1, bf2},
                        env.two_center_helper<op, kind>(
                            idxs, opt.optimizer_ptr(), buffer.get(), nullptr),
                        buffer.get()};
            if (args.dims[0] > -1)
                f(args);
        }
    }
}

template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center_with_shellpairs(Lambda &f,
                                         cint::IntegralEnvironment &env,
                                         const AOBasis &basis,
                                         const ShellPairList &shellpairs,
                                         int thread_id = 0) {
    using Result = IntegralEngine::IntegralResult<2>;
    occ::qm::cint::Optimizer opt(env, op, 2);
    auto nthreads = occ::parallel::get_num_threads();
    auto bufsize = env.buffer_size_1e(op);

    auto buffer = std::make_unique<double[]>(bufsize);
    const auto &first_bf = basis.first_bf();
    for (int p = 0, pq = 0; p < basis.size(); p++) {
        int bf1 = first_bf[p];
        for (const auto &q : shellpairs[p]) {
            if (pq++ % nthreads != thread_id)
                continue;
            int bf2 = first_bf[q];
            std::array<int, 2> idxs{p, static_cast<int>(q)};
            Result args{thread_id,
                        idxs,
                        {bf1, bf2},
                        env.two_center_helper<op, kind>(
                            idxs, opt.optimizer_ptr(), buffer.get(), nullptr),
                        buffer.get()};
            if (args.dims[0] > -1)
                f(args);
        }
    }
}

template <Op op, ShellKind kind = ShellKind::Cartesian>
Mat one_electron_operator_kernel(const AOBasis &basis,
                                 cint::IntegralEnvironment &env,
                                 const ShellPairList &shellpairs) {
    using Result = IntegralEngine::IntegralResult<2>;
    auto nthreads = occ::parallel::get_num_threads();
    const auto nbf = basis.nbf();
    Mat result = Mat::Zero(nbf, nbf);
    std::vector<Mat> results;
    results.emplace_back(Mat::Zero(nbf, nbf));
    for (size_t i = 1; i < nthreads; i++) {
        results.push_back(results[0]);
    }
    auto f = [&results](const Result &args) {
        auto &result = results[args.thread];
        Eigen::Map<const occ::Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
        result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmp;
        if (args.shell[0] != args.shell[1]) {
            result.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) =
                tmp.transpose();
        }
    };

    auto lambda = [&](int thread_id) {
        if (shellpairs.size() > 0) {
            evaluate_two_center_with_shellpairs<op, kind>(
                f, env, basis, shellpairs, thread_id);
        } else {
            evaluate_two_center<op, kind>(f, env, basis, thread_id);
        }
    };
    occ::parallel::parallel_do(lambda);

    for (auto i = 1; i < nthreads; ++i) {
        results[0].noalias() += results[i];
    }
    return results[0];
}

Mat IntegralEngine::one_electron_operator(Op op,
                                          bool use_shellpair_list) const {
    bool spherical = is_spherical();
    constexpr auto Cart = ShellKind::Cartesian;
    constexpr auto Sph = ShellKind::Spherical;
    ShellPairList empty_shellpairs = {};
    const auto &shellpairs =
        use_shellpair_list ? m_shellpairs : empty_shellpairs;
    switch (op) {
    case Op::overlap: {
        if (spherical) {
            return one_electron_operator_kernel<Op::overlap, Sph>(
                m_aobasis, m_env, shellpairs);
        } else {
            return one_electron_operator_kernel<Op::overlap, Cart>(
                m_aobasis, m_env, shellpairs);
        }
        break;
    }
    case Op::nuclear: {
        if (spherical) {
            return one_electron_operator_kernel<Op::nuclear, Sph>(
                m_aobasis, m_env, shellpairs);
        } else {
            return one_electron_operator_kernel<Op::nuclear, Cart>(
                m_aobasis, m_env, shellpairs);
        }
        break;
    }
    case Op::kinetic: {
        if (spherical) {
            return one_electron_operator_kernel<Op::kinetic, Sph>(
                m_aobasis, m_env, shellpairs);
        } else {
            return one_electron_operator_kernel<Op::kinetic, Cart>(
                m_aobasis, m_env, shellpairs);
        }
        break;
    }
    case Op::coulomb: {
        if (spherical) {
            return one_electron_operator_kernel<Op::coulomb, Sph>(
                m_aobasis, m_env, shellpairs);
        } else {
            return one_electron_operator_kernel<Op::coulomb, Cart>(
                m_aobasis, m_env, shellpairs);
        }
        break;
    }
    default:
        throw std::runtime_error("Invalid operator for two-center integral");
        break;
    }
}


Mat IntegralEngine::rinv_operator_atom_center(size_t atom_index, bool use_shellpair_list) const {
    const auto &atoms = m_aobasis.atoms();
    if(atom_index > atoms.size()) throw std::runtime_error("Invalid atom index for rinv operator");

    bool spherical = is_spherical();
    constexpr auto Cart = ShellKind::Cartesian;
    constexpr auto Sph = ShellKind::Spherical;
    ShellPairList empty_shellpairs = {};
    const auto &shellpairs =
        use_shellpair_list ? m_shellpairs : empty_shellpairs;

    std::array<double, 3> origin{atoms[atom_index].x, atoms[atom_index].y, atoms[atom_index].z};
    m_env.set_rinv_origin(origin);
    Mat result;
    if(spherical) {
	result = one_electron_operator_kernel<Op::rinv, Sph>(m_aobasis, m_env, shellpairs);
    }
    else {
	result = one_electron_operator_kernel<Op::rinv, Cart>(m_aobasis, m_env, shellpairs);
    }
    m_env.set_rinv_origin({0.0, 0.0, 0.0});
    return result;
}

#if HAVE_ECPINT
template <typename Lambda>
void evaluate_two_center_ecp_with_shellpairs(
    Lambda &f, const std::vector<libecpint::GaussianShell> &shells,
    const std::vector<libecpint::ECP> &ecps, int lmax1, int lmax2,
    const ShellPairList &shellpairs, int thread_id = 0) {
    // TODO maybe share this? not sure if it's thread safe.
    libecpint::ECPIntegral ecp_integral(lmax1, lmax2, 0);
    using Result = IntegralEngine::IntegralResult<2>;
    auto nthreads = occ::parallel::get_num_threads();

    std::array<int, 2> dims;
    std::array<int, 2> bfs{-1, -1}; // ignore this

    libecpint::TwoIndex<double> tmp;
    for (int p = 0, pq = 0; p < shells.size(); p++) {
        const auto &sh1 = shells[p];

        dims[0] = sh1.ncartesian();

        for (const auto &q : shellpairs[p]) {
            if (pq++ % nthreads != thread_id)
                continue;
            const auto &sh2 = shells[q];
            dims[1] = sh2.ncartesian();
            libecpint::TwoIndex<double> buffer(dims[0], dims[1], 0.0);

            std::array<int, 2> idxs{p, static_cast<int>(q)};
            for (const auto &U : ecps) {
                ecp_integral.compute_shell_pair(U, sh1, sh2, tmp);
                buffer.add(tmp);
            }
            Result args{thread_id, idxs, bfs, dims, buffer.data.data()};
            f(args);
        }
    }
}

template <ShellKind kind = ShellKind::Cartesian>
Mat ecp_operator_kernel(const AOBasis &aobasis,
                        const std::vector<libecpint::GaussianShell> &aoshells,
                        const std::vector<libecpint::ECP> &ecps, int ao_max_l,
                        int ecp_max_l, const ShellPairList &shellpairs) {
    using Result = IntegralEngine::IntegralResult<2>;
    auto nthreads = occ::parallel::get_num_threads();

    std::vector<Mat> results;
    results.emplace_back(Mat::Zero(aobasis.nbf(), aobasis.nbf()));
    for (size_t i = 1; i < nthreads; i++) {
        results.push_back(results[0]);
    }

    if constexpr (kind == ShellKind::Spherical) {
	std::vector<Mat> cart2sph;
	for (int i = 0; i <= ao_max_l; i++) {
	    cart2sph.push_back(
		occ::gto::cartesian_to_spherical_transformation_matrix(i));
	}

	auto f = [&results, &aobasis, &cart2sph](const Result &args) {
	    auto &result = results[args.thread];
	    Eigen::Map<const occ::MatRM> tmp(args.buffer, args.dims[0],
					     args.dims[1]);
	    const int bf0 = aobasis.first_bf()[args.shell[0]];
	    const int bf1 = aobasis.first_bf()[args.shell[1]];
	    const int dim0 = aobasis[args.shell[0]].size();
	    const int dim1 = aobasis[args.shell[1]].size();
	    const int l0 = aobasis[args.shell[0]].l;
	    const int l1 = aobasis[args.shell[1]].l;
	    result.block(bf0, bf1, dim0, dim1) =
		cart2sph[l0] * tmp * cart2sph[l1].transpose();
	    if (args.shell[0] != args.shell[1]) {
		result.block(bf1, bf0, dim1, dim0) =
		    result.block(bf0, bf1, dim0, dim1).transpose();
	    }
	};
	auto lambda = [&](int thread_id) {
	    evaluate_two_center_ecp_with_shellpairs(
		f, aoshells, ecps, ao_max_l, ecp_max_l, shellpairs, thread_id);
	};
	occ::parallel::parallel_do(lambda);

    } else {
	auto f = [&results, &aobasis](const Result &args) {
	    auto &result = results[args.thread];
	    Eigen::Map<const occ::MatRM> tmp(args.buffer, args.dims[0],
					     args.dims[1]);
	    const int bf0 = aobasis.first_bf()[args.shell[0]];
	    const int bf1 = aobasis.first_bf()[args.shell[1]];

	    result.block(bf0, bf1, args.dims[0], args.dims[1]) = tmp;
	    if (args.shell[0] != args.shell[1]) {
		result.block(bf1, bf0, args.dims[1], args.dims[0]) =
		    tmp.transpose();
	    }
	};
	auto lambda = [&](int thread_id) {
	    evaluate_two_center_ecp_with_shellpairs(
		f, aoshells, ecps, ao_max_l, ecp_max_l, shellpairs, thread_id);
	};
	occ::parallel::parallel_do(lambda);

    };


    for (auto i = 1; i < nthreads; ++i) {
        results[0].noalias() += results[i];
    }
    return results[0];
}


Mat IntegralEngine::effective_core_potential(bool use_shellpair_list) const {
    if (!have_effective_core_potentials())
        throw std::runtime_error(
            "Called effective_core_potential without any ECPs");

    occ::timing::start(occ::timing::category::ecp);
    bool spherical = is_spherical();
    constexpr auto Cart = ShellKind::Cartesian;
    constexpr auto Sph = ShellKind::Spherical;
    ShellPairList empty_shellpairs = {};
    const auto &shellpairs =
        use_shellpair_list ? m_shellpairs : empty_shellpairs;
    Mat result;
    if (spherical) {
        result =
            ecp_operator_kernel<Sph>(m_aobasis, m_ecp_gaussian_shells, m_ecp,
                                     m_ecp_ao_max_l, m_ecp_max_l, shellpairs);
    } else {
        result =
            ecp_operator_kernel<Cart>(m_aobasis, m_ecp_gaussian_shells, m_ecp,
                                      m_ecp_ao_max_l, m_ecp_max_l, shellpairs);
    }
    occ::timing::stop(occ::timing::category::ecp);
    return result;
}

void IntegralEngine::set_effective_core_potentials(
    const ShellList &ecp_shells, const std::vector<int> &ecp_electrons) {
    const auto &atoms = m_aobasis.atoms();
    for (const auto &sh : m_aobasis.shells()) {
        libecpint::GaussianShell ecpint_shell(
            {sh.origin(0), sh.origin(1), sh.origin(2)}, sh.l);
        const Mat &coeffs_norm = sh.contraction_coefficients;
        m_ecp_ao_max_l = std::max(static_cast<int>(sh.l), m_ecp_ao_max_l);
        for (int i = 0; i < sh.num_primitives(); i++) {
            double c = coeffs_norm(i, 0);
            if (sh.l == 0) {
                c *= 0.28209479177387814; // 1 / (2 * sqrt(pi))
            }
            if (sh.l == 1) {
                c *= 0.4886025119029199; // sqrt(3) / (2 * sqrt(pi))
            }
            ecpint_shell.addPrim(sh.exponents(i), c);
        }
        m_ecp_gaussian_shells.push_back(ecpint_shell);
    }
    for (int i = 0; i < ecp_electrons.size(); i++) {
        int charge = atoms[i].atomic_number - ecp_electrons[i];
        occ::log::debug("setting atom {} charge to {}", i, charge);
        m_env.set_atom_charge(i, charge);
    }

    Vec3 pt = ecp_shells[0].origin;
    // For some reason need to merge all shells that share a center.
    // This code relies on shells with the same center being grouped.
    //
    libecpint::ECP ecp(pt.data());
    for (const auto &sh : ecp_shells) {
        if ((pt - sh.origin).norm() > 1e-3) {
            ecp.sort();
            ecp.atom_id = m_ecp.size();
            m_ecp.push_back(ecp);
            pt = sh.origin;
            ecp = libecpint::ECP(pt.data());
        }
        for (int i = 0; i < sh.num_primitives(); i++) {
            m_ecp_max_l = std::max(static_cast<int>(sh.l), m_ecp_max_l);
            ecp.addPrimitive(sh.ecp_r_exponents(i), sh.l, sh.exponents(i),
                             sh.contraction_coefficients(i, 0), false);
        }
    }

    // add the last ECP to the end
    ecp.sort();
    ecp.atom_id = m_ecp.size();
    m_ecp.push_back(ecp);

    m_have_ecp = true;
}
#else

Mat IntegralEngine::effective_core_potential(bool use_shellpair_list) const {
  throw std::runtime_error("Not compiled with libecpint");
}
void IntegralEngine::set_effective_core_potentials(
    const ShellList &ecp_shells, const std::vector<int> &ecp_electrons) {
  throw std::runtime_error("Not compiled with libecpint");
}

#endif

template <int order, SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Vec multipole_kernel(const AOBasis &basis, cint::IntegralEnvironment &env,
                     const ShellPairList &shellpairs,
                     const MolecularOrbitals &mo, const Vec3 &origin) {
    using Result = IntegralEngine::IntegralResult<2>;
    constexpr std::array<Op, 5> ops{Op::overlap, Op::dipole, Op::quadrupole,
                                    Op::octapole, Op::hexadecapole};
    constexpr Op op = ops[order];

    auto nthreads = occ::parallel::get_num_threads();
    size_t num_components = occ::core::num_multipole_components_tensor(order);
    env.set_common_origin({origin.x(), origin.y(), origin.z()});
    std::vector<Vec> results(nthreads, Vec::Zero(num_components));
    const auto &D = mo.D;
    /*
     * For symmetric matrices
     * the of a matrix product tr(D @ O) is equal to
     * the sum of the elementwise product with the transpose:
     * tr(D @ O) == sum(D * O^T)
     * since expectation is -2 tr(D @ O) we factor that into the
     * inner loop
     */
    auto f = [&D, &results, &num_components](const Result &args) {
        auto &result = results[args.thread];
        size_t offset = 0;
        double scale = (args.shell[0] != args.shell[1]) ? 2.0 : 1.0;
        // TODO avoid redundant tensor calcs
        for (size_t n = 0; n < num_components; n++) {
            Eigen::Map<const occ::Mat> tmp(args.buffer + offset, args.dims[0],
                                           args.dims[1]);
            if constexpr (sk == SpinorbitalKind::Restricted) {
                result(n) += scale * (D.block(args.bf[0], args.bf[1],
                                              args.dims[0], args.dims[1])
                                          .array() *
                                      tmp.array())
                                         .sum();
            } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
                const auto Da = qm::block::a(D);
                const auto Db = qm::block::b(D);
                result(n) += scale * (Da.block(args.bf[0], args.bf[1],
                                               args.dims[0], args.dims[1])
                                          .array() *
                                      tmp.array())
                                         .sum();
                result(n) += scale * (Db.block(args.bf[0], args.bf[1],
                                               args.dims[0], args.dims[1])
                                          .array() *
                                      tmp.array())
                                         .sum();
            } else if constexpr (sk == SpinorbitalKind::General) {
                const auto Daa = qm::block::aa(D);
                const auto Dab = qm::block::ab(D);
                const auto Dba = qm::block::ba(D);
                const auto Dbb = qm::block::bb(D);
                result(n) += scale * (Daa.block(args.bf[0], args.bf[1],
                                                args.dims[0], args.dims[1])
                                          .array() *
                                      tmp.array())
                                         .sum();
                result(n) += scale * (Dab.block(args.bf[0], args.bf[1],
                                                args.dims[0], args.dims[1])
                                          .array() *
                                      tmp.array())
                                         .sum();
                result(n) += scale * (Dba.block(args.bf[0], args.bf[1],
                                                args.dims[0], args.dims[1])
                                          .array() *
                                      tmp.array())
                                         .sum();
                result(n) += scale * (Dbb.block(args.bf[0], args.bf[1],
                                                args.dims[0], args.dims[1])
                                          .array() *
                                      tmp.array())
                                         .sum();
            }
            offset += tmp.size();
        }
    };

    auto lambda = [&](int thread_id) {
        evaluate_two_center_with_shellpairs<op, kind>(f, env, basis, shellpairs,
                                                      thread_id);
    };
    occ::parallel::parallel_do(lambda);

    for (auto i = 1; i < nthreads; ++i) {
        results[0].noalias() += results[i];
    }

    results[0] *= -2;
    // TODO refactor this
    Vec unique(occ::core::num_unique_multipole_components(order));
    if constexpr (order <= 1) {
        return results[0];
    } else if constexpr (order == 2) {
        const auto &Q = results[0];
        unique(0) = Q(0); // xx
        unique(1) = Q(1); // xy
        unique(2) = Q(2); // xz
        unique(3) = Q(4); // yy
        unique(4) = Q(5); // yz
        unique(5) = Q(8); // zz
    } else if constexpr (order == 3) {
        const auto &O = results[0];
        unique(0) = O(0);  // xxx
        unique(1) = O(1);  // xxy
        unique(2) = O(2);  // xxz
        unique(3) = O(4);  // xyy
        unique(4) = O(5);  // xyz
        unique(5) = O(8);  // xzz
        unique(6) = O(13); // yyy
        unique(7) = O(14); // yyz
        unique(8) = O(17); // yzz
        unique(9) = O(26); // zzz
    } else if constexpr (order == 4) {
        const auto &H = results[0];
        unique(0) = H(0);   // xxxx
        unique(1) = H(1);   // xxxy
        unique(2) = H(2);   // xxxz
        unique(3) = H(4);   // xxyy
        unique(4) = H(5);   // xxyz
        unique(5) = H(8);   // xxzz
        unique(6) = H(13);  // xyyy
        unique(7) = H(14);  // xyyz
        unique(8) = H(17);  // xyzz
        unique(9) = H(26);  // xzzz
        unique(10) = H(40); // yyyy
        unique(11) = H(41); // yyyz
        unique(12) = H(44); // yyzz
        unique(13) = H(53); // yzzz
        unique(14) = H(80); // zzzz
    }
    return unique;
}

Vec IntegralEngine::multipole(int order, const MolecularOrbitals &mo,
                              const Vec3 &origin) const {
    bool spherical = is_spherical();
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Cart = ShellKind::Cartesian;
    constexpr auto Sph = ShellKind::Spherical;
    if (mo.kind == R) {
        switch (order) {
        case 0:
            if (spherical) {
                return multipole_kernel<0, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<0, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 1:
            if (spherical) {
                return multipole_kernel<1, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<1, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 2:
            if (spherical) {
                return multipole_kernel<2, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<2, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 3:
            if (spherical) {
                return multipole_kernel<3, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<3, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 4:
            if (spherical) {
                return multipole_kernel<4, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<4, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        default:
            throw std::runtime_error("Invalid multipole order");
            break;
        }
    } else if (mo.kind == U) {
        switch (order) {
        case 0:
            if (spherical) {
                return multipole_kernel<0, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<0, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 1:
            if (spherical) {
                return multipole_kernel<1, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<1, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 2:
            if (spherical) {
                return multipole_kernel<2, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<2, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 3:
            if (spherical) {
                return multipole_kernel<3, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<3, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 4:
            if (spherical) {
                return multipole_kernel<4, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<4, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        default:
            throw std::runtime_error("Invalid multipole order");
            break;
        }
    } else { // if (sk == G)
        switch (order) {
        case 0:
            if (spherical) {
                return multipole_kernel<0, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<0, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 1:
            if (spherical) {
                return multipole_kernel<1, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<1, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 2:
            if (spherical) {
                return multipole_kernel<2, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<2, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 3:
            if (spherical) {
                return multipole_kernel<3, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<3, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        case 4:
            if (spherical) {
                return multipole_kernel<4, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
            } else {
                return multipole_kernel<4, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
            }
            break;
        default:
            throw std::runtime_error("Invalid multipole order");
            break;
        }
    }
}

// helper functions to make chained calls to std::max more clear
inline double max_of(double p, double q, double r) {
    return std::max(p, std::max(q, r));
}

inline double max_of(double p, double q, double r, double s) {
    return std::max(p, std::max(q, std::max(r, s)));
}

template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center(Lambda &f, cint::IntegralEnvironment &env,
                          const AOBasis &basis, const ShellPairList &shellpairs,
                          const Mat &Dnorm = Mat(), const Mat &Schwarz = Mat(),
                          double precision = 1e-12, int thread_id = 0) {
    using Result = IntegralEngine::IntegralResult<4>;
    occ::timing::start(occ::timing::category::ints4c2e);
    auto nthreads = occ::parallel::get_num_threads();
    occ::qm::cint::Optimizer opt(env, Op::coulomb, 4);
    auto buffer = std::make_unique<double[]>(env.buffer_size_2e());
    std::array<int, 4> shell_idx;
    std::array<int, 4> bf;

    const auto &first_bf = basis.first_bf();
    const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
    // <pq|rs>
    for (size_t p = 0, pqrs = 0; p < basis.size(); p++) {
        bf[0] = first_bf[p];
        const auto &plist = shellpairs[p];
        for (const auto q : plist) {
            bf[1] = first_bf[q];

            // for Schwarz screening
            const double norm_pq = do_schwarz_screen ? Dnorm(p, q) : 0.0;

            for (size_t r = 0; r <= p; r++) {
                bf[2] = first_bf[r];
                // check if <pq|ps>, if so ensure s <= q else s <= r
                const auto s_max = (p == r) ? q : r;

                const double norm_pqr =
                    do_schwarz_screen
                        ? max_of(Dnorm(p, r), Dnorm(q, r), norm_pq)
                        : 0.0;

                for (const auto s : shellpairs[r]) {
                    if (s > s_max)
                        break;
                    if (pqrs++ % nthreads != thread_id)
                        continue;
                    const double norm_pqrs =
                        do_schwarz_screen ? max_of(Dnorm(p, s), Dnorm(q, s),
                                                   Dnorm(r, s), norm_pqr)
                                          : 0.0;
                    if (do_schwarz_screen &&
                        norm_pqrs * Schwarz(p, q) * Schwarz(r, s) < precision)
                        continue;

                    bf[3] = first_bf[s];
                    shell_idx = {static_cast<int>(p), static_cast<int>(q),
                                 static_cast<int>(r), static_cast<int>(s)};

                    Result args{thread_id, shell_idx, bf,
                                env.four_center_helper<Op::coulomb, kind>(
                                    shell_idx, opt.optimizer_ptr(),
                                    buffer.get(), nullptr),
                                buffer.get()};
                    if (args.dims[0] > -1)
                        f(args);
                }
            }
        }
    }
    occ::timing::stop(occ::timing::category::ints4c2e);
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Mat fock_operator_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                         const ShellPairList &shellpairs,
                         const MolecularOrbitals &mo, double precision = 1e-12,
                         const Mat &Schwarz = Mat()) {
    using Result = IntegralEngine::IntegralResult<4>;
    auto nthreads = occ::parallel::get_num_threads();
    constexpr Op op = Op::coulomb;
    std::vector<Mat> Fmats(nthreads, Mat::Zero(mo.D.rows(), mo.D.cols()));
    Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

    const auto &D = mo.D;
    auto f = [&D, &Fmats](const Result &args) {
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
        evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm,
                                       Schwarz, precision, thread_id);
    };
    occ::timing::start(occ::timing::category::fock);
    occ::parallel::parallel_do(lambda);
    occ::timing::stop(occ::timing::category::fock);

    Mat F = Mat::Zero(Fmats[0].rows(), Fmats[0].cols());

    for (const auto &part : Fmats) {
        impl::accumulate_operator_symmetric<sk>(part, F);
    }
    F *= 0.5;

    return F;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Mat coulomb_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                   const ShellPairList &shellpairs, const MolecularOrbitals &mo,
                   double precision = 1e-12, const Mat &Schwarz = Mat()) {
    using Result = IntegralEngine::IntegralResult<4>;
    auto nthreads = occ::parallel::get_num_threads();
    constexpr Op op = Op::coulomb;
    std::vector<Mat> Jmats(nthreads, Mat::Zero(mo.D.rows(), mo.D.cols()));
    Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

    const auto &D = mo.D;
    auto f = [&D, &Jmats](const Result &args) {
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
                        impl::delegate_j<sk>(D, J, bf0, bf1, bf2, bf3, value);
                    }
                }
            }
        }
    };
    auto lambda = [&](int thread_id) {
        evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm,
                                       Schwarz, precision, thread_id);
    };
    occ::timing::start(occ::timing::category::fock);
    occ::parallel::parallel_do(lambda);
    occ::timing::stop(occ::timing::category::fock);

    Mat J = Mat::Zero(Jmats[0].rows(), Jmats[0].cols());

    for (size_t i = 0; i < nthreads; i++) {
        impl::accumulate_operator_symmetric<sk>(Jmats[i], J);
    }
    J *= 0.5;
    return J;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
JKPair coulomb_and_exchange_kernel(cint::IntegralEnvironment &env,
                                   const AOBasis &basis,
                                   const ShellPairList &shellpairs,
                                   const MolecularOrbitals &mo,
                                   double precision = 1e-12,
                                   const Mat &Schwarz = Mat()) {
    using Result = IntegralEngine::IntegralResult<4>;
    auto nthreads = occ::parallel::get_num_threads();
    constexpr Op op = Op::coulomb;
    std::vector<Mat> Jmats(nthreads, Mat::Zero(mo.D.rows(), mo.D.cols()));
    std::vector<Mat> Kmats(nthreads, Mat::Zero(mo.D.rows(), mo.D.cols()));
    Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

    const auto &D = mo.D;
    auto f = [&D, &Jmats, &Kmats](const Result &args) {
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
        evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm,
                                       Schwarz, precision, thread_id);
    };
    occ::timing::start(occ::timing::category::fock);
    occ::parallel::parallel_do(lambda);
    occ::timing::stop(occ::timing::category::fock);

    JKPair result{Mat::Zero(Jmats[0].rows(), Jmats[0].cols()),
                  Mat::Zero(Kmats[0].rows(), Kmats[0].cols())};

    for (size_t i = 0; i < nthreads; i++) {
        impl::accumulate_operator_symmetric<sk>(Jmats[i], result.J);
        impl::accumulate_operator_symmetric<sk>(Kmats[i], result.K);
    }
    result.J *= 0.5;
    result.K *= 0.5;
    return result;
}

Mat IntegralEngine::fock_operator(SpinorbitalKind sk,
                                  const MolecularOrbitals &mo,
                                  const Mat &Schwarz) const {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    bool spherical = is_spherical();
    switch (sk) {
    default:
    case R:
        if (spherical) {
            return fock_operator_kernel<R, Sph>(m_env, m_aobasis, m_shellpairs,
                                                mo, m_precision, Schwarz);
        } else {
            return fock_operator_kernel<R, Cart>(m_env, m_aobasis, m_shellpairs,
                                                 mo, m_precision, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return fock_operator_kernel<U, Sph>(m_env, m_aobasis, m_shellpairs,
                                                mo, m_precision, Schwarz);
        } else {
            return fock_operator_kernel<U, Cart>(m_env, m_aobasis, m_shellpairs,
                                                 mo, m_precision, Schwarz);
        }

    case G:
        if (spherical) {
            return fock_operator_kernel<G, Sph>(m_env, m_aobasis, m_shellpairs,
                                                mo, m_precision, Schwarz);
        } else {
            return fock_operator_kernel<G, Cart>(m_env, m_aobasis, m_shellpairs,
                                                 mo, m_precision, Schwarz);
        }
    }
}

Mat IntegralEngine::coulomb(SpinorbitalKind sk, const MolecularOrbitals &mo,
                            const Mat &Schwarz) const {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    bool spherical = is_spherical();
    switch (sk) {
    default:
    case R:
        if (spherical) {
            return coulomb_kernel<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                          m_precision, Schwarz);
        } else {
            return coulomb_kernel<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                           m_precision, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return coulomb_kernel<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                          m_precision, Schwarz);
        } else {
            return coulomb_kernel<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                           m_precision, Schwarz);
        }

    case G:
        if (spherical) {
            return coulomb_kernel<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                          m_precision, Schwarz);
        } else {
            return coulomb_kernel<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                           m_precision, Schwarz);
        }
    }
}

JKPair IntegralEngine::coulomb_and_exchange(SpinorbitalKind sk,
                                            const MolecularOrbitals &mo,
                                            const Mat &Schwarz) const {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    bool spherical = is_spherical();
    switch (sk) {
    default:
    case R:
        if (spherical) {
            return coulomb_and_exchange_kernel<R, Sph>(
                m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
        } else {
            return coulomb_and_exchange_kernel<R, Cart>(
                m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return coulomb_and_exchange_kernel<U, Sph>(
                m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
        } else {
            return coulomb_and_exchange_kernel<U, Cart>(
                m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
        }

    case G:
        if (spherical) {
            return coulomb_and_exchange_kernel<G, Sph>(
                m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
        } else {
            return coulomb_and_exchange_kernel<G, Cart>(
                m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
        }
    }
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
std::vector<Mat> coulomb_kernel_list(
    cint::IntegralEnvironment &env, const AOBasis &basis,
    const ShellPairList &shellpairs, const std::vector<MolecularOrbitals> &mos,
    double precision = 1e-12, const Mat &Schwarz = Mat()) {

    using Result = IntegralEngine::IntegralResult<4>;
    auto nthreads = occ::parallel::get_num_threads();
    constexpr Op op = Op::coulomb;

    const int rows = mos[0].D.rows();
    const int cols = mos[0].D.cols();

    std::vector<std::vector<Mat>> js(mos.size(), std::vector<Mat>(nthreads, Mat::Zero(rows, cols)));

    Mat Dnorm = shellblock_norm<sk, kind>(basis, mos[0].D);

    auto f = [&mos, &js](const Result &args) {
        auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
        auto pr_qs_degree = (args.shell[0] == args.shell[2])
                                ? (args.shell[1] == args.shell[3] ? 1 : 2)
                                : 2;
        auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
        auto scale = pq_degree * rs_degree * pr_qs_degree;

        for (int mo_index = 0; mo_index < mos.size(); mo_index++) {
            const auto &D = mos[mo_index].D;
            auto &J = js[mo_index][args.thread];

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
        }
    };
    auto lambda = [&](int thread_id) {
        evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm,
                                       Schwarz, precision, thread_id);
    };
    occ::timing::start(occ::timing::category::fock);
    occ::parallel::parallel_do(lambda);
    occ::timing::stop(occ::timing::category::fock);

    std::vector<Mat> results;
    for (size_t mo_index = 0; mo_index < mos.size(); mo_index++) {
        Mat result = Mat::Zero(rows, cols);
        const auto mo_jk = js[mo_index];
        for (size_t i = 0; i < nthreads; i++) {
            const auto &J = mo_jk[i];
            impl::accumulate_operator_symmetric<sk>(J, result);
        }
        result *= 0.5;
        results.push_back(result);
    }
    return results;
}

std::vector<Mat> IntegralEngine::coulomb_list(
    SpinorbitalKind sk, const std::vector<MolecularOrbitals> &mos,
    const Mat &Schwarz) const {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    bool spherical = is_spherical();
    switch (sk) {
    default:
    case R:
        if (spherical) {
            return coulomb_kernel_list<R, Sph>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        } else {
            return coulomb_kernel_list<R, Cart>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return coulomb_kernel_list<U, Sph>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        } else {
            return coulomb_kernel_list<U, Cart>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        }

    case G:
        if (spherical) {
            return coulomb_kernel_list<G, Sph>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        } else {
            return coulomb_kernel_list<G, Cart>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        }
    }
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
std::vector<JKPair> coulomb_and_exchange_kernel_list(
    cint::IntegralEnvironment &env, const AOBasis &basis,
    const ShellPairList &shellpairs, const std::vector<MolecularOrbitals> &mos,
    double precision = 1e-12, const Mat &Schwarz = Mat()) {

    using Result = IntegralEngine::IntegralResult<4>;
    auto nthreads = occ::parallel::get_num_threads();
    constexpr Op op = Op::coulomb;

    const int rows = mos[0].D.rows();
    const int cols = mos[0].D.cols();

    std::vector<std::vector<JKPair>> jkpairs(
	    mos.size(), 
	    std::vector<JKPair>(nthreads, JKPair{Mat::Zero(rows, cols), Mat::Zero(rows, cols)})
    );

    Mat Dnorm = shellblock_norm<sk, kind>(basis, mos[0].D);

    auto f = [&mos, &jkpairs](const Result &args) {
        auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
        auto pr_qs_degree = (args.shell[0] == args.shell[2])
                                ? (args.shell[1] == args.shell[3] ? 1 : 2)
                                : 2;
        auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
        auto scale = pq_degree * rs_degree * pr_qs_degree;

        for (int mo_index = 0; mo_index < mos.size(); mo_index++) {
            const auto &D = mos[mo_index].D;
            auto &J = jkpairs[mo_index][args.thread].J;
            auto &K = jkpairs[mo_index][args.thread].K;

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
        }
    };
    auto lambda = [&](int thread_id) {
        evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm,
                                       Schwarz, precision, thread_id);
    };
    occ::timing::start(occ::timing::category::fock);
    occ::parallel::parallel_do(lambda);
    occ::timing::stop(occ::timing::category::fock);

    std::vector<JKPair> results;
    for (size_t mo_index = 0; mo_index < mos.size(); mo_index++) {
        JKPair result{Mat::Zero(rows, cols), Mat::Zero(rows, cols)};
        const auto mo_jk = jkpairs[mo_index];
        for (size_t i = 0; i < nthreads; i++) {
            const auto &J = mo_jk[i].J;
            const auto &K = mo_jk[i].K;
            impl::accumulate_operator_symmetric<sk>(J, result.J);
            impl::accumulate_operator_symmetric<sk>(K, result.K);
        }
        result.J *= 0.5;
        result.K *= 0.5;
        results.push_back(result);
    }
    return results;
}

std::vector<JKPair> IntegralEngine::coulomb_and_exchange_list(
    SpinorbitalKind sk, const std::vector<MolecularOrbitals> &mos,
    const Mat &Schwarz) const {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    bool spherical = is_spherical();
    switch (sk) {
    default:
    case R:
        if (spherical) {
            return coulomb_and_exchange_kernel_list<R, Sph>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        } else {
            return coulomb_and_exchange_kernel_list<R, Cart>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return coulomb_and_exchange_kernel_list<U, Sph>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        } else {
            return coulomb_and_exchange_kernel_list<U, Cart>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        }

    case G:
        if (spherical) {
            return coulomb_and_exchange_kernel_list<G, Sph>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        } else {
            return coulomb_and_exchange_kernel_list<G, Cart>(
                m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
        }
    }
}

Mat IntegralEngine::fock_operator_mixed_basis(const Mat &D, const AOBasis &D_bs,
                                              bool is_shell_diagonal) {
    set_auxiliary_basis(D_bs.shells(), false);
    constexpr Op op = Op::coulomb;
    auto nthreads = occ::parallel::get_num_threads();

    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    bool spherical = is_spherical();
    const int nbf = m_aobasis.nbf();
    const int nsh = m_aobasis.size();
    const int nbf_aux = m_auxbasis.nbf();
    const int nsh_aux = m_auxbasis.size();
    assert(D.cols() == D.rows() && D.cols() == nbf_aux);

    std::vector<Mat> Fmats(nthreads, Mat::Zero(nbf, nbf));

    // construct the 2-electron repulsion integrals engine
    auto shell2bf = m_aobasis.first_bf();
    auto shell2bf_D = m_auxbasis.first_bf();

    auto lambda = [&](int thread_id) {
        auto &F = Fmats[thread_id];
        occ::qm::cint::Optimizer opt(m_env, Op::coulomb, 4);
        auto buffer = std::make_unique<double[]>(m_env.buffer_size_2e());

        // loop over permutationally-unique set of shells
        for (int s1 = 0, s1234 = 0; s1 != nsh; ++s1) {
            int bf1_first = shell2bf[s1]; // first basis function in this shell
            int n1 =
                m_aobasis[s1].size(); // number of basis functions in this shell

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

                        // compute the permutational degeneracy (i.e. #
                        // of equivalents) of the given shell set
                        double s12_deg = (s1 == s2) ? 1.0 : 2.0;

                        std::array<int, 4> dims;
                        if (s3 >= s4) {
                            double s34_deg = (s3 == s4) ? 1.0 : 2.0;
                            double s1234_deg = s12_deg * s34_deg;
                            // auto s1234_deg = s12_deg;
                            std::array<int, 4> idxs{s1, s2, s3 + nsh, s4 + nsh};
                            if (spherical) {
                                dims = m_env.four_center_helper<op, Sph>(
                                    idxs, opt.optimizer_ptr(), buffer.get(),
                                    nullptr);
                            } else {
                                dims = m_env.four_center_helper<op, Cart>(
                                    idxs, opt.optimizer_ptr(), buffer.get(),
                                    nullptr);
                            }

                            if (dims[0] >= 0) {
                                const auto *buf_1234 = buffer.get();
                                for (auto f4 = 0, f1234 = 0; f4 != n4; ++f4) {
                                    const auto bf4 = f4 + bf4_first;
                                    for (auto f3 = 0; f3 != n3; ++f3) {
                                        const auto bf3 = f3 + bf3_first;
                                        for (auto f2 = 0; f2 != n2; ++f2) {
                                            const auto bf2 = f2 + bf2_first;
                                            for (auto f1 = 0; f1 != n1;
                                                 ++f1, ++f1234) {
                                                const auto bf1 = f1 + bf1_first;

                                                const auto value =
                                                    buf_1234[f1234];
                                                const auto value_scal_by_deg =
                                                    value * s1234_deg;
                                                F(bf1, bf2) +=
                                                    2.0 * D(bf3, bf4) *
                                                    value_scal_by_deg;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        std::array<int, 4> idxs{s1, s3 + nsh, s2, s4 + nsh};
                        if (spherical) {
                            dims = m_env.four_center_helper<op, Sph>(
                                idxs, opt.optimizer_ptr(), buffer.get(),
                                nullptr);
                        } else {
                            dims = m_env.four_center_helper<op, Cart>(
                                idxs, opt.optimizer_ptr(), buffer.get(),
                                nullptr);
                        }
                        if (dims[0] < 0)
                            continue;

                        const auto *buf_1324 = buffer.get();

                        for (auto f4 = 0, f1324 = 0; f4 != n4; ++f4) {
                            const auto bf4 = f4 + bf4_first;
                            for (auto f2 = 0; f2 != n2; ++f2) {
                                const auto bf2 = f2 + bf2_first;
                                for (auto f3 = 0; f3 != n3; ++f3) {
                                    const auto bf3 = f3 + bf3_first;
                                    for (auto f1 = 0; f1 != n1; ++f1, ++f1324) {
                                        const auto bf1 = f1 + bf1_first;

                                        const auto value = buf_1324[f1324];
                                        const auto value_scal_by_deg =
                                            value * s12_deg;
                                        F(bf1, bf2) -=
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
        Fmats[0] += Fmats[i];
    }

    clear_auxiliary_basis();
    // symmetrize the result and return
    return 0.5 * (Fmats[0] + Fmats[0].transpose());
}

template <ShellKind kind = ShellKind::Cartesian>
Mat schwarz_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                   const ShellPairList &shellpairs) {
    constexpr auto op = Op::coulomb;
    using Result = IntegralEngine::IntegralResult<4>;
    auto nthreads = occ::parallel::get_num_threads();
    constexpr bool use_euclidean_norm{false};
    const auto nsh = basis.size();
    const auto &first_bf = basis.first_bf();
    std::vector<Mat> results;
    results.emplace_back(Mat::Zero(nsh, nsh));
    for (size_t i = 1; i < nthreads; i++) {
        results.push_back(results[0]);
    }

    auto f = [&results](const Result &args) {
        auto &result = results[args.thread];
        auto N = args.dims[0] * args.dims[1];
        Eigen::Map<const occ::Mat> tmp(args.buffer, N, N);
        double sq_norm =
            use_euclidean_norm ? tmp.norm() : tmp.array().abs().maxCoeff();
        double norm = std::sqrt(sq_norm);
        result(args.shell[0], args.shell[1]) = norm;
        result(args.shell[1], args.shell[0]) = norm;
    };

    auto lambda = [&](int thread_id) {
        auto buffer = std::make_unique<double[]>(env.buffer_size_2e());
        for (int p = 0, pq = 0; p < nsh; p++) {
            int bf1 = first_bf[p];
            for (const auto &q : shellpairs[p]) {
                if (pq++ % nthreads != thread_id)
                    continue;
                int bf2 = first_bf[q];
                std::array<int, 4> idxs{p, static_cast<int>(q), p, static_cast<int>(q)};
                Result args{thread_id,
                            idxs,
                            {bf1, bf2, bf1, bf2},
                            env.four_center_helper<op, kind>(
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

Mat IntegralEngine::schwarz() const {
    if (is_spherical()) {
        return schwarz_kernel<ShellKind::Spherical>(m_env, m_aobasis,
                                                    m_shellpairs);
    } else {
        return schwarz_kernel<ShellKind::Cartesian>(m_env, m_aobasis,
                                                    m_shellpairs);
    }
}

/*
 * Three-center integrals
 */
template <ShellKind kind, typename Lambda>
void three_center_aux_kernel(Lambda &f, qm::cint::IntegralEnvironment &env,
                             const qm::AOBasis &aobasis,
                             const qm::AOBasis &auxbasis,
                             const ShellPairList &shellpairs,
                             int thread_id = 0) noexcept {
    using Result = IntegralEngine::IntegralResult<3>;
    occ::timing::start(occ::timing::category::ints3c2e);
    auto nthreads = occ::parallel::get_num_threads();
    occ::qm::cint::Optimizer opt(env, Op::coulomb, 3);
    size_t bufsize = aobasis.max_shell_size() * aobasis.max_shell_size() *
                     auxbasis.max_shell_size();
    auto buffer = std::make_unique<double[]>(bufsize);
    Result args;
    args.thread = thread_id;
    args.buffer = buffer.get();
    std::array<int, 3> shell_idx;
    const auto &first_bf_ao = aobasis.first_bf();
    const auto &first_bf_aux = auxbasis.first_bf();
    for (int auxP = 0; auxP < auxbasis.size(); auxP++) {
        if (auxP % nthreads != thread_id)
            continue;
        args.bf[2] = first_bf_aux[auxP];
        args.shell[2] = auxP;
        for (int p = 0; p < aobasis.size(); p++) {
            args.bf[0] = first_bf_ao[p];
            args.shell[0] = p;
            const auto &plist = shellpairs[p];
            for (const auto &q : plist) {
                args.bf[1] = first_bf_ao[q];
                args.shell[1] = q;
                shell_idx = {p, static_cast<int>(q), auxP + static_cast<int>(aobasis.size())};
                args.dims = env.three_center_helper<Op::coulomb, kind>(
                    shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr);
                if (args.dims[0] > -1) {
                    f(args);
                }
            }
        }
    }
    occ::timing::stop(occ::timing::category::ints3c2e);
}

template <ShellKind kind = ShellKind::Cartesian>
Mat point_charge_potential_kernel(cint::IntegralEnvironment &env,
                                  const AOBasis &aobasis,
                                  const AOBasis &auxbasis,
                                  const ShellPairList &shellpairs) {
    using Result = IntegralEngine::IntegralResult<3>;
    auto nthreads = occ::parallel::get_num_threads();
    const auto nbf = aobasis.nbf();
    std::vector<Mat> results(nthreads, Mat::Zero(nbf, nbf));
    auto f = [&results](const Result &args) {
        auto &result = results[args.thread];
        Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
        result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += tmp;
        if (args.shell[0] != args.shell[1]) {
            result.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
                tmp.transpose();
        }
    };

    auto lambda = [&](int thread_id) {
        three_center_aux_kernel<kind>(f, env, aobasis, auxbasis, shellpairs,
                                      thread_id);
    };
    occ::parallel::parallel_do(lambda);

    for (auto i = 1; i < nthreads; i++) {
        results[0] += results[i];
    }
    return results[0];
}

Mat IntegralEngine::point_charge_potential(
    const std::vector<occ::core::PointCharge> &charges) {
    ShellList dummy_shells;
    dummy_shells.reserve(charges.size());
    for (size_t i = 0; i < charges.size(); i++) {
        dummy_shells.push_back(Shell(charges[i]));
    }
    set_auxiliary_basis(dummy_shells, true);
    if (is_spherical()) {
        return point_charge_potential_kernel<ShellKind::Spherical>(
            m_env, m_aobasis, m_auxbasis, m_shellpairs);
    } else {
        return point_charge_potential_kernel<ShellKind::Cartesian>(
            m_env, m_aobasis, m_auxbasis, m_shellpairs);
    }
}

#if HAVE_ECPINT
Vec electric_potential_ecp_kernel(std::vector<libecpint::ECP> &ecps,
                                  int ecp_lmax, const Mat3N &points) {
    Vec result = Vec::Zero(points.cols());
    for (int pt = 0; pt < points.cols(); pt++) {
        for (const auto &U : ecps) {
            double dx = points(0, pt) - U.center_[0];
            double dy = points(1, pt) - U.center_[1];
            double dz = points(2, pt) - U.center_[2];
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);

            double fac = 1.0;
            for (int l = 0; l <= U.getL(); l++) {
                result(pt) += fac * U.evaluate(r, l);
                fac *= r;
            }
        }
    }
    return result;
}
#endif

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Vec electric_potential_kernel(cint::IntegralEnvironment &env,
                              const AOBasis &aobasis, const AOBasis &auxbasis,
                              const ShellPairList &shellpairs,
                              const MolecularOrbitals &mo) {
    using Result = IntegralEngine::IntegralResult<3>;
    auto nthreads = occ::parallel::get_num_threads();
    size_t npts = auxbasis.size();
    std::vector<Vec> results(nthreads, Vec::Zero(npts));

    const auto &D = mo.D;
    auto f = [&D, &results](const Result &args) {
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
            const auto beta = occ::qm::block::b(D).block(
                args.bf[0], args.bf[1], args.dims[0], args.dims[1]);
            v(args.shell[2]) +=
                ((alpha.array() + beta.array()) * tmp.array()).sum() * scale;
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
        three_center_aux_kernel<kind>(f, env, aobasis, auxbasis, shellpairs,
                                      thread_id);
    };
    occ::parallel::parallel_do(lambda);

    for (auto i = 1; i < nthreads; i++) {
        results[0] += results[i];
    }
    return 2 * results[0];
}

Vec IntegralEngine::electric_potential(const MolecularOrbitals &mo,
                                       const Mat3N &points) {
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Sph = ShellKind::Spherical;
    constexpr auto Cart = ShellKind::Cartesian;
    ShellList dummy_shells;
    dummy_shells.reserve(points.cols());
    Vec result = Vec::Zero(points.cols());
    for (size_t i = 0; i < points.cols(); i++) {
        dummy_shells.push_back(
            Shell({1.0, {points(0, i), points(1, i), points(2, i)}}));
    }
    set_auxiliary_basis(dummy_shells, true);

    // Below code could be used if ECPs are needed for electric potential,
    // not sure if it's correct
    /*
    if (m_have_ecp) {
        result += electric_potential_ecp_kernel(m_ecp, m_ecp_max_l, points);
    }
    */

    if (is_spherical()) {
        switch (mo.kind) {
        default: // Restricted
            result += electric_potential_kernel<R, Sph>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        case U:
            result += electric_potential_kernel<U, Sph>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        case G:
            result += electric_potential_kernel<G, Sph>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        }
    } else {
        switch (mo.kind) {
        default: // Restricted
            result += electric_potential_kernel<R, Cart>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        case U:
            result += electric_potential_kernel<U, Cart>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        case G:
            result += electric_potential_kernel<G, Cart>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        }
    }
    return result;
}

} // namespace occ::qm
