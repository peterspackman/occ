#include <occ/qm/integral_engine.h>

namespace occ::qm {

namespace impl {
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
    Fbb(bf0, bf2) -= 0.5 * D(bf1, bf3) * value;
    Fbb(bf1, bf3) -= 0.5 * D(bf0, bf2) * value;
    Fbb(bf0, bf3) -= 0.5 * D(bf1, bf2) * value;
    Fbb(bf1, bf2) -= 0.5 * D(bf0, bf3) * value;

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
    auto Jab = occ::qm::block::ab(J);
    auto Jba = occ::qm::block::ba(J);
    auto Jbb = occ::qm::block::bb(J);
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
}

inline void jk_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J,
                       Eigen::Ref<Mat> K, int bf0, int bf1, int bf2, int bf3,
                       double value) {
    auto Jaa = occ::qm::block::aa(J);
    auto Jab = occ::qm::block::ab(J);
    auto Jba = occ::qm::block::ba(J);
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
    Kbb(bf0, bf2) += 0.5 * D(bf1, bf3) * value;
    Kbb(bf1, bf3) += 0.5 * D(bf0, bf2) * value;
    Kbb(bf0, bf3) += 0.5 * D(bf1, bf2) * value;
    Kbb(bf1, bf2) += 0.5 * D(bf0, bf3) * value;

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

using ShellList = std::vector<OccShell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = OccShell::Kind;
using Op = cint::Operator;

size_t buffer_size_1e(const AOBasis &basis, Op op = Op::overlap) {
    auto bufsize = basis.max_shell_size() * basis.max_shell_size();
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

size_t buffer_size_2e(const AOBasis &basis) {
    return buffer_size_1e(basis) * buffer_size_1e(basis);
}

template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center(Lambda &f, cint::IntegralEnvironment &env,
                         const AOBasis &basis, int thread_id = 0) {
    using Result = IntegralEngine::IntegralResult<2>;
    occ::qm::cint::Optimizer opt(env, op, 2);
    auto nthreads = occ::parallel::get_num_threads();
    auto bufsize = buffer_size_1e(basis, op);
    const auto nsh = basis.size();

    auto buffer = std::make_unique<double[]>(bufsize);
    const auto &first_bf = basis.first_bf();
    for (int p = 0, pq = 0; p < nsh; p++) {
        int bf1 = first_bf[p];
        const auto &sh1 = basis[p];
        for (int q = 0; q <= p; q++) {
            if (pq++ % nthreads != thread_id)
                continue;
            int bf2 = first_bf[q];
            const auto &sh2 = basis[q];
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
    auto bufsize = buffer_size_1e(basis, op);

    auto buffer = std::make_unique<double[]>(bufsize);
    const auto &first_bf = basis.first_bf();
    for (int p = 0, pq = 0; p < basis.size(); p++) {
        int bf1 = first_bf[p];
        const auto &sh1 = basis[p];
        for (const int &q : shellpairs.at(p)) {
            if (pq++ % nthreads != thread_id)
                continue;
            int bf2 = first_bf[q];
            const auto &sh2 = basis[q];
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

template <int order, SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Vec multipole_kernel(const AOBasis &basis, cint::IntegralEnvironment &env,
                     const ShellPairList &shellpairs,
                     const MolecularOrbitals &mo, const Vec3 &origin) {

    using Result = IntegralEngine::IntegralResult<2>;
    constexpr std::array<Op, 5> ops{Op::overlap, Op::dipole, Op::quadrupole,
                                    Op::octapole, Op::hexadecapole};
    constexpr Op op = ops[order];

    auto nthreads = occ::parallel::get_num_threads();
    size_t num_components = occ::core::num_unique_multipole_components(order);
    env.set_common_origin({origin.x(), origin.y(), origin.z()});
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
    auto f = [&D, &results, &num_components](const Result &args) {
        auto &result = results[args.thread];
        size_t offset = 0;
        double scale = (args.shell[0] != args.shell[1]) ? 2.0 : 1.0;
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
    return results[0];
}

Vec IntegralEngine::multipole(SpinorbitalKind sk, int order,
                              const MolecularOrbitals &mo,
                              const Vec3 &origin) const {

    bool spherical = is_spherical();
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto U = SpinorbitalKind::Unrestricted;
    constexpr auto G = SpinorbitalKind::General;
    constexpr auto Cart = ShellKind::Cartesian;
    constexpr auto Sph = ShellKind::Spherical;
    if (sk == R) {
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
    } else if (sk == U) {
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

template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center(Lambda &f, cint::IntegralEnvironment &env,
                          const AOBasis &basis, const ShellPairList &shellpairs,
                          const Mat &Dnorm = Mat(), const Mat &Schwarz = Mat(),
                          double precision = 1e-12, int thread_id = 0) {
    using Result = IntegralEngine::IntegralResult<4>;
    auto nthreads = occ::parallel::get_num_threads();
    occ::qm::cint::Optimizer opt(env, Op::coulomb, 4);
    auto buffer = std::make_unique<double[]>(buffer_size_2e(basis));
    std::array<int, 4> shell_idx;
    std::array<int, 4> bf;

    const auto &first_bf = basis.first_bf();
    const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
    for (int p = 0, pqrs = 0; p < basis.size(); p++) {
        const auto &sh1 = basis[p];
        bf[0] = first_bf[p];
        const auto &plist = shellpairs.at(p);
        for (const int &q : plist) {
            bf[1] = first_bf[q];
            const auto &sh2 = basis[q];
            const auto DnormPQ = do_schwarz_screen ? Dnorm(p, q) : 0.;
            for (int r = 0; r <= p; r++) {
                const auto &sh3 = basis[r];
                bf[2] = first_bf[r];
                const auto s_max = (p == r) ? q : r;
                const auto DnormPQR =
                    do_schwarz_screen
                        ? std::max(Dnorm(p, r), std::max(Dnorm(q, r), DnormPQ))
                        : 0.;

                for (const int s : shellpairs.at(r)) {
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
                        DnormPQRS * Schwarz(p, q) * Schwarz(r, s) < precision)
                        continue;

                    bf[3] = first_bf[s];
                    const auto &sh4 = basis[s];
                    shell_idx = {p, q, r, s};

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
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Mat fock_operator_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                         const ShellPairList &shellpairs,
                         const MolecularOrbitals &mo,
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
                                       Schwarz, 1e-12, thread_id);
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

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Mat coulomb_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                   const ShellPairList &shellpairs, const MolecularOrbitals &mo,
                   const Mat &Schwarz = Mat()) {
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
                                       Schwarz, 1e-12, thread_id);
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
std::pair<Mat, Mat> coulomb_and_exchange_kernel(cint::IntegralEnvironment &env,
                                                const AOBasis &basis,
                                                const ShellPairList &shellpairs,
                                                const MolecularOrbitals &mo,
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
                                       Schwarz, 1e-12, thread_id);
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
                occ::qm::block::a(K).noalias() += (Ka + Ka.transpose());
                occ::qm::block::b(K).noalias() += (Kb + Kb.transpose());
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
                occ::qm::block::aa(K).noalias() += (Kaa + Kaa.transpose());
                occ::qm::block::ab(K).noalias() += (Kab + Kab.transpose());
                occ::qm::block::ba(K).noalias() += (Kba + Kba.transpose());
                occ::qm::block::bb(K).noalias() += (Kbb + Kbb.transpose());
            }
        }
    }
    J *= 0.5;
    K *= 0.5;
    return JK;
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
                                                mo, Schwarz);
        } else {
            return fock_operator_kernel<R, Cart>(m_env, m_aobasis, m_shellpairs,
                                                 mo, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return fock_operator_kernel<U, Sph>(m_env, m_aobasis, m_shellpairs,
                                                mo, Schwarz);
        } else {
            return fock_operator_kernel<U, Cart>(m_env, m_aobasis, m_shellpairs,
                                                 mo, Schwarz);
        }

    case G:
        if (spherical) {
            return fock_operator_kernel<G, Sph>(m_env, m_aobasis, m_shellpairs,
                                                mo, Schwarz);
        } else {
            return fock_operator_kernel<G, Cart>(m_env, m_aobasis, m_shellpairs,
                                                 mo, Schwarz);
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
                                          Schwarz);
        } else {
            return coulomb_kernel<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                           Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return coulomb_kernel<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                          Schwarz);
        } else {
            return coulomb_kernel<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                           Schwarz);
        }

    case G:
        if (spherical) {
            return coulomb_kernel<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                          Schwarz);
        } else {
            return coulomb_kernel<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                           Schwarz);
        }
    }
}

std::pair<Mat, Mat> IntegralEngine::coulomb_and_exchange(
    SpinorbitalKind sk, const MolecularOrbitals &mo, const Mat &Schwarz) const {
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
                m_env, m_aobasis, m_shellpairs, mo, Schwarz);
        } else {
            return coulomb_and_exchange_kernel<R, Cart>(
                m_env, m_aobasis, m_shellpairs, mo, Schwarz);
        }
        break;
    case U:
        if (spherical) {
            return coulomb_and_exchange_kernel<U, Sph>(
                m_env, m_aobasis, m_shellpairs, mo, Schwarz);
        } else {
            return coulomb_and_exchange_kernel<U, Cart>(
                m_env, m_aobasis, m_shellpairs, mo, Schwarz);
        }

    case G:
        if (spherical) {
            return coulomb_and_exchange_kernel<G, Sph>(
                m_env, m_aobasis, m_shellpairs, mo, Schwarz);
        } else {
            return coulomb_and_exchange_kernel<G, Cart>(
                m_env, m_aobasis, m_shellpairs, mo, Schwarz);
        }
    }
}

Mat IntegralEngine::fock_operator_mixed_basis(const Mat &D, const AOBasis &D_bs,
                                              bool is_shell_diagonal) {
    set_auxiliary_basis(D_bs.shells(), false);
    constexpr Op op = Op::coulomb;
    occ::timing::start(occ::timing::category::ints2e);
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
        auto buffer = std::make_unique<double[]>(buffer_size_2e());

        std::array<int, 4> idxs;
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

                        // compute the permutational degeneracy (i.e. # of
                        // equivalents) of the given shell set
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
    occ::timing::stop(occ::timing::category::ints2e);

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
            use_euclidean_norm ? tmp.norm() : tmp.lpNorm<Eigen::Infinity>();
        double norm = std::sqrt(sq_norm);
        result(args.shell[0], args.shell[1]) = norm;
        result(args.shell[1], args.shell[0]) = norm;
    };

    auto lambda = [&](int thread_id) {
        auto buffer = std::make_unique<double[]>(buffer_size_2e(basis));
        for (int p = 0, pq = 0; p < nsh; p++) {
            int bf1 = first_bf[p];
            const auto &sh1 = basis[p];
            for (const int &q : shellpairs.at(p)) {
                if (pq++ % nthreads != thread_id)
                    continue;
                int bf2 = first_bf[q];
                const auto &sh2 = basis[q];
                std::array<int, 4> idxs{p, q, p, q};
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
        const auto &shauxP = auxbasis[auxP];
        args.bf[2] = first_bf_aux[auxP];
        args.shell[2] = auxP;
        for (int p = 0; p < aobasis.size(); p++) {
            args.bf[0] = first_bf_ao[p];
            args.shell[0] = p;
            const auto &shp = aobasis[p];
            const auto &plist = shellpairs.at(p);
            for (const int &q : plist) {
                args.bf[1] = first_bf_ao[q];
                args.shell[1] = q;
                const auto &shq = aobasis[q];
                shell_idx = {p, q, auxP + static_cast<int>(aobasis.size())};
                args.dims = env.three_center_helper<Op::coulomb, kind>(
                    shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr);
                if (args.dims[0] > -1) {
                    f(args);
                }
            }
        }
    }
}

template <ShellKind kind = ShellKind::Cartesian>
Mat point_charge_potential_kernel(cint::IntegralEnvironment &env,
                                  const AOBasis &aobasis,
                                  const AOBasis &auxbasis,
                                  const ShellPairList &shellpairs) {
    using Result = IntegralEngine::IntegralResult<3>;
    auto nthreads = occ::parallel::get_num_threads();
    size_t nsh = aobasis.size();
    const auto nbf = aobasis.nbf();
    std::vector<Mat> results(nthreads, Mat::Zero(nbf, nbf));
    auto f = [nsh, &results](const Result &args) {
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
        dummy_shells.push_back(OccShell(charges[i]));
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
            const auto beta = occ::qm::block::a(D).block(
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
    for (size_t i = 0; i < points.cols(); i++) {
        dummy_shells.push_back(
            OccShell({1.0, {points(0, i), points(1, i), points(2, i)}}));
    }
    set_auxiliary_basis(dummy_shells, true);
    if (is_spherical()) {
        switch (mo.kind) {
        default: // Restricted
            return electric_potential_kernel<R, Sph>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        case U:
            return electric_potential_kernel<U, Sph>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        case G:
            return electric_potential_kernel<G, Sph>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        }
    } else {
        switch (mo.kind) {
        default: // Restricted
            return electric_potential_kernel<R, Cart>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        case U:
            return electric_potential_kernel<U, Cart>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        case G:
            return electric_potential_kernel<G, Cart>(
                m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
            break;
        }
    }
}

} // namespace occ::qm
