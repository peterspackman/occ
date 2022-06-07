#include <occ/qm/integral_engine.h>

namespace occ::qm {

namespace impl {
void fock_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0, int bf1,
                  int bf2, int bf3, double value) noexcept {
    F(bf0, bf1) += D(bf2, bf3) * value;
    F(bf2, bf3) += D(bf0, bf1) * value;
    // K
    F(bf0, bf2) -= 0.25 * D(bf1, bf3) * value;
    F(bf1, bf3) -= 0.25 * D(bf0, bf2) * value;
    F(bf0, bf3) -= 0.25 * D(bf1, bf2) * value;
    F(bf1, bf2) -= 0.25 * D(bf0, bf3) * value;
}

void j_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0, int bf1,
               int bf2, int bf3, double value) noexcept {
    J(bf0, bf1) += D(bf2, bf3) * value;
    J(bf2, bf3) += D(bf0, bf1) * value;
}

void jk_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, Eigen::Ref<Mat> K,
                int bf0, int bf1, int bf2, int bf3, double value) noexcept {
    J(bf0, bf1) += D(bf2, bf3) * value;
    J(bf2, bf3) += D(bf0, bf1) * value;
    // K
    K(bf0, bf2) += 0.25 * D(bf1, bf3) * value;
    K(bf1, bf3) += 0.25 * D(bf0, bf2) * value;
    K(bf0, bf3) += 0.25 * D(bf1, bf2) * value;
    K(bf1, bf2) += 0.25 * D(bf0, bf3) * value;
}

void fock_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0, int bf1,
                  int bf2, int bf3, double value) noexcept {
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

void j_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0, int bf1,
               int bf2, int bf3, double value) noexcept {
    auto Ja = occ::qm::block::a(J);
    auto Jb = occ::qm::block::b(J);
    const auto Da = occ::qm::block::a(D);
    const auto Db = occ::qm::block::b(D);
    Ja(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
    Ja(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
    Jb(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
    Jb(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
}

void jk_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, Eigen::Ref<Mat> K,
                int bf0, int bf1, int bf2, int bf3, double value) noexcept {
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

void fock_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0, int bf1,
                  int bf2, int bf3, double value) noexcept {
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

void j_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0, int bf1,
               int bf2, int bf3, double value) noexcept {
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

void jk_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, Eigen::Ref<Mat> K,
                int bf0, int bf1, int bf2, int bf3, double value) noexcept {
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

} // namespace impl

namespace cint {

Optimizer::Optimizer(IntegralEnvironment &env, Operator op, int num_center)
    : m_op(op), m_num_center(num_center) {
    switch (m_num_center) {
    case 1:
    case 2:
        create1or2c(env);
        break;
    case 3:
        create3c(env);
        break;
    case 4:
        create4c(env);
        break;
    default:
        throw std::runtime_error("Invalid num centers for cint::Optimizer");
    }
}

Optimizer::~Optimizer() { libcint::CINTdel_optimizer(&m_optimizer); }

void Optimizer::create1or2c(IntegralEnvironment &env) {
    switch (m_op) {
    case Operator::coulomb:
        libcint::int2c2e_optimizer(&m_optimizer, env.atom_data_ptr(),
                                   env.num_atoms(), env.basis_data_ptr(),
                                   env.num_basis(), env.env_data_ptr());
        break;
    case Operator::nuclear:
        libcint::int1e_nuc_optimizer(&m_optimizer, env.atom_data_ptr(),
                                     env.num_atoms(), env.basis_data_ptr(),
                                     env.num_basis(), env.env_data_ptr());
        break;
    case Operator::kinetic:
        libcint::int1e_kin_optimizer(&m_optimizer, env.atom_data_ptr(),
                                     env.num_atoms(), env.basis_data_ptr(),
                                     env.num_basis(), env.env_data_ptr());
        break;
    case Operator::overlap:
        libcint::int1e_ovlp_optimizer(&m_optimizer, env.atom_data_ptr(),
                                      env.num_atoms(), env.basis_data_ptr(),
                                      env.num_basis(), env.env_data_ptr());
        break;
    }
}
void Optimizer::create3c(IntegralEnvironment &env) {
    switch (m_op) {
    case Operator::coulomb:
        libcint::int3c2e_optimizer(&m_optimizer, env.atom_data_ptr(),
                                   env.num_atoms(), env.basis_data_ptr(),
                                   env.num_basis(), env.env_data_ptr());
        break;
    default:
        throw std::runtime_error(
            "Invalid operator for 3-center integral optimizer");
    }
}
void Optimizer::create4c(IntegralEnvironment &env) {
    switch (m_op) {
    case Operator::coulomb:
        libcint::int2e_optimizer(&m_optimizer, env.atom_data_ptr(),
                                 env.num_atoms(), env.basis_data_ptr(),
                                 env.num_basis(), env.env_data_ptr());
        break;
    default:
        throw std::runtime_error(
            "Invalid operator for 4-center integral optimizer");
    }
}
} // namespace cint

using ShellList = std::vector<OccShell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = OccShell::Kind;
using Op = cint::Operator;

inline size_t buffer_size_1e(const AOBasis &basis, Op op = Op::overlap) {
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

template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center(Lambda &f, cint::IntegralEnvironment &env,
                         const AOBasis &basis, const ShellPairList &shellpairs,
                         int thread_id = 0) noexcept {
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
                                 const ShellPairList &shellpairs) noexcept {
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
        evaluate_two_center<op, kind>(f, env, basis, shellpairs, thread_id);
    };
    occ::parallel::parallel_do(lambda);

    for (auto i = 1; i < nthreads; ++i) {
        results[0].noalias() += results[i];
    }
    return results[0];
}

Mat IntegralEngine::one_electron_operator(Op op) const noexcept {
    bool spherical = is_spherical();
    constexpr auto Cart = ShellKind::Cartesian;
    constexpr auto Sph = ShellKind::Spherical;
    switch (op) {
    case Op::overlap: {
        if (spherical) {
            return one_electron_operator_kernel<Op::overlap, Sph>(
                m_aobasis, m_env, m_shellpairs);
        } else {
            return one_electron_operator_kernel<Op::overlap, Cart>(
                m_aobasis, m_env, m_shellpairs);
        }
        break;
    }
    case Op::nuclear: {
        if (spherical) {
            return one_electron_operator_kernel<Op::nuclear, Sph>(
                m_aobasis, m_env, m_shellpairs);
        } else {
            return one_electron_operator_kernel<Op::nuclear, Cart>(
                m_aobasis, m_env, m_shellpairs);
        }
        break;
    }
    case Op::kinetic: {
        if (spherical) {
            return one_electron_operator_kernel<Op::kinetic, Sph>(
                m_aobasis, m_env, m_shellpairs);
        } else {
            return one_electron_operator_kernel<Op::kinetic, Cart>(
                m_aobasis, m_env, m_shellpairs);
        }
        break;
    }
    case Op::coulomb: {
        if (spherical) {
            return one_electron_operator_kernel<Op::coulomb, Sph>(
                m_aobasis, m_env, m_shellpairs);
        } else {
            return one_electron_operator_kernel<Op::coulomb, Cart>(
                m_aobasis, m_env, m_shellpairs);
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
    static_assert(sk == SpinorbitalKind::Restricted,
                  "Unrestricted and General cases not implemented for "
                  "multipoles yet");
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
            result(n) += scale * (D.block(args.bf[0], args.bf[1], args.dims[0],
                                          args.dims[1])
                                      .array() *
                                  tmp.array())
                                     .sum();
            offset += tmp.size();
        }
    };

    auto lambda = [&](int thread_id) {
        evaluate_two_center<op, kind>(f, env, basis, shellpairs, thread_id);
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
    if (sk != SpinorbitalKind::Restricted) {
        throw std::runtime_error(
            "Multipole integrals only implemented for restricted case");
    }
    constexpr auto R = SpinorbitalKind::Restricted;
    constexpr auto Cart = ShellKind::Cartesian;
    constexpr auto Sph = ShellKind::Spherical;
    switch (order) {
    case 0:
        if (spherical) {
            return multipole_kernel<0, R, Sph>(m_aobasis, m_env, m_shellpairs,
                                               mo, origin);
        } else {
            return multipole_kernel<0, R, Cart>(m_aobasis, m_env, m_shellpairs,
                                                mo, origin);
        }
        break;
    case 1:
        if (spherical) {
            return multipole_kernel<1, R, Sph>(m_aobasis, m_env, m_shellpairs,
                                               mo, origin);
        } else {
            return multipole_kernel<1, R, Cart>(m_aobasis, m_env, m_shellpairs,
                                                mo, origin);
        }
        break;
    case 2:
        if (spherical) {
            return multipole_kernel<2, R, Sph>(m_aobasis, m_env, m_shellpairs,
                                               mo, origin);
        } else {
            return multipole_kernel<2, R, Cart>(m_aobasis, m_env, m_shellpairs,
                                                mo, origin);
        }
        break;
    case 3:
        if (spherical) {
            return multipole_kernel<3, R, Sph>(m_aobasis, m_env, m_shellpairs,
                                               mo, origin);
        } else {
            return multipole_kernel<3, R, Cart>(m_aobasis, m_env, m_shellpairs,
                                                mo, origin);
        }
        break;
    case 4:
        if (spherical) {
            return multipole_kernel<4, R, Sph>(m_aobasis, m_env, m_shellpairs,
                                               mo, origin);
        } else {
            return multipole_kernel<4, R, Cart>(m_aobasis, m_env, m_shellpairs,
                                                mo, origin);
        }
        break;
    default:
        throw std::runtime_error("Invalid multipole order");
        break;
    }
}

} // namespace occ::qm
