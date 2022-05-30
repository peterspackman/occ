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

} // namespace occ::qm
