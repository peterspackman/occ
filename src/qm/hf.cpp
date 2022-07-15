#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/qm/hf.h>

namespace occ::hf {

Vec3 HartreeFock::center_of_mass() const {
    auto mol = occ::core::Molecule(m_atoms);
    return mol.center_of_mass() * occ::units::ANGSTROM_TO_BOHR;
}

void HartreeFock::set_system_charge(int charge) {
    m_num_e += m_charge;
    m_charge = charge;
    m_num_e -= m_charge;
}

void HartreeFock::set_density_fitting_basis(
    const std::string &density_fitting_basis) {
    occ::qm::AOBasis dfbasis =
        occ::qm::AOBasis::load(m_atoms, density_fitting_basis);
    dfbasis.set_kind(m_engine.aobasis().kind());
    m_df_engine.emplace(m_atoms, m_engine.aobasis().shells(), dfbasis.shells());
}

HartreeFock::HartreeFock(const AOBasis &basis)
    : m_atoms(basis.atoms()), m_engine(basis.atoms(), basis.shells()) {

    for (const auto &a : m_atoms) {
        m_num_e += a.atomic_number;
    }
    m_num_e -= m_charge;
}

double HartreeFock::nuclear_repulsion_energy() const {
    double enuc = 0.0;
    for (auto i = 0; i < m_atoms.size(); i++)
        for (auto j = i + 1; j < m_atoms.size(); j++) {
            auto xij = m_atoms[i].x - m_atoms[j].x;
            auto yij = m_atoms[i].y - m_atoms[j].y;
            auto zij = m_atoms[i].z - m_atoms[j].z;
            auto r2 = xij * xij + yij * yij + zij * zij;
            auto r = sqrt(r2);
            enuc += m_atoms[i].atomic_number * m_atoms[j].atomic_number / r;
        }
    return enuc;
}

Mat HartreeFock::compute_fock(SpinorbitalKind kind, const MolecularOrbitals &mo,
                              double precision, const Mat &Schwarz) const {
    if (m_df_engine && kind == SpinorbitalKind::Restricted) {
        return (*m_df_engine).fock_operator(mo);
    } else {
        return m_engine.fock_operator(kind, mo, Schwarz);
    }
}

Mat HartreeFock::compute_fock_mixed_basis(SpinorbitalKind kind, const Mat &D_bs,
                                          const qm::AOBasis &bs,
                                          bool is_shell_diagonal) {
    using Kind = occ::qm::Shell::Kind;
    if (kind == SpinorbitalKind::Restricted) {
        return m_engine.fock_operator_mixed_basis(D_bs, bs, is_shell_diagonal);
    } else if (kind == SpinorbitalKind::Unrestricted) {
        const auto [rows, cols] =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(
                m_engine.aobasis().nbf());
        Mat F = Mat::Zero(rows, cols);
        qm::block::a(F) =
            m_engine.fock_operator_mixed_basis(D_bs, bs, is_shell_diagonal);
        qm::block::b(F) = qm::block::a(F);
        return F;
    } else { // kind == SpinorbitalKind::General
        const auto [rows, cols] =
            occ::qm::matrix_dimensions<SpinorbitalKind::General>(
                m_engine.aobasis().nbf());
        Mat F = Mat::Zero(rows, cols);
        qm::block::aa(F) =
            m_engine.fock_operator_mixed_basis(D_bs, bs, is_shell_diagonal);
        qm::block::bb(F) = qm::block::aa(F);
        return F;
    }
}

std::pair<Mat, Mat> HartreeFock::compute_JK(SpinorbitalKind kind,
                                            const MolecularOrbitals &mo,
                                            double precision,
                                            const Mat &Schwarz) const {
    if (m_df_engine && kind == SpinorbitalKind::Restricted) {
        return (*m_df_engine).coulomb_and_exchange(mo);
    } else {
        return m_engine.coulomb_and_exchange(kind, mo, Schwarz);
    }
}

Mat HartreeFock::compute_J(SpinorbitalKind kind, const MolecularOrbitals &mo,
                           double precision, const Mat &Schwarz) const {
    if (m_df_engine && kind == SpinorbitalKind::Restricted) {
        return (*m_df_engine).coulomb(mo);
    } else {
        return m_engine.coulomb(kind, mo, Schwarz);
    }
}

Mat HartreeFock::compute_kinetic_matrix() const {
    using Kind = occ::qm::Shell::Kind;
    using Op = occ::qm::cint::Operator;
    return m_engine.one_electron_operator(Op::kinetic);
}

Mat HartreeFock::compute_overlap_matrix() const {
    using Kind = occ::qm::Shell::Kind;
    using Op = occ::qm::cint::Operator;
    return m_engine.one_electron_operator(Op::overlap);
}

Mat HartreeFock::compute_nuclear_attraction_matrix() const {
    using Kind = occ::qm::Shell::Kind;
    using Op = occ::qm::cint::Operator;
    return m_engine.one_electron_operator(Op::nuclear);
}

Mat HartreeFock::compute_point_charge_interaction_matrix(
    const PointChargeList &point_charges) const {
    return m_engine.point_charge_potential(point_charges);
}

Mat3N HartreeFock::nuclear_electric_field_contribution(
    const Mat3N &positions) const {
    Mat3N result = Mat3N::Zero(3, positions.cols());
    for (const auto &atom : m_atoms) {
        double Z = atom.atomic_number;
        Vec3 atom_pos{atom.x, atom.y, atom.z};
        auto ab = positions.colwise() - atom_pos;
        auto r = ab.colwise().norm();
        auto r3 = r.array() * r.array() * r.array();
        result.array() += (Z * (ab.array().rowwise() / r3));
    }
    return result;
}

Mat3N HartreeFock::electronic_electric_field_contribution(
    SpinorbitalKind kind, const MolecularOrbitals &mo,
    const Mat3N &positions) const {
    const auto &D = mo.D;
    double delta = 1e-8;
    occ::Mat3N efield_fd(positions.rows(), positions.cols());
    for (size_t i = 0; i < 3; i++) {
        auto pts_delta = positions;
        pts_delta.row(i).array() += delta;
        auto esp_f =
            electronic_electric_potential_contribution(kind, mo, pts_delta);
        pts_delta.row(i).array() -= 2 * delta;
        auto esp_b =
            electronic_electric_potential_contribution(kind, mo, pts_delta);
        efield_fd.row(i) = -(esp_f - esp_b) / (2 * delta);
    }
    return efield_fd;
}

Vec HartreeFock::electronic_electric_potential_contribution(
    SpinorbitalKind kind, const MolecularOrbitals &mo,
    const Mat3N &positions) const {
    const auto &D = mo.D;
    return m_engine.electric_potential(mo, positions);
}

Vec HartreeFock::nuclear_electric_potential_contribution(
    const Mat3N &positions) const {
    Vec result = Vec::Zero(positions.cols());
    for (const auto &atom : m_atoms) {
        double Z = atom.atomic_number;
        Vec3 atom_pos{atom.x, atom.y, atom.z};
        auto ab = positions.colwise() - atom_pos;
        auto r = ab.colwise().norm();
        result.array() += Z / r.array();
    }
    return result;
}

Mat HartreeFock::compute_schwarz_ints() const { return m_engine.schwarz(); }

} // namespace occ::hf
