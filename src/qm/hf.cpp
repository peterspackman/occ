#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/qm/hf.h>

namespace occ::qm {

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
    m_df_engine = std::make_unique<IntegralEngineDF>(
        m_atoms, m_engine.aobasis().shells(), dfbasis.shells());
}

HartreeFock::HartreeFock(const AOBasis &basis)
    : m_atoms(basis.atoms()),
      m_frozen_electrons(basis.atoms().size(), 0),
      m_engine(basis) {

    for (const auto &a : m_atoms) {
        m_num_e += a.atomic_number;
    }
    m_num_e -= m_charge;
    m_num_frozen = basis.total_ecp_electrons();
    if (m_num_frozen > 0) {
        m_frozen_electrons = basis.ecp_electrons();
    }
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
            enuc += (m_atoms[i].atomic_number - m_frozen_electrons[i]) *
                    (m_atoms[j].atomic_number - m_frozen_electrons[j]) / r;
        }
    return enuc;
}

Mat3N HartreeFock::nuclear_repulsion_gradient() const {
    Mat3N grad = Mat3N::Zero(3, m_atoms.size());
    for (auto i = 0; i < m_atoms.size(); i++)
        for (auto j = i + 1; j < m_atoms.size(); j++) {
            auto xij = m_atoms[i].x - m_atoms[j].x;
            auto yij = m_atoms[i].y - m_atoms[j].y;
            auto zij = m_atoms[i].z - m_atoms[j].z;
            auto r2 = xij * xij + yij * yij + zij * zij;
            auto r = sqrt(r2);
	    double fac = (m_atoms[i].atomic_number - m_frozen_electrons[i]) *
                         (m_atoms[j].atomic_number - m_frozen_electrons[j]) / 
			 (r * r2);
            grad(0, i) -= fac * xij;
            grad(1, i) -= fac * yij;
            grad(2, i) -= fac * zij;

            grad(0, j) += fac * xij;
            grad(1, j) += fac * yij;
            grad(2, j) += fac * zij;
        }
    return grad;
}

double HartreeFock::nuclear_point_charge_interaction_energy(
    const PointChargeList &pc) const {
    double etot = 0.0;
    for (auto i = 0; i < m_atoms.size(); i++) {
        double Z = m_atoms[i].atomic_number - m_frozen_electrons[i];
        for (const auto &[q, pos] : pc) {
            auto xij = m_atoms[i].x - pos[0];
            auto yij = m_atoms[i].y - pos[1];
            auto zij = m_atoms[i].z - pos[2];
            auto r2 = xij * xij + yij * yij + zij * zij;
            auto r = sqrt(r2);
            etot += Z * q / r;
        }
    }
    return etot;
}

Mat HartreeFock::compute_fock(const MolecularOrbitals &mo,
                              const Mat &Schwarz) const {
    if (m_df_engine) {
        return m_df_engine->fock_operator(mo);
    } else {
        return m_engine.fock_operator(mo.kind, mo, Schwarz);
    }
}

MatTriple HartreeFock::compute_fock_gradient(const MolecularOrbitals &mo,
					     const Mat &Schwarz) const {
    return m_engine.fock_operator_grad(mo.kind, mo, Schwarz);
}

Mat HartreeFock::compute_effective_core_potential_matrix() const {
    return m_engine.effective_core_potential();
}

Mat HartreeFock::compute_fock_mixed_basis(const MolecularOrbitals &mo_minbs,
                                          const qm::AOBasis &bs,
                                          bool is_shell_diagonal) {
    if (mo_minbs.kind == SpinorbitalKind::Restricted) {
        return m_engine.fock_operator_mixed_basis(mo_minbs.D, bs,
                                                  is_shell_diagonal);
    } else if (mo_minbs.kind == SpinorbitalKind::Unrestricted) {
        const auto [rows, cols] =
            occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(
                m_engine.aobasis().nbf());
        Mat F = Mat::Zero(rows, cols);
        qm::block::a(F) = m_engine.fock_operator_mixed_basis(mo_minbs.D, bs,
                                                             is_shell_diagonal);
        qm::block::b(F) = qm::block::a(F);
        return F;
    } else { // kind == SpinorbitalKind::General
        const auto [rows, cols] =
            occ::qm::matrix_dimensions<SpinorbitalKind::General>(
                m_engine.aobasis().nbf());
        Mat F = Mat::Zero(rows, cols);
        qm::block::aa(F) = m_engine.fock_operator_mixed_basis(
            mo_minbs.D, bs, is_shell_diagonal);
        qm::block::bb(F) = qm::block::aa(F);
        return F;
    }
}

JKPair HartreeFock::compute_JK(const MolecularOrbitals &mo,
                               const Mat &Schwarz) const {
    if (m_df_engine) {
        return m_df_engine->coulomb_and_exchange(mo);
    } else {
        return m_engine.coulomb_and_exchange(mo.kind, mo, Schwarz);
    }
}

std::vector<JKPair> HartreeFock::compute_JK_list(const std::vector<MolecularOrbitals> &mos,
                                            const Mat &Schwarz) const {
    return m_engine.coulomb_and_exchange_list(mos[0].kind, mos, Schwarz);
}

std::vector<Mat> HartreeFock::compute_J_list(const std::vector<MolecularOrbitals> &mos,
                                            const Mat &Schwarz) const {
    return m_engine.coulomb_list(mos[0].kind, mos, Schwarz);
}


Mat HartreeFock::compute_J(const MolecularOrbitals &mo,
                           const Mat &Schwarz) const {
    if (m_df_engine) {
        return m_df_engine->coulomb(mo);
    } else {
        return m_engine.coulomb(mo.kind, mo, Schwarz);
    }
}

MatTriple HartreeFock::compute_J_gradient(const MolecularOrbitals &mo,
					  const Mat &Schwarz) const {
    return m_engine.coulomb_grad(mo.kind, mo, Schwarz);
}

JKTriple HartreeFock::compute_JK_gradient(const MolecularOrbitals &mo,
				          const Mat &Schwarz) const {
    return m_engine.coulomb_exchange_grad(mo.kind, mo, Schwarz);
}

Mat HartreeFock::compute_kinetic_matrix() const {
    using Op = occ::qm::cint::Operator;
    return m_engine.one_electron_operator(Op::kinetic);
}

Mat HartreeFock::compute_overlap_matrix() const {
    using Op = occ::qm::cint::Operator;
    return m_engine.one_electron_operator(Op::overlap);
}


Mat HartreeFock::compute_overlap_matrix_for_basis(const occ::qm::AOBasis &basis) const {
    using Op = occ::qm::cint::Operator;
    occ::qm::IntegralEngine temporary_engine(basis);
    return temporary_engine.one_electron_operator(Op::overlap);
}

Mat HartreeFock::compute_nuclear_attraction_matrix() const {
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
    int atom_index = 0;
    for (const auto &atom : m_atoms) {
        double Z = atom.atomic_number - m_frozen_electrons[atom_index];
        Vec3 atom_pos{atom.x, atom.y, atom.z};
        auto ab = positions.colwise() - atom_pos;
        auto r = ab.colwise().norm();
        auto r3 = r.array() * r.array() * r.array();
        result.array() += (Z * (ab.array().rowwise() / r3));
        atom_index++;
    }
    return result;
}

Mat3N HartreeFock::electronic_electric_field_contribution(
    const MolecularOrbitals &mo, const Mat3N &positions) const {
    double delta = 1e-8;
    occ::Mat3N efield_fd(positions.rows(), positions.cols());
    for (size_t i = 0; i < 3; i++) {
        auto pts_delta = positions;
        pts_delta.row(i).array() += delta;
        auto esp_f = electronic_electric_potential_contribution(mo, pts_delta);
        pts_delta.row(i).array() -= 2 * delta;
        auto esp_b = electronic_electric_potential_contribution(mo, pts_delta);
        efield_fd.row(i) = -(esp_f - esp_b) / (2 * delta);
    }
    return efield_fd;
}

Vec HartreeFock::electronic_electric_potential_contribution(
    const MolecularOrbitals &mo, const Mat3N &positions) const {
    return m_engine.electric_potential(mo, positions);
}

Vec HartreeFock::nuclear_electric_potential_contribution(
    const Mat3N &positions) const {
    Vec result = Vec::Zero(positions.cols());
    int atom_index = 0;
    for (const auto &atom : m_atoms) {
        double Z = atom.atomic_number - m_frozen_electrons[atom_index];
        Vec3 atom_pos{atom.x, atom.y, atom.z};
        auto ab = positions.colwise() - atom_pos;
        auto r = ab.colwise().norm();
        result.array() += Z / r.array();
        atom_index++;
    }
    return result;
}

Mat HartreeFock::compute_schwarz_ints() const { return m_engine.schwarz(); }

MatTriple HartreeFock::compute_kinetic_gradient() const {
    using Op = occ::qm::cint::Operator;
    return m_engine.one_electron_operator_grad(Op::kinetic);
}

MatTriple HartreeFock::compute_overlap_gradient() const {
    using Op = occ::qm::cint::Operator;
    return m_engine.one_electron_operator_grad(Op::overlap);
}

MatTriple HartreeFock::compute_nuclear_attraction_gradient() const {
    using Op = occ::qm::cint::Operator;
    return m_engine.one_electron_operator_grad(Op::nuclear);
}

MatTriple HartreeFock::compute_rinv_gradient_for_atom(size_t atom_index) const {
    return m_engine.rinv_operator_grad_atom(atom_index);
}

} // namespace occ::qm
