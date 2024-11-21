#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/qm/hf.h>

namespace occ::qm {

void HartreeFock::set_density_fitting_basis(
    const std::string &density_fitting_basis) {
  occ::qm::AOBasis dfbasis =
      occ::qm::AOBasis::load(atoms(), density_fitting_basis);
  dfbasis.set_kind(m_engine.aobasis().kind());
  m_df_engine = std::make_unique<IntegralEngineDF>(
      atoms(), m_engine.aobasis().shells(), dfbasis.shells());
}

HartreeFock::HartreeFock(const AOBasis &basis)
    : SCFMethodBase(basis.atoms()), m_engine(basis) {

  update_electron_count();

  std::vector<int> frozen(basis.atoms().size(), 0);
  int num_frozen = basis.total_ecp_electrons();
  if (m_num_frozen > 0) {
    frozen = basis.ecp_electrons();
  }
  set_frozen_electrons(frozen);
}

double HartreeFock::nuclear_point_charge_interaction_energy(
    const PointChargeList &pc) const {
  double etot = 0.0;

  int i = 0;
  for (const auto &atom : atoms()) {
    double Z = atom.atomic_number - m_frozen_electrons[i];

    for (const auto &[q, pos] : pc) {
      auto xij = atom.x - pos[0];
      auto yij = atom.y - pos[1];
      auto zij = atom.z - pos[2];
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = std::sqrt(r2);
      etot += Z * q / r;
    }
    i++;
  }
  return etot;
}

/*
double HartreeFock::nuclear_point_charge_interaction_energy(
    const PointChargeList &pc) const {
  double etot = 0.0;
  const double alpha = 0.01;                             // damping parameter
  const double rc = 12.0 * occ::units::ANGSTROM_TO_BOHR; // cutoff radius

  // Pre-calculate Wolf self-terms
  const double erfc_term = std::erfc(alpha * rc) / rc;
  const double gaussian_term =
      (2.0 * alpha / std::sqrt(M_PI)) * std::exp(-alpha * alpha * rc * rc);
  const double self_term = erfc_term + gaussian_term;

  // Nuclear-point charge interactions
  int i = 0;
  for (const auto &atom : atoms()) {
    double Z = atom.atomic_number - m_frozen_electrons[i];

    for (const auto &[q, pos] : pc) {
      auto xij = atom.x - pos[0];
      auto yij = atom.y - pos[1];
      auto zij = atom.z - pos[2];
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = std::sqrt(r2);

      if (r > rc)
        continue;

      // Direct Coulomb with error function damping
      etot += Z * q * std::erfc(alpha * r) / r;

      // Wolf correction term
      etot -= Z * q * self_term;
    }
    i++;
  }

  // Point charge self-terms
  for (const auto &[q, pos] : pc) {
    etot -= 0.5 * q * q * self_term;
  }

  return etot;
}
*/

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
    qm::block::a(F) =
        m_engine.fock_operator_mixed_basis(mo_minbs.D, bs, is_shell_diagonal);
    qm::block::b(F) = qm::block::a(F);
    return F;
  } else { // kind == SpinorbitalKind::General
    const auto [rows, cols] =
        occ::qm::matrix_dimensions<SpinorbitalKind::General>(
            m_engine.aobasis().nbf());
    Mat F = Mat::Zero(rows, cols);
    qm::block::aa(F) =
        m_engine.fock_operator_mixed_basis(mo_minbs.D, bs, is_shell_diagonal);
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

std::vector<JKPair>
HartreeFock::compute_JK_list(const std::vector<MolecularOrbitals> &mos,
                             const Mat &Schwarz) const {
  return m_engine.coulomb_and_exchange_list(mos[0].kind, mos, Schwarz);
}

std::vector<Mat>
HartreeFock::compute_J_list(const std::vector<MolecularOrbitals> &mos,
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

Mat HartreeFock::compute_overlap_matrix_for_basis(
    const occ::qm::AOBasis &basis) const {
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
  /*
  const double alpha = 0.01;
  return m_engine.wolf_point_charge_potential(
      point_charges, 12.0 * occ::units::ANGSTROM_TO_BOHR, alpha);
  */
  return m_engine.point_charge_potential(point_charges);
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
