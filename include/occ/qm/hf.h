#pragma once
#include <occ/qm/fock.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/energy_components.h>

namespace occ::hf {

using occ::qm::SpinorbitalKind;
using occ::MatRM;
using occ::ints::BasisSet;
using occ::ints::compute_1body_ints;
using occ::ints::compute_1body_ints_deriv;
using occ::ints::Operator;
using occ::ints::shellpair_data_t;
using occ::ints::shellpair_list_t;

/// to use precomputed shell pair data must decide on max precision a priori
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

class HartreeFock {
public:
  HartreeFock(const std::vector<libint2::Atom> &atoms, const BasisSet &basis);
  const auto &shellpair_list() const { return m_shellpair_list; }
  const auto &shellpair_data() const { return m_shellpair_data; }
  const auto &atoms() const { return m_atoms; }
  const auto &basis() const { return m_basis; }

  void set_system_charge(int charge) {
    m_num_e += m_charge;
    m_charge = charge;
    m_num_e -= m_charge;
  }
  int system_charge() const { return m_charge; }
  int num_e() const { return m_num_e; }

  double two_electron_energy_alpha() const { return m_e_alpha; }
  double two_electron_energy_beta() const { return m_e_beta; }
  double two_electron_energy() const { return m_e_alpha + m_e_beta; }
  bool usual_scf_energy() const { return true; }
  void update_scf_energy(occ::qm::EnergyComponents &energy) const { return; }
  bool supports_incremental_fock_build() const { return true; }

  double nuclear_repulsion_energy() const;

  MatRM compute_fock(SpinorbitalKind kind, const MatRM &D,
                    double precision = std::numeric_limits<double>::epsilon(),
                    const MatRM &Schwarz = MatRM()) const
  {
      if(kind == SpinorbitalKind::General) return occ::ints::compute_fock<SpinorbitalKind::General>(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
      if(kind == SpinorbitalKind::Unrestricted) return occ::ints::compute_fock<SpinorbitalKind::Unrestricted>(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
      return occ::ints::compute_fock<SpinorbitalKind::Restricted>(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
  }

  std::pair<MatRM, MatRM> compute_JK(SpinorbitalKind kind, const MatRM &D,
                    double precision = std::numeric_limits<double>::epsilon(),
                    const MatRM &Schwarz = MatRM()) const
  {
      if(kind == SpinorbitalKind::General) return occ::ints::compute_JK<SpinorbitalKind::General>(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
      if(kind == SpinorbitalKind::Unrestricted) return occ::ints::compute_JK<SpinorbitalKind::Unrestricted>(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
      return occ::ints::compute_JK<SpinorbitalKind::Restricted>(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
  }

  MatRM compute_J(SpinorbitalKind kind, const MatRM &D,
                  double precision = std::numeric_limits<double>::epsilon(),
                  const MatRM &Schwarz = MatRM()) const
  {
      if(kind == SpinorbitalKind::General) return occ::ints::compute_J<SpinorbitalKind::General>(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
      if(kind == SpinorbitalKind::Unrestricted) return occ::ints::compute_J<SpinorbitalKind::Unrestricted>(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
      return occ::ints::compute_J<SpinorbitalKind::Restricted>(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
  }


  auto compute_kinetic_matrix() {
    return compute_1body_ints<Operator::kinetic>(m_basis, m_shellpair_list)[0];
  }
  auto compute_overlap_matrix() {
    return compute_1body_ints<Operator::overlap>(m_basis, m_shellpair_list)[0];
  }
  auto compute_nuclear_attraction_matrix() {
    return compute_1body_ints<Operator::nuclear>(
        m_basis, m_shellpair_list, libint2::make_point_charges(m_atoms))[0];
  }

  auto compute_point_charge_interaction_matrix(const std::vector<std::pair<double, std::array<double, 3>>> &point_charges) {
    return compute_1body_ints<Operator::nuclear>(m_basis, m_shellpair_list, point_charges)[0];
  }

  auto compute_kinetic_energy_derivatives(unsigned derivative) {
    return compute_1body_ints_deriv<Operator::kinetic>(
        derivative, m_basis, m_shellpair_list, m_atoms);
  }

  auto compute_nuclear_attraction_derivatives(unsigned derivative) {
    return compute_1body_ints_deriv<Operator::nuclear>(
        derivative, m_basis, m_shellpair_list, m_atoms);
  }

  auto compute_overlap_derivatives(unsigned derivative) {
    return compute_1body_ints_deriv<Operator::overlap>(
        derivative, m_basis, m_shellpair_list, m_atoms);
  }

  Mat3N nuclear_electric_field_contribution(const Mat3N&) const;
  Mat3N electronic_electric_field_contribution(const MatRM&, const Mat3N&) const;
  Vec electronic_electric_potential_contribution(const MatRM&, const Mat3N&) const;
  Vec nuclear_electric_potential_contribution(const Mat3N&) const;

  MatRM compute_shellblock_norm(const MatRM &A) const;

  auto compute_schwarz_ints() {
    return occ::ints::compute_schwarz_ints<>(m_basis);
  }

  void update_core_hamiltonian(occ::qm::SpinorbitalKind k, const MatRM &D, MatRM &H) { return; }

private:
  int m_charge{0};
  int m_num_e{0};
  std::vector<libint2::Atom> m_atoms;
  BasisSet m_basis;
  shellpair_list_t m_shellpair_list; // shellpair list for OBS
  shellpair_data_t m_shellpair_data; // shellpair data for OBS
  mutable double m_e_alpha{0};
  mutable double m_e_beta{0};
};

} // namespace occ::hf
