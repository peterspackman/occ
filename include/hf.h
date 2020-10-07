#pragma once
#include "ints.h"

namespace tonto::hf {

using tonto::MatRM;
using tonto::ints::BasisSet;
using tonto::ints::compute_1body_ints;
using tonto::ints::compute_1body_ints_deriv;
using tonto::ints::Operator;
using tonto::ints::shellpair_data_t;
using tonto::ints::shellpair_list_t;

/// to use precomputed shell pair data must decide on max precision a priori
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

std::tuple<shellpair_list_t, shellpair_data_t>
compute_shellpairs(const BasisSet &bs1, const BasisSet &bs2 = BasisSet(),
                   double threshold = 1e-12);

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

  double nuclear_repulsion_energy() const;
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

  MatRM
  compute_2body_fock(const MatRM &D,
                     double precision = std::numeric_limits<double>::epsilon(),
                     const MatRM &Schwarz = MatRM()) const;

  std::pair<MatRM, MatRM>
  compute_JK(const MatRM &D,
             double precision = std::numeric_limits<double>::epsilon(),
             const MatRM &Schwarz = MatRM()) const;

  MatRM
  compute_J(const MatRM &D,
            double precision = std::numeric_limits<double>::epsilon(),
            const MatRM &Schwarz = MatRM()) const;

  std::pair<MatRM, MatRM> compute_2body_fock_unrestricted(
      const MatRM &Da, const MatRM &Db,
      double precision = std::numeric_limits<double>::epsilon(),
      const MatRM &Schwarz = MatRM()) const;

  std::tuple<MatRM, MatRM, MatRM, MatRM> compute_JK_unrestricted(
      const MatRM &Da, const MatRM &Db,
      double precision = std::numeric_limits<double>::epsilon(),
      const MatRM &Schwarz = MatRM()) const;

  MatRM
  compute_2body_fock_general(const MatRM &D,
                     double precision = std::numeric_limits<double>::epsilon(),
                     const MatRM &Schwarz = MatRM()) const;

  std::pair<MatRM, MatRM>
  compute_JK_general(const MatRM &D,
                     double precision = std::numeric_limits<double>::epsilon(),
                     const MatRM &Schwarz = MatRM()) const;

  MatRM compute_shellblock_norm(const MatRM &A) const;

  auto compute_schwarz_ints() {
    return tonto::ints::compute_schwarz_ints<>(m_basis);
  }

private:
  int m_charge{0};
  int m_num_e{0};
  std::vector<libint2::Atom> m_atoms;
  BasisSet m_basis;
  shellpair_list_t m_shellpair_list; // shellpair list for OBS
  shellpair_data_t m_shellpair_data; // shellpair data for OBS
};

} // namespace tonto::hf
