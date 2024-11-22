#pragma once
#include <occ/qm/orbital_smearing.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

class AOBasis;

struct MolecularOrbitals {
  MolecularOrbitals(const MolecularOrbitals &) = default;
  MolecularOrbitals() = default;
  MolecularOrbitals(const MolecularOrbitals &, const MolecularOrbitals &);
  SpinorbitalKind kind{SpinorbitalKind::Restricted};
  size_t n_alpha{0}, n_beta{0}, n_ao{0};
  Mat C;
  Mat Cocc;
  Mat D;
  Vec energies;
  Vec occupation;
  OrbitalSmearing smearing;

  void update(const Mat &ortho, const Mat &potential);
  void update_density_matrix();
  void update_occupied_orbitals();
  void update_occupied_orbitals_fractional();
  void rotate(const AOBasis &basis, const occ::Mat3 &rotation);
  void to_cartesian(const AOBasis &bspure, const AOBasis &bscart);
  void to_spherical(const AOBasis &bscart, const AOBasis &bspure);
  void print() const;
  void incorporate_norm(Eigen::Ref<const Vec> norms);
  double expectation_value(const Mat &op) const;

  inline const auto occ_alpha() const {
    return Cocc.block(0, 0, n_ao,
                      n_alpha); // block for consistent return type
  }

  inline const auto occ_beta() const {
    switch (kind) {
    default:
      // restricted
      return Cocc.block(0, 0, n_ao,
                        n_alpha); // block for consistent return type
    case SpinorbitalKind::Unrestricted:
      return Cocc.block(n_ao, 0, n_ao, n_beta);
    case SpinorbitalKind::General: // due to peculiarities of n_alpha/n_beta
                                   // in general case
      return Cocc.block(n_ao, 0, n_ao, n_alpha);
    }
  }

  Mat energy_weighted_density_matrix() const;
  Mat density_matrix_single_mo(int mo_index) const;

  MolecularOrbitals
  symmetrically_orthonormalized(const Mat &overlap_matrix) const;
  MolecularOrbitals as_kind(SpinorbitalKind) const;
};

} // namespace occ::qm
