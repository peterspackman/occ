#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm::orb {

inline auto occupied_restricted(Eigen::Ref<const Mat> orbitals,
                                size_t num_occ) {
  return orbitals.leftCols(num_occ);
}

inline auto occupied_restricted_fractional(Eigen::Ref<const Mat> orbitals,
                                           Eigen::Ref<const Vec> occupations) {
  return orbitals * occupations.array().sqrt().matrix().asDiagonal();
}

inline auto occupied_unrestricted(Eigen::Ref<const Mat> orbitals,
                                  size_t num_alpha, size_t num_beta) {
  size_t nbf = orbitals.rows() / 2;
  Mat occ = Mat::Zero(2 * nbf, std::max(num_alpha, num_beta));
  occ.block(0, 0, nbf, num_alpha) =
      occ::qm::block::a(orbitals).leftCols(num_alpha);
  occ.block(nbf, 0, nbf, num_beta) =
      occ::qm::block::b(orbitals).leftCols(num_beta);
  return occ;
}

inline auto
occupied_unrestricted_fractional(Eigen::Ref<const Mat> orbitals,
                                 Eigen::Ref<const Vec> occupations) {

  size_t nbf = orbitals.rows() / 2;
  auto [rows, cols] =
      occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
  Mat occ(rows, cols);
  block::a(occ) = block::a(orbitals) *
                  block::a(occupations).array().sqrt().matrix().asDiagonal();
  block::b(occ) = block::b(orbitals) *
                  block::b(occupations).array().sqrt().matrix().asDiagonal();
  return occ;
}

inline auto weighted_density_matrix_restricted(Eigen::Ref<const Mat> orbitals,
                                               Eigen::Ref<const Vec> weights) {

  return orbitals * weights.asDiagonal() * orbitals.transpose();
}

inline auto
weighted_density_matrix_unrestricted(Eigen::Ref<const Mat> orbitals,
                                     Eigen::Ref<const Vec> weights) {

  size_t nbf = orbitals.rows() / 2;
  auto [rows, cols] =
      occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
  Mat D(rows, cols);
  block::a(D) = block::a(orbitals) * block::a(weights).asDiagonal() *
                block::a(orbitals).transpose();
  block::b(D) = block::b(orbitals) * block::b(weights).asDiagonal() *
                block::b(orbitals).transpose();
  return D;
}

inline auto density_matrix_restricted(Eigen::Ref<const Mat> occupied_orbitals) {
  return occupied_orbitals * occupied_orbitals.transpose();
}

inline auto density_matrix_unrestricted(Eigen::Ref<const Mat> occupied_orbitals,
                                        size_t num_alpha, size_t num_beta) {
  size_t rows, cols;
  size_t nbf = occupied_orbitals.rows() / 2;
  std::tie(rows, cols) =
      occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
  Mat D(rows, cols);
  occ::qm::block::a(D) =
      occupied_orbitals.block(0, 0, nbf, num_alpha) *
      occupied_orbitals.block(0, 0, nbf, num_alpha).transpose();
  occ::qm::block::b(D) =
      occupied_orbitals.block(nbf, 0, nbf, num_beta) *
      occupied_orbitals.block(nbf, 0, nbf, num_beta).transpose();
  D *= 0.5;
  return D;
}

inline auto density_matrix_general(Eigen::Ref<const Mat> occupied_orbitals) {
  return 0.5 * occupied_orbitals * occupied_orbitals.transpose();
}

} // namespace occ::qm::orb
