#include <cmath>
#include <fmt/core.h>
#include <occ/gto/gto.h>
#include <occ/gto/rotation.h>
#include <occ/gto/shell_order.h>

namespace occ::gto {

// The code for these matrices has been modified from
// https://github.com/google/spherical-harmonics
// which is licensed under the Apache-2.0 license
// (compatible with GPLv3)
//
// Some comments have been retained for clarity,
// particularly regarding differences in the implementation
// from the original paper in J. Phys. Chem

inline constexpr double kronecker_delta(int a, int b) {
  return (a == b) ? 1.0 : 0.0;
}

Vec3 calculate_uvw_coefficients(int m, int n, int l) {
  double dm0 = kronecker_delta(m, 0);
  double denom =
      (std::abs(n) == l ? 2.0 * l * (2.0 * l - 1) : (l + n) * (l - n));

  Vec3 uvw;
  uvw(0) = std::sqrt((l + m) * (l - m) / denom);
  uvw(1) = 0.5 *
           std::sqrt((1 + dm0) * (l + std::abs(m) - 1.0) * (l + std::abs(m)) /
                     denom) *
           (1 - 2 * dm0);
  uvw(2) = -0.5 * std::sqrt((l - std::abs(m) - 1) * (l - std::abs(m)) / denom) *
           (1 - dm0);
  return uvw;
}

double get_centered_element(const Mat &r, int i, int j) {
  int offset = (r.rows() - 1) / 2;
  return r(i + offset, j + offset);
}

double P(int i, int a, int b, int l,
         const std::vector<Mat> &rotation_matrices) {
  if (b == l) {
    return (get_centered_element(rotation_matrices[1], i, 1) *
                get_centered_element(rotation_matrices[l - 1], a, l - 1) -
            get_centered_element(rotation_matrices[1], i, -1) *
                get_centered_element(rotation_matrices[l - 1], a, -l + 1));
  } else if (b == -l) {
    return (get_centered_element(rotation_matrices[1], i, 1) *
                get_centered_element(rotation_matrices[l - 1], a, -l + 1) +
            get_centered_element(rotation_matrices[1], i, -1) *
                get_centered_element(rotation_matrices[l - 1], a, l - 1));
  } else {
    return get_centered_element(rotation_matrices[1], i, 0) *
           get_centered_element(rotation_matrices[l - 1], a, b);
  }
}

double U(int m, int n, int l, const std::vector<Mat> &rotation_matrices) {
  return P(0, m, n, l, rotation_matrices);
}

double V(int m, int n, int l, std::vector<Mat> &rotation_matrices) {
  if (m == 0) {
    return P(1, 1, n, l, rotation_matrices) +
           P(-1, -1, n, l, rotation_matrices);
  } else if (m > 0) {
    return P(1, m - 1, n, l, rotation_matrices) *
               std::sqrt(1 + kronecker_delta(m, 1)) -
           P(-1, -m + 1, n, l, rotation_matrices) * (1 - kronecker_delta(m, 1));
  } else {
    // Note there is apparent errata in[1, 4, 4b] dealing with this
    // particular case.[4b] writes it should be P *(1 - d) + P *(1 - d) ^
    // 0.5
    // [1] writes it as P *(1 + d) + P *(1 - d) ^ 0.5, but going through the
    // math by hand, you must have it as P *(1 - d) + P *(1 + d) ^ 0.5 to form a
    // 2 ^ .5 term, which parallels the case where m> 0.
    return P(1, m + 1, n, l, rotation_matrices) * (1 - kronecker_delta(m, -1)) +
           P(-1, -m - 1, n, l, rotation_matrices) *
               std::sqrt(1 + kronecker_delta(m, -1));
  }
}

double W(int m, int n, int l, const std::vector<Mat> &rotation_matrices) {
  if (m == 0) {
    // whenever this happens, w is also 0 so W can be anything
    return 0.0;
  } else if (m > 0) {
    return P(1, m + 1, n, l, rotation_matrices) +
           P(-1, -m - 1, n, l, rotation_matrices);
  } else {
    return P(1, m - 1, n, l, rotation_matrices) -
           P(-1, -m + 1, n, l, rotation_matrices);
  }
}

template <ShellOrder order = ShellOrder::Default>
void populate_rotation_matrix_for_l(int l,
                                    std::vector<Mat> &rotation_matrices) {
  const double eps = 32 * std::numeric_limits<double>::epsilon();
  Mat rot(2 * l + 1, 2 * l + 1);
  for (int m = -l; m <= l; m++) {
    for (int n = -l; n <= l; n++) {
      Vec3 uvw = calculate_uvw_coefficients(m, n, l);
      if (std::abs(uvw(0)) > eps) {
        uvw(0) *= U(m, n, l, rotation_matrices);
      }
      if (std::abs(uvw(1)) > eps) {
        uvw(1) *= V(m, n, l, rotation_matrices);
      }
      if (std::abs(uvw(2)) > eps) {
        uvw(2) *= W(m, n, l, rotation_matrices);
      }

      rot(shell_index_spherical<order>(l, m),
          shell_index_spherical<order>(l, n)) = uvw.sum();
    }
  }
  rotation_matrices.push_back(rot);
}

std::vector<Mat> spherical_gaussian_rotation_matrices(int lmax,
                                                      const Mat3 &rotation) {

  std::vector<Mat> rotation_matrices{Mat(1, 1)};

  rotation_matrices[0].setConstant(1.0);
  // TODO ensure order is correct for l == 1
  // Apparently no condon-shortley phase factor for
  // the p functions... very consistent.
  Mat rot(3, 3);
  rot(0, 0) = rotation(1, 1);
  rot(0, 1) = rotation(1, 2);
  rot(0, 2) = rotation(1, 0);
  rot(1, 0) = rotation(2, 1);
  rot(1, 1) = rotation(2, 2);
  rot(1, 2) = rotation(2, 0);
  rot(2, 0) = rotation(0, 1);
  rot(2, 1) = rotation(0, 2);
  rot(2, 2) = rotation(0, 0);
  rotation_matrices.push_back(rot);

  for (int l = 2; l <= lmax; l++) {
    populate_rotation_matrix_for_l(l, rotation_matrices);
  }

  return rotation_matrices;
}

/*
 * Result should be R: an MxM rotation matrix for P: a MxN set of coordinates
 * giving results P' = R P
 */
template <int l> Mat cartesian_gaussian_rotation_matrix(const Mat3 &rotation) {
  constexpr int num_moments = (l + 1) * (l + 2) / 2;
  Mat result = Mat::Zero(num_moments, num_moments);
  auto cg_powers = cartesian_gaussian_power_index_arrays<l>();
  int p1_idx = 0;
  for (const auto &p1 : cg_powers) {
    int p2_idx = 0;
    // copy as we're permuting p2
    for (auto p2 : cg_powers) {
      do {
        double tmp{1.0};
        for (int k = 0; k < l; k++) {
          tmp *= rotation(p2[k], p1[k]);
        }
        result(p2_idx, p1_idx) += tmp;
      } while (std::next_permutation(p2.begin(), p2.end()));
      p2_idx++;
    }
    p1_idx++;
  }
  return result;
}

Mat cartesian_gaussian_rotation_matrix(int l, const Mat3 &rotation) {
  Mat rot;
  switch (l) {
  case 0:
    rot = occ::gto::cartesian_gaussian_rotation_matrix<0>(rotation);
    break;
  case 1:
    rot = occ::gto::cartesian_gaussian_rotation_matrix<1>(rotation);
    break;
  case 2:
    rot = occ::gto::cartesian_gaussian_rotation_matrix<2>(rotation);
    break;
  case 3:
    rot = occ::gto::cartesian_gaussian_rotation_matrix<3>(rotation);
    break;
  case 4:
    rot = occ::gto::cartesian_gaussian_rotation_matrix<4>(rotation);
  default:
    throw std::runtime_error(
        "MO rotation not implemented for angular momentum > 4");
  }

  return rot;
}

std::vector<Mat> cartesian_gaussian_rotation_matrices(int lmax,
                                                      const Mat3 &rotation) {

  std::vector<Mat> rotation_matrices{Mat(1, 1)};

  rotation_matrices[0].setConstant(1.0);

  for (int l = 1; l <= lmax; l++) {
    rotation_matrices.push_back(
        cartesian_gaussian_rotation_matrix(l, rotation));
  }

  return rotation_matrices;
}
} // namespace occ::gto
