/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#include <occ/opt/angle_coordinate.h>
#include <occ/core/units.h>
#include <cmath>

namespace occ::opt {

AngleCoordinate::AngleCoordinate(int i_in, int j_in, int k_in) {
  if (i_in > k_in) {
    i = k_in;
    j = j_in;
    k = i_in;
  } else {
    i = i_in;
    j = j_in;
    k = k_in;
  }
}

double AngleCoordinate::operator()(const occ::Mat3N &coords) const {
  occ::Vec3 v1 = (coords.col(i) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR;
  occ::Vec3 v2 = (coords.col(k) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR;

  double dot_product = v1.dot(v2) / (v1.norm() * v2.norm());

  if (dot_product < -1) {
    dot_product = -1;
  } else if (dot_product > 1) {
    dot_product = 1;
  }

  return std::acos(dot_product);
}

occ::Mat3N AngleCoordinate::gradient(const occ::Mat3N &coords) const {
  occ::Vec3 v1 = (coords.col(i) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR;
  occ::Vec3 v2 = (coords.col(k) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR;

  double dot_product = v1.dot(v2) / (v1.norm() * v2.norm());

  if (dot_product < -1) {
    dot_product = -1;
  } else if (dot_product > 1) {
    dot_product = 1;
  }

  double phi = std::acos(dot_product);
  
  occ::Mat3N grad(3, 3);

  if (std::abs(phi) > M_PI - 1e-6) {
    grad.col(0) = (M_PI - phi) / (2 * v1.squaredNorm()) * v1;
    grad.col(1) = (1.0 / v1.norm() - 1.0 / v2.norm()) *
                             (M_PI - phi) / (2 * v1.norm()) * v1;
    grad.col(2) = (M_PI - phi) / (2 * v2.squaredNorm()) * v2;
  } else {
    double cot_phi = std::cos(phi) / std::sin(phi); // 1 / np.tan(phi)
    double sin_phi = std::sin(phi);

    grad.col(0) = cot_phi * v1 / v1.squaredNorm() -
                             v2 / (v1.norm() * v2.norm() * sin_phi);

    grad.col(1) =
        (v1 + v2) / (v1.norm() * v2.norm() * sin_phi) -
        cot_phi * (v1 / v1.squaredNorm() + v2 / v2.squaredNorm());

    grad.col(2) = cot_phi * v2 / v2.squaredNorm() -
                             v1 / (v1.norm() * v2.norm() * sin_phi);
  }

  return grad;
}

double AngleCoordinate::hessian(const occ::Mat &rho) const {
  return 0.15 * (rho(i, j) * rho(j, k));
}

double AngleCoordinate::weight(const occ::Mat &rho, const occ::Mat3N &coords) const {
  double f = 0.12;
  double phi = (*this)(coords);
  return std::sqrt(rho(i, j) * rho(j, k)) * (f + (1 - f) * std::sin(phi));
}

} // namespace occ::opt