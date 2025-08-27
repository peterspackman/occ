/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#include <occ/opt/dihedral_coordinate.h>
#include <occ/core/units.h>
#include <cmath>

namespace occ::opt {

DihedralCoordinate::DihedralCoordinate(int i_in, int j_in, int k_in, int l_in) {
  if (j_in > k_in) {
    i = l_in;
    j = k_in;
    k = j_in;
    l = i_in;
  } else {
    i = i_in;
    j = j_in;
    k = k_in;
    l = l_in;
  }
}

double DihedralCoordinate::operator()(const occ::Mat3N &coords) const {
  occ::Vec3 v1 = (coords.col(i) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR ;
  occ::Vec3 v2 = (coords.col(l) - coords.col(k)) * occ::units::ANGSTROM_TO_BOHR ;
  occ::Vec3 w = (coords.col(k) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR ;

  // Lines 151-153
  occ::Vec3 ew = w / w.norm();
  occ::Vec3 a1 = v1 - v1.dot(ew) * ew;
  occ::Vec3 a2 = v2 - v2.dot(ew) * ew;

  // Lines 154-155
  occ::Mat3 det_matrix;
  det_matrix.col(0) = v2;
  det_matrix.col(1) = v1;
  det_matrix.col(2) = w;
  double sgn = (det_matrix.determinant() > 0) ? 1.0 : -1.0;
  if (sgn == 0)
    sgn = 1.0;

  // Lines 156-161
  double dot_product = a1.dot(a2) / (a1.norm() * a2.norm());
  if (dot_product < -1) {
    dot_product = -1;
  } else if (dot_product > 1) {
    dot_product = 1;
  }
  
  return std::acos(dot_product) * sgn;
}

occ::Mat3N DihedralCoordinate::gradient(const occ::Mat3N &coords) const {
  occ::Vec3 v1 = (coords.col(i) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR ;
  occ::Vec3 v2 = (coords.col(l) - coords.col(k)) * occ::units::ANGSTROM_TO_BOHR ;
  occ::Vec3 w = (coords.col(k) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR ;

  // Lines 151-153
  occ::Vec3 ew = w / w.norm();
  occ::Vec3 a1 = v1 - v1.dot(ew) * ew;
  occ::Vec3 a2 = v2 - v2.dot(ew) * ew;

  // Lines 154-155
  occ::Mat3 det_matrix;
  det_matrix.col(0) = v2;
  det_matrix.col(1) = v1;
  det_matrix.col(2) = w;
  double sgn = (det_matrix.determinant() > 0) ? 1.0 : -1.0;
  if (sgn == 0)
    sgn = 1.0;

  // Lines 156-161
  double dot_product = a1.dot(a2) / (a1.norm() * a2.norm());
  if (dot_product < -1) {
    dot_product = -1;
  } else if (dot_product > 1) {
    dot_product = 1;
  }
  double phi = std::acos(dot_product) * sgn;

  occ::Mat3N grad(3, 4);

  // Lines 164-174 - case abs(phi) > pi - 1e-6
  if (std::abs(phi) > M_PI - 1e-6) {
    occ::Vec3 g = cross(w, a1);
    g = g / g.norm();
    double A = v1.dot(ew) / w.norm();
    double B = v2.dot(ew) / w.norm();

    grad.col(0) = g / (g.norm() * a1.norm());
    grad.col(1) = -((1 - A) / a1.norm() - B / a2.norm()) * g;
    grad.col(2) = -((1 + B) / a2.norm() + A / a1.norm()) * g;
    grad.col(3) = g / (g.norm() * a2.norm());
  }
  // Lines 175-185 - case abs(phi) < 1e-6
  else if (std::abs(phi) < 1e-6) {
    occ::Vec3 g = cross(w, a1);
    g = g / g.norm();
    double A = v1.dot(ew) / w.norm();
    double B = v2.dot(ew) / w.norm();

    grad.col(0) = g / (g.norm() * a1.norm());
    grad.col(1) = -((1 - A) / a1.norm() + B / a2.norm()) * g;
    grad.col(2) = ((1 + B) / a2.norm() - A / a1.norm()) * g;
    grad.col(3) = -g / (g.norm() * a2.norm());
  }
  // Lines 186-210+ - normal case
  else {
    double A = v1.dot(ew) / w.norm();
    double B = v2.dot(ew) / w.norm();
    double cot_phi = std::cos(phi) / std::sin(phi);
    double sin_phi = std::sin(phi);

    grad.col(0) = cot_phi * a1 / a1.squaredNorm() -
                             a2 / (a1.norm() * a2.norm() * sin_phi);
    grad.col(1) =
        ((1 - A) * a2 - B * a1) / (a1.norm() * a2.norm() * sin_phi) -
        cot_phi *
            ((1 - A) * a1 / a1.squaredNorm() - B * a2 / a2.squaredNorm());
    grad.col(2) =
        ((1 + B) * a1 + A * a2) / (a1.norm() * a2.norm() * sin_phi) -
        cot_phi *
            ((1 + B) * a2 / a2.squaredNorm() + A * a1 / a1.squaredNorm());
    grad.col(3) = cot_phi * a2 / a2.squaredNorm() -
                             a1 / (a1.norm() * a2.norm() * sin_phi);
  }

  return grad;
}

double DihedralCoordinate::hessian(const occ::Mat &rho) const {
  return 0.005 * rho(i, j) * rho(j, k) * rho(k, l);
}

double DihedralCoordinate::weight(const occ::Mat &rho, const occ::Mat3N &coords) const {
  double f = 0.12;
  AngleCoordinate th1(i, j, k);
  AngleCoordinate th2(j, k, l);
  double th1_val = th1(coords);
  double th2_val = th2(coords);
  return std::pow(rho(i, j) * rho(j, k) * rho(k, l), 1.0 / 3.0) *
         (f + (1 - f) * std::sin(th1_val)) *
         (f + (1 - f) * std::sin(th2_val));
}

occ::Vec3 DihedralCoordinate::cross(const occ::Vec3 &a, const occ::Vec3 &b) const {
  return a.cross(b);
}

} // namespace occ::opt