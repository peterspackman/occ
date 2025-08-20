/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#include <occ/opt/bond_coordinate.h>
#include <occ/core/units.h>
#include <cmath>

namespace occ::opt {

BondCoordinate::BondCoordinate(int i_in, int j_in, Type type) : bond_type(type) {
  if (i_in > j_in) {
    i = j_in;
    j = i_in;
  } else {
    i = i_in;
    j = j_in;
  }
}

double BondCoordinate::operator()(const occ::Mat3N &coords) const {
  occ::Vec3 v = (coords.col(i) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR;
  return v.norm();
}

occ::Mat3N BondCoordinate::gradient(const occ::Mat3N &coords) const {
  occ::Vec3 v = (coords.col(i) - coords.col(j)) * occ::units::ANGSTROM_TO_BOHR;
  double distance = v.norm();
  
  occ::Mat3N grad(3, 2);
  grad.col(0) = v / distance;  // gradient w.r.t. atom i (Bohr/Angstrom units)
  grad.col(1) = -v / distance; // gradient w.r.t. atom j (Bohr/Angstrom units)
  return grad;
}

double BondCoordinate::hessian(const occ::Mat &rho) const {
  return 0.45 * rho(i, j);
}

double BondCoordinate::weight(const occ::Mat &rho) const {
  return rho(i, j);
}

} // namespace occ::opt