/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *
 * This file contains code derived from pyberny (https://github.com/jhrmnn/pyberny)
 * by Jan Hermann <dev@jan.hermann.name>, licensed under MPL-2.0.
 */

#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/opt/angle_coordinate.h>

namespace occ::opt {

class DihedralCoordinate {
public:
  int i, j, k, l;

  DihedralCoordinate(int i_in, int j_in, int k_in, int l_in);

  // Returns the dihedral angle value in radians
  double operator()(const occ::Mat3N &coords) const;

  // Returns the gradient matrix (3 x 4, columns are gradients w.r.t. atoms i, j, k, l)
  occ::Mat3N gradient(const occ::Mat3N &coords) const;

  // Hessian diagonal element for this coordinate
  double hessian(const occ::Mat &rho) const;

  // Weight for this coordinate
  double weight(const occ::Mat &rho, const occ::Mat3N &coords) const;

private:
  // Cross product helper
  occ::Vec3 cross(const occ::Vec3 &a, const occ::Vec3 &b) const;
};

} // namespace occ::opt
