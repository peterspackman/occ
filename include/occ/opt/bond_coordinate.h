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

namespace occ::opt {

class BondCoordinate {
public:
  enum class Type { COVALENT, VDW };
  
  int i, j;
  Type bond_type;

  BondCoordinate(int i_in, int j_in, Type type = Type::COVALENT);

  // Returns the bond distance value
  double operator()(const occ::Mat3N &coords) const;

  // Returns the gradient matrix (3 x 2, columns are gradients w.r.t. atoms i and j)
  occ::Mat3N gradient(const occ::Mat3N &coords) const;

  // Hessian diagonal element for this coordinate
  double hessian(const occ::Mat &rho) const;

  // Weight for this coordinate  
  double weight(const occ::Mat &rho) const;
};

} // namespace occ::opt
