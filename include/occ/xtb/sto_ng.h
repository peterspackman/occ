#pragma once
#include <vector>

namespace occ::xtb {

// Coefficients are taken from:
//   R. F. Stewart, Small Gaussian Expansions of Slater-Type Orbitals,
//   J. Chem. Phys. 52, 431 (1970).  DOI: 10.1063/1.1672702
// Re-uses the encoding from Grimme's xtb (src/slater.f90).
//
// Restrictions: 1 <= NG <= 6, l in [0..4], n in [1..5] (with the special-case
// (n=6, l=0) and (n=6, l=1) supported only for NG=6).

struct StoNgFit {
  std::vector<double> alpha; // primitive exponents (already scaled by zeta^2)
  std::vector<double> coeff; // contraction coefficients
};

// Expand a Slater-type orbital with principal QN n, angular momentum l, and
// exponent zeta as a sum of `ng` primitive Gaussians. If `normalize` is true,
// each contraction coefficient is multiplied by the Cartesian-Gaussian
// self-normalization factor (matching xtb's slaterToGauss output).
StoNgFit slater_to_gauss(int ng, int n, int l, double zeta,
                         bool normalize = true);

} // namespace occ::xtb
