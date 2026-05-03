#include <array>
#include <cmath>
#include <occ/xtb/sto_ng.h>
#include <stdexcept>
#include <utility>

namespace occ::xtb {
namespace detail {

// Stewart 1970 fits, generated from xtb's slater.f90 by
// scripts/extract_stong_tables.py. The data is included directly so the
// arrays have a single TU and need no extern declarations.
#include "sto_ng_data.inc"

namespace {
// (2l - 1)!! for l = 0..4. Used for Cartesian-Gaussian normalization.
constexpr double dfact_2lm1[5] = {1.0, 1.0, 3.0, 15.0, 105.0};

// xtb stores fits indexed by an integer "ityp" combining (n, l).
//   l = 0 (s):  ityp = n
//   l = 1 (p):  ityp = 4 + n
//   l = 2 (d):  ityp = 7 + n
//   l = 3 (f):  ityp = 9 + n
//   l = 4 (g):  ityp = 10 + n
// Returns -1 if the (n, l) pair is unsupported.
int ityp_index(int n, int l) {
  switch (l) {
  case 0: return n;            // 1..5
  case 1: return 4 + n;        // 6..9 (n in 2..5)
  case 2: return 7 + n;        // 10..12 (n in 3..5)
  case 3: return 9 + n;        // 13..14 (n in 4..5)
  case 4: return 10 + n;       // 15 (n = 5)
  default: return -1;
  }
}
} // namespace

} // namespace detail

StoNgFit slater_to_gauss(int ng, int n, int l, double zeta, bool normalize) {
  using namespace detail;

  if (zeta <= 0.0) {
    throw std::invalid_argument("slater_to_gauss: zeta must be positive");
  }
  if (ng < 1 || ng > 6) {
    throw std::invalid_argument("slater_to_gauss: ng must be in [1, 6]");
  }
  if (n <= l || l < 0 || l > 4) {
    throw std::invalid_argument("slater_to_gauss: invalid (n, l) — require l in [0,4] and l < n");
  }

  // n = 6 is supported only for ng = 6, l in {0, 1}; everything else
  // requires n <= 5.
  const bool n6_special = (n == 6) && (ng == 6) && (l == 0 || l == 1);
  if (n > 5 && !n6_special) {
    throw std::invalid_argument(
        "slater_to_gauss: n > 5 only supported for (n=6, l in {0,1}, ng=6)");
  }

  StoNgFit fit;
  fit.alpha.resize(ng);
  fit.coeff.resize(ng);

  if (n6_special) {
    if (l == 0) {
      for (int i = 0; i < ng; ++i) {
        fit.alpha[i] = alpha_ng6_6s[i] * zeta * zeta;
        fit.coeff[i] = coeff_ng6_6s[i];
      }
    } else { // l == 1
      for (int i = 0; i < ng; ++i) {
        fit.alpha[i] = alpha_ng6_6p[i] * zeta * zeta;
        fit.coeff[i] = coeff_ng6_6p[i];
      }
    }
  } else {
    int ityp = ityp_index(n, l) - 1; // convert to 0-based
    if (ityp < 0 || ityp >= 15) {
      throw std::invalid_argument(
          "slater_to_gauss: no Stewart fit for this (n, l)");
    }
    switch (ng) {
    case 1:
      fit.alpha[0] = alpha_ng1[ityp] * zeta * zeta;
      fit.coeff[0] = 1.0;
      break;
    case 2:
      for (int i = 0; i < 2; ++i) {
        fit.alpha[i] = alpha_ng2[ityp][i] * zeta * zeta;
        fit.coeff[i] = coeff_ng2[ityp][i];
      }
      break;
    case 3:
      for (int i = 0; i < 3; ++i) {
        fit.alpha[i] = alpha_ng3[ityp][i] * zeta * zeta;
        fit.coeff[i] = coeff_ng3[ityp][i];
      }
      break;
    case 4:
      for (int i = 0; i < 4; ++i) {
        fit.alpha[i] = alpha_ng4[ityp][i] * zeta * zeta;
        fit.coeff[i] = coeff_ng4[ityp][i];
      }
      break;
    case 5:
      for (int i = 0; i < 5; ++i) {
        fit.alpha[i] = alpha_ng5[ityp][i] * zeta * zeta;
        fit.coeff[i] = coeff_ng5[ityp][i];
      }
      break;
    case 6:
      for (int i = 0; i < 6; ++i) {
        fit.alpha[i] = alpha_ng6[ityp][i] * zeta * zeta;
        fit.coeff[i] = coeff_ng6[ityp][i];
      }
      break;
    }
  }

  if (normalize) {
    // Cartesian-Gaussian primitive normalization, matching xtb's
    // slaterToGauss with norm = .true.:
    //   c_i *= (2 alpha_i / pi)^(3/4) * (4 alpha_i)^(l/2) / sqrt((2l - 1)!!)
    constexpr double two_over_pi = 2.0 / M_PI;
    const double inv_sqrt_dfact = 1.0 / std::sqrt(dfact_2lm1[l]);
    for (int i = 0; i < ng; ++i) {
      const double a = fit.alpha[i];
      double f = std::pow(two_over_pi * a, 0.75);
      if (l > 0) {
        f *= std::pow(std::sqrt(4.0 * a), l);
      }
      fit.coeff[i] *= f * inv_sqrt_dfact;
    }
  }

  return fit;
}

} // namespace occ::xtb
