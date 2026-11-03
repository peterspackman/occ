#pragma once
#include <occ/ints/ecp.h>
#include <vector>

namespace occ::ints::ecp_detail {

inline constexpr int nmono(int d) { return (d + 1) * (d + 2) / 2; }
/// Number of monomials of degree < d.
inline constexpr int mono_offset(int d) { return d * (d + 1) * (d + 2) / 6; }
/// Number of monomials of degree <= d.
inline constexpr int nmono_cumulative(int d) { return mono_offset(d + 1); }
inline constexpr int mono_index(int i, int j, int k) {
  return mono_offset(i + j + k) + ecp_cart_index(i, j, k);
}

/// 4 pi (a-1)!!(b-1)!!(c-1)!!/(a+b+c+1)!! for even a, b, c; zero otherwise.
double sphere_monomial_integral(int a, int b, int c);

/// Gauss-Legendre order for each radial window: ~1e-14 relative accuracy on
/// adversarial single integrals (exponents 0.03..3e4, plane-wave orders to
/// 11, r powers to 16) against a dense composite reference; doubling it
/// leaves the blocks for the def2 test systems unchanged.
inline constexpr int num_radial_points = 48;
/// The window extends sqrt(window_drop / p) beyond the integrand's extreme
/// possible peak positions on either side, so its edges are at least
/// exp(-window_drop) below the peak.
inline constexpr double window_drop = 44.0;

} // namespace occ::ints::ecp_detail

namespace occ::ints {

struct EcpEvaluator::Tables {
  Tables(int max_shell_l, int max_semilocal_l);
  int max_shell_l{0};       // basis l including derivative shifts
  int max_semilocal_l{-1};  // highest semilocal channel
  int max_lambda{0};        // highest plane-wave order
  int max_monomial_degree{0};
  /// Y[l](m, local monomial of degree l): orthonormal real harmonics.
  std::vector<Mat> Y;
  /// S[l](m, global monomial M) = int r^M Y_lm dOmega.
  std::vector<Mat> S;
  std::vector<double> leg_x, leg_w;
  std::vector<std::vector<double>> binom;
};

namespace ecp_detail {
using Tables = EcpEvaluator::Tables;
}

} // namespace occ::ints
