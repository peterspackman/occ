#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::qm::cc {

/// Least-squares "Laplace" quadrature for the inverse function on a range:
///   1/x  ~=  sum_k weights(k) * exp(-x * points(k))   for x in [xmin, xmax].
/// Used to make the MP2 energy denominator 1/(e_a+e_b-e_i-e_j) separable so the
/// THC factors can collapse the orbital sums. All points and weights are
/// strictly positive (no catastrophic cancellation).
struct LaplaceGrid {
  Vec points;  ///< t_k  (length n)
  Vec weights; ///< w_k  (length n)
  int size() const { return static_cast<int>(points.size()); }
};

/// Build an n-point Laplace grid approximating 1/x over [xmin, xmax]
/// (require 0 < xmin <= xmax). Derivation: the canonical inverse on [1,R] with
/// R=xmax/xmin satisfies 1/x' = \int_0^1 u^{x'-1} du; applying n-point
/// Gauss-Legendre on [0,1] and undoing the x'=x/xmin scaling gives
///   t_k = -ln(u_k)/xmin,  w_k = omega_k/(u_k*xmin).
LaplaceGrid laplace_grid(double xmin, double xmax, int n);

/// Max relative error  max_x |1 - x * sum_k w_k e^{-x t_k}|  of `grid`,
/// sampled on `n_sample` log-spaced x in [xmin, xmax] (testing / diagnostics).
double laplace_max_rel_error(const LaplaceGrid &grid, double xmin, double xmax,
                             int n_sample = 64);

} // namespace occ::qm::cc
