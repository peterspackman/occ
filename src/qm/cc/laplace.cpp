#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <occ/qm/cc/laplace.h>
#include <stdexcept>
#include <utility>

namespace occ::qm::cc {

// Least-squares Laplace quadrature for 1/x on [xmin, xmax]. The nodes t_k are
// FIXED, log-spaced over the range of denominator peaks (the integrand
// e^{-x t} of 1/x = \int_0^\infty e^{-x t} dt peaks at t = 1/x, so over
// x in [xmin,xmax] the peaks span [1/xmax, 1/xmin]; nodes bracket that with
// margin). The weights are then the linear least-squares solution of
//   minimise_w  sum_m ( 1 - x_m sum_k w_k e^{-x_m t_k} )^2
// over a dense log grid {x_m} -- a relative-error fit. Linear (well-conditioned
// via QR), monotone in n, and needs no nonlinear optimisation. Nonlinear
// minimax would shave the point count, but the THC factorisation error (~1e-3)
// dominates the Laplace error here long before that matters.

LaplaceGrid laplace_grid(double xmin, double xmax, int n) {
  if (n <= 0)
    throw std::invalid_argument("laplace_grid: need n > 0");
  if (!(xmin > 0.0))
    throw std::invalid_argument("laplace_grid: need xmin > 0");
  if (xmax < xmin)
    std::swap(xmin, xmax);

  // Fixed log-spaced nodes bracketing [1/xmax, 1/xmin] with a few decades of
  // margin on each side so the least-squares basis spans the tails.
  const double t_lo = 0.1 / xmax;
  const double t_hi = 30.0 / xmin;
  LaplaceGrid g;
  g.points.resize(n);
  if (n == 1) {
    g.points(0) = std::sqrt(t_lo * t_hi);
  } else {
    const double dl = std::log(t_hi / t_lo) / (n - 1);
    for (int k = 0; k < n; ++k)
      g.points(k) = t_lo * std::exp(dl * k);
  }

  // Dense log grid of x for the relative-error fit, and design matrix
  // A(m,k) = x_m e^{-x_m t_k}; solve A w = 1 in least squares.
  const int M = std::max(8 * n, 200);
  const double lmin = std::log(xmin), lmax = std::log(xmax);
  Mat A(M, n);
  Vec rhs = Vec::Ones(M);
  for (int m = 0; m < M; ++m) {
    const double frac = (M == 1) ? 0.0 : static_cast<double>(m) / (M - 1);
    const double x = std::exp(lmin + frac * (lmax - lmin));
    for (int k = 0; k < n; ++k)
      A(m, k) = x * std::exp(-x * g.points(k));
  }
  g.weights = A.colPivHouseholderQr().solve(rhs);
  return g;
}

double laplace_max_rel_error(const LaplaceGrid &grid, double xmin, double xmax,
                             int n_sample) {
  if (n_sample < 2)
    n_sample = 2;
  const double lmin = std::log(xmin);
  const double lmax = std::log(xmax);
  double worst = 0.0;
  for (int s = 0; s < n_sample; ++s) {
    const double frac = static_cast<double>(s) / (n_sample - 1);
    const double x = std::exp(lmin + frac * (lmax - lmin));
    double approx = 0.0;
    for (int k = 0; k < grid.size(); ++k)
      approx += grid.weights(k) * std::exp(-x * grid.points(k));
    worst = std::max(worst, std::abs(1.0 - x * approx));
  }
  return worst;
}

} // namespace occ::qm::cc
