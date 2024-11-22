#include <cmath>
#include <occ/core/log.h>
#include <occ/sht/quadrature.h>

namespace occ::sht {

std::pair<Vec, Vec> gauss_legendre_quadrature(int N) {
  Vec roots(N), weights(N);
  gauss_legendre_quadrature(roots, weights, N);
  return {roots, weights};
}

void gauss_legendre_quadrature(Vec &roots, Vec &weights, int N) {

  const double eps{2.3e-16};
  const long m = (N + 1) / 2;
  for (long i = 0; i < m; ++i) {
    double z1, deriv, point0, point1;
    // maximum Newton iteration count
    int iteration_count = 10;
    // initial guess
    double z = (1.0 - (N - 1) / (8.0 * N * N * N)) *
               std::cos((M_PI * (4 * i + 3)) / (4.0 * N + 2));
    do {
      point1 = z;   // P_1
      point0 = 1.0; // P_0
      for (long l = 2; l <= N; ++l) {
        // recurrence : l P_l = (2l-1) z P_{l-1} - (l-1) P_{l-2}
        // (works ok up to l=100000)
        double point3 = point0;
        point0 = point1;
        // Legendre polynomial
        point1 = ((2 * l - 1) * z * point0 - (l - 1) * point3) / l;
      }
      // Approximate derivative of Legendre Polynomial
      deriv = N * (point0 - z * point1);
      z1 = z;
      // Newton's method step
      z -= point1 * (1.0 - z * z) / deriv;
    } while ((std::fabs(z - z1) > (z1 + z) * 0.5 * eps) &&
             (--iteration_count > 0));

    if (iteration_count == 0)
      occ::log::warn("Iterations exceeded when finding Gauss-Legendre roots");

    double s2 = 1.0 - z * z;
    // Build up the abscissas.
    roots(i) = z;
    roots(N - 1 - i) = -z;
    // Build up the weights.
    weights(i) = 2.0 * s2 / (deriv * deriv);
    weights(N - 1 - i) = weights(i);
  }
  // if n is even
  if (N & 1) {
    roots(N / 2) = 0.0; // exactly zero.
    weights(N / 2) = 1.0;
    double point0 = 1.0; // P_0
    for (long l = 2; l <= N; l += 2) {
      // recurrence : l P_l = (2l-1) z P_{l-1} - (l-1) P_{l-2}	(works
      // ok up to l=100000) The Legendre polynomial...
      point0 *= (1.0 - l) / l;
    }
    // ... and its inverse derivative.
    double deriv = 1.0 / (N * point0);
    weights(N / 2) = 2.0 * deriv * deriv;
  }
  // as we started with initial guesses, we should check if the gauss points
  // are actually unique and ordered.
  for (long i = m - 1; i > 0; i--) {
    if (roots(i) >= roots(i - 1))
      occ::log::error("Invalid Gauss-Legendre points");
  }
}

} // namespace occ::sht
