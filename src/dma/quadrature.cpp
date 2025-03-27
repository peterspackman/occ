#include <cmath>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/dma/quadrature.h>
#include <stdexcept>

namespace occ::dma {

BinomialCoefficients::BinomialCoefficients(int max_order)
    : m_max_order(max_order) {

  // Initialize the binomial coefficient matrices
  m_binomial = Mat::Zero(max_order + 1, max_order + 1);
  m_sqrt_binomial = Mat::Zero(max_order + 1, max_order + 1);

  // Base case
  for (int k = 0; k <= max_order; k++) {
    m_binomial(k, 0) = 1.0;
    m_sqrt_binomial(k, 0) = 1.0;
  }

  // Compute Pascal's triangle
  for (int k = 1; k <= max_order; k++) {
    for (int m = 1; m <= k; m++) {
      m_binomial(k, m) = m_binomial(k - 1, m - 1) + m_binomial(k - 1, m);
      m_sqrt_binomial(k, m) = std::sqrt(m_binomial(k, m));
    }
  }
}

double BinomialCoefficients::binomial(int k, int m) const {
  if (k < 0 || m < 0 || m > k || k > m_max_order) {
    return 0.0;
  }
  return m_binomial(k, m);
}

double BinomialCoefficients::sqrt_binomial(int k, int m) const {
  if (k < 0 || m < 0 || m > k || k > m_max_order) {
    return 0.0;
  }
  return m_sqrt_binomial(k, m);
}

const Mat &BinomialCoefficients::binomial_matrix() const { return m_binomial; }

const Mat &BinomialCoefficients::sqrt_binomial_matrix() const {
  return m_sqrt_binomial;
}

std::pair<Vec, Vec> compute_gauss_hermite(int n) {
  if (n < 1) {
    throw std::invalid_argument("n must be a positive integer");
  }

  // Create vectors for points and weights
  Vec points = Vec::Zero(n);
  Vec weights = Vec::Zero(n);

  // For Hermite polynomials:
  // a_n = 0
  // b_n = sqrt(n/2)

  // Create the symmetric tridiagonal matrix
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(n, n);

  // Off-diagonal elements (use only upper diagonal)
  for (int i = 0; i < n - 1; i++) {
    J(i, i + 1) = std::sqrt((i + 1) / 2.0);
  }

  // The matrix is symmetric, so we can use a specialized eigenvalue solver
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(J);

  // The eigenvalues are the points (initial guess)
  points = solver.eigenvalues();

  // Improve the zeros by a Newton iteration
  for (int i = 0; i < n; i++) {
    double x = points(i);

    // Newton iteration
    // Evaluate Hermite polynomial and its derivative at x
    double p0 = 1.0;     // H_0(x)
    double p1 = 2.0 * x; // H_1(x)
    double dp = 0.0;

    // Recurrence relation for Hermite polynomials
    for (int j = 1; j < n; j++) {
      double p2 = 2.0 * x * p1 - 2.0 * j * p0;
      p0 = p1;
      p1 = p2;
    }

    // Derivative of H_n(x) is 2n*H_{n-1}(x)
    dp = 2.0 * n * p0;

    // Newton step
    points(i) = x - p1 / dp;
  }

  // Calculate weights using formula w_i = sqrt(π)/[H'_n(x_i)]²
  const double sqrt_pi = std::sqrt(M_PI);
  for (int i = 0; i < n; i++) {
    double x = points(i);
    double p0 = 1.0;
    double p1 = 2.0 * x;

    for (int j = 1; j < n; j++) {
      double p2 = 2.0 * x * p1 - 2.0 * j * p0;
      p0 = p1;
      p1 = p2;
    }

    double dp = 2.0 * n * p0;
    weights(i) = sqrt_pi / (dp * dp);
  }

  return {points, weights};
}

GaussHermite::GaussHermite(int n) {
  if (n < 1 || n > 20) {
    throw std::invalid_argument(
        "Gauss-Hermite quadrature requires 1 <= n <= 20");
  }

  // First try to use the hardcoded values for better precision
  bool use_hardcoded = true;

  if (use_hardcoded) {
    m_points = Vec::Zero(n);
    m_weights = Vec::Zero(n);

    if (n == 1) {
      m_points(0) = 0.00000000000000;
      m_weights(0) = 1.77245385090552;
    } else if (n == 2) {
      m_points(0) = -0.70710678118654;
      m_points(1) = 0.70710678118654;
      m_weights(0) = 0.88622692545275;
      m_weights(1) = 0.88622692545275;
    } else if (n == 3) {
      m_points(0) = -1.22474487139159;
      m_points(1) = 0.00000000000000;
      m_points(2) = 1.22474487139159;
      m_weights(0) = 0.29540897515092;
      m_weights(1) = 1.18163590060368;
      m_weights(2) = 0.29540897515092;
    } else if (n == 4) {
      m_points(0) = -1.65068012388578;
      m_points(1) = -0.52464762327529;
      m_points(2) = 0.52464762327529;
      m_points(3) = 1.65068012388578;
      m_weights(0) = 0.08131283544724;
      m_weights(1) = 0.80491409000551;
      m_weights(2) = 0.80491409000551;
      m_weights(3) = 0.08131283544724;
    } else if (n == 5) {
      m_points(0) = -2.02018287045608;
      m_points(1) = -0.95857246461381;
      m_points(2) = 0.00000000000000;
      m_points(3) = 0.95857246461381;
      m_points(4) = 2.02018287045608;
      m_weights(0) = 0.01995324205904;
      m_weights(1) = 0.39361932315224;
      m_weights(2) = 0.94530872048294;
      m_weights(3) = 0.39361932315224;
      m_weights(4) = 0.01995324205904;
    } else if (n == 6) {
      m_points(0) = -2.35060497367449;
      m_points(1) = -1.33584907401370;
      m_points(2) = -0.43607741192761;
      m_points(3) = 0.43607741192761;
      m_points(4) = 1.33584907401370;
      m_points(5) = 2.35060497367449;
      m_weights(0) = 0.00453000990550;
      m_weights(1) = 0.15706732032285;
      m_weights(2) = 0.72462959522439;
      m_weights(3) = 0.72462959522439;
      m_weights(4) = 0.15706732032285;
      m_weights(5) = 0.00453000990550;
    } else if (n == 7) {
      m_points(0) = -2.65196135683523;
      m_points(1) = -1.67355162876747;
      m_points(2) = -0.81628788285896;
      m_points(3) = 0.00000000000000;
      m_points(4) = 0.81628788285896;
      m_points(5) = 1.67355162876747;
      m_points(6) = 2.65196135683523;
      m_weights(0) = 0.00097178124509;
      m_weights(1) = 0.05451558281912;
      m_weights(2) = 0.42560725261012;
      m_weights(3) = 0.81026461755680;
      m_weights(4) = 0.42560725261012;
      m_weights(5) = 0.05451558281912;
      m_weights(6) = 0.00097178124509;
    } else if (n == 8) {
      m_points(0) = -2.93063742025724;
      m_points(1) = -1.98165675669584;
      m_points(2) = -1.15719371244678;
      m_points(3) = -0.38118699020732;
      m_points(4) = 0.38118699020732;
      m_points(5) = 1.15719371244678;
      m_points(6) = 1.98165675669584;
      m_points(7) = 2.93063742025724;
      m_weights(0) = 0.00019960407221;
      m_weights(1) = 0.01707798300741;
      m_weights(2) = 0.20780232581489;
      m_weights(3) = 0.66114701255824;
      m_weights(4) = 0.66114701255824;
      m_weights(5) = 0.20780232581489;
      m_weights(6) = 0.01707798300741;
      m_weights(7) = 0.00019960407221;
    } else if (n == 9) {
      m_points(0) = -3.19099320178152;
      m_points(1) = -2.26658058453184;
      m_points(2) = -1.46855328921666;
      m_points(3) = -0.72355101875283;
      m_points(4) = 0.00000000000000;
      m_points(5) = 0.72355101875283;
      m_points(6) = 1.46855328921666;
      m_points(7) = 2.26658058453184;
      m_points(8) = 3.19099320178152;
      m_weights(0) = 0.00003960697726;
      m_weights(1) = 0.00494362427553;
      m_weights(2) = 0.08847452739437;
      m_weights(3) = 0.43265155900255;
      m_weights(4) = 0.72023521560605;
      m_weights(5) = 0.43265155900255;
      m_weights(6) = 0.08847452739437;
      m_weights(7) = 0.00494362427553;
      m_weights(8) = 0.00003960697726;
    } else if (n == 10) {
      m_points(0) = -3.43615911883773;
      m_points(1) = -2.53273167423278;
      m_points(2) = -1.75668364929988;
      m_points(3) = -1.03661082978951;
      m_points(4) = -0.34290132722370;
      m_points(5) = 0.34290132722370;
      m_points(6) = 1.03661082978951;
      m_points(7) = 1.75668364929988;
      m_points(8) = 2.53273167423278;
      m_points(9) = 3.43615911883773;
      m_weights(0) = 0.00000764043285;
      m_weights(1) = 0.00134364574678;
      m_weights(2) = 0.03387439445548;
      m_weights(3) = 0.24013861108231;
      m_weights(4) = 0.61086263373532;
      m_weights(5) = 0.61086263373532;
      m_weights(6) = 0.24013861108231;
      m_weights(7) = 0.03387439445548;
      m_weights(8) = 0.00134364574678;
      m_weights(9) = 0.00000764043285;
    } else {
      // For n > 10, compute the points and weights
      use_hardcoded = false;
    }
  }

  if (!use_hardcoded) {
    // Compute the points and weights using the algorithm
    auto result = compute_gauss_hermite(n);
    m_points = result.first;
    m_weights = result.second;
  }
}

const Vec &GaussHermite::points() const { return m_points; }

const Vec &GaussHermite::weights() const { return m_weights; }

int GaussHermite::size() const { return m_points.size(); }

// Method to compute and compare with hardcoded values
bool GaussHermite::validate_computed_values(int n, double tolerance) {
  if (n < 1 || n > 10) {
    throw std::invalid_argument("Validation only works for 1 <= n <= 10");
  }

  // Get hardcoded values
  GaussHermite hardcoded(n);

  // Compute values using algorithm
  auto computed = compute_gauss_hermite(n);

  occ::log::error("N = {}", n);
  // Compare points
  for (int i = 0; i < n; i++) {
    occ::log::error("Point {}: hardcoded = {}, computed = {}", i,
                    hardcoded.points()(i), computed.first(i));
    if (std::abs(computed.first(i) - hardcoded.points()(i)) > tolerance) {
      occ::log::error("Point {} mismatch: hardcoded = {}, computed = {}", i,
                      hardcoded.points()(i), computed.first(i));
      return false;
    }
  }

  // Compare weights
  for (int i = 0; i < n; i++) {
    occ::log::error("Weight {}: hardcoded = {}, computed = {}", i,
                    hardcoded.weights()(i), computed.second(i));
    if (std::abs(computed.second(i) - hardcoded.weights()(i)) > tolerance) {
      occ::log::error("Weight {} mismatch: hardcoded = {}, computed = {}", i,
                      hardcoded.weights()(i), computed.second(i));
      return false;
    }
  }

  return true;
}



} // namespace occ::dma
