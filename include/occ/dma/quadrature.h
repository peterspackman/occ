#pragma once
#include <occ/core/linear_algebra.h>
#include <utility>

namespace occ::dma {

/**
 * @brief Class to compute and store binomial coefficients
 */
class BinomialCoefficients {
public:
  /**
   * @brief Constructor
   * @param max_order Maximum order of binomial coefficients to compute
   */
  BinomialCoefficients(int max_order = 20);

  /**
   * @brief Get binomial coefficient (n choose k)
   * @param k First parameter
   * @param m Second parameter
   * @return Binomial coefficient
   */
  double binomial(int k, int m) const;

  /**
   * @brief Get square root of binomial coefficient
   * @param k First parameter
   * @param m Second parameter
   * @return Square root of binomial coefficient
   */
  double sqrt_binomial(int k, int m) const;

  /**
   * @brief Get matrix of all binomial coefficients
   * @return Reference to matrix of binomial coefficients
   */
  const Mat &binomial_matrix() const;

  /**
   * @brief Get matrix of all square roots of binomial coefficients
   * @return Reference to matrix of square roots of binomial coefficients
   */
  const Mat &sqrt_binomial_matrix() const;

private:
  int m_max_order;
  Mat m_binomial;
  Mat m_sqrt_binomial;
};

/**
 * @brief Compute Gauss-Hermite quadrature points and weights
 * @param n Number of quadrature points
 * @return Pair of vectors containing points and weights
 */
std::pair<Vec, Vec> compute_gauss_hermite(int n);

/**
 * @brief Class for Gauss-Hermite quadrature
 */
class GaussHermite {
public:
  /**
   * @brief Constructor
   * @param n Number of quadrature points
   */
  GaussHermite(int n);

  /**
   * @brief Get quadrature points
   * @return Vector of quadrature points
   */
  const Vec &points() const;

  /**
   * @brief Get quadrature weights
   * @return Vector of quadrature weights
   */
  const Vec &weights() const;

  /**
   * @brief Get number of quadrature points
   * @return Number of points
   */
  int size() const;

  /**
   * @brief Validate computed values against hardcoded ones
   * @param n Number of quadrature points
   * @param tolerance Tolerance for comparison
   * @return True if computed values match hardcoded ones within tolerance
   */
  static bool validate_computed_values(int n, double tolerance = 1e-10);

private:
  Vec m_points;
  Vec m_weights;
};

} // namespace occ::dma
