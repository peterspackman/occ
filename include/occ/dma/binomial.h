#pragma once
#include <occ/core/linear_algebra.h>

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
  int m_max_order{20};
  Mat m_binomial;
  Mat m_sqrt_binomial;
};

} // namespace occ::dma
