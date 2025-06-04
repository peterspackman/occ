#include <occ/dma/binomial.h>

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

} // namespace occ::dma
