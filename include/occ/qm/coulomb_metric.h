#pragma once
#include <Eigen/Cholesky>
#include <occ/core/linear_algebra.h>

namespace occ::qm {

/**
 * @brief Factorization of a density-fitting Coulomb metric V = (P|Q).
 *
 * Cholesky whenever V is numerically positive definite, otherwise a symmetric
 * eigendecomposition with the near-null space discarded. Two situations need
 * the second path, and both otherwise fail badly:
 *
 *  - an auxiliary basis that is close to linearly dependent for the given
 *    geometry (Eigen's LLT then reports failure, and solving with the failed
 *    factorization returns garbage);
 *  - the long-range metric (P|erf(omega r)/r|Q) of a range-separated hybrid,
 *    whose attenuated kernel damps the high-exponent auxiliary functions so
 *    hard that a JK-fitting basis is near-singular under it.
 */
class CoulombMetric {
public:
  /// Factorize V. `lindep` is relative to the largest eigenvalue.
  void compute(const Mat &V, double lindep = 1e-12);

  inline bool uses_cholesky() const { return m_cholesky; }
  inline Eigen::Index num_discarded() const { return m_discarded; }
  inline bool initialized() const { return m_size > 0; }

  /// V^-1 b
  template <typename Derived>
  Mat solve(const Eigen::MatrixBase<Derived> &b) const {
    if (m_cholesky)
      return m_llt.solve(b);
    return m_half * (m_half.transpose() * b);
  }

  /// L^-1 b, where L L^T = V (or its eigen equivalent U diag(w)^-1/2): the
  /// half-inverse that turns 3-centre integrals into DF B tensors. Has fewer
  /// rows than naux when the eigen path drops vectors.
  template <typename Derived>
  Mat half_inverse_apply(const Eigen::MatrixBase<Derived> &b) const {
    if (m_cholesky)
      return m_llt.matrixL().solve(Mat(b));
    return m_half.transpose() * b;
  }

private:
  bool m_cholesky{true};
  Eigen::Index m_size{0};
  Eigen::Index m_discarded{0};
  Eigen::LLT<Mat> m_llt;
  Mat m_half; ///< U diag(1/sqrt(w)) over the retained eigenvectors
};

} // namespace occ::qm
