#include <Eigen/Eigenvalues>
#include <occ/core/log.h>
#include <occ/qm/coulomb_metric.h>
#include <stdexcept>

namespace occ::qm {

void CoulombMetric::compute(const Mat &V, double lindep) {
  m_size = V.rows();
  m_discarded = 0;
  m_llt = Eigen::LLT<Mat>(V);
  if (m_llt.info() == Eigen::Success) {
    m_cholesky = true;
    m_half.resize(0, 0);
    return;
  }
  // Not numerically positive definite: fall back to a symmetric
  // eigendecomposition and drop the near-null space.
  m_cholesky = false;
  Eigen::SelfAdjointEigenSolver<Mat> es(V);
  if (es.info() != Eigen::Success)
    throw std::runtime_error(
        "Eigendecomposition of the density-fitting Coulomb metric failed");
  const Vec &w = es.eigenvalues();
  const double cutoff = lindep * std::max(w.maxCoeff(), 0.0);
  Eigen::Index nkept = 0;
  for (Eigen::Index i = 0; i < w.size(); ++i)
    if (w(i) > cutoff)
      ++nkept;
  if (nkept == 0)
    throw std::runtime_error(
        "Density-fitting Coulomb metric has no positive eigenvalues");
  m_discarded = w.size() - nkept;
  m_half.resize(w.size(), nkept);
  Eigen::Index col = 0;
  for (Eigen::Index i = 0; i < w.size(); ++i) {
    if (w(i) <= cutoff)
      continue;
    m_half.col(col++) = es.eigenvectors().col(i) / std::sqrt(w(i));
  }
  occ::log::debug("Coulomb metric eigendecomposition: discarded {} of {} "
                  "vectors (eigenvalue range [{:.3e}, {:.3e}])",
                  m_discarded, w.size(), w.minCoeff(), w.maxCoeff());
}

} // namespace occ::qm
