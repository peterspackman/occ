#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/integral_engine_df.h>

namespace occ::qm {

/// Builds metric-folded density-fitting "B" tensors for correlated methods.
///
/// Given the 3-center AO integrals (μν|P) and the Coulomb metric V=(P|Q) owned
/// by an IntegralEngineDF, build_b_tilde returns B̃ such that
///   (ia|jb) = Σ_P B̃(ia,P) B̃(jb,P)
/// i.e. the metric is folded in once (B̃ = B·L⁻ᵀ with V = L Lᵀ), so downstream
/// code recovers MO integrals with a single GEMM and no per-pair metric solves.
///
/// Depending on the memory budget, the 3-center integrals are either stored
/// densely once and reused (fast, O(nbf²·naux) memory) or streamed
/// integral-direct per call (bounded memory, no dense store). Either way the
/// returned B̃ is identical.
class DFIntegrals {
public:
  /// memory_budget (bytes): if the dense (μν|P) store comfortably fits, it is
  /// built once and reused; otherwise B is built integral-direct per call.
  DFIntegrals(IntegralEngineDF &df_engine, size_t memory_budget);

  size_t nbf() const { return m_nbf; }
  size_t naux() const { return m_naux; }
  /// True if B is built integral-direct (no dense (μν|P) store).
  bool uses_direct() const { return m_direct; }

  /// Build the metric-folded B tensor for the given MO coefficient blocks.
  ///   C_left:  (nbf x nL)   C_right: (nbf x nR)
  /// Returns (nL*nR x naux); the row for (i,a) is i*nR + a.
  Mat build_b_tilde(Eigen::Ref<const Mat> C_left,
                    Eigen::Ref<const Mat> C_right) const;

private:
  Mat build_b_stored(Eigen::Ref<const Mat> C_left,
                     Eigen::Ref<const Mat> C_right) const;

  IntegralEngineDF &m_df;
  size_t m_nbf{0};
  size_t m_naux{0};
  bool m_direct{false};
};

} // namespace occ::qm
