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
/// The builder works on arbitrary MO coefficient blocks, so callers can batch
/// over occupied orbitals to bound memory.
class DFIntegrals {
public:
  explicit DFIntegrals(IntegralEngineDF &df_engine);

  size_t nbf() const { return m_nbf; }
  size_t naux() const { return m_naux; }

  /// Build the metric-folded B tensor for the given MO coefficient blocks.
  ///   C_left:  (nbf x nL)   C_right: (nbf x nR)
  /// Returns (nL*nR x naux); the row for (i,a) is i*nR + a (i over C_left,
  /// a over C_right).
  Mat build_b_tilde(Eigen::Ref<const Mat> C_left,
                    Eigen::Ref<const Mat> C_right) const;

private:
  IntegralEngineDF &m_df;
  size_t m_nbf{0};
  size_t m_naux{0};
};

} // namespace occ::qm
