#include <occ/core/log.h>
#include <occ/qm/correlation/df_integrals.h>

namespace occ::qm {

DFIntegrals::DFIntegrals(IntegralEngineDF &df_engine, size_t memory_budget)
    : m_df(df_engine), m_store_limit(memory_budget / 2) {
  m_nbf = m_df.ao_engine().aobasis().nbf();
  m_naux = m_df.aux_engine().nbf();

  // The packed (μν|P) store is shared with the SCF on the same engine: reuse
  // it when that already built it whole, else build it if it comfortably
  // fits the budget; otherwise build B integral-direct per call to keep
  // memory bounded.
  m_direct = !m_df.ensure_complete_packed_store(m_store_limit);

  occ::log::debug(
      "DFIntegrals: {} ({} aux fns, packed (μν|P) store {:.1f} MiB, budget "
      "{:.1f} MiB)",
      m_direct ? "integral-direct" : "stored 3-center", m_naux,
      m_df.packed_storage_bytes() / 1048576.0, memory_budget / 1048576.0);
}

Mat DFIntegrals::build_b_tilde(Eigen::Ref<const Mat> C_left,
                               Eigen::Ref<const Mat> C_right) const {
  // A complete store already has the metric folded in. Checked again here,
  // not only at construction: a policy change on the engine drops its store.
  if (!m_direct && m_df.ensure_complete_packed_store(m_store_limit))
    return m_df.packed_store().half_transform(C_left, C_right);
  Mat B = m_df.build_b_direct(C_left, C_right);
  // Fold the Coulomb metric once: B̃ = B·L⁻ᵀ with V = L Lᵀ, so that
  //   (ia|jb) = Σ_P B̃(ia,P) B̃(jb,P).
  // If the metric was too ill-conditioned for a Cholesky the half-inverse
  // comes from its eigendecomposition instead, with fewer than naux rows.
  // The identity above holds either way.
  Mat Bt = m_df.coulomb_metric().half_inverse_apply(B.transpose()); // (naux x M)
  return Bt.transpose();                                            // (M x naux)
}

} // namespace occ::qm
