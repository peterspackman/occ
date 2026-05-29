#include <occ/core/log.h>
#include <occ/qm/correlation/df_integrals.h>

namespace occ::qm {

DFIntegrals::DFIntegrals(IntegralEngineDF &df_engine, size_t memory_budget)
    : m_df(df_engine) {
  m_nbf = m_df.ao_engine().aobasis().nbf();
  m_naux = m_df.aux_engine().nbf();

  // The dense (μν|P) store costs nbf²·naux doubles. If it comfortably fits the
  // budget, build it once and reuse across blocks (faster); otherwise build B
  // integral-direct per call to keep memory bounded.
  const size_t store_bytes = m_nbf * m_nbf * m_naux * sizeof(double);
  m_direct = store_bytes > memory_budget / 2;
  if (!m_direct)
    m_df.compute_stored_integrals();

  occ::log::debug(
      "DFIntegrals: {} ({} aux fns, (μν|P) store {:.1f} MiB, budget {:.1f} MiB)",
      m_direct ? "integral-direct" : "stored 3-center", m_naux,
      store_bytes / 1048576.0, memory_budget / 1048576.0);
}

Mat DFIntegrals::build_b_stored(Eigen::Ref<const Mat> C_left,
                                Eigen::Ref<const Mat> C_right) const {
  const Eigen::Index nL = C_left.cols();
  const Eigen::Index nR = C_right.cols();
  const Eigen::Index nbf = static_cast<Eigen::Index>(m_nbf);
  const Eigen::Index naux = static_cast<Eigen::Index>(m_naux);
  const Mat &eri3 = m_df.integral_store(); // (nbf*nbf x naux), col P = (μν|P)

  Mat B(nL * nR, naux);
  for (Eigen::Index P = 0; P < naux; ++P) {
    Eigen::Map<const Mat> Mp(eri3.col(P).data(), nbf, nbf);
    Mat A = Mp * C_left; // (nbf x nL)
    Eigen::Map<Mat>(B.col(P).data(), nR, nL).noalias() =
        C_right.transpose() * A;
  }
  return B;
}

Mat DFIntegrals::build_b_tilde(Eigen::Ref<const Mat> C_left,
                               Eigen::Ref<const Mat> C_right) const {
  Mat B = m_direct ? m_df.build_b_direct(C_left, C_right)
                   : build_b_stored(C_left, C_right);
  // Fold the Coulomb metric once: B̃ = B·L⁻ᵀ with V = L Lᵀ, so that
  //   (ia|jb) = Σ_P B̃(ia,P) B̃(jb,P).
  // B̃ᵀ = L⁻¹ Bᵀ via a single triangular solve.
  Mat Bt = m_df.coulomb_metric().matrixL().solve(B.transpose()); // (naux x M)
  return Bt.transpose();                                         // (M x naux)
}

} // namespace occ::qm
