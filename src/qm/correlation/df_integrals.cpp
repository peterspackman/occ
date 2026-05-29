#include <occ/qm/correlation/df_integrals.h>

namespace occ::qm {

DFIntegrals::DFIntegrals(IntegralEngineDF &df_engine) : m_df(df_engine) {
  // Ensure the (μν|P) integrals are available as a dense store.
  m_df.compute_stored_integrals();
  m_nbf = m_df.ao_engine().aobasis().nbf();
  m_naux = m_df.aux_engine().nbf();
}

Mat DFIntegrals::build_b_tilde(Eigen::Ref<const Mat> C_left,
                               Eigen::Ref<const Mat> C_right) const {
  const Eigen::Index nL = C_left.cols();
  const Eigen::Index nR = C_right.cols();
  const Eigen::Index nbf = static_cast<Eigen::Index>(m_nbf);
  const Eigen::Index naux = static_cast<Eigen::Index>(m_naux);

  // (μν|P) store: column P is the nbf x nbf matrix M_P with M_P(μ,ν)=(μν|P).
  const Mat &eri3 = m_df.integral_store();

  // Half + quarter transform per auxiliary function:
  //   T_P(a,i) = C_rightᵀ M_P C_left   ->  B(i*nR + a, P) = T_P(a,i)
  Mat B(nL * nR, naux);
  for (Eigen::Index P = 0; P < naux; ++P) {
    Eigen::Map<const Mat> Mp(eri3.col(P).data(), nbf, nbf);
    Mat A = Mp * C_left; // (nbf x nL)
    // B.col(P) viewed column-major as (nR x nL) stores element (a,i) at a + i*nR
    Eigen::Map<Mat>(B.col(P).data(), nR, nL).noalias() =
        C_right.transpose() * A;
  }

  // Fold the Coulomb metric once: B̃ = B·L⁻ᵀ with V = L Lᵀ, so that
  //   (ia|jb) = Σ_P B̃(ia,P) B̃(jb,P).
  // Compute B̃ᵀ = L⁻¹ Bᵀ via a single triangular solve, then transpose.
  Mat Bt = m_df.coulomb_metric().matrixL().solve(B.transpose()); // (naux x M)
  return Bt.transpose();                                         // (M x naux)
}

} // namespace occ::qm
