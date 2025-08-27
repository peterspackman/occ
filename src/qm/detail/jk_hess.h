#include "kernel_traits.h"
#include <occ/qm/integral_engine.h>

namespace occ::qm::detail {

// Coulomb (J) Hessian inner loop functions for different spin cases
inline void j_inner_r_hess(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                           int bf1, int bf2, int bf3, double value) {
  J(bf0, bf1) += D(bf2, bf3) * value;
}

inline void j_inner_g_hess(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                           int bf1, int bf2, int bf3, double value) {
  auto Jaa = occ::qm::block::aa(J);
  auto Jbb = occ::qm::block::bb(J);
  const auto Daa = occ::qm::block::aa(D);
  const auto Dbb = occ::qm::block::bb(D);
  Jaa(bf0, bf1) += Daa(bf2, bf3) * value;
  Jbb(bf0, bf1) += Dbb(bf2, bf3) * value;
}

inline void j_inner_u_hess(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                           int bf1, int bf2, int bf3, double value) {
  auto Ja = occ::qm::block::a(J);
  auto Jb = occ::qm::block::b(J);
  const auto Da = occ::qm::block::a(D);
  const auto Db = occ::qm::block::b(D);
  Ja(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
  Jb(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
}

// Exchange (K) Hessian inner loop functions for different spin cases
// NOTE: Exchange Hessian is more complex and involves additional derivatives
// For now, provide placeholders that can be implemented later
inline void k_inner_r_hess(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> K, int bf0,
                           int bf1, int bf2, int bf3, double value) {
  // TODO: Implement exchange Hessian for restricted case
  // This involves second derivatives of exchange integrals which is more complex
  K(bf0, bf2) -= D(bf1, bf3) * value;
  K(bf0, bf3) -= D(bf1, bf2) * value;
}

inline void k_inner_g_hess(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> K, int bf0,
                           int bf1, int bf2, int bf3, double value) {
  // TODO: Implement exchange Hessian for general case
  auto Kaa = occ::qm::block::aa(K);
  auto Kab = occ::qm::block::ab(K);
  auto Kba = occ::qm::block::ba(K);
  auto Kbb = occ::qm::block::bb(K);
  const auto Daa = occ::qm::block::aa(D);
  const auto Dab = occ::qm::block::ab(D);
  const auto Dba = occ::qm::block::ba(D);
  const auto Dbb = occ::qm::block::bb(D);
  Kaa(bf0, bf2) -= Daa(bf1, bf3) * value;
  Kaa(bf0, bf3) -= Daa(bf1, bf2) * value;
  Kbb(bf0, bf2) -= Dbb(bf1, bf3) * value;
  Kbb(bf0, bf3) -= Dbb(bf1, bf2) * value;
  Kab(bf0, bf2) -= (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
  Kab(bf0, bf3) -= (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
  Kba(bf0, bf2) -= (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
  Kba(bf0, bf3) -= (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
}

inline void k_inner_u_hess(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> K, int bf0,
                           int bf1, int bf2, int bf3, double value) {
  // TODO: Implement exchange Hessian for unrestricted case
  auto Ka = occ::qm::block::a(K);
  auto Kb = occ::qm::block::b(K);
  const auto Da = occ::qm::block::a(D);
  const auto Db = occ::qm::block::b(D);
  Ka(bf0, bf2) -= 2 * Da(bf1, bf3) * value;
  Ka(bf0, bf3) -= 2 * Da(bf1, bf2) * value;
  Kb(bf0, bf2) -= 2 * Db(bf1, bf3) * value;
  Kb(bf0, bf3) -= 2 * Db(bf1, bf2) * value;
}

// Template delegates for Coulomb Hessian
template <occ::qm::SpinorbitalKind sk>
void delegate_j_hess(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                     int bf1, int bf2, int bf3, double value) {
  if constexpr (sk == SpinorbitalKind::Restricted) {
    j_inner_r_hess(D, J, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
    j_inner_u_hess(D, J, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::General) {
    j_inner_g_hess(D, J, bf0, bf1, bf2, bf3, value);
  }
}

// Template delegates for Exchange Hessian
template <occ::qm::SpinorbitalKind sk>
void delegate_k_hess(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> K, int bf0,
                     int bf1, int bf2, int bf3, double value) {
  if constexpr (sk == SpinorbitalKind::Restricted) {
    k_inner_r_hess(D, K, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
    k_inner_u_hess(D, K, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::General) {
    k_inner_g_hess(D, K, bf0, bf1, bf2, bf3, value);
  }
}

} // namespace occ::qm::detail