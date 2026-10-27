#include "kernel_traits.h"
#include <occ/qm/integral_engine.h>

namespace occ::qm::detail {

inline void fock_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0,
                         int bf1, int bf2, int bf3, double value) {
  F(bf0, bf1) += D(bf2, bf3) * value;
  F(bf2, bf3) += D(bf0, bf1) * value;
  // K
  F(bf0, bf2) -= 0.25 * D(bf1, bf3) * value;
  F(bf1, bf3) -= 0.25 * D(bf0, bf2) * value;
  F(bf0, bf3) -= 0.25 * D(bf1, bf2) * value;
  F(bf1, bf2) -= 0.25 * D(bf0, bf3) * value;
}

inline void j_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                      int bf1, int bf2, int bf3, double value) {
  J(bf0, bf1) += D(bf2, bf3) * value;
  J(bf2, bf3) += D(bf0, bf1) * value;
}

inline void jk_inner_r(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J,
                       Eigen::Ref<Mat> K, int bf0, int bf1, int bf2, int bf3,
                       double value) {
  J(bf0, bf1) += D(bf2, bf3) * value;
  J(bf2, bf3) += D(bf0, bf1) * value;
  // K
  K(bf0, bf2) += 0.25 * D(bf1, bf3) * value;
  K(bf1, bf3) += 0.25 * D(bf0, bf2) * value;
  K(bf0, bf3) += 0.25 * D(bf1, bf2) * value;
  K(bf1, bf2) += 0.25 * D(bf0, bf3) * value;
}

inline void fock_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0,
                         int bf1, int bf2, int bf3, double value) {
  auto Fa = occ::qm::block::a(F);
  auto Fb = occ::qm::block::b(F);
  const auto Da = occ::qm::block::a(D);
  const auto Db = occ::qm::block::b(D);
  Fa(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
  Fa(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
  Fb(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
  Fb(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;

  Fa(bf0, bf2) -= 0.5 * Da(bf1, bf3) * value;
  Fa(bf1, bf3) -= 0.5 * Da(bf0, bf2) * value;
  Fa(bf0, bf3) -= 0.5 * Da(bf1, bf2) * value;
  Fa(bf1, bf2) -= 0.5 * Da(bf0, bf3) * value;

  Fb(bf0, bf2) -= 0.5 * Db(bf1, bf3) * value;
  Fb(bf1, bf3) -= 0.5 * Db(bf0, bf2) * value;
  Fb(bf0, bf3) -= 0.5 * Db(bf1, bf2) * value;
  Fb(bf1, bf2) -= 0.5 * Db(bf0, bf3) * value;
}

inline void j_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                      int bf1, int bf2, int bf3, double value) {
  auto Ja = occ::qm::block::a(J);
  auto Jb = occ::qm::block::b(J);
  const auto Da = occ::qm::block::a(D);
  const auto Db = occ::qm::block::b(D);
  Ja(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
  Ja(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
  Jb(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
  Jb(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
}

inline void jk_inner_u(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J,
                       Eigen::Ref<Mat> K, int bf0, int bf1, int bf2, int bf3,
                       double value) {
  auto Ja = occ::qm::block::a(J);
  auto Jb = occ::qm::block::b(J);
  auto Ka = occ::qm::block::a(K);
  auto Kb = occ::qm::block::b(K);
  const auto Da = occ::qm::block::a(D);
  const auto Db = occ::qm::block::b(D);
  Ja(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
  Ja(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;
  Jb(bf0, bf1) += (Da(bf2, bf3) + Db(bf2, bf3)) * value;
  Jb(bf2, bf3) += (Da(bf0, bf1) + Db(bf0, bf1)) * value;

  Ka(bf0, bf2) += 0.5 * Da(bf1, bf3) * value;
  Ka(bf1, bf3) += 0.5 * Da(bf0, bf2) * value;
  Ka(bf0, bf3) += 0.5 * Da(bf1, bf2) * value;
  Ka(bf1, bf2) += 0.5 * Da(bf0, bf3) * value;

  Kb(bf0, bf2) += 0.5 * Db(bf1, bf3) * value;
  Kb(bf1, bf3) += 0.5 * Db(bf0, bf2) * value;
  Kb(bf0, bf3) += 0.5 * Db(bf1, bf2) * value;
  Kb(bf1, bf2) += 0.5 * Db(bf0, bf3) * value;
}

// General (two-component) spinorbitals. Every spin block of D carries the
// same factor 1/2 as the unrestricted ones, so the diagonal blocks follow
// `fock_inner_u`: both see the Coulomb field of the whole density, Daa + Dbb,
// and exchange with their own block.
//
// The off-diagonal exchange is K[D_ab], and D_ab is not symmetric. The
// kernel visits each unique integral once and the accumulation finishes the
// other half by a transpose, which is only right for a symmetric density, so
// D_ab is split: its symmetric part accumulates in the ab block, its
// antisymmetric part in the ba block, and `accumulate_operator_symmetric`
// finishes them as X + X^T and X - X^T. (D_ba = D_ab^T, as D is symmetric.)
inline void j_inner_g_diag(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                           int bf1, int bf2, int bf3, double value) {
  auto Jaa = occ::qm::block::aa(J);
  auto Jbb = occ::qm::block::bb(J);
  const auto Daa = occ::qm::block::aa(D);
  const auto Dbb = occ::qm::block::bb(D);
  const double d23 = (Daa(bf2, bf3) + Dbb(bf2, bf3)) * value;
  const double d01 = (Daa(bf0, bf1) + Dbb(bf0, bf1)) * value;
  Jaa(bf0, bf1) += d23;
  Jaa(bf2, bf3) += d01;
  Jbb(bf0, bf1) += d23;
  Jbb(bf2, bf3) += d01;
}

inline void k_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> K, int bf0,
                      int bf1, int bf2, int bf3, double value) {
  auto Kaa = occ::qm::block::aa(K);
  auto Kbb = occ::qm::block::bb(K);
  auto Ksym = occ::qm::block::ab(K);
  auto Kanti = occ::qm::block::ba(K);
  const auto Daa = occ::qm::block::aa(D);
  const auto Dab = occ::qm::block::ab(D);
  const auto Dba = occ::qm::block::ba(D);
  const auto Dbb = occ::qm::block::bb(D);
  const double w = 0.5 * value;
  const double ws = 0.25 * value;

  Kaa(bf0, bf2) += w * Daa(bf1, bf3);
  Kaa(bf1, bf3) += w * Daa(bf0, bf2);
  Kaa(bf0, bf3) += w * Daa(bf1, bf2);
  Kaa(bf1, bf2) += w * Daa(bf0, bf3);

  Kbb(bf0, bf2) += w * Dbb(bf1, bf3);
  Kbb(bf1, bf3) += w * Dbb(bf0, bf2);
  Kbb(bf0, bf3) += w * Dbb(bf1, bf2);
  Kbb(bf1, bf2) += w * Dbb(bf0, bf3);

  Ksym(bf0, bf2) += ws * (Dab(bf1, bf3) + Dba(bf1, bf3));
  Ksym(bf1, bf3) += ws * (Dab(bf0, bf2) + Dba(bf0, bf2));
  Ksym(bf0, bf3) += ws * (Dab(bf1, bf2) + Dba(bf1, bf2));
  Ksym(bf1, bf2) += ws * (Dab(bf0, bf3) + Dba(bf0, bf3));

  Kanti(bf0, bf2) += ws * (Dab(bf1, bf3) - Dba(bf1, bf3));
  Kanti(bf1, bf3) += ws * (Dab(bf0, bf2) - Dba(bf0, bf2));
  Kanti(bf0, bf3) += ws * (Dab(bf1, bf2) - Dba(bf1, bf2));
  Kanti(bf1, bf2) += ws * (Dab(bf0, bf3) - Dba(bf0, bf3));
}

inline void fock_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0,
                         int bf1, int bf2, int bf3, double value) {
  j_inner_g_diag(D, F, bf0, bf1, bf2, bf3, value);
  k_inner_g(D, F, bf0, bf1, bf2, bf3, -value);
}

inline void j_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                      int bf1, int bf2, int bf3, double value) {
  j_inner_g_diag(D, J, bf0, bf1, bf2, bf3, value);
}

inline void jk_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J,
                       Eigen::Ref<Mat> K, int bf0, int bf1, int bf2, int bf3,
                       double value) {
  j_inner_g_diag(D, J, bf0, bf1, bf2, bf3, value);
  k_inner_g(D, K, bf0, bf1, bf2, bf3, value);
}

template <occ::qm::SpinorbitalKind sk>
void delegate_fock(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0, int bf1,
                   int bf2, int bf3, double value) {
  if constexpr (sk == SpinorbitalKind::Restricted) {
    fock_inner_r(D, F, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
    fock_inner_u(D, F, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::General) {
    fock_inner_g(D, F, bf0, bf1, bf2, bf3, value);
  }
}

template <occ::qm::SpinorbitalKind sk>
void delegate_jk(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, Eigen::Ref<Mat> K,
                 int bf0, int bf1, int bf2, int bf3, double value) {
  if constexpr (sk == SpinorbitalKind::Restricted) {
    jk_inner_r(D, J, K, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
    jk_inner_u(D, J, K, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::General) {
    jk_inner_g(D, J, K, bf0, bf1, bf2, bf3, value);
  }
}

template <occ::qm::SpinorbitalKind sk>
void delegate_j(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0, int bf1,
                int bf2, int bf3, double value) {
  if constexpr (sk == SpinorbitalKind::Restricted) {
    j_inner_r(D, J, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
    j_inner_u(D, J, bf0, bf1, bf2, bf3, value);
  } else if constexpr (sk == SpinorbitalKind::General) {
    j_inner_g(D, J, bf0, bf1, bf2, bf3, value);
  }
}

} // namespace occ::qm::detail
