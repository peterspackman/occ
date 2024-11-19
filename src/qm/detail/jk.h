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

inline void fock_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> F, int bf0,
                         int bf1, int bf2, int bf3, double value) {
  auto Faa = occ::qm::block::aa(F);
  auto Fab = occ::qm::block::ab(F);
  auto Fba = occ::qm::block::ba(F);
  auto Fbb = occ::qm::block::bb(F);
  const auto Daa = occ::qm::block::aa(D);
  const auto Dab = occ::qm::block::ab(D);
  const auto Dba = occ::qm::block::ba(D);
  const auto Dbb = occ::qm::block::bb(D);

  // J aa
  Faa(bf0, bf1) += 2 * Daa(bf2, bf3) * value;
  Faa(bf2, bf3) += 2 * Daa(bf0, bf1) * value;
  // J bb
  Fbb(bf0, bf1) += 2 * Dbb(bf2, bf3) * value;
  Fbb(bf2, bf3) += 2 * Dbb(bf0, bf1) * value;

  // K aa
  Faa(bf0, bf2) -= 0.5 * Daa(bf1, bf3) * value;
  Faa(bf1, bf3) -= 0.5 * Daa(bf0, bf2) * value;
  Faa(bf0, bf3) -= 0.5 * Daa(bf1, bf2) * value;
  Faa(bf1, bf2) -= 0.5 * Daa(bf0, bf3) * value;

  // K bb
  Fbb(bf0, bf2) -= 0.5 * Dbb(bf1, bf3) * value;
  Fbb(bf1, bf3) -= 0.5 * Dbb(bf0, bf2) * value;
  Fbb(bf0, bf3) -= 0.5 * Dbb(bf1, bf2) * value;
  Fbb(bf1, bf2) -= 0.5 * Dbb(bf0, bf3) * value;

  // Kab, Kba
  Fab(bf0, bf2) -= 0.5 * (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
  Fab(bf1, bf3) -= 0.5 * (Dab(bf0, bf2) + Dba(bf0, bf2)) * value;
  Fab(bf0, bf3) -= 0.5 * (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
  Fab(bf1, bf2) -= 0.5 * (Dab(bf0, bf3) + Dba(bf0, bf3)) * value;
  Fba(bf0, bf2) -= 0.5 * (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
  Fba(bf1, bf3) -= 0.5 * (Dab(bf0, bf2) + Dba(bf0, bf2)) * value;
  Fba(bf0, bf3) -= 0.5 * (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
  Fba(bf1, bf2) -= 0.5 * (Dab(bf0, bf3) + Dba(bf0, bf3)) * value;
}

inline void j_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J, int bf0,
                      int bf1, int bf2, int bf3, double value) {
  auto Jaa = occ::qm::block::aa(J);
  auto Jbb = occ::qm::block::bb(J);
  const auto Daa = occ::qm::block::aa(D);
  const auto Dbb = occ::qm::block::bb(D);

  // J aa
  Jaa(bf0, bf1) += 2 * Daa(bf2, bf3) * value;
  Jaa(bf2, bf3) += 2 * Daa(bf0, bf1) * value;
  // J bb
  Jbb(bf0, bf1) += 2 * Dbb(bf2, bf3) * value;
  Jbb(bf2, bf3) += 2 * Dbb(bf0, bf1) * value;
}

inline void jk_inner_g(Eigen::Ref<const Mat> D, Eigen::Ref<Mat> J,
                       Eigen::Ref<Mat> K, int bf0, int bf1, int bf2, int bf3,
                       double value) {
  auto Jaa = occ::qm::block::aa(J);
  auto Jbb = occ::qm::block::bb(J);
  auto Kaa = occ::qm::block::aa(K);
  auto Kab = occ::qm::block::ab(K);
  auto Kba = occ::qm::block::ba(K);
  auto Kbb = occ::qm::block::bb(K);

  const auto Daa = occ::qm::block::aa(D);
  const auto Dab = occ::qm::block::ab(D);
  const auto Dba = occ::qm::block::ba(D);
  const auto Dbb = occ::qm::block::bb(D);

  // J aa
  Jaa(bf0, bf1) += 2 * Daa(bf2, bf3) * value;
  Jaa(bf2, bf3) += 2 * Daa(bf0, bf1) * value;
  // J bb
  Jbb(bf0, bf1) += 2 * Dbb(bf2, bf3) * value;
  Jbb(bf2, bf3) += 2 * Dbb(bf0, bf1) * value;

  // K aa
  Kaa(bf0, bf2) += 0.5 * Daa(bf1, bf3) * value;
  Kaa(bf1, bf3) += 0.5 * Daa(bf0, bf2) * value;
  Kaa(bf0, bf3) += 0.5 * Daa(bf1, bf2) * value;
  Kaa(bf1, bf2) += 0.5 * Daa(bf0, bf3) * value;

  // K bb
  Kbb(bf0, bf2) += 0.5 * Dbb(bf1, bf3) * value;
  Kbb(bf1, bf3) += 0.5 * Dbb(bf0, bf2) * value;
  Kbb(bf0, bf3) += 0.5 * Dbb(bf1, bf2) * value;
  Kbb(bf1, bf2) += 0.5 * Dbb(bf0, bf3) * value;

  // Kab, Kba
  Kab(bf0, bf2) += 0.5 * (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
  Kab(bf1, bf3) += 0.5 * (Dab(bf0, bf2) + Dba(bf0, bf2)) * value;
  Kab(bf0, bf3) += 0.5 * (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
  Kab(bf1, bf2) += 0.5 * (Dab(bf0, bf3) + Dba(bf0, bf3)) * value;
  Kba(bf0, bf2) += 0.5 * (Dab(bf1, bf3) + Dba(bf1, bf3)) * value;
  Kba(bf1, bf3) += 0.5 * (Dab(bf0, bf2) + Dba(bf0, bf2)) * value;
  Kba(bf0, bf3) += 0.5 * (Dab(bf1, bf2) + Dba(bf1, bf2)) * value;
  Kba(bf1, bf2) += 0.5 * (Dab(bf0, bf3) + Dba(bf0, bf3)) * value;
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
