// Nuclear gradient of the density-fitted (RI-J) Coulomb energy.
//
// With g_P = sum_mn (mn|P) D_mn and the Coulomb metric V_PQ = (P|Q), the
// fitted coefficients are d = V^-1 g. The engine's J is the Coulomb matrix of
// the total density P = 2D, J = 2 (mn|P) d_P, so
//
//   E_J = tr(D J) = 2 g^T V^-1 g,
//
// the convention of the four-centre compute_jk_gradient(mo, 1, 0), and with
// every term a derivative of the integrals only,
//
//   dE_J/dX = 4 d^T (dg/dX) - 2 d^T (dV/dX) d.
//
// The 3-centre term d^T dg/dX is formed in Hermite space by the split RI-J
// engine (SplitRIJ::three_center_gradient), which fits d on the way. The
// rest uses libcint's ip1 integrals, which differentiate the first function
// with respect to the electron coordinate, minus the derivative with respect
// to its centre: over all ordered AO pairs the first-index derivative of
// (mn|P) D_mn covers half of the AO part of dg (the other half is the same
// sum with m and n swapped), and the auxiliary centre follows from
// translational invariance. Likewise for the metric, d^T dV d = 2 sum over
// ordered (P, Q) of d_P d_Q (dP|Q) on P's centre.

#include <occ/core/parallel.h>
#include <occ/qm/cint_interface.h>
#include <occ/core/timings.h>
#include <occ/qm/integral_engine_df.h>
#include "detail/df_derivatives.h"

namespace occ::qm {

namespace {

using namespace detail::dfd;

// g_P = sum_mn (mn|P) D_k,mn for each density D_k (naux x nd).
template <ShellKind kind>
Mat fitting_rhs(IntegralEngine &ao, const std::vector<Mat> &D) {
  const auto &first_ao = ao.aobasis().first_bf();
  const Eigen::Index naux = ao.auxbasis().nbf();
  const Eigen::Index nd = D.size();
  occ::parallel::thread_local_storage<Mat> tl_g(Mat::Zero(naux, nd));
  occ::parallel::thread_local_storage<Mat> tl_d;
  for_each_pair_block<kind>(ao, [&](int p, int q, int na, int nb,
                                    const Mat &I) {
    auto &Dp = tl_d.local();
    Dp.resize(na * nb, nd);
    const double fac = (p == q) ? 1.0 : 2.0;
    for (Eigen::Index k = 0; k < nd; k++)
      Eigen::Map<Mat>(Dp.col(k).data(), na, nb) =
          fac * D[k].block(first_ao[p], first_ao[q], na, nb);
    tl_g.local().noalias() += I.transpose() * Dp;
  });
  Mat g = Mat::Zero(naux, nd);
  for (const auto &t : tl_g)
    g += t;
  return g;
}

// d^T dg/dX over all atoms (3 x natom) from libcint: the fallback for
// angular momenta beyond the split RI-J kernels.
template <ShellKind kind>
Mat3N three_center_term(IntegralEngine &ao, const Mat &D, const Vec &d,
                        int natom) {
  const auto &first_ao = ao.aobasis().first_bf();
  const auto &first_aux = ao.auxbasis().first_bf();
  const auto &ao_atom = ao.aobasis().shell_to_atom();
  const auto &aux_atom = ao.auxbasis().shell_to_atom();
  occ::parallel::thread_local_storage<Mat3N> tl(Mat3N::Zero(3, natom));
  for_each_ip1_triple<kind>(ao, [&](int a, int b, int R, int na, int nb,
                                    int nr, const double *buf) {
    const size_t n = static_cast<size_t>(na) * nb * nr;
    const auto Db = D.block(first_ao[a], first_ao[b], na, nb);
    const auto dR = d.segment(first_aux[R], nr);
    Vec3 s = Vec3::Zero();
    for (int x = 0; x < 3; x++)
      for (int r = 0; r < nr; r++) {
        Eigen::Map<const Mat> I(buf + x * n + r * na * nb, na, nb);
        s(x) += dR(r) * Db.cwiseProduct(I).sum();
      }
    // Centre derivative of the first function is -ip1; both orders of the
    // pair make up the AO part, so each counts twice.
    auto &grad = tl.local();
    grad.col(ao_atom[a]) -= 2.0 * s;
    grad.col(aux_atom[R]) += 2.0 * s;
  });
  Mat3N grad = Mat3N::Zero(3, natom);
  for (const auto &t : tl)
    grad += t;
  return grad;
}

template <ShellKind kind>
Mat3N coulomb_gradient_impl(IntegralEngine &ao, IntegralEngine &aux,
                            const Mat &D, const CoulombMetric &metric) {
  const int natom = static_cast<int>(ao.aobasis().atoms().size());
  const Vec d = metric.solve(fitting_rhs<kind>(ao, {D}).col(0));
  return 4.0 * three_center_term<kind>(ao, D, d, natom) -
         2.0 * metric_term(aux, d, natom);
}

// The per-spin density of a restricted calculation, or its unrestricted
// counterpart: the blocks are half the spin densities, so their sum.
Mat restricted_like_density(const MolecularOrbitals &mo, const char *what) {
  switch (mo.kind) {
  case SpinorbitalKind::Restricted:
    return mo.D;
  case SpinorbitalKind::Unrestricted:
    return block::a(mo.D) + block::b(mo.D);
  default:
    throw std::runtime_error(std::string(what) +
                             ": general spinorbitals not supported");
  }
}

// ---------------------------------------------------------------------------
// RI-K gradient
// ---------------------------------------------------------------------------
//
// Per spin, with occupied coefficients C (nbf x o), T^P_ij = (ij|P) and
// c = V^-1 T (over P, for every pair ij), the exchange energy is
//
//   E_K = sum_ij sum_PQ T^P_ij V^-1_PQ T^Q_ij
//
// and its integral derivative is
//
//   dE_K/dX = 2 sum_P sum_mn (mn|P)^X G^P_mn - sum_PQ (V^X)_PQ (c c^T)_PQ,
//
// with G^P = C c^P C^T, formed block by block from Y^P = C c^P as needed.
// The 3-centre part uses the same ordered-pair ip1 scheme as the RI-J
// gradient; the metric part contracts ordered auxiliary pairs with c c^T.

template <ShellKind kind>
Mat3N exchange_gradient_spin(IntegralEngine &ao, IntegralEngine &aux,
                             const Mat &C, const CoulombMetric &metric) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const int natom = static_cast<int>(aobasis.atoms().size());
  const auto &first_ao = aobasis.first_bf();
  const auto &first_aux = auxbasis.first_bf();
  const auto &ao_atom = aobasis.shell_to_atom();
  const auto &aux_atom = auxbasis.shell_to_atom();
  const int nsh_ao = static_cast<int>(aobasis.size());
  const auto &shellpairs = ao.shellpairs();
  const Eigen::Index nbf = aobasis.nbf(), naux = auxbasis.nbf();
  const Eigen::Index o = C.cols();
  if (o == 0)
    return Mat3N::Zero(3, natom);

  // X^P = (mn|P) C, nbf x o per auxiliary function; tasks own aux shells.
  std::vector<Mat> X(naux, Mat::Zero(nbf, o));
  {
    cint::Optimizer opt(ao.env(), Op::coulomb, 3);
    occ::parallel::parallel_for(size_t(0), auxbasis.size(), [&](size_t R) {
      std::vector<double> buf(aobasis.max_shell_size() *
                              aobasis.max_shell_size() *
                              auxbasis.max_shell_size());
      for (int p = 0; p < nsh_ao; p++)
        for (const auto q : shellpairs[p]) {
          const auto dims = ao.env().three_center_helper<Op::coulomb, kind>(
              {p, static_cast<int>(q), static_cast<int>(R) + nsh_ao},
              opt.optimizer_ptr(), buf.data(), nullptr);
          if (dims[0] < 0)
            continue;
          const int na = dims[0], nb = dims[1];
          for (int r = 0; r < dims[2]; r++) {
            Eigen::Map<const Mat> I(buf.data() + r * na * nb, na, nb);
            auto &Xr = X[first_aux[R] + r];
            Xr.middleRows(first_ao[p], na).noalias() +=
                I * C.middleRows(first_ao[q], nb);
            if (p != static_cast<int>(q))
              Xr.middleRows(first_ao[q], nb).noalias() +=
                  I.transpose() * C.middleRows(first_ao[p], na);
          }
        }
    });
  }
  // T (o^2 x naux), c = V^-1 T^T (naux x o^2).
  Mat T(o * o, naux);
  occ::parallel::parallel_for(Eigen::Index(0), naux, [&](Eigen::Index P) {
    const Mat t = C.transpose() * X[P];
    T.col(P) = Eigen::Map<const Vec>(t.data(), o * o);
  });
  const Mat c = metric.solve(Mat(T.transpose()));
  const Mat cct = c * c.transpose();
  // Y^P = C c^P, reusing X.
  occ::parallel::parallel_for(Eigen::Index(0), naux, [&](Eigen::Index P) {
    const Vec row = c.row(P).transpose();
    X[P].noalias() = C * Eigen::Map<const Mat>(row.data(), o, o);
  });

  // 3-centre part: sum over ordered pairs of first-index ip1 integrals with
  // G^P(a, b) = Y^P[a] C[b]^T, doubled for the other index, the auxiliary
  // centre by translational invariance.
  occ::parallel::thread_local_storage<Mat3N> tl(Mat3N::Zero(3, natom));
  for_each_ip1_triple<kind>(ao, [&](int a, int b, int R, int na, int nb,
                                    int nr, const double *buf) {
    const size_t n = static_cast<size_t>(na) * nb * nr;
    Vec3 s = Vec3::Zero();
    for (int r = 0; r < nr; r++) {
      const Mat G = X[first_aux[R] + r].middleRows(first_ao[a], na) *
                    C.middleRows(first_ao[b], nb).transpose();
      for (int x = 0; x < 3; x++)
        s(x) += G.cwiseProduct(
                     Eigen::Map<const Mat>(buf + x * n + r * na * nb, na, nb))
                    .sum();
    }
    auto &grad = tl.local();
    grad.col(ao_atom[a]) -= 2.0 * s;
    grad.col(aux_atom[R]) += 2.0 * s;
  });
  Mat3N t3 = Mat3N::Zero(3, natom);
  for (const auto &t : tl)
    t3 += t;
  return 2.0 * t3 - metric_term_matrix(aux, cct, natom);
}

} // namespace

Mat3N IntegralEngineDF::coulomb_gradient(const MolecularOrbitals &mo) {
  if (m_omega != 0.0)
    throw std::runtime_error(
        "RI-J gradient: the active metric is the attenuated one");
  const Mat D = restricted_like_density(mo, "RI-J gradient");
  if (!SplitRIJ::supports(m_ao_engine.aobasis(), m_ao_engine.auxbasis())) {
    occ::timing::start(occ::timing::category::df);
    Mat3N grad = m_ao_engine.is_spherical()
                     ? coulomb_gradient_impl<ShellKind::Spherical>(
                           m_ao_engine, m_aux_engine, D, V_LLt)
                     : coulomb_gradient_impl<ShellKind::Cartesian>(
                           m_ao_engine, m_aux_engine, D, V_LLt);
    occ::timing::stop(occ::timing::category::df);
    return grad;
  }
  // The 3-centre term in Hermite space (split RI-J, which also fits d), the
  // metric term from libcint.
  if (!m_split_rij) {
    m_split_rij = std::make_unique<SplitRIJ>(
        m_ao_engine.aobasis(), m_ao_engine.auxbasis(),
        m_ao_engine.shellpairs(), m_ao_engine.schwarz(), m_V_LLt_full,
        m_V_diag_sqrt);
  }
  Vec d;
  const Mat3N t3 = m_split_rij->three_center_gradient(D, &d);
  occ::timing::start(occ::timing::category::df);
  const int natom = static_cast<int>(m_ao_engine.aobasis().atoms().size());
  const Mat3N t2 = metric_term(m_aux_engine, d, natom);
  occ::timing::stop(occ::timing::category::df);
  return 4.0 * t3 - 2.0 * t2;
}

// ---------------------------------------------------------------------------
// RI-J second derivatives
// ---------------------------------------------------------------------------
//
// With U^X = g^X - V^X d (first-derivative integrals only) and W = V^-1,
//
//   d2E_J/dXdY = 4 d^T g^XY - 2 d^T V^XY d + 4 U^X^T W U^Y,
//
// the skeleton Fock derivative at fixed D is
//
//   J^X = 2 (mn|P)^X d_P + 2 (mn|P) (W U^X)_P,
//
// and the response to a density D1 is the RI-J build on D1 itself.

namespace {

struct RIJFirstOrder {
  Vec d;  // fitted coefficients V^-1 g
  Mat U;  // naux x 3N: g^X - V^X d
  Mat WU; // V^-1 U
};

template <ShellKind kind>
RIJFirstOrder rij_first_order(IntegralEngine &ao, IntegralEngine &aux,
                              const Mat &D, const CoulombMetric &metric) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const int natom = static_cast<int>(aobasis.atoms().size());
  const auto &first_ao = aobasis.first_bf();
  const auto &first_aux = auxbasis.first_bf();
  const auto &ao_atom = aobasis.shell_to_atom();
  const auto &aux_atom = auxbasis.shell_to_atom();
  const Eigen::Index naux = auxbasis.nbf();

  RIJFirstOrder out;
  out.d = metric.solve(fitting_rhs<kind>(ao, {D}).col(0));

  // g^X: over ordered pairs the first-index derivative is half the AO part;
  // the auxiliary centre by translational invariance.
  occ::parallel::thread_local_storage<Mat> tl_G(Mat::Zero(naux, 3 * natom));
  for_each_ip1_triple<kind>(ao, [&](int a, int b, int R, int na, int nb,
                                    int nr, const double *buf) {
    auto &G = tl_G.local();
    const size_t n = static_cast<size_t>(na) * nb * nr;
    const auto Db = D.block(first_ao[a], first_ao[b], na, nb);
    const int A = ao_atom[a], C = aux_atom[R];
    for (int x = 0; x < 3; x++)
      for (int r = 0; r < nr; r++) {
        Eigen::Map<const Mat> I(buf + x * n + r * na * nb, na, nb);
        const double v = -2.0 * Db.cwiseProduct(I).sum();
        G(first_aux[R] + r, 3 * A + x) += v;
        G(first_aux[R] + r, 3 * C + x) -= v;
      }
  });
  Mat G = Mat::Zero(naux, 3 * natom);
  for (const auto &t : tl_G)
    G += t;

  // V^X d: (dP|Q) d_Q on P's centre, minus the same on Q's. Tasks own P.
  const auto &first = aux.aobasis().first_bf();
  const auto &atom = aux.aobasis().shell_to_atom();
  Mat M = Mat::Zero(naux, 3 * natom);
  for_each_metric_ip1(aux, [&](int P, int Q, int np, int nq,
                               const double *buf) {
    const auto dQ = out.d.segment(first[Q], nq);
    for (int x = 0; x < 3; x++) {
      Eigen::Map<const Mat> I(buf + x * np * nq, np, nq);
      const Vec w = -(I * dQ); // centre derivative of P
      M.block(first[P], 3 * atom[P] + x, np, 1) += w;
      M.block(first[P], 3 * atom[Q] + x, np, 1) -= w;
    }
  });
  out.U = G - M;
  out.WU = metric.solve(out.U);
  return out;
}

template <ShellKind kind>
Mat rij_hessian_impl(IntegralEngine &ao, IntegralEngine &aux, const Mat &D,
                     const CoulombMetric &metric) {
  const auto &aobasis = ao.aobasis();
  const auto &first_ao = aobasis.first_bf();
  const auto &first_aux = ao.auxbasis().first_bf();

  const auto fo = rij_first_order<kind>(ao, aux, D, metric);
  const Vec &d = fo.d;
  Mat H = 4.0 * fo.U.transpose() * fo.WU;

  // 4 d^T g^XY and -2 d^T V^XY d.
  H += three_center_hessian<kind>(
      ao, 4.0, [&](int p, int q, int R, int nr, std::vector<Mat> &W) {
        const auto Db = D.block(first_ao[p], first_ao[q], aobasis[p].size(),
                                aobasis[q].size());
        for (int r = 0; r < nr; r++)
          W[r] = d(first_aux[R] + r) * Db;
      });
  H += metric_hessian(aux, -2.0, vector_metric_contraction(aux, d));
  return 0.5 * (H + H.transpose());
}

template <ShellKind kind>
std::vector<Mat> rij_derivative_fock_impl(IntegralEngine &ao,
                                          IntegralEngine &aux, const Mat &D,
                                          const CoulombMetric &metric) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const int natom = static_cast<int>(aobasis.atoms().size());
  const int n3 = 3 * natom;
  const size_t nbf = aobasis.nbf();
  const auto &first_ao = aobasis.first_bf();
  const auto &first_aux = auxbasis.first_bf();
  const auto &ao_atom = aobasis.shell_to_atom();
  const auto &aux_atom = auxbasis.shell_to_atom();

  const auto fo = rij_first_order<kind>(ao, aux, D, metric);
  std::vector<Mat> F(n3, Mat::Zero(nbf, nbf));

  // 2 (mn|P)^X d_P. Tasks own unordered pairs, so blocks (a, b) and (b, a)
  // are written by one task only.
  for_each_ip1_triple<kind, TripleTasks::ShellPair>(
      ao, [&](int a, int b, int R, int na, int nb, int nr, const double *buf) {
    const size_t n = static_cast<size_t>(na) * nb * nr;
    const auto dR = fo.d.segment(first_aux[R], nr);
    const int A = ao_atom[a], C = aux_atom[R];
    for (int x = 0; x < 3; x++) {
      Mat T = Mat::Zero(na, nb);
      for (int r = 0; r < nr; r++)
        T -= dR(r) * Eigen::Map<const Mat>(buf + x * n + r * na * nb, na, nb);
      T *= 2.0;
      for (const auto [X, sgn] : {std::pair{3 * A + x, 1.0},
                                  std::pair{3 * C + x, -1.0}}) {
        F[X].block(first_ao[a], first_ao[b], na, nb) += sgn * T;
        F[X].block(first_ao[b], first_ao[a], nb, na) += sgn * T.transpose();
      }
    }
  });

  // 2 (mn|P) (W U^X)_P: an RI-J build per coordinate on shared integrals,
  // one GEMM per pair over all coordinates.
  for_each_pair_block<kind>(ao, [&](int p, int q, int na, int nb,
                                    const Mat &I) {
    const Mat T = 2.0 * I * fo.WU; // (na nb) x 3N
    for (int X = 0; X < n3; X++) {
      Eigen::Map<const Mat> TX(T.col(X).data(), na, nb);
      F[X].block(first_ao[p], first_ao[q], na, nb) += TX;
      if (p != q)
        F[X].block(first_ao[q], first_ao[p], nb, na) += TX.transpose();
    }
  });
  return F;
}

template <ShellKind kind>
std::vector<Mat> rij_response_impl(IntegralEngine &ao,
                                   const std::vector<Mat> &D1,
                                   const CoulombMetric &metric) {
  const size_t nbf = ao.aobasis().nbf();
  const auto &first_ao = ao.aobasis().first_bf();
  const Eigen::Index nd = D1.size();
  // C = V^-1 g for every density at once, then J = 2 I C as one GEMM per
  // pair.
  const Mat g = fitting_rhs<kind>(ao, D1);
  const Mat C = metric.solve(g);
  std::vector<Mat> J(nd, Mat::Zero(nbf, nbf));
  for_each_pair_block<kind>(ao, [&](int p, int q, int na, int nb,
                                    const Mat &I) {
    const Mat T = 2.0 * I * C; // (na nb) x nd
    for (Eigen::Index k = 0; k < nd; k++) {
      Eigen::Map<const Mat> Tk(T.col(k).data(), na, nb);
      J[k].block(first_ao[p], first_ao[q], na, nb) += Tk;
      if (p != q)
        J[k].block(first_ao[q], first_ao[p], nb, na) += Tk.transpose();
    }
  });
  return J;
}

} // namespace

Mat IntegralEngineDF::coulomb_hessian(const MolecularOrbitals &mo) {
  if (m_omega != 0.0)
    throw std::runtime_error("RI-J Hessian: the active metric is attenuated");
  const Mat D = restricted_like_density(mo, "RI-J Hessian");
  occ::timing::start(occ::timing::category::df);
  Mat H = m_ao_engine.is_spherical()
              ? rij_hessian_impl<ShellKind::Spherical>(m_ao_engine,
                                                       m_aux_engine, D, V_LLt)
              : rij_hessian_impl<ShellKind::Cartesian>(m_ao_engine,
                                                       m_aux_engine, D, V_LLt);
  occ::timing::stop(occ::timing::category::df);
  return H;
}

std::vector<Mat>
IntegralEngineDF::coulomb_derivative_fock(const MolecularOrbitals &mo) {
  if (m_omega != 0.0)
    throw std::runtime_error(
        "RI-J derivative Fock: the active metric is attenuated");
  const Mat D = restricted_like_density(mo, "RI-J derivative Fock");
  occ::timing::start(occ::timing::category::df);
  auto F = m_ao_engine.is_spherical()
               ? rij_derivative_fock_impl<ShellKind::Spherical>(
                     m_ao_engine, m_aux_engine, D, V_LLt)
               : rij_derivative_fock_impl<ShellKind::Cartesian>(
                     m_ao_engine, m_aux_engine, D, V_LLt);
  occ::timing::stop(occ::timing::category::df);
  return F;
}

std::vector<Mat>
IntegralEngineDF::coulomb_response(const std::vector<Mat> &D1) {
  if (m_omega != 0.0)
    throw std::runtime_error("RI-J response: the active metric is attenuated");
  occ::timing::start(occ::timing::category::df);
  auto J = m_ao_engine.is_spherical()
               ? rij_response_impl<ShellKind::Spherical>(m_ao_engine, D1,
                                                         V_LLt)
               : rij_response_impl<ShellKind::Cartesian>(m_ao_engine, D1,
                                                         V_LLt);
  occ::timing::stop(occ::timing::category::df);
  return J;
}

Mat3N IntegralEngineDF::exchange_gradient(const MolecularOrbitals &mo) {
  // With omega set, the 3-centre integrals, their derivatives and the metric
  // are all for erf(omega r)/r, so the same expression applies.
  occ::timing::start(occ::timing::category::df);
  auto run = [&](const Mat &C) {
    return m_ao_engine.is_spherical()
               ? exchange_gradient_spin<ShellKind::Spherical>(
                     m_ao_engine, m_aux_engine, C, V_LLt)
               : exchange_gradient_spin<ShellKind::Cartesian>(
                     m_ao_engine, m_aux_engine, C, V_LLt);
  };
  Mat3N grad;
  switch (mo.kind) {
  case SpinorbitalKind::Restricted:
    grad = run(mo.Cocc);
    break;
  case SpinorbitalKind::Unrestricted:
    // The density blocks are half the spin densities, D^s = C^s C^s^T / 2.
    grad = 0.5 * (run(block::a(mo.Cocc)) + run(block::b(mo.Cocc)));
    break;
  default:
    occ::timing::stop(occ::timing::category::df);
    throw std::runtime_error(
        "RI-K gradient: general spinorbitals not supported");
  }
  occ::timing::stop(occ::timing::category::df);
  return grad;
}

} // namespace occ::qm
