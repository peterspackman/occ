// RI-K second derivatives (restricted): skeleton Hessian, skeleton Fock
// derivatives and the exchange response to perturbed densities.
//
// With occupied coefficients C (nbf x o, D = C C^T), B^P = (. nu|P) C,
// E = V^-1 B over the auxiliary index, t^P = C^T B^P and c^P = C^T E^P, the
// exchange matrix and energy are
//
//   K = sum_P B^P E^P^T,    E_K = tr(D K) = sum_P t^P . c^P.
//
// Its integral derivatives, with B^{P,X} = (. nu|P)^X C (both AO indices
// differentiated, the auxiliary centre by translational invariance):
//
//   K^X = Z + Z^T - (S + S^T),  Z = sum_P B^{P,X} E^P^T,
//                               S = sum_P E^P (sum_Q V^X_PQ E^Q)^T,
//
//   d2E_K/dXdY = 2 sum_P sum_mn (mn|P)^XY G^P_mn - sum_PQ (c c^T)_PQ V^XY_PQ
//                + 2 sum_ij u^X_ij^T V^-1 u^Y_ij,
//
// where G^P = C c^P C^T and u^X = t^X - V^X c. The last term is assembled
// from L^-1 u^X (V = L L^T) for every coordinate, which costs
// 8 naux o^2 3N bytes. The response to D1 = Cx C^T + C Cx^T is
// K[D1] = Z + Z^T with Z = sum_P (. nu|P) Cx E^P^T.
//
// Tensors over (orbital, auxiliary) are stored nbf x (o naux) with column
// i + o P, which maps without copying onto (nbf o) x naux for the metric.

#include "detail/df_derivatives.h"
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/integral_engine_df.h>

namespace occ::qm {

namespace {

using namespace detail::dfd;

// (mu nu|r) as full nbf x nbf matrices for every function r of aux shell R.
template <ShellKind kind>
std::vector<Mat> aux_shell_matrices(IntegralEngine &ao, cint::Optimizer &opt,
                                    size_t R) {
  const auto &aobasis = ao.aobasis();
  const auto &first_ao = aobasis.first_bf();
  const auto &shellpairs = ao.shellpairs();
  const int nsh_ao = static_cast<int>(aobasis.size());
  const Eigen::Index nbf = aobasis.nbf();
  const int nr = ao.auxbasis()[R].size();
  std::vector<double> buf(aobasis.max_shell_size() * aobasis.max_shell_size() *
                          ao.auxbasis().max_shell_size());
  std::vector<Mat> M(nr, Mat::Zero(nbf, nbf));
  for (int p = 0; p < nsh_ao; p++)
    for (const auto qq : shellpairs[p]) {
      const int q = static_cast<int>(qq);
      const auto dims = ao.env().three_center_helper<Op::coulomb, kind>(
          {p, q, static_cast<int>(R) + nsh_ao}, opt.optimizer_ptr(),
          buf.data(), nullptr);
      if (dims[0] < 0)
        continue;
      const int na = dims[0], nb = dims[1];
      for (int r = 0; r < nr; r++) {
        Eigen::Map<const Mat> I(buf.data() + r * na * nb, na, nb);
        M[r].block(first_ao[p], first_ao[q], na, nb) = I;
        if (p != q)
          M[r].block(first_ao[q], first_ao[p], nb, na) = I.transpose();
      }
    }
  return M;
}

// (. nu|P) Cs[k] for several coefficient sets at once (each nbf x o), as
// nbf x (o naux) tensors; tasks own auxiliary shells.
template <ShellKind kind>
std::vector<Mat> half_transform(IntegralEngine &ao,
                                const std::vector<const Mat *> &Cs) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const auto &first_aux = auxbasis.first_bf();
  const Eigen::Index nbf = aobasis.nbf(), naux = auxbasis.nbf();
  const size_t m = Cs.size();
  const Eigen::Index o = m ? Cs[0]->cols() : 0;
  Mat Cstack(nbf, o * m);
  for (size_t k = 0; k < m; k++)
    Cstack.middleCols(o * k, o) = *Cs[k];
  std::vector<Mat> B(m, Mat::Zero(nbf, o * naux));
  cint::Optimizer opt(ao.env(), Op::coulomb, 3);
  // Per auxiliary shell: assemble each function's full (mu nu|r) matrix from
  // the pair blocks, then one GEMM against every coefficient set at once.
  occ::parallel::parallel_for(size_t(0), auxbasis.size(), [&](size_t R) {
    const int nr = auxbasis[R].size();
    const auto M = aux_shell_matrices<kind>(ao, opt, R);
    Mat T;
    for (int r = 0; r < nr; r++) {
      T.noalias() = M[r] * Cstack;
      const Eigen::Index col = o * (first_aux[R] + r);
      for (size_t k = 0; k < m; k++)
        B[k].middleCols(col, o) = T.middleCols(o * k, o);
    }
  });
  return B;
}

// V^-1 applied over the auxiliary index of an nbf x (o naux) tensor.
Mat fit(const Mat &B, Eigen::Index nbf, Eigen::Index o, Eigen::Index naux,
        const CoulombMetric &metric) {
  Eigen::Map<const Mat> Bm(B.data(), nbf * o, naux);
  const Mat Et = metric.solve(Bm.transpose());
  Mat E(nbf, o * naux);
  Eigen::Map<Mat>(E.data(), nbf * o, naux) = Et.transpose();
  return E;
}

// B^{P,X} for the three coordinates of atom A.
template <ShellKind kind>
std::array<Mat, 3> half_transform_derivative(IntegralEngine &ao,
                                             const Mat &C, int A) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const auto &first_ao = aobasis.first_bf();
  const auto &first_aux = auxbasis.first_bf();
  const auto &ao_atom = aobasis.shell_to_atom();
  const auto &aux_atom = auxbasis.shell_to_atom();
  const int nsh_ao = static_cast<int>(aobasis.size());
  const auto pairs = unordered_pairs(ao);
  const Eigen::Index nbf = aobasis.nbf(), naux = auxbasis.nbf();
  const Eigen::Index o = C.cols();
  std::array<Mat, 3> B;
  for (auto &b : B)
    b = Mat::Zero(nbf, o * naux);
  cint::Optimizer opt(ao.env(), Op::coulomb, 3, 1);
  occ::parallel::parallel_for(size_t(0), auxbasis.size(), [&](size_t R) {
    std::vector<double> buf(3 * aobasis.max_shell_size() *
                            aobasis.max_shell_size() *
                            auxbasis.max_shell_size());
    const bool onR = aux_atom[R] == A;
    auto ordered = [&](int a, int b) {
      const bool onA = ao_atom[a] == A;
      if (onA == onR)
        return; // both centres on A cancel; neither is not this atom
      const auto dims = ao.env().three_center_helper_grad<Op::coulomb, kind>(
          {a, b, static_cast<int>(R) + nsh_ao}, opt.optimizer_ptr(),
          buf.data(), nullptr);
      if (dims[0] < 0)
        return;
      const int na = dims[0], nb = dims[1], nr = dims[2];
      const size_t n = static_cast<size_t>(na) * nb * nr;
      // Centre derivative of a is -ip1; the auxiliary centre's is +ip1.
      const double sgn = onA ? -1.0 : 1.0;
      for (int x = 0; x < 3; x++)
        for (int r = 0; r < nr; r++) {
          Eigen::Map<const Mat> J(buf.data() + x * n + r * na * nb, na, nb);
          const Eigen::Index col = o * (first_aux[R] + r);
          B[x].block(first_ao[a], col, na, o).noalias() +=
              sgn * J * C.middleRows(first_ao[b], nb);
          B[x].block(first_ao[b], col, nb, o).noalias() +=
              sgn * J.transpose() * C.middleRows(first_ao[a], na);
        }
    };
    for (const auto &[p, q] : pairs) {
      ordered(p, q);
      if (p != q)
        ordered(q, p);
    }
  });
  return B;
}

// For atom A: the auxiliary functions on it and, per coordinate, VA (naux x
// nA) with VA(Q, p) = d/dX (P|Q) for P = aux_A[p] on A and Q not on A.
template <ShellKind kind>
std::array<Mat, 3> metric_derivative_columns(IntegralEngine &aux, int A,
                                             std::vector<int> &aux_A) {
  const auto &basis = aux.aobasis();
  const auto &first = basis.first_bf();
  const auto &atom = basis.shell_to_atom();
  const int nsh = static_cast<int>(basis.size());
  aux_A.clear();
  std::vector<int> local(basis.nbf(), -1);
  for (int P = 0; P < nsh; P++)
    if (atom[P] == A)
      for (size_t i = 0; i < basis[P].size(); i++) {
        local[first[P] + i] = static_cast<int>(aux_A.size());
        aux_A.push_back(first[P] + i);
      }
  std::array<Mat, 3> VA;
  for (auto &v : VA)
    v = Mat::Zero(basis.nbf(), aux_A.size());
  cint::Optimizer opt(aux.env(), Op::coulomb, 2, 1);
  std::vector<double> buf(3 * basis.max_shell_size() * basis.max_shell_size());
  for (int P = 0; P < nsh; P++) {
    if (atom[P] != A)
      continue;
    const int np = basis[P].size();
    for (int Q = 0; Q < nsh; Q++) {
      if (atom[Q] == A)
        continue;
      const auto dims = aux.env().two_center_helper_grad<Op::coulomb, kind>(
          {P, Q}, opt.optimizer_ptr(), buf.data(), nullptr);
      const int nq = dims[1];
      for (int x = 0; x < 3; x++) {
        Eigen::Map<const Mat> I(buf.data() + x * np * nq, np, nq);
        for (int i = 0; i < np; i++)
          for (int j = 0; j < nq; j++)
            VA[x](first[Q] + j, local[first[P] + i]) = -I(i, j);
      }
    }
  }
  return VA;
}

struct RIKSecondOrder {
  Mat H;
  std::vector<Mat> F;
};

template <ShellKind kind>
RIKSecondOrder rik_second_order(IntegralEngine &ao, IntegralEngine &aux,
                                const Mat &C, const CoulombMetric &metric,
                                bool want_hessian, bool want_fock,
                                bool fock_occ = false) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const int natom = static_cast<int>(aobasis.atoms().size());
  const int n3 = 3 * natom;
  const Eigen::Index nbf = aobasis.nbf(), naux = auxbasis.nbf();
  const Eigen::Index o = C.cols();
  const Eigen::Index o2 = o * o;
  const bool aux_sph = aux.is_spherical();

  RIKSecondOrder out;
  if (want_fock)
    out.F.assign(n3, Mat::Zero(nbf, fock_occ ? o : nbf));
  if (want_hessian)
    out.H = Mat::Zero(n3, n3);
  if (o == 0)
    return out;

  const Mat Bc = half_transform<kind>(ao, {&C})[0];
  const Mat Ec = fit(Bc, nbf, o, naux, metric);
  const Mat cmat = C.transpose() * Ec; // o x (o naux): c^P blocks
  Eigen::Map<const Mat> c_map(cmat.data(), o2, naux);
  Eigen::Map<const Mat> E_map(Ec.data(), nbf * o, naux);
  const Mat EcTC = fock_occ ? Mat(Ec.transpose() * C) : Mat(); // (o naux) x o

  // L^-1 u^X for every coordinate: (nkept o^2) x 3N, where nkept is naux
  // less any null space the metric factorisation dropped (the attenuated
  // metric's eigendecomposition), so sized on first use.
  Mat Ut;
  if (want_hessian)
    occ::log::debug("RI-K Hessian: {:.1f} MB for the coupling term",
                    8.0 * naux * o2 * n3 / 1e6);

  for (int A = 0; A < natom; A++) {
    const auto BX = half_transform_derivative<kind>(ao, C, A);
    std::vector<int> aux_A;
    const auto VA = aux_sph
                        ? metric_derivative_columns<ShellKind::Spherical>(
                              aux, A, aux_A)
                        : metric_derivative_columns<ShellKind::Cartesian>(
                              aux, A, aux_A);
    const Eigen::Index nA = aux_A.size();
    for (int x = 0; x < 3; x++) {
      const int X = 3 * A + x;
      if (want_fock && fock_occ) {
        // K^X C without nbf x nbf products: with Z = BX Ec^T - EA Gr^T,
        // K^X C = Z C + Z^T C.
        Mat ZC = BX[x] * EcTC;                      // nbf x o
        ZC.noalias() += Ec * (BX[x].transpose() * C); // Z^T C, first part
        if (nA > 0) {
          Mat G = E_map * VA[x]; // (nbf o) x nA
          Mat EA(nbf, o * nA);
          for (Eigen::Index p = 0; p < nA; p++)
            EA.middleCols(o * p, o) = Ec.middleCols(o * aux_A[p], o);
          Eigen::Map<const Mat> Gr(G.data(), nbf, o * nA);
          ZC.noalias() -= EA * (Gr.transpose() * C);
          ZC.noalias() -= Gr * (EA.transpose() * C);
        }
        out.F[X] = ZC;
      } else if (want_fock) {
        Mat Z = BX[x] * Ec.transpose();
        if (nA > 0) {
          Mat G = E_map * VA[x]; // (nbf o) x nA
          Mat EA(nbf, o * nA);
          for (Eigen::Index p = 0; p < nA; p++)
            EA.middleCols(o * p, o) = Ec.middleCols(o * aux_A[p], o);
          Eigen::Map<const Mat> Gr(G.data(), nbf, o * nA);
          Z -= EA * Gr.transpose();
        }
        out.F[X] = Z + Z.transpose();
      }
      if (want_hessian) {
        const Mat tX = C.transpose() * BX[x]; // o x (o naux)
        Mat u = Eigen::Map<const Mat>(tX.data(), o2, naux);
        if (nA > 0) {
          Mat cA(o2, nA);
          for (Eigen::Index p = 0; p < nA; p++)
            cA.col(p) = c_map.col(aux_A[p]);
          const Mat Vc_A = c_map * VA[x]; // rows P on A
          for (Eigen::Index p = 0; p < nA; p++)
            u.col(aux_A[p]) -= Vc_A.col(p);
          u.noalias() -= cA * VA[x].transpose(); // rows P not on A
        }
        const Mat ut = metric.half_inverse_apply(u.transpose()); // nkept x o2
        if (Ut.rows() != ut.size())
          Ut.setZero(ut.size(), n3);
        Ut.col(X) = Eigen::Map<const Vec>(ut.data(), ut.size());
      }
    }
  }
  if (!want_hessian)
    return out;

  out.H = 2.0 * Ut.transpose() * Ut;
  Ut.resize(0, 0);

  // 2 sum_P sum_mn (mn|P)^XY G^P_mn, G^P = Y^P C^T with Y = C c^P, and
  // -sum_PQ (c c^T)_PQ V^XY_PQ.
  const Mat Yc = C * cmat; // nbf x (o naux)
  const auto &first_ao = aobasis.first_bf();
  const auto &first_aux = auxbasis.first_bf();
  out.H += three_center_hessian<kind>(
      ao, 2.0, [&](int p, int q, int R, int nr, std::vector<Mat> &W) {
        for (int r = 0; r < nr; r++)
          W[r] = Yc.block(first_ao[p], o * (first_aux[R] + r),
                          aobasis[p].size(), o) *
                 C.middleRows(first_ao[q], aobasis[q].size()).transpose();
      });
  const Mat M = c_map.transpose() * c_map; // naux x naux
  out.H += metric_hessian(aux, -1.0, matrix_metric_contraction(aux, M));
  out.H = 0.5 * (out.H + out.H.transpose());
  return out;
}

template <ShellKind kind>
std::vector<Mat> rik_response_impl(IntegralEngine &ao, const Mat &C,
                                   const std::vector<Mat> &Cx,
                                   const CoulombMetric &metric) {
  const Eigen::Index nbf = ao.aobasis().nbf(), naux = ao.auxbasis().nbf();
  const Eigen::Index o = C.cols();
  std::vector<Mat> K(Cx.size(), Mat::Zero(nbf, nbf));
  if (o == 0 || Cx.empty())
    return K;
  const Mat Ec = fit(half_transform<kind>(ao, {&C})[0], nbf, o, naux, metric);
  // Batches bounded to ~1 GB of half-transformed tensors.
  const size_t per = static_cast<size_t>(nbf) * o * naux * sizeof(double);
  const size_t batch = std::max<size_t>(1, (size_t(1) << 30) / per);
  for (size_t k0 = 0; k0 < Cx.size(); k0 += batch) {
    std::vector<const Mat *> Cs;
    for (size_t k = k0; k < std::min(Cx.size(), k0 + batch); k++)
      Cs.push_back(&Cx[k]);
    const auto Bx = half_transform<kind>(ao, Cs);
    for (size_t k = 0; k < Bx.size(); k++) {
      const Mat Z = Bx[k] * Ec.transpose();
      K[k0 + k] = Z + Z.transpose();
    }
  }
  return K;
}

// K[D1] C for D1 = Cx C^T + C Cx^T, all densities in one integral pass:
//
//   K[D1] C = Z C + Z^T C,  Z = sum_P Bx^P E^P^T,  Bx^P = (. nu|P) Cx,
//   Z C   = sum_P Bx^P (E^P^T C),
//   Z^T C = sum_P E^P (Bx^P^T C),
//
// so each auxiliary function's Bx^P is contracted as soon as it is formed
// and never stored.
template <ShellKind kind>
std::vector<Mat> rik_response_occ_impl(IntegralEngine &ao, const Mat &C,
                                       const std::vector<Mat> &Cx,
                                       const CoulombMetric &metric) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const auto &first_aux = auxbasis.first_bf();
  const Eigen::Index nbf = aobasis.nbf(), naux = auxbasis.nbf();
  const Eigen::Index o = C.cols();
  const Eigen::Index m = Cx.size();
  std::vector<Mat> out(m, Mat::Zero(nbf, o));
  if (o == 0 || m == 0)
    return out;
  const Mat Ec = fit(half_transform<kind>(ao, {&C})[0], nbf, o, naux, metric);
  // F^P = E^P^T C, o x o per auxiliary function, stacked as (o naux) x o.
  Mat F(o * naux, o);
  for (Eigen::Index P = 0; P < naux; P++)
    F.middleRows(o * P, o) = Ec.middleCols(o * P, o).transpose() * C;
  Mat Cstack(nbf, o * m);
  for (Eigen::Index k = 0; k < m; k++)
    Cstack.middleCols(o * k, o) = Cx[k];

  // Per thread: sum_P Bx^P F^P (nbf x o m), and the stacked Bx^P^T C
  // ((o naux) x (o m)) whose rows for P a task owns are written by it alone.
  Mat G(o * naux, o * m);
  occ::parallel::thread_local_storage<Mat> tl_zc(
      [nbf, o, m]() { return Mat(Mat::Zero(nbf, o * m)); });
  cint::Optimizer opt(ao.env(), Op::coulomb, 3);
  occ::parallel::parallel_for(size_t(0), auxbasis.size(), [&](size_t R) {
    const int nr = auxbasis[R].size();
    const auto M = aux_shell_matrices<kind>(ao, opt, R);
    auto &zc = tl_zc.local();
    Mat T;
    for (int r = 0; r < nr; r++) {
      const Eigen::Index P = first_aux[R] + r;
      T.noalias() = M[r] * Cstack; // Bx^P for every density, nbf x (o m)
      const auto FP = F.middleRows(o * P, o);
      for (Eigen::Index k = 0; k < m; k++)
        zc.middleCols(o * k, o).noalias() += T.middleCols(o * k, o) * FP;
      G.middleRows(o * P, o).noalias() = C.transpose() * T; // (Bx^P^T C)^T
    }
  });
  Mat ZC = Mat::Zero(nbf, o * m);
  for (const auto &t : tl_zc)
    ZC += t;
  // Z^T C = sum_P E^P (Bx^P^T C); G holds C^T Bx^P, i.e. its transpose
  // per o x o block.
  for (Eigen::Index k = 0; k < m; k++) {
    Mat Gk(o * naux, o);
    for (Eigen::Index P = 0; P < naux; P++)
      Gk.middleRows(o * P, o) =
          G.block(o * P, o * k, o, o).transpose();
    out[k] = ZC.middleCols(o * k, o) + Ec * Gk;
  }
  return out;
}

// Occupied coefficients per spin: the restricted per-spin density is
// C C^T, the unrestricted spin densities P^s = C^s C^s^T.
std::vector<Mat> spin_occupied(const MolecularOrbitals &mo, const char *what) {
  switch (mo.kind) {
  case SpinorbitalKind::Restricted:
    return {mo.Cocc};
  case SpinorbitalKind::Unrestricted:
    return {block::a(mo.Cocc).leftCols(mo.n_alpha),
            block::b(mo.Cocc).leftCols(mo.n_beta)};
  default:
    throw std::runtime_error(std::string(what) +
                             ": general spinorbitals not supported");
  }
}

RIKSecondOrder rik_second_order(IntegralEngine &ao, IntegralEngine &aux,
                                const Mat &C, const CoulombMetric &metric,
                                bool hessian, bool fock,
                                bool fock_occ = false) {
  return ao.is_spherical()
             ? rik_second_order<ShellKind::Spherical>(ao, aux, C, metric,
                                                      hessian, fock, fock_occ)
             : rik_second_order<ShellKind::Cartesian>(ao, aux, C, metric,
                                                      hessian, fock, fock_occ);
}
} // namespace

Mat IntegralEngineDF::exchange_hessian(const MolecularOrbitals &mo) {
  const auto C = spin_occupied(mo, "RI-K Hessian");
  occ::timing::start(occ::timing::category::df);
  // Unrestricted E_K = 1/2 sum_s tr(P^s K[P^s]), restricted tr(D K[D]).
  const double scale = C.size() == 1 ? 1.0 : 0.5;
  const Eigen::Index n3 = 3 * m_ao_engine.aobasis().atoms().size();
  Mat H = Mat::Zero(n3, n3);
  for (const auto &c : C)
    H += scale *
         rik_second_order(m_ao_engine, m_aux_engine, c, V_LLt, true, false).H;
  occ::timing::stop(occ::timing::category::df);
  return H;
}

// Unrestricted: K^x[P^s] per spin, alpha rows over beta rows.
std::vector<Mat>
IntegralEngineDF::exchange_derivative_fock(const MolecularOrbitals &mo) {
  const auto C = spin_occupied(mo, "RI-K derivative Fock");
  occ::timing::start(occ::timing::category::df);
  std::vector<Mat> F;
  for (size_t s = 0; s < C.size(); s++) {
    const auto Fs =
        rik_second_order(m_ao_engine, m_aux_engine, C[s], V_LLt, false, true)
            .F;
    const Eigen::Index nbf = Fs[0].rows();
    F.resize(Fs.size(), Mat(C.size() * nbf, nbf));
    for (size_t k = 0; k < Fs.size(); k++)
      F[k].middleRows(s * nbf, nbf) = Fs[k];
  }
  occ::timing::stop(occ::timing::category::df);
  return F;
}

std::vector<Mat> IntegralEngineDF::exchange_response(const Mat &C,
                                                     const std::vector<Mat> &Cx) {
  occ::timing::start(occ::timing::category::df);
  auto K = m_ao_engine.is_spherical()
               ? rik_response_impl<ShellKind::Spherical>(m_ao_engine, C, Cx,
                                                         V_LLt)
               : rik_response_impl<ShellKind::Cartesian>(m_ao_engine, C, Cx,
                                                         V_LLt);
  occ::timing::stop(occ::timing::category::df);
  return K;
}

std::vector<Mat>
IntegralEngineDF::exchange_response_occ(const Mat &C,
                                        const std::vector<Mat> &Cx) {
  occ::timing::start(occ::timing::category::df);
  auto K = m_ao_engine.is_spherical()
               ? rik_response_occ_impl<ShellKind::Spherical>(m_ao_engine, C,
                                                             Cx, V_LLt)
               : rik_response_occ_impl<ShellKind::Cartesian>(m_ao_engine, C,
                                                             Cx, V_LLt);
  occ::timing::stop(occ::timing::category::df);
  return K;
}

// Unrestricted: K^x[P^s] C^s side by side, alpha then beta.
std::vector<Mat>
IntegralEngineDF::exchange_derivative_fock_occ(const MolecularOrbitals &mo) {
  const auto C = spin_occupied(mo, "RI-K derivative Fock");
  occ::timing::start(occ::timing::category::df);
  std::vector<Mat> F;
  Eigen::Index col = 0, ncol = 0;
  for (const auto &c : C)
    ncol += c.cols();
  for (const auto &c : C) {
    const auto Fs = rik_second_order(m_ao_engine, m_aux_engine, c, V_LLt,
                                     false, true, true)
                        .F;
    F.resize(Fs.size(), Mat(Fs[0].rows(), ncol));
    for (size_t k = 0; k < Fs.size(); k++)
      F[k].middleCols(col, c.cols()) = Fs[k];
    col += c.cols();
  }
  occ::timing::stop(occ::timing::category::df);
  return F;
}

} // namespace occ::qm
