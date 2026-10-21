// Relaxed MP2 density of a closed-shell RHF reference: the MP2 Lagrangian with
// the Z-vector orbital response (PySCF grad/mp2.py; Weigend & Haser, Theor.
// Chem. Acc. 97, 331 (1997) for the RI form).
//
// The MP2 part (amplitudes, the unrelaxed oo / vv blocks and the Lagrangian I)
// comes either from occ's exact MO integral blocks or, with the df backend,
// from the metric-folded B tensors one occupied orbital at a time: nothing of
// size o v^3 or v^4 is formed, only O(o v n_aux) B and Gamma plus an o v^2
// amplitude buffer and B_vv streamed in blocks. The SCF-like parts (the
// relaxed density's Fock response X and the orbital Hessian in the Z-vector
// equations) are AO Coulomb / exchange builds, with the SCF's own engines when
// opts.scf_jk is given.
#include "detail/density.h"
#include <Eigen/Dense>
#include <chrono>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/qm/cc/gemm.h>
#include <occ/qm/correlation/df_integrals.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>
#include <tbb/enumerable_thread_specific.h>

namespace occ::qm::cc {

namespace {
using namespace detail;

// The MP2 Lagrangian's pieces over all o occupied orbitals (frozen core
// included, with zero amplitudes) and v virtuals; PySCF grad/mp2.py:
//   P_ij = -2 t_ki^ab tt_kj^ab, P_ab = 2 t_ij^ca tt_ij^cb (tt = 2 t - t(ba)),
//   I_kc = -2 (ik|jb) tt_ij^cb, I_dk = -2 (ad|jb) tt_kj^ab,
//   I_mk = -2 (ma|jb) tt_kj^ab
struct Mp2Lagrangian {
  double e2 = 0.0;
  occ::Mat p_oo, p_vv; // (o x o), (v x v)
  occ::Mat i_oc;       // (o x v)
  occ::Mat i_vo;       // (v x o)
  occ::Mat i_oo;       // (o x o)
};

// from occ's MO integral blocks (built with no frozen core)
Mp2Lagrangian lagrangian_from_blocks(const CCIntegrals &e, const int nf) {
  const int o = e.nocc, v = e.nvir;
  const occ::Vec eo = e.mo_energy.head(o), ev = e.mo_energy.tail(v);
  const T4 g = transpose(e.ovov, {0, 2, 1, 3});
  T4 t2 = g;
  for (int b = 0; b < v; b++)
    for (int a = 0; a < v; a++)
      for (int j = 0; j < o; j++)
        for (int i = 0; i < o; i++)
          t2(i, j, a, b) =
              i < nf || j < nf
                  ? 0.0
                  : g(i, j, a, b) / (eo(i) + eo(j) - ev(a) - ev(b));
  const T4 tt = t2 * 2.0 - transpose(t2, {0, 1, 3, 2});
  Mp2Lagrangian L;
  L.e2 = dot(tt, g);
  L.p_vv = as_matrix(einsum<2>("ijca,ijcb->ab", t2, tt) * 2.0);
  L.p_oo = as_matrix(einsum<2>("kiab,kjab->ij", t2, tt) * -2.0);
  L.i_oc = as_matrix(einsum<2>("jbik,ijcb->kc", e.ovoo, tt) * -2.0);
  L.i_vo = as_matrix(einsum<2>("jbad,kjab->dk", e.ovvv, tt) * -2.0);
  L.i_oo = as_matrix(einsum<2>("majb,kjab->mk", e.ovov, tt) * -2.0);
  return L;
}

// from the metric-folded B tensors, (ia|jb) = sum_P B(ia,P) B(jb,P), one
// correlated occupied k at a time:
//   K_k(jb, a) = B B_k^T            one GEMM for every (jb|ka)
//   t_kj = K_kj / D, tt_kj = 2 t_kj - t_kj^T
//   Gamma_k(a,P) = sum_jb tt_kj(a,b) B(jb,P)
// and from Gamma the three I blocks: I_mk and I_kc through B_ov and B_oo, I_dk
// through B_vv in blocks of a
Mp2Lagrangian lagrangian_from_df(DFIntegrals &df, const occ::Mat &C_occ,
                                 const occ::Mat &C_vir, const occ::Vec &eo,
                                 const occ::Vec &ev, const int nf,
                                 const size_t budget) {
  const int o = static_cast<int>(C_occ.cols()),
            v = static_cast<int>(C_vir.cols());
  const int oc = o - nf; // correlated occupied
  const Eigen::Index naux = static_cast<Eigen::Index>(df.naux());
  const Eigen::Index vv = static_cast<Eigen::Index>(v) * v;
  const occ::Mat B = df.build_b_tilde(C_occ, C_vir); // row i v + a
  const auto Bc = B.middleRows(static_cast<Eigen::Index>(nf) * v,
                               static_cast<Eigen::Index>(oc) * v);
  Mp2Lagrangian L;
  L.p_oo = occ::Mat::Zero(o, o);
  L.p_vv = occ::Mat::Zero(v, v);
  occ::Mat Gamma = occ::Mat::Zero(static_cast<Eigen::Index>(oc) * v, naux);
  occ::Mat K(static_cast<Eigen::Index>(oc) * v, v),
      Tk(v, static_cast<Eigen::Index>(oc) * v), TTk(v, Tk.cols());
  tbb::enumerable_thread_specific<occ::Mat> p_vv_local(
      [v] { return occ::Mat::Zero(v, v); });
  tbb::enumerable_thread_specific<double> e2_local(0.0);
  for (int k = 0; k < oc; k++) {
    // K(j v + b, a) = (jb|ka)
    const occ::Mat Bk_t =
        Bc.middleRows(static_cast<Eigen::Index>(k) * v, v).transpose();
    pgemm(K, Bc, Bk_t);
    const double ek = eo(nf + k);
    occ::parallel::parallel_for(size_t(0), size_t(oc), [&](size_t ju) {
      const Eigen::Index j = static_cast<Eigen::Index>(ju);
      const double ekj = ek + eo(nf + j);
      auto t = Tk.middleCols(j * v, v);
      auto tt = TTk.middleCols(j * v, v);
      const auto Kkj_t = K.middleRows(j * v, v); // (b, a)
      for (int b = 0; b < v; b++)
        for (int a = 0; a < v; a++)
          t(a, b) = Kkj_t(b, a) / (ekj - ev(a) - ev(b));
      tt = 2.0 * t - t.transpose();
      e2_local.local() += tt.cwiseProduct(Kkj_t.transpose()).sum();
      p_vv_local.local().noalias() += 2.0 * t.transpose() * tt;
    });
    // P_ij -= 2 <t_ki | tt_kj>: the columns of T_k are vec(t_kj)
    const Eigen::Map<const occ::Mat> Tv(Tk.data(), vv, oc),
        TTv(TTk.data(), vv, oc);
    occ::Mat pk(oc, oc);
    pgemm(pk, Tv, TTv, false, true);
    L.p_oo.block(nf, nf, oc, oc) -= 2.0 * pk;
    // Gamma_k = TT_k B (columns j v + b of TT_k pair rows j v + b of B)
    pgemm(Gamma.middleRows(static_cast<Eigen::Index>(k) * v, v), TTk, Bc);
  }
  for (const double x : e2_local)
    L.e2 += x;
  for (const occ::Mat &x : p_vv_local)
    L.p_vv += x;

  // I_mk = -2 sum_aP B(ma,P) Gamma(ka,P)
  L.i_oo = occ::Mat::Zero(o, o);
  occ::parallel::parallel_for(size_t(0), size_t(o), [&](size_t mu) {
    const Eigen::Index m = static_cast<Eigen::Index>(mu);
    for (int k = 0; k < oc; k++)
      L.i_oo(m, nf + k) = -2.0 * B.middleRows(m * v, v)
                                     .cwiseProduct(Gamma.middleRows(
                                         static_cast<Eigen::Index>(k) * v, v))
                                     .sum();
  });
  // I_kc = -2 sum_iP B(ik,P) Gamma(ic,P), i correlated
  {
    const occ::Mat Boo = df.build_b_tilde(C_occ, C_occ); // row i o + k
    L.i_oc = occ::Mat::Zero(o, v);
    for (int i = 0; i < oc; i++)
      L.i_oc.noalias() -=
          2.0 * Boo.middleRows(static_cast<Eigen::Index>(nf + i) * o, o) *
          Gamma.middleRows(static_cast<Eigen::Index>(i) * v, v).transpose();
  }
  // I_dk = -2 sum_aP B(ad,P) Gamma(ka,P), B_vv built na rows of a at a time
  L.i_vo = occ::Mat::Zero(v, o);
  {
    const size_t row_bytes = static_cast<size_t>(v) * naux * sizeof(double);
    const int na = static_cast<int>(std::clamp<size_t>(
        budget / std::max<size_t>(1, 4 * row_bytes), 1, size_t(v)));
    occ::Mat G(oc, naux);
    for (int a0 = 0; a0 < v; a0 += na) {
      const int nb = std::min(na, v - a0);
      const occ::Mat Bvv = df.build_b_tilde(C_vir.middleCols(a0, nb), C_vir);
      for (int al = 0; al < nb; al++) {
        for (int k = 0; k < oc; k++)
          G.row(k) = Gamma.row(static_cast<Eigen::Index>(k) * v + a0 + al);
        L.i_vo.rightCols(oc).noalias() -=
            2.0 * Bvv.middleRows(static_cast<Eigen::Index>(al) * v, v) *
            G.transpose();
      }
    }
  }
  return L;
}

} // namespace

CorrelatedDensity mp2_relaxed_density(const occ::qm::Wavefunction &scf,
                                      const CorrelatedDensityOptions &opts) {
  check(scf.mo.kind == occ::qm::SpinorbitalKind::Restricted &&
            scf.mo.n_alpha == scf.mo.n_beta,
        "The MP2 density needs a closed-shell (restricted) reference");
  CorrelatedDensity result;
  auto clock = std::chrono::steady_clock::now();
  auto mark = [&clock](const char *stage) {
    const auto now = std::chrono::steady_clock::now();
    occ::log::info("MP2 density: {} took {:.3f} s", stage,
                   std::chrono::duration<double>(now - clock).count());
    clock = now;
  };
  const int o = static_cast<int>(scf.mo.n_alpha),
            nmo = static_cast<int>(scf.mo.C.cols()), v = nmo - o;
  result.n_frozen = frozen_core_count(scf, opts.n_frozen);
  const int nf = result.n_frozen;
  const occ::Mat C_occ = scf.mo.C.leftCols(o), C_vir = scf.mo.C.rightCols(v);
  const occ::Vec eo = scf.mo.energies.head(o),
                 ev = scf.mo.energies.segment(o, v);

  // the MP2 part over every occupied orbital: the orbital response couples
  // the frozen core
  std::string aux_name;
  result.backend = resolve_backend(opts);
  Mp2Lagrangian L;
  if (result.backend == "exact") {
    L = lagrangian_from_blocks(
        exact_eris(scf.basis, scf.mo, 0, memory_budget(opts), false), nf);
  } else {
    const auto aux = load_aux_basis(scf.basis, opts, aux_name);
    IntegralEngineDF df_engine(scf.basis.atoms(), scf.basis.shells(),
                               aux.shells());
    DFIntegrals df(df_engine, memory_budget(opts));
    L = lagrangian_from_df(df, C_occ, C_vir, eo, ev, nf, memory_budget(opts));
  }
  result.e_corr = L.e2;
  mark("MP2 Lagrangian");
  // the frozen-core block of P: P_fk = I_fk / (e_f - e_k)
  for (int f = 0; f < nf; f++)
    for (int k = nf; k < o; k++)
      L.p_oo(f, k) = L.p_oo(k, f) = L.i_oo(f, k) / (eo(f) - eo(k));

  // AO Coulomb and exchange of a symmetric MO-basis density M (D = C M C^T).
  // occ's restricted engines return 2 J[D] and K[D] for the density in mo.D
  const IntegralEngine engine(scf.basis);
  const occ::Mat schwarz = engine.schwarz();
  const HartreeFock *scf_jk = opts.scf_jk;
  // DF exchange is built from mo.Cocc, not mo.D (COSX and the four-centre
  // build read mo.D), so it needs D = C+ C+^T - C- C-^T, from M's eigenpairs
  const bool factored =
      scf_jk && scf_jk->using_density_fitting() && !scf_jk->using_cosx();
  occ::qm::MolecularOrbitals jk_mo = scf.mo;
  auto build = [&](const occ::Mat &D, const occ::Mat &Cocc) {
    jk_mo.D = D;
    jk_mo.Cocc = Cocc;
    return scf_jk ? scf_jk->compute_JK(jk_mo, schwarz)
                  : engine.coulomb_and_exchange(
                        occ::qm::SpinorbitalKind::Restricted, jk_mo, schwarz);
  };
  auto jk = [&](const occ::Mat &M) {
    if (!factored)
      return build(scf.mo.C * M * scf.mo.C.transpose(), scf.mo.Cocc);
    Eigen::SelfAdjointEigenSolver<occ::Mat> eig(M);
    const occ::Vec &w = eig.eigenvalues();
    const double cut = 1e-14 * w.cwiseAbs().maxCoeff();
    const auto nbf = scf.mo.C.rows();
    occ::qm::JKPair r{occ::Mat::Zero(nbf, nbf), occ::Mat::Zero(nbf, nbf)};
    for (const double sign : {1.0, -1.0}) {
      std::vector<int> keep;
      for (int k = 0; k < w.size(); k++)
        if (sign * w(k) > cut)
          keep.push_back(k);
      if (keep.empty())
        continue;
      occ::Mat Cf(scf.mo.C.rows(), keep.size());
      for (size_t k = 0; k < keep.size(); k++)
        Cf.col(k) = std::sqrt(sign * w(keep[k])) *
                    (scf.mo.C * eig.eigenvectors().col(keep[k]));
      const auto part = build(Cf * Cf.transpose(), Cf);
      r.J += sign * part.J;
      r.K += sign * part.K;
    }
    return r;
  };
  // X_am = sum_pq P_pq [2 (am|pq) - (ap|qm)] + I_ma - I_am
  occ::Mat x;
  {
    occ::Mat P = occ::Mat::Zero(nmo, nmo);
    P.topLeftCorner(o, o) = L.p_oo;
    P.bottomRightCorner(v, v) = L.p_vv;
    const auto r = jk(P);
    x = C_vir.transpose() * (r.J - r.K) * C_occ + L.i_oc.transpose() - L.i_vo;
  }
  // Z-vector: (e_a - e_m) z_am + sum_bn [4 (am|bn) - (ab|mn) - (an|bm)] z_bn
  // = -X_am, i.e. C_v^T (4 J - 2 K)[sym(C_v z C_o^T)] C_o; conjugate gradients
  // preconditioned by e_a - e_m (the RHF orbital Hessian is positive definite)
  occ::Mat d_vo(v, o);
  for (int m = 0; m < o; m++)
    for (int a = 0; a < v; a++)
      d_vo(a, m) = ev(a) - eo(m);
  // applied to z scaled to unit size and scaled back: the engine screens
  // absolutely (on integral times density), so the small late CG directions
  // would otherwise lose relative precision and stall CG on a noise floor
  auto hessian = [&](const occ::Mat &z) {
    const double scale = z.cwiseAbs().maxCoeff();
    if (scale == 0.0)
      return occ::Mat(occ::Mat::Zero(v, o));
    occ::Mat Mz = occ::Mat::Zero(nmo, nmo);
    Mz.bottomLeftCorner(v, o) = 0.5 * z / scale;
    Mz.topRightCorner(o, v) = Mz.bottomLeftCorner(v, o).transpose();
    const auto r = jk(Mz);
    return occ::Mat(scale * C_vir.transpose() * (2.0 * r.J - 2.0 * r.K) *
                        C_occ +
                    d_vo.cwiseProduct(z));
  };
  const occ::Mat b = -x;
  occ::Mat z = b.cwiseQuotient(d_vo);
  occ::Mat res = b - hessian(z);
  occ::Mat s = res.cwiseQuotient(d_vo), dir = s;
  double rs = res.cwiseProduct(s).sum();
  int it = 0;
  // converged on max |residual|, opts.tol as for the CC amplitudes and Lambda
  for (; it < opts.max_cycle && res.cwiseAbs().maxCoeff() > opts.tol; it++) {
    const occ::Mat hd = hessian(dir);
    const double alpha = rs / dir.cwiseProduct(hd).sum();
    z += alpha * dir;
    res -= alpha * hd;
    s = res.cwiseQuotient(d_vo);
    const double rs_new = res.cwiseProduct(s).sum();
    dir = s + (rs_new / rs) * dir;
    rs = rs_new;
  }
  check(res.cwiseAbs().maxCoeff() <= opts.tol,
        "The MP2 Z-vector equations did not converge in " +
            std::to_string(opts.max_cycle) + " iterations");
  result.lambda_iterations = it;
  mark("Z-vector");

  occ::Mat dm = occ::Mat::Zero(nmo, nmo);
  dm.topLeftCorner(o, o) = L.p_oo;
  dm.topLeftCorner(o, o).diagonal().array() += 2.0;
  dm.bottomLeftCorner(v, o) = z;
  dm.topRightCorner(o, v) = z.transpose();
  dm.bottomRightCorner(v, v) = L.p_vv;
  natural_orbitals(scf, dm, "MP2", scf.energy.total + result.e_corr, result);
  mark("density and natural orbitals");
  result.n_occ = o - nf;
  result.n_vir = v;
  result.aux_basis = aux_name;
  return result;
}

} // namespace occ::qm::cc
