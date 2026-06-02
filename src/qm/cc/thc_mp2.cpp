#include <algorithm>
#include <cmath>
#include <limits>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/cc/laplace.h>
#include <occ/qm/cc/thc_mp2.h>
#include <occ/qm/correlation/df_integrals.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/opmatrix.h> // block::a / block::b

namespace occ::qm::cc {

namespace {

using occ::qm::DFIntegrals;
using occ::qm::IntegralEngineDF;

// Scale the rows of an (nmo_block x P) collocation by exp(s * e_row), i.e.
// diag(exp(s*e)) * X. For occupied rows use s = +t/2, for virtual s = -t/2, so
// the squared scaled integral carries exp(-(e_a+e_b-e_i-e_j) t).
Mat scale_rows(const Mat &X, const Vec &e, double s) {
  Mat out = X;
  for (Eigen::Index r = 0; r < X.rows(); ++r)
    out.row(r) *= std::exp(s * e(r));
  return out;
}

// Frobenius inner product sum_PR A(P,R) B(P,R).
inline double frob(const Mat &A, const Mat &B) {
  return (A.cwiseProduct(B)).sum();
}

// Coulomb contraction <B, Vl B' Vr>_F for one Laplace point. For same-spin
// B == B' and Vl == Vr == V (symmetric); for cross-spin B=Ba, B'=Bb, Vl=Vab,
// Vr=Vab^T. Equals sum_{ijab} (ia|jb)^2 (or (iαaα|jβbβ)^2) at this point.
inline double coulomb_term(const Mat &B, const Mat &Vl, const Mat &Bp,
                           const Mat &Vr) {
  return frob(B, Vl * Bp * Vr);
}

// Same-spin exchange sum_{ijab} (ia|jb)(ib|ja) for one Laplace point, in the
// orbital-intermediate form: O(o^2 v^2 P) and GEMM-heavy, vs the equivalent
// O(P^4) interpolation-point contraction (P = c*nbf is large, so this is ~100x
// fewer flops at realistic sizes). Mo (o x P) and Mv (v x P) are the
// Laplace-scaled occupied/virtual collocations; V is the THC core (P x P,
// symmetric for same spin).
//   MvO_i(b,Q) = Mo(i,Q) Mv(b,Q)                 [v x P]
//   Z_i(a,Q)   = sum_P Mo(i,P) Mv(a,P) V(P,Q)    = (MvO_i V)            [v x P]
//   K_ij(a,b)  = sum_Q Z_i(a,Q) MvO_j(b,Q)       = Z_i MvO_j^T = (ia|jb)[v x v]
//   result     = sum_ij sum_ab K_ij(a,b) K_ij(b,a)
// Parallel over the outer occupied index with i>=j symmetry (factor 2
// off-diag).
double exchange_orbital(const Mat &Mo, const Mat &Mv, const Mat &V) {
  const size_t o = static_cast<size_t>(Mo.rows());
  const size_t v = static_cast<size_t>(Mv.rows());
  if (o == 0 || v == 0)
    return 0.0;

  // MvO[i] (v x P): Mv with column Q scaled by Mo(i,Q). Needed as the right
  // factor of K for every pair, so build all o of them once.
  std::vector<Mat> MvO(o);
  occ::parallel::parallel_for(size_t(0), o, [&](size_t i) {
    MvO[i] =
        (Mv.array().rowwise() * Mo.row(static_cast<Eigen::Index>(i)).array())
            .matrix();
  });

  std::vector<double> partial(o, 0.0);
  occ::parallel::parallel_for(size_t(0), o, [&](size_t i) {
    const Mat Zi = MvO[i] * V; // (v x P)
    double e = 0.0;
    for (size_t j = 0; j <= i; ++j) {
      const Mat K = Zi * MvO[j].transpose(); // (v x v) = (ia|jb)
      const double tr = (K.cwiseProduct(K.transpose())).sum();
      e += (i == j ? 1.0 : 2.0) * tr;
    }
    partial[i] = e;
  });

  double total = 0.0;
  for (double e : partial)
    total += e;
  return total;
}

// Make a kind-matched DFIntegrals for the given AO + auxiliary basis.
DFIntegrals make_df(const AOBasis &basis, const AOBasis &aux_basis,
                    std::size_t budget,
                    std::unique_ptr<IntegralEngineDF> &engine_out) {
  AOBasis aux = aux_basis;
  aux.set_kind(basis.kind());
  engine_out = std::make_unique<IntegralEngineDF>(basis.atoms(), basis.shells(),
                                                  aux.shells());
  return DFIntegrals(*engine_out, budget);
}

ThcMP2Result thc_mp2_restricted(const AOBasis &basis, const AOBasis &aux_basis,
                                const MolecularOrbitals &mo_in,
                                const ThcMP2Options &opts) {
  // Active space: drop the lowest n_frozen occupied orbitals.
  const int nf = opts.n_frozen;
  const int o = static_cast<int>(mo_in.n_alpha) - nf;
  const int nmo = static_cast<int>(mo_in.C.cols()) - nf;
  const int v = nmo - o;

  ThcMP2Result r;
  if (o <= 0 || v <= 0)
    return r;

  const Mat C = mo_in.C.middleCols(nf, nmo).eval();
  const Vec eps = mo_in.energies.segment(nf, nmo).eval();
  const Vec eo = eps.head(o);
  const Vec ev = eps.tail(v);

  // LS-THC factors over the active MOs, with the core V fit to the occ-virt
  // block only -- the integrals MP2 needs. Much cheaper than the all-pairs fit
  // (o*v << nmo^2 reference columns) and more accurate where it counts.
  std::unique_ptr<IntegralEngineDF> engine;
  DFIntegrals df = make_df(basis, aux_basis, opts.memory_budget, engine);
  MolecularOrbitals mo = mo_in;
  mo.C = C;
  occ::timing::start(occ::timing::category::thc_factorize);
  const Mat X = thc_select_collocation(basis, mo, opts.thc); // (nmo x P)
  const int P = static_cast<int>(X.cols());
  const Mat Xo = X.topRows(o);    // (o x P)
  const Mat Xv = X.bottomRows(v); // (v x P)
  const Mat B_ov =
      df.build_b_tilde(C.leftCols(o), C.middleCols(o, v)); // (o*v x naux)
  occ::timing::start(occ::timing::category::thc_fit);
  const Mat V = fit_core_ov(Xo, Xv, B_ov, opts.thc.reg, opts.thc.reg_type);
  occ::timing::stop(occ::timing::category::thc_fit);
  occ::timing::stop(occ::timing::category::thc_factorize);

  // Laplace grid for 1/(e_a+e_b-e_i-e_j) over the whole pair-gap range.
  const double dmin = 2.0 * (ev.minCoeff() - eo.maxCoeff());
  const double dmax = 2.0 * (ev.maxCoeff() - eo.minCoeff());
  const LaplaceGrid g = laplace_grid(dmin, dmax, opts.n_laplace);

  double coul = 0.0, exch = 0.0;
  occ::timing::start(occ::timing::category::mp2_energy);
  for (int k = 0; k < g.size(); ++k) {
    const double t = g.points(k), w = g.weights(k);
    const Mat Mo = scale_rows(Xo, eo, 0.5 * t);  // exp(+e_i t/2)
    const Mat Mv = scale_rows(Xv, ev, -0.5 * t); // exp(-e_a t/2)
    const Mat Go = Mo.transpose() * Mo;          // (P x P)
    const Mat Gv = Mv.transpose() * Mv;          // (P x P)
    const Mat Bm = Go.cwiseProduct(Gv);
    coul += w * coulomb_term(Bm, V, Bm, V); // O(P^3) cubic opposite-spin
    if (!opts.opposite_spin_only)
      exch += w * exchange_orbital(Mo, Mv, V); // O(o^2 v^2 P) same-spin
  }
  occ::timing::stop(occ::timing::category::mp2_energy);

  r.opposite_spin = -coul;
  r.same_spin = opts.opposite_spin_only ? 0.0 : (-coul + exch);
  r.total = r.same_spin + r.opposite_spin;
  r.n_isdf = P;
  r.n_laplace = g.size();
  r.laplace_max_rel_error = laplace_max_rel_error(g, dmin, dmax);
  occ::log::debug("THC-MP2 (restricted): {} interp pts, {} Laplace pts "
                  "(max rel err {:.2e}); E_os={:.8f} E_ss={:.8f}",
                  P, g.size(), r.laplace_max_rel_error, r.opposite_spin,
                  r.same_spin);
  return r;
}

ThcMP2Result thc_mp2_unrestricted(const AOBasis &basis,
                                  const AOBasis &aux_basis,
                                  const MolecularOrbitals &mo,
                                  const ThcMP2Options &opts) {
  using occ::qm::block::a;
  using occ::qm::block::b;
  const int nf = opts.n_frozen;
  const Eigen::Index nbf = static_cast<Eigen::Index>(mo.n_ao);
  const Mat Ca = a(mo.C);
  const Mat Cb = b(mo.C);
  const Vec ea_all = mo.energies.head(nbf);
  const Vec eb_all = mo.energies.segment(nbf, nbf);

  const int oa = static_cast<int>(mo.n_alpha) - nf;
  const int ob = static_cast<int>(mo.n_beta) - nf;
  const int nmoa = static_cast<int>(Ca.cols()) - nf;
  const int nmob = static_cast<int>(Cb.cols()) - nf;
  const int va = nmoa - oa, vb = nmob - ob;

  ThcMP2Result r;
  if (oa <= 0 && ob <= 0)
    return r;

  const Mat Caa = Ca.middleCols(nf, nmoa).eval(); // active alpha
  const Mat Cbb = Cb.middleCols(nf, nmob).eval(); // active beta
  const Vec ea = ea_all.segment(nf, nmoa).eval();
  const Vec eb = eb_all.segment(nf, nmob).eval();
  const Vec eoa = ea.head(oa), eva = ea.tail(va);
  const Vec eob = eb.head(ob), evb = eb.tail(vb);

  std::unique_ptr<IntegralEngineDF> engine;
  DFIntegrals df = make_df(basis, aux_basis, opts.memory_budget, engine);
  const Mat Ba = df.build_b_tilde(Caa, Caa);
  const Mat Bb = df.build_b_tilde(Cbb, Cbb);
  const UThcFactors f = build_uthc(basis, Caa, Cbb, Ba, Bb, opts.thc);
  const int P = f.n_isdf;
  const Mat Xoa = f.Xa.topRows(oa), Xva = f.Xa.bottomRows(va);
  const Mat Xob = f.Xb.topRows(ob), Xvb = f.Xb.bottomRows(vb);

  const bool have_a = (oa > 0 && va > 0);
  const bool have_b = (ob > 0 && vb > 0);

  // Laplace range over all active channels (αα, ββ, αβ).
  double dmin = std::numeric_limits<double>::infinity();
  double dmax = -std::numeric_limits<double>::infinity();
  auto extend = [&](double lo, double hi) {
    dmin = std::min(dmin, lo);
    dmax = std::max(dmax, hi);
  };
  if (have_a)
    extend(2.0 * (eva.minCoeff() - eoa.maxCoeff()),
           2.0 * (eva.maxCoeff() - eoa.minCoeff()));
  if (have_b)
    extend(2.0 * (evb.minCoeff() - eob.maxCoeff()),
           2.0 * (evb.maxCoeff() - eob.minCoeff()));
  if (have_a && have_b)
    extend(
        (eva.minCoeff() + evb.minCoeff()) - (eoa.maxCoeff() + eob.maxCoeff()),
        (eva.maxCoeff() + evb.maxCoeff()) - (eoa.minCoeff() + eob.minCoeff()));
  const LaplaceGrid g = laplace_grid(dmin, dmax, opts.n_laplace);

  const bool ss = !opts.opposite_spin_only;
  double coul_a = 0.0, exch_a = 0.0, coul_b = 0.0, exch_b = 0.0, coul_ab = 0.0;
  occ::timing::start(occ::timing::category::mp2_energy);
  for (int k = 0; k < g.size(); ++k) {
    const double t = g.points(k), w = g.weights(k);
    Mat Ba_h, Bb_h;
    if (have_a) {
      const Mat Mo = scale_rows(Xoa, eoa, 0.5 * t);
      const Mat Mv = scale_rows(Xva, eva, -0.5 * t);
      Ba_h = (Mo.transpose() * Mo).cwiseProduct(Mv.transpose() * Mv);
      coul_a += w * coulomb_term(Ba_h, f.Vaa, Ba_h, f.Vaa);
      if (ss)
        exch_a += w * exchange_orbital(Mo, Mv, f.Vaa);
    }
    if (have_b) {
      const Mat Mo = scale_rows(Xob, eob, 0.5 * t);
      const Mat Mv = scale_rows(Xvb, evb, -0.5 * t);
      Bb_h = (Mo.transpose() * Mo).cwiseProduct(Mv.transpose() * Mv);
      coul_b += w * coulomb_term(Bb_h, f.Vbb, Bb_h, f.Vbb);
      if (ss)
        exch_b += w * exchange_orbital(Mo, Mv, f.Vbb);
    }
    if (have_a && have_b)
      coul_ab += w * coulomb_term(Ba_h, f.Vab, Bb_h, f.Vab.transpose());
  }
  occ::timing::stop(occ::timing::category::mp2_energy);

  // E_σσ = ½(exch_σ - coul_σ); E_αβ = -coul_ab. (Restricted limit: α==β gives
  // same_spin = exch - coul, opposite_spin = -coul.)
  r.same_spin = ss ? (0.5 * (exch_a - coul_a) + 0.5 * (exch_b - coul_b)) : 0.0;
  r.opposite_spin = -coul_ab;
  r.total = r.same_spin + r.opposite_spin;
  r.n_isdf = P;
  r.n_laplace = g.size();
  r.laplace_max_rel_error = laplace_max_rel_error(g, dmin, dmax);
  occ::log::debug("THC-MP2 (unrestricted): {} interp pts, {} Laplace pts "
                  "(max rel err {:.2e}); E_os={:.8f} E_ss={:.8f}",
                  P, g.size(), r.laplace_max_rel_error, r.opposite_spin,
                  r.same_spin);
  return r;
}

} // namespace

ThcMP2Result thc_mp2(const AOBasis &basis, const AOBasis &aux_basis,
                     const MolecularOrbitals &mo, const ThcMP2Options &opts) {
  if (mo.kind == occ::qm::SpinorbitalKind::Unrestricted)
    return thc_mp2_unrestricted(basis, aux_basis, mo, opts);
  if (mo.kind == occ::qm::SpinorbitalKind::Restricted)
    return thc_mp2_restricted(basis, aux_basis, mo, opts);
  throw std::runtime_error("THC-MP2: only restricted/unrestricted supported");
}

} // namespace occ::qm::cc
