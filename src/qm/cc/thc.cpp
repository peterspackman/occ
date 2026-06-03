#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <cmath>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/numint/molecular_grid.h>
#include <occ/qm/cc/thc.h>
#include <occ/qm/correlation/df_integrals.h>
#include <occ/qm/integral_engine_df.h>

namespace occ::qm::cc {

namespace {

using T4 = Eigen::Tensor<double, 4>;

// Pivoted Cholesky of the grid Gram K(g,g') = (W Wᵀ)∘(W Wᵀ), with columns
// evaluated lazily (a factor nfunc cheaper than QRCP on the pair-collocation).
// Selects grid-point indices. target<=0 -> stop at tol * diag_max.
std::vector<int> pivoted_cholesky_points(const Mat &W, int target, double tol) {
  const Eigen::Index ng = W.rows();
  Vec d(ng);
  for (Eigen::Index g = 0; g < ng; ++g) {
    const double n2 = W.row(g).squaredNorm();
    d(g) = n2 * n2; // (||W_g||²)² since X1 == X2 == W
  }
  const double dmax0 = d.maxCoeff();
  const int maxk = (target > 0) ? std::min<int>(target, static_cast<int>(ng))
                                : static_cast<int>(ng);
  const double thresh = (target > 0) ? -1.0 : tol * dmax0;

  Mat L(ng, std::max(1, maxk));
  std::vector<int> piv;
  piv.reserve(maxk);
  for (int k = 0; k < maxk; ++k) {
    Eigen::Index p;
    const double dp = d.maxCoeff(&p);
    if (dp <= thresh || dp <= 0.0)
      break;
    // K[:, p] = (W W_pᵀ)∘(W W_pᵀ). Copy the pivot rows to contiguous vectors so
    // the GEMVs read column-major W / L with unit stride (W.row(p) is strided).
    const Vec wp = W.row(p).transpose();
    Vec Lk = (W * wp).array().square().matrix();
    if (k > 0) {
      const Vec lp = L.row(p).head(k).transpose();
      Lk.noalias() -= L.leftCols(k) * lp;
    }
    Lk /= std::sqrt(dp);
    L.col(k) = Lk;
    d = (d - Lk.array().square().matrix()).cwiseMax(0.0);
    d(p) = 0.0;
    piv.push_back(static_cast<int>(p));
  }
  return piv;
}

// Finish a semidirect transform: given the half-transformed H(p, ν, ρ, σ)
// (left index already in MO for a block), contract ν,ρ,σ with the MO blocks
// Cq, Cr, Cs to give (p q | r s). Contracting index 1 (with the new MO index
// appended last) three times lands directly on [p, q, r, s].
T4 finish_transform(const T4 &H, const Mat &Cq, const Mat &Cr, const Mat &Cs) {
  using T2 = Eigen::Tensor<double, 2>;
  using T3 = Eigen::Tensor<double, 3>;
  const Eigen::Index nL = H.dimension(0);
  const Eigen::Index nao = Cq.rows();
  const Eigen::Index nq = Cq.cols(), nr = Cr.cols(), ns = Cs.cols();
  const Eigen::TensorMap<const T2> Tq(Cq.data(), nao, nq);
  const Eigen::TensorMap<const T2> Tr(Cr.data(), nao, nr);
  const Eigen::TensorMap<const T2> Ts(Cs.data(), nao, ns);
  const Eigen::array<Eigen::IndexPair<int>, 1> c00 = {
      Eigen::IndexPair<int>(0, 0)};
  // Eigen tensor contractions are single-threaded; parallelize over the left
  // index p (each slice is independent). Store with p as the *outer* index so
  // each thread writes a contiguous block (no false sharing), then shuffle.
  T4 tmp(nq, nr, ns, nL);
  occ::parallel::parallel_for(
      size_t(0), static_cast<size_t>(nL), [&](size_t p) {
        const Eigen::Index pi = static_cast<Eigen::Index>(p);
        const T3 Hp = H.chip(pi, 0);           // [ν, ρ, σ]
        const T3 a = Hp.contract(Tq, c00);     // [ρ, σ, q]
        const T3 b = a.contract(Tr, c00);      // [σ, q, r]
        tmp.chip(pi, 3) = b.contract(Ts, c00); // [q, r, s]
      });
  return tmp.shuffle(Eigen::array<int, 4>{3, 0, 1, 2}); // (p, q, r, s)
}

} // namespace

std::vector<int> select_isdf_points(const Mat &coll, IsdfMethod method,
                                    int target, double tol) {
  if (method == IsdfMethod::Cholesky)
    return pivoted_cholesky_points(coll, target, tol);

  // Pivoted QR on the (symmetric) pair-density collocation, lower triangle with
  // off-diagonal pairs scaled by √2 (so ‖Z column‖ matches the symmetric pair).
  const Eigen::Index npts = coll.rows();
  const Eigen::Index nf = coll.cols();
  const Eigen::Index npairs = nf * (nf + 1) / 2;
  Mat Z(npts, npairs);
  const double s2 = std::sqrt(2.0);
  Eigen::Index col = 0;
  for (Eigen::Index p = 0; p < nf; ++p)
    for (Eigen::Index q = 0; q <= p; ++q, ++col) {
      Z.col(col) = coll.col(p).cwiseProduct(coll.col(q));
      if (p != q)
        Z.col(col) *= s2;
    }

  const Mat Zt = Z.transpose(); // (npairs x npts): columns are grid points
  Eigen::ColPivHouseholderQR<Mat> qr(Zt);
  const auto &perm = qr.colsPermutation().indices();

  // Numerical rank of the pair-collocation: pivots below rank_tol*|R00| live in
  // the null space. Selecting them as interpolation points only adds redundant
  // (often clustered) points that collapse the LS-THC metric rank, so cap the
  // selection at the rank regardless of the requested count.
  const Eigen::Index k = std::min(npairs, npts);
  const Mat &QR = qr.matrixQR();
  const double d0 = std::abs(QR(0, 0));
  constexpr double rank_tol = 1e-10;
  int rank = 0;
  while (rank < static_cast<int>(k) && std::abs(QR(rank, rank)) > rank_tol * d0)
    ++rank;
  rank = std::max(rank, 1);

  int t;
  if (target > 0) {
    t = std::min(target, rank);
  } else {
    int cnt = 0;
    while (cnt < rank && std::abs(QR(cnt, cnt)) > tol * d0)
      ++cnt;
    t = std::max(cnt, 1);
  }
  t = std::min<int>(t, static_cast<int>(npts));
  std::vector<int> sel(t);
  for (int i = 0; i < t; ++i)
    sel[i] = static_cast<int>(perm[i]);
  return sel;
}

namespace {

// Regularised inverse of the (ill-conditioned) symmetric LS-THC metric S:
// Eig drops eigenvalues below reg*lambda_max (truncated pinv); Tikhonov shifts
// by reg*lambda_max. Reports the condition number and #eigenvalues kept.
Mat reg_inverse(const Mat &S, double reg, ThcRegType reg_type,
                double *condition_out, int *n_kept_out) {
  const Eigen::Index npt = S.rows();
  Eigen::SelfAdjointEigenSolver<Mat> es(S);
  const Vec w = es.eigenvalues();
  const Mat &U = es.eigenvectors();
  const double lmax = w.maxCoeff();
  const double lmin = w.minCoeff();
  if (condition_out)
    *condition_out = lmax / std::max(lmin, 1e-300);

  Vec dinv(npt);
  int kept = 0;
  if (reg_type == ThcRegType::Tikhonov) {
    const double shift = reg * lmax;
    for (Eigen::Index i = 0; i < npt; ++i)
      dinv(i) = 1.0 / (w(i) + shift);
    kept = static_cast<int>(npt);
  } else {
    const double cut = reg * lmax;
    for (Eigen::Index i = 0; i < npt; ++i) {
      if (w(i) > cut) {
        dinv(i) = 1.0 / w(i);
        ++kept;
      } else {
        dinv(i) = 0.0;
      }
    }
  }
  if (n_kept_out)
    *n_kept_out = kept;
  return U * dinv.asDiagonal() * U.transpose();
}

// V = Sinv R Sinv, symmetrised (the LS-THC normal-equation solution).
Mat solve_core(const Mat &S, const Mat &R, double reg, ThcRegType reg_type,
               double *condition_out, int *n_kept_out) {
  const Mat Sinv = reg_inverse(S, reg, reg_type, condition_out, n_kept_out);
  Mat V = Sinv * R * Sinv;
  return 0.5 * (V + V.transpose()); // symmetric in exact arithmetic; enforce
}

} // namespace

Mat fit_core(const Mat &X, const Mat &B, double reg, ThcRegType reg_type,
             double *condition_out, int *n_kept_out) {
  const Eigen::Index nmo = X.rows();
  const Eigen::Index npt = X.cols();

  // E(P, p*nmo+q) = X(p,P) X(q,P) -- the pair collocation, row-major over (p,q)
  // to match the B tensor's row layout (p*nmo+q).
  Mat E(npt, nmo * nmo);
  for (Eigen::Index P = 0; P < npt; ++P)
    for (Eigen::Index p = 0; p < nmo; ++p) {
      const double xp = X(p, P);
      for (Eigen::Index q = 0; q < nmo; ++q)
        E(P, p * nmo + q) = xp * X(q, P);
    }

  const Mat EBt = E * B;               // (npt x naux): E Bᵀ
  const Mat G = X.transpose() * X;     // (npt x npt): orbital Gram
  const Mat S = G.cwiseProduct(G);     // LS-THC metric (Hadamard square)
  const Mat R = EBt * EBt.transpose(); // (npt x npt): E M Eᵀ via DF
  return solve_core(S, R, reg, reg_type, condition_out, n_kept_out);
}

Mat fit_core_ov(const Mat &Xo, const Mat &Xv, const Mat &B_ov, double reg,
                ThcRegType reg_type, double *condition_out, int *n_kept_out) {
  const Eigen::Index o = Xo.rows();
  const Eigen::Index v = Xv.rows();
  const Eigen::Index npt = Xo.cols();

  // E(P, i*v+a) = Xo(i,P) Xv(a,P) -- the occ-virt pair collocation, matching
  // B_ov's row layout (i*v+a). Only o*v columns (vs nmo^2 for fit_core).
  Mat E(npt, o * v);
  for (Eigen::Index P = 0; P < npt; ++P)
    for (Eigen::Index i = 0; i < o; ++i) {
      const double xi = Xo(i, P);
      for (Eigen::Index a = 0; a < v; ++a)
        E(P, i * v + a) = xi * Xv(a, P);
    }

  const Mat EBt = E * B_ov;            // (npt x naux)
  const Mat Go = Xo.transpose() * Xo;  // (npt x npt): occ Gram
  const Mat Gv = Xv.transpose() * Xv;  // (npt x npt): virt Gram
  const Mat S = Go.cwiseProduct(Gv);   // ov LS-THC metric
  const Mat R = EBt * EBt.transpose(); // (npt x npt)
  return solve_core(S, R, reg, reg_type, condition_out, n_kept_out);
}

Eigen::Tensor<double, 4> reconstruct_eri(const Mat &X, const Mat &V) {
  const Eigen::Index nmo = X.rows();
  const Eigen::Index npt = X.cols();
  Mat E(npt, nmo * nmo);
  for (Eigen::Index P = 0; P < npt; ++P)
    for (Eigen::Index p = 0; p < nmo; ++p) {
      const double xp = X(p, P);
      for (Eigen::Index q = 0; q < nmo; ++q)
        E(P, p * nmo + q) = xp * X(q, P);
    }
  const Mat W = V * E;             // W(P, rs) = Σ_Q V(P,Q) E(Q, rs)
  const Mat A = E.transpose() * W; // A(pq, rs) = (pq|rs)
  T4 out(nmo, nmo, nmo, nmo);
  for (Eigen::Index p = 0; p < nmo; ++p)
    for (Eigen::Index q = 0; q < nmo; ++q)
      for (Eigen::Index r = 0; r < nmo; ++r)
        for (Eigen::Index s = 0; s < nmo; ++s)
          out(p, q, r, s) = A(p * nmo + q, r * nmo + s);
  return out;
}

Eigen::Tensor<double, 4> mo_eri_general(const IntegralEngine &engine,
                                        const Mat &C_L, const Mat &C_q,
                                        const Mat &C_r, const Mat &C_s,
                                        size_t budget) {
  const Eigen::Index nao = C_L.rows();
  const Eigen::Index nL = C_L.cols();
  const Eigen::Index nq = C_q.cols(), nr = C_r.cols(), ns = C_s.cols();
  T4 out(nL, nq, nr, ns);

  const size_t per = static_cast<size_t>(nao) * nao * nao * sizeof(double);
  Eigen::Index blk = std::max<Eigen::Index>(
      1, static_cast<Eigen::Index>(budget / std::max<size_t>(1, 4 * per)));
  blk = std::min(blk, nL);

  for (Eigen::Index s0 = 0; s0 < nL; s0 += blk) {
    const Eigen::Index b = std::min(blk, nL - s0);
    // H(p, ν, ρ, σ) = Σ_μ C_L(μ,p) (μν|ρσ) -- semidirect, no nao^4 tensor
    const T4 H = engine.ao_direct_half_transform(C_L.middleCols(s0, b));
    const T4 g = finish_transform(H, C_q, C_r, C_s); // (b, nq, nr, ns)
    out.slice(Eigen::array<Eigen::Index, 4>{s0, 0, 0, 0},
              Eigen::array<Eigen::Index, 4>{b, nq, nr, ns}) = g;
  }
  return out;
}

double reconstruction_error(const AOBasis &basis, const MolecularOrbitals &mo,
                            const Mat &X, const Mat &V) {
  IntegralEngine engine(basis);
  const Mat &C = mo.C;
  const T4 exact = mo_eri_general(engine, C, C, C, C);
  const T4 approx = reconstruct_eri(X, V);
  const T4 diff = approx - exact;
  const Eigen::Tensor<double, 0> dn = diff.square().sum();
  const Eigen::Tensor<double, 0> en = exact.square().sum();
  return std::sqrt(dn(0)) / std::sqrt(en(0));
}

Mat thc_select_collocation(const AOBasis &basis, const MolecularOrbitals &mo,
                           const ThcOptions &opts) {
  namespace tc = occ::timing;
  const Mat &C = mo.C;
  const Eigen::Index nmo = C.cols();

  // --- candidate grid + AO/MO collocation -------------------------------
  tc::start(tc::category::thc_grid);
  occ::io::GridSettings gs;
  gs.max_angular_points = static_cast<size_t>(opts.grid_max_angular);
  gs.min_angular_points =
      std::min<size_t>(50, static_cast<size_t>(opts.grid_max_angular));
  gs.radial_precision = opts.grid_radial_precision;
  occ::dft::MolecularGrid grid(basis, gs);
  grid.populate_molecular_grid_points();
  const auto &gp = grid.get_molecular_grid_points();
  const Mat3N pts = gp.points(); // 3 x npts
  const Vec wts = gp.weights();  // npts

  const occ::gto::GTOValues vals = occ::gto::evaluate_basis(basis, pts, 0);
  const Mat &phi = vals.phi;   // npts x nbf
  const Mat mo_grid = phi * C; // npts x nmo

  const Vec sqw = wts.array().abs().sqrt();
  const Mat sel_w = (opts.select_basis == ThcSelectBasis::AO)
                        ? Mat(sqw.asDiagonal() * phi)
                        : Mat(sqw.asDiagonal() * mo_grid);
  const Eigen::Index nselfun = sel_w.cols();
  tc::stop(tc::category::thc_grid);

  int target = -1;
  if (opts.n_isdf > 0)
    target = opts.n_isdf;
  else if (opts.c_isdf > 0)
    target = static_cast<int>(
        std::llround(opts.c_isdf * static_cast<double>(nselfun)));

  // Cap the rank at the MO pair-space size nmo(nmo+1)/2: selecting more points
  // than that cannot add independent THC terms, only makes the LS-THC metric
  // singular -- which amplifies FP noise from the parallel integral build into
  // an unstable (run-to-run non-deterministic) fit. Binds only for small
  // systems (large systems have c*nbf << nmo^2/2).
  const int pair_rank = static_cast<int>(nmo * (nmo + 1) / 2);
  if (target > pair_rank) {
    occ::log::debug("THC: capping rank target {} at MO pair-space rank {}",
                    target, pair_rank);
    target = pair_rank;
  }

  tc::start(tc::category::thc_select);
  const std::vector<int> sel =
      select_isdf_points(sel_w, opts.method, target, opts.tol);
  const int nsel = static_cast<int>(sel.size());
  tc::stop(tc::category::thc_select);

  Mat X(nmo, nsel);
  for (int k = 0; k < nsel; ++k)
    X.col(k) = mo_grid.row(sel[k]).transpose();
  occ::log::debug("THC: {} interpolation points from {} grid candidates "
                  "({} funcs)",
                  nsel, pts.cols(), nselfun);
  return X;
}

ThcFactors build_thc_from_B(const AOBasis &basis, const MolecularOrbitals &mo,
                            const ThcOptions &opts, const Mat &B) {
  namespace tc = occ::timing;
  tc::start(tc::category::thc_factorize);
  ThcFactors f;
  f.X = thc_select_collocation(basis, mo, opts);
  f.n_isdf = static_cast<int>(f.X.cols());

  // --- robust LS-THC core fit -------------------------------------------
  tc::start(tc::category::thc_fit);
  f.V = fit_core(f.X, B, opts.reg, opts.reg_type, &f.metric_condition,
                 &f.metric_n_kept);
  tc::stop(tc::category::thc_fit);
  occ::log::debug("THC: metric cond={:.3e}, kept {}/{}", f.metric_condition,
                  f.metric_n_kept, f.n_isdf);
  tc::stop(tc::category::thc_factorize);
  return f;
}

ThcFactors build_thc(const AOBasis &basis, const AOBasis &aux_basis,
                     const MolecularOrbitals &mo, const ThcOptions &opts) {
  AOBasis aux = aux_basis;
  aux.set_kind(basis.kind());
  occ::qm::IntegralEngineDF df_engine(basis.atoms(), basis.shells(),
                                      aux.shells());
  occ::qm::DFIntegrals df(df_engine, opts.memory_budget);
  const Mat B = df.build_b_tilde(mo.C, mo.C); // (nmo^2 x naux), row p*nmo+q
  return build_thc_from_B(basis, mo, opts, B);
}

namespace {

// Regularised pseudo-inverse of a symmetric metric S (drop eigenvalues below
// reg*lambda_max), matching fit_core's Eig regularisation.
Mat reg_pinv(const Mat &S, double reg) {
  Eigen::SelfAdjointEigenSolver<Mat> es(S);
  const Vec w = es.eigenvalues();
  const Mat &U = es.eigenvectors();
  const double cut = reg * w.maxCoeff();
  Vec d(w.size());
  for (Eigen::Index i = 0; i < w.size(); ++i)
    d(i) = (w(i) > cut) ? 1.0 / w(i) : 0.0;
  return U * d.asDiagonal() * U.transpose();
}

// (E B) where E(P, p*nmo+q) = X(p,P) X(q,P) -- pair collocation times DF
// tensor.
Mat pair_coll_B(const Mat &X, const Mat &B) {
  const Eigen::Index nmo = X.rows(), npt = X.cols();
  Mat E(npt, nmo * nmo);
  for (Eigen::Index P = 0; P < npt; ++P)
    for (Eigen::Index p = 0; p < nmo; ++p) {
      const double xp = X(p, P);
      for (Eigen::Index q = 0; q < nmo; ++q)
        E(P, p * nmo + q) = xp * X(q, P);
    }
  return E * B;
}

} // namespace

UThcFactors build_uthc(const AOBasis &basis, const Mat &Ca, const Mat &Cb,
                       const Mat &Ba, const Mat &Bb, const ThcOptions &opts) {
  namespace tc = occ::timing;
  tc::start(tc::category::thc_factorize);

  // candidate grid + AO collocation (orbital-independent point selection)
  tc::start(tc::category::thc_grid);
  occ::io::GridSettings gs;
  gs.max_angular_points = static_cast<size_t>(opts.grid_max_angular);
  gs.min_angular_points =
      std::min<size_t>(50, static_cast<size_t>(opts.grid_max_angular));
  gs.radial_precision = opts.grid_radial_precision;
  occ::dft::MolecularGrid grid(basis, gs);
  grid.populate_molecular_grid_points();
  const auto &gp = grid.get_molecular_grid_points();
  const Mat3N pts = gp.points();
  const Vec wts = gp.weights();
  const occ::gto::GTOValues vals = occ::gto::evaluate_basis(basis, pts, 0);
  const Mat &phi = vals.phi; // npts x nbf
  const Vec sqw = wts.array().abs().sqrt();
  const Mat sel_w = sqw.asDiagonal() * phi; // AO selection collocation
  tc::stop(tc::category::thc_grid);

  const Eigen::Index nmoa = Ca.cols(), nmob = Cb.cols();
  int target = -1;
  if (opts.n_isdf > 0)
    target = opts.n_isdf;
  else if (opts.c_isdf > 0)
    target = static_cast<int>(
        std::llround(opts.c_isdf * static_cast<double>(phi.cols())));
  // cap at the smaller spin's MO pair-space rank to keep both metrics solvable
  const int pair_rank =
      static_cast<int>(std::min(nmoa * (nmoa + 1) / 2, nmob * (nmob + 1) / 2));
  if (target > pair_rank)
    target = pair_rank;

  tc::start(tc::category::thc_select);
  const std::vector<int> sel =
      select_isdf_points(sel_w, opts.method, target, opts.tol);
  const int nsel = static_cast<int>(sel.size());
  tc::stop(tc::category::thc_select);

  const Mat moa = phi * Ca; // npts x nmoa
  const Mat mob = phi * Cb;
  UThcFactors f;
  f.n_isdf = nsel;
  f.Xa = Mat(nmoa, nsel);
  f.Xb = Mat(nmob, nsel);
  for (int k = 0; k < nsel; ++k) {
    f.Xa.col(k) = moa.row(sel[k]).transpose();
    f.Xb.col(k) = mob.row(sel[k]).transpose();
  }

  tc::start(tc::category::thc_fit);
  f.Vaa = fit_core(f.Xa, Ba, opts.reg, opts.reg_type);
  f.Vbb = fit_core(f.Xb, Bb, opts.reg, opts.reg_type);
  // cross core: Vab = Sa^-1 (EBa EBb^T) Sb^-1
  const Mat EBa = pair_coll_B(f.Xa, Ba);
  const Mat EBb = pair_coll_B(f.Xb, Bb);
  const Mat Rab = EBa * EBb.transpose();
  const Mat Ga = f.Xa.transpose() * f.Xa, Gb = f.Xb.transpose() * f.Xb;
  const Mat Sa = Ga.cwiseProduct(Ga), Sb = Gb.cwiseProduct(Gb);
  f.Vab = reg_pinv(Sa, opts.reg) * Rab * reg_pinv(Sb, opts.reg);
  tc::stop(tc::category::thc_fit);

  occ::log::debug("U-THC: {} interpolation points ({} grid candidates)", nsel,
                  pts.cols());
  tc::stop(tc::category::thc_factorize);
  return f;
}

} // namespace occ::qm::cc
