#include <algorithm>
#include <memory>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/cc/gemm.h>
#include <occ/qm/cc/integrals.h>
#include <occ/qm/cc/thc.h>
#include <occ/qm/correlation/df_integrals.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>

namespace occ::qm::cc {

using T4 = Eigen::Tensor<double, 4>;

namespace {

// Slice G(i, q, r, s) [i over occ] -> block over the (q,r,s) sub-ranges.
T4 slice_block(const T4 &G, int nocc, int q0, int qn, int r0, int rn, int s0,
               int sn) {
  return G.slice(Eigen::array<Eigen::Index, 4>{0, q0, r0, s0},
                 Eigen::array<Eigen::Index, 4>{nocc, qn, rn, sn});
}

// Unpack a (np*nq x nr*ns) matrix product of two B blocks into a chemist tensor
// result(p,q,r,s) = M(p*nq+q, r*ns+s) -- matches DFIntegrals' row layout left*nR+r.
T4 unpack(const Mat &M, int np, int nq, int nr, int ns) {
  T4 result(np, nq, nr, ns);
  for (int p = 0; p < np; ++p)
    for (int q = 0; q < nq; ++q)
      for (int r = 0; r < nr; ++r)
        for (int s = 0; s < ns; ++s)
          result(p, q, r, s) = M(p * nq + q, r * ns + s);
  return result;
}

// Drop the lowest n_frozen occupied orbitals from the correlation space by
// returning MOs restricted to the active range [n_frozen, nmo). The frozen
// orbitals still contributed to the SCF; they just aren't correlated.
MolecularOrbitals freeze_core(const MolecularOrbitals &mo, int n_frozen) {
  if (n_frozen <= 0)
    return mo;
  const Eigen::Index nmo = mo.C.cols();
  MolecularOrbitals a = mo;
  a.C = mo.C.middleCols(n_frozen, nmo - n_frozen).eval();
  a.energies = mo.energies.segment(n_frozen, mo.energies.size() - n_frozen).eval();
  a.n_alpha = mo.n_alpha - static_cast<size_t>(n_frozen);
  a.n_beta = mo.n_beta > static_cast<size_t>(n_frozen)
                 ? mo.n_beta - static_cast<size_t>(n_frozen)
                 : 0;
  return a;
}

} // namespace

namespace {

// In-core transform from the PACKED AO matrix A(P,Q)=(μν|λσ) [P=pair(μ≤ν)].
// Two half-transforms, each parallel over the disjoint output pair index: unpack
// one packed column to a symmetric nao×nao square and GEMM C^T M C. Same O(N^5)
// and GEMM-bound, but the AO store is ~8x smaller than the dense tensor.
T4 mo_transform_packed(const Mat &A, const Mat &Cp, const Mat &Cq,
                       const Mat &Cr, const Mat &Cs) {
  const int n = static_cast<int>(Cp.rows());
  const int np = Cp.cols(), nq = Cq.cols(), nr = Cr.cols(), ns = Cs.cols();
  const Eigen::Index npair = static_cast<Eigen::Index>(n) * (n + 1) / 2;
  auto unpack = [n](const double *col, Mat &M) {
    for (int b = 0; b < n; ++b) {
      const int base = b * (b + 1) / 2;
      for (int a = 0; a <= b; ++a) {
        const double v = col[base + a];
        M(a, b) = v;
        M(b, a) = v;
      }
    }
  };

  // Phase 1 (bra): T1(pq, Q) = Cp^T unpack(A[:,Q]) Cq, parallel over ket pairs Q.
  Mat T1(static_cast<Eigen::Index>(np) * nq, npair);
  occ::parallel::parallel_for(size_t(0), static_cast<size_t>(npair),
                              [&](size_t Q) {
    Mat M(n, n);
    unpack(A.col(static_cast<Eigen::Index>(Q)).data(), M);
    Eigen::Map<Mat>(T1.col(static_cast<Eigen::Index>(Q)).data(), np, nq)
        .noalias() = Cp.transpose() * M * Cq;
  });

  Mat T1t = T1.transpose(); // (npair x np*nq), contiguous columns for phase 2
  T1.resize(0, 0);

  // Phase 2 (ket): T2(rs, pq) = Cr^T unpack(T1t[:,pq]) Cs, parallel over pq.
  // (named T2, not OUT: "OUT" is an empty SAL macro in the Windows headers.)
  const Eigen::Index npq = static_cast<Eigen::Index>(np) * nq;
  Mat T2(static_cast<Eigen::Index>(nr) * ns, npq);
  occ::parallel::parallel_for(size_t(0), static_cast<size_t>(npq),
                              [&](size_t pq) {
    Mat M(n, n);
    unpack(T1t.col(static_cast<Eigen::Index>(pq)).data(), M);
    Eigen::Map<Mat>(T2.col(static_cast<Eigen::Index>(pq)).data(), nr, ns)
        .noalias() = Cr.transpose() * M * Cs;
  });
  T1t.resize(0, 0);

  T4 result(np, nq, nr, ns); // result(p,q,r,s) = T2(r+nr*s, p+np*q)
  for (int s = 0; s < ns; ++s)
    for (int r = 0; r < nr; ++r)
      for (int q = 0; q < nq; ++q)
        for (int p = 0; p < np; ++p)
          result(p, q, r, s) = T2(r + nr * s, p + np * q);
  return result;
}

} // namespace

int num_frozen_core(const AOBasis &basis) {
  int n = 0;
  for (const auto &atom : basis.atoms()) {
    const int z = atom.atomic_number;
    if (z >= 21)
      n += 9; // [Ar] 3d core for Sc and beyond
    else if (z >= 11)
      n += 5; // [Ne] core for Na-Ar
    else if (z >= 3)
      n += 1; // 1s core for Li-Ne
  }
  return n;
}

CCIntegrals exact_eris(const AOBasis &basis, const MolecularOrbitals &mo_in,
                       int n_frozen, size_t memory_budget) {
  const MolecularOrbitals mo = freeze_core(mo_in, n_frozen);
  const Mat &C = mo.C;
  const int nmo = static_cast<int>(C.cols());
  const int nocc = static_cast<int>(mo.n_alpha);
  const int nvir = nmo - nocc;

  CCIntegrals e;
  e.nocc = nocc;
  e.nvir = nvir;
  e.mo_energy = mo.energies.head(nmo);
  e.fock = mo.energies.head(nmo).asDiagonal();

  IntegralEngine engine(basis);
  const Mat C_occ = C.leftCols(nocc);
  const Mat C_virt = C.middleCols(nocc, nvir);
  const int o = nocc, v = nvir;

  // Choose the AO->MO path. In-core (build the packed AO matrix once, then GEMM
  // half-transforms) is much faster; the packed store is ~npair^2 (8x smaller
  // than the dense nao^4). Fall back to the semidirect occ-blocked transform if
  // even the packed AO matrix won't comfortably fit the memory budget.
  const size_t nao = static_cast<size_t>(C.rows());
  const size_t npair = nao * (nao + 1) / 2;
  const size_t packed_bytes = npair * npair * sizeof(double);
  const bool in_core = (2 * packed_bytes <= memory_budget);
  const double vvvv_mb =
      static_cast<double>(nvir) * nvir * nvir * nvir * 8.0 / 1024.0 / 1024.0;
  occ::log::info("Exact backend: AO->MO transform ({}); stores vvvv = {:.1f} MB",
                 in_core ? "in-core packed GEMM" : "semidirect, occ-blocked",
                 vvvv_mb);
  occ::timing::start(occ::timing::category::cc_ao2mo);

  // G(i, q, r, s) = (iq|rs) for i in occ; every <=2-virtual block and ovvv
  // has an occupied index in chemist slot 0, so all are slices of G.
  T4 G, vvvv_full;
  if (in_core) {
    const Mat A = engine.four_center_integrals_packed(); // packed, one AO pass
    G = mo_transform_packed(A, C_occ, C, C, C);
    vvvv_full = mo_transform_packed(A, C_virt, C_virt, C_virt, C_virt); // (ab|cd)
  } else {
    G = mo_eri_general(engine, C_occ, C, C, C, memory_budget);
    vvvv_full = mo_eri_general(engine, C_virt, C_virt, C_virt, C_virt,
                               memory_budget);
  }
  e.oooo = slice_block(G, o, 0, o, 0, o, 0, o);
  e.ooov = slice_block(G, o, 0, o, 0, o, o, v);
  e.oovv = slice_block(G, o, 0, o, o, v, o, v);
  e.ovoo = slice_block(G, o, o, v, 0, o, 0, o);
  e.ovov = slice_block(G, o, o, v, 0, o, o, v);
  e.ovvo = slice_block(G, o, o, v, o, v, 0, o);
  e.ovvv = slice_block(G, o, o, v, o, v, o, v);

  // Exact vvvv ladder. Store a single O(V^4) matrix (the dense vvvv is inherent
  // to the *exact* backend -- use df/thc to avoid it), laid out once as
  // W(c + v*d, a + v*b) = (ac|bd) so that the ladder
  //   L(ij, ab) = sum_cd tau(ij, cd) W(cd, ab)
  // is one parallel GEMM straight on the column-major tau and result tensors:
  // no permute per call, and no serial Eigen tensor contraction.
  // The relayout is done in place: (ac|bd) = (ca|db) = vvvv_full(c,a,d,b) sits
  // at c + v*a + v^2*d + v^3*b, so W is vvvv_full with a and d swapped for each
  // (c, b), and no second V^4 buffer is ever allocated.
  auto vvvv = std::make_shared<T4>(std::move(vvvv_full));
  {
    double *x = vvvv->data();
    const size_t sv = static_cast<size_t>(v);
    occ::parallel::parallel_for(size_t(0), sv, [&](size_t b) {
      for (size_t a = 0; a < sv; ++a)
        for (size_t d = a + 1; d < sv; ++d)
          for (size_t c = 0; c < sv; ++c)
            std::swap(x[c + sv * (a + sv * (d + sv * b))],
                      x[c + sv * (d + sv * (a + sv * b))]);
    });
  }
  e.ladder = [vvvv, o, v](const T4 &tau) -> T4 {
    const Eigen::Index oo = static_cast<Eigen::Index>(o) * o;
    const Eigen::Index vv = static_cast<Eigen::Index>(v) * v;
    T4 out(o, o, v, v);
    Eigen::Map<Mat> L(out.data(), oo, vv);
    pgemm(L, Eigen::Map<const Mat>(tau.data(), oo, vv),
          Eigen::Map<const Mat>(vvvv->data(), vv, vv));
    return out;
  };
  occ::timing::stop(occ::timing::category::cc_ao2mo);
  return e;
}

// Build all CCIntegrals blocks + the DF ladder from a prepared DFIntegrals.
// Shared by df_eris and thc_eris so the 3-center store is built only once.
static CCIntegrals df_blocks(DFIntegrals &df, const MolecularOrbitals &mo) {
  const Mat &C = mo.C;
  const int nmo = static_cast<int>(C.cols());
  const int nocc = static_cast<int>(mo.n_alpha);
  const int nvir = nmo - nocc;
  const int o = nocc, v = nvir;

  CCIntegrals e;
  e.nocc = nocc;
  e.nvir = nvir;
  e.mo_energy = mo.energies.head(nmo);
  e.fock = mo.energies.head(nmo).asDiagonal();

  const Mat C_occ = C.leftCols(nocc);
  const Mat C_virt = C.middleCols(nocc, nvir);
  const Mat Boo = df.build_b_tilde(C_occ, C_occ);   // (o*o   x naux)
  const Mat Bov = df.build_b_tilde(C_occ, C_virt);  // (o*v   x naux)
  const Mat Bvv = df.build_b_tilde(C_virt, C_virt); // (v*v   x naux)
  const Mat Bvo = df.build_b_tilde(C_virt, C_occ);  // (v*o   x naux)

  e.oooo = unpack(Boo * Boo.transpose(), o, o, o, o);
  e.ooov = unpack(Boo * Bov.transpose(), o, o, o, v);
  e.oovv = unpack(Boo * Bvv.transpose(), o, o, v, v);
  e.ovoo = unpack(Bov * Boo.transpose(), o, v, o, o);
  e.ovov = unpack(Bov * Bov.transpose(), o, v, o, v);
  e.ovvo = unpack(Bov * Bvo.transpose(), o, v, v, o);
  e.ovvv = unpack(Bov * Bvv.transpose(), o, v, v, v);

  // DF ladder, (ac|bd) built in blocks of a and never stored whole. Bvv row
  // a*v + c holds (ac|P). For a block of na values of a:
  //   X(al*v + c, b*v + d) = sum_P (ac|P)(bd|P)            one GEMM
  //   W(c + v*d, al + na*b) = X(al*v + c, b*v + d)         re-laid out
  //   L(ij, al + na*b)    = sum_cd tau(ij, cd) W(cd, .)    one GEMM
  // n_aux v^4 + o^2 v^4 flops per call, as parallel GEMMs, instead of
  // 4 o^2 v^3 n_aux in serial per-pair tensor contractions; X and W take
  // 2 na v^3 doubles, sized from the budget below (na >= 1).
  auto BT = std::make_shared<Mat>(Bvv.transpose()); // (naux x v*v)
  e.ladder = [BT, o, v](const T4 &tau) -> T4 {
    constexpr size_t block_bytes = size_t(1) << 28; // X and W together
    const Eigen::Index oo = static_cast<Eigen::Index>(o) * o;
    const Eigen::Index vv = static_cast<Eigen::Index>(v) * v;
    const Eigen::Index na = std::clamp<Eigen::Index>(
        static_cast<Eigen::Index>(block_bytes / (2 * sizeof(double) * vv * v)), 1, v);
    const Eigen::Map<const Mat> Tau(tau.data(), oo, vv);
    T4 out(o, o, v, v);
    Eigen::Map<Mat> L(out.data(), oo, vv); // column a + v*b
    Mat X, W, Lb;
    for (Eigen::Index a0 = 0; a0 < v; a0 += na) {
      const Eigen::Index nb = std::min(na, static_cast<Eigen::Index>(v) - a0);
      X.resize(nb * v, vv);
      pgemm(X, BT->middleCols(a0 * v, nb * v), *BT, false, true);
      W.resize(vv, nb * v);
      occ::parallel::parallel_for(size_t(0), static_cast<size_t>(v), [&](size_t bu) {
        const Eigen::Index b = static_cast<Eigen::Index>(bu);
        for (Eigen::Index al = 0; al < nb; ++al)
          for (Eigen::Index d = 0; d < v; ++d)
            for (Eigen::Index c = 0; c < v; ++c)
              W(c + v * d, al + nb * b) = X(al * v + c, b * v + d);
      });
      Lb.resize(oo, nb * v);
      pgemm(Lb, Tau, W);
      occ::parallel::parallel_for(size_t(0), static_cast<size_t>(v), [&](size_t bu) {
        const Eigen::Index b = static_cast<Eigen::Index>(bu);
        for (Eigen::Index al = 0; al < nb; ++al)
          L.col(a0 + al + v * b) = Lb.col(al + nb * b);
      });
    }
    return out;
  };
  return e;
}

// Construct a DFIntegrals for the given AO + auxiliary basis (kind-matched).
static DFIntegrals make_df(const AOBasis &basis, const AOBasis &aux_basis,
                           size_t memory_budget,
                           std::unique_ptr<IntegralEngineDF> &engine_out) {
  AOBasis aux = aux_basis;
  aux.set_kind(basis.kind());
  engine_out = std::make_unique<IntegralEngineDF>(basis.atoms(), basis.shells(),
                                                  aux.shells());
  return DFIntegrals(*engine_out, memory_budget);
}

CCIntegrals df_eris(const AOBasis &basis, const AOBasis &aux_basis,
                    const MolecularOrbitals &mo_in, int n_frozen,
                    size_t memory_budget) {
  const MolecularOrbitals mo = freeze_core(mo_in, n_frozen);
  std::unique_ptr<IntegralEngineDF> engine;
  DFIntegrals df = make_df(basis, aux_basis, memory_budget, engine);
  return df_blocks(df, mo);
}

CCIntegrals thc_eris(const AOBasis &basis, const AOBasis &aux_basis,
                     const MolecularOrbitals &mo_in, const ThcOptions &opts,
                     int n_frozen, size_t memory_budget) {
  const MolecularOrbitals mo = freeze_core(mo_in, n_frozen);
  // The cheap (<=2-virtual) blocks come from DF; ovvv and the ladder are then
  // overridden with the THC factors. (The reference uses exact cheap blocks;
  // DF here keeps everything O(N^4)-free and scalable.) A single DFIntegrals is
  // shared by the cheap-block build and the LS-THC reference, so the 3-center
  // integrals are computed and stored only once.
  std::unique_ptr<IntegralEngineDF> engine;
  DFIntegrals df = make_df(basis, aux_basis, memory_budget, engine);
  CCIntegrals e = df_blocks(df, mo);
  const int o = e.nocc, v = e.nvir;

  occ::timing::start(occ::timing::category::thc_bbuild);
  const Mat B = df.build_b_tilde(mo.C, mo.C); // reuse the same 3-center store
  occ::timing::stop(occ::timing::category::thc_bbuild);
  const ThcFactors f = build_thc_from_B(basis, mo, opts, B);
  const int P = f.n_isdf;
  const Mat Xo = f.X.topRows(o);    // (o x P): MO value at each interp point
  const Mat Xv = f.X.bottomRows(v); // (v x P)
  occ::log::debug("THC-CCSD: {} interpolation points, metric cond={:.3e}", P,
                  f.metric_condition);

  // ovvv = (ia|bc) = sum_PQ Xo(i,P) Xv(a,P) V(P,Q) Xv(b,Q) Xv(c,Q)
  Mat Ev(P, v * v); // Ev(Q, b*v+c) = Xv(b,Q) Xv(c,Q)
  for (int Q = 0; Q < P; ++Q)
    for (int b = 0; b < v; ++b)
      for (int c = 0; c < v; ++c)
        Ev(Q, b * v + c) = Xv(b, Q) * Xv(c, Q);
  const Mat Wv = f.V * Ev; // (P x v*v): Wv(P, bc)
  Mat Eov(P, o * v);       // Eov(P, i*v+a) = Xo(i,P) Xv(a,P)
  for (int p = 0; p < P; ++p)
    for (int i = 0; i < o; ++i)
      for (int a = 0; a < v; ++a)
        Eov(p, i * v + a) = Xo(i, p) * Xv(a, p);
  e.ovvv = unpack(Eov.transpose() * Wv, o, v, v, v);

  // vvvv ladder, per occupied pair (i,j) as a GEMM chain, parallel over pairs:
  //   T_ij = Xvᵀ tau_ij Xv (PxP);  M_ij = V∘T_ij;  L_ij = Xv M_ij Xvᵀ (vxv).
  // Never forms vvvv; peak memory is one PxP block per thread (not o²·P²).
  auto Xv_sh = std::make_shared<Mat>(Xv);
  auto V_sh = std::make_shared<Mat>(f.V);
  e.ladder = [Xv_sh, V_sh, o, v](const T4 &tau) -> T4 {
    const Mat &Xvm = *Xv_sh; // (v x P)
    const Mat &Vm = *V_sh;   // (P x P)
    T4 result(o, o, v, v);
    occ::parallel::parallel_for(size_t(0), static_cast<size_t>(o) * o,
                                [&](size_t ij) {
      const int i = static_cast<int>(ij / o);
      const int j = static_cast<int>(ij % o);
      Mat tij(v, v);
      for (int c = 0; c < v; ++c)
        for (int d = 0; d < v; ++d)
          tij(c, d) = tau(i, j, c, d);
      const Mat T = Xvm.transpose() * tij * Xvm;  // (P x P)
      const Mat M = Vm.cwiseProduct(T);           // (P x P)
      const Mat L = Xvm * M * Xvm.transpose();     // (v x v)
      for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b)
          result(i, j, a, b) = L(a, b);
    });
    return result;
  };
  return e;
}

} // namespace occ::qm::cc
