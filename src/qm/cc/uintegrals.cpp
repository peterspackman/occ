#include <memory>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/cc/thc.h> // mo_eri_general
#include <occ/qm/cc/uintegrals.h>
#include <occ/qm/correlation/df_integrals.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/opmatrix.h> // block::a / block::b

// Spin-blocked spatial MO integrals for spin-adapted unrestricted CCSD. Every
// block PySCF's UCCSD update_amps reads is built here, in chemist notation.
// exact: the three chemist tensors (Gaa, Gbb, Gab) via the shared semidirect
// AO->MO transform, with the spatial vvvv tensors stored for the ladders.
// df: all blocks from the metric-folded DF B-tensor; the vvvv ladders contract
// through the per-spin B-tensors (alpha left, beta right for the ab ladder), so
// no O(V^4) block is ever formed.

namespace occ::qm::cc {

using occ::Mat;
using occ::Vec;
using T3 = Eigen::Tensor<double, 3>;
using T4 = Eigen::Tensor<double, 4>;
using Idx4 = Eigen::array<Eigen::Index, 4>;
using Sh4 = Eigen::array<int, 4>;
template <int N> using IA = Eigen::array<Eigen::IndexPair<int>, N>;
inline Eigen::IndexPair<int> ip(int a, int b) { return {a, b}; }

namespace {

// Chemist-tensor sub-block G[o0:o0+n0, o1:o1+n1, o2:o2+n2, o3:o3+n3].
T4 blk(const T4 &G, int o0, int n0, int o1, int n1, int o2, int n2, int o3,
       int n3) {
  return G.slice(Idx4{o0, o1, o2, o3}, Idx4{n0, n1, n2, n3});
}

// Unpack a (np*nq x nr*ns) B-block product into chemist tensor out(p,q,r,s).
T4 unpack(const Mat &M, int np, int nq, int nr, int ns) {
  T4 out(np, nq, nr, ns);
  for (int p = 0; p < np; ++p)
    for (int q = 0; q < nq; ++q)
      for (int r = 0; r < nr; ++r)
        for (int s = 0; s < ns; ++s)
          out(p, q, r, s) = M(p * nq + q, r * ns + s);
  return out;
}

// Per-spin active orbitals after dropping n_frozen core orbitals.
struct SpinMO {
  Mat Ca, Cb;       // active coefficients (nbf x nmoa/nmob)
  Vec ea, eb;       // active orbital energies (occ then vir)
  int oa, ob, va, vb;
  Mat Coa, Cva;     // alpha occ / virt
  Mat Cob, Cvb;     // beta occ / virt
};

SpinMO extract(const MolecularOrbitals &mo, int n_frozen) {
  const int nbf = static_cast<int>(mo.n_ao);
  Mat Ca, Cb;
  Vec ea, eb;
  int na, nb;
  if (mo.kind == SpinorbitalKind::Unrestricted) {
    Ca = occ::qm::block::a(mo.C);
    Cb = occ::qm::block::b(mo.C);
    ea = mo.energies.head(nbf);
    eb = mo.energies.segment(nbf, nbf);
    na = static_cast<int>(mo.n_alpha);
    nb = static_cast<int>(mo.n_beta);
  } else {
    Ca = mo.C;
    Cb = mo.C;
    ea = mo.energies.head(nbf);
    eb = ea;
    na = static_cast<int>(mo.n_alpha);
    nb = na;
  }
  const int nf = n_frozen;
  SpinMO s;
  const int nmoa = static_cast<int>(Ca.cols()) - nf;
  const int nmob = static_cast<int>(Cb.cols()) - nf;
  s.Ca = Ca.middleCols(nf, nmoa);
  s.Cb = Cb.middleCols(nf, nmob);
  s.ea = ea.segment(nf, nmoa);
  s.eb = eb.segment(nf, nmob);
  s.oa = na - nf;
  s.ob = nb - nf;
  s.va = nmoa - s.oa;
  s.vb = nmob - s.ob;
  s.Coa = s.Ca.leftCols(s.oa);
  s.Cva = s.Ca.middleCols(s.oa, s.va);
  s.Cob = s.Cb.leftCols(s.ob);
  s.Cvb = s.Cb.middleCols(s.ob, s.vb);
  return s;
}

// B3[x,y,P] = Bvv row x*v+y, for the DF ladder.
std::shared_ptr<T3> make_B3(const Mat &Bvv, int v) {
  const Eigen::Index naux = Bvv.cols();
  auto B3 = std::make_shared<T3>(v, v, naux);
  for (int x = 0; x < v; ++x)
    for (int y = 0; y < v; ++y)
      for (Eigen::Index P = 0; P < naux; ++P)
        (*B3)(x, y, P) = Bvv(x * v + y, P);
  return B3;
}

// DF/ladder closure: out(I,J,a,b) = sum_{c,d} (ac|bd) tau(I,J,c,d), contracting
// through B3l (left virtuals) and B3r (right virtuals). aa/bb use B3l==B3r; the
// ab ladder uses alpha B3 on the left, beta B3 on the right.
std::function<T4(const T4 &)> df_ladder(std::shared_ptr<T3> B3l,
                                        std::shared_ptr<T3> B3r, int nol,
                                        int nor, int vl, int vr) {
  return [B3l, B3r, nol, nor, vl, vr](const T4 &tau) -> T4 {
    T4 out(nol, nor, vl, vr);
    const IA<1> c1 = {ip(1, 0)};           // contract c
    const IA<2> Pd = {ip(1, 2), ip(2, 1)}; // contract P, d
    occ::parallel::parallel_for(
        size_t(0), static_cast<size_t>(nol) * nor, [&](size_t ij) {
          const int i = static_cast<int>(ij / nor);
          const int j = static_cast<int>(ij % nor);
          const Eigen::Tensor<double, 2> tau_ij = tau.chip(i, 0).chip(j, 0);
          const T3 M = B3l->contract(tau_ij, c1);     // (a, P, d)
          const Eigen::Tensor<double, 2> Lij = M.contract(*B3r, Pd); // (a, b)
          for (int a = 0; a < vl; ++a)
            for (int b = 0; b < vr; ++b)
              out(i, j, a, b) = Lij(a, b);
        });
    return out;
  };
}

// Assemble all UCCIntegrals blocks from the alpha/beta DF B-blocks.
struct BSet {
  Mat Boo, Bov, Bvv, Bvo;
};
BSet build_bset(DFIntegrals &df, const Mat &Cocc, const Mat &Cvirt) {
  return {df.build_b_tilde(Cocc, Cocc), df.build_b_tilde(Cocc, Cvirt),
          df.build_b_tilde(Cvirt, Cvirt), df.build_b_tilde(Cvirt, Cocc)};
}

// Fill every stored (<= 2-virtual + ovvv) block of `e` from the per-spin DF
// B-blocks A (alpha) and B (beta). The vvvv ladders are set by the caller.
void fill_df_blocks(UCCIntegrals &e, const BSet &A, const BSet &B) {
  const int oa = e.nocca, ob = e.noccb, va = e.nvira, vb = e.nvirb;
  e.oooo = unpack(A.Boo * A.Boo.transpose(), oa, oa, oa, oa);
  e.ovoo = unpack(A.Bov * A.Boo.transpose(), oa, va, oa, oa);
  e.ovov = unpack(A.Bov * A.Bov.transpose(), oa, va, oa, va);
  e.oovv = unpack(A.Boo * A.Bvv.transpose(), oa, oa, va, va);
  e.ovvo = unpack(A.Bov * A.Bvo.transpose(), oa, va, va, oa);
  e.ovvv = unpack(A.Bov * A.Bvv.transpose(), oa, va, va, va);

  e.OOOO = unpack(B.Boo * B.Boo.transpose(), ob, ob, ob, ob);
  e.OVOO = unpack(B.Bov * B.Boo.transpose(), ob, vb, ob, ob);
  e.OVOV = unpack(B.Bov * B.Bov.transpose(), ob, vb, ob, vb);
  e.OOVV = unpack(B.Boo * B.Bvv.transpose(), ob, ob, vb, vb);
  e.OVVO = unpack(B.Bov * B.Bvo.transpose(), ob, vb, vb, ob);
  e.OVVV = unpack(B.Bov * B.Bvv.transpose(), ob, vb, vb, vb);

  e.ooOO = unpack(A.Boo * B.Boo.transpose(), oa, oa, ob, ob);
  e.ovOO = unpack(A.Bov * B.Boo.transpose(), oa, va, ob, ob);
  e.ovOV = unpack(A.Bov * B.Bov.transpose(), oa, va, ob, vb);
  e.ooVV = unpack(A.Boo * B.Bvv.transpose(), oa, oa, vb, vb);
  e.ovVO = unpack(A.Bov * B.Bvo.transpose(), oa, va, vb, ob);
  e.ovVV = unpack(A.Bov * B.Bvv.transpose(), oa, va, vb, vb);

  e.OVoo = unpack(B.Bov * A.Boo.transpose(), ob, vb, oa, oa);
  e.OOvv = unpack(B.Boo * A.Bvv.transpose(), ob, ob, va, va);
  e.OVvo = unpack(B.Bov * A.Bvo.transpose(), ob, vb, va, oa);
  e.OVvv = unpack(B.Bov * A.Bvv.transpose(), ob, vb, va, va);
}

// THC ladder: out(I,J,a,b) = sum_{c,d} (ac|bd) tau(I,J,c,d), with
// (ac|bd) ~ sum_PQ Xvl(a,P) Xvl(c,P) V(P,Q) Xvr(b,Q) Xvr(d,Q). Per occ pair:
// T = Xvl^T tau_IJ Xvr, M = V .* T, L = Xvl M Xvr^T -- all GEMMs.
std::function<T4(const T4 &)> thc_ladder(Mat Xvl, Mat V, Mat Xvr, int nol,
                                         int nor) {
  const int vl = static_cast<int>(Xvl.rows());
  const int vr = static_cast<int>(Xvr.rows());
  return [Xvl, V, Xvr, nol, nor, vl, vr](const T4 &tau) -> T4 {
    T4 out(nol, nor, vl, vr);
    occ::parallel::parallel_for(
        size_t(0), static_cast<size_t>(nol) * nor, [&](size_t ij) {
          const int i = static_cast<int>(ij / nor);
          const int j = static_cast<int>(ij % nor);
          Eigen::Tensor<double, 2> tt = tau.chip(i, 0).chip(j, 0); // (vl,vr)
          const Eigen::Map<const Mat> tauM(tt.data(), vl, vr);
          const Mat T = Xvl.transpose() * tauM * Xvr; // (P x Q)
          const Mat M = V.cwiseProduct(T);
          const Mat L = Xvl * M * Xvr.transpose(); // (vl x vr)
          for (int a = 0; a < vl; ++a)
            for (int b = 0; b < vr; ++b)
              out(i, j, a, b) = L(a, b);
        });
    return out;
  };
}

} // namespace

UCCIntegrals u_exact_eris(const AOBasis &basis, const MolecularOrbitals &mo,
                          int n_frozen, std::size_t memory_budget) {
  occ::timing::start(occ::timing::category::cc_ao2mo);
  const SpinMO s = extract(mo, n_frozen);
  const int oa = s.oa, ob = s.ob, va = s.va, vb = s.vb;

  UCCIntegrals e;
  e.nocca = oa;
  e.noccb = ob;
  e.nvira = va;
  e.nvirb = vb;
  e.mo_energy_a = s.ea;
  e.mo_energy_b = s.eb;

  IntegralEngine engine(basis);
  const T4 Gaa = mo_eri_general(engine, s.Ca, s.Ca, s.Ca, s.Ca, memory_budget);
  const T4 Gbb = mo_eri_general(engine, s.Cb, s.Cb, s.Cb, s.Cb, memory_budget);
  const T4 Gab = mo_eri_general(engine, s.Ca, s.Ca, s.Cb, s.Cb, memory_budget);

  e.oooo = blk(Gaa, 0, oa, 0, oa, 0, oa, 0, oa);
  e.ovoo = blk(Gaa, 0, oa, oa, va, 0, oa, 0, oa);
  e.ovov = blk(Gaa, 0, oa, oa, va, 0, oa, oa, va);
  e.oovv = blk(Gaa, 0, oa, 0, oa, oa, va, oa, va);
  e.ovvo = blk(Gaa, 0, oa, oa, va, oa, va, 0, oa);
  e.ovvv = blk(Gaa, 0, oa, oa, va, oa, va, oa, va);

  e.OOOO = blk(Gbb, 0, ob, 0, ob, 0, ob, 0, ob);
  e.OVOO = blk(Gbb, 0, ob, ob, vb, 0, ob, 0, ob);
  e.OVOV = blk(Gbb, 0, ob, ob, vb, 0, ob, ob, vb);
  e.OOVV = blk(Gbb, 0, ob, 0, ob, ob, vb, ob, vb);
  e.OVVO = blk(Gbb, 0, ob, ob, vb, ob, vb, 0, ob);
  e.OVVV = blk(Gbb, 0, ob, ob, vb, ob, vb, ob, vb);

  e.ooOO = blk(Gab, 0, oa, 0, oa, 0, ob, 0, ob);
  e.ovOO = blk(Gab, 0, oa, oa, va, 0, ob, 0, ob);
  e.ovOV = blk(Gab, 0, oa, oa, va, 0, ob, ob, vb);
  e.ooVV = blk(Gab, 0, oa, 0, oa, ob, vb, ob, vb);
  e.ovVO = blk(Gab, 0, oa, oa, va, ob, vb, 0, ob);
  e.ovVV = blk(Gab, 0, oa, oa, va, ob, vb, ob, vb);

  e.OVoo = T4(blk(Gab, 0, oa, 0, oa, 0, ob, ob, vb)).shuffle(Sh4{2, 3, 0, 1});
  e.OOvv = T4(blk(Gab, oa, va, oa, va, 0, ob, 0, ob)).shuffle(Sh4{2, 3, 0, 1});
  e.OVvo = T4(blk(Gab, oa, va, 0, oa, 0, ob, ob, vb)).shuffle(Sh4{2, 3, 0, 1});
  e.OVvv = T4(blk(Gab, oa, va, oa, va, 0, ob, ob, vb)).shuffle(Sh4{2, 3, 0, 1});

  auto vvvv_aa = std::make_shared<T4>(blk(Gaa, oa, va, oa, va, oa, va, oa, va));
  auto vvvv_bb = std::make_shared<T4>(blk(Gbb, ob, vb, ob, vb, ob, vb, ob, vb));
  auto vvVV = std::make_shared<T4>(blk(Gab, oa, va, oa, va, ob, vb, ob, vb));
  e.ladder_aa = [vvvv_aa](const T4 &tau) -> T4 {
    return tau.contract(*vvvv_aa, IA<2>{ip(2, 1), ip(3, 3)});
  };
  e.ladder_bb = [vvvv_bb](const T4 &tau) -> T4 {
    return tau.contract(*vvvv_bb, IA<2>{ip(2, 1), ip(3, 3)});
  };
  e.ladder_ab = [vvVV](const T4 &tau) -> T4 {
    return tau.contract(*vvVV, IA<2>{ip(2, 1), ip(3, 3)});
  };

  occ::timing::stop(occ::timing::category::cc_ao2mo);
  return e;
}

UCCIntegrals u_df_eris(const AOBasis &basis, const AOBasis &aux_basis,
                       const MolecularOrbitals &mo, int n_frozen,
                       std::size_t memory_budget) {
  occ::timing::start(occ::timing::category::cc_ao2mo);
  const SpinMO s = extract(mo, n_frozen);
  const int oa = s.oa, ob = s.ob, va = s.va, vb = s.vb;

  AOBasis aux = aux_basis;
  aux.set_kind(basis.kind());
  IntegralEngineDF df_engine(basis.atoms(), basis.shells(), aux.shells());
  DFIntegrals df(df_engine, memory_budget);

  const BSet A = build_bset(df, s.Coa, s.Cva); // alpha
  const BSet B = build_bset(df, s.Cob, s.Cvb); // beta

  UCCIntegrals e;
  e.nocca = oa;
  e.noccb = ob;
  e.nvira = va;
  e.nvirb = vb;
  e.mo_energy_a = s.ea;
  e.mo_energy_b = s.eb;
  fill_df_blocks(e, A, B);

  // vvvv ladders through the per-spin B-tensors
  auto B3a = make_B3(A.Bvv, va);
  auto B3b = make_B3(B.Bvv, vb);
  e.ladder_aa = df_ladder(B3a, B3a, oa, oa, va, va);
  e.ladder_bb = df_ladder(B3b, B3b, ob, ob, vb, vb);
  e.ladder_ab = df_ladder(B3a, B3b, oa, ob, va, vb);

  occ::timing::stop(occ::timing::category::cc_ao2mo);
  return e;
}

UCCIntegrals u_thc_eris(const AOBasis &basis, const AOBasis &aux_basis,
                        const MolecularOrbitals &mo, const ThcOptions &opts,
                        int n_frozen, std::size_t memory_budget) {
  occ::timing::start(occ::timing::category::cc_ao2mo);
  const SpinMO s = extract(mo, n_frozen);
  const int oa = s.oa, ob = s.ob, va = s.va, vb = s.vb;

  AOBasis aux = aux_basis;
  aux.set_kind(basis.kind());
  IntegralEngineDF df_engine(basis.atoms(), basis.shells(), aux.shells());
  DFIntegrals df(df_engine, memory_budget);

  const BSet A = build_bset(df, s.Coa, s.Cva);
  const BSet B = build_bset(df, s.Cob, s.Cvb);

  UCCIntegrals e;
  e.nocca = oa;
  e.noccb = ob;
  e.nvira = va;
  e.nvirb = vb;
  e.mo_energy_a = s.ea;
  e.mo_energy_b = s.eb;
  fill_df_blocks(e, A, B); // cheap blocks from DF

  // cross-spin THC factors (shared ISDF points, three cores), using the same
  // DF store for the LS-THC reference.
  const Mat Ba = df.build_b_tilde(s.Ca, s.Ca); // (nmoa^2 x naux)
  const Mat Bb = df.build_b_tilde(s.Cb, s.Cb);
  const UThcFactors f = build_uthc(basis, s.Ca, s.Cb, Ba, Bb, opts);
  const Mat Xva = f.Xa.middleRows(oa, va); // (va x n_isdf)
  const Mat Xvb = f.Xb.middleRows(ob, vb); // (vb x n_isdf)

  e.ladder_aa = thc_ladder(Xva, f.Vaa, Xva, oa, oa);
  e.ladder_bb = thc_ladder(Xvb, f.Vbb, Xvb, ob, ob);
  e.ladder_ab = thc_ladder(Xva, f.Vab, Xvb, oa, ob);

  occ::timing::stop(occ::timing::category::cc_ao2mo);
  return e;
}

} // namespace occ::qm::cc
