// Correlated one-particle densities of a closed-shell RHF reference: relaxed
// MP2, unrelaxed MP3 / MP4, and unrelaxed CCSD and CCSD(T) (Lambda equations).
// The equations are PySCF's ccsd_lambda / ccsd_rdm, their (T) versions and
// grad/mp2.py, term for term, on occ's CCIntegrals blocks. The
// cluster-in-molecule route is in local_density.cpp.
#include "detail/density.h"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <occ/core/diis.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/util.h>
#include <occ/qm/cc/ccsd.h>
#include <occ/qm/fitting_basis.h>
#include <tbb/enumerable_thread_specific.h>

namespace occ::qm::cc {

namespace detail {
namespace {

// occ's integral blocks in the row-major form the equations use, plus the vvvv
// ladder
struct Eris {
  int o = 0, v = 0;
  Tn foo, fov, fvv, oooo, ovoo, oovv, ovov, ovvo, ovvv;
  occ::Vec eo, ev;
  std::function<Eigen::Tensor<double, 4>(const Eigen::Tensor<double, 4> &)>
      ladder;
};

Eris convert(const occ::qm::cc::CCIntegrals &e) {
  Eris r;
  r.o = e.nocc;
  r.v = e.nvir;
  r.eo = e.mo_energy.head(e.nocc);
  r.ev = e.mo_energy.tail(e.nvir);
  occ::Mat f = e.fock;
  if (f.rows() != e.nocc + e.nvir)
    f = e.mo_energy.asDiagonal();
  r.foo = block(f, 0, 0, r.o, r.o);
  r.fov = block(f, 0, r.o, r.o, r.v);
  r.fvv = block(f, r.o, r.o, r.v, r.v);
  r.oooo = from_tensor(e.oooo);
  r.ovoo = from_tensor(e.ovoo);
  r.oovv = from_tensor(e.oovv);
  r.ovov = from_tensor(e.ovov);
  r.ovvo = from_tensor(e.ovvo);
  r.ovvv = from_tensor(e.ovvv);
  r.ladder = e.ladder;
  return r;
}

struct Imds {
  Tn w1, w2, w3, w4, woooo, wvooo, wVOov, wvOOv, wvvov;
};

// PySCF ccsd_lambda.make_intermediates, unblocked
Imds make_intermediates(const Tn &t1, const Tn &t2, const Eris &e) {
  const Tn &ovvv = e.ovvv, &ovoo = e.ovoo, &ovvo = e.ovvo, &oovv = e.oovv,
           &oooo = e.oooo, &fov = e.fov;
  Tn w1 = e.fvv - einsum("ja,jb->ba", fov, t1);
  Tn w2 = e.foo + einsum("ib,jb->ij", fov, t1);
  Tn w3 = einsum("kc,jkbc->bj", fov, t2) * 2.0 + T(fov, {1, 0});
  w3 -= einsum("kc,kjbc->bj", fov, t2);
  w3 += einsum("cb,jc->bj", einsum("kc,kb->cb", fov, t1), t1);
  Tn w4 = fov;
  const Tn vvov = T(ovvv, {2, 3, 0, 1});
  w1 += einsum("jcba,jc->ba", ovvv, t1 * 2.0);
  w1 -= einsum("jabc,jc->ba", ovvv, t1);
  Tn theta = t2 * 2.0 - T(t2, {1, 0, 2, 3});
  w3 += einsum("jkcd,kdcb->bj", theta, ovvv);
  Tn wVOov = einsum("jbcd,kd->bjkc", ovvv, t1);
  Tn wvOOv = einsum("cbjd,kd->cjkb", vvov, t1 * -1.0);
  const Tn tau = t2 + einsum("ia,jb->ijab", t1, t1);
  Tn wvooo = einsum("cibd,jkbd->ckij",
                    T(vvov, {0, 2, 1, 3}) * 2.0 - T(vvov, {0, 2, 3, 1}), tau);

  Tn wvvov = einsum("jabd,jkcd->abkc", ovvv, t2) * -1.5;
  wvvov += T(vvov, {0, 3, 2, 1}) * 2.0;
  wvvov -= vvov;
  {
    const Tn vackb =
        einsum("acjd,kjbd->ackb", vvov * 2.0 - T(ovvv, {1, 2, 0, 3}),
               t2 * 2.0 - T(t2, {0, 1, 3, 2}));
    wvvov += T(vackb, {0, 3, 2, 1});
    wvvov -= vackb * 0.5;
  }

  w2 += einsum("kbij,kb->ij", ovoo, t1) * 2.0;
  w2 -= einsum("ibkj,kb->ij", ovoo, t1);
  theta = T(t2, {1, 0, 2, 3}) * 2.0 - t2;
  w3 -= einsum("lckj,klcb->bj", ovoo, theta);

  Tn woooo = einsum("lc,jcik->ijkl", t1, ovoo);
  woooo += T(woooo, {1, 0, 3, 2});
  wvOOv += einsum("lbjk,lc->bjkc", ovoo, t1);
  wVOov -= einsum("jbkl,lc->bjkc", ovoo, t1);
  wvooo += T(ovoo, {1, 3, 2, 0}) * 2.0;
  wvooo -= T(ovoo, {1, 0, 2, 3});
  wvooo -= einsum("klbc,iblj->ckij", t2, ovoo * 1.5);
  const Tn g2ovoo = ovoo * 2.0 - T(ovoo, {2, 1, 0, 3});
  {
    const Tn vcjik =
        einsum("jlcb,lbki->cjki", t2 * 2.0 - T(t2, {1, 0, 2, 3}), g2ovoo);
    wvooo += T(vcjik, {0, 3, 2, 1});
    wvooo -= vcjik * 0.5;
  }

  const Tn voov = T(ovvo, {1, 0, 3, 2});
  woooo += einsum("cijd,klcd->ijkl", voov, tau);
  const Tn g2voov = voov * 2.0 - T(voov, {0, 2, 1, 3});
  const Tn tmpw4 = einsum("ckld,ld->kc", g2voov, t1);
  w1 -= einsum("ckja,kjcb->ba", g2voov, t2);
  w1 -= einsum("ja,jb->ba", tmpw4, t1);
  w2 += einsum("jkbc,bikc->ij", t2, g2voov);
  w2 += einsum("ib,jb->ij", tmpw4, t1);
  w3 += einsum("bc,jc->bj", einsum("kb,kc->bc", t1, tmpw4), t1);
  w4 += tmpw4;
  wvOOv += einsum("bljk,lc->bjkc", einsum("bljd,kd->bljk", voov, t1), t1);
  wVOov -= einsum("bjlk,lc->bjkc", einsum("bjld,kd->bjlk", voov, t1), t1);
  Tn VOov = einsum("bjld,klcd->bjkc", g2voov, t2);
  VOov -= einsum("bjld,kldc->bjkc", voov, t2);
  VOov += voov;
  Tn vOOv = einsum("bljd,kldc->bjkc", voov, t2);
  vOOv -= T(oovv, {2, 1, 0, 3});
  wVOov += VOov;
  wvOOv += vOOv;
  const Tn ov1 = vOOv * 2.0 + VOov, ov2 = VOov * 2.0 + vOOv;
  wvooo -= einsum("jb,bikc->ckij", t1, ov1);
  wvooo += einsum("kb,bijc->ckij", t1, ov2);
  w3 += einsum("ckjb,kc->bj", ov2, t1);
  wvvov += einsum("ajkc,jb->abkc", ov1, t1);
  wvvov -= einsum("ajkb,jc->abkc", ov2, t1);
  wvvov += einsum("laki,klbc->abic", g2ovoo, tau);
  woooo += T(oooo, {0, 2, 1, 3});
  w3 += einsum("bc,jc->bj", w1, t1);
  w3 -= einsum("kj,kb->bj", w2, t1);
  return {w1, w2, w3, w4, woooo, wvooo, wVOov, wvOOv, wvvov};
}

// PySCF ccsd_lambda.update_lambda, unblocked: the next (l1, l2) from the
// current ones
void update_lambda(const Tn &t1, const Tn &t2, const Tn &l1, const Tn &l2,
                   const Eris &e, const Imds &m, Tn &l1new, Tn &l2new) {
  const int o = e.o, v = e.v;
  const Tn theta = t2 * 2.0 - T(t2, {0, 1, 3, 2});
  const Tn mba = einsum("klca,klcb->ba", l2, theta);
  const Tn mij = einsum("ikcd,jkcd->ij", l2, theta);
  const Tn mba1 = einsum("jc,jb->bc", l1, t1) + mba;
  const Tn mij1 = einsum("kb,jb->kj", l1, t1) + mij;
  Tn mia1 = t1 + einsum("kc,jkbc->jb", l1, t2) * 2.0;
  mia1 -= einsum("kc,jkcb->jb", l1, t2);
  mia1 -= einsum("jk,kb->jb", einsum("ja,ka->jk", t1, l1), t1);
  mia1 -= einsum("bd,jd->jb", mba, t1);
  mia1 -= einsum("lj,lb->jb", mij, t1);
  l2new = from_tensor(e.ladder(to_tensor(l2))); // sum_cd (ac|bd) l2(i,j,c,d)
  l1new = einsum("ijab,jb->ia", l2new, t1) * 2.0;
  l1new -= einsum("jiab,jb->ia", l2new, t1);
  l2new *= 0.5; // the (ia)<->(jb) symmetrisation at the end doubles it back

  Tn w1 = m.w1, w2 = m.w2;
  for (int a = 0; a < v; a++)
    w1.at(a, a) -= e.ev(a);
  for (int i = 0; i < o; i++)
    w2.at(i, i) -= e.eo(i);
  l1new += e.fov;
  l1new += einsum("ib,ba->ia", l1, w1);
  l1new -= einsum("ja,ij->ia", l1, w2);
  l1new -= einsum("ik,ka->ia", mij, m.w4);
  l1new -= einsum("ca,ic->ia", mba, m.w4);
  l1new += einsum("ijab,bj->ia", l2, m.w3) * 2.0;
  l1new -= einsum("ijba,bj->ia", l2, m.w3);
  l2new += einsum("ia,jb->ijab", l1, m.w4);
  l2new += einsum("jibc,ca->jiba", l2, w1);
  l2new -= einsum("jk,kiba->jiba", w2, l2);
  l1new -= einsum("iajk,kj->ia", e.ovoo, mij1) * 2.0;
  l1new += einsum("jaik,kj->ia", e.ovoo, mij1);
  l2new -= einsum("jbki,ka->jiba", e.ovoo, l1);
  const Tn tau = t2 + einsum("ia,jb->ijab", t1, t1);
  const Tn l2tau = einsum("ijcd,klcd->ijkl", l2, tau);
  const Tn l2t1 = einsum("jidc,kc->ijkd", l2, t1);
  l1new -= einsum("jb,jiab->ia", l1, e.oovv);
  l1new += einsum("iabc,bc->ia", e.ovvv, mba1) * 2.0;
  l1new -= einsum("ibca,bc->ia", e.ovvv, mba1);
  l2new += einsum("jbac,ic->jiba", e.ovvv, l1);
  {
    const Tn m4 = einsum("ijkd,kadb->ijab", l2t1, e.ovvv);
    l2new -= m4;
    l1new -= einsum("ijab,jb->ia", m4, t1) * 2.0;
    l1new -= einsum("ijab,ia->jb", m4, t1) * 2.0;
    l1new += einsum("jiab,jb->ia", m4, t1);
    l1new += einsum("jiab,ia->jb", m4, t1);
  }

  const Tn voov = T(e.ovvo, {1, 0, 3, 2});
  l1new += einsum("jb,aijb->ia", l1, voov) * 2.0;
  l2new += T(voov, {1, 2, 0, 3}) * 0.5;
  l2new -= einsum("bjic,ca->jiba", voov, mba1);
  l2new -= einsum("bjka,ik->jiba", voov, mij1);
  l1new += einsum("aijb,jb->ia", voov, mia1) * 2.0;
  l1new -= einsum("bija,jb->ia", voov, mia1);
  {
    const Tn m4 = einsum("ijkl,aklb->ijab", l2tau, voov);
    l2new += m4 * 0.5;
    l1new += einsum("ijab,jb->ia", m4, t1) * 2.0;
    l1new -= einsum("ijba,jb->ia", m4, t1);
  }

  l1new -= einsum("ckij,jkca->ia", m.wvooo, l2);
  l1new += einsum("abkc,kibc->ia", m.wvvov, l2);
  l2new +=
      einsum("iakc,bjkc->jiba", T(l2, {0, 2, 1, 3}) - T(l2, {0, 3, 1, 2}) * 0.5,
             m.wVOov * 2.0 + m.wvOOv);
  {
    const Tn tmp = einsum("jkca,bikc->jiba", l2, m.wvOOv);
    l2new += tmp;
    l2new += T(tmp, {1, 0, 2, 3}) * 0.5;
  }
  {
    const Tn m3 = einsum("ijkl,klab->ijab", m.woooo, l2);
    l2new += m3 * 0.5;
    l1new += einsum("ijab,jb->ia", m3, t1) * 2.0;
    l1new -= einsum("ijba,jb->ia", m3, t1);
  }

  for (int i = 0; i < o; i++)
    for (int a = 0; a < v; a++)
      l1new.at(i, a) /= e.eo(i) - e.ev(a);
  l2new += T(l2new, {1, 0, 3, 2});
  size_t w = 0;
  for (int i = 0; i < o; i++)
    for (int j = 0; j < o; j++)
      for (int a = 0; a < v; a++)
        for (int b = 0; b < v; b++)
          l2new.v[w++] /= e.eo(i) - e.ev(a) + e.eo(j) - e.ev(b);
}

// t2_vo: the doubles r6(W) is contracted with for the vo block (t2 if null)
Triples triples(const Tn &t1, const Tn &t2, const Eris &e,
                const occ::Mat *proj = nullptr, const Tn *t2_vo = nullptr) {
  const int o = e.o, v = e.v;
  const Tn &tvo = t2_vo ? *t2_vo : t2;
  const size_t vv = size_t(v) * v, v3 = vv * v;
  const Tn L1 = T(e.ovvv, {0, 1, 3, 2}); //[i][a][b][f] = (ia|fb)
  const Tn L2 = T(e.ovvv, {0, 3, 1, 2}); //[k][e][f][b] = (kf|be)
  const Tn t2T = T(t2, {0, 1, 3, 2});    //[k][j][f][c] = t2(kj,cf)
  const Tn t2_t = T(t2, {1, 0, 2, 3});   //[k][m][b][c] = t2(mk,bc)
  const Tn ovoo_t = T(e.ovoo, {0, 2, 1, 3}),
           ovov_t = T(e.ovov, {0, 2, 1, 3}); //[i][j][a][m], [i][j][a][b]
  auto at = [](const Tn &a, const size_t block, const size_t size) {
    return a.v.data() + block * size;
  };
  auto ix = [v](const int a, const int b, const int c) {
    return (size_t(a) * v + b) * v + c;
  };

  // PySCF's un-symmetrised w for one ordered triple: (iafb,kjcf) - (iajm,mkbc),
  // into out as abc
  const int iv = v, ivv = static_cast<int>(vv);
  auto gemm = [](const bool ta, const bool tb, const int m, const int n,
                 const int k, const double alpha, const double *a,
                 const int lda, const double *b, const int ldb,
                 const double beta, double *c, const int ldc) {
    gemm_rm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  };
  auto base_w = [&](const int i, const int j, const int k, double *out) {
    gemm(false, false, ivv, iv, iv, 1.0, at(L1, i, v3), iv,
         at(t2T, size_t(k) * o + j, vv), iv, 0.0, out, iv);
    gemm(false, false, iv, ivv, o, -1.0,
         at(ovoo_t, size_t(i) * o + j, size_t(v) * o), o,
         at(t2_t, k, size_t(o) * vv), ivv, 1.0, out, ivv);
  };

  Triples r;
  r.l1_t = Tn({o, v});
  r.goo = Tn({o, o});
  r.gvv = Tn({v, v});
  r.dvo = Tn({v, o});
  if (proj)
    r.gvv_c = Tn({v, v});
  Tn joovv({o, o, v, v});
  std::vector<Tn> gvv_i(o, Tn({v, v})), dvo_i(o, Tn({v, o}));
  vec e_i(o, 0.0);
  // W + V/2 and r6(W) of every i of the current (j, k): the oo block pairs
  // W[ijk] with W[i'jk]
  RowMat WV(o, v3), RW(o, v3);
  // per-thread v^3 buffers: six unsymmetrised w blocks, W and V
  tbb::enumerable_thread_specific<std::array<vec, 8>> bufs([v3] {
    std::array<vec, 8> b;
    for (vec &x : b)
      x.assign(v3, 0.0);
    return b;
  });

  // block (i, j, k) and, for j > k, its b<->c mirror (i, k, j) (joint
  // permutation symmetry), given W and V of (i, j, k). The mirror's r6(W) and W
  // + V/2 are the swapped ones, so its E(T), l1, vv and oo shares equal those
  // of (i, j, k): weight 2. as_r6 does not commute with the swap: m' = as_r6(Y)
  // of the swapped Y = 2W + V/2. F_ov = 0, as in occ's CCSD.
  auto block = [&](const int i, const int j, const int k, const double *W,
                   const double *V, double *wv, double *rw, double *m,
                   double *mm) {
    const bool mirror = j != k;
    const double eijk = e.eo(i) + e.eo(j) + e.eo(k),
                 weight = mirror ? 2.0 : 1.0;
    double et = 0.0;
    for (int a = 0; a < v; a++)
      for (int b = 0; b < v; b++)
        for (int c = 0; c < v; c++) {
          const size_t x = ix(a, b, c), cab = ix(c, a, b), bca = ix(b, c, a),
                       cba = ix(c, b, a), acb = ix(a, c, b), bac = ix(b, a, c);
          // r6 and as_r6 act on abc only: r6(w) = 4w + w(cab) + w(bca) - 2
          // [w(cba) + w(acb) + w(bac)]
          rw[x] =
              4.0 * W[x] + W[cab] + W[bca] - 2.0 * (W[cba] + W[acb] + W[bac]);
          wv[x] = W[x] + 0.5 * V[x];
          const double y = 2.0 * W[x] + 0.5 * V[x],
                       y_acb = 2.0 * W[acb] + 0.5 * V[acb];
          m[x] = 2.0 * y - (2.0 * W[cba] + 0.5 * V[cba]) - y_acb;
          if (mirror)
            mm[x] = 2.0 * y_acb - (2.0 * W[cab] + 0.5 * V[cab]) - y;
          // E(T) = 1/3 sum W D r6(W + V/2) = 1/3 sum D r6(W) (W + V/2), r6
          // self-adjoint and D symmetric in abc
          et += (eijk - e.ev(a) - e.ev(b) - e.ev(c)) * rw[x] * wv[x];
        }
    e_i[i] += weight * et / 3.0;
    // l1_t[i] and joovv[i] are this i's alone; the vv, vo parts are per i and
    // summed after the loop
    MapR(r.l1_t.v.data() + size_t(i) * v, 1, v).noalias() +=
        weight *
        (CMapR(rw, v, vv) * CMapR(at(ovov_t, size_t(j) * o + k, vv), vv, 1))
            .transpose();
    gemm(false, true, iv, iv, ivv, weight, wv, ivv, rw, ivv, 1.0,
         gvv_i[i].v.data(), iv);
    // joovv[i, j] += m L2[k]; joovv[i, J, ab] -= sum_c (kc|jJ) m[ab, c]; the
    // mirror with j <-> k and m'
    gemm(false, false, iv, iv, ivv, 1.0, m, ivv, at(L2, k, v3), iv, 1.0,
         joovv.v.data() + (size_t(i) * o + j) * vv, iv);
    gemm(true, true, o, ivv, iv, -1.0,
         at(ovoo_t, size_t(k) * o + j, size_t(v) * o), o, m, iv, 1.0,
         joovv.v.data() + size_t(i) * o * vv, ivv);
    const Eigen::VectorXd dv = CMapR(rw, vv, v).transpose() *
                               CMapR(at(tvo, size_t(i) * o + j, vv), vv, 1);
    for (int c = 0; c < v; c++)
      dvo_i[i].at(c, k) += 0.5 * dv(c);
    if (!mirror)
      return;
    gemm(false, false, iv, iv, ivv, 1.0, mm, ivv, at(L2, j, v3), iv, 1.0,
         joovv.v.data() + (size_t(i) * o + k) * vv, iv);
    gemm(true, true, o, ivv, iv, -1.0,
         at(ovoo_t, size_t(j) * o + k, size_t(v) * o), o, mm, iv, 1.0,
         joovv.v.data() + size_t(i) * o * vv, ivv);
    // the mirror's dv(c) = sum_ab r6(W)[acb] t2(ik, ab)
    Eigen::VectorXd dvm = Eigen::VectorXd::Zero(v);
    for (int a = 0; a < v; a++)
      dvm.noalias() +=
          CMapR(rw + size_t(a) * vv, v, v) *
          CMapR(at(tvo, size_t(i) * o + k, vv) + size_t(a) * v, v, 1);
    for (int c = 0; c < v; c++)
      dvo_i[i].at(c, j) += 0.5 * dvm(c);
  };

  for (int j = 0; j < o; j++)
    for (int k = 0; k <= j; k++) {
      // the occupied triple (i, j, k)
      auto triple = [&](const int i) {
        std::array<vec, 8> &B = bufs.local();
        double *b0 = B[0].data(), *b1 = B[1].data(), *b2 = B[2].data(),
               *b3 = B[3].data(), *b4 = B[4].data(), *b5 = B[5].data();
        double *W = B[6].data(), *V = B[7].data();
        base_w(i, j, k, b0);
        base_w(i, k, j, b1);
        base_w(j, i, k, b2);
        base_w(j, k, i, b3);
        base_w(k, i, j, b4);
        base_w(k, j, i, b5);
        // p6 and the division by D in one pass; v's six permutations coincide
        // in pairs, so three terms twice
        const double eijk = e.eo(i) + e.eo(j) + e.eo(k);
        const double *Xij = at(ovov_t, size_t(i) * o + j, vv),
                     *Xik = at(ovov_t, size_t(i) * o + k, vv),
                     *Xjk = at(ovov_t, size_t(j) * o + k, vv);
        const double *t1i = at(t1, i, v), *t1j = at(t1, j, v),
                     *t1k = at(t1, k, v);
        for (int a = 0; a < v; a++)
          for (int b = 0; b < v; b++)
            for (int c = 0; c < v; c++) {
              const size_t x = ix(a, b, c);
              const double d = eijk - e.ev(a) - e.ev(b) - e.ev(c);
              W[x] = (b0[x] + b1[ix(a, c, b)] + b2[ix(b, a, c)] +
                      b3[ix(b, c, a)] + b4[ix(c, a, b)] + b5[ix(c, b, a)]) /
                     d;
              const size_t ab = size_t(a) * v + b, ac = size_t(a) * v + c,
                           bc = size_t(b) * v + c;
              V[x] = 2.0 *
                     (Xij[ab] * t1k[c] + Xik[ac] * t1j[b] + Xjk[bc] * t1i[a]) /
                     d;
            }
        block(i, j, k, W, V, WV.row(i).data(), RW.row(i).data(), b0, b1);
      };
      occ::parallel::parallel_for(size_t(0), size_t(o), [&](const size_t i) {
        triple(static_cast<int>(i));
      });
      // the vv block and E(T) attributed by the block's first occupied index,
      // projected onto proj's directions. E(T) = 1/3 sum_i <D_i r6(W)_i |
      // (W + V/2)_i>, a sum over i of products, so rotating i to the projected
      // directions partitions it exactly when proj spans the occupied space
      if (proj) {
        Eigen::VectorXd base(static_cast<Eigen::Index>(v3));
        const double ejk = e.eo(j) + e.eo(k);
        for (int a = 0; a < v; a++)
          for (int b = 0; b < v; b++)
            for (int c = 0; c < v; c++)
              base(ix(a, b, c)) = ejk - e.ev(a) - e.ev(b) - e.ev(c);
        const double weight = j == k ? 1.0 : 2.0;
        for (int c = 0; c < proj->cols(); c++) {
          const Eigen::VectorXd wp = WV.transpose() * proj->col(c),
                                rp = RW.transpose() * proj->col(c);
          gemm(false, true, iv, iv, ivv, weight, wp.data(), ivv, rp.data(), ivv,
               1.0, r.gvv_c.v.data(), iv);
          // sum_i p_i D_i r6(W)_i with D_i = e_i + base
          const Eigen::VectorXd zp =
              base.cwiseProduct(rp) +
              RW.transpose() * proj->col(c).cwiseProduct(e.eo);
          r.e_t_c += weight * zp.dot(wp) / 3.0;
        }
      }
      // the response form (PySCF for_grad=True): the full oo block, not only
      // its diagonal; the output is only o x o, so the v^3 sum is split over
      // the threads
      {
        const size_t nb = std::min<size_t>(
                         v3,
                         static_cast<size_t>(occ::parallel::get_num_threads())),
                     per = (v3 + nb - 1) / nb;
        std::vector<occ::Mat> part(nb, occ::Mat::Zero(o, o));
        occ::parallel::parallel_for(size_t(0), nb, [&](const size_t b) {
          const Eigen::Index c0 = static_cast<Eigen::Index>(b * per),
                             nc = static_cast<Eigen::Index>(
                                      std::min(v3, (b + 1) * per)) -
                                  c0;
          if (nc > 0)
            part[b].noalias() =
                WV.middleCols(c0, nc) * RW.middleCols(c0, nc).transpose();
        });
        MapR goo(r.goo.v.data(), o, o);
        for (size_t b = 0; b < nb; b++)
          goo += (j == k ? 1.0 : 2.0) * part[b];
      }
    }
  for (int i = 0; i < o; i++) {
    r.gvv += gvv_i[i];
    r.dvo += dvo_i[i];
    r.e_t += e_i[i];
    for (int a = 0; a < v; a++)
      r.l1_t.at(i, a) *= 0.5 / (e.eo(i) - e.ev(a));
  }
  joovv += T(joovv, {1, 0, 3, 2});
  r.l2_t = joovv;
  size_t n = 0;
  for (int i = 0; i < o; i++)
    for (int j = 0; j < o; j++)
      for (int a = 0; a < v; a++)
        for (int b = 0; b < v; b++)
          r.l2_t.v[n++] /= e.eo(i) - e.ev(a) + e.eo(j) - e.ev(b);
  return r;
}

// PySCF ccsd_rdm._make_rdm1: the spin-traced density in the correlated MOs,
// reference occupation included; symmetric, so no question of <p+q> against
// <q+p>
occ::Mat rdm1(const Gamma1 &g, const Tn &l1) {
  const int o = l1.d[0], v = l1.d[1];
  occ::Mat dm = occ::Mat::Zero(o + v, o + v);
  for (int i = 0; i < o; i++) {
    for (int j = 0; j < o; j++)
      dm(i, j) = g.doo.at(i, j) + g.doo.at(j, i);
    dm(i, i) += 2.0;
    for (int a = 0; a < v; a++)
      dm(i, o + a) = dm(o + a, i) = l1.at(i, a) + g.dvo.at(a, i);
  }
  for (int a = 0; a < v; a++)
    for (int b = 0; b < v; b++)
      dm(o + a, o + b) = g.dvv.at(a, b) + g.dvv.at(b, a);
  return dm;
}

occ::Mat flatten(const Tn &l1, const Tn &l2) {
  occ::Mat x(l1.v.size() + l2.v.size(), 1);
  std::copy(l1.v.begin(), l1.v.end(), x.data());
  std::copy(l2.v.begin(), l2.v.end(), x.data() + l1.v.size());
  return x;
}

void unflatten(const occ::Mat &x, Tn &l1, Tn &l2) {
  std::copy(x.data(), x.data() + l1.v.size(), l1.v.begin());
  std::copy(x.data() + l1.v.size(), x.data() + x.size(), l2.v.begin());
}

// W + V/2 and r6(W) for every i of one (j, k), two re-laid-out ovvv, eight v^3
// buffers per thread
void check_triples_memory(const Eris &e, const double max_gb) {
  const double v3 = std::pow(double(e.v), 3);
  const double gb =
      8.0 * (4.0 * e.o * v3 + 8.0 * v3 * occ::parallel::get_num_threads()) /
      (1024.0 * 1024.0 * 1024.0);
  check(gb <= max_gb, "The triples terms of the density need about " +
                          std::to_string(gb) + " GB here, over the " +
                          std::to_string(max_gb) +
                          " GB budget (max_memory_gb)");
}

} // namespace

Gamma1 gamma1(const Tn &t1, const Tn &t2, const Tn &l1, const Tn &l2) {
  Gamma1 g;
  g.doo = einsum("ja,ia->ij", t1, l1) * -1.0;
  g.dvv = einsum("ia,ib->ab", t1, l1);
  g.dvo = T(t1, {1, 0}) - einsum("im,ma->ai", einsum("ie,me->im", t1, l1), t1);
  const Tn theta = t2 * 2.0 - T(t2, {0, 1, 3, 2});
  g.doo -= einsum("jkab,ikab->ij", theta, l2);
  g.dvv += einsum("jica,jicb->ab", theta, l2);
  g.dvo += einsum("imae,me->ai", theta, l1);
  g.dvo -= einsum("mi,ma->ai", einsum("mnef,inef->mi", l2, theta), t1);
  g.dvo -= einsum("ie,ae->ai", t1, einsum("mnaf,mnef->ea", l2, theta));
  return g;
}

CcSolution solve_cc(const occ::qm::cc::CCIntegrals &cc_eris,
                    const CorrelatedDensityOptions &opts,
                    const std::function<void(const char *)> &mark,
                    const occ::Mat *proj) {
  CcSolution s;
  occ::qm::cc::CCSDOptions copts;
  copts.max_cycle = opts.max_cycle;
  copts.residual_tol = opts.tol;
  const occ::qm::cc::CCSDResult cc = occ::qm::cc::ccsd(cc_eris, copts);
  check(cc.converged, "CCSD did not converge in " +
                          std::to_string(opts.max_cycle) + " iterations");
  s.e_corr = cc.e_corr;
  s.cc_iterations = cc.iterations;
  mark("CCSD amplitudes");
  const Eris e = convert(cc_eris);
  s.t1 = from_tensor(cc.t1);
  s.t2 = from_tensor(cc.t2);
  const Tn &t1 = s.t1, &t2 = s.t2;
  const Imds m = make_intermediates(t1, t2, e);
  mark("Lambda intermediates");
  if (opts.triples) {
    check_triples_memory(e, opts.max_memory_gb);
    s.tr = triples(t1, t2, e, proj);
    mark("triples terms");
  }
  s.l1 = t1;
  s.l2 = t2;
  Tn &l1 = s.l1, &l2 = s.l2;
  Tn l1new, l2new;
  occ::core::diis::DIIS diis(2, 8);
  bool converged = false;
  for (int it = 1; it <= opts.max_cycle && !converged; it++) {
    update_lambda(t1, t2, l1, l2, e, m, l1new, l2new);
    if (opts.triples) {
      l1new += s.tr.l1_t;
      l2new += s.tr.l2_t;
    }
    occ::Mat x = flatten(l1new, l2new);
    occ::Mat dl = x - flatten(l1, l2);
    // max |R|, R = D (l_new - l_old)
    double rmax = 0.0;
    size_t n = 0;
    for (int i = 0; i < e.o; i++)
      for (int a = 0; a < e.v; a++)
        rmax = std::max(rmax, std::abs(dl(n++) * (e.eo(i) - e.ev(a))));
    for (int i = 0; i < e.o; i++)
      for (int j = 0; j < e.o; j++)
        for (int a = 0; a < e.v; a++)
          for (int b = 0; b < e.v; b++)
            rmax = std::max(rmax, std::abs(dl(n++) * (e.eo(i) + e.eo(j) -
                                                      e.ev(a) - e.ev(b))));
    diis.extrapolate(x, dl);
    unflatten(x, l1, l2);
    s.lambda_iterations = it;
    converged = rmax < opts.tol;
  }
  mark("Lambda iterations");
  check(converged, "The CCSD Lambda equations did not converge in " +
                       std::to_string(opts.max_cycle) + " iterations");
  return s;
}

occ::qm::cc::CCIntegrals build_eris(const occ::gto::AOBasis &basis,
                                    const occ::qm::MolecularOrbitals &mo,
                                    const int n_frozen,
                                    const CorrelatedDensityOptions &opts,
                                    std::string &backend, std::string &aux_name,
                                    const bool with_ladder) {
  backend = occ::util::to_lower_copy(opts.backend);
  if (backend.empty() || backend == "auto" || backend == "conventional")
    backend = "exact";
  if (backend == "ri")
    backend = "df";
  check(backend == "exact" || backend == "df",
        "Correlated density backend '" + opts.backend +
            "' is not supported (exact | df)");
  const size_t budget =
      static_cast<size_t>(opts.max_memory_gb * 1024.0 * 1024.0 * 1024.0);
  if (backend == "exact")
    return occ::qm::cc::exact_eris(basis, mo, n_frozen, budget, with_ladder);
  aux_name = opts.aux_basis.empty()
                 ? occ::qm::resolve_fitting_basis(
                       basis.name(), occ::qm::FittingKind::Correlation)
                 : opts.aux_basis;
  auto aux = occ::gto::AOBasis::load(basis.atoms(), aux_name);
  aux.set_pure(basis.is_pure());
  return occ::qm::cc::df_eris(basis, aux, mo, n_frozen, budget);
}

void natural_orbitals(const occ::qm::Wavefunction &scf, const occ::Mat &dm,
                      const std::string &name, const double e_total,
                      CorrelatedDensity &result) {
  const int nmo = static_cast<int>(dm.rows());
  Eigen::SelfAdjointEigenSolver<occ::Mat> es(dm);
  check(es.info() == Eigen::Success,
        "CCSD density: the MO density could not be diagonalised");
  result.occupations = es.eigenvalues().reverse();
  const occ::Mat U = es.eigenvectors().rowwise().reverse();
  occ::qm::Wavefunction &nat = result.natural;
  nat = scf;
  nat.mo.C = scf.mo.C * U;
  nat.mo.occupation = 0.5 * result.occupations;
  nat.mo.energies =
      occ::Vec::Zero(nmo); // natural orbitals have occupations, not energies
  nat.mo.D = 0.5 * scf.mo.C * dm * scf.mo.C.transpose();
  nat.method = name;
  nat.energy.total = e_total;
}

} // namespace detail

using namespace detail;

CorrelatedDensity ccsd_density(const occ::qm::Wavefunction &scf,
                               const CorrelatedDensityOptions &opts) {
  check(scf.mo.kind == occ::qm::SpinorbitalKind::Restricted &&
            scf.mo.n_alpha == scf.mo.n_beta,
        "The CCSD density needs a closed-shell (restricted) reference; "
        "open-shell Lambda equations are not implemented");
  CorrelatedDensity result;
  auto clock = std::chrono::steady_clock::now();
  auto mark = [&clock](const char *stage) {
    const auto now = std::chrono::steady_clock::now();
    occ::log::info("CCSD density: {} took {:.3f} s", stage,
                   std::chrono::duration<double>(now - clock).count());
    clock = now;
  };
  const int nocc_total = static_cast<int>(scf.mo.n_alpha);
  const int nmo = static_cast<int>(scf.mo.C.cols());
  // as occ's CCSD driver: -1 is the chemical core, and at least one occupied
  // orbital stays correlated
  result.n_frozen = std::max(
      0, std::min(opts.n_frozen < 0 ? occ::qm::cc::num_frozen_core(scf.basis)
                                    : opts.n_frozen,
                  nocc_total - 1));
  std::string backend, aux_name;
  const occ::qm::cc::CCIntegrals cc_eris =
      build_eris(scf.basis, scf.mo, result.n_frozen, opts, backend, aux_name);
  mark("MO integrals");
  check(result.n_frozen + cc_eris.nocc + cc_eris.nvir == nmo,
        "CCSD density: the correlated space does not cover the MOs");
  const CcSolution s = solve_cc(cc_eris, opts, mark);
  result.e_corr = s.e_corr;
  result.e_triples = s.tr.e_t;
  result.lambda_iterations = s.lambda_iterations;
  const Tn &t1 = s.t1, &t2 = s.t2, &l1 = s.l1, &l2 = s.l2;
  const Triples &tr = s.tr;
  result.t1_norm = t1.norm();
  result.t2_norm = t2.norm();
  result.l1_norm = l1.norm();
  result.l2_norm = l2.norm();

  // frozen orbitals stay doubly occupied and uncoupled
  const int nf = result.n_frozen;
  occ::Mat dm = occ::Mat::Zero(nmo, nmo);
  for (int k = 0; k < nf; k++)
    dm(k, k) = 2.0;
  Gamma1 g = gamma1(t1, t2, l1, l2);
  if (opts.triples) {
    g.doo -= tr.goo * 0.5;
    g.dvv += tr.gvv * 0.5;
    g.dvo += tr.dvo;
  }
  dm.block(nf, nf, nmo - nf, nmo - nf) = rdm1(g, l1);
  const std::string name = opts.triples ? "CCSD(T)" : "CCSD";
  natural_orbitals(scf, dm, name,
                   scf.energy.total + s.e_corr + result.e_triples, result);
  mark("density and natural orbitals");
  result.n_occ = cc_eris.nocc;
  result.n_vir = cc_eris.nvir;
  result.cc_iterations = s.cc_iterations;
  result.backend = backend;
  result.aux_basis = aux_name;
  return result;
}

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
            nmo = static_cast<int>(scf.mo.C.cols());
  result.n_frozen = std::max(
      0, std::min(opts.n_frozen < 0 ? occ::qm::cc::num_frozen_core(scf.basis)
                                    : opts.n_frozen,
                  o - 1));
  const int nf = result.n_frozen;
  std::string backend, aux_name;
  // every occupied orbital: the orbital response couples the frozen core
  const Eris e =
      convert(build_eris(scf.basis, scf.mo, 0, opts, backend, aux_name, false));
  mark("MO integrals");
  const int v = e.v;
  // t_ij^ab = (ia|jb) / D_ij^ab on correlated pairs, tt = 2 t - t(ba); E2 = sum
  // tt (ia|jb)
  const Tn g = T(e.ovov, {0, 2, 1, 3});
  Tn t2 = g;
  size_t w = 0;
  for (int i = 0; i < o; i++)
    for (int j = 0; j < o; j++)
      for (int a = 0; a < v; a++)
        for (int b = 0; b < v; b++, w++)
          t2.v[w] = i < nf || j < nf
                        ? 0.0
                        : g.v[w] / (e.eo(i) + e.eo(j) - e.ev(a) - e.ev(b));
  const Tn tt = t2 * 2.0 - T(t2, {0, 1, 3, 2});
  result.e_corr = dot(tt, g);
  // PySCF grad/mp2.py: P_ij = -2 t_ki^ab tt_kj^ab, P_ab = 2 t_ij^ca tt_ij^cb;
  // Lagrangian I_kc = -2 (ik|jb) tt_ij^cb, I_dk = -2 (ad|jb) tt_kj^ab, I_mk =
  // -2 (ma|jb) tt_kj^ab; frozen-core block P_fk = I_fk / (e_f - e_k)
  const Tn p_vv = einsum("ijca,ijcb->ab", t2, tt) * 2.0;
  Tn p_oo = einsum("kiab,kjab->ij", t2, tt) * -2.0;
  const Tn i_oc = einsum("jbik,ijcb->kc", e.ovoo, tt) * -2.0;
  const Tn i_vo = einsum("jbad,kjab->dk", e.ovvv, tt) * -2.0;
  const Tn i_oo = einsum("majb,kjab->mk", e.ovov, tt) * -2.0;
  for (int f = 0; f < nf; f++)
    for (int k = nf; k < o; k++)
      p_oo.at(f, k) = p_oo.at(k, f) = i_oo.at(f, k) / (e.eo(f) - e.eo(k));
  // X_am = sum_pq P_pq [2 (am|pq) - (ap|qm)] + I_ma - I_am
  Tn x = einsum("majk,jk->am", e.ovoo, p_oo) * 2.0 -
         einsum("jakm,jk->am", e.ovoo, p_oo);
  x += einsum("mabc,bc->am", e.ovvv, p_vv) * 2.0 -
       einsum("mcab,bc->am", e.ovvv, p_vv);
  x += T(i_oc, {1, 0}) - i_vo;
  // Z-vector: (e_a - e_m) z_am + sum_bn [4 (am|bn) - (ab|mn) - (an|bm)] z_bn =
  // -X_am, conjugate gradients preconditioned by e_a - e_m (the RHF orbital
  // Hessian is positive definite)
  auto hessian = [&](const Tn &z) {
    Tn r = einsum("manb,bn->am", e.ovov, z) * 4.0 -
           einsum("mnab,bn->am", e.oovv, z) - einsum("namb,bn->am", e.ovov, z);
    for (int a = 0; a < v; a++)
      for (int m = 0; m < o; m++)
        r.at(a, m) += (e.ev(a) - e.eo(m)) * z.at(a, m);
    return r;
  };
  auto precondition = [&](const Tn &r) {
    Tn s = r;
    for (int a = 0; a < v; a++)
      for (int m = 0; m < o; m++)
        s.at(a, m) /= e.ev(a) - e.eo(m);
    return s;
  };
  Tn z = precondition(x * -1.0), res = x * -1.0 - hessian(z);
  Tn s = precondition(res), dir = s;
  double rs = dot(res, s);
  int it = 0;
  const auto max_abs = [](const Tn &a) {
    double m = 0.0;
    for (size_t k = 0; k < a.v.size(); k++)
      m = std::max(m, std::abs(a.v[k]));
    return m;
  };
  for (; it < opts.max_cycle && max_abs(res) > 1e-10; it++) {
    const Tn hd = hessian(dir);
    const double alpha = rs / dot(dir, hd);
    z += dir * alpha;
    res -= hd * alpha;
    s = precondition(res);
    const double rs_new = dot(res, s);
    dir = s + dir * (rs_new / rs);
    rs = rs_new;
  }
  check(max_abs(res) <= 1e-10,
        "The MP2 Z-vector equations did not converge in " +
            std::to_string(opts.max_cycle) + " iterations");
  result.lambda_iterations = it;
  mark("Z-vector");
  occ::Mat dm = occ::Mat::Zero(nmo, nmo);
  for (int i = 0; i < o; i++) {
    dm(i, i) = 2.0;
    for (int j = 0; j < o; j++)
      dm(i, j) += p_oo.at(i, j);
    for (int a = 0; a < v; a++)
      dm(o + a, i) = dm(i, o + a) = z.at(a, i);
  }
  for (int a = 0; a < v; a++)
    for (int b = 0; b < v; b++)
      dm(o + a, o + b) = p_vv.at(a, b);
  natural_orbitals(scf, dm, "MP2", scf.energy.total + result.e_corr, result);
  mark("density and natural orbitals");
  result.n_occ = o - nf;
  result.n_vir = v;
  result.backend = backend;
  result.aux_basis = aux_name;
  return result;
}

CorrelatedDensity mpn_density(const occ::qm::Wavefunction &scf, const int order,
                              const CorrelatedDensityOptions &opts) {
  check(order == 3 || order == 4, "The MPn density is for order 3 or 4");
  check(scf.mo.kind == occ::qm::SpinorbitalKind::Restricted &&
            scf.mo.n_alpha == scf.mo.n_beta,
        "The MPn density needs a closed-shell (restricted) reference");
  const bool with_t = order == 4 && opts.triples;
  const std::string name = order == 3 ? "MP3" : (with_t ? "MP4" : "MP4(SDQ)");
  CorrelatedDensity result;
  auto clock = std::chrono::steady_clock::now();
  auto mark = [&clock, &name](const char *stage) {
    const auto now = std::chrono::steady_clock::now();
    occ::log::info("{} density: {} took {:.3f} s", name, stage,
                   std::chrono::duration<double>(now - clock).count());
    clock = now;
  };
  const int nocc_total = static_cast<int>(scf.mo.n_alpha);
  const int nmo = static_cast<int>(scf.mo.C.cols());
  result.n_frozen = std::max(
      0, std::min(opts.n_frozen < 0 ? occ::qm::cc::num_frozen_core(scf.basis)
                                    : opts.n_frozen,
                  nocc_total - 1));
  std::string backend, aux_name;
  const occ::qm::cc::CCIntegrals cc_eris =
      build_eris(scf.basis, scf.mo, result.n_frozen, opts, backend, aux_name);
  mark("MO integrals");
  check(result.n_frozen + cc_eris.nocc + cc_eris.nvir == nmo,
        "MPn density: the correlated space does not cover the MOs");
  const occ::qm::cc::MPnAmplitudes a = occ::qm::cc::mpn_amplitudes(cc_eris);
  mark("amplitudes");
  const Eris e = convert(cc_eris);
  const Tn t = from_tensor(a.t), u = from_tensor(a.u);
  const Tn zero1({e.o, e.v});
  // the Lagrangian's multiplier of D t = g is -l2 below; gamma1 with (t, l2)
  // gives its share of the density
  Gamma1 g;
  if (order == 3) {
    // E3's gradient is 2 L(t), L self-adjoint
    g = gamma1(zero1, t, zero1, t + u * 2.0);
    result.e_corr = a.energies.mp3();
  } else {
    const Tn s = from_tensor(a.s), q = from_tensor(a.q);
    // the Lambda update at t1 = 0 is t plus adjoint terms over D: (0, s, u)
    // carries S^+(s) and L(u), (t, 0, t) L(t) and Q'(t)^+ t
    auto lambda = [&](const Tn &t2, const Tn &l1, const Tn &l2) {
      Tn l1new, l2new;
      update_lambda(zero1, t2, l1, l2, e, make_intermediates(zero1, t2, e),
                    l1new, l2new);
      return l2new;
    };
    Tn l2 = u - t * 2.0 + lambda(Tn({e.o, e.o, e.v, e.v}), s, u) * 2.0 + q +
            lambda(t, zero1, t);
    mark("Lagrange multipliers");
    Triples tr;
    if (with_t) {
      check_triples_memory(e, opts.max_memory_gb);
      tr = triples(zero1, t, e, nullptr, &u);
      l2 += tr.l2_t;
      mark("triples terms");
    }
    g = gamma1(zero1, t, zero1, l2);
    const Gamma1 gu = gamma1(zero1, u, zero1, u);
    g.doo += gu.doo - einsum("ia,ja->ij", s, s);
    g.dvv += gu.dvv + einsum("ia,ib->ab", s, s);
    // ov: the singles-doubles coupling of the second-order wavefunction
    g.dvo += einsum("imae,me->ai", u * 2.0 - T(u, {0, 1, 3, 2}), s);
    if (with_t) {
      g.doo -= tr.goo * 0.5;
      g.dvv += tr.gvv * 0.5;
      g.dvo += tr.dvo;
    }
    result.e_triples = tr.e_t;
    result.e_corr = a.energies.mp4_sdq() + tr.e_t;
  }
  const int nf = result.n_frozen;
  occ::Mat dm = occ::Mat::Zero(nmo, nmo);
  for (int k = 0; k < nf; k++)
    dm(k, k) = 2.0;
  dm.block(nf, nf, nmo - nf, nmo - nf) = rdm1(g, zero1);
  natural_orbitals(scf, dm, name, scf.energy.total + result.e_corr, result);
  mark("density and natural orbitals");
  occ::log::info("{} correlation energy {:.12f}", name, result.e_corr);
  result.n_occ = cc_eris.nocc;
  result.n_vir = cc_eris.nvir;
  result.t2_norm = t.norm();
  result.backend = backend;
  result.aux_basis = aux_name;
  return result;
}

} // namespace occ::qm::cc
