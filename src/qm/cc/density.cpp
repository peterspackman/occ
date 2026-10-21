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

using vec = std::vector<double>;
using RowMat =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapR = Eigen::Map<RowMat>;
using CMapR = Eigen::Map<const RowMat>;

// C = alpha op(A) op(B) + beta C on row-major arrays with leading dimensions
// (cblas_dgemm's row-major form), serial: the (T) kernels' per-triple products,
// already inside a parallel loop
void gemm_rm(const bool ta, const bool tb, const int m, const int n,
             const int k, const double alpha, const double *A, const int lda,
             const double *B, const int ldb, const double beta, double *C,
             const int ldc) {
  using Stride = Eigen::OuterStride<>;
  Eigen::Map<RowMat, 0, Stride> Cm(C, m, n, Stride(ldc));
  const Eigen::Map<const RowMat, 0, Stride> Am(A, ta ? k : m, ta ? m : k,
                                               Stride(lda)),
      Bm(B, tb ? n : k, tb ? k : n, Stride(ldb));
  if (beta == 0.0)
    Cm.setZero();
  else if (beta != 1.0)
    Cm *= beta;
  if (!ta && !tb)
    Cm.noalias() += alpha * Am * Bm;
  else if (ta && !tb)
    Cm.noalias() += alpha * Am.transpose() * Bm;
  else if (!ta && tb)
    Cm.noalias() += alpha * Am * Bm.transpose();
  else
    Cm.noalias() += alpha * Am.transpose() * Bm.transpose();
}

// occ's integral blocks under the names the equations use (references, no
// copies), the Fock blocks and the orbital energies
struct Eris {
  explicit Eris(const CCIntegrals &e)
      : o(e.nocc), v(e.nvir), eo(e.mo_energy.head(e.nocc)),
        ev(e.mo_energy.tail(e.nvir)), oooo(e.oooo), ovoo(e.ovoo), oovv(e.oovv),
        ovov(e.ovov), ovvo(e.ovvo), ovvv(e.ovvv), ladder(e.ladder) {
    const occ::Mat f =
        e.fock.rows() == o + v ? e.fock : occ::Mat(e.mo_energy.asDiagonal());
    foo = as_tensor(f.topLeftCorner(o, o));
    fov = as_tensor(f.topRightCorner(o, v));
    fvv = as_tensor(f.bottomRightCorner(v, v));
  }
  int o, v;
  occ::Vec eo, ev;
  T2 foo, fov, fvv;
  const T4 &oooo, &ovoo, &oovv, &ovov, &ovvo, &ovvv;
  const std::function<T4(const T4 &)> &ladder;
};

struct Imds {
  T2 w1, w2, w3, w4;
  T4 woooo, wvooo, wVOov, wvOOv, wvvov;
};

// PySCF ccsd_lambda.make_intermediates, unblocked
Imds make_intermediates(const T2 &t1, const T4 &t2, const Eris &e) {
  const T4 &ovvv = e.ovvv, &ovoo = e.ovoo, &ovvo = e.ovvo, &oovv = e.oovv,
           &oooo = e.oooo;
  const T2 &fov = e.fov;
  T2 w1 = e.fvv - einsum<2>("ja,jb->ba", fov, t1);
  T2 w2 = e.foo + einsum<2>("ib,jb->ij", fov, t1);
  T2 w3 = einsum<2>("kc,jkbc->bj", fov, t2) * 2.0 + transpose(fov, {1, 0});
  w3 -= einsum<2>("kc,kjbc->bj", fov, t2);
  w3 += einsum<2>("cb,jc->bj", einsum<2>("kc,kb->cb", fov, t1), t1);
  T2 w4 = fov;
  const T4 vvov = transpose(ovvv, {2, 3, 0, 1});
  w1 += einsum<2>("jcba,jc->ba", ovvv, t1 * 2.0);
  w1 -= einsum<2>("jabc,jc->ba", ovvv, t1);
  T4 theta = t2 * 2.0 - transpose(t2, {1, 0, 2, 3});
  w3 += einsum<2>("jkcd,kdcb->bj", theta, ovvv);
  T4 wVOov = einsum<4>("jbcd,kd->bjkc", ovvv, t1);
  T4 wvOOv = einsum<4>("cbjd,kd->cjkb", vvov, t1 * -1.0);
  const T4 tau = t2 + einsum<4>("ia,jb->ijab", t1, t1);
  T4 wvooo = einsum<4>(
      "cibd,jkbd->ckij",
      transpose(vvov, {0, 2, 1, 3}) * 2.0 - transpose(vvov, {0, 2, 3, 1}), tau);

  T4 wvvov = einsum<4>("jabd,jkcd->abkc", ovvv, t2) * -1.5;
  wvvov += transpose(vvov, {0, 3, 2, 1}) * 2.0;
  wvvov -= vvov;
  {
    const T4 vackb =
        einsum<4>("acjd,kjbd->ackb", vvov * 2.0 - transpose(ovvv, {1, 2, 0, 3}),
                  t2 * 2.0 - transpose(t2, {0, 1, 3, 2}));
    wvvov += transpose(vackb, {0, 3, 2, 1});
    wvvov -= vackb * 0.5;
  }

  w2 += einsum<2>("kbij,kb->ij", ovoo, t1) * 2.0;
  w2 -= einsum<2>("ibkj,kb->ij", ovoo, t1);
  theta = transpose(t2, {1, 0, 2, 3}) * 2.0 - t2;
  w3 -= einsum<2>("lckj,klcb->bj", ovoo, theta);

  T4 woooo = einsum<4>("lc,jcik->ijkl", t1, ovoo);
  woooo += transpose(woooo, {1, 0, 3, 2});
  wvOOv += einsum<4>("lbjk,lc->bjkc", ovoo, t1);
  wVOov -= einsum<4>("jbkl,lc->bjkc", ovoo, t1);
  wvooo += transpose(ovoo, {1, 3, 2, 0}) * 2.0;
  wvooo -= transpose(ovoo, {1, 0, 2, 3});
  wvooo -= einsum<4>("klbc,iblj->ckij", t2, ovoo * 1.5);
  const T4 g2ovoo = ovoo * 2.0 - transpose(ovoo, {2, 1, 0, 3});
  {
    const T4 vcjik = einsum<4>("jlcb,lbki->cjki",
                               t2 * 2.0 - transpose(t2, {1, 0, 2, 3}), g2ovoo);
    wvooo += transpose(vcjik, {0, 3, 2, 1});
    wvooo -= vcjik * 0.5;
  }

  const T4 voov = transpose(ovvo, {1, 0, 3, 2});
  woooo += einsum<4>("cijd,klcd->ijkl", voov, tau);
  const T4 g2voov = voov * 2.0 - transpose(voov, {0, 2, 1, 3});
  const T2 tmpw4 = einsum<2>("ckld,ld->kc", g2voov, t1);
  w1 -= einsum<2>("ckja,kjcb->ba", g2voov, t2);
  w1 -= einsum<2>("ja,jb->ba", tmpw4, t1);
  w2 += einsum<2>("jkbc,bikc->ij", t2, g2voov);
  w2 += einsum<2>("ib,jb->ij", tmpw4, t1);
  w3 += einsum<2>("bc,jc->bj", einsum<2>("kb,kc->bc", t1, tmpw4), t1);
  w4 += tmpw4;
  wvOOv += einsum<4>("bljk,lc->bjkc", einsum<4>("bljd,kd->bljk", voov, t1), t1);
  wVOov -= einsum<4>("bjlk,lc->bjkc", einsum<4>("bjld,kd->bjlk", voov, t1), t1);
  T4 VOov = einsum<4>("bjld,klcd->bjkc", g2voov, t2);
  VOov -= einsum<4>("bjld,kldc->bjkc", voov, t2);
  VOov += voov;
  T4 vOOv = einsum<4>("bljd,kldc->bjkc", voov, t2);
  vOOv -= transpose(oovv, {2, 1, 0, 3});
  wVOov += VOov;
  wvOOv += vOOv;
  const T4 ov1 = vOOv * 2.0 + VOov, ov2 = VOov * 2.0 + vOOv;
  wvooo -= einsum<4>("jb,bikc->ckij", t1, ov1);
  wvooo += einsum<4>("kb,bijc->ckij", t1, ov2);
  w3 += einsum<2>("ckjb,kc->bj", ov2, t1);
  wvvov += einsum<4>("ajkc,jb->abkc", ov1, t1);
  wvvov -= einsum<4>("ajkb,jc->abkc", ov2, t1);
  wvvov += einsum<4>("laki,klbc->abic", g2ovoo, tau);
  woooo += transpose(oooo, {0, 2, 1, 3});
  w3 += einsum<2>("bc,jc->bj", w1, t1);
  w3 -= einsum<2>("kj,kb->bj", w2, t1);
  return {w1, w2, w3, w4, woooo, wvooo, wVOov, wvOOv, wvvov};
}

// PySCF ccsd_lambda.update_lambda, unblocked: the next (l1, l2) from the
// current ones
void update_lambda(const T2 &t1, const T4 &t2, const T2 &l1, const T4 &l2,
                   const Eris &e, const Imds &m, T2 &l1new, T4 &l2new) {
  const int o = e.o, v = e.v;
  const T4 theta = t2 * 2.0 - transpose(t2, {0, 1, 3, 2});
  const T2 mba = einsum<2>("klca,klcb->ba", l2, theta);
  const T2 mij = einsum<2>("ikcd,jkcd->ij", l2, theta);
  const T2 mba1 = einsum<2>("jc,jb->bc", l1, t1) + mba;
  const T2 mij1 = einsum<2>("kb,jb->kj", l1, t1) + mij;
  T2 mia1 = t1 + einsum<2>("kc,jkbc->jb", l1, t2) * 2.0;
  mia1 -= einsum<2>("kc,jkcb->jb", l1, t2);
  mia1 -= einsum<2>("jk,kb->jb", einsum<2>("ja,ka->jk", t1, l1), t1);
  mia1 -= einsum<2>("bd,jd->jb", mba, t1);
  mia1 -= einsum<2>("lj,lb->jb", mij, t1);
  l2new = e.ladder(l2); // sum_cd (ac|bd) l2(i,j,c,d)
  l1new = einsum<2>("ijab,jb->ia", l2new, t1) * 2.0;
  l1new -= einsum<2>("jiab,jb->ia", l2new, t1);
  l2new = l2new * 0.5; // the (ia)<->(jb) symmetrisation doubles it back

  T2 w1 = m.w1, w2 = m.w2;
  for (int a = 0; a < v; a++)
    w1(a, a) -= e.ev(a);
  for (int i = 0; i < o; i++)
    w2(i, i) -= e.eo(i);
  l1new += e.fov;
  l1new += einsum<2>("ib,ba->ia", l1, w1);
  l1new -= einsum<2>("ja,ij->ia", l1, w2);
  l1new -= einsum<2>("ik,ka->ia", mij, m.w4);
  l1new -= einsum<2>("ca,ic->ia", mba, m.w4);
  l1new += einsum<2>("ijab,bj->ia", l2, m.w3) * 2.0;
  l1new -= einsum<2>("ijba,bj->ia", l2, m.w3);
  l2new += einsum<4>("ia,jb->ijab", l1, m.w4);
  l2new += einsum<4>("jibc,ca->jiba", l2, w1);
  l2new -= einsum<4>("jk,kiba->jiba", w2, l2);
  l1new -= einsum<2>("iajk,kj->ia", e.ovoo, mij1) * 2.0;
  l1new += einsum<2>("jaik,kj->ia", e.ovoo, mij1);
  l2new -= einsum<4>("jbki,ka->jiba", e.ovoo, l1);
  const T4 tau = t2 + einsum<4>("ia,jb->ijab", t1, t1);
  const T4 l2tau = einsum<4>("ijcd,klcd->ijkl", l2, tau);
  const T4 l2t1 = einsum<4>("jidc,kc->ijkd", l2, t1);
  l1new -= einsum<2>("jb,jiab->ia", l1, e.oovv);
  l1new += einsum<2>("iabc,bc->ia", e.ovvv, mba1) * 2.0;
  l1new -= einsum<2>("ibca,bc->ia", e.ovvv, mba1);
  l2new += einsum<4>("jbac,ic->jiba", e.ovvv, l1);
  {
    const T4 m4 = einsum<4>("ijkd,kadb->ijab", l2t1, e.ovvv);
    l2new -= m4;
    l1new -= einsum<2>("ijab,jb->ia", m4, t1) * 2.0;
    l1new -= einsum<2>("ijab,ia->jb", m4, t1) * 2.0;
    l1new += einsum<2>("jiab,jb->ia", m4, t1);
    l1new += einsum<2>("jiab,ia->jb", m4, t1);
  }

  const T4 voov = transpose(e.ovvo, {1, 0, 3, 2});
  l1new += einsum<2>("jb,aijb->ia", l1, voov) * 2.0;
  l2new += transpose(voov, {1, 2, 0, 3}) * 0.5;
  l2new -= einsum<4>("bjic,ca->jiba", voov, mba1);
  l2new -= einsum<4>("bjka,ik->jiba", voov, mij1);
  l1new += einsum<2>("aijb,jb->ia", voov, mia1) * 2.0;
  l1new -= einsum<2>("bija,jb->ia", voov, mia1);
  {
    const T4 m4 = einsum<4>("ijkl,aklb->ijab", l2tau, voov);
    l2new += m4 * 0.5;
    l1new += einsum<2>("ijab,jb->ia", m4, t1) * 2.0;
    l1new -= einsum<2>("ijba,jb->ia", m4, t1);
  }

  l1new -= einsum<2>("ckij,jkca->ia", m.wvooo, l2);
  l1new += einsum<2>("abkc,kibc->ia", m.wvvov, l2);
  l2new +=
      einsum<4>("iakc,bjkc->jiba",
                transpose(l2, {0, 2, 1, 3}) - transpose(l2, {0, 3, 1, 2}) * 0.5,
                m.wVOov * 2.0 + m.wvOOv);
  {
    const T4 tmp = einsum<4>("jkca,bikc->jiba", l2, m.wvOOv);
    l2new += tmp;
    l2new += transpose(tmp, {1, 0, 2, 3}) * 0.5;
  }
  {
    const T4 m3 = einsum<4>("ijkl,klab->ijab", m.woooo, l2);
    l2new += m3 * 0.5;
    l1new += einsum<2>("ijab,jb->ia", m3, t1) * 2.0;
    l1new -= einsum<2>("ijba,jb->ia", m3, t1);
  }

  for (int a = 0; a < v; a++)
    for (int i = 0; i < o; i++)
      l1new(i, a) /= e.eo(i) - e.ev(a);
  l2new += transpose(l2new, {1, 0, 3, 2});
  for (int b = 0; b < v; b++)
    for (int a = 0; a < v; a++)
      for (int j = 0; j < o; j++)
        for (int i = 0; i < o; i++)
          l2new(i, j, a, b) /= e.eo(i) - e.ev(a) + e.eo(j) - e.ev(b);
}

// The (T) kernels index v^3 blocks by raw offsets, PySCF-style, in row-major
// order: row_major(x, perm) is a tensor whose memory is the row-major layout of
// transpose(x, perm) (a row-major [i0]..[in] array is the column-major one
// with its axes reversed), and the outputs are accumulated the same way and
// turned back at the end.
template <int R>
Eigen::Tensor<double, R> row_major(const Eigen::Tensor<double, R> &x,
                                   std::array<int, R> perm) {
  std::reverse(perm.begin(), perm.end());
  return transpose(x, perm);
}

template <int R>
Eigen::Tensor<double, R> zeros(Eigen::array<Eigen::Index, R> d) {
  Eigen::Tensor<double, R> t(d);
  t.setZero();
  return t;
}

// t2_vo: the doubles r6(W) is contracted with for the vo block (t2 if null)
Triples triples(const T2 &t1_in, const T4 &t2, const Eris &e,
                const occ::Mat *proj = nullptr, const T4 *t2_vo = nullptr) {
  const int o = e.o, v = e.v;
  const size_t vv = size_t(v) * v, v3 = vv * v;
  const T4 L1 = row_major(e.ovvv, {0, 1, 3, 2}); //[i][a][b][f] = (ia|fb)
  const T4 L2 = row_major(e.ovvv, {0, 3, 1, 2}); //[k][e][f][b] = (kf|be)
  const T4 t2T = row_major(t2, {0, 1, 3, 2});    //[k][j][f][c] = t2(kj,cf)
  const T4 t2_t = row_major(t2, {1, 0, 2, 3});   //[k][m][b][c] = t2(mk,bc)
  const T4 ovoo_t = row_major(e.ovoo, {0, 2, 1, 3}),
           ovov_t =
               row_major(e.ovov, {0, 2, 1, 3}); //[i][j][a][m], [i][j][a][b]
  const T4 tvo = row_major(t2_vo ? *t2_vo : t2, {0, 1, 2, 3});
  const T2 t1 = row_major(t1_in, {0, 1});
  auto at = [](const auto &a, const size_t block, const size_t size) {
    return a.data() + block * size;
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
  // row-major accumulators (see row_major): l1 (o, v), goo (o, o), gvv and its
  // projected share (v, v), joovv (o, o, v, v); dvo is indexed logically
  T2 l1_rm = zeros<2>({v, o}), goo_rm = zeros<2>({o, o}),
     gvv_c_rm = zeros<2>({v, v});
  T4 joovv_rm = zeros<4>({v, v, o, o});
  std::vector<T2> gvv_i(o, zeros<2>({v, v})), dvo_i(o, zeros<2>({v, o}));
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
    MapR(l1_rm.data() + size_t(i) * v, 1, v).noalias() +=
        weight *
        (CMapR(rw, v, vv) * CMapR(at(ovov_t, size_t(j) * o + k, vv), vv, 1))
            .transpose();
    gemm(false, true, iv, iv, ivv, weight, wv, ivv, rw, ivv, 1.0,
         gvv_i[i].data(), iv);
    // joovv[i, j] += m L2[k]; joovv[i, J, ab] -= sum_c (kc|jJ) m[ab, c]; the
    // mirror with j <-> k and m'
    gemm(false, false, iv, iv, ivv, 1.0, m, ivv, at(L2, k, v3), iv, 1.0,
         joovv_rm.data() + (size_t(i) * o + j) * vv, iv);
    gemm(true, true, o, ivv, iv, -1.0,
         at(ovoo_t, size_t(k) * o + j, size_t(v) * o), o, m, iv, 1.0,
         joovv_rm.data() + size_t(i) * o * vv, ivv);
    const Eigen::VectorXd dv = CMapR(rw, vv, v).transpose() *
                               CMapR(at(tvo, size_t(i) * o + j, vv), vv, 1);
    for (int c = 0; c < v; c++)
      dvo_i[i](c, k) += 0.5 * dv(c);
    if (!mirror)
      return;
    gemm(false, false, iv, iv, ivv, 1.0, mm, ivv, at(L2, j, v3), iv, 1.0,
         joovv_rm.data() + (size_t(i) * o + k) * vv, iv);
    gemm(true, true, o, ivv, iv, -1.0,
         at(ovoo_t, size_t(j) * o + k, size_t(v) * o), o, mm, iv, 1.0,
         joovv_rm.data() + size_t(i) * o * vv, ivv);
    // the mirror's dv(c) = sum_ab r6(W)[acb] t2(ik, ab)
    Eigen::VectorXd dvm = Eigen::VectorXd::Zero(v);
    for (int a = 0; a < v; a++)
      dvm.noalias() +=
          CMapR(rw + size_t(a) * vv, v, v) *
          CMapR(at(tvo, size_t(i) * o + k, vv) + size_t(a) * v, v, 1);
    for (int c = 0; c < v; c++)
      dvo_i[i](c, j) += 0.5 * dvm(c);
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
               1.0, gvv_c_rm.data(), iv);
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
        MapR goo(goo_rm.data(), o, o);
        for (size_t b = 0; b < nb; b++)
          goo += (j == k ? 1.0 : 2.0) * part[b];
      }
    }
  T2 gvv_rm = zeros<2>({v, v});
  r.dvo = zeros<2>({v, o});
  for (int i = 0; i < o; i++) {
    gvv_rm += gvv_i[i];
    r.dvo += dvo_i[i];
    r.e_t += e_i[i];
  }
  r.gvv = transpose(gvv_rm, {1, 0});
  r.goo = transpose(goo_rm, {1, 0});
  if (proj)
    r.gvv_c = transpose(gvv_c_rm, {1, 0});
  r.l1_t = transpose(l1_rm, {1, 0});
  for (int a = 0; a < v; a++)
    for (int i = 0; i < o; i++)
      r.l1_t(i, a) *= 0.5 / (e.eo(i) - e.ev(a));
  const T4 joovv = transpose(joovv_rm, {3, 2, 1, 0});
  r.l2_t = joovv + transpose(joovv, {1, 0, 3, 2});
  for (int b = 0; b < v; b++)
    for (int a = 0; a < v; a++)
      for (int j = 0; j < o; j++)
        for (int i = 0; i < o; i++)
          r.l2_t(i, j, a, b) /= e.eo(i) - e.ev(a) + e.eo(j) - e.ev(b);
  return r;
}

// PySCF ccsd_rdm._make_rdm1: the spin-traced density in the correlated MOs,
// reference occupation included; symmetric, so no question of <p+q> against
// <q+p>
occ::Mat rdm1(const Gamma1 &g, const T2 &l1) {
  const int o = static_cast<int>(l1.dimension(0)),
            v = static_cast<int>(l1.dimension(1));
  const occ::Mat doo = as_matrix(g.doo), dvv = as_matrix(g.dvv);
  occ::Mat dm = occ::Mat::Zero(o + v, o + v);
  dm.topLeftCorner(o, o) = doo + doo.transpose();
  dm.topLeftCorner(o, o).diagonal().array() += 2.0;
  dm.topRightCorner(o, v) = as_matrix(l1) + as_matrix(g.dvo).transpose();
  dm.bottomLeftCorner(v, o) = dm.topRightCorner(o, v).transpose();
  dm.bottomRightCorner(v, v) = dvv + dvv.transpose();
  return dm;
}

// (l1, l2) as one vector for DIIS, and back
occ::Mat flatten(const T2 &l1, const T4 &l2) {
  occ::Mat x(l1.size() + l2.size(), 1);
  std::copy_n(l1.data(), l1.size(), x.data());
  std::copy_n(l2.data(), l2.size(), x.data() + l1.size());
  return x;
}

void unflatten(const occ::Mat &x, T2 &l1, T4 &l2) {
  std::copy_n(x.data(), l1.size(), l1.data());
  std::copy_n(x.data() + l1.size(), l2.size(), l2.data());
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

Gamma1 gamma1(const T2 &t1, const T4 &t2, const T2 &l1, const T4 &l2) {
  Gamma1 g;
  g.doo = einsum<2>("ja,ia->ij", t1, l1) * -1.0;
  g.dvv = einsum<2>("ia,ib->ab", t1, l1);
  g.dvo = transpose(t1, {1, 0}) -
          einsum<2>("im,ma->ai", einsum<2>("ie,me->im", t1, l1), t1);
  const T4 theta = t2 * 2.0 - transpose(t2, {0, 1, 3, 2});
  g.doo -= einsum<2>("jkab,ikab->ij", theta, l2);
  g.dvv += einsum<2>("jica,jicb->ab", theta, l2);
  g.dvo += einsum<2>("imae,me->ai", theta, l1);
  g.dvo -= einsum<2>("mi,ma->ai", einsum<2>("mnef,inef->mi", l2, theta), t1);
  g.dvo -= einsum<2>("ie,ae->ai", t1, einsum<2>("mnaf,mnef->ea", l2, theta));
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
  const Eris e(cc_eris);
  s.t1 = cc.t1;
  s.t2 = cc.t2;
  const T2 &t1 = s.t1;
  const T4 &t2 = s.t2;
  const Imds m = make_intermediates(t1, t2, e);
  mark("Lambda intermediates");
  if (opts.triples) {
    check_triples_memory(e, opts.max_memory_gb);
    s.tr = triples(t1, t2, e, proj);
    mark("triples terms");
  }
  s.l1 = t1;
  s.l2 = t2;
  T2 &l1 = s.l1, l1new;
  T4 &l2 = s.l2, l2new;
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
    // max |R|, R = D (l_new - l_old), in flatten's (column-major) order
    double rmax = 0.0;
    Eigen::Index n = 0;
    for (int a = 0; a < e.v; a++)
      for (int i = 0; i < e.o; i++)
        rmax = std::max(rmax, std::abs(dl(n++) * (e.eo(i) - e.ev(a))));
    for (int b = 0; b < e.v; b++)
      for (int a = 0; a < e.v; a++)
        for (int j = 0; j < e.o; j++)
          for (int i = 0; i < e.o; i++)
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

std::string resolve_backend(const CorrelatedDensityOptions &opts) {
  std::string backend = occ::util::to_lower_copy(opts.backend);
  if (backend.empty() || backend == "auto" || backend == "conventional")
    backend = "exact";
  if (backend == "ri")
    backend = "df";
  check(backend == "exact" || backend == "df",
        "Correlated density backend '" + opts.backend +
            "' is not supported (exact | df)");
  return backend;
}

occ::gto::AOBasis load_aux_basis(const occ::gto::AOBasis &basis,
                                 const CorrelatedDensityOptions &opts,
                                 std::string &aux_name) {
  aux_name = opts.aux_basis.empty()
                 ? occ::qm::resolve_fitting_basis(
                       basis.name(), occ::qm::FittingKind::Correlation)
                 : opts.aux_basis;
  auto aux = occ::gto::AOBasis::load(basis.atoms(), aux_name);
  aux.set_pure(basis.is_pure());
  return aux;
}

int frozen_core_count(const occ::qm::Wavefunction &scf, const int n_frozen) {
  const int nocc = static_cast<int>(scf.mo.n_alpha);
  return std::max(0, std::min(n_frozen < 0
                                  ? occ::qm::cc::num_frozen_core(scf.basis)
                                  : n_frozen,
                              nocc - 1));
}

size_t memory_budget(const CorrelatedDensityOptions &opts) {
  return static_cast<size_t>(opts.max_memory_gb * 1024.0 * 1024.0 * 1024.0);
}

occ::qm::cc::CCIntegrals build_eris(const occ::gto::AOBasis &basis,
                                    const occ::qm::MolecularOrbitals &mo,
                                    const int n_frozen,
                                    const CorrelatedDensityOptions &opts,
                                    std::string &backend, std::string &aux_name,
                                    const bool with_ladder) {
  backend = resolve_backend(opts);
  if (backend == "exact")
    return occ::qm::cc::exact_eris(basis, mo, n_frozen, memory_budget(opts),
                                   with_ladder);
  const auto aux = load_aux_basis(basis, opts, aux_name);
  return occ::qm::cc::df_eris(basis, aux, mo, n_frozen, memory_budget(opts));
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
  const int nmo = static_cast<int>(scf.mo.C.cols());
  result.n_frozen = frozen_core_count(scf, opts.n_frozen);
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
  const T2 &t1 = s.t1, &l1 = s.l1;
  const T4 &t2 = s.t2, &l2 = s.l2;
  const Triples &tr = s.tr;
  result.t1_norm = norm(t1);
  result.t2_norm = norm(t2);
  result.l1_norm = norm(l1);
  result.l2_norm = norm(l2);

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
  const int nmo = static_cast<int>(scf.mo.C.cols());
  result.n_frozen = frozen_core_count(scf, opts.n_frozen);
  std::string backend, aux_name;
  const occ::qm::cc::CCIntegrals cc_eris =
      build_eris(scf.basis, scf.mo, result.n_frozen, opts, backend, aux_name);
  mark("MO integrals");
  check(result.n_frozen + cc_eris.nocc + cc_eris.nvir == nmo,
        "MPn density: the correlated space does not cover the MOs");
  const occ::qm::cc::MPnAmplitudes a = occ::qm::cc::mpn_amplitudes(cc_eris);
  mark("amplitudes");
  const Eris e(cc_eris);
  const T4 &t = a.t, &u = a.u;
  const T2 zero1 = zeros<2>({e.o, e.v});
  // the Lagrangian's multiplier of D t = g is -l2 below; gamma1 with (t, l2)
  // gives its share of the density
  Gamma1 g;
  if (order == 3) {
    // E3's gradient is 2 L(t), L self-adjoint
    g = gamma1(zero1, t, zero1, t + u * 2.0);
    result.e_corr = a.energies.mp3();
  } else {
    const T2 &s = a.s;
    const T4 &q = a.q;
    // the Lambda update at t1 = 0 is t plus adjoint terms over D: (0, s, u)
    // carries S^+(s) and L(u), (t, 0, t) L(t) and Q'(t)^+ t
    auto lambda = [&](const T4 &t2, const T2 &l1, const T4 &l2) {
      T2 l1new;
      T4 l2new;
      update_lambda(zero1, t2, l1, l2, e, make_intermediates(zero1, t2, e),
                    l1new, l2new);
      return l2new;
    };
    T4 l2 = u - t * 2.0 + lambda(zeros<4>({e.o, e.o, e.v, e.v}), s, u) * 2.0 +
            q + lambda(t, zero1, t);
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
    g.doo += gu.doo - einsum<2>("ia,ja->ij", s, s);
    g.dvv += gu.dvv + einsum<2>("ia,ib->ab", s, s);
    // ov: the singles-doubles coupling of the second-order wavefunction
    g.dvo += einsum<2>("imae,me->ai", u * 2.0 - transpose(u, {0, 1, 3, 2}), s);
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
  result.t2_norm = norm(t);
  result.backend = backend;
  result.aux_basis = aux_name;
  return result;
}

} // namespace occ::qm::cc
