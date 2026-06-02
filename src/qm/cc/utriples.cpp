#include <occ/core/timings.h>
#include <occ/qm/cc/utriples.h>

// Spin-adapted unrestricted (T) -- a direct port of PySCF uccsd_t_slow.py
// (kernel), evaluated natively from the spin-blocked UCCIntegrals. Four spin
// cases: aaa, bbb, baa (beta-alpha-alpha), bba (alpha-beta-beta). A canonical
// UHF reference is assumed so the f_vo disconnected terms drop. This builds the
// full 6D (ijkabc) tensors (the "slow" reference); a loop-restructured,
// memory-bounded version validates against it.
//
// numpy transpose(perm) == Eigen shuffle(perm).

namespace occ::qm::cc {

namespace {

using occ::Mat;
using occ::Vec;
using T2 = Eigen::Tensor<double, 2>;
using T4 = Eigen::Tensor<double, 4>;
using T6 = Eigen::Tensor<double, 6>;
using Sh6 = Eigen::array<int, 6>;
template <int N> using IA = Eigen::array<Eigen::IndexPair<int>, N>;
inline Eigen::IndexPair<int> ip(int a, int b) { return {a, b}; }
inline double sum0(const T6 &x) {
  Eigen::Tensor<double, 0> s = x.sum();
  return s(0);
}

// P(6): the symmetric 6-fold permutation of (i,j,k)<->(a,b,c) pairs.
T6 p6(const T6 &t) {
  return t + t.shuffle(Sh6{1, 2, 0, 4, 5, 3}) + t.shuffle(Sh6{2, 0, 1, 5, 3, 4}) +
         t.shuffle(Sh6{0, 2, 1, 3, 5, 4}) + t.shuffle(Sh6{2, 1, 0, 5, 4, 3}) +
         t.shuffle(Sh6{1, 0, 2, 4, 3, 5});
}
// R(6): antisymmetriser over the occupied (i,j,k) triple.
T6 r6(const T6 &w) {
  return w + w.shuffle(Sh6{2, 0, 1, 3, 4, 5}) + w.shuffle(Sh6{1, 2, 0, 3, 4, 5}) -
         w.shuffle(Sh6{2, 1, 0, 3, 4, 5}) - w.shuffle(Sh6{0, 2, 1, 3, 4, 5}) -
         w.shuffle(Sh6{1, 0, 2, 3, 4, 5});
}

// eia(i,a) = e_occ(i) - e_vir(a) for an active (occ then vir) energy vector.
Mat make_eia(const Vec &e, int o, int v) {
  Mat eia(o, v);
  for (int i = 0; i < o; ++i)
    for (int a = 0; a < v; ++a)
      eia(i, a) = e(i) - e(o + a);
  return eia;
}

// d3(i,j,k,a,b,c) = eP(i,a) + eQ(j,b) + eR(k,c).
T6 make_d3(const Mat &eP, const Mat &eQ, const Mat &eR) {
  const int oi = eP.rows(), oj = eQ.rows(), ok = eR.rows();
  const int va = eP.cols(), vb = eQ.cols(), vc = eR.cols();
  T6 d(oi, oj, ok, va, vb, vc);
  for (int i = 0; i < oi; ++i)
    for (int j = 0; j < oj; ++j)
      for (int k = 0; k < ok; ++k)
        for (int a = 0; a < va; ++a)
          for (int b = 0; b < vb; ++b)
            for (int c = 0; c < vc; ++c)
              d(i, j, k, a, b, c) = eP(i, a) + eQ(j, b) + eR(k, c);
  return d;
}

} // namespace

double uccsd_t(const UCCIntegrals &e, const T2 &t1a, const T2 &t1b,
               const T4 &t2aa, const T4 &t2ab, const T4 &t2bb) {
  occ::timing::start(occ::timing::category::ccsd_triples);
  const Mat eia = make_eia(e.mo_energy_a, e.nocca, e.nvira);
  const Mat eIA = make_eia(e.mo_energy_b, e.noccb, e.nvirb);
  double et = 0.0;

  // --- aaa ----------------------------------------------------------------
  {
    T6 w = t2aa.contract(e.ovvv, IA<1>{ip(3, 2)}).shuffle(Sh6{0, 1, 3, 2, 5, 4});
    w -= e.ovoo.contract(t2aa, IA<1>{ip(3, 0)}).shuffle(Sh6{0, 2, 3, 1, 4, 5});
    const T6 r = r6(w);
    const T6 v = e.ovov.contract(t1a, IA<0>{}).shuffle(Sh6{4, 0, 2, 5, 1, 3});
    const T6 d3 = make_d3(eia, eia, eia);
    et += sum0(p6(w + v) / d3 * r);
  }
  // --- bbb ----------------------------------------------------------------
  {
    T6 w = t2bb.contract(e.OVVV, IA<1>{ip(3, 2)}).shuffle(Sh6{0, 1, 3, 2, 5, 4});
    w -= e.OVOO.contract(t2bb, IA<1>{ip(3, 0)}).shuffle(Sh6{0, 2, 3, 1, 4, 5});
    const T6 r = r6(w);
    const T6 v = e.OVOV.contract(t1b, IA<0>{}).shuffle(Sh6{4, 0, 2, 5, 1, 3});
    const T6 d3 = make_d3(eIA, eIA, eIA);
    et += sum0(p6(w + v) / d3 * r);
  }
  // --- baa  (I=beta occ; j,k=alpha occ; A=beta vir; b,c=alpha vir) ---------
  {
    T6 w =
        t2ab.contract(e.ovvv, IA<1>{ip(2, 2)}).shuffle(Sh6{1, 0, 3, 2, 5, 4}) *
        2.0;
    w += t2ab.contract(e.ovVV, IA<1>{ip(3, 2)}).shuffle(Sh6{1, 0, 3, 5, 2, 4}) *
         2.0;
    w += t2aa.contract(e.OVvv, IA<1>{ip(3, 2)}).shuffle(Sh6{3, 0, 1, 4, 2, 5});
    w -= e.ovoo.contract(t2ab, IA<1>{ip(3, 0)}).shuffle(Sh6{3, 2, 0, 5, 4, 1}) *
         2.0;
    w -= e.ovOO.contract(t2ab, IA<1>{ip(3, 1)}).shuffle(Sh6{2, 3, 0, 5, 4, 1}) *
         2.0;
    w -= e.OVoo.contract(t2aa, IA<1>{ip(3, 1)}).shuffle(Sh6{0, 3, 2, 1, 4, 5});
    const T6 r0 = w - w.shuffle(Sh6{0, 2, 1, 3, 4, 5});
    const T6 r = r0 + r0.shuffle(Sh6{0, 2, 1, 3, 5, 4}); // fresh: avoid self-alias
    T6 v = e.ovov.contract(t1b, IA<0>{}).shuffle(Sh6{4, 0, 2, 5, 1, 3});
    v += e.ovOV.contract(t1a, IA<0>{}).shuffle(Sh6{2, 4, 0, 3, 5, 1}) * 2.0;
    w += v;
    const T6 d3 = make_d3(eIA, eia, eia);
    et += sum0(w * (r / d3));
  }
  // --- bba  (i=alpha occ; j,k=beta occ; a=alpha vir; b,c=beta vir) ---------
  {
    T6 w =
        t2ab.contract(e.OVVV, IA<1>{ip(3, 2)}).shuffle(Sh6{0, 1, 3, 2, 5, 4}) *
        2.0;
    w += t2ab.contract(e.OVvv, IA<1>{ip(2, 2)}).shuffle(Sh6{0, 1, 3, 5, 2, 4}) *
         2.0;
    w += t2bb.contract(e.ovVV, IA<1>{ip(3, 2)}).shuffle(Sh6{3, 0, 1, 4, 2, 5});
    w -= e.OVOO.contract(t2ab, IA<1>{ip(3, 1)}).shuffle(Sh6{3, 2, 0, 4, 5, 1}) *
         2.0;
    w -= e.OVoo.contract(t2ab, IA<1>{ip(3, 0)}).shuffle(Sh6{2, 3, 0, 4, 5, 1}) *
         2.0;
    w -= e.ovOO.contract(t2bb, IA<1>{ip(3, 1)}).shuffle(Sh6{0, 3, 2, 1, 4, 5});
    const T6 r0 = w - w.shuffle(Sh6{0, 2, 1, 3, 4, 5});
    const T6 r = r0 + r0.shuffle(Sh6{0, 2, 1, 3, 5, 4}); // fresh: avoid self-alias
    T6 v = e.OVOV.contract(t1a, IA<0>{}).shuffle(Sh6{4, 0, 2, 5, 1, 3});
    v += e.ovOV.contract(t1b, IA<0>{}).shuffle(Sh6{0, 4, 2, 1, 5, 3}) * 2.0;
    w += v;
    const T6 d3 = make_d3(eia, eIA, eIA);
    et += sum0(w * (r / d3));
  }

  occ::timing::stop(occ::timing::category::ccsd_triples);
  return et * 0.25;
}

} // namespace occ::qm::cc
