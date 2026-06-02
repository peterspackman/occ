#include <atomic>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/cc/utriples.h>
#include <vector>

// Spin-adapted unrestricted (T) -- a memory-bounded, loop-based evaluation of
// PySCF uccsd_t_slow.py. Four spin cases (aaa, bbb, baa, bba); a canonical UHF
// reference is assumed so the f_vo disconnected terms drop. Like the restricted
// triples.cpp, each case loops over virtual triples and builds only O(no^3)
// occupied tensors (via two GEMMs), so the O(no^3 nv^3) array is never formed.
//
// Per-triple algebra from the dense kernel (numpy transpose==Eigen shuffle):
//   p6(T)[ijk,abc] = T_abc[ijk]+T_cab[kij]+T_bca[jki]+T_acb[ikj]+T_cba[kji]+T_bac[jik]
//   r6(W)[ijk,abc] = W_abc[ijk]+W_abc[jki]+W_abc[kij]-W_abc[kji]-W_abc[ikj]-W_abc[jik]

namespace occ::qm::cc {

namespace {

using occ::Mat;
using occ::Vec;
using T2 = Eigen::Tensor<double, 2>;
using T3 = Eigen::Tensor<double, 3>;
using T4 = Eigen::Tensor<double, 4>;
using Sh3 = Eigen::array<int, 3>;
template <int N> using IA = Eigen::array<Eigen::IndexPair<int>, N>;
inline Eigen::IndexPair<int> ip(int a, int b) { return {a, b}; }

// occupied-index permutations of an o^3 tensor (the six S3 orderings)
inline T3 sh(const T3 &w, int p0, int p1, int p2) {
  return w.shuffle(Sh3{p0, p1, p2});
}
inline double dot3(const T3 &a, const T3 &b) {
  const Eigen::Tensor<double, 0> s = (a * b).sum();
  return s(0);
}

// e_occ(i)+e_occ(j)+e_occ(k) for an active (occ then vir) energy vector.
T3 make_eo3(const Vec &e, int o) {
  T3 t(o, o, o);
  for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
      for (int k = 0; k < o; ++k)
        t(i, j, k) = e(i) + e(j) + e(k);
  return t;
}

} // namespace

double uccsd_t(const UCCIntegrals &e, const T2 &t1a, const T2 &t1b,
               const T4 &t2aa, const T4 &t2ab, const T4 &t2bb) {
  occ::timing::start(occ::timing::category::ccsd_triples);
  const int oa = e.nocca, ob = e.noccb, va = e.nvira, vb = e.nvirb;
  const Vec &ea = e.mo_energy_a, &eb = e.mo_energy_b;

  // virtual-orbital energies
  auto evir = [](const Vec &en, int o, int n) {
    Vec v(n);
    for (int x = 0; x < n; ++x)
      v(x) = en(o + x);
    return v;
  };
  const Vec eva = evir(ea, oa, va), evb = evir(eb, ob, vb);
  const T3 eo3a = make_eo3(ea, oa), eo3b = make_eo3(eb, ob);

  double et = 0.0;

  // 10% progress ticks for the outer loop of a spin case (only when worthwhile).
  auto tick = [](const char *label, std::atomic<int> &done, int n) {
    if (n < 40)
      return;
    const int d = done.fetch_add(1) + 1;
    if ((100 * d) / n / 10 > (100 * (d - 1)) / n / 10)
      occ::log::info("  (T) {} {:3d}%", label, ((100 * d) / n / 10) * 10);
  };

  // ===================== same-spin: aaa / bbb ============================
  // Loops over all virtual triples (a,b,c); per triple builds W_p[ijk] for the
  // six virtual permutations p of (a,b,c) via GEMMs, then applies p6/r6.
  auto same_spin = [&](const char *label, const T4 &t2, const T4 &ovvv,
                       const T4 &ovoo, const T4 &ovov, const T2 &t1, int o,
                       int v, const Vec &ev, const T3 &eo3) -> double {
    // contiguous slices for the get_w GEMMs
    const int oo = o * o;
    std::vector<Mat> T2A(v);          // [a](i*o+j, e) = t2(i,j,a,e)
    std::vector<Mat> ovvvCB(static_cast<size_t>(v) * v); // [c*v+b](k,e)=ovvv(k,c,e,b)
    std::vector<Mat> t2bc(static_cast<size_t>(v) * v);   // [b*v+c](m,k)=t2(m,k,b,c)
    std::vector<Mat> ovooA(v);        // [a](i*o+j, m) = ovoo(i,a,j,m)
    std::vector<Mat> ovovBC(static_cast<size_t>(v) * v); // [b*v+c](j,k)=ovov(j,b,k,c)
    for (int a = 0; a < v; ++a) {
      Mat ta(oo, v), oa_(oo, o);
      for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j) {
          for (int x = 0; x < v; ++x)
            ta(i * o + j, x) = t2(i, j, a, x);
          for (int m = 0; m < o; ++m)
            oa_(i * o + j, m) = ovoo(i, a, j, m);
        }
      T2A[a] = std::move(ta);
      ovooA[a] = std::move(oa_);
    }
    for (int b = 0; b < v; ++b)
      for (int c = 0; c < v; ++c) {
        Mat ob_(o, v), tb(o, o), ov(o, o);
        for (int k = 0; k < o; ++k)
          for (int x = 0; x < v; ++x)
            ob_(k, x) = ovvv(k, c, x, b);
        for (int m = 0; m < o; ++m)
          for (int k = 0; k < o; ++k) {
            tb(m, k) = t2(m, k, b, c);
            ov(m, k) = ovov(m, b, k, c);
          }
        ovvvCB[c * v + b] = std::move(ob_);
        t2bc[b * v + c] = std::move(tb);
        ovovBC[b * v + c] = std::move(ov);
      }

    auto get_w = [&](int a, int b, int c) -> T3 {
      const Mat g1 = T2A[a] * ovvvCB[c * v + b].transpose(); // (oo x o)[ij,k]
      const Mat g2 = ovooA[a] * t2bc[b * v + c];             // (oo x o)[ij,k]
      T3 w(o, o, o);
      for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
          for (int k = 0; k < o; ++k)
            w(i, j, k) = g1(i * o + j, k) - g2(i * o + j, k);
      return w;
    };
    auto get_v = [&](int a, int b, int c) -> T3 {
      const Mat &ov = ovovBC[b * v + c]; // (j,k)
      T3 out(o, o, o);
      for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
          for (int k = 0; k < o; ++k)
            out(i, j, k) = t1(i, a) * ov(j, k);
      return out;
    };

    std::atomic<int> done{0};
    occ::parallel::thread_local_storage<double> acc(0.0);
    occ::parallel::parallel_for(size_t(0), static_cast<size_t>(v), [&](size_t au) {
      const int a = static_cast<int>(au);
      double &s = acc.local();
      for (int b = 0; b < v; ++b)
        for (int c = 0; c < v; ++c) {
          const T3 Wabc = get_w(a, b, c), Wcab = get_w(c, a, b),
                   Wbca = get_w(b, c, a), Wacb = get_w(a, c, b),
                   Wcba = get_w(c, b, a), Wbac = get_w(b, a, c);
          const T3 Tabc = Wabc + get_v(a, b, c), Tcab = Wcab + get_v(c, a, b),
                   Tbca = Wbca + get_v(b, c, a), Tacb = Wacb + get_v(a, c, b),
                   Tcba = Wcba + get_v(c, b, a), Tbac = Wbac + get_v(b, a, c);
          // p6: T_abc[ijk]+T_cab[kij]+T_bca[jki]+T_acb[ikj]+T_cba[kji]+T_bac[jik]
          const T3 p6 = Tabc + sh(Tcab, 1, 2, 0) + sh(Tbca, 2, 0, 1) +
                        sh(Tacb, 0, 2, 1) + sh(Tcba, 2, 1, 0) + sh(Tbac, 1, 0, 2);
          // r6: W_abc[ijk]+W[jki]+W[kij]-W[kji]-W[ikj]-W[jik]
          const T3 r6 = Wabc + sh(Wabc, 2, 0, 1) + sh(Wabc, 1, 2, 0) -
                        sh(Wabc, 2, 1, 0) - sh(Wabc, 0, 2, 1) - sh(Wabc, 1, 0, 2);
          T3 d3(o, o, o);
          const double evv = ev(a) + ev(b) + ev(c);
          for (int i = 0; i < o; ++i)
            for (int j = 0; j < o; ++j)
              for (int k = 0; k < o; ++k)
                d3(i, j, k) = eo3(i, j, k) - evv;
          s += dot3(p6 / d3, r6);
        }
      tick(label, done, v);
    });
    double s = 0.0;
    for (double x : acc)
      s += x;
    return s * 0.25;
  };

  et += same_spin("aaa", t2aa, e.ovvv, e.ovoo, e.ovov, t1a, oa, va, eva, eo3a);
  et += same_spin("bbb", t2bb, e.OVVV, e.OVOO, e.OVOV, t1b, ob, vb, evb, eo3b);

  // ===================== baa  (I beta-occ; j,k alpha-occ) =================
  // W(A,b,c)[I,j,k]; loop over (A,b,c); r needs W at (A,b,c) and (A,c,b).
  {
    auto get_W = [&](int A, int b, int c) -> T3 {
      const T3 t2abA = t2ab.chip(A, 3);                  // (oa,ob,va)[j,I,e]
      const T2 ovvv_cb = e.ovvv.chip(b, 3).chip(c, 1);   // (oa,va)[k,e]
      T3 W = t2abA.contract(ovvv_cb, IA<1>{ip(2, 1)}).shuffle(Sh3{1, 0, 2}) * 2.0;
      const T3 t2abB = t2ab.chip(b, 2);                  // (oa,ob,vb)[j,I,E]
      const T2 ovVV_cA = e.ovVV.chip(A, 3).chip(c, 1);   // (oa,vb)[k,E]
      W += t2abB.contract(ovVV_cA, IA<1>{ip(2, 1)}).shuffle(Sh3{1, 0, 2}) * 2.0;
      const T3 t2aaB = t2aa.chip(b, 2);                  // (oa,oa,va)[j,k,e]
      const T2 OVvv_Ac = e.OVvv.chip(c, 3).chip(A, 1);   // (ob,va)[I,e]
      W += t2aaB.contract(OVvv_Ac, IA<1>{ip(2, 1)}).shuffle(Sh3{2, 0, 1});
      const T2 t2ab_bA = t2ab.chip(A, 3).chip(b, 2);     // (oa,ob)[m/j,I/M]
      const T3 ovoo_c = e.ovoo.chip(c, 1);               // (oa,oa,oa)[k,j,m]
      W -= t2ab_bA.contract(ovoo_c, IA<1>{ip(0, 2)}).shuffle(Sh3{0, 2, 1}) * 2.0;
      const T3 ovOO_c = e.ovOO.chip(c, 1);               // (oa,ob,ob)[k,I,M]
      W -= t2ab_bA.contract(ovOO_c, IA<1>{ip(1, 2)}).shuffle(Sh3{2, 0, 1}) * 2.0;
      const T2 t2aa_bc = t2aa.chip(c, 3).chip(b, 2);     // (oa,oa)[j,m]
      const T3 OVoo_A = e.OVoo.chip(A, 1);               // (ob,oa,oa)[I,k,m]
      W -= t2aa_bc.contract(OVoo_A, IA<1>{ip(1, 2)}).shuffle(Sh3{1, 0, 2});
      return W;
    };
    std::atomic<int> done{0};
    occ::parallel::thread_local_storage<double> acc(0.0);
    occ::parallel::parallel_for(size_t(0), static_cast<size_t>(vb), [&](size_t Au) {
      const int A = static_cast<int>(Au);
      double &s = acc.local();
      for (int b = 0; b < va; ++b)
        for (int c = 0; c < va; ++c) {
          const T3 Wabc = get_W(A, b, c), Wacb = get_W(A, c, b);
          const T3 D = Wabc - Wacb;
          const T3 r = D - D.shuffle(Sh3{0, 2, 1});
          const T2 ovov_bc = e.ovov.chip(c, 3).chip(b, 1); // (oa,oa)[j,k]
          const T2 ovOV_cA = e.ovOV.chip(A, 3).chip(c, 1); // (oa,ob)[k,I]
          for (int I = 0; I < ob; ++I)
            for (int j = 0; j < oa; ++j)
              for (int k = 0; k < oa; ++k) {
                const double num = Wabc(I, j, k) + ovov_bc(j, k) * t1b(I, A) +
                                   2.0 * ovOV_cA(k, I) * t1a(j, b);
                const double d3 = (eb(I) - evb(A)) + (ea(j) - eva(b)) +
                                  (ea(k) - eva(c));
                s += num * r(I, j, k) / d3;
              }
        }
      tick("baa", done, vb);
    });
    double s = 0.0;
    for (double x : acc)
      s += x;
    et += s * 0.25;
  }

  // ===================== bba  (i alpha-occ; j,k beta-occ) =================
  {
    auto get_W = [&](int a, int b, int c) -> T3 {
      const T3 t2abA = t2ab.chip(a, 2);                  // (oa,ob,vb)[i,j,e]
      const T2 OVVV_cb = e.OVVV.chip(b, 3).chip(c, 1);   // (ob,vb)[k,e]
      T3 W = t2abA.contract(OVVV_cb, IA<1>{ip(2, 1)});   // (i,j,k)
      W = W * 2.0;
      const T3 t2abB = t2ab.chip(b, 3);                  // (oa,ob,va)[i,j,e]
      const T2 OVvv_ca = e.OVvv.chip(a, 3).chip(c, 1);   // (ob,va)[k,e]
      W += t2abB.contract(OVvv_ca, IA<1>{ip(2, 1)}) * 2.0;
      const T3 t2bbB = t2bb.chip(b, 2);                  // (ob,ob,vb)[j,k,e]
      const T2 ovVV_ac = e.ovVV.chip(c, 3).chip(a, 1);   // (oa,vb)[i,e]
      W += t2bbB.contract(ovVV_ac, IA<1>{ip(2, 1)}).shuffle(Sh3{2, 0, 1});
      const T2 t2ab_ab = t2ab.chip(b, 3).chip(a, 2);     // (oa,ob)[i/m,m/j]
      const T3 OVOO_c = e.OVOO.chip(c, 1);               // (ob,ob,ob)[k,j,m]
      W -= t2ab_ab.contract(OVOO_c, IA<1>{ip(1, 2)}).shuffle(Sh3{0, 2, 1}) * 2.0;
      const T3 OVoo_c = e.OVoo.chip(c, 1);               // (ob,oa,oa)[k,i,m]
      W -= t2ab_ab.contract(OVoo_c, IA<1>{ip(0, 2)}).shuffle(Sh3{2, 0, 1}) * 2.0;
      const T2 t2bb_bc = t2bb.chip(c, 3).chip(b, 2);     // (ob,ob)[j,m]
      const T3 ovOO_a = e.ovOO.chip(a, 1);               // (oa,ob,ob)[i,k,m]
      W -= t2bb_bc.contract(ovOO_a, IA<1>{ip(1, 2)}).shuffle(Sh3{1, 0, 2});
      return W;
    };
    std::atomic<int> done{0};
    occ::parallel::thread_local_storage<double> acc(0.0);
    occ::parallel::parallel_for(size_t(0), static_cast<size_t>(va), [&](size_t au) {
      const int a = static_cast<int>(au);
      double &s = acc.local();
      for (int b = 0; b < vb; ++b)
        for (int c = 0; c < vb; ++c) {
          const T3 Wabc = get_W(a, b, c), Wacb = get_W(a, c, b);
          const T3 D = Wabc - Wacb;
          const T3 r = D - D.shuffle(Sh3{0, 2, 1});
          const T2 OVOV_bc = e.OVOV.chip(c, 3).chip(b, 1); // (ob,ob)[j,k]
          const T2 ovOV_ac = e.ovOV.chip(c, 3).chip(a, 1); // (oa,ob)[i,k]
          for (int i = 0; i < oa; ++i)
            for (int j = 0; j < ob; ++j)
              for (int k = 0; k < ob; ++k) {
                const double num = Wabc(i, j, k) + OVOV_bc(j, k) * t1a(i, a) +
                                   2.0 * ovOV_ac(i, k) * t1b(j, b);
                const double d3 = (ea(i) - eva(a)) + (eb(j) - evb(b)) +
                                  (eb(k) - evb(c));
                s += num * r(i, j, k) / d3;
              }
        }
      tick("bba", done, va);
    });
    double s = 0.0;
    for (double x : acc)
      s += x;
    et += s * 0.25;
  }

  occ::timing::stop(occ::timing::category::ccsd_triples);
  return et;
}

} // namespace occ::qm::cc
