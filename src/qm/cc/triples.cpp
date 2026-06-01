#include <atomic>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/cc/triples.h>
#include <vector>

// Restricted (T) -- a direct port of the thc_cct reference rtriples.py
// (PySCF ccsd_t_slow). Loops over virtual triples (a>=b>=c); per triple it
// builds only O(nocc^3) tensors, so the O(o^3 v^3) triples array is never
// formed. Canonical reference assumed (f_vo = 0), so the disconnected term
// keeps only the t1 (vvoo) contribution.

namespace occ::qm::cc {

namespace {

using occ::Mat;
using T2 = Eigen::Tensor<double, 2>;
using T3 = Eigen::Tensor<double, 3>;
using T4 = Eigen::Tensor<double, 4>;
using Sh3 = Eigen::array<int, 3>;

// r3(w) = 4 w + w(jki) + w(kij) - 2 w(kji) - 2 w(ikj) - 2 w(jik)
T3 r3(const T3 &w) {
  return w * 4.0 + w.shuffle(Sh3{1, 2, 0}) + w.shuffle(Sh3{2, 0, 1}) -
         w.shuffle(Sh3{2, 1, 0}) * 2.0 - w.shuffle(Sh3{0, 2, 1}) * 2.0 -
         w.shuffle(Sh3{1, 0, 2}) * 2.0;
}

// sum_ijk w[perm] * z
inline double dot3(const T3 &w, const Sh3 &perm, const T3 &z) {
  const Eigen::Tensor<double, 0> s = (w.shuffle(perm) * z).sum();
  return s(0);
}

// index permutations (label of w against fixed "ijk" of z)
constexpr Sh3 P_ijk{0, 1, 2};
constexpr Sh3 P_ikj{0, 2, 1};
constexpr Sh3 P_jik{1, 0, 2};
constexpr Sh3 P_jki{2, 0, 1};
constexpr Sh3 P_kij{1, 2, 0};
constexpr Sh3 P_kji{2, 1, 0};

} // namespace

double ccsd_t(const T2 &t1, const T4 &t2, const CCIntegrals &e) {
  occ::timing::start(occ::timing::category::ccsd_triples);
  const int o = e.nocc, v = e.nvir;
  const Vec &mo_e = e.mo_energy;

  // Precompute contiguous matrix slices so each triple's get_w is two BLAS
  // GEMMs with no per-call copies. (The tensor chip/contract path is
  // overhead-bound for the small o^3 intermediates.) Footprint is the same
  // order as the ovvv / ovov blocks.
  const int oo = o * o;
  const size_t vv = static_cast<size_t>(v) * v;
  std::vector<Mat> vvov_slc(vv); // [a*v+b](i,f)      = (ia|fb) = ovvv(i,a,f,b)
  std::vector<Mat> vvoo_slc(vv); // [a*v+b](i,j)      = (ia|jb) = ovov(i,a,j,b)
  std::vector<Mat> t2bc_slc(vv); // [b*v+c](m,k)      = t2(m,k,b,c)
  for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
      Mat sv(o, v), so(o, o), st(o, o);
      for (int i = 0; i < o; ++i) {
        for (int f = 0; f < v; ++f)
          sv(i, f) = e.ovvv(i, a, f, b);
        for (int j = 0; j < o; ++j)
          so(i, j) = e.ovov(i, a, j, b);
      }
      for (int m = 0; m < o; ++m)
        for (int k = 0; k < o; ++k)
          st(m, k) = t2(m, k, a, b);
      vvov_slc[a * v + b] = std::move(sv);
      vvoo_slc[a * v + b] = std::move(so);
      t2bc_slc[a * v + b] = std::move(st);
    }
  std::vector<Mat> vooo_slc(v);  // [a](i*o+j, m)     = (ia|jm) = ovoo(i,a,j,m)
  std::vector<Mat> t2T_slc(v);   // [c](f, k*o+j)     = t2(k,j,c,f)
  for (int a = 0; a < v; ++a) {
    Mat m(oo, o);
    for (int i = 0; i < o; ++i)
      for (int j = 0; j < o; ++j)
        for (int mm = 0; mm < o; ++mm)
          m(i * o + j, mm) = e.ovoo(i, a, j, mm);
    vooo_slc[a] = std::move(m);
  }
  for (int c = 0; c < v; ++c) {
    Mat m(v, oo);
    for (int f = 0; f < v; ++f)
      for (int k = 0; k < o; ++k)
        for (int j = 0; j < o; ++j)
          m(f, k * o + j) = t2(k, j, c, f);
    t2T_slc[c] = std::move(m);
  }

  T3 eijk(o, o, o);
  for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
      for (int k = 0; k < o; ++k)
        eijk(i, j, k) = mo_e(i) + mo_e(j) + mo_e(k);

  auto get_w = [&](int a, int b, int c) -> T3 {
    const Mat g1 = vvov_slc[a * v + b] * t2T_slc[c]; // (o x o^2): [i, k*o+j]
    const Mat g2 = vooo_slc[a] * t2bc_slc[b * v + c]; // (o^2 x o): [i*o+j, k]
    T3 w(o, o, o);
    for (int i = 0; i < o; ++i)
      for (int j = 0; j < o; ++j)
        for (int k = 0; k < o; ++k)
          w(i, j, k) = g1(i, k * o + j) - g2(i * o + j, k);
    return w;
  };
  auto get_v = [&](int a, int b, int c) -> T3 {
    const Mat &vvoo_ab = vvoo_slc[a * v + b]; // (i,j)
    // v(i,j,k) = (ia|jb) t1(k,c); f_vo = 0 (canonical) drops the t2 term.
    T3 out(o, o, o);
    for (int i = 0; i < o; ++i)
      for (int j = 0; j < o; ++j)
        for (int k = 0; k < o; ++k)
          out(i, j, k) = vvoo_ab(i, j) * t1(k, c);
    return out;
  };

  // Parallelize over the outer virtual index a; each (a,b<=a,c<=b) triple is
  // independent. TBB work-stealing balances the triangular load. For non-trivial
  // sizes report progress (the (T) step is O(nocc^3 nvir^4)).
  const bool report = v >= 40;
  std::atomic<int> completed{0};
  occ::parallel::thread_local_storage<double> et_local(0.0);
  occ::parallel::parallel_for(size_t(0), static_cast<size_t>(v), [&](size_t au) {
    const int a = static_cast<int>(au);
    double &et = et_local.local();
    {
      for (int b = 0; b <= a; ++b) {
        for (int c = 0; c <= b; ++c) {
        T3 d3 = eijk - eijk.constant(mo_e(o + a) + mo_e(o + b) + mo_e(o + c));
        if (a == c)
          d3 = d3 * 6.0;
        else if (a == b || b == c)
          d3 = d3 * 2.0;

        const T3 wabc = get_w(a, b, c), wacb = get_w(a, c, b);
        const T3 wbac = get_w(b, a, c), wbca = get_w(b, c, a);
        const T3 wcab = get_w(c, a, b), wcba = get_w(c, b, a);
        const T3 vabc = get_v(a, b, c), vacb = get_v(a, c, b);
        const T3 vbac = get_v(b, a, c), vbca = get_v(b, c, a);
        const T3 vcab = get_v(c, a, b), vcba = get_v(c, b, a);
        const T3 zabc = r3(wabc + vabc * 0.5) / d3;
        const T3 zacb = r3(wacb + vacb * 0.5) / d3;
        const T3 zbac = r3(wbac + vbac * 0.5) / d3;
        const T3 zbca = r3(wbca + vbca * 0.5) / d3;
        const T3 zcab = r3(wcab + vcab * 0.5) / d3;
        const T3 zcba = r3(wcba + vcba * 0.5) / d3;

        et += dot3(wabc, P_ijk, zabc) + dot3(wacb, P_ikj, zabc) +
              dot3(wbac, P_jik, zabc) + dot3(wbca, P_jki, zabc) +
              dot3(wcab, P_kij, zabc) + dot3(wcba, P_kji, zabc);
        et += dot3(wacb, P_ijk, zacb) + dot3(wabc, P_ikj, zacb) +
              dot3(wcab, P_jik, zacb) + dot3(wcba, P_jki, zacb) +
              dot3(wbac, P_kij, zacb) + dot3(wbca, P_kji, zacb);
        et += dot3(wbac, P_ijk, zbac) + dot3(wbca, P_ikj, zbac) +
              dot3(wabc, P_jik, zbac) + dot3(wacb, P_jki, zbac) +
              dot3(wcba, P_kij, zbac) + dot3(wcab, P_kji, zbac);
        et += dot3(wbca, P_ijk, zbca) + dot3(wbac, P_ikj, zbca) +
              dot3(wcba, P_jik, zbca) + dot3(wcab, P_jki, zbca) +
              dot3(wabc, P_kij, zbca) + dot3(wacb, P_kji, zbca);
        et += dot3(wcab, P_ijk, zcab) + dot3(wcba, P_ikj, zcab) +
              dot3(wacb, P_jik, zcab) + dot3(wabc, P_jki, zcab) +
              dot3(wbca, P_kij, zcab) + dot3(wbac, P_kji, zcab);
        et += dot3(wcba, P_ijk, zcba) + dot3(wcab, P_ikj, zcba) +
              dot3(wbca, P_jik, zcba) + dot3(wbac, P_jki, zcba) +
              dot3(wacb, P_kij, zcba) + dot3(wabc, P_kji, zcba);
        }
      }
    }
    if (report) {
      const int c = completed.fetch_add(1) + 1;
      if ((100 * c) / v / 10 > (100 * (c - 1)) / v / 10)
        occ::log::info("  (T) {:3d}% done", ((100 * c) / v / 10) * 10);
    }
  });
  double et = 0.0;
  for (const double e_t : et_local)
    et += e_t;
  occ::timing::stop(occ::timing::category::ccsd_triples);
  return et * 2.0;
}

} // namespace occ::qm::cc
