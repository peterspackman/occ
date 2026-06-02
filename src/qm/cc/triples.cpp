#include <array>
#include <atomic>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/cc/triples.h>
#include <vector>

// Restricted (T). Loops over virtual triples (a>=b>=c); per triple only
// O(nocc^3) tensors are built, so the O(o^3 v^3) triples array is never formed.
// Canonical reference assumed (f_vo = 0), so the disconnected term keeps only
// the t1 (vvoo) contribution.
//
// Layout follows PySCF's production kernel (pyscf/lib/cc/ccsd_t.c): both GEMMs
// emit w in [i,j,k] (k-fastest) CONTIGUOUS order, so the per-ordering fold is a
// contiguous read + indexed scatter in tight raw-pointer loops -- the energy
// contraction is then ~9 vectorisable passes over the o^3 working set instead
// of the strided Eigen element accesses (operator() on (i,k*o+j)) that ran the
// bookkeeping at ~40 cycles/element. The GEMM that dominates (particle term) is
// also cast as M=o^2 (fat) rather than M=o (skinny), ~1.7x faster on BLAS.

namespace occ::qm::cc {

namespace {
using occ::Mat;
using T2 = Eigen::Tensor<double, 2>;
using T4 = Eigen::Tensor<double, 4>;
} // namespace

double ccsd_t(const T2 &t1, const T4 &t2, const CCIntegrals &e) {
  occ::timing::start(occ::timing::category::ccsd_triples);
  const int o = e.nocc, v = e.nvir;
  const int oo = o * o;
  const int ooo = oo * o;
  const Vec &mo_e = e.mo_energy;

  // Contiguous matrix slices, laid out so each ordering's two GEMMs land in
  // [i,j,k] (k-fastest) order:
  //   cache1[i,j,k] = sum_f t2kjf[j*o+k, f] vvovT[f, i]      (oo x v)*(v x o)
  //   cache2[i,j,k] = sum_m t2km[k, m]      voooT[m, i*o+j]  (o  x o)*(o x oo)
  //   v[i,j,k]      = vvoo[i, j] t1(k, c)
  const size_t vv = static_cast<size_t>(v) * v;
  std::vector<Mat> t2kjf_slc(v);     // [c](j*o+k, f)  = t2(k,j,c,f)
  std::vector<Mat> vvovT_slc(vv);    // [a*v+b](f, i)  = ovvv(i,a,f,b)
  std::vector<Mat> t2km_slc(vv);     // [b*v+c](k, m)  = t2(m,k,b,c)
  std::vector<Mat> voooT_slc(v);     // [a](m, i*o+j)  = ovoo(i,a,j,m)
  std::vector<Mat> vvoo_slc(vv);     // [a*v+b](i, j)  = ovov(i,a,j,b)
  for (int c = 0; c < v; ++c) {
    Mat m(oo, v);
    for (int j = 0; j < o; ++j)
      for (int k = 0; k < o; ++k)
        for (int f = 0; f < v; ++f)
          m(j * o + k, f) = t2(k, j, c, f);
    t2kjf_slc[c] = std::move(m);
  }
  for (int a = 0; a < v; ++a) {
    Mat m(o, oo);
    for (int mm = 0; mm < o; ++mm)
      for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
          m(mm, i * o + j) = e.ovoo(i, a, j, mm);
    voooT_slc[a] = std::move(m);
  }
  for (int a = 0; a < v; ++a)
    for (int b = 0; b < v; ++b) {
      Mat sv(v, o), so(o, o), st(o, o);
      for (int f = 0; f < v; ++f)
        for (int i = 0; i < o; ++i)
          sv(f, i) = e.ovvv(i, a, f, b);
      for (int i = 0; i < o; ++i)
        for (int j = 0; j < o; ++j)
          so(i, j) = e.ovov(i, a, j, b);
      for (int k = 0; k < o; ++k)
        for (int mm = 0; mm < o; ++mm)
          st(k, mm) = t2(mm, k, a, b);
      vvovT_slc[a * v + b] = std::move(sv);
      vvoo_slc[a * v + b] = std::move(so);
      t2km_slc[a * v + b] = std::move(st);
    }

  std::vector<double> eijk(ooo);
  for (int i = 0, n = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
      for (int k = 0; k < o; ++k, ++n)
        eijk[n] = mo_e(i) + mo_e(j) + mo_e(k);

  // Six permutation index maps (PySCF _make_permute_indices): ordering tau
  // scatters its w/v into w0/v0 at pidx[tau][ijk], folding all six orderings.
  std::array<std::vector<int>, 6> pidx;
  for (auto &p : pidx)
    p.resize(ooo);
  for (int i = 0, m = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
      for (int k = 0; k < o; ++k, ++m) {
        pidx[0][m] = i * oo + j * o + k;
        pidx[1][m] = i * oo + k * o + j;
        pidx[2][m] = j * oo + i * o + k;
        pidx[3][m] = k * oo + i * o + j;
        pidx[4][m] = j * oo + k * o + i;
        pidx[5][m] = k * oo + j * o + i;
      }

  // Parallelize over the outer virtual index a; each (a,b<=a,c<=b) triple is
  // independent and TBB work-stealing balances the triangular load.
  const bool report = v >= 40;
  std::atomic<int> completed{0};
  occ::parallel::thread_local_storage<double> et_local(0.0);
  occ::parallel::parallel_for(size_t(0), static_cast<size_t>(v), [&](size_t au) {
    const int a = static_cast<int>(au);
    double &et = et_local.local();
    std::vector<double> w0(ooo), v0(ooo), z0(ooo), c1(ooo), c2(ooo);

    // accumulate ordering (A,B,C)'s connected (w) and disconnected (v) pieces
    // into w0/v0 via the permutation index idx. Both GEMMs are written in
    // [i,j,k] contiguous order, so c1/c2 are read contiguously.
    auto accum = [&](int A, int B, int C, const int *idx) {
      Eigen::Map<Mat>(c1.data(), oo, o).noalias() =
          t2kjf_slc[C] * vvovT_slc[A * v + B];        // (oo x o): w1[ijk]
      Eigen::Map<Mat>(c2.data(), o, oo).noalias() =
          t2km_slc[B * v + C] * voooT_slc[A];         // (o x oo): w2[ijk]
      const double *vvoo = vvoo_slc[A * v + B].data(); // (o x o): [i + j*o]
      const double *t1c = t1.data() + static_cast<size_t>(C) * o;
      for (int i = 0, n = 0; i < o; ++i)
        for (int j = 0; j < o; ++j) {
          const double vij = vvoo[i + j * o];
          for (int k = 0; k < o; ++k, ++n) {
            w0[idx[n]] += c1[n] - c2[n];
            v0[idx[n]] += vij * t1c[k];
          }
        }
    };

    for (int b = 0; b <= a; ++b) {
      for (int c = 0; c <= b; ++c) {
        std::fill(w0.begin(), w0.end(), 0.0);
        std::fill(v0.begin(), v0.end(), 0.0);
        accum(a, b, c, pidx[0].data());
        accum(a, c, b, pidx[1].data());
        accum(b, a, c, pidx[2].data());
        accum(b, c, a, pidx[3].data());
        accum(c, a, b, pidx[4].data());
        accum(c, b, a, pidx[5].data());

        // u = w0 + 0.5 v0 (stored back into v0), then z0 = r3(u):
        //   z0[ijk] = 4u[ijk] + u[jki] + u[kij] - 2u[kji] - 2u[ikj] - 2u[jik]
        for (int n = 0; n < ooo; ++n)
          v0[n] = w0[n] + 0.5 * v0[n];
        const double *u = v0.data();
        for (int i = 0; i < o; ++i)
          for (int j = 0; j < o; ++j)
            for (int k = 0; k < o; ++k)
              z0[i * oo + j * o + k] =
                  4.0 * u[i * oo + j * o + k] + u[j * oo + k * o + i] +
                  u[k * oo + i * o + j] - 2.0 * u[k * oo + j * o + i] -
                  2.0 * u[i * oo + k * o + j] - 2.0 * u[j * oo + i * o + k];

        const double eabc = mo_e(o + a) + mo_e(o + b) + mo_e(o + c);
        const double fac =
            (a == c) ? (1.0 / 6.0) : ((a == b || b == c) ? 0.5 : 1.0);
        double s = 0.0;
        for (int n = 0; n < ooo; ++n)
          s += w0[n] * z0[n] / (eijk[n] - eabc);
        et += fac * s;
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
