#pragma once
// Boys function F_m(T), m = 0..L, for a vector of independent arguments.
//
// Shared by the kernels that evaluate Hermite Coulomb integrals for a batch of
// arguments (ESP grid points, split RI-J auxiliary primitives). Written with
// GCC/Clang vector types rather than intrinsics, so the same code is AVX2 or
// AVX-512 on x86-64 and NEON on AArch64. Baseline x86-64 builds compile the
// callers a second time inside __attribute__((target("avx2,fma"))) functions
// and pick that copy at run time with cpu_has_avx2_fma(); everything here is
// always_inline so it takes on the caller's target.
#include <occ/ints/boys.h>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) &&       \
    !defined(__EMSCRIPTEN__) && !defined(OCC_DISABLE_SIMD_DISPATCH)
#define OCC_SIMD_X86_DISPATCH 1
#else
#define OCC_SIMD_X86_DISPATCH 0
#endif

#define OCC_SIMD_INLINE __attribute__((always_inline)) inline

#if defined(__AVX512F__) && defined(__AVX512VL__)
#include <immintrin.h>
#endif

namespace occ::ints::simd {

#if OCC_SIMD_X86_DISPATCH
inline bool cpu_has_avx2_fma() {
  static const bool has = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  }();
  return has;
}
#endif

// N doubles, and the integers of the same width (comparison results, bits).
template <int N> struct Vec {
  typedef double D __attribute__((vector_size(N * sizeof(double))));
  typedef long long I __attribute__((vector_size(N * sizeof(double))));
  typedef int I32 __attribute__((vector_size(N * sizeof(int))));
  // the same at any element alignment, for loads and stores in place
  typedef double DU __attribute__((vector_size(N * sizeof(double)),
                                   aligned(sizeof(double)), may_alias));
};

template <typename V> constexpr int lanes() {
  return sizeof(V) / sizeof(double);
}

// F_m(x), m = 0..N-1, for one argument, written with stride `ld`. As
// boys_evaluate, but the segment index comes from a multiply by the (exact)
// reciprocal segment width rather than a division.
template <typename T, int N>
inline void boys_point(const T *table, T x, T *F, int ld) {
  using P = BoysParamsDefault;
  constexpr T inv_delta = T(P::Segments) / T(P::MaxT);
  static_assert(P::Segments % P::MaxT == 0, "segment width must be 1/integer");
  if (x < T(P::MaxT)) {
    const T xs = x * inv_delta;
    const int seg = static_cast<int>(xs);
    const T xd = xs - T(seg) - T(0.5);
    const T *base = table + seg * P::MaxM * (P::Order + 1);
    for (int m = 0; m < N; ++m) {
      const T *c = base + m * (P::Order + 1);
      T r = c[P::Order];
      for (int k = P::Order - 1; k >= 0; --k)
        r = r * xd + c[k];
      F[m * ld] = r;
    }
  } else {
    const T inv_x = T(1) / x;
    T f = BoysConstants<T>::sqrt_pi_over_4 * std::sqrt(inv_x);
    F[0] = f;
    for (int m = 1; m < N; ++m) {
      f *= (T(m) - T(0.5)) * inv_x;
      F[m * ld] = f;
    }
  }
}

// a where mask, else b (mask lanes all ones or all zeros)
template <typename V, typename I>
OCC_SIMD_INLINE V select(I mask, V a, V b) {
  return (V)((mask & (I)a) | (~mask & (I)b));
}

// e^{-t} for 0 <= t <= 64 to a couple of ulp: t = n ln 2 - r, |r| <= ln2 / 2,
// a degree-13 Taylor polynomial in r and 2^n put in the exponent. n is rounded
// by adding 1.5 * 2^52, which also leaves it in the low mantissa bits.
template <typename V> OCC_SIMD_INLINE V exp_neg(V t) {
  using I = typename Vec<lanes<V>()>::I;
  constexpr double shift = 6755399441055744.0; // 1.5 * 2^52
  const V x = -t;
  const V ns = x * 1.4426950408889634074 + shift;
  const V n = ns - shift;
  V r = x - n * 6.93147180369123816490e-01;
  r = r - n * 1.90821492927058770002e-10;
  constexpr double inv_fact[14] = {1.0,
                                   1.0,
                                   1.0 / 2,
                                   1.0 / 6,
                                   1.0 / 24,
                                   1.0 / 120,
                                   1.0 / 720,
                                   1.0 / 5040,
                                   1.0 / 40320,
                                   1.0 / 362880,
                                   1.0 / 3628800,
                                   1.0 / 39916800,
                                   1.0 / 479001600,
                                   1.0 / 6227020800.0};
  V p = V{} + inv_fact[13];
  for (int k = 12; k >= 0; --k)
    p = p * r + inv_fact[k];
  const I bits = ((I)ns - (I)(V{} + shift) + 1023) << 52;
  return p * (V)bits;
}

// 1/sqrt(x) to about an ulp for 1e-300 < x < 1e300, without the divider: an
// estimate and Newton steps, two from AVX-512's 14-bit estimate where the
// build targets it, else four from the bit-pattern one (3.4% error).
template <typename V> OCC_SIMD_INLINE V rsqrt(V x) {
  using I = typename Vec<lanes<V>()>::I;
  V y = (V)(0x5FE6EB50C7B537A9LL - ((I)x >> 1));
  int steps = 4;
#if defined(__AVX512F__) && defined(__AVX512VL__)
  if constexpr (lanes<V>() == 8) {
    y = (V)_mm512_rsqrt14_pd((__m512d)x);
    steps = 2;
  } else if constexpr (lanes<V>() == 4) {
    y = (V)_mm256_rsqrt14_pd((__m256d)x);
    steps = 2;
  }
#endif
  const V hx = x * 0.5;
  for (int i = 0; i < steps; ++i)
    y = y + y * (0.5 - hx * y * y);
  return y;
}

// The eight interpolation coefficients of order L in the segments idx[0..3],
// lane j of k[i] coefficient i of segment j: two contiguous loads per segment
// and a 4x4 transpose per half rather than gathers (unpack and permute on
// x86, zip on NEON).
template <typename V4>
OCC_SIMD_INLINE void transpose4(const V4 *in, V4 *o) {
  const V4 t0 = __builtin_shufflevector(in[0], in[1], 0, 4, 2, 6);
  const V4 t1 = __builtin_shufflevector(in[0], in[1], 1, 5, 3, 7);
  const V4 t2 = __builtin_shufflevector(in[2], in[3], 0, 4, 2, 6);
  const V4 t3 = __builtin_shufflevector(in[2], in[3], 1, 5, 3, 7);
  o[0] = __builtin_shufflevector(t0, t2, 0, 1, 4, 5);
  o[1] = __builtin_shufflevector(t1, t3, 0, 1, 4, 5);
  o[2] = __builtin_shufflevector(t0, t2, 2, 3, 6, 7);
  o[3] = __builtin_shufflevector(t1, t3, 2, 3, 6, 7);
}

template <typename V4>
OCC_SIMD_INLINE void load_coefficients4(const double *c, const int *idx,
                                        V4 *k) {
  using DU = typename Vec<4>::DU;
  V4 lo[4], hi[4];
  for (int j = 0; j < 4; ++j) {
    lo[j] = *reinterpret_cast<const DU *>(c + idx[j]);
    hi[j] = *reinterpret_cast<const DU *>(c + idx[j] + 4);
  }
  transpose4(lo, k);
  transpose4(hi, k + 4);
}

// F_m(t), m = 0..L, for the four lanes of t. Only the top order
// is interpolated; the rest follow by downward recursion,
// F_m = (2 T F_{m+1} + e^{-T}) / (2m + 1), which is stable. A vector whose
// arguments all lie on one side of the interpolation range skips the other
// branch.
template <int L>
OCC_SIMD_INLINE void boys4(const double *table, typename Vec<4>::D t,
                           typename Vec<4>::D *F) {
  constexpr int N = 4;
  using V = typename Vec<N>::D;
  using I = typename Vec<N>::I;
  using I32 = typename Vec<N>::I32;
  using P = BoysParamsDefault;
  constexpr int stride = P::MaxM * (P::Order + 1);
  static_assert(L < P::MaxM, "Boys table order exceeded");
  static_assert(P::Order == 7, "transposed table load assumes 8 coefficients");
  const I inside = t < double(P::MaxT);
  // lanes reduced in registers: any / all of the mask
  I m_or = inside | __builtin_shufflevector(inside, inside, 2, 3, 0, 1);
  I m_and = inside & __builtin_shufflevector(inside, inside, 2, 3, 0, 1);
  m_or |= __builtin_shufflevector(m_or, m_or, 1, 0, 3, 2);
  m_and &= __builtin_shufflevector(m_and, m_and, 1, 0, 3, 2);
  const bool any = m_or[0] != 0, all = m_and[0] != 0;
  V fin[L + 1], fas[L + 1];
  if (any) {
    // arguments outside the table are clamped to segment 0 and discarded
    const V tin = (V)((I)t & inside);
    const V xs = tin * (double(P::Segments) / P::MaxT);
    const I32 seg = __builtin_convertvector(xs, I32);
    const V xd = xs - __builtin_convertvector(seg, V) - 0.5;
    alignas(16) int idx[N];
    *reinterpret_cast<I32 *>(idx) = seg * stride;
    const double *c = table + L * (P::Order + 1);
    V k[8];
    load_coefficients4(c, idx, k);
    V r = k[7];
    for (int j = 6; j >= 0; --j)
      r = r * xd + k[j];
    fin[L] = r;
    if constexpr (L > 0) {
      const V e = exp_neg(tin);
      const V t2 = tin + tin;
      for (int m = L - 1; m >= 0; --m)
        fin[m] = (t2 * fin[m + 1] + e) * (1.0 / (2 * m + 1));
    }
  }
  if (!all) {
    const V rs = rsqrt(select(inside, V{} + 1.0, t));
    const V inv = rs * rs;
    V f = BoysConstants<double>::sqrt_pi_over_4 * rs;
    fas[0] = f;
    for (int m = 1; m <= L; ++m) {
      f = f * ((m - 0.5) * inv);
      fas[m] = f;
    }
  }
  for (int m = 0; m <= L; ++m)
    F[m] = all ? fin[m] : !any ? fas[m] : select(inside, fin[m], fas[m]);
}

template <int L, typename V>
OCC_SIMD_INLINE void boys(const double *table, V t, V *F) {
  constexpr int N = lanes<V>();
  static_assert(N == 4 || N == 8, "Boys vectors are four or eight lanes");
  if constexpr (N == 4) {
    boys4<L>(table, t, F);
  } else {
    // two halves of four, moved in registers
    using V4 = typename Vec<4>::D;
    V4 lo[L + 1], hi[L + 1];
    boys4<L>(table, __builtin_shufflevector(t, t, 0, 1, 2, 3), lo);
    boys4<L>(table, __builtin_shufflevector(t, t, 4, 5, 6, 7), hi);
    for (int m = 0; m <= L; ++m)
      F[m] = __builtin_shufflevector(lo[m], hi[m], 0, 1, 2, 3, 4, 5, 6, 7);
  }
}

} // namespace occ::ints::simd
