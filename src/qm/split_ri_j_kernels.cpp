#include "split_ri_j_kernels.h"
#include <utility>

namespace occ::qm::srij {

namespace {

// Recursion schedule for the Hermite Coulomb integrals R^{(n)}_{tuv}. Entry k
// (canonical Hermite order, so the entries of total degree d are exactly
// nhermsum(d - 1) .. nhermsum(d) - 1) is built along one Cartesian direction
// from the level above:
//   R^{(n)}_k = X_d R^{(n+1)}_{k - e_d} + (c - 1) R^{(n+1)}_{k - 2 e_d}.
template <int L> struct Schedule {
  static constexpr int N = nhermsum(L);
  std::array<unsigned char, N> dir{};
  std::array<short, N> src1{}, src2{}, coef{};
  constexpr Schedule() {
    for (int deg = 1; deg <= L; ++deg) {
      for (int t = deg; t >= 0; --t) {
        for (int u = deg - t; u >= 0; --u) {
          const int v = deg - t - u;
          const int k = hermite_index(t, u, v);
          if (t > 0) {
            dir[k] = 0;
            src1[k] = hermite_index(t - 1, u, v);
            coef[k] = t - 1;
            src2[k] = t > 1 ? hermite_index(t - 2, u, v) : 0;
          } else if (u > 0) {
            dir[k] = 1;
            src1[k] = hermite_index(t, u - 1, v);
            coef[k] = u - 1;
            src2[k] = u > 1 ? hermite_index(t, u - 2, v) : 0;
          } else {
            dir[k] = 2;
            src1[k] = hermite_index(t, u, v - 1);
            coef[k] = v - 1;
            src2[k] = v > 1 ? hermite_index(t, u, v - 2) : 0;
          }
        }
      }
    }
  }
};

// idx[a * NC + c] = position of R_{t_a + t_c} in the canonical order.
template <int LA, int LC> struct RIndex {
  static constexpr int NA = nhermsum(LA), NC = nhermsum(LC);
  std::array<short, NA * NC> idx{};
  constexpr RIndex() {
    std::array<int, nhermsum(LA + LC) * 3> tuv{};
    for (int t = 0; t <= LA + LC; ++t)
      for (int u = 0; t + u <= LA + LC; ++u)
        for (int v = 0; t + u + v <= LA + LC; ++v) {
          const int h = hermite_index(t, u, v);
          tuv[3 * h] = t;
          tuv[3 * h + 1] = u;
          tuv[3 * h + 2] = v;
        }
    for (int a = 0; a < NA; ++a)
      for (int c = 0; c < NC; ++c)
        idx[a * NC + c] =
            hermite_index(tuv[3 * a] + tuv[3 * c], tuv[3 * a + 1] + tuv[3 * c + 1],
                          tuv[3 * a + 2] + tuv[3 * c + 2]);
  }
};

// Lanes are processed in groups of W doubles held in one GCC vector: four
// (one ymm, or two xmm on baseline x86-64), or eight (one zmm) where the build
// targets AVX-512. The recursion below is then explicitly vectorised
// regardless of the auto-vectoriser, and every load is the same width as the
// store that produced it, so store-to-load forwarding never splits.
template <bool Avx2> constexpr int group_width() {
#if defined(__AVX512F__)
  return Avx2 ? 8 : 4;
#else
  return 4;
#endif
}

// The in-place recursion as a flat list of steps: step s either builds entry
// op_k[s] > 0 of the current level from the level above, or (op_k[s] == 0)
// sets R_000 to level op_n[s]. Level n overwrites degrees L - n .. 1 from the
// top down, so the lower degrees it reads still hold level n + 1.
template <int L> struct Steps {
  static constexpr int count() {
    // R_000 at level L; then per level n, degrees L - n .. 1 and R_000
    int c = 1;
    for (int n = L - 1; n >= 0; --n)
      c += nhermsum(L - n);
    return c;
  }
  static constexpr int N = count();
  std::array<short, N> op_k{}, op_n{};
  constexpr Steps() {
    int s = 0;
    op_k[s] = 0;
    op_n[s++] = L;
    for (int n = L - 1; n >= 0; --n) {
      for (int d = L - n; d >= 1; --d)
        for (int k = nhermsum(d - 1); k < nhermsum(d); ++k) {
          op_k[s] = k;
          op_n[s++] = n;
        }
      op_k[s] = 0;
      op_n[s++] = n;
    }
  }
};

// R entry k of one lane group lives at R[k * NG] (groups interleaved, so the
// chunk's R is [k][NP] in doubles).
template <int L, int S, int NG, typename V>
__attribute__((always_inline)) inline void step(V *R, const V *F, const V &X,
                                                const V &Y, const V &Z) {
  static constexpr Steps<L> st{};
  static constexpr Schedule<L> sc{};
  constexpr int k = st.op_k[S];
  if constexpr (k == 0) {
    R[0] = F[st.op_n[S]];
  } else {
    constexpr int dir = sc.dir[k];
    const V &XD = dir == 0 ? X : (dir == 1 ? Y : Z);
    if constexpr (sc.coef[k] > 0)
      R[k * NG] = XD * R[sc.src1[k] * NG] + double(sc.coef[k]) * R[sc.src2[k] * NG];
    else
      R[k * NG] = XD * R[sc.src1[k] * NG];
  }
}

template <int L, int NG, typename V, size_t... S>
__attribute__((always_inline)) inline void
recursion_unrolled(V *R, const V *F, const V &X, const V &Y, const V &Z,
                   std::index_sequence<S...>) {
  (step<L, S, NG>(R, F, X, Y, Z), ...);
}

// Level-0 Hermite integrals S_k R_t(alpha_k, P - C_k), t of total degree <= L,
// for aux primitives k0 .. k0 + NP - 1, into R[t * NP + k - k0].
template <int L, bool Avx2>
__attribute__((always_inline)) inline void
hermite_chunk(const double *boys, const PrimPair &pp, const AuxView &a, int k0,
              double *Rd) {
  static_assert(L <= MAX_L);
  constexpr int W = group_width<Avx2>();
  constexpr int NG = NP / W;
  using V = typename ints::simd::Vec<W>::D;
  using VU = typename ints::simd::Vec<W>::DU;
  const double p = pp.p;
  for (int gi = 0; gi < NG; ++gi) {
    const int k = k0 + gi * W;
    V *R = reinterpret_cast<V *>(Rd) + gi;
    const V g = *reinterpret_cast<const VU *>(a.gamma + k);
    const V cp = *reinterpret_cast<const VU *>(a.cpref + k);
    const V X = pp.Px - *reinterpret_cast<const VU *>(a.cx + k);
    const V Y = pp.Py - *reinterpret_cast<const VU *>(a.cy + k);
    const V Z = pp.Pz - *reinterpret_cast<const VU *>(a.cz + k);
    const V pq = p + g;
    const V sq = ints::simd::rsqrt(pq); // 1/sqrt(p + gamma)
    const V al = p * g * (sq * sq);
    V F[L + 1];
    ints::simd::boys<L>(boys, al * (X * X + Y * Y + Z * Z), F);
    // F_n <- S (-2 alpha)^n F_n = R^{(n)}_{000}
    V S = pp.pref * cp * sq;
    const V m2a = -2.0 * al;
    for (int n = 0; n <= L; ++n) {
      F[n] *= S;
      S *= m2a;
    }
    if constexpr (L <= 10) {
      recursion_unrolled<L, NG>(R, F, X, Y, Z,
                                std::make_index_sequence<Steps<L>::N>{});
    } else {
      // The same steps as a loop: fully unrolled, the highest degrees cost
      // far more compile time than they save.
      static constexpr Schedule<L> sched{};
      R[0] = F[L];
      for (int n = L - 1; n >= 0; --n) {
        for (int d = L - n; d >= 1; --d) {
          for (int kk = nhermsum(d - 1); kk < nhermsum(d); ++kk) {
            const V &XD = sched.dir[kk] == 0 ? X : (sched.dir[kk] == 1 ? Y : Z);
            if (sched.coef[kk] > 0)
              R[kk * NG] = XD * R[sched.src1[kk] * NG] +
                           double(sched.coef[kk]) * R[sched.src2[kk] * NG];
            else
              R[kk * NG] = XD * R[sched.src1[kk] * NG];
          }
        }
        R[0] = F[n];
      }
    }
  }
}

// One out-of-line copy per degree and target, shared by every (LA, LC) with
// LA + LC = L.
template <int L>
__attribute__((noinline)) void hermite_chunk_generic(const double *boys,
                                                     const PrimPair &pp,
                                                     const AuxView &a, int k0,
                                                     double *R) {
  hermite_chunk<L, false>(boys, pp, a, k0, R);
}

#if OCC_SIMD_X86_DISPATCH
template <int L>
__attribute__((target("avx2,fma"), noinline, flatten)) void
hermite_chunk_avx2(const double *boys, const PrimPair &pp, const AuxView &a,
                   int k0, double *R) {
  hermite_chunk<L, true>(boys, pp, a, k0, R);
}
#endif

template <int L, bool Avx2>
__attribute__((always_inline)) inline void
hermite(const double *boys, const PrimPair &pp, const AuxView &a, int k0,
        double *R) {
#if OCC_SIMD_X86_DISPATCH
  if constexpr (Avx2) {
    hermite_chunk_avx2<L>(boys, pp, a, k0, R);
    return;
  }
#endif
  hermite_chunk_generic<L>(boys, pp, a, k0, R);
}

template <bool Avx2> constexpr int vector_doubles() {
  return Avx2 ? group_width<Avx2>() : 2;
}

// Contraction vectors: VB doubles each, NV of them per chunk of NP aux
// primitives, with accumulators held for HB rows at a time.
template <bool Avx2> struct Blocking {
  static constexpr int VB = vector_doubles<Avx2>();
  static constexpr int NV = NP / VB;
  static constexpr int HB = NV >= 4 ? 3 : 12 / NV;
  using V = typename ints::simd::Vec<VB>::D;
  using VU = typename ints::simd::Vec<VB>::DU;
};

// Forward contraction of rows c0 .. c0 + NB - 1 of the aux side.
template <int LA, int LC, bool Avx2, int NB>
__attribute__((always_inline)) inline void
fwd_block(const RIndex<LA, LC> &ri, int c0, const double *R, const double *Xi,
          double *Yk, int npad) {
  using W = Blocking<Avx2>;
  using V = typename W::V;
  using VU = typename W::VU;
  constexpr int NV = W::NV, VB = W::VB, NA = nhermsum(LA), NC = nhermsum(LC);
  V acc[NB][NV];
  for (int j = 0; j < NB; ++j)
    for (int v = 0; v < NV; ++v)
      acc[j][v] = *reinterpret_cast<const VU *>(Yk + (c0 + j) * npad + v * VB);
  for (int t = 0; t < NA; ++t) {
    const double x = Xi[t];
    const short *row = ri.idx.data() + t * NC + c0;
    for (int j = 0; j < NB; ++j)
      for (int v = 0; v < NV; ++v)
        acc[j][v] += x * *reinterpret_cast<const VU *>(R + row[j] * NP + v * VB);
  }
  for (int j = 0; j < NB; ++j)
    for (int v = 0; v < NV; ++v)
      *reinterpret_cast<VU *>(Yk + (c0 + j) * npad + v * VB) = acc[j][v];
}

template <int LA, int LC, bool Avx2>
__attribute__((always_inline)) inline void
fwd_body(const double *boys, const PrimPair *pp, int nprim, const double *X,
         int ldx, const AuxView &a, int kend, double *Y) {
  constexpr int L = LA + LC, NC = nhermsum(LC);
  constexpr int HB = Blocking<Avx2>::HB;
  constexpr int NFULL = NC / HB * HB, NREM = NC - NFULL;
  static constexpr RIndex<LA, LC> ri{};
  alignas(64) double R[nhermsum(L) * NP];
  for (int ip = 0; ip < nprim; ++ip) {
    const double *Xi = X + ip * ldx;
    for (int k0 = 0; k0 < kend; k0 += NP) {
      hermite<L, Avx2>(boys, pp[ip], a, k0, R);
      double *Yk = Y + k0;
      for (int c0 = 0; c0 < NFULL; c0 += HB)
        fwd_block<LA, LC, Avx2, HB>(ri, c0, R, Xi, Yk, a.npad);
      if constexpr (NREM > 0)
        fwd_block<LA, LC, Avx2, NREM>(ri, NFULL, R, Xi, Yk, a.npad);
    }
  }
}

// Backward contraction into rows t0 .. t0 + NB - 1 of the pair side.
template <int LA, int LC, bool Avx2, int NB>
__attribute__((always_inline)) inline void
bwd_block(const RIndex<LA, LC> &ri, int t0, const double *R, const double *Tk,
          int npad, double *acc_mem) {
  using W = Blocking<Avx2>;
  using V = typename W::V;
  using VU = typename W::VU;
  constexpr int NV = W::NV, VB = W::VB, NC = nhermsum(LC);
  V acc[NB][NV];
  for (int j = 0; j < NB; ++j)
    for (int v = 0; v < NV; ++v)
      acc[j][v] = *reinterpret_cast<const VU *>(acc_mem + (t0 + j) * NP + v * VB);
  for (int c = 0; c < NC; ++c) {
    V tv[NV];
    for (int v = 0; v < NV; ++v)
      tv[v] = *reinterpret_cast<const VU *>(Tk + c * npad + v * VB);
    for (int j = 0; j < NB; ++j) {
      const int r = ri.idx[(t0 + j) * NC + c];
      for (int v = 0; v < NV; ++v)
        acc[j][v] += tv[v] * *reinterpret_cast<const VU *>(R + r * NP + v * VB);
    }
  }
  for (int j = 0; j < NB; ++j)
    for (int v = 0; v < NV; ++v)
      *reinterpret_cast<VU *>(acc_mem + (t0 + j) * NP + v * VB) = acc[j][v];
}

template <int LA, int LC, bool Avx2>
__attribute__((always_inline)) inline void
bwd_body(const double *boys, const PrimPair *pp, int nprim, const AuxView &a,
         int kend, const double *T, double *U, int ldu) {
  constexpr int L = LA + LC, NA = nhermsum(LA);
  constexpr int HB = Blocking<Avx2>::HB;
  constexpr int NFULL = NA / HB * HB, NREM = NA - NFULL;
  static constexpr RIndex<LA, LC> ri{};
  alignas(64) double R[nhermsum(L) * NP];
  alignas(64) double acc[NA * NP];
  for (int ip = 0; ip < nprim; ++ip) {
    std::fill(acc, acc + NA * NP, 0.0);
    for (int k0 = 0; k0 < kend; k0 += NP) {
      hermite<L, Avx2>(boys, pp[ip], a, k0, R);
      const double *Tk = T + k0;
      for (int t0 = 0; t0 < NFULL; t0 += HB)
        bwd_block<LA, LC, Avx2, HB>(ri, t0, R, Tk, a.npad, acc);
      if constexpr (NREM > 0)
        bwd_block<LA, LC, Avx2, NREM>(ri, NFULL, R, Tk, a.npad, acc);
    }
    double *Ui = U + ip * ldu;
    for (int t = 0; t < NA; ++t) {
      double s = 0.0;
      for (int i = 0; i < NP; ++i)
        s += acc[t * NP + i];
      Ui[t] += s;
    }
  }
}

template <int LA, int LC>
void fwd_generic(const double *boys, const PrimPair *pp, int nprim,
                 const double *X, int ldx, const AuxView &a, int kend,
                 double *Y) {
  fwd_body<LA, LC, false>(boys, pp, nprim, X, ldx, a, kend, Y);
}

template <int LA, int LC>
void bwd_generic(const double *boys, const PrimPair *pp, int nprim,
                 const AuxView &a, int kend, const double *T, double *U,
                 int ldu) {
  bwd_body<LA, LC, false>(boys, pp, nprim, a, kend, T, U, ldu);
}

#if OCC_SIMD_X86_DISPATCH
template <int LA, int LC>
__attribute__((target("avx2,fma"))) void
fwd_avx2(const double *boys, const PrimPair *pp, int nprim, const double *X,
         int ldx, const AuxView &a, int kend, double *Y) {
  fwd_body<LA, LC, true>(boys, pp, nprim, X, ldx, a, kend, Y);
}

template <int LA, int LC>
__attribute__((target("avx2,fma"))) void
bwd_avx2(const double *boys, const PrimPair *pp, int nprim, const AuxView &a,
         int kend, const double *T, double *U, int ldu) {
  bwd_body<LA, LC, true>(boys, pp, nprim, a, kend, T, U, ldu);
}
#endif

constexpr int NTAB = (MAX_LA + 1) * (MAX_LC + 1);

template <int K> constexpr bool compiled() {
  return K / (MAX_LC + 1) + K % (MAX_LC + 1) <= MAX_L;
}

template <bool Avx2, size_t... K>
constexpr std::array<FwdFn, NTAB> make_fwd(std::index_sequence<K...>) {
#if OCC_SIMD_X86_DISPATCH
  if constexpr (Avx2)
    return {(compiled<K>() ? &fwd_avx2<K / (MAX_LC + 1) * compiled<K>(),
                                       K % (MAX_LC + 1) * compiled<K>()>
                           : nullptr)...};
  else
#endif
    return {(compiled<K>() ? &fwd_generic<K / (MAX_LC + 1) * compiled<K>(),
                                          K % (MAX_LC + 1) * compiled<K>()>
                           : nullptr)...};
}

template <bool Avx2, size_t... K>
constexpr std::array<BwdFn, NTAB> make_bwd(std::index_sequence<K...>) {
#if OCC_SIMD_X86_DISPATCH
  if constexpr (Avx2)
    return {(compiled<K>() ? &bwd_avx2<K / (MAX_LC + 1) * compiled<K>(),
                                       K % (MAX_LC + 1) * compiled<K>()>
                           : nullptr)...};
  else
#endif
    return {(compiled<K>() ? &bwd_generic<K / (MAX_LC + 1) * compiled<K>(),
                                          K % (MAX_LC + 1) * compiled<K>()>
                           : nullptr)...};
}

bool use_avx2() {
#if OCC_SIMD_X86_DISPATCH
  return ints::simd::cpu_has_avx2_fma();
#else
  return false;
#endif
}

} // namespace

FwdFn forward_kernel(int la, int lc) {
  if (la < 0 || la > MAX_LA || lc < 0 || lc > MAX_LC)
    return nullptr;
  static const auto gen = make_fwd<false>(std::make_index_sequence<NTAB>{});
#if OCC_SIMD_X86_DISPATCH
  static const auto avx = make_fwd<true>(std::make_index_sequence<NTAB>{});
  if (use_avx2())
    return avx[la * (MAX_LC + 1) + lc];
#endif
  return gen[la * (MAX_LC + 1) + lc];
}

BwdFn backward_kernel(int la, int lc) {
  if (la < 0 || la > MAX_LA || lc < 0 || lc > MAX_LC)
    return nullptr;
  static const auto gen = make_bwd<false>(std::make_index_sequence<NTAB>{});
#if OCC_SIMD_X86_DISPATCH
  static const auto avx = make_bwd<true>(std::make_index_sequence<NTAB>{});
  if (use_avx2())
    return avx[la * (MAX_LC + 1) + lc];
#endif
  return gen[la * (MAX_LC + 1) + lc];
}

} // namespace occ::qm::srij
