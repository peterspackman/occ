#include <occ/ints/esp.h>
#include <occ/ints/kernels.h>
#include <occ/ints/rints_specialized.h>
#include <occ/gto/gto.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <type_traits>

// The structure-of-arrays kernels are written with vector types (see
// boys_simd.h); on baseline x86-64 builds they are compiled a second time for
// AVX2/FMA (function-level target attributes) and the CPU picks one.
#include <occ/ints/boys_simd.h>

namespace occ::ints {

// Number of spherical harmonics for angular momentum l
constexpr int nsph(int l) { return 2 * l + 1; }

// Build combined transformation matrix T_ab = kron(T_a, T_b)
// Transforms from Cartesian product basis to spherical product basis
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
build_cart_to_sph_transform(int la, int lb) {
    using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Get individual transformation matrices
    occ::Mat T_a = occ::gto::cartesian_to_spherical_transformation_matrix(la);
    occ::Mat T_b = occ::gto::cartesian_to_spherical_transformation_matrix(lb);

    int nsph_a = nsph(la);
    int nsph_b = nsph(lb);
    int ncart_a = ncart(la);
    int ncart_b = ncart(lb);

    // Build Kronecker product: T_ab[i*nsph_b + j, k*ncart_b + l] = T_a[i,k] * T_b[j,l]
    Mat T_ab(nsph_a * nsph_b, ncart_a * ncart_b);

    for (int i = 0; i < nsph_a; ++i) {
        for (int j = 0; j < nsph_b; ++j) {
            for (int k = 0; k < ncart_a; ++k) {
                for (int l = 0; l < ncart_b; ++l) {
                    T_ab(i * nsph_b + j, k * ncart_b + l) =
                        static_cast<T>(T_a(i, k) * T_b(j, l));
                }
            }
        }
    }

    return T_ab;
}

// Type aliases matching ESPEvaluator
template <typename T>
using Mat3N = Eigen::Matrix<T, 3, Eigen::Dynamic>;
template <typename T>
using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Type-erased shell pair storage
template <typename T>
struct ShellPairHolder {
    virtual ~ShellPairHolder() = default;
    virtual int la() const = 0;
    virtual int lb() const = 0;
    virtual int nab() const = 0;
    virtual int nherm() const = 0;
    virtual bool is_spherical() const = 0;
    virtual void evaluate(const Eigen::Ref<const Mat3N<T>>& C,
                          const T* boys_table,
                          Eigen::Ref<MatRM<T>> integrals,
                          Eigen::Ref<MatRM<T>> workspace,
                          T omega) const = 0;
    virtual void evaluate_overlap(T* integrals) const = 0;
    virtual void evaluate_soa(const T *x, const T *y, const T *z, int npts,
                              const T *boys_table, T *V, int ldv,
                              T omega) const = 0;
    virtual void evaluate_soa_contract(const T *x, const T *y, const T *z,
                                       int npts, const T *boys_table,
                                       const T *F, T *G, int ld, int bf_p,
                                       int bf_q, bool both, T omega) const = 0;

    // Screening support: pair center and extent based on most diffuse primitive
    virtual std::array<T, 3> pair_center() const = 0;
    virtual T pair_extent() const = 0;
    virtual T pair_estimate() const = 0;
    virtual std::array<T, 2> pair_far_estimate() const = 0;
};

// Grid-point-independent bound on a pair's ESP integrals: the integral
// prefactor 2 pi / p * c_a c_b * exp(-mu R_AB^2), with the Boys factor
// F_0 <= 1 bounded out. exp(-mu R_AB^2) lives in the (0,0,0) Hermite column
// of the E matrix. Pairs sharing an origin drop that factor entirely: their
// monopole coefficient can vanish by symmetry (an s-p pair, say) while the
// ESP integral does not, so bounding E by 1 there keeps them unscreened.
template <typename T>
T primitive_pair_estimate(const ShellPairData<T> &data, int nab_val,
                          int nherm_val) {
    const bool same_origin = (data.A == data.B);
    T est = T(0);
    for (const auto& prim : data.primitives) {
        T emax = T(1);
        if (!same_origin) {
            emax = T(0);
            for (int ab = 0; ab < nab_val; ++ab)
                emax = std::max(emax, std::abs(prim.E_matrix[ab * nherm_val]));
        }
        est += std::abs(prim.prefactor) * emax;
    }
    return est;
}

// Far-field companion of primitive_pair_estimate: each primitive's Boys
// factor is also bounded by F_0(T) <= sqrt(pi / T) / 2, T = p |P - C|^2, so
// beyond the primitives' centres the pair's ESP is at most
//     q1 / (|C - center| - spread),   q1 = sum_prim est_prim sqrt(pi / p) / 2,
// spread the farthest primitive centre P from `center`. Returns {q1, spread}.
template <typename T>
std::array<T, 2> primitive_pair_far_estimate(const ShellPairData<T> &data,
                                             int nab_val, int nherm_val,
                                             const std::array<T, 3> &center) {
    const bool same_origin = (data.A == data.B);
    T q1 = T(0), spread = T(0);
    for (const auto& prim : data.primitives) {
        T emax = T(1);
        if (!same_origin) {
            emax = T(0);
            for (int ab = 0; ab < nab_val; ++ab)
                emax = std::max(emax, std::abs(prim.E_matrix[ab * nherm_val]));
        }
        q1 += std::abs(prim.prefactor) * emax * T(0.886226925452758) /
              std::sqrt(prim.p);
        const T dx = prim.Px - center[0], dy = prim.Py - center[1],
                dz = prim.Pz - center[2];
        spread = std::max(spread, std::sqrt(dx * dx + dy * dy + dz * dz));
    }
    return {q1, spread};
}


// ============================================================================
// Structure-of-arrays ESP kernel
//
// The row-major evaluate() path computes the Hermite R-integrals point by point
// and then contracts them with the E-matrix through an Eigen GEMM per
// primitive. At the sizes involved (a few to a few dozen columns, one call per
// primitive pair) that GEMM is all overhead, and the point-major output forces
// the caller into a per-point mat-vec. Here every intermediate is laid out
// point-fastest, in fixed chunks of NP points, so the Hermite recursion, the
// E-contraction and the caller's contraction all vectorise over points.
// ============================================================================

namespace {

// Recursion schedule for the Hermite Coulomb integrals R^{(n)}_{tuv}. Entry k
// (canonical Hermite order, so the entries of total degree <= K are exactly the
// first nhermsum(K)) is built along one Cartesian direction d from the level
// above:  R^{(n)}_k = X_d R^{(n+1)}_{k - e_d} + (c-1) R^{(n+1)}_{k - 2 e_d}.
template <int L> struct HermiteSchedule {
  static constexpr int N = nhermsum(L);
  std::array<int, N> dir{}, src1{}, src2{}, coef{};
  constexpr HermiteSchedule() {
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

// Offsets of each recursion level n in the scratch buffer: level n holds the
// nhermsum(L - n) entries of total degree <= L - n.
template <int L> constexpr std::array<int, L + 2> level_offsets() {
  std::array<int, L + 2> off{};
  off[0] = 0;
  for (int n = 0; n <= L; ++n)
    off[n + 1] = off[n] + nhermsum(L - n);
  return off;
}

using simd::boys_point;

// The fused gradient pass keeps 32-point chunks up to L = 5 (d-d pairs): its
// per-column loop overhead is paid once per chunk, which outweighs R
// outgrowing L1 there.
template <int L> constexpr int fused_chunk() { return L <= 5 ? 32 : 16; }
template <int L> constexpr int soa_chunk() {
  return L <= 4 ? 32 : (L <= 7 ? 16 : 8);
}

// Width in bytes of the kernels' explicit point vectors (GCC/Clang vector
// extensions): AVX-512 when the build targets it, AVX2 in the dispatched
// copies, otherwise 16 (SSE2, NEON).
template <bool Avx2> constexpr int vector_bytes() {
#if defined(__AVX512F__)
  return 64;
#else
  return Avx2 ? 32 : 16;
#endif
}

// N lanes of T; VU is the same vector at any element alignment, for loads
// and stores in place.
template <typename T, int N> struct PointVector {
  typedef T V __attribute__((vector_size(N * sizeof(T))));
  typedef T VU __attribute__((vector_size(N * sizeof(T)), aligned(sizeof(T)),
                              may_alias));
};

// Boys values F_m(T_i), m = 0..L, for one chunk of NP points, from the point
// offsets X, Y, Z relative to the primitive pair's centre: Fm[m * NP + i].
template <typename T, int L, int NP>
inline void boys_chunk_generic(const T *table, T p, const T *X, const T *Y,
                               const T *Z, T *Fm) {
  for (int i = 0; i < NP; ++i) {
    const T Tp = p * (X[i] * X[i] + Y[i] * Y[i] + Z[i] * Z[i]);
    boys_point<T, L + 1>(table, Tp, Fm + i, NP);
  }
}

// The same, four points at a time (see simd::boys).
template <int L, int NP>
OCC_SIMD_INLINE void boys_chunk_simd(const double *table, double p,
                                     const double *X, const double *Y,
                                     const double *Z, double *Fm) {
  static_assert(NP % 4 == 0);
  using V = simd::Vec<4>::D;
  using VU = simd::Vec<4>::DU;
  for (int i = 0; i < NP; i += 4) {
    const V x = *reinterpret_cast<const VU *>(X + i);
    const V y = *reinterpret_cast<const VU *>(Y + i);
    const V z = *reinterpret_cast<const VU *>(Z + i);
    V f[L + 1];
    simd::boys<L>(table, p * (x * x + y * y + z * z), f);
    for (int m = 0; m <= L; ++m)
      *reinterpret_cast<VU *>(Fm + m * NP + i) = f[m];
  }
}

// The nonzero entries of each primitive's prefactor-scaled E-matrix, row by
// row: for primitive ip and row ab, entries start[ip * nab + ab] up to
// start[ip * nab + ab + 1] of (t, c). Cartesian E-matrices are mostly
// structural zeros (E_ab,tuv needs t <= ia + ib, and so on), so this halves the
// contraction work for d shells and lets it run from registers.
template <typename T> struct SparseE {
  std::vector<int> start;
  std::vector<int> t;
  std::vector<T> c;
};

template <typename T>
SparseE<T> build_sparse_e(const ShellPairData<T> &data, int nab, int nherm) {
  SparseE<T> out;
  out.start.push_back(0);
  for (const auto &prim : data.primitives) {
    for (int ab = 0; ab < nab; ++ab) {
      for (int t = 0; t < nherm; ++t) {
        const T e = prim.E_matrix[ab * nherm + t];
        if (e != T(0)) {
          out.t.push_back(t);
          out.c.push_back(prim.prefactor * e);
        }
      }
      out.start.push_back(static_cast<int>(out.t.size()));
    }
  }
  return out;
}

// Hermite integrals R_t(i), t of total degree <= L, of one primitive pair for
// one chunk of NP points, into R (level 0 of the recursion is R[t * NP + i]).
// Forced inline, so it is vectorised for whichever target its caller has.
template <typename T, int L, int NP = soa_chunk<L>()>
__attribute__((always_inline)) inline void
soa_prim_hermite(const PrimitivePairData<T> &prim, const T *boys_table,
                 const T *cx, const T *cy, const T *cz, T *R, T omega) {
  constexpr auto off = level_offsets<L>();
  static constexpr HermiteSchedule<L> sched{};
  alignas(64) T X[NP], Y[NP], Z[NP];
  alignas(64) T Fm[(L + 1) * NP];
  {
    T p_used = prim.p;
    T rs = T(1);
    if (omega > T(0)) {
      const T w2 = omega * omega;
      const T scale = w2 / (prim.p + w2);
      p_used = prim.p * scale;
      rs = std::sqrt(scale);
    }

    for (int i = 0; i < NP; ++i) {
      X[i] = prim.Px - cx[i];
      Y[i] = prim.Py - cy[i];
      Z[i] = prim.Pz - cz[i];
    }
    if constexpr (std::is_same_v<T, double>)
      boys_chunk_simd<L, NP>(boys_table, p_used, X, Y, Z, Fm);
    else
      boys_chunk_generic<T, L, NP>(boys_table, p_used, X, Y, Z, Fm);

    // R^{(n)}_{000} = rs * (-2p)^n F_n
    {
      T pw = rs;
      const T neg_2p = T(-2) * p_used;
      for (int n = 0; n <= L; ++n) {
        T *dst = R + off[n] * NP;
        const T *f = Fm + n * NP;
        for (int i = 0; i < NP; ++i)
          dst[i] = pw * f[i];
        pw *= neg_2p;
      }
    }
    for (int n = L - 1; n >= 0; --n) {
      T *dst = R + off[n] * NP;
      const T *src = R + off[n + 1] * NP;
      const int nk = nhermsum(L - n);
      for (int k = 1; k < nk; ++k) {
        const T *XD = sched.dir[k] == 0 ? X : (sched.dir[k] == 1 ? Y : Z);
        const T *s1 = src + sched.src1[k] * NP;
        T *d = dst + k * NP;
        if (sched.coef[k] > 0) {
          const T c = T(sched.coef[k]);
          const T *s2 = src + sched.src2[k] * NP;
          for (int i = 0; i < NP; ++i)
            d[i] = XD[i] * s1[i] + c * s2[i];
        } else {
          for (int i = 0; i < NP; ++i)
            d[i] = XD[i] * s1[i];
        }
      }
    }
  }
}

template <typename T, int L, bool Avx2>
__attribute__((always_inline)) inline void
soa_chunk_body(const ShellPairData<T> &data, const SparseE<T> &se, int nab,
               const T *boys_table, const T *cx, const T *cy, const T *cz,
               T *V, int ldv, T omega) {
  constexpr int NP = soa_chunk<L>();
  constexpr auto off = level_offsets<L>();
  alignas(64) T R[off[L + 1] * NP];

  for (size_t ip = 0; ip < data.primitives.size(); ++ip) {
    soa_prim_hermite<T, L>(data.primitives[ip], boys_table, cx, cy, cz,
                                 R, omega);

    // E-contraction, level 0 holds R^{(0)}_t for all t: each row's nonzero
    // entries accumulated over the whole chunk in explicit vectors, so the
    // row's loop overhead is paid once per chunk and its accumulators are
    // independent.
    constexpr int VB0 = vector_bytes<Avx2>() / int(sizeof(T));
    constexpr int VB = NP < VB0 ? NP : VB0;
    constexpr int NV = NP / VB;
    constexpr int NB = VB * NV;
    using Vv = typename PointVector<T, VB>::V;
    using VU = typename PointVector<T, VB>::VU;
    const int *rs_start = se.start.data() + ip * nab;
    const int *ts = se.t.data();
    const T *cs = se.c.data();
    for (int i0 = 0; i0 < NP; i0 += NB) {
      for (int ab = 0; ab < nab; ++ab) {
        VU *v = reinterpret_cast<VU *>(V + ab * ldv + i0);
        Vv acc[NV];
        for (int j = 0; j < NV; ++j)
          acc[j] = v[j];
        for (int k = rs_start[ab]; k < rs_start[ab + 1]; ++k) {
          const T c = cs[k];
          const VU *r = reinterpret_cast<const VU *>(R + ts[k] * NP + i0);
          for (int j = 0; j < NV; ++j)
            acc[j] += c * r[j];
        }
        for (int j = 0; j < NV; ++j)
          v[j] = acc[j];
      }
    }
  }
}

template <typename T, int L>
void soa_chunk_generic(const ShellPairData<T> &data, const SparseE<T> &se,
                       int nab, const T *boys_table, const T *cx, const T *cy,
                       const T *cz, T *V, int ldv, T omega) {
  soa_chunk_body<T, L, false>(data, se, nab, boys_table, cx, cy, cz, V, ldv,
                              omega);
}

#if OCC_SIMD_X86_DISPATCH
template <typename T, int L>
__attribute__((target("avx2,fma"))) void
soa_chunk_avx2(const ShellPairData<T> &data, const SparseE<T> &se, int nab,
               const T *boys_table, const T *cx, const T *cy, const T *cz,
               T *V, int ldv, T omega) {
  soa_chunk_body<T, L, true>(data, se, nab, boys_table, cx, cy, cz, V, ldv,
                             omega);
}

using simd::cpu_has_avx2_fma;
#endif

template <typename T, int L>
void soa_eval(const ShellPairData<T> &data, const SparseE<T> &se, int nab,
              const T *boys_table, const T *x, const T *y, const T *z,
              int npts, T *V, int ldv, T omega) {
  constexpr int NP = soa_chunk<L>();
  for (int ab = 0; ab < nab; ++ab)
    std::fill(V + ab * ldv, V + ab * ldv + npts, T(0));
#if OCC_SIMD_X86_DISPATCH
  if (cpu_has_avx2_fma()) {
    for (int c = 0; c < npts; c += NP)
      soa_chunk_avx2<T, L>(data, se, nab, boys_table, x + c, y + c, z + c, V + c,
                           ldv, omega);
    return;
  }
#endif
  for (int c = 0; c < npts; c += NP)
    soa_chunk_generic<T, L>(data, se, nab, boys_table, x + c, y + c, z + c, V + c,
                            ldv, omega);
}

// Values contracted as they are formed, for the seminumerical exchange: per
// chunk of NP points the pair's integrals V_ab go to a small buffer that stays
// in L1, and straight into
//   G_q,b -= sum_a F_p,a V_ab   and, when both, G_p,a -= sum_b F_q,b V_ab,
// F and G column-major with leading dimension ld (columns bf + a). The full
// nab x npts V is never written.
template <typename T, int L, int NABMAX, bool Avx2>
__attribute__((always_inline)) inline void
soa_chunk_contract_body(const ShellPairData<T> &data, const SparseE<T> &se,
                        int nab, const T *boys_table, const T *cx,
                        const T *cy, const T *cz, const T *F, T *G, int ld,
                        int bf_p, int np, int bf_q, int nq, bool both,
                        T omega) {
  constexpr int NP = soa_chunk<L>();
  constexpr int VB0 = vector_bytes<Avx2>() / int(sizeof(T));
  constexpr int VB = NP < VB0 ? NP : VB0;
  constexpr int NV = NP / VB;
  using Vv = typename PointVector<T, VB>::V;
  using VU = typename PointVector<T, VB>::VU;
  alignas(64) T Vb[NABMAX * NP];
  for (int i = 0; i < nab * NP; ++i)
    Vb[i] = T(0);
  soa_chunk_body<T, L, Avx2>(data, se, nab, boys_table, cx, cy, cz, Vb, NP,
                             omega);
  for (int b = 0; b < nq; ++b) {
    Vv acc[NV] = {};
    for (int a = 0; a < np; ++a) {
      const VU *f = reinterpret_cast<const VU *>(F + size_t(bf_p + a) * ld);
      const VU *v = reinterpret_cast<const VU *>(Vb + (a * nq + b) * NP);
      for (int j = 0; j < NV; ++j)
        acc[j] += f[j] * v[j];
    }
    VU *g = reinterpret_cast<VU *>(G + size_t(bf_q + b) * ld);
    for (int j = 0; j < NV; ++j)
      g[j] -= acc[j];
  }
  if (!both)
    return;
  for (int a = 0; a < np; ++a) {
    Vv acc[NV] = {};
    for (int b = 0; b < nq; ++b) {
      const VU *f = reinterpret_cast<const VU *>(F + size_t(bf_q + b) * ld);
      const VU *v = reinterpret_cast<const VU *>(Vb + (a * nq + b) * NP);
      for (int j = 0; j < NV; ++j)
        acc[j] += f[j] * v[j];
    }
    VU *g = reinterpret_cast<VU *>(G + size_t(bf_p + a) * ld);
    for (int j = 0; j < NV; ++j)
      g[j] -= acc[j];
  }
}

template <typename T, int L, int NABMAX>
void soa_chunk_contract_generic(const ShellPairData<T> &data,
                                const SparseE<T> &se, int nab,
                                const T *boys_table, const T *cx, const T *cy,
                                const T *cz, const T *F, T *G, int ld,
                                int bf_p, int np, int bf_q, int nq, bool both,
                                T omega) {
  soa_chunk_contract_body<T, L, NABMAX, false>(data, se, nab, boys_table, cx, cy, cz,
                                       F, G, ld, bf_p, np, bf_q, nq, both,
                                       omega);
}

#if OCC_SIMD_X86_DISPATCH
template <typename T, int L, int NABMAX>
__attribute__((target("avx2,fma"))) void
soa_chunk_contract_avx2(const ShellPairData<T> &data, const SparseE<T> &se,
                        int nab, const T *boys_table, const T *cx,
                        const T *cy, const T *cz, const T *F, T *G, int ld,
                        int bf_p, int np, int bf_q, int nq, bool both,
                        T omega) {
  soa_chunk_contract_body<T, L, NABMAX, true>(data, se, nab, boys_table, cx, cy, cz,
                                      F, G, ld, bf_p, np, bf_q, nq, both,
                                      omega);
}
#endif

template <typename T, int L, int NABMAX>
void soa_eval_contract(const ShellPairData<T> &data, const SparseE<T> &se,
                       int nab, const T *boys_table, const T *x, const T *y,
                       const T *z, int npts, const T *F, T *G, int ld,
                       int bf_p, int np, int bf_q, int nq, bool both,
                       T omega) {
  constexpr int NP = soa_chunk<L>();
#if OCC_SIMD_X86_DISPATCH
  if (cpu_has_avx2_fma()) {
    for (int c = 0; c < npts; c += NP)
      soa_chunk_contract_avx2<T, L, NABMAX>(data, se, nab, boys_table, x + c, y + c,
                                    z + c, F + c, G + c, ld, bf_p, np, bf_q,
                                    nq, both, omega);
    return;
  }
#endif
  for (int c = 0; c < npts; c += NP)
    soa_chunk_contract_generic<T, L, NABMAX>(data, se, nab, boys_table, x + c, y + c,
                                     z + c, F + c, G + c, ld, bf_p, np, bf_q,
                                     nq, both, omega);
}

// Value and contracted centre derivatives of a derivative pair in one pass.
//
// Per primitive the pair's product is sum_t E_ab,t Lambda_t(r; P), and with
// the 1D identity E^{i+1,j}_t = E^{ij}_{t-1} / 2p + X_PA E^{ij}_t
// + (t+1) E^{ij}_{t+1}, the derivative in one centre X (exponent x) is
//   d/dX_d [ab] = 2x X_PX,d sum_t E_ab,t R_t + 2x sum_t t_d E_ab,t R_{t-e_d}
//               - sum_t K^d_ab,t R_t + (x/p) sum_t E_ab,t R_{t+e_d},
// K^d_ab = a_d E^{a-e_d, b} (the pair with X's shell lowered; zero for an s
// shell), while the two centres together move P only:
//   (d/dA_d + d/dB_d) [ab] = sum_t E_ab,t R_{t+e_d}.
// Contracting with the weights first, into the Hermite densities
// g_t = sum_ab W_ab E_ab,t and k^d_t = sum_ab W_ab K^d_ab,t, leaves per point
// one E-sized and three (sparser) K-sized contractions and a few Hermite dots.
// X is the centre with the lower angular momentum, so K is as small as it gets.
template <typename T> struct FusedE {
  int nab = 0, nv = 0, ne = 0;
  bool explicit_b = false; // X = B (else A)
  // value rows, pattern shared by every primitive: row ab has entries
  // vstart[ab] .. vstart[ab + 1] of Hermite column vt; coefficient vc[ip * nv + k]
  std::vector<int> vstart, vt;
  std::vector<T> vc;
  // g by Hermite column t < nhv: entries estart[t] .. estart[t + 1] over rows
  // eab, coefficients ec[ip * ne + k]
  std::vector<int> estart, eab;
  std::vector<T> ec;
  // The lowered-shell term through the lowered pair's own integrals:
  //   sum_ab W_ab K^d_ab . R = sum_{s,o,c'} T^d_{s c'} W_{(s,o)} Alow_{c' o},
  // Alow_{c'o} = sum_t El_{c'o,t} R_t the ESP integrals of the pair (X's
  // shell lowered to Cartesian c', the other shell's functions o), summed
  // over primitives (the term carries no per-primitive factor), and
  // T^d_{s c'} = (c'_d + 1) S_{s, c' + e_d}, S X's Cartesian-to-function
  // transform. Rows c' * no + o: entries lstart[r] .. lstart[r + 1] of
  // Hermite columns lt, coefficients lc[ip * nl + k].
  int nlow = 0, nl = 0, no = 0, nx = 0;
  std::vector<int> lstart, lt;
  std::vector<T> lc;
  // T^d entries: direction td, X function ts, lowered Cartesian tc, value tv
  std::vector<int> td, ts, tc;
  std::vector<T> tv;
  // per primitive: 2x, 2x X_PX (3), x / p
  std::vector<T> scal;
};

// GCC 13.3 at -O3 (with AVX-512) miscompiles this kernel through value-range
// propagation: one direction's lowered-Hermite term (t_d > 0) comes out
// wrong, varying with code shape and memory layout; -O2, other compilers and
// -fno-tree-vrp are correct (cf. GCC PR 109934, fixed after 13.3.0). VRP buys
// nothing here, so it is off for the fused kernel.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vrp")
#endif
template <typename T, int L, bool HasK, bool Avx2>
__attribute__((always_inline)) inline void
soa_fused_chunk_body(const ShellPairData<T> &data, const FusedE<T> &fe,
                     int ncentres, const T *boys_table, const T *cx,
                     const T *cy, const T *cz, T *A, int lda, const T *W,
                     int ldw, T *out, T omega) {
  constexpr int NP = fused_chunk<L>();
  constexpr auto off = level_offsets<L>();
  // Points go in explicit vectors (GCC/Clang vector extensions) of VB lanes,
  // up to 4 of them per block (a whole 32-point chunk with AVX-512): each
  // column's loop overhead is paid once per block, and the 7 dot accumulators
  // mostly stay in registers (the rest spill cheaply).
  constexpr int NVMAX = 4;
  constexpr int VB0 = vector_bytes<Avx2>() / int(sizeof(T));
  constexpr int VB = NP < VB0 ? NP : VB0;
  constexpr int NV = NP / VB < NVMAX ? NP / VB : NVMAX;
  constexpr int NB = VB * NV;
  using V = typename PointVector<T, VB>::V;
  using VU = typename PointVector<T, VB>::VU;
  auto at = [](const T *p) -> const VU & {
    return *reinterpret_cast<const VU *>(p);
  };
  alignas(64) T R[off[L + 1] * NP];
  alignas(64) T sX[3][NP] = {}, sT[3][NP] = {};
  const int nab = fe.nab;
  const int *vstart = fe.vstart.data(), *vt = fe.vt.data();
  const int *estart = fe.estart.data(), *eab = fe.eab.data();
  const int *lstart = fe.lstart.data(), *lt = fe.lt.data();
  // the lowered pair's integrals, summed over primitives (HasK only)
  static thread_local std::vector<T> alow_buf;
  T *Alow = nullptr;
  if constexpr (HasK) {
    alow_buf.assign(static_cast<size_t>(fe.nlow) * NP, T(0));
    Alow = alow_buf.data();
  }

  for (size_t ip = 0; ip < data.primitives.size(); ++ip) {
    soa_prim_hermite<T, L, NP>(data.primitives[ip], boys_table, cx, cy,
                                     cz, R, omega);
    const T *vc = fe.vc.data() + ip * fe.nv;
    const T *ec = fe.ec.data() + ip * fe.ne;
    const T *lc = HasK ? fe.lc.data() + ip * fe.nl : nullptr;
    const T *sc = fe.scal.data() + ip * 5;
    const T c2x = sc[0], rx = sc[4];
    for (int i0 = 0; i0 < NP; i0 += NB) {
      for (int ab = 0; ab < nab; ++ab) {
        T *v = A + ab * lda + i0;
        V acc[NV];
        for (int j = 0; j < NV; ++j)
          acc[j] = at(v + j * VB);
        for (int k = vstart[ab]; k < vstart[ab + 1]; ++k) {
          const T c = vc[k];
          const T *r = R + vt[k] * NP + i0;
          for (int j = 0; j < NV; ++j)
            acc[j] += c * at(r + j * VB);
        }
        for (int j = 0; j < NV; ++j)
          *reinterpret_cast<VU *>(v + j * VB) = acc[j];
      }
      if constexpr (HasK) {
        for (int r = 0; r < fe.nlow; ++r) {
          T *v = Alow + r * NP + i0;
          V acc[NV];
          for (int j = 0; j < NV; ++j)
            acc[j] = at(v + j * VB);
          for (int k = lstart[r]; k < lstart[r + 1]; ++k) {
            const T c = lc[k];
            const T *rr = R + lt[k] * NP + i0;
            for (int j = 0; j < NV; ++j)
              acc[j] += c * at(rr + j * VB);
          }
          for (int j = 0; j < NV; ++j)
            *reinterpret_cast<VU *>(v + j * VB) = acc[j];
        }
      }
      // Hermite densities of the weights, column by column, each dotted
      // with R as soon as it is formed:
      //   P0 = g . R,  M_d = 2x sum t_d g_t R_{t-e_d} - k^d . R,
      //   P2_d = g . R_{+e_d}
      V P0[NV] = {}, Mx[NV] = {}, My[NV] = {}, Mz[NV] = {}, Px[NV] = {},
        Py[NV] = {}, Pz[NV] = {};
      // columns in canonical Hermite order: degree, then t descending, then
      // u descending, so the column index is a running count
      int t = 0;
      for (int deg = 0; deg < L; ++deg)
        for (int tx = deg; tx >= 0; --tx)
          for (int ty = deg - tx; ty >= 0; --ty, ++t) {
            const int tz = deg - tx - ty;
            V g[NV] = {};
            for (int k = estart[t]; k < estart[t + 1]; ++k) {
              const T c = ec[k];
              const T *w = W + eab[k] * ldw + i0;
              for (int j = 0; j < NV; ++j)
                g[j] += c * at(w + j * VB);
            }
            const T *r = R + t * NP + i0;
            const T *rux = R + hermite_index(tx + 1, ty, tz) * NP + i0;
            const T *ruy = R + hermite_index(tx, ty + 1, tz) * NP + i0;
            const T *ruz = R + hermite_index(tx, ty, tz + 1) * NP + i0;
            for (int j = 0; j < NV; ++j) {
              P0[j] += g[j] * at(r + j * VB);
              Px[j] += g[j] * at(rux + j * VB);
              Py[j] += g[j] * at(ruy + j * VB);
              Pz[j] += g[j] * at(ruz + j * VB);
            }
            // 2x t_d g_t R_{t-e_d}, for t_d > 0
            if (tx > 0) {
              const T s = c2x * T(tx);
              const T *rd = R + hermite_index(tx - 1, ty, tz) * NP + i0;
              for (int j = 0; j < NV; ++j)
                Mx[j] += s * g[j] * at(rd + j * VB);
            }
            if (ty > 0) {
              const T s = c2x * T(ty);
              const T *rd = R + hermite_index(tx, ty - 1, tz) * NP + i0;
              for (int j = 0; j < NV; ++j)
                My[j] += s * g[j] * at(rd + j * VB);
            }
            if (tz > 0) {
              const T s = c2x * T(tz);
              const T *rd = R + hermite_index(tx, ty, tz - 1) * NP + i0;
              for (int j = 0; j < NV; ++j)
                Mz[j] += s * g[j] * at(rd + j * VB);
            }
          }
      const V *M[3] = {Mx, My, Mz}, *P2[3] = {Px, Py, Pz};
      for (int d = 0; d < 3; ++d) {
        const T c0 = sc[1 + d];
        for (int j = 0; j < NV; ++j) {
          *reinterpret_cast<VU *>(&sX[d][i0 + j * VB]) +=
              c0 * P0[j] + M[d][j] + rx * P2[d][j];
          *reinterpret_cast<VU *>(&sT[d][i0 + j * VB]) += P2[d][j];
        }
      }
    }
  }
  // lowered-shell term: sX_d -= sum T^d_{s c'} W_(s,o) Alow_(c' o)
  if constexpr (HasK) {
    const int no = fe.no, nx = fe.nx, nt = static_cast<int>(fe.tv.size());
    for (int i0 = 0; i0 < NP; i0 += NB) {
      V Md[3][NV] = {};
      for (int e = 0; e < nt; ++e) {
        const int d = fe.td[e], sx = fe.ts[e];
        const T tv = fe.tv[e];
        const T *al = Alow + fe.tc[e] * no * NP + i0;
        V acc[NV] = {};
        for (int o = 0; o < no; ++o) {
          const int ab = fe.explicit_b ? o * nx + sx : sx * no + o;
          const T *w = W + ab * ldw + i0;
          for (int j = 0; j < NV; ++j)
            acc[j] += at(w + j * VB) * at(al + o * NP + j * VB);
        }
        for (int j = 0; j < NV; ++j)
          Md[d][j] += tv * acc[j];
      }
      for (int d = 0; d < 3; ++d)
        for (int j = 0; j < NV; ++j)
          *reinterpret_cast<VU *>(&sX[d][i0 + j * VB]) -= Md[d][j];
    }
  }
  for (int d = 0; d < 3; ++d) {
    T x = T(0), tot = T(0);
    for (int q = 0; q < NP; ++q) {
      x += sX[d][q];
      tot += sT[d][q];
    }
    const T a = fe.explicit_b ? tot - x : x;
    out[d] += a;
    if (ncentres == 2)
      out[3 + d] += tot - a;
  }
}

template <typename T, int L, bool HasK>
void soa_fused_chunk_generic(const ShellPairData<T> &data, const FusedE<T> &fe,
                             int ncentres, const T *boys_table, const T *cx,
                             const T *cy, const T *cz, T *A, int lda,
                             const T *W, int ldw, T *out, T omega) {
  soa_fused_chunk_body<T, L, HasK, false>(data, fe, ncentres, boys_table, cx, cy, cz,
                                    A, lda, W, ldw, out, omega);
}

#if OCC_SIMD_X86_DISPATCH
template <typename T, int L, bool HasK>
__attribute__((target("avx2,fma"))) void
soa_fused_chunk_avx2(const ShellPairData<T> &data, const FusedE<T> &fe,
                     int ncentres, const T *boys_table, const T *cx,
                     const T *cy, const T *cz, T *A, int lda, const T *W,
                     int ldw, T *out, T omega) {
  soa_fused_chunk_body<T, L, HasK, true>(data, fe, ncentres, boys_table, cx, cy, cz,
                                   A, lda, W, ldw, out, omega);
}
#endif

template <typename T, int L, bool HasK>
void soa_fused_eval(const ShellPairData<T> &data, const FusedE<T> &fe,
                    int ncentres, const T *boys_table, const T *x, const T *y,
                    const T *z, int npts, T *A, int lda, const T *W, int ldw,
                    T *out, T omega) {
  constexpr int NP = fused_chunk<L>();
  for (int ab = 0; ab < fe.nab; ++ab)
    std::fill(A + ab * lda, A + ab * lda + npts, T(0));
#if OCC_SIMD_X86_DISPATCH
  if (cpu_has_avx2_fma()) {
    for (int c = 0; c < npts; c += NP)
      soa_fused_chunk_avx2<T, L, HasK>(data, fe, ncentres, boys_table, x + c, y + c,
                                 z + c, A + c, lda, W + c, ldw, out, omega);
    return;
  }
#endif
  for (int c = 0; c < npts; c += NP)
    soa_fused_chunk_generic<T, L, HasK>(data, fe, ncentres, boys_table, x + c, y + c,
                                  z + c, A + c, lda, W + c, ldw, out, omega);
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

} // namespace

// One shell pair at fixed (LA, LB). A spherical pair's E-matrices are
// already transformed to its spherical functions.
template <typename T, int LA, int LB, bool Spherical>
struct ShellPairHolderImpl : ShellPairHolder<T> {
    static constexpr int NAB =
        Spherical ? nsph(LA) * nsph(LB) : ncart(LA) * ncart(LB);
    static constexpr int NHERM = nhermsum(LA + LB);
    ShellPairData<T> data;
    SparseE<T> m_sparse_e;
    std::array<T, 3> m_pair_center;
    T m_pair_extent;
    T m_pair_estimate;
    std::array<T, 2> m_pair_far;

    ShellPairHolderImpl(ShellPairData<T>&& d) : data(std::move(d)) {
        // The most diffuse primitive (smallest exponent p) sets the pair's
        // centre and extent.
        T p_min = std::numeric_limits<T>::max();
        m_pair_center = {T(0), T(0), T(0)};
        for (const auto& prim : data.primitives) {
            if (prim.p < p_min) {
                p_min = prim.p;
                m_pair_center = {prim.Px, prim.Py, prim.Pz};
            }
        }
        // Extent: radius where exp(-p * r^2) < 1e-12
        constexpr T log_threshold = T(27.631021115928547);  // -ln(1e-12)
        m_pair_extent = std::sqrt(log_threshold / p_min);
        m_pair_estimate = primitive_pair_estimate<T>(data, NAB, NHERM);
        m_pair_far =
            primitive_pair_far_estimate<T>(data, NAB, NHERM, m_pair_center);
        m_sparse_e = build_sparse_e<T>(data, NAB, NHERM);
    }

    int la() const override { return LA; }
    int lb() const override { return LB; }
    int nab() const override { return NAB; }
    int nherm() const override { return NHERM; }
    bool is_spherical() const override { return Spherical; }

    std::array<T, 3> pair_center() const override { return m_pair_center; }
    T pair_extent() const override { return m_pair_extent; }
    T pair_estimate() const override { return m_pair_estimate; }
    std::array<T, 2> pair_far_estimate() const override { return m_pair_far; }

    void evaluate_soa(const T *x, const T *y, const T *z, int npts,
                      const T *boys_table, T *V, int ldv,
                      T omega) const override {
        soa_eval<T, LA + LB>(data, m_sparse_e, NAB, boys_table, x, y, z, npts,
                             V, ldv, omega);
    }

    void evaluate_soa_contract(const T *x, const T *y, const T *z, int npts,
                               const T *boys_table, const T *F, T *G, int ld,
                               int bf_p, int bf_q, bool both,
                               T omega) const override {
        constexpr int na = Spherical ? nsph(LA) : ncart(LA);
        constexpr int nb = Spherical ? nsph(LB) : ncart(LB);
        soa_eval_contract<T, LA + LB, ncart(LA) * ncart(LB)>(
            data, m_sparse_e, NAB, boys_table, x, y, z, npts, F, G, ld, bf_p,
            na, bf_q, nb, both, omega);
    }

    void evaluate(const Eigen::Ref<const Mat3N<T>>& C,
                  const T* boys_table,
                  Eigen::Ref<MatRM<T>> integrals,
                  Eigen::Ref<MatRM<T>> workspace,
                  T omega) const override {
        const int npts = static_cast<int>(C.cols());
        integrals.setZero();
        for (const auto& prim : data.primitives) {
            compute_r_ints_batch_direct<T, LA + LB, BoysParamsDefault>(
                boys_table, prim.p, npts, prim.Px, prim.Py, prim.Pz, C,
                workspace, omega);
            Eigen::Map<const MatRM<T>> E_matrix(prim.E_matrix.data(), NAB,
                                                NHERM);
            integrals.noalias() +=
                prim.prefactor * workspace * E_matrix.transpose();
        }
    }

    void evaluate_overlap(T* integrals) const override {
        for (int ab = 0; ab < NAB; ++ab)
            integrals[ab] = T(0);
        // Overlap prefactor ratio to ESP prefactor:
        // (pi/p)^{3/2} / (2 pi/p) = sqrt(pi/p) / 2; only the (0,0,0)
        // Hermite component contributes.
        for (const auto& prim : data.primitives) {
            const T overlap_prefactor =
                prim.prefactor *
                (std::sqrt(BoysConstants<T>::pi / prim.p) / T(2));
            for (int ab = 0; ab < NAB; ++ab)
                integrals[ab] += overlap_prefactor * prim.E_matrix[ab * NHERM];
        }
    }
};

// Transform E-matrices from Cartesian to spherical
template <typename T, int LA, int LB>
ShellPairData<T> transform_shell_pair_to_spherical(ShellPairData<T>&& cart_data) {
    constexpr int nab_cart = ncart(LA) * ncart(LB);
    const int nab_sph = nsph(LA) * nsph(LB);
    constexpr int nherm = nhermsum(LA + LB);

    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Build transformation matrix
    MatRM T_ab = build_cart_to_sph_transform<T>(LA, LB);

    // Transform each primitive's E-matrix
    for (auto& prim : cart_data.primitives) {
        Eigen::Map<MatRM> E_cart(prim.E_matrix.data(), nab_cart, nherm);

        // E_sph = T_ab @ E_cart
        MatRM E_sph = T_ab * E_cart;

        // Resize and copy
        prim.E_matrix.resize(nab_sph * nherm);
        prim.nab = nab_sph;
        Eigen::Map<MatRM>(prim.E_matrix.data(), nab_sph, nherm) = E_sph;
    }

    return std::move(cart_data);
}

// Primitive pairs of a derivative pair (see FusedE): the pair's E-matrix in
// the Hermite columns of order <= la + lb + 1, per primitive the scalars
// 2x, 2x X_PX (three) and x / p of the explicitly differentiated centre X
// (`scal`), and the Cartesian E of the pair with X's shell lowered by one
// (`low`, rows c' * ncart(l_O) + o).
template <typename T, int LA, int LB>
ShellPairData<T> precompute_shell_pair_derivative(
    int na_prim, int nb_prim, const T *exponents_a, const T *exponents_b,
    const T *coeffs_a, const T *coeffs_b, const T *A, const T *B,
    std::vector<T> &scal, std::vector<T> &low) {
  constexpr int L1 = LA + LB + 1;
  constexpr int na = ncart(LA), nb = ncart(LB), nab = na * nb;
  constexpr int nherm = nhermsum(L1);
  constexpr T sph_fac =
      spherical_harmonic_factor<T>(LA) * spherical_harmonic_factor<T>(LB);
  ShellPairData<T> data;
  data.la = LA;
  data.lb = LB;
  data.A = {A[0], A[1], A[2]};
  data.B = {B[0], B[1], B[2]};
  data.primitives.reserve(na_prim * nb_prim);
  const T rab2 = (A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) +
                 (A[2] - B[2]) * (A[2] - B[2]);
  for (int ia = 0; ia < na_prim; ++ia) {
    const T a = exponents_a[ia], ca = coeffs_a[ia];
    for (int ib = 0; ib < nb_prim; ++ib) {
      const T b = exponents_b[ib], cb = coeffs_b[ib];
      // Primitive pairs whose overlap factor exp(-ab/(a+b) R_AB^2) leaves
      // them below 1e-18 (with the 2a, 2b of the centre derivatives) are
      // dropped: tight primitives on different atoms contribute nothing.
      if (std::abs(T(2) * BoysConstants<T>::pi / (a + b) * ca * cb) *
              std::exp(-a * b / (a + b) * rab2) * (T(1) + T(2) * a) *
              (T(1) + T(2) * b) <
          T(1e-18))
        continue;
      PrimitivePairData<T> prim;
      prim.p = a + b;
      prim.Px = (a * A[0] + b * B[0]) / prim.p;
      prim.Py = (a * A[1] + b * B[1]) / prim.p;
      prim.Pz = (a * A[2] + b * B[2]) / prim.p;
      prim.prefactor =
          T(2) * BoysConstants<T>::pi / prim.p * sph_fac * ca * cb;
      prim.nab = nab;
      prim.nherm = nherm;
      prim.E_matrix.assign(nab * nherm, T(0));
      // The fused pass differentiates explicitly in the centre with the lower
      // angular momentum, X.
      constexpr bool xb = LB < LA;
      const T x = xb ? b : a;
      const T *X = xb ? B : A;
      scal.push_back(T(2) * x);
      scal.push_back(T(2) * x * (prim.Px - X[0]));
      scal.push_back(T(2) * x * (prim.Py - X[1]));
      scal.push_back(T(2) * x * (prim.Pz - X[2]));
      scal.push_back(x / prim.p);
      ECoeffs3D<T, LA + 1, LB + 1> E;
      compute_e_coeffs_3d<T, LA + 1, LB + 1>(a, b, A, B, E);
      constexpr int lX = xb ? LB : LA, lO = xb ? LA : LB;
      auto e1 = [](const auto &E1, int i, int j, int t) -> T {
        return (i < 0 || j < 0 || t > i + j) ? T(0) : E1(i, j, t);
      };
      if (lX > 0) {
        for (int cx = lX - 1; cx >= 0; --cx)
          for (int cy = lX - 1 - cx; cy >= 0; --cy) {
            const int cz = lX - 1 - cx - cy;
            for (int ox = lO; ox >= 0; --ox)
              for (int oy = lO - ox; oy >= 0; --oy) {
                const int oz = lO - ox - oy;
                for (int k = 0; k < nherm; ++k)
                  low.push_back(T(0));
                T *row = low.data() + low.size() - nherm;
                for (int t = 0; t <= L1; ++t)
                  for (int u = 0; u <= L1 - t; ++u)
                    for (int v = 0; v <= L1 - t - u; ++v)
                      row[hermite_index(t, u, v)] =
                          xb ? e1(E.x, ox, cx, t) * e1(E.y, oy, cy, u) *
                                   e1(E.z, oz, cz, v)
                             : e1(E.x, cx, ox, t) * e1(E.y, cy, oy, u) *
                                   e1(E.z, cz, oz, v);
              }
          }
      }
      int row = 0;
      for (int ax = LA; ax >= 0; --ax)
        for (int ay = LA - ax; ay >= 0; --ay) {
          const int az = LA - ax - ay;
          for (int bx = LB; bx >= 0; --bx)
            for (int by = LB - bx; by >= 0; --by) {
              const int bz = LB - bx - by;
              for (int t = 0; t <= L1; ++t)
                for (int u = 0; u <= L1 - t; ++u)
                  for (int v = 0; v <= L1 - t - u; ++v)
                    prim.E_matrix[row * nherm + hermite_index(t, u, v)] =
                        e1(E.x, ax, bx, t) * e1(E.y, ay, by, u) *
                        e1(E.z, az, bz, v);
              ++row;
            }
        }
      data.primitives.push_back(std::move(prim));
    }
  }
  return data;
}

template <typename T, int LA, int LB>
ShellPairData<T> transform_derivative_pair_to_spherical(ShellPairData<T> &&d) {
  constexpr int nab_cart = ncart(LA) * ncart(LB);
  const int nab_sph = nsph(LA) * nsph(LB);
  using MatRMT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const MatRMT T_ab = build_cart_to_sph_transform<T>(LA, LB);
  for (auto &prim : d.primitives) {
    const int nblk = prim.nab / nab_cart;
    const int nherm = prim.nherm;
    MatRMT out(nblk * nab_sph, nherm);
    for (int k = 0; k < nblk; ++k) {
      Eigen::Map<const MatRMT> Ec(prim.E_matrix.data() + k * nab_cart * nherm,
                                  nab_cart, nherm);
      out.middleRows(k * nab_sph, nab_sph) = T_ab * Ec;
    }
    prim.E_matrix.assign(out.data(), out.data() + out.size());
    prim.nab = nblk * nab_sph;
  }
  return std::move(d);
}

template <typename T> struct DerivativePairHolder {
  virtual ~DerivativePairHolder() = default;
  virtual void evaluate_fused(const T *x, const T *y, const T *z, int npts,
                              const T *boys_table, T *A, int lda, const T *W,
                              int ldw, int ncentres, T *out,
                              T omega) const = 0;
};

// Index of the Cartesian function (x, y, z), x + y + z = l, in the order the
// pairs are built in (x descending, then y descending).
inline int cart_index(int l, int x, int y) {
  int idx = 0;
  for (int ax = l; ax > x; --ax)
    idx += l - ax + 1;
  return idx + (l - x - y);
}

// Fused-pass coefficients: value rows and g entries (from the pair's
// E-matrix), columns of order <= la + lb, each over the union of every
// primitive's nonzero pattern; and the lowered-pair rows and T^d entries of
// the lowered-shell term from `low` (per primitive, Cartesian, as written by
// precompute_shell_pair_derivative).
template <typename T, int LA, int LB>
FusedE<T> build_fused_e(const ShellPairData<T> &data, std::vector<T> scal,
                        const std::vector<T> &low, bool spherical, int nab,
                        int nherm, int nhv) {
  constexpr bool xb = LB < LA;
  constexpr int lX = xb ? LB : LA, lO = xb ? LA : LB;
  FusedE<T> fe;
  fe.nab = nab;
  fe.explicit_b = xb;
  fe.scal = std::move(scal);
  auto entry = [&](const PrimitivePairData<T> &prim, int ab, int t) {
    return prim.E_matrix[ab * nherm + t];
  };
  auto nonzero = [&](int ab, int t) {
    for (const auto &prim : data.primitives)
      if (entry(prim, ab, t) != T(0))
        return true;
    return false;
  };
  fe.vstart.push_back(0);
  for (int ab = 0; ab < nab; ++ab) {
    for (int t = 0; t < nhv; ++t)
      if (nonzero(ab, t))
        fe.vt.push_back(t);
    fe.vstart.push_back(static_cast<int>(fe.vt.size()));
  }
  fe.nv = static_cast<int>(fe.vt.size());
  fe.estart.push_back(0);
  for (int t = 0; t < nhv; ++t) {
    for (int ab = 0; ab < nab; ++ab)
      if (nonzero(ab, t))
        fe.eab.push_back(ab);
    fe.estart.push_back(static_cast<int>(fe.eab.size()));
  }
  fe.ne = static_cast<int>(fe.eab.size());
  for (const auto &prim : data.primitives) {
    for (int ab = 0; ab < nab; ++ab)
      for (int k = fe.vstart[ab]; k < fe.vstart[ab + 1]; ++k)
        fe.vc.push_back(prim.prefactor * entry(prim, ab, fe.vt[k]));
    for (int t = 0; t < nhv; ++t)
      for (int k = fe.estart[t]; k < fe.estart[t + 1]; ++k)
        fe.ec.push_back(prim.prefactor * entry(prim, fe.eab[k], t));
  }
  if constexpr (lX > 0) {
    using MatRMT =
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const int nlc = ncart(lX - 1), noc = ncart(lO);
    const int no = spherical ? nsph(lO) : noc;
    const int nx = spherical ? nsph(lX) : ncart(lX);
    const size_t nprim = data.primitives.size();
    const size_t per = static_cast<size_t>(nlc) * noc * nherm;
    // the other shell to its own functions: El[c', o] = sum_b S_O[o, b] E[c', b]
    std::vector<MatRMT> El(nprim);
    const MatRMT SO =
        spherical ? MatRMT(occ::gto::cartesian_to_spherical_transformation_matrix(lO).template cast<T>())
                  : MatRMT(MatRMT::Identity(noc, noc));
    for (size_t ip = 0; ip < nprim; ++ip) {
      El[ip].resize(nlc * no, nherm);
      for (int c = 0; c < nlc; ++c) {
        Eigen::Map<const MatRMT> Ec(low.data() + ip * per +
                                        static_cast<size_t>(c) * noc * nherm,
                                    noc, nherm);
        El[ip].middleRows(c * no, no) = SO * Ec;
      }
    }
    fe.nlow = nlc * no;
    fe.no = no;
    fe.nx = nx;
    fe.lstart.push_back(0);
    for (int r = 0; r < fe.nlow; ++r) {
      for (int t = 0; t < nhv; ++t) {
        bool nz = false;
        for (size_t ip = 0; ip < nprim && !nz; ++ip)
          nz = El[ip](r, t) != T(0);
        if (nz)
          fe.lt.push_back(t);
      }
      fe.lstart.push_back(static_cast<int>(fe.lt.size()));
    }
    fe.nl = static_cast<int>(fe.lt.size());
    for (size_t ip = 0; ip < nprim; ++ip)
      for (int r = 0; r < fe.nlow; ++r)
        for (int k = fe.lstart[r]; k < fe.lstart[r + 1]; ++k)
          fe.lc.push_back(data.primitives[ip].prefactor * El[ip](r, fe.lt[k]));
    // T^d_{s c'} = (c'_d + 1) S_X[s, c' + e_d]
    const MatRMT SX =
        spherical ? MatRMT(occ::gto::cartesian_to_spherical_transformation_matrix(lX).template cast<T>())
                  : MatRMT(MatRMT::Identity(ncart(lX), ncart(lX)));
    for (int d = 0; d < 3; ++d)
      for (int sx = 0; sx < nx; ++sx) {
        int c = 0;
        for (int cx = lX - 1; cx >= 0; --cx)
          for (int cy = lX - 1 - cx; cy >= 0; --cy, ++c) {
            const int cz = lX - 1 - cx - cy;
            const int ux = cx + (d == 0), uy = cy + (d == 1);
            const int cd = d == 0 ? cx : d == 1 ? cy : cz;
            const T v = T(cd + 1) * SX(sx, cart_index(lX, ux, uy));
            if (v != T(0)) {
              fe.td.push_back(d);
              fe.ts.push_back(sx);
              fe.tc.push_back(c);
              fe.tv.push_back(v);
            }
          }
      }
  }
  return fe;
}

template <typename T, int LA, int LB>
struct DerivativePairHolderImpl : DerivativePairHolder<T> {
  ShellPairData<T> data;
  FusedE<T> fe;
  DerivativePairHolderImpl(int na_prim, int nb_prim, const T *exponents_a,
                           const T *exponents_b, const T *coeffs_a,
                           const T *coeffs_b, const T *A, const T *B,
                           bool spherical) {
    constexpr int nherm = nhermsum(LA + LB + 1);
    const int nab = spherical ? nsph(LA) * nsph(LB) : ncart(LA) * ncart(LB);
    std::vector<T> scal, low;
    data = precompute_shell_pair_derivative<T, LA, LB>(
        na_prim, nb_prim, exponents_a, exponents_b, coeffs_a, coeffs_b, A, B,
        scal, low);
    if (spherical)
      data = transform_derivative_pair_to_spherical<T, LA, LB>(std::move(data));
    fe = build_fused_e<T, LA, LB>(data, std::move(scal), low, spherical, nab,
                                  nherm, nhermsum(LA + LB));
    // the kernel reads only the primitives' exponents, centres and Boys
    // prefactors; the E blocks now live in fe
    for (auto &prim : data.primitives)
      std::vector<T>().swap(prim.E_matrix);
  }
  void evaluate_fused(const T *x, const T *y, const T *z, int npts,
                      const T *boys_table, T *A, int lda, const T *W, int ldw,
                      int ncentres, T *out, T omega) const override {
    // K vanishes when the explicitly differentiated shell is an s shell
    constexpr bool has_k = LA > 0 && LB > 0;
    soa_fused_eval<T, LA + LB + 1, has_k>(data, fe, ncentres, boys_table, x, y, z,
                                   npts, A, lda, W, ldw, out, omega);
  }
};

// ============================================================================
// Second derivatives of the ESP integrals, for seminumerical exchange
// Hessians.
//
// X is the pair's centre of lower angular momentum (exponent x), a its
// shell's Cartesian index, P the product centre. Raising X's function along
// d is, in Hermite form, R_t -> (R_d R)_t = R_{t+e_d} / 2p + X_PX,d R_t
// + t_d R_{t-e_d}, so with E the pair's expansion and the lowered ones
// K^d = a_d E^{a-e_d}, K^de = a_d (a - e_d)_e E^{a-e_d-e_e}:
//   [ab]           = E . R
//   d/dX_d         = 2x E . R_d R - K^d . R
//   d/dX_d d/dX_e  = 4x^2 E . R_d R_e R - 2x (K^d . R_e R + K^e . R_d R)
//                    - 2x delta_de E . R + K^de . R
// and moving both centres moves P only (d/dP_e R_t = R_{t+e_e}):
//   d/dX_d d/dP_e  = 2x (E . (R_d R)_{+e} - delta_de E . R) - K^d . R_{+e}
//   d/dP_d d/dP_e  = E . R_{+d+e},
// the other centre's derivatives following from d/dO = d/dP - d/dX. The
// second derivatives are only needed contracted with per-point weights W, so
// W goes into the Hermite densities W.E, W.K^d and W.K^de first.
// ============================================================================

namespace {

// Hermite index tables up to total degree L: components and the indices one
// step up (degree < L) or down (-1 where the component is zero) along d.
template <int L> struct HermiteTable {
  static constexpr int N = nhermsum(L);
  std::array<int, N> c[3]{}, up[3]{}, dn[3]{};
  constexpr HermiteTable() {
    for (int deg = 0; deg <= L; ++deg)
      for (int t = deg; t >= 0; --t)
        for (int u = deg - t; u >= 0; --u) {
          const int v = deg - t - u, k = hermite_index(t, u, v);
          const int tuv[3] = {t, u, v};
          for (int d = 0; d < 3; ++d) {
            c[d][k] = tuv[d];
            int s[3] = {t, u, v};
            s[d] += 1;
            up[d][k] = deg < L ? hermite_index(s[0], s[1], s[2]) : -1;
            s[d] -= 2;
            dn[d][k] = tuv[d] > 0 ? hermite_index(s[0], s[1], s[2]) : -1;
          }
        }
  }
};

// Pair data for the kernel: blocks 0 (E), 1..3 (K^d) and 4..9 (K^de, d <= e
// in the order xx, xy, xz, yy, yz, zz) of each primitive's prefactor-scaled
// expansion, over the union of the primitives' nonzero patterns.
template <typename T> struct HessE {
  int nab = 0, nh = 0;
  bool explicit_b = false; // X = B (else A)
  // rows of blocks 0..3, for the integrals and first derivatives: row
  // b * nab + ab has entries rstart[r] .. rstart[r + 1] of Hermite columns
  // rt, coefficients rc[ip * nr + k]
  int nr = 0;
  std::vector<int> rstart, rt;
  std::vector<T> rc;
  // columns of all ten blocks, for the densities: column b * nh + t has
  // entries cstart[.] .. cstart[. + 1] of rows crow, coefficients
  // cc[ip * nc + k]
  int nc = 0;
  std::vector<int> cstart, crow;
  std::vector<T> cc;
  // per primitive: 2x, 1 / 2p, X_PX (3)
  std::vector<T> scal;
};

template <typename T, int LA, int LB>
ShellPairData<T> precompute_shell_pair_hessian(
    int na_prim, int nb_prim, const T *exponents_a, const T *exponents_b,
    const T *coeffs_a, const T *coeffs_b, const T *A, const T *B,
    std::vector<T> &scal) {
  constexpr int L = LA + LB;
  constexpr int na = ncart(LA), nb = ncart(LB), nab = na * nb;
  constexpr int nherm = nhermsum(L);
  constexpr bool xb = LB < LA;
  constexpr T sph_fac =
      spherical_harmonic_factor<T>(LA) * spherical_harmonic_factor<T>(LB);
  ShellPairData<T> data;
  data.la = LA;
  data.lb = LB;
  data.A = {A[0], A[1], A[2]};
  data.B = {B[0], B[1], B[2]};
  const T rab2 = (A[0] - B[0]) * (A[0] - B[0]) + (A[1] - B[1]) * (A[1] - B[1]) +
                 (A[2] - B[2]) * (A[2] - B[2]);
  for (int ia = 0; ia < na_prim; ++ia) {
    const T a = exponents_a[ia], ca = coeffs_a[ia];
    for (int ib = 0; ib < nb_prim; ++ib) {
      const T b = exponents_b[ib], cb = coeffs_b[ib];
      // as for the derivative pairs, with the 4a^2, 4b^2 of the second
      // derivatives
      if (std::abs(T(2) * BoysConstants<T>::pi / (a + b) * ca * cb) *
              std::exp(-a * b / (a + b) * rab2) * (T(1) + T(2) * a) *
              (T(1) + T(2) * a) * (T(1) + T(2) * b) * (T(1) + T(2) * b) <
          T(1e-18))
        continue;
      PrimitivePairData<T> prim;
      prim.p = a + b;
      prim.Px = (a * A[0] + b * B[0]) / prim.p;
      prim.Py = (a * A[1] + b * B[1]) / prim.p;
      prim.Pz = (a * A[2] + b * B[2]) / prim.p;
      prim.prefactor =
          T(2) * BoysConstants<T>::pi / prim.p * sph_fac * ca * cb;
      prim.nab = 10 * nab;
      prim.nherm = nherm;
      prim.E_matrix.assign(10 * nab * nherm, T(0));
      const T x = xb ? b : a;
      const T *X = xb ? B : A;
      scal.push_back(T(2) * x);
      scal.push_back(T(0.5) / prim.p);
      scal.push_back(prim.Px - X[0]);
      scal.push_back(prim.Py - X[1]);
      scal.push_back(prim.Pz - X[2]);
      ECoeffs3D<T, LA, LB> E;
      compute_e_coeffs_3d<T, LA, LB>(a, b, A, B, E);
      auto e1 = [](const auto &E1, int i, int j, int t) -> T {
        return (i < 0 || j < 0 || t > i + j) ? T(0) : E1(i, j, t);
      };
      // X's index lowered once (times the index) and twice in one direction
      auto lw = [&](const auto &E1, int i, int j, int t) {
        return xb ? T(j) * e1(E1, i, j - 1, t) : T(i) * e1(E1, i - 1, j, t);
      };
      auto lw2 = [&](const auto &E1, int i, int j, int t) {
        return xb ? T(j * (j - 1)) * e1(E1, i, j - 2, t)
                  : T(i * (i - 1)) * e1(E1, i - 2, j, t);
      };
      int row = 0;
      for (int ax = LA; ax >= 0; --ax)
        for (int ay = LA - ax; ay >= 0; --ay) {
          const int az = LA - ax - ay;
          for (int bx = LB; bx >= 0; --bx)
            for (int by = LB - bx; by >= 0; --by) {
              const int bz = LB - bx - by;
              for (int t = 0; t <= L; ++t)
                for (int u = 0; u <= L - t; ++u)
                  for (int v = 0; v <= L - t - u; ++v) {
                    const T ex = e1(E.x, ax, bx, t), ey = e1(E.y, ay, by, u),
                            ez = e1(E.z, az, bz, v);
                    const T lx = lw(E.x, ax, bx, t), ly = lw(E.y, ay, by, u),
                            lz = lw(E.z, az, bz, v);
                    const T f[10] = {ex * ey * ez,
                                     lx * ey * ez,
                                     ex * ly * ez,
                                     ex * ey * lz,
                                     lw2(E.x, ax, bx, t) * ey * ez,
                                     lx * ly * ez,
                                     lx * ey * lz,
                                     ex * lw2(E.y, ay, by, u) * ez,
                                     ex * ly * lz,
                                     ex * ey * lw2(E.z, az, bz, v)};
                    const int col = hermite_index(t, u, v);
                    for (int k = 0; k < 10; ++k)
                      prim.E_matrix[(k * nab + row) * nherm + col] = f[k];
                  }
              ++row;
            }
        }
      data.primitives.push_back(std::move(prim));
    }
  }
  return data;
}

template <typename T>
HessE<T> build_hessian_e(const ShellPairData<T> &data, std::vector<T> scal,
                         int nab, int nh, bool explicit_b) {
  HessE<T> he;
  he.nab = nab;
  he.nh = nh;
  he.explicit_b = explicit_b;
  he.scal = std::move(scal);
  auto entry = [&](const PrimitivePairData<T> &prim, int blk, int ab, int t) {
    return prim.E_matrix[(blk * nab + ab) * nh + t];
  };
  auto nonzero = [&](int blk, int ab, int t) {
    for (const auto &prim : data.primitives)
      if (entry(prim, blk, ab, t) != T(0))
        return true;
    return false;
  };
  he.rstart.push_back(0);
  for (int blk = 0; blk < 4; ++blk)
    for (int ab = 0; ab < nab; ++ab) {
      for (int t = 0; t < nh; ++t)
        if (nonzero(blk, ab, t))
          he.rt.push_back(t);
      he.rstart.push_back(static_cast<int>(he.rt.size()));
    }
  he.nr = static_cast<int>(he.rt.size());
  he.cstart.push_back(0);
  for (int blk = 0; blk < 10; ++blk)
    for (int t = 0; t < nh; ++t) {
      for (int ab = 0; ab < nab; ++ab)
        if (nonzero(blk, ab, t))
          he.crow.push_back(ab);
      he.cstart.push_back(static_cast<int>(he.crow.size()));
    }
  he.nc = static_cast<int>(he.crow.size());
  for (const auto &prim : data.primitives) {
    int k = 0;
    for (int r = 0; r < 4 * nab; ++r)
      for (; k < he.rstart[r + 1]; ++k)
        he.rc.push_back(prim.prefactor * entry(prim, r / nab, r % nab, he.rt[k]));
    k = 0;
    for (int c = 0; c < 10 * nh; ++c)
      for (; k < he.cstart[c + 1]; ++k)
        he.cc.push_back(prim.prefactor * entry(prim, c / nh, he.crow[k], c % nh));
  }
  return he;
}

// No VRP here either (see the fused kernel).
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-tree-vrp")
#endif
// One chunk of NP points. V rows: the integrals, d/dX_d and d/dP_d (turned
// into d/dA, d/dB by the caller); acc[k * NP + i]: second derivatives XX
// (6), XP (X_d P_e at 3 d + e) and PP (6), per point.
template <typename T, int L>
__attribute__((always_inline)) inline void
soa_hessian_chunk_body(const ShellPairData<T> &data, const HessE<T> &he,
                       const T *boys_table, const T *cx, const T *cy,
                       const T *cz, T *V, int ldv, const T *W, int ldw, T *acc,
                       T omega) {
  constexpr int NP = soa_chunk<L + 2>();
  constexpr auto off = level_offsets<L + 2>();
  constexpr int N0 = nhermsum(L), N1 = nhermsum(L - 1), N2 = nhermsum(L - 2),
                NU = nhermsum(L + 1);
  static constexpr HermiteTable<L + 2> ht{};
  static constexpr int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  alignas(64) T R[off[L + 3] * NP];
  alignas(64) T Rd[3][NU * NP];
  const int nab = he.nab, nh = he.nh;
  for (size_t ip = 0; ip < data.primitives.size(); ++ip) {
    soa_prim_hermite<T, L + 2, NP>(data.primitives[ip], boys_table, cx,
                                         cy, cz, R, omega);
    const T *sc = he.scal.data() + ip * 5;
    const T x2 = sc[0], h = sc[1];
    // (R_d R)_t for t of degree <= L + 1
    for (int d = 0; d < 3; ++d) {
      const T X = sc[2 + d];
      for (int t = 0; t < NU; ++t) {
        const T *ru = R + ht.up[d][t] * NP, *r0 = R + t * NP;
        T *o = Rd[d] + t * NP;
        if (ht.dn[d][t] >= 0) {
          const T c = T(ht.c[d][t]);
          const T *rn = R + ht.dn[d][t] * NP;
          for (int i = 0; i < NP; ++i)
            o[i] = h * ru[i] + X * r0[i] + c * rn[i];
        } else {
          for (int i = 0; i < NP; ++i)
            o[i] = h * ru[i] + X * r0[i];
        }
      }
    }
    // integrals and first derivatives
    const T *rc = he.rc.data() + ip * he.nr;
    for (int ab = 0; ab < nab; ++ab) {
      alignas(64) T val[NP] = {}, dx[3][NP] = {}, dp[3][NP] = {};
      for (int k = he.rstart[ab]; k < he.rstart[ab + 1]; ++k) {
        const T c = rc[k], cx2 = c * x2;
        const int t = he.rt[k];
        const T *r0 = R + t * NP;
        for (int i = 0; i < NP; ++i)
          val[i] += c * r0[i];
        for (int d = 0; d < 3; ++d) {
          const T *rd = Rd[d] + t * NP, *ru = R + ht.up[d][t] * NP;
          for (int i = 0; i < NP; ++i) {
            dx[d][i] += cx2 * rd[i];
            dp[d][i] += c * ru[i];
          }
        }
      }
      for (int d = 0; d < 3; ++d)
        for (int k = he.rstart[(1 + d) * nab + ab];
             k < he.rstart[(1 + d) * nab + ab + 1]; ++k) {
          const T c = rc[k];
          const T *r0 = R + he.rt[k] * NP;
          for (int i = 0; i < NP; ++i)
            dx[d][i] -= c * r0[i];
        }
      T *v = V + ab * ldv;
      for (int i = 0; i < NP; ++i)
        v[i] += val[i];
      for (int d = 0; d < 3; ++d) {
        T *vx = V + ((1 + d) * nab + ab) * ldv, *vp = V + ((4 + d) * nab + ab) * ldv;
        for (int i = 0; i < NP; ++i) {
          vx[i] += dx[d][i];
          vp[i] += dp[d][i];
        }
      }
    }
    if (!W)
      continue;
    // Hermite densities of W: g (block 0), k^d (1..3), k^de (4..9)
    alignas(64) T g[N0 * NP], kd[3][(N1 > 0 ? N1 : 1) * NP],
        kde[6][(N2 > 0 ? N2 : 1) * NP];
    const T *cc = he.cc.data() + ip * he.nc;
    auto density = [&](int blk, int nt, T *out) {
      for (int t = 0; t < nt; ++t) {
        T *o = out + t * NP;
        for (int i = 0; i < NP; ++i)
          o[i] = T(0);
        for (int k = he.cstart[blk * nh + t]; k < he.cstart[blk * nh + t + 1];
             ++k) {
          const T c = cc[k];
          const T *w = W + he.crow[k] * ldw;
          for (int i = 0; i < NP; ++i)
            o[i] += c * w[i];
        }
      }
    };
    density(0, N0, g);
    for (int d = 0; d < 3; ++d)
      density(1 + d, N1, kd[d]);
    for (int c = 0; c < 6; ++c)
      density(4 + c, N2, kde[c]);
    alignas(64) T gR[NP] = {};
    for (int t = 0; t < N0; ++t)
      for (int i = 0; i < NP; ++i)
        gR[i] += g[t * NP + i] * R[t * NP + i];
    for (int d = 0; d < 3; ++d)
      for (int e = d; e < 3; ++e) {
        const int s = sym[d][e];
        const T X = sc[2 + d];
        T *xx = acc + s * NP, *pp = acc + (15 + s) * NP;
        alignas(64) T a4[NP] = {}, ak[NP] = {}, app[NP] = {};
        // E . R_d R_e R, from (R_e R) at degree <= L + 1
        for (int t = 0; t < N0; ++t) {
          const T *gt = g + t * NP;
          const T *re = Rd[e] + t * NP, *reu = Rd[e] + ht.up[d][t] * NP;
          const T *rpp = R + ht.up[d][ht.up[e][t]] * NP;
          if (ht.dn[d][t] >= 0) {
            const T c = T(ht.c[d][t]);
            const T *ren = Rd[e] + ht.dn[d][t] * NP;
            for (int i = 0; i < NP; ++i)
              a4[i] += gt[i] * (h * reu[i] + X * re[i] + c * ren[i]);
          } else {
            for (int i = 0; i < NP; ++i)
              a4[i] += gt[i] * (h * reu[i] + X * re[i]);
          }
          for (int i = 0; i < NP; ++i)
            app[i] += gt[i] * rpp[i];
        }
        for (int t = 0; t < N1; ++t) {
          const T *kdt = kd[d] + t * NP, *ket = kd[e] + t * NP;
          const T *re = Rd[e] + t * NP, *rd = Rd[d] + t * NP;
          for (int i = 0; i < NP; ++i)
            ak[i] += kdt[i] * re[i] + ket[i] * rd[i];
        }
        for (int t = 0; t < N2; ++t) {
          const T *k2 = kde[s] + t * NP, *r0 = R + t * NP;
          for (int i = 0; i < NP; ++i)
            xx[i] += k2[i] * r0[i];
        }
        const T dg = d == e ? x2 : T(0);
        for (int i = 0; i < NP; ++i) {
          xx[i] += x2 * x2 * a4[i] - x2 * ak[i] - dg * gR[i];
          pp[i] += app[i];
        }
      }
    for (int d = 0; d < 3; ++d)
      for (int e = 0; e < 3; ++e) {
        T *xp = acc + (6 + 3 * d + e) * NP;
        alignas(64) T a1[NP] = {}, a2[NP] = {};
        for (int t = 0; t < N0; ++t) {
          const T *gt = g + t * NP, *rd = Rd[d] + ht.up[e][t] * NP;
          for (int i = 0; i < NP; ++i)
            a1[i] += gt[i] * rd[i];
        }
        for (int t = 0; t < N1; ++t) {
          const T *kdt = kd[d] + t * NP, *ru = R + ht.up[e][t] * NP;
          for (int i = 0; i < NP; ++i)
            a2[i] += kdt[i] * ru[i];
        }
        const T dg = d == e ? x2 : T(0);
        for (int i = 0; i < NP; ++i)
          xp[i] += x2 * a1[i] - dg * gR[i] - a2[i];
      }
  }
}

template <typename T, int L>
void soa_hessian_chunk_generic(const ShellPairData<T> &data,
                               const HessE<T> &he, const T *boys_table,
                               const T *cx, const T *cy, const T *cz, T *V,
                               int ldv, const T *W, int ldw, T *acc, T omega) {
  soa_hessian_chunk_body<T, L>(data, he, boys_table, cx, cy, cz, V, ldv,
                                      W, ldw, acc, omega);
}

#if OCC_SIMD_X86_DISPATCH
template <typename T, int L>
__attribute__((target("avx2,fma"))) void
soa_hessian_chunk_avx2(const ShellPairData<T> &data, const HessE<T> &he,
                       const T *boys_table, const T *cx, const T *cy,
                       const T *cz, T *V, int ldv, const T *W, int ldw, T *acc,
                       T omega) {
  soa_hessian_chunk_body<T, L>(data, he, boys_table, cx, cy, cz, V, ldv,
                                     W, ldw, acc, omega);
}
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

} // namespace

template <typename T> struct HessianPairHolder {
  virtual ~HessianPairHolder() = default;
  virtual void evaluate(const T *x, const T *y, const T *z, int npts,
                        const T *boys_table, T *V, int ldv, const T *W,
                        int ldw, T *out, T omega) const = 0;
};

template <typename T, int LA, int LB>
struct HessianPairHolderImpl : HessianPairHolder<T> {
  ShellPairData<T> data;
  HessE<T> he;
  HessianPairHolderImpl(int na_prim, int nb_prim, const T *exponents_a,
                        const T *exponents_b, const T *coeffs_a,
                        const T *coeffs_b, const T *A, const T *B,
                        bool spherical) {
    std::vector<T> scal;
    data = precompute_shell_pair_hessian<T, LA, LB>(
        na_prim, nb_prim, exponents_a, exponents_b, coeffs_a, coeffs_b, A, B,
        scal);
    if (spherical)
      data = transform_derivative_pair_to_spherical<T, LA, LB>(std::move(data));
    const int nab = spherical ? nsph(LA) * nsph(LB) : ncart(LA) * ncart(LB);
    he = build_hessian_e<T>(data, std::move(scal), nab, nhermsum(LA + LB),
                            LB < LA);
    for (auto &prim : data.primitives)
      std::vector<T>().swap(prim.E_matrix);
  }
  void evaluate(const T *x, const T *y, const T *z, int npts,
                const T *boys_table, T *V, int ldv, const T *W, int ldw,
                T *out, T omega) const override {
    constexpr int L = LA + LB;
    constexpr int NP = soa_chunk<L + 2>();
    const int nab = he.nab;
    for (int r = 0; r < 7 * nab; ++r)
      std::fill(V + r * ldv, V + r * ldv + npts, T(0));
    std::vector<T> acc(W ? 21 * NP : 0, T(0));
    auto chunk = [&](int c) {
#if OCC_SIMD_X86_DISPATCH
      if (cpu_has_avx2_fma())
        return soa_hessian_chunk_avx2<T, L>(data, he, boys_table, x + c, y + c,
                                            z + c, V + c, ldv,
                                            W ? W + c : nullptr, ldw,
                                            acc.data(), omega);
#endif
      soa_hessian_chunk_generic<T, L>(data, he, boys_table, x + c, y + c, z + c,
                                      V + c, ldv, W ? W + c : nullptr, ldw,
                                      acc.data(), omega);
    };
    for (int c = 0; c < npts; c += NP)
      chunk(c);
    // d/dO = d/dP - d/dX for the other centre, then to A, B order
    for (int d = 0; d < 3; ++d) {
      T *vx = V + (1 + d) * nab * ldv, *vp = V + (4 + d) * nab * ldv;
      for (int r = 0; r < nab; ++r)
        for (int i = 0; i < npts; ++i) {
          const T dx = vx[r * ldv + i], dpo = vp[r * ldv + i] - dx;
          vx[r * ldv + i] = he.explicit_b ? dpo : dx;
          vp[r * ldv + i] = he.explicit_b ? dx : dpo;
        }
    }
    if (!W)
      return;
    T s[21];
    for (int k = 0; k < 21; ++k) {
      s[k] = T(0);
      for (int i = 0; i < NP; ++i)
        s[k] += acc[k * NP + i];
    }
    static constexpr int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    auto XX = [&](int d, int e) { return s[sym[d][e]]; };
    auto XP = [&](int d, int e) { return s[6 + 3 * d + e]; };
    auto PP = [&](int d, int e) { return s[15 + sym[d][e]]; };
    for (int d = 0; d < 3; ++d)
      for (int e = 0; e < 3; ++e) {
        const T xo_de = XP(d, e) - XX(d, e), xo_ed = XP(e, d) - XX(e, d);
        const T oo = PP(d, e) - XP(e, d) - XP(d, e) + XX(d, e);
        out[6 + 3 * d + e] += he.explicit_b ? xo_ed : xo_de;
        if (e >= d) {
          out[sym[d][e]] += he.explicit_b ? oo : XX(d, e);
          out[15 + sym[d][e]] += he.explicit_b ? XX(d, e) : oo;
        }
      }
  }
};

template <typename T>
struct ESPEvaluator<T>::Impl {
    const T* boys_table;
    std::vector<std::unique_ptr<ShellPairHolder<T>>> shell_pairs;
    std::vector<std::unique_ptr<DerivativePairHolder<T>>> derivative_pairs;
    std::vector<std::unique_ptr<HessianPairHolder<T>>> hessian_pairs;

    explicit Impl(const T* table) : boys_table(table) {}
};

template <typename T>
ESPEvaluator<T>::ESPEvaluator(const T* boys_table)
    : impl_(std::make_unique<Impl>(boys_table)) {}

template <typename T>
ESPEvaluator<T>::~ESPEvaluator() = default;

template <typename T>
ESPEvaluator<T>::ESPEvaluator(ESPEvaluator&&) noexcept = default;

template <typename T>
ESPEvaluator<T>& ESPEvaluator<T>::operator=(ESPEvaluator&&) noexcept = default;

// Helper macros for dispatch
#define DISPATCH_CARTESIAN(LA, LB) \
    if (la == LA && lb == LB) { \
        auto data = precompute_shell_pair<T, LA, LB>( \
            na_prim, nb_prim, exponents_a, exponents_b, \
            coeffs_a, coeffs_b, A, B); \
        impl_->shell_pairs.push_back( \
            std::make_unique<ShellPairHolderImpl<T, LA, LB, false>>(std::move(data))); \
        return impl_->shell_pairs.size() - 1; \
    }

#define DISPATCH_SPHERICAL(LA, LB) \
    if (la == LA && lb == LB) { \
        auto cart_data = precompute_shell_pair<T, LA, LB>( \
            na_prim, nb_prim, exponents_a, exponents_b, \
            coeffs_a, coeffs_b, A, B); \
        auto sph_data = transform_shell_pair_to_spherical<T, LA, LB>(std::move(cart_data)); \
        impl_->shell_pairs.push_back( \
            std::make_unique<ShellPairHolderImpl<T, LA, LB, true>>(std::move(sph_data))); \
        return impl_->shell_pairs.size() - 1; \
    }

template <typename T>
size_t ESPEvaluator<T>::add_shell_pair(int la, int lb,
                                        int na_prim, int nb_prim,
                                        const T* exponents_a, const T* exponents_b,
                                        const T* coeffs_a, const T* coeffs_b,
                                        const T* A, const T* B) {
    // Default: Cartesian
    return add_shell_pair(la, lb, na_prim, nb_prim, exponents_a, exponents_b,
                          coeffs_a, coeffs_b, A, B, false);
}

template <typename T>
size_t ESPEvaluator<T>::add_shell_pair(int la, int lb,
                                        int na_prim, int nb_prim,
                                        const T* exponents_a, const T* exponents_b,
                                        const T* coeffs_a, const T* coeffs_b,
                                        const T* A, const T* B,
                                        bool spherical) {
    if (spherical) {
        DISPATCH_SPHERICAL(0, 0)
        DISPATCH_SPHERICAL(0, 1)
        DISPATCH_SPHERICAL(0, 2)
        DISPATCH_SPHERICAL(0, 3)
        DISPATCH_SPHERICAL(1, 0)
        DISPATCH_SPHERICAL(1, 1)
        DISPATCH_SPHERICAL(1, 2)
        DISPATCH_SPHERICAL(1, 3)
        DISPATCH_SPHERICAL(2, 0)
        DISPATCH_SPHERICAL(2, 1)
        DISPATCH_SPHERICAL(2, 2)
        DISPATCH_SPHERICAL(2, 3)
        DISPATCH_SPHERICAL(3, 0)
        DISPATCH_SPHERICAL(3, 1)
        DISPATCH_SPHERICAL(3, 2)
        DISPATCH_SPHERICAL(3, 3)
        DISPATCH_SPHERICAL(0, 4)
        DISPATCH_SPHERICAL(1, 4)
        DISPATCH_SPHERICAL(2, 4)
        DISPATCH_SPHERICAL(3, 4)
        DISPATCH_SPHERICAL(4, 0)
        DISPATCH_SPHERICAL(4, 1)
        DISPATCH_SPHERICAL(4, 2)
        DISPATCH_SPHERICAL(4, 3)
        DISPATCH_SPHERICAL(4, 4)
        DISPATCH_SPHERICAL(0, 5)
        DISPATCH_SPHERICAL(1, 5)
        DISPATCH_SPHERICAL(2, 5)
        DISPATCH_SPHERICAL(3, 5)
        DISPATCH_SPHERICAL(4, 5)
        DISPATCH_SPHERICAL(5, 0)
        DISPATCH_SPHERICAL(5, 1)
        DISPATCH_SPHERICAL(5, 2)
        DISPATCH_SPHERICAL(5, 3)
        DISPATCH_SPHERICAL(5, 4)
        DISPATCH_SPHERICAL(5, 5)
    } else {
        DISPATCH_CARTESIAN(0, 0)
        DISPATCH_CARTESIAN(0, 1)
        DISPATCH_CARTESIAN(0, 2)
        DISPATCH_CARTESIAN(0, 3)
        DISPATCH_CARTESIAN(1, 0)
        DISPATCH_CARTESIAN(1, 1)
        DISPATCH_CARTESIAN(1, 2)
        DISPATCH_CARTESIAN(1, 3)
        DISPATCH_CARTESIAN(2, 0)
        DISPATCH_CARTESIAN(2, 1)
        DISPATCH_CARTESIAN(2, 2)
        DISPATCH_CARTESIAN(2, 3)
        DISPATCH_CARTESIAN(3, 0)
        DISPATCH_CARTESIAN(3, 1)
        DISPATCH_CARTESIAN(3, 2)
        DISPATCH_CARTESIAN(3, 3)
        DISPATCH_CARTESIAN(0, 4)
        DISPATCH_CARTESIAN(1, 4)
        DISPATCH_CARTESIAN(2, 4)
        DISPATCH_CARTESIAN(3, 4)
        DISPATCH_CARTESIAN(4, 0)
        DISPATCH_CARTESIAN(4, 1)
        DISPATCH_CARTESIAN(4, 2)
        DISPATCH_CARTESIAN(4, 3)
        DISPATCH_CARTESIAN(4, 4)
        DISPATCH_CARTESIAN(0, 5)
        DISPATCH_CARTESIAN(1, 5)
        DISPATCH_CARTESIAN(2, 5)
        DISPATCH_CARTESIAN(3, 5)
        DISPATCH_CARTESIAN(4, 5)
        DISPATCH_CARTESIAN(5, 0)
        DISPATCH_CARTESIAN(5, 1)
        DISPATCH_CARTESIAN(5, 2)
        DISPATCH_CARTESIAN(5, 3)
        DISPATCH_CARTESIAN(5, 4)
        DISPATCH_CARTESIAN(5, 5)
    }

    throw std::runtime_error("Unsupported angular momentum combination: la=" +
                             std::to_string(la) + ", lb=" + std::to_string(lb));
}

#undef DISPATCH_CARTESIAN
#undef DISPATCH_SPHERICAL

template <typename T>
void ESPEvaluator<T>::evaluate(size_t shell_idx,
                                const Eigen::Ref<const Mat3N>& C,
                                Eigen::Ref<MatRM> integrals,
                                Eigen::Ref<MatRM> workspace,
                                T omega) {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }

    const auto& sp = impl_->shell_pairs[shell_idx];
    sp->evaluate(C, impl_->boys_table, integrals, workspace, omega);
}

template <typename T>
void ESPEvaluator<T>::evaluate_soa(size_t shell_idx, const T *x, const T *y,
                                   const T *z, int npts, T *V, int ldv,
                                   T omega) const {
    impl_->shell_pairs[shell_idx]->evaluate_soa(x, y, z, npts,
                                                impl_->boys_table, V, ldv,
                                                omega);
}

template <typename T>
void ESPEvaluator<T>::evaluate_soa_contract(size_t shell_idx, const T *x,
                                            const T *y, const T *z, int npts,
                                            const T *F, T *G, int ld,
                                            int bf_p, int bf_q, bool both,
                                            T omega) const {
    impl_->shell_pairs[shell_idx]->evaluate_soa_contract(
        x, y, z, npts, impl_->boys_table, F, G, ld, bf_p, bf_q, both, omega);
}

template <typename T>
size_t ESPEvaluator<T>::workspace_size(size_t shell_idx, int npts) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return static_cast<size_t>(npts) * impl_->shell_pairs[shell_idx]->nherm();
}

template <typename T>
void ESPEvaluator<T>::evaluate_overlap(size_t shell_idx, T* integrals) {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }

    impl_->shell_pairs[shell_idx]->evaluate_overlap(integrals);
}

template <typename T>
int ESPEvaluator<T>::nab(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->nab();
}

template <typename T>
int ESPEvaluator<T>::nherm(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->nherm();
}

template <typename T>
std::pair<int, int> ESPEvaluator<T>::angular_momenta(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    const auto& sp = impl_->shell_pairs[shell_idx];
    return {sp->la(), sp->lb()};
}

template <typename T>
void ESPEvaluator<T>::clear() {
    impl_->shell_pairs.clear();
}

template <typename T>
size_t ESPEvaluator<T>::num_shell_pairs() const {
    return impl_->shell_pairs.size();
}

template <typename T>
bool ESPEvaluator<T>::is_spherical(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->is_spherical();
}

template <typename T>
std::array<T, 3> ESPEvaluator<T>::pair_center(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->pair_center();
}

template <typename T>
T ESPEvaluator<T>::pair_extent(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->pair_extent();
}

template <typename T>
T ESPEvaluator<T>::pair_estimate(size_t shell_idx) const {
    if (shell_idx >= impl_->shell_pairs.size()) {
        throw std::out_of_range("Invalid shell pair index");
    }
    return impl_->shell_pairs[shell_idx]->pair_estimate();
}

template <typename T>
std::array<T, 2> ESPEvaluator<T>::pair_far_estimate(size_t shell_idx) const {
    return impl_->shell_pairs[shell_idx]->pair_far_estimate();
}

template <typename T>
void ESPEvaluator<T>::resize_derivative_pairs(size_t n) {
  impl_->derivative_pairs.resize(n);
}

template <typename T>
void ESPEvaluator<T>::set_shell_pair_derivative(
    size_t idx, int la, int lb, int na_prim, int nb_prim,
    const T *exponents_a, const T *exponents_b, const T *coeffs_a,
    const T *coeffs_b, const T *A, const T *B, bool spherical) {
#define OCC_DERIV_CASE(LA, LB)                                                \
  if (la == LA && lb == LB) {                                                 \
    impl_->derivative_pairs[idx] =                                            \
        std::make_unique<DerivativePairHolderImpl<T, LA, LB>>(                \
            na_prim, nb_prim, exponents_a, exponents_b, coeffs_a, coeffs_b,   \
            A, B, spherical);                                                 \
    return;                                                                   \
  }
#define OCC_DERIV_ROW(LA)                                                     \
  OCC_DERIV_CASE(LA, 0)                                                       \
  OCC_DERIV_CASE(LA, 1)                                                       \
  OCC_DERIV_CASE(LA, 2)                                                       \
  OCC_DERIV_CASE(LA, 3)                                                       \
  OCC_DERIV_CASE(LA, 4)                                                       \
  OCC_DERIV_CASE(LA, 5)
  OCC_DERIV_ROW(0)
  OCC_DERIV_ROW(1)
  OCC_DERIV_ROW(2)
  OCC_DERIV_ROW(3)
  OCC_DERIV_ROW(4)
  OCC_DERIV_ROW(5)
#undef OCC_DERIV_ROW
#undef OCC_DERIV_CASE
  throw std::runtime_error("ESP derivative pair: angular momentum > 5");
}

template <typename T>
void ESPEvaluator<T>::evaluate_soa_value_and_derivative_contraction(
    size_t idx, const T *x, const T *y, const T *z, int npts, T *A, int lda,
    const T *W, int ldw, int ncentres, T *out, T omega) const {
  impl_->derivative_pairs[idx]->evaluate_fused(x, y, z, npts,
                                               impl_->boys_table, A, lda, W,
                                               ldw, ncentres, out, omega);
}

template <typename T>
void ESPEvaluator<T>::resize_hessian_pairs(size_t n) {
  impl_->hessian_pairs.resize(n);
}

template <typename T>
void ESPEvaluator<T>::set_shell_pair_hessian(
    size_t idx, int la, int lb, int na_prim, int nb_prim,
    const T *exponents_a, const T *exponents_b, const T *coeffs_a,
    const T *coeffs_b, const T *A, const T *B, bool spherical) {
#define OCC_HESS_CASE(LA, LB)                                                 \
  if (la == LA && lb == LB) {                                                 \
    impl_->hessian_pairs[idx] = std::make_unique<HessianPairHolderImpl<T, LA, LB>>( \
        na_prim, nb_prim, exponents_a, exponents_b, coeffs_a, coeffs_b, A, B, \
        spherical);                                                           \
    return;                                                                   \
  }
#define OCC_HESS_ROW(LA)                                                      \
  OCC_HESS_CASE(LA, 0)                                                        \
  OCC_HESS_CASE(LA, 1)                                                        \
  OCC_HESS_CASE(LA, 2)                                                        \
  OCC_HESS_CASE(LA, 3)                                                        \
  OCC_HESS_CASE(LA, 4)
  OCC_HESS_ROW(0)
  OCC_HESS_ROW(1)
  OCC_HESS_ROW(2)
  OCC_HESS_ROW(3)
  OCC_HESS_ROW(4)
#undef OCC_HESS_ROW
#undef OCC_HESS_CASE
  throw std::runtime_error("ESP Hessian pair: angular momentum > 4");
}

template <typename T>
void ESPEvaluator<T>::evaluate_soa_hessian(size_t idx, const T *x, const T *y,
                                           const T *z, int npts, T *V, int ldv,
                                           const T *W, int ldw, T *out,
                                           T omega) const {
  impl_->hessian_pairs[idx]->evaluate(x, y, z, npts, impl_->boys_table, V, ldv,
                                      W, ldw, out, omega);
}

template class ESPEvaluator<float>;
template class ESPEvaluator<double>;

} // namespace occ::ints
