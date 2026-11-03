#pragma once
// Shell-pair contraction shared by the seminumerical exchange builds and
// their derivatives.
#include <cstddef>
#include <occ/ints/boys_simd.h>

namespace occ::qm::cosx::detail {

// G_q -= F_p V and, for p != q, G_p -= F_q V^T, for one shell pair's
// integrals V[(a * nq + b) * ld + i] on a block of ld points (a multiple of
// NB). F and G are column-major with leading dimension ld. Each output column
// is accumulated over the other shell's functions in registers, NB points at
// a time, so G is read and written once per pair rather than once per term.
template <int NB>
__attribute__((always_inline)) inline void
contract_pair_body(const double *F, double *G, int ld, const double *V,
                   int bf_p, int np, int bf_q, int nq, bool both) {
  for (int i0 = 0; i0 < ld; i0 += NB) {
    for (int b = 0; b < nq; ++b) {
      double acc[NB] = {};
      for (int a = 0; a < np; ++a) {
        const double *f = F + static_cast<size_t>(bf_p + a) * ld + i0;
        const double *v = V + static_cast<size_t>(a * nq + b) * ld + i0;
        for (int i = 0; i < NB; ++i)
          acc[i] += f[i] * v[i];
      }
      double *g = G + static_cast<size_t>(bf_q + b) * ld + i0;
      for (int i = 0; i < NB; ++i)
        g[i] -= acc[i];
    }
    if (!both)
      continue;
    for (int a = 0; a < np; ++a) {
      double acc[NB] = {};
      for (int b = 0; b < nq; ++b) {
        const double *f = F + static_cast<size_t>(bf_q + b) * ld + i0;
        const double *v = V + static_cast<size_t>(a * nq + b) * ld + i0;
        for (int i = 0; i < NB; ++i)
          acc[i] += f[i] * v[i];
      }
      double *g = G + static_cast<size_t>(bf_p + a) * ld + i0;
      for (int i = 0; i < NB; ++i)
        g[i] -= acc[i];
    }
  }
}

inline void contract_pair_generic(const double *F, double *G, int ld,
                                  const double *V, int bf_p, int np, int bf_q,
                                  int nq, bool both) {
  contract_pair_body<16>(F, G, ld, V, bf_p, np, bf_q, nq, both);
}

// The same with point-independent F, G_j for the columns j of a coefficient
// matrix C (nbf x nc, column-major, leading dimension ldc):
// G_j,q += C_p,j V and, for p != q, G_j,p += C_q,j V^T, G_j at G + j * ldg.
template <int NB>
__attribute__((always_inline)) inline void
contract_pair_coefficients_body(const double *C, int ldc, int nc, double *G,
                                size_t ldg, int ld, const double *V, int bf_p,
                                int np, int bf_q, int nq, bool both) {
  for (int j = 0; j < nc; ++j) {
    const double *c = C + static_cast<size_t>(j) * ldc;
    double *Gj = G + j * ldg;
    for (int i0 = 0; i0 < ld; i0 += NB) {
      for (int b = 0; b < nq; ++b) {
        double acc[NB] = {};
        for (int a = 0; a < np; ++a) {
          const double f = c[bf_p + a];
          const double *v = V + static_cast<size_t>(a * nq + b) * ld + i0;
          for (int i = 0; i < NB; ++i)
            acc[i] += f * v[i];
        }
        double *g = Gj + static_cast<size_t>(bf_q + b) * ld + i0;
        for (int i = 0; i < NB; ++i)
          g[i] += acc[i];
      }
      if (!both)
        continue;
      for (int a = 0; a < np; ++a) {
        double acc[NB] = {};
        for (int b = 0; b < nq; ++b) {
          const double f = c[bf_q + b];
          const double *v = V + static_cast<size_t>(a * nq + b) * ld + i0;
          for (int i = 0; i < NB; ++i)
            acc[i] += f * v[i];
        }
        double *g = Gj + static_cast<size_t>(bf_p + a) * ld + i0;
        for (int i = 0; i < NB; ++i)
          g[i] += acc[i];
      }
    }
  }
}

inline void contract_pair_coefficients_generic(
    const double *C, int ldc, int nc, double *G, size_t ldg, int ld,
    const double *V, int bf_p, int np, int bf_q, int nq, bool both) {
  contract_pair_coefficients_body<16>(C, ldc, nc, G, ldg, ld, V, bf_p, np,
                                      bf_q, nq, both);
}

#if OCC_SIMD_X86_DISPATCH
__attribute__((target("avx2,fma"))) inline void
contract_pair_avx2(const double *F, double *G, int ld, const double *V,
                   int bf_p, int np, int bf_q, int nq, bool both) {
  contract_pair_body<16>(F, G, ld, V, bf_p, np, bf_q, nq, both);
}

__attribute__((target("avx2,fma"))) inline void contract_pair_coefficients_avx2(
    const double *C, int ldc, int nc, double *G, size_t ldg, int ld,
    const double *V, int bf_p, int np, int bf_q, int nq, bool both) {
  contract_pair_coefficients_body<16>(C, ldc, nc, G, ldg, ld, V, bf_p, np,
                                      bf_q, nq, both);
}

inline void contract_pair(const double *F, double *G, int ld, const double *V,
                          int bf_p, int np, int bf_q, int nq, bool both) {
  if (occ::ints::simd::cpu_has_avx2_fma())
    contract_pair_avx2(F, G, ld, V, bf_p, np, bf_q, nq, both);
  else
    contract_pair_generic(F, G, ld, V, bf_p, np, bf_q, nq, both);
}

inline void contract_pair_coefficients(const double *C, int ldc, int nc,
                                       double *G, size_t ldg, int ld,
                                       const double *V, int bf_p, int np,
                                       int bf_q, int nq, bool both) {
  if (occ::ints::simd::cpu_has_avx2_fma())
    contract_pair_coefficients_avx2(C, ldc, nc, G, ldg, ld, V, bf_p, np, bf_q,
                                    nq, both);
  else
    contract_pair_coefficients_generic(C, ldc, nc, G, ldg, ld, V, bf_p, np,
                                       bf_q, nq, both);
}
#else
inline void contract_pair(const double *F, double *G, int ld, const double *V,
                          int bf_p, int np, int bf_q, int nq, bool both) {
  contract_pair_generic(F, G, ld, V, bf_p, np, bf_q, nq, both);
}

inline void contract_pair_coefficients(const double *C, int ldc, int nc,
                                       double *G, size_t ldg, int ld,
                                       const double *V, int bf_p, int np,
                                       int bf_q, int nq, bool both) {
  contract_pair_coefficients_generic(C, ldc, nc, G, ldg, ld, V, bf_p, np,
                                     bf_q, nq, both);
}
#endif

} // namespace occ::qm::cosx::detail
