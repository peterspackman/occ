#pragma once
#include <occ/ints/rints.h>
#include <occ/mults/cartesian_kernels.h>
#include <cmath>
#include <cstring>

// Platform detection (same pattern as sorted_k_distances.h)
#if !defined(OCC_DISABLE_SIMD) && (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#define MULTS_HAS_NEON 1
#define MULTS_HAS_AVX2 0
#define MULTS_SIMD_WIDTH 2
#elif !defined(OCC_DISABLE_SIMD) && (defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__)))
#include <immintrin.h>
#define MULTS_HAS_NEON 0
#define MULTS_HAS_AVX2 1
#define MULTS_SIMD_WIDTH 4
#else
#define MULTS_HAS_NEON 0
#define MULTS_HAS_AVX2 0
#define MULTS_SIMD_WIDTH 2 // scalar batch of 2
#endif

namespace occ::mults {

using occ::ints::nherm;
using occ::ints::nhermsum;
using occ::ints::hermite_index;

/// Batch size for SIMD T-tensor computation.
inline constexpr int simd_batch_size = MULTS_SIMD_WIDTH;

/// SOA interaction tensor batch: BatchSize tensors stored interleaved.
///
/// Layout: data[hermite_index(t,u,v) * BatchSize + batch_idx]
/// This enables vectorized load/store across the batch dimension.
///
/// @tparam MaxL Maximum total rank
/// @tparam BatchSize Number of tensors in the batch
template <int MaxL, int BatchSize = simd_batch_size>
struct InteractionTensorBatch {
    static constexpr int entries = nhermsum(MaxL);
    static constexpr int total_size = entries * BatchSize;
    alignas(64) double data[total_size];

    InteractionTensorBatch() { std::memset(data, 0, sizeof(data)); }

    double &operator()(int t, int u, int v, int b) {
        return data[hermite_index(t, u, v) * BatchSize + b];
    }

    double operator()(int t, int u, int v, int b) const {
        return data[hermite_index(t, u, v) * BatchSize + b];
    }

    /// Pointer to the BatchSize-wide slot for a given (t,u,v).
    double *slot(int t, int u, int v) {
        return &data[hermite_index(t, u, v) * BatchSize];
    }

    const double *slot(int t, int u, int v) const {
        return &data[hermite_index(t, u, v) * BatchSize];
    }
};

// ======================================================================
// NEON (aarch64) batch T-tensor: 2-wide
// ======================================================================

#if MULTS_HAS_NEON

template <int MaxL>
void compute_interaction_tensor_batch_neon(
    const double *__restrict__ Rx,
    const double *__restrict__ Ry,
    const double *__restrict__ Rz,
    InteractionTensorBatch<MaxL, 2> &T) {

    constexpr int B = 2;
    constexpr int nsz = nhermsum(MaxL);
    constexpr int num_aux = MaxL + 2;

    // Working memory: num_aux auxiliary levels x nsz entries x B batch
    alignas(64) double R_all[num_aux * nsz * B];
    std::memset(R_all, 0, sizeof(R_all));

    auto R_m = [&](int m, int idx) -> double * {
        return &R_all[(m * nsz + idx) * B];
    };

    // Load displacement components
    float64x2_t vRx = vld1q_f64(Rx);
    float64x2_t vRy = vld1q_f64(Ry);
    float64x2_t vRz = vld1q_f64(Rz);

    // R^2 = Rx^2 + Ry^2 + Rz^2
    float64x2_t vR2 = vmulq_f64(vRx, vRx);
    vR2 = vfmaq_f64(vR2, vRy, vRy);
    vR2 = vfmaq_f64(vR2, vRz, vRz);

    // R = sqrt(R2), invR = 1/R, invR2 = 1/R2
    float64x2_t vR = vsqrtq_f64(vR2);
    float64x2_t vOne = vdupq_n_f64(1.0);
    float64x2_t vInvR = vdivq_f64(vOne, vR);
    float64x2_t vInvR2 = vmulq_f64(vInvR, vInvR);

    // Base case: T^{(0)}_{000} = 1/R
    vst1q_f64(R_m(0, 0), vInvR);

    // T^{(m)}_{000} = -(2m-1) * invR2 * T^{(m-1)}_{000}
    for (int m = 1; m <= MaxL; ++m) {
        float64x2_t prev = vld1q_f64(R_m(m - 1, 0));
        float64x2_t coeff = vdupq_n_f64(-(2.0 * m - 1.0));
        float64x2_t result = vmulq_f64(coeff, vmulq_f64(vInvR2, prev));
        vst1q_f64(R_m(m, 0), result);
    }

    // Build up t index (x-direction)
    for (int t = 0; t < MaxL; ++t) {
        int hi_t = hermite_index(t, 0, 0);
        int hi_t1 = hermite_index(t + 1, 0, 0);
        int hi_tm1 = (t > 0) ? hermite_index(t - 1, 0, 0) : 0;

        for (int m = 0; m <= MaxL - t - 1; ++m) {
            float64x2_t val = vmulq_f64(vRx, vld1q_f64(R_m(m + 1, hi_t)));
            if (t > 0) {
                float64x2_t prev = vld1q_f64(R_m(m + 1, hi_tm1));
                val = vfmaq_f64(val, vdupq_n_f64((double)t), prev);
            }
            vst1q_f64(R_m(m, hi_t1), val);
        }
    }

    // Build up u index (y-direction)
    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u < MaxL - t; ++u) {
            int hi_tu = hermite_index(t, u, 0);
            int hi_tu1 = hermite_index(t, u + 1, 0);
            int hi_tum1 = (u > 0) ? hermite_index(t, u - 1, 0) : 0;

            for (int m = 0; m <= MaxL - t - u - 1; ++m) {
                float64x2_t val = vmulq_f64(vRy, vld1q_f64(R_m(m + 1, hi_tu)));
                if (u > 0) {
                    float64x2_t prev = vld1q_f64(R_m(m + 1, hi_tum1));
                    val = vfmaq_f64(val, vdupq_n_f64((double)u), prev);
                }
                vst1q_f64(R_m(m, hi_tu1), val);
            }
        }
    }

    // Build up v index (z-direction)
    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u <= MaxL - t; ++u) {
            for (int v = 0; v < MaxL - t - u; ++v) {
                int hi_tuv = hermite_index(t, u, v);
                int hi_tuv1 = hermite_index(t, u, v + 1);
                int hi_tuvm1 = (v > 0) ? hermite_index(t, u, v - 1) : 0;

                for (int m = 0; m <= MaxL - t - u - v - 1; ++m) {
                    float64x2_t val = vmulq_f64(vRz, vld1q_f64(R_m(m + 1, hi_tuv)));
                    if (v > 0) {
                        float64x2_t prev = vld1q_f64(R_m(m + 1, hi_tuvm1));
                        val = vfmaq_f64(val, vdupq_n_f64((double)v), prev);
                    }
                    vst1q_f64(R_m(m, hi_tuv1), val);
                }
            }
        }
    }

    // Extract m=0 results into the batch tensor
    for (int i = 0; i < nsz; ++i) {
        vst1q_f64(&T.data[i * B], vld1q_f64(R_m(0, i)));
    }
}

#endif // MULTS_HAS_NEON

// ======================================================================
// AVX2 (x86-64) batch T-tensor: 4-wide
// ======================================================================

#if MULTS_HAS_AVX2

template <int MaxL>
void compute_interaction_tensor_batch_avx2(
    const double *__restrict__ Rx,
    const double *__restrict__ Ry,
    const double *__restrict__ Rz,
    InteractionTensorBatch<MaxL, 4> &T) {

    constexpr int B = 4;
    constexpr int nsz = nhermsum(MaxL);
    constexpr int num_aux = MaxL + 2;

    alignas(64) double R_all[num_aux * nsz * B];
    std::memset(R_all, 0, sizeof(R_all));

    auto R_m = [&](int m, int idx) -> double * {
        return &R_all[(m * nsz + idx) * B];
    };

    __m256d vRx = _mm256_load_pd(Rx);
    __m256d vRy = _mm256_load_pd(Ry);
    __m256d vRz = _mm256_load_pd(Rz);

    __m256d vR2 = _mm256_mul_pd(vRx, vRx);
    vR2 = _mm256_fmadd_pd(vRy, vRy, vR2);
    vR2 = _mm256_fmadd_pd(vRz, vRz, vR2);

    __m256d vR = _mm256_sqrt_pd(vR2);
    __m256d vOne = _mm256_set1_pd(1.0);
    __m256d vInvR = _mm256_div_pd(vOne, vR);
    __m256d vInvR2 = _mm256_mul_pd(vInvR, vInvR);

    _mm256_store_pd(R_m(0, 0), vInvR);

    for (int m = 1; m <= MaxL; ++m) {
        __m256d prev = _mm256_load_pd(R_m(m - 1, 0));
        __m256d coeff = _mm256_set1_pd(-(2.0 * m - 1.0));
        __m256d result = _mm256_mul_pd(coeff, _mm256_mul_pd(vInvR2, prev));
        _mm256_store_pd(R_m(m, 0), result);
    }

    for (int t = 0; t < MaxL; ++t) {
        int hi_t = hermite_index(t, 0, 0);
        int hi_t1 = hermite_index(t + 1, 0, 0);
        int hi_tm1 = (t > 0) ? hermite_index(t - 1, 0, 0) : 0;
        for (int m = 0; m <= MaxL - t - 1; ++m) {
            __m256d val = _mm256_mul_pd(vRx, _mm256_load_pd(R_m(m + 1, hi_t)));
            if (t > 0) {
                val = _mm256_fmadd_pd(_mm256_set1_pd((double)t),
                                       _mm256_load_pd(R_m(m + 1, hi_tm1)), val);
            }
            _mm256_store_pd(R_m(m, hi_t1), val);
        }
    }

    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u < MaxL - t; ++u) {
            int hi_tu = hermite_index(t, u, 0);
            int hi_tu1 = hermite_index(t, u + 1, 0);
            int hi_tum1 = (u > 0) ? hermite_index(t, u - 1, 0) : 0;
            for (int m = 0; m <= MaxL - t - u - 1; ++m) {
                __m256d val = _mm256_mul_pd(vRy, _mm256_load_pd(R_m(m + 1, hi_tu)));
                if (u > 0) {
                    val = _mm256_fmadd_pd(_mm256_set1_pd((double)u),
                                           _mm256_load_pd(R_m(m + 1, hi_tum1)), val);
                }
                _mm256_store_pd(R_m(m, hi_tu1), val);
            }
        }
    }

    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u <= MaxL - t; ++u) {
            for (int v = 0; v < MaxL - t - u; ++v) {
                int hi_tuv = hermite_index(t, u, v);
                int hi_tuv1 = hermite_index(t, u, v + 1);
                int hi_tuvm1 = (v > 0) ? hermite_index(t, u, v - 1) : 0;
                for (int m = 0; m <= MaxL - t - u - v - 1; ++m) {
                    __m256d val = _mm256_mul_pd(vRz, _mm256_load_pd(R_m(m + 1, hi_tuv)));
                    if (v > 0) {
                        val = _mm256_fmadd_pd(_mm256_set1_pd((double)v),
                                               _mm256_load_pd(R_m(m + 1, hi_tuvm1)), val);
                    }
                    _mm256_store_pd(R_m(m, hi_tuv1), val);
                }
            }
        }
    }

    for (int i = 0; i < nsz; ++i) {
        _mm256_store_pd(&T.data[i * B], _mm256_load_pd(R_m(0, i)));
    }
}

#endif // MULTS_HAS_AVX2

// ======================================================================
// Scalar fallback batch T-tensor: BatchSize-wide via loop
// ======================================================================

template <int MaxL, int BatchSize>
void compute_interaction_tensor_batch_scalar(
    const double *__restrict__ Rx,
    const double *__restrict__ Ry,
    const double *__restrict__ Rz,
    InteractionTensorBatch<MaxL, BatchSize> &T) {

    constexpr int B = BatchSize;
    constexpr int nsz = nhermsum(MaxL);
    constexpr int num_aux = MaxL + 2;

    alignas(64) double R_all[num_aux * nsz * B];
    std::memset(R_all, 0, sizeof(R_all));

    auto R_m = [&](int m, int idx) -> double * {
        return &R_all[(m * nsz + idx) * B];
    };

    // Base case
    for (int b = 0; b < B; ++b) {
        double R2 = Rx[b] * Rx[b] + Ry[b] * Ry[b] + Rz[b] * Rz[b];
        double invR = 1.0 / std::sqrt(R2);
        double invR2 = invR * invR;
        R_m(0, 0)[b] = invR;
        for (int m = 1; m <= MaxL; ++m) {
            R_m(m, 0)[b] = -(2.0 * m - 1.0) * invR2 * R_m(m - 1, 0)[b];
        }
    }

    // Build up t index
    for (int t = 0; t < MaxL; ++t) {
        int hi_t = hermite_index(t, 0, 0);
        int hi_t1 = hermite_index(t + 1, 0, 0);
        int hi_tm1 = (t > 0) ? hermite_index(t - 1, 0, 0) : 0;
        for (int m = 0; m <= MaxL - t - 1; ++m) {
            for (int b = 0; b < B; ++b) {
                double val = Rx[b] * R_m(m + 1, hi_t)[b];
                if (t > 0)
                    val += t * R_m(m + 1, hi_tm1)[b];
                R_m(m, hi_t1)[b] = val;
            }
        }
    }

    // Build up u index
    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u < MaxL - t; ++u) {
            int hi_tu = hermite_index(t, u, 0);
            int hi_tu1 = hermite_index(t, u + 1, 0);
            int hi_tum1 = (u > 0) ? hermite_index(t, u - 1, 0) : 0;
            for (int m = 0; m <= MaxL - t - u - 1; ++m) {
                for (int b = 0; b < B; ++b) {
                    double val = Ry[b] * R_m(m + 1, hi_tu)[b];
                    if (u > 0)
                        val += u * R_m(m + 1, hi_tum1)[b];
                    R_m(m, hi_tu1)[b] = val;
                }
            }
        }
    }

    // Build up v index
    for (int t = 0; t <= MaxL; ++t) {
        for (int u = 0; u <= MaxL - t; ++u) {
            for (int v = 0; v < MaxL - t - u; ++v) {
                int hi_tuv = hermite_index(t, u, v);
                int hi_tuv1 = hermite_index(t, u, v + 1);
                int hi_tuvm1 = (v > 0) ? hermite_index(t, u, v - 1) : 0;
                for (int m = 0; m <= MaxL - t - u - v - 1; ++m) {
                    for (int b = 0; b < B; ++b) {
                        double val = Rz[b] * R_m(m + 1, hi_tuv)[b];
                        if (v > 0)
                            val += v * R_m(m + 1, hi_tuvm1)[b];
                        R_m(m, hi_tuv1)[b] = val;
                    }
                }
            }
        }
    }

    // Extract m=0
    for (int i = 0; i < nsz; ++i) {
        for (int b = 0; b < B; ++b) {
            T.data[i * B + b] = R_m(0, i)[b];
        }
    }
}

// ======================================================================
// Dispatch to best available implementation
// ======================================================================

#if MULTS_HAS_NEON

template <int MaxL>
inline void compute_interaction_tensor_batch(
    const double *__restrict__ Rx,
    const double *__restrict__ Ry,
    const double *__restrict__ Rz,
    InteractionTensorBatch<MaxL, 2> &T) {
    compute_interaction_tensor_batch_neon<MaxL>(Rx, Ry, Rz, T);
}

#elif MULTS_HAS_AVX2

template <int MaxL>
inline void compute_interaction_tensor_batch(
    const double *__restrict__ Rx,
    const double *__restrict__ Ry,
    const double *__restrict__ Rz,
    InteractionTensorBatch<MaxL, 4> &T) {
    compute_interaction_tensor_batch_avx2<MaxL>(Rx, Ry, Rz, T);
}

#else

template <int MaxL>
inline void compute_interaction_tensor_batch(
    const double *__restrict__ Rx,
    const double *__restrict__ Ry,
    const double *__restrict__ Rz,
    InteractionTensorBatch<MaxL, simd_batch_size> &T) {
    compute_interaction_tensor_batch_scalar<MaxL, simd_batch_size>(Rx, Ry, Rz, T);
}

#endif

/// Contract one pair from a batch tensor with precomputed multipoles.
///
/// Extracts the b-th tensor from the batch and contracts with
/// the preweighted multipoles.
template <int Order, int BatchSize>
double contract_ranked_from_batch(
    const CartesianMultipole<4> &A, int rankA,
    const InteractionTensorBatch<Order, BatchSize> &T, int b,
    const CartesianMultipole<4> &B, int rankB) {

    using namespace kernel_detail;

    const int nA = nhermsum(rankA);
    const int nB = nhermsum(rankB);

    alignas(64) double wA[nhermsum(4)];
    alignas(64) double wB[nhermsum(4)];

    for (int i = 0; i < nA; ++i)
        wA[i] = weights4.sign_inv_fact[i] * A.data[i];
    for (int j = 0; j < nB; ++j)
        wB[j] = weights4.inv_fact[j] * B.data[j];

    double energy = 0.0;
    for (int i = 0; i < nA; ++i) {
        if (wA[i] == 0.0) continue;
        auto [ta, ua, va] = tuv4.entries[i];
        double eA = 0.0;
        for (int j = 0; j < nB; ++j) {
            auto [tb, ub, vb] = tuv4.entries[j];
            eA += T.data[hermite_index(ta + tb, ua + ub, va + vb) * BatchSize + b]
                  * wB[j];
        }
        energy += wA[i] * eA;
    }
    return energy;
}

} // namespace occ::mults

// Clean up macros (don't leak into user code)
#undef MULTS_HAS_NEON
#undef MULTS_HAS_AVX2
