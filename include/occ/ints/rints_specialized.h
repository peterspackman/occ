#pragma once
#include <occ/ints/boys.h>
#include <occ/ints/rints.h>
#include <Eigen/Core>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OCC_GPU_ENABLED __host__ __device__
#define OCC_GPU_INLINE __forceinline__
#else
#define OCC_GPU_ENABLED
#define OCC_GPU_INLINE inline
#endif

namespace occ::ints {

// ============================================================================
// Specialized R-integral kernels for L=0 to L=6
// These kernels are optimized for batch processing over grid points
// ============================================================================

// ----------------------------------------------------------------------------
// L = 0: 1 R-integral component
// R(0,0,0;m=0)
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_L0_batch(const T* boys_table, T p, int npts,
                              const T* PCx, const T* PCy, const T* PCz,
                              T* R_out) {
    // For each point, compute R(0,0,0;0) = F0(p*PC²)
    for (int pt = 0; pt < npts; ++pt) {
        const T PC2 = PCx[pt] * PCx[pt] + PCy[pt] * PCy[pt] + PCz[pt] * PCz[pt];
        const T Tp = p * PC2;

        T Fm[1];
        boys_evaluate<T, 1, BoysParams>(boys_table, Tp, 0, Fm);

        R_out[pt] = Fm[0];
    }
}

// ----------------------------------------------------------------------------
// L = 1: 4 R-integral components
// R(0,0,0), R(1,0,0), R(0,1,0), R(0,0,1)
// Indices: 0, 1, 2, 3
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_L1_batch(const T* boys_table, T p, int npts,
                              const T* PCx, const T* PCy, const T* PCz,
                              T* R_out) {
    constexpr int nherm = 4;
    const T neg_2p = T(-2) * p;

    for (int pt = 0; pt < npts; ++pt) {
        const T PCx_val = PCx[pt];
        const T PCy_val = PCy[pt];
        const T PCz_val = PCz[pt];
        const T PC2 = PCx_val * PCx_val + PCy_val * PCy_val + PCz_val * PCz_val;
        const T Tp = p * PC2;

        T Fm[2];
        boys_evaluate<T, 2, BoysParams>(boys_table, Tp, 0, Fm);

        const T R_000_m0 = Fm[0];
        const T R_000_m1 = neg_2p * Fm[1];

        T* out = R_out + pt * nherm;
        out[0] = R_000_m0;
        out[1] = PCx_val * R_000_m1;
        out[2] = PCy_val * R_000_m1;
        out[3] = PCz_val * R_000_m1;
    }
}

// ----------------------------------------------------------------------------
// L = 2: 10 R-integral components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_L2_batch(const T* boys_table, T p, int npts,
                              const T* PCx, const T* PCy, const T* PCz,
                              T* R_out) {
    constexpr int nherm = 10;
    const T neg_2p = T(-2) * p;

    for (int pt = 0; pt < npts; ++pt) {
        const T PCx_val = PCx[pt];
        const T PCy_val = PCy[pt];
        const T PCz_val = PCz[pt];
        const T PC2 = PCx_val * PCx_val + PCy_val * PCy_val + PCz_val * PCz_val;
        const T Tp = p * PC2;

        T Fm[3];
        boys_evaluate<T, 3, BoysParams>(boys_table, Tp, 0, Fm);

        const T R_000_m0 = Fm[0];
        const T R_000_m1 = neg_2p * Fm[1];
        const T R_000_m2 = neg_2p * neg_2p * Fm[2];

        const T R_100_m0 = PCx_val * R_000_m1;
        const T R_100_m1 = PCx_val * R_000_m2;
        const T R_010_m0 = PCy_val * R_000_m1;
        const T R_010_m1 = PCy_val * R_000_m2;
        const T R_001_m0 = PCz_val * R_000_m1;
        const T R_001_m1 = PCz_val * R_000_m2;

        T* out = R_out + pt * nherm;
        out[0] = R_000_m0;
        out[1] = R_100_m0;
        out[2] = R_010_m0;
        out[3] = R_001_m0;
        out[4] = PCx_val * R_100_m1 + R_000_m1;
        out[5] = PCy_val * R_100_m1;
        out[6] = PCz_val * R_100_m1;
        out[7] = PCy_val * R_010_m1 + R_000_m1;
        out[8] = PCz_val * R_010_m1;
        out[9] = PCz_val * R_001_m1 + R_000_m1;
    }
}

// ----------------------------------------------------------------------------
// L = 3: 20 R-integral components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_L3_batch(const T* boys_table, T p, int npts,
                              const T* PCx, const T* PCy, const T* PCz,
                              T* R_out) {
    constexpr int nherm = 20;
    const T neg_2p = T(-2) * p;

    for (int pt = 0; pt < npts; ++pt) {
        const T PCx_val = PCx[pt];
        const T PCy_val = PCy[pt];
        const T PCz_val = PCz[pt];
        const T PC2 = PCx_val * PCx_val + PCy_val * PCy_val + PCz_val * PCz_val;
        const T Tp = p * PC2;

        T Fm[4];
        boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

        const T R_000_m0 = Fm[0];
        const T R_000_m1 = neg_2p * Fm[1];
        const T R_000_m2 = neg_2p * neg_2p * Fm[2];
        const T R_000_m3 = neg_2p * neg_2p * neg_2p * Fm[3];

        // X-direction
        const T R_100_m0 = PCx_val * R_000_m1;
        const T R_100_m1 = PCx_val * R_000_m2;
        const T R_100_m2 = PCx_val * R_000_m3;
        const T R_200_m0 = PCx_val * R_100_m1 + R_000_m1;
        const T R_200_m1 = PCx_val * R_100_m2 + R_000_m2;
        const T R_300_m0 = PCx_val * R_200_m1 + T(2) * R_100_m1;

        // Y-direction
        const T R_010_m0 = PCy_val * R_000_m1;
        const T R_010_m1 = PCy_val * R_000_m2;
        const T R_010_m2 = PCy_val * R_000_m3;
        const T R_110_m0 = PCy_val * R_100_m1;
        const T R_110_m1 = PCy_val * R_100_m2;
        const T R_210_m0 = PCy_val * R_200_m1;
        const T R_020_m0 = PCy_val * R_010_m1 + R_000_m1;
        const T R_020_m1 = PCy_val * R_010_m2 + R_000_m2;
        const T R_120_m0 = PCy_val * R_110_m1 + R_100_m1;
        const T R_030_m0 = PCy_val * R_020_m1 + T(2) * R_010_m1;

        // Z-direction
        const T R_001_m0 = PCz_val * R_000_m1;
        const T R_001_m1 = PCz_val * R_000_m2;
        const T R_001_m2 = PCz_val * R_000_m3;
        const T R_101_m0 = PCz_val * R_100_m1;
        const T R_101_m1 = PCz_val * R_100_m2;
        const T R_201_m0 = PCz_val * R_200_m1;
        const T R_011_m0 = PCz_val * R_010_m1;
        const T R_011_m1 = PCz_val * R_010_m2;
        const T R_111_m0 = PCz_val * R_110_m1;
        const T R_021_m0 = PCz_val * R_020_m1;
        const T R_002_m0 = PCz_val * R_001_m1 + R_000_m1;
        const T R_002_m1 = PCz_val * R_001_m2 + R_000_m2;
        const T R_102_m0 = PCz_val * R_101_m1 + R_100_m1;
        const T R_012_m0 = PCz_val * R_011_m1 + R_010_m1;
        const T R_003_m0 = PCz_val * R_002_m1 + T(2) * R_001_m1;

        T* out = R_out + pt * nherm;
        out[0] = R_000_m0;   out[1] = R_100_m0;   out[2] = R_010_m0;
        out[3] = R_001_m0;   out[4] = R_200_m0;   out[5] = R_110_m0;
        out[6] = R_101_m0;   out[7] = R_020_m0;   out[8] = R_011_m0;
        out[9] = R_002_m0;   out[10] = R_300_m0;  out[11] = R_210_m0;
        out[12] = R_201_m0;  out[13] = R_120_m0;  out[14] = R_111_m0;
        out[15] = R_102_m0;  out[16] = R_030_m0;  out[17] = R_021_m0;
        out[18] = R_012_m0;  out[19] = R_003_m0;
    }
}

// ----------------------------------------------------------------------------
// L = 4: 35 R-integral components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_L4_batch(const T* boys_table, T p, int npts,
                              const T* PCx, const T* PCy, const T* PCz,
                              T* R_out) {
    constexpr int nherm = 35;
    const T neg_2p = T(-2) * p;

    for (int pt = 0; pt < npts; ++pt) {
        const T PCx_val = PCx[pt];
        const T PCy_val = PCy[pt];
        const T PCz_val = PCz[pt];
        const T PC2 = PCx_val * PCx_val + PCy_val * PCy_val + PCz_val * PCz_val;
        const T Tp = p * PC2;

        T Fm[5];
        boys_evaluate<T, 5, BoysParams>(boys_table, Tp, 0, Fm);

        const T R_000_m0 = Fm[0];
        const T R_000_m1 = neg_2p * Fm[1];
        const T R_000_m2 = neg_2p * neg_2p * Fm[2];
        const T R_000_m3 = neg_2p * neg_2p * neg_2p * Fm[3];
        const T R_000_m4 = neg_2p * neg_2p * neg_2p * neg_2p * Fm[4];

        // X-direction
        const T R_100_m0 = PCx_val * R_000_m1;
        const T R_100_m1 = PCx_val * R_000_m2;
        const T R_100_m2 = PCx_val * R_000_m3;
        const T R_100_m3 = PCx_val * R_000_m4;
        const T R_200_m0 = PCx_val * R_100_m1 + R_000_m1;
        const T R_200_m1 = PCx_val * R_100_m2 + R_000_m2;
        const T R_200_m2 = PCx_val * R_100_m3 + R_000_m3;
        const T R_300_m0 = PCx_val * R_200_m1 + T(2) * R_100_m1;
        const T R_300_m1 = PCx_val * R_200_m2 + T(2) * R_100_m2;
        const T R_400_m0 = PCx_val * R_300_m1 + T(3) * R_200_m1;

        // Y-direction
        const T R_010_m0 = PCy_val * R_000_m1;
        const T R_010_m1 = PCy_val * R_000_m2;
        const T R_010_m2 = PCy_val * R_000_m3;
        const T R_010_m3 = PCy_val * R_000_m4;
        const T R_020_m0 = PCy_val * R_010_m1 + R_000_m1;
        const T R_020_m1 = PCy_val * R_010_m2 + R_000_m2;
        const T R_020_m2 = PCy_val * R_010_m3 + R_000_m3;
        const T R_030_m0 = PCy_val * R_020_m1 + T(2) * R_010_m1;
        const T R_030_m1 = PCy_val * R_020_m2 + T(2) * R_010_m2;
        const T R_040_m0 = PCy_val * R_030_m1 + T(3) * R_020_m1;

        const T R_110_m0 = PCy_val * R_100_m1;
        const T R_110_m1 = PCy_val * R_100_m2;
        const T R_110_m2 = PCy_val * R_100_m3;
        const T R_120_m0 = PCy_val * R_110_m1 + R_100_m1;
        const T R_120_m1 = PCy_val * R_110_m2 + R_100_m2;
        const T R_130_m0 = PCy_val * R_120_m1 + T(2) * R_110_m1;
        const T R_210_m0 = PCy_val * R_200_m1;
        const T R_210_m1 = PCy_val * R_200_m2;
        const T R_220_m0 = PCy_val * R_210_m1 + R_200_m1;
        const T R_310_m0 = PCy_val * R_300_m1;

        // Z-direction
        const T R_001_m0 = PCz_val * R_000_m1;
        const T R_001_m1 = PCz_val * R_000_m2;
        const T R_001_m2 = PCz_val * R_000_m3;
        const T R_001_m3 = PCz_val * R_000_m4;
        const T R_002_m0 = PCz_val * R_001_m1 + R_000_m1;
        const T R_002_m1 = PCz_val * R_001_m2 + R_000_m2;
        const T R_002_m2 = PCz_val * R_001_m3 + R_000_m3;
        const T R_003_m0 = PCz_val * R_002_m1 + T(2) * R_001_m1;
        const T R_003_m1 = PCz_val * R_002_m2 + T(2) * R_001_m2;
        const T R_004_m0 = PCz_val * R_003_m1 + T(3) * R_002_m1;

        const T R_101_m0 = PCz_val * R_100_m1;
        const T R_101_m1 = PCz_val * R_100_m2;
        const T R_101_m2 = PCz_val * R_100_m3;
        const T R_102_m0 = PCz_val * R_101_m1 + R_100_m1;
        const T R_102_m1 = PCz_val * R_101_m2 + R_100_m2;
        const T R_103_m0 = PCz_val * R_102_m1 + T(2) * R_101_m1;
        const T R_201_m0 = PCz_val * R_200_m1;
        const T R_201_m1 = PCz_val * R_200_m2;
        const T R_202_m0 = PCz_val * R_201_m1 + R_200_m1;
        const T R_301_m0 = PCz_val * R_300_m1;

        const T R_011_m0 = PCz_val * R_010_m1;
        const T R_011_m1 = PCz_val * R_010_m2;
        const T R_011_m2 = PCz_val * R_010_m3;
        const T R_012_m0 = PCz_val * R_011_m1 + R_010_m1;
        const T R_012_m1 = PCz_val * R_011_m2 + R_010_m2;
        const T R_013_m0 = PCz_val * R_012_m1 + T(2) * R_011_m1;
        const T R_111_m0 = PCz_val * R_110_m1;
        const T R_111_m1 = PCz_val * R_110_m2;
        const T R_112_m0 = PCz_val * R_111_m1 + R_110_m1;
        const T R_211_m0 = PCz_val * R_210_m1;

        const T R_021_m0 = PCz_val * R_020_m1;
        const T R_021_m1 = PCz_val * R_020_m2;
        const T R_022_m0 = PCz_val * R_021_m1 + R_020_m1;
        const T R_121_m0 = PCz_val * R_120_m1;
        const T R_031_m0 = PCz_val * R_030_m1;

        T* out = R_out + pt * nherm;
        out[0]=R_000_m0;  out[1]=R_100_m0;  out[2]=R_010_m0;  out[3]=R_001_m0;
        out[4]=R_200_m0;  out[5]=R_110_m0;  out[6]=R_101_m0;  out[7]=R_020_m0;
        out[8]=R_011_m0;  out[9]=R_002_m0;  out[10]=R_300_m0; out[11]=R_210_m0;
        out[12]=R_201_m0; out[13]=R_120_m0; out[14]=R_111_m0; out[15]=R_102_m0;
        out[16]=R_030_m0; out[17]=R_021_m0; out[18]=R_012_m0; out[19]=R_003_m0;
        out[20]=R_400_m0; out[21]=R_310_m0; out[22]=R_301_m0; out[23]=R_220_m0;
        out[24]=R_211_m0; out[25]=R_202_m0; out[26]=R_130_m0; out[27]=R_121_m0;
        out[28]=R_112_m0; out[29]=R_103_m0; out[30]=R_040_m0; out[31]=R_031_m0;
        out[32]=R_022_m0; out[33]=R_013_m0; out[34]=R_004_m0;
    }
}

// ----------------------------------------------------------------------------
// L = 5: 56 R-integral components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_L5_batch(const T* boys_table, T p, int npts,
                              const T* PCx, const T* PCy, const T* PCz,
                              T* R_out) {
    constexpr int nherm = 56;
    const T neg_2p = T(-2) * p;

    for (int pt = 0; pt < npts; ++pt) {
        const T PCx_val = PCx[pt];
        const T PCy_val = PCy[pt];
        const T PCz_val = PCz[pt];
        const T PC2 = PCx_val * PCx_val + PCy_val * PCy_val + PCz_val * PCz_val;
        const T Tp = p * PC2;

        T Fm[6];
        boys_evaluate<T, 6, BoysParams>(boys_table, Tp, 0, Fm);

        // Initialize base: R(0,0,0;m) = (-2p)^m * Fm
        const T R_000_m0 = Fm[0];
        const T R_000_m1 = neg_2p * Fm[1];
        const T R_000_m2 = neg_2p * neg_2p * Fm[2];
        const T R_000_m3 = neg_2p * neg_2p * neg_2p * Fm[3];
        const T R_000_m4 = neg_2p * neg_2p * neg_2p * neg_2p * Fm[4];
        const T R_000_m5 = neg_2p * neg_2p * neg_2p * neg_2p * neg_2p * Fm[5];

        // X-direction recursion: build up to t=5
        const T R_100_m0 = PCx_val * R_000_m1;
        const T R_100_m1 = PCx_val * R_000_m2;
        const T R_100_m2 = PCx_val * R_000_m3;
        const T R_100_m3 = PCx_val * R_000_m4;
        const T R_100_m4 = PCx_val * R_000_m5;

        const T R_200_m0 = PCx_val * R_100_m1 + R_000_m1;
        const T R_200_m1 = PCx_val * R_100_m2 + R_000_m2;
        const T R_200_m2 = PCx_val * R_100_m3 + R_000_m3;
        const T R_200_m3 = PCx_val * R_100_m4 + R_000_m4;

        const T R_300_m0 = PCx_val * R_200_m1 + T(2) * R_100_m1;
        const T R_300_m1 = PCx_val * R_200_m2 + T(2) * R_100_m2;
        const T R_300_m2 = PCx_val * R_200_m3 + T(2) * R_100_m3;

        const T R_400_m0 = PCx_val * R_300_m1 + T(3) * R_200_m1;
        const T R_400_m1 = PCx_val * R_300_m2 + T(3) * R_200_m2;

        const T R_500_m0 = PCx_val * R_400_m1 + T(4) * R_300_m1;

        // Y-direction: for each t, build u
        // t=0, u: 0->1->2->3->4->5
        const T R_010_m0 = PCy_val * R_000_m1;
        const T R_010_m1 = PCy_val * R_000_m2;
        const T R_010_m2 = PCy_val * R_000_m3;
        const T R_010_m3 = PCy_val * R_000_m4;
        const T R_010_m4 = PCy_val * R_000_m5;

        const T R_020_m0 = PCy_val * R_010_m1 + R_000_m1;
        const T R_020_m1 = PCy_val * R_010_m2 + R_000_m2;
        const T R_020_m2 = PCy_val * R_010_m3 + R_000_m3;
        const T R_020_m3 = PCy_val * R_010_m4 + R_000_m4;

        const T R_030_m0 = PCy_val * R_020_m1 + T(2) * R_010_m1;
        const T R_030_m1 = PCy_val * R_020_m2 + T(2) * R_010_m2;
        const T R_030_m2 = PCy_val * R_020_m3 + T(2) * R_010_m3;

        const T R_040_m0 = PCy_val * R_030_m1 + T(3) * R_020_m1;
        const T R_040_m1 = PCy_val * R_030_m2 + T(3) * R_020_m2;

        const T R_050_m0 = PCy_val * R_040_m1 + T(4) * R_030_m1;

        // t=1, u: 0->1->2->3->4
        const T R_110_m0 = PCy_val * R_100_m1;
        const T R_110_m1 = PCy_val * R_100_m2;
        const T R_110_m2 = PCy_val * R_100_m3;
        const T R_110_m3 = PCy_val * R_100_m4;

        const T R_120_m0 = PCy_val * R_110_m1 + R_100_m1;
        const T R_120_m1 = PCy_val * R_110_m2 + R_100_m2;
        const T R_120_m2 = PCy_val * R_110_m3 + R_100_m3;

        const T R_130_m0 = PCy_val * R_120_m1 + T(2) * R_110_m1;
        const T R_130_m1 = PCy_val * R_120_m2 + T(2) * R_110_m2;

        const T R_140_m0 = PCy_val * R_130_m1 + T(3) * R_120_m1;

        // t=2, u: 0->1->2->3
        const T R_210_m0 = PCy_val * R_200_m1;
        const T R_210_m1 = PCy_val * R_200_m2;
        const T R_210_m2 = PCy_val * R_200_m3;

        const T R_220_m0 = PCy_val * R_210_m1 + R_200_m1;
        const T R_220_m1 = PCy_val * R_210_m2 + R_200_m2;

        const T R_230_m0 = PCy_val * R_220_m1 + T(2) * R_210_m1;

        // t=3, u: 0->1->2
        const T R_310_m0 = PCy_val * R_300_m1;
        const T R_310_m1 = PCy_val * R_300_m2;

        const T R_320_m0 = PCy_val * R_310_m1 + R_300_m1;

        // t=4, u: 0->1
        const T R_410_m0 = PCy_val * R_400_m1;

        // Z-direction: for each (t,u), build v
        // t=0, u=0, v: 0->1->2->3->4->5
        const T R_001_m0 = PCz_val * R_000_m1;
        const T R_001_m1 = PCz_val * R_000_m2;
        const T R_001_m2 = PCz_val * R_000_m3;
        const T R_001_m3 = PCz_val * R_000_m4;
        const T R_001_m4 = PCz_val * R_000_m5;

        const T R_002_m0 = PCz_val * R_001_m1 + R_000_m1;
        const T R_002_m1 = PCz_val * R_001_m2 + R_000_m2;
        const T R_002_m2 = PCz_val * R_001_m3 + R_000_m3;
        const T R_002_m3 = PCz_val * R_001_m4 + R_000_m4;

        const T R_003_m0 = PCz_val * R_002_m1 + T(2) * R_001_m1;
        const T R_003_m1 = PCz_val * R_002_m2 + T(2) * R_001_m2;
        const T R_003_m2 = PCz_val * R_002_m3 + T(2) * R_001_m3;

        const T R_004_m0 = PCz_val * R_003_m1 + T(3) * R_002_m1;
        const T R_004_m1 = PCz_val * R_003_m2 + T(3) * R_002_m2;

        const T R_005_m0 = PCz_val * R_004_m1 + T(4) * R_003_m1;

        // t=1, u=0, v: 0->1->2->3->4
        const T R_101_m0 = PCz_val * R_100_m1;
        const T R_101_m1 = PCz_val * R_100_m2;
        const T R_101_m2 = PCz_val * R_100_m3;
        const T R_101_m3 = PCz_val * R_100_m4;

        const T R_102_m0 = PCz_val * R_101_m1 + R_100_m1;
        const T R_102_m1 = PCz_val * R_101_m2 + R_100_m2;
        const T R_102_m2 = PCz_val * R_101_m3 + R_100_m3;

        const T R_103_m0 = PCz_val * R_102_m1 + T(2) * R_101_m1;
        const T R_103_m1 = PCz_val * R_102_m2 + T(2) * R_101_m2;

        const T R_104_m0 = PCz_val * R_103_m1 + T(3) * R_102_m1;

        // t=2, u=0, v: 0->1->2->3
        const T R_201_m0 = PCz_val * R_200_m1;
        const T R_201_m1 = PCz_val * R_200_m2;
        const T R_201_m2 = PCz_val * R_200_m3;

        const T R_202_m0 = PCz_val * R_201_m1 + R_200_m1;
        const T R_202_m1 = PCz_val * R_201_m2 + R_200_m2;

        const T R_203_m0 = PCz_val * R_202_m1 + T(2) * R_201_m1;

        // t=3, u=0, v: 0->1->2
        const T R_301_m0 = PCz_val * R_300_m1;
        const T R_301_m1 = PCz_val * R_300_m2;

        const T R_302_m0 = PCz_val * R_301_m1 + R_300_m1;

        // t=4, u=0, v: 0->1
        const T R_401_m0 = PCz_val * R_400_m1;

        // t=0, u=1, v: 0->1->2->3->4
        const T R_011_m0 = PCz_val * R_010_m1;
        const T R_011_m1 = PCz_val * R_010_m2;
        const T R_011_m2 = PCz_val * R_010_m3;
        const T R_011_m3 = PCz_val * R_010_m4;

        const T R_012_m0 = PCz_val * R_011_m1 + R_010_m1;
        const T R_012_m1 = PCz_val * R_011_m2 + R_010_m2;
        const T R_012_m2 = PCz_val * R_011_m3 + R_010_m3;

        const T R_013_m0 = PCz_val * R_012_m1 + T(2) * R_011_m1;
        const T R_013_m1 = PCz_val * R_012_m2 + T(2) * R_011_m2;

        const T R_014_m0 = PCz_val * R_013_m1 + T(3) * R_012_m1;

        // t=1, u=1, v: 0->1->2->3
        const T R_111_m0 = PCz_val * R_110_m1;
        const T R_111_m1 = PCz_val * R_110_m2;
        const T R_111_m2 = PCz_val * R_110_m3;

        const T R_112_m0 = PCz_val * R_111_m1 + R_110_m1;
        const T R_112_m1 = PCz_val * R_111_m2 + R_110_m2;

        const T R_113_m0 = PCz_val * R_112_m1 + T(2) * R_111_m1;

        // t=2, u=1, v: 0->1->2
        const T R_211_m0 = PCz_val * R_210_m1;
        const T R_211_m1 = PCz_val * R_210_m2;

        const T R_212_m0 = PCz_val * R_211_m1 + R_210_m1;

        // t=3, u=1, v: 0->1
        const T R_311_m0 = PCz_val * R_310_m1;

        // t=0, u=2, v: 0->1->2->3
        const T R_021_m0 = PCz_val * R_020_m1;
        const T R_021_m1 = PCz_val * R_020_m2;
        const T R_021_m2 = PCz_val * R_020_m3;

        const T R_022_m0 = PCz_val * R_021_m1 + R_020_m1;
        const T R_022_m1 = PCz_val * R_021_m2 + R_020_m2;

        const T R_023_m0 = PCz_val * R_022_m1 + T(2) * R_021_m1;

        // t=1, u=2, v: 0->1->2
        const T R_121_m0 = PCz_val * R_120_m1;
        const T R_121_m1 = PCz_val * R_120_m2;

        const T R_122_m0 = PCz_val * R_121_m1 + R_120_m1;

        // t=2, u=2, v: 0->1
        const T R_221_m0 = PCz_val * R_220_m1;

        // t=0, u=3, v: 0->1->2
        const T R_031_m0 = PCz_val * R_030_m1;
        const T R_031_m1 = PCz_val * R_030_m2;

        const T R_032_m0 = PCz_val * R_031_m1 + R_030_m1;

        // t=1, u=3, v: 0->1
        const T R_131_m0 = PCz_val * R_130_m1;

        // t=0, u=4, v: 0->1
        const T R_041_m0 = PCz_val * R_040_m1;

        // Write output in canonical Hermite order
        T* out = R_out + pt * nherm;
        out[0] = R_000_m0;   // R(0,0,0)
        out[1] = R_100_m0;   // R(1,0,0)
        out[2] = R_010_m0;   // R(0,1,0)
        out[3] = R_001_m0;   // R(0,0,1)
        out[4] = R_200_m0;   // R(2,0,0)
        out[5] = R_110_m0;   // R(1,1,0)
        out[6] = R_101_m0;   // R(1,0,1)
        out[7] = R_020_m0;   // R(0,2,0)
        out[8] = R_011_m0;   // R(0,1,1)
        out[9] = R_002_m0;   // R(0,0,2)
        out[10] = R_300_m0;  // R(3,0,0)
        out[11] = R_210_m0;  // R(2,1,0)
        out[12] = R_201_m0;  // R(2,0,1)
        out[13] = R_120_m0;  // R(1,2,0)
        out[14] = R_111_m0;  // R(1,1,1)
        out[15] = R_102_m0;  // R(1,0,2)
        out[16] = R_030_m0;  // R(0,3,0)
        out[17] = R_021_m0;  // R(0,2,1)
        out[18] = R_012_m0;  // R(0,1,2)
        out[19] = R_003_m0;  // R(0,0,3)
        out[20] = R_400_m0;  // R(4,0,0)
        out[21] = R_310_m0;  // R(3,1,0)
        out[22] = R_301_m0;  // R(3,0,1)
        out[23] = R_220_m0;  // R(2,2,0)
        out[24] = R_211_m0;  // R(2,1,1)
        out[25] = R_202_m0;  // R(2,0,2)
        out[26] = R_130_m0;  // R(1,3,0)
        out[27] = R_121_m0;  // R(1,2,1)
        out[28] = R_112_m0;  // R(1,1,2)
        out[29] = R_103_m0;  // R(1,0,3)
        out[30] = R_040_m0;  // R(0,4,0)
        out[31] = R_031_m0;  // R(0,3,1)
        out[32] = R_022_m0;  // R(0,2,2)
        out[33] = R_013_m0;  // R(0,1,3)
        out[34] = R_004_m0;  // R(0,0,4)
        out[35] = R_500_m0;  // R(5,0,0)
        out[36] = R_410_m0;  // R(4,1,0)
        out[37] = R_401_m0;  // R(4,0,1)
        out[38] = R_320_m0;  // R(3,2,0)
        out[39] = R_311_m0;  // R(3,1,1)
        out[40] = R_302_m0;  // R(3,0,2)
        out[41] = R_230_m0;  // R(2,3,0)
        out[42] = R_221_m0;  // R(2,2,1)
        out[43] = R_212_m0;  // R(2,1,2)
        out[44] = R_203_m0;  // R(2,0,3)
        out[45] = R_140_m0;  // R(1,4,0)
        out[46] = R_131_m0;  // R(1,3,1)
        out[47] = R_122_m0;  // R(1,2,2)
        out[48] = R_113_m0;  // R(1,1,3)
        out[49] = R_104_m0;  // R(1,0,4)
        out[50] = R_050_m0;  // R(0,5,0)
        out[51] = R_041_m0;  // R(0,4,1)
        out[52] = R_032_m0;  // R(0,3,2)
        out[53] = R_023_m0;  // R(0,2,3)
        out[54] = R_014_m0;  // R(0,1,4)
        out[55] = R_005_m0;  // R(0,0,5)
    }
}

// ----------------------------------------------------------------------------
// L = 6: 84 R-integral components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_L6_batch(const T* boys_table, T p, int npts,
                              const T* PCx, const T* PCy, const T* PCz,
                              T* R_out) {
    constexpr int nherm = 84;
    const T neg_2p = T(-2) * p;
    const T neg_2p_2 = neg_2p * neg_2p;
    const T neg_2p_3 = neg_2p_2 * neg_2p;
    const T neg_2p_4 = neg_2p_3 * neg_2p;
    const T neg_2p_5 = neg_2p_4 * neg_2p;
    const T neg_2p_6 = neg_2p_5 * neg_2p;

    for (int pt = 0; pt < npts; ++pt) {
        const T pcx = PCx[pt];
        const T pcy = PCy[pt];
        const T pcz = PCz[pt];
        const T Tp = p * (pcx * pcx + pcy * pcy + pcz * pcz);

        T Fm[7];
        boys_evaluate<T, 7, BoysParams>(boys_table, Tp, 0, Fm);

        // Base values R(0,0,0;m) for m=0..6
        T R_000_m0 = Fm[0], R_000_m1 = neg_2p * Fm[1], R_000_m2 = neg_2p_2 * Fm[2];
        T R_000_m3 = neg_2p_3 * Fm[3], R_000_m4 = neg_2p_4 * Fm[4];
        T R_000_m5 = neg_2p_5 * Fm[5], R_000_m6 = neg_2p_6 * Fm[6];

        // X-direction: R(t,0,0;m) for t=1..6
        T R_100_m0 = pcx * R_000_m1, R_100_m1 = pcx * R_000_m2, R_100_m2 = pcx * R_000_m3;
        T R_100_m3 = pcx * R_000_m4, R_100_m4 = pcx * R_000_m5, R_100_m5 = pcx * R_000_m6;

        T R_200_m0 = pcx * R_100_m1 + R_000_m1, R_200_m1 = pcx * R_100_m2 + R_000_m2;
        T R_200_m2 = pcx * R_100_m3 + R_000_m3, R_200_m3 = pcx * R_100_m4 + R_000_m4;
        T R_200_m4 = pcx * R_100_m5 + R_000_m5;

        T R_300_m0 = pcx * R_200_m1 + T(2) * R_100_m1, R_300_m1 = pcx * R_200_m2 + T(2) * R_100_m2;
        T R_300_m2 = pcx * R_200_m3 + T(2) * R_100_m3, R_300_m3 = pcx * R_200_m4 + T(2) * R_100_m4;

        T R_400_m0 = pcx * R_300_m1 + T(3) * R_200_m1, R_400_m1 = pcx * R_300_m2 + T(3) * R_200_m2;
        T R_400_m2 = pcx * R_300_m3 + T(3) * R_200_m3;

        T R_500_m0 = pcx * R_400_m1 + T(4) * R_300_m1, R_500_m1 = pcx * R_400_m2 + T(4) * R_300_m2;

        T R_600_m0 = pcx * R_500_m1 + T(5) * R_400_m1;

        // Y-direction: R(0,u,0;m) for u=1..6
        T R_010_m0 = pcy * R_000_m1, R_010_m1 = pcy * R_000_m2, R_010_m2 = pcy * R_000_m3;
        T R_010_m3 = pcy * R_000_m4, R_010_m4 = pcy * R_000_m5, R_010_m5 = pcy * R_000_m6;

        T R_020_m0 = pcy * R_010_m1 + R_000_m1, R_020_m1 = pcy * R_010_m2 + R_000_m2;
        T R_020_m2 = pcy * R_010_m3 + R_000_m3, R_020_m3 = pcy * R_010_m4 + R_000_m4;
        T R_020_m4 = pcy * R_010_m5 + R_000_m5;

        T R_030_m0 = pcy * R_020_m1 + T(2) * R_010_m1, R_030_m1 = pcy * R_020_m2 + T(2) * R_010_m2;
        T R_030_m2 = pcy * R_020_m3 + T(2) * R_010_m3, R_030_m3 = pcy * R_020_m4 + T(2) * R_010_m4;

        T R_040_m0 = pcy * R_030_m1 + T(3) * R_020_m1, R_040_m1 = pcy * R_030_m2 + T(3) * R_020_m2;
        T R_040_m2 = pcy * R_030_m3 + T(3) * R_020_m3;

        T R_050_m0 = pcy * R_040_m1 + T(4) * R_030_m1, R_050_m1 = pcy * R_040_m2 + T(4) * R_030_m2;

        T R_060_m0 = pcy * R_050_m1 + T(5) * R_040_m1;

        // Mixed X-Y: R(t,u,0;m)
        T R_110_m0 = pcy * R_100_m1, R_110_m1 = pcy * R_100_m2, R_110_m2 = pcy * R_100_m3;
        T R_110_m3 = pcy * R_100_m4, R_110_m4 = pcy * R_100_m5;

        T R_120_m0 = pcy * R_110_m1 + R_100_m1, R_120_m1 = pcy * R_110_m2 + R_100_m2;
        T R_120_m2 = pcy * R_110_m3 + R_100_m3, R_120_m3 = pcy * R_110_m4 + R_100_m4;

        T R_130_m0 = pcy * R_120_m1 + T(2) * R_110_m1, R_130_m1 = pcy * R_120_m2 + T(2) * R_110_m2;
        T R_130_m2 = pcy * R_120_m3 + T(2) * R_110_m3;

        T R_140_m0 = pcy * R_130_m1 + T(3) * R_120_m1, R_140_m1 = pcy * R_130_m2 + T(3) * R_120_m2;

        T R_150_m0 = pcy * R_140_m1 + T(4) * R_130_m1;

        T R_210_m0 = pcy * R_200_m1, R_210_m1 = pcy * R_200_m2, R_210_m2 = pcy * R_200_m3;
        T R_210_m3 = pcy * R_200_m4;

        T R_220_m0 = pcy * R_210_m1 + R_200_m1, R_220_m1 = pcy * R_210_m2 + R_200_m2;
        T R_220_m2 = pcy * R_210_m3 + R_200_m3;

        T R_230_m0 = pcy * R_220_m1 + T(2) * R_210_m1, R_230_m1 = pcy * R_220_m2 + T(2) * R_210_m2;

        T R_240_m0 = pcy * R_230_m1 + T(3) * R_220_m1;

        T R_310_m0 = pcy * R_300_m1, R_310_m1 = pcy * R_300_m2, R_310_m2 = pcy * R_300_m3;

        T R_320_m0 = pcy * R_310_m1 + R_300_m1, R_320_m1 = pcy * R_310_m2 + R_300_m2;

        T R_330_m0 = pcy * R_320_m1 + T(2) * R_310_m1;

        T R_410_m0 = pcy * R_400_m1, R_410_m1 = pcy * R_400_m2;

        T R_420_m0 = pcy * R_410_m1 + R_400_m1;

        T R_510_m0 = pcy * R_500_m1;

        // Z-direction: R(0,0,v;m) and mixed terms
        T R_001_m0 = pcz * R_000_m1, R_001_m1 = pcz * R_000_m2, R_001_m2 = pcz * R_000_m3;
        T R_001_m3 = pcz * R_000_m4, R_001_m4 = pcz * R_000_m5, R_001_m5 = pcz * R_000_m6;

        T R_002_m0 = pcz * R_001_m1 + R_000_m1, R_002_m1 = pcz * R_001_m2 + R_000_m2;
        T R_002_m2 = pcz * R_001_m3 + R_000_m3, R_002_m3 = pcz * R_001_m4 + R_000_m4;
        T R_002_m4 = pcz * R_001_m5 + R_000_m5;

        T R_003_m0 = pcz * R_002_m1 + T(2) * R_001_m1, R_003_m1 = pcz * R_002_m2 + T(2) * R_001_m2;
        T R_003_m2 = pcz * R_002_m3 + T(2) * R_001_m3, R_003_m3 = pcz * R_002_m4 + T(2) * R_001_m4;

        T R_004_m0 = pcz * R_003_m1 + T(3) * R_002_m1, R_004_m1 = pcz * R_003_m2 + T(3) * R_002_m2;
        T R_004_m2 = pcz * R_003_m3 + T(3) * R_002_m3;

        T R_005_m0 = pcz * R_004_m1 + T(4) * R_003_m1, R_005_m1 = pcz * R_004_m2 + T(4) * R_003_m2;

        T R_006_m0 = pcz * R_005_m1 + T(5) * R_004_m1;

        // Mixed X-Z: R(t,0,v;m)
        T R_101_m0 = pcz * R_100_m1, R_101_m1 = pcz * R_100_m2, R_101_m2 = pcz * R_100_m3;
        T R_101_m3 = pcz * R_100_m4, R_101_m4 = pcz * R_100_m5;

        T R_102_m0 = pcz * R_101_m1 + R_100_m1, R_102_m1 = pcz * R_101_m2 + R_100_m2;
        T R_102_m2 = pcz * R_101_m3 + R_100_m3, R_102_m3 = pcz * R_101_m4 + R_100_m4;

        T R_103_m0 = pcz * R_102_m1 + T(2) * R_101_m1, R_103_m1 = pcz * R_102_m2 + T(2) * R_101_m2;
        T R_103_m2 = pcz * R_102_m3 + T(2) * R_101_m3;

        T R_104_m0 = pcz * R_103_m1 + T(3) * R_102_m1, R_104_m1 = pcz * R_103_m2 + T(3) * R_102_m2;

        T R_105_m0 = pcz * R_104_m1 + T(4) * R_103_m1;

        T R_201_m0 = pcz * R_200_m1, R_201_m1 = pcz * R_200_m2, R_201_m2 = pcz * R_200_m3;
        T R_201_m3 = pcz * R_200_m4;

        T R_202_m0 = pcz * R_201_m1 + R_200_m1, R_202_m1 = pcz * R_201_m2 + R_200_m2;
        T R_202_m2 = pcz * R_201_m3 + R_200_m3;

        T R_203_m0 = pcz * R_202_m1 + T(2) * R_201_m1, R_203_m1 = pcz * R_202_m2 + T(2) * R_201_m2;

        T R_204_m0 = pcz * R_203_m1 + T(3) * R_202_m1;

        T R_301_m0 = pcz * R_300_m1, R_301_m1 = pcz * R_300_m2, R_301_m2 = pcz * R_300_m3;

        T R_302_m0 = pcz * R_301_m1 + R_300_m1, R_302_m1 = pcz * R_301_m2 + R_300_m2;

        T R_303_m0 = pcz * R_302_m1 + T(2) * R_301_m1;

        T R_401_m0 = pcz * R_400_m1, R_401_m1 = pcz * R_400_m2;

        T R_402_m0 = pcz * R_401_m1 + R_400_m1;

        T R_501_m0 = pcz * R_500_m1;

        // Mixed Y-Z: R(0,u,v;m)
        T R_011_m0 = pcz * R_010_m1, R_011_m1 = pcz * R_010_m2, R_011_m2 = pcz * R_010_m3;
        T R_011_m3 = pcz * R_010_m4, R_011_m4 = pcz * R_010_m5;

        T R_012_m0 = pcz * R_011_m1 + R_010_m1, R_012_m1 = pcz * R_011_m2 + R_010_m2;
        T R_012_m2 = pcz * R_011_m3 + R_010_m3, R_012_m3 = pcz * R_011_m4 + R_010_m4;

        T R_013_m0 = pcz * R_012_m1 + T(2) * R_011_m1, R_013_m1 = pcz * R_012_m2 + T(2) * R_011_m2;
        T R_013_m2 = pcz * R_012_m3 + T(2) * R_011_m3;

        T R_014_m0 = pcz * R_013_m1 + T(3) * R_012_m1, R_014_m1 = pcz * R_013_m2 + T(3) * R_012_m2;

        T R_015_m0 = pcz * R_014_m1 + T(4) * R_013_m1;

        T R_021_m0 = pcz * R_020_m1, R_021_m1 = pcz * R_020_m2, R_021_m2 = pcz * R_020_m3;
        T R_021_m3 = pcz * R_020_m4;

        T R_022_m0 = pcz * R_021_m1 + R_020_m1, R_022_m1 = pcz * R_021_m2 + R_020_m2;
        T R_022_m2 = pcz * R_021_m3 + R_020_m3;

        T R_023_m0 = pcz * R_022_m1 + T(2) * R_021_m1, R_023_m1 = pcz * R_022_m2 + T(2) * R_021_m2;

        T R_024_m0 = pcz * R_023_m1 + T(3) * R_022_m1;

        T R_031_m0 = pcz * R_030_m1, R_031_m1 = pcz * R_030_m2, R_031_m2 = pcz * R_030_m3;

        T R_032_m0 = pcz * R_031_m1 + R_030_m1, R_032_m1 = pcz * R_031_m2 + R_030_m2;

        T R_033_m0 = pcz * R_032_m1 + T(2) * R_031_m1;

        T R_041_m0 = pcz * R_040_m1, R_041_m1 = pcz * R_040_m2;

        T R_042_m0 = pcz * R_041_m1 + R_040_m1;

        T R_051_m0 = pcz * R_050_m1;

        // Mixed X-Y-Z: R(t,u,v;m)
        T R_111_m0 = pcz * R_110_m1, R_111_m1 = pcz * R_110_m2, R_111_m2 = pcz * R_110_m3;
        T R_111_m3 = pcz * R_110_m4;

        T R_112_m0 = pcz * R_111_m1 + R_110_m1, R_112_m1 = pcz * R_111_m2 + R_110_m2;
        T R_112_m2 = pcz * R_111_m3 + R_110_m3;

        T R_113_m0 = pcz * R_112_m1 + T(2) * R_111_m1, R_113_m1 = pcz * R_112_m2 + T(2) * R_111_m2;

        T R_114_m0 = pcz * R_113_m1 + T(3) * R_112_m1;

        T R_121_m0 = pcz * R_120_m1, R_121_m1 = pcz * R_120_m2, R_121_m2 = pcz * R_120_m3;

        T R_122_m0 = pcz * R_121_m1 + R_120_m1, R_122_m1 = pcz * R_121_m2 + R_120_m2;

        T R_123_m0 = pcz * R_122_m1 + T(2) * R_121_m1;

        T R_131_m0 = pcz * R_130_m1, R_131_m1 = pcz * R_130_m2;

        T R_132_m0 = pcz * R_131_m1 + R_130_m1;

        T R_141_m0 = pcz * R_140_m1;

        T R_211_m0 = pcz * R_210_m1, R_211_m1 = pcz * R_210_m2, R_211_m2 = pcz * R_210_m3;

        T R_212_m0 = pcz * R_211_m1 + R_210_m1, R_212_m1 = pcz * R_211_m2 + R_210_m2;

        T R_213_m0 = pcz * R_212_m1 + T(2) * R_211_m1;

        T R_221_m0 = pcz * R_220_m1, R_221_m1 = pcz * R_220_m2;

        T R_222_m0 = pcz * R_221_m1 + R_220_m1;

        T R_231_m0 = pcz * R_230_m1;

        T R_311_m0 = pcz * R_310_m1, R_311_m1 = pcz * R_310_m2;

        T R_312_m0 = pcz * R_311_m1 + R_310_m1;

        T R_321_m0 = pcz * R_320_m1;

        T R_411_m0 = pcz * R_410_m1;

        // Write output (84 components)
        T* out = R_out + pt * nherm;
        out[0] = R_000_m0;  out[1] = R_100_m0;  out[2] = R_010_m0;
        out[3] = R_001_m0;  out[4] = R_200_m0;  out[5] = R_110_m0;
        out[6] = R_101_m0;  out[7] = R_020_m0;  out[8] = R_011_m0;
        out[9] = R_002_m0;  out[10] = R_300_m0; out[11] = R_210_m0;
        out[12] = R_201_m0; out[13] = R_120_m0; out[14] = R_111_m0;
        out[15] = R_102_m0; out[16] = R_030_m0; out[17] = R_021_m0;
        out[18] = R_012_m0; out[19] = R_003_m0; out[20] = R_400_m0;
        out[21] = R_310_m0; out[22] = R_301_m0; out[23] = R_220_m0;
        out[24] = R_211_m0; out[25] = R_202_m0; out[26] = R_130_m0;
        out[27] = R_121_m0; out[28] = R_112_m0; out[29] = R_103_m0;
        out[30] = R_040_m0; out[31] = R_031_m0; out[32] = R_022_m0;
        out[33] = R_013_m0; out[34] = R_004_m0; out[35] = R_500_m0;
        out[36] = R_410_m0; out[37] = R_401_m0; out[38] = R_320_m0;
        out[39] = R_311_m0; out[40] = R_302_m0; out[41] = R_230_m0;
        out[42] = R_221_m0; out[43] = R_212_m0; out[44] = R_203_m0;
        out[45] = R_140_m0; out[46] = R_131_m0; out[47] = R_122_m0;
        out[48] = R_113_m0; out[49] = R_104_m0; out[50] = R_050_m0;
        out[51] = R_041_m0; out[52] = R_032_m0; out[53] = R_023_m0;
        out[54] = R_014_m0; out[55] = R_005_m0; out[56] = R_600_m0;
        out[57] = R_510_m0; out[58] = R_501_m0; out[59] = R_420_m0;
        out[60] = R_411_m0; out[61] = R_402_m0; out[62] = R_330_m0;
        out[63] = R_321_m0; out[64] = R_312_m0; out[65] = R_303_m0;
        out[66] = R_240_m0; out[67] = R_231_m0; out[68] = R_222_m0;
        out[69] = R_213_m0; out[70] = R_204_m0; out[71] = R_150_m0;
        out[72] = R_141_m0; out[73] = R_132_m0; out[74] = R_123_m0;
        out[75] = R_114_m0; out[76] = R_105_m0; out[77] = R_060_m0;
        out[78] = R_051_m0; out[79] = R_042_m0; out[80] = R_033_m0;
        out[81] = R_024_m0; out[82] = R_015_m0; out[83] = R_006_m0;
    }
}

// ============================================================================
// Dispatch function
// ============================================================================

template <typename T, int L, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_batch(const T* boys_table, T p, int npts,
                          const T* PCx, const T* PCy, const T* PCz,
                          T* R_out) {
    if constexpr (L == 0) {
        compute_r_ints_L0_batch<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 1) {
        compute_r_ints_L1_batch<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 2) {
        compute_r_ints_L2_batch<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 3) {
        compute_r_ints_L3_batch<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 4) {
        compute_r_ints_L4_batch<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 5) {
        compute_r_ints_L5_batch<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 6) {
        compute_r_ints_L6_batch<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else {
        // For L >= 7, use general implementation point-by-point
        constexpr int nherm = nhermsum(L);
        for (int pt = 0; pt < npts; ++pt) {
            RInts<T, L> R_temp;
            compute_r_ints<T, L, BoysParams>(
                boys_table, p, PCx[pt], PCy[pt], PCz[pt], R_temp);
            for (int i = 0; i < nherm; ++i) {
                R_out[pt * nherm + i] = R_temp.data[i];
            }
        }
    }
}

// ============================================================================
// Eigen-vectorized R-integral kernels
// These perform ALL recurrence operations using Eigen array operations
// ============================================================================

template <typename T, typename BoysParams = BoysParamsDefault>
void compute_r_ints_L0_eigen(const T* boys_table, T p, int npts,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCx,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCy,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCz,
                              Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> R_out) {

    // Process all points
    for (int pt = 0; pt < npts; ++pt) {
        T Fm;
        const T pcx = PCx(pt);
        const T pcy = PCy(pt);
        const T pcz = PCz(pt);
        const T Tp = p * (pcx * pcx + pcy * pcy + pcz * pcz);
        boys_evaluate<T, 1, BoysParams>(boys_table, Tp, 0, &Fm);
        R_out(pt, 0) = Fm;
    }
}

template <typename T, typename BoysParams = BoysParamsDefault>
void compute_r_ints_L1_eigen(const T* boys_table, T p, int npts,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCx,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCy,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCz,
                              Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> R_out) {
    const T neg_2p = T(-2) * p;

    // Lambda for MD recurrence - computes 4 R-integrals for one point
    auto compute_point = [&](int pt, T pcx, T pcy, T pcz, const T* Fm) {
        // L=1: R(0,0,0;m) for m=0,1
        T R_000_m0 = Fm[0];
        T R_000_m1 = neg_2p * Fm[1];

        // Write output (4 components)
        R_out(pt, 0) = R_000_m0;
        R_out(pt, 1) = pcx * R_000_m1;
        R_out(pt, 2) = pcy * R_000_m1;
        R_out(pt, 3) = pcz * R_000_m1;
    };

    // Stack-allocated temporaries
    std::array<T, 2> Fm;

    // Process all points
    for (int pt = 0; pt < npts; ++pt) {
        const T pcx = PCx(pt);
        const T pcy = PCy(pt);
        const T pcz = PCz(pt);
        const T Tp = p * (pcx * pcx + pcy * pcy + pcz * pcz);

        boys_evaluate<T, 2, BoysParams>(boys_table, Tp, 0, Fm.data());
        compute_point(pt, pcx, pcy, pcz, Fm.data());
    }
}

template <typename T, typename BoysParams = BoysParamsDefault>
void compute_r_ints_L2_eigen(const T* boys_table, T p, int npts,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCx,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCy,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCz,
                              Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> R_out) {
    const T neg_2p = T(-2) * p;
    const T neg_2p_2 = neg_2p * neg_2p;

    // Lambda for MD recurrence - computes 10 R-integrals for one point
    auto compute_point = [&](int pt, T pcx, T pcy, T pcz, const T* Fm) {
        // L=2: R(0,0,0;m) for m=0,1,2
        T R_000_m0 = Fm[0];
        T R_000_m1 = neg_2p * Fm[1];
        T R_000_m2 = neg_2p_2 * Fm[2];

        // X, Y, Z direction intermediates
        T R_100_m1 = pcx * R_000_m2;
        T R_010_m1 = pcy * R_000_m2;
        T R_001_m1 = pcz * R_000_m2;

        // Write output (10 components)
        R_out(pt, 0) = R_000_m0;
        R_out(pt, 1) = pcx * R_000_m1;
        R_out(pt, 2) = pcy * R_000_m1;
        R_out(pt, 3) = pcz * R_000_m1;
        R_out(pt, 4) = pcx * R_100_m1 + R_000_m1;
        R_out(pt, 5) = pcy * R_100_m1;
        R_out(pt, 6) = pcz * R_100_m1;
        R_out(pt, 7) = pcy * R_010_m1 + R_000_m1;
        R_out(pt, 8) = pcz * R_010_m1;
        R_out(pt, 9) = pcz * R_001_m1 + R_000_m1;
    };

    // Stack-allocated temporaries
    std::array<T, 3> Fm;

    // Process all points
    for (int pt = 0; pt < npts; ++pt) {
        const T pcx = PCx(pt);
        const T pcy = PCy(pt);
        const T pcz = PCz(pt);
        const T Tp = p * (pcx * pcx + pcy * pcy + pcz * pcz);

        boys_evaluate<T, 3, BoysParams>(boys_table, Tp, 0, Fm.data());
        compute_point(pt, pcx, pcy, pcz, Fm.data());
    }
}

template <typename T, typename BoysParams = BoysParamsDefault>
void compute_r_ints_L3_eigen(const T* boys_table, T p, int npts,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCx,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCy,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCz,
                              Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> R_out) {
    const T neg_2p = T(-2) * p;
    const T neg_2p_2 = neg_2p * neg_2p;
    const T neg_2p_3 = neg_2p_2 * neg_2p;

    // Lambda for MD recurrence - computes 20 R-integrals for one point
    auto compute_point = [&](int pt, T pcx, T pcy, T pcz, const T* Fm) {
        // L=3: R(0,0,0;m) for m=0,1,2,3
        T R_000_m0 = Fm[0], R_000_m1 = neg_2p * Fm[1];
        T R_000_m2 = neg_2p_2 * Fm[2], R_000_m3 = neg_2p_3 * Fm[3];

        // X-direction
        T R_100_m0 = pcx * R_000_m1, R_100_m1 = pcx * R_000_m2, R_100_m2 = pcx * R_000_m3;
        T R_200_m0 = pcx * R_100_m1 + R_000_m1, R_200_m1 = pcx * R_100_m2 + R_000_m2;
        T R_300_m0 = pcx * R_200_m1 + T(2) * R_100_m1;

        // Y-direction
        T R_010_m0 = pcy * R_000_m1, R_010_m1 = pcy * R_000_m2, R_010_m2 = pcy * R_000_m3;
        T R_110_m0 = pcy * R_100_m1, R_110_m1 = pcy * R_100_m2;
        T R_210_m0 = pcy * R_200_m1;
        T R_020_m0 = pcy * R_010_m1 + R_000_m1, R_020_m1 = pcy * R_010_m2 + R_000_m2;
        T R_120_m0 = pcy * R_110_m1 + R_100_m1;
        T R_030_m0 = pcy * R_020_m1 + T(2) * R_010_m1;

        // Z-direction
        T R_001_m0 = pcz * R_000_m1, R_001_m1 = pcz * R_000_m2, R_001_m2 = pcz * R_000_m3;
        T R_101_m0 = pcz * R_100_m1, R_101_m1 = pcz * R_100_m2;
        T R_201_m0 = pcz * R_200_m1;
        T R_011_m0 = pcz * R_010_m1, R_011_m1 = pcz * R_010_m2;
        T R_111_m0 = pcz * R_110_m1;
        T R_021_m0 = pcz * R_020_m1;
        T R_002_m0 = pcz * R_001_m1 + R_000_m1, R_002_m1 = pcz * R_001_m2 + R_000_m2;
        T R_102_m0 = pcz * R_101_m1 + R_100_m1;
        T R_012_m0 = pcz * R_011_m1 + R_010_m1;
        T R_003_m0 = pcz * R_002_m1 + T(2) * R_001_m1;

        // Write output (20 components)
        R_out(pt, 0) = R_000_m0;   R_out(pt, 1) = R_100_m0;   R_out(pt, 2) = R_010_m0;
        R_out(pt, 3) = R_001_m0;   R_out(pt, 4) = R_200_m0;   R_out(pt, 5) = R_110_m0;
        R_out(pt, 6) = R_101_m0;   R_out(pt, 7) = R_020_m0;   R_out(pt, 8) = R_011_m0;
        R_out(pt, 9) = R_002_m0;   R_out(pt, 10) = R_300_m0;  R_out(pt, 11) = R_210_m0;
        R_out(pt, 12) = R_201_m0;  R_out(pt, 13) = R_120_m0;  R_out(pt, 14) = R_111_m0;
        R_out(pt, 15) = R_102_m0;  R_out(pt, 16) = R_030_m0;  R_out(pt, 17) = R_021_m0;
        R_out(pt, 18) = R_012_m0;  R_out(pt, 19) = R_003_m0;
    };

    // Stack-allocated temporaries
    std::array<T, 4> Fm;

    // Process all points
    for (int pt = 0; pt < npts; ++pt) {
        const T pcx = PCx(pt);
        const T pcy = PCy(pt);
        const T pcz = PCz(pt);
        const T Tp = p * (pcx * pcx + pcy * pcy + pcz * pcz);

        boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm.data());
        compute_point(pt, pcx, pcy, pcz, Fm.data());
    }
}

template <typename T, typename BoysParams = BoysParamsDefault>
void compute_r_ints_L4_eigen(const T* boys_table, T p, int npts,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCx,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCy,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCz,
                              Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> R_out) {
    const T neg_2p = T(-2) * p;
    const T neg_2p_2 = neg_2p * neg_2p;
    const T neg_2p_3 = neg_2p_2 * neg_2p;
    const T neg_2p_4 = neg_2p_3 * neg_2p;

    // Lambda for MD recurrence - computes all 35 R-integrals for one point
    auto compute_point = [&](int pt, T pcx, T pcy, T pcz, const T* Fm) {
        T R_000_m0 = Fm[0], R_000_m1 = neg_2p * Fm[1], R_000_m2 = neg_2p_2 * Fm[2];
        T R_000_m3 = neg_2p_3 * Fm[3], R_000_m4 = neg_2p_4 * Fm[4];

        // X-direction
        T R_100_m0 = pcx * R_000_m1, R_100_m1 = pcx * R_000_m2;
        T R_100_m2 = pcx * R_000_m3, R_100_m3 = pcx * R_000_m4;
        T R_200_m0 = pcx * R_100_m1 + R_000_m1, R_200_m1 = pcx * R_100_m2 + R_000_m2;
        T R_200_m2 = pcx * R_100_m3 + R_000_m3;
        T R_300_m0 = pcx * R_200_m1 + T(2) * R_100_m1, R_300_m1 = pcx * R_200_m2 + T(2) * R_100_m2;
        T R_400_m0 = pcx * R_300_m1 + T(3) * R_200_m1;

        // Y-direction
        T R_010_m0 = pcy * R_000_m1, R_010_m1 = pcy * R_000_m2;
        T R_010_m2 = pcy * R_000_m3, R_010_m3 = pcy * R_000_m4;
        T R_020_m0 = pcy * R_010_m1 + R_000_m1, R_020_m1 = pcy * R_010_m2 + R_000_m2;
        T R_020_m2 = pcy * R_010_m3 + R_000_m3;
        T R_030_m0 = pcy * R_020_m1 + T(2) * R_010_m1, R_030_m1 = pcy * R_020_m2 + T(2) * R_010_m2;
        T R_040_m0 = pcy * R_030_m1 + T(3) * R_020_m1;

        T R_110_m0 = pcy * R_100_m1, R_110_m1 = pcy * R_100_m2, R_110_m2 = pcy * R_100_m3;
        T R_120_m0 = pcy * R_110_m1 + R_100_m1, R_120_m1 = pcy * R_110_m2 + R_100_m2;
        T R_130_m0 = pcy * R_120_m1 + T(2) * R_110_m1;
        T R_210_m0 = pcy * R_200_m1, R_210_m1 = pcy * R_200_m2;
        T R_220_m0 = pcy * R_210_m1 + R_200_m1;
        T R_310_m0 = pcy * R_300_m1;

        // Z-direction
        T R_001_m0 = pcz * R_000_m1, R_001_m1 = pcz * R_000_m2;
        T R_001_m2 = pcz * R_000_m3, R_001_m3 = pcz * R_000_m4;
        T R_002_m0 = pcz * R_001_m1 + R_000_m1, R_002_m1 = pcz * R_001_m2 + R_000_m2;
        T R_002_m2 = pcz * R_001_m3 + R_000_m3;
        T R_003_m0 = pcz * R_002_m1 + T(2) * R_001_m1, R_003_m1 = pcz * R_002_m2 + T(2) * R_001_m2;
        T R_004_m0 = pcz * R_003_m1 + T(3) * R_002_m1;

        T R_101_m0 = pcz * R_100_m1, R_101_m1 = pcz * R_100_m2, R_101_m2 = pcz * R_100_m3;
        T R_102_m0 = pcz * R_101_m1 + R_100_m1, R_102_m1 = pcz * R_101_m2 + R_100_m2;
        T R_103_m0 = pcz * R_102_m1 + T(2) * R_101_m1;
        T R_201_m0 = pcz * R_200_m1, R_201_m1 = pcz * R_200_m2;
        T R_202_m0 = pcz * R_201_m1 + R_200_m1;
        T R_301_m0 = pcz * R_300_m1;

        T R_011_m0 = pcz * R_010_m1, R_011_m1 = pcz * R_010_m2, R_011_m2 = pcz * R_010_m3;
        T R_012_m0 = pcz * R_011_m1 + R_010_m1, R_012_m1 = pcz * R_011_m2 + R_010_m2;
        T R_013_m0 = pcz * R_012_m1 + T(2) * R_011_m1;
        T R_111_m0 = pcz * R_110_m1, R_111_m1 = pcz * R_110_m2;
        T R_112_m0 = pcz * R_111_m1 + R_110_m1;
        T R_211_m0 = pcz * R_210_m1;
        T R_021_m0 = pcz * R_020_m1, R_021_m1 = pcz * R_020_m2;
        T R_022_m0 = pcz * R_021_m1 + R_020_m1;
        T R_121_m0 = pcz * R_120_m1;
        T R_031_m0 = pcz * R_030_m1;

        // Write output (35 components)
        R_out(pt, 0) = R_000_m0;  R_out(pt, 1) = R_100_m0;  R_out(pt, 2) = R_010_m0;
        R_out(pt, 3) = R_001_m0;  R_out(pt, 4) = R_200_m0;  R_out(pt, 5) = R_110_m0;
        R_out(pt, 6) = R_101_m0;  R_out(pt, 7) = R_020_m0;  R_out(pt, 8) = R_011_m0;
        R_out(pt, 9) = R_002_m0;  R_out(pt, 10) = R_300_m0; R_out(pt, 11) = R_210_m0;
        R_out(pt, 12) = R_201_m0; R_out(pt, 13) = R_120_m0; R_out(pt, 14) = R_111_m0;
        R_out(pt, 15) = R_102_m0; R_out(pt, 16) = R_030_m0; R_out(pt, 17) = R_021_m0;
        R_out(pt, 18) = R_012_m0; R_out(pt, 19) = R_003_m0; R_out(pt, 20) = R_400_m0;
        R_out(pt, 21) = R_310_m0; R_out(pt, 22) = R_301_m0; R_out(pt, 23) = R_220_m0;
        R_out(pt, 24) = R_211_m0; R_out(pt, 25) = R_202_m0; R_out(pt, 26) = R_130_m0;
        R_out(pt, 27) = R_121_m0; R_out(pt, 28) = R_112_m0; R_out(pt, 29) = R_103_m0;
        R_out(pt, 30) = R_040_m0; R_out(pt, 31) = R_031_m0; R_out(pt, 32) = R_022_m0;
        R_out(pt, 33) = R_013_m0; R_out(pt, 34) = R_004_m0;
    };

    // Stack-allocated temporaries
    std::array<T, 5> Fm;

    // Process all points
    for (int pt = 0; pt < npts; ++pt) {
        const T pcx = PCx(pt);
        const T pcy = PCy(pt);
        const T pcz = PCz(pt);
        const T Tp = p * (pcx * pcx + pcy * pcy + pcz * pcz);

        boys_evaluate<T, 5, BoysParams>(boys_table, Tp, 0, Fm.data());
        compute_point(pt, pcx, pcy, pcz, Fm.data());
    }
}

template <typename T, typename BoysParams = BoysParamsDefault>
void compute_r_ints_L6_eigen(const T* boys_table, T p, int npts,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCx,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCy,
                              const Eigen::Array<T, Eigen::Dynamic, 1>& PCz,
                              Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> R_out) {
    const T neg_2p = T(-2) * p;
    const T neg_2p_2 = neg_2p * neg_2p;
    const T neg_2p_3 = neg_2p_2 * neg_2p;
    const T neg_2p_4 = neg_2p_3 * neg_2p;
    const T neg_2p_5 = neg_2p_4 * neg_2p;
    const T neg_2p_6 = neg_2p_5 * neg_2p;

    // Lambda for MD recurrence - computes all 84 R-integrals for one point
    auto compute_point = [&](int pt, T pcx, T pcy, T pcz, const T* Fm) {
        // Base values R(0,0,0;m) for m=0..6
        T R_000_m0 = Fm[0], R_000_m1 = neg_2p * Fm[1], R_000_m2 = neg_2p_2 * Fm[2];
        T R_000_m3 = neg_2p_3 * Fm[3], R_000_m4 = neg_2p_4 * Fm[4];
        T R_000_m5 = neg_2p_5 * Fm[5], R_000_m6 = neg_2p_6 * Fm[6];

        // X-direction: R(t,0,0;m) for t=1..6
        T R_100_m0 = pcx * R_000_m1, R_100_m1 = pcx * R_000_m2, R_100_m2 = pcx * R_000_m3;
        T R_100_m3 = pcx * R_000_m4, R_100_m4 = pcx * R_000_m5, R_100_m5 = pcx * R_000_m6;

        T R_200_m0 = pcx * R_100_m1 + R_000_m1, R_200_m1 = pcx * R_100_m2 + R_000_m2;
        T R_200_m2 = pcx * R_100_m3 + R_000_m3, R_200_m3 = pcx * R_100_m4 + R_000_m4;
        T R_200_m4 = pcx * R_100_m5 + R_000_m5;

        T R_300_m0 = pcx * R_200_m1 + T(2) * R_100_m1, R_300_m1 = pcx * R_200_m2 + T(2) * R_100_m2;
        T R_300_m2 = pcx * R_200_m3 + T(2) * R_100_m3, R_300_m3 = pcx * R_200_m4 + T(2) * R_100_m4;

        T R_400_m0 = pcx * R_300_m1 + T(3) * R_200_m1, R_400_m1 = pcx * R_300_m2 + T(3) * R_200_m2;
        T R_400_m2 = pcx * R_300_m3 + T(3) * R_200_m3;

        T R_500_m0 = pcx * R_400_m1 + T(4) * R_300_m1, R_500_m1 = pcx * R_400_m2 + T(4) * R_300_m2;

        T R_600_m0 = pcx * R_500_m1 + T(5) * R_400_m1;

        // Y-direction: R(0,u,0;m) for u=1..6
        T R_010_m0 = pcy * R_000_m1, R_010_m1 = pcy * R_000_m2, R_010_m2 = pcy * R_000_m3;
        T R_010_m3 = pcy * R_000_m4, R_010_m4 = pcy * R_000_m5, R_010_m5 = pcy * R_000_m6;

        T R_020_m0 = pcy * R_010_m1 + R_000_m1, R_020_m1 = pcy * R_010_m2 + R_000_m2;
        T R_020_m2 = pcy * R_010_m3 + R_000_m3, R_020_m3 = pcy * R_010_m4 + R_000_m4;
        T R_020_m4 = pcy * R_010_m5 + R_000_m5;

        T R_030_m0 = pcy * R_020_m1 + T(2) * R_010_m1, R_030_m1 = pcy * R_020_m2 + T(2) * R_010_m2;
        T R_030_m2 = pcy * R_020_m3 + T(2) * R_010_m3, R_030_m3 = pcy * R_020_m4 + T(2) * R_010_m4;

        T R_040_m0 = pcy * R_030_m1 + T(3) * R_020_m1, R_040_m1 = pcy * R_030_m2 + T(3) * R_020_m2;
        T R_040_m2 = pcy * R_030_m3 + T(3) * R_020_m3;

        T R_050_m0 = pcy * R_040_m1 + T(4) * R_030_m1, R_050_m1 = pcy * R_040_m2 + T(4) * R_030_m2;

        T R_060_m0 = pcy * R_050_m1 + T(5) * R_040_m1;

        // Mixed X-Y: R(t,u,0;m)
        T R_110_m0 = pcy * R_100_m1, R_110_m1 = pcy * R_100_m2, R_110_m2 = pcy * R_100_m3;
        T R_110_m3 = pcy * R_100_m4, R_110_m4 = pcy * R_100_m5;

        T R_120_m0 = pcy * R_110_m1 + R_100_m1, R_120_m1 = pcy * R_110_m2 + R_100_m2;
        T R_120_m2 = pcy * R_110_m3 + R_100_m3, R_120_m3 = pcy * R_110_m4 + R_100_m4;

        T R_130_m0 = pcy * R_120_m1 + T(2) * R_110_m1, R_130_m1 = pcy * R_120_m2 + T(2) * R_110_m2;
        T R_130_m2 = pcy * R_120_m3 + T(2) * R_110_m3;

        T R_140_m0 = pcy * R_130_m1 + T(3) * R_120_m1, R_140_m1 = pcy * R_130_m2 + T(3) * R_120_m2;

        T R_150_m0 = pcy * R_140_m1 + T(4) * R_130_m1;

        T R_210_m0 = pcy * R_200_m1, R_210_m1 = pcy * R_200_m2, R_210_m2 = pcy * R_200_m3;
        T R_210_m3 = pcy * R_200_m4;

        T R_220_m0 = pcy * R_210_m1 + R_200_m1, R_220_m1 = pcy * R_210_m2 + R_200_m2;
        T R_220_m2 = pcy * R_210_m3 + R_200_m3;

        T R_230_m0 = pcy * R_220_m1 + T(2) * R_210_m1, R_230_m1 = pcy * R_220_m2 + T(2) * R_210_m2;

        T R_240_m0 = pcy * R_230_m1 + T(3) * R_220_m1;

        T R_310_m0 = pcy * R_300_m1, R_310_m1 = pcy * R_300_m2, R_310_m2 = pcy * R_300_m3;

        T R_320_m0 = pcy * R_310_m1 + R_300_m1, R_320_m1 = pcy * R_310_m2 + R_300_m2;

        T R_330_m0 = pcy * R_320_m1 + T(2) * R_310_m1;

        T R_410_m0 = pcy * R_400_m1, R_410_m1 = pcy * R_400_m2;

        T R_420_m0 = pcy * R_410_m1 + R_400_m1;

        T R_510_m0 = pcy * R_500_m1;

        // Z-direction: R(0,0,v;m) and mixed terms
        T R_001_m0 = pcz * R_000_m1, R_001_m1 = pcz * R_000_m2, R_001_m2 = pcz * R_000_m3;
        T R_001_m3 = pcz * R_000_m4, R_001_m4 = pcz * R_000_m5, R_001_m5 = pcz * R_000_m6;

        T R_002_m0 = pcz * R_001_m1 + R_000_m1, R_002_m1 = pcz * R_001_m2 + R_000_m2;
        T R_002_m2 = pcz * R_001_m3 + R_000_m3, R_002_m3 = pcz * R_001_m4 + R_000_m4;
        T R_002_m4 = pcz * R_001_m5 + R_000_m5;

        T R_003_m0 = pcz * R_002_m1 + T(2) * R_001_m1, R_003_m1 = pcz * R_002_m2 + T(2) * R_001_m2;
        T R_003_m2 = pcz * R_002_m3 + T(2) * R_001_m3, R_003_m3 = pcz * R_002_m4 + T(2) * R_001_m4;

        T R_004_m0 = pcz * R_003_m1 + T(3) * R_002_m1, R_004_m1 = pcz * R_003_m2 + T(3) * R_002_m2;
        T R_004_m2 = pcz * R_003_m3 + T(3) * R_002_m3;

        T R_005_m0 = pcz * R_004_m1 + T(4) * R_003_m1, R_005_m1 = pcz * R_004_m2 + T(4) * R_003_m2;

        T R_006_m0 = pcz * R_005_m1 + T(5) * R_004_m1;

        // Mixed X-Z: R(t,0,v;m)
        T R_101_m0 = pcz * R_100_m1, R_101_m1 = pcz * R_100_m2, R_101_m2 = pcz * R_100_m3;
        T R_101_m3 = pcz * R_100_m4, R_101_m4 = pcz * R_100_m5;

        T R_102_m0 = pcz * R_101_m1 + R_100_m1, R_102_m1 = pcz * R_101_m2 + R_100_m2;
        T R_102_m2 = pcz * R_101_m3 + R_100_m3, R_102_m3 = pcz * R_101_m4 + R_100_m4;

        T R_103_m0 = pcz * R_102_m1 + T(2) * R_101_m1, R_103_m1 = pcz * R_102_m2 + T(2) * R_101_m2;
        T R_103_m2 = pcz * R_102_m3 + T(2) * R_101_m3;

        T R_104_m0 = pcz * R_103_m1 + T(3) * R_102_m1, R_104_m1 = pcz * R_103_m2 + T(3) * R_102_m2;

        T R_105_m0 = pcz * R_104_m1 + T(4) * R_103_m1;

        T R_201_m0 = pcz * R_200_m1, R_201_m1 = pcz * R_200_m2, R_201_m2 = pcz * R_200_m3;
        T R_201_m3 = pcz * R_200_m4;

        T R_202_m0 = pcz * R_201_m1 + R_200_m1, R_202_m1 = pcz * R_201_m2 + R_200_m2;
        T R_202_m2 = pcz * R_201_m3 + R_200_m3;

        T R_203_m0 = pcz * R_202_m1 + T(2) * R_201_m1, R_203_m1 = pcz * R_202_m2 + T(2) * R_201_m2;

        T R_204_m0 = pcz * R_203_m1 + T(3) * R_202_m1;

        T R_301_m0 = pcz * R_300_m1, R_301_m1 = pcz * R_300_m2, R_301_m2 = pcz * R_300_m3;

        T R_302_m0 = pcz * R_301_m1 + R_300_m1, R_302_m1 = pcz * R_301_m2 + R_300_m2;

        T R_303_m0 = pcz * R_302_m1 + T(2) * R_301_m1;

        T R_401_m0 = pcz * R_400_m1, R_401_m1 = pcz * R_400_m2;

        T R_402_m0 = pcz * R_401_m1 + R_400_m1;

        T R_501_m0 = pcz * R_500_m1;

        // Mixed Y-Z: R(0,u,v;m)
        T R_011_m0 = pcz * R_010_m1, R_011_m1 = pcz * R_010_m2, R_011_m2 = pcz * R_010_m3;
        T R_011_m3 = pcz * R_010_m4, R_011_m4 = pcz * R_010_m5;

        T R_012_m0 = pcz * R_011_m1 + R_010_m1, R_012_m1 = pcz * R_011_m2 + R_010_m2;
        T R_012_m2 = pcz * R_011_m3 + R_010_m3, R_012_m3 = pcz * R_011_m4 + R_010_m4;

        T R_013_m0 = pcz * R_012_m1 + T(2) * R_011_m1, R_013_m1 = pcz * R_012_m2 + T(2) * R_011_m2;
        T R_013_m2 = pcz * R_012_m3 + T(2) * R_011_m3;

        T R_014_m0 = pcz * R_013_m1 + T(3) * R_012_m1, R_014_m1 = pcz * R_013_m2 + T(3) * R_012_m2;

        T R_015_m0 = pcz * R_014_m1 + T(4) * R_013_m1;

        T R_021_m0 = pcz * R_020_m1, R_021_m1 = pcz * R_020_m2, R_021_m2 = pcz * R_020_m3;
        T R_021_m3 = pcz * R_020_m4;

        T R_022_m0 = pcz * R_021_m1 + R_020_m1, R_022_m1 = pcz * R_021_m2 + R_020_m2;
        T R_022_m2 = pcz * R_021_m3 + R_020_m3;

        T R_023_m0 = pcz * R_022_m1 + T(2) * R_021_m1, R_023_m1 = pcz * R_022_m2 + T(2) * R_021_m2;

        T R_024_m0 = pcz * R_023_m1 + T(3) * R_022_m1;

        T R_031_m0 = pcz * R_030_m1, R_031_m1 = pcz * R_030_m2, R_031_m2 = pcz * R_030_m3;

        T R_032_m0 = pcz * R_031_m1 + R_030_m1, R_032_m1 = pcz * R_031_m2 + R_030_m2;

        T R_033_m0 = pcz * R_032_m1 + T(2) * R_031_m1;

        T R_041_m0 = pcz * R_040_m1, R_041_m1 = pcz * R_040_m2;

        T R_042_m0 = pcz * R_041_m1 + R_040_m1;

        T R_051_m0 = pcz * R_050_m1;

        // Mixed X-Y-Z: R(t,u,v;m)
        T R_111_m0 = pcz * R_110_m1, R_111_m1 = pcz * R_110_m2, R_111_m2 = pcz * R_110_m3;
        T R_111_m3 = pcz * R_110_m4;

        T R_112_m0 = pcz * R_111_m1 + R_110_m1, R_112_m1 = pcz * R_111_m2 + R_110_m2;
        T R_112_m2 = pcz * R_111_m3 + R_110_m3;

        T R_113_m0 = pcz * R_112_m1 + T(2) * R_111_m1, R_113_m1 = pcz * R_112_m2 + T(2) * R_111_m2;

        T R_114_m0 = pcz * R_113_m1 + T(3) * R_112_m1;

        T R_121_m0 = pcz * R_120_m1, R_121_m1 = pcz * R_120_m2, R_121_m2 = pcz * R_120_m3;

        T R_122_m0 = pcz * R_121_m1 + R_120_m1, R_122_m1 = pcz * R_121_m2 + R_120_m2;

        T R_123_m0 = pcz * R_122_m1 + T(2) * R_121_m1;

        T R_131_m0 = pcz * R_130_m1, R_131_m1 = pcz * R_130_m2;

        T R_132_m0 = pcz * R_131_m1 + R_130_m1;

        T R_141_m0 = pcz * R_140_m1;

        T R_211_m0 = pcz * R_210_m1, R_211_m1 = pcz * R_210_m2, R_211_m2 = pcz * R_210_m3;

        T R_212_m0 = pcz * R_211_m1 + R_210_m1, R_212_m1 = pcz * R_211_m2 + R_210_m2;

        T R_213_m0 = pcz * R_212_m1 + T(2) * R_211_m1;

        T R_221_m0 = pcz * R_220_m1, R_221_m1 = pcz * R_220_m2;

        T R_222_m0 = pcz * R_221_m1 + R_220_m1;

        T R_231_m0 = pcz * R_230_m1;

        T R_311_m0 = pcz * R_310_m1, R_311_m1 = pcz * R_310_m2;

        T R_312_m0 = pcz * R_311_m1 + R_310_m1;

        T R_321_m0 = pcz * R_320_m1;

        T R_411_m0 = pcz * R_410_m1;

        // Write output (84 components in canonical Hermite order)
        R_out(pt, 0) = R_000_m0;  R_out(pt, 1) = R_100_m0;  R_out(pt, 2) = R_010_m0;
        R_out(pt, 3) = R_001_m0;  R_out(pt, 4) = R_200_m0;  R_out(pt, 5) = R_110_m0;
        R_out(pt, 6) = R_101_m0;  R_out(pt, 7) = R_020_m0;  R_out(pt, 8) = R_011_m0;
        R_out(pt, 9) = R_002_m0;  R_out(pt, 10) = R_300_m0; R_out(pt, 11) = R_210_m0;
        R_out(pt, 12) = R_201_m0; R_out(pt, 13) = R_120_m0; R_out(pt, 14) = R_111_m0;
        R_out(pt, 15) = R_102_m0; R_out(pt, 16) = R_030_m0; R_out(pt, 17) = R_021_m0;
        R_out(pt, 18) = R_012_m0; R_out(pt, 19) = R_003_m0; R_out(pt, 20) = R_400_m0;
        R_out(pt, 21) = R_310_m0; R_out(pt, 22) = R_301_m0; R_out(pt, 23) = R_220_m0;
        R_out(pt, 24) = R_211_m0; R_out(pt, 25) = R_202_m0; R_out(pt, 26) = R_130_m0;
        R_out(pt, 27) = R_121_m0; R_out(pt, 28) = R_112_m0; R_out(pt, 29) = R_103_m0;
        R_out(pt, 30) = R_040_m0; R_out(pt, 31) = R_031_m0; R_out(pt, 32) = R_022_m0;
        R_out(pt, 33) = R_013_m0; R_out(pt, 34) = R_004_m0; R_out(pt, 35) = R_500_m0;
        R_out(pt, 36) = R_410_m0; R_out(pt, 37) = R_401_m0; R_out(pt, 38) = R_320_m0;
        R_out(pt, 39) = R_311_m0; R_out(pt, 40) = R_302_m0; R_out(pt, 41) = R_230_m0;
        R_out(pt, 42) = R_221_m0; R_out(pt, 43) = R_212_m0; R_out(pt, 44) = R_203_m0;
        R_out(pt, 45) = R_140_m0; R_out(pt, 46) = R_131_m0; R_out(pt, 47) = R_122_m0;
        R_out(pt, 48) = R_113_m0; R_out(pt, 49) = R_104_m0; R_out(pt, 50) = R_050_m0;
        R_out(pt, 51) = R_041_m0; R_out(pt, 52) = R_032_m0; R_out(pt, 53) = R_023_m0;
        R_out(pt, 54) = R_014_m0; R_out(pt, 55) = R_005_m0; R_out(pt, 56) = R_600_m0;
        R_out(pt, 57) = R_510_m0; R_out(pt, 58) = R_501_m0; R_out(pt, 59) = R_420_m0;
        R_out(pt, 60) = R_411_m0; R_out(pt, 61) = R_402_m0; R_out(pt, 62) = R_330_m0;
        R_out(pt, 63) = R_321_m0; R_out(pt, 64) = R_312_m0; R_out(pt, 65) = R_303_m0;
        R_out(pt, 66) = R_240_m0; R_out(pt, 67) = R_231_m0; R_out(pt, 68) = R_222_m0;
        R_out(pt, 69) = R_213_m0; R_out(pt, 70) = R_204_m0; R_out(pt, 71) = R_150_m0;
        R_out(pt, 72) = R_141_m0; R_out(pt, 73) = R_132_m0; R_out(pt, 74) = R_123_m0;
        R_out(pt, 75) = R_114_m0; R_out(pt, 76) = R_105_m0; R_out(pt, 77) = R_060_m0;
        R_out(pt, 78) = R_051_m0; R_out(pt, 79) = R_042_m0; R_out(pt, 80) = R_033_m0;
        R_out(pt, 81) = R_024_m0; R_out(pt, 82) = R_015_m0; R_out(pt, 83) = R_006_m0;
    };

    // Stack-allocated temporaries
    std::array<T, 7> Fm;

    // Process all points
    for (int pt = 0; pt < npts; ++pt) {
        const T pcx = PCx(pt);
        const T pcy = PCy(pt);
        const T pcz = PCz(pt);
        const T Tp = p * (pcx * pcx + pcy * pcy + pcz * pcz);

        boys_evaluate<T, 7, BoysParams>(boys_table, Tp, 0, Fm.data());
        compute_point(pt, pcx, pcy, pcz, Fm.data());
    }
}

// ============================================================================
// Direct batch interface - takes primitive center P and grid points C directly
// Uses Eigen-vectorized kernels for L=0-6, falls back for L>=7
// ============================================================================

template <typename T, int L, typename BoysParams = BoysParamsDefault>
void compute_r_ints_batch_direct(const T* boys_table, T p, int npts,
                                  T Px, T Py, T Pz,
                                  const Eigen::Ref<const Eigen::Matrix<T, 3, Eigen::Dynamic>>& C,
                                  Eigen::Ref<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> R_out) {
    // Compute PC = P - C using Eigen vectorization
    using Vec = Eigen::Array<T, Eigen::Dynamic, 1>;
    Vec PCx = Px - C.row(0).array().transpose();
    Vec PCy = Py - C.row(1).array().transpose();
    Vec PCz = Pz - C.row(2).array().transpose();

    // Use Eigen-vectorized kernels for L=0-4
    if constexpr (L == 0) {
        compute_r_ints_L0_eigen<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 1) {
        compute_r_ints_L1_eigen<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 2) {
        compute_r_ints_L2_eigen<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 3) {
        compute_r_ints_L3_eigen<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 4) {
        compute_r_ints_L4_eigen<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else if constexpr (L == 6) {
        compute_r_ints_L6_eigen<T, BoysParams>(boys_table, p, npts, PCx, PCy, PCz, R_out);
    } else {
        // For L=5, L>=7, use the point-by-point batched kernel
        compute_r_ints_batch<T, L, BoysParams>(
            boys_table, p, npts,
            PCx.data(), PCy.data(), PCz.data(),
            R_out.data());
    }
}

} // namespace occ::ints
