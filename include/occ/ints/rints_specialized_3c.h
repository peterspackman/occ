#pragma once
#include <occ/ints/boys.h>
#include <occ/ints/rints.h>

namespace occ::ints {

// ============================================================================
// Single-point specialized R-integral kernels for 3-center integrals
// These are optimized for Split-RI-J where we compute one R at a time
// ============================================================================

// ----------------------------------------------------------------------------
// L = 0: 1 component - R(0,0,0)
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_3c_L0(const T* boys_table, T p, T PCx, T PCy, T PCz, T* R) {
    const T Tp = p * (PCx * PCx + PCy * PCy + PCz * PCz);
    T Fm[1];
    boys_evaluate<T, 1, BoysParams>(boys_table, Tp, 0, Fm);
    R[0] = Fm[0];
}

// ----------------------------------------------------------------------------
// L = 1: 4 components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_3c_L1(const T* boys_table, T p, T PCx, T PCy, T PCz, T* R) {
    const T Tp = p * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2p = T(-2) * p;

    T Fm[2];
    boys_evaluate<T, 2, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0];
    const T R000_m1 = neg_2p * Fm[1];

    R[0] = R000_m0;
    R[1] = PCx * R000_m1;
    R[2] = PCy * R000_m1;
    R[3] = PCz * R000_m1;
}

// ----------------------------------------------------------------------------
// L = 2: 10 components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_3c_L2(const T* boys_table, T p, T PCx, T PCy, T PCz, T* R) {
    const T Tp = p * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2p = T(-2) * p;

    T Fm[3];
    boys_evaluate<T, 3, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0];
    const T R000_m1 = neg_2p * Fm[1];
    const T R000_m2 = neg_2p * neg_2p * Fm[2];

    const T R100_m0 = PCx * R000_m1;
    const T R100_m1 = PCx * R000_m2;
    const T R010_m0 = PCy * R000_m1;
    const T R010_m1 = PCy * R000_m2;
    const T R001_m0 = PCz * R000_m1;
    const T R001_m1 = PCz * R000_m2;

    R[0] = R000_m0;
    R[1] = R100_m0;
    R[2] = R010_m0;
    R[3] = R001_m0;
    R[4] = PCx * R100_m1 + R000_m1;  // R(2,0,0)
    R[5] = PCy * R100_m1;             // R(1,1,0)
    R[6] = PCz * R100_m1;             // R(1,0,1)
    R[7] = PCy * R010_m1 + R000_m1;  // R(0,2,0)
    R[8] = PCz * R010_m1;             // R(0,1,1)
    R[9] = PCz * R001_m1 + R000_m1;  // R(0,0,2)
}

// ----------------------------------------------------------------------------
// L = 3: 20 components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_3c_L3(const T* boys_table, T p, T PCx, T PCy, T PCz, T* R) {
    const T Tp = p * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2p = T(-2) * p;

    T Fm[4];
    boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0];
    const T R000_m1 = neg_2p * Fm[1];
    const T R000_m2 = neg_2p * neg_2p * Fm[2];
    const T R000_m3 = neg_2p * neg_2p * neg_2p * Fm[3];

    // X-direction
    const T R100_m0 = PCx * R000_m1;
    const T R100_m1 = PCx * R000_m2;
    const T R100_m2 = PCx * R000_m3;
    const T R200_m0 = PCx * R100_m1 + R000_m1;
    const T R200_m1 = PCx * R100_m2 + R000_m2;
    const T R300_m0 = PCx * R200_m1 + T(2) * R100_m1;

    // Y-direction
    const T R010_m0 = PCy * R000_m1;
    const T R010_m1 = PCy * R000_m2;
    const T R010_m2 = PCy * R000_m3;
    const T R110_m0 = PCy * R100_m1;
    const T R110_m1 = PCy * R100_m2;
    const T R210_m0 = PCy * R200_m1;
    const T R020_m0 = PCy * R010_m1 + R000_m1;
    const T R020_m1 = PCy * R010_m2 + R000_m2;
    const T R120_m0 = PCy * R110_m1 + R100_m1;
    const T R030_m0 = PCy * R020_m1 + T(2) * R010_m1;

    // Z-direction
    const T R001_m0 = PCz * R000_m1;
    const T R001_m1 = PCz * R000_m2;
    const T R001_m2 = PCz * R000_m3;
    const T R101_m0 = PCz * R100_m1;
    const T R101_m1 = PCz * R100_m2;
    const T R201_m0 = PCz * R200_m1;
    const T R011_m0 = PCz * R010_m1;
    const T R011_m1 = PCz * R010_m2;
    const T R111_m0 = PCz * R110_m1;
    const T R021_m0 = PCz * R020_m1;
    const T R002_m0 = PCz * R001_m1 + R000_m1;
    const T R002_m1 = PCz * R001_m2 + R000_m2;
    const T R102_m0 = PCz * R101_m1 + R100_m1;
    const T R012_m0 = PCz * R011_m1 + R010_m1;
    const T R003_m0 = PCz * R002_m1 + T(2) * R001_m1;

    R[0] = R000_m0;   R[1] = R100_m0;   R[2] = R010_m0;
    R[3] = R001_m0;   R[4] = R200_m0;   R[5] = R110_m0;
    R[6] = R101_m0;   R[7] = R020_m0;   R[8] = R011_m0;
    R[9] = R002_m0;   R[10] = R300_m0;  R[11] = R210_m0;
    R[12] = R201_m0;  R[13] = R120_m0;  R[14] = R111_m0;
    R[15] = R102_m0;  R[16] = R030_m0;  R[17] = R021_m0;
    R[18] = R012_m0;  R[19] = R003_m0;
}

// ----------------------------------------------------------------------------
// L = 4: 35 components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_3c_L4(const T* boys_table, T p, T PCx, T PCy, T PCz, T* R) {
    const T Tp = p * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2p = T(-2) * p;
    const T neg_2p_2 = neg_2p * neg_2p;
    const T neg_2p_3 = neg_2p_2 * neg_2p;
    const T neg_2p_4 = neg_2p_3 * neg_2p;

    T Fm[5];
    boys_evaluate<T, 5, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2p * Fm[1], R000_m2 = neg_2p_2 * Fm[2];
    const T R000_m3 = neg_2p_3 * Fm[3], R000_m4 = neg_2p_4 * Fm[4];

    // X-direction
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3, R100_m3 = PCx * R000_m4;
    const T R200_m0 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2, R200_m2 = PCx * R100_m3 + R000_m3;
    const T R300_m0 = PCx * R200_m1 + T(2) * R100_m1, R300_m1 = PCx * R200_m2 + T(2) * R100_m2;
    const T R400_m0 = PCx * R300_m1 + T(3) * R200_m1;

    // Y-direction
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3, R010_m3 = PCy * R000_m4;
    const T R020_m0 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2, R020_m2 = PCy * R010_m3 + R000_m3;
    const T R030_m0 = PCy * R020_m1 + T(2) * R010_m1, R030_m1 = PCy * R020_m2 + T(2) * R010_m2;
    const T R040_m0 = PCy * R030_m1 + T(3) * R020_m1;

    // Mixed X-Y
    const T R110_m0 = PCy * R100_m1, R110_m1 = PCy * R100_m2, R110_m2 = PCy * R100_m3;
    const T R120_m0 = PCy * R110_m1 + R100_m1, R120_m1 = PCy * R110_m2 + R100_m2;
    const T R130_m0 = PCy * R120_m1 + T(2) * R110_m1;
    const T R210_m0 = PCy * R200_m1, R210_m1 = PCy * R200_m2;
    const T R220_m0 = PCy * R210_m1 + R200_m1;
    const T R310_m0 = PCy * R300_m1;

    // Z-direction
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3, R001_m3 = PCz * R000_m4;
    const T R002_m0 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2, R002_m2 = PCz * R001_m3 + R000_m3;
    const T R003_m0 = PCz * R002_m1 + T(2) * R001_m1, R003_m1 = PCz * R002_m2 + T(2) * R001_m2;
    const T R004_m0 = PCz * R003_m1 + T(3) * R002_m1;

    // Mixed X-Z
    const T R101_m0 = PCz * R100_m1, R101_m1 = PCz * R100_m2, R101_m2 = PCz * R100_m3;
    const T R102_m0 = PCz * R101_m1 + R100_m1, R102_m1 = PCz * R101_m2 + R100_m2;
    const T R103_m0 = PCz * R102_m1 + T(2) * R101_m1;
    const T R201_m0 = PCz * R200_m1, R201_m1 = PCz * R200_m2;
    const T R202_m0 = PCz * R201_m1 + R200_m1;
    const T R301_m0 = PCz * R300_m1;

    // Mixed Y-Z
    const T R011_m0 = PCz * R010_m1, R011_m1 = PCz * R010_m2, R011_m2 = PCz * R010_m3;
    const T R012_m0 = PCz * R011_m1 + R010_m1, R012_m1 = PCz * R011_m2 + R010_m2;
    const T R013_m0 = PCz * R012_m1 + T(2) * R011_m1;
    const T R021_m0 = PCz * R020_m1, R021_m1 = PCz * R020_m2;
    const T R022_m0 = PCz * R021_m1 + R020_m1;
    const T R031_m0 = PCz * R030_m1;

    // Mixed X-Y-Z
    const T R111_m0 = PCz * R110_m1, R111_m1 = PCz * R110_m2;
    const T R112_m0 = PCz * R111_m1 + R110_m1;
    const T R121_m0 = PCz * R120_m1;
    const T R211_m0 = PCz * R210_m1;

    R[0] = R000_m0;   R[1] = R100_m0;   R[2] = R010_m0;   R[3] = R001_m0;
    R[4] = R200_m0;   R[5] = R110_m0;   R[6] = R101_m0;   R[7] = R020_m0;
    R[8] = R011_m0;   R[9] = R002_m0;   R[10] = R300_m0;  R[11] = R210_m0;
    R[12] = R201_m0;  R[13] = R120_m0;  R[14] = R111_m0;  R[15] = R102_m0;
    R[16] = R030_m0;  R[17] = R021_m0;  R[18] = R012_m0;  R[19] = R003_m0;
    R[20] = R400_m0;  R[21] = R310_m0;  R[22] = R301_m0;  R[23] = R220_m0;
    R[24] = R211_m0;  R[25] = R202_m0;  R[26] = R130_m0;  R[27] = R121_m0;
    R[28] = R112_m0;  R[29] = R103_m0;  R[30] = R040_m0;  R[31] = R031_m0;
    R[32] = R022_m0;  R[33] = R013_m0;  R[34] = R004_m0;
}

// ----------------------------------------------------------------------------
// L = 5: 56 components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_3c_L5(const T* boys_table, T p, T PCx, T PCy, T PCz, T* R) {
    const T Tp = p * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2p = T(-2) * p;
    const T neg_2p_2 = neg_2p * neg_2p;
    const T neg_2p_3 = neg_2p_2 * neg_2p;
    const T neg_2p_4 = neg_2p_3 * neg_2p;
    const T neg_2p_5 = neg_2p_4 * neg_2p;

    T Fm[6];
    boys_evaluate<T, 6, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2p * Fm[1], R000_m2 = neg_2p_2 * Fm[2];
    const T R000_m3 = neg_2p_3 * Fm[3], R000_m4 = neg_2p_4 * Fm[4], R000_m5 = neg_2p_5 * Fm[5];

    // X-direction
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R100_m3 = PCx * R000_m4, R100_m4 = PCx * R000_m5;
    const T R200_m0 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R200_m2 = PCx * R100_m3 + R000_m3, R200_m3 = PCx * R100_m4 + R000_m4;
    const T R300_m0 = PCx * R200_m1 + T(2) * R100_m1, R300_m1 = PCx * R200_m2 + T(2) * R100_m2;
    const T R300_m2 = PCx * R200_m3 + T(2) * R100_m3;
    const T R400_m0 = PCx * R300_m1 + T(3) * R200_m1, R400_m1 = PCx * R300_m2 + T(3) * R200_m2;
    const T R500_m0 = PCx * R400_m1 + T(4) * R300_m1;

    // Y-direction
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R010_m3 = PCy * R000_m4, R010_m4 = PCy * R000_m5;
    const T R020_m0 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R020_m2 = PCy * R010_m3 + R000_m3, R020_m3 = PCy * R010_m4 + R000_m4;
    const T R030_m0 = PCy * R020_m1 + T(2) * R010_m1, R030_m1 = PCy * R020_m2 + T(2) * R010_m2;
    const T R030_m2 = PCy * R020_m3 + T(2) * R010_m3;
    const T R040_m0 = PCy * R030_m1 + T(3) * R020_m1, R040_m1 = PCy * R030_m2 + T(3) * R020_m2;
    const T R050_m0 = PCy * R040_m1 + T(4) * R030_m1;

    // Mixed X-Y
    const T R110_m0 = PCy * R100_m1, R110_m1 = PCy * R100_m2, R110_m2 = PCy * R100_m3, R110_m3 = PCy * R100_m4;
    const T R120_m0 = PCy * R110_m1 + R100_m1, R120_m1 = PCy * R110_m2 + R100_m2, R120_m2 = PCy * R110_m3 + R100_m3;
    const T R130_m0 = PCy * R120_m1 + T(2) * R110_m1, R130_m1 = PCy * R120_m2 + T(2) * R110_m2;
    const T R140_m0 = PCy * R130_m1 + T(3) * R120_m1;
    const T R210_m0 = PCy * R200_m1, R210_m1 = PCy * R200_m2, R210_m2 = PCy * R200_m3;
    const T R220_m0 = PCy * R210_m1 + R200_m1, R220_m1 = PCy * R210_m2 + R200_m2;
    const T R230_m0 = PCy * R220_m1 + T(2) * R210_m1;
    const T R310_m0 = PCy * R300_m1, R310_m1 = PCy * R300_m2;
    const T R320_m0 = PCy * R310_m1 + R300_m1;
    const T R410_m0 = PCy * R400_m1;

    // Z-direction
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R001_m3 = PCz * R000_m4, R001_m4 = PCz * R000_m5;
    const T R002_m0 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R002_m2 = PCz * R001_m3 + R000_m3, R002_m3 = PCz * R001_m4 + R000_m4;
    const T R003_m0 = PCz * R002_m1 + T(2) * R001_m1, R003_m1 = PCz * R002_m2 + T(2) * R001_m2;
    const T R003_m2 = PCz * R002_m3 + T(2) * R001_m3;
    const T R004_m0 = PCz * R003_m1 + T(3) * R002_m1, R004_m1 = PCz * R003_m2 + T(3) * R002_m2;
    const T R005_m0 = PCz * R004_m1 + T(4) * R003_m1;

    // Mixed X-Z
    const T R101_m0 = PCz * R100_m1, R101_m1 = PCz * R100_m2, R101_m2 = PCz * R100_m3, R101_m3 = PCz * R100_m4;
    const T R102_m0 = PCz * R101_m1 + R100_m1, R102_m1 = PCz * R101_m2 + R100_m2, R102_m2 = PCz * R101_m3 + R100_m3;
    const T R103_m0 = PCz * R102_m1 + T(2) * R101_m1, R103_m1 = PCz * R102_m2 + T(2) * R101_m2;
    const T R104_m0 = PCz * R103_m1 + T(3) * R102_m1;
    const T R201_m0 = PCz * R200_m1, R201_m1 = PCz * R200_m2, R201_m2 = PCz * R200_m3;
    const T R202_m0 = PCz * R201_m1 + R200_m1, R202_m1 = PCz * R201_m2 + R200_m2;
    const T R203_m0 = PCz * R202_m1 + T(2) * R201_m1;
    const T R301_m0 = PCz * R300_m1, R301_m1 = PCz * R300_m2;
    const T R302_m0 = PCz * R301_m1 + R300_m1;
    const T R401_m0 = PCz * R400_m1;

    // Mixed Y-Z
    const T R011_m0 = PCz * R010_m1, R011_m1 = PCz * R010_m2, R011_m2 = PCz * R010_m3, R011_m3 = PCz * R010_m4;
    const T R012_m0 = PCz * R011_m1 + R010_m1, R012_m1 = PCz * R011_m2 + R010_m2, R012_m2 = PCz * R011_m3 + R010_m3;
    const T R013_m0 = PCz * R012_m1 + T(2) * R011_m1, R013_m1 = PCz * R012_m2 + T(2) * R011_m2;
    const T R014_m0 = PCz * R013_m1 + T(3) * R012_m1;
    const T R021_m0 = PCz * R020_m1, R021_m1 = PCz * R020_m2, R021_m2 = PCz * R020_m3;
    const T R022_m0 = PCz * R021_m1 + R020_m1, R022_m1 = PCz * R021_m2 + R020_m2;
    const T R023_m0 = PCz * R022_m1 + T(2) * R021_m1;
    const T R031_m0 = PCz * R030_m1, R031_m1 = PCz * R030_m2;
    const T R032_m0 = PCz * R031_m1 + R030_m1;
    const T R041_m0 = PCz * R040_m1;

    // Mixed X-Y-Z
    const T R111_m0 = PCz * R110_m1, R111_m1 = PCz * R110_m2, R111_m2 = PCz * R110_m3;
    const T R112_m0 = PCz * R111_m1 + R110_m1, R112_m1 = PCz * R111_m2 + R110_m2;
    const T R113_m0 = PCz * R112_m1 + T(2) * R111_m1;
    const T R121_m0 = PCz * R120_m1, R121_m1 = PCz * R120_m2;
    const T R122_m0 = PCz * R121_m1 + R120_m1;
    const T R131_m0 = PCz * R130_m1;
    const T R211_m0 = PCz * R210_m1, R211_m1 = PCz * R210_m2;
    const T R212_m0 = PCz * R211_m1 + R210_m1;
    const T R221_m0 = PCz * R220_m1;
    const T R311_m0 = PCz * R310_m1;

    // L=0
    R[0] = R000_m0;
    // L=1
    R[1] = R100_m0; R[2] = R010_m0; R[3] = R001_m0;
    // L=2
    R[4] = R200_m0; R[5] = R110_m0; R[6] = R101_m0; R[7] = R020_m0; R[8] = R011_m0; R[9] = R002_m0;
    // L=3
    R[10] = R300_m0; R[11] = R210_m0; R[12] = R201_m0; R[13] = R120_m0; R[14] = R111_m0;
    R[15] = R102_m0; R[16] = R030_m0; R[17] = R021_m0; R[18] = R012_m0; R[19] = R003_m0;
    // L=4
    R[20] = R400_m0; R[21] = R310_m0; R[22] = R301_m0; R[23] = R220_m0; R[24] = R211_m0;
    R[25] = R202_m0; R[26] = R130_m0; R[27] = R121_m0; R[28] = R112_m0; R[29] = R103_m0;
    R[30] = R040_m0; R[31] = R031_m0; R[32] = R022_m0; R[33] = R013_m0; R[34] = R004_m0;
    // L=5
    R[35] = R500_m0; R[36] = R410_m0; R[37] = R401_m0; R[38] = R320_m0; R[39] = R311_m0;
    R[40] = R302_m0; R[41] = R230_m0; R[42] = R221_m0; R[43] = R212_m0; R[44] = R203_m0;
    R[45] = R140_m0; R[46] = R131_m0; R[47] = R122_m0; R[48] = R113_m0; R[49] = R104_m0;
    R[50] = R050_m0; R[51] = R041_m0; R[52] = R032_m0; R[53] = R023_m0; R[54] = R014_m0;
    R[55] = R005_m0;
}

// ----------------------------------------------------------------------------
// L = 6: 84 components
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_3c_L6(const T* boys_table, T p, T PCx, T PCy, T PCz, T* R) {
    const T Tp = p * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2p = T(-2) * p;
    const T neg_2p_2 = neg_2p * neg_2p;
    const T neg_2p_3 = neg_2p_2 * neg_2p;
    const T neg_2p_4 = neg_2p_3 * neg_2p;
    const T neg_2p_5 = neg_2p_4 * neg_2p;
    const T neg_2p_6 = neg_2p_5 * neg_2p;

    T Fm[7];
    boys_evaluate<T, 7, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2p * Fm[1], R000_m2 = neg_2p_2 * Fm[2];
    const T R000_m3 = neg_2p_3 * Fm[3], R000_m4 = neg_2p_4 * Fm[4];
    const T R000_m5 = neg_2p_5 * Fm[5], R000_m6 = neg_2p_6 * Fm[6];

    // X-direction: R(t,0,0;m)
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R100_m3 = PCx * R000_m4, R100_m4 = PCx * R000_m5, R100_m5 = PCx * R000_m6;
    const T R200_m0 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R200_m2 = PCx * R100_m3 + R000_m3, R200_m3 = PCx * R100_m4 + R000_m4;
    const T R200_m4 = PCx * R100_m5 + R000_m5;
    const T R300_m0 = PCx * R200_m1 + T(2) * R100_m1, R300_m1 = PCx * R200_m2 + T(2) * R100_m2;
    const T R300_m2 = PCx * R200_m3 + T(2) * R100_m3, R300_m3 = PCx * R200_m4 + T(2) * R100_m4;
    const T R400_m0 = PCx * R300_m1 + T(3) * R200_m1, R400_m1 = PCx * R300_m2 + T(3) * R200_m2;
    const T R400_m2 = PCx * R300_m3 + T(3) * R200_m3;
    const T R500_m0 = PCx * R400_m1 + T(4) * R300_m1, R500_m1 = PCx * R400_m2 + T(4) * R300_m2;
    const T R600_m0 = PCx * R500_m1 + T(5) * R400_m1;

    // Y-direction: R(0,u,0;m)
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R010_m3 = PCy * R000_m4, R010_m4 = PCy * R000_m5, R010_m5 = PCy * R000_m6;
    const T R020_m0 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R020_m2 = PCy * R010_m3 + R000_m3, R020_m3 = PCy * R010_m4 + R000_m4;
    const T R020_m4 = PCy * R010_m5 + R000_m5;
    const T R030_m0 = PCy * R020_m1 + T(2) * R010_m1, R030_m1 = PCy * R020_m2 + T(2) * R010_m2;
    const T R030_m2 = PCy * R020_m3 + T(2) * R010_m3, R030_m3 = PCy * R020_m4 + T(2) * R010_m4;
    const T R040_m0 = PCy * R030_m1 + T(3) * R020_m1, R040_m1 = PCy * R030_m2 + T(3) * R020_m2;
    const T R040_m2 = PCy * R030_m3 + T(3) * R020_m3;
    const T R050_m0 = PCy * R040_m1 + T(4) * R030_m1, R050_m1 = PCy * R040_m2 + T(4) * R030_m2;
    const T R060_m0 = PCy * R050_m1 + T(5) * R040_m1;

    // Z-direction: R(0,0,v;m)
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R001_m3 = PCz * R000_m4, R001_m4 = PCz * R000_m5, R001_m5 = PCz * R000_m6;
    const T R002_m0 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R002_m2 = PCz * R001_m3 + R000_m3, R002_m3 = PCz * R001_m4 + R000_m4;
    const T R002_m4 = PCz * R001_m5 + R000_m5;
    const T R003_m0 = PCz * R002_m1 + T(2) * R001_m1, R003_m1 = PCz * R002_m2 + T(2) * R001_m2;
    const T R003_m2 = PCz * R002_m3 + T(2) * R001_m3, R003_m3 = PCz * R002_m4 + T(2) * R001_m4;
    const T R004_m0 = PCz * R003_m1 + T(3) * R002_m1, R004_m1 = PCz * R003_m2 + T(3) * R002_m2;
    const T R004_m2 = PCz * R003_m3 + T(3) * R002_m3;
    const T R005_m0 = PCz * R004_m1 + T(4) * R003_m1, R005_m1 = PCz * R004_m2 + T(4) * R003_m2;
    const T R006_m0 = PCz * R005_m1 + T(5) * R004_m1;

    // Mixed X-Y
    const T R110_m0 = PCy * R100_m1, R110_m1 = PCy * R100_m2, R110_m2 = PCy * R100_m3;
    const T R110_m3 = PCy * R100_m4, R110_m4 = PCy * R100_m5;
    const T R120_m0 = PCy * R110_m1 + R100_m1, R120_m1 = PCy * R110_m2 + R100_m2;
    const T R120_m2 = PCy * R110_m3 + R100_m3, R120_m3 = PCy * R110_m4 + R100_m4;
    const T R130_m0 = PCy * R120_m1 + T(2) * R110_m1, R130_m1 = PCy * R120_m2 + T(2) * R110_m2;
    const T R130_m2 = PCy * R120_m3 + T(2) * R110_m3;
    const T R140_m0 = PCy * R130_m1 + T(3) * R120_m1, R140_m1 = PCy * R130_m2 + T(3) * R120_m2;
    const T R150_m0 = PCy * R140_m1 + T(4) * R130_m1;
    const T R210_m0 = PCy * R200_m1, R210_m1 = PCy * R200_m2, R210_m2 = PCy * R200_m3, R210_m3 = PCy * R200_m4;
    const T R220_m0 = PCy * R210_m1 + R200_m1, R220_m1 = PCy * R210_m2 + R200_m2, R220_m2 = PCy * R210_m3 + R200_m3;
    const T R230_m0 = PCy * R220_m1 + T(2) * R210_m1, R230_m1 = PCy * R220_m2 + T(2) * R210_m2;
    const T R240_m0 = PCy * R230_m1 + T(3) * R220_m1;
    const T R310_m0 = PCy * R300_m1, R310_m1 = PCy * R300_m2, R310_m2 = PCy * R300_m3;
    const T R320_m0 = PCy * R310_m1 + R300_m1, R320_m1 = PCy * R310_m2 + R300_m2;
    const T R330_m0 = PCy * R320_m1 + T(2) * R310_m1;
    const T R410_m0 = PCy * R400_m1, R410_m1 = PCy * R400_m2;
    const T R420_m0 = PCy * R410_m1 + R400_m1;
    const T R510_m0 = PCy * R500_m1;

    // Mixed X-Z
    const T R101_m0 = PCz * R100_m1, R101_m1 = PCz * R100_m2, R101_m2 = PCz * R100_m3;
    const T R101_m3 = PCz * R100_m4, R101_m4 = PCz * R100_m5;
    const T R102_m0 = PCz * R101_m1 + R100_m1, R102_m1 = PCz * R101_m2 + R100_m2;
    const T R102_m2 = PCz * R101_m3 + R100_m3, R102_m3 = PCz * R101_m4 + R100_m4;
    const T R103_m0 = PCz * R102_m1 + T(2) * R101_m1, R103_m1 = PCz * R102_m2 + T(2) * R101_m2;
    const T R103_m2 = PCz * R102_m3 + T(2) * R101_m3;
    const T R104_m0 = PCz * R103_m1 + T(3) * R102_m1, R104_m1 = PCz * R103_m2 + T(3) * R102_m2;
    const T R105_m0 = PCz * R104_m1 + T(4) * R103_m1;
    const T R201_m0 = PCz * R200_m1, R201_m1 = PCz * R200_m2, R201_m2 = PCz * R200_m3, R201_m3 = PCz * R200_m4;
    const T R202_m0 = PCz * R201_m1 + R200_m1, R202_m1 = PCz * R201_m2 + R200_m2, R202_m2 = PCz * R201_m3 + R200_m3;
    const T R203_m0 = PCz * R202_m1 + T(2) * R201_m1, R203_m1 = PCz * R202_m2 + T(2) * R201_m2;
    const T R204_m0 = PCz * R203_m1 + T(3) * R202_m1;
    const T R301_m0 = PCz * R300_m1, R301_m1 = PCz * R300_m2, R301_m2 = PCz * R300_m3;
    const T R302_m0 = PCz * R301_m1 + R300_m1, R302_m1 = PCz * R301_m2 + R300_m2;
    const T R303_m0 = PCz * R302_m1 + T(2) * R301_m1;
    const T R401_m0 = PCz * R400_m1, R401_m1 = PCz * R400_m2;
    const T R402_m0 = PCz * R401_m1 + R400_m1;
    const T R501_m0 = PCz * R500_m1;

    // Mixed Y-Z
    const T R011_m0 = PCz * R010_m1, R011_m1 = PCz * R010_m2, R011_m2 = PCz * R010_m3;
    const T R011_m3 = PCz * R010_m4, R011_m4 = PCz * R010_m5;
    const T R012_m0 = PCz * R011_m1 + R010_m1, R012_m1 = PCz * R011_m2 + R010_m2;
    const T R012_m2 = PCz * R011_m3 + R010_m3, R012_m3 = PCz * R011_m4 + R010_m4;
    const T R013_m0 = PCz * R012_m1 + T(2) * R011_m1, R013_m1 = PCz * R012_m2 + T(2) * R011_m2;
    const T R013_m2 = PCz * R012_m3 + T(2) * R011_m3;
    const T R014_m0 = PCz * R013_m1 + T(3) * R012_m1, R014_m1 = PCz * R013_m2 + T(3) * R012_m2;
    const T R015_m0 = PCz * R014_m1 + T(4) * R013_m1;
    const T R021_m0 = PCz * R020_m1, R021_m1 = PCz * R020_m2, R021_m2 = PCz * R020_m3, R021_m3 = PCz * R020_m4;
    const T R022_m0 = PCz * R021_m1 + R020_m1, R022_m1 = PCz * R021_m2 + R020_m2, R022_m2 = PCz * R021_m3 + R020_m3;
    const T R023_m0 = PCz * R022_m1 + T(2) * R021_m1, R023_m1 = PCz * R022_m2 + T(2) * R021_m2;
    const T R024_m0 = PCz * R023_m1 + T(3) * R022_m1;
    const T R031_m0 = PCz * R030_m1, R031_m1 = PCz * R030_m2, R031_m2 = PCz * R030_m3;
    const T R032_m0 = PCz * R031_m1 + R030_m1, R032_m1 = PCz * R031_m2 + R030_m2;
    const T R033_m0 = PCz * R032_m1 + T(2) * R031_m1;
    const T R041_m0 = PCz * R040_m1, R041_m1 = PCz * R040_m2;
    const T R042_m0 = PCz * R041_m1 + R040_m1;
    const T R051_m0 = PCz * R050_m1;

    // Mixed X-Y-Z (just need m=0)
    const T R111_m0 = PCz * R110_m1, R111_m1 = PCz * R110_m2, R111_m2 = PCz * R110_m3, R111_m3 = PCz * R110_m4;
    const T R112_m0 = PCz * R111_m1 + R110_m1, R112_m1 = PCz * R111_m2 + R110_m2, R112_m2 = PCz * R111_m3 + R110_m3;
    const T R113_m0 = PCz * R112_m1 + T(2) * R111_m1, R113_m1 = PCz * R112_m2 + T(2) * R111_m2;
    const T R114_m0 = PCz * R113_m1 + T(3) * R112_m1;
    const T R121_m0 = PCz * R120_m1, R121_m1 = PCz * R120_m2, R121_m2 = PCz * R120_m3;
    const T R122_m0 = PCz * R121_m1 + R120_m1, R122_m1 = PCz * R121_m2 + R120_m2;
    const T R123_m0 = PCz * R122_m1 + T(2) * R121_m1;
    const T R131_m0 = PCz * R130_m1, R131_m1 = PCz * R130_m2;
    const T R132_m0 = PCz * R131_m1 + R130_m1;
    const T R141_m0 = PCz * R140_m1;
    const T R211_m0 = PCz * R210_m1, R211_m1 = PCz * R210_m2, R211_m2 = PCz * R210_m3;
    const T R212_m0 = PCz * R211_m1 + R210_m1, R212_m1 = PCz * R211_m2 + R210_m2;
    const T R213_m0 = PCz * R212_m1 + T(2) * R211_m1;
    const T R221_m0 = PCz * R220_m1, R221_m1 = PCz * R220_m2;
    const T R222_m0 = PCz * R221_m1 + R220_m1;
    const T R231_m0 = PCz * R230_m1;
    const T R311_m0 = PCz * R310_m1, R311_m1 = PCz * R310_m2;
    const T R312_m0 = PCz * R311_m1 + R310_m1;
    const T R321_m0 = PCz * R320_m1;
    const T R411_m0 = PCz * R410_m1;

    // Output in hermite_index order
    // L=0
    R[0] = R000_m0;
    // L=1
    R[1] = R100_m0; R[2] = R010_m0; R[3] = R001_m0;
    // L=2
    R[4] = R200_m0; R[5] = R110_m0; R[6] = R101_m0; R[7] = R020_m0; R[8] = R011_m0; R[9] = R002_m0;
    // L=3
    R[10] = R300_m0; R[11] = R210_m0; R[12] = R201_m0; R[13] = R120_m0; R[14] = R111_m0;
    R[15] = R102_m0; R[16] = R030_m0; R[17] = R021_m0; R[18] = R012_m0; R[19] = R003_m0;
    // L=4
    R[20] = R400_m0; R[21] = R310_m0; R[22] = R301_m0; R[23] = R220_m0; R[24] = R211_m0;
    R[25] = R202_m0; R[26] = R130_m0; R[27] = R121_m0; R[28] = R112_m0; R[29] = R103_m0;
    R[30] = R040_m0; R[31] = R031_m0; R[32] = R022_m0; R[33] = R013_m0; R[34] = R004_m0;
    // L=5
    R[35] = R500_m0; R[36] = R410_m0; R[37] = R401_m0; R[38] = R320_m0; R[39] = R311_m0;
    R[40] = R302_m0; R[41] = R230_m0; R[42] = R221_m0; R[43] = R212_m0; R[44] = R203_m0;
    R[45] = R140_m0; R[46] = R131_m0; R[47] = R122_m0; R[48] = R113_m0; R[49] = R104_m0;
    R[50] = R050_m0; R[51] = R041_m0; R[52] = R032_m0; R[53] = R023_m0; R[54] = R014_m0;
    R[55] = R005_m0;
    // L=6
    R[56] = R600_m0; R[57] = R510_m0; R[58] = R501_m0; R[59] = R420_m0; R[60] = R411_m0;
    R[61] = R402_m0; R[62] = R330_m0; R[63] = R321_m0; R[64] = R312_m0; R[65] = R303_m0;
    R[66] = R240_m0; R[67] = R231_m0; R[68] = R222_m0; R[69] = R213_m0; R[70] = R204_m0;
    R[71] = R150_m0; R[72] = R141_m0; R[73] = R132_m0; R[74] = R123_m0; R[75] = R114_m0;
    R[76] = R105_m0; R[77] = R060_m0; R[78] = R051_m0; R[79] = R042_m0; R[80] = R033_m0;
    R[81] = R024_m0; R[82] = R015_m0; R[83] = R006_m0;
}

// ============================================================================
// Fused forward/backward kernels for common (L_ab, lc) combinations
// These compute Boys values and directly contract without intermediate R storage
// ============================================================================

// Forward pass: Y[h_c] += prefactor * sum_h_ab(X[h_ab] * R[idx(h_ab,h_c)])
// Backward pass: U[h_ab] += prefactor * sum_h_c(R[idx(h_ab,h_c)] * T[h_c])

// ----------------------------------------------------------------------------
// L_ab=0, lc=0: (s|s) - L_total=0, nherm_ab=1, nherm_c=1
// R_pq[0,0] = R[0] = F0
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_0_0(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    T F0;
    boys_evaluate<T, 1, BoysParams>(boys_table, Tp, 0, &F0);
    Y[0] += prefactor * X[0] * F0;
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_0_0(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    T F0;
    boys_evaluate<T, 1, BoysParams>(boys_table, Tp, 0, &F0);
    U[0] += prefactor * F0 * T_vec[0];
}

// ----------------------------------------------------------------------------
// L_ab=0, lc=1: (s|p) - L_total=1, nherm_ab=1, nherm_c=4
// R_pq[0, h_c] = R[h_c] for h_c in 0-3
// Forward: Y[h_c] += pf * X[0] * R[h_c]
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_0_1(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    T Fm[2];
    boys_evaluate<T, 2, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0];
    const T R000_m1 = neg_2a * Fm[1];
    const T R100 = PCx * R000_m1;
    const T R010 = PCy * R000_m1;
    const T R001 = PCz * R000_m1;

    const T pX0 = prefactor * X[0];
    Y[0] += pX0 * R000;
    Y[1] += pX0 * R100;
    Y[2] += pX0 * R010;
    Y[3] += pX0 * R001;
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_0_1(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    T Fm[2];
    boys_evaluate<T, 2, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0];
    const T R000_m1 = neg_2a * Fm[1];
    const T R100 = PCx * R000_m1;
    const T R010 = PCy * R000_m1;
    const T R001 = PCz * R000_m1;

    // U[0] += pf * (R[0]*T[0] + R[1]*T[1] + R[2]*T[2] + R[3]*T[3])
    U[0] += prefactor * (R000 * T_vec[0] + R100 * T_vec[1] + R010 * T_vec[2] + R001 * T_vec[3]);
}

// ----------------------------------------------------------------------------
// L_ab=1, lc=0: (p|s) - L_total=1, nherm_ab=4, nherm_c=1
// R_pq[h_ab, 0] = R[h_ab] for h_ab in 0-3
// Forward: Y[0] += pf * sum(X[h_ab] * R[h_ab])
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_1_0(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    T Fm[2];
    boys_evaluate<T, 2, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0];
    const T R000_m1 = neg_2a * Fm[1];
    const T R100 = PCx * R000_m1;
    const T R010 = PCy * R000_m1;
    const T R001 = PCz * R000_m1;

    Y[0] += prefactor * (X[0] * R000 + X[1] * R100 + X[2] * R010 + X[3] * R001);
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_1_0(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    T Fm[2];
    boys_evaluate<T, 2, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0];
    const T R000_m1 = neg_2a * Fm[1];
    const T R100 = PCx * R000_m1;
    const T R010 = PCy * R000_m1;
    const T R001 = PCz * R000_m1;

    const T pT0 = prefactor * T_vec[0];
    U[0] += pT0 * R000;
    U[1] += pT0 * R100;
    U[2] += pT0 * R010;
    U[3] += pT0 * R001;
}

// ----------------------------------------------------------------------------
// L_ab=0, lc=2: (s|d) - L_total=2, nherm_ab=1, nherm_c=10
// R_pq[0, h_c] = R[h_c] for h_c in 0-9
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_0_2(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    T Fm[3];
    boys_evaluate<T, 3, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2];
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2;
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2;
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2;
    const T R200_m0 = PCx * R100_m1 + R000_m1;
    const T R110_m0 = PCy * R100_m1;
    const T R101_m0 = PCz * R100_m1;
    const T R020_m0 = PCy * R010_m1 + R000_m1;
    const T R011_m0 = PCz * R010_m1;
    const T R002_m0 = PCz * R001_m1 + R000_m1;

    const T pX0 = prefactor * X[0];
    Y[0] += pX0 * R000_m0;
    Y[1] += pX0 * R100_m0;
    Y[2] += pX0 * R010_m0;
    Y[3] += pX0 * R001_m0;
    Y[4] += pX0 * R200_m0;
    Y[5] += pX0 * R110_m0;
    Y[6] += pX0 * R101_m0;
    Y[7] += pX0 * R020_m0;
    Y[8] += pX0 * R011_m0;
    Y[9] += pX0 * R002_m0;
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_0_2(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    T Fm[3];
    boys_evaluate<T, 3, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2];
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2;
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2;
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2;
    const T R200_m0 = PCx * R100_m1 + R000_m1;
    const T R110_m0 = PCy * R100_m1;
    const T R101_m0 = PCz * R100_m1;
    const T R020_m0 = PCy * R010_m1 + R000_m1;
    const T R011_m0 = PCz * R010_m1;
    const T R002_m0 = PCz * R001_m1 + R000_m1;

    U[0] += prefactor * (R000_m0 * T_vec[0] + R100_m0 * T_vec[1] + R010_m0 * T_vec[2] +
                         R001_m0 * T_vec[3] + R200_m0 * T_vec[4] + R110_m0 * T_vec[5] +
                         R101_m0 * T_vec[6] + R020_m0 * T_vec[7] + R011_m0 * T_vec[8] +
                         R002_m0 * T_vec[9]);
}

// ----------------------------------------------------------------------------
// L_ab=2, lc=0: (d|s) - L_total=2, nherm_ab=10, nherm_c=1
// R_pq[h_ab, 0] = R[h_ab] for h_ab in 0-9
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_2_0(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    T Fm[3];
    boys_evaluate<T, 3, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2];
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2;
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2;
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2;
    const T R200_m0 = PCx * R100_m1 + R000_m1;
    const T R110_m0 = PCy * R100_m1;
    const T R101_m0 = PCz * R100_m1;
    const T R020_m0 = PCy * R010_m1 + R000_m1;
    const T R011_m0 = PCz * R010_m1;
    const T R002_m0 = PCz * R001_m1 + R000_m1;

    Y[0] += prefactor * (X[0] * R000_m0 + X[1] * R100_m0 + X[2] * R010_m0 +
                         X[3] * R001_m0 + X[4] * R200_m0 + X[5] * R110_m0 +
                         X[6] * R101_m0 + X[7] * R020_m0 + X[8] * R011_m0 +
                         X[9] * R002_m0);
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_2_0(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    T Fm[3];
    boys_evaluate<T, 3, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2];
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2;
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2;
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2;
    const T R200_m0 = PCx * R100_m1 + R000_m1;
    const T R110_m0 = PCy * R100_m1;
    const T R101_m0 = PCz * R100_m1;
    const T R020_m0 = PCy * R010_m1 + R000_m1;
    const T R011_m0 = PCz * R010_m1;
    const T R002_m0 = PCz * R001_m1 + R000_m1;

    const T pT0 = prefactor * T_vec[0];
    U[0] += pT0 * R000_m0;
    U[1] += pT0 * R100_m0;
    U[2] += pT0 * R010_m0;
    U[3] += pT0 * R001_m0;
    U[4] += pT0 * R200_m0;
    U[5] += pT0 * R110_m0;
    U[6] += pT0 * R101_m0;
    U[7] += pT0 * R020_m0;
    U[8] += pT0 * R011_m0;
    U[9] += pT0 * R002_m0;
}

// ----------------------------------------------------------------------------
// L_ab=1, lc=1: (p|p) - L_total=2, nherm_ab=4, nherm_c=4
// R_pq is 4x4, symmetric in some indices
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_1_1(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    T Fm[3];
    boys_evaluate<T, 3, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2];
    const T R100 = PCx * R000_m1, R100_m1 = PCx * R000_m2;
    const T R010 = PCy * R000_m1, R010_m1 = PCy * R000_m2;
    const T R001 = PCz * R000_m1, R001_m1 = PCz * R000_m2;
    const T R200 = PCx * R100_m1 + R000_m1;
    const T R110 = PCy * R100_m1;
    const T R101 = PCz * R100_m1;
    const T R020 = PCy * R010_m1 + R000_m1;
    const T R011 = PCz * R010_m1;
    const T R002 = PCz * R001_m1 + R000_m1;

    // R_pq^T @ X where R_pq[h_ab, h_c] = R[combined]
    // Y[0] += pf * (R000*X0 + R100*X1 + R010*X2 + R001*X3)
    // Y[1] += pf * (R100*X0 + R200*X1 + R110*X2 + R101*X3)
    // Y[2] += pf * (R010*X0 + R110*X1 + R020*X2 + R011*X3)
    // Y[3] += pf * (R001*X0 + R101*X1 + R011*X2 + R002*X3)
    Y[0] += prefactor * (R000 * X[0] + R100 * X[1] + R010 * X[2] + R001 * X[3]);
    Y[1] += prefactor * (R100 * X[0] + R200 * X[1] + R110 * X[2] + R101 * X[3]);
    Y[2] += prefactor * (R010 * X[0] + R110 * X[1] + R020 * X[2] + R011 * X[3]);
    Y[3] += prefactor * (R001 * X[0] + R101 * X[1] + R011 * X[2] + R002 * X[3]);
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_1_1(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    T Fm[3];
    boys_evaluate<T, 3, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2];
    const T R100 = PCx * R000_m1, R100_m1 = PCx * R000_m2;
    const T R010 = PCy * R000_m1, R010_m1 = PCy * R000_m2;
    const T R001 = PCz * R000_m1, R001_m1 = PCz * R000_m2;
    const T R200 = PCx * R100_m1 + R000_m1;
    const T R110 = PCy * R100_m1;
    const T R101 = PCz * R100_m1;
    const T R020 = PCy * R010_m1 + R000_m1;
    const T R011 = PCz * R010_m1;
    const T R002 = PCz * R001_m1 + R000_m1;

    // U += pf * R_pq @ T
    // U[0] += pf * (R000*T0 + R100*T1 + R010*T2 + R001*T3)
    // U[1] += pf * (R100*T0 + R200*T1 + R110*T2 + R101*T3)
    // U[2] += pf * (R010*T0 + R110*T1 + R020*T2 + R011*T3)
    // U[3] += pf * (R001*T0 + R101*T1 + R011*T2 + R002*T3)
    U[0] += prefactor * (R000 * T_vec[0] + R100 * T_vec[1] + R010 * T_vec[2] + R001 * T_vec[3]);
    U[1] += prefactor * (R100 * T_vec[0] + R200 * T_vec[1] + R110 * T_vec[2] + R101 * T_vec[3]);
    U[2] += prefactor * (R010 * T_vec[0] + R110 * T_vec[1] + R020 * T_vec[2] + R011 * T_vec[3]);
    U[3] += prefactor * (R001 * T_vec[0] + R101 * T_vec[1] + R011 * T_vec[2] + R002 * T_vec[3]);
}

// ----------------------------------------------------------------------------
// L_ab=0, lc=3: (s|f) - L_total=3, nherm_ab=1, nherm_c=20
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_0_3(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    const T neg_2a_3 = neg_2a_2 * neg_2a;
    T Fm[4];
    boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2], R000_m3 = neg_2a_3 * Fm[3];
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R200_m0 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R110_m0 = PCy * R100_m1, R110_m1 = PCy * R100_m2;
    const T R101_m0 = PCz * R100_m1, R101_m1 = PCz * R100_m2;
    const T R020_m0 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R011_m0 = PCz * R010_m1, R011_m1 = PCz * R010_m2;
    const T R002_m0 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R300_m0 = PCx * R200_m1 + T(2) * R100_m1;
    const T R210_m0 = PCy * R200_m1;
    const T R201_m0 = PCz * R200_m1;
    const T R120_m0 = PCy * R110_m1 + R100_m1;
    const T R111_m0 = PCz * R110_m1;
    const T R102_m0 = PCz * R101_m1 + R100_m1;
    const T R030_m0 = PCy * R020_m1 + T(2) * R010_m1;
    const T R021_m0 = PCz * R020_m1;
    const T R012_m0 = PCz * R011_m1 + R010_m1;
    const T R003_m0 = PCz * R002_m1 + T(2) * R001_m1;

    const T pX0 = prefactor * X[0];
    Y[0] += pX0 * R000_m0; Y[1] += pX0 * R100_m0; Y[2] += pX0 * R010_m0; Y[3] += pX0 * R001_m0;
    Y[4] += pX0 * R200_m0; Y[5] += pX0 * R110_m0; Y[6] += pX0 * R101_m0;
    Y[7] += pX0 * R020_m0; Y[8] += pX0 * R011_m0; Y[9] += pX0 * R002_m0;
    Y[10] += pX0 * R300_m0; Y[11] += pX0 * R210_m0; Y[12] += pX0 * R201_m0;
    Y[13] += pX0 * R120_m0; Y[14] += pX0 * R111_m0; Y[15] += pX0 * R102_m0;
    Y[16] += pX0 * R030_m0; Y[17] += pX0 * R021_m0; Y[18] += pX0 * R012_m0; Y[19] += pX0 * R003_m0;
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_0_3(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    const T neg_2a_3 = neg_2a_2 * neg_2a;
    T Fm[4];
    boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2], R000_m3 = neg_2a_3 * Fm[3];
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R200_m0 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R110_m0 = PCy * R100_m1, R110_m1 = PCy * R100_m2;
    const T R101_m0 = PCz * R100_m1, R101_m1 = PCz * R100_m2;
    const T R020_m0 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R011_m0 = PCz * R010_m1, R011_m1 = PCz * R010_m2;
    const T R002_m0 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R300_m0 = PCx * R200_m1 + T(2) * R100_m1;
    const T R210_m0 = PCy * R200_m1;
    const T R201_m0 = PCz * R200_m1;
    const T R120_m0 = PCy * R110_m1 + R100_m1;
    const T R111_m0 = PCz * R110_m1;
    const T R102_m0 = PCz * R101_m1 + R100_m1;
    const T R030_m0 = PCy * R020_m1 + T(2) * R010_m1;
    const T R021_m0 = PCz * R020_m1;
    const T R012_m0 = PCz * R011_m1 + R010_m1;
    const T R003_m0 = PCz * R002_m1 + T(2) * R001_m1;

    U[0] += prefactor * (R000_m0 * T_vec[0] + R100_m0 * T_vec[1] + R010_m0 * T_vec[2] + R001_m0 * T_vec[3] +
                         R200_m0 * T_vec[4] + R110_m0 * T_vec[5] + R101_m0 * T_vec[6] +
                         R020_m0 * T_vec[7] + R011_m0 * T_vec[8] + R002_m0 * T_vec[9] +
                         R300_m0 * T_vec[10] + R210_m0 * T_vec[11] + R201_m0 * T_vec[12] +
                         R120_m0 * T_vec[13] + R111_m0 * T_vec[14] + R102_m0 * T_vec[15] +
                         R030_m0 * T_vec[16] + R021_m0 * T_vec[17] + R012_m0 * T_vec[18] + R003_m0 * T_vec[19]);
}

// ----------------------------------------------------------------------------
// L_ab=3, lc=0: (f|s) - L_total=3, nherm_ab=20, nherm_c=1
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_3_0(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    const T neg_2a_3 = neg_2a_2 * neg_2a;
    T Fm[4];
    boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2], R000_m3 = neg_2a_3 * Fm[3];
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R200_m0 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R110_m0 = PCy * R100_m1, R110_m1 = PCy * R100_m2;
    const T R101_m0 = PCz * R100_m1, R101_m1 = PCz * R100_m2;
    const T R020_m0 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R011_m0 = PCz * R010_m1, R011_m1 = PCz * R010_m2;
    const T R002_m0 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R300_m0 = PCx * R200_m1 + T(2) * R100_m1;
    const T R210_m0 = PCy * R200_m1;
    const T R201_m0 = PCz * R200_m1;
    const T R120_m0 = PCy * R110_m1 + R100_m1;
    const T R111_m0 = PCz * R110_m1;
    const T R102_m0 = PCz * R101_m1 + R100_m1;
    const T R030_m0 = PCy * R020_m1 + T(2) * R010_m1;
    const T R021_m0 = PCz * R020_m1;
    const T R012_m0 = PCz * R011_m1 + R010_m1;
    const T R003_m0 = PCz * R002_m1 + T(2) * R001_m1;

    Y[0] += prefactor * (X[0] * R000_m0 + X[1] * R100_m0 + X[2] * R010_m0 + X[3] * R001_m0 +
                         X[4] * R200_m0 + X[5] * R110_m0 + X[6] * R101_m0 +
                         X[7] * R020_m0 + X[8] * R011_m0 + X[9] * R002_m0 +
                         X[10] * R300_m0 + X[11] * R210_m0 + X[12] * R201_m0 +
                         X[13] * R120_m0 + X[14] * R111_m0 + X[15] * R102_m0 +
                         X[16] * R030_m0 + X[17] * R021_m0 + X[18] * R012_m0 + X[19] * R003_m0);
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_3_0(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    const T neg_2a_3 = neg_2a_2 * neg_2a;
    T Fm[4];
    boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000_m0 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2], R000_m3 = neg_2a_3 * Fm[3];
    const T R100_m0 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R010_m0 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R001_m0 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R200_m0 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R110_m0 = PCy * R100_m1, R110_m1 = PCy * R100_m2;
    const T R101_m0 = PCz * R100_m1, R101_m1 = PCz * R100_m2;
    const T R020_m0 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R011_m0 = PCz * R010_m1, R011_m1 = PCz * R010_m2;
    const T R002_m0 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R300_m0 = PCx * R200_m1 + T(2) * R100_m1;
    const T R210_m0 = PCy * R200_m1;
    const T R201_m0 = PCz * R200_m1;
    const T R120_m0 = PCy * R110_m1 + R100_m1;
    const T R111_m0 = PCz * R110_m1;
    const T R102_m0 = PCz * R101_m1 + R100_m1;
    const T R030_m0 = PCy * R020_m1 + T(2) * R010_m1;
    const T R021_m0 = PCz * R020_m1;
    const T R012_m0 = PCz * R011_m1 + R010_m1;
    const T R003_m0 = PCz * R002_m1 + T(2) * R001_m1;

    const T pT0 = prefactor * T_vec[0];
    U[0] += pT0 * R000_m0; U[1] += pT0 * R100_m0; U[2] += pT0 * R010_m0; U[3] += pT0 * R001_m0;
    U[4] += pT0 * R200_m0; U[5] += pT0 * R110_m0; U[6] += pT0 * R101_m0;
    U[7] += pT0 * R020_m0; U[8] += pT0 * R011_m0; U[9] += pT0 * R002_m0;
    U[10] += pT0 * R300_m0; U[11] += pT0 * R210_m0; U[12] += pT0 * R201_m0;
    U[13] += pT0 * R120_m0; U[14] += pT0 * R111_m0; U[15] += pT0 * R102_m0;
    U[16] += pT0 * R030_m0; U[17] += pT0 * R021_m0; U[18] += pT0 * R012_m0; U[19] += pT0 * R003_m0;
}

// ----------------------------------------------------------------------------
// L_ab=1, lc=2: (p|d) - L_total=3, nherm_ab=4, nherm_c=10
// R_pq[h_ab, h_c] where h_ab in 0-3, h_c in 0-9
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_1_2(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    const T neg_2a_3 = neg_2a_2 * neg_2a;
    T Fm[4];
    boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2], R000_m3 = neg_2a_3 * Fm[3];
    const T R100 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R010 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R001 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R200 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R110 = PCy * R100_m1, R110_m1 = PCy * R100_m2;
    const T R101 = PCz * R100_m1, R101_m1 = PCz * R100_m2;
    const T R020 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R011 = PCz * R010_m1, R011_m1 = PCz * R010_m2;
    const T R002 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R300 = PCx * R200_m1 + T(2) * R100_m1;
    const T R210 = PCy * R200_m1;
    const T R201 = PCz * R200_m1;
    const T R120 = PCy * R110_m1 + R100_m1;
    const T R111 = PCz * R110_m1;
    const T R102 = PCz * R101_m1 + R100_m1;
    const T R030 = PCy * R020_m1 + T(2) * R010_m1;
    const T R021 = PCz * R020_m1;
    const T R012 = PCz * R011_m1 + R010_m1;
    const T R003 = PCz * R002_m1 + T(2) * R001_m1;

    // Y[h_c] += pf * sum_h_ab(X[h_ab] * R_pq[h_ab, h_c])
    // h_ab: 0=(000), 1=(100), 2=(010), 3=(001)
    // h_c:  0=(000), 1=(100), 2=(010), 3=(001), 4=(200), 5=(110), 6=(101), 7=(020), 8=(011), 9=(002)
    Y[0] += prefactor * (X[0] * R000 + X[1] * R100 + X[2] * R010 + X[3] * R001);
    Y[1] += prefactor * (X[0] * R100 + X[1] * R200 + X[2] * R110 + X[3] * R101);
    Y[2] += prefactor * (X[0] * R010 + X[1] * R110 + X[2] * R020 + X[3] * R011);
    Y[3] += prefactor * (X[0] * R001 + X[1] * R101 + X[2] * R011 + X[3] * R002);
    Y[4] += prefactor * (X[0] * R200 + X[1] * R300 + X[2] * R210 + X[3] * R201);
    Y[5] += prefactor * (X[0] * R110 + X[1] * R210 + X[2] * R120 + X[3] * R111);
    Y[6] += prefactor * (X[0] * R101 + X[1] * R201 + X[2] * R111 + X[3] * R102);
    Y[7] += prefactor * (X[0] * R020 + X[1] * R120 + X[2] * R030 + X[3] * R021);
    Y[8] += prefactor * (X[0] * R011 + X[1] * R111 + X[2] * R021 + X[3] * R012);
    Y[9] += prefactor * (X[0] * R002 + X[1] * R102 + X[2] * R012 + X[3] * R003);
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_1_2(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    const T neg_2a_3 = neg_2a_2 * neg_2a;
    T Fm[4];
    boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2], R000_m3 = neg_2a_3 * Fm[3];
    const T R100 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R010 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R001 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R200 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R110 = PCy * R100_m1, R110_m1 = PCy * R100_m2;
    const T R101 = PCz * R100_m1, R101_m1 = PCz * R100_m2;
    const T R020 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R011 = PCz * R010_m1, R011_m1 = PCz * R010_m2;
    const T R002 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R300 = PCx * R200_m1 + T(2) * R100_m1;
    const T R210 = PCy * R200_m1;
    const T R201 = PCz * R200_m1;
    const T R120 = PCy * R110_m1 + R100_m1;
    const T R111 = PCz * R110_m1;
    const T R102 = PCz * R101_m1 + R100_m1;
    const T R030 = PCy * R020_m1 + T(2) * R010_m1;
    const T R021 = PCz * R020_m1;
    const T R012 = PCz * R011_m1 + R010_m1;
    const T R003 = PCz * R002_m1 + T(2) * R001_m1;

    // U[h_ab] += pf * sum_h_c(R_pq[h_ab, h_c] * T[h_c])
    U[0] += prefactor * (R000 * T_vec[0] + R100 * T_vec[1] + R010 * T_vec[2] + R001 * T_vec[3] +
                         R200 * T_vec[4] + R110 * T_vec[5] + R101 * T_vec[6] +
                         R020 * T_vec[7] + R011 * T_vec[8] + R002 * T_vec[9]);
    U[1] += prefactor * (R100 * T_vec[0] + R200 * T_vec[1] + R110 * T_vec[2] + R101 * T_vec[3] +
                         R300 * T_vec[4] + R210 * T_vec[5] + R201 * T_vec[6] +
                         R120 * T_vec[7] + R111 * T_vec[8] + R102 * T_vec[9]);
    U[2] += prefactor * (R010 * T_vec[0] + R110 * T_vec[1] + R020 * T_vec[2] + R011 * T_vec[3] +
                         R210 * T_vec[4] + R120 * T_vec[5] + R111 * T_vec[6] +
                         R030 * T_vec[7] + R021 * T_vec[8] + R012 * T_vec[9]);
    U[3] += prefactor * (R001 * T_vec[0] + R101 * T_vec[1] + R011 * T_vec[2] + R002 * T_vec[3] +
                         R201 * T_vec[4] + R111 * T_vec[5] + R102 * T_vec[6] +
                         R021 * T_vec[7] + R012 * T_vec[8] + R003 * T_vec[9]);
}

// ----------------------------------------------------------------------------
// L_ab=2, lc=1: (d|p) - L_total=3, nherm_ab=10, nherm_c=4
// R_pq[h_ab, h_c] where h_ab in 0-9, h_c in 0-3
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_forward_2_1(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                        T prefactor, const T* X, T* Y) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    const T neg_2a_3 = neg_2a_2 * neg_2a;
    T Fm[4];
    boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2], R000_m3 = neg_2a_3 * Fm[3];
    const T R100 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R010 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R001 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R200 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R110 = PCy * R100_m1, R110_m1 = PCy * R100_m2;
    const T R101 = PCz * R100_m1, R101_m1 = PCz * R100_m2;
    const T R020 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R011 = PCz * R010_m1, R011_m1 = PCz * R010_m2;
    const T R002 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R300 = PCx * R200_m1 + T(2) * R100_m1;
    const T R210 = PCy * R200_m1;
    const T R201 = PCz * R200_m1;
    const T R120 = PCy * R110_m1 + R100_m1;
    const T R111 = PCz * R110_m1;
    const T R102 = PCz * R101_m1 + R100_m1;
    const T R030 = PCy * R020_m1 + T(2) * R010_m1;
    const T R021 = PCz * R020_m1;
    const T R012 = PCz * R011_m1 + R010_m1;
    const T R003 = PCz * R002_m1 + T(2) * R001_m1;

    // Y[h_c] += pf * sum_h_ab(X[h_ab] * R_pq[h_ab, h_c])
    // h_ab: 0-9, h_c: 0-3
    Y[0] += prefactor * (X[0] * R000 + X[1] * R100 + X[2] * R010 + X[3] * R001 +
                         X[4] * R200 + X[5] * R110 + X[6] * R101 +
                         X[7] * R020 + X[8] * R011 + X[9] * R002);
    Y[1] += prefactor * (X[0] * R100 + X[1] * R200 + X[2] * R110 + X[3] * R101 +
                         X[4] * R300 + X[5] * R210 + X[6] * R201 +
                         X[7] * R120 + X[8] * R111 + X[9] * R102);
    Y[2] += prefactor * (X[0] * R010 + X[1] * R110 + X[2] * R020 + X[3] * R011 +
                         X[4] * R210 + X[5] * R120 + X[6] * R111 +
                         X[7] * R030 + X[8] * R021 + X[9] * R012);
    Y[3] += prefactor * (X[0] * R001 + X[1] * R101 + X[2] * R011 + X[3] * R002 +
                         X[4] * R201 + X[5] * R111 + X[6] * R102 +
                         X[7] * R021 + X[8] * R012 + X[9] * R003);
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void fused_backward_2_1(const T* boys_table, T alpha, T PCx, T PCy, T PCz,
                         T prefactor, const T* T_vec, T* U) {
    const T Tp = alpha * (PCx * PCx + PCy * PCy + PCz * PCz);
    const T neg_2a = T(-2) * alpha;
    const T neg_2a_2 = neg_2a * neg_2a;
    const T neg_2a_3 = neg_2a_2 * neg_2a;
    T Fm[4];
    boys_evaluate<T, 4, BoysParams>(boys_table, Tp, 0, Fm);

    const T R000 = Fm[0], R000_m1 = neg_2a * Fm[1], R000_m2 = neg_2a_2 * Fm[2], R000_m3 = neg_2a_3 * Fm[3];
    const T R100 = PCx * R000_m1, R100_m1 = PCx * R000_m2, R100_m2 = PCx * R000_m3;
    const T R010 = PCy * R000_m1, R010_m1 = PCy * R000_m2, R010_m2 = PCy * R000_m3;
    const T R001 = PCz * R000_m1, R001_m1 = PCz * R000_m2, R001_m2 = PCz * R000_m3;
    const T R200 = PCx * R100_m1 + R000_m1, R200_m1 = PCx * R100_m2 + R000_m2;
    const T R110 = PCy * R100_m1, R110_m1 = PCy * R100_m2;
    const T R101 = PCz * R100_m1, R101_m1 = PCz * R100_m2;
    const T R020 = PCy * R010_m1 + R000_m1, R020_m1 = PCy * R010_m2 + R000_m2;
    const T R011 = PCz * R010_m1, R011_m1 = PCz * R010_m2;
    const T R002 = PCz * R001_m1 + R000_m1, R002_m1 = PCz * R001_m2 + R000_m2;
    const T R300 = PCx * R200_m1 + T(2) * R100_m1;
    const T R210 = PCy * R200_m1;
    const T R201 = PCz * R200_m1;
    const T R120 = PCy * R110_m1 + R100_m1;
    const T R111 = PCz * R110_m1;
    const T R102 = PCz * R101_m1 + R100_m1;
    const T R030 = PCy * R020_m1 + T(2) * R010_m1;
    const T R021 = PCz * R020_m1;
    const T R012 = PCz * R011_m1 + R010_m1;
    const T R003 = PCz * R002_m1 + T(2) * R001_m1;

    // U[h_ab] += pf * sum_h_c(R_pq[h_ab, h_c] * T[h_c])
    U[0] += prefactor * (R000 * T_vec[0] + R100 * T_vec[1] + R010 * T_vec[2] + R001 * T_vec[3]);
    U[1] += prefactor * (R100 * T_vec[0] + R200 * T_vec[1] + R110 * T_vec[2] + R101 * T_vec[3]);
    U[2] += prefactor * (R010 * T_vec[0] + R110 * T_vec[1] + R020 * T_vec[2] + R011 * T_vec[3]);
    U[3] += prefactor * (R001 * T_vec[0] + R101 * T_vec[1] + R011 * T_vec[2] + R002 * T_vec[3]);
    U[4] += prefactor * (R200 * T_vec[0] + R300 * T_vec[1] + R210 * T_vec[2] + R201 * T_vec[3]);
    U[5] += prefactor * (R110 * T_vec[0] + R210 * T_vec[1] + R120 * T_vec[2] + R111 * T_vec[3]);
    U[6] += prefactor * (R101 * T_vec[0] + R201 * T_vec[1] + R111 * T_vec[2] + R102 * T_vec[3]);
    U[7] += prefactor * (R020 * T_vec[0] + R120 * T_vec[1] + R030 * T_vec[2] + R021 * T_vec[3]);
    U[8] += prefactor * (R011 * T_vec[0] + R111 * T_vec[1] + R021 * T_vec[2] + R012 * T_vec[3]);
    U[9] += prefactor * (R002 * T_vec[0] + R102 * T_vec[1] + R012 * T_vec[2] + R003 * T_vec[3]);
}

// ----------------------------------------------------------------------------
// Dispatch functions for fused kernels
// Returns true if handled by fused kernel, false if fallback needed
// ----------------------------------------------------------------------------
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
bool fused_forward_dispatch(const T* boys_table, int L_ab, int lc,
                             T alpha, T PCx, T PCy, T PCz,
                             T prefactor, const T* X, T* Y) {
    if (L_ab == 0) {
        if (lc == 0) { fused_forward_0_0<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
        if (lc == 1) { fused_forward_0_1<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
        if (lc == 2) { fused_forward_0_2<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
        if (lc == 3) { fused_forward_0_3<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
    } else if (L_ab == 1) {
        if (lc == 0) { fused_forward_1_0<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
        if (lc == 1) { fused_forward_1_1<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
        if (lc == 2) { fused_forward_1_2<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
    } else if (L_ab == 2) {
        if (lc == 0) { fused_forward_2_0<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
        if (lc == 1) { fused_forward_2_1<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
    } else if (L_ab == 3) {
        if (lc == 0) { fused_forward_3_0<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, X, Y); return true; }
    }
    return false;
}

template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
bool fused_backward_dispatch(const T* boys_table, int L_ab, int lc,
                              T alpha, T PCx, T PCy, T PCz,
                              T prefactor, const T* T_vec, T* U) {
    if (L_ab == 0) {
        if (lc == 0) { fused_backward_0_0<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
        if (lc == 1) { fused_backward_0_1<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
        if (lc == 2) { fused_backward_0_2<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
        if (lc == 3) { fused_backward_0_3<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
    } else if (L_ab == 1) {
        if (lc == 0) { fused_backward_1_0<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
        if (lc == 1) { fused_backward_1_1<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
        if (lc == 2) { fused_backward_1_2<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
    } else if (L_ab == 2) {
        if (lc == 0) { fused_backward_2_0<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
        if (lc == 1) { fused_backward_2_1<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
    } else if (L_ab == 3) {
        if (lc == 0) { fused_backward_3_0<T, BoysParams>(boys_table, alpha, PCx, PCy, PCz, prefactor, T_vec, U); return true; }
    }
    return false;
}

// ============================================================================
// Runtime dispatch for specialized R-integral kernels
// ============================================================================

/// Dispatch to specialized kernel based on runtime L value
/// Falls back to generic compute_r_ints_dynamic for L > 6
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_3c_dispatch(const T* boys_table, int L, T p,
                                 T PCx, T PCy, T PCz, T* R) {
    switch (L) {
        case 0: compute_r_ints_3c_L0<T, BoysParams>(boys_table, p, PCx, PCy, PCz, R); break;
        case 1: compute_r_ints_3c_L1<T, BoysParams>(boys_table, p, PCx, PCy, PCz, R); break;
        case 2: compute_r_ints_3c_L2<T, BoysParams>(boys_table, p, PCx, PCy, PCz, R); break;
        case 3: compute_r_ints_3c_L3<T, BoysParams>(boys_table, p, PCx, PCy, PCz, R); break;
        case 4: compute_r_ints_3c_L4<T, BoysParams>(boys_table, p, PCx, PCy, PCz, R); break;
        case 5: compute_r_ints_3c_L5<T, BoysParams>(boys_table, p, PCx, PCy, PCz, R); break;
        case 6: compute_r_ints_3c_L6<T, BoysParams>(boys_table, p, PCx, PCy, PCz, R); break;
        default: {
            // Fallback to generic implementation
            RIntsDynamic<T> R_dyn;
            compute_r_ints_dynamic<T, BoysParams>(boys_table, L, p, PCx, PCy, PCz, R_dyn);
            const int nherm = nhermsum(L);
            for (int i = 0; i < nherm; ++i) {
                R[i] = R_dyn.data[i];
            }
            break;
        }
    }
}

/// Version that writes directly to RIntsDynamic
template <typename T, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED OCC_GPU_INLINE
void compute_r_ints_3c_dispatch(const T* boys_table, int L, T p,
                                 T PCx, T PCy, T PCz, RIntsDynamic<T>& R) {
    R.set_L(L);
    compute_r_ints_3c_dispatch<T, BoysParams>(boys_table, L, p, PCx, PCy, PCz, R.data);
}

} // namespace occ::ints
