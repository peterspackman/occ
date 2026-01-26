#include <occ/mults/sfunctions.h>
#include <cmath>

namespace occ::mults::kernels {

// Mathematical constants
constexpr double rt3 = 1.7320508075688772935;
constexpr double rt5 = 2.2360679774997896964;
constexpr double rt6 = 2.4494897427831780982;
constexpr double rt10 = 3.1622776601683793320;
constexpr double rt15 = 3.8729833462074168852;
constexpr double rt35 = 5.9160797830996161426;
constexpr double rt70 = 8.3666002653407554798;

// ============================================================================
// QUADRUPOLE-QUADRUPOLE KERNELS (Orient cases 101-125)
// Quadrupole @ A (uses rax, ray, raz), Quadrupole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Quadrupole-20 × Quadrupole-20 kernel
 * Orient case 101
 */
void quadrupole_20_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 8.0 * sf.raz() * sf.raz() * sf.rbz() * sf.rbz() -
                5.0 / 8.0 * sf.raz() * sf.raz() -
                5.0 / 8.0 * sf.rbz() * sf.rbz() +
                5.0 / 2.0 * sf.raz() * sf.rbz() * sf.czz() +
                sf.czz() * sf.czz() / 4.0 +
                1.0 / 8.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = 35.0 / 4.0 * sf.rbz() * sf.rbz() * sf.raz() -
                       5.0 / 4.0 * sf.raz() +
                       5.0 / 2.0 * sf.rbz() * sf.czz(); // d/d(raz)
        result.s1[5] = 35.0 / 4.0 * sf.raz() * sf.raz() * sf.rbz() -
                       5.0 / 4.0 * sf.rbz() +
                       5.0 / 2.0 * sf.raz() * sf.czz(); // d/d(rbz)
        result.s1[14] = 5.0 / 2.0 * sf.raz() * sf.rbz() +
                        sf.czz() / 2.0; // d/d(czz)

        if (level >= 2) {
            // Second derivatives
            result.s2[5] = 35.0 / 4.0 * sf.rbz() * sf.rbz() - 5.0 / 4.0;  // d²/d(raz)²
            result.s2[17] = 35.0 / 2.0 * sf.raz() * sf.rbz() + 5.0 / 2.0 * sf.czz(); // d²/d(raz)d(rbz)
            result.s2[20] = 35.0 / 4.0 * sf.raz() * sf.raz() - 5.0 / 4.0; // d²/d(rbz)²
            result.s2[107] = 5.0 / 2.0 * sf.rbz(); // d²/d(raz)d(czz)
            result.s2[110] = 5.0 / 2.0 * sf.raz(); // d²/d(rbz)d(czz)
            result.s2[119] = 1.0 / 2.0; // d²/d(czz)²
        }
    }
}

/**
 * Quadrupole-20 × Quadrupole-21c kernel
 * Orient case 102
 */
void quadrupole_20_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = rt3 *
                (35.0 * sf.raz() * sf.raz() * sf.rbx() * sf.rbz() -
                 5.0 * sf.rbx() * sf.rbz() +
                 10.0 * sf.raz() * sf.rbx() * sf.czz() +
                 10.0 * sf.raz() * sf.rbz() * sf.czx() +
                 2.0 * sf.czx() * sf.czz()) /
                12.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt3 * (70.0 * sf.rbx() * sf.rbz() * sf.raz() +
                              10.0 * sf.rbx() * sf.czz() +
                              10.0 * sf.rbz() * sf.czx()) / 12.0; // d/d(raz)
        result.s1[3] = rt3 * (35.0 * sf.raz() * sf.raz() * sf.rbz() -
                              5.0 * sf.rbz() +
                              10.0 * sf.raz() * sf.czz()) / 12.0; // d/d(rbx)
        result.s1[5] = rt3 * (35.0 * sf.raz() * sf.raz() * sf.rbx() -
                              5.0 * sf.rbx() +
                              10.0 * sf.raz() * sf.czx()) / 12.0; // d/d(rbz)
        result.s1[12] = rt3 * (10.0 * sf.raz() * sf.rbz() + 2.0 * sf.czz()) / 12.0; // d/d(czx)
        result.s1[14] = rt3 * (10.0 * sf.raz() * sf.rbx() + 2.0 * sf.czx()) / 12.0; // d/d(czz)

        if (level >= 2) {
            // Second derivatives
            result.s2[5] = 35.0 / 6.0 * rt3 * sf.rbx() * sf.rbz();   // d²/d(raz)²
            result.s2[8] = rt3 * (70.0 * sf.raz() * sf.rbz() + 10.0 * sf.czz()) / 12.0; // d²/d(raz)d(rbx)
            result.s2[17] = rt3 * (70.0 * sf.raz() * sf.rbx() + 10.0 * sf.czx()) / 12.0; // d²/d(raz)d(rbz)
            result.s2[18] = rt3 * (35.0 * sf.raz() * sf.raz() - 5.0) / 12.0; // d²/d(rbx)d(rbz)
            result.s2[38] = 5.0 / 6.0 * rt3 * sf.rbz(); // d²/d(raz)d(czx)
            result.s2[41] = 5.0 / 6.0 * rt3 * sf.raz(); // d²/d(rbx)d(czx)
            result.s2[107] = 5.0 / 6.0 * rt3 * sf.rbx(); // d²/d(raz)d(czz)
            result.s2[108] = 5.0 / 6.0 * rt3 * sf.raz(); // d²/d(rbz)d(czx)
            result.s2[113] = rt3 / 6.0; // d²/d(czx)d(czz)
        }
    }
}

/**
 * Quadrupole-20 × Quadrupole-21s kernel
 * Orient case 103
 * Formula: S0 = rt3*(35*raz²*rby*rbz - 5*rby*rbz + 10*rbz*rby*czz + 10*rbz*raz*czy + 2*czy*czz)/12
 */
void quadrupole_20_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = rt3 *
                (35.0 * sf.raz() * sf.raz() * sf.rby() * sf.rbz() -
                 5.0 * sf.rby() * sf.rbz() +
                 10.0 * sf.raz() * sf.rby() * sf.czz() +
                 10.0 * sf.raz() * sf.rbz() * sf.czy() +
                 2.0 * sf.czy() * sf.czz()) /
                12.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt3 * (70.0 * sf.rby() * sf.rbz() * sf.raz() +
                              10.0 * sf.rby() * sf.czz() +
                              10.0 * sf.rbz() * sf.czy()) / 12.0; // d/d(raz)
        result.s1[4] = rt3 * (35.0 * sf.raz() * sf.raz() * sf.rbz() -
                              5.0 * sf.rbz() +
                              10.0 * sf.raz() * sf.czz()) / 12.0; // d/d(rby)
        result.s1[5] = rt3 * (35.0 * sf.raz() * sf.raz() * sf.rby() -
                              5.0 * sf.rby() +
                              10.0 * sf.raz() * sf.czy()) / 12.0; // d/d(rbz)
        result.s1[13] = rt3 * (10.0 * sf.raz() * sf.rbz() + 2.0 * sf.czz()) / 12.0; // d/d(czy)
        result.s1[14] = rt3 * (10.0 * sf.raz() * sf.rby() + 2.0 * sf.czy()) / 12.0; // d/d(czz)

        if (level >= 2) {
            // Second derivatives
            result.s2[5] = 35.0 / 6.0 * rt3 * sf.rby() * sf.rbz();   // d²/d(raz)²
            result.s2[12] = rt3 * (70.0 * sf.raz() * sf.rbz() + 10.0 * sf.czz()) / 12.0; // d²/d(raz)d(rby)
            result.s2[17] = rt3 * (70.0 * sf.raz() * sf.rby() + 10.0 * sf.czy()) / 12.0; // d²/d(raz)d(rbz)
            result.s2[19] = rt3 * (35.0 * sf.raz() * sf.raz() - 5.0) / 12.0; // d²/d(rby)d(rbz)
            result.s2[68] = 5.0 / 6.0 * rt3 * sf.rbz(); // d²/d(raz)d(czy)
            result.s2[71] = 5.0 / 6.0 * rt3 * sf.raz(); // d²/d(rby)d(czy)
            result.s2[107] = 5.0 / 6.0 * rt3 * sf.rby(); // d²/d(raz)d(czz)
            result.s2[109] = 5.0 / 6.0 * rt3 * sf.raz(); // d²/d(rbz)d(czy)
            result.s2[110] = 5.0 / 6.0 * rt3 * sf.raz(); // d²/d(rby)d(czz)
            result.s2[116] = rt3 / 6.0; // d²/d(czy)d(czz)
        }
    }
}

/**
 * Quadrupole-20 × Quadrupole-22c kernel
 * Orient case 104
 * Formula: S0 = rt3*(35*rbz²*rax²-35*rbz²*ray²-5*rax²+5*ray²+20*rbz*rax*czx-20*rbz*ray*czy+2*czx²-2*czy²)/24
 * Note: Orient's (rbz,rax,ray) correspond to OCC's (raz,rbx,rby)
 */
void quadrupole_20_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // Orient case 104: S0 = rt3*(35*rbz²*rax²-35*rbz²*ray²-5*rax²+5*ray²+20*rbz*rax*czx-20*rbz*ray*czy+2*czx²-2*czy²)/24
    // Note: Orient's rbz,rax,ray correspond to OCC's raz,rbx,rby (site B coordinates)
    result.s0 =
        rt3 *
        (35.0 * sf.raz() * sf.raz() * sf.rbx() * sf.rbx() -
         35.0 * sf.raz() * sf.raz() * sf.rby() * sf.rby() -
         5.0 * sf.rbx() * sf.rbx() +
         5.0 * sf.rby() * sf.rby() +
         20.0 * sf.raz() * sf.rbx() * sf.czx() -
         20.0 * sf.raz() * sf.rby() * sf.czy() +
         2.0 * sf.czx() * sf.czx() -
         2.0 * sf.czy() * sf.czy()) /
        24.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt3 * (70.0 * sf.raz() * sf.rbx() * sf.rbx() -
                              70.0 * sf.raz() * sf.rby() * sf.rby() +
                              20.0 * sf.rbx() * sf.czx() -
                              20.0 * sf.rby() * sf.czy()) / 24.0; // d/d(raz)
        result.s1[3] = rt3 * (70.0 * sf.raz() * sf.raz() * sf.rbx() -
                              10.0 * sf.rbx() +
                              20.0 * sf.raz() * sf.czx()) / 24.0; // d/d(rbx)
        result.s1[4] = rt3 * (-70.0 * sf.raz() * sf.raz() * sf.rby() +
                              10.0 * sf.rby() -
                              20.0 * sf.raz() * sf.czy()) / 24.0; // d/d(rby)
        result.s1[SFunctions::S1_CZX] = rt3 * (20.0 * sf.raz() * sf.rbx() +
                                                4.0 * sf.czx()) / 24.0; // d/d(czx)
        result.s1[SFunctions::S1_CZY] = rt3 * (-20.0 * sf.raz() * sf.rby() -
                                                4.0 * sf.czy()) / 24.0; // d/d(czy)

        if (level >= 2) {
            // Second derivatives
            result.s2[2] = rt3 * (70.0 * sf.rbx() * sf.rbx() -
                                  70.0 * sf.rby() * sf.rby()) / 24.0; // d²/d(raz)²
            result.s2[8] = rt3 * (140.0 * sf.raz() * sf.rbx() +
                                  20.0 * sf.czx()) / 24.0;  // d²/d(raz)d(rbx)
            result.s2[9] = rt3 * (70.0 * sf.raz() * sf.raz() - 10.0) / 24.0; // d²/d(rbx)²
            result.s2[12] = rt3 * (-140.0 * sf.raz() * sf.rby() -
                                   20.0 * sf.czy()) / 24.0; // d²/d(raz)d(rby)
            result.s2[14] = rt3 * (-70.0 * sf.raz() * sf.raz() + 10.0) / 24.0; // d²/d(rby)²
            result.s2[41] = rt3 * 4.0 / 24.0; // d²/d(czx)²
            result.s2[70] = rt3 * (-4.0) / 24.0; // d²/d(czy)²
            result.s2[105] = rt3 * 20.0 / 24.0 * sf.rbx(); // d²/d(raz)d(czx)
            result.s2[106] = rt3 * 20.0 / 24.0 * sf.raz(); // d²/d(rbx)d(czx)
            result.s2[109] = rt3 * (-20.0) / 24.0 * sf.rby(); // d²/d(raz)d(czy)
            result.s2[111] = rt3 * (-20.0) / 24.0 * sf.raz(); // d²/d(rby)d(czy)
        }
    }
}

/**
 * Quadrupole-20 × Quadrupole-22s kernel
 * Orient case 105
 * Formula: S0 = rt3*(35*raz²*rbx*rby - 5*rbx*rby + 10*raz*rbx*czy + 10*raz*rby*czx + 2*czx*czy)/12
 */
void quadrupole_20_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = rt3 *
                (35.0 * sf.raz() * sf.raz() * sf.rbx() * sf.rby() -
                 5.0 * sf.rbx() * sf.rby() +
                 10.0 * sf.raz() * sf.rbx() * sf.czy() +
                 10.0 * sf.raz() * sf.rby() * sf.czx() +
                 2.0 * sf.czx() * sf.czy()) /
                12.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt3 * 70.0 / 12.0 * sf.raz() * sf.rbx() * sf.rby(); // d/d(raz)
        result.s1[3] = rt3 * (35.0 * sf.raz() * sf.raz() * sf.rby() - 5.0 * sf.rby()) / 12.0; // d/d(rbx)
        result.s1[4] = rt3 * (35.0 * sf.raz() * sf.raz() * sf.rbx() - 5.0 * sf.rbx()) / 12.0; // d/d(rby)
        result.s1[SFunctions::S1_CZX] = rt3 * (10.0 * sf.raz() * sf.rby() + 2.0 * sf.czy()) / 12.0; // d/d(czx)
        result.s1[SFunctions::S1_CZY] = rt3 * (10.0 * sf.raz() * sf.rbx() + 2.0 * sf.czx()) / 12.0; // d/d(czy)

        if (level >= 2) {
            // Second derivatives
            result.s2[2] = rt3 * 70.0 / 12.0 * sf.rbx() * sf.rby(); // d²/d(raz)²
            result.s2[8] = rt3 * 70.0 / 12.0 * sf.raz() * sf.rby(); // d²/d(raz)d(rbx)
            result.s2[12] = rt3 * 70.0 / 12.0 * sf.raz() * sf.rbx(); // d²/d(raz)d(rby)
            result.s2[13] = rt3 * (35.0 * sf.raz() * sf.raz() - 5.0) / 12.0; // d²/d(rbx)d(rby)
        }
    }
}

/**
 * Quadrupole-21c × Quadrupole-20 kernel
 * Orient case 106 (symmetry)
 * Formula: S0 = rt3*(35*rbz²*rax*raz - 5*rax*raz + 10*rbz*rax*czz + 10*rbz*raz*cxz + 2*cxz*czz)/12
 */
void quadrupole_21c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = rt3 *
                (35.0 * sf.rbz() * sf.rbz() * sf.rax() * sf.raz() -
                 5.0 * sf.rax() * sf.raz() + 10.0 * sf.rbz() * sf.rax() * sf.czz() +
                 10.0 * sf.rbz() * sf.raz() * sf.cxz() + 2.0 * sf.cxz() * sf.czz()) /
                12.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt3 * (35.0 * sf.rbz() * sf.rbz() * sf.raz() - 5.0 * sf.raz() +
                              10.0 * sf.rbz() * sf.czz()) / 12.0; // d/d(rax)
        result.s1[2] = rt3 * (35.0 * sf.rbz() * sf.rbz() * sf.rax() - 5.0 * sf.rax() +
                              10.0 * sf.rbz() * sf.cxz()) / 12.0; // d/d(raz)
        result.s1[5] = rt3 * (70.0 * sf.rbz() * sf.rax() * sf.raz() + 10.0 * sf.rax() * sf.czz() +
                              10.0 * sf.raz() * sf.cxz()) / 12.0; // d/d(rbz)
        result.s1[8] = rt3 * (10.0 * sf.rbz() * sf.raz() + 2.0 * sf.czz()) / 12.0; // d/d(cxz)
        result.s1[14] = rt3 * (10.0 * sf.rbz() * sf.rax() + 2.0 * sf.cxz()) / 12.0; // d/d(czz)

        if (level >= 2) {
            // Second derivatives (Orient case 106)
            result.s2[2] = rt3 * (35.0 * sf.rbz() * sf.rbz() - 5.0) / 12.0;  // ∂²/∂raz∂rax
            result.s2[15] = rt3 * (70.0 * sf.rbz() * sf.raz() + 10.0 * sf.czz()) / 12.0; // ∂²/∂rbz∂rax
            result.s2[17] = rt3 * (70.0 * sf.rbz() * sf.rax() + 10.0 * sf.cxz()) / 12.0;  // ∂²/∂rbz∂raz
            result.s2[20] = rt3 * (70.0 * sf.rax() * sf.raz()) / 12.0; // ∂²/∂rbz²
        }
    }
}

/**
 * Quadrupole-21c × Quadrupole-21c kernel
 * Orient case 107
 * Formula: S0 = 35/6*rax*raz*rbx*rbz + 5/6*rax*rbx*czz + 5/6*rax*rbz*czx + 5/6*raz*rbx*cxz + 5/6*raz*rbz*cxx + cxx*czz/6 + cxz*czx/6
 */
void quadrupole_21c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rbx() * sf.rbz() +
                5.0 / 6.0 * sf.rax() * sf.rbx() * sf.czz() +
                5.0 / 6.0 * sf.rax() * sf.rbz() * sf.czx() +
                5.0 / 6.0 * sf.raz() * sf.rbx() * sf.cxz() +
                5.0 / 6.0 * sf.raz() * sf.rbz() * sf.cxx() +
                sf.cxx() * sf.czz() / 6.0 +
                sf.cxz() * sf.czx() / 6.0;

    if (level >= 1) {
        // First derivatives
        // d/d(rax) = 35/6*raz*rbx*rbz + 5/6*rbx*czz + 5/6*rbz*czx
        result.s1[0] = 35.0 / 6.0 * sf.raz() * sf.rbx() * sf.rbz() +
                       5.0 / 6.0 * sf.rbx() * sf.czz() +
                       5.0 / 6.0 * sf.rbz() * sf.czx();

        // d/d(raz) = 35/6*rax*rbx*rbz + 5/6*rbx*cxz + 5/6*rbz*cxx
        result.s1[2] = 35.0 / 6.0 * sf.rax() * sf.rbx() * sf.rbz() +
                       5.0 / 6.0 * sf.rbx() * sf.cxz() +
                       5.0 / 6.0 * sf.rbz() * sf.cxx();

        // d/d(rbx) = 35/6*rax*raz*rbz + 5/6*rax*czz + 5/6*raz*cxz
        result.s1[3] = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rbz() +
                       5.0 / 6.0 * sf.rax() * sf.czz() +
                       5.0 / 6.0 * sf.raz() * sf.cxz();

        // d/d(rbz) = 35/6*rax*raz*rbx + 5/6*rax*czx + 5/6*raz*cxx
        result.s1[5] = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rbx() +
                       5.0 / 6.0 * sf.rax() * sf.czx() +
                       5.0 / 6.0 * sf.raz() * sf.cxx();

        // d/d(cxx) = 5/6*raz*rbz + czz/6
        result.s1[6] = 5.0 / 6.0 * sf.raz() * sf.rbz() + sf.czz() / 6.0;

        // d/d(cxz) = 5/6*raz*rbx + czx/6
        result.s1[8] = 5.0 / 6.0 * sf.raz() * sf.rbx() + sf.czx() / 6.0;

        // d/d(czx) = 5/6*rax*rbz + cxz/6
        result.s1[12] = 5.0 / 6.0 * sf.rax() * sf.rbz() + sf.cxz() / 6.0;

        // d/d(czz) = 5/6*rax*rbx + cxx/6
        result.s1[14] = 5.0 / 6.0 * sf.rax() * sf.rbx() + sf.cxx() / 6.0;

        if (level >= 2) {
            // Second derivatives (Orient case 107)
            // ∂²/∂raz∂rax = 35/6*rbx*rbz
            result.s2[2] = 35.0 / 6.0 * sf.rbx() * sf.rbz();

            // ∂²/∂rbx∂rax = 35/6*raz*rbz + 5/6*czz
            result.s2[6] = 35.0 / 6.0 * sf.raz() * sf.rbz() + 5.0 / 6.0 * sf.czz();

            // ∂²/∂rbx∂raz = 35/6*rax*rbz + 5/6*cxz
            result.s2[8] = 35.0 / 6.0 * sf.rax() * sf.rbz() + 5.0 / 6.0 * sf.cxz();

            // ∂²/∂rbz∂rax = 35/6*raz*rbx + 5/6*czx
            result.s2[15] = 35.0 / 6.0 * sf.raz() * sf.rbx() + 5.0 / 6.0 * sf.czx();

            // ∂²/∂rbz∂raz = 35/6*rax*rbx + 5/6*cxx
            result.s2[17] = 35.0 / 6.0 * sf.rax() * sf.rbx() + 5.0 / 6.0 * sf.cxx();

            // ∂²/∂rbz∂rbx = 35/6*rax*raz
            result.s2[18] = 35.0 / 6.0 * sf.rax() * sf.raz();

            // ∂²/∂czz∂rax = 5/6*rbx
            result.s2[42] = 5.0 / 6.0 * sf.rbx();

            // ∂²/∂czx∂rax = 5/6*rbz
            result.s2[45] = 5.0 / 6.0 * sf.rbz();

            // ∂²/∂cxx∂raz = 5/6*rbz
            result.s2[47] = 5.0 / 6.0 * sf.rbz();

            // ∂²/∂cxz∂raz = 5/6*rbx
            result.s2[50] = 5.0 / 6.0 * sf.rbx();

            // ∂²/∂czz∂rbx = 5/6*rax
            result.s2[59] = 5.0 / 6.0 * sf.rax();

            // ∂²/∂cxz∂rbx = 5/6*raz
            result.s2[62] = 5.0 / 6.0 * sf.raz();

            // ∂²/∂czx∂rbz = 5/6*rax
            result.s2[80] = 5.0 / 6.0 * sf.rax();

            // ∂²/∂cxx∂rbz = 5/6*raz
            result.s2[83] = 5.0 / 6.0 * sf.raz();

            // ∂²/∂czz∂cxx = 1/6
            result.s2[104] = 1.0 / 6.0;

            // ∂²/∂czx∂cxz = 1/6
            result.s2[117] = 1.0 / 6.0;
        }
    }
}

/**
 * Quadrupole-21c × Quadrupole-21s kernel
 * Orient case 108
 * Formula: S0 = 35/6*rax*raz*rby*rbz + 5/6*rax*rby*czz + 5/6*rax*rbz*czy + 5/6*raz*rby*cxz + 5/6*raz*rbz*cxy + cxy*czz/6 + cxz*czy/6
 */
void quadrupole_21c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rby() * sf.rbz() +
                5.0 / 6.0 * sf.rax() * sf.rby() * sf.czz() +
                5.0 / 6.0 * sf.rax() * sf.rbz() * sf.czy() +
                5.0 / 6.0 * sf.raz() * sf.rby() * sf.cxz() +
                5.0 / 6.0 * sf.raz() * sf.rbz() * sf.cxy() +
                1.0 / 6.0 * sf.cxy() * sf.czz() +
                1.0 / 6.0 * sf.cxz() * sf.czy();

    if (level >= 1) {
        // First derivatives
        result.s1[0] = 35.0 / 6.0 * sf.raz() * sf.rby() * sf.rbz() +
                       5.0 / 6.0 * sf.rby() * sf.czz() +
                       5.0 / 6.0 * sf.rbz() * sf.czy(); // d/d(rax)
        result.s1[2] = 35.0 / 6.0 * sf.rax() * sf.rby() * sf.rbz() +
                       5.0 / 6.0 * sf.rby() * sf.cxz() +
                       5.0 / 6.0 * sf.rbz() * sf.cxy(); // d/d(raz)
        result.s1[4] = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rbz() +
                       5.0 / 6.0 * sf.rax() * sf.czz() +
                       5.0 / 6.0 * sf.raz() * sf.cxz(); // d/d(rby)
        result.s1[5] = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rby() +
                       5.0 / 6.0 * sf.rax() * sf.czy() +
                       5.0 / 6.0 * sf.raz() * sf.cxy(); // d/d(rbz)
        result.s1[7] = 5.0 / 6.0 * sf.raz() * sf.rbz() + 1.0 / 6.0 * sf.czz(); // d/d(cxy)
        result.s1[8] = 5.0 / 6.0 * sf.raz() * sf.rby() + 1.0 / 6.0 * sf.czy(); // d/d(cxz)
        result.s1[13] = 5.0 / 6.0 * sf.rax() * sf.rbz() + 1.0 / 6.0 * sf.cxz(); // d/d(czy)
        result.s1[14] = 5.0 / 6.0 * sf.rax() * sf.rby() + 1.0 / 6.0 * sf.cxy(); // d/d(czz)

        if (level >= 2) {
            // Second derivatives (Orient case 108)
            result.s2[2] = 35.0 / 6.0 * sf.rby() * sf.rbz();   // ∂²/∂raz∂rax
            result.s2[10] = 35.0 / 6.0 * sf.raz() * sf.rbz() + 5.0 / 6.0 * sf.czz(); // ∂²/∂rby∂rax
            result.s2[11] = 5.0 / 6.0 * sf.czy(); // ∂²/∂rbz∂rax
            result.s2[12] = 35.0 / 6.0 * sf.rax() * sf.rbz() + 5.0 / 6.0 * sf.cxz();  // ∂²/∂rby∂raz
            result.s2[13] = 5.0 / 6.0 * sf.cxy(); // ∂²/∂rbz∂raz
            result.s2[15] = 35.0 / 6.0 * sf.raz() * sf.rby() + 5.0 / 6.0 * sf.czy();  // ∂²/∂rbz∂rax
            result.s2[17] = 35.0 / 6.0 * sf.rax() * sf.rby() + 5.0 / 6.0 * sf.cxy();  // ∂²/∂rbz∂raz
            result.s2[19] = 35.0 / 6.0 * sf.rax() * sf.raz() + 5.0 / 6.0 * sf.cxz();  // ∂²/∂rbz∂rby

            // c-term derivatives
            result.s2[68] = 1.0 / 6.0 * sf.czz(); // ∂²/∂czz∂cxy
            result.s2[74] = 1.0 / 6.0 * sf.czy(); // ∂²/∂czy∂cxz
            result.s2[79] = 5.0 / 6.0 * sf.rax() * sf.rby(); // ∂²/∂czz∂czz
            result.s2[86] = 1.0 / 6.0 * sf.cxz(); // ∂²/∂czy∂czz
        }
    }
}

/**
 * Quadrupole-21s × Quadrupole-20 kernel
 * Orient case 109 (by analogy)
 * Formula: S0 = rt3*(35*rbz²*ray*raz - 5*ray*raz + 10*rbz*ray)/12
 */
void quadrupole_21s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = rt3 *
                (35.0 * sf.rbz() * sf.rbz() * sf.ray() * sf.raz() -
                 5.0 * sf.ray() * sf.raz() + 10.0 * sf.rbz() * sf.ray()) /
                12.0;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt3 * (35.0 * sf.rbz() * sf.rbz() * sf.raz() - 5.0 * sf.raz() + 10.0 * sf.rbz()) / 12.0; // d/d(ray)
        result.s1[2] = rt3 * (35.0 * sf.rbz() * sf.rbz() * sf.ray() - 5.0 * sf.ray()) / 12.0; // d/d(raz)
        result.s1[5] = rt3 * (70.0 * sf.rbz() * sf.ray() * sf.raz() + 10.0 * sf.ray()) / 12.0; // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 111)
            result.s2[4] = rt3 * (35.0 * sf.rbz() * sf.rbz() - 5.0) / 12.0;  // ∂²/∂raz∂ray
            result.s2[16] = rt3 * (70.0 * sf.rbz() * sf.raz() + 10.0) / 12.0; // ∂²/∂rbz∂ray
            result.s2[17] = rt3 * 70.0 / 12.0 * sf.rbz() * sf.ray();  // ∂²/∂rbz∂raz
            result.s2[20] = rt3 * (70.0 * sf.ray() * sf.raz()) / 12.0; // ∂²/∂rbz²
        }
    }
}

/**
 * Quadrupole-21s × Quadrupole-21c kernel
 * Orient case (by symmetry)
 * Formula: S0 = 35/6*ray*raz*rbx*rbz + 5/6*ray*rbx
 */
void quadrupole_21s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rbx() * sf.rbz() +
                5.0 / 6.0 * sf.ray() * sf.rbx();

    if (level >= 1) {
        // First derivatives
        result.s1[1] = 35.0 / 6.0 * sf.raz() * sf.rbx() * sf.rbz() + 5.0 / 6.0 * sf.rbx(); // d/d(ray)
        result.s1[2] = 35.0 / 6.0 * sf.ray() * sf.rbx() * sf.rbz(); // d/d(raz)
        result.s1[3] = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rbz() + 5.0 / 6.0 * sf.ray(); // d/d(rbx)
        result.s1[5] = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rbx(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 112)
            result.s2[4] = 35.0 / 6.0 * sf.rbx() * sf.rbz();   // ∂²/∂raz∂ray
            result.s2[7] = 35.0 / 6.0 * sf.raz() * sf.rbz() + 5.0 / 6.0; // ∂²/∂rbx∂ray
            result.s2[8] = 35.0 / 6.0 * sf.ray() * sf.rbz();   // ∂²/∂rbx∂raz
            result.s2[16] = 35.0 / 6.0 * sf.raz() * sf.rbx();  // ∂²/∂rbz∂ray
            result.s2[17] = 35.0 / 6.0 * sf.ray() * sf.rbx();  // ∂²/∂rbz∂raz
            result.s2[18] = 35.0 / 6.0 * sf.ray() * sf.raz();  // ∂²/∂rbz∂rbx
        }
    }
}

/**
 * Quadrupole-21s × Quadrupole-21s kernel
 * Orient case (by analogy to case 107)
 * Formula: S0 = 35/6*ray*raz*rby*rbz + 5/6*ray*rby + 5/6*raz*rbz + 1/6
 */
void quadrupole_21s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rby() * sf.rbz() +
                5.0 / 6.0 * sf.ray() * sf.rby() + 5.0 / 6.0 * sf.raz() * sf.rbz() +
                1.0 / 6.0;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = 35.0 / 6.0 * sf.raz() * sf.rby() * sf.rbz() + 5.0 / 6.0 * sf.rby(); // d/d(ray)
        result.s1[2] = 35.0 / 6.0 * sf.ray() * sf.rby() * sf.rbz() + 5.0 / 6.0 * sf.rbz(); // d/d(raz)
        result.s1[4] = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rbz() + 5.0 / 6.0 * sf.ray(); // d/d(rby)
        result.s1[5] = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rby() + 5.0 / 6.0 * sf.raz(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 113)
            result.s2[4] = 35.0 / 6.0 * sf.rby() * sf.rbz();   // ∂²/∂raz∂ray
            result.s2[11] = 35.0 / 6.0 * sf.raz() * sf.rbz() + 5.0 / 6.0; // ∂²/∂rby∂ray
            result.s2[12] = 35.0 / 6.0 * sf.ray() * sf.rbz();  // ∂²/∂rby∂raz
            result.s2[16] = 35.0 / 6.0 * sf.raz() * sf.rby();  // ∂²/∂rbz∂ray
            result.s2[17] = 35.0 / 6.0 * sf.ray() * sf.rby() + 5.0 / 6.0;  // ∂²/∂rbz∂raz
            result.s2[19] = 35.0 / 6.0 * sf.ray() * sf.raz();  // ∂²/∂rbz∂rby
        }
    }
}

/**
 * Quadrupole-22c × Quadrupole-20 kernel
 * Orient case (by symmetry with case 104)
 * Formula: S0 = rt3*(35*rbz²*(rax²-ray²) - 5*(rax²-ray²))/24
 */
void quadrupole_22c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 =
        rt3 *
        (35.0 * sf.rbz() * sf.rbz() * (sf.rax() * sf.rax() - sf.ray() * sf.ray()) -
         5.0 * (sf.rax() * sf.rax() - sf.ray() * sf.ray())) /
        24.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt3 * (70.0 * sf.rbz() * sf.rbz() * sf.rax() - 10.0 * sf.rax()) / 24.0; // d/d(rax)
        result.s1[1] = rt3 * (-70.0 * sf.rbz() * sf.rbz() * sf.ray() + 10.0 * sf.ray()) / 24.0; // d/d(ray)
        result.s1[5] = rt3 * 70.0 / 24.0 * sf.rbz() * (sf.rax() * sf.rax() - sf.ray() * sf.ray()); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 116)
            result.s2[0] = rt3 * (70.0 * sf.rbz() * sf.rbz() - 10.0) / 24.0; // ∂²/∂rax²
            result.s2[3] = rt3 * (-70.0 * sf.rbz() * sf.rbz() + 10.0) / 24.0; // ∂²/∂ray²
            result.s2[15] = rt3 * 140.0 / 24.0 * sf.rbz() * sf.rax(); // ∂²/∂rbz∂rax
            result.s2[16] = rt3 * (-140.0) / 24.0 * sf.rbz() * sf.ray(); // ∂²/∂rbz∂ray
            result.s2[20] = rt3 * 70.0 / 24.0 * (sf.rax() * sf.rax() - sf.ray() * sf.ray()); // ∂²/∂rbz²
        }
    }
}

/**
 * Quadrupole-22c × Quadrupole-22c kernel
 * Orient case 119
 * Formula: S0 = 35/24*(rax²*rbx² - rax²*rby² - ray²*rbx² + ray²*rby²)
 *              + 5/6*(rax*rbx - ray*rby) + 1/6
 */
void quadrupole_22c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // Orient case 119: 22c × 22c
    result.s0 = 35.0 / 24.0 * sf.rax() * sf.rax() * sf.rbx() * sf.rbx()
              - 35.0 / 24.0 * sf.rax() * sf.rax() * sf.rby() * sf.rby()
              - 35.0 / 24.0 * sf.ray() * sf.ray() * sf.rbx() * sf.rbx()
              + 35.0 / 24.0 * sf.ray() * sf.ray() * sf.rby() * sf.rby()
              + 5.0 / 6.0 * sf.rax() * sf.rbx() * sf.cxx()
              - 5.0 / 6.0 * sf.rax() * sf.rby() * sf.cxy()
              - 5.0 / 6.0 * sf.ray() * sf.rbx() * sf.cyx()
              + 5.0 / 6.0 * sf.ray() * sf.rby() * sf.cyy()
              + sf.cxx() * sf.cxx() / 12.0
              - sf.cxy() * sf.cxy() / 12.0
              - sf.cyx() * sf.cyx() / 12.0
              + sf.cyy() * sf.cyy() / 12.0;

    if (level >= 1) {
        // First derivatives - Orient S1(1), S1(2), S1(4), S1(5), S1(7), S1(8), S1(10), S1(11)
        result.s1[0] = 35.0 / 12.0 * sf.rbx() * sf.rbx() * sf.rax() - 35.0 / 12.0 * sf.rby() * sf.rby() * sf.rax() + 5.0 / 6.0 * sf.rbx() * sf.cxx() - 5.0 / 6.0 * sf.rby() * sf.cxy(); // S1(1)
        result.s1[1] = -35.0 / 12.0 * sf.rbx() * sf.rbx() * sf.ray() + 35.0 / 12.0 * sf.rby() * sf.rby() * sf.ray() - 5.0 / 6.0 * sf.rbx() * sf.cyx() + 5.0 / 6.0 * sf.rby() * sf.cyy(); // S1(2)
        result.s1[3] = 35.0 / 12.0 * sf.rax() * sf.rax() * sf.rbx() - 35.0 / 12.0 * sf.ray() * sf.ray() * sf.rbx() + 5.0 / 6.0 * sf.rax() * sf.cxx() - 5.0 / 6.0 * sf.ray() * sf.cyx(); // S1(4)
        result.s1[4] = -35.0 / 12.0 * sf.rax() * sf.rax() * sf.rby() + 35.0 / 12.0 * sf.ray() * sf.ray() * sf.rby() - 5.0 / 6.0 * sf.rax() * sf.cxy() + 5.0 / 6.0 * sf.ray() * sf.cyy(); // S1(5)
        result.s1[6] = 5.0 / 6.0 * sf.rax() * sf.rbx() + sf.cxx() / 6.0; // S1(7) - d/d(cxx)
        result.s1[7] = -5.0 / 6.0 * sf.rax() * sf.rby() - sf.cxy() / 6.0; // S1(8) - d/d(cxy)
        result.s1[9] = -5.0 / 6.0 * sf.rbx() * sf.ray() - sf.cyx() / 6.0; // S1(10) - d/d(cyx)
        result.s1[10] = 5.0 / 6.0 * sf.ray() * sf.rby() + sf.cyy() / 6.0; // S1(11) - d/d(cyy)

        if (level >= 2) {
            // Second derivatives (Orient case 119)
            result.s2[0] = 35.0 / 12.0 * sf.rbx() * sf.rbx() - 35.0 / 12.0 * sf.rby() * sf.rby();   // S2(1): ∂²/∂rax²
            result.s2[2] = -35.0 / 12.0 * sf.rbx() * sf.rbx() + 35.0 / 12.0 * sf.rby() * sf.rby(); // S2(3): ∂²/∂ray²
            result.s2[6] = 35.0 / 6.0 * sf.rax() * sf.rbx() + 5.0 / 6.0 * sf.cxx();   // S2(7): ∂²/∂rbx∂rax
            result.s2[7] = -35.0 / 6.0 * sf.rbx() * sf.ray() - 5.0 / 6.0 * sf.cyx();  // S2(8): ∂²/∂rbx∂ray
            result.s2[9] = 35.0 / 12.0 * sf.rax() * sf.rax() - 35.0 / 12.0 * sf.ray() * sf.ray(); // S2(10): ∂²/∂rbx²
            result.s2[10] = -35.0 / 6.0 * sf.rax() * sf.rby() - 5.0 / 6.0 * sf.cxy();  // S2(11): ∂²/∂rby∂rax
            result.s2[11] = 35.0 / 6.0 * sf.ray() * sf.rby() + 5.0 / 6.0 * sf.cyy();  // S2(12): ∂²/∂rby∂ray
            result.s2[14] = -35.0 / 12.0 * sf.rax() * sf.rax() + 35.0 / 12.0 * sf.ray() * sf.ray();  // S2(15): ∂²/∂rby²
        }
    }
}

/**
 * Quadrupole-22s × Quadrupole-20 kernel
 * Orient case (by analogy)
 * Formula: S0 = rt3*(35*rbz²*rax*ray - 5*rax*ray)/12
 */
void quadrupole_22s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = rt3 *
                (35.0 * sf.rbz() * sf.rbz() * sf.rax() * sf.ray() -
                 5.0 * sf.rax() * sf.ray()) /
                12.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt3 * (35.0 * sf.rbz() * sf.rbz() * sf.ray() - 5.0 * sf.ray()) / 12.0; // d/d(rax)
        result.s1[1] = rt3 * (35.0 * sf.rbz() * sf.rbz() * sf.rax() - 5.0 * sf.rax()) / 12.0; // d/d(ray)
        result.s1[5] = rt3 * 70.0 / 12.0 * sf.rbz() * sf.rax() * sf.ray(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case similar to 116)
            result.s2[1] = rt3 * (35.0 * sf.rbz() * sf.rbz() - 5.0) / 12.0;  // ∂²/∂ray∂rax
            result.s2[15] = rt3 * 70.0 / 12.0 * sf.rbz() * sf.ray(); // ∂²/∂rbz∂rax
            result.s2[16] = rt3 * 70.0 / 12.0 * sf.rbz() * sf.rax(); // ∂²/∂rbz∂ray
            result.s2[20] = rt3 * 70.0 / 12.0 * sf.rax() * sf.ray(); // ∂²/∂rbz²
        }
    }
}

/**
 * Quadrupole-22s × Quadrupole-22s kernel
 * Orient case 125
 * Formula: S0 = 35/6*rax*ray*rbx*rby + 5/6*(rax*rbx + ray*rby) + 1/6
 */
void quadrupole_22s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 6.0 * sf.rax() * sf.ray() * sf.rbx() * sf.rby() +
                5.0 / 6.0 * (sf.rax() * sf.rbx() + sf.ray() * sf.rby()) + 1.0 / 6.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = 35.0 / 6.0 * sf.ray() * sf.rbx() * sf.rby() + 5.0 / 6.0 * sf.rbx(); // d/d(rax)
        result.s1[1] = 35.0 / 6.0 * sf.rax() * sf.rbx() * sf.rby() + 5.0 / 6.0 * sf.rby(); // d/d(ray)
        result.s1[3] = 35.0 / 6.0 * sf.rax() * sf.ray() * sf.rby() + 5.0 / 6.0 * sf.rax(); // d/d(rbx)
        result.s1[4] = 35.0 / 6.0 * sf.rax() * sf.ray() * sf.rbx() + 5.0 / 6.0 * sf.ray(); // d/d(rby)

        if (level >= 2) {
            // Second derivatives (Orient case 125)
            result.s2[1] = 35.0 / 6.0 * sf.rbx() * sf.rby();   // ∂²/∂ray∂rax
            result.s2[6] = 35.0 / 6.0 * sf.ray() * sf.rby() + 5.0 / 6.0; // ∂²/∂rbx∂rax
            result.s2[7] = 35.0 / 6.0 * sf.rax() * sf.rby();   // ∂²/∂rbx∂ray
            result.s2[10] = 35.0 / 6.0 * sf.ray() * sf.rbx();  // ∂²/∂rby∂rax
            result.s2[11] = 35.0 / 6.0 * sf.rax() * sf.rbx() + 5.0 / 6.0; // ∂²/∂rby∂ray
            result.s2[13] = 35.0 / 6.0 * sf.rax() * sf.ray();  // ∂²/∂rby∂rbx
        }
    }
}

/**
 * Quadrupole-21c × Quadrupole-22c kernel
 * Orient case 109
 * Formula (static): S0 = (35/12)*rax*raz*rbx^2 - (35/12)*rax*raz*rby^2
 *                      + (5/6)*rax*rbx*czx - (5/6)*rax*rby*czy
 *                      + (5/6)*raz*rbx*cxx - (5/6)*raz*rby*cxy
 *                      + cxx*czx/6 - cxy*czy/6
 */
void quadrupole_21c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 12.0 * sf.rax() * sf.raz() * sf.rbx() * sf.rbx()
              - 35.0 / 12.0 * sf.rax() * sf.raz() * sf.rby() * sf.rby()
              + 5.0 / 6.0 * sf.rax() * sf.rbx() * sf.czx()
              - 5.0 / 6.0 * sf.rax() * sf.rby() * sf.czy()
              + 5.0 / 6.0 * sf.raz() * sf.rbx() * sf.cxx()
              - 5.0 / 6.0 * sf.raz() * sf.rby() * sf.cxy()
              + sf.cxx() * sf.czx() / 6.0
              - sf.cxy() * sf.czy() / 6.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = 35.0 / 12.0 * sf.raz() * (sf.rbx() * sf.rbx() - sf.rby() * sf.rby()); // d/d(rax)
        result.s1[2] = 35.0 / 12.0 * sf.rax() * (sf.rbx() * sf.rbx() - sf.rby() * sf.rby()); // d/d(raz)
        result.s1[3] = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rbx(); // d/d(rbx)
        result.s1[4] = -35.0 / 6.0 * sf.rax() * sf.raz() * sf.rby(); // d/d(rby)
        result.s1[6] = 5.0 / 6.0 * sf.raz() * sf.rbx() + sf.czx() / 6.0; // d/d(cxx)
        result.s1[7] = -5.0 / 6.0 * sf.raz() * sf.rby() - sf.czy() / 6.0; // d/d(cxy)
        result.s1[12] = 5.0 / 6.0 * sf.rax() * sf.rbx() + sf.cxx() / 6.0; // d/d(czx)
        result.s1[13] = -5.0 / 6.0 * sf.rax() * sf.rby() - sf.cxy() / 6.0; // d/d(czy)

        if (level >= 2) {
            // Second derivatives
            result.s2[2] = 35.0 / 12.0 * (sf.rbx() * sf.rbx() - sf.rby() * sf.rby());   // ∂²/∂raz∂rax
            result.s2[6] = 35.0 / 6.0 * sf.raz() * sf.rbx();   // ∂²/∂rbx∂rax
            result.s2[8] = 35.0 / 6.0 * sf.rax() * sf.rbx(); // ∂²/∂rbx∂raz
            result.s2[9] = 35.0 / 6.0 * sf.rax() * sf.raz();   // ∂²/∂rbx²
            result.s2[10] = -35.0 / 6.0 * sf.raz() * sf.rby();  // ∂²/∂rby∂rax
            result.s2[12] = -35.0 / 6.0 * sf.rax() * sf.rby(); // ∂²/∂rby∂raz
            result.s2[14] = -35.0 / 6.0 * sf.rax() * sf.raz();  // ∂²/∂rby²
        }
    }
}

/**
 * Quadrupole-21c × Quadrupole-22s kernel
 * Orient case 108
 * Formula (static): S0 = 35/6*rax*raz*rby*rbz + 5/6*rax*rby*czz + 5/6*rax*rbz*czy
 *                      + 5/6*raz*rby*cxz + 5/6*raz*rbz*cxy + cxy*czz/6 + cxz*czy/6
 */
void quadrupole_21c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // Orient case 110: Q21c × Q22s
    result.s0 = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rbx() * sf.rby()
              + 5.0 / 6.0 * sf.rax() * sf.rbx() * sf.czy()
              + 5.0 / 6.0 * sf.rax() * sf.rby() * sf.czx()
              + 5.0 / 6.0 * sf.raz() * sf.rbx() * sf.cxy()
              + 5.0 / 6.0 * sf.raz() * sf.rby() * sf.cxx()
              + sf.cxx() * sf.czy() / 6.0
              + sf.cxy() * sf.czx() / 6.0;

    if (level >= 1) {
        // First derivatives from Orient case 110
        // S1(1) → s1[0]: d/d(rax)
        result.s1[0] = 35.0 / 6.0 * sf.rbx() * sf.rby() * sf.raz()
                     + 5.0 / 6.0 * sf.rbx() * sf.czy()
                     + 5.0 / 6.0 * sf.rby() * sf.czx();

        // S1(3) → s1[2]: d/d(raz)
        result.s1[2] = 35.0 / 6.0 * sf.rbx() * sf.rby() * sf.rax()
                     + 5.0 / 6.0 * sf.rbx() * sf.cxy()
                     + 5.0 / 6.0 * sf.rby() * sf.cxx();

        // S1(4) → s1[3]: d/d(rbx)
        result.s1[3] = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rby()
                     + 5.0 / 6.0 * sf.rax() * sf.czy()
                     + 5.0 / 6.0 * sf.raz() * sf.cxy();

        // S1(5) → s1[4]: d/d(rby)
        result.s1[4] = 35.0 / 6.0 * sf.rax() * sf.raz() * sf.rbx()
                     + 5.0 / 6.0 * sf.rax() * sf.czx()
                     + 5.0 / 6.0 * sf.raz() * sf.cxx();

        // d/d(cxx) → s1[6]
        result.s1[6] = 5.0 / 6.0 * sf.raz() * sf.rby()
                     + sf.czy() / 6.0;

        // d/d(cxy) → s1[7]
        result.s1[7] = 5.0 / 6.0 * sf.raz() * sf.rbx()
                     + sf.czx() / 6.0;

        // d/d(czx) → s1[12]
        result.s1[12] = 5.0 / 6.0 * sf.rax() * sf.rby()
                     + sf.cxy() / 6.0;

        // d/d(czy) → s1[13]
        result.s1[13] = 5.0 / 6.0 * sf.rax() * sf.rbx()
                      + sf.cxx() / 6.0;

        if (level >= 2) {
            // Second derivatives
            result.s2[2] = 35.0 / 6.0 * sf.rby() * sf.rbz();   // ∂²/∂raz∂rax
            result.s2[8] = 35.0 / 6.0 * sf.rax() * sf.rbz();   // ∂²/∂rby∂raz
            result.s2[9] = 35.0 / 6.0 * sf.rax() * sf.raz();   // ∂²/∂rby∂rbz
            result.s2[10] = 35.0 / 6.0 * sf.raz() * sf.rby(); // ∂²/∂rbz∂rax
            result.s2[12] = 35.0 / 6.0 * sf.rax() * sf.rby();  // ∂²/∂rbz∂raz
        }
    }
}

/**
 * Quadrupole-21s × Quadrupole-22c kernel
 * Orient case 114: 21s × 22c
 * Formula (static): S0 = 35/12*ray*raz*rbx^2 - 35/12*ray*raz*rby^2 + 5/6*ray*rbx*czx - 5/6*ray*rby*czy + 5/6*raz*rbx*cyx - 5/6*raz*rby*cyy + cyx*czx/6 - cyy*czy/6
 */
void quadrupole_21s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // Orient case 114: 21s × 22c
    result.s0 = 35.0 / 12.0 * sf.ray() * sf.raz() * sf.rbx() * sf.rbx()
              - 35.0 / 12.0 * sf.ray() * sf.raz() * sf.rby() * sf.rby()
              + 5.0 / 6.0 * sf.ray() * sf.rbx() * sf.czx()
              - 5.0 / 6.0 * sf.ray() * sf.rby() * sf.czy()
              + 5.0 / 6.0 * sf.raz() * sf.rbx() * sf.cyx()
              - 5.0 / 6.0 * sf.raz() * sf.rby() * sf.cyy()
              + sf.cyx() * sf.czx() / 6.0
              - sf.cyy() * sf.czy() / 6.0;

    if (level >= 1) {
        // First derivatives - Orient S1(2), S1(3), S1(4), S1(5), S1(8), S1(9), S1(11), S1(12)
        result.s1[1] = 35.0 / 12.0 * sf.rbx() * sf.rbx() * sf.raz() - 35.0 / 12.0 * sf.rby() * sf.rby() * sf.raz() + 5.0 / 6.0 * sf.rbx() * sf.czx() - 5.0 / 6.0 * sf.rby() * sf.czy(); // S1(2)
        result.s1[2] = 35.0 / 12.0 * sf.rbx() * sf.rbx() * sf.ray() - 35.0 / 12.0 * sf.rby() * sf.rby() * sf.ray() + 5.0 / 6.0 * sf.rbx() * sf.cyx() - 5.0 / 6.0 * sf.rby() * sf.cyy(); // S1(3)
        result.s1[3] = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rbx() + 5.0 / 6.0 * sf.ray() * sf.czx() + 5.0 / 6.0 * sf.raz() * sf.cyx(); // S1(4)
        result.s1[4] = -35.0 / 6.0 * sf.ray() * sf.raz() * sf.rby() - 5.0 / 6.0 * sf.ray() * sf.czy() - 5.0 / 6.0 * sf.raz() * sf.cyy(); // S1(5)
        result.s1[9] = 5.0 / 6.0 * sf.raz() * sf.rbx() + sf.czx() / 6.0; // S1(8)
        result.s1[12] = 5.0 / 6.0 * sf.rbx() * sf.ray() + sf.cyx() / 6.0; // S1(9)
        result.s1[10] = -5.0 / 6.0 * sf.raz() * sf.rby() - sf.czy() / 6.0; // S1(11)
        result.s1[13] = -5.0 / 6.0 * sf.ray() * sf.rby() - sf.cyy() / 6.0; // S1(12)

        if (level >= 2) {
            // Second derivatives
            result.s2[4] = 35.0 / 12.0 * sf.rbx() * sf.rbx() - 35.0 / 12.0 * sf.rby() * sf.rby();  // S2(5): ∂²/∂raz∂ray
            result.s2[7] = 35.0 / 6.0 * sf.raz() * sf.rbx() + 5.0 / 6.0 * sf.czx();  // S2(8): ∂²/∂rbx∂ray
            result.s2[8] = 35.0 / 6.0 * sf.rbx() * sf.ray() + 5.0 / 6.0 * sf.cyx(); // S2(9): ∂²/∂rbx∂raz
            result.s2[9] = 35.0 / 6.0 * sf.ray() * sf.raz();  // S2(10): ∂²/∂rbx²
            result.s2[11] = -35.0 / 6.0 * sf.raz() * sf.rby() - 5.0 / 6.0 * sf.czy();  // S2(12): ∂²/∂rby∂ray
            result.s2[12] = -35.0 / 6.0 * sf.ray() * sf.rby() - 5.0 / 6.0 * sf.cyy();  // S2(13): ∂²/∂rby∂raz
            result.s2[14] = -35.0 / 6.0 * sf.ray() * sf.raz();  // S2(15): ∂²/∂rby²
        }
    }
}

/**
 * Quadrupole-21s × Quadrupole-22s kernel
 * Orient case 115
 * Formula (static): S0 = 35/6*ray*raz*rbx*rby
 */
void quadrupole_21s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rbx() * sf.rby()
              + 5.0 / 6.0 * sf.ray() * sf.rbx() * sf.czy()
              + 5.0 / 6.0 * sf.ray() * sf.rby() * sf.czx()
              + 5.0 / 6.0 * sf.raz() * sf.rbx() * sf.cyy()
              + 5.0 / 6.0 * sf.raz() * sf.rby() * sf.cyx()
              + sf.cyx() * sf.czy() / 6.0
              + sf.cyy() * sf.czx() / 6.0;

    if (level >= 1) {
        // First derivatives - Orient S1(2), S1(3), S1(4), S1(5), S1(8), S1(9), S1(11), S1(12)
        result.s1[1] = 35.0 / 6.0 * sf.rbx() * sf.rby() * sf.raz() + 5.0 / 6.0 * sf.rbx() * sf.czy() + 5.0 / 6.0 * sf.rby() * sf.czx(); // S1(2)
        result.s1[2] = 35.0 / 6.0 * sf.rbx() * sf.rby() * sf.ray() + 5.0 / 6.0 * sf.rbx() * sf.cyy() + 5.0 / 6.0 * sf.rby() * sf.cyx(); // S1(3)
        result.s1[3] = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rby() + 5.0 / 6.0 * sf.ray() * sf.czy() + 5.0 / 6.0 * sf.raz() * sf.cyy(); // S1(4)
        result.s1[4] = 35.0 / 6.0 * sf.ray() * sf.raz() * sf.rbx() + 5.0 / 6.0 * sf.ray() * sf.czx() + 5.0 / 6.0 * sf.raz() * sf.cyx(); // S1(5)
        result.s1[9] = 5.0 / 6.0 * sf.raz() * sf.rby() + sf.czy() / 6.0; // S1(8)
        result.s1[12] = 5.0 / 6.0 * sf.ray() * sf.rby() + sf.cyy() / 6.0; // S1(9)
        result.s1[10] = 5.0 / 6.0 * sf.raz() * sf.rbx() + sf.czx() / 6.0; // S1(11)
        result.s1[13] = 5.0 / 6.0 * sf.rbx() * sf.ray() + sf.cyx() / 6.0; // S1(12)

        if (level >= 2) {
            // Second derivatives
            result.s2[4] = 35.0 / 6.0 * sf.rbx() * sf.rby();   // S2(5): ∂²/∂raz∂ray
            result.s2[7] = 35.0 / 6.0 * sf.raz() * sf.rby() + 5.0 / 6.0 * sf.czy();  // S2(8): ∂²/∂rbx∂ray
            result.s2[8] = 35.0 / 6.0 * sf.ray() * sf.rby() + 5.0 / 6.0 * sf.cyy(); // S2(9): ∂²/∂rbx∂raz
            result.s2[11] = 35.0 / 6.0 * sf.raz() * sf.rbx() + 5.0 / 6.0 * sf.czx(); // S2(12): ∂²/∂rby∂ray
            result.s2[12] = 35.0 / 6.0 * sf.rbx() * sf.ray() + 5.0 / 6.0 * sf.cyx();  // S2(13): ∂²/∂rby∂raz
            result.s2[13] = 35.0 / 6.0 * sf.ray() * sf.raz();  // S2(14): ∂²/∂rby∂rbx
        }
    }
}

/**
 * Quadrupole-22c × Quadrupole-21c kernel
 * Orient case 117: 22c × 21c
 * Formula (static): S0 = 35/12*rbx*rbz*rax^2 - 35/12*rbx*rbz*ray^2 + 5/6*rbx*rax*cxz - 5/6*rbx*ray*cyz + 5/6*rbz*rax*cxx - 5/6*rbz*ray*cyx + cxx*cxz/6 - cyx*cyz/6
 */
void quadrupole_22c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 12.0 * sf.rbx() * sf.rbz() * sf.rax() * sf.rax()
              - 35.0 / 12.0 * sf.rbx() * sf.rbz() * sf.ray() * sf.ray()
              + 5.0 / 6.0 * sf.rbx() * sf.rax() * sf.cxz()
              - 5.0 / 6.0 * sf.rbx() * sf.ray() * sf.cyz()
              + 5.0 / 6.0 * sf.rbz() * sf.rax() * sf.cxx()
              - 5.0 / 6.0 * sf.rbz() * sf.ray() * sf.cyx()
              + sf.cxx() * sf.cxz() / 6.0
              - sf.cyx() * sf.cyz() / 6.0;

    if (level >= 1) {
        // First derivatives - Orient S1(1), S1(2), S1(4), S1(6), S1(7), S1(8), S1(13), S1(14)
        result.s1[0] = 35.0 / 6.0 * sf.rbx() * sf.rbz() * sf.rax() + 5.0 / 6.0 * sf.rbx() * sf.cxz() + 5.0 / 6.0 * sf.rbz() * sf.cxx(); // S1(1)
        result.s1[1] = -35.0 / 6.0 * sf.rbx() * sf.rbz() * sf.ray() - 5.0 / 6.0 * sf.rbx() * sf.cyz() - 5.0 / 6.0 * sf.rbz() * sf.cyx(); // S1(2)
        result.s1[3] = 35.0 / 12.0 * sf.rax() * sf.rax() * sf.rbz() - 35.0 / 12.0 * sf.ray() * sf.ray() * sf.rbz() + 5.0 / 6.0 * sf.rax() * sf.cxz() - 5.0 / 6.0 * sf.ray() * sf.cyz(); // S1(4)
        result.s1[5] = 35.0 / 12.0 * sf.rax() * sf.rax() * sf.rbx() - 35.0 / 12.0 * sf.ray() * sf.ray() * sf.rbx() + 5.0 / 6.0 * sf.rax() * sf.cxx() - 5.0 / 6.0 * sf.ray() * sf.cyx(); // S1(6)
        result.s1[6] = 5.0 / 6.0 * sf.rbz() * sf.rax() + sf.cxz() / 6.0; // S1(7)
        result.s1[9] = -5.0 / 6.0 * sf.rbz() * sf.ray() - sf.cyz() / 6.0; // S1(8)
        result.s1[8] = 5.0 / 6.0 * sf.rax() * sf.rbx() + sf.cxx() / 6.0; // S1(13)
        result.s1[11] = -5.0 / 6.0 * sf.rbx() * sf.ray() - sf.cyx() / 6.0; // S1(14)

        if (level >= 2) {
            // Second derivatives
            result.s2[0] = 35.0 / 6.0 * sf.rbx() * sf.rbz();  // S2(1): ∂²/∂rax²
            result.s2[2] = -35.0 / 6.0 * sf.rbx() * sf.rbz();   // S2(3): ∂²/∂ray²
            result.s2[6] = 35.0 / 6.0 * sf.rbz() * sf.rax() + 5.0 / 6.0 * sf.cxz();  // S2(7): ∂²/∂rbx∂rax
            result.s2[7] = -35.0 / 6.0 * sf.rbz() * sf.ray() - 5.0 / 6.0 * sf.cyz();  // S2(8): ∂²/∂rbx∂ray
            result.s2[15] = 35.0 / 6.0 * sf.rax() * sf.rbx() + 5.0 / 6.0 * sf.cxx(); // S2(16): ∂²/∂rbz∂rax
            result.s2[16] = -35.0 / 6.0 * sf.rbx() * sf.ray() - 5.0 / 6.0 * sf.cyx(); // S2(17): ∂²/∂rbz∂ray
            result.s2[18] = 35.0 / 12.0 * sf.rax() * sf.rax() - 35.0 / 12.0 * sf.ray() * sf.ray();  // S2(19): ∂²/∂rbz∂rbx
        }
    }
}

/**
 * Quadrupole-22c × Quadrupole-21s kernel
 * Orient case 118: 22c × 21s
 * Formula (static): S0 = 35/12*rby*rbz*rax^2 - 35/12*rby*rbz*ray^2 + 5/6*rby*rax*cxz - 5/6*rby*ray*cyz + 5/6*rbz*rax*cxy - 5/6*rbz*ray*cyy + cxy*cxz/6 - cyy*cyz/6
 */
void quadrupole_22c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 12.0 * sf.rby() * sf.rbz() * sf.rax() * sf.rax()
              - 35.0 / 12.0 * sf.rby() * sf.rbz() * sf.ray() * sf.ray()
              + 5.0 / 6.0 * sf.rby() * sf.rax() * sf.cxz()
              - 5.0 / 6.0 * sf.rby() * sf.ray() * sf.cyz()
              + 5.0 / 6.0 * sf.rbz() * sf.rax() * sf.cxy()
              - 5.0 / 6.0 * sf.rbz() * sf.ray() * sf.cyy()
              + sf.cxy() * sf.cxz() / 6.0
              - sf.cyy() * sf.cyz() / 6.0;

    if (level >= 1) {
        // First derivatives - Orient S1(1), S1(2), S1(5), S1(6), S1(10), S1(11), S1(13), S1(14)
        result.s1[0] = 35.0 / 6.0 * sf.rby() * sf.rbz() * sf.rax() + 5.0 / 6.0 * sf.rby() * sf.cxz() + 5.0 / 6.0 * sf.rbz() * sf.cxy(); // S1(1)
        result.s1[1] = -35.0 / 6.0 * sf.rby() * sf.rbz() * sf.ray() - 5.0 / 6.0 * sf.rby() * sf.cyz() - 5.0 / 6.0 * sf.rbz() * sf.cyy(); // S1(2)
        result.s1[4] = 35.0 / 12.0 * sf.rax() * sf.rax() * sf.rbz() - 35.0 / 12.0 * sf.ray() * sf.ray() * sf.rbz() + 5.0 / 6.0 * sf.rax() * sf.cxz() - 5.0 / 6.0 * sf.ray() * sf.cyz(); // S1(5)
        result.s1[5] = 35.0 / 12.0 * sf.rax() * sf.rax() * sf.rby() - 35.0 / 12.0 * sf.ray() * sf.ray() * sf.rby() + 5.0 / 6.0 * sf.rax() * sf.cxy() - 5.0 / 6.0 * sf.ray() * sf.cyy(); // S1(6)
        result.s1[7] = 5.0 / 6.0 * sf.rbz() * sf.rax() + sf.cxz() / 6.0; // S1(10)
        result.s1[10] = -5.0 / 6.0 * sf.rbz() * sf.ray() - sf.cyz() / 6.0; // S1(11)
        result.s1[8] = 5.0 / 6.0 * sf.rax() * sf.rby() + sf.cxy() / 6.0; // S1(13)
        result.s1[11] = -5.0 / 6.0 * sf.ray() * sf.rby() - sf.cyy() / 6.0; // S1(14)

        if (level >= 2) {
            // Second derivatives
            result.s2[0] = 35.0 / 6.0 * sf.rby() * sf.rbz();  // S2(1): ∂²/∂rax²
            result.s2[2] = -35.0 / 6.0 * sf.rby() * sf.rbz();   // S2(3): ∂²/∂ray²
            result.s2[10] = 35.0 / 6.0 * sf.rbz() * sf.rax() + 5.0 / 6.0 * sf.cxz();  // S2(11): ∂²/∂rby∂rax
            result.s2[11] = -35.0 / 6.0 * sf.rbz() * sf.ray() - 5.0 / 6.0 * sf.cyz();  // S2(12): ∂²/∂rby∂ray
            result.s2[15] = 35.0 / 6.0 * sf.rax() * sf.rby() + 5.0 / 6.0 * sf.cxy(); // S2(16): ∂²/∂rbz∂rax
            result.s2[16] = -35.0 / 6.0 * sf.ray() * sf.rby() - 5.0 / 6.0 * sf.cyy(); // S2(17): ∂²/∂rbz∂ray
            result.s2[19] = 35.0 / 12.0 * sf.rax() * sf.rax() - 35.0 / 12.0 * sf.ray() * sf.ray();  // S2(20): ∂²/∂rbz∂rby
        }
    }
}

/**
 * Quadrupole-22c × Quadrupole-22s kernel
 * Orient case 120
 * Formula (static): S0 = 35/12*rax²*rbx*rby - 35/12*ray²*rbx*rby
 */
void quadrupole_22c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 12.0 * sf.rax() * sf.rax() * sf.rbx() * sf.rby()
              - 35.0 / 12.0 * sf.ray() * sf.ray() * sf.rbx() * sf.rby()
              + 5.0 / 6.0 * sf.rax() * sf.rbx() * sf.cxy()
              + 5.0 / 6.0 * sf.rax() * sf.rby() * sf.cxx()
              - 5.0 / 6.0 * sf.ray() * sf.rbx() * sf.cyy()
              - 5.0 / 6.0 * sf.ray() * sf.rby() * sf.cyx()
              + sf.cxx() * sf.cxy() / 6.0
              - sf.cyx() * sf.cyy() / 6.0;

    if (level >= 1) {
        // First derivatives - Orient S1(1), S1(2), S1(4), S1(5), S1(7), S1(8), S1(10), S1(11)
        result.s1[0] = 35.0 / 6.0 * sf.rbx() * sf.rby() * sf.rax() + 5.0 / 6.0 * sf.rbx() * sf.cxy() + 5.0 / 6.0 * sf.rby() * sf.cxx(); // S1(1)
        result.s1[1] = -35.0 / 6.0 * sf.rbx() * sf.rby() * sf.ray() - 5.0 / 6.0 * sf.rbx() * sf.cyy() - 5.0 / 6.0 * sf.rby() * sf.cyx(); // S1(2)
        result.s1[3] = 35.0 / 12.0 * sf.rax() * sf.rax() * sf.rby() - 35.0 / 12.0 * sf.ray() * sf.ray() * sf.rby() + 5.0 / 6.0 * sf.rax() * sf.cxy() - 5.0 / 6.0 * sf.ray() * sf.cyy(); // S1(4)
        result.s1[4] = 35.0 / 12.0 * sf.rax() * sf.rax() * sf.rbx() - 35.0 / 12.0 * sf.ray() * sf.ray() * sf.rbx() + 5.0 / 6.0 * sf.rax() * sf.cxx() - 5.0 / 6.0 * sf.ray() * sf.cyx(); // S1(5)
        result.s1[6] = 5.0 / 6.0 * sf.rax() * sf.rby() + sf.cxy() / 6.0; // S1(7)
        result.s1[9] = -5.0 / 6.0 * sf.ray() * sf.rby() - sf.cyy() / 6.0; // S1(8)
        result.s1[7] = 5.0 / 6.0 * sf.rax() * sf.rbx() + sf.cxx() / 6.0; // S1(10)
        result.s1[10] = -5.0 / 6.0 * sf.rbx() * sf.ray() - sf.cyx() / 6.0; // S1(11)

        if (level >= 2) {
            // Second derivatives
            result.s2[0] = 35.0 / 6.0 * sf.rbx() * sf.rby();   // S2(1): ∂²/∂rax²
            result.s2[2] = -35.0 / 6.0 * sf.rbx() * sf.rby();  // S2(3): ∂²/∂ray²
            result.s2[6] = 35.0 / 6.0 * sf.rax() * sf.rby() + 5.0 / 6.0 * sf.cxy(); // S2(7): ∂²/∂rbx∂rax
            result.s2[7] = -35.0 / 6.0 * sf.ray() * sf.rby() - 5.0 / 6.0 * sf.cyy(); // S2(8): ∂²/∂rbx∂ray
            result.s2[10] = 35.0 / 6.0 * sf.rax() * sf.rbx() + 5.0 / 6.0 * sf.cxx(); // S2(11): ∂²/∂rby∂rax
            result.s2[11] = -35.0 / 6.0 * sf.rbx() * sf.ray() - 5.0 / 6.0 * sf.cyx(); // S2(12): ∂²/∂rby∂ray
            result.s2[13] = 35.0 / 12.0 * sf.rax() * sf.rax() - 35.0 / 12.0 * sf.ray() * sf.ray(); // S2(14): ∂²/∂rby∂rbx
        }
    }
}

/**
 * Quadrupole-22s × Quadrupole-21c kernel
 * Orient case 122: 22s × 21c
 * Formula (static): S0 = 35/6*rbx*rbz*rax*ray + 5/6*rbx*rax*cyz + 5/6*rbx*ray*cxz + 5/6*rbz*rax*cyx + 5/6*rbz*ray*cxx + cxx*cyz/6 + cyx*cxz/6
 */
void quadrupole_22s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // Orient case 122: 22s × 21c
    result.s0 = 35.0 / 6.0 * sf.rbx() * sf.rbz() * sf.rax() * sf.ray()
              + 5.0 / 6.0 * sf.rbx() * sf.rax() * sf.cyz()
              + 5.0 / 6.0 * sf.rbx() * sf.ray() * sf.cxz()
              + 5.0 / 6.0 * sf.rbz() * sf.rax() * sf.cyx()
              + 5.0 / 6.0 * sf.rbz() * sf.ray() * sf.cxx()
              + sf.cxx() * sf.cyz() / 6.0
              + sf.cyx() * sf.cxz() / 6.0;

    if (level >= 1) {
        // First derivatives - Orient S1(1), S1(2), S1(4), S1(6), S1(7), S1(8), S1(13), S1(14)
        result.s1[0] = 35.0 / 6.0 * sf.rbx() * sf.rbz() * sf.ray() + 5.0 / 6.0 * sf.rbx() * sf.cyz() + 5.0 / 6.0 * sf.rbz() * sf.cyx(); // S1(1)
        result.s1[1] = 35.0 / 6.0 * sf.rbx() * sf.rbz() * sf.rax() + 5.0 / 6.0 * sf.rbx() * sf.cxz() + 5.0 / 6.0 * sf.rbz() * sf.cxx(); // S1(2)
        result.s1[3] = 35.0 / 6.0 * sf.rax() * sf.ray() * sf.rbz() + 5.0 / 6.0 * sf.rax() * sf.cyz() + 5.0 / 6.0 * sf.ray() * sf.cxz(); // S1(4)
        result.s1[5] = 35.0 / 6.0 * sf.rax() * sf.ray() * sf.rbx() + 5.0 / 6.0 * sf.rax() * sf.cyx() + 5.0 / 6.0 * sf.ray() * sf.cxx(); // S1(6)
        result.s1[6] = 5.0 / 6.0 * sf.rbz() * sf.ray() + sf.cyz() / 6.0; // S1(7)
        result.s1[9] = 5.0 / 6.0 * sf.rbz() * sf.rax() + sf.cxz() / 6.0; // S1(8)
        result.s1[8] = 5.0 / 6.0 * sf.rbx() * sf.ray() + sf.cyx() / 6.0; // S1(13)
        result.s1[11] = 5.0 / 6.0 * sf.rax() * sf.rbx() + sf.cxx() / 6.0; // S1(14)

        if (level >= 2) {
            // Second derivatives
            result.s2[1] = 35.0 / 6.0 * sf.rbx() * sf.rbz();   // S2(2): ∂²/∂ray∂rax
            result.s2[6] = 35.0 / 6.0 * sf.rbz() * sf.ray() + 5.0 / 6.0 * sf.cyz(); // S2(7): ∂²/∂rbx∂rax
            result.s2[7] = 35.0 / 6.0 * sf.rbz() * sf.rax() + 5.0 / 6.0 * sf.cxz(); // S2(8): ∂²/∂rbx∂ray
            result.s2[15] = 35.0 / 6.0 * sf.rbx() * sf.ray() + 5.0 / 6.0 * sf.cyx(); // S2(16): ∂²/∂rbz∂rax
            result.s2[16] = 35.0 / 6.0 * sf.rax() * sf.rbx() + 5.0 / 6.0 * sf.cxx(); // S2(17): ∂²/∂rbz∂ray
            result.s2[18] = 35.0 / 6.0 * sf.rax() * sf.ray(); // S2(19): ∂²/∂rbz∂rbx
        }
    }
}

/**
 * Quadrupole-22s × Quadrupole-21s kernel
 * Orient case 123: 22s × 21s
 * Formula (static): S0 = 35/6*rby*rbz*rax*ray + 5/6*rby*rax*cyz + 5/6*rby*ray*cxz + 5/6*rbz*rax*cyy + 5/6*rbz*ray*cxy + cxy*cyz/6 + cyy*cxz/6
 */
void quadrupole_22s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 6.0 * sf.rby() * sf.rbz() * sf.rax() * sf.ray()
              + 5.0 / 6.0 * sf.rby() * sf.rax() * sf.cyz()
              + 5.0 / 6.0 * sf.rby() * sf.ray() * sf.cxz()
              + 5.0 / 6.0 * sf.rbz() * sf.rax() * sf.cyy()
              + 5.0 / 6.0 * sf.rbz() * sf.ray() * sf.cxy()
              + sf.cxy() * sf.cyz() / 6.0
              + sf.cyy() * sf.cxz() / 6.0;

    if (level >= 1) {
        // First derivatives - Orient S1(1), S1(2), S1(5), S1(6), S1(10), S1(11), S1(13), S1(14)
        result.s1[0] = 35.0 / 6.0 * sf.rby() * sf.rbz() * sf.ray() + 5.0 / 6.0 * sf.rby() * sf.cyz() + 5.0 / 6.0 * sf.rbz() * sf.cyy(); // S1(1)
        result.s1[1] = 35.0 / 6.0 * sf.rby() * sf.rbz() * sf.rax() + 5.0 / 6.0 * sf.rby() * sf.cxz() + 5.0 / 6.0 * sf.rbz() * sf.cxy(); // S1(2)
        result.s1[4] = 35.0 / 6.0 * sf.rax() * sf.ray() * sf.rbz() + 5.0 / 6.0 * sf.rax() * sf.cyz() + 5.0 / 6.0 * sf.ray() * sf.cxz(); // S1(5)
        result.s1[5] = 35.0 / 6.0 * sf.rax() * sf.ray() * sf.rby() + 5.0 / 6.0 * sf.rax() * sf.cyy() + 5.0 / 6.0 * sf.ray() * sf.cxy(); // S1(6)
        result.s1[7] = 5.0 / 6.0 * sf.rbz() * sf.ray() + sf.cyz() / 6.0; // S1(10)
        result.s1[10] = 5.0 / 6.0 * sf.rbz() * sf.rax() + sf.cxz() / 6.0; // S1(11)
        result.s1[8] = 5.0 / 6.0 * sf.ray() * sf.rby() + sf.cyy() / 6.0; // S1(13)
        result.s1[11] = 5.0 / 6.0 * sf.rax() * sf.rby() + sf.cxy() / 6.0; // S1(14)

        if (level >= 2) {
            // Second derivatives
            result.s2[1] = 35.0 / 6.0 * sf.rby() * sf.rbz();   // S2(2): ∂²/∂ray∂rax
            result.s2[10] = 35.0 / 6.0 * sf.rbz() * sf.ray() + 5.0 / 6.0 * sf.cyz(); // S2(11): ∂²/∂rby∂rax
            result.s2[11] = 35.0 / 6.0 * sf.rbz() * sf.rax() + 5.0 / 6.0 * sf.cxz(); // S2(12): ∂²/∂rby∂ray
            result.s2[15] = 35.0 / 6.0 * sf.ray() * sf.rby() + 5.0 / 6.0 * sf.cyy(); // S2(16): ∂²/∂rbz∂rax
            result.s2[16] = 35.0 / 6.0 * sf.rax() * sf.rby() + 5.0 / 6.0 * sf.cxy(); // S2(17): ∂²/∂rbz∂ray
            result.s2[19] = 35.0 / 6.0 * sf.rax() * sf.ray(); // S2(20): ∂²/∂rbz∂rby
        }
    }
}

/**
 * Quadrupole-22s × Quadrupole-22c kernel
 * Orient case 124
 * Formula (static): S0 = (35/12)*(rbx^2 - rby^2)*rax*ray
 */
void quadrupole_22s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 35.0 / 12.0 * (sf.rbx() * sf.rbx() - sf.rby() * sf.rby()) * sf.rax() * sf.ray();

    if (level >= 1) {
        // First derivatives
        result.s1[0] = 35.0 / 12.0 * (sf.rbx() * sf.rbx() - sf.rby() * sf.rby()) * sf.ray(); // d/d(rax)
        result.s1[1] = 35.0 / 12.0 * (sf.rbx() * sf.rbx() - sf.rby() * sf.rby()) * sf.rax(); // d/d(ray)
        result.s1[3] = 35.0 / 6.0 * sf.rax() * sf.ray() * sf.rbx(); // d/d(rbx)
        result.s1[4] = -35.0 / 6.0 * sf.rax() * sf.ray() * sf.rby(); // d/d(rby)

        if (level >= 2) {
            // Second derivatives
            result.s2[1] = 35.0 / 12.0 * (sf.rbx() * sf.rbx() - sf.rby() * sf.rby());   // ∂²/∂ray∂rax
            result.s2[6] = 35.0 / 6.0 * sf.ray() * sf.rbx(); // ∂²/∂rbx∂rax
            result.s2[7] = 35.0 / 6.0 * sf.rax() * sf.rbx(); // ∂²/∂rbx∂ray
            result.s2[9] = 35.0 / 6.0 * sf.rax() * sf.ray(); // ∂²/∂rbx²
            result.s2[10] = -35.0 / 6.0 * sf.ray() * sf.rby(); // ∂²/∂rby∂rax
            result.s2[11] = -35.0 / 6.0 * sf.rax() * sf.rby(); // ∂²/∂rby∂ray
            result.s2[14] = -35.0 / 6.0 * sf.rax() * sf.ray(); // ∂²/∂rby²
        }
    }
}

// ============================================================================
// CHARGE-HEXADECAPOLE KERNELS (Orient S4 subroutine, rank 0+4)
// Charge @ A, Hexadecapole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Charge × Hexadecapole-40 kernel
 * Formula: S0 = 35z⁴/8 - 15z²/4 + 3/8
 */
void charge_hexadecapole_40(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = 35.0 / 8.0 * std::pow(sf.rbz(), 4) - 15.0 / 4.0 * sf.rbz() * sf.rbz() +
                3.0 / 8.0;
    if (level >= 1) {
        result.s1[5] = 35.0 / 2.0 * std::pow(sf.rbz(), 3) - 15.0 / 2.0 * sf.rbz();
        if (level >= 2) {
            result.s2[20] = 105.0 / 2.0 * sf.rbz() * sf.rbz() - 15.0 / 2.0;
        }
    }
}

/**
 * Charge × Hexadecapole-41c kernel
 * Formula: S0 = √10(7xz³-3xz)/4
 */
void charge_hexadecapole_41c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 =
        rt10 * (7.0 * sf.rbx() * std::pow(sf.rbz(), 3) - 3.0 * sf.rbx() * sf.rbz()) / 4.0;
    if (level >= 1) {
        result.s1[3] = rt10 * (7.0 * std::pow(sf.rbz(), 3) - 3.0 * sf.rbz()) / 4.0;
        result.s1[5] =
            rt10 * (21.0 * sf.rbx() * sf.rbz() * sf.rbz() - 3.0 * sf.rbx()) / 4.0;
        if (level >= 2) {
            result.s2[18] = rt10 * (21.0 * sf.rbz() * sf.rbz() - 3.0) / 4.0;
            result.s2[20] = 21.0 / 2.0 * rt10 * sf.rbx() * sf.rbz();
        }
    }
}

/**
 * Charge × Hexadecapole-41s kernel
 * Formula: S0 = √10(7yz³-3yz)/4
 */
void charge_hexadecapole_41s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 =
        rt10 * (7.0 * sf.rby() * std::pow(sf.rbz(), 3) - 3.0 * sf.rby() * sf.rbz()) / 4.0;
    if (level >= 1) {
        result.s1[4] = rt10 * (7.0 * std::pow(sf.rbz(), 3) - 3.0 * sf.rbz()) / 4.0;
        result.s1[5] =
            rt10 * (21.0 * sf.rby() * sf.rbz() * sf.rbz() - 3.0 * sf.rby()) / 4.0;
        if (level >= 2) {
            result.s2[19] = rt10 * (21.0 * sf.rbz() * sf.rbz() - 3.0) / 4.0;
            result.s2[20] = 21.0 / 2.0 * rt10 * sf.rby() * sf.rbz();
        }
    }
}

/**
 * Charge × Hexadecapole-42c kernel
 * Formula: S0 = √5(7x²z²-7y²z²-x²+y²)/4
 */
void charge_hexadecapole_42c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt5 *
                (7.0 * sf.rbx() * sf.rbx() * sf.rbz() * sf.rbz() -
                 7.0 * sf.rby() * sf.rby() * sf.rbz() * sf.rbz() - sf.rbx() * sf.rbx() +
                 sf.rby() * sf.rby()) /
                4.0;
    if (level >= 1) {
        result.s1[3] = rt5 * (14.0 * sf.rbx() * sf.rbz() * sf.rbz() - 2.0 * sf.rbx()) / 4.0;
        result.s1[4] =
            rt5 * (-14.0 * sf.rby() * sf.rbz() * sf.rbz() + 2.0 * sf.rby()) / 4.0;
        result.s1[5] =
            rt5 *
            (14.0 * sf.rbx() * sf.rbx() * sf.rbz() - 14.0 * sf.rby() * sf.rby() * sf.rbz()) / 4.0;
        if (level >= 2) {
            result.s2[9] = rt5 * (14.0 * sf.rbz() * sf.rbz() - 2.0) / 4.0;
            result.s2[14] = rt5 * (-14.0 * sf.rbz() * sf.rbz() + 2.0) / 4.0;
            result.s2[18] = 7.0 * rt5 * sf.rbx() * sf.rbz();
            result.s2[19] = -7.0 * rt5 * sf.rby() * sf.rbz();
            result.s2[20] =
                rt5 * (14.0 * sf.rbx() * sf.rbx() - 14.0 * sf.rby() * sf.rby()) / 4.0;
        }
    }
}

/**
 * Charge × Hexadecapole-42s kernel
 * Formula: S0 = √5(7xyz²-xy)/2
 */
void charge_hexadecapole_42s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 =
        rt5 * (7.0 * sf.rbx() * sf.rby() * sf.rbz() * sf.rbz() - sf.rbx() * sf.rby()) / 2.0;
    if (level >= 1) {
        result.s1[3] = rt5 * (7.0 * sf.rby() * sf.rbz() * sf.rbz() - sf.rby()) / 2.0;
        result.s1[4] = rt5 * (7.0 * sf.rbx() * sf.rbz() * sf.rbz() - sf.rbx()) / 2.0;
        result.s1[5] = 7.0 * rt5 * sf.rbx() * sf.rby() * sf.rbz();
        if (level >= 2) {
            result.s2[13] = rt5 * (7.0 * sf.rbz() * sf.rbz() - 1.0) / 2.0;
            result.s2[18] = 7.0 * rt5 * sf.rby() * sf.rbz();
            result.s2[19] = 7.0 * rt5 * sf.rbx() * sf.rbz();
            result.s2[20] = 7.0 * rt5 * sf.rbx() * sf.rby();
        }
    }
}

/**
 * Charge × Hexadecapole-43c kernel
 * Formula: S0 = √70(x³z-3xy²z)/4
 */
void charge_hexadecapole_43c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt70 *
                (sf.rbx() * sf.rbx() * sf.rbx() * sf.rbz() -
                 3.0 * sf.rbx() * sf.rby() * sf.rby() * sf.rbz()) /
                4.0;
    if (level >= 1) {
        result.s1[3] =
            rt70 * (3.0 * sf.rbx() * sf.rbx() * sf.rbz() - 3.0 * sf.rby() * sf.rby() * sf.rbz()) /
            4.0;
        result.s1[4] = -1.5 * rt70 * sf.rbx() * sf.rby() * sf.rbz();
        result.s1[5] =
            rt70 * (sf.rbx() * sf.rbx() * sf.rbx() - 3.0 * sf.rbx() * sf.rby() * sf.rby()) / 4.0;
        if (level >= 2) {
            result.s2[9] = 1.5 * rt70 * sf.rbx() * sf.rbz();
            result.s2[13] = -1.5 * rt70 * sf.rby() * sf.rbz();
            result.s2[14] = -1.5 * rt70 * sf.rbx() * sf.rbz();
            result.s2[18] =
                rt70 * (3.0 * sf.rbx() * sf.rbx() - 3.0 * sf.rby() * sf.rby()) / 4.0;
            result.s2[19] = -1.5 * rt70 * sf.rbx() * sf.rby();
        }
    }
}

/**
 * Charge × Hexadecapole-43s kernel
 * Formula: S0 = √70(3x²yz-y³z)/4
 */
void charge_hexadecapole_43s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt70 *
                (3.0 * sf.rbx() * sf.rbx() * sf.rby() * sf.rbz() -
                 sf.rby() * sf.rby() * sf.rby() * sf.rbz()) /
                4.0;
    if (level >= 1) {
        result.s1[3] = 1.5 * rt70 * sf.rbx() * sf.rby() * sf.rbz();
        result.s1[4] =
            rt70 * (3.0 * sf.rbx() * sf.rbx() * sf.rbz() - 3.0 * sf.rby() * sf.rby() * sf.rbz()) /
            4.0;
        result.s1[5] =
            rt70 * (3.0 * sf.rbx() * sf.rbx() * sf.rby() - sf.rby() * sf.rby() * sf.rby()) / 4.0;
        if (level >= 2) {
            result.s2[9] = 1.5 * rt70 * sf.rby() * sf.rbz();
            result.s2[13] = 1.5 * rt70 * sf.rbx() * sf.rbz();
            result.s2[14] = -1.5 * rt70 * sf.rby() * sf.rbz();
            result.s2[18] = 1.5 * rt70 * sf.rbx() * sf.rby();
            result.s2[19] =
                rt70 * (3.0 * sf.rbx() * sf.rbx() - 3.0 * sf.rby() * sf.rby()) / 4.0;
        }
    }
}

/**
 * Charge × Hexadecapole-44c kernel
 * Formula: S0 = √35(x⁴-6x²y²+y⁴)/8
 */
void charge_hexadecapole_44c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt35 *
                (std::pow(sf.rbx(), 4) - 6.0 * sf.rbx() * sf.rbx() * sf.rby() * sf.rby() +
                 std::pow(sf.rby(), 4)) /
                8.0;
    if (level >= 1) {
        result.s1[3] =
            rt35 * (4.0 * std::pow(sf.rbx(), 3) - 12.0 * sf.rbx() * sf.rby() * sf.rby()) /
            8.0;
        result.s1[4] =
            rt35 * (-12.0 * sf.rbx() * sf.rbx() * sf.rby() + 4.0 * std::pow(sf.rby(), 3)) /
            8.0;
        if (level >= 2) {
            result.s2[9] =
                rt35 * (12.0 * sf.rbx() * sf.rbx() - 12.0 * sf.rby() * sf.rby()) / 8.0;
            result.s2[13] = -3.0 * rt35 * sf.rbx() * sf.rby();
            result.s2[14] =
                rt35 * (-12.0 * sf.rbx() * sf.rbx() + 12.0 * sf.rby() * sf.rby()) / 8.0;
        }
    }
}

/**
 * Charge × Hexadecapole-44s kernel
 * Formula: S0 = √35(x³y-xy³)/2
 */
void charge_hexadecapole_44s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 =
        rt35 *
        (sf.rbx() * sf.rbx() * sf.rbx() * sf.rby() - sf.rbx() * sf.rby() * sf.rby() * sf.rby()) / 2.0;
    if (level >= 1) {
        result.s1[3] =
            rt35 * (3.0 * sf.rbx() * sf.rbx() * sf.rby() - sf.rby() * sf.rby() * sf.rby()) / 2.0;
        result.s1[4] =
            rt35 * (sf.rbx() * sf.rbx() * sf.rbx() - 3.0 * sf.rbx() * sf.rby() * sf.rby()) / 2.0;
        if (level >= 2) {
            result.s2[9] = 3.0 * rt35 * sf.rbx() * sf.rby();
            result.s2[13] =
                rt35 * (3.0 * sf.rbx() * sf.rbx() - 3.0 * sf.rby() * sf.rby()) / 2.0;
            result.s2[14] = -3.0 * rt35 * sf.rbx() * sf.rby();
        }
    }
}

// ============================================================================
// DIPOLE-OCTOPOLE KERNELS (Orient cases 80-99 in S4 subroutine)
// Dipole @ A (uses rax, ray, raz), Octopole @ B (uses rbx, rby, rbz)
// CRITICAL: Only Q10 × octopole cases need scaling factor of 0.4 (1/2.5)
// Q11c/Q11s × octopole cases use formulas as-is
// ============================================================================

/**
 * Dipole-z × Octopole-30 kernel
 * Orient case 80: Q10 × Q30
 * Formula: S0 = 0.4 * (35/8*rbz³*raz - 15/8*raz*rbz)
 * NEEDS SCALING: factor 0.4 applied
 */
void dipole_z_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double scale = 0.4; // Empirical factor needed for Q10 interactions
    result.s0 = scale * (35.0 / 8.0 * sf.rbz() * sf.rbz() * sf.rbz() * sf.raz() -
                         15.0 / 8.0 * sf.raz() * sf.rbz());

    if (level >= 1) {
        // First derivatives
        result.s1[2] = scale * (35.0 / 8.0 * sf.rbz() * sf.rbz() * sf.rbz() - 15.0 / 8.0 * sf.rbz()); // d/d(raz)
        result.s1[5] = scale * (105.0 / 8.0 * sf.rbz() * sf.rbz() * sf.raz() - 15.0 / 8.0 * sf.raz()); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 80: S2(18), S2(21))
            result.s2[17] = scale * (105.0 / 8.0 * sf.rbz() * sf.rbz() - 15.0 / 8.0); // d²/d(raz)d(rbz) -> d²/d(rbz)²
            result.s2[20] = scale * 105.0 / 4.0 * sf.raz() * sf.rbz(); // d²/d(raz)d(rbz)
        }
    }
}

/**
 * Dipole-z × Octopole-31c kernel
 * Orient case 81: Q10 × Q31c
 * Formula: S0 = 0.4 * rt6 * (35*rbx*rbz²*raz - 5*raz*rbx)/16
 * NEEDS SCALING: factor 0.4 applied
 */
void dipole_z_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double scale = 0.4;
    result.s0 =
        scale * rt6 *
        (35.0 * sf.rbx() * sf.rbz() * sf.rbz() * sf.raz() - 5.0 * sf.raz() * sf.rbx()) / 16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = scale * rt6 * (35.0 * sf.rbx() * sf.rbz() * sf.rbz() - 5.0 * sf.rbx()) / 16.0; // d/d(raz)
        result.s1[3] = scale * rt6 * (35.0 * sf.rbz() * sf.rbz() * sf.raz() - 5.0 * sf.raz()) / 16.0; // d/d(rbx)
        result.s1[5] = scale * rt6 * 70.0 / 16.0 * sf.rbx() * sf.rbz() * sf.raz(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 81: S2(9), S2(18), S2(19), S2(21))
            result.s2[8] = scale * rt6 * (35.0 * sf.rbz() * sf.rbz() - 5.0) / 16.0; // d²/d(raz)d(rbx)
            result.s2[17] = scale * rt6 * 35.0 / 8.0 * sf.rbx() * sf.rbz(); // d²/d(rbx)d(rbz)
            result.s2[18] = scale * rt6 * 70.0 / 16.0 * sf.raz() * sf.rbz(); // d²/d(raz)d(rbz)
            result.s2[20] = scale * rt6 * 70.0 / 16.0 * sf.raz() * sf.rbx(); // d²/d(raz)d(rbz) [different pair]
        }
    }
}

/**
 * Dipole-z × Octopole-31s kernel
 * Orient case 82: Q10 × Q31s
 * Formula: S0 = 0.4 * rt6 * (35*rby*rbz²*raz - 5*raz*rby)/16
 * NEEDS SCALING: factor 0.4 applied
 */
void dipole_z_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double scale = 0.4;
    result.s0 =
        scale * rt6 *
        (35.0 * sf.rby() * sf.rbz() * sf.rbz() * sf.raz() - 5.0 * sf.raz() * sf.rby()) / 16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = scale * rt6 * (35.0 * sf.rby() * sf.rbz() * sf.rbz() - 5.0 * sf.rby()) / 16.0; // d/d(raz)
        result.s1[4] = scale * rt6 * (35.0 * sf.rbz() * sf.rbz() * sf.raz() - 5.0 * sf.raz()) / 16.0; // d/d(rby)
        result.s1[5] = scale * rt6 * 70.0 / 16.0 * sf.rby() * sf.rbz() * sf.raz(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 82: S2(13), S2(18), S2(20), S2(21))
            result.s2[12] = scale * rt6 * (35.0 * sf.rbz() * sf.rbz() - 5.0) / 16.0; // d²/d(raz)d(rby)
            result.s2[17] = scale * rt6 * 35.0 / 8.0 * sf.rby() * sf.rbz(); // d²/d(rby)d(rbz)
            result.s2[19] = scale * rt6 * 70.0 / 16.0 * sf.raz() * sf.rbz(); // d²/d(raz)d(rbz)
            result.s2[20] = scale * rt6 * 70.0 / 16.0 * sf.raz() * sf.rby(); // d²/d(raz)d(rby) [cross]
        }
    }
}

/**
 * Dipole-z × Octopole-32c kernel
 * Orient case 83: Q10 × Q32c
 * Formula: S0 = 0.4 * rt15 * (7*rbx²*rbz*raz - 7*rby²*rbz*raz)/8
 * NEEDS SCALING: factor 0.4 applied
 */
void dipole_z_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double scale = 0.4;
    result.s0 = scale * rt15 *
                (7.0 * sf.rbx() * sf.rbx() * sf.rbz() * sf.raz() -
                 7.0 * sf.rby() * sf.rby() * sf.rbz() * sf.raz()) /
                8.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = scale * rt15 * (7.0 * sf.rbx() * sf.rbx() * sf.rbz() - 7.0 * sf.rby() * sf.rby() * sf.rbz()) / 8.0; // d/d(raz)
        result.s1[3] = scale * rt15 * 14.0 / 8.0 * sf.rbx() * sf.rbz() * sf.raz(); // d/d(rbx)
        result.s1[4] = scale * rt15 * (-14.0) / 8.0 * sf.rby() * sf.rbz() * sf.raz(); // d/d(rby)
        result.s1[5] = scale * rt15 * (7.0 * sf.rbx() * sf.rbx() * sf.raz() - 7.0 * sf.rby() * sf.rby() * sf.raz()) / 8.0; // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 83: S2(9), S2(10), S2(13), S2(14), S2(18), S2(20))
            result.s2[8] = scale * rt15 * 7.0 / 4.0 * sf.rbx() * sf.rbz(); // d²/d(raz)d(rbx)
            result.s2[9] = scale * rt15 * 14.0 / 8.0 * sf.raz() * sf.rbz(); // d²/d(rbx)d(rbz)
            result.s2[12] = scale * rt15 * (-7.0) / 4.0 * sf.rby() * sf.rbz(); // d²/d(raz)d(rby)
            result.s2[13] = scale * rt15 * (-14.0) / 8.0 * sf.raz() * sf.rbz(); // d²/d(rby)d(rbz)
            result.s2[17] = scale * rt15 * (7.0 * sf.rbx() * sf.rbx() - 7.0 * sf.rby() * sf.rby()) / 8.0; // d²/d(rbz)d(raz)
            result.s2[19] = scale * rt15 * (7.0 * sf.rbx() * sf.rbx() - 7.0 * sf.rby() * sf.rby()) / 8.0; // d²/d(rbx)d(raz) [symmetry]
        }
    }
}

/**
 * Dipole-z × Octopole-32s kernel
 * Orient case 84: Q10 × Q32s
 * Formula: S0 = 0.4 * rt15 * 7*rbx*rby*rbz*raz/4
 * NEEDS SCALING: factor 0.4 applied
 */
void dipole_z_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double scale = 0.4;
    result.s0 = scale * rt15 * 7.0 * sf.rbx() * sf.rby() * sf.rbz() * sf.raz() / 4.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = scale * rt15 * 7.0 / 4.0 * sf.rbx() * sf.rby() * sf.rbz(); // d/d(raz)
        result.s1[3] = scale * rt15 * 7.0 / 4.0 * sf.rby() * sf.rbz() * sf.raz(); // d/d(rbx)
        result.s1[4] = scale * rt15 * 7.0 / 4.0 * sf.rbx() * sf.rbz() * sf.raz(); // d/d(rby)
        result.s1[5] = scale * rt15 * 7.0 / 4.0 * sf.rbx() * sf.rby() * sf.raz(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 84: S2(9), S2(13), S2(14), S2(18), S2(19), S2(20))
            result.s2[8] = scale * rt15 * 7.0 / 4.0 * sf.rby() * sf.rbz(); // d²/d(raz)d(rbx)
            result.s2[12] = scale * rt15 * 7.0 / 4.0 * sf.rbx() * sf.rbz(); // d²/d(raz)d(rby)
            result.s2[13] = scale * rt15 * 7.0 / 4.0 * sf.raz() * sf.rbz(); // d²/d(rbx)d(rby)
            result.s2[17] = scale * rt15 * 7.0 / 4.0 * sf.rbx() * sf.rby(); // d²/d(rby)d(rbz)
            result.s2[18] = scale * rt15 * 7.0 / 4.0 * sf.raz() * sf.rby(); // d²/d(raz)d(rbz)
            result.s2[19] = scale * rt15 * 7.0 / 4.0 * sf.raz() * sf.rbx(); // d²/d(rbx)d(rbz)
        }
    }
}

/**
 * Dipole-z × Octopole-33c kernel
 * Orient case 85: Q10 × Q33c
 * Formula: S0 = 0.4 * rt10 * (7*rbx³*raz - 21*rbx*rby²*raz)/16
 * NEEDS SCALING: factor 0.4 applied
 */
void dipole_z_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double scale = 0.4;
    result.s0 = scale * rt10 *
                (7.0 * sf.rbx() * sf.rbx() * sf.rbx() * sf.raz() -
                 21.0 * sf.rbx() * sf.rby() * sf.rby() * sf.raz()) /
                16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = scale * rt10 * (7.0 * sf.rbx() * sf.rbx() * sf.rbx() - 21.0 * sf.rbx() * sf.rby() * sf.rby()) / 16.0; // d/d(raz)
        result.s1[3] = scale * rt10 * (21.0 * sf.rbx() * sf.rbx() * sf.raz() - 21.0 * sf.rby() * sf.rby() * sf.raz()) / 16.0; // d/d(rbx)
        result.s1[4] = scale * rt10 * (-42.0) / 16.0 * sf.rbx() * sf.rby() * sf.raz(); // d/d(rby)

        if (level >= 2) {
            // Second derivatives (Orient case 85: S2(9), S2(10), S2(13), S2(14), S2(15))
            result.s2[8] = scale * rt10 * (21.0 * sf.rbx() * sf.rbx() - 21.0 * sf.rby() * sf.rby()) / 16.0; // d²/d(raz)d(rbx)
            result.s2[9] = scale * rt10 * 42.0 / 16.0 * sf.raz() * sf.rbx(); // d²/d(rbx)²
            result.s2[12] = scale * rt10 * (-21.0) / 8.0 * sf.rbx() * sf.rby(); // d²/d(raz)d(rby)
            result.s2[13] = scale * rt10 * (-42.0) / 16.0 * sf.raz() * sf.rby(); // d²/d(rbx)d(rby)
            result.s2[14] = scale * rt10 * (-42.0) / 16.0 * sf.raz() * sf.rbx(); // d²/d(rby)d(rbx)
        }
    }
}

/**
 * Dipole-z × Octopole-33s kernel
 * Orient case 86: Q10 × Q33s
 * Formula: S0 = 0.4 * rt10 * (21*rbx²*rby*raz - 7*rby³*raz)/16
 * NEEDS SCALING: factor 0.4 applied
 */
void dipole_z_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double scale = 0.4;
    result.s0 = scale * rt10 *
                (21.0 * sf.rbx() * sf.rbx() * sf.rby() * sf.raz() -
                 7.0 * sf.rby() * sf.rby() * sf.rby() * sf.raz()) /
                16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = scale * rt10 * (21.0 * sf.rbx() * sf.rbx() * sf.rby() - 7.0 * sf.rby() * sf.rby() * sf.rby()) / 16.0; // d/d(raz)
        result.s1[3] = scale * rt10 * 42.0 / 16.0 * sf.rbx() * sf.rby() * sf.raz(); // d/d(rbx)
        result.s1[4] = scale * rt10 * (21.0 * sf.rbx() * sf.rbx() * sf.raz() - 21.0 * sf.rby() * sf.rby() * sf.raz()) / 16.0; // d/d(rby)

        if (level >= 2) {
            // Second derivatives (Orient case 86: S2(9), S2(10), S2(13), S2(14), S2(15))
            result.s2[8] = scale * rt10 * 21.0 / 8.0 * sf.rbx() * sf.rby(); // d²/d(raz)d(rbx)
            result.s2[9] = scale * rt10 * 42.0 / 16.0 * sf.raz() * sf.rby(); // d²/d(rbx)d(rby)
            result.s2[12] = scale * rt10 * (21.0 * sf.rbx() * sf.rbx() - 21.0 * sf.rby() * sf.rby()) / 16.0; // d²/d(raz)d(rby)
            result.s2[13] = scale * rt10 * 42.0 / 16.0 * sf.raz() * sf.rbx(); // d²/d(rbx)d(rby) [cross]
            result.s2[14] = scale * rt10 * (-42.0) / 16.0 * sf.raz() * sf.rby(); // d²/d(rby)²
        }
    }
}

/**
 * Dipole-x × Octopole-30 kernel
 * Orient case 87: Q11c × Q30
 * Formula: S0 = 35/8*rbz³*rax - 15/8*rbz*rax
 * NO SCALING
 */
void dipole_x_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = 35.0 / 8.0 * sf.rbz() * sf.rbz() * sf.rbz() * sf.rax() -
                15.0 / 8.0 * sf.rbz() * sf.rax();

    if (level >= 1) {
        // First derivatives
        result.s1[0] = 35.0 / 8.0 * sf.rbz() * sf.rbz() * sf.rbz() - 15.0 / 8.0 * sf.rbz(); // d/d(rax)
        result.s1[5] = 105.0 / 8.0 * sf.rbz() * sf.rbz() * sf.rax() - 15.0 / 8.0 * sf.rax(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 87: S2(16), S2(21))
            result.s2[15] = 105.0 / 8.0 * sf.rbz() * sf.rbz() - 15.0 / 8.0; // d²/d(rbz)²
            result.s2[20] = 105.0 / 4.0 * sf.rbz() * sf.rax(); // d²/d(rax)d(rbz)
        }
    }
}

/**
 * Dipole-x × Octopole-31c kernel
 * Orient case 88: Q11c × Q31c (cxx=1)
 * Formula: S0 = rt6*(35*rbx*rbz²*rax - 5*rax*rbx + 5*rbz² - 1)/16
 * NO SCALING
 */
void dipole_x_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt6 *
                (35.0 * sf.rbx() * sf.rbz() * sf.rbz() * sf.rax() -
                 5.0 * sf.rax() * sf.rbx() + 5.0 * sf.rbz() * sf.rbz() - 1.0) /
                16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt6 * (35.0 * sf.rbx() * sf.rbz() * sf.rbz() - 5.0 * sf.rbx()) / 16.0; // d/d(rax)
        result.s1[3] = rt6 * (35.0 * sf.rbz() * sf.rbz() * sf.rax() - 5.0 * sf.rax()) / 16.0; // d/d(rbx)
        result.s1[5] = rt6 * (70.0 * sf.rbx() * sf.rbz() * sf.rax() + 10.0 * sf.rbz()) / 16.0; // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 88: S2(7), S2(16), S2(19), S2(21))
            result.s2[6] = rt6 * (35.0 * sf.rbz() * sf.rbz() - 5.0) / 16.0; // d²/d(rax)d(rbx)
            result.s2[15] = rt6 * 35.0 / 8.0 * sf.rbx() * sf.rbz(); // d²/d(rbx)d(rbz)
            result.s2[18] = rt6 * 70.0 / 16.0 * sf.rbz() * sf.rax(); // d²/d(rax)d(rbz)
            result.s2[20] = rt6 * 70.0 / 16.0 * sf.rax() * sf.rbx(); // d²/d(rax)d(rbz) [cross]
        }
    }
}

/**
 * Dipole-x × Octopole-31s kernel
 * Orient case 89: Q11c × Q31s
 * Formula: S0 = rt6*(35*rby*rbz²*rax - 5*rax*rby)/16
 * NO SCALING
 */
void dipole_x_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 =
        rt6 * (35.0 * sf.rby() * sf.rbz() * sf.rbz() * sf.rax() - 5.0 * sf.rax() * sf.rby()) /
        16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt6 * (35.0 * sf.rby() * sf.rbz() * sf.rbz() - 5.0 * sf.rby()) / 16.0; // d/d(rax)
        result.s1[4] = rt6 * (35.0 * sf.rbz() * sf.rbz() * sf.rax() - 5.0 * sf.rax()) / 16.0; // d/d(rby)
        result.s1[5] = rt6 * 70.0 / 16.0 * sf.rby() * sf.rbz() * sf.rax(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 89: S2(11), S2(16), S2(20), S2(21))
            result.s2[10] = rt6 * (35.0 * sf.rbz() * sf.rbz() - 5.0) / 16.0; // d²/d(rax)d(rby)
            result.s2[15] = rt6 * 35.0 / 8.0 * sf.rby() * sf.rbz(); // d²/d(rby)d(rbz)
            result.s2[19] = rt6 * 70.0 / 16.0 * sf.rbz() * sf.rax(); // d²/d(rax)d(rbz)
            result.s2[20] = rt6 * 70.0 / 16.0 * sf.rax() * sf.rby(); // d²/d(rax)d(rby) [cross]
        }
    }
}

/**
 * Dipole-x × Octopole-32c kernel
 * Orient case 90: Q11c × Q32c
 * Formula: S0 = rt15*(7*rbx²*rbz*rax - 7*rby²*rbz*rax + rbx²*cxz + 2*rbx*rbz*cxx - rby²*cxz - 2*rby*rbz*cxy)/8
 * For static multipoles (cxx=1, cxy=0, cxz=0): S0 = rt15*(7*rbx²*rbz*rax - 7*rby²*rbz*rax + 2*rbx*rbz)/8
 * NO SCALING
 */
void dipole_x_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt15 *
                (7.0 * sf.rbx() * sf.rbx() * sf.rbz() * sf.rax() -
                 7.0 * sf.rby() * sf.rby() * sf.rbz() * sf.rax() +
                 2.0 * sf.rbx() * sf.rbz()) /
                8.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt15 * (7.0 * sf.rbx() * sf.rbx() * sf.rbz() - 7.0 * sf.rby() * sf.rby() * sf.rbz()) / 8.0; // d/d(rax)
        result.s1[3] = rt15 * (14.0 * sf.rbx() * sf.rbz() * sf.rax() + 2.0 * sf.rbz()) / 8.0; // d/d(rbx)
        result.s1[4] = rt15 * (-14.0 * sf.rby() * sf.rbz() * sf.rax()) / 8.0; // d/d(rby)
        result.s1[5] = rt15 * (7.0 * sf.rbx() * sf.rbx() * sf.rax() - 7.0 * sf.rby() * sf.rby() * sf.rax() + 2.0 * sf.rbx()) / 8.0; // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 90: S2(7), S2(10), S2(11), S2(15), S2(16), S2(19), S2(20))
            result.s2[6] = rt15 * 7.0 / 4.0 * sf.rbx() * sf.rbz(); // d²/d(rax)d(rbx)
            result.s2[9] = rt15 * (14.0 * sf.rax() * sf.rbz() + 2.0) / 8.0; // d²/d(rbx)²
            result.s2[10] = rt15 * (-7.0) / 4.0 * sf.rby() * sf.rbz(); // d²/d(rax)d(rby)
            result.s2[14] = rt15 * (-14.0 * sf.rax() * sf.rbz()) / 8.0; // d²/d(rby)²
            result.s2[15] = rt15 * (7.0 * sf.rbx() * sf.rbx() - 7.0 * sf.rby() * sf.rby()) / 8.0; // d²/d(rbz)²
            result.s2[18] = rt15 * (14.0 * sf.rax() * sf.rbx() + 2.0) / 8.0; // d²/d(rax)d(rbz)
            result.s2[19] = rt15 * (-14.0 * sf.rax() * sf.rby()) / 8.0; // d²/d(rby)d(rbz)
        }
    }
}

/**
 * Dipole-x × Octopole-32s kernel
 * Orient case 91: Q11c × Q32s
 * Formula: S0 = rt15*(7*rbx*rby*rbz*rax + rbx*rby*cxz + rbx*rbz*cxy + rby*rbz*cxx)/4
 * For static multipoles (cxx=1, cxy=0, cxz=0): S0 = rt15*(7*rbx*rby*rbz*rax + rby*rbz)/4
 * NO SCALING
 */
void dipole_x_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt15 * (7.0 * sf.rbx() * sf.rby() * sf.rbz() * sf.rax() + sf.rby() * sf.rbz()) / 4.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt15 * 7.0 / 4.0 * sf.rbx() * sf.rby() * sf.rbz(); // d/d(rax)
        result.s1[3] = rt15 * (7.0 * sf.rby() * sf.rbz() * sf.rax()) / 4.0; // d/d(rbx)
        result.s1[4] = rt15 * (7.0 * sf.rbx() * sf.rbz() * sf.rax() + sf.rbz()) / 4.0; // d/d(rby)
        result.s1[5] = rt15 * (7.0 * sf.rbx() * sf.rby() * sf.rax() + sf.rby()) / 4.0; // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 91: S2(7), S2(11), S2(14), S2(16), S2(19), S2(20))
            result.s2[6] = rt15 * 7.0 / 4.0 * sf.rby() * sf.rbz(); // d²/d(rax)d(rbx)
            result.s2[10] = rt15 * 7.0 / 4.0 * sf.rbx() * sf.rbz(); // d²/d(rax)d(rby)
            result.s2[13] = rt15 * (7.0 * sf.rax() * sf.rbz()) / 4.0; // d²/d(rbx)d(rby)
            result.s2[15] = rt15 * 7.0 / 4.0 * sf.rbx() * sf.rby(); // d²/d(rbz)²
            result.s2[18] = rt15 * (7.0 * sf.rax() * sf.rby() + 1.0) / 4.0; // d²/d(rax)d(rbz)
            result.s2[19] = rt15 * (7.0 * sf.rax() * sf.rbx()) / 4.0; // d²/d(rby)d(rbz)
        }
    }
}

/**
 * Dipole-x × Octopole-33c kernel
 * Orient case 92: Q11c × Q33c (cxx=1, cxy=0)
 * Formula: S0 = rt10*(7*rbx³*rax - 21*rbx*rby²*rax + 3*rbx² - 3*rby²)/16
 * NO SCALING
 */
void dipole_x_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt10 *
                (7.0 * sf.rbx() * sf.rbx() * sf.rbx() * sf.rax() -
                 21.0 * sf.rbx() * sf.rby() * sf.rby() * sf.rax() +
                 3.0 * sf.rbx() * sf.rbx() - 3.0 * sf.rby() * sf.rby()) /
                16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt10 * (7.0 * sf.rbx() * sf.rbx() * sf.rbx() - 21.0 * sf.rbx() * sf.rby() * sf.rby()) / 16.0; // d/d(rax)
        result.s1[3] = rt10 * (21.0 * sf.rbx() * sf.rbx() * sf.rax() - 21.0 * sf.rby() * sf.rby() * sf.rax() + 6.0 * sf.rbx()) / 16.0; // d/d(rbx)
        result.s1[4] = rt10 * (-42.0 * sf.rbx() * sf.rby() * sf.rax() - 6.0 * sf.rby()) / 16.0; // d/d(rby)

        if (level >= 2) {
            // Second derivatives (Orient case 92: S2(7), S2(10), S2(11), S2(14), S2(15))
            result.s2[6] = rt10 * (21.0 * sf.rbx() * sf.rbx() - 21.0 * sf.rby() * sf.rby()) / 16.0; // d²/d(rax)d(rbx)
            result.s2[9] = rt10 * 42.0 / 16.0 * sf.rax() * sf.rbx(); // d²/d(rbx)²
            result.s2[10] = rt10 * (-21.0) / 8.0 * sf.rbx() * sf.rby(); // d²/d(rax)d(rby)
            result.s2[13] = rt10 * (-42.0) / 16.0 * sf.rax() * sf.rby(); // d²/d(rbx)d(rby)
            result.s2[14] = rt10 * (-42.0) / 16.0 * sf.rax() * sf.rbx(); // d²/d(rby)d(rbx) [cross]
        }
    }
}

/**
 * Dipole-x × Octopole-33s kernel
 * Orient case 93: Q11c × Q33s
 * Formula: S0 = rt10*(21*rbx²*rby*rax - 7*rby³*rax + 3*rbx²*cxy + 6*rbx*rby*cxx - 3*rby²*cxy)/16
 * For static multipoles (cxx=1, cxy=0): S0 = rt10*(21*rbx²*rby*rax - 7*rby³*rax + 6*rbx*rby)/16
 * NO SCALING
 */
void dipole_x_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt10 *
                (21.0 * sf.rbx() * sf.rbx() * sf.rby() * sf.rax() -
                 7.0 * sf.rby() * sf.rby() * sf.rby() * sf.rax() +
                 6.0 * sf.rbx() * sf.rby()) /
                16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt10 * (21.0 * sf.rbx() * sf.rbx() * sf.rby() - 7.0 * sf.rby() * sf.rby() * sf.rby()) / 16.0; // d/d(rax)
        result.s1[3] = rt10 * (42.0 * sf.rbx() * sf.rby() * sf.rax() + 6.0 * sf.rby()) / 16.0; // d/d(rbx)
        result.s1[4] = rt10 * (21.0 * sf.rbx() * sf.rbx() * sf.rax() - 21.0 * sf.rby() * sf.rby() * sf.rax() + 6.0 * sf.rbx()) / 16.0; // d/d(rby)

        if (level >= 2) {
            // Second derivatives (Orient case 93: S2(7), S2(10), S2(11), S2(14), S2(15))
            result.s2[6] = rt10 * 21.0 / 8.0 * sf.rbx() * sf.rby(); // d²/d(rax)d(rbx)
            result.s2[9] = rt10 * (42.0 * sf.rax() * sf.rby() + 6.0) / 16.0; // d²/d(rbx)d(rby)
            result.s2[10] = rt10 * (21.0 * sf.rbx() * sf.rbx() - 21.0 * sf.rby() * sf.rby()) / 16.0; // d²/d(rax)d(rby)
            result.s2[13] = rt10 * 42.0 / 16.0 * sf.rax() * sf.rbx(); // d²/d(rbx)d(rby)
            result.s2[14] = rt10 * (-42.0) / 16.0 * sf.rax() * sf.rby(); // d²/d(rby)²
        }
    }
}

/**
 * Dipole-y × Octopole-30 kernel
 * Orient case 94: Q11s × Q30
 * Formula: S0 = 35/8*rbz³*ray - 15/8*rbz*ray
 * NO SCALING
 */
void dipole_y_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = 35.0 / 8.0 * sf.rbz() * sf.rbz() * sf.rbz() * sf.ray() -
                15.0 / 8.0 * sf.rbz() * sf.ray();

    if (level >= 1) {
        // First derivatives
        result.s1[1] = 35.0 / 8.0 * sf.rbz() * sf.rbz() * sf.rbz() - 15.0 / 8.0 * sf.rbz(); // d/d(ray)
        result.s1[5] = 105.0 / 8.0 * sf.rbz() * sf.rbz() * sf.ray() - 15.0 / 8.0 * sf.ray(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 94: S2(17), S2(21))
            result.s2[16] = 105.0 / 8.0 * sf.rbz() * sf.rbz() - 15.0 / 8.0; // d²/d(rbz)²
            result.s2[20] = 105.0 / 4.0 * sf.rbz() * sf.ray(); // d²/d(ray)d(rbz)
        }
    }
}

/**
 * Dipole-y × Octopole-31c kernel
 * Orient case 95: Q11s × Q31c
 * Formula: S0 = rt6*(35*rbx*rbz²*ray - 5*rbx*ray)/16
 * NO SCALING
 */
void dipole_y_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 =
        rt6 * (35.0 * sf.rbx() * sf.rbz() * sf.rbz() * sf.ray() - 5.0 * sf.rbx() * sf.ray()) /
        16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt6 * (35.0 * sf.rbx() * sf.rbz() * sf.rbz() - 5.0 * sf.rbx()) / 16.0; // d/d(ray)
        result.s1[3] = rt6 * (35.0 * sf.rbz() * sf.rbz() * sf.ray() - 5.0 * sf.ray()) / 16.0; // d/d(rbx)
        result.s1[5] = rt6 * 70.0 / 16.0 * sf.rbx() * sf.rbz() * sf.ray(); // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 95: S2(8), S2(17), S2(19), S2(21))
            result.s2[7] = rt6 * (35.0 * sf.rbz() * sf.rbz() - 5.0) / 16.0; // d²/d(ray)d(rbx)
            result.s2[16] = rt6 * 35.0 / 8.0 * sf.rbx() * sf.rbz(); // d²/d(rbx)d(rbz)
            result.s2[18] = rt6 * 70.0 / 16.0 * sf.rbz() * sf.ray(); // d²/d(ray)d(rbz)
            result.s2[20] = rt6 * 70.0 / 16.0 * sf.rbx() * sf.ray(); // d²/d(ray)d(rbx) [cross]
        }
    }
}

/**
 * Dipole-y × Octopole-31s kernel
 * Orient case 96: Q11s × Q31s (cyy=1)
 * Formula: S0 = rt6*(35*rby*rbz²*ray - 5*ray*rby + 5*rbz² - 1)/16
 * NO SCALING
 */
void dipole_y_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt6 *
                (35.0 * sf.rby() * sf.rbz() * sf.rbz() * sf.ray() -
                 5.0 * sf.ray() * sf.rby() + 5.0 * sf.rbz() * sf.rbz() - 1.0) /
                16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt6 * (35.0 * sf.rby() * sf.rbz() * sf.rbz() - 5.0 * sf.rby()) / 16.0; // d/d(ray)
        result.s1[4] = rt6 * (35.0 * sf.rbz() * sf.rbz() * sf.ray() - 5.0 * sf.ray()) / 16.0; // d/d(rby)
        result.s1[5] = rt6 * (70.0 * sf.rby() * sf.rbz() * sf.ray() + 10.0 * sf.rbz()) / 16.0; // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 96: S2(12), S2(17), S2(20), S2(21))
            result.s2[11] = rt6 * (35.0 * sf.rbz() * sf.rbz() - 5.0) / 16.0; // d²/d(ray)d(rby)
            result.s2[16] = rt6 * 35.0 / 8.0 * sf.rby() * sf.rbz(); // d²/d(rby)d(rbz)
            result.s2[19] = rt6 * 70.0 / 16.0 * sf.rbz() * sf.ray(); // d²/d(ray)d(rbz)
            result.s2[20] = rt6 * 70.0 / 16.0 * sf.ray() * sf.rby(); // d²/d(ray)d(rby) [cross]
        }
    }
}

/**
 * Dipole-y × Octopole-32c kernel
 * Orient case 97: Q11s × Q32c
 * Formula: S0 = rt15*(7*rbx²*rbz*ray - 7*rby²*rbz*ray + rbx²*cyz + 2*rbx*rbz*cyx - rby²*cyz - 2*rby*rbz*cyy)/8
 * For static multipoles (cyx=0, cyy=1, cyz=0): S0 = rt15*(7*rbx²*rbz*ray - 7*rby²*rbz*ray - 2*rby*rbz)/8
 * NO SCALING
 */
void dipole_y_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt15 *
                (7.0 * sf.rbx() * sf.rbx() * sf.rbz() * sf.ray() -
                 7.0 * sf.rby() * sf.rby() * sf.rbz() * sf.ray() -
                 2.0 * sf.rby() * sf.rbz()) /
                8.0;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt15 * (7.0 * sf.rbx() * sf.rbx() * sf.rbz() - 7.0 * sf.rby() * sf.rby() * sf.rbz()) / 8.0; // d/d(ray)
        result.s1[3] = rt15 * (14.0 * sf.rbx() * sf.rbz() * sf.ray()) / 8.0; // d/d(rbx)
        result.s1[4] = rt15 * (-14.0 * sf.rby() * sf.rbz() * sf.ray() - 2.0 * sf.rbz()) / 8.0; // d/d(rby)
        result.s1[5] = rt15 * (7.0 * sf.rbx() * sf.rbx() * sf.ray() - 7.0 * sf.rby() * sf.rby() * sf.ray() - 2.0 * sf.rby()) / 8.0; // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 97: S2(8), S2(10), S2(12), S2(15), S2(17), S2(19), S2(20))
            result.s2[7] = rt15 * 7.0 / 4.0 * sf.rbx() * sf.rbz(); // d²/d(ray)d(rbx)
            result.s2[9] = rt15 * (14.0 * sf.ray() * sf.rbz()) / 8.0; // d²/d(rbx)²
            result.s2[11] = rt15 * (-7.0) / 4.0 * sf.rby() * sf.rbz(); // d²/d(ray)d(rby)
            result.s2[14] = rt15 * (-14.0 * sf.ray() * sf.rbz() - 2.0) / 8.0; // d²/d(rby)²
            result.s2[16] = rt15 * (7.0 * sf.rbx() * sf.rbx() - 7.0 * sf.rby() * sf.rby()) / 8.0; // d²/d(rbz)²
            result.s2[18] = rt15 * (14.0 * sf.ray() * sf.rbx()) / 8.0; // d²/d(ray)d(rbz)
            result.s2[19] = rt15 * (-14.0 * sf.ray() * sf.rby() - 2.0) / 8.0; // d²/d(rby)d(rbz)
        }
    }
}

/**
 * Dipole-y × Octopole-32s kernel
 * Orient case 98: Q11s × Q32s
 * Formula: S0 = rt15*(7*rbx*rby*rbz*ray + rbx*rby*cyz + rbx*rbz*cyy + rby*rbz*cyx)/4
 * For static multipoles (cyx=0, cyy=1, cyz=0): S0 = rt15*(7*rbx*rby*rbz*ray + rbx*rbz)/4
 * NO SCALING
 */
void dipole_y_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt15 * (7.0 * sf.rbx() * sf.rby() * sf.rbz() * sf.ray() + sf.rbx() * sf.rbz()) / 4.0;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt15 * 7.0 / 4.0 * sf.rbx() * sf.rby() * sf.rbz(); // d/d(ray)
        result.s1[3] = rt15 * (7.0 * sf.rby() * sf.rbz() * sf.ray() + sf.rbz()) / 4.0; // d/d(rbx)
        result.s1[4] = rt15 * (7.0 * sf.rbx() * sf.rbz() * sf.ray()) / 4.0; // d/d(rby)
        result.s1[5] = rt15 * (7.0 * sf.rbx() * sf.rby() * sf.ray() + sf.rbx()) / 4.0; // d/d(rbz)

        if (level >= 2) {
            // Second derivatives (Orient case 98: S2(8), S2(12), S2(14), S2(17), S2(19), S2(20))
            result.s2[7] = rt15 * 7.0 / 4.0 * sf.rby() * sf.rbz(); // d²/d(ray)d(rbx)
            result.s2[11] = rt15 * 7.0 / 4.0 * sf.rbx() * sf.rbz(); // d²/d(ray)d(rby)
            result.s2[13] = rt15 * (7.0 * sf.ray() * sf.rbz()) / 4.0; // d²/d(rbx)d(rby)
            result.s2[16] = rt15 * 7.0 / 4.0 * sf.rbx() * sf.rby(); // d²/d(rbz)²
            result.s2[18] = rt15 * (7.0 * sf.ray() * sf.rby() + 1.0) / 4.0; // d²/d(ray)d(rbz)
            result.s2[19] = rt15 * (7.0 * sf.ray() * sf.rbx()) / 4.0; // d²/d(rby)d(rbz)
        }
    }
}

/**
 * Dipole-y × Octopole-33c kernel
 * Orient case 99: Q11s × Q33c
 * Formula: S0 = rt10*(7*rbx³*ray - 21*rbx*rby²*ray + 3*rbx²*cyx - 6*rbx*rby*cyy - 3*rby²*cyx)/16
 * For static multipoles (cyx=0, cyy=1): S0 = rt10*(7*rbx³*ray - 21*rbx*rby²*ray - 6*rbx*rby)/16
 * NO SCALING
 */
void dipole_y_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt10 *
                (7.0 * sf.rbx() * sf.rbx() * sf.rbx() * sf.ray() -
                 21.0 * sf.rbx() * sf.rby() * sf.rby() * sf.ray() -
                 6.0 * sf.rbx() * sf.rby()) /
                16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt10 * (7.0 * sf.rbx() * sf.rbx() * sf.rbx() - 21.0 * sf.rbx() * sf.rby() * sf.rby()) / 16.0; // d/d(ray)
        result.s1[3] = rt10 * (21.0 * sf.rbx() * sf.rbx() * sf.ray() - 21.0 * sf.rby() * sf.rby() * sf.ray() - 6.0 * sf.rby()) / 16.0; // d/d(rbx)
        result.s1[4] = rt10 * (-42.0 * sf.rbx() * sf.rby() * sf.ray() - 6.0 * sf.rbx()) / 16.0; // d/d(rby)

        if (level >= 2) {
            // Second derivatives (Orient case 99: S2(8), S2(10), S2(12), S2(14), S2(15))
            result.s2[7] = rt10 * (21.0 * sf.rbx() * sf.rbx() - 21.0 * sf.rby() * sf.rby()) / 16.0; // d²/d(ray)d(rbx)
            result.s2[9] = rt10 * 42.0 / 16.0 * sf.ray() * sf.rbx(); // d²/d(rbx)²
            result.s2[11] = rt10 * (-21.0) / 8.0 * sf.rbx() * sf.rby(); // d²/d(ray)d(rby)
            result.s2[13] = rt10 * (-42.0) / 16.0 * sf.ray() * sf.rby(); // d²/d(rbx)d(rby)
            result.s2[14] = rt10 * (-42.0) / 16.0 * sf.ray() * sf.rbx(); // d²/d(rby)d(rbx) [cross]
        }
    }
}

/**
 * Dipole-y × Octopole-33s kernel
 * Orient case 100: Q11s × Q33s
 * Formula: S0 = rt10*(21*rbx²*rby*ray - 7*rby³*ray + 3*rbx²*cyy + 6*rbx*rby*cyx - 3*rby²*cyy)/16
 * For static multipoles (cyx=0, cyy=1): S0 = rt10*(21*rbx²*rby*ray - 7*rby³*ray + 3*rbx² - 3*rby²)/16
 * NO SCALING
 */
void dipole_y_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = rt10 *
                (21.0 * sf.rbx() * sf.rbx() * sf.rby() * sf.ray() -
                 7.0 * sf.rby() * sf.rby() * sf.rby() * sf.ray() +
                 3.0 * sf.rbx() * sf.rbx() - 3.0 * sf.rby() * sf.rby()) /
                16.0;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt10 * (21.0 * sf.rbx() * sf.rbx() * sf.rby() - 7.0 * sf.rby() * sf.rby() * sf.rby()) / 16.0; // d/d(ray)
        result.s1[3] = rt10 * (42.0 * sf.rbx() * sf.rby() * sf.ray() + 6.0 * sf.rbx()) / 16.0; // d/d(rbx)
        result.s1[4] = rt10 * (21.0 * sf.rbx() * sf.rbx() * sf.ray() - 21.0 * sf.rby() * sf.rby() * sf.ray() - 6.0 * sf.rby()) / 16.0; // d/d(rby)

        if (level >= 2) {
            // Second derivatives (Orient case 100: S2(8), S2(10), S2(12), S2(14), S2(15))
            result.s2[7] = rt10 * 21.0 / 8.0 * sf.rbx() * sf.rby(); // d²/d(ray)d(rbx)
            result.s2[9] = rt10 * (42.0 * sf.ray() * sf.rby() + 6.0) / 16.0; // d²/d(rbx)d(rby)
            result.s2[11] = rt10 * (21.0 * sf.rbx() * sf.rbx() - 21.0 * sf.rby() * sf.rby()) / 16.0; // d²/d(ray)d(rby)
            result.s2[13] = rt10 * 42.0 / 16.0 * sf.ray() * sf.rbx(); // d²/d(rbx)d(rby)
            result.s2[14] = rt10 * (-42.0) / 16.0 * sf.ray() * sf.rby(); // d²/d(rby)²
        }
    }
}

// ============================================================================
// OCTOPOLE-DIPOLE KERNELS (Orient S4 subroutine)
// Octopole @ A (uses rax, ray, raz), Dipole @ B (uses rbx, rby, rbz)
// Orient uses DIFFERENT formulas than dipole-octopole, so can't simply swap!
// ============================================================================

// ============================================================================
// OCTOPOLE-DIPOLE KERNELS (Orient cases 126-146)
// Octopole @ A (uses rax, ray, raz), Dipole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Octopole-30 × Dipole-10 kernel
 * Orient case 126: Q30 × Q10
 */
void octopole_30_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = 35.0/8.0 * sf.raz()*sf.raz()*sf.raz() * sf.rbz() -
                15.0/8.0 * sf.raz() * sf.rbz() +
                15.0/8.0 * sf.raz()*sf.raz() * sf.czz() -
                3.0/8.0 * sf.czz();

    if (level >= 1) {
        result.s1[2] = 105.0/8.0 * sf.raz()*sf.raz() * sf.rbz() - 15.0/8.0 * sf.rbz() + 15.0/4.0 * sf.raz() * sf.czz(); // d/d(raz)
        result.s1[5] = 35.0/8.0 * sf.raz()*sf.raz()*sf.raz() - 15.0/8.0 * sf.raz(); // d/d(rbz)
        result.s1[14] = 15.0/8.0 * sf.raz()*sf.raz() - 3.0/8.0; // d/d(czz)

        if (level >= 2) {
            result.s2[5] = 105.0/4.0 * sf.raz() * sf.rbz() + 15.0/4.0 * sf.czz(); // d²/d(raz)²
            result.s2[17] = 105.0/8.0 * sf.raz()*sf.raz() - 15.0/8.0; // d²/d(raz)d(rbz)
            result.s2[107] = 15.0/4.0 * sf.raz(); // d²/d(raz)d(czz)
        }
    }
}

/**
 * Octopole-31c × Dipole-10 kernel
 * Orient case 129: Q31c × Q10
 */
void octopole_31c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = rt6 * (35.0 * sf.rax() * sf.raz()*sf.raz() * sf.rbz() -
                       5.0 * sf.rbz() * sf.rax() +
                       10.0 * sf.rax() * sf.raz() * sf.czz() +
                       5.0 * sf.raz()*sf.raz() * sf.cxz() -
                       sf.cxz()) / 16.0;

    if (level >= 1) {
        result.s1[0] = rt6 * (35.0 * sf.raz()*sf.raz() * sf.rbz() - 5.0 * sf.rbz() + 10.0 * sf.raz() * sf.czz()) / 16.0; // d/d(rax)
        result.s1[2] = rt6 * (70.0 * sf.rax() * sf.raz() * sf.rbz() + 10.0 * sf.rax() * sf.czz() + 10.0 * sf.raz() * sf.cxz()) / 16.0; // d/d(raz)
        result.s1[5] = rt6 * (35.0 * sf.rax() * sf.raz()*sf.raz() - 5.0 * sf.rax()) / 16.0; // d/d(rbz)
        result.s1[8] = rt6 * (5.0 * sf.raz()*sf.raz() - 1.0) / 16.0; // d/d(cxz)
        result.s1[14] = 5.0/8.0 * rt6 * sf.rax() * sf.raz(); // d/d(czz)

        if (level >= 2) {
            result.s2[3] = rt6 * (70.0 * sf.raz() * sf.rbz() + 10.0 * sf.czz()) / 16.0; // d²/d(rax)d(raz)
            result.s2[5] = rt6 * (70.0 * sf.rbz() * sf.rax() + 10.0 * sf.cxz()) / 16.0; // d²/d(raz)²
            result.s2[15] = rt6 * (35.0 * sf.raz()*sf.raz() - 5.0) / 16.0; // d²/d(rax)d(rbz)
            result.s2[17] = 35.0/8.0 * rt6 * sf.rax() * sf.raz(); // d²/d(raz)d(rbz)
            result.s2[80] = 5.0/8.0 * rt6 * sf.raz(); // d²/d(raz)d(cxz)
            result.s2[105] = 5.0/8.0 * rt6 * sf.raz(); // d²/d(rax)d(czz)
            result.s2[107] = 5.0/8.0 * rt6 * sf.rax(); // d²/d(raz)d(czz)
        }
    }
}

/**
 * Octopole-31s × Dipole-10 kernel
 * Orient case 132: Q31s × Q10
 */
void octopole_31s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = rt6 * (35.0 * sf.ray() * sf.raz()*sf.raz() * sf.rbz() -
                       5.0 * sf.rbz() * sf.ray() +
                       10.0 * sf.ray() * sf.raz() * sf.czz() +
                       5.0 * sf.raz()*sf.raz() * sf.cyz() -
                       sf.cyz()) / 16.0;

    if (level >= 1) {
        result.s1[1] = rt6 * (35.0 * sf.raz()*sf.raz() * sf.rbz() - 5.0 * sf.rbz() + 10.0 * sf.raz() * sf.czz()) / 16.0; // d/d(ray)
        result.s1[2] = rt6 * (70.0 * sf.ray() * sf.raz() * sf.rbz() + 10.0 * sf.ray() * sf.czz() + 10.0 * sf.raz() * sf.cyz()) / 16.0; // d/d(raz)
        result.s1[5] = rt6 * (35.0 * sf.ray() * sf.raz()*sf.raz() - 5.0 * sf.ray()) / 16.0; // d/d(rbz)
        result.s1[11] = rt6 * (5.0 * sf.raz()*sf.raz() - 1.0) / 16.0; // d/d(cyz)
        result.s1[14] = 5.0/8.0 * rt6 * sf.ray() * sf.raz(); // d/d(czz)

        if (level >= 2) {
            result.s2[4] = rt6 * (70.0 * sf.raz() * sf.rbz() + 10.0 * sf.czz()) / 16.0; // d²/d(ray)d(raz)
            result.s2[5] = rt6 * (70.0 * sf.rbz() * sf.ray() + 10.0 * sf.cyz()) / 16.0; // d²/d(raz)²
            result.s2[16] = rt6 * (35.0 * sf.raz()*sf.raz() - 5.0) / 16.0; // d²/d(ray)d(rbz)
            result.s2[17] = 35.0/8.0 * rt6 * sf.ray() * sf.raz(); // d²/d(raz)d(rbz)
            result.s2[93] = 5.0/8.0 * rt6 * sf.raz(); // d²/d(raz)d(cyz)
            result.s2[106] = 5.0/8.0 * rt6 * sf.raz(); // d²/d(ray)d(czz)
            result.s2[107] = 5.0/8.0 * rt6 * sf.ray(); // d²/d(raz)d(czz)
        }
    }
}

/**
 * Octopole-32c × Dipole-10 kernel
 * Orient case 135: Q32c × Q10
 */
void octopole_32c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = rt15 * (7.0 * sf.rax()*sf.rax() * sf.raz() * sf.rbz() -
                        7.0 * sf.ray()*sf.ray() * sf.raz() * sf.rbz() +
                        sf.rax()*sf.rax() * sf.czz() +
                        2.0 * sf.rax() * sf.raz() * sf.cxz() -
                        sf.ray()*sf.ray() * sf.czz() -
                        2.0 * sf.ray() * sf.raz() * sf.cyz()) / 8.0;

    if (level >= 1) {
        result.s1[0] = rt15 * (14.0 * sf.rax() * sf.raz() * sf.rbz() + 2.0 * sf.rax() * sf.czz() + 2.0 * sf.raz() * sf.cxz()) / 8.0; // d/d(rax)
        result.s1[1] = rt15 * (-14.0 * sf.ray() * sf.raz() * sf.rbz() - 2.0 * sf.ray() * sf.czz() - 2.0 * sf.raz() * sf.cyz()) / 8.0; // d/d(ray)
        result.s1[2] = rt15 * (7.0 * sf.rax()*sf.rax() * sf.rbz() - 7.0 * sf.ray()*sf.ray() * sf.rbz() + 2.0 * sf.rax() * sf.cxz() - 2.0 * sf.ray() * sf.cyz()) / 8.0; // d/d(raz)
        result.s1[5] = rt15 * (7.0 * sf.rax()*sf.rax() * sf.raz() - 7.0 * sf.ray()*sf.ray() * sf.raz()) / 8.0; // d/d(rbz)
        result.s1[8] = rt15 * sf.rax() * sf.raz() / 4.0; // d/d(cxz)
        result.s1[11] = -rt15 * sf.ray() * sf.raz() / 4.0; // d/d(cyz)
        result.s1[14] = rt15 * (sf.rax()*sf.rax() - sf.ray()*sf.ray()) / 8.0; // d/d(czz)

        if (level >= 2) {
            result.s2[0] = rt15 * (14.0 * sf.raz() * sf.rbz() + 2.0 * sf.czz()) / 8.0; // d²/d(rax)²
            result.s2[2] = rt15 * (-14.0 * sf.raz() * sf.rbz() - 2.0 * sf.czz()) / 8.0; // d²/d(ray)²
            result.s2[3] = rt15 * (14.0 * sf.rbz() * sf.rax() + 2.0 * sf.cxz()) / 8.0; // d²/d(rax)d(raz)
            result.s2[4] = rt15 * (-14.0 * sf.rbz() * sf.ray() - 2.0 * sf.cyz()) / 8.0; // d²/d(ray)d(raz)
            result.s2[15] = 7.0/4.0 * rt15 * sf.rax() * sf.raz(); // d²/d(rax)d(rbz)
            result.s2[16] = -7.0/4.0 * rt15 * sf.ray() * sf.raz(); // d²/d(ray)d(rbz)
            result.s2[17] = rt15 * (7.0 * sf.rax()*sf.rax() - 7.0 * sf.ray()*sf.ray()) / 8.0; // d²/d(raz)d(rbz)
            result.s2[78] = rt15 * sf.raz() / 4.0; // d²/d(rax)d(cxz)
            result.s2[80] = rt15 * sf.rax() / 4.0; // d²/d(raz)d(cxz)
            result.s2[92] = -rt15 * sf.raz() / 4.0; // d²/d(ray)d(cyz)
            result.s2[93] = -rt15 * sf.ray() / 4.0; // d²/d(raz)d(cyz)
            result.s2[105] = rt15 * sf.rax() / 4.0; // d²/d(rax)d(czz)
            result.s2[106] = -rt15 * sf.ray() / 4.0; // d²/d(ray)d(czz)
        }
    }
}

/**
 * Octopole-32s × Dipole-10 kernel
 * Orient case 138: Q32s × Q10
 */
void octopole_32s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = rt15 * (7.0 * sf.rax() * sf.ray() * sf.raz() * sf.rbz() +
                        sf.rax() * sf.ray() * sf.czz() +
                        sf.rax() * sf.raz() * sf.cyz() +
                        sf.ray() * sf.raz() * sf.cxz()) / 4.0;

    if (level >= 1) {
        result.s1[0] = rt15 * (7.0 * sf.ray() * sf.raz() * sf.rbz() + sf.ray() * sf.czz() + sf.raz() * sf.cyz()) / 4.0; // d/d(rax)
        result.s1[1] = rt15 * (7.0 * sf.rax() * sf.raz() * sf.rbz() + sf.rax() * sf.czz() + sf.raz() * sf.cxz()) / 4.0; // d/d(ray)
        result.s1[2] = rt15 * (7.0 * sf.rax() * sf.ray() * sf.rbz() + sf.rax() * sf.cyz() + sf.ray() * sf.cxz()) / 4.0; // d/d(raz)
        result.s1[5] = 7.0/4.0 * rt15 * sf.rax() * sf.ray() * sf.raz(); // d/d(rbz)
        result.s1[8] = rt15 * sf.ray() * sf.raz() / 4.0; // d/d(cxz)
        result.s1[11] = rt15 * sf.rax() * sf.raz() / 4.0; // d/d(cyz)
        result.s1[14] = rt15 * sf.rax() * sf.ray() / 4.0; // d/d(czz)

        if (level >= 2) {
            result.s2[1] = rt15 * (7.0 * sf.raz() * sf.rbz() + sf.czz()) / 4.0; // d²/d(rax)d(ray)
            result.s2[3] = rt15 * (7.0 * sf.rbz() * sf.ray() + sf.cyz()) / 4.0; // d²/d(rax)d(raz)
            result.s2[4] = rt15 * (7.0 * sf.rbz() * sf.rax() + sf.cxz()) / 4.0; // d²/d(ray)d(raz)
            result.s2[15] = 7.0/4.0 * rt15 * sf.ray() * sf.raz(); // d²/d(rax)d(rbz)
            result.s2[16] = 7.0/4.0 * rt15 * sf.rax() * sf.raz(); // d²/d(ray)d(rbz)
            result.s2[17] = 7.0/4.0 * rt15 * sf.rax() * sf.ray(); // d²/d(raz)d(rbz)
            result.s2[79] = rt15 * sf.raz() / 4.0; // d²/d(rax)d(cxz)
            result.s2[80] = rt15 * sf.ray() / 4.0; // d²/d(raz)d(cxz)
            result.s2[91] = rt15 * sf.raz() / 4.0; // d²/d(ray)d(cyz)
            result.s2[93] = rt15 * sf.rax() / 4.0; // d²/d(raz)d(cyz)
            result.s2[105] = rt15 * sf.ray() / 4.0; // d²/d(rax)d(czz)
            result.s2[106] = rt15 * sf.rax() / 4.0; // d²/d(ray)d(czz)
        }
    }
}

/**
 * Octopole-33c × Dipole-10 kernel
 * Orient case 141: Q33c × Q10
 */
void octopole_33c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = rt10 * (7.0 * sf.rax()*sf.rax()*sf.rax() * sf.rbz() -
                        21.0 * sf.rax() * sf.ray()*sf.ray() * sf.rbz() +
                        3.0 * sf.rax()*sf.rax() * sf.cxz() -
                        6.0 * sf.rax() * sf.ray() * sf.cyz() -
                        3.0 * sf.ray()*sf.ray() * sf.cxz()) / 16.0;

    if (level >= 1) {
        result.s1[0] = rt10 * (21.0 * sf.rax()*sf.rax() * sf.rbz() - 21.0 * sf.ray()*sf.ray() * sf.rbz() + 6.0 * sf.rax() * sf.cxz() - 6.0 * sf.ray() * sf.cyz()) / 16.0; // d/d(rax)
        result.s1[1] = rt10 * (-42.0 * sf.rax() * sf.ray() * sf.rbz() - 6.0 * sf.rax() * sf.cyz() - 6.0 * sf.ray() * sf.cxz()) / 16.0; // d/d(ray)
        result.s1[5] = rt10 * (7.0 * sf.rax()*sf.rax()*sf.rax() - 21.0 * sf.rax() * sf.ray()*sf.ray()) / 16.0; // d/d(rbz)
        result.s1[8] = rt10 * (3.0 * sf.rax()*sf.rax() - 3.0 * sf.ray()*sf.ray()) / 16.0; // d/d(cxz)
        result.s1[11] = -3.0/8.0 * rt10 * sf.rax() * sf.ray(); // d/d(cyz)

        if (level >= 2) {
            result.s2[0] = rt10 * (42.0 * sf.rbz() * sf.rax() + 6.0 * sf.cxz()) / 16.0; // d²/d(rax)²
            result.s2[1] = rt10 * (-42.0 * sf.rbz() * sf.ray() - 6.0 * sf.cyz()) / 16.0; // d²/d(rax)d(ray)
            result.s2[2] = rt10 * (-42.0 * sf.rbz() * sf.rax() - 6.0 * sf.cxz()) / 16.0; // d²/d(ray)²
            result.s2[15] = rt10 * (21.0 * sf.rax()*sf.rax() - 21.0 * sf.ray()*sf.ray()) / 16.0; // d²/d(rax)d(rbz)
            result.s2[16] = -21.0/8.0 * rt10 * sf.rax() * sf.ray(); // d²/d(ray)d(rbz)
            result.s2[78] = 3.0/8.0 * rt10 * sf.rax(); // d²/d(rax)d(cxz)
            result.s2[79] = -3.0/8.0 * rt10 * sf.ray(); // d²/d(rax)d(cyz)
            result.s2[91] = -3.0/8.0 * rt10 * sf.ray(); // d²/d(ray)d(cyz)
            result.s2[92] = -3.0/8.0 * rt10 * sf.rax(); // d²/d(ray)d(cxz)
        }
    }
}

/**
 * Octopole-33c × Dipole-11c kernel
 * Orient case 142: Q33c × Q11c (cxx=1, cyx=0)
 */
void octopole_33c_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term - INCLUDES C-TERMS
    result.s0 = rt10 * (7.0 * sf.rax()*sf.rax()*sf.rax() * sf.rbx() -
                        21.0 * sf.rax() * sf.ray()*sf.ray() * sf.rbx() +
                        3.0 * sf.rax()*sf.rax() * sf.cxx() -
                        6.0 * sf.rax() * sf.ray() * sf.cyx() -
                        3.0 * sf.ray()*sf.ray() * sf.cxx()) / 16.0;

    if (level >= 1) {
        result.s1[0] = rt10 * (21.0 * sf.rax()*sf.rax() * sf.rbx() - 21.0 * sf.ray()*sf.ray() * sf.rbx() + 6.0 * sf.rax() * sf.cxx() - 6.0 * sf.ray() * sf.cyx()) / 16.0; // d/d(rax)
        result.s1[1] = rt10 * (-42.0 * sf.rax() * sf.ray() * sf.rbx() - 6.0 * sf.rax() * sf.cyx() - 6.0 * sf.ray() * sf.cxx()) / 16.0; // d/d(ray)
        result.s1[3] = rt10 * (7.0 * sf.rax()*sf.rax()*sf.rax() - 21.0 * sf.rax() * sf.ray()*sf.ray()) / 16.0; // d/d(rbx)
        result.s1[6] = rt10 * (3.0 * sf.rax()*sf.rax() - 3.0 * sf.ray()*sf.ray()) / 16.0; // d/d(cxx)
        result.s1[9] = -3.0/8.0 * rt10 * sf.rax() * sf.ray(); // d/d(cyx)

        if (level >= 2) {
            result.s2[0] = rt10 * (42.0 * sf.rax() * sf.rbx() + 6.0 * sf.cxx()) / 16.0; // d²/d(rax)²
            result.s2[1] = rt10 * (-42.0 * sf.rbx() * sf.ray() - 6.0 * sf.cyx()) / 16.0; // d²/d(rax)d(ray)
            result.s2[2] = rt10 * (-42.0 * sf.rax() * sf.rbx() - 6.0 * sf.cxx()) / 16.0; // d²/d(ray)²
            result.s2[6] = rt10 * (21.0 * sf.rax()*sf.rax() - 21.0 * sf.ray()*sf.ray()) / 16.0; // d²/d(rax)d(rbx)
            result.s2[7] = -21.0/8.0 * rt10 * sf.rax() * sf.ray(); // d²/d(ray)d(rbx)
            result.s2[21] = 3.0/8.0 * rt10 * sf.rax(); // d²/d(rax)d(cxx)
            result.s2[22] = -3.0/8.0 * rt10 * sf.ray(); // d²/d(rax)d(cyx)
            result.s2[28] = -3.0/8.0 * rt10 * sf.ray(); // d²/d(ray)d(cyx)
            result.s2[29] = -3.0/8.0 * rt10 * sf.rax(); // d²/d(ray)d(cxx)
        }
    }
}

/**
 * Octopole-33c × Dipole-11s kernel
 * Orient case 143: Q33c × Q11s
 */
void octopole_33c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = rt10 * (7.0 * sf.rax()*sf.rax()*sf.rax() * sf.rby() -
                        21.0 * sf.rax() * sf.ray()*sf.ray() * sf.rby() +
                        3.0 * sf.rax()*sf.rax() * sf.cxy() -
                        6.0 * sf.rax() * sf.ray() * sf.cyy() -
                        3.0 * sf.ray()*sf.ray() * sf.cxy()) / 16.0;

    if (level >= 1) {
        result.s1[0] = rt10 * (21.0 * sf.rax()*sf.rax() * sf.rby() - 21.0 * sf.ray()*sf.ray() * sf.rby() + 6.0 * sf.rax() * sf.cxy() - 6.0 * sf.ray() * sf.cyy()) / 16.0; // d/d(rax)
        result.s1[1] = rt10 * (-42.0 * sf.rax() * sf.ray() * sf.rby() - 6.0 * sf.rax() * sf.cyy() - 6.0 * sf.ray() * sf.cxy()) / 16.0; // d/d(ray)
        result.s1[4] = rt10 * (7.0 * sf.rax()*sf.rax()*sf.rax() - 21.0 * sf.rax() * sf.ray()*sf.ray()) / 16.0; // d/d(rby)
        result.s1[7] = rt10 * (3.0 * sf.rax()*sf.rax() - 3.0 * sf.ray()*sf.ray()) / 16.0; // d/d(cxy)
        result.s1[10] = -3.0/8.0 * rt10 * sf.rax() * sf.ray(); // d/d(cyy)

        if (level >= 2) {
            result.s2[0] = rt10 * (42.0 * sf.rax() * sf.rby() + 6.0 * sf.cxy()) / 16.0; // d²/d(rax)²
            result.s2[1] = rt10 * (-42.0 * sf.ray() * sf.rby() - 6.0 * sf.cyy()) / 16.0; // d²/d(rax)d(ray)
            result.s2[2] = rt10 * (-42.0 * sf.rax() * sf.rby() - 6.0 * sf.cxy()) / 16.0; // d²/d(ray)²
            result.s2[10] = rt10 * (21.0 * sf.rax()*sf.rax() - 21.0 * sf.ray()*sf.ray()) / 16.0; // d²/d(rax)d(rby)
            result.s2[11] = -21.0/8.0 * rt10 * sf.rax() * sf.ray(); // d²/d(ray)d(rby)
            result.s2[45] = 3.0/8.0 * rt10 * sf.rax(); // d²/d(rax)d(cxy)
            result.s2[46] = -3.0/8.0 * rt10 * sf.ray(); // d²/d(rax)d(cyy)
            result.s2[55] = -3.0/8.0 * rt10 * sf.ray(); // d²/d(ray)d(cyy)
            result.s2[56] = -3.0/8.0 * rt10 * sf.rax(); // d²/d(ray)d(cxy)
        }
    }
}

/**
 * Octopole-33s × Dipole-10 kernel
 * Orient case 144: Q33s × Q10
 */
void octopole_33s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = rt10 * (21.0 * sf.rax()*sf.rax() * sf.ray() * sf.rbz() -
                        7.0 * sf.ray()*sf.ray()*sf.ray() * sf.rbz() +
                        3.0 * sf.rax()*sf.rax() * sf.cyz() +
                        6.0 * sf.rax() * sf.ray() * sf.cxz() -
                        3.0 * sf.ray()*sf.ray() * sf.cyz()) / 16.0;

    if (level >= 1) {
        result.s1[0] = rt10 * (42.0 * sf.rax() * sf.ray() * sf.rbz() + 6.0 * sf.rax() * sf.cyz() + 6.0 * sf.ray() * sf.cxz()) / 16.0; // d/d(rax)
        result.s1[1] = rt10 * (21.0 * sf.rax()*sf.rax() * sf.rbz() - 21.0 * sf.ray()*sf.ray() * sf.rbz() + 6.0 * sf.rax() * sf.cxz() - 6.0 * sf.ray() * sf.cyz()) / 16.0; // d/d(ray)
        result.s1[5] = rt10 * (21.0 * sf.rax()*sf.rax() * sf.ray() - 7.0 * sf.ray()*sf.ray()*sf.ray()) / 16.0; // d/d(rbz)
        result.s1[8] = 3.0/8.0 * rt10 * sf.rax() * sf.ray(); // d/d(cxz)
        result.s1[11] = rt10 * (3.0 * sf.rax()*sf.rax() - 3.0 * sf.ray()*sf.ray()) / 16.0; // d/d(cyz)

        if (level >= 2) {
            result.s2[0] = rt10 * (42.0 * sf.rbz() * sf.ray() + 6.0 * sf.cyz()) / 16.0; // d²/d(rax)²
            result.s2[1] = rt10 * (42.0 * sf.rbz() * sf.rax() + 6.0 * sf.cxz()) / 16.0; // d²/d(rax)d(ray)
            result.s2[2] = rt10 * (-42.0 * sf.rbz() * sf.ray() - 6.0 * sf.cyz()) / 16.0; // d²/d(ray)²
            result.s2[15] = 21.0/8.0 * rt10 * sf.rax() * sf.ray(); // d²/d(rax)d(rbz)
            result.s2[16] = rt10 * (21.0 * sf.rax()*sf.rax() - 21.0 * sf.ray()*sf.ray()) / 16.0; // d²/d(ray)d(rbz)
            result.s2[78] = 3.0/8.0 * rt10 * sf.ray(); // d²/d(rax)d(cxz)
            result.s2[79] = 3.0/8.0 * rt10 * sf.rax(); // d²/d(rax)d(cyz)
            result.s2[91] = 3.0/8.0 * rt10 * sf.rax(); // d²/d(ray)d(cxz)
            result.s2[92] = -3.0/8.0 * rt10 * sf.ray(); // d²/d(ray)d(cyz)
        }
    }
}

/**
 * Octopole-33s × Dipole-11c kernel
 * Orient case 145: Q33s × Q11c
 */
void octopole_33s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = rt10 * (21.0 * sf.rax()*sf.rax() * sf.ray() * sf.rbx() -
                        7.0 * sf.ray()*sf.ray()*sf.ray() * sf.rbx() +
                        3.0 * sf.rax()*sf.rax() * sf.cyx() +
                        6.0 * sf.rax() * sf.ray() * sf.cxx() -
                        3.0 * sf.ray()*sf.ray() * sf.cyx()) / 16.0;

    if (level >= 1) {
        result.s1[0] = rt10 * (42.0 * sf.rax() * sf.ray() * sf.rbx() + 6.0 * sf.rax() * sf.cyx() + 6.0 * sf.ray() * sf.cxx()) / 16.0; // d/d(rax)
        result.s1[1] = rt10 * (21.0 * sf.rax()*sf.rax() * sf.rbx() - 21.0 * sf.ray()*sf.ray() * sf.rbx() + 6.0 * sf.rax() * sf.cxx() - 6.0 * sf.ray() * sf.cyx()) / 16.0; // d/d(ray)
        result.s1[3] = rt10 * (21.0 * sf.rax()*sf.rax() * sf.ray() - 7.0 * sf.ray()*sf.ray()*sf.ray()) / 16.0; // d/d(rbx)
        result.s1[6] = 3.0/8.0 * rt10 * sf.rax() * sf.ray(); // d/d(cxx)
        result.s1[9] = rt10 * (3.0 * sf.rax()*sf.rax() - 3.0 * sf.ray()*sf.ray()) / 16.0; // d/d(cyx)

        if (level >= 2) {
            result.s2[0] = rt10 * (42.0 * sf.rbx() * sf.ray() + 6.0 * sf.cyx()) / 16.0; // d²/d(rax)²
            result.s2[1] = rt10 * (42.0 * sf.rax() * sf.rbx() + 6.0 * sf.cxx()) / 16.0; // d²/d(rax)d(ray)
            result.s2[2] = rt10 * (-42.0 * sf.rbx() * sf.ray() - 6.0 * sf.cyx()) / 16.0; // d²/d(ray)²
            result.s2[6] = 21.0/8.0 * rt10 * sf.rax() * sf.ray(); // d²/d(rax)d(rbx)
            result.s2[7] = rt10 * (21.0 * sf.rax()*sf.rax() - 21.0 * sf.ray()*sf.ray()) / 16.0; // d²/d(ray)d(rbx)
            result.s2[21] = 3.0/8.0 * rt10 * sf.ray(); // d²/d(rax)d(cxx)
            result.s2[22] = 3.0/8.0 * rt10 * sf.rax(); // d²/d(rax)d(cyx)
            result.s2[28] = 3.0/8.0 * rt10 * sf.rax(); // d²/d(ray)d(cxx)
            result.s2[29] = -3.0/8.0 * rt10 * sf.ray(); // d²/d(ray)d(cyx)
        }
    }
}

/**
 * Octopole-33s × Dipole-11s kernel
 * Orient case 146: Q33s × Q11s
 */
void octopole_33s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    // S0 term
    result.s0 = rt10 * (21.0 * sf.rax()*sf.rax() * sf.ray() * sf.rby() -
                        7.0 * sf.ray()*sf.ray()*sf.ray() * sf.rby() +
                        3.0 * sf.rax()*sf.rax() * sf.cyy() +
                        6.0 * sf.rax() * sf.ray() * sf.cxy() -
                        3.0 * sf.ray()*sf.ray() * sf.cyy()) / 16.0;

    if (level >= 1) {
        result.s1[0] = rt10 * (42.0 * sf.rax() * sf.ray() * sf.rby() + 6.0 * sf.rax() * sf.cyy() + 6.0 * sf.ray() * sf.cxy()) / 16.0; // d/d(rax)
        result.s1[1] = rt10 * (21.0 * sf.rax()*sf.rax() * sf.rby() - 21.0 * sf.ray()*sf.ray() * sf.rby() + 6.0 * sf.rax() * sf.cxy() - 6.0 * sf.ray() * sf.cyy()) / 16.0; // d/d(ray)
        result.s1[4] = rt10 * (21.0 * sf.rax()*sf.rax() * sf.ray() - 7.0 * sf.ray()*sf.ray()*sf.ray()) / 16.0; // d/d(rby)
        result.s1[7] = 3.0/8.0 * rt10 * sf.rax() * sf.ray(); // d/d(cxy)
        result.s1[10] = rt10 * (3.0 * sf.rax()*sf.rax() - 3.0 * sf.ray()*sf.ray()) / 16.0; // d/d(cyy)

        if (level >= 2) {
            result.s2[0] = rt10 * (42.0 * sf.ray() * sf.rby() + 6.0 * sf.cyy()) / 16.0; // d²/d(rax)²
            result.s2[1] = rt10 * (42.0 * sf.rax() * sf.rby() + 6.0 * sf.cxy()) / 16.0; // d²/d(rax)d(ray)
            result.s2[2] = rt10 * (-42.0 * sf.ray() * sf.rby() - 6.0 * sf.cyy()) / 16.0; // d²/d(ray)²
            result.s2[10] = 21.0/8.0 * rt10 * sf.rax() * sf.ray(); // d²/d(rax)d(rby)
            result.s2[11] = rt10 * (21.0 * sf.rax()*sf.rax() - 21.0 * sf.ray()*sf.ray()) / 16.0; // d²/d(ray)d(rby)
            result.s2[45] = 3.0/8.0 * rt10 * sf.ray(); // d²/d(rax)d(cxy)
            result.s2[46] = 3.0/8.0 * rt10 * sf.rax(); // d²/d(rax)d(cyy)
            result.s2[55] = 3.0/8.0 * rt10 * sf.rax(); // d²/d(ray)d(cxy)
            result.s2[56] = -3.0/8.0 * rt10 * sf.ray(); // d²/d(ray)d(cyy)
        }
    }
}

} // namespace occ::mults::kernels
