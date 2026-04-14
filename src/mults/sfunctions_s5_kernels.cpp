#include <occ/mults/sfunctions.h>
#include <cmath>

namespace occ::mults::kernels {

// Mathematical constants
constexpr double rt5 = 2.2360679774997896964;
constexpr double rt7 = 2.6457513110645905905;
constexpr double rt10 = 3.1622776601683793320;
constexpr double rt35 = 5.9160797830996161426;
constexpr double rt70 = 8.3666002653407554798;
constexpr double rt3 = 1.7320508075688772935;
constexpr double rt6 = 2.4494897427831780982;
constexpr double rt2 = 1.4142135623730950488;
constexpr double rt15 = 3.8729833462074168852;
constexpr double rt30 = 5.4772255750516611346;

// ============================================================================
// DIPOLE-HEXADECAPOLE KERNELS (Orient cases 167-193)
// Dipole @ A (uses rax, ray, raz), Hexadecapole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Dipole-z × Hexadecapole-40 kernel
 * Orient case 167: Q10 × Q40
 */
void dipole_z_hexadecapole_40(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        63.0 / 8.0 * rbz_val * rbz_val * rbz_val * rbz_val * raz_val -
        21.0 / 4.0 * rbz_val * rbz_val * raz_val +
        7.0 / 2.0 * rbz_val * rbz_val * rbz_val * sf.czz() +
        3.0 / 8.0 * raz_val - 3.0 / 2.0 * rbz_val * sf.czz();

    if (level >= 1) {
        // First derivatives
        result.s1[2] = 63.0 / 8.0 * rbz_val * rbz_val * rbz_val * rbz_val - 21.0 / 4.0 * rbz_val * rbz_val + 3.0 / 8.0; // d/d(raz)
        result.s1[5] = 252.0 / 8.0 * rbz_val * rbz_val * rbz_val * raz_val - 42.0 / 4.0 * rbz_val * raz_val + 21.0 / 2.0 * rbz_val * rbz_val * sf.czz() - 3.0 / 2.0 * sf.czz(); // d/d(rbz)
        result.s1[14] = 7.0 / 2.0 * rbz_val * rbz_val * rbz_val - 3.0 / 2.0 * rbz_val; // d/d(czz)

        if (level >= 2) {
            // Second derivatives - Orient case 167
            result.s2[17] = 63.0 / 2.0 * rbz_val * rbz_val * rbz_val - 21.0 / 2.0 * rbz_val; // d²/d(rbz)²
            result.s2[20] = 189.0 / 2.0 * rbz_val * rbz_val * raz_val - 21.0 / 2.0 * raz_val + 21.0 * rbz_val * sf.czz(); // d²/d(rbz)d(raz)
            result.s2[110] = 21.0 / 2.0 * rbz_val * rbz_val - 3.0 / 2.0; // d²/d(czz)²
        }
    }
}

/**
 * Dipole-z × Hexadecapole-41c kernel
 * Orient case 168: Q10 × Q41c
 */
void dipole_z_hexadecapole_41c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt10 *
                (63 * rbx_val * rbz_val * rbz_val * rbz_val * raz_val -
                 21 * rbx_val * rbz_val * raz_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt10 * (63 * rbx_val * rbz_val * rbz_val * rbz_val - 21 * rbx_val * rbz_val) / 20; // d/d(raz)
        result.s1[3] = rt10 * (63 * rbz_val * rbz_val * rbz_val * raz_val - 21 * rbz_val * raz_val + 21 * rbz_val * rbz_val * sf.czz() - 3 * sf.czz()) / 20; // d/d(rbx)
        result.s1[5] = rt10 * (189 * rbx_val * rbz_val * rbz_val * raz_val - 21 * rbx_val * raz_val + 42 * rbx_val * rbz_val * sf.czz() + 21 * rbz_val * rbz_val * sf.czx() - 3 * sf.czx()) / 20; // d/d(rbz)
        result.s1[8] = rt10 * (7 * rbz_val * rbz_val * rbz_val - 3 * rbz_val) / 20; // d/d(czx)
        result.s1[14] = rt10 * (21 * rbx_val * rbz_val * rbz_val - 3 * rbx_val) / 20; // d/d(czz)

        if (level >= 2) {
            // Second derivatives - Orient case 168
            result.s2[8] = rt10 * (63 * rbz_val * rbz_val * rbz_val - 21 * rbz_val) / 20; // d²/d(rbx)²
            result.s2[17] = rt10 * (189 * rbx_val * rbz_val * rbz_val - 21 * rbx_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt10 * (189 * rbz_val * rbz_val * raz_val - 21 * raz_val + 42 * rbz_val * sf.czz()) / 20; // d²/d(rbz)d(rbx)
            result.s2[20] = rt10 * (378 * rbx_val * rbz_val * raz_val + 42 * rbx_val * sf.czz() + 42 * rbz_val * sf.czx()) / 20; // d²/d(rbz)d(raz)
            result.s2[41] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(czx)²
            result.s2[108] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(czx)d(czz)
            result.s2[110] = 21.0 / 10.0 * rt10 * rbx_val * rbz_val; // d²/d(czz)²
        }
    }
}

/**
 * Dipole-z × Hexadecapole-41s kernel
 * Orient case 169: Q10 × Q41s
 */
void dipole_z_hexadecapole_41s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt10 *
                (63 * rby_val * rbz_val * rbz_val * rbz_val * raz_val -
                 21 * rby_val * rbz_val * raz_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt10 * (63 * rby_val * rbz_val * rbz_val * rbz_val - 21 * rby_val * rbz_val) / 20; // d/d(raz)
        result.s1[4] = rt10 * (63 * rbz_val * rbz_val * rbz_val * raz_val - 21 * rbz_val * raz_val + 21 * rbz_val * rbz_val * sf.czz() - 3 * sf.czz()) / 20; // d/d(rby)
        result.s1[5] = rt10 * (189 * rby_val * rbz_val * rbz_val * raz_val - 21 * rby_val * raz_val + 42 * rby_val * rbz_val * sf.czz() + 21 * rbz_val * rbz_val * sf.czy() - 3 * sf.czy()) / 20; // d/d(rbz)
        result.s1[11] = rt10 * (7 * rbz_val * rbz_val * rbz_val - 3 * rbz_val) / 20; // d/d(czy)
        result.s1[14] = rt10 * (21 * rby_val * rbz_val * rbz_val - 3 * rby_val) / 20; // d/d(czz)

        if (level >= 2) {
            // Second derivatives - Orient case 169
            result.s2[12] = rt10 * (63 * rbz_val * rbz_val * rbz_val - 21 * rbz_val) / 20; // d²/d(rby)²
            result.s2[17] = rt10 * (189 * rby_val * rbz_val * rbz_val - 21 * rby_val) / 20; // d²/d(rbz)²
            result.s2[19] = rt10 * (189 * rbz_val * rbz_val * raz_val - 21 * raz_val + 42 * rbz_val * sf.czz()) / 20; // d²/d(rbz)d(rby)
            result.s2[20] = rt10 * (378 * rby_val * rbz_val * raz_val + 42 * rby_val * sf.czz() + 42 * rbz_val * sf.czy()) / 20; // d²/d(rbz)d(raz)
            result.s2[71] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(czy)²
            result.s2[109] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(czy)d(czz)
            result.s2[110] = 21.0 / 10.0 * rt10 * rby_val * rbz_val; // d²/d(czz)²
        }
    }
}

/**
 * Dipole-z × Hexadecapole-42c kernel
 * Orient case 170: Q10 × Q42c
 */
void dipole_z_hexadecapole_42c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rbx_val * rbz_val * rbz_val * raz_val -
                 63 * rby_val * rby_val * rbz_val * rbz_val * raz_val -
                 7 * rbx_val * rbx_val * raz_val +
                 7 * rby_val * rby_val * raz_val +
                 14 * rbx_val * rbx_val * rbz_val * sf.czz() +
                 14 * rbx_val * rbz_val * rbz_val * sf.czx() -
                 14 * rby_val * rby_val * rbz_val * sf.czz() -
                 14 * rby_val * rbz_val * rbz_val * sf.czy() -
                 2 * rbx_val * sf.czx() +
                 2 * rby_val * sf.czy()) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt5 * (63 * rbx_val * rbx_val * rbz_val * rbz_val - 63 * rby_val * rby_val * rbz_val * rbz_val - 7 * rbx_val * rbx_val + 7 * rby_val * rby_val) / 20; // d/d(raz)
        result.s1[3] = rt5 * (126 * rbx_val * rbz_val * rbz_val * raz_val - 14 * rbx_val * raz_val + 28 * rbx_val * rbz_val * sf.czz() + 14 * rbz_val * rbz_val * sf.czx() - 2 * sf.czx()) / 20; // d/d(rbx)
        result.s1[4] = rt5 * (-126 * rby_val * rbz_val * rbz_val * raz_val + 14 * rby_val * raz_val - 28 * rby_val * rbz_val * sf.czz() - 14 * rbz_val * rbz_val * sf.czy() + 2 * sf.czy()) / 20; // d/d(rby)
        result.s1[5] = rt5 * (126 * rbx_val * rbx_val * rbz_val * raz_val - 126 * rby_val * rby_val * rbz_val * raz_val + 14 * rbx_val * rbx_val * sf.czz() + 28 * rbx_val * rbz_val * sf.czx() - 14 * rby_val * rby_val * sf.czz() - 28 * rby_val * rbz_val * sf.czy()) / 20; // d/d(rbz)
        result.s1[8] = rt5 * (14 * rbx_val * rbz_val * rbz_val - 2 * rbx_val) / 20; // d/d(czx)
        result.s1[11] = rt5 * (-14 * rby_val * rbz_val * rbz_val + 2 * rby_val) / 20; // d/d(czy)
        result.s1[14] = rt5 * (14 * rbx_val * rbx_val * rbz_val - 14 * rby_val * rby_val * rbz_val) / 20; // d/d(czz)

        if (level >= 2) {
            // Second derivatives - Orient case 170
            result.s2[8] = rt5 * (126 * rbx_val * rbz_val * rbz_val - 14 * rbx_val) / 20; // d²/d(rbx)²
            result.s2[9] = rt5 * (126 * rbz_val * rbz_val * raz_val - 14 * raz_val + 28 * rbz_val * sf.czz()) / 20; // d²/d(rbx)d(rby)
            result.s2[12] = rt5 * (-126 * rby_val * rbz_val * rbz_val + 14 * rby_val) / 20; // d²/d(rby)²
            result.s2[14] = rt5 * (-126 * rbz_val * rbz_val * raz_val + 14 * raz_val - 28 * rbz_val * sf.czz()) / 20; // d²/d(rby)d(rbx)
            result.s2[17] = rt5 * (126 * rbx_val * rbx_val * rbz_val - 126 * rby_val * rby_val * rbz_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt5 * (252 * rbx_val * rbz_val * raz_val + 28 * rbx_val * sf.czz() + 28 * rbz_val * sf.czx()) / 20; // d²/d(rbz)d(rbx)
            result.s2[19] = rt5 * (-252 * rby_val * rbz_val * raz_val - 28 * rby_val * sf.czz() - 28 * rbz_val * sf.czy()) / 20; // d²/d(rbz)d(rby)
            result.s2[20] = rt5 * (126 * rbx_val * rbx_val * raz_val - 126 * rby_val * rby_val * raz_val + 28 * rbx_val * sf.czx() - 28 * rby_val * sf.czy()) / 20; // d²/d(rbz)d(raz)
            result.s2[39] = rt5 * (14 * rbz_val * rbz_val - 2) / 20; // d²/d(czx)²
            result.s2[41] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(czx)d(czz)
            result.s2[70] = rt5 * (-14 * rbz_val * rbz_val + 2) / 20; // d²/d(czy)²
            result.s2[71] = -7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(czy)d(czx)
            result.s2[108] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(czx)d(czz)
            result.s2[109] = -7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(czy)d(czz)
            result.s2[110] = rt5 * (14 * rbx_val * rbx_val - 14 * rby_val * rby_val) / 20; // d²/d(czz)²
        }
    }
}

/**
 * Dipole-z × Hexadecapole-42s kernel
 * Orient case 171: Q10 × Q42s
 */
void dipole_z_hexadecapole_42s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rby_val * rbz_val * rbz_val * raz_val -
                 7 * rbx_val * rby_val * raz_val +
                 14 * rbx_val * rby_val * rbz_val * sf.czz() +
                 7 * rbx_val * rbz_val * rbz_val * sf.czy() +
                 7 * rby_val * rbz_val * rbz_val * sf.czx() -
                 rbx_val * sf.czy() -
                 rby_val * sf.czx()) /
                10;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt5 * (63 * rbx_val * rby_val * rbz_val * rbz_val - 7 * rbx_val * rby_val) / 10; // d/d(raz)
        result.s1[3] = rt5 * (63 * rby_val * rbz_val * rbz_val * raz_val - 7 * rby_val * raz_val + 14 * rby_val * rbz_val * sf.czz() + 7 * rbz_val * rbz_val * sf.czy() - sf.czy()) / 10; // d/d(rbx)
        result.s1[4] = rt5 * (63 * rbx_val * rbz_val * rbz_val * raz_val - 7 * rbx_val * raz_val + 14 * rbx_val * rbz_val * sf.czz() + 7 * rbz_val * rbz_val * sf.czx() - sf.czx()) / 10; // d/d(rby)
        result.s1[5] = rt5 * (126 * rbx_val * rby_val * rbz_val * raz_val + 14 * rbx_val * rby_val * sf.czz() + 14 * rbx_val * rbz_val * sf.czy() + 14 * rby_val * rbz_val * sf.czx()) / 10; // d/d(rbz)
        result.s1[8] = rt5 * (7 * rby_val * rbz_val * rbz_val - rby_val) / 10; // d/d(czx)
        result.s1[11] = rt5 * (7 * rbx_val * rbz_val * rbz_val - rbx_val) / 10; // d/d(czy)
        result.s1[14] = 7.0 / 5.0 * rt5 * rbx_val * rby_val * rbz_val; // d/d(czz)

        if (level >= 2) {
            // Second derivatives - Orient case 171
            result.s2[8] = rt5 * (63 * rby_val * rbz_val * rbz_val - 7 * rby_val) / 10; // d²/d(rbx)²
            result.s2[12] = rt5 * (63 * rbx_val * rbz_val * rbz_val - 7 * rbx_val) / 10; // d²/d(rby)²
            result.s2[13] = rt5 * (63 * rbz_val * rbz_val * raz_val - 7 * raz_val + 14 * rbz_val * sf.czz()) / 10; // d²/d(rby)d(rbx)
            result.s2[17] = 63.0 / 5.0 * rt5 * rbx_val * rby_val * rbz_val; // d²/d(rbz)²
            result.s2[18] = rt5 * (126 * rby_val * rbz_val * raz_val + 14 * rby_val * sf.czz() + 14 * rbz_val * sf.czy()) / 10; // d²/d(rbz)d(rbx)
            result.s2[19] = rt5 * (126 * rbx_val * rbz_val * raz_val + 14 * rbx_val * sf.czz() + 14 * rbz_val * sf.czx()) / 10; // d²/d(rbz)d(rby)
            result.s2[20] = rt5 * (126 * rbx_val * rby_val * raz_val + 14 * rbx_val * sf.czy() + 14 * rby_val * sf.czx()) / 10; // d²/d(rbz)d(raz)
            result.s2[40] = rt5 * (7 * rbz_val * rbz_val - 1) / 10; // d²/d(czx)²
            result.s2[41] = 7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(czx)d(czz)
            result.s2[69] = rt5 * (7 * rbz_val * rbz_val - 1) / 10; // d²/d(czy)d(czx)
            result.s2[71] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(czy)d(czx)
            result.s2[108] = 7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(czx)d(czz)
            result.s2[109] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(czy)d(czz)
            result.s2[110] = 7.0 / 5.0 * rt5 * rbx_val * rby_val; // d²/d(czz)²
        }
    }
}

/**
 * Dipole-z × Hexadecapole-43c kernel
 * Orient case 172: Q10 × Q43c
 */
void dipole_z_hexadecapole_43c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt70 *
                (9 * rbx_val * rbx_val * rbx_val * rbz_val * raz_val -
                 27 * rbx_val * rby_val * rby_val * rbz_val * raz_val +
                 rbx_val * rbx_val * rbx_val * sf.czz() +
                 3 * rbx_val * rbx_val * rbz_val * sf.czx() -
                 3 * rbx_val * rby_val * rby_val * sf.czz() -
                 6 * rbx_val * rby_val * rbz_val * sf.czy() -
                 3 * rby_val * rby_val * rbz_val * sf.czx()) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt70 * (9 * rbx_val * rbx_val * rbx_val * rbz_val - 27 * rbx_val * rby_val * rby_val * rbz_val) / 20; // d/d(raz)
        result.s1[3] = rt70 * (27 * rbx_val * rbx_val * rbz_val * raz_val - 27 * rby_val * rby_val * rbz_val * raz_val + 3 * rbx_val * rbx_val * sf.czz() + 6 * rbx_val * rbz_val * sf.czx() - 3 * rby_val * rby_val * sf.czz() - 6 * rby_val * rbz_val * sf.czy()) / 20; // d/d(rbx)
        result.s1[4] = rt70 * (-54 * rbx_val * rby_val * rbz_val * raz_val - 6 * rbx_val * rby_val * sf.czz() - 6 * rbx_val * rbz_val * sf.czy() - 6 * rby_val * rbz_val * sf.czx()) / 20; // d/d(rby)
        result.s1[5] = rt70 * (9 * rbx_val * rbx_val * rbx_val * raz_val - 27 * rbx_val * rby_val * rby_val * raz_val + 3 * rbx_val * rbx_val * sf.czx() - 6 * rbx_val * rby_val * sf.czy() - 3 * rby_val * rby_val * sf.czx()) / 20; // d/d(rbz)
        result.s1[8] = rt70 * (3 * rbx_val * rbx_val * rbz_val - 3 * rby_val * rby_val * rbz_val) / 20; // d/d(czx)
        result.s1[11] = -3.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d/d(czy)
        result.s1[14] = rt70 * (rbx_val * rbx_val * rbx_val - 3 * rbx_val * rby_val * rby_val) / 20; // d/d(czz)

        if (level >= 2) {
            // Second derivatives - Orient case 172
            result.s2[8] = rt70 * (27 * rbx_val * rbx_val * rbz_val - 27 * rby_val * rby_val * rbz_val) / 20; // d²/d(rbx)²
            result.s2[9] = rt70 * (54 * rbx_val * rbz_val * raz_val + 6 * rbx_val * sf.czz() + 6 * rbz_val * sf.czx()) / 20; // d²/d(rbx)d(rby)
            result.s2[12] = -27.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d²/d(rby)²
            result.s2[13] = rt70 * (-54 * rby_val * rbz_val * raz_val - 6 * rby_val * sf.czz() - 6 * rbz_val * sf.czy()) / 20; // d²/d(rby)d(rbx)
            result.s2[14] = rt70 * (-54 * rbx_val * rbz_val * raz_val - 6 * rbx_val * sf.czz() - 6 * rbz_val * sf.czx()) / 20; // d²/d(rby)d(rbx) (symmetric)
            result.s2[17] = rt70 * (9 * rbx_val * rbx_val * rbx_val - 27 * rbx_val * rby_val * rby_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt70 * (27 * rbx_val * rbx_val * raz_val - 27 * rby_val * rby_val * raz_val + 6 * rbx_val * sf.czx() - 6 * rby_val * sf.czy()) / 20; // d²/d(rbz)d(rbx)
            result.s2[19] = rt70 * (-54 * rbx_val * rby_val * raz_val - 6 * rbx_val * sf.czy() - 6 * rby_val * sf.czx()) / 20; // d²/d(rbz)d(rby)
            result.s2[39] = 3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(czx)²
            result.s2[40] = -3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(czx)d(czy)
            result.s2[41] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(czx)d(czz)
            result.s2[69] = -3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(czy)d(czx)
            result.s2[70] = -3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(czy)²
            result.s2[71] = -3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(czy)d(czz)
            result.s2[108] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(czx)d(czz)
            result.s2[109] = -3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(czy)d(czz)
        }
    }
}

/**
 * Dipole-z × Hexadecapole-43s kernel
 * Orient case 173: Q10 × Q43s
 */
void dipole_z_hexadecapole_43s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt70 *
                (27 * rbx_val * rbx_val * rby_val * rbz_val * raz_val -
                 9 * rby_val * rby_val * rby_val * rbz_val * raz_val +
                 3 * rbx_val * rbx_val * rby_val * sf.czz() +
                 3 * rbx_val * rbx_val * rbz_val * sf.czy() +
                 6 * rbx_val * rby_val * rbz_val * sf.czx() -
                 rby_val * rby_val * rby_val * sf.czz() -
                 3 * rby_val * rby_val * rbz_val * sf.czy()) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt70 * (27 * rbx_val * rbx_val * rby_val * rbz_val - 9 * rby_val * rby_val * rby_val * rbz_val) / 20; // d/d(raz)
        result.s1[3] = rt70 * (54 * rbx_val * rby_val * rbz_val * raz_val + 6 * rbx_val * rby_val * sf.czz() + 6 * rbx_val * rbz_val * sf.czy() + 6 * rby_val * rbz_val * sf.czx()) / 20; // d/d(rbx)
        result.s1[4] = rt70 * (27 * rbx_val * rbx_val * rbz_val * raz_val - 27 * rby_val * rby_val * rbz_val * raz_val + 3 * rbx_val * rbx_val * sf.czz() + 6 * rbx_val * rbz_val * sf.czx() - 3 * rby_val * rby_val * sf.czz() - 6 * rby_val * rbz_val * sf.czy()) / 20; // d/d(rby)
        result.s1[5] = rt70 * (27 * rbx_val * rbx_val * rby_val * raz_val - 9 * rby_val * rby_val * rby_val * raz_val + 3 * rbx_val * rbx_val * sf.czy() + 6 * rbx_val * rby_val * sf.czx() - 3 * rby_val * rby_val * sf.czy()) / 20; // d/d(rbz)
        result.s1[8] = 3.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d/d(czx)
        result.s1[11] = rt70 * (3 * rbx_val * rbx_val * rbz_val - 3 * rby_val * rby_val * rbz_val) / 20; // d/d(czy)
        result.s1[14] = rt70 * (3 * rbx_val * rbx_val * rby_val - rby_val * rby_val * rby_val) / 20; // d/d(czz)

        if (level >= 2) {
            // Second derivatives - Orient case 173
            result.s2[8] = 27.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d²/d(rbx)²
            result.s2[9] = rt70 * (54 * rby_val * rbz_val * raz_val + 6 * rby_val * sf.czz() + 6 * rbz_val * sf.czy()) / 20; // d²/d(rbx)d(rby)
            result.s2[12] = rt70 * (27 * rbx_val * rbx_val * rbz_val - 27 * rby_val * rby_val * rbz_val) / 20; // d²/d(rby)²
            result.s2[13] = rt70 * (54 * rbx_val * rbz_val * raz_val + 6 * rbx_val * sf.czz() + 6 * rbz_val * sf.czx()) / 20; // d²/d(rby)d(rbx)
            result.s2[14] = rt70 * (-54 * rby_val * rbz_val * raz_val - 6 * rby_val * sf.czz() - 6 * rbz_val * sf.czy()) / 20; // d²/d(rby)d(rbx) (cross)
            result.s2[17] = rt70 * (27 * rbx_val * rbx_val * rby_val - 9 * rby_val * rby_val * rby_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt70 * (54 * rbx_val * rby_val * raz_val + 6 * rbx_val * sf.czy() + 6 * rby_val * sf.czx()) / 20; // d²/d(rbz)d(rbx)
            result.s2[19] = rt70 * (27 * rbx_val * rbx_val * raz_val - 27 * rby_val * rby_val * raz_val + 6 * rbx_val * sf.czx() - 6 * rby_val * sf.czy()) / 20; // d²/d(rbz)d(rby)
            result.s2[39] = 3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(czx)²
            result.s2[40] = 3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(czx)d(czy)
            result.s2[41] = 3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(czx)d(czz)
            result.s2[69] = 3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(czy)d(czx)
            result.s2[70] = -3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(czy)²
            result.s2[71] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(czy)d(czz)
            result.s2[108] = 3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(czx)d(czz)
            result.s2[109] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(czy)d(czz)
        }
    }
}

/**
 * Dipole-z × Hexadecapole-44c kernel
 * Orient case 174: Q10 × Q44c
 */
void dipole_z_hexadecapole_44c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt35 *
                (9 * rbx_val * rbx_val * rbx_val * rbx_val * raz_val -
                 54 * rbx_val * rbx_val * rby_val * rby_val * raz_val +
                 9 * rby_val * rby_val * rby_val * rby_val * raz_val +
                 4 * rbx_val * rbx_val * rbx_val * sf.czx() -
                 12 * rbx_val * rbx_val * rby_val * sf.czy() -
                 12 * rbx_val * rby_val * rby_val * sf.czx() +
                 4 * rby_val * rby_val * rby_val * sf.czy()) /
                40;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt35 * (9 * rbx_val * rbx_val * rbx_val * rbx_val - 54 * rbx_val * rbx_val * rby_val * rby_val + 9 * rby_val * rby_val * rby_val * rby_val) / 40; // d/d(raz)
        result.s1[3] = rt35 * (36 * rbx_val * rbx_val * rbx_val * raz_val - 108 * rbx_val * rby_val * rby_val * raz_val + 12 * rbx_val * rbx_val * sf.czx() - 24 * rbx_val * rby_val * sf.czy() - 12 * rby_val * rby_val * sf.czx()) / 40; // d/d(rbx)
        result.s1[4] = rt35 * (-108 * rbx_val * rbx_val * rby_val * raz_val + 36 * rby_val * rby_val * rby_val * raz_val - 12 * rbx_val * rbx_val * sf.czy() - 24 * rbx_val * rby_val * sf.czx() + 12 * rby_val * rby_val * sf.czy()) / 40; // d/d(rby)
        result.s1[8] = rt35 * (4 * rbx_val * rbx_val * rbx_val - 12 * rbx_val * rby_val * rby_val) / 40; // d/d(czx)
        result.s1[11] = rt35 * (-12 * rbx_val * rbx_val * rby_val + 4 * rby_val * rby_val * rby_val) / 40; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 174
            result.s2[8] = rt35 * (36 * rbx_val * rbx_val * rbx_val - 108 * rbx_val * rby_val * rby_val) / 40; // d²/d(rbx)²
            result.s2[9] = rt35 * (108 * rbx_val * rbx_val * raz_val - 108 * rby_val * rby_val * raz_val + 24 * rbx_val * sf.czx() - 24 * rby_val * sf.czy()) / 40; // d²/d(rbx)d(rby)
            result.s2[12] = rt35 * (-108 * rbx_val * rbx_val * rby_val + 36 * rby_val * rby_val * rby_val) / 40; // d²/d(rby)²
            result.s2[13] = rt35 * (-216 * rbx_val * rby_val * raz_val - 24 * rbx_val * sf.czy() - 24 * rby_val * sf.czx()) / 40; // d²/d(rby)d(rbx)
            result.s2[14] = rt35 * (-108 * rbx_val * rbx_val * raz_val + 108 * rby_val * rby_val * raz_val - 24 * rbx_val * sf.czx() + 24 * rby_val * sf.czy()) / 40; // d²/d(rby)d(rbx) (cross)
            result.s2[39] = rt35 * (12 * rbx_val * rbx_val - 12 * rby_val * rby_val) / 40; // d²/d(czx)²
            result.s2[40] = -3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(czx)d(czy)
            result.s2[69] = -3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(czy)d(czx)
            result.s2[70] = rt35 * (-12 * rbx_val * rbx_val + 12 * rby_val * rby_val) / 40; // d²/d(czy)²
        }
    }
}

/**
 * Dipole-z × Hexadecapole-44s kernel
 * Orient case 175: Q10 × Q44s
 */
void dipole_z_hexadecapole_44s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt35 *
                (9 * rbx_val * rbx_val * rbx_val * rby_val * raz_val -
                 9 * rbx_val * rby_val * rby_val * rby_val * raz_val +
                 rbx_val * rbx_val * rbx_val * sf.czy() +
                 3 * rbx_val * rbx_val * rby_val * sf.czx() -
                 3 * rbx_val * rby_val * rby_val * sf.czy() -
                 rby_val * rby_val * rby_val * sf.czx()) /
                10;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt35 * (9 * rbx_val * rbx_val * rbx_val * rby_val - 9 * rbx_val * rby_val * rby_val * rby_val) / 10; // d/d(raz)
        result.s1[3] = rt35 * (27 * rbx_val * rbx_val * rby_val * raz_val - 9 * rby_val * rby_val * rby_val * raz_val + 3 * rbx_val * rbx_val * sf.czy() + 6 * rbx_val * rby_val * sf.czx() - 3 * rby_val * rby_val * sf.czy()) / 10; // d/d(rbx)
        result.s1[4] = rt35 * (9 * rbx_val * rbx_val * rbx_val * raz_val - 27 * rbx_val * rby_val * rby_val * raz_val + 3 * rbx_val * rbx_val * sf.czx() - 6 * rbx_val * rby_val * sf.czy() - 3 * rby_val * rby_val * sf.czx()) / 10; // d/d(rby)
        result.s1[8] = rt35 * (3 * rbx_val * rbx_val * rby_val - rby_val * rby_val * rby_val) / 10; // d/d(czx)
        result.s1[11] = rt35 * (rbx_val * rbx_val * rbx_val - 3 * rbx_val * rby_val * rby_val) / 10; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 175
            result.s2[8] = rt35 * (27 * rbx_val * rbx_val * rby_val - 9 * rby_val * rby_val * rby_val) / 10; // d²/d(rbx)²
            result.s2[9] = rt35 * (54 * rbx_val * rby_val * raz_val + 6 * rbx_val * sf.czy() + 6 * rby_val * sf.czx()) / 10; // d²/d(rbx)d(rby)
            result.s2[12] = rt35 * (9 * rbx_val * rbx_val * rbx_val - 27 * rbx_val * rby_val * rby_val) / 10; // d²/d(rby)²
            result.s2[13] = rt35 * (27 * rbx_val * rbx_val * raz_val - 27 * rby_val * rby_val * raz_val + 6 * rbx_val * sf.czx() - 6 * rby_val * sf.czy()) / 10; // d²/d(rby)d(rbx)
            result.s2[14] = rt35 * (-54 * rbx_val * rby_val * raz_val - 6 * rbx_val * sf.czy() - 6 * rby_val * sf.czx()) / 10; // d²/d(rby)d(rbx) (cross)
            result.s2[39] = 3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(czx)²
            result.s2[40] = rt35 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 10; // d²/d(czx)d(czy)
            result.s2[69] = rt35 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 10; // d²/d(czy)d(czx)
            result.s2[70] = -3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(czy)²
        }
    }
}

/**
 * Dipole-x × Hexadecapole-40 kernel
 * Orient case 176: Q11c × Q40
 */
void dipole_x_hexadecapole_40(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        63.0 / 8.0 * rbz_val * rbz_val * rbz_val * rbz_val * rax_val -
        21.0 / 4.0 * rbz_val * rbz_val * rax_val + 3.0 / 8.0 * rax_val;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = 63.0 / 8.0 * rbz_val * rbz_val * rbz_val * rbz_val - 21.0 / 4.0 * rbz_val * rbz_val + 3.0 / 8.0; // d/d(rax)
        result.s1[5] = 252.0 / 8.0 * rbz_val * rbz_val * rbz_val * rax_val - 42.0 / 4.0 * rbz_val * rax_val + 21.0 / 2.0 * rbz_val * rbz_val * sf.cxz() - 3.0 / 2.0 * sf.cxz(); // d/d(rbz)
        result.s1[8] = 7.0 / 2.0 * rbz_val * rbz_val * rbz_val - 3.0 / 2.0 * rbz_val; // d/d(cxz)

        if (level >= 2) {
            // Second derivatives - Orient case 176
            result.s2[15] = 63.0 / 2.0 * rbz_val * rbz_val * rbz_val - 21.0 / 2.0 * rbz_val; // d²/d(rbz)²
            result.s2[20] = 189.0 / 2.0 * rbz_val * rbz_val * rax_val - 21.0 / 2.0 * rax_val + 21.0 * rbz_val * sf.cxz(); // d²/d(rbz)d(rax)
            result.s2[83] = 21.0 / 2.0 * rbz_val * rbz_val - 3.0 / 2.0; // d²/d(cxz)²
        }
    }
}

/**
 * Dipole-x × Hexadecapole-41c kernel
 * Orient case 177: Q11c × Q41c
 */
void dipole_x_hexadecapole_41c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt10 *
                (63 * rbx_val * rbz_val * rbz_val * rbz_val * rax_val -
                 21 * rbx_val * rbz_val * rax_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt10 * (63 * rbx_val * rbz_val * rbz_val * rbz_val - 21 * rbx_val * rbz_val) / 20; // d/d(rax)
        result.s1[3] = rt10 * (63 * rbz_val * rbz_val * rbz_val * rax_val - 21 * rbz_val * rax_val + 21 * rbz_val * rbz_val * sf.cxz() - 3 * sf.cxz()) / 20; // d/d(rbx)
        result.s1[5] = rt10 * (189 * rbx_val * rbz_val * rbz_val * rax_val - 21 * rbx_val * rax_val + 42 * rbx_val * rbz_val * sf.cxz() + 21 * rbz_val * rbz_val * sf.cxx() - 3 * sf.cxx()) / 20; // d/d(rbz)
        result.s1[6] = rt10 * (7 * rbz_val * rbz_val * rbz_val - 3 * rbz_val) / 20; // d/d(cxx)
        result.s1[8] = rt10 * (21 * rbx_val * rbz_val * rbz_val - 3 * rbx_val) / 20; // d/d(cxz)

        if (level >= 2) {
            // Second derivatives - Orient case 177
            result.s2[6] = rt10 * (63 * rbz_val * rbz_val * rbz_val - 21 * rbz_val) / 20; // d²/d(rbx)²
            result.s2[15] = rt10 * (189 * rbx_val * rbz_val * rbz_val - 21 * rbx_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt10 * (189 * rbz_val * rbz_val * rax_val - 21 * rax_val + 42 * rbz_val * sf.cxz()) / 20; // d²/d(rbz)d(rbx)
            result.s2[20] = rt10 * (378 * rbx_val * rbz_val * rax_val + 42 * rbx_val * sf.cxz() + 42 * rbz_val * sf.cxx()) / 20; // d²/d(rbz)d(rax)
            result.s2[26] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(cxx)²
            result.s2[81] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(cxx)d(cxz)
            result.s2[83] = 21.0 / 10.0 * rt10 * rbx_val * rbz_val; // d²/d(cxz)²
        }
    }
}

/**
 * Dipole-x × Hexadecapole-41s kernel
 * Orient case 178: Q11c × Q41s
 */
void dipole_x_hexadecapole_41s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt10 *
                (63 * rby_val * rbz_val * rbz_val * rbz_val * rax_val -
                 21 * rby_val * rbz_val * rax_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt10 * (63 * rby_val * rbz_val * rbz_val * rbz_val - 21 * rby_val * rbz_val) / 20; // d/d(rax)
        result.s1[4] = rt10 * (63 * rbz_val * rbz_val * rbz_val * rax_val - 21 * rbz_val * rax_val + 21 * rbz_val * rbz_val * sf.cxz() - 3 * sf.cxz()) / 20; // d/d(rby)
        result.s1[5] = rt10 * (189 * rby_val * rbz_val * rbz_val * rax_val - 21 * rby_val * rax_val + 42 * rby_val * rbz_val * sf.cxz() + 21 * rbz_val * rbz_val * sf.cxy() - 3 * sf.cxy()) / 20; // d/d(rbz)
        result.s1[7] = rt10 * (7 * rbz_val * rbz_val * rbz_val - 3 * rbz_val) / 20; // d/d(cxy)
        result.s1[8] = rt10 * (21 * rby_val * rbz_val * rbz_val - 3 * rby_val) / 20; // d/d(cxz)

        if (level >= 2) {
            // Second derivatives - Orient case 178
            result.s2[10] = rt10 * (63 * rbz_val * rbz_val * rbz_val - 21 * rbz_val) / 20; // d²/d(rby)²
            result.s2[15] = rt10 * (189 * rby_val * rbz_val * rbz_val - 21 * rby_val) / 20; // d²/d(rbz)²
            result.s2[19] = rt10 * (189 * rbz_val * rbz_val * rax_val - 21 * rax_val + 42 * rbz_val * sf.cxz()) / 20; // d²/d(rbz)d(rby)
            result.s2[20] = rt10 * (378 * rby_val * rbz_val * rax_val + 42 * rby_val * sf.cxz() + 42 * rbz_val * sf.cxy()) / 20; // d²/d(rbz)d(rax)
            result.s2[50] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(cxy)²
            result.s2[82] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(cxy)d(cxz)
            result.s2[83] = 21.0 / 10.0 * rt10 * rby_val * rbz_val; // d²/d(cxz)²
        }
    }
}

/**
 * Dipole-x × Hexadecapole-42c kernel
 * Orient case 179: Q11c × Q42c
 */
void dipole_x_hexadecapole_42c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rbx_val * rbz_val * rbz_val * rax_val -
                 63 * rby_val * rby_val * rbz_val * rbz_val * rax_val -
                 7 * rbx_val * rbx_val * rax_val +
                 7 * rby_val * rby_val * rax_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt5 * (63 * rbx_val * rbx_val * rbz_val * rbz_val - 63 * rby_val * rby_val * rbz_val * rbz_val - 7 * rbx_val * rbx_val + 7 * rby_val * rby_val) / 20; // d/d(rax)
        result.s1[3] = rt5 * (126 * rbx_val * rbz_val * rbz_val * rax_val - 14 * rbx_val * rax_val + 28 * rbx_val * rbz_val * sf.cxz() + 14 * rbz_val * rbz_val * sf.cxx() - 2 * sf.cxx()) / 20; // d/d(rbx)
        result.s1[4] = rt5 * (-126 * rby_val * rbz_val * rbz_val * rax_val + 14 * rby_val * rax_val - 28 * rby_val * rbz_val * sf.cxz() - 14 * rbz_val * rbz_val * sf.cxy() + 2 * sf.cxy()) / 20; // d/d(rby)
        result.s1[5] = rt5 * (126 * rbx_val * rbx_val * rbz_val * rax_val - 126 * rby_val * rby_val * rbz_val * rax_val + 14 * rbx_val * rbx_val * sf.cxz() + 28 * rbx_val * rbz_val * sf.cxx() - 14 * rby_val * rby_val * sf.cxz() - 28 * rby_val * rbz_val * sf.cxy()) / 20; // d/d(rbz)
        result.s1[6] = rt5 * (14 * rbx_val * rbz_val * rbz_val - 2 * rbx_val) / 20; // d/d(cxx)
        result.s1[7] = rt5 * (-14 * rby_val * rbz_val * rbz_val + 2 * rby_val) / 20; // d/d(cxy)
        result.s1[8] = rt5 * (14 * rbx_val * rbx_val * rbz_val - 14 * rby_val * rby_val * rbz_val) / 20; // d/d(cxz)

        if (level >= 2) {
            // Second derivatives - Orient case 179
            result.s2[6] = rt5 * (126 * rbx_val * rbz_val * rbz_val - 14 * rbx_val) / 20; // d²/d(rbx)²
            result.s2[9] = rt5 * (126 * rbz_val * rbz_val * rax_val - 14 * rax_val + 28 * rbz_val * sf.cxz()) / 20; // d²/d(rbx)d(rax)
            result.s2[10] = rt5 * (-126 * rby_val * rbz_val * rbz_val + 14 * rby_val) / 20; // d²/d(rby)²
            result.s2[14] = rt5 * (-126 * rbz_val * rbz_val * rax_val + 14 * rax_val - 28 * rbz_val * sf.cxz()) / 20; // d²/d(rby)d(rax)
            result.s2[15] = rt5 * (126 * rbx_val * rbx_val * rbz_val - 126 * rby_val * rby_val * rbz_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt5 * (252 * rbx_val * rbz_val * rax_val + 28 * rbx_val * sf.cxz() + 28 * rbz_val * sf.cxx()) / 20; // d²/d(rbz)d(rbx)
            result.s2[19] = rt5 * (-252 * rby_val * rbz_val * rax_val - 28 * rby_val * sf.cxz() - 28 * rbz_val * sf.cxy()) / 20; // d²/d(rbz)d(rby)
            result.s2[20] = rt5 * (126 * rbx_val * rbx_val * rax_val - 126 * rby_val * rby_val * rax_val + 28 * rbx_val * sf.cxx() - 28 * rby_val * sf.cxy()) / 20; // d²/d(rbz)d(rax)
            result.s2[24] = rt5 * (14 * rbz_val * rbz_val - 2) / 20; // d²/d(cxx)²
            result.s2[26] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(cxx)d(rbz)
            result.s2[49] = rt5 * (-14 * rbz_val * rbz_val + 2) / 20; // d²/d(cxy)²
            result.s2[50] = -7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(cxy)d(rbz)
            result.s2[81] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(cxx)d(cxz)
            result.s2[82] = -7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(cxy)d(cxz)
            result.s2[83] = rt5 * (14 * rbx_val * rbx_val - 14 * rby_val * rby_val) / 20; // d²/d(cxz)²
        }
    }
}

/**
 * Dipole-x × Hexadecapole-42s kernel
 * Orient case 180: Q11c × Q42s
 */
void dipole_x_hexadecapole_42s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rby_val * rbz_val * rbz_val * rax_val -
                 7 * rbx_val * rby_val * rax_val) /
                10;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt5 * (63 * rbx_val * rby_val * rbz_val * rbz_val - 7 * rbx_val * rby_val) / 10; // d/d(rax)
        result.s1[3] = rt5 * (63 * rby_val * rbz_val * rbz_val * rax_val - 7 * rby_val * rax_val + 14 * rby_val * rbz_val * sf.cxz() + 7 * rbz_val * rbz_val * sf.cxy() - sf.cxy()) / 10; // d/d(rbx)
        result.s1[4] = rt5 * (63 * rbx_val * rbz_val * rbz_val * rax_val - 7 * rbx_val * rax_val + 14 * rbx_val * rbz_val * sf.cxz() + 7 * rbz_val * rbz_val * sf.cxx() - sf.cxx()) / 10; // d/d(rby)
        result.s1[5] = rt5 * (126 * rbx_val * rby_val * rbz_val * rax_val + 14 * rbx_val * rby_val * sf.cxz() + 14 * rbx_val * rbz_val * sf.cxy() + 14 * rby_val * rbz_val * sf.cxx()) / 10; // d/d(rbz)
        result.s1[6] = rt5 * (7 * rby_val * rbz_val * rbz_val - rby_val) / 10; // d/d(cxx)
        result.s1[7] = rt5 * (7 * rbx_val * rbz_val * rbz_val - rbx_val) / 10; // d/d(cxy)
        result.s1[8] = 7.0 / 5.0 * rt5 * rbx_val * rby_val * rbz_val; // d/d(cxz)

        if (level >= 2) {
            // Second derivatives - Orient case 180
            result.s2[6] = rt5 * (63 * rby_val * rbz_val * rbz_val - 7 * rby_val) / 10; // d²/d(rbx)²
            result.s2[10] = rt5 * (63 * rbx_val * rbz_val * rbz_val - 7 * rbx_val) / 10; // d²/d(rby)²
            result.s2[13] = rt5 * (63 * rbz_val * rbz_val * rax_val - 7 * rax_val + 14 * rbz_val * sf.cxz()) / 10; // d²/d(rbx)d(rby)
            result.s2[15] = 63.0 / 5.0 * rt5 * rbx_val * rby_val * rbz_val; // d²/d(rbz)²
            result.s2[18] = rt5 * (126 * rby_val * rbz_val * rax_val + 14 * rby_val * sf.cxz() + 14 * rbz_val * sf.cxy()) / 10; // d²/d(rbz)d(rbx)
            result.s2[19] = rt5 * (126 * rbx_val * rbz_val * rax_val + 14 * rbx_val * sf.cxz() + 14 * rbz_val * sf.cxx()) / 10; // d²/d(rbz)d(rby)
            result.s2[20] = rt5 * (126 * rbx_val * rby_val * rax_val + 14 * rbx_val * sf.cxy() + 14 * rby_val * sf.cxx()) / 10; // d²/d(rbz)d(rax)
            result.s2[25] = rt5 * (7 * rbz_val * rbz_val - 1) / 10; // d²/d(cxx)²
            result.s2[26] = 7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(cxx)d(rbz)
            result.s2[48] = rt5 * (7 * rbz_val * rbz_val - 1) / 10; // d²/d(cxy)²
            result.s2[50] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(cxy)d(rbz)
            result.s2[81] = 7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(cxx)d(cxz)
            result.s2[82] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(cxy)d(cxz)
            result.s2[83] = 7.0 / 5.0 * rt5 * rbx_val * rby_val; // d²/d(cxz)²
        }
    }
}

/**
 * Dipole-x × Hexadecapole-43c kernel
 * Orient case 181: Q11c × Q43c
 */
void dipole_x_hexadecapole_43c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt70 *
                (9 * rbx_val * rbx_val * rbx_val * rbz_val * rax_val -
                 27 * rbx_val * rby_val * rby_val * rbz_val * rax_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt70 * (9 * rbx_val * rbx_val * rbx_val * rbz_val - 27 * rbx_val * rby_val * rby_val * rbz_val) / 20; // d/d(rax)
        result.s1[3] = rt70 * (27 * rbx_val * rbx_val * rbz_val * rax_val - 27 * rby_val * rby_val * rbz_val * rax_val + 3 * rbx_val * rbx_val * sf.cxz() + 6 * rbx_val * rbz_val * sf.cxx() - 3 * rby_val * rby_val * sf.cxz() - 6 * rby_val * rbz_val * sf.cxy()) / 20; // d/d(rbx)
        result.s1[4] = rt70 * (-54 * rbx_val * rby_val * rbz_val * rax_val - 6 * rbx_val * rby_val * sf.cxz() - 6 * rbx_val * rbz_val * sf.cxy() - 6 * rby_val * rbz_val * sf.cxx()) / 20; // d/d(rby)
        result.s1[5] = rt70 * (9 * rbx_val * rbx_val * rbx_val * rax_val - 27 * rbx_val * rby_val * rby_val * rax_val + 3 * rbx_val * rbx_val * sf.cxx() - 6 * rbx_val * rby_val * sf.cxy() - 3 * rby_val * rby_val * sf.cxx()) / 20; // d/d(rbz)
        result.s1[6] = rt70 * (3 * rbx_val * rbx_val * rbz_val - 3 * rby_val * rby_val * rbz_val) / 20; // d/d(cxx)
        result.s1[7] = -3.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d/d(cxy)
        result.s1[8] = rt70 * (rbx_val * rbx_val * rbx_val - 3 * rbx_val * rby_val * rby_val) / 20; // d/d(cxz)

        if (level >= 2) {
            // Second derivatives - Orient case 181
            result.s2[6] = rt70 * (27 * rbx_val * rbx_val * rbz_val - 27 * rby_val * rby_val * rbz_val) / 20; // d²/d(rbx)²
            result.s2[9] = rt70 * (54 * rbx_val * rbz_val * rax_val + 6 * rbx_val * sf.cxz() + 6 * rbz_val * sf.cxx()) / 20; // d²/d(rbx)d(rax)
            result.s2[10] = -27.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d²/d(rby)²
            result.s2[13] = rt70 * (-54 * rby_val * rbz_val * rax_val - 6 * rby_val * sf.cxz() - 6 * rbz_val * sf.cxy()) / 20; // d²/d(rbx)d(rby)
            result.s2[14] = rt70 * (-54 * rbx_val * rbz_val * rax_val - 6 * rbx_val * sf.cxz() - 6 * rbz_val * sf.cxx()) / 20; // d²/d(rby)d(rax)
            result.s2[15] = rt70 * (9 * rbx_val * rbx_val * rbx_val - 27 * rbx_val * rby_val * rby_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt70 * (27 * rbx_val * rbx_val * rax_val - 27 * rby_val * rby_val * rax_val + 6 * rbx_val * sf.cxx() - 6 * rby_val * sf.cxy()) / 20; // d²/d(rbz)d(rbx)
            result.s2[19] = rt70 * (-54 * rbx_val * rby_val * rax_val - 6 * rbx_val * sf.cxy() - 6 * rby_val * sf.cxx()) / 20; // d²/d(rbz)d(rby)
            result.s2[24] = 3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(cxx)²
            result.s2[25] = -3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(cxx)d(cxy)
            result.s2[26] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(cxx)d(rbz)
            result.s2[48] = -3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(cxy)²
            result.s2[49] = -3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(cxy)d(rby)
            result.s2[50] = -3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(cxy)d(rbz)
            result.s2[81] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(cxx)d(cxz)
            result.s2[82] = -3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(cxy)d(cxz)
        }
    }
}

/**
 * Dipole-x × Hexadecapole-43s kernel
 * Orient case 182: Q11c × Q43s
 */
void dipole_x_hexadecapole_43s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt70 *
                (27 * rbx_val * rbx_val * rby_val * rbz_val * rax_val -
                 9 * rby_val * rby_val * rby_val * rbz_val * rax_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt70 * (27 * rbx_val * rbx_val * rby_val * rbz_val - 9 * rby_val * rby_val * rby_val * rbz_val) / 20; // d/d(rax)
        result.s1[3] = rt70 * (54 * rbx_val * rby_val * rbz_val * rax_val + 6 * rbx_val * rby_val * sf.cxz() + 6 * rbx_val * rbz_val * sf.cxy() + 6 * rby_val * rbz_val * sf.cxx()) / 20; // d/d(rbx)
        result.s1[4] = rt70 * (27 * rbx_val * rbx_val * rbz_val * rax_val - 27 * rby_val * rby_val * rbz_val * rax_val + 3 * rbx_val * rbx_val * sf.cxz() + 6 * rbx_val * rbz_val * sf.cxx() - 3 * rby_val * rby_val * sf.cxz() - 6 * rby_val * rbz_val * sf.cxy()) / 20; // d/d(rby)
        result.s1[5] = rt70 * (27 * rbx_val * rbx_val * rby_val * rax_val - 9 * rby_val * rby_val * rby_val * rax_val + 3 * rbx_val * rbx_val * sf.cxy() + 6 * rbx_val * rby_val * sf.cxx() - 3 * rby_val * rby_val * sf.cxy()) / 20; // d/d(rbz)
        result.s1[6] = 3.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d/d(cxx)
        result.s1[7] = rt70 * (3 * rbx_val * rbx_val * rbz_val - 3 * rby_val * rby_val * rbz_val) / 20; // d/d(cxy)
        result.s1[8] = rt70 * (3 * rbx_val * rbx_val * rby_val - rby_val * rby_val * rby_val) / 20; // d/d(cxz)

        if (level >= 2) {
            // Second derivatives - Orient case 182
            result.s2[6] = 27.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d²/d(rbx)²
            result.s2[9] = rt70 * (54 * rby_val * rbz_val * rax_val + 6 * rby_val * sf.cxz() + 6 * rbz_val * sf.cxy()) / 20; // d²/d(rbx)d(rax)
            result.s2[10] = rt70 * (27 * rbx_val * rbx_val * rbz_val - 27 * rby_val * rby_val * rbz_val) / 20; // d²/d(rby)²
            result.s2[13] = rt70 * (54 * rbx_val * rbz_val * rax_val + 6 * rbx_val * sf.cxz() + 6 * rbz_val * sf.cxx()) / 20; // d²/d(rbx)d(rby)
            result.s2[14] = rt70 * (-54 * rby_val * rbz_val * rax_val - 6 * rby_val * sf.cxz() - 6 * rbz_val * sf.cxy()) / 20; // d²/d(rby)d(rax)
            result.s2[15] = rt70 * (27 * rbx_val * rbx_val * rby_val - 9 * rby_val * rby_val * rby_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt70 * (54 * rbx_val * rby_val * rax_val + 6 * rbx_val * sf.cxy() + 6 * rby_val * sf.cxx()) / 20; // d²/d(rbz)d(rbx)
            result.s2[19] = rt70 * (27 * rbx_val * rbx_val * rax_val - 27 * rby_val * rby_val * rax_val + 6 * rbx_val * sf.cxx() - 6 * rby_val * sf.cxy()) / 20; // d²/d(rbz)d(rby)
            result.s2[24] = 3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(cxx)²
            result.s2[25] = 3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(cxx)d(cxy)
            result.s2[26] = 3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(cxx)d(rbz)
            result.s2[48] = 3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(cxy)²
            result.s2[49] = -3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(cxy)d(rby)
            result.s2[50] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(cxy)d(rbz)
            result.s2[81] = 3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(cxx)d(cxz)
            result.s2[82] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(cxy)d(cxz)
        }
    }
}

/**
 * Dipole-x × Hexadecapole-44c kernel
 * Orient case 183: Q11c × Q44c
 */
void dipole_x_hexadecapole_44c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt35 *
                (9 * rbx_val * rbx_val * rbx_val * rbx_val * rax_val -
                 54 * rbx_val * rbx_val * rby_val * rby_val * rax_val +
                 9 * rby_val * rby_val * rby_val * rby_val * rax_val) /
                40;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt35 * (9 * rbx_val * rbx_val * rbx_val * rbx_val - 54 * rbx_val * rbx_val * rby_val * rby_val + 9 * rby_val * rby_val * rby_val * rby_val) / 40; // d/d(rax)
        result.s1[3] = rt35 * (36 * rbx_val * rbx_val * rbx_val * rax_val - 108 * rbx_val * rby_val * rby_val * rax_val + 12 * rbx_val * rbx_val * sf.cxx() - 24 * rbx_val * rby_val * sf.cxy() - 12 * rby_val * rby_val * sf.cxx()) / 40; // d/d(rbx)
        result.s1[4] = rt35 * (-108 * rbx_val * rbx_val * rby_val * rax_val + 36 * rby_val * rby_val * rby_val * rax_val - 12 * rbx_val * rbx_val * sf.cxy() - 24 * rbx_val * rby_val * sf.cxx() + 12 * rby_val * rby_val * sf.cxy()) / 40; // d/d(rby)
        result.s1[6] = rt35 * (4 * rbx_val * rbx_val * rbx_val - 12 * rbx_val * rby_val * rby_val) / 40; // d/d(cxx)
        result.s1[7] = rt35 * (-12 * rbx_val * rbx_val * rby_val + 4 * rby_val * rby_val * rby_val) / 40; // d/d(cxy)

        if (level >= 2) {
            // Second derivatives - Orient case 183
            result.s2[6] = rt35 * (36 * rbx_val * rbx_val * rbx_val - 108 * rbx_val * rby_val * rby_val) / 40; // d²/d(rbx)²
            result.s2[9] = rt35 * (108 * rbx_val * rbx_val * rax_val - 108 * rby_val * rby_val * rax_val + 24 * rbx_val * sf.cxx() - 24 * rby_val * sf.cxy()) / 40; // d²/d(rbx)d(rax)
            result.s2[10] = rt35 * (-108 * rbx_val * rbx_val * rby_val + 36 * rby_val * rby_val * rby_val) / 40; // d²/d(rby)²
            result.s2[13] = rt35 * (-216 * rbx_val * rby_val * rax_val - 24 * rbx_val * sf.cxy() - 24 * rby_val * sf.cxx()) / 40; // d²/d(rbx)d(rby)
            result.s2[14] = rt35 * (-108 * rbx_val * rbx_val * rax_val + 108 * rby_val * rby_val * rax_val - 24 * rbx_val * sf.cxx() + 24 * rby_val * sf.cxy()) / 40; // d²/d(rby)d(rax)
            result.s2[24] = rt35 * (12 * rbx_val * rbx_val - 12 * rby_val * rby_val) / 40; // d²/d(cxx)²
            result.s2[25] = -3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(cxx)d(cxy)
            result.s2[48] = -3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(cxy)²
            result.s2[49] = rt35 * (-12 * rbx_val * rbx_val + 12 * rby_val * rby_val) / 40; // d²/d(cxy)d(rby)
        }
    }
}

/**
 * Dipole-x × Hexadecapole-44s kernel
 * Orient case 184: Q11c × Q44s
 */
void dipole_x_hexadecapole_44s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt35 *
                (9 * rbx_val * rbx_val * rbx_val * rby_val * rax_val -
                 9 * rbx_val * rby_val * rby_val * rby_val * rax_val) /
                10;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt35 * (9 * rbx_val * rbx_val * rbx_val * rby_val - 9 * rbx_val * rby_val * rby_val * rby_val) / 10; // d/d(rax)
        result.s1[3] = rt35 * (27 * rbx_val * rbx_val * rby_val * rax_val - 9 * rby_val * rby_val * rby_val * rax_val + 3 * rbx_val * rbx_val * sf.cxy() + 6 * rbx_val * rby_val * sf.cxx() - 3 * rby_val * rby_val * sf.cxy()) / 10; // d/d(rbx)
        result.s1[4] = rt35 * (9 * rbx_val * rbx_val * rbx_val * rax_val - 27 * rbx_val * rby_val * rby_val * rax_val + 3 * rbx_val * rbx_val * sf.cxx() - 6 * rbx_val * rby_val * sf.cxy() - 3 * rby_val * rby_val * sf.cxx()) / 10; // d/d(rby)
        result.s1[6] = rt35 * (3 * rbx_val * rbx_val * rby_val - rby_val * rby_val * rby_val) / 10; // d/d(cxx)
        result.s1[7] = rt35 * (rbx_val * rbx_val * rbx_val - 3 * rbx_val * rby_val * rby_val) / 10; // d/d(cxy)

        if (level >= 2) {
            // Second derivatives - Orient case 184
            result.s2[6] = rt35 * (27 * rbx_val * rbx_val * rby_val - 9 * rby_val * rby_val * rby_val) / 10; // d²/d(rbx)²
            result.s2[9] = rt35 * (54 * rbx_val * rby_val * rax_val + 6 * rbx_val * sf.cxy() + 6 * rby_val * sf.cxx()) / 10; // d²/d(rbx)d(rax)
            result.s2[10] = rt35 * (9 * rbx_val * rbx_val * rbx_val - 27 * rbx_val * rby_val * rby_val) / 10; // d²/d(rby)²
            result.s2[13] = rt35 * (27 * rbx_val * rbx_val * rax_val - 27 * rby_val * rby_val * rax_val + 6 * rbx_val * sf.cxx() - 6 * rby_val * sf.cxy()) / 10; // d²/d(rbx)d(rby)
            result.s2[14] = rt35 * (-54 * rbx_val * rby_val * rax_val - 6 * rbx_val * sf.cxy() - 6 * rby_val * sf.cxx()) / 10; // d²/d(rby)d(rax)
            result.s2[24] = 3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(cxx)²
            result.s2[25] = rt35 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 10; // d²/d(cxx)d(cxy)
            result.s2[48] = rt35 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 10; // d²/d(cxy)²
            result.s2[49] = -3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(cxy)d(rby)
        }
    }
}

/**
 * Dipole-y × Hexadecapole-40 kernel
 * Orient case 185: Q11s × Q40
 */
void dipole_y_hexadecapole_40(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        63.0 / 8.0 * rbz_val * rbz_val * rbz_val * rbz_val * ray_val -
        21.0 / 4.0 * rbz_val * rbz_val * ray_val + 3.0 / 8.0 * ray_val;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = 63.0 / 8.0 * rbz_val * rbz_val * rbz_val * rbz_val - 21.0 / 4.0 * rbz_val * rbz_val + 3.0 / 8.0; // d/d(ray)
        result.s1[5] = 252.0 / 8.0 * rbz_val * rbz_val * rbz_val * ray_val - 42.0 / 4.0 * rbz_val * ray_val + 21.0 / 2.0 * rbz_val * rbz_val * sf.cyz() - 3.0 / 2.0 * sf.cyz(); // d/d(rbz)
        result.s1[11] = 7.0 / 2.0 * rbz_val * rbz_val * rbz_val - 3.0 / 2.0 * rbz_val; // d/d(cyz)

        if (level >= 2) {
            // Second derivatives - Orient case 185
            result.s2[16] = 63.0 / 2.0 * rbz_val * rbz_val * rbz_val - 21.0 / 2.0 * rbz_val; // d²/d(rbz)²
            result.s2[20] = 189.0 / 2.0 * rbz_val * rbz_val * ray_val - 21.0 / 2.0 * ray_val + 21.0 * rbz_val * sf.cyz(); // d²/d(rbz)d(ray)
            result.s2[96] = 21.0 / 2.0 * rbz_val * rbz_val - 3.0 / 2.0; // d²/d(cyz)²
        }
    }
}

/**
 * Dipole-y × Hexadecapole-41c kernel
 * Orient case 186: Q11s × Q41c
 */
void dipole_y_hexadecapole_41c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt10 *
                (63 * rbx_val * rbz_val * rbz_val * rbz_val * ray_val -
                 21 * rbx_val * rbz_val * ray_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt10 * (63 * rbx_val * rbz_val * rbz_val * rbz_val - 21 * rbx_val * rbz_val) / 20; // d/d(ray)
        result.s1[3] = rt10 * (63 * rbz_val * rbz_val * rbz_val * ray_val - 21 * rbz_val * ray_val + 21 * rbz_val * rbz_val * sf.cyz() - 3 * sf.cyz()) / 20; // d/d(rbx)
        result.s1[5] = rt10 * (189 * rbx_val * rbz_val * rbz_val * ray_val - 21 * rbx_val * ray_val + 42 * rbx_val * rbz_val * sf.cyz() + 21 * rbz_val * rbz_val * sf.cyx() - 3 * sf.cyx()) / 20; // d/d(rbz)
        result.s1[9] = rt10 * (7 * rbz_val * rbz_val * rbz_val - 3 * rbz_val) / 20; // d/d(cyx)
        result.s1[11] = rt10 * (21 * rbx_val * rbz_val * rbz_val - 3 * rbx_val) / 20; // d/d(cyz)

        if (level >= 2) {
            // Second derivatives - Orient case 186
            result.s2[7] = rt10 * (63 * rbz_val * rbz_val * rbz_val - 21 * rbz_val) / 20; // d²/d(rbx)²
            result.s2[16] = rt10 * (189 * rbx_val * rbz_val * rbz_val - 21 * rbx_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt10 * (189 * rbz_val * rbz_val * ray_val - 21 * ray_val + 42 * rbz_val * sf.cyz()) / 20; // d²/d(rbz)d(rbx)
            result.s2[20] = rt10 * (378 * rbx_val * rbz_val * ray_val + 42 * rbx_val * sf.cyz() + 42 * rbz_val * sf.cyx()) / 20; // d²/d(rbz)d(ray)
            result.s2[33] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(cyx)²
            result.s2[94] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(cyx)d(cyz)
            result.s2[96] = 21.0 / 10.0 * rt10 * rbx_val * rbz_val; // d²/d(cyz)²
        }
    }
}

/**
 * Dipole-y × Hexadecapole-41s kernel
 * Orient case 187: Q11s × Q41s
 */
void dipole_y_hexadecapole_41s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt10 *
                (63 * rby_val * rbz_val * rbz_val * rbz_val * ray_val -
                 21 * rby_val * rbz_val * ray_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt10 * (63 * rby_val * rbz_val * rbz_val * rbz_val - 21 * rby_val * rbz_val) / 20; // d/d(ray)
        result.s1[4] = rt10 * (63 * rbz_val * rbz_val * rbz_val * ray_val - 21 * rbz_val * ray_val + 21 * rbz_val * rbz_val * sf.cyz() - 3 * sf.cyz()) / 20; // d/d(rby)
        result.s1[5] = rt10 * (189 * rby_val * rbz_val * rbz_val * ray_val - 21 * rby_val * ray_val + 42 * rby_val * rbz_val * sf.cyz() + 21 * rbz_val * rbz_val * sf.cyy() - 3 * sf.cyy()) / 20; // d/d(rbz)
        result.s1[10] = rt10 * (7 * rbz_val * rbz_val * rbz_val - 3 * rbz_val) / 20; // d/d(cyy)
        result.s1[11] = rt10 * (21 * rby_val * rbz_val * rbz_val - 3 * rby_val) / 20; // d/d(cyz)

        if (level >= 2) {
            // Second derivatives - Orient case 187
            result.s2[11] = rt10 * (63 * rbz_val * rbz_val * rbz_val - 21 * rbz_val) / 20; // d²/d(rby)²
            result.s2[16] = rt10 * (189 * rby_val * rbz_val * rbz_val - 21 * rby_val) / 20; // d²/d(rbz)²
            result.s2[19] = rt10 * (189 * rbz_val * rbz_val * ray_val - 21 * ray_val + 42 * rbz_val * sf.cyz()) / 20; // d²/d(rbz)d(rby)
            result.s2[20] = rt10 * (378 * rby_val * rbz_val * ray_val + 42 * rby_val * sf.cyz() + 42 * rbz_val * sf.cyy()) / 20; // d²/d(rbz)d(ray)
            result.s2[60] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(cyy)²
            result.s2[95] = rt10 * (21 * rbz_val * rbz_val - 3) / 20; // d²/d(cyy)d(cyz)
            result.s2[96] = 21.0 / 10.0 * rt10 * rby_val * rbz_val; // d²/d(cyz)²
        }
    }
}

/**
 * Dipole-y × Hexadecapole-42c kernel
 * Orient case 188: Q11s × Q42c
 */
void dipole_y_hexadecapole_42c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rbx_val * rbz_val * rbz_val * ray_val -
                 63 * rby_val * rby_val * rbz_val * rbz_val * ray_val -
                 7 * rbx_val * rbx_val * ray_val +
                 7 * rby_val * rby_val * ray_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt5 * (63 * rbx_val * rbx_val * rbz_val * rbz_val - 63 * rby_val * rby_val * rbz_val * rbz_val - 7 * rbx_val * rbx_val + 7 * rby_val * rby_val) / 20; // d/d(ray)
        result.s1[3] = rt5 * (126 * rbx_val * rbz_val * rbz_val * ray_val - 14 * rbx_val * ray_val + 28 * rbx_val * rbz_val * sf.cyz() + 14 * rbz_val * rbz_val * sf.cyx() - 2 * sf.cyx()) / 20; // d/d(rbx)
        result.s1[4] = rt5 * (-126 * rby_val * rbz_val * rbz_val * ray_val + 14 * rby_val * ray_val - 28 * rby_val * rbz_val * sf.cyz() - 14 * rbz_val * rbz_val * sf.cyy() + 2 * sf.cyy()) / 20; // d/d(rby)
        result.s1[5] = rt5 * (126 * rbx_val * rbx_val * rbz_val * ray_val - 126 * rby_val * rby_val * rbz_val * ray_val + 14 * rbx_val * rbx_val * sf.cyz() + 28 * rbx_val * rbz_val * sf.cyx() - 14 * rby_val * rby_val * sf.cyz() - 28 * rby_val * rbz_val * sf.cyy()) / 20; // d/d(rbz)
        result.s1[9] = rt5 * (14 * rbx_val * rbz_val * rbz_val - 2 * rbx_val) / 20; // d/d(cyx)
        result.s1[10] = rt5 * (-14 * rby_val * rbz_val * rbz_val + 2 * rby_val) / 20; // d/d(cyy)
        result.s1[11] = rt5 * (14 * rbx_val * rbx_val * rbz_val - 14 * rby_val * rby_val * rbz_val) / 20; // d/d(cyz)

        if (level >= 2) {
            // Second derivatives - Orient case 188
            result.s2[7] = rt5 * (126 * rbx_val * rbz_val * rbz_val - 14 * rbx_val) / 20; // d²/d(rbx)²
            result.s2[9] = rt5 * (126 * rbz_val * rbz_val * ray_val - 14 * ray_val + 28 * rbz_val * sf.cyz()) / 20; // d²/d(rbx)d(ray)
            result.s2[11] = rt5 * (-126 * rby_val * rbz_val * rbz_val + 14 * rby_val) / 20; // d²/d(rby)²
            result.s2[14] = rt5 * (-126 * rbz_val * rbz_val * ray_val + 14 * ray_val - 28 * rbz_val * sf.cyz()) / 20; // d²/d(rby)d(ray)
            result.s2[16] = rt5 * (126 * rbx_val * rbx_val * rbz_val - 126 * rby_val * rby_val * rbz_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt5 * (252 * rbx_val * rbz_val * ray_val + 28 * rbx_val * sf.cyz() + 28 * rbz_val * sf.cyx()) / 20; // d²/d(rbz)d(rbx)
            result.s2[19] = rt5 * (-252 * rby_val * rbz_val * ray_val - 28 * rby_val * sf.cyz() - 28 * rbz_val * sf.cyy()) / 20; // d²/d(rbz)d(rby)
            result.s2[20] = rt5 * (126 * rbx_val * rbx_val * ray_val - 126 * rby_val * rby_val * ray_val + 28 * rbx_val * sf.cyx() - 28 * rby_val * sf.cyy()) / 20; // d²/d(rbz)d(ray)
            result.s2[31] = rt5 * (14 * rbz_val * rbz_val - 2) / 20; // d²/d(cyx)²
            result.s2[33] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(cyx)d(rbz)
            result.s2[59] = rt5 * (-14 * rbz_val * rbz_val + 2) / 20; // d²/d(cyy)²
            result.s2[60] = -7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(cyy)d(rbz)
            result.s2[94] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(cyx)d(cyz)
            result.s2[95] = -7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(cyy)d(cyz)
            result.s2[96] = rt5 * (14 * rbx_val * rbx_val - 14 * rby_val * rby_val) / 20; // d²/d(cyz)²
        }
    }
}

/**
 * Dipole-y × Hexadecapole-42s kernel
 * Orient case 189: Q11s × Q42s
 */
void dipole_y_hexadecapole_42s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rby_val * rbz_val * rbz_val * ray_val -
                 7 * rbx_val * rby_val * ray_val) /
                10;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt5 * (63 * rbx_val * rby_val * rbz_val * rbz_val - 7 * rbx_val * rby_val) / 10; // d/d(ray)
        result.s1[3] = rt5 * (63 * rby_val * rbz_val * rbz_val * ray_val - 7 * rby_val * ray_val + 14 * rby_val * rbz_val * sf.cyz() + 7 * rbz_val * rbz_val * sf.cyy() - sf.cyy()) / 10; // d/d(rbx)
        result.s1[4] = rt5 * (63 * rbx_val * rbz_val * rbz_val * ray_val - 7 * rbx_val * ray_val + 14 * rbx_val * rbz_val * sf.cyz() + 7 * rbz_val * rbz_val * sf.cyx() - sf.cyx()) / 10; // d/d(rby)
        result.s1[5] = rt5 * (126 * rbx_val * rby_val * rbz_val * ray_val + 14 * rbx_val * rby_val * sf.cyz() + 14 * rbx_val * rbz_val * sf.cyy() + 14 * rby_val * rbz_val * sf.cyx()) / 10; // d/d(rbz)
        result.s1[9] = rt5 * (7 * rby_val * rbz_val * rbz_val - rby_val) / 10; // d/d(cyx)
        result.s1[10] = rt5 * (7 * rbx_val * rbz_val * rbz_val - rbx_val) / 10; // d/d(cyy)
        result.s1[11] = 7.0 / 5.0 * rt5 * rbx_val * rby_val * rbz_val; // d/d(cyz)

        if (level >= 2) {
            // Second derivatives - Orient case 189
            result.s2[7] = rt5 * (63 * rby_val * rbz_val * rbz_val - 7 * rby_val) / 10; // d²/d(rbx)²
            result.s2[11] = rt5 * (63 * rbx_val * rbz_val * rbz_val - 7 * rbx_val) / 10; // d²/d(rby)²
            result.s2[13] = rt5 * (63 * rbz_val * rbz_val * ray_val - 7 * ray_val + 14 * rbz_val * sf.cyz()) / 10; // d²/d(rbx)d(rby)
            result.s2[16] = 63.0 / 5.0 * rt5 * rbx_val * rby_val * rbz_val; // d²/d(rbz)²
            result.s2[18] = rt5 * (126 * rby_val * rbz_val * ray_val + 14 * rby_val * sf.cyz() + 14 * rbz_val * sf.cyy()) / 10; // d²/d(rbz)d(rbx)
            result.s2[19] = rt5 * (126 * rbx_val * rbz_val * ray_val + 14 * rbx_val * sf.cyz() + 14 * rbz_val * sf.cyx()) / 10; // d²/d(rbz)d(rby)
            result.s2[20] = rt5 * (126 * rbx_val * rby_val * ray_val + 14 * rbx_val * sf.cyy() + 14 * rby_val * sf.cyx()) / 10; // d²/d(rbz)d(ray)
            result.s2[32] = rt5 * (7 * rbz_val * rbz_val - 1) / 10; // d²/d(cyx)²
            result.s2[33] = 7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(cyx)d(rbz)
            result.s2[58] = rt5 * (7 * rbz_val * rbz_val - 1) / 10; // d²/d(cyy)²
            result.s2[60] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(cyy)d(rbz)
            result.s2[94] = 7.0 / 5.0 * rt5 * rby_val * rbz_val; // d²/d(cyx)d(cyz)
            result.s2[95] = 7.0 / 5.0 * rt5 * rbx_val * rbz_val; // d²/d(cyy)d(cyz)
            result.s2[96] = 7.0 / 5.0 * rt5 * rbx_val * rby_val; // d²/d(cyz)²
        }
    }
}

/**
 * Dipole-y × Hexadecapole-43c kernel
 * Orient case 190: Q11s × Q43c
 */
void dipole_y_hexadecapole_43c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt70 *
                (9 * rbx_val * rbx_val * rbx_val * rbz_val * ray_val -
                 27 * rbx_val * rby_val * rby_val * rbz_val * ray_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt70 * (9 * rbx_val * rbx_val * rbx_val * rbz_val - 27 * rbx_val * rby_val * rby_val * rbz_val) / 20; // d/d(ray)
        result.s1[3] = rt70 * (27 * rbx_val * rbx_val * rbz_val * ray_val - 27 * rby_val * rby_val * rbz_val * ray_val + 3 * rbx_val * rbx_val * sf.cyz() + 6 * rbx_val * rbz_val * sf.cyx() - 3 * rby_val * rby_val * sf.cyz() - 6 * rby_val * rbz_val * sf.cyy()) / 20; // d/d(rbx)
        result.s1[4] = rt70 * (-54 * rbx_val * rby_val * rbz_val * ray_val - 6 * rbx_val * rby_val * sf.cyz() - 6 * rbx_val * rbz_val * sf.cyy() - 6 * rby_val * rbz_val * sf.cyx()) / 20; // d/d(rby)
        result.s1[5] = rt70 * (9 * rbx_val * rbx_val * rbx_val * ray_val - 27 * rbx_val * rby_val * rby_val * ray_val + 3 * rbx_val * rbx_val * sf.cyx() - 6 * rbx_val * rby_val * sf.cyy() - 3 * rby_val * rby_val * sf.cyx()) / 20; // d/d(rbz)
        result.s1[9] = rt70 * (3 * rbx_val * rbx_val * rbz_val - 3 * rby_val * rby_val * rbz_val) / 20; // d/d(cyx)
        result.s1[10] = -3.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d/d(cyy)
        result.s1[11] = rt70 * (rbx_val * rbx_val * rbx_val - 3 * rbx_val * rby_val * rby_val) / 20; // d/d(cyz)

        if (level >= 2) {
            // Second derivatives - Orient case 190
            result.s2[7] = rt70 * (27 * rbx_val * rbx_val * rbz_val - 27 * rby_val * rby_val * rbz_val) / 20; // d²/d(rbx)²
            result.s2[9] = rt70 * (54 * rbx_val * rbz_val * ray_val + 6 * rbx_val * sf.cyz() + 6 * rbz_val * sf.cyx()) / 20; // d²/d(rbx)d(ray)
            result.s2[11] = -27.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d²/d(rby)²
            result.s2[13] = rt70 * (-54 * rby_val * rbz_val * ray_val - 6 * rby_val * sf.cyz() - 6 * rbz_val * sf.cyy()) / 20; // d²/d(rbx)d(rby)
            result.s2[14] = rt70 * (-54 * rbx_val * rbz_val * ray_val - 6 * rbx_val * sf.cyz() - 6 * rbz_val * sf.cyx()) / 20; // d²/d(rby)d(ray)
            result.s2[16] = rt70 * (9 * rbx_val * rbx_val * rbx_val - 27 * rbx_val * rby_val * rby_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt70 * (27 * rbx_val * rbx_val * ray_val - 27 * rby_val * rby_val * ray_val + 6 * rbx_val * sf.cyx() - 6 * rby_val * sf.cyy()) / 20; // d²/d(rbz)d(rbx)
            result.s2[19] = rt70 * (-54 * rbx_val * rby_val * ray_val - 6 * rbx_val * sf.cyy() - 6 * rby_val * sf.cyx()) / 20; // d²/d(rbz)d(rby)
            result.s2[31] = 3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(cyx)²
            result.s2[32] = -3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(cyx)d(cyy)
            result.s2[33] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(cyx)d(rbz)
            result.s2[58] = -3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(cyy)²
            result.s2[59] = -3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(cyy)d(rby)
            result.s2[60] = -3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(cyy)d(rbz)
            result.s2[94] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(cyx)d(cyz)
            result.s2[95] = -3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(cyy)d(cyz)
        }
    }
}

/**
 * Dipole-y × Hexadecapole-43s kernel
 * Orient case 191: Q11s × Q43s
 */
void dipole_y_hexadecapole_43s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt70 *
                (27 * rbx_val * rbx_val * rby_val * rbz_val * ray_val -
                 9 * rby_val * rby_val * rby_val * rbz_val * ray_val) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt70 * (27 * rbx_val * rbx_val * rby_val * rbz_val - 9 * rby_val * rby_val * rby_val * rbz_val) / 20; // d/d(ray)
        result.s1[3] = rt70 * (54 * rbx_val * rby_val * rbz_val * ray_val + 6 * rbx_val * rby_val * sf.cyz() + 6 * rbx_val * rbz_val * sf.cyy() + 6 * rby_val * rbz_val * sf.cyx()) / 20; // d/d(rbx)
        result.s1[4] = rt70 * (27 * rbx_val * rbx_val * rbz_val * ray_val - 27 * rby_val * rby_val * rbz_val * ray_val + 3 * rbx_val * rbx_val * sf.cyz() + 6 * rbx_val * rbz_val * sf.cyx() - 3 * rby_val * rby_val * sf.cyz() - 6 * rby_val * rbz_val * sf.cyy()) / 20; // d/d(rby)
        result.s1[5] = rt70 * (27 * rbx_val * rbx_val * rby_val * ray_val - 9 * rby_val * rby_val * rby_val * ray_val + 3 * rbx_val * rbx_val * sf.cyy() + 6 * rbx_val * rby_val * sf.cyx() - 3 * rby_val * rby_val * sf.cyy()) / 20; // d/d(rbz)
        result.s1[9] = 3.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d/d(cyx)
        result.s1[10] = rt70 * (3 * rbx_val * rbx_val * rbz_val - 3 * rby_val * rby_val * rbz_val) / 20; // d/d(cyy)
        result.s1[11] = rt70 * (3 * rbx_val * rbx_val * rby_val - rby_val * rby_val * rby_val) / 20; // d/d(cyz)

        if (level >= 2) {
            // Second derivatives - Orient case 191
            result.s2[7] = 27.0 / 10.0 * rt70 * rbx_val * rby_val * rbz_val; // d²/d(rbx)²
            result.s2[9] = rt70 * (54 * rby_val * rbz_val * ray_val + 6 * rby_val * sf.cyz() + 6 * rbz_val * sf.cyy()) / 20; // d²/d(rbx)d(ray)
            result.s2[11] = rt70 * (27 * rbx_val * rbx_val * rbz_val - 27 * rby_val * rby_val * rbz_val) / 20; // d²/d(rby)²
            result.s2[13] = rt70 * (54 * rbx_val * rbz_val * ray_val + 6 * rbx_val * sf.cyz() + 6 * rbz_val * sf.cyx()) / 20; // d²/d(rbx)d(rby)
            result.s2[14] = rt70 * (-54 * rby_val * rbz_val * ray_val - 6 * rby_val * sf.cyz() - 6 * rbz_val * sf.cyy()) / 20; // d²/d(rby)d(ray)
            result.s2[16] = rt70 * (27 * rbx_val * rbx_val * rby_val - 9 * rby_val * rby_val * rby_val) / 20; // d²/d(rbz)²
            result.s2[18] = rt70 * (54 * rbx_val * rby_val * ray_val + 6 * rbx_val * sf.cyy() + 6 * rby_val * sf.cyx()) / 20; // d²/d(rbz)d(rbx)
            result.s2[19] = rt70 * (27 * rbx_val * rbx_val * ray_val - 27 * rby_val * rby_val * ray_val + 6 * rbx_val * sf.cyx() - 6 * rby_val * sf.cyy()) / 20; // d²/d(rbz)d(rby)
            result.s2[31] = 3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(cyx)²
            result.s2[32] = 3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(cyx)d(cyy)
            result.s2[33] = 3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(cyx)d(rbz)
            result.s2[58] = 3.0 / 10.0 * rt70 * rbx_val * rbz_val; // d²/d(cyy)²
            result.s2[59] = -3.0 / 10.0 * rt70 * rby_val * rbz_val; // d²/d(cyy)d(rby)
            result.s2[60] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(cyy)d(rbz)
            result.s2[94] = 3.0 / 10.0 * rt70 * rbx_val * rby_val; // d²/d(cyx)d(cyz)
            result.s2[95] = rt70 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 20; // d²/d(cyy)d(cyz)
        }
    }
}

/**
 * Dipole-y × Hexadecapole-44c kernel
 * Orient case 192: Q11s × Q44c
 */
void dipole_y_hexadecapole_44c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt35 *
                (9 * rbx_val * rbx_val * rbx_val * rbx_val * ray_val -
                 54 * rbx_val * rbx_val * rby_val * rby_val * ray_val +
                 9 * rby_val * rby_val * rby_val * rby_val * ray_val) /
                40;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt35 * (9 * rbx_val * rbx_val * rbx_val * rbx_val - 54 * rbx_val * rbx_val * rby_val * rby_val + 9 * rby_val * rby_val * rby_val * rby_val) / 40; // d/d(ray)
        result.s1[3] = rt35 * (36 * rbx_val * rbx_val * rbx_val * ray_val - 108 * rbx_val * rby_val * rby_val * ray_val + 12 * rbx_val * rbx_val * sf.cyx() - 24 * rbx_val * rby_val * sf.cyy() - 12 * rby_val * rby_val * sf.cyx()) / 40; // d/d(rbx)
        result.s1[4] = rt35 * (-108 * rbx_val * rbx_val * rby_val * ray_val + 36 * rby_val * rby_val * rby_val * ray_val - 12 * rbx_val * rbx_val * sf.cyy() - 24 * rbx_val * rby_val * sf.cyx() + 12 * rby_val * rby_val * sf.cyy()) / 40; // d/d(rby)
        result.s1[9] = rt35 * (4 * rbx_val * rbx_val * rbx_val - 12 * rbx_val * rby_val * rby_val) / 40; // d/d(cyx)
        result.s1[10] = rt35 * (-12 * rbx_val * rbx_val * rby_val + 4 * rby_val * rby_val * rby_val) / 40; // d/d(cyy)

        if (level >= 2) {
            // Second derivatives - Orient case 192
            result.s2[7] = rt35 * (36 * rbx_val * rbx_val * rbx_val - 108 * rbx_val * rby_val * rby_val) / 40; // d²/d(rbx)²
            result.s2[9] = rt35 * (108 * rbx_val * rbx_val * ray_val - 108 * rby_val * rby_val * ray_val + 24 * rbx_val * sf.cyx() - 24 * rby_val * sf.cyy()) / 40; // d²/d(rbx)d(ray)
            result.s2[11] = rt35 * (-108 * rbx_val * rbx_val * rby_val + 36 * rby_val * rby_val * rby_val) / 40; // d²/d(rby)²
            result.s2[13] = rt35 * (-216 * rbx_val * rby_val * ray_val - 24 * rbx_val * sf.cyy() - 24 * rby_val * sf.cyx()) / 40; // d²/d(rbx)d(rby)
            result.s2[14] = rt35 * (-108 * rbx_val * rbx_val * ray_val + 108 * rby_val * rby_val * ray_val - 24 * rbx_val * sf.cyx() + 24 * rby_val * sf.cyy()) / 40; // d²/d(rby)d(ray)
            result.s2[31] = rt35 * (12 * rbx_val * rbx_val - 12 * rby_val * rby_val) / 40; // d²/d(cyx)²
            result.s2[32] = -3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(cyx)d(cyy)
            result.s2[58] = -3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(cyy)²
            result.s2[59] = rt35 * (-12 * rbx_val * rbx_val + 12 * rby_val * rby_val) / 40; // d²/d(cyy)d(rby)
        }
    }
}

/**
 * Dipole-y × Hexadecapole-44s kernel
 * Orient case 193: Q11s × Q44s
 */
void dipole_y_hexadecapole_44s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt35 *
                (9 * rbx_val * rbx_val * rbx_val * rby_val * ray_val -
                 9 * rbx_val * rby_val * rby_val * rby_val * ray_val) /
                10;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt35 * (9 * rbx_val * rbx_val * rbx_val * rby_val - 9 * rbx_val * rby_val * rby_val * rby_val) / 10; // d/d(ray)
        result.s1[3] = rt35 * (27 * rbx_val * rbx_val * rby_val * ray_val - 9 * rby_val * rby_val * rby_val * ray_val + 3 * rbx_val * rbx_val * sf.cyy() + 6 * rbx_val * rby_val * sf.cyx() - 3 * rby_val * rby_val * sf.cyy()) / 10; // d/d(rbx)
        result.s1[4] = rt35 * (9 * rbx_val * rbx_val * rbx_val * ray_val - 27 * rbx_val * rby_val * rby_val * ray_val + 3 * rbx_val * rbx_val * sf.cyx() - 6 * rbx_val * rby_val * sf.cyy() - 3 * rby_val * rby_val * sf.cyx()) / 10; // d/d(rby)
        result.s1[9] = rt35 * (3 * rbx_val * rbx_val * rby_val - rby_val * rby_val * rby_val) / 10; // d/d(cyx)
        result.s1[10] = rt35 * (rbx_val * rbx_val * rbx_val - 3 * rbx_val * rby_val * rby_val) / 10; // d/d(cyy)

        if (level >= 2) {
            // Second derivatives - Orient case 193
            result.s2[7] = rt35 * (27 * rbx_val * rbx_val * rby_val - 9 * rby_val * rby_val * rby_val) / 10; // d²/d(rbx)²
            result.s2[9] = rt35 * (54 * rbx_val * rby_val * ray_val + 6 * rbx_val * sf.cyy() + 6 * rby_val * sf.cyx()) / 10; // d²/d(rbx)d(ray)
            result.s2[11] = rt35 * (9 * rbx_val * rbx_val * rbx_val - 27 * rbx_val * rby_val * rby_val) / 10; // d²/d(rby)²
            result.s2[13] = rt35 * (27 * rbx_val * rbx_val * ray_val - 27 * rby_val * rby_val * ray_val + 6 * rbx_val * sf.cyx() - 6 * rby_val * sf.cyy()) / 10; // d²/d(rbx)d(rby)
            result.s2[14] = rt35 * (-54 * rbx_val * rby_val * ray_val - 6 * rbx_val * sf.cyy() - 6 * rby_val * sf.cyx()) / 10; // d²/d(rby)d(ray)
            result.s2[31] = 3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(cyx)²
            result.s2[32] = rt35 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 10; // d²/d(cyx)d(cyy)
            result.s2[58] = rt35 * (3 * rbx_val * rbx_val - 3 * rby_val * rby_val) / 10; // d²/d(cyy)²
            result.s2[59] = -3.0 / 5.0 * rt35 * rbx_val * rby_val; // d²/d(cyy)d(rby)
        }
    }
}

// ============================================================================
// QUADRUPOLE-OCTOPOLE KERNELS (Orient cases 194-228)
// Quadrupole @ A (uses rax, ray, raz), Octopole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Quadrupole-20 × Octopole-30 kernel
 * Orient case 194: Q20 × Q30
 */
void quadrupole_20_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        63.0 / 8.0 * rbz_val * rbz_val * rbz_val * raz_val * raz_val -
        7.0 / 8.0 * rbz_val * rbz_val * rbz_val -
        21.0 / 8.0 * raz_val * raz_val * rbz_val +
        21.0 / 4.0 * rbz_val * rbz_val * raz_val * sf.czz() +
        3.0 / 8.0 * rbz_val - 3.0 / 4.0 * raz_val * sf.czz() +
        3.0 / 4.0 * rbz_val * sf.czz() * sf.czz();

    if (level >= 1) {
        // Coordinate derivatives
        result.s1[2] = 126.0 / 8.0 * rbz_val * rbz_val * rbz_val * raz_val - 42.0 / 8.0 * raz_val * rbz_val + 21.0 / 4.0 * rbz_val * rbz_val * sf.czz() - 3.0 / 4.0 * sf.czz(); // d/d(raz)
        result.s1[5] = 189.0 / 8.0 * rbz_val * rbz_val * raz_val * raz_val - 21.0 / 8.0 * rbz_val * rbz_val - 21.0 / 8.0 * raz_val * raz_val + 42.0 / 4.0 * rbz_val * raz_val * sf.czz() + 3.0 / 8.0 + 3.0 / 4.0 * sf.czz() * sf.czz(); // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[14] = 21.0 / 4.0 * rbz_val * rbz_val * raz_val - 3.0 / 4.0 * raz_val + 3.0 / 2.0 * rbz_val * sf.czz(); // ∂s0/∂czz

        if (level >= 2) {
            // Orient case 194: Q20 × O30
            result.s2[5] = 63.0 / 4.0 * rbz_val * rbz_val * rbz_val - 21.0 / 4.0 * rbz_val;
            result.s2[17] = 189.0 / 4.0 * rbz_val * rbz_val * raz_val - 21.0 / 4.0 * raz_val + 21.0 / 2.0 * rbz_val * sf.czz();
            result.s2[20] = 189.0 / 4.0 * raz_val * raz_val * rbz_val - 21.0 / 4.0 * rbz_val + 21.0 / 2.0 * raz_val * sf.czz();
            result.s2[107] = 21.0 / 4.0 * rbz_val * rbz_val - 3.0 / 4.0;
            result.s2[110] = 21.0 / 2.0 * raz_val * rbz_val + 3.0 / 2.0 * sf.czz();
            result.s2[119] = 3.0 / 2.0 * rbz_val;
        }
    }
}

/**
 * Quadrupole-20 × Octopole-31c kernel
 * Orient case 195: Q20 × Q31c
 */
void quadrupole_20_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt6 *
        (63 * rbx_val * rbz_val * rbz_val * raz_val * raz_val -
         7 * rbx_val * rbz_val * rbz_val - 7 * raz_val * raz_val * rbx_val +
         28 * rbx_val * rbz_val * raz_val * sf.czz() +
         14 * rbz_val * rbz_val * raz_val * sf.czx() + rbx_val -
         2 * raz_val * sf.czx() + 2 * rbx_val * sf.czz() * sf.czz() +
         4 * rbz_val * sf.czx() * sf.czz()) /
        16;

    if (level >= 1) {
        // Coordinate derivatives
        result.s1[2] = rt6 * (126 * rbx_val * rbz_val * rbz_val * raz_val - 14 * rbx_val + 28 * rbx_val * rbz_val * sf.czz() + 14 * rbz_val * rbz_val * sf.czx() - 2 * sf.czx()) / 16; // d/d(raz)
        result.s1[3] = rt6 * (63 * rbz_val * rbz_val * raz_val * raz_val - 7 * rbz_val * rbz_val - 7 * raz_val * raz_val + 28 * rbz_val * raz_val * sf.czz() + 1 + 2 * sf.czz() * sf.czz()) / 16; // d/d(rbx)
        result.s1[5] = rt6 * (126 * rbx_val * rbz_val * raz_val * raz_val - 14 * rbx_val * rbz_val + 28 * rbx_val * raz_val * sf.czz() + 28 * rbz_val * raz_val * sf.czx() + 4 * sf.czx() * sf.czz()) / 16; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[12] = rt6 * (14 * rbz_val * rbz_val * raz_val - 2 * raz_val + 4 * rbz_val * sf.czz()) / 16; // ∂s0/∂czx
        result.s1[14] = rt6 * (28 * rbx_val * rbz_val * raz_val + 4 * rbx_val * sf.czz() + 4 * rbz_val * sf.czx()) / 16; // ∂s0/∂czz

        if (level >= 2) {
            // Orient case 195: Q20 × O31c
            result.s2[5] = rt6 * (126 * rbx_val * rbz_val * rbz_val - 14 * rbx_val) / 16;
            result.s2[8] = rt6 * (126 * rbz_val * rbz_val * raz_val - 14 * raz_val + 28 * rbz_val * sf.czz()) / 16;
            result.s2[17] = rt6 * (252 * rbx_val * rbz_val * raz_val + 28 * rbx_val * sf.czz() + 28 * rbz_val * sf.czx()) / 16;
            result.s2[18] = rt6 * (126 * raz_val * raz_val * rbz_val - 14 * rbz_val + 28 * raz_val * sf.czz()) / 16;
            result.s2[20] = rt6 * (126 * raz_val * raz_val * rbx_val - 14 * rbx_val + 28 * raz_val * sf.czx()) / 16;
            result.s2[38] = rt6 * (14 * rbz_val * rbz_val - 2) / 16;
            result.s2[41] = rt6 * (28 * raz_val * rbz_val + 4 * sf.czz()) / 16;
            result.s2[107] = 7.0 / 4.0 * rt6 * rbx_val * rbz_val;
            result.s2[108] = rt6 * (28 * raz_val * rbz_val + 4 * sf.czz()) / 16;
            result.s2[110] = rt6 * (28 * raz_val * rbx_val + 4 * sf.czx()) / 16;
            result.s2[113] = rt6 * rbz_val / 4;
            result.s2[119] = rt6 * rbx_val / 4;
        }
    }
}

/**
 * Quadrupole-20 × Octopole-31s kernel
 * Orient case 196: Q20 × Q31s
 */
void quadrupole_20_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt6 *
        (63 * rby_val * rbz_val * rbz_val * raz_val * raz_val -
         7 * rby_val * rbz_val * rbz_val - 7 * raz_val * raz_val * rby_val +
         28 * rby_val * rbz_val * raz_val * sf.czz() +
         14 * rbz_val * rbz_val * raz_val * sf.czy() + rby_val -
         2 * raz_val * sf.czy() + 2 * rby_val * sf.czz() * sf.czz() +
         4 * rbz_val * sf.czy() * sf.czz()) /
        16;

    if (level >= 1) {
        // Coordinate derivatives
        result.s1[2] = rt6 * (126 * rby_val * rbz_val * rbz_val * raz_val - 14 * rby_val + 28 * rby_val * rbz_val * sf.czz() + 14 * rbz_val * rbz_val * sf.czy() - 2 * sf.czy()) / 16; // d/d(raz)
        result.s1[4] = rt6 * (63 * rbz_val * rbz_val * raz_val * raz_val - 7 * rbz_val * rbz_val - 7 * raz_val * raz_val + 28 * rbz_val * raz_val * sf.czz() + 1 + 2 * sf.czz() * sf.czz()) / 16; // d/d(rby)
        result.s1[5] = rt6 * (126 * rby_val * rbz_val * raz_val * raz_val - 14 * rby_val * rbz_val + 28 * rby_val * raz_val * sf.czz() + 28 * rbz_val * raz_val * sf.czy() + 4 * sf.czy() * sf.czz()) / 16; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[13] = rt6 * (14 * rbz_val * rbz_val * raz_val - 2 * raz_val + 4 * rbz_val * sf.czz()) / 16; // ∂s0/∂czy
        result.s1[14] = rt6 * (28 * rby_val * rbz_val * raz_val + 4 * rby_val * sf.czz() + 4 * rbz_val * sf.czy()) / 16; // ∂s0/∂czz

        if (level >= 2) {
            // Orient case 196: Q20 × O31s
            result.s2[5] = rt6 * (126 * rby_val * rbz_val * rbz_val - 14 * rby_val) / 16;
            result.s2[12] = rt6 * (126 * rbz_val * rbz_val * raz_val - 14 * raz_val + 28 * rbz_val * sf.czz()) / 16;
            result.s2[17] = rt6 * (252 * rby_val * rbz_val * raz_val + 28 * rby_val * sf.czz() + 28 * rbz_val * sf.czy()) / 16;
            result.s2[19] = rt6 * (126 * raz_val * raz_val * rbz_val - 14 * rbz_val + 28 * raz_val * sf.czz()) / 16;
            result.s2[20] = rt6 * (126 * raz_val * raz_val * rby_val - 14 * rby_val + 28 * raz_val * sf.czy()) / 16;
            result.s2[68] = rt6 * (14 * rbz_val * rbz_val - 2) / 16;
            result.s2[71] = rt6 * (28 * raz_val * rbz_val + 4 * sf.czz()) / 16;
            result.s2[107] = 7.0 / 4.0 * rt6 * rby_val * rbz_val;
            result.s2[109] = rt6 * (28 * raz_val * rbz_val + 4 * sf.czz()) / 16;
            result.s2[110] = rt6 * (28 * raz_val * rby_val + 4 * sf.czy()) / 16;
            result.s2[116] = rt6 * rbz_val / 4;
            result.s2[119] = rt6 * rby_val / 4;
        }
    }
}

/**
 * Quadrupole-20 × Octopole-32c kernel
 * Orient case 197: Q20 × Q32c
 */
void quadrupole_20_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt15 *
        (63 * rbx_val * rbx_val * rbz_val * raz_val * raz_val -
         63 * rby_val * rby_val * rbz_val * raz_val * raz_val -
         7 * rbx_val * rbx_val * rbz_val + 7 * rby_val * rby_val * rbz_val +
         14 * rbx_val * rbx_val * raz_val * sf.czz() +
         28 * rbx_val * rbz_val * raz_val * sf.czx() -
         14 * rby_val * rby_val * raz_val * sf.czz() -
         28 * rby_val * rbz_val * raz_val * sf.czy() + 4 * rbx_val * sf.czx() * sf.czz() -
         4 * rby_val * sf.czy() * sf.czz() + 2 * rbz_val * sf.czx() * sf.czx() -
         2 * rbz_val * sf.czy() * sf.czy()) /
        40;

    if (level >= 1) {
        // Coordinate derivatives
        result.s1[2] = rt15 * (126 * rbx_val * rbx_val * rbz_val * raz_val - 126 * rby_val * rby_val * rbz_val * raz_val + 14 * rbx_val * rbx_val * sf.czz() + 28 * rbx_val * rbz_val * sf.czx() - 14 * rby_val * rby_val * sf.czz() - 28 * rby_val * rbz_val * sf.czy()) / 40; // d/d(raz)
        result.s1[3] = rt15 * (126 * rbx_val * rbz_val * raz_val * raz_val - 14 * rbx_val * rbz_val + 28 * rbx_val * raz_val * sf.czz() + 28 * rbz_val * raz_val * sf.czx() + 4 * sf.czx() * sf.czz()) / 40; // d/d(rbx)
        result.s1[4] = rt15 * (-126 * rby_val * rbz_val * raz_val * raz_val + 14 * rby_val * rbz_val - 28 * rby_val * raz_val * sf.czz() - 28 * rbz_val * raz_val * sf.czy() - 4 * sf.czy() * sf.czz()) / 40; // d/d(rby)
        result.s1[5] = rt15 * (63 * rbx_val * rbx_val * raz_val * raz_val - 63 * rby_val * rby_val * raz_val * raz_val - 7 * rbx_val * rbx_val + 7 * rby_val * rby_val + 28 * rbx_val * raz_val * sf.czx() - 28 * rby_val * raz_val * sf.czy() + 2 * sf.czx() * sf.czx() - 2 * sf.czy() * sf.czy()) / 40; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[12] = rt15 * (28 * rbx_val * rbz_val * raz_val + 4 * rbx_val * sf.czz() + 4 * rbz_val * sf.czx()) / 40; // ∂s0/∂czx
        result.s1[13] = rt15 * (-28 * rby_val * rbz_val * raz_val - 4 * rby_val * sf.czz() - 4 * rbz_val * sf.czy()) / 40; // ∂s0/∂czy
        result.s1[14] = rt15 * (14 * rbx_val * rbx_val * raz_val - 14 * rby_val * rby_val * raz_val + 4 * rbx_val * sf.czx() - 4 * rby_val * sf.czy()) / 40; // ∂s0/∂czz

        if (level >= 2) {
            // Orient case 197: Q20 × O32c
            result.s2[5] = rt15 * (126 * rbx_val * rbx_val * rbz_val - 126 * rby_val * rby_val * rbz_val) / 40;
            result.s2[8] = rt15 * (252 * rbx_val * rbz_val * raz_val + 28 * rbx_val * sf.czz() + 28 * rbz_val * sf.czx()) / 40;
            result.s2[9] = rt15 * (126 * raz_val * raz_val * rbz_val - 14 * rbz_val + 28 * raz_val * sf.czz()) / 40;
            result.s2[12] = rt15 * (-252 * rby_val * rbz_val * raz_val - 28 * rby_val * sf.czz() - 28 * rbz_val * sf.czy()) / 40;
            result.s2[14] = rt15 * (-126 * raz_val * raz_val * rbz_val + 14 * rbz_val - 28 * raz_val * sf.czz()) / 40;
            result.s2[17] = rt15 * (126 * rbx_val * rbx_val * raz_val - 126 * rby_val * rby_val * raz_val + 28 * rbx_val * sf.czx() - 28 * rby_val * sf.czy()) / 40;
            result.s2[18] = rt15 * (126 * raz_val * raz_val * rbx_val - 14 * rbx_val + 28 * raz_val * sf.czx()) / 40;
            result.s2[19] = rt15 * (-126 * raz_val * raz_val * rby_val + 14 * rby_val - 28 * raz_val * sf.czy()) / 40;
            result.s2[38] = 7.0 / 10.0 * rt15 * rbx_val * rbz_val;
            result.s2[39] = rt15 * (28 * raz_val * rbz_val + 4 * sf.czz()) / 40;
            result.s2[41] = rt15 * (28 * raz_val * rbx_val + 4 * sf.czx()) / 40;
            result.s2[44] = rt15 * rbz_val / 10;
            result.s2[68] = -7.0 / 10.0 * rt15 * rby_val * rbz_val;
            result.s2[70] = rt15 * (-28 * raz_val * rbz_val - 4 * sf.czz()) / 40;
            result.s2[71] = rt15 * (-28 * raz_val * rby_val - 4 * sf.czy()) / 40;
            result.s2[77] = -rt15 * rbz_val / 10;
            result.s2[107] = rt15 * (14 * rbx_val * rbx_val - 14 * rby_val * rby_val) / 40;
            result.s2[108] = rt15 * (28 * raz_val * rbx_val + 4 * sf.czx()) / 40;
            result.s2[109] = rt15 * (-28 * raz_val * rby_val - 4 * sf.czy()) / 40;
            result.s2[113] = rt15 * rbx_val / 10;
            result.s2[116] = -rt15 * rby_val / 10;
        }
    }
}

/**
 * Quadrupole-20 × Octopole-32s kernel
 * Orient case 198: Q20 × Q32s
 */
void quadrupole_20_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt15 *
        (63 * rbx_val * rby_val * rbz_val * raz_val * raz_val -
         7 * rbx_val * rby_val * rbz_val +
         14 * rbx_val * rby_val * raz_val * sf.czz() +
         14 * rbx_val * rbz_val * raz_val * sf.czy() +
         14 * rby_val * rbz_val * raz_val * sf.czx() + 2 * rbx_val * sf.czy() * sf.czz() +
         2 * rby_val * sf.czx() * sf.czz() + 2 * rbz_val * sf.czx() * sf.czy()) /
        20;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt15 * (126 * rbx_val * rby_val * rbz_val * raz_val + 14 * rbx_val * rby_val * sf.czz() + 14 * rbx_val * rbz_val * sf.czy() + 14 * rby_val * rbz_val * sf.czx()) / 20; // d/d(raz)
        result.s1[3] = rt15 * (63 * rby_val * rbz_val * raz_val * raz_val - 7 * rby_val * rbz_val + 14 * rby_val * raz_val * sf.czz() + 14 * rbz_val * raz_val * sf.czy() + 2 * sf.czy() * sf.czz()) / 20; // d/d(rbx)
        result.s1[4] = rt15 * (63 * rbx_val * rbz_val * raz_val * raz_val - 7 * rbx_val * rbz_val + 14 * rbx_val * raz_val * sf.czz() + 14 * rbz_val * raz_val * sf.czx() + 2 * sf.czx() * sf.czz()) / 20; // d/d(rby)
        result.s1[5] = rt15 * (63 * rbx_val * rby_val * raz_val * raz_val - 7 * rbx_val * rby_val + 14 * rbx_val * raz_val * sf.czy() + 14 * rby_val * raz_val * sf.czx() + 2 * sf.czx() * sf.czy()) / 20; // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[12] = rt15 * (14 * rby_val * rbz_val * raz_val + 2 * rby_val * sf.czz() + 2 * rbz_val * sf.czy()) / 20; // ∂s0/∂czx
        result.s1[13] = rt15 * (14 * rbx_val * rbz_val * raz_val + 2 * rbx_val * sf.czz() + 2 * rbz_val * sf.czx()) / 20; // ∂s0/∂czy
        result.s1[14] = rt15 * (14 * rbx_val * rby_val * raz_val + 2 * rbx_val * sf.czy() + 2 * rby_val * sf.czx()) / 20; // ∂s0/∂czz

        if (level >= 2) {
            // Orient case 198: Q20 × O32s
            result.s2[5] = 63.0 / 10.0 * rt15 * rbx_val * rby_val * rbz_val;
            result.s2[8] = rt15 * (126 * rby_val * rbz_val * raz_val + 14 * rby_val * sf.czz() + 14 * rbz_val * sf.czy()) / 20;
            result.s2[12] = rt15 * (126 * rbx_val * rbz_val * raz_val + 14 * rbx_val * sf.czz() + 14 * rbz_val * sf.czx()) / 20;
            result.s2[13] = rt15 * (63 * raz_val * raz_val * rbz_val - 7 * rbz_val + 14 * raz_val * sf.czz()) / 20;
            result.s2[17] = rt15 * (126 * rbx_val * rby_val * raz_val + 14 * rbx_val * sf.czy() + 14 * rby_val * sf.czx()) / 20;
            result.s2[18] = rt15 * (63 * raz_val * raz_val * rby_val - 7 * rby_val + 14 * raz_val * sf.czy()) / 20;
            result.s2[19] = rt15 * (63 * raz_val * raz_val * rbx_val - 7 * rbx_val + 14 * raz_val * sf.czx()) / 20;
            result.s2[38] = 7.0 / 10.0 * rt15 * rby_val * rbz_val;
            result.s2[40] = rt15 * (14 * raz_val * rbz_val + 2 * sf.czz()) / 20;
            result.s2[41] = rt15 * (14 * raz_val * rby_val + 2 * sf.czy()) / 20;
            result.s2[68] = 7.0 / 10.0 * rt15 * rbx_val * rbz_val;
            result.s2[69] = rt15 * (14 * raz_val * rbz_val + 2 * sf.czz()) / 20;
            result.s2[71] = rt15 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 20;
            result.s2[74] = rt15 * rbz_val / 10;
            result.s2[107] = 7.0 / 10.0 * rt15 * rbx_val * rby_val;
            result.s2[108] = rt15 * (14 * raz_val * rby_val + 2 * sf.czy()) / 20;
            result.s2[109] = rt15 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 20;
            result.s2[113] = rt15 * rby_val / 10;
            result.s2[116] = rt15 * rbx_val / 10;
        }
    }
}

/**
 * Quadrupole-20 × Octopole-33c kernel
 * Orient case 199: Q20 × Q33c
 */
void quadrupole_20_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt10 *
        (63 * rbx_val * rbx_val * rbx_val * raz_val * raz_val -
         189 * rbx_val * rby_val * rby_val * raz_val * raz_val -
         7 * rbx_val * rbx_val * rbx_val +
         21 * rbx_val * rby_val * rby_val +
         42 * rbx_val * rbx_val * raz_val * sf.czx() -
         84 * rbx_val * rby_val * raz_val * sf.czy() -
         42 * rby_val * rby_val * raz_val * sf.czx() + 6 * rbx_val * sf.czx() * sf.czx() -
         6 * rbx_val * sf.czy() * sf.czy() - 12 * rby_val * sf.czx() * sf.czy()) /
        80;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt10 * (126 * rbx_val * rbx_val * rbx_val * raz_val - 378 * rbx_val * rby_val * rby_val * raz_val + 42 * rbx_val * rbx_val * sf.czx() - 84 * rbx_val * rby_val * sf.czy() - 42 * rby_val * rby_val * sf.czx()) / 80; // d/d(raz)
        result.s1[3] = rt10 * (189 * rbx_val * rbx_val * raz_val * raz_val - 189 * rby_val * rby_val * raz_val * raz_val - 21 * rbx_val * rbx_val + 21 * rby_val * rby_val + 84 * rbx_val * raz_val * sf.czx() - 84 * rby_val * raz_val * sf.czy() + 6 * sf.czx() * sf.czx() - 6 * sf.czy() * sf.czy()) / 80; // d/d(rbx)
        result.s1[4] = rt10 * (-378 * rbx_val * rby_val * raz_val * raz_val + 42 * rbx_val * rby_val - 84 * rbx_val * raz_val * sf.czy() - 84 * rby_val * raz_val * sf.czx() - 12 * sf.czx() * sf.czy()) / 80; // d/d(rby)
        // Orientation matrix derivatives
        result.s1[12] = rt10 * (42 * rbx_val * rbx_val * raz_val - 42 * rby_val * rby_val * raz_val + 12 * rbx_val * sf.czx() - 12 * rby_val * sf.czy()) / 80; // ∂s0/∂czx
        result.s1[13] = rt10 * (-84 * rbx_val * rby_val * raz_val - 12 * rbx_val * sf.czy() - 12 * rby_val * sf.czx()) / 80; // ∂s0/∂czy

        if (level >= 2) {
            // Orient case 199: Q20 × O33c
            result.s2[5] = rt10 * (126 * rbx_val * rbx_val * rbx_val - 378 * rbx_val * rby_val * rby_val) / 80;
            result.s2[8] = rt10 * (378 * rbx_val * rbx_val * raz_val - 378 * rby_val * rby_val * raz_val + 84 * rbx_val * sf.czx() - 84 * rby_val * sf.czy()) / 80;
            result.s2[9] = rt10 * (378 * raz_val * raz_val * rbx_val - 42 * rbx_val + 84 * raz_val * sf.czx()) / 80;
            result.s2[12] = rt10 * (-756 * rbx_val * rby_val * raz_val - 84 * rbx_val * sf.czy() - 84 * rby_val * sf.czx()) / 80;
            result.s2[13] = rt10 * (-378 * raz_val * raz_val * rby_val + 42 * rby_val - 84 * raz_val * sf.czy()) / 80;
            result.s2[14] = rt10 * (-378 * raz_val * raz_val * rbx_val + 42 * rbx_val - 84 * raz_val * sf.czx()) / 80;
            result.s2[38] = rt10 * (42 * rbx_val * rbx_val - 42 * rby_val * rby_val) / 80;
            result.s2[39] = rt10 * (84 * raz_val * rbx_val + 12 * sf.czx()) / 80;
            result.s2[40] = rt10 * (-84 * raz_val * rby_val - 12 * sf.czy()) / 80;
            result.s2[44] = 3.0 / 20.0 * rt10 * rbx_val;
            result.s2[68] = -21.0 / 20.0 * rt10 * rbx_val * rby_val;
            result.s2[69] = rt10 * (-84 * raz_val * rby_val - 12 * sf.czy()) / 80;
            result.s2[70] = rt10 * (-84 * raz_val * rbx_val - 12 * sf.czx()) / 80;
            result.s2[74] = -3.0 / 20.0 * rt10 * rby_val;
            result.s2[77] = -3.0 / 20.0 * rt10 * rbx_val;
        }
    }
}

/**
 * Quadrupole-20 × Octopole-33s kernel
 * Orient case 200: Q20 × Q33s
 */
void quadrupole_20_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt10 *
        (189 * rbx_val * rbx_val * rby_val * raz_val * raz_val -
         63 * rby_val * rby_val * rby_val * raz_val * raz_val -
         21 * rbx_val * rbx_val * rby_val +
         7 * rby_val * rby_val * rby_val +
         42 * rbx_val * rbx_val * raz_val * sf.czy() +
         84 * rbx_val * rby_val * raz_val * sf.czx() -
         42 * rby_val * rby_val * raz_val * sf.czy() + 12 * rbx_val * sf.czx() * sf.czy() +
         6 * rby_val * sf.czx() * sf.czx() - 6 * rby_val * sf.czy() * sf.czy()) /
        80;

    if (level >= 1) {
        // First derivatives
        result.s1[2] = rt10 * (378 * rbx_val * rbx_val * rby_val * raz_val - 126 * rby_val * rby_val * rby_val * raz_val + 42 * rbx_val * rbx_val * sf.czy() + 84 * rbx_val * rby_val * sf.czx() - 42 * rby_val * rby_val * sf.czy()) / 80; // d/d(raz)
        result.s1[3] = rt10 * (378 * rbx_val * rby_val * raz_val * raz_val - 42 * rbx_val * rby_val + 84 * rbx_val * raz_val * sf.czy() + 84 * rby_val * raz_val * sf.czx() + 12 * sf.czx() * sf.czy()) / 80; // d/d(rbx)
        result.s1[4] = rt10 * (189 * rbx_val * rbx_val * raz_val * raz_val - 189 * rby_val * rby_val * raz_val * raz_val - 21 * rbx_val * rbx_val + 21 * rby_val * rby_val + 84 * rbx_val * raz_val * sf.czx() - 84 * rby_val * raz_val * sf.czy() + 6 * sf.czx() * sf.czx() - 6 * sf.czy() * sf.czy()) / 80; // d/d(rby)
        // Orientation matrix derivatives
        result.s1[12] = rt10 * (84 * rbx_val * rby_val * raz_val + 12 * rbx_val * sf.czy() + 12 * rby_val * sf.czx()) / 80; // ∂s0/∂czx
        result.s1[13] = rt10 * (42 * rbx_val * rbx_val * raz_val - 42 * rby_val * rby_val * raz_val + 12 * rbx_val * sf.czx() - 12 * rby_val * sf.czy()) / 80; // ∂s0/∂czy

        if (level >= 2) {
            // Orient case 200: Q20 × O33s
            result.s2[5] = rt10 * (378 * rbx_val * rbx_val * rby_val - 126 * rby_val * rby_val * rby_val) / 80;
            result.s2[8] = rt10 * (756 * rbx_val * rby_val * raz_val + 84 * rbx_val * sf.czy() + 84 * rby_val * sf.czx()) / 80;
            result.s2[9] = rt10 * (378 * raz_val * raz_val * rby_val - 42 * rby_val + 84 * raz_val * sf.czy()) / 80;
            result.s2[12] = rt10 * (378 * rbx_val * rbx_val * raz_val - 378 * rby_val * rby_val * raz_val + 84 * rbx_val * sf.czx() - 84 * rby_val * sf.czy()) / 80;
            result.s2[13] = rt10 * (378 * raz_val * raz_val * rbx_val - 42 * rbx_val + 84 * raz_val * sf.czx()) / 80;
            result.s2[14] = rt10 * (-378 * raz_val * raz_val * rby_val + 42 * rby_val - 84 * raz_val * sf.czy()) / 80;
            result.s2[38] = 21.0 / 20.0 * rt10 * rbx_val * rby_val;
            result.s2[39] = rt10 * (84 * raz_val * rby_val + 12 * sf.czy()) / 80;
            result.s2[40] = rt10 * (84 * raz_val * rbx_val + 12 * sf.czx()) / 80;
            result.s2[44] = 3.0 / 20.0 * rt10 * rby_val;
            result.s2[68] = rt10 * (42 * rbx_val * rbx_val - 42 * rby_val * rby_val) / 80;
            result.s2[69] = rt10 * (84 * raz_val * rbx_val + 12 * sf.czx()) / 80;
            result.s2[70] = rt10 * (-84 * raz_val * rby_val - 12 * sf.czy()) / 80;
            result.s2[74] = 3.0 / 20.0 * rt10 * rbx_val;
            result.s2[77] = -3.0 / 20.0 * rt10 * rby_val;
        }
    }
}

/**
 * Quadrupole-21c × Octopole-30 kernel
 * Orient case 201: Q21c × Q30
 */
void quadrupole_21c_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        63.0 / 8.0 * rbz_val * rbz_val * rbz_val * rax_val * raz_val -
        7.0 / 8.0 * rax_val * raz_val * rbz_val +
        21.0 / 4.0 * rbz_val * rbz_val * rax_val * sf.czz() +
        21.0 / 4.0 * rbz_val * rbz_val * raz_val * sf.czx() -
        3.0 / 4.0 * rax_val * sf.czz() - 3.0 / 4.0 * raz_val * sf.czx() +
        3.0 / 2.0 * rbz_val * sf.czx() * sf.czz();

    if (level >= 1) {
        // First derivatives
        result.s1[0] = 63.0 / 8.0 * rbz_val * rbz_val * rbz_val * raz_val - 7.0 / 8.0 * raz_val * rbz_val + 21.0 / 4.0 * rbz_val * rbz_val * sf.czz() - 3.0 / 4.0 * sf.czz(); // d/d(rax)
        result.s1[2] = 63.0 / 8.0 * rbz_val * rbz_val * rbz_val * rax_val - 7.0 / 8.0 * rax_val * rbz_val + 21.0 / 4.0 * rbz_val * rbz_val * sf.czx() - 3.0 / 4.0 * sf.czx(); // d/d(raz)
        result.s1[5] = 189.0 / 8.0 * rbz_val * rbz_val * rax_val * raz_val - 7.0 / 8.0 * rax_val * raz_val + 42.0 / 4.0 * rbz_val * rax_val * sf.czz() + 42.0 / 4.0 * rbz_val * raz_val * sf.czx() + 3.0 / 2.0 * sf.czx() * sf.czz(); // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[8] = rt3 * (7 * rbz_val * rbz_val * raz_val - raz_val + 2 * rbz_val * sf.czz()) / 4; // ∂s0/∂cxz
        result.s1[14] = rt3 * (7 * rbz_val * rbz_val * rax_val - rax_val + 2 * rbz_val * sf.cxz()) / 4; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 201
            result.s2[3] = rt3 * (21 * rbz_val * rbz_val * rbz_val - 7 * rbz_val) / 4;
            result.s2[15] = rt3 * (63 * rbz_val * rbz_val * raz_val - 7 * raz_val + 14 * rbz_val * sf.czz()) / 4;
            result.s2[17] = rt3 * (63 * rbz_val * rbz_val * rax_val - 7 * rax_val + 14 * rbz_val * sf.cxz()) / 4;
            result.s2[20] = rt3 * (126 * rax_val * raz_val * rbz_val + 14 * rax_val * sf.czz() + 14 * raz_val * sf.cxz()) / 4;
            result.s2[80] = rt3 * (7 * rbz_val * rbz_val - 1) / 4;
            result.s2[83] = rt3 * (14 * raz_val * rbz_val + 2 * sf.czz()) / 4;
            result.s2[105] = rt3 * (7 * rbz_val * rbz_val - 1) / 4;
            result.s2[110] = rt3 * (14 * rbz_val * rax_val + 2 * sf.cxz()) / 4;
            result.s2[117] = rt3 * rbz_val / 2;
        }
    }
}

/**
 * Quadrupole-21c × Octopole-31c kernel
 * Orient case 202: Q21c × Q31c
 */
void quadrupole_21c_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt2 *
                (63 * rbx_val * rbz_val * rbz_val * rax_val * raz_val -
                 7 * rax_val * raz_val * rbx_val +
                 14 * rbx_val * rbz_val * rax_val * sf.czz() +
                 14 * rbx_val * rbz_val * raz_val * sf.czx() +
                 7 * rbz_val * rbz_val * rax_val * sf.czx() +
                 7 * rbz_val * rbz_val * raz_val * sf.cxx() - rax_val * sf.czx() -
                 raz_val * sf.cxx() + 2 * rbx_val * sf.czx() * sf.czz() +
                 2 * rbz_val * sf.cxx() * sf.czz() + 2 * rbz_val * sf.czx() * sf.czx()) /
                8;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt2 * (63 * rbx_val * rbz_val * rbz_val * raz_val - 7 * raz_val * rbx_val + 14 * rbx_val * rbz_val * sf.czz() + 7 * rbz_val * rbz_val * sf.czx() - sf.czx()) / 8; // d/d(rax)
        result.s1[2] = rt2 * (63 * rbx_val * rbz_val * rbz_val * rax_val - 7 * rax_val * rbx_val + 14 * rbx_val * rbz_val * sf.czx() + 7 * rbz_val * rbz_val * sf.cxx() - sf.cxx()) / 8; // d/d(raz)
        result.s1[3] = rt2 * (63 * rbz_val * rbz_val * rax_val * raz_val - 7 * rax_val * raz_val + 14 * rbz_val * rax_val * sf.czz() + 14 * rbz_val * raz_val * sf.czx() + 2 * sf.czx() * sf.czz()) / 8; // d/d(rbx)
        result.s1[5] = rt2 * (126 * rbx_val * rbz_val * rax_val * raz_val + 14 * rbx_val * rax_val * sf.czz() + 14 * rbx_val * raz_val * sf.czx() + 14 * rbz_val * rax_val * sf.czx() + 14 * rbz_val * raz_val * sf.cxx() + 2 * sf.cxx() * sf.czz() + 4 * rbz_val * sf.czx() * sf.czx()) / 8; // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[6] = rt2 * (7 * rbz_val * rbz_val * raz_val - raz_val + 2 * rbz_val * sf.czz()) / 8; // ∂s0/∂cxx
        result.s1[12] = rt2 * (7 * rbz_val * rbz_val * rax_val - rax_val + 2 * rbz_val * sf.cxz()) / 8; // ∂s0/∂czx
        result.s1[8] = rt2 * (14 * rbx_val * rbz_val * raz_val + 2 * rbx_val * sf.czz() + 2 * rbz_val * sf.czx()) / 8; // ∂s0/∂cxz
        result.s1[14] = rt2 * (14 * rbx_val * rbz_val * rax_val + 2 * rbx_val * sf.cxz() + 2 * rbz_val * sf.cxx()) / 8; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 202
            result.s2[3] = rt2 * (63 * rbx_val * rbz_val * rbz_val - 7 * rbx_val) / 8;
            result.s2[6] = rt2 * (63 * rbz_val * rbz_val * raz_val - 7 * raz_val + 14 * rbz_val * sf.czz()) / 8;
            result.s2[8] = rt2 * (63 * rbz_val * rbz_val * rax_val - 7 * rax_val + 14 * rbz_val * sf.cxz()) / 8;
            result.s2[15] = rt2 * (126 * rbx_val * rbz_val * raz_val + 14 * rbx_val * sf.czz() + 14 * rbz_val * sf.czx()) / 8;
            result.s2[17] = rt2 * (126 * rbx_val * rbz_val * rax_val + 14 * rbx_val * sf.cxz() + 14 * rbz_val * sf.cxx()) / 8;
            result.s2[18] = rt2 * (126 * rax_val * raz_val * rbz_val + 14 * rax_val * sf.czz() + 14 * raz_val * sf.cxz()) / 8;
            result.s2[20] = rt2 * (126 * rax_val * raz_val * rbx_val + 14 * rax_val * sf.czx() + 14 * raz_val * sf.cxx()) / 8;
            result.s2[23] = rt2 * (7 * rbz_val * rbz_val - 1) / 8;
            result.s2[26] = rt2 * (14 * raz_val * rbz_val + 2 * sf.czz()) / 8;
            result.s2[36] = rt2 * (7 * rbz_val * rbz_val - 1) / 8;
            result.s2[41] = rt2 * (14 * rbz_val * rax_val + 2 * sf.cxz()) / 8;
            result.s2[80] = 7.0 / 4.0 * rt2 * rbx_val * rbz_val;
            result.s2[81] = rt2 * (14 * raz_val * rbz_val + 2 * sf.czz()) / 8;
            result.s2[83] = rt2 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 8;
            result.s2[86] = rt2 * rbz_val / 4;
            result.s2[105] = 7.0 / 4.0 * rt2 * rbx_val * rbz_val;
            result.s2[108] = rt2 * (14 * rbz_val * rax_val + 2 * sf.cxz()) / 8;
            result.s2[110] = rt2 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 8;
            result.s2[111] = rt2 * rbz_val / 4;
            result.s2[117] = rt2 * rbx_val / 4;
        }
    }
}

/**
 * Quadrupole-21c × Octopole-31s kernel
 * Orient case 203: Q21c × Q31s
 */
void quadrupole_21c_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt2 *
                (63 * rby_val * rbz_val * rbz_val * rax_val * raz_val -
                 7 * rax_val * raz_val * rby_val +
                 14 * rby_val * rbz_val * rax_val * sf.czz() +
                 14 * rby_val * rbz_val * raz_val * sf.czx() +
                 7 * rbz_val * rbz_val * rax_val * sf.czy() +
                 7 * rbz_val * rbz_val * raz_val * sf.cxy() - rax_val * sf.czy() -
                 raz_val * sf.cxy() + 2 * rby_val * sf.czx() * sf.czz() +
                 2 * rbz_val * sf.cxy() * sf.czz() + 2 * rbz_val * sf.czy() * sf.czx()) /
                8;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt2 * (63 * rby_val * rbz_val * rbz_val * raz_val - 7 * raz_val * rby_val + 14 * rby_val * rbz_val * sf.czz() + 7 * rbz_val * rbz_val * sf.czy() - sf.czy()) / 8; // d/d(rax)
        result.s1[2] = rt2 * (63 * rby_val * rbz_val * rbz_val * rax_val - 7 * rax_val * rby_val + 14 * rby_val * rbz_val * sf.czx() + 7 * rbz_val * rbz_val * sf.cxy() - sf.cxy()) / 8; // d/d(raz)
        result.s1[4] = rt2 * (63 * rbz_val * rbz_val * rax_val * raz_val - 7 * rax_val * raz_val + 14 * rbz_val * rax_val * sf.czz() + 14 * rbz_val * raz_val * sf.czx() + 2 * sf.czx() * sf.czz()) / 8; // d/d(rby)
        result.s1[5] = rt2 * (126 * rby_val * rbz_val * rax_val * raz_val + 14 * rby_val * rax_val * sf.czz() + 14 * rby_val * raz_val * sf.czx() + 14 * rbz_val * rax_val * sf.czy() + 14 * rbz_val * raz_val * sf.cxy() + 2 * sf.cxy() * sf.czz() + 2 * sf.czy() * sf.czx()) / 8; // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[7] = rt2 * (7 * rbz_val * rbz_val * raz_val - raz_val + 2 * rbz_val * sf.czz()) / 8; // ∂s0/∂cxy
        result.s1[13] = rt2 * (7 * rbz_val * rbz_val * rax_val - rax_val + 2 * rbz_val * sf.cxz()) / 8; // ∂s0/∂czy
        result.s1[8] = rt2 * (14 * rby_val * rbz_val * raz_val + 2 * rby_val * sf.czz() + 2 * rbz_val * sf.czy()) / 8; // ∂s0/∂cxz
        result.s1[14] = rt2 * (14 * rby_val * rbz_val * rax_val + 2 * rby_val * sf.cxz() + 2 * rbz_val * sf.cxy()) / 8; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 203
            result.s2[3] = rt2 * (63 * rby_val * rbz_val * rbz_val - 7 * rby_val) / 8;
            result.s2[10] = rt2 * (63 * rbz_val * rbz_val * raz_val - 7 * raz_val + 14 * rbz_val * sf.czz()) / 8;
            result.s2[12] = rt2 * (63 * rbz_val * rbz_val * rax_val - 7 * rax_val + 14 * rbz_val * sf.cxz()) / 8;
            result.s2[15] = rt2 * (126 * rby_val * rbz_val * raz_val + 14 * rby_val * sf.czz() + 14 * rbz_val * sf.czy()) / 8;
            result.s2[17] = rt2 * (126 * rby_val * rbz_val * rax_val + 14 * rby_val * sf.cxz() + 14 * rbz_val * sf.cxy()) / 8;
            result.s2[19] = rt2 * (126 * rax_val * raz_val * rbz_val + 14 * rax_val * sf.czz() + 14 * raz_val * sf.cxz()) / 8;
            result.s2[20] = rt2 * (126 * rax_val * raz_val * rby_val + 14 * rax_val * sf.czy() + 14 * raz_val * sf.cxy()) / 8;
            result.s2[47] = rt2 * (7 * rbz_val * rbz_val - 1) / 8;
            result.s2[50] = rt2 * (14 * raz_val * rbz_val + 2 * sf.czz()) / 8;
            result.s2[66] = rt2 * (7 * rbz_val * rbz_val - 1) / 8;
            result.s2[71] = rt2 * (14 * rbz_val * rax_val + 2 * sf.cxz()) / 8;
            result.s2[80] = 7.0 / 4.0 * rt2 * rby_val * rbz_val;
            result.s2[82] = rt2 * (14 * raz_val * rbz_val + 2 * sf.czz()) / 8;
            result.s2[83] = rt2 * (14 * raz_val * rby_val + 2 * sf.czy()) / 8;
            result.s2[89] = rt2 * rbz_val / 4;
            result.s2[105] = 7.0 / 4.0 * rt2 * rby_val * rbz_val;
            result.s2[109] = rt2 * (14 * rbz_val * rax_val + 2 * sf.cxz()) / 8;
            result.s2[110] = rt2 * (14 * rax_val * rby_val + 2 * sf.cxy()) / 8;
            result.s2[114] = rt2 * rbz_val / 4;
            result.s2[117] = rt2 * rby_val / 4;
        }
    }
}

/**
 * Quadrupole-21c × Octopole-32c kernel
 * Orient case 204: Q21c × Q32c
 */
void quadrupole_21c_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rbx_val * rbz_val * rax_val * raz_val -
                 63 * rby_val * rby_val * rbz_val * rax_val * raz_val +
                 7 * rbx_val * rbx_val * rax_val * sf.czz() +
                 7 * rbx_val * rbx_val * raz_val * sf.czx() +
                 14 * rbx_val * rbz_val * rax_val * sf.czx() +
                 14 * rbx_val * rbz_val * raz_val * sf.cxx() -
                 7 * rby_val * rby_val * rax_val * sf.czz() -
                 7 * rby_val * rby_val * raz_val * sf.czx() -
                 14 * rby_val * rbz_val * rax_val * sf.czy() -
                 14 * rby_val * rbz_val * raz_val * sf.cxy() +
                 2 * rbx_val * sf.cxx() * sf.czz() + 2 * rbx_val * sf.czx() * sf.czx() -
                 2 * rby_val * sf.cxy() * sf.czz() - 2 * rby_val * sf.czy() * sf.czx() +
                 2 * rbz_val * sf.cxx() * sf.czx() - 2 * rbz_val * sf.cxy() * sf.czy()) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt5 * (63 * rbx_val * rbx_val * rbz_val * raz_val - 63 * rby_val * rby_val * rbz_val * raz_val + 7 * rbx_val * rbx_val * sf.czz() + 14 * rbx_val * rbz_val * sf.czx() - 7 * rby_val * rby_val * sf.czz() - 14 * rby_val * rbz_val * sf.czy()) / 20; // d/d(rax)
        result.s1[2] = rt5 * (63 * rbx_val * rbx_val * rbz_val * rax_val - 63 * rby_val * rby_val * rbz_val * rax_val + 7 * rbx_val * rbx_val * sf.czx() + 14 * rbx_val * rbz_val * sf.cxx() - 7 * rby_val * rby_val * sf.czx() - 14 * rby_val * rbz_val * sf.cxy()) / 20; // d/d(raz)
        result.s1[3] = rt5 * (126 * rbx_val * rbz_val * rax_val * raz_val + 14 * rbx_val * rax_val * sf.czz() + 14 * rbx_val * raz_val * sf.czx() + 14 * rbz_val * rax_val * sf.czx() + 14 * rbz_val * raz_val * sf.cxx() + 2 * sf.cxx() * sf.czz() + 4 * sf.czx() * sf.czx() + 2 * rbz_val * sf.cxx() * sf.czx()) / 20; // d/d(rbx)
        result.s1[4] = rt5 * (-126 * rby_val * rbz_val * rax_val * raz_val - 14 * rby_val * rax_val * sf.czz() - 14 * rby_val * raz_val * sf.czx() - 14 * rbz_val * rax_val * sf.czy() - 14 * rbz_val * raz_val * sf.cxy() - 2 * sf.cxy() * sf.czz() - 2 * sf.czy() * sf.czx() - 2 * rbz_val * sf.cxy() * sf.czy()) / 20; // d/d(rby)
        result.s1[5] = rt5 * (63 * rbx_val * rbx_val * rax_val * raz_val - 63 * rby_val * rby_val * rax_val * raz_val + 14 * rbx_val * rax_val * sf.czx() + 14 * rbx_val * raz_val * sf.cxx() - 14 * rby_val * rax_val * sf.czy() - 14 * rby_val * raz_val * sf.cxy() + 2 * sf.cxx() * sf.czx() - 2 * sf.cxy() * sf.czy()) / 20; // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[6] = rt5 * (14 * rbx_val * rbz_val * raz_val + 2 * rbx_val * sf.czz() + 2 * rbz_val * sf.czx()) / 20; // ∂s0/∂cxx
        result.s1[12] = rt5 * (14 * rbx_val * rbz_val * rax_val + 2 * rbx_val * sf.cxz() + 2 * rbz_val * sf.cxx()) / 20; // ∂s0/∂czx
        result.s1[7] = rt5 * (-14 * rby_val * rbz_val * raz_val - 2 * rby_val * sf.czz() - 2 * rbz_val * sf.czy()) / 20; // ∂s0/∂cxy
        result.s1[13] = rt5 * (-14 * rby_val * rbz_val * rax_val - 2 * rby_val * sf.cxz() - 2 * rbz_val * sf.cxy()) / 20; // ∂s0/∂czy
        result.s1[8] = rt5 * (7 * rbx_val * rbx_val * raz_val - 7 * rby_val * rby_val * raz_val + 2 * rbx_val * sf.czx() - 2 * rby_val * sf.czy()) / 20; // ∂s0/∂cxz
        result.s1[14] = rt5 * (7 * rbx_val * rbx_val * rax_val - 7 * rby_val * rby_val * rax_val + 2 * rbx_val * sf.cxx() - 2 * rby_val * sf.cxy()) / 20; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 204
            result.s2[3] = rt5*(63*rbx_val * rbx_val*rbz_val-63*rby_val * rby_val*rbz_val)/20;
            result.s2[6] = rt5*(126*rbx_val*rbz_val*raz_val+14*rbx_val*sf.czz()+14*rbz_val*sf.czx())/20;
            result.s2[8] = rt5*(126*rbx_val*rbz_val*rax_val+14*rbx_val*sf.cxz()+14*rbz_val*sf.cxx())/20;
            result.s2[9] = rt5*(126*rax_val*raz_val*rbz_val+14*rax_val*sf.czz()+14*raz_val*sf.cxz())/20;
            result.s2[10] = rt5*(-126*rby_val*rbz_val*raz_val-14*rby_val*sf.czz()-14*rbz_val*sf.czy())/20;
            result.s2[12] = rt5*(-126*rby_val*rbz_val*rax_val-14*rby_val*sf.cxz()-14*rbz_val*sf.cxy())/20;
            result.s2[14] = rt5*(-126*rax_val*raz_val*rbz_val-14*rax_val*sf.czz()-14*raz_val*sf.cxz())/20;
            result.s2[15] = rt5*(63*rbx_val * rbx_val*raz_val-63*rby_val * rby_val*raz_val+14*rbx_val*sf.czx()-14*rby_val*sf.czy())/20;
            result.s2[17] = rt5*(63*rbx_val * rbx_val*rax_val-63*rby_val * rby_val*rax_val+14*rbx_val*sf.cxx()-14*rby_val*sf.cxy())/20;
            result.s2[18] = rt5*(126*rax_val*raz_val*rbx_val+14*rax_val*sf.czx()+14*raz_val*sf.cxx())/20;
            result.s2[19] = rt5*(-126*rax_val*raz_val*rby_val-14*rax_val*sf.czy()-14*raz_val*sf.cxy())/20;
            result.s2[23] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[24] = rt5*(14*raz_val*rbz_val+2*sf.czz())/20;
            result.s2[26] = rt5*(14*raz_val*rbx_val+2*sf.czx())/20;
            result.s2[36] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[39] = rt5*(14*rbz_val*rax_val+2*sf.cxz())/20;
            result.s2[41] = rt5*(14*rax_val*rbx_val+2*sf.cxx())/20;
            result.s2[42] = rt5*rbz_val/10;
            result.s2[47] = -7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[49] = rt5*(-14*raz_val*rbz_val-2*sf.czz())/20;
            result.s2[50] = rt5*(-14*raz_val*rby_val-2*sf.czy())/20;
            result.s2[66] = -7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[70] = rt5*(-14*rbz_val*rax_val-2*sf.cxz())/20;
            result.s2[71] = rt5*(-14*rax_val*rby_val-2*sf.cxy())/20;
            result.s2[75] = -rt5*rbz_val/10;
            result.s2[80] = rt5*(7*rbx_val * rbx_val-7*rby_val * rby_val)/20;
            result.s2[81] = rt5*(14*raz_val*rbx_val+2*sf.czx())/20;
            result.s2[82] = rt5*(-14*raz_val*rby_val-2*sf.czy())/20;
            result.s2[86] = rt5*rbx_val/10;
            result.s2[89] = -rt5*rby_val/10;
            result.s2[105] = rt5*(7*rbx_val * rbx_val-7*rby_val * rby_val)/20;
            result.s2[108] = rt5*(14*rax_val*rbx_val+2*sf.cxx())/20;
            result.s2[109] = rt5*(-14*rax_val*rby_val-2*sf.cxy())/20;
            result.s2[111] = rt5*rbx_val/10;
            result.s2[114] = -rt5*rby_val/10;
        }
    }
}

/**
 * Quadrupole-21c × Octopole-32s kernel
 * Orient case 205: Q21c × Q32s
 */
void quadrupole_21c_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt5 *
        (63 * rbx_val * rby_val * rbz_val * rax_val * raz_val +
         7 * rbx_val * rby_val * rax_val * sf.czz() +
         7 * rbx_val * rby_val * raz_val * sf.czx() +
         7 * rbx_val * rbz_val * rax_val * sf.czy() +
         7 * rbx_val * rbz_val * raz_val * sf.cxy() +
         7 * rby_val * rbz_val * rax_val * sf.czx() +
         7 * rby_val * rbz_val * raz_val * sf.cxx() + rbx_val * sf.cxy() * sf.czz() +
         rbx_val * sf.czy() * sf.czx() + rby_val * sf.cxx() * sf.czz() + rby_val * sf.czx() * sf.czx() +
         rbz_val * sf.cxx() * sf.czy() + rbz_val * sf.cxy() * sf.czx()) /
        10;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt5 * (63 * rbx_val * rby_val * rbz_val * raz_val + 7 * rbx_val * rby_val * sf.czz() + 7 * rbx_val * rbz_val * sf.czy() + 7 * rby_val * rbz_val * sf.czx()) / 10; // d/d(rax)
        result.s1[2] = rt5 * (63 * rbx_val * rby_val * rbz_val * rax_val + 7 * rbx_val * rby_val * sf.czx() + 7 * rbx_val * rbz_val * sf.cxy() + 7 * rby_val * rbz_val * sf.cxx()) / 10; // d/d(raz)
        result.s1[3] = rt5 * (63 * rby_val * rbz_val * rax_val * raz_val + 7 * rby_val * rax_val * sf.czz() + 7 * rby_val * raz_val * sf.czx() + 7 * rbz_val * rax_val * sf.czy() + 7 * rbz_val * raz_val * sf.cxy() + sf.cxy() * sf.czz() + sf.czy() * sf.czx() + rbz_val * sf.cxy() * sf.czx()) / 10; // d/d(rbx)
        result.s1[4] = rt5 * (63 * rbx_val * rbz_val * rax_val * raz_val + 7 * rbx_val * rax_val * sf.czz() + 7 * rbx_val * raz_val * sf.czx() + 7 * rbz_val * rax_val * sf.czx() + 7 * rbz_val * raz_val * sf.cxx() + sf.cxx() * sf.czz() + sf.czx() * sf.czx() + rbz_val * sf.cxx() * sf.czy()) / 10; // d/d(rby)
        result.s1[5] = rt5 * (63 * rbx_val * rby_val * rax_val * raz_val + 7 * rbx_val * rax_val * sf.czy() + 7 * rbx_val * raz_val * sf.cxy() + 7 * rby_val * rax_val * sf.czx() + 7 * rby_val * raz_val * sf.cxx() + sf.cxx() * sf.czy() + sf.cxy() * sf.czx()) / 10; // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[6] = rt5 * (7 * rby_val * rbz_val * raz_val + rby_val * sf.czz() + rbz_val * sf.czy()) / 10; // ∂s0/∂cxx
        result.s1[12] = rt5 * (7 * rby_val * rbz_val * rax_val + rby_val * sf.cxz() + rbz_val * sf.cxy()) / 10; // ∂s0/∂czx
        result.s1[7] = rt5 * (7 * rbx_val * rbz_val * raz_val + rbx_val * sf.czz() + rbz_val * sf.czx()) / 10; // ∂s0/∂cxy
        result.s1[13] = rt5 * (7 * rbx_val * rbz_val * rax_val + rbx_val * sf.cxz() + rbz_val * sf.cxx()) / 10; // ∂s0/∂czy
        result.s1[8] = rt5 * (7 * rbx_val * rby_val * raz_val + rbx_val * sf.czy() + rby_val * sf.czx()) / 10; // ∂s0/∂cxz
        result.s1[14] = rt5 * (7 * rbx_val * rby_val * rax_val + rbx_val * sf.cxy() + rby_val * sf.cxx()) / 10; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 205
            result.s2[3] = 63.0/10.0*rt5*rbx_val*rby_val*rbz_val;
            result.s2[6] = rt5*(63*rby_val*rbz_val*raz_val+7*rby_val*sf.czz()+7*rbz_val*sf.czy())/10;
            result.s2[8] = rt5*(63*rby_val*rbz_val*rax_val+7*rby_val*sf.cxz()+7*rbz_val*sf.cxy())/10;
            result.s2[10] = rt5*(63*rbx_val*rbz_val*raz_val+7*rbx_val*sf.czz()+7*rbz_val*sf.czx())/10;
            result.s2[12] = rt5*(63*rbx_val*rbz_val*rax_val+7*rbx_val*sf.cxz()+7*rbz_val*sf.cxx())/10;
            result.s2[13] = rt5*(63*rax_val*raz_val*rbz_val+7*rax_val*sf.czz()+7*raz_val*sf.cxz())/10;
            result.s2[15] = rt5*(63*rbx_val*rby_val*raz_val+7*rbx_val*sf.czy()+7*rby_val*sf.czx())/10;
            result.s2[17] = rt5*(63*rbx_val*rby_val*rax_val+7*rbx_val*sf.cxy()+7*rby_val*sf.cxx())/10;
            result.s2[18] = rt5*(63*rax_val*raz_val*rby_val+7*rax_val*sf.czy()+7*raz_val*sf.cxy())/10;
            result.s2[19] = rt5*(63*rax_val*raz_val*rbx_val+7*rax_val*sf.czx()+7*raz_val*sf.cxx())/10;
            result.s2[23] = 7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[25] = rt5*(14*rby_val*rbz_val+2*sf.czy())/10;
            result.s2[26] = rt5*(7*raz_val*rby_val+sf.czy())/10;
            result.s2[36] = 7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[40] = rt5*(14*rby_val*rax_val+2*sf.cxy())/10;
            result.s2[41] = rt5*(7*rax_val*rby_val+sf.cxy())/10;
            result.s2[47] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[48] = rt5*(7*raz_val*rbz_val+sf.czz())/10;
            result.s2[50] = rt5*(7*raz_val*rbx_val+sf.czx())/10;
            result.s2[53] = rt5*rbz_val/10;
            result.s2[66] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[69] = rt5*(7*rbz_val*rax_val+sf.cxz())/10;
            result.s2[71] = rt5*(7*rax_val*rbx_val+sf.cxx())/10;
            result.s2[72] = rt5*rbz_val/10;
            result.s2[80] = 7.0/10.0*rt5*rbx_val*rby_val;
            result.s2[81] = rt5*(7*raz_val*rby_val+sf.czy())/10;
            result.s2[82] = rt5*(7*raz_val*rbx_val+sf.czx())/10;
            result.s2[86] = rt5*rby_val/10;
            result.s2[89] = rt5*rbx_val/10;
            result.s2[105] = 7.0/10.0*rt5*rbx_val*rby_val;
            result.s2[108] = rt5*(7*rax_val*rby_val+sf.cxy())/10;
            result.s2[109] = rt5*(7*rax_val*rbx_val+sf.cxx())/10;
            result.s2[111] = rt5*rby_val/10;
            result.s2[114] = rt5*rbx_val/10;
        }
    }
}

/**
 * Quadrupole-21c × Octopole-33c kernel
 * Orient case 206: Q21c × Q33c
 */
void quadrupole_21c_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt30 *
                (21 * rbx_val * rbx_val * rbx_val * rax_val * raz_val -
                 63 * rbx_val * rby_val * rby_val * rax_val * raz_val +
                 7 * rbx_val * rbx_val * rax_val * sf.czx() +
                 7 * rbx_val * rbx_val * raz_val * sf.cxx() -
                 14 * rbx_val * rby_val * rax_val * sf.czy() -
                 14 * rbx_val * rby_val * raz_val * sf.cxy() -
                 7 * rby_val * rby_val * rax_val * sf.czx() -
                 7 * rby_val * rby_val * raz_val * sf.cxx() +
                 2 * rbx_val * sf.cxx() * sf.czx() - 2 * rbx_val * sf.cxy() * sf.czy() -
                 2 * rby_val * sf.cxx() * sf.czy() - 2 * rby_val * sf.czx() * sf.cxy()) /
                40;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt30 * (21 * rbx_val * rbx_val * rbx_val * raz_val - 63 * rbx_val * rby_val * rby_val * raz_val + 7 * rbx_val * rbx_val * sf.czx() - 14 * rbx_val * rby_val * sf.czy() - 7 * rby_val * rby_val * sf.czx()) / 40; // d/d(rax)
        result.s1[2] = rt30 * (21 * rbx_val * rbx_val * rbx_val * rax_val - 63 * rbx_val * rby_val * rby_val * rax_val + 7 * rbx_val * rbx_val * sf.cxx() - 14 * rbx_val * rby_val * sf.cxy() - 7 * rby_val * rby_val * sf.cxx()) / 40; // d/d(raz)
        result.s1[3] = rt30 * (63 * rbx_val * rbx_val * rax_val * raz_val - 63 * rby_val * rby_val * rax_val * raz_val + 14 * rbx_val * rax_val * sf.czx() + 14 * rbx_val * raz_val * sf.cxx() - 14 * rby_val * rax_val * sf.czy() - 14 * rby_val * raz_val * sf.cxy() + 2 * sf.cxx() * sf.czx() - 2 * sf.cxy() * sf.czy()) / 40; // d/d(rbx)
        result.s1[4] = rt30 * (-126 * rbx_val * rby_val * rax_val * raz_val - 14 * rbx_val * rax_val * sf.czy() - 14 * rbx_val * raz_val * sf.cxy() - 14 * rby_val * rax_val * sf.czx() - 14 * rby_val * raz_val * sf.cxx() - 2 * sf.cxx() * sf.czy() - 2 * sf.czx() * sf.cxy()) / 40; // d/d(rby)
        // Orientation matrix derivatives
        result.s1[6] = rt30 * (7 * rbx_val * rbx_val * raz_val - 7 * rby_val * rby_val * raz_val + 2 * rbx_val * sf.czx() - 2 * rby_val * sf.czy()) / 40; // ∂s0/∂cxx
        result.s1[12] = rt30 * (7 * rbx_val * rbx_val * rax_val - 7 * rby_val * rby_val * rax_val + 2 * rbx_val * sf.cxx() - 2 * rby_val * sf.cxy()) / 40; // ∂s0/∂czx
        result.s1[7] = rt30 * (-14 * rbx_val * rby_val * raz_val - 2 * rbx_val * sf.czy() - 2 * rby_val * sf.czx()) / 40; // ∂s0/∂cxy
        result.s1[13] = rt30 * (-14 * rbx_val * rby_val * rax_val - 2 * rbx_val * sf.cxy() - 2 * rby_val * sf.cxx()) / 40; // ∂s0/∂czy

        if (level >= 2) {
            // Second derivatives - Orient case 206
            result.s2[3] = rt30 * (21 * rbx_val * rbx_val * rbx_val - 63 * rbx_val * rby_val * rby_val) / 40;
            result.s2[6] = rt30 * (63 * rbx_val * rbx_val * raz_val - 63 * rby_val * rby_val * raz_val + 14 * rbx_val * sf.czx() - 14 * rby_val * sf.czy()) / 40;
            result.s2[8] = rt30 * (63 * rbx_val * rbx_val * rax_val - 63 * rby_val * rby_val * rax_val + 14 * rbx_val * sf.cxx() - 14 * rby_val * sf.cxy()) / 40;
            result.s2[9] = rt30 * (126 * rax_val * raz_val * rbx_val + 14 * rax_val * sf.czx() + 14 * raz_val * sf.cxx()) / 40;
            result.s2[10] = rt30 * (-126 * rbx_val * rby_val * raz_val - 14 * rbx_val * sf.czy() - 14 * rby_val * sf.czx()) / 40;
            result.s2[12] = rt30 * (-126 * rbx_val * rby_val * rax_val - 14 * rbx_val * sf.cxy() - 14 * rby_val * sf.cxx()) / 40;
            result.s2[13] = rt30 * (-126 * rax_val * raz_val * rby_val - 14 * rax_val * sf.czy() - 14 * raz_val * sf.cxy()) / 40;
            result.s2[14] = rt30 * (-126 * rax_val * raz_val * rbx_val - 14 * rax_val * sf.czx() - 14 * raz_val * sf.cxx()) / 40;
            result.s2[23] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[24] = rt30 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 40;
            result.s2[25] = rt30 * (-14 * raz_val * rby_val - 2 * sf.czy()) / 40;
            result.s2[36] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[39] = rt30 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 40;
            result.s2[40] = rt30 * (-14 * rax_val * rby_val - 2 * sf.cxy()) / 40;
            result.s2[42] = rt30 * rbx_val / 20;
            result.s2[47] = -7.0 / 20.0 * rt30 * rbx_val * rby_val;
            result.s2[48] = rt30 * (-14 * raz_val * rby_val - 2 * sf.czy()) / 40;
            result.s2[49] = rt30 * (-14 * raz_val * rbx_val - 2 * sf.czx()) / 40;
            result.s2[53] = -rt30 * rby_val / 20;
            result.s2[66] = -7.0 / 20.0 * rt30 * rbx_val * rby_val;
            result.s2[69] = rt30 * (-14 * rax_val * rby_val - 2 * sf.cxy()) / 40;
            result.s2[70] = rt30 * (-14 * rax_val * rbx_val - 2 * sf.cxx()) / 40;
            result.s2[72] = -rt30 * rby_val / 20;
            result.s2[75] = -rt30 * rbx_val / 20;
        }
    }
}

/**
 * Quadrupole-21c × Octopole-33s kernel
 * Orient case 207: Q21c × Q33s
 */
void quadrupole_21c_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt30 *
                (63 * rbx_val * rbx_val * rby_val * rax_val * raz_val -
                 21 * rby_val * rby_val * rby_val * rax_val * raz_val +
                 7 * rbx_val * rbx_val * rax_val * sf.czy() +
                 7 * rbx_val * rbx_val * raz_val * sf.cxy() +
                 14 * rbx_val * rby_val * rax_val * sf.czx() +
                 14 * rbx_val * rby_val * raz_val * sf.cxx() -
                 7 * rby_val * rby_val * rax_val * sf.czy() -
                 7 * rby_val * rby_val * raz_val * sf.cxy() +
                 2 * rbx_val * sf.cxy() * sf.czx() + 2 * rbx_val * sf.cxx() * sf.czy() +
                 2 * rby_val * sf.cxx() * sf.czx() - 2 * rby_val * sf.cxy() * sf.czy()) /
                40;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt30 * (63 * rbx_val * rbx_val * rby_val * raz_val - 21 * rby_val * rby_val * rby_val * raz_val + 7 * rbx_val * rbx_val * sf.czy() + 14 * rbx_val * rby_val * sf.czx() - 7 * rby_val * rby_val * sf.czy()) / 40; // d/d(rax)
        result.s1[2] = rt30 * (63 * rbx_val * rbx_val * rby_val * rax_val - 21 * rby_val * rby_val * rby_val * rax_val + 7 * rbx_val * rbx_val * sf.cxy() + 14 * rbx_val * rby_val * sf.cxx() - 7 * rby_val * rby_val * sf.cxy()) / 40; // d/d(raz)
        result.s1[3] = rt30 * (126 * rbx_val * rby_val * rax_val * raz_val + 14 * rbx_val * rax_val * sf.czy() + 14 * rbx_val * raz_val * sf.cxy() + 14 * rby_val * rax_val * sf.czx() + 14 * rby_val * raz_val * sf.cxx() + 2 * sf.cxy() * sf.czx() + 2 * sf.cxx() * sf.czy()) / 40; // d/d(rbx)
        result.s1[4] = rt30 * (63 * rbx_val * rbx_val * rax_val * raz_val - 63 * rby_val * rby_val * rax_val * raz_val + 14 * rbx_val * rax_val * sf.czx() + 14 * rbx_val * raz_val * sf.cxx() - 14 * rby_val * rax_val * sf.czy() - 14 * rby_val * raz_val * sf.cxy() + 2 * sf.cxx() * sf.czx() - 2 * sf.cxy() * sf.czy()) / 40; // d/d(rby)
        // Orientation matrix derivatives
        result.s1[6] = rt30 * (14 * rbx_val * rby_val * raz_val + 2 * rbx_val * sf.czy() + 2 * rby_val * sf.czx()) / 40; // ∂s0/∂cxx
        result.s1[12] = rt30 * (14 * rbx_val * rby_val * rax_val + 2 * rbx_val * sf.cxy() + 2 * rby_val * sf.cxx()) / 40; // ∂s0/∂czx
        result.s1[7] = rt30 * (7 * rbx_val * rbx_val * raz_val - 7 * rby_val * rby_val * raz_val + 2 * rbx_val * sf.czx() - 2 * rby_val * sf.czy()) / 40; // ∂s0/∂cxy
        result.s1[13] = rt30 * (7 * rbx_val * rbx_val * rax_val - 7 * rby_val * rby_val * rax_val + 2 * rbx_val * sf.cxx() - 2 * rby_val * sf.cxy()) / 40; // ∂s0/∂czy

        if (level >= 2) {
            // Second derivatives - Orient case 207
            result.s2[3] = rt30 * (63 * rbx_val * rbx_val * rby_val - 21 * rby_val * rby_val * rby_val) / 40;
            result.s2[6] = rt30 * (126 * rbx_val * rby_val * raz_val + 14 * rbx_val * sf.czy() + 14 * rby_val * sf.czx()) / 40;
            result.s2[8] = rt30 * (126 * rbx_val * rby_val * rax_val + 14 * rbx_val * sf.cxy() + 14 * rby_val * sf.cxx()) / 40;
            result.s2[9] = rt30 * (126 * rax_val * raz_val * rby_val + 14 * rax_val * sf.czy() + 14 * raz_val * sf.cxy()) / 40;
            result.s2[10] = rt30 * (63 * rbx_val * rbx_val * raz_val - 63 * rby_val * rby_val * raz_val + 14 * rbx_val * sf.czx() - 14 * rby_val * sf.czy()) / 40;
            result.s2[12] = rt30 * (63 * rbx_val * rbx_val * rax_val - 63 * rby_val * rby_val * rax_val + 14 * rbx_val * sf.cxx() - 14 * rby_val * sf.cxy()) / 40;
            result.s2[13] = rt30 * (126 * rax_val * raz_val * rbx_val + 14 * rax_val * sf.czx() + 14 * raz_val * sf.cxx()) / 40;
            result.s2[14] = rt30 * (-126 * rax_val * raz_val * rby_val - 14 * rax_val * sf.czy() - 14 * raz_val * sf.cxy()) / 40;
            result.s2[23] = 7.0 / 20.0 * rt30 * rbx_val * rby_val;
            result.s2[24] = rt30 * (14 * raz_val * rby_val + 2 * sf.czy()) / 40;
            result.s2[25] = rt30 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 40;
            result.s2[36] = 7.0 / 20.0 * rt30 * rbx_val * rby_val;
            result.s2[39] = rt30 * (14 * rax_val * rby_val + 2 * sf.cxy()) / 40;
            result.s2[40] = rt30 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 40;
            result.s2[42] = rt30 * rby_val / 20;
            result.s2[47] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[48] = rt30 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 40;
            result.s2[49] = rt30 * (-14 * raz_val * rby_val - 2 * sf.czy()) / 40;
            result.s2[53] = rt30 * rbx_val / 20;
            result.s2[66] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[69] = rt30 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 40;
            result.s2[70] = rt30 * (-14 * rax_val * rby_val - 2 * sf.cxy()) / 40;
            result.s2[72] = rt30 * rbx_val / 20;
            result.s2[75] = -rt30 * rby_val / 20;
        }
    }
}

/**
 * Quadrupole-21s × Octopole-30 kernel
 * Orient case 208: Q21s × Q30
 */
void quadrupole_21s_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        63.0 / 8.0 * rbz_val * rbz_val * rbz_val * ray_val * raz_val -
        7.0 / 8.0 * ray_val * raz_val * rbz_val +
        21.0 / 4.0 * rbz_val * rbz_val * ray_val * sf.czz() +
        21.0 / 4.0 * rbz_val * rbz_val * raz_val * sf.czy() -
        3.0 / 4.0 * ray_val * sf.czz() - 3.0 / 4.0 * raz_val * sf.czy() +
        3.0 / 2.0 * rbz_val * sf.czy() * sf.czz();

    if (level >= 1) {
        // First derivatives
        result.s1[1] = 63.0 / 8.0 * rbz_val * rbz_val * rbz_val * raz_val - 7.0 / 8.0 * raz_val * rbz_val + 21.0 / 4.0 * rbz_val * rbz_val * sf.czz() - 3.0 / 4.0 * sf.czz(); // d/d(ray)
        result.s1[2] = 63.0 / 8.0 * rbz_val * rbz_val * rbz_val * ray_val - 7.0 / 8.0 * ray_val * rbz_val + 21.0 / 4.0 * rbz_val * rbz_val * sf.czy() - 3.0 / 4.0 * sf.czy(); // d/d(raz)
        result.s1[5] = 189.0 / 8.0 * rbz_val * rbz_val * ray_val * raz_val - 7.0 / 8.0 * ray_val * raz_val + 42.0 / 4.0 * rbz_val * ray_val * sf.czz() + 42.0 / 4.0 * rbz_val * raz_val * sf.czy() + 3.0 / 2.0 * sf.czy() * sf.czz(); // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[11] = rt3 * (7 * rbz_val * rbz_val * raz_val - raz_val + 2 * rbz_val * sf.czz()) / 4; // ∂s0/∂cyz
        result.s1[14] = rt3 * (7 * rbz_val * rbz_val * ray_val - ray_val + 2 * rbz_val * sf.cyz()) / 4; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 208
            result.s2[4] = rt3*(21*rbz_val * rbz_val * rbz_val-7*rbz_val)/4;
            result.s2[16] = rt3*(63*rbz_val * rbz_val*raz_val-7*raz_val+14*rbz_val*sf.czz())/4;
            result.s2[17] = rt3*(63*rbz_val * rbz_val*ray_val-7*ray_val+14*rbz_val*sf.cyz())/4;
            result.s2[20] = rt3*(126*ray_val*raz_val*rbz_val+14*ray_val*sf.czz()+14*raz_val*sf.cyz())/4;
            result.s2[93] = rt3*(7*rbz_val * rbz_val-1)/4;
            result.s2[96] = rt3*(14*raz_val*rbz_val+2*sf.czz())/4;
            result.s2[106] = rt3*(7*rbz_val * rbz_val-1)/4;
            result.s2[110] = rt3*(14*rbz_val*ray_val+2*sf.cyz())/4;
            result.s2[118] = rt3*rbz_val/2;
        }
    }
}

/**
 * Quadrupole-21s × Octopole-31c kernel
 * Orient case 209: Q21s × Q31c
 */
void quadrupole_21s_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt2 *
                (63 * rbx_val * rbz_val * rbz_val * ray_val * raz_val -
                 7 * ray_val * raz_val * rbx_val +
                 14 * rbx_val * rbz_val * ray_val * sf.czz() +
                 14 * rbx_val * rbz_val * raz_val * sf.czy() +
                 7 * rbz_val * rbz_val * ray_val * sf.czx() +
                 7 * rbz_val * rbz_val * raz_val * sf.cyx() - ray_val * sf.czx() -
                 raz_val * sf.cyx() + 2 * rbx_val * sf.czy() * sf.czz() +
                 2 * rbz_val * sf.cyx() * sf.czz() + 2 * rbz_val * sf.czx() * sf.czy()) /
                8;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt2 * (63 * rbx_val * rbz_val * rbz_val * raz_val - 7 * raz_val * rbx_val + 14 * rbx_val * rbz_val * sf.czz() + 7 * rbz_val * rbz_val * sf.czx() - sf.czx()) / 8; // d/d(ray)
        result.s1[2] = rt2 * (63 * rbx_val * rbz_val * rbz_val * ray_val - 7 * ray_val * rbx_val + 14 * rbx_val * rbz_val * sf.czy() + 7 * rbz_val * rbz_val * sf.cyx() - sf.cyx()) / 8; // d/d(raz)
        result.s1[3] = rt2 * (63 * rbz_val * rbz_val * ray_val * raz_val - 7 * ray_val * raz_val + 14 * rbz_val * ray_val * sf.czz() + 14 * rbz_val * raz_val * sf.czy() + 2 * sf.czy() * sf.czz()) / 8; // d/d(rbx)
        result.s1[5] = rt2 * (126 * rbx_val * rbz_val * ray_val * raz_val + 14 * rbx_val * ray_val * sf.czz() + 14 * rbx_val * raz_val * sf.czy() + 14 * rbz_val * ray_val * sf.czx() + 14 * rbz_val * raz_val * sf.cyx() + 2 * sf.cyx() * sf.czz() + 2 * sf.czx() * sf.czy()) / 8; // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[9] = rt2 * (7 * rbz_val * rbz_val * raz_val - raz_val + 2 * rbz_val * sf.czz()) / 8; // ∂s0/∂cyx
        result.s1[12] = rt2 * (7 * rbz_val * rbz_val * ray_val - ray_val + 2 * rbz_val * sf.cyz()) / 8; // ∂s0/∂czx
        result.s1[11] = rt2 * (14 * rbx_val * rbz_val * raz_val + 2 * rbx_val * sf.czz() + 2 * rbz_val * sf.czx()) / 8; // ∂s0/∂cyz
        result.s1[14] = rt2 * (14 * rbx_val * rbz_val * ray_val + 2 * rbx_val * sf.cyz() + 2 * rbz_val * sf.cyx()) / 8; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 209
            result.s2[4] = rt2*(63*rbx_val*rbz_val * rbz_val-7*rbx_val)/8;
            result.s2[7] = rt2*(63*rbz_val * rbz_val*raz_val-7*raz_val+14*rbz_val*sf.czz())/8;
            result.s2[8] = rt2*(63*rbz_val * rbz_val*ray_val-7*ray_val+14*rbz_val*sf.cyz())/8;
            result.s2[16] = rt2*(126*rbx_val*rbz_val*raz_val+14*rbx_val*sf.czz()+14*rbz_val*sf.czx())/8;
            result.s2[17] = rt2*(126*rbx_val*rbz_val*ray_val+14*rbx_val*sf.cyz()+14*rbz_val*sf.cyx())/8;
            result.s2[18] = rt2*(126*ray_val*raz_val*rbz_val+14*ray_val*sf.czz()+14*raz_val*sf.cyz())/8;
            result.s2[20] = rt2*(126*ray_val*raz_val*rbx_val+14*ray_val*sf.czx()+14*raz_val*sf.cyx())/8;
            result.s2[30] = rt2*(7*rbz_val * rbz_val-1)/8;
            result.s2[33] = rt2*(14*raz_val*rbz_val+2*sf.czz())/8;
            result.s2[37] = rt2*(7*rbz_val * rbz_val-1)/8;
            result.s2[41] = rt2*(14*rbz_val*ray_val+2*sf.cyz())/8;
            result.s2[93] = 7.0/4.0*rt2*rbx_val*rbz_val;
            result.s2[94] = rt2*(14*raz_val*rbz_val+2*sf.czz())/8;
            result.s2[96] = rt2*(14*raz_val*rbx_val+2*sf.czx())/8;
            result.s2[99] = rt2*rbz_val/4;
            result.s2[106] = 7.0/4.0*rt2*rbx_val*rbz_val;
            result.s2[108] = rt2*(14*rbz_val*ray_val+2*sf.cyz())/8;
            result.s2[110] = rt2*(14*rbx_val*ray_val+2*sf.cyx())/8;
            result.s2[112] = rt2*rbz_val/4;
            result.s2[118] = rt2*rbx_val/4;
        }
    }
}

/**
 * Quadrupole-21s × Octopole-31s kernel
 * Orient case 210: Q21s × Q31s
 */
void quadrupole_21s_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt2 *
                (63 * rby_val * rbz_val * rbz_val * ray_val * raz_val -
                 7 * ray_val * raz_val * rby_val +
                 14 * rby_val * rbz_val * ray_val * sf.czz() +
                 14 * rby_val * rbz_val * raz_val * sf.czy() +
                 7 * rbz_val * rbz_val * ray_val * sf.czy() +
                 7 * rbz_val * rbz_val * raz_val * sf.cyy() - ray_val * sf.czy() -
                 raz_val * sf.cyy() + 2 * rby_val * sf.czy() * sf.czz() +
                 2 * rbz_val * sf.cyy() * sf.czz() + 2 * rbz_val * sf.czy() * sf.czy()) /
                8;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt2 * (63 * rby_val * rbz_val * rbz_val * raz_val - 7 * raz_val * rby_val + 14 * rby_val * rbz_val * sf.czz() + 7 * rbz_val * rbz_val * sf.czy() - sf.czy()) / 8; // d/d(ray)
        result.s1[2] = rt2 * (63 * rby_val * rbz_val * rbz_val * ray_val - 7 * ray_val * rby_val + 14 * rby_val * rbz_val * sf.czy() + 7 * rbz_val * rbz_val * sf.cyy() - sf.cyy()) / 8; // d/d(raz)
        result.s1[4] = rt2 * (63 * rbz_val * rbz_val * ray_val * raz_val - 7 * ray_val * raz_val + 14 * rbz_val * ray_val * sf.czz() + 14 * rbz_val * raz_val * sf.czy() + 2 * sf.czy() * sf.czz()) / 8; // d/d(rby)
        result.s1[5] = rt2 * (126 * rby_val * rbz_val * ray_val * raz_val + 14 * rby_val * ray_val * sf.czz() + 14 * rby_val * raz_val * sf.czy() + 14 * rbz_val * ray_val * sf.czy() + 14 * rbz_val * raz_val * sf.cyy() + 2 * sf.cyy() * sf.czz() + 4 * rbz_val * sf.czy() * sf.czy()) / 8; // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[10] = rt2 * (7 * rbz_val * rbz_val * raz_val - raz_val + 2 * rbz_val * sf.czz()) / 8; // ∂s0/∂cyy
        result.s1[13] = rt2 * (7 * rbz_val * rbz_val * ray_val - ray_val + 2 * rbz_val * sf.cyz()) / 8; // ∂s0/∂czy
        result.s1[11] = rt2 * (14 * rby_val * rbz_val * raz_val + 2 * rby_val * sf.czz() + 2 * rbz_val * sf.czy()) / 8; // ∂s0/∂cyz
        result.s1[14] = rt2 * (14 * rby_val * rbz_val * ray_val + 2 * rby_val * sf.cyz() + 2 * rbz_val * sf.cyy()) / 8; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 210
            result.s2[4] = rt2*(63*rby_val*rbz_val * rbz_val-7*rby_val)/8;
            result.s2[11] = rt2*(63*rbz_val * rbz_val*raz_val-7*raz_val+14*rbz_val*sf.czz())/8;
            result.s2[12] = rt2*(63*rbz_val * rbz_val*ray_val-7*ray_val+14*rbz_val*sf.cyz())/8;
            result.s2[16] = rt2*(126*rby_val*rbz_val*raz_val+14*rby_val*sf.czz()+14*rbz_val*sf.czy())/8;
            result.s2[17] = rt2*(126*rby_val*rbz_val*ray_val+14*rby_val*sf.cyz()+14*rbz_val*sf.cyy())/8;
            result.s2[19] = rt2*(126*ray_val*raz_val*rbz_val+14*ray_val*sf.czz()+14*raz_val*sf.cyz())/8;
            result.s2[20] = rt2*(126*ray_val*raz_val*rby_val+14*ray_val*sf.czy()+14*raz_val*sf.cyy())/8;
            result.s2[57] = rt2*(7*rbz_val * rbz_val-1)/8;
            result.s2[60] = rt2*(14*raz_val*rbz_val+2*sf.czz())/8;
            result.s2[67] = rt2*(7*rbz_val * rbz_val-1)/8;
            result.s2[71] = rt2*(14*rbz_val*ray_val+2*sf.cyz())/8;
            result.s2[93] = 7.0/4.0*rt2*rby_val*rbz_val;
            result.s2[95] = rt2*(14*raz_val*rbz_val+2*sf.czz())/8;
            result.s2[96] = rt2*(14*raz_val*rby_val+2*sf.czy())/8;
            result.s2[102] = rt2*rbz_val/4;
            result.s2[106] = 7.0/4.0*rt2*rby_val*rbz_val;
            result.s2[109] = rt2*(14*rbz_val*ray_val+2*sf.cyz())/8;
            result.s2[110] = rt2*(14*ray_val*rby_val+2*sf.cyy())/8;
            result.s2[115] = rt2*rbz_val/4;
            result.s2[118] = rt2*rby_val/4;
        }
    }
}

/**
 * Quadrupole-21s × Octopole-32c kernel
 * Orient case 211: Q21s × Q32c
 */
void quadrupole_21s_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rbx_val * rbz_val * ray_val * raz_val -
                 63 * rby_val * rby_val * rbz_val * ray_val * raz_val +
                 7 * rbx_val * rbx_val * ray_val * sf.czz() +
                 7 * rbx_val * rbx_val * raz_val * sf.czy() +
                 14 * rbx_val * rbz_val * ray_val * sf.czx() +
                 14 * rbx_val * rbz_val * raz_val * sf.cyx() -
                 7 * rby_val * rby_val * ray_val * sf.czz() -
                 7 * rby_val * rby_val * raz_val * sf.czy() -
                 14 * rby_val * rbz_val * ray_val * sf.czy() -
                 14 * rby_val * rbz_val * raz_val * sf.cyy() +
                 2 * rbx_val * sf.cyx() * sf.czz() + 2 * rbx_val * sf.czx() * sf.czy() -
                 2 * rby_val * sf.cyy() * sf.czz() - 2 * rby_val * sf.czy() * sf.czy() +
                 2 * rbz_val * sf.cyx() * sf.czx() - 2 * rbz_val * sf.cyy() * sf.czy()) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt5 * (63 * rbx_val * rbx_val * rbz_val * raz_val - 63 * rby_val * rby_val * rbz_val * raz_val + 7 * rbx_val * rbx_val * sf.czz() + 14 * rbx_val * rbz_val * sf.czx() - 7 * rby_val * rby_val * sf.czz() - 14 * rby_val * rbz_val * sf.czy()) / 20; // d/d(ray)
        result.s1[2] = rt5 * (63 * rbx_val * rbx_val * rbz_val * ray_val - 63 * rby_val * rby_val * rbz_val * ray_val + 7 * rbx_val * rbx_val * sf.czy() + 14 * rbx_val * rbz_val * sf.cyx() - 7 * rby_val * rby_val * sf.czy() - 14 * rby_val * rbz_val * sf.cyy()) / 20; // d/d(raz)
        result.s1[3] = rt5 * (126 * rbx_val * rbz_val * ray_val * raz_val + 14 * rbx_val * ray_val * sf.czz() + 14 * rbx_val * raz_val * sf.czy() + 14 * rbz_val * ray_val * sf.czx() + 14 * rbz_val * raz_val * sf.cyx() + 2 * sf.cyx() * sf.czz() + 2 * sf.czx() * sf.czy() + 2 * rbz_val * sf.cyx() * sf.czx()) / 20; // d/d(rbx)
        result.s1[4] = rt5 * (-126 * rby_val * rbz_val * ray_val * raz_val - 14 * rby_val * ray_val * sf.czz() - 14 * rby_val * raz_val * sf.czy() - 14 * rbz_val * ray_val * sf.czy() - 14 * rbz_val * raz_val * sf.cyy() - 2 * sf.cyy() * sf.czz() - 4 * rby_val * sf.czy() * sf.czy() - 2 * rbz_val * sf.cyy() * sf.czy()) / 20; // d/d(rby)
        result.s1[5] = rt5 * (63 * rbx_val * rbx_val * ray_val * raz_val - 63 * rby_val * rby_val * ray_val * raz_val + 14 * rbx_val * ray_val * sf.czx() + 14 * rbx_val * raz_val * sf.cyx() - 14 * rby_val * ray_val * sf.czy() - 14 * rby_val * raz_val * sf.cyy() + 2 * sf.cyx() * sf.czx() - 2 * sf.cyy() * sf.czy()) / 20; // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[9] = rt5 * (14 * rbx_val * rbz_val * raz_val + 2 * rbx_val * sf.czz() + 2 * rbz_val * sf.czx()) / 20; // ∂s0/∂cyx
        result.s1[12] = rt5 * (14 * rbx_val * rbz_val * ray_val + 2 * rbx_val * sf.cyz() + 2 * rbz_val * sf.cyx()) / 20; // ∂s0/∂czx
        result.s1[10] = rt5 * (-14 * rby_val * rbz_val * raz_val - 2 * rby_val * sf.czz() - 2 * rbz_val * sf.czy()) / 20; // ∂s0/∂cyy
        result.s1[13] = rt5 * (-14 * rby_val * rbz_val * ray_val - 2 * rby_val * sf.cyz() - 2 * rbz_val * sf.cyy()) / 20; // ∂s0/∂czy
        result.s1[11] = rt5 * (7 * rbx_val * rbx_val * raz_val - 7 * rby_val * rby_val * raz_val + 2 * rbx_val * sf.czx() - 2 * rby_val * sf.czy()) / 20; // ∂s0/∂cyz
        result.s1[14] = rt5 * (7 * rbx_val * rbx_val * ray_val - 7 * rby_val * rby_val * ray_val + 2 * rbx_val * sf.cyx() - 2 * rby_val * sf.cyy()) / 20; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 211
            result.s2[4] = rt5 * (63 * rbx_val * rbx_val * rbz_val - 63 * rby_val * rby_val * rbz_val) / 20;
            result.s2[7] = rt5 * (126 * rbx_val * rbz_val * raz_val + 14 * rbx_val * sf.czz() + 14 * rbz_val * sf.czx()) / 20;
            result.s2[8] = rt5 * (126 * rbx_val * rbz_val * ray_val + 14 * rbx_val * sf.czy() + 14 * rbz_val * sf.cyx()) / 20;
            result.s2[9] = rt5 * (126 * ray_val * raz_val * rbz_val + 14 * ray_val * sf.czz() + 14 * raz_val * sf.czy()) / 20;
            result.s2[11] = rt5 * (-126 * rby_val * rbz_val * raz_val - 14 * rby_val * sf.czz() - 14 * rbz_val * sf.czy()) / 20;
            result.s2[12] = rt5 * (-126 * rby_val * rbz_val * ray_val - 14 * rby_val * sf.czy() - 14 * rbz_val * sf.cyy()) / 20;
            result.s2[14] = rt5 * (-126 * ray_val * raz_val * rbz_val - 14 * ray_val * sf.czz() - 14 * raz_val * sf.czy()) / 20;
            result.s2[16] = rt5 * (63 * rbx_val * rbx_val * raz_val - 63 * rby_val * rby_val * raz_val + 14 * rbx_val * sf.czx() - 14 * rby_val * sf.czy()) / 20;
            result.s2[17] = rt5 * (63 * rbx_val * rbx_val * ray_val - 63 * rby_val * rby_val * ray_val + 14 * rbx_val * sf.cyx() - 14 * rby_val * sf.cyy()) / 20;
            result.s2[18] = rt5 * (126 * ray_val * raz_val * rbx_val + 14 * ray_val * sf.czx() + 14 * raz_val * sf.cyx()) / 20;
            result.s2[19] = rt5 * (-126 * ray_val * raz_val * rby_val - 14 * ray_val * sf.czy() - 14 * raz_val * sf.cyy()) / 20;
            result.s2[30] = 7.0 / 10.0 * rt5 * rbx_val * rbz_val;
            result.s2[31] = rt5 * (14 * raz_val * rbz_val + 2 * sf.czz()) / 20;
            result.s2[33] = rt5 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 20;
            result.s2[37] = 7.0 / 10.0 * rt5 * rbx_val * rbz_val;
            result.s2[39] = rt5 * (14 * rbz_val * ray_val + 2 * sf.czy()) / 20;
            result.s2[41] = rt5 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 20;
            result.s2[43] = rt5 * rbz_val / 10;
            result.s2[57] = -7.0 / 10.0 * rt5 * rby_val * rbz_val;
            result.s2[59] = rt5 * (-14 * raz_val * rbz_val - 2 * sf.czz()) / 20;
            result.s2[60] = rt5 * (-14 * raz_val * rby_val - 2 * sf.czy()) / 20;
            result.s2[67] = -7.0 / 10.0 * rt5 * rby_val * rbz_val;
            result.s2[70] = rt5 * (-14 * rbz_val * ray_val - 2 * sf.czy()) / 20;
            result.s2[71] = rt5 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 20;
            result.s2[76] = -rt5 * rbz_val / 10;
            result.s2[93] = rt5 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 20;
            result.s2[94] = rt5 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 20;
            result.s2[95] = rt5 * (-14 * raz_val * rby_val - 2 * sf.czy()) / 20;
            result.s2[99] = rt5 * rbx_val / 10;
            result.s2[102] = -rt5 * rby_val / 10;
            result.s2[106] = rt5 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 20;
            result.s2[108] = rt5 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 20;
            result.s2[109] = rt5 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 20;
            result.s2[112] = rt5 * rbx_val / 10;
            result.s2[115] = -rt5 * rby_val / 10;
        }
    }
}

/**
 * Quadrupole-21s × Octopole-32s kernel
 * Orient case 212: Q21s × Q32s
 */
void quadrupole_21s_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt5 *
        (63 * rbx_val * rby_val * rbz_val * ray_val * raz_val +
         7 * rbx_val * rby_val * ray_val * sf.czz() +
         7 * rbx_val * rby_val * raz_val * sf.czy() +
         7 * rbx_val * rbz_val * ray_val * sf.czy() +
         7 * rbx_val * rbz_val * raz_val * sf.cyy() +
         7 * rby_val * rbz_val * ray_val * sf.czx() +
         7 * rby_val * rbz_val * raz_val * sf.cyx() + rbx_val * sf.cyy() * sf.czz() +
         rbx_val * sf.czy() * sf.czy() + rby_val * sf.cyx() * sf.czz() + rby_val * sf.czx() * sf.czy() +
         rbz_val * sf.cyx() * sf.czy() + rbz_val * sf.cyy() * sf.czx()) /
        10;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt5 * (63 * rbx_val * rby_val * rbz_val * raz_val + 7 * rbx_val * rby_val * sf.czz() + 7 * rbx_val * rbz_val * sf.czy() + 7 * rby_val * rbz_val * sf.czx()) / 10; // d/d(ray)
        result.s1[2] = rt5 * (63 * rbx_val * rby_val * rbz_val * ray_val + 7 * rbx_val * rby_val * sf.czy() + 7 * rbx_val * rbz_val * sf.cyy() + 7 * rby_val * rbz_val * sf.cyx()) / 10; // d/d(raz)
        result.s1[3] = rt5 * (63 * rby_val * rbz_val * ray_val * raz_val + 7 * rby_val * ray_val * sf.czz() + 7 * rby_val * raz_val * sf.czy() + 7 * rbz_val * ray_val * sf.czy() + 7 * rbz_val * raz_val * sf.cyy() + sf.cyy() * sf.czz() + sf.czy() * sf.czy() + rbz_val * sf.cyy() * sf.czx()) / 10; // d/d(rbx)
        result.s1[4] = rt5 * (63 * rbx_val * rbz_val * ray_val * raz_val + 7 * rbx_val * ray_val * sf.czz() + 7 * rbx_val * raz_val * sf.czy() + 7 * rbz_val * ray_val * sf.czx() + 7 * rbz_val * raz_val * sf.cyx() + sf.cyx() * sf.czz() + sf.czx() * sf.czy() + rbz_val * sf.cyx() * sf.czy()) / 10; // d/d(rby)
        result.s1[5] = rt5 * (63 * rbx_val * rby_val * ray_val * raz_val + 7 * rbx_val * ray_val * sf.czy() + 7 * rbx_val * raz_val * sf.cyy() + 7 * rby_val * ray_val * sf.czx() + 7 * rby_val * raz_val * sf.cyx() + sf.cyx() * sf.czy() + sf.cyy() * sf.czx()) / 10; // d/d(rbz)
        // Orientation matrix derivatives
        result.s1[9] = rt5 * (7 * rby_val * rbz_val * raz_val + rby_val * sf.czz() + rbz_val * sf.czy()) / 10; // ∂s0/∂cyx
        result.s1[12] = rt5 * (7 * rby_val * rbz_val * ray_val + rby_val * sf.cyz() + rbz_val * sf.cyy()) / 10; // ∂s0/∂czx
        result.s1[10] = rt5 * (7 * rbx_val * rbz_val * raz_val + rbx_val * sf.czz() + rbz_val * sf.czx()) / 10; // ∂s0/∂cyy
        result.s1[13] = rt5 * (7 * rbx_val * rbz_val * ray_val + rbx_val * sf.cyz() + rbz_val * sf.cyx()) / 10; // ∂s0/∂czy
        result.s1[11] = rt5 * (7 * rbx_val * rby_val * raz_val + rbx_val * sf.czy() + rby_val * sf.czx()) / 10; // ∂s0/∂cyz
        result.s1[14] = rt5 * (7 * rbx_val * rby_val * ray_val + rbx_val * sf.cyy() + rby_val * sf.cyx()) / 10; // ∂s0/∂czz

        if (level >= 2) {
            // Second derivatives - Orient case 212
            result.s2[4] = 63.0/10.0*rt5*rbx_val*rby_val*rbz_val;
            result.s2[7] = rt5*(63*rby_val*rbz_val*raz_val+7*rby_val*sf.czz()+7*rbz_val*sf.czy())/10;
            result.s2[8] = rt5*(63*rby_val*rbz_val*ray_val+7*rby_val*sf.cyz()+7*rbz_val*sf.cyy())/10;
            result.s2[11] = rt5*(63*rbx_val*rbz_val*raz_val+7*rbx_val*sf.czz()+7*rbz_val*sf.czx())/10;
            result.s2[12] = rt5*(63*rbx_val*rbz_val*ray_val+7*rbx_val*sf.cyz()+7*rbz_val*sf.cyx())/10;
            result.s2[13] = rt5*(63*ray_val*raz_val*rbz_val+7*ray_val*sf.czz()+7*raz_val*sf.cyz())/10;
            result.s2[16] = rt5*(63*rbx_val*rby_val*raz_val+7*rbx_val*sf.czy()+7*rby_val*sf.czx())/10;
            result.s2[17] = rt5*(63*rbx_val*rby_val*ray_val+7*rbx_val*sf.cyy()+7*rby_val*sf.cyx())/10;
            result.s2[18] = rt5*(63*ray_val*raz_val*rby_val+7*ray_val*sf.czy()+7*raz_val*sf.cyy())/10;
            result.s2[19] = rt5*(63*ray_val*raz_val*rbx_val+7*ray_val*sf.czx()+7*raz_val*sf.cyx())/10;
            result.s2[30] = 7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[32] = rt5*(7*raz_val*rbz_val+sf.czz())/10;
            result.s2[33] = rt5*(7*raz_val*rby_val+sf.czy())/10;
            result.s2[37] = 7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[40] = rt5*(7*rbz_val*ray_val+sf.cyz())/10;
            result.s2[41] = rt5*(7*ray_val*rby_val+sf.cyy())/10;
            result.s2[57] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[58] = rt5*(7*raz_val*rbz_val+sf.czz())/10;
            result.s2[60] = rt5*(7*raz_val*rbx_val+sf.czx())/10;
            result.s2[63] = rt5*rbz_val/10;
            result.s2[67] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[69] = rt5*(7*rbz_val*ray_val+sf.cyz())/10;
            result.s2[71] = rt5*(7*rbx_val*ray_val+sf.cyx())/10;
            result.s2[73] = rt5*rbz_val/10;
            result.s2[93] = 7.0/10.0*rt5*rbx_val*rby_val;
            result.s2[94] = rt5*(7*raz_val*rby_val+sf.czy())/10;
            result.s2[95] = rt5*(7*raz_val*rbx_val+sf.czx())/10;
            result.s2[99] = rt5*rby_val/10;
            result.s2[102] = rt5*rbx_val/10;
            result.s2[106] = 7.0/10.0*rt5*rbx_val*rby_val;
            result.s2[108] = rt5*(7*ray_val*rby_val+sf.cyy())/10;
            result.s2[109] = rt5*(7*rbx_val*ray_val+sf.cyx())/10;
            result.s2[112] = rt5*rby_val/10;
            result.s2[115] = rt5*rbx_val/10;
        }
    }
}

/**
 * Quadrupole-21s × Octopole-33c kernel
 * Orient case 213: Q21s × Q33c
 */
void quadrupole_21s_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt30 *
                (21 * rbx_val * rbx_val * rbx_val * ray_val * raz_val -
                 63 * rbx_val * rby_val * rby_val * ray_val * raz_val +
                 7 * rbx_val * rbx_val * ray_val * sf.czx() +
                 7 * rbx_val * rbx_val * raz_val * sf.cyx() -
                 14 * rbx_val * rby_val * ray_val * sf.czy() -
                 14 * rbx_val * rby_val * raz_val * sf.cyy() -
                 7 * rby_val * rby_val * ray_val * sf.czx() -
                 7 * rby_val * rby_val * raz_val * sf.cyx() +
                 2 * rbx_val * sf.cyx() * sf.czx() - 2 * rbx_val * sf.cyy() * sf.czy() -
                 2 * rby_val * sf.cyx() * sf.czy() - 2 * rby_val * sf.czx() * sf.cyy()) /
                40;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt30 * (21 * rbx_val * rbx_val * rbx_val * raz_val - 63 * rbx_val * rby_val * rby_val * raz_val + 7 * rbx_val * rbx_val * sf.czx() - 14 * rbx_val * rby_val * sf.czy() - 7 * rby_val * rby_val * sf.czx()) / 40; // d/d(ray)
        result.s1[2] = rt30 * (21 * rbx_val * rbx_val * rbx_val * ray_val - 63 * rbx_val * rby_val * rby_val * ray_val + 7 * rbx_val * rbx_val * sf.cyx() - 14 * rbx_val * rby_val * sf.cyy() - 7 * rby_val * rby_val * sf.cyx()) / 40; // d/d(raz)
        result.s1[3] = rt30 * (63 * rbx_val * rbx_val * ray_val * raz_val - 63 * rby_val * rby_val * ray_val * raz_val + 14 * rbx_val * ray_val * sf.czx() + 14 * rbx_val * raz_val * sf.cyx() - 14 * rby_val * ray_val * sf.czy() - 14 * rby_val * raz_val * sf.cyy() + 2 * sf.cyx() * sf.czx() - 2 * sf.cyy() * sf.czy()) / 40; // d/d(rbx)
        result.s1[4] = rt30 * (-126 * rbx_val * rby_val * ray_val * raz_val - 14 * rbx_val * ray_val * sf.czy() - 14 * rbx_val * raz_val * sf.cyy() - 14 * rby_val * ray_val * sf.czx() - 14 * rby_val * raz_val * sf.cyx() - 2 * sf.cyx() * sf.czy() - 2 * sf.czx() * sf.cyy()) / 40; // d/d(rby)
        // Orientation matrix derivatives
        result.s1[9] = rt30 * (7 * rbx_val * rbx_val * raz_val - 7 * rby_val * rby_val * raz_val + 2 * rbx_val * sf.czx() - 2 * rby_val * sf.czy()) / 40; // ∂s0/∂cyx
        result.s1[12] = rt30 * (7 * rbx_val * rbx_val * ray_val - 7 * rby_val * rby_val * ray_val + 2 * rbx_val * sf.cyx() - 2 * rby_val * sf.cyy()) / 40; // ∂s0/∂czx
        result.s1[10] = rt30 * (-14 * rbx_val * rby_val * raz_val - 2 * rbx_val * sf.czy() - 2 * rby_val * sf.czx()) / 40; // ∂s0/∂cyy
        result.s1[13] = rt30 * (-14 * rbx_val * rby_val * ray_val - 2 * rbx_val * sf.cyy() - 2 * rby_val * sf.cyx()) / 40; // ∂s0/∂czy

        if (level >= 2) {
            // Second derivatives - Orient case 213
            result.s2[4] = rt30 * (21 * rbx_val * rbx_val * rbx_val - 63 * rbx_val * rby_val * rby_val) / 40;
            result.s2[7] = rt30 * (63 * rbx_val * rbx_val * raz_val - 63 * rby_val * rby_val * raz_val + 14 * rbx_val * sf.czx() - 14 * rby_val * sf.czy()) / 40;
            result.s2[8] = rt30 * (63 * rbx_val * rbx_val * ray_val - 63 * rby_val * rby_val * ray_val + 14 * rbx_val * sf.cyx() - 14 * rby_val * sf.cyy()) / 40;
            result.s2[9] = rt30 * (126 * ray_val * raz_val * rbx_val + 14 * ray_val * sf.czx() + 14 * raz_val * sf.cyx()) / 40;
            result.s2[11] = rt30 * (-126 * rbx_val * rby_val * raz_val - 14 * rbx_val * sf.czy() - 14 * rby_val * sf.czx()) / 40;
            result.s2[12] = rt30 * (-126 * rbx_val * rby_val * ray_val - 14 * rbx_val * sf.cyy() - 14 * rby_val * sf.cyx()) / 40;
            result.s2[13] = rt30 * (-126 * ray_val * raz_val * rby_val - 14 * ray_val * sf.czy() - 14 * raz_val * sf.cyy()) / 40;
            result.s2[14] = rt30 * (-126 * ray_val * raz_val * rbx_val - 14 * ray_val * sf.czx() - 14 * raz_val * sf.cyx()) / 40;
            result.s2[30] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[31] = rt30 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 40;
            result.s2[32] = rt30 * (-14 * raz_val * rby_val - 2 * sf.czy()) / 40;
            result.s2[37] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[39] = rt30 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 40;
            result.s2[40] = rt30 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 40;
            result.s2[43] = rt30 * rbx_val / 20;
            result.s2[57] = -7.0 / 20.0 * rt30 * rbx_val * rby_val;
            result.s2[58] = rt30 * (-14 * raz_val * rby_val - 2 * sf.czy()) / 40;
            result.s2[59] = rt30 * (-14 * raz_val * rbx_val - 2 * sf.czx()) / 40;
            result.s2[63] = -rt30 * rby_val / 20;
            result.s2[67] = -7.0 / 20.0 * rt30 * rbx_val * rby_val;
            result.s2[69] = rt30 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 40;
            result.s2[70] = rt30 * (-14 * rbx_val * ray_val - 2 * sf.cyx()) / 40;
            result.s2[73] = -rt30 * rby_val / 20;
            result.s2[76] = -rt30 * rbx_val / 20;
        }
    }
}

/**
 * Quadrupole-21s × Octopole-33s kernel
 * Orient case 214: Q21s × Q33s
 */
void quadrupole_21s_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt30 *
                (63 * rbx_val * rbx_val * rby_val * ray_val * raz_val -
                 21 * rby_val * rby_val * rby_val * ray_val * raz_val +
                 7 * rbx_val * rbx_val * ray_val * sf.czy() +
                 7 * rbx_val * rbx_val * raz_val * sf.cyy() +
                 14 * rbx_val * rby_val * ray_val * sf.czx() +
                 14 * rbx_val * rby_val * raz_val * sf.cyx() -
                 7 * rby_val * rby_val * ray_val * sf.czy() -
                 7 * rby_val * rby_val * raz_val * sf.cyy() +
                 2 * rbx_val * sf.cyx() * sf.czy() + 2 * rbx_val * sf.czx() * sf.cyy() +
                 2 * rby_val * sf.cyx() * sf.czx() - 2 * rby_val * sf.cyy() * sf.czy()) /
                40;

    if (level >= 1) {
        // First derivatives
        result.s1[1] = rt30 * (63 * rbx_val * rbx_val * rby_val * raz_val - 21 * rby_val * rby_val * rby_val * raz_val + 7 * rbx_val * rbx_val * sf.czy() + 14 * rbx_val * rby_val * sf.czx() - 7 * rby_val * rby_val * sf.czy()) / 40; // d/d(ray)
        result.s1[2] = rt30 * (63 * rbx_val * rbx_val * rby_val * ray_val - 21 * rby_val * rby_val * rby_val * ray_val + 7 * rbx_val * rbx_val * sf.cyy() + 14 * rbx_val * rby_val * sf.cyx() - 7 * rby_val * rby_val * sf.cyy()) / 40; // d/d(raz)
        result.s1[3] = rt30 * (126 * rbx_val * rby_val * ray_val * raz_val + 14 * rbx_val * ray_val * sf.czy() + 14 * rbx_val * raz_val * sf.cyy() + 14 * rby_val * ray_val * sf.czx() + 14 * rby_val * raz_val * sf.cyx() + 2 * sf.cyx() * sf.czy() + 2 * sf.czx() * sf.cyy()) / 40; // d/d(rbx)
        result.s1[4] = rt30 * (63 * rbx_val * rbx_val * ray_val * raz_val - 63 * rby_val * rby_val * ray_val * raz_val + 14 * rbx_val * ray_val * sf.czx() + 14 * rbx_val * raz_val * sf.cyx() - 14 * rby_val * ray_val * sf.czy() - 14 * rby_val * raz_val * sf.cyy() + 2 * sf.cyx() * sf.czx() - 2 * sf.cyy() * sf.czy()) / 40; // d/d(rby)
        // Orientation matrix derivatives
        result.s1[9] = rt30 * (14 * rbx_val * rby_val * raz_val + 2 * rbx_val * sf.czy() + 2 * rby_val * sf.czx()) / 40; // ∂s0/∂cyx
        result.s1[12] = rt30 * (14 * rbx_val * rby_val * ray_val + 2 * rbx_val * sf.cyy() + 2 * rby_val * sf.cyx()) / 40; // ∂s0/∂czx
        result.s1[10] = rt30 * (7 * rbx_val * rbx_val * raz_val - 7 * rby_val * rby_val * raz_val + 2 * rbx_val * sf.czx() - 2 * rby_val * sf.czy()) / 40; // ∂s0/∂cyy
        result.s1[13] = rt30 * (7 * rbx_val * rbx_val * ray_val - 7 * rby_val * rby_val * ray_val + 2 * rbx_val * sf.cyx() - 2 * rby_val * sf.cyy()) / 40; // ∂s0/∂czy

        if (level >= 2) {
            // Second derivatives - Orient case 214
            result.s2[4] = rt30 * (63 * rbx_val * rbx_val * rby_val - 21 * rby_val * rby_val * rby_val) / 40;
            result.s2[7] = rt30 * (126 * rbx_val * rby_val * raz_val + 14 * rbx_val * sf.czy() + 14 * rby_val * sf.czx()) / 40;
            result.s2[8] = rt30 * (126 * rbx_val * rby_val * ray_val + 14 * rbx_val * sf.cyy() + 14 * rby_val * sf.cyx()) / 40;
            result.s2[9] = rt30 * (126 * ray_val * raz_val * rby_val + 14 * ray_val * sf.czy() + 14 * raz_val * sf.cyy()) / 40;
            result.s2[11] = rt30 * (63 * rbx_val * rbx_val * raz_val - 63 * rby_val * rby_val * raz_val + 14 * rbx_val * sf.czx() - 14 * rby_val * sf.czy()) / 40;
            result.s2[12] = rt30 * (63 * rbx_val * rbx_val * ray_val - 63 * rby_val * rby_val * ray_val + 14 * rbx_val * sf.cyx() - 14 * rby_val * sf.cyy()) / 40;
            result.s2[13] = rt30 * (126 * ray_val * raz_val * rbx_val + 14 * ray_val * sf.czx() + 14 * raz_val * sf.cyx()) / 40;
            result.s2[14] = rt30 * (-126 * ray_val * raz_val * rby_val - 14 * ray_val * sf.czy() - 14 * raz_val * sf.cyy()) / 40;
            result.s2[30] = 7.0 / 20.0 * rt30 * rbx_val * rby_val;
            result.s2[31] = rt30 * (14 * raz_val * rby_val + 2 * sf.czy()) / 40;
            result.s2[32] = rt30 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 40;
            result.s2[37] = 7.0 / 20.0 * rt30 * rbx_val * rby_val;
            result.s2[39] = rt30 * (14 * ray_val * rby_val + 2 * sf.cyy()) / 40;
            result.s2[40] = rt30 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 40;
            result.s2[43] = rt30 * rby_val / 20;
            result.s2[57] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[58] = rt30 * (14 * raz_val * rbx_val + 2 * sf.czx()) / 40;
            result.s2[59] = rt30 * (-14 * raz_val * rby_val - 2 * sf.czy()) / 40;
            result.s2[63] = rt30 * rbx_val / 20;
            result.s2[67] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[69] = rt30 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 40;
            result.s2[70] = rt30 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 40;
            result.s2[73] = rt30 * rbx_val / 20;
            result.s2[76] = -rt30 * rby_val / 20;
        }
    }
}

/**
 * Quadrupole-22c × Octopole-30 kernel
 * Orient case 215: Q22c × Q30
 */
void quadrupole_22c_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt3 *
        (21 * rbz_val * rbz_val * rbz_val * rax_val * rax_val -
         21 * rbz_val * rbz_val * rbz_val * ray_val * ray_val -
         7 * rax_val * rax_val * rbz_val + 7 * ray_val * ray_val * rbz_val +
         14 * rbz_val * rbz_val * rax_val * sf.czx() -
         14 * rbz_val * rbz_val * ray_val * sf.czy() - 2 * rax_val * sf.czx() +
         2 * ray_val * sf.czy() + 2 * rbz_val * sf.czx() * sf.czx() -
         2 * rbz_val * sf.czy() * sf.czy()) /
        8;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt3 * (42 * rbz_val * rbz_val * rbz_val * rax_val - 14 * rax_val * rbz_val + 14 * rbz_val * rbz_val * sf.czx() - 2 * sf.czx()) / 8; // d/d(rax)
        result.s1[1] = rt3 * (-42 * rbz_val * rbz_val * rbz_val * ray_val + 14 * ray_val * rbz_val - 14 * rbz_val * rbz_val * sf.czy() + 2 * sf.czy()) / 8; // d/d(ray)
        result.s1[5] = rt3 * (63 * rbz_val * rbz_val * rax_val * rax_val - 63 * rbz_val * rbz_val * ray_val * ray_val - 7 * rax_val * rax_val + 7 * ray_val * ray_val + 28 * rbz_val * rax_val * sf.czx() - 28 * rbz_val * ray_val * sf.czy() + 2 * sf.czx() * sf.czx() - 2 * sf.czy() * sf.czy()) / 8; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[8] = rt3 * (14 * rbz_val * rbz_val * rax_val - 2 * rax_val + 4 * rbz_val * sf.czx()) / 8; // d/d(czx)
        result.s1[11] = rt3 * (-14 * rbz_val * rbz_val * ray_val + 2 * ray_val - 4 * rbz_val * sf.czy()) / 8; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 215
            result.s2[0] = rt3 * (42 * rbz_val * rbz_val * rbz_val - 14 * rbz_val) / 8;
            result.s2[2] = rt3 * (-42 * rbz_val * rbz_val * rbz_val + 14 * rbz_val) / 8;
            result.s2[15] = rt3 * (126 * rbz_val * rbz_val * rax_val - 14 * rax_val + 28 * rbz_val * sf.czx()) / 8;
            result.s2[16] = rt3 * (-126 * rbz_val * rbz_val * ray_val + 14 * ray_val - 28 * rbz_val * sf.czy()) / 8;
            result.s2[20] = rt3 * (126 * rax_val * rax_val * rbz_val - 126 * ray_val * ray_val * rbz_val + 28 * rax_val * sf.czx() - 28 * ray_val * sf.czy()) / 8;
            result.s2[78] = rt3 * (14 * rbz_val * rbz_val - 2) / 8;
            result.s2[83] = rt3 * (28 * rbz_val * rax_val + 4 * sf.czx()) / 8;
            result.s2[90] = rt3 * rbz_val / 2;
            result.s2[92] = rt3 * (-14 * rbz_val * rbz_val + 2) / 8;
            result.s2[96] = rt3 * (-28 * rbz_val * ray_val - 4 * sf.czy()) / 8;
            result.s2[104] = -rt3 * rbz_val / 2;
        }
    }
}

/**
 * Quadrupole-22c × Octopole-31c kernel
 * Orient case 216: Q22c × Q31c
 */
void quadrupole_22c_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt2 *
        (63 * rbx_val * rbz_val * rbz_val * rax_val * rax_val -
         63 * rbx_val * rbz_val * rbz_val * ray_val * ray_val -
         7 * rax_val * rax_val * rbx_val + 7 * ray_val * ray_val * rbx_val +
         28 * rbx_val * rbz_val * rax_val * sf.czx() -
         28 * rbx_val * rbz_val * ray_val * sf.czy() +
         14 * rbz_val * rbz_val * rax_val * sf.cxx() -
         14 * rbz_val * rbz_val * ray_val * sf.cyx() - 2 * rax_val * sf.cxx() +
         2 * ray_val * sf.cyx() + 2 * rbx_val * sf.czx() * sf.czx() -
         2 * rbx_val * sf.czy() * sf.czy() + 4 * rbz_val * sf.cxx() * sf.czx() -
         4 * rbz_val * sf.cyx() * sf.czy()) /
        16;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt2 * (126 * rbx_val * rbz_val * rbz_val * rax_val - 14 * rax_val * rbx_val + 28 * rbx_val * rbz_val * sf.czx() + 14 * rbz_val * rbz_val * sf.cxx() - 2 * sf.cxx()) / 16; // d/d(rax)
        result.s1[1] = rt2 * (-126 * rbx_val * rbz_val * rbz_val * ray_val + 14 * ray_val * rbx_val - 28 * rbx_val * rbz_val * sf.czy() - 14 * rbz_val * rbz_val * sf.cyx() + 2 * sf.cyx()) / 16; // d/d(ray)
        result.s1[3] = rt2 * (63 * rbz_val * rbz_val * rax_val * rax_val - 63 * rbz_val * rbz_val * ray_val * ray_val - 7 * rax_val * rax_val + 7 * ray_val * ray_val + 28 * rbz_val * rax_val * sf.czx() - 28 * rbz_val * ray_val * sf.czy() + 2 * sf.czx() * sf.czx() - 2 * sf.czy() * sf.czy()) / 16; // d/d(rbx)
        result.s1[5] = rt2 * (126 * rbx_val * rbz_val * rax_val * rax_val - 126 * rbx_val * rbz_val * ray_val * ray_val + 28 * rbx_val * rax_val * sf.czx() - 28 * rbx_val * ray_val * sf.czy() + 28 * rbz_val * rax_val * sf.cxx() - 28 * rbz_val * ray_val * sf.cyx() + 4 * sf.cxx() * sf.czx() - 4 * sf.cyx() * sf.czy()) / 16; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[6] = rt2 * (14 * rbz_val * rbz_val * rax_val - 2 * rax_val + 4 * rbz_val * sf.czx()) / 16; // d/d(cxx)
        result.s1[9] = rt2 * (-14 * rbz_val * rbz_val * ray_val + 2 * ray_val - 4 * rbz_val * sf.czy()) / 16; // d/d(cyx)
        result.s1[8] = rt2 * (28 * rbx_val * rbz_val * rax_val + 2 * rbx_val * sf.czx() + 4 * rbz_val * sf.cxx()) / 16; // d/d(czx)
        result.s1[11] = rt2 * (-28 * rbx_val * rbz_val * ray_val - 2 * rbx_val * sf.czy() - 4 * rbz_val * sf.cyx()) / 16; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 216
            result.s2[0] = rt2 * (126 * rbx_val * rbz_val * rbz_val - 14 * rbx_val) / 16;
            result.s2[2] = rt2 * (-126 * rbx_val * rbz_val * rbz_val + 14 * rbx_val) / 16;
            result.s2[6] = rt2 * (126 * rbz_val * rbz_val * rax_val - 14 * rax_val + 28 * rbz_val * sf.czx()) / 16;
            result.s2[7] = rt2 * (-126 * rbz_val * rbz_val * ray_val + 14 * ray_val - 28 * rbz_val * sf.czy()) / 16;
            result.s2[15] = rt2 * (252 * rbx_val * rbz_val * rax_val + 28 * rbx_val * sf.czx() + 28 * rbz_val * sf.cxx()) / 16;
            result.s2[16] = rt2 * (-252 * rbx_val * rbz_val * ray_val - 28 * rbx_val * sf.czy() - 28 * rbz_val * sf.cyx()) / 16;
            result.s2[18] = rt2 * (126 * rax_val * rax_val * rbz_val - 126 * ray_val * ray_val * rbz_val + 28 * rax_val * sf.czx() - 28 * ray_val * sf.czy()) / 16;
            result.s2[20] = rt2 * (126 * rax_val * rax_val * rbx_val - 126 * ray_val * ray_val * rbx_val + 28 * rax_val * sf.cxx() - 28 * ray_val * sf.cyx()) / 16;
            result.s2[21] = rt2 * (14 * rbz_val * rbz_val - 2) / 16;
            result.s2[26] = rt2 * (28 * rbz_val * rax_val + 4 * sf.czx()) / 16;
            result.s2[29] = rt2 * (-14 * rbz_val * rbz_val + 2) / 16;
            result.s2[33] = rt2 * (-28 * rbz_val * ray_val - 4 * sf.czy()) / 16;
            result.s2[78] = 7.0 / 4.0 * rt2 * rbx_val * rbz_val;
            result.s2[81] = rt2 * (28 * rbz_val * rax_val + 4 * sf.czx()) / 16;
            result.s2[83] = rt2 * (28 * rax_val * rbx_val + 4 * sf.cxx()) / 16;
            result.s2[84] = rt2 * rbz_val / 4;
            result.s2[90] = rt2 * rbx_val / 4;
            result.s2[92] = -7.0 / 4.0 * rt2 * rbx_val * rbz_val;
            result.s2[94] = rt2 * (-28 * rbz_val * ray_val - 4 * sf.czy()) / 16;
            result.s2[96] = rt2 * (-28 * rbx_val * ray_val - 4 * sf.cyx()) / 16;
            result.s2[98] = -rt2 * rbz_val / 4;
            result.s2[104] = -rt2 * rbx_val / 4;
        }
    }
}

/**
 * Quadrupole-22c × Octopole-31s kernel
 * Orient case 217: Q22c × Q31s
 */
void quadrupole_22c_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt2 *
        (63 * rby_val * rbz_val * rbz_val * rax_val * rax_val -
         63 * rby_val * rbz_val * rbz_val * ray_val * ray_val -
         7 * rax_val * rax_val * rby_val + 7 * ray_val * ray_val * rby_val +
         28 * rby_val * rbz_val * rax_val * sf.czx() -
         28 * rby_val * rbz_val * ray_val * sf.czy() +
         14 * rbz_val * rbz_val * rax_val * sf.cxy() -
         14 * rbz_val * rbz_val * ray_val * sf.cyy() - 2 * rax_val * sf.cxy() +
         2 * ray_val * sf.cyy() + 2 * rby_val * sf.czx() * sf.czx() -
         2 * rby_val * sf.czy() * sf.czy() + 4 * rbz_val * sf.cxy() * sf.czx() -
         4 * rbz_val * sf.cyy() * sf.czy()) /
        16;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt2 * (126 * rby_val * rbz_val * rbz_val * rax_val - 14 * rax_val * rby_val + 28 * rby_val * rbz_val * sf.czx() + 14 * rbz_val * rbz_val * sf.cxy() - 2 * sf.cxy()) / 16; // d/d(rax)
        result.s1[1] = rt2 * (-126 * rby_val * rbz_val * rbz_val * ray_val + 14 * ray_val * rby_val - 28 * rby_val * rbz_val * sf.czy() - 14 * rbz_val * rbz_val * sf.cyy() + 2 * sf.cyy()) / 16; // d/d(ray)
        result.s1[4] = rt2 * (63 * rbz_val * rbz_val * rax_val * rax_val - 63 * rbz_val * rbz_val * ray_val * ray_val - 7 * rax_val * rax_val + 7 * ray_val * ray_val + 28 * rbz_val * rax_val * sf.czx() - 28 * rbz_val * ray_val * sf.czy() + 2 * sf.czx() * sf.czx() - 2 * sf.czy() * sf.czy()) / 16; // d/d(rby)
        result.s1[5] = rt2 * (126 * rby_val * rbz_val * rax_val * rax_val - 126 * rby_val * rbz_val * ray_val * ray_val + 28 * rby_val * rax_val * sf.czx() - 28 * rby_val * ray_val * sf.czy() + 28 * rbz_val * rax_val * sf.cxy() - 28 * rbz_val * ray_val * sf.cyy() + 4 * sf.cxy() * sf.czx() - 4 * sf.cyy() * sf.czy()) / 16; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[8] = rt2 * (28 * rby_val * rbz_val * rax_val + 2 * rby_val * sf.czx() + 4 * rbz_val * sf.cxy()) / 16; // d/d(czx)
        result.s1[7] = rt2 * (14 * rbz_val * rbz_val * rax_val - 2 * rax_val + 4 * rbz_val * sf.czx()) / 16; // d/d(cxy)
        result.s1[10] = rt2 * (-14 * rbz_val * rbz_val * ray_val + 2 * ray_val - 4 * rbz_val * sf.czy()) / 16; // d/d(cyy)
        result.s1[11] = rt2 * (-28 * rby_val * rbz_val * ray_val - 2 * rby_val * sf.czy() - 4 * rbz_val * sf.cyy()) / 16; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 217
            result.s2[0] = rt2 * (126 * rby_val * rbz_val * rbz_val - 14 * rby_val) / 16;
            result.s2[2] = rt2 * (-126 * rby_val * rbz_val * rbz_val + 14 * rby_val) / 16;
            result.s2[10] = rt2 * (126 * rbz_val * rbz_val * rax_val - 14 * rax_val + 28 * rbz_val * sf.czx()) / 16;
            result.s2[11] = rt2 * (-126 * rbz_val * rbz_val * ray_val + 14 * ray_val - 28 * rbz_val * sf.czy()) / 16;
            result.s2[15] = rt2 * (252 * rby_val * rbz_val * rax_val + 28 * rby_val * sf.czx() + 28 * rbz_val * sf.cxy()) / 16;
            result.s2[16] = rt2 * (-252 * rby_val * rbz_val * ray_val - 28 * rby_val * sf.czy() - 28 * rbz_val * sf.cyy()) / 16;
            result.s2[19] = rt2 * (126 * rax_val * rax_val * rbz_val - 126 * ray_val * ray_val * rbz_val + 28 * rax_val * sf.czx() - 28 * ray_val * sf.czy()) / 16;
            result.s2[20] = rt2 * (126 * rax_val * rax_val * rby_val - 126 * ray_val * ray_val * rby_val + 28 * rax_val * sf.cxy() - 28 * ray_val * sf.cyy()) / 16;
            result.s2[45] = rt2 * (14 * rbz_val * rbz_val - 2) / 16;
            result.s2[50] = rt2 * (28 * rbz_val * rax_val + 4 * sf.czx()) / 16;
            result.s2[56] = rt2 * (-14 * rbz_val * rbz_val + 2) / 16;
            result.s2[60] = rt2 * (-28 * rbz_val * ray_val - 4 * sf.czy()) / 16;
            result.s2[78] = 7.0/4.0*rt2*rby_val*rbz_val;
            result.s2[82] = rt2 * (28 * rbz_val * rax_val + 4 * sf.czx()) / 16;
            result.s2[83] = rt2 * (28 * rax_val * rby_val + 4 * sf.cxy()) / 16;
            result.s2[87] = rt2*rbz_val/4;
            result.s2[90] = rt2*rby_val/4;
            result.s2[92] = -7.0/4.0*rt2*rby_val*rbz_val;
            result.s2[95] = rt2 * (-28 * rbz_val * ray_val - 4 * sf.czy()) / 16;
            result.s2[96] = rt2 * (-28 * ray_val * rby_val - 4 * sf.cyy()) / 16;
            result.s2[101] = -rt2*rbz_val/4;
            result.s2[104] = -rt2*rby_val/4;
        }
    }
}

/**
 * Quadrupole-22c × Octopole-32c kernel
 * Orient case 218: Q22c × Q32c
 */
void quadrupole_22c_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rbx_val * rbz_val * rax_val * rax_val -
                 63 * rbx_val * rbx_val * rbz_val * ray_val * ray_val -
                 63 * rby_val * rby_val * rbz_val * rax_val * rax_val +
                 63 * rby_val * rby_val * rbz_val * ray_val * ray_val +
                 14 * rbx_val * rbx_val * rax_val * sf.czx() -
                 14 * rbx_val * rbx_val * ray_val * sf.czy() +
                 28 * rbx_val * rbz_val * rax_val * sf.cxx() -
                 28 * rbx_val * rbz_val * ray_val * sf.cyx() -
                 14 * rby_val * rby_val * rax_val * sf.czx() +
                 14 * rby_val * rby_val * ray_val * sf.czy() -
                 28 * rby_val * rbz_val * rax_val * sf.cxy() +
                 28 * rby_val * rbz_val * ray_val * sf.cyy() +
                 4 * rbx_val * sf.cxx() * sf.czx() - 4 * rbx_val * sf.cyx() * sf.czy() -
                 4 * rby_val * sf.cxy() * sf.czx() + 4 * rby_val * sf.cyy() * sf.czy() +
                 2 * rbz_val * sf.cxx() * sf.cxx() - 2 * rbz_val * sf.cyx() * sf.cyx() -
                 2 * rbz_val * sf.cxy() * sf.cxy() + 2 * rbz_val * sf.cyy() * sf.cyy()) /
                40;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt5 * (126 * rbx_val * rbx_val * rbz_val * rax_val - 126 * rby_val * rby_val * rbz_val * rax_val + 14 * rbx_val * rbx_val * sf.czx() + 28 * rbx_val * rbz_val * sf.cxx() - 14 * rby_val * rby_val * sf.czx() - 28 * rby_val * rbz_val * sf.cxy()) / 40; // d/d(rax)
        result.s1[1] = rt5 * (-126 * rbx_val * rbx_val * rbz_val * ray_val + 126 * rby_val * rby_val * rbz_val * ray_val - 14 * rbx_val * rbx_val * sf.czy() - 28 * rbx_val * rbz_val * sf.cyx() + 14 * rby_val * rby_val * sf.czy() + 28 * rby_val * rbz_val * sf.cyy()) / 40; // d/d(ray)
        result.s1[3] = rt5 * (126 * rbx_val * rbz_val * rax_val * rax_val - 126 * rbx_val * rbz_val * ray_val * ray_val + 28 * rbx_val * rax_val * sf.czx() - 28 * rbx_val * ray_val * sf.czy() + 28 * rbz_val * rax_val * sf.cxx() - 28 * rbz_val * ray_val * sf.cyx() + 4 * sf.cxx() * sf.czx() - 4 * sf.cyx() * sf.czy() + 2 * rbz_val * sf.cxx() * sf.cxx() - 2 * rbz_val * sf.cyx() * sf.cyx()) / 40; // d/d(rbx)
        result.s1[4] = rt5 * (-126 * rby_val * rbz_val * rax_val * rax_val + 126 * rby_val * rbz_val * ray_val * ray_val - 28 * rby_val * rax_val * sf.czx() + 28 * rby_val * ray_val * sf.czy() - 28 * rbz_val * rax_val * sf.cxy() + 28 * rbz_val * ray_val * sf.cyy() - 4 * sf.cxy() * sf.czx() + 4 * sf.cyy() * sf.czy() - 2 * rbz_val * sf.cxy() * sf.cxy() + 2 * rbz_val * sf.cyy() * sf.cyy()) / 40; // d/d(rby)
        result.s1[5] = rt5 * (63 * rbx_val * rbx_val * rax_val * rax_val - 63 * rbx_val * rbx_val * ray_val * ray_val - 63 * rby_val * rby_val * rax_val * rax_val + 63 * rby_val * rby_val * ray_val * ray_val + 28 * rbx_val * rax_val * sf.cxx() - 28 * rbx_val * ray_val * sf.cyx() - 28 * rby_val * rax_val * sf.cxy() + 28 * rby_val * ray_val * sf.cyy() + 2 * sf.cxx() * sf.cxx() - 2 * sf.cyx() * sf.cyx() - 2 * sf.cxy() * sf.cxy() + 2 * sf.cyy() * sf.cyy()) / 40; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[6] = rt5 * (28 * rbx_val * rbz_val * rax_val + 4 * sf.czx() + 4 * rbz_val * sf.cxx()) / 40; // d/d(cxx)
        result.s1[9] = rt5 * (-28 * rbx_val * rbz_val * ray_val - 4 * sf.czy() - 4 * rbz_val * sf.cyx()) / 40; // d/d(cyx)
        result.s1[8] = rt5 * (14 * rbx_val * rbx_val * rax_val - 14 * rby_val * rby_val * rax_val + 4 * sf.cxx()) / 40; // d/d(czx)
        result.s1[7] = rt5 * (-28 * rby_val * rbz_val * rax_val - 4 * sf.czx() - 4 * rbz_val * sf.cxy()) / 40; // d/d(cxy)
        result.s1[10] = rt5 * (28 * rby_val * rbz_val * ray_val + 4 * sf.czy() + 4 * rbz_val * sf.cyy()) / 40; // d/d(cyy)
        result.s1[11] = rt5 * (-14 * rbx_val * rbx_val * ray_val + 14 * rby_val * rby_val * ray_val - 4 * sf.cyx() + 4 * sf.cyy()) / 40; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 218
            result.s2[0] = rt5 * (126 * rbx_val * rbx_val * rbz_val - 126 * rby_val * rby_val * rbz_val) / 40;
            result.s2[2] = rt5 * (-126 * rbx_val * rbx_val * rbz_val + 126 * rby_val * rby_val * rbz_val) / 40;
            result.s2[6] = rt5 * (252 * rbx_val * rbz_val * rax_val + 28 * rbx_val * sf.czx() + 28 * rbz_val * sf.cxx()) / 40;
            result.s2[7] = rt5 * (-252 * rbx_val * rbz_val * ray_val - 28 * rbx_val * sf.czy() - 28 * rbz_val * sf.cyx()) / 40;
            result.s2[9] = rt5 * (126 * rax_val * rax_val * rbz_val - 126 * ray_val * ray_val * rbz_val + 28 * rax_val * sf.czx() - 28 * ray_val * sf.czy()) / 40;
            result.s2[10] = rt5 * (-252 * rby_val * rbz_val * rax_val - 28 * rby_val * sf.czx() - 28 * rbz_val * sf.cxy()) / 40;
            result.s2[11] = rt5 * (252 * rby_val * rbz_val * ray_val + 28 * rby_val * sf.czy() + 28 * rbz_val * sf.cyy()) / 40;
            result.s2[14] = rt5 * (-126 * rax_val * rax_val * rbz_val + 126 * ray_val * ray_val * rbz_val - 28 * rax_val * sf.czx() + 28 * ray_val * sf.czy()) / 40;
            result.s2[15] = rt5 * (126 * rbx_val * rbx_val * rax_val - 126 * rby_val * rby_val * rax_val + 28 * rbx_val * sf.cxx() - 28 * rby_val * sf.cxy()) / 40;
            result.s2[16] = rt5 * (-126 * rbx_val * rbx_val * ray_val + 126 * rby_val * rby_val * ray_val - 28 * rbx_val * sf.cyx() + 28 * rby_val * sf.cyy()) / 40;
            result.s2[18] = rt5 * (126 * rax_val * rax_val * rbx_val - 126 * ray_val * ray_val * rbx_val + 28 * rax_val * sf.cxx() - 28 * ray_val * sf.cyx()) / 40;
            result.s2[19] = rt5 * (-126 * rax_val * rax_val * rby_val + 126 * ray_val * ray_val * rby_val - 28 * rax_val * sf.cxy() + 28 * ray_val * sf.cyy()) / 40;
            result.s2[21] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[24] = rt5 * (28 * rbz_val * rax_val + 4 * sf.czx()) / 40;
            result.s2[26] = rt5 * (28 * rax_val * rbx_val + 4 * sf.cxx()) / 40;
            result.s2[27] = rt5*rbz_val/10;
            result.s2[29] = -7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[31] = rt5 * (-28 * rbz_val * ray_val - 4 * sf.czy()) / 40;
            result.s2[33] = rt5 * (-28 * rbx_val * ray_val - 4 * sf.cyx()) / 40;
            result.s2[35] = -rt5*rbz_val/10;
            result.s2[45] = -7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[49] = rt5 * (-28 * rbz_val * rax_val - 4 * sf.czx()) / 40;
            result.s2[50] = rt5 * (-28 * rax_val * rby_val - 4 * sf.cxy()) / 40;
            result.s2[54] = -rt5*rbz_val/10;
            result.s2[56] = 7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[59] = rt5 * (28 * rbz_val * ray_val + 4 * sf.czy()) / 40;
            result.s2[60] = rt5 * (28 * ray_val * rby_val + 4 * sf.cyy()) / 40;
            result.s2[65] = rt5*rbz_val/10;
            result.s2[78] = rt5 * (14 * rbx_val * rbx_val - 14 * rby_val * rby_val) / 40;
            result.s2[81] = rt5 * (28 * rax_val * rbx_val + 4 * sf.cxx()) / 40;
            result.s2[82] = rt5 * (-28 * rax_val * rby_val - 4 * sf.cxy()) / 40;
            result.s2[84] = rt5*rbx_val/10;
            result.s2[87] = -rt5*rby_val/10;
            result.s2[92] = rt5 * (-14 * rbx_val * rbx_val + 14 * rby_val * rby_val) / 40;
            result.s2[94] = rt5 * (-28 * rbx_val * ray_val - 4 * sf.cyx()) / 40;
            result.s2[95] = rt5 * (28 * ray_val * rby_val + 4 * sf.cyy()) / 40;
            result.s2[98] = -rt5*rbx_val/10;
            result.s2[101] = rt5*rby_val/10;
        }
    }
}

/**
 * Quadrupole-22c × Octopole-32s kernel
 * Orient case 219: Q22c × Q32s
 */
void quadrupole_22c_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rby_val * rbz_val * rax_val * rax_val -
                 63 * rbx_val * rby_val * rbz_val * ray_val * ray_val +
                 14 * rbx_val * rby_val * rax_val * sf.czx() -
                 14 * rbx_val * rby_val * ray_val * sf.czy() +
                 14 * rbx_val * rbz_val * rax_val * sf.cxy() -
                 14 * rbx_val * rbz_val * ray_val * sf.cyy() +
                 14 * rby_val * rbz_val * rax_val * sf.cxx() -
                 14 * rby_val * rbz_val * ray_val * sf.cyx() +
                 2 * rbx_val * sf.cxy() * sf.czx() - 2 * rbx_val * sf.cyy() * sf.czy() +
                 2 * rby_val * sf.cxx() * sf.czx() - 2 * rby_val * sf.cyx() * sf.czy() +
                 2 * rbz_val * sf.cxx() * sf.cxy() - 2 * rbz_val * sf.cyx() * sf.cyy()) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt5 * (126 * rbx_val * rby_val * rbz_val * rax_val + 14 * rbx_val * rby_val * sf.czx() + 14 * rbx_val * rbz_val * sf.cxy() + 14 * rby_val * rbz_val * sf.cxx()) / 20; // d/d(rax)
        result.s1[1] = rt5 * (-126 * rbx_val * rby_val * rbz_val * ray_val - 14 * rbx_val * rby_val * sf.czy() - 14 * rbx_val * rbz_val * sf.cyy() - 14 * rby_val * rbz_val * sf.cyx()) / 20; // d/d(ray)
        result.s1[3] = rt5 * (63 * rby_val * rbz_val * rax_val * rax_val - 63 * rby_val * rbz_val * ray_val * ray_val + 14 * rby_val * rax_val * sf.czx() - 14 * rby_val * ray_val * sf.czy() + 14 * rbz_val * rax_val * sf.cxy() - 14 * rbz_val * ray_val * sf.cyy() + 2 * sf.cxy() * sf.czx() - 2 * sf.cyy() * sf.czy() + 2 * rbz_val * sf.cxx() * sf.cxy() - 2 * rbz_val * sf.cyx() * sf.cyy()) / 20; // d/d(rbx)
        result.s1[4] = rt5 * (63 * rbx_val * rbz_val * rax_val * rax_val - 63 * rbx_val * rbz_val * ray_val * ray_val + 14 * rbx_val * rax_val * sf.czx() - 14 * rbx_val * ray_val * sf.czy() + 14 * rbz_val * rax_val * sf.cxx() - 14 * rbz_val * ray_val * sf.cyx() + 2 * sf.cxx() * sf.czx() - 2 * sf.cyx() * sf.czy()) / 20; // d/d(rby)
        result.s1[5] = rt5 * (63 * rbx_val * rby_val * rax_val * rax_val - 63 * rbx_val * rby_val * ray_val * ray_val + 14 * rbx_val * rax_val * sf.cxy() - 14 * rbx_val * ray_val * sf.cyy() + 14 * rby_val * rax_val * sf.cxx() - 14 * rby_val * ray_val * sf.cyx() + 2 * sf.cxx() * sf.cxy() - 2 * sf.cyx() * sf.cyy()) / 20; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[6] = rt5 * (14 * rby_val * rbz_val * rax_val + 2 * sf.czx() + 2 * rbz_val * sf.cxy()) / 20; // d/d(cxx)
        result.s1[9] = rt5 * (-14 * rby_val * rbz_val * ray_val - 2 * sf.czy() - 2 * rbz_val * sf.cyy()) / 20; // d/d(cyx)
        result.s1[8] = rt5 * (14 * rbx_val * rby_val * rax_val + 2 * sf.cxy() + 2 * sf.cxx()) / 20; // d/d(czx)
        result.s1[7] = rt5 * (14 * rbx_val * rbz_val * rax_val + 2 * sf.czx() + 2 * rbz_val * sf.cxx()) / 20; // d/d(cxy)
        result.s1[10] = rt5 * (-14 * rbx_val * rbz_val * ray_val - 2 * sf.czy() - 2 * rbz_val * sf.cyx()) / 20; // d/d(cyy)
        result.s1[11] = rt5 * (-14 * rbx_val * rby_val * ray_val - 2 * sf.cyy() - 2 * sf.cyx()) / 20; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 219
            result.s2[0] = 63.0/10.0*rt5*rbx_val*rby_val*rbz_val;
            result.s2[2] = -63.0/10.0*rt5*rbx_val*rby_val*rbz_val;
            result.s2[6] = rt5 * (126 * rby_val * rbz_val * rax_val + 14 * rby_val * sf.czx() + 14 * rbz_val * sf.cxy()) / 20;
            result.s2[7] = rt5 * (-126 * rby_val * rbz_val * ray_val - 14 * rby_val * sf.czy() - 14 * rbz_val * sf.cyy()) / 20;
            result.s2[10] = rt5 * (126 * rbx_val * rbz_val * rax_val + 14 * rbx_val * sf.czx() + 14 * rbz_val * sf.cxx()) / 20;
            result.s2[11] = rt5 * (-126 * rbx_val * rbz_val * ray_val - 14 * rbx_val * sf.czy() - 14 * rbz_val * sf.cyx()) / 20;
            result.s2[13] = rt5 * (63 * rax_val * rax_val * rbz_val - 63 * ray_val * ray_val * rbz_val + 14 * rax_val * sf.czx() - 14 * ray_val * sf.czy()) / 20;
            result.s2[15] = rt5 * (126 * rbx_val * rby_val * rax_val + 14 * rbx_val * sf.cxy() + 14 * rby_val * sf.cxx()) / 20;
            result.s2[16] = rt5 * (-126 * rbx_val * rby_val * ray_val - 14 * rbx_val * sf.cyy() - 14 * rby_val * sf.cyx()) / 20;
            result.s2[18] = rt5 * (63 * rax_val * rax_val * rby_val - 63 * ray_val * ray_val * rby_val + 14 * rax_val * sf.cxy() - 14 * ray_val * sf.cyy()) / 20;
            result.s2[19] = rt5 * (63 * rax_val * rax_val * rbx_val - 63 * ray_val * ray_val * rbx_val + 14 * rax_val * sf.cxx() - 14 * ray_val * sf.cyx()) / 20;
            result.s2[21] = 7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[25] = rt5 * (14 * rbz_val * rax_val + 2 * sf.czx()) / 20;
            result.s2[26] = rt5 * (14 * rax_val * rby_val + 2 * sf.cxy()) / 20;
            result.s2[29] = -7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[32] = rt5 * (-14 * rbz_val * ray_val - 2 * sf.czy()) / 20;
            result.s2[33] = rt5 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 20;
            result.s2[45] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[48] = rt5 * (14 * rbz_val * rax_val + 2 * sf.czx()) / 20;
            result.s2[50] = rt5 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 20;
            result.s2[51] = rt5*rbz_val/10;
            result.s2[56] = -7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[58] = rt5 * (-14 * rbz_val * ray_val - 2 * sf.czy()) / 20;
            result.s2[60] = rt5 * (-14 * rbx_val * ray_val - 2 * sf.cyx()) / 20;
            result.s2[62] = -rt5*rbz_val/10;
            result.s2[78] = 7.0/10.0*rt5*rbx_val*rby_val;
            result.s2[81] = rt5 * (14 * rax_val * rby_val + 2 * sf.cxy()) / 20;
            result.s2[82] = rt5 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 20;
            result.s2[84] = rt5*rby_val/10;
            result.s2[87] = rt5*rbx_val/10;
            result.s2[92] = -7.0/10.0*rt5*rbx_val*rby_val;
            result.s2[94] = rt5 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 20;
            result.s2[95] = rt5 * (-14 * rbx_val * ray_val - 2 * sf.cyx()) / 20;
            result.s2[98] = -rt5*rby_val/10;
            result.s2[101] = -rt5*rbx_val/10;
        }
    }
}

/**
 * Quadrupole-22c × Octopole-33c kernel
 * Orient case 220: Q22c × Q33c
 */
void quadrupole_22c_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt30 *
                (21 * rbx_val * rbx_val * rbx_val * rax_val * rax_val -
                 21 * rbx_val * rbx_val * rbx_val * ray_val * ray_val -
                 63 * rbx_val * rby_val * rby_val * rax_val * rax_val +
                 63 * rbx_val * rby_val * rby_val * ray_val * ray_val +
                 14 * rbx_val * rbx_val * rax_val * sf.cxx() -
                 14 * rbx_val * rbx_val * ray_val * sf.cyx() -
                 28 * rbx_val * rby_val * rax_val * sf.cxy() +
                 28 * rbx_val * rby_val * ray_val * sf.cyy() -
                 14 * rby_val * rby_val * rax_val * sf.cxx() +
                 14 * rby_val * rby_val * ray_val * sf.cyx() +
                 2 * rbx_val * sf.cxx() * sf.cxx() - 2 * rbx_val * sf.cyx() * sf.cyx() -
                 2 * rbx_val * sf.cxy() * sf.cxy() + 2 * rbx_val * sf.cyy() * sf.cyy() -
                 4 * rby_val * sf.cxx() * sf.cxy() + 4 * rby_val * sf.cyx() * sf.cyy()) /
                80;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt30 * (42 * rbx_val * rbx_val * rbx_val * rax_val - 126 * rbx_val * rby_val * rby_val * rax_val + 14 * rbx_val * rbx_val * sf.cxx() - 28 * rbx_val * rby_val * sf.cxy() - 14 * rby_val * rby_val * sf.cxx()) / 80; // d/d(rax)
        result.s1[1] = rt30 * (-42 * rbx_val * rbx_val * rbx_val * ray_val + 126 * rbx_val * rby_val * rby_val * ray_val - 14 * rbx_val * rbx_val * sf.cyx() + 28 * rbx_val * rby_val * sf.cyy() + 14 * rby_val * rby_val * sf.cyx()) / 80; // d/d(ray)
        result.s1[3] = rt30 * (63 * rbx_val * rbx_val * rax_val * rax_val - 63 * rbx_val * rbx_val * ray_val * ray_val - 63 * rby_val * rby_val * rax_val * rax_val + 63 * rby_val * rby_val * ray_val * ray_val + 28 * rbx_val * rax_val * sf.cxx() - 28 * rbx_val * ray_val * sf.cyx() - 28 * rby_val * rax_val * sf.cxy() + 28 * rby_val * ray_val * sf.cyy() + 2 * sf.cxx() * sf.cxx() - 2 * sf.cyx() * sf.cyx() - 2 * sf.cxy() * sf.cxy() + 2 * sf.cyy() * sf.cyy()) / 80; // d/d(rbx)
        result.s1[4] = rt30 * (-126 * rbx_val * rby_val * rax_val * rax_val + 126 * rbx_val * rby_val * ray_val * ray_val - 28 * rbx_val * rax_val * sf.cxy() + 28 * rbx_val * ray_val * sf.cyy() - 28 * rby_val * rax_val * sf.cxx() + 28 * rby_val * ray_val * sf.cyx() - 4 * sf.cxx() * sf.cxy() + 4 * sf.cyx() * sf.cyy()) / 80; // d/d(rby)

        // Orientation matrix derivatives
        result.s1[6] = rt30 * (14 * rbx_val * rbx_val * rax_val - 14 * rby_val * rby_val * rax_val + 4 * sf.cxx()) / 80; // d/d(cxx)
        result.s1[9] = rt30 * (-14 * rbx_val * rbx_val * ray_val + 14 * rby_val * rby_val * ray_val - 4 * sf.cyx()) / 80; // d/d(cyx)
        result.s1[7] = rt30 * (-28 * rbx_val * rby_val * rax_val - 4 * sf.cxx() - 4 * sf.cxy()) / 80; // d/d(cxy)
        result.s1[10] = rt30 * (28 * rbx_val * rby_val * ray_val + 4 * sf.cyx() + 4 * sf.cyy()) / 80; // d/d(cyy)

        if (level >= 2) {
            // Second derivatives - Orient case 220
            result.s2[0] = rt30 * (42 * rbx_val * rbx_val * rbx_val - 126 * rbx_val * rby_val * rby_val) / 80;
            result.s2[2] = rt30 * (-42 * rbx_val * rbx_val * rbx_val + 126 * rbx_val * rby_val * rby_val) / 80;
            result.s2[6] = rt30 * (126 * rbx_val * rbx_val * rax_val - 126 * rby_val * rby_val * rax_val + 28 * rbx_val * sf.cxx() - 28 * rby_val * sf.cxy()) / 80;
            result.s2[7] = rt30 * (-126 * rbx_val * rbx_val * ray_val + 126 * rby_val * rby_val * ray_val - 28 * rbx_val * sf.cyx() + 28 * rby_val * sf.cyy()) / 80;
            result.s2[9] = rt30 * (126 * rax_val * rax_val * rbx_val - 126 * ray_val * ray_val * rbx_val + 28 * rax_val * sf.cxx() - 28 * ray_val * sf.cyx()) / 80;
            result.s2[10] = rt30 * (-252 * rbx_val * rby_val * rax_val - 28 * rbx_val * sf.cxy() - 28 * rby_val * sf.cxx()) / 80;
            result.s2[11] = rt30 * (252 * rbx_val * rby_val * ray_val + 28 * rbx_val * sf.cyy() + 28 * rby_val * sf.cyx()) / 80;
            result.s2[13] = rt30 * (-126 * rax_val * rax_val * rby_val + 126 * ray_val * ray_val * rby_val - 28 * rax_val * sf.cxy() + 28 * ray_val * sf.cyy()) / 80;
            result.s2[14] = rt30 * (-126 * rax_val * rax_val * rbx_val + 126 * ray_val * ray_val * rbx_val - 28 * rax_val * sf.cxx() + 28 * ray_val * sf.cyx()) / 80;
            result.s2[21] = rt30 * (14 * rbx_val * rbx_val - 14 * rby_val * rby_val) / 80;
            result.s2[24] = rt30 * (28 * rax_val * rbx_val + 4 * sf.cxx()) / 80;
            result.s2[25] = rt30 * (-28 * rax_val * rby_val - 4 * sf.cxy()) / 80;
            result.s2[27] = rt30*rbx_val/20;
            result.s2[29] = rt30 * (-14 * rbx_val * rbx_val + 14 * rby_val * rby_val) / 80;
            result.s2[31] = rt30 * (-28 * rbx_val * ray_val - 4 * sf.cyx()) / 80;
            result.s2[32] = rt30 * (28 * ray_val * rby_val + 4 * sf.cyy()) / 80;
            result.s2[35] = -rt30*rbx_val/20;
            result.s2[45] = -7.0/20.0*rt30*rbx_val*rby_val;
            result.s2[48] = rt30 * (-28 * rax_val * rby_val - 4 * sf.cxy()) / 80;
            result.s2[49] = rt30 * (-28 * rax_val * rbx_val - 4 * sf.cxx()) / 80;
            result.s2[51] = -rt30*rby_val/20;
            result.s2[54] = -rt30*rbx_val/20;
            result.s2[56] = 7.0/20.0*rt30*rbx_val*rby_val;
            result.s2[58] = rt30 * (28 * ray_val * rby_val + 4 * sf.cyy()) / 80;
            result.s2[59] = rt30 * (28 * rbx_val * ray_val + 4 * sf.cyx()) / 80;
            result.s2[62] = rt30*rby_val/20;
            result.s2[65] = rt30*rbx_val/20;
        }
    }
}

/**
 * Quadrupole-22c × Octopole-33s kernel
 * Orient case 221: Q22c × Q33s
 */
void quadrupole_22c_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt30 *
                (63 * rbx_val * rbx_val * rby_val * rax_val * rax_val -
                 63 * rbx_val * rbx_val * rby_val * ray_val * ray_val -
                 21 * rby_val * rby_val * rby_val * rax_val * rax_val +
                 21 * rby_val * rby_val * rby_val * ray_val * ray_val +
                 14 * rbx_val * rbx_val * rax_val * sf.cxy() -
                 14 * rbx_val * rbx_val * ray_val * sf.cyy() +
                 28 * rbx_val * rby_val * rax_val * sf.cxx() -
                 28 * rbx_val * rby_val * ray_val * sf.cyx() -
                 14 * rby_val * rby_val * rax_val * sf.cxy() +
                 14 * rby_val * rby_val * ray_val * sf.cyy() +
                 2 * rbx_val * sf.cxx() * sf.cxy() - 2 * rbx_val * sf.cyx() * sf.cyy() +
                 2 * rby_val * sf.cxx() * sf.cxx() - 2 * rby_val * sf.cyx() * sf.cyx() -
                 2 * rby_val * sf.cxy() * sf.cxy() + 2 * rby_val * sf.cyy() * sf.cyy()) /
                80;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt30 * (126 * rbx_val * rbx_val * rby_val * rax_val - 42 * rby_val * rby_val * rby_val * rax_val + 14 * rbx_val * rbx_val * sf.cxy() + 28 * rbx_val * rby_val * sf.cxx() - 14 * rby_val * rby_val * sf.cxy()) / 80; // d/d(rax)
        result.s1[1] = rt30 * (-126 * rbx_val * rbx_val * rby_val * ray_val + 42 * rby_val * rby_val * rby_val * ray_val - 14 * rbx_val * rbx_val * sf.cyy() - 28 * rbx_val * rby_val * sf.cyx() + 14 * rby_val * rby_val * sf.cyy()) / 80; // d/d(ray)
        result.s1[3] = rt30 * (126 * rbx_val * rby_val * rax_val * rax_val - 126 * rbx_val * rby_val * ray_val * ray_val + 28 * rbx_val * rax_val * sf.cxy() - 28 * rbx_val * ray_val * sf.cyy() + 28 * rby_val * rax_val * sf.cxx() - 28 * rby_val * ray_val * sf.cyx() + 2 * sf.cxx() * sf.cxy() - 2 * sf.cyx() * sf.cyy()) / 80; // d/d(rbx)
        result.s1[4] = rt30 * (63 * rbx_val * rbx_val * rax_val * rax_val - 63 * rbx_val * rbx_val * ray_val * ray_val - 63 * rby_val * rby_val * rax_val * rax_val + 63 * rby_val * rby_val * ray_val * ray_val + 28 * rbx_val * rax_val * sf.cxx() - 28 * rbx_val * ray_val * sf.cyx() - 28 * rby_val * rax_val * sf.cxy() + 28 * rby_val * ray_val * sf.cyy() + 2 * sf.cxx() * sf.cxx() - 2 * sf.cyx() * sf.cyx() - 2 * sf.cxy() * sf.cxy() + 2 * sf.cyy() * sf.cyy()) / 80; // d/d(rby)

        // Orientation matrix derivatives
        result.s1[6] = rt30 * (28 * rbx_val * rby_val * rax_val + 2 * sf.cxy() + 4 * sf.cxx()) / 80; // d/d(cxx)
        result.s1[9] = rt30 * (-28 * rbx_val * rby_val * ray_val - 2 * sf.cyy() - 4 * sf.cyx()) / 80; // d/d(cyx)
        result.s1[7] = rt30 * (14 * rbx_val * rbx_val * rax_val - 14 * rby_val * rby_val * rax_val + 2 * sf.cxx()) / 80; // d/d(cxy)
        result.s1[10] = rt30 * (-14 * rbx_val * rbx_val * ray_val + 14 * rby_val * rby_val * ray_val - 2 * sf.cyx() + 4 * sf.cyy()) / 80; // d/d(cyy)

        if (level >= 2) {
            // Second derivatives - Orient case 221
            result.s2[0] = rt30 * (126 * rbx_val * rbx_val * rby_val - 42 * rby_val * rby_val * rby_val) / 80;
            result.s2[2] = rt30 * (-126 * rbx_val * rbx_val * rby_val + 42 * rby_val * rby_val * rby_val) / 80;
            result.s2[6] = rt30 * (252 * rbx_val * rby_val * rax_val + 28 * rbx_val * sf.cxy() + 28 * rby_val * sf.cxx()) / 80;
            result.s2[7] = rt30 * (-252 * rbx_val * rby_val * ray_val - 28 * rbx_val * sf.cyy() - 28 * rby_val * sf.cyx()) / 80;
            result.s2[9] = rt30 * (126 * rax_val * rax_val * rby_val - 126 * ray_val * ray_val * rby_val + 28 * rax_val * sf.cxy() - 28 * ray_val * sf.cyy()) / 80;
            result.s2[10] = rt30 * (126 * rbx_val * rbx_val * rax_val - 126 * rby_val * rby_val * rax_val + 28 * rbx_val * sf.cxx() - 28 * rby_val * sf.cxy()) / 80;
            result.s2[11] = rt30 * (-126 * rbx_val * rbx_val * ray_val + 126 * rby_val * rby_val * ray_val - 28 * rbx_val * sf.cyx() + 28 * rby_val * sf.cyy()) / 80;
            result.s2[13] = rt30 * (126 * rax_val * rax_val * rbx_val - 126 * ray_val * ray_val * rbx_val + 28 * rax_val * sf.cxx() - 28 * ray_val * sf.cyx()) / 80;
            result.s2[14] = rt30 * (-126 * rax_val * rax_val * rby_val + 126 * ray_val * ray_val * rby_val - 28 * rax_val * sf.cxy() + 28 * ray_val * sf.cyy()) / 80;
            result.s2[21] = 7.0/20.0*rt30*rbx_val*rby_val;
            result.s2[24] = rt30 * (28 * rax_val * rby_val + 4 * sf.cxy()) / 80;
            result.s2[25] = rt30 * (28 * rax_val * rbx_val + 4 * sf.cxx()) / 80;
            result.s2[27] = rt30*rby_val/20;
            result.s2[29] = -7.0/20.0*rt30*rbx_val*rby_val;
            result.s2[31] = rt30 * (-28 * ray_val * rby_val - 4 * sf.cyy()) / 80;
            result.s2[32] = rt30 * (-28 * rbx_val * ray_val - 4 * sf.cyx()) / 80;
            result.s2[35] = -rt30*rby_val/20;
            result.s2[45] = rt30 * (14 * rbx_val * rbx_val - 14 * rby_val * rby_val) / 80;
            result.s2[48] = rt30 * (28 * rax_val * rbx_val + 4 * sf.cxx()) / 80;
            result.s2[49] = rt30 * (-28 * rax_val * rby_val - 4 * sf.cxy()) / 80;
            result.s2[51] = rt30*rbx_val/20;
            result.s2[54] = -rt30*rby_val/20;
            result.s2[56] = rt30 * (-14 * rbx_val * rbx_val + 14 * rby_val * rby_val) / 80;
            result.s2[58] = rt30 * (-28 * rbx_val * ray_val - 4 * sf.cyx()) / 80;
            result.s2[59] = rt30 * (28 * ray_val * rby_val + 4 * sf.cyy()) / 80;
            result.s2[62] = -rt30*rbx_val/20;
            result.s2[65] = rt30*rby_val/20;
        }
    }
}

/**
 * Quadrupole-22s × Octopole-30 kernel
 * Orient case 222: Q22s × Q30
 */
void quadrupole_22s_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt3 *
        (21 * rbz_val * rbz_val * rbz_val * rax_val * ray_val -
         7 * rax_val * ray_val * rbz_val +
         14 * rbz_val * rbz_val * rax_val * sf.czy() +
         14 * rbz_val * rbz_val * ray_val * sf.czx() - 2 * rax_val * sf.czy() -
         2 * ray_val * sf.czx() + 4 * rbz_val * sf.czx() * sf.czy()) /
        4;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt3 * (21 * rbz_val * rbz_val * rbz_val * ray_val - 7 * ray_val * rbz_val + 14 * rbz_val * rbz_val * sf.czy() - 2 * sf.czy()) / 4; // d/d(rax)
        result.s1[1] = rt3 * (21 * rbz_val * rbz_val * rbz_val * rax_val - 7 * rax_val * rbz_val + 14 * rbz_val * rbz_val * sf.czx() - 2 * sf.czx()) / 4; // d/d(ray)
        result.s1[5] = rt3 * (63 * rbz_val * rbz_val * rax_val * ray_val - 7 * rax_val * ray_val + 28 * rbz_val * rax_val * sf.czy() + 28 * rbz_val * ray_val * sf.czx() + 4 * sf.czx() * sf.czy()) / 4; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[8] = rt3 * (14 * rbz_val * rbz_val * ray_val - 2 * ray_val + 4 * rbz_val * sf.czy()) / 4; // d/d(czx)
        result.s1[11] = rt3 * (14 * rbz_val * rbz_val * rax_val - 2 * rax_val + 4 * rbz_val * sf.czx()) / 4; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 222
            result.s2[1] = rt3*(21*rbz_val * rbz_val * rbz_val-7*rbz_val)/4;
            result.s2[15] = rt3*(63*rbz_val * rbz_val*ray_val-7*ray_val+14*rbz_val*sf.cyz())/4;
            result.s2[16] = rt3*(63*rbz_val * rbz_val*rax_val-7*rax_val+14*rbz_val*sf.cxz())/4;
            result.s2[20] = rt3*(126*rax_val*ray_val*rbz_val+14*rax_val*sf.cyz()+14*ray_val*sf.cxz())/4;
            result.s2[79] = rt3*(7*rbz_val * rbz_val-1)/4;
            result.s2[83] = rt3*(14*rbz_val*ray_val+2*sf.cyz())/4;
            result.s2[91] = rt3*(7*rbz_val * rbz_val-1)/4;
            result.s2[96] = rt3*(14*rbz_val*rax_val+2*sf.cxz())/4;
            result.s2[103] = rt3*rbz_val/2;
        }
    }
}

/**
 * Quadrupole-22s × Octopole-31c kernel
 * Orient case 223: Q22s × Q31c
 */
void quadrupole_22s_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt2 *
        (63 * rbx_val * rbz_val * rbz_val * rax_val * ray_val -
         7 * rax_val * ray_val * rbx_val +
         14 * rbx_val * rbz_val * rax_val * sf.czy() +
         14 * rbx_val * rbz_val * ray_val * sf.czx() +
         14 * rbz_val * rbz_val * rax_val * sf.cxy() +
         14 * rbz_val * rbz_val * ray_val * sf.cxx() - 2 * rax_val * sf.cxy() -
         2 * ray_val * sf.cxx() + 4 * rbx_val * sf.czx() * sf.czy() +
         4 * rbz_val * sf.cxx() * sf.czy() + 4 * rbz_val * sf.cxy() * sf.czx()) /
        16;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt2 * (63 * rbx_val * rbz_val * rbz_val * ray_val - 7 * ray_val * rbx_val + 14 * rbx_val * rbz_val * sf.czy() + 14 * rbz_val * rbz_val * sf.cxy() - 2 * sf.cxy()) / 16; // d/d(rax)
        result.s1[1] = rt2 * (63 * rbx_val * rbz_val * rbz_val * rax_val - 7 * rax_val * rbx_val + 14 * rbx_val * rbz_val * sf.czx() + 14 * rbz_val * rbz_val * sf.cxx() - 2 * sf.cxx()) / 16; // d/d(ray)
        result.s1[3] = rt2 * (63 * rbz_val * rbz_val * rax_val * ray_val - 7 * rax_val * ray_val + 14 * rbz_val * rax_val * sf.czy() + 14 * rbz_val * ray_val * sf.czx() + 4 * sf.czx() * sf.czy()) / 16; // d/d(rbx)
        result.s1[5] = rt2 * (126 * rbx_val * rbz_val * rax_val * ray_val + 14 * rbx_val * rax_val * sf.czy() + 14 * rbx_val * ray_val * sf.czx() + 28 * rbz_val * rax_val * sf.cxy() + 28 * rbz_val * ray_val * sf.cxx() + 4 * sf.cxx() * sf.czy() + 4 * sf.cxy() * sf.czx()) / 16; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[6] = rt2 * (14 * rbz_val * rbz_val * ray_val - 2 * ray_val + 4 * rbz_val * sf.czy()) / 16; // d/d(cxx)
        result.s1[8] = rt2 * (14 * rbx_val * rbz_val * ray_val + 14 * rbz_val * ray_val + 4 * rbx_val * sf.czy() + 4 * sf.cxy()) / 16; // d/d(czx)
        result.s1[7] = rt2 * (14 * rbz_val * rbz_val * rax_val - 2 * rax_val + 4 * rbz_val * sf.czx()) / 16; // d/d(cxy)
        result.s1[11] = rt2 * (14 * rbx_val * rbz_val * rax_val + 14 * rbz_val * rax_val + 4 * rbx_val * sf.czx() + 4 * sf.cxx()) / 16; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 223
            result.s2[1] = rt2*(63*rbx_val*rbz_val * rbz_val-7*rbx_val)/8;
            result.s2[6] = rt2*(63*rbz_val * rbz_val*ray_val-7*ray_val+14*rbz_val*sf.cyz())/8;
            result.s2[7] = rt2*(63*rbz_val * rbz_val*rax_val-7*rax_val+14*rbz_val*sf.cxz())/8;
            result.s2[15] = rt2*(126*rbx_val*rbz_val*ray_val+14*rbx_val*sf.cyz()+14*rbz_val*sf.cyx())/8;
            result.s2[16] = rt2*(126*rbx_val*rbz_val*rax_val+14*rbx_val*sf.cxz()+14*rbz_val*sf.cxx())/8;
            result.s2[18] = rt2*(126*rax_val*ray_val*rbz_val+14*rax_val*sf.cyz()+14*ray_val*sf.cxz())/8;
            result.s2[20] = rt2*(126*rax_val*ray_val*rbx_val+14*rax_val*sf.cyx()+14*ray_val*sf.cxx())/8;
            result.s2[22] = rt2*(7*rbz_val * rbz_val-1)/8;
            result.s2[26] = rt2*(14*rbz_val*ray_val+2*sf.cyz())/8;
            result.s2[28] = rt2*(7*rbz_val * rbz_val-1)/8;
            result.s2[33] = rt2*(14*rbz_val*rax_val+2*sf.cxz())/8;
            result.s2[79] = 7.0/4.0*rt2*rbx_val*rbz_val;
            result.s2[81] = rt2*(14*rbz_val*ray_val+2*sf.cyz())/8;
            result.s2[83] = rt2*(14*rbx_val*ray_val+2*sf.cyx())/8;
            result.s2[85] = rt2*rbz_val/4;
            result.s2[91] = 7.0/4.0*rt2*rbx_val*rbz_val;
            result.s2[94] = rt2*(14*rbz_val*rax_val+2*sf.cxz())/8;
            result.s2[96] = rt2*(14*rax_val*rbx_val+2*sf.cxx())/8;
            result.s2[97] = rt2*rbz_val/4;
            result.s2[103] = rt2*rbx_val/4;
        }
    }
}

/**
 * Quadrupole-22s × Octopole-31s kernel
 * Orient case 224: Q22s × Q31s
 */
void quadrupole_22s_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 =
        rt2 *
        (63 * rby_val * rbz_val * rbz_val * rax_val * ray_val -
         7 * rax_val * ray_val * rby_val +
         14 * rby_val * rbz_val * rax_val * sf.czy() +
         14 * rby_val * rbz_val * ray_val * sf.czx() +
         14 * rbz_val * rbz_val * rax_val * sf.cyy() +
         14 * rbz_val * rbz_val * ray_val * sf.cxy() - 2 * rax_val * sf.cyy() -
         2 * ray_val * sf.cxy() + 4 * rby_val * sf.czx() * sf.czy() +
         4 * rbz_val * sf.cxy() * sf.czy() + 4 * rbz_val * sf.cyy() * sf.czx()) /
        16;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt2 * (63 * rby_val * rbz_val * rbz_val * ray_val - 7 * ray_val * rby_val + 14 * rby_val * rbz_val * sf.czy() + 14 * rbz_val * rbz_val * sf.cyy() - 2 * sf.cyy()) / 16; // d/d(rax)
        result.s1[1] = rt2 * (63 * rby_val * rbz_val * rbz_val * rax_val - 7 * rax_val * rby_val + 14 * rby_val * rbz_val * sf.czx() + 14 * rbz_val * rbz_val * sf.cxy() - 2 * sf.cxy()) / 16; // d/d(ray)
        result.s1[4] = rt2 * (63 * rbz_val * rbz_val * rax_val * ray_val - 7 * rax_val * ray_val + 14 * rbz_val * rax_val * sf.czy() + 14 * rbz_val * ray_val * sf.czx() + 4 * sf.czx() * sf.czy()) / 16; // d/d(rby)
        result.s1[5] = rt2 * (126 * rby_val * rbz_val * rax_val * ray_val + 14 * rby_val * rax_val * sf.czy() + 14 * rby_val * ray_val * sf.czx() + 28 * rbz_val * rax_val * sf.cyy() + 28 * rbz_val * ray_val * sf.cxy() + 4 * sf.cxy() * sf.czy() + 4 * rbz_val * sf.cyy() * sf.czx()) / 16; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[8] = rt2 * (14 * rby_val * rbz_val * ray_val + 14 * rbz_val * ray_val + 4 * rby_val * sf.czy() + 4 * rbz_val * sf.cyy()) / 16; // d/d(czx)
        result.s1[7] = rt2 * (14 * rbz_val * rbz_val * rax_val - 2 * rax_val + 4 * rbz_val * sf.czy()) / 16; // d/d(cxy)
        result.s1[10] = rt2 * (14 * rbz_val * rbz_val * ray_val - 2 * ray_val + 4 * rbz_val * sf.czx()) / 16; // d/d(cyy)
        result.s1[11] = rt2 * (14 * rby_val * rbz_val * rax_val + 14 * rbz_val * rax_val + 4 * rby_val * sf.czx() + 4 * sf.cxy()) / 16; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 224
            result.s2[1] = rt2*(63*rby_val*rbz_val * rbz_val-7*rby_val)/8;
            result.s2[10] = rt2*(63*rbz_val * rbz_val*ray_val-7*ray_val+14*rbz_val*sf.cyz())/8;
            result.s2[11] = rt2*(63*rbz_val * rbz_val*rax_val-7*rax_val+14*rbz_val*sf.cxz())/8;
            result.s2[15] = rt2*(126*rby_val*rbz_val*ray_val+14*rby_val*sf.cyz()+14*rbz_val*sf.cyy())/8;
            result.s2[16] = rt2*(126*rby_val*rbz_val*rax_val+14*rby_val*sf.cxz()+14*rbz_val*sf.cxy())/8;
            result.s2[19] = rt2*(126*rax_val*ray_val*rbz_val+14*rax_val*sf.cyz()+14*ray_val*sf.cxz())/8;
            result.s2[20] = rt2*(126*rax_val*ray_val*rby_val+14*rax_val*sf.cyy()+14*ray_val*sf.cxy())/8;
            result.s2[46] = rt2*(7*rbz_val * rbz_val-1)/8;
            result.s2[50] = rt2*(14*rbz_val*ray_val+2*sf.cyz())/8;
            result.s2[55] = rt2*(7*rbz_val * rbz_val-1)/8;
            result.s2[60] = rt2*(14*rbz_val*rax_val+2*sf.cxz())/8;
            result.s2[79] = 7.0/4.0*rt2*rby_val*rbz_val;
            result.s2[82] = rt2*(14*rbz_val*ray_val+2*sf.cyz())/8;
            result.s2[83] = rt2*(14*ray_val*rby_val+2*sf.cyy())/8;
            result.s2[88] = rt2*rbz_val/4;
            result.s2[91] = 7.0/4.0*rt2*rby_val*rbz_val;
            result.s2[95] = rt2*(14*rbz_val*rax_val+2*sf.cxz())/8;
            result.s2[96] = rt2*(14*rax_val*rby_val+2*sf.cxy())/8;
            result.s2[100] = rt2*rbz_val/4;
            result.s2[103] = rt2*rby_val/4;
        }
    }
}

/**
 * Quadrupole-22s × Octopole-32c kernel
 * Orient case 225: Q22s × Q32c
 */
void quadrupole_22s_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rbx_val * rbz_val * rax_val * ray_val -
                 63 * rby_val * rby_val * rbz_val * rax_val * ray_val +
                 14 * rbx_val * rbx_val * rax_val * sf.czy() +
                 14 * rbx_val * rbx_val * ray_val * sf.czx() +
                 28 * rbx_val * rbz_val * rax_val * sf.cxy() +
                 28 * rbx_val * rbz_val * ray_val * sf.cxx() -
                 14 * rby_val * rby_val * rax_val * sf.czy() -
                 14 * rby_val * rby_val * ray_val * sf.czx() -
                 28 * rby_val * rbz_val * rax_val * sf.cyy() -
                 28 * rby_val * rbz_val * ray_val * sf.cxy() +
                 4 * rbx_val * sf.cxx() * sf.czy() + 4 * rbx_val * sf.cxy() * sf.czx() -
                 4 * rby_val * sf.cxy() * sf.czy() - 4 * rby_val * sf.cyy() * sf.czx() +
                 4 * rbz_val * sf.cxx() * sf.cxy() - 4 * rbz_val * sf.cxy() * sf.cxy()) /
                40;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt5 * (63 * rbx_val * rbx_val * rbz_val * ray_val - 63 * rby_val * rby_val * rbz_val * ray_val + 14 * rbx_val * rbx_val * sf.czy() + 28 * rbx_val * rbz_val * sf.cxy() - 14 * rby_val * rby_val * sf.czy() - 28 * rby_val * rbz_val * sf.cyy()) / 40; // d/d(rax)
        result.s1[1] = rt5 * (63 * rbx_val * rbx_val * rbz_val * rax_val - 63 * rby_val * rby_val * rbz_val * rax_val + 14 * rbx_val * rbx_val * sf.czx() + 28 * rbx_val * rbz_val * sf.cxx() - 14 * rby_val * rby_val * sf.czx() - 28 * rby_val * rbz_val * sf.cxy()) / 40; // d/d(ray)
        result.s1[3] = rt5 * (126 * rbx_val * rbz_val * rax_val * ray_val + 28 * rbx_val * rax_val * sf.czy() + 28 * rbx_val * ray_val * sf.czx() + 28 * rbz_val * rax_val * sf.cxy() + 28 * rbz_val * ray_val * sf.cxx() + 4 * sf.cxx() * sf.czy() + 4 * sf.cxy() * sf.czx() + 4 * rbz_val * sf.cxx() * sf.cxy()) / 40; // d/d(rbx)
        result.s1[4] = rt5 * (-126 * rby_val * rbz_val * rax_val * ray_val - 28 * rby_val * rax_val * sf.czy() - 28 * rby_val * ray_val * sf.czx() - 28 * rbz_val * rax_val * sf.cyy() - 28 * rbz_val * ray_val * sf.cxy() - 4 * sf.cxy() * sf.czy() - 4 * sf.cyy() * sf.czx() - 8 * rbz_val * sf.cxy() * sf.cxy()) / 40; // d/d(rby)
        result.s1[5] = rt5 * (63 * rbx_val * rbx_val * rax_val * ray_val - 63 * rby_val * rby_val * rax_val * ray_val + 28 * rbx_val * rax_val * sf.cxy() + 28 * rbx_val * ray_val * sf.cxx() - 28 * rby_val * rax_val * sf.cyy() - 28 * rby_val * ray_val * sf.cxy() + 4 * sf.cxx() * sf.cxy() - 4 * sf.cxy() * sf.cxy()) / 40; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[6] = rt5 * (28 * rbx_val * rbz_val * ray_val + 4 * sf.czy() + 4 * rbz_val * sf.cxy()) / 40; // d/d(cxx)
        result.s1[9] = rt5 * 0; // d/d(cyx) - no terms
        result.s1[8] = rt5 * (14 * rbx_val * rbx_val * ray_val - 14 * rby_val * rby_val * ray_val + 4 * sf.cxy()) / 40; // d/d(czx)
        result.s1[7] = rt5 * (28 * rbx_val * rbz_val * rax_val + 4 * sf.czx() + 4 * rbz_val * sf.cxx()) / 40; // d/d(cxy)
        result.s1[10] = rt5 * (-28 * rby_val * rbz_val * rax_val - 4 * sf.czx() - 8 * rbz_val * sf.cxy()) / 40; // d/d(cyy)
        result.s1[11] = rt5 * (14 * rbx_val * rbx_val * rax_val - 14 * rby_val * rby_val * rax_val + 4 * sf.cxx() - 4 * sf.cyy()) / 40; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 225
            result.s2[1] = rt5 * (63 * rbx_val * rbx_val * rbz_val - 63 * rby_val * rby_val * rbz_val) / 20;
            result.s2[6] = rt5 * (126 * rbx_val * rbz_val * ray_val + 14 * rbx_val * sf.czy() + 14 * rbz_val * sf.cyx()) / 20;
            result.s2[7] = rt5 * (126 * rbx_val * rbz_val * rax_val + 14 * rbx_val * sf.czx() + 14 * rbz_val * sf.cxx()) / 20;
            result.s2[9] = rt5 * (126 * rax_val * ray_val * rbz_val + 14 * rax_val * sf.czy() + 14 * ray_val * sf.czx()) / 20;
            result.s2[10] = rt5 * (-126 * rby_val * rbz_val * ray_val - 14 * rby_val * sf.czy() - 14 * rbz_val * sf.cyy()) / 20;
            result.s2[11] = rt5 * (-126 * rby_val * rbz_val * rax_val - 14 * rby_val * sf.czx() - 14 * rbz_val * sf.cxy()) / 20;
            result.s2[14] = rt5 * (-126 * rax_val * ray_val * rbz_val - 14 * rax_val * sf.czy() - 14 * ray_val * sf.czx()) / 20;
            result.s2[15] = rt5 * (63 * rbx_val * rbx_val * ray_val - 63 * rby_val * rby_val * ray_val + 14 * rbx_val * sf.cyx() - 14 * rby_val * sf.cyy()) / 20;
            result.s2[16] = rt5 * (63 * rbx_val * rbx_val * rax_val - 63 * rby_val * rby_val * rax_val + 14 * rbx_val * sf.cxx() - 14 * rby_val * sf.cxy()) / 20;
            result.s2[18] = rt5 * (126 * rax_val * ray_val * rbx_val + 14 * rax_val * sf.cyx() + 14 * ray_val * sf.cxx()) / 20;
            result.s2[19] = rt5 * (-126 * rax_val * ray_val * rby_val - 14 * rax_val * sf.cyy() - 14 * ray_val * sf.cxy()) / 20;
            result.s2[22] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[24] = rt5 * (14 * rbz_val * ray_val + 2 * sf.czy()) / 20;
            result.s2[26] = rt5 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 20;
            result.s2[28] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[31] = rt5 * (14 * rbz_val * rax_val + 2 * sf.czx()) / 20;
            result.s2[33] = rt5 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 20;
            result.s2[34] = rt5*rbz_val/10;
            result.s2[46] = -7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[49] = rt5 * (-14 * rbz_val * ray_val - 2 * sf.czy()) / 20;
            result.s2[50] = rt5 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 20;
            result.s2[55] = -7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[59] = rt5 * (-14 * rbz_val * rax_val - 2 * sf.czx()) / 20;
            result.s2[60] = rt5 * (-14 * rax_val * rby_val - 2 * sf.cxy()) / 20;
            result.s2[64] = -rt5*rbz_val/10;
            result.s2[79] = rt5 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 20;
            result.s2[81] = rt5 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 20;
            result.s2[82] = rt5 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 20;
            result.s2[85] = rt5*rbx_val/10;
            result.s2[88] = -rt5*rby_val/10;
            result.s2[91] = rt5 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 20;
            result.s2[94] = rt5 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 20;
            result.s2[95] = rt5 * (-14 * rax_val * rby_val - 2 * sf.cxy()) / 20;
            result.s2[97] = rt5*rbx_val/10;
            result.s2[100] = -rt5*rby_val/10;
        }
    }
}

/**
 * Quadrupole-22s × Octopole-32s kernel
 * Orient case 226: Q22s × Q32s
 */
void quadrupole_22s_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt5 *
                (63 * rbx_val * rby_val * rbz_val * rax_val * ray_val +
                 7 * rbx_val * rby_val * rax_val * sf.czy() +
                 7 * rbx_val * rby_val * ray_val * sf.czx() +
                 14 * rbx_val * rbz_val * rax_val * sf.cyy() +
                 14 * rbx_val * rbz_val * ray_val * sf.cxy() +
                 14 * rby_val * rbz_val * rax_val * sf.cxy() +
                 14 * rby_val * rbz_val * ray_val * sf.cxx() +
                 2 * rbx_val * sf.cxy() * sf.czy() + 2 * rbx_val * sf.cyy() * sf.czx() +
                 2 * rby_val * sf.cxx() * sf.czy() + 2 * rby_val * sf.cxy() * sf.czx() +
                 2 * rbz_val * sf.cxx() * sf.cyy() + 2 * rbz_val * sf.cxy() * sf.cxy()) /
                20;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt5 * (63 * rbx_val * rby_val * rbz_val * ray_val + 7 * rbx_val * rby_val * sf.czy() + 14 * rbx_val * rbz_val * sf.cyy() + 14 * rby_val * rbz_val * sf.cxy()) / 20; // d/d(rax)
        result.s1[1] = rt5 * (63 * rbx_val * rby_val * rbz_val * rax_val + 7 * rbx_val * rby_val * sf.czx() + 14 * rbx_val * rbz_val * sf.cxy() + 14 * rby_val * rbz_val * sf.cxx()) / 20; // d/d(ray)
        result.s1[3] = rt5 * (63 * rby_val * rbz_val * rax_val * ray_val + 7 * rby_val * rax_val * sf.czy() + 7 * rby_val * ray_val * sf.czx() + 14 * rbz_val * rax_val * sf.cyy() + 14 * rbz_val * ray_val * sf.cxy() + 2 * sf.cxy() * sf.czy() + 2 * sf.cyy() * sf.czx() + 2 * rbz_val * sf.cxy() * sf.cxy()) / 20; // d/d(rbx)
        result.s1[4] = rt5 * (63 * rbx_val * rbz_val * rax_val * ray_val + 7 * rbx_val * rax_val * sf.czy() + 7 * rbx_val * ray_val * sf.czx() + 14 * rbz_val * rax_val * sf.cxy() + 14 * rbz_val * ray_val * sf.cxx() + 2 * sf.cxx() * sf.czy() + 2 * sf.cxy() * sf.czx() + 2 * rbz_val * sf.cxx() * sf.cyy()) / 20; // d/d(rby)
        result.s1[5] = rt5 * (63 * rbx_val * rby_val * rax_val * ray_val + 14 * rbx_val * rax_val * sf.cyy() + 14 * rbx_val * ray_val * sf.cxy() + 14 * rby_val * rax_val * sf.cxy() + 14 * rby_val * ray_val * sf.cxx() + 2 * sf.cxx() * sf.cyy() + 4 * rbz_val * sf.cxy() * sf.cxy()) / 20; // d/d(rbz)

        // Orientation matrix derivatives
        result.s1[6] = rt5 * (14 * rby_val * rbz_val * ray_val + 2 * sf.czy() + 2 * rbz_val * sf.cyy()) / 20; // d/d(cxx)
        result.s1[8] = rt5 * (7 * rbx_val * rby_val * ray_val + 7 * rby_val * ray_val + 2 * sf.cxy()) / 20; // d/d(czx)
        result.s1[7] = rt5 * (14 * rbx_val * rbz_val * ray_val + 14 * rby_val * rbz_val * rax_val + 2 * sf.czx() + 4 * rbz_val * sf.cxy()) / 20; // d/d(cxy)
        result.s1[10] = rt5 * (14 * rbx_val * rbz_val * rax_val + 2 * sf.czx() + 2 * rbz_val * sf.cxx()) / 20; // d/d(cyy)
        result.s1[11] = rt5 * (7 * rbx_val * rby_val * rax_val + 7 * rby_val * rax_val + 2 * sf.cxx() + 2 * sf.cyy()) / 20; // d/d(czy)

        if (level >= 2) {
            // Second derivatives - Orient case 226
            result.s2[1] = 63.0/10.0*rt5*rbx_val*rby_val*rbz_val;
            result.s2[6] = rt5*(63*rby_val*rbz_val*ray_val+7*rby_val*sf.cyz()+7*rbz_val*sf.cyy())/10;
            result.s2[7] = rt5*(63*rby_val*rbz_val*rax_val+7*rby_val*sf.cxz()+7*rbz_val*sf.cxy())/10;
            result.s2[10] = rt5*(63*rbx_val*rbz_val*ray_val+7*rbx_val*sf.cyz()+7*rbz_val*sf.cyx())/10;
            result.s2[11] = rt5*(63*rbx_val*rbz_val*rax_val+7*rbx_val*sf.cxz()+7*rbz_val*sf.cxx())/10;
            result.s2[13] = rt5*(63*rax_val*ray_val*rbz_val+7*rax_val*sf.cyz()+7*ray_val*sf.cxz())/10;
            result.s2[15] = rt5*(63*rbx_val*rby_val*ray_val+7*rbx_val*sf.cyy()+7*rby_val*sf.cyx())/10;
            result.s2[16] = rt5*(63*rbx_val*rby_val*rax_val+7*rbx_val*sf.cxy()+7*rby_val*sf.cxx())/10;
            result.s2[18] = rt5*(63*rax_val*ray_val*rby_val+7*rax_val*sf.cyy()+7*ray_val*sf.cxy())/10;
            result.s2[19] = rt5*(63*rax_val*ray_val*rbx_val+7*rax_val*sf.cyx()+7*ray_val*sf.cxx())/10;
            result.s2[22] = 7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[25] = rt5*(7*rbz_val*ray_val+sf.cyz())/10;
            result.s2[26] = rt5*(7*ray_val*rby_val+sf.cyy())/10;
            result.s2[28] = 7.0/10.0*rt5*rby_val*rbz_val;
            result.s2[32] = rt5*(7*rbz_val*rax_val+sf.cxz())/10;
            result.s2[33] = rt5*(7*rax_val*rby_val+sf.cxy())/10;
            result.s2[46] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[48] = rt5*(7*rbz_val*ray_val+sf.cyz())/10;
            result.s2[50] = rt5*(7*rbx_val*ray_val+sf.cyx())/10;
            result.s2[52] = rt5*rbz_val/10;
            result.s2[55] = 7.0/10.0*rt5*rbx_val*rbz_val;
            result.s2[58] = rt5*(7*rbz_val*rax_val+sf.cxz())/10;
            result.s2[60] = rt5*(7*rax_val*rbx_val+sf.cxx())/10;
            result.s2[61] = rt5*rbz_val/10;
            result.s2[79] = 7.0/10.0*rt5*rbx_val*rby_val;
            result.s2[81] = rt5*(7*ray_val*rby_val+sf.cyy())/10;
            result.s2[82] = rt5*(7*rbx_val*ray_val+sf.cyx())/10;
            result.s2[85] = rt5*rby_val/10;
            result.s2[88] = rt5*rbx_val/10;
            result.s2[91] = 7.0/10.0*rt5*rbx_val*rby_val;
            result.s2[94] = rt5*(7*rax_val*rby_val+sf.cxy())/10;
            result.s2[95] = rt5*(7*rax_val*rbx_val+sf.cxx())/10;
            result.s2[97] = rt5*rby_val/10;
            result.s2[100] = rt5*rbx_val/10;
        }
    }
}

/**
 * Quadrupole-22s × Octopole-33c kernel
 * Orient case 227: Q22s × Q33c
 */
void quadrupole_22s_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt30 *
                (21 * rbx_val * rbx_val * rbx_val * rax_val * ray_val -
                 63 * rbx_val * rby_val * rby_val * rax_val * ray_val +
                 14 * rbx_val * rbx_val * rax_val * sf.cxy() +
                 14 * rbx_val * rbx_val * ray_val * sf.cxx() -
                 28 * rbx_val * rby_val * rax_val * sf.cyy() -
                 28 * rbx_val * rby_val * ray_val * sf.cxy() -
                 14 * rby_val * rby_val * rax_val * sf.cxy() -
                 14 * rby_val * rby_val * ray_val * sf.cxx() +
                 4 * rbx_val * sf.cxx() * sf.cxy() - 4 * rbx_val * sf.cxy() * sf.cxy() -
                 4 * rby_val * sf.cxx() * sf.cyy() - 4 * rby_val * sf.cxy() * sf.cxy()) /
                80;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt30 * (21 * rbx_val * rbx_val * rbx_val * ray_val - 63 * rbx_val * rby_val * rby_val * ray_val + 14 * rbx_val * rbx_val * sf.cxy() - 28 * rbx_val * rby_val * sf.cyy() - 14 * rby_val * rby_val * sf.cxy()) / 80; // d/d(rax)
        result.s1[1] = rt30 * (21 * rbx_val * rbx_val * rbx_val * rax_val - 63 * rbx_val * rby_val * rby_val * rax_val + 14 * rbx_val * rbx_val * sf.cxx() - 28 * rbx_val * rby_val * sf.cxy() - 14 * rby_val * rby_val * sf.cxx()) / 80; // d/d(ray)
        result.s1[3] = rt30 * (63 * rbx_val * rbx_val * rax_val * ray_val - 63 * rby_val * rby_val * rax_val * ray_val + 28 * rbx_val * rax_val * sf.cxy() + 28 * rbx_val * ray_val * sf.cxx() - 28 * rby_val * rax_val * sf.cyy() - 28 * rby_val * ray_val * sf.cxy() + 4 * sf.cxx() * sf.cxy() - 8 * rbx_val * sf.cxy() * sf.cxy()) / 80; // d/d(rbx)
        result.s1[4] = rt30 * (-126 * rbx_val * rby_val * rax_val * ray_val - 28 * rbx_val * rax_val * sf.cyy() - 28 * rbx_val * ray_val * sf.cxy() - 28 * rby_val * rax_val * sf.cxy() - 28 * rby_val * ray_val * sf.cxx() - 4 * sf.cxx() * sf.cyy() - 8 * rby_val * sf.cxy() * sf.cxy()) / 80; // d/d(rby)

        // Orientation matrix derivatives
        result.s1[6] = rt30 * (14 * rbx_val * rbx_val * ray_val - 14 * rby_val * rby_val * ray_val + 4 * sf.cxy()) / 80; // d/d(cxx)
        result.s1[7] = rt30 * (14 * rbx_val * rbx_val * rax_val - 14 * rby_val * rby_val * rax_val + 4 * sf.cxx() - 8 * sf.cxy()) / 80; // d/d(cxy)
        result.s1[10] = rt30 * (-28 * rbx_val * rby_val * rax_val - 4 * sf.cxx() - 8 * sf.cxy()) / 80; // d/d(cyy)

        if (level >= 2) {
            // Second derivatives - Orient case 227
            result.s2[1] = rt30 * (21 * rbx_val * rbx_val * rbx_val - 63 * rbx_val * rby_val * rby_val) / 40;
            result.s2[6] = rt30 * (63 * rbx_val * rbx_val * ray_val - 63 * rby_val * rby_val * ray_val + 14 * rbx_val * sf.cyx() - 14 * rby_val * sf.cyy()) / 40;
            result.s2[7] = rt30 * (63 * rbx_val * rbx_val * rax_val - 63 * rby_val * rby_val * rax_val + 14 * rbx_val * sf.cxx() - 14 * rby_val * sf.cxy()) / 40;
            result.s2[9] = rt30 * (126 * rax_val * ray_val * rbx_val + 14 * rax_val * sf.cyx() + 14 * ray_val * sf.cxx()) / 40;
            result.s2[10] = rt30 * (-126 * rbx_val * rby_val * ray_val - 14 * rbx_val * sf.cyy() - 14 * rby_val * sf.cyx()) / 40;
            result.s2[11] = rt30 * (-126 * rbx_val * rby_val * rax_val - 14 * rbx_val * sf.cxy() - 14 * rby_val * sf.cxx()) / 40;
            result.s2[13] = rt30 * (-126 * rax_val * ray_val * rby_val - 14 * rax_val * sf.cyy() - 14 * ray_val * sf.cxy()) / 40;
            result.s2[14] = rt30 * (-126 * rax_val * ray_val * rbx_val - 14 * rax_val * sf.cyx() - 14 * ray_val * sf.cxx()) / 40;
            result.s2[22] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[24] = rt30 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 40;
            result.s2[25] = rt30 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 40;
            result.s2[28] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[31] = rt30 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 40;
            result.s2[32] = rt30 * (-14 * rax_val * rby_val - 2 * sf.cxy()) / 40;
            result.s2[34] = rt30*rbx_val/20;
            result.s2[46] = -7.0/20.0*rt30*rbx_val*rby_val;
            result.s2[48] = rt30 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 40;
            result.s2[49] = rt30 * (-14 * rbx_val * ray_val - 2 * sf.cyx()) / 40;
            result.s2[52] = -rt30*rby_val/20;
            result.s2[55] = -7.0/20.0*rt30*rbx_val*rby_val;
            result.s2[58] = rt30 * (-14 * rax_val * rby_val - 2 * sf.cxy()) / 40;
            result.s2[59] = rt30 * (-14 * rax_val * rbx_val - 2 * sf.cxx()) / 40;
            result.s2[61] = -rt30*rby_val/20;
            result.s2[64] = -rt30*rbx_val/20;
        }
    }
}

/**
 * Quadrupole-22s × Octopole-33s kernel
 * Orient case 228: Q22s × Q33s
 */
void quadrupole_22s_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();

    result.s0 = rt30 *
                (63 * rbx_val * rbx_val * rby_val * rax_val * ray_val -
                 21 * rby_val * rby_val * rby_val * rax_val * ray_val +
                 14 * rbx_val * rbx_val * rax_val * sf.cyy() +
                 14 * rbx_val * rbx_val * ray_val * sf.cxy() +
                 28 * rbx_val * rby_val * rax_val * sf.cxy() +
                 28 * rbx_val * rby_val * ray_val * sf.cxx() -
                 14 * rby_val * rby_val * rax_val * sf.cyy() -
                 14 * rby_val * rby_val * ray_val * sf.cxy() +
                 4 * rbx_val * sf.cxx() * sf.cyy() + 4 * rbx_val * sf.cxy() * sf.cxy() +
                 4 * rby_val * sf.cxx() * sf.cxy() - 4 * rby_val * sf.cxy() * sf.cyy()) /
                80;

    if (level >= 1) {
        // First derivatives
        result.s1[0] = rt30 * (63 * rbx_val * rbx_val * rby_val * ray_val - 21 * rby_val * rby_val * rby_val * ray_val + 14 * rbx_val * rbx_val * sf.cyy() + 28 * rbx_val * rby_val * sf.cxy() - 14 * rby_val * rby_val * sf.cyy()) / 80; // d/d(rax)
        result.s1[1] = rt30 * (63 * rbx_val * rbx_val * rby_val * rax_val - 21 * rby_val * rby_val * rby_val * rax_val + 14 * rbx_val * rbx_val * sf.cxy() + 28 * rbx_val * rby_val * sf.cxx() - 14 * rby_val * rby_val * sf.cxy()) / 80; // d/d(ray)
        result.s1[3] = rt30 * (126 * rbx_val * rby_val * rax_val * ray_val + 28 * rbx_val * rax_val * sf.cyy() + 28 * rbx_val * ray_val * sf.cxy() + 28 * rby_val * rax_val * sf.cxy() + 28 * rby_val * ray_val * sf.cxx() + 4 * sf.cxx() * sf.cyy() + 8 * rbx_val * sf.cxy() * sf.cxy() + 4 * rby_val * sf.cxx() * sf.cxy()) / 80; // d/d(rbx)
        result.s1[4] = rt30 * (63 * rbx_val * rbx_val * rax_val * ray_val - 63 * rby_val * rby_val * rax_val * ray_val + 28 * rbx_val * rax_val * sf.cxy() + 28 * rbx_val * ray_val * sf.cxx() - 28 * rby_val * rax_val * sf.cyy() - 28 * rby_val * ray_val * sf.cxy() + 4 * sf.cxx() * sf.cxy() - 4 * sf.cxy() * sf.cyy()) / 80; // d/d(rby)

        // Orientation matrix derivatives
        result.s1[6] = rt30 * (28 * rbx_val * rby_val * ray_val + 4 * sf.cyy() + 4 * sf.cxy()) / 80; // d/d(cxx)
        result.s1[7] = rt30 * (14 * rbx_val * rbx_val * ray_val - 14 * rby_val * rby_val * ray_val + 8 * sf.cxy()) / 80; // d/d(cxy)
        result.s1[10] = rt30 * (14 * rbx_val * rbx_val * rax_val - 14 * rby_val * rby_val * rax_val + 4 * sf.cxx() - 4 * sf.cxy()) / 80; // d/d(cyy)

        if (level >= 2) {
            // Second derivatives - Orient case 228
            result.s2[1] = rt30 * (63 * rbx_val * rbx_val * rby_val - 21 * rby_val * rby_val * rby_val) / 40;
            result.s2[6] = rt30 * (126 * rbx_val * rby_val * ray_val + 14 * rbx_val * sf.cyy() + 14 * rby_val * sf.cyx()) / 40;
            result.s2[7] = rt30 * (126 * rbx_val * rby_val * rax_val + 14 * rbx_val * sf.cxy() + 14 * rby_val * sf.cxx()) / 40;
            result.s2[9] = rt30 * (126 * rax_val * ray_val * rby_val + 14 * rax_val * sf.cyy() + 14 * ray_val * sf.cxy()) / 40;
            result.s2[10] = rt30 * (63 * rbx_val * rbx_val * ray_val - 63 * rby_val * rby_val * ray_val + 14 * rbx_val * sf.cyx() - 14 * rby_val * sf.cyy()) / 40;
            result.s2[11] = rt30 * (63 * rbx_val * rbx_val * rax_val - 63 * rby_val * rby_val * rax_val + 14 * rbx_val * sf.cxx() - 14 * rby_val * sf.cxy()) / 40;
            result.s2[13] = rt30 * (126 * rax_val * ray_val * rbx_val + 14 * rax_val * sf.cyx() + 14 * ray_val * sf.cxx()) / 40;
            result.s2[14] = rt30 * (-126 * rax_val * ray_val * rby_val - 14 * rax_val * sf.cyy() - 14 * ray_val * sf.cxy()) / 40;
            result.s2[22] = 7.0/20.0*rt30*rbx_val*rby_val;
            result.s2[24] = rt30 * (14 * ray_val * rby_val + 2 * sf.cyy()) / 40;
            result.s2[25] = rt30 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 40;
            result.s2[28] = 7.0/20.0*rt30*rbx_val*rby_val;
            result.s2[31] = rt30 * (14 * rax_val * rby_val + 2 * sf.cxy()) / 40;
            result.s2[32] = rt30 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 40;
            result.s2[34] = rt30*rby_val/20;
            result.s2[46] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[48] = rt30 * (14 * rbx_val * ray_val + 2 * sf.cyx()) / 40;
            result.s2[49] = rt30 * (-14 * ray_val * rby_val - 2 * sf.cyy()) / 40;
            result.s2[52] = rt30*rbx_val/20;
            result.s2[55] = rt30 * (7 * rbx_val * rbx_val - 7 * rby_val * rby_val) / 40;
            result.s2[58] = rt30 * (14 * rax_val * rbx_val + 2 * sf.cxx()) / 40;
            result.s2[59] = rt30 * (-14 * rax_val * rby_val - 2 * sf.cxy()) / 40;
            result.s2[61] = rt30*rbx_val/20;
            result.s2[64] = -rt30*rby_val/20;
        }
    }
}

// ============================================================================
// OCTOPOLE-QUADRUPOLE KERNELS (Orient cases 229-263)
// Octopole @ A (uses rax, ray, raz), Quadrupole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Octopole-30 × Quadrupole-20 kernel
 * Orient case 229: Q30 × Q20
 */
void octopole_30_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double raz = sf.raz();
    double rbz = sf.rbz();
    double czz = sf.czz();

    result.s0 = 63.0/8.0*raz*raz*raz*rbz*rbz - 7.0/8.0*raz*raz*raz
              - 21.0/8.0*rbz*rbz*raz + 21.0/4.0*raz*raz*rbz*czz
              + 3.0/8.0*raz - 3.0/4.0*rbz*czz + 3.0/4.0*raz*czz*czz;

    if (level >= 1) {
        result.s1[2] = 189.0/8.0*raz*raz*rbz*rbz - 21.0/8.0*raz*raz
                     - 21.0/8.0*rbz*rbz + 21.0/2.0*raz*rbz*czz
                     + 3.0/8.0 + 3.0/4.0*czz*czz;
        result.s1[5] = 63.0/4.0*raz*raz*raz*rbz - 21.0/4.0*raz*rbz
                     + 21.0/4.0*raz*raz*czz - 3.0/4.0*czz;
        result.s1[14] = 21.0/4.0*raz*raz*rbz - 3.0/4.0*rbz + 3.0/2.0*raz*czz;

        if (level >= 2) {
            result.s2[5] = 189.0/4.0*rbz*rbz*raz - 21.0/4.0*raz + 21.0/2.0*rbz*czz;
            result.s2[17] = 189.0/4.0*raz*raz*rbz - 21.0/4.0*rbz + 21.0/2.0*raz*czz;
            result.s2[20] = 63.0/4.0*raz*raz*raz - 21.0/4.0*raz;
            result.s2[107] = 21.0/2.0*raz*rbz + 3.0/2.0*czz;
            result.s2[110] = 21.0/4.0*raz*raz - 3.0/4.0;
            result.s2[119] = 3.0/2.0*raz;
        }
    }
}

/**
 * Octopole-30 × Quadrupole-21c kernel
 * Orient case 230: Q30 × Q21c
 */
void octopole_30_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rbz = sf.rbz();
    double czx = sf.czx();
    double czz = sf.czz();

    result.s0 = rt3*(21*raz*raz*raz*rbx*rbz - 7*rbx*rbz*raz
                    + 7*raz*raz*rbx*czz + 7*raz*raz*rbz*czx
                    - rbx*czz - rbz*czx + 2*raz*czx*czz)/4;

    if (level >= 1) {
        result.s1[2] = rt3*(63*raz*raz*rbx*rbz - 7*rbx*rbz
                          + 14*raz*rbx*czz + 14*raz*rbz*czx + 2*czx*czz)/4;
        result.s1[3] = rt3*(21*raz*raz*raz*rbz - 7*raz*rbz + 7*raz*raz*czz - czz)/4;
        result.s1[5] = rt3*(21*raz*raz*raz*rbx - 7*raz*rbx + 7*raz*raz*czx - czx)/4;
        result.s1[12] = rt3*(7*raz*raz*rbz - rbz + 2*raz*czz)/4;
        result.s1[14] = rt3*(7*raz*raz*rbx - rbx + 2*raz*czx)/4;

        if (level >= 2) {
            result.s2[5] = rt3*(126*rbx*rbz*raz + 14*rbx*czz + 14*rbz*czx)/4;
            result.s2[8] = rt3*(63*raz*raz*rbz - 7*rbz + 14*raz*czz)/4;
            result.s2[17] = rt3*(63*raz*raz*rbx - 7*rbx + 14*raz*czx)/4;
            result.s2[18] = rt3*(21*raz*raz*raz - 7*raz)/4;
            result.s2[38] = rt3*(14*raz*rbz + 2*czz)/4;
            result.s2[41] = rt3*(7*raz*raz - 1)/4;
            result.s2[107] = rt3*(14*raz*rbx + 2*czx)/4;
            result.s2[108] = rt3*(7*raz*raz - 1)/4;
            result.s2[113] = rt3*raz/2;
        }
    }
}

/**
 * Octopole-30 × Quadrupole-21s kernel
 * Orient case 231: Q30 × Q21s
 */
void octopole_30_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double raz = sf.raz();
    double rby = sf.rby();
    double rbz = sf.rbz();
    double czy = sf.czy();
    double czz = sf.czz();

    result.s0 = rt3*(21*raz*raz*raz*rby*rbz - 7*rby*rbz*raz
                    + 7*raz*raz*rby*czz + 7*raz*raz*rbz*czy
                    - rby*czz - rbz*czy + 2*raz*czy*czz)/4;

    if (level >= 1) {
        result.s1[2] = rt3*(63*raz*raz*rby*rbz - 7*rby*rbz
                          + 14*raz*rby*czz + 14*raz*rbz*czy + 2*czy*czz)/4;
        result.s1[4] = rt3*(21*raz*raz*raz*rbz - 7*raz*rbz + 7*raz*raz*czz - czz)/4;
        result.s1[5] = rt3*(21*raz*raz*raz*rby - 7*raz*rby + 7*raz*raz*czy - czy)/4;
        result.s1[13] = rt3*(7*raz*raz*rbz - rbz + 2*raz*czz)/4;
        result.s1[14] = rt3*(7*raz*raz*rby - rby + 2*raz*czy)/4;

        if (level >= 2) {
            result.s2[5] = rt3*(126*rby*rbz*raz + 14*rby*czz + 14*rbz*czy)/4;
            result.s2[12] = rt3*(63*raz*raz*rbz - 7*rbz + 14*raz*czz)/4;
            result.s2[17] = rt3*(63*raz*raz*rby - 7*rby + 14*raz*czy)/4;
            result.s2[19] = rt3*(21*raz*raz*raz - 7*raz)/4;
            result.s2[68] = rt3*(14*raz*rbz + 2*czz)/4;
            result.s2[71] = rt3*(7*raz*raz - 1)/4;
            result.s2[107] = rt3*(14*raz*rby + 2*czy)/4;
            result.s2[109] = rt3*(7*raz*raz - 1)/4;
            result.s2[116] = rt3*raz/2;
        }
    }
}

/**
 * Octopole-30 × Quadrupole-22c kernel
 * Orient case 232: Q30 × Q22c
 */
void octopole_30_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double czx = sf.czx();
    double czy = sf.czy();

    result.s0 = rt3*(21*raz*raz*raz*rbx*rbx - 21*raz*raz*raz*rby*rby
                    - 7*rbx*rbx*raz + 7*rby*rby*raz
                    + 14*raz*raz*rbx*czx - 14*raz*raz*rby*czy
                    - 2*rbx*czx + 2*rby*czy
                    + 2*raz*czx*czx - 2*raz*czy*czy)/8;

    if (level >= 1) {
        result.s1[2] = rt3*(63*raz*raz*rbx*rbx - 63*raz*raz*rby*rby
                          - 7*rbx*rbx + 7*rby*rby
                          + 28*raz*rbx*czx - 28*raz*rby*czy
                          + 2*czx*czx - 2*czy*czy)/8;
        result.s1[3] = rt3*(42*raz*raz*raz*rbx - 14*raz*rbx + 14*raz*raz*czx - 2*czx)/8;
        result.s1[4] = rt3*(-42*raz*raz*raz*rby + 14*raz*rby - 14*raz*raz*czy + 2*czy)/8;
        result.s1[12] = rt3*(14*raz*raz*rbx - 2*rbx + 4*raz*czx)/8;
        result.s1[13] = rt3*(-14*raz*raz*rby + 2*rby - 4*raz*czy)/8;

        if (level >= 2) {
            result.s2[5] = rt3*(126*rbx*rbx*raz - 126*rby*rby*raz + 28*rbx*czx - 28*rby*czy)/8;
            result.s2[8] = rt3*(126*raz*raz*rbx - 14*rbx + 28*raz*czx)/8;
            result.s2[9] = rt3*(42*raz*raz*raz - 14*raz)/8;
            result.s2[12] = rt3*(-126*raz*raz*rby + 14*rby - 28*raz*czy)/8;
            result.s2[14] = rt3*(-42*raz*raz*raz + 14*raz)/8;
            result.s2[38] = rt3*(28*raz*rbx + 4*czx)/8;
            result.s2[39] = rt3*(14*raz*raz - 2)/8;
            result.s2[44] = rt3*raz/2;
            result.s2[68] = rt3*(-28*raz*rby - 4*czy)/8;
            result.s2[70] = rt3*(-14*raz*raz + 2)/8;
            result.s2[77] = -rt3*raz/2;
        }
    }
}

/**
 * Octopole-30 × Quadrupole-22s kernel
 * Orient case 233: Q30 × Q22s
 */
void octopole_30_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double czx = sf.czx();
    double czy = sf.czy();

    result.s0 = rt3*(21*raz*raz*raz*rbx*rby - 7*rbx*rby*raz
                    + 7*raz*raz*rbx*czy + 7*raz*raz*rby*czx
                    - rbx*czy - rby*czx + 2*raz*czx*czy)/4;

    if (level >= 1) {
        result.s1[2] = rt3*(63*raz*raz*rbx*rby - 7*rbx*rby
                          + 14*raz*rbx*czy + 14*raz*rby*czx + 2*czx*czy)/4;
        result.s1[3] = rt3*(21*raz*raz*raz*rby - 7*raz*rby + 7*raz*raz*czy - czy)/4;
        result.s1[4] = rt3*(21*raz*raz*raz*rbx - 7*raz*rbx + 7*raz*raz*czx - czx)/4;
        result.s1[12] = rt3*(7*raz*raz*rby - rby + 2*raz*czy)/4;
        result.s1[13] = rt3*(7*raz*raz*rbx - rbx + 2*raz*czx)/4;

        if (level >= 2) {
            result.s2[5] = rt3*(126*rbx*rby*raz + 14*rbx*czy + 14*rby*czx)/4;
            result.s2[8] = rt3*(63*raz*raz*rby - 7*rby + 14*raz*czy)/4;
            result.s2[12] = rt3*(63*raz*raz*rbx - 7*rbx + 14*raz*czx)/4;
            result.s2[13] = rt3*(21*raz*raz*raz - 7*raz)/4;
            result.s2[38] = rt3*(14*raz*rby + 2*czy)/4;
            result.s2[40] = rt3*(7*raz*raz - 1)/4;
            result.s2[68] = rt3*(14*raz*rbx + 2*czx)/4;
            result.s2[69] = rt3*(7*raz*raz - 1)/4;
            result.s2[74] = rt3*raz/2;
        }
    }
}

// ============================================================================
// BATCH 2: Q31c × Q2* (cases 234-238)
// ============================================================================

/**
 * Octopole-31c × Quadrupole-20 kernel
 * Orient case 234: Q31c × Q20
 */
void octopole_31c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double raz = sf.raz();
    double rbz = sf.rbz();
    double czz = sf.czz();
    double cxz = sf.cxz();

    result.s0 = rt6*(63*rax*raz*raz*rbz*rbz - 7*rax*raz*raz
                   - 7*rbz*rbz*rax + 28*rax*raz*rbz*czz + 14*raz*raz*rbz*cxz + rax
                   - 2*rbz*cxz + 2*rax*czz*czz + 4*raz*cxz*czz)/16;

    if (level >= 1) {
        result.s1[0] = rt6*(63*raz*raz*rbz*rbz - 7*raz*raz - 7*rbz*rbz
                          + 28*raz*rbz*czz + 1 + 2*czz*czz)/16;
        result.s1[2] = rt6*(126*rbz*rbz*rax*raz - 14*rax*raz
                          + 28*rbz*rax*czz + 28*rbz*raz*cxz + 4*cxz*czz)/16;
        result.s1[5] = rt6*(126*rax*raz*raz*rbz - 14*rbz*rax
                          + 28*rax*raz*czz + 14*raz*raz*cxz - 2*cxz)/16;
        result.s1[8] = rt6*(14*raz*raz*rbz - 2*rbz + 4*raz*czz)/16;
        result.s1[14] = rt6*(28*rax*raz*rbz + 4*rax*czz + 4*raz*cxz)/16;

        if (level >= 2) {
            result.s2[3] = rt6*(126*rbz*rbz*raz - 14*raz + 28*rbz*czz)/16;
            result.s2[5] = rt6*(126*rbz*rbz*rax - 14*rax + 28*rbz*cxz)/16;
            result.s2[15] = rt6*(126*raz*raz*rbz - 14*rbz + 28*raz*czz)/16;
            result.s2[17] = rt6*(252*rax*raz*rbz + 28*rax*czz + 28*raz*cxz)/16;
            result.s2[20] = rt6*(126*rax*raz*raz - 14*rax)/16;
            result.s2[80] = rt6*(28*raz*rbz + 4*czz)/16;
            result.s2[83] = rt6*(14*raz*raz - 2)/16;
            result.s2[105] = rt6*(28*raz*rbz + 4*czz)/16;
            result.s2[107] = rt6*(28*rbz*rax + 4*cxz)/16;
            result.s2[110] = 7.0/4.0*rt6*rax*raz;
            result.s2[117] = rt6*raz/4;
            result.s2[119] = rt6*rax/4;
        }
    }
}

/**
 * Octopole-31c × Quadrupole-21c kernel
 * Orient case 235: Q31c × Q21c
 */
void octopole_31c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rbz = sf.rbz();
    double czx = sf.czx();
    double czz = sf.czz();
    double cxz = sf.cxz();
    double cxx = sf.cxx();

    result.s0 = rt2*(63*rax*raz*raz*rbx*rbz - 7*rbx*rbz*rax
                   + 14*rax*raz*rbx*czz + 14*rax*raz*rbz*czx + 7*raz*raz*rbx*cxz
                   + 7*raz*raz*rbz*cxx - rbx*cxz - rbz*cxx + 2*rax*czx*czz + 2*raz*cxx*czz
                   + 2*raz*cxz*czx)/8;

    if (level >= 1) {
        result.s1[0] = rt2*(63*raz*raz*rbx*rbz - 7*rbx*rbz
                          + 14*raz*rbx*czz + 14*raz*rbz*czx + 2*czx*czz)/8;
        result.s1[2] = rt2*(126*rax*raz*rbx*rbz + 14*rax*rbx*czz
                          + 14*rax*rbz*czx + 14*raz*rbx*cxz + 14*raz*rbz*cxx + 2*cxx*czz
                          + 2*cxz*czx)/8;
        result.s1[3] = rt2*(63*rax*raz*raz*rbz - 7*rbz*rax
                          + 14*rax*raz*czz + 7*raz*raz*cxz - cxz)/8;
        result.s1[5] = rt2*(63*rax*raz*raz*rbx - 7*rax*rbx
                          + 14*rax*raz*czx + 7*raz*raz*cxx - cxx)/8;
        result.s1[6] = rt2*(7*raz*raz*rbz - rbz + 2*raz*czz)/8;
        result.s1[12] = rt2*(14*rax*raz*rbz + 2*rax*czz + 2*raz*cxz)/8;
        result.s1[8] = rt2*(7*raz*raz*rbx - rbx + 2*raz*czx)/8;
        result.s1[14] = rt2*(14*rax*raz*rbx + 2*rax*czx + 2*raz*cxx)/8;

        if (level >= 2) {
            result.s2[3] = rt2*(126*rbx*rbz*raz + 14*rbx*czz + 14*rbz*czx)/8;
            result.s2[5] = rt2*(126*rbx*rbz*rax + 14*rbx*cxz + 14*rbz*cxx)/8;
            result.s2[6] = rt2*(63*raz*raz*rbz - 7*rbz + 14*raz*czz)/8;
            result.s2[8] = rt2*(126*rax*raz*rbz + 14*rax*czz + 14*raz*cxz)/8;
            result.s2[15] = rt2*(63*raz*raz*rbx - 7*rbx + 14*raz*czx)/8;
            result.s2[17] = rt2*(126*rax*raz*rbx + 14*rax*czx + 14*raz*cxx)/8;
            result.s2[18] = rt2*(63*rax*raz*raz - 7*rax)/8;
            result.s2[23] = rt2*(14*raz*rbz + 2*czz)/8;
            result.s2[26] = rt2*(7*raz*raz - 1)/8;
            result.s2[36] = rt2*(14*raz*rbz + 2*czz)/8;
            result.s2[38] = rt2*(14*rbz*rax + 2*cxz)/8;
            result.s2[41] = 7.0/4.0*rt2*rax*raz;
            result.s2[80] = rt2*(14*raz*rbx + 2*czx)/8;
            result.s2[81] = rt2*(7*raz*raz - 1)/8;
            result.s2[86] = rt2*raz/4;
            result.s2[105] = rt2*(14*raz*rbx + 2*czx)/8;
            result.s2[107] = rt2*(14*rax*rbx + 2*cxx)/8;
            result.s2[108] = 7.0/4.0*rt2*rax*raz;
            result.s2[111] = rt2*raz/4;
            result.s2[113] = rt2*rax/4;
        }
    }
}

/**
 * Octopole-31c × Quadrupole-21s kernel
 * Orient case 236: Q31c × Q21s
 */
void octopole_31c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double raz = sf.raz();
    double rby = sf.rby();
    double rbz = sf.rbz();
    double czy = sf.czy();
    double czz = sf.czz();
    double cxz = sf.cxz();
    double cxy = sf.cxy();

    result.s0 = rt2*(63*rax*raz*raz*rby*rbz - 7*rby*rbz*rax
                   + 14*rax*raz*rby*czz + 14*rax*raz*rbz*czy + 7*raz*raz*rby*cxz
                   + 7*raz*raz*rbz*cxy - rby*cxz - rbz*cxy + 2*rax*czy*czz + 2*raz*cxy*czz
                   + 2*raz*cxz*czy)/8;

    if (level >= 1) {
        result.s1[0] = rt2*(63*raz*raz*rby*rbz - 7*rby*rbz
                          + 14*raz*rby*czz + 14*raz*rbz*czy + 2*czy*czz)/8;
        result.s1[2] = rt2*(126*rax*raz*rby*rbz + 14*rax*rby*czz
                          + 14*rax*rbz*czy + 14*raz*rby*cxz + 14*raz*rbz*cxy + 2*cxy*czz
                          + 2*cxz*czy)/8;
        result.s1[4] = rt2*(63*rax*raz*raz*rbz - 7*rbz*rax
                          + 14*rax*raz*czz + 7*raz*raz*cxz - cxz)/8;
        result.s1[5] = rt2*(63*rax*raz*raz*rby - 7*rax*rby
                          + 14*rax*raz*czy + 7*raz*raz*cxy - cxy)/8;
        result.s1[7] = rt2*(7*raz*raz*rbz - rbz + 2*raz*czz)/8;
        result.s1[13] = rt2*(14*rax*raz*rbz + 2*rax*czz + 2*raz*cxz)/8;
        result.s1[8] = rt2*(7*raz*raz*rby - rby + 2*raz*czy)/8;
        result.s1[14] = rt2*(14*rax*raz*rby + 2*rax*czy + 2*raz*cxy)/8;

        if (level >= 2) {
            result.s2[3] = rt2*(126*rby*rbz*raz + 14*rby*czz + 14*rbz*czy)/8;
            result.s2[5] = rt2*(126*rby*rbz*rax + 14*rby*cxz + 14*rbz*cxy)/8;
            result.s2[10] = rt2*(63*raz*raz*rbz - 7*rbz + 14*raz*czz)/8;
            result.s2[12] = rt2*(126*rax*raz*rbz + 14*rax*czz + 14*raz*cxz)/8;
            result.s2[15] = rt2*(63*raz*raz*rby - 7*rby + 14*raz*czy)/8;
            result.s2[17] = rt2*(126*rax*raz*rby + 14*rax*czy + 14*raz*cxy)/8;
            result.s2[19] = rt2*(63*rax*raz*raz - 7*rax)/8;
            result.s2[47] = rt2*(14*raz*rbz + 2*czz)/8;
            result.s2[50] = rt2*(7*raz*raz - 1)/8;
            result.s2[66] = rt2*(14*raz*rbz + 2*czz)/8;
            result.s2[68] = rt2*(14*rbz*rax + 2*cxz)/8;
            result.s2[71] = 7.0/4.0*rt2*rax*raz;
            result.s2[80] = rt2*(14*raz*rby + 2*czy)/8;
            result.s2[82] = rt2*(7*raz*raz - 1)/8;
            result.s2[89] = rt2*raz/4;
            result.s2[105] = rt2*(14*raz*rby + 2*czy)/8;
            result.s2[107] = rt2*(14*rax*rby + 2*cxy)/8;
            result.s2[109] = 7.0/4.0*rt2*rax*raz;
            result.s2[114] = rt2*raz/4;
            result.s2[116] = rt2*rax/4;
        }
    }
}

/**
 * Octopole-31c × Quadrupole-22c kernel
 * Orient case 237: Q31c × Q22c
 */
void octopole_31c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double czx = sf.czx();
    double czy = sf.czy();
    double cxx = sf.cxx();
    double cxy = sf.cxy();

    result.s0 = rt2*(63*rax*raz*raz*rbx*rbx - 63*rax*raz*raz*rby*rby
                   - 7*rbx*rbx*rax + 7*rby*rby*rax + 28*rax*raz*rbx*czx
                   - 28*rax*raz*rby*czy + 14*raz*raz*rbx*cxx - 14*raz*raz*rby*cxy
                   - 2*rbx*cxx + 2*rby*cxy + 2*rax*czx*czx - 2*rax*czy*czy + 4*raz*cxx*czx
                   - 4*raz*cxy*czy)/16;

    if (level >= 1) {
        result.s1[0] = rt2*(63*raz*raz*rbx*rbx - 63*raz*raz*rby*rby
                          - 7*rbx*rbx + 7*rby*rby + 28*raz*rbx*czx - 28*raz*rby*czy + 2*czx*czx
                          - 2*czy*czy)/16;
        result.s1[2] = rt2*(126*rax*raz*rbx*rbx - 126*rax*raz*rby*rby
                          + 28*rax*rbx*czx - 28*rax*rby*czy + 28*raz*rbx*cxx - 28*raz*rby*cxy
                          + 4*cxx*czx - 4*cxy*czy)/16;
        result.s1[3] = rt2*(126*rax*raz*raz*rbx - 14*rax*rbx
                          + 28*rax*raz*czx + 14*raz*raz*cxx - 2*cxx)/16;
        result.s1[4] = rt2*(-126*rax*raz*raz*rby + 14*rax*rby
                          - 28*rax*raz*czy - 14*raz*raz*cxy + 2*cxy)/16;
        result.s1[6] = rt2*(14*raz*raz*rbx - 2*rbx + 4*raz*czx)/16;
        result.s1[12] = rt2*(28*rax*raz*rbx + 4*rax*czx + 4*raz*cxx)/16;
        result.s1[7] = rt2*(-14*raz*raz*rby + 2*rby - 4*raz*czy)/16;
        result.s1[13] = rt2*(-28*rax*raz*rby - 4*rax*czy - 4*raz*cxy)/16;

        if (level >= 2) {
            result.s2[3] = rt2*(126*rbx*rbx*raz - 126*rby*rby*raz + 28*rbx*czx
                              - 28*rby*czy)/16;
            result.s2[5] = rt2*(126*rbx*rbx*rax - 126*rby*rby*rax + 28*rbx*cxx
                              - 28*rby*cxy)/16;
            result.s2[6] = rt2*(126*raz*raz*rbx - 14*rbx + 28*raz*czx)/16;
            result.s2[8] = rt2*(252*rax*raz*rbx + 28*rax*czx + 28*raz*cxx)/16;
            result.s2[9] = rt2*(126*rax*raz*raz - 14*rax)/16;
            result.s2[10] = rt2*(-126*raz*raz*rby + 14*rby - 28*raz*czy)/16;
            result.s2[12] = rt2*(-252*rax*raz*rby - 28*rax*czy - 28*raz*cxy)/16;
            result.s2[14] = rt2*(-126*rax*raz*raz + 14*rax)/16;
            result.s2[23] = rt2*(28*raz*rbx + 4*czx)/16;
            result.s2[24] = rt2*(14*raz*raz - 2)/16;
            result.s2[36] = rt2*(28*raz*rbx + 4*czx)/16;
            result.s2[38] = rt2*(28*rax*rbx + 4*cxx)/16;
            result.s2[39] = 7.0/4.0*rt2*rax*raz;
            result.s2[42] = rt2*raz/4;
            result.s2[44] = rt2*rax/4;
            result.s2[47] = rt2*(-28*raz*rby - 4*czy)/16;
            result.s2[49] = rt2*(-14*raz*raz + 2)/16;
            result.s2[66] = rt2*(-28*raz*rby - 4*czy)/16;
            result.s2[68] = rt2*(-28*rax*rby - 4*cxy)/16;
            result.s2[70] = -7.0/4.0*rt2*rax*raz;
            result.s2[75] = -rt2*raz/4;
            result.s2[77] = -rt2*rax/4;
        }
    }
}

/**
 * Octopole-31c × Quadrupole-22s kernel
 * Orient case 238: Q31c × Q22s
 */
void octopole_31c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double czx = sf.czx();
    double czy = sf.czy();
    double cxx = sf.cxx();
    double cxy = sf.cxy();

    result.s0 = rt2*(63*rax*raz*raz*rbx*rby - 7*rbx*rby*rax
                   + 14*rax*raz*rbx*czy + 14*rax*raz*rby*czx + 7*raz*raz*rbx*cxy
                   + 7*raz*raz*rby*cxx - rbx*cxy - rby*cxx + 2*rax*czx*czy + 2*raz*cxx*czy
                   + 2*raz*cxy*czx)/8;

    if (level >= 1) {
        result.s1[0] = rt2*(63*raz*raz*rbx*rby - 7*rbx*rby
                          + 14*raz*rbx*czy + 14*raz*rby*czx + 2*czx*czy)/8;
        result.s1[2] = rt2*(126*rax*raz*rbx*rby + 14*rax*rbx*czy
                          + 14*rax*rby*czx + 14*raz*rbx*cxy + 14*raz*rby*cxx + 2*cxx*czy
                          + 2*cxy*czx)/8;
        result.s1[3] = rt2*(63*rax*raz*raz*rby - 7*rax*rby
                          + 14*rax*raz*czy + 7*raz*raz*cxy - cxy)/8;
        result.s1[4] = rt2*(63*rax*raz*raz*rbx - 7*rax*rbx
                          + 14*rax*raz*czx + 7*raz*raz*cxx - cxx)/8;
        result.s1[6] = rt2*(7*raz*raz*rby - rby + 2*raz*czy)/8;
        result.s1[12] = rt2*(14*rax*raz*rby + 2*rax*czy + 2*raz*cxy)/8;
        result.s1[7] = rt2*(7*raz*raz*rbx - rbx + 2*raz*czx)/8;
        result.s1[13] = rt2*(14*rax*raz*rbx + 2*rax*czx + 2*raz*cxx)/8;

        if (level >= 2) {
            result.s2[3] = rt2*(126*rbx*rby*raz + 14*rbx*czy + 14*rby*czx)/8;
            result.s2[5] = rt2*(126*rbx*rby*rax + 14*rbx*cxy + 14*rby*cxx)/8;
            result.s2[6] = rt2*(63*raz*raz*rby - 7*rby + 14*raz*czy)/8;
            result.s2[8] = rt2*(126*rax*raz*rby + 14*rax*czy + 14*raz*cxy)/8;
            result.s2[10] = rt2*(63*raz*raz*rbx - 7*rbx + 14*raz*czx)/8;
            result.s2[12] = rt2*(126*rax*raz*rbx + 14*rax*czx + 14*raz*cxx)/8;
            result.s2[13] = rt2*(63*rax*raz*raz - 7*rax)/8;
            result.s2[23] = rt2*(14*raz*rby + 2*czy)/8;
            result.s2[25] = rt2*(7*raz*raz - 1)/8;
            result.s2[36] = rt2*(14*raz*rby + 2*czy)/8;
            result.s2[38] = rt2*(14*rax*rby + 2*cxy)/8;
            result.s2[40] = 7.0/4.0*rt2*rax*raz;
            result.s2[47] = rt2*(14*raz*rbx + 2*czx)/8;
            result.s2[48] = rt2*(7*raz*raz - 1)/8;
            result.s2[53] = rt2*raz/4;
            result.s2[66] = rt2*(14*raz*rbx + 2*czx)/8;
            result.s2[68] = rt2*(14*rax*rbx + 2*cxx)/8;
            result.s2[69] = 7.0/4.0*rt2*rax*raz;
            result.s2[72] = rt2*raz/4;
            result.s2[74] = rt2*rax/4;
        }
    }
}

// ============================================================================
// BATCH 3: Q31s × Q2* (cases 239-243)
// ============================================================================

/**
 * Octopole-31s × Quadrupole-20 kernel
 * Orient case 239: Q31s × Q20
 */
void octopole_31s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double ray = sf.ray();
    double raz = sf.raz();
    double rbz = sf.rbz();
    double czz = sf.czz();
    double cyz = sf.cyz();

    result.s0 = rt6*(63*ray*raz*raz*rbz*rbz - 7*ray*raz*raz
                   - 7*rbz*rbz*ray + 28*ray*raz*rbz*czz + 14*raz*raz*rbz*cyz + ray
                   - 2*rbz*cyz + 2*ray*czz*czz + 4*raz*cyz*czz)/16;

    if (level >= 1) {
        result.s1[1] = rt6*(63*raz*raz*rbz*rbz - 7*raz*raz - 7*rbz*rbz
                          + 28*raz*rbz*czz + 1 + 2*czz*czz)/16;
        result.s1[2] = rt6*(126*rbz*rbz*ray*raz - 14*ray*raz
                          + 28*rbz*ray*czz + 28*rbz*raz*cyz + 4*cyz*czz)/16;
        result.s1[5] = rt6*(126*ray*raz*raz*rbz - 14*rbz*ray
                          + 28*ray*raz*czz + 14*raz*raz*cyz - 2*cyz)/16;
        result.s1[11] = rt6*(14*raz*raz*rbz - 2*rbz + 4*raz*czz)/16;
        result.s1[14] = rt6*(28*ray*raz*rbz + 4*ray*czz + 4*raz*cyz)/16;

        if (level >= 2) {
            result.s2[4] = rt6*(126*rbz*rbz*raz - 14*raz + 28*rbz*czz)/16;
            result.s2[5] = rt6*(126*rbz*rbz*ray - 14*ray + 28*rbz*cyz)/16;
            result.s2[16] = rt6*(126*raz*raz*rbz - 14*rbz + 28*raz*czz)/16;
            result.s2[17] = rt6*(252*ray*raz*rbz + 28*ray*czz + 28*raz*cyz)/16;
            result.s2[20] = rt6*(126*ray*raz*raz - 14*ray)/16;
            result.s2[93] = rt6*(28*raz*rbz + 4*czz)/16;
            result.s2[96] = rt6*(14*raz*raz - 2)/16;
            result.s2[106] = rt6*(28*raz*rbz + 4*czz)/16;
            result.s2[107] = rt6*(28*rbz*ray + 4*cyz)/16;
            result.s2[110] = 7.0/4.0*rt6*ray*raz;
            result.s2[118] = rt6*raz/4;
            result.s2[119] = rt6*ray/4;
        }
    }
}

/**
 * Octopole-31s × Quadrupole-21c kernel
 * Orient case 240: Q31s × Q21c
 */
void octopole_31s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double ray = sf.ray();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rbz = sf.rbz();
    double czx = sf.czx();
    double czz = sf.czz();
    double cyz = sf.cyz();
    double cyx = sf.cyx();

    result.s0 = rt2*(63*ray*raz*raz*rbx*rbz - 7*rbx*rbz*ray
                   + 14*ray*raz*rbx*czz + 14*ray*raz*rbz*czx + 7*raz*raz*rbx*cyz
                   + 7*raz*raz*rbz*cyx - rbx*cyz - rbz*cyx + 2*ray*czx*czz + 2*raz*cyx*czz
                   + 2*raz*cyz*czx)/8;

    if (level >= 1) {
        result.s1[1] = rt2*(63*raz*raz*rbx*rbz - 7*rbx*rbz
                          + 14*raz*rbx*czz + 14*raz*rbz*czx + 2*czx*czz)/8;
        result.s1[2] = rt2*(126*rbx*rbz*ray*raz + 14*rbx*ray*czz
                          + 14*rbz*ray*czx + 14*rbx*raz*cyz + 14*rbz*raz*cyx + 2*cyx*czz
                          + 2*czx*cyz)/8;
        result.s1[3] = rt2*(63*ray*raz*raz*rbz - 7*rbz*ray
                          + 14*ray*raz*czz + 7*raz*raz*cyz - cyz)/8;
        result.s1[5] = rt2*(63*ray*raz*raz*rbx - 7*rbx*ray
                          + 14*ray*raz*czx + 7*raz*raz*cyx - cyx)/8;
        result.s1[9] = rt2*(7*raz*raz*rbz - rbz + 2*raz*czz)/8;
        result.s1[12] = rt2*(14*ray*raz*rbz + 2*ray*czz + 2*raz*cyz)/8;
        result.s1[11] = rt2*(7*raz*raz*rbx - rbx + 2*raz*czx)/8;
        result.s1[14] = rt2*(14*ray*raz*rbx + 2*ray*czx + 2*raz*cyx)/8;

        if (level >= 2) {
            result.s2[4] = rt2*(126*rbx*rbz*raz + 14*rbx*czz + 14*rbz*czx)/8;
            result.s2[5] = rt2*(126*rbx*rbz*ray + 14*rbx*cyz + 14*rbz*cyx)/8;
            result.s2[7] = rt2*(63*raz*raz*rbz - 7*rbz + 14*raz*czz)/8;
            result.s2[8] = rt2*(126*ray*raz*rbz + 14*ray*czz + 14*raz*cyz)/8;
            result.s2[16] = rt2*(63*raz*raz*rbx - 7*rbx + 14*raz*czx)/8;
            result.s2[17] = rt2*(126*ray*raz*rbx + 14*ray*czx + 14*raz*cyx)/8;
            result.s2[18] = rt2*(63*ray*raz*raz - 7*ray)/8;
            result.s2[30] = rt2*(14*raz*rbz + 2*czz)/8;
            result.s2[33] = rt2*(7*raz*raz - 1)/8;
            result.s2[37] = rt2*(14*raz*rbz + 2*czz)/8;
            result.s2[38] = rt2*(14*rbz*ray + 2*cyz)/8;
            result.s2[41] = 7.0/4.0*rt2*ray*raz;
            result.s2[93] = rt2*(14*raz*rbx + 2*czx)/8;
            result.s2[94] = rt2*(7*raz*raz - 1)/8;
            result.s2[99] = rt2*raz/4;
            result.s2[106] = rt2*(14*raz*rbx + 2*czx)/8;
            result.s2[107] = rt2*(14*rbx*ray + 2*cyx)/8;
            result.s2[108] = 7.0/4.0*rt2*ray*raz;
            result.s2[112] = rt2*raz/4;
            result.s2[113] = rt2*ray/4;
        }
    }
}

/**
 * Octopole-31s × Quadrupole-21s kernel
 * Orient case 241: Q31s × Q21s
 */
void octopole_31s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double ray = sf.ray();
    double raz = sf.raz();
    double rby = sf.rby();
    double rbz = sf.rbz();
    double czy = sf.czy();
    double czz = sf.czz();
    double cyz = sf.cyz();
    double cyy = sf.cyy();

    result.s0 = rt2*(63*ray*raz*raz*rby*rbz - 7*rby*rbz*ray
                   + 14*ray*raz*rby*czz + 14*ray*raz*rbz*czy + 7*raz*raz*rby*cyz
                   + 7*raz*raz*rbz*cyy - rby*cyz - rbz*cyy + 2*ray*czy*czz + 2*raz*cyy*czz
                   + 2*raz*cyz*czy)/8;

    if (level >= 1) {
        result.s1[1] = rt2*(63*raz*raz*rby*rbz - 7*rby*rbz
                          + 14*raz*rby*czz + 14*raz*rbz*czy + 2*czy*czz)/8;
        result.s1[2] = rt2*(126*ray*raz*rby*rbz + 14*ray*rby*czz
                          + 14*ray*rbz*czy + 14*raz*rby*cyz + 14*raz*rbz*cyy + 2*cyy*czz
                          + 2*cyz*czy)/8;
        result.s1[4] = rt2*(63*ray*raz*raz*rbz - 7*rbz*ray
                          + 14*ray*raz*czz + 7*raz*raz*cyz - cyz)/8;
        result.s1[5] = rt2*(63*ray*raz*raz*rby - 7*ray*rby
                          + 14*ray*raz*czy + 7*raz*raz*cyy - cyy)/8;
        result.s1[10] = rt2*(7*raz*raz*rbz - rbz + 2*raz*czz)/8;
        result.s1[13] = rt2*(14*ray*raz*rbz + 2*ray*czz + 2*raz*cyz)/8;
        result.s1[11] = rt2*(7*raz*raz*rby - rby + 2*raz*czy)/8;
        result.s1[14] = rt2*(14*ray*raz*rby + 2*ray*czy + 2*raz*cyy)/8;

        if (level >= 2) {
            result.s2[4] = rt2*(126*rby*rbz*raz + 14*rby*czz + 14*rbz*czy)/8;
            result.s2[5] = rt2*(126*rby*rbz*ray + 14*rby*cyz + 14*rbz*cyy)/8;
            result.s2[11] = rt2*(63*raz*raz*rbz - 7*rbz + 14*raz*czz)/8;
            result.s2[12] = rt2*(126*ray*raz*rbz + 14*ray*czz + 14*raz*cyz)/8;
            result.s2[16] = rt2*(63*raz*raz*rby - 7*rby + 14*raz*czy)/8;
            result.s2[17] = rt2*(126*ray*raz*rby + 14*ray*czy + 14*raz*cyy)/8;
            result.s2[19] = rt2*(63*ray*raz*raz - 7*ray)/8;
            result.s2[57] = rt2*(14*raz*rbz + 2*czz)/8;
            result.s2[60] = rt2*(7*raz*raz - 1)/8;
            result.s2[67] = rt2*(14*raz*rbz + 2*czz)/8;
            result.s2[68] = rt2*(14*rbz*ray + 2*cyz)/8;
            result.s2[71] = 7.0/4.0*rt2*ray*raz;
            result.s2[93] = rt2*(14*raz*rby + 2*czy)/8;
            result.s2[95] = rt2*(7*raz*raz - 1)/8;
            result.s2[102] = rt2*raz/4;
            result.s2[106] = rt2*(14*raz*rby + 2*czy)/8;
            result.s2[107] = rt2*(14*ray*rby + 2*cyy)/8;
            result.s2[109] = 7.0/4.0*rt2*ray*raz;
            result.s2[115] = rt2*raz/4;
            result.s2[116] = rt2*ray/4;
        }
    }
}

/**
 * Octopole-31s × Quadrupole-22c kernel
 * Orient case 242: Q31s × Q22c
 */
void octopole_31s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double ray = sf.ray();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double czx = sf.czx();
    double czy = sf.czy();
    double cyx = sf.cyx();
    double cyy = sf.cyy();

    result.s0 = rt2*(63*ray*raz*raz*rbx*rbx - 63*ray*raz*raz*rby*rby
                   - 7*rbx*rbx*ray + 7*rby*rby*ray + 28*ray*raz*rbx*czx
                   - 28*ray*raz*rby*czy + 14*raz*raz*rbx*cyx - 14*raz*raz*rby*cyy
                   - 2*rbx*cyx + 2*rby*cyy + 2*ray*czx*czx - 2*ray*czy*czy + 4*raz*cyx*czx
                   - 4*raz*cyy*czy)/16;

    if (level >= 1) {
        result.s1[1] = rt2*(63*raz*raz*rbx*rbx - 63*raz*raz*rby*rby
                          - 7*rbx*rbx + 7*rby*rby + 28*raz*rbx*czx - 28*raz*rby*czy + 2*czx*czx
                          - 2*czy*czy)/16;
        result.s1[2] = rt2*(126*ray*raz*rbx*rbx - 126*ray*raz*rby*rby
                          + 28*ray*rbx*czx - 28*ray*rby*czy + 28*raz*rbx*cyx - 28*raz*rby*cyy
                          + 4*cyx*czx - 4*cyy*czy)/16;
        result.s1[3] = rt2*(126*ray*raz*raz*rbx - 14*rbx*ray
                          + 28*ray*raz*czx + 14*raz*raz*cyx - 2*cyx)/16;
        result.s1[4] = rt2*(-126*ray*raz*raz*rby + 14*ray*rby
                          - 28*ray*raz*czy - 14*raz*raz*cyy + 2*cyy)/16;
        result.s1[9] = rt2*(14*raz*raz*rbx - 2*rbx + 4*raz*czx)/16;
        result.s1[12] = rt2*(28*ray*raz*rbx + 4*ray*czx + 4*raz*cyx)/16;
        result.s1[10] = rt2*(-14*raz*raz*rby + 2*rby - 4*raz*czy)/16;
        result.s1[13] = rt2*(-28*ray*raz*rby - 4*ray*czy - 4*raz*cyy)/16;

        if (level >= 2) {
            result.s2[4] = rt2*(126*rbx*rbx*raz - 126*rby*rby*raz + 28*rbx*czx
                              - 28*rby*czy)/16;
            result.s2[5] = rt2*(126*rbx*rbx*ray - 126*rby*rby*ray + 28*rbx*cyx
                              - 28*rby*cyy)/16;
            result.s2[7] = rt2*(126*raz*raz*rbx - 14*rbx + 28*raz*czx)/16;
            result.s2[8] = rt2*(252*ray*raz*rbx + 28*ray*czx + 28*raz*cyx)/16;
            result.s2[9] = rt2*(126*ray*raz*raz - 14*ray)/16;
            result.s2[11] = rt2*(-126*raz*raz*rby + 14*rby - 28*raz*czy)/16;
            result.s2[12] = rt2*(-252*ray*raz*rby - 28*ray*czy - 28*raz*cyy)/16;
            result.s2[14] = rt2*(-126*ray*raz*raz + 14*ray)/16;
            result.s2[30] = rt2*(28*raz*rbx + 4*czx)/16;
            result.s2[31] = rt2*(14*raz*raz - 2)/16;
            result.s2[37] = rt2*(28*raz*rbx + 4*czx)/16;
            result.s2[38] = rt2*(28*rbx*ray + 4*cyx)/16;
            result.s2[39] = 7.0/4.0*rt2*ray*raz;
            result.s2[43] = rt2*raz/4;
            result.s2[44] = rt2*ray/4;
            result.s2[57] = rt2*(-28*raz*rby - 4*czy)/16;
            result.s2[59] = rt2*(-14*raz*raz + 2)/16;
            result.s2[67] = rt2*(-28*raz*rby - 4*czy)/16;
            result.s2[68] = rt2*(-28*ray*rby - 4*cyy)/16;
            result.s2[70] = -7.0/4.0*rt2*ray*raz;
            result.s2[76] = -rt2*raz/4;
            result.s2[77] = -rt2*ray/4;
        }
    }
}

/**
 * Octopole-31s × Quadrupole-22s kernel
 * Orient case 243: Q31s × Q22s
 */
void octopole_31s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double ray = sf.ray();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double czx = sf.czx();
    double czy = sf.czy();
    double cyx = sf.cyx();
    double cyy = sf.cyy();

    result.s0 = rt2*(63*ray*raz*raz*rbx*rby - 7*rbx*rby*ray
                   + 14*ray*raz*rbx*czy + 14*ray*raz*rby*czx + 7*raz*raz*rbx*cyy
                   + 7*raz*raz*rby*cyx - rbx*cyy - rby*cyx + 2*ray*czx*czy + 2*raz*cyx*czy
                   + 2*raz*cyy*czx)/8;

    if (level >= 1) {
        result.s1[1] = rt2*(63*raz*raz*rbx*rby - 7*rbx*rby
                          + 14*raz*rbx*czy + 14*raz*rby*czx + 2*czx*czy)/8;
        result.s1[2] = rt2*(126*ray*raz*rbx*rby + 14*ray*rbx*czy
                          + 14*ray*rby*czx + 14*raz*rbx*cyy + 14*raz*rby*cyx + 2*cyx*czy
                          + 2*cyy*czx)/8;
        result.s1[3] = rt2*(63*ray*raz*raz*rby - 7*ray*rby
                          + 14*ray*raz*czy + 7*raz*raz*cyy - cyy)/8;
        result.s1[4] = rt2*(63*ray*raz*raz*rbx - 7*rbx*ray
                          + 14*ray*raz*czx + 7*raz*raz*cyx - cyx)/8;
        result.s1[9] = rt2*(7*raz*raz*rby - rby + 2*raz*czy)/8;
        result.s1[12] = rt2*(14*ray*raz*rby + 2*ray*czy + 2*raz*cyy)/8;
        result.s1[10] = rt2*(7*raz*raz*rbx - rbx + 2*raz*czx)/8;
        result.s1[13] = rt2*(14*ray*raz*rbx + 2*ray*czx + 2*raz*cyx)/8;

        if (level >= 2) {
            result.s2[4] = rt2*(126*rbx*rby*raz + 14*rbx*czy + 14*rby*czx)/8;
            result.s2[5] = rt2*(126*rbx*rby*ray + 14*rbx*cyy + 14*rby*cyx)/8;
            result.s2[7] = rt2*(63*raz*raz*rby - 7*rby + 14*raz*czy)/8;
            result.s2[8] = rt2*(126*ray*raz*rby + 14*ray*czy + 14*raz*cyy)/8;
            result.s2[11] = rt2*(63*raz*raz*rbx - 7*rbx + 14*raz*czx)/8;
            result.s2[12] = rt2*(126*ray*raz*rbx + 14*ray*czx + 14*raz*cyx)/8;
            result.s2[13] = rt2*(63*ray*raz*raz - 7*ray)/8;
            result.s2[30] = rt2*(14*raz*rby + 2*czy)/8;
            result.s2[32] = rt2*(7*raz*raz - 1)/8;
            result.s2[37] = rt2*(14*raz*rby + 2*czy)/8;
            result.s2[38] = rt2*(14*ray*rby + 2*cyy)/8;
            result.s2[40] = 7.0/4.0*rt2*ray*raz;
            result.s2[57] = rt2*(14*raz*rbx + 2*czx)/8;
            result.s2[58] = rt2*(7*raz*raz - 1)/8;
            result.s2[63] = rt2*raz/4;
            result.s2[67] = rt2*(14*raz*rbx + 2*czx)/8;
            result.s2[68] = rt2*(14*rbx*ray + 2*cyx)/8;
            result.s2[69] = 7.0/4.0*rt2*ray*raz;
            result.s2[73] = rt2*raz/4;
            result.s2[74] = rt2*ray/4;
        }
    }
}

// ============================================================================
// BATCH 4: Q32c × Q2* (cases 244-248)
// ============================================================================

/**
 * Octopole-32c × Quadrupole-20 kernel
 * Orient case 244: Q32c × Q20
 */
void octopole_32c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rbz = sf.rbz();
    double czz = sf.czz();
    double cxz = sf.cxz();
    double cyz = sf.cyz();

    result.s0 = rt15*(63*rax*rax*raz*rbz*rbz - 63*ray*ray*raz*rbz*rbz
                    - 7*rax*rax*raz + 7*ray*ray*raz + 14*rax*rax*rbz*czz
                    + 28*rax*raz*rbz*cxz - 14*ray*ray*rbz*czz - 28*ray*raz*rbz*cyz
                    + 4*rax*cxz*czz - 4*ray*cyz*czz + 2*raz*cxz*cxz - 2*raz*cyz*cyz)/40;

    if (level >= 1) {
        result.s1[0] = rt15*(126*rbz*rbz*rax*raz - 14*rax*raz
                           + 28*rbz*rax*czz + 28*rbz*raz*cxz + 4*cxz*czz)/40;
        result.s1[1] = rt15*(-126*rbz*rbz*ray*raz + 14*ray*raz
                           - 28*rbz*ray*czz - 28*rbz*raz*cyz - 4*cyz*czz)/40;
        result.s1[2] = rt15*(63*rbz*rbz*rax*rax - 63*rbz*rbz*ray*ray
                           - 7*rax*rax + 7*ray*ray + 28*rbz*rax*cxz - 28*rbz*ray*cyz + 2*cxz*cxz
                           - 2*cyz*cyz)/40;
        result.s1[5] = rt15*(126*rax*rax*raz*rbz - 126*ray*ray*raz*rbz
                           + 14*rax*rax*czz + 28*rax*raz*cxz - 14*ray*ray*czz
                           - 28*ray*raz*cyz)/40;
        result.s1[8] = rt15*(28*rax*raz*rbz + 4*rax*czz + 4*raz*cxz)/40;
        result.s1[11] = rt15*(-28*ray*raz*rbz - 4*ray*czz - 4*raz*cyz)/40;
        result.s1[14] = rt15*(14*rax*rax*rbz - 14*ray*ray*rbz + 4*rax*cxz
                            - 4*ray*cyz)/40;

        if (level >= 2) {
            result.s2[0] = rt15*(126*rbz*rbz*raz - 14*raz + 28*rbz*czz)/40;
            result.s2[2] = rt15*(-126*rbz*rbz*raz + 14*raz - 28*rbz*czz)/40;
            result.s2[3] = rt15*(126*rbz*rbz*rax - 14*rax + 28*rbz*cxz)/40;
            result.s2[4] = rt15*(-126*rbz*rbz*ray + 14*ray - 28*rbz*cyz)/40;
            result.s2[15] = rt15*(252*rax*raz*rbz + 28*rax*czz + 28*raz*cxz)/40;
            result.s2[16] = rt15*(-252*ray*raz*rbz - 28*ray*czz - 28*raz*cyz)/40;
            result.s2[17] = rt15*(126*rax*rax*rbz - 126*ray*ray*rbz
                                + 28*rax*cxz - 28*ray*cyz)/40;
            result.s2[20] = rt15*(126*rax*rax*raz - 126*ray*ray*raz)/40;
            result.s2[78] = rt15*(28*raz*rbz + 4*czz)/40;
            result.s2[80] = rt15*(28*rbz*rax + 4*cxz)/40;
            result.s2[83] = 7.0/10.0*rt15*rax*raz;
            result.s2[90] = rt15*raz/10;
            result.s2[92] = rt15*(-28*raz*rbz - 4*czz)/40;
            result.s2[93] = rt15*(-28*rbz*ray - 4*cyz)/40;
            result.s2[96] = -7.0/10.0*rt15*ray*raz;
            result.s2[104] = -rt15*raz/10;
            result.s2[105] = rt15*(28*rbz*rax + 4*cxz)/40;
            result.s2[106] = rt15*(-28*rbz*ray - 4*cyz)/40;
            result.s2[110] = rt15*(14*rax*rax - 14*ray*ray)/40;
            result.s2[117] = rt15*rax/10;
            result.s2[118] = -rt15*ray/10;
        }
    }
}

/**
 * Octopole-32c × Quadrupole-21c kernel
 * Orient case 245: Q32c × Q21c
 */
void octopole_32c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rbz = sf.rbz();
    double czx = sf.czx();
    double czz = sf.czz();
    double cxx = sf.cxx();
    double cxz = sf.cxz();
    double cyx = sf.cyx();
    double cyz = sf.cyz();

    result.s0 = rt5*(63*rax*rax*raz*rbx*rbz - 63*ray*ray*raz*rbx*rbz
                   + 7*rax*rax*rbx*czz + 7*rax*rax*rbz*czx + 14*rax*raz*rbx*cxz
                   + 14*rax*raz*rbz*cxx - 7*ray*ray*rbx*czz - 7*ray*ray*rbz*czx
                   - 14*ray*raz*rbx*cyz - 14*ray*raz*rbz*cyx + 2*rax*cxx*czz
                   + 2*rax*cxz*czx - 2*ray*cyx*czz - 2*ray*cyz*czx + 2*raz*cxx*cxz
                   - 2*raz*cyx*cyz)/20;

    if (level >= 1) {
        result.s1[0] = rt5*(126*rax*raz*rbx*rbz + 14*rax*rbx*czz
                          + 14*rax*rbz*czx + 14*raz*rbx*cxz + 14*raz*rbz*cxx + 2*cxx*czz
                          + 2*cxz*czx)/20;
        result.s1[1] = rt5*(-126*rbx*rbz*ray*raz - 14*rbx*ray*czz
                          - 14*rbz*ray*czx - 14*rbx*raz*cyz - 14*rbz*raz*cyx - 2*cyx*czz
                          - 2*czx*cyz)/20;
        result.s1[2] = rt5*(63*rbx*rbz*rax*rax - 63*rbx*rbz*ray*ray
                          + 14*rbx*rax*cxz + 14*rbz*rax*cxx - 14*rbx*ray*cyz - 14*rbz*ray*cyx
                          + 2*cxx*cxz - 2*cyx*cyz)/20;
        result.s1[3] = rt5*(63*rax*rax*raz*rbz - 63*ray*ray*raz*rbz
                          + 7*rax*rax*czz + 14*rax*raz*cxz - 7*ray*ray*czz - 14*ray*raz*cyz)/20;
        result.s1[5] = rt5*(63*rax*rax*raz*rbx - 63*ray*ray*raz*rbx
                          + 7*rax*rax*czx + 14*rax*raz*cxx - 7*ray*ray*czx - 14*ray*raz*cyx)/20;
        result.s1[6] = rt5*(14*rax*raz*rbz + 2*rax*czz + 2*raz*cxz)/20;
        result.s1[9] = rt5*(-14*ray*raz*rbz - 2*ray*czz - 2*raz*cyz)/20;
        result.s1[12] = rt5*(7*rax*rax*rbz - 7*ray*ray*rbz + 2*rax*cxz
                          - 2*ray*cyz)/20;
        result.s1[8] = rt5*(14*rax*raz*rbx + 2*rax*czx + 2*raz*cxx)/20;
        result.s1[11] = rt5*(-14*ray*raz*rbx - 2*ray*czx - 2*raz*cyx)/20;
        result.s1[14] = rt5*(7*rax*rax*rbx - 7*ray*ray*rbx + 2*rax*cxx
                           - 2*ray*cyx)/20;

        if (level >= 2) {
            result.s2[0] = rt5*(126*rbx*rbz*raz + 14*rbx*czz + 14*rbz*czx)/20;
            result.s2[2] = rt5*(-126*rbx*rbz*raz - 14*rbx*czz - 14*rbz*czx)/20;
            result.s2[3] = rt5*(126*rbx*rbz*rax + 14*rbx*cxz + 14*rbz*cxx)/20;
            result.s2[4] = rt5*(-126*rbx*rbz*ray - 14*rbx*cyz - 14*rbz*cyx)/20;
            result.s2[6] = rt5*(126*rax*raz*rbz + 14*rax*czz + 14*raz*cxz)/20;
            result.s2[7] = rt5*(-126*ray*raz*rbz - 14*ray*czz - 14*raz*cyz)/20;
            result.s2[8] = rt5*(63*rax*rax*rbz - 63*ray*ray*rbz + 14*rax*cxz
                              - 14*ray*cyz)/20;
            result.s2[15] = rt5*(126*rax*raz*rbx + 14*rax*czx + 14*raz*cxx)/20;
            result.s2[16] = rt5*(-126*ray*raz*rbx - 14*ray*czx - 14*raz*cyx)/20;
            result.s2[17] = rt5*(63*rax*rax*rbx - 63*ray*ray*rbx + 14*rax*cxx
                               - 14*ray*cyx)/20;
            result.s2[18] = rt5*(63*rax*rax*raz - 63*ray*ray*raz)/20;
            result.s2[21] = rt5*(14*raz*rbz + 2*czz)/20;
            result.s2[23] = rt5*(14*rbz*rax + 2*cxz)/20;
            result.s2[26] = 7.0/10.0*rt5*rax*raz;
            result.s2[29] = rt5*(-14*raz*rbz - 2*czz)/20;
            result.s2[30] = rt5*(-14*rbz*ray - 2*cyz)/20;
            result.s2[33] = -7.0/10.0*rt5*ray*raz;
            result.s2[36] = rt5*(14*rbz*rax + 2*cxz)/20;
            result.s2[37] = rt5*(-14*rbz*ray - 2*cyz)/20;
            result.s2[41] = rt5*(7*rax*rax - 7*ray*ray)/20;
            result.s2[78] = rt5*(14*raz*rbx + 2*czx)/20;
            result.s2[80] = rt5*(14*rax*rbx + 2*cxx)/20;
            result.s2[81] = 7.0/10.0*rt5*rax*raz;
            result.s2[84] = rt5*raz/10;
            result.s2[86] = rt5*rax/10;
            result.s2[92] = rt5*(-14*raz*rbx - 2*czx)/20;
            result.s2[93] = rt5*(-14*rbx*ray - 2*cyx)/20;
            result.s2[94] = -7.0/10.0*rt5*ray*raz;
            result.s2[98] = -rt5*raz/10;
            result.s2[99] = -rt5*ray/10;
            result.s2[105] = rt5*(14*rax*rbx + 2*cxx)/20;
            result.s2[106] = rt5*(-14*rbx*ray - 2*cyx)/20;
            result.s2[108] = rt5*(7*rax*rax - 7*ray*ray)/20;
            result.s2[111] = rt5*rax/10;
            result.s2[112] = -rt5*ray/10;
        }
    }
}

/**
 * Octopole-32c × Quadrupole-21s kernel
 * Orient case 246: Q32c × Q21s
 */
void octopole_32c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rby = sf.rby();
    double rbz = sf.rbz();
    double czy = sf.czy();
    double czz = sf.czz();
    double cxy = sf.cxy();
    double cxz = sf.cxz();
    double cyy = sf.cyy();
    double cyz = sf.cyz();

    result.s0 = rt5*(63*rax*rax*raz*rby*rbz - 63*ray*ray*raz*rby*rbz
                   + 7*rax*rax*rby*czz + 7*rax*rax*rbz*czy + 14*rax*raz*rby*cxz
                   + 14*rax*raz*rbz*cxy - 7*ray*ray*rby*czz - 7*ray*ray*rbz*czy
                   - 14*ray*raz*rby*cyz - 14*ray*raz*rbz*cyy + 2*rax*cxy*czz
                   + 2*rax*cxz*czy - 2*ray*cyy*czz - 2*ray*cyz*czy + 2*raz*cxy*cxz
                   - 2*raz*cyy*cyz)/20;

    if (level >= 1) {
        result.s1[0] = rt5*(126*rax*raz*rby*rbz + 14*rax*rby*czz
                          + 14*rax*rbz*czy + 14*raz*rby*cxz + 14*raz*rbz*cxy + 2*cxy*czz
                          + 2*cxz*czy)/20;
        result.s1[1] = rt5*(-126*ray*raz*rby*rbz - 14*ray*rby*czz
                          - 14*ray*rbz*czy - 14*raz*rby*cyz - 14*raz*rbz*cyy - 2*cyy*czz
                          - 2*cyz*czy)/20;
        result.s1[2] = rt5*(63*rby*rbz*rax*rax - 63*rby*rbz*ray*ray
                          + 14*rby*rax*cxz + 14*rbz*rax*cxy - 14*rby*ray*cyz - 14*rbz*ray*cyy
                          + 2*cxy*cxz - 2*cyy*cyz)/20;
        result.s1[4] = rt5*(63*rax*rax*raz*rbz - 63*ray*ray*raz*rbz
                          + 7*rax*rax*czz + 14*rax*raz*cxz - 7*ray*ray*czz - 14*ray*raz*cyz)/20;
        result.s1[5] = rt5*(63*rax*rax*raz*rby - 63*ray*ray*raz*rby
                          + 7*rax*rax*czy + 14*rax*raz*cxy - 7*ray*ray*czy - 14*ray*raz*cyy)/20;
        result.s1[7] = rt5*(14*rax*raz*rbz + 2*rax*czz + 2*raz*cxz)/20;
        result.s1[10] = rt5*(-14*ray*raz*rbz - 2*ray*czz - 2*raz*cyz)/20;
        result.s1[13] = rt5*(7*rax*rax*rbz - 7*ray*ray*rbz + 2*rax*cxz
                           - 2*ray*cyz)/20;
        result.s1[8] = rt5*(14*rax*raz*rby + 2*rax*czy + 2*raz*cxy)/20;
        result.s1[11] = rt5*(-14*ray*raz*rby - 2*ray*czy - 2*raz*cyy)/20;
        result.s1[14] = rt5*(7*rax*rax*rby - 7*ray*ray*rby + 2*rax*cxy
                           - 2*ray*cyy)/20;

        if (level >= 2) {
            result.s2[0] = rt5*(126*rby*rbz*raz + 14*rby*czz + 14*rbz*czy)/20;
            result.s2[2] = rt5*(-126*rby*rbz*raz - 14*rby*czz - 14*rbz*czy)/20;
            result.s2[3] = rt5*(126*rby*rbz*rax + 14*rby*cxz + 14*rbz*cxy)/20;
            result.s2[4] = rt5*(-126*rby*rbz*ray - 14*rby*cyz - 14*rbz*cyy)/20;
            result.s2[10] = rt5*(126*rax*raz*rbz + 14*rax*czz + 14*raz*cxz)/20;
            result.s2[11] = rt5*(-126*ray*raz*rbz - 14*ray*czz - 14*raz*cyz)/20;
            result.s2[12] = rt5*(63*rax*rax*rbz - 63*ray*ray*rbz + 14*rax*cxz
                               - 14*ray*cyz)/20;
            result.s2[15] = rt5*(126*rax*raz*rby + 14*rax*czy + 14*raz*cxy)/20;
            result.s2[16] = rt5*(-126*ray*raz*rby - 14*ray*czy - 14*raz*cyy)/20;
            result.s2[17] = rt5*(63*rax*rax*rby - 63*ray*ray*rby + 14*rax*cxy
                               - 14*ray*cyy)/20;
            result.s2[19] = rt5*(63*rax*rax*raz - 63*ray*ray*raz)/20;
            result.s2[45] = rt5*(14*raz*rbz + 2*czz)/20;
            result.s2[47] = rt5*(14*rbz*rax + 2*cxz)/20;
            result.s2[50] = 7.0/10.0*rt5*rax*raz;
            result.s2[56] = rt5*(-14*raz*rbz - 2*czz)/20;
            result.s2[57] = rt5*(-14*rbz*ray - 2*cyz)/20;
            result.s2[60] = -7.0/10.0*rt5*ray*raz;
            result.s2[66] = rt5*(14*rbz*rax + 2*cxz)/20;
            result.s2[67] = rt5*(-14*rbz*ray - 2*cyz)/20;
            result.s2[71] = rt5*(7*rax*rax - 7*ray*ray)/20;
            result.s2[78] = rt5*(14*raz*rby + 2*czy)/20;
            result.s2[80] = rt5*(14*rax*rby + 2*cxy)/20;
            result.s2[82] = 7.0/10.0*rt5*rax*raz;
            result.s2[87] = rt5*raz/10;
            result.s2[89] = rt5*rax/10;
            result.s2[92] = rt5*(-14*raz*rby - 2*czy)/20;
            result.s2[93] = rt5*(-14*ray*rby - 2*cyy)/20;
            result.s2[95] = -7.0/10.0*rt5*ray*raz;
            result.s2[101] = -rt5*raz/10;
            result.s2[102] = -rt5*ray/10;
            result.s2[105] = rt5*(14*rax*rby + 2*cxy)/20;
            result.s2[106] = rt5*(-14*ray*rby - 2*cyy)/20;
            result.s2[109] = rt5*(7*rax*rax - 7*ray*ray)/20;
            result.s2[114] = rt5*rax/10;
            result.s2[115] = -rt5*ray/10;
        }
    }
}

/**
 * Octopole-32c × Quadrupole-22c kernel
 * Orient case 247: Q32c × Q22c
 */
void octopole_32c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double czx = sf.czx();
    double czy = sf.czy();
    double cxx = sf.cxx();
    double cxy = sf.cxy();
    double cyx = sf.cyx();
    double cyy = sf.cyy();

    result.s0 = rt5*(63*rax*rax*raz*rbx*rbx - 63*rax*rax*raz*rby*rby
                   - 63*ray*ray*raz*rbx*rbx + 63*ray*ray*raz*rby*rby + 14*rax*rax*rbx*czx
                   - 14*rax*rax*rby*czy + 28*rax*raz*rbx*cxx - 28*rax*raz*rby*cxy
                   - 14*ray*ray*rbx*czx + 14*ray*ray*rby*czy - 28*ray*raz*rbx*cyx
                   + 28*ray*raz*rby*cyy + 4*rax*cxx*czx - 4*rax*cxy*czy - 4*ray*cyx*czx
                   + 4*ray*cyy*czy + 2*raz*cxx*cxx - 2*raz*cxy*cxy - 2*raz*cyx*cyx
                   + 2*raz*cyy*cyy)/40;

    if (level >= 1) {
        result.s1[0] = rt5*(126*rax*raz*rbx*rbx - 126*rax*raz*rby*rby
                          + 28*rax*rbx*czx - 28*rax*rby*czy + 28*raz*rbx*cxx - 28*raz*rby*cxy
                          + 4*cxx*czx - 4*cxy*czy)/40;
        result.s1[1] = rt5*(-126*ray*raz*rbx*rbx + 126*ray*raz*rby*rby
                          - 28*ray*rbx*czx + 28*ray*rby*czy - 28*raz*rbx*cyx + 28*raz*rby*cyy
                          - 4*cyx*czx + 4*cyy*czy)/40;
        result.s1[2] = rt5*(63*rax*rax*rbx*rbx - 63*rax*rax*rby*rby
                          - 63*ray*ray*rbx*rbx + 63*ray*ray*rby*rby + 28*rax*rbx*cxx
                          - 28*rax*rby*cxy - 28*ray*rbx*cyx + 28*ray*rby*cyy + 2*cxx*cxx
                          - 2*cxy*cxy - 2*cyx*cyx + 2*cyy*cyy)/40;
        result.s1[3] = rt5*(126*rax*rax*raz*rbx - 126*ray*ray*raz*rbx
                          + 14*rax*rax*czx + 28*rax*raz*cxx - 14*ray*ray*czx
                          - 28*ray*raz*cyx)/40;
        result.s1[4] = rt5*(-126*rax*rax*raz*rby + 126*ray*ray*raz*rby
                          - 14*rax*rax*czy - 28*rax*raz*cxy + 14*ray*ray*czy
                          + 28*ray*raz*cyy)/40;
        result.s1[6] = rt5*(28*rax*raz*rbx + 4*rax*czx + 4*raz*cxx)/40;
        result.s1[9] = rt5*(-28*ray*raz*rbx - 4*ray*czx - 4*raz*cyx)/40;
        result.s1[12] = rt5*(14*rax*rax*rbx - 14*ray*ray*rbx + 4*rax*cxx
                          - 4*ray*cyx)/40;
        result.s1[7] = rt5*(-28*rax*raz*rby - 4*rax*czy - 4*raz*cxy)/40;
        result.s1[10] = rt5*(28*ray*raz*rby + 4*ray*czy + 4*raz*cyy)/40;
        result.s1[13] = rt5*(-14*rax*rax*rby + 14*ray*ray*rby - 4*rax*cxy
                           + 4*ray*cyy)/40;

        if (level >= 2) {
            result.s2[0] = rt5*(126*rbx*rbx*raz - 126*rby*rby*raz + 28*rbx*czx
                              - 28*rby*czy)/40;
            result.s2[2] = rt5*(-126*rbx*rbx*raz + 126*rby*rby*raz - 28*rbx*czx
                              + 28*rby*czy)/40;
            result.s2[3] = rt5*(126*rbx*rbx*rax - 126*rby*rby*rax + 28*rbx*cxx
                              - 28*rby*cxy)/40;
            result.s2[4] = rt5*(-126*rbx*rbx*ray + 126*rby*rby*ray - 28*rbx*cyx
                              + 28*rby*cyy)/40;
            result.s2[6] = rt5*(252*rax*raz*rbx + 28*rax*czx + 28*raz*cxx)/40;
            result.s2[7] = rt5*(-252*ray*raz*rbx - 28*ray*czx - 28*raz*cyx)/40;
            result.s2[8] = rt5*(126*rax*rax*rbx - 126*ray*ray*rbx + 28*rax*cxx
                              - 28*ray*cyx)/40;
            result.s2[9] = rt5*(126*rax*rax*raz - 126*ray*ray*raz)/40;
            result.s2[10] = rt5*(-252*rax*raz*rby - 28*rax*czy - 28*raz*cxy)/40;
            result.s2[11] = rt5*(252*ray*raz*rby + 28*ray*czy + 28*raz*cyy)/40;
            result.s2[12] = rt5*(-126*rax*rax*rby + 126*ray*ray*rby
                               - 28*rax*cxy + 28*ray*cyy)/40;
            result.s2[14] = rt5*(-126*rax*rax*raz + 126*ray*ray*raz)/40;
            result.s2[21] = rt5*(28*raz*rbx + 4*czx)/40;
            result.s2[23] = rt5*(28*rax*rbx + 4*cxx)/40;
            result.s2[24] = 7.0/10.0*rt5*rax*raz;
            result.s2[27] = rt5*raz/10;
            result.s2[29] = rt5*(-28*raz*rbx - 4*czx)/40;
            result.s2[30] = rt5*(-28*rbx*ray - 4*cyx)/40;
            result.s2[31] = -7.0/10.0*rt5*ray*raz;
            result.s2[35] = -rt5*raz/10;
            result.s2[36] = rt5*(28*rax*rbx + 4*cxx)/40;
            result.s2[37] = rt5*(-28*rbx*ray - 4*cyx)/40;
            result.s2[39] = rt5*(14*rax*rax - 14*ray*ray)/40;
            result.s2[42] = rt5*rax/10;
            result.s2[43] = -rt5*ray/10;
            result.s2[45] = rt5*(-28*raz*rby - 4*czy)/40;
            result.s2[47] = rt5*(-28*rax*rby - 4*cxy)/40;
            result.s2[49] = -7.0/10.0*rt5*rax*raz;
            result.s2[54] = -rt5*raz/10;
            result.s2[56] = rt5*(28*raz*rby + 4*czy)/40;
            result.s2[57] = rt5*(28*ray*rby + 4*cyy)/40;
            result.s2[59] = 7.0/10.0*rt5*ray*raz;
            result.s2[65] = rt5*raz/10;
            result.s2[66] = rt5*(-28*rax*rby - 4*cxy)/40;
            result.s2[67] = rt5*(28*ray*rby + 4*cyy)/40;
            result.s2[70] = rt5*(-14*rax*rax + 14*ray*ray)/40;
            result.s2[75] = -rt5*rax/10;
            result.s2[76] = rt5*ray/10;
        }
    }
}

/**
 * Octopole-32c × Quadrupole-22s kernel
 * Orient case 248: Q32c × Q22s
 */
void octopole_32c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double czx = sf.czx();
    double czy = sf.czy();
    double cxx = sf.cxx();
    double cxy = sf.cxy();
    double cyx = sf.cyx();
    double cyy = sf.cyy();

    result.s0 = rt5*(63*rax*rax*raz*rbx*rby - 63*ray*ray*raz*rbx*rby
                   + 7*rax*rax*rbx*czy + 7*rax*rax*rby*czx + 14*rax*raz*rbx*cxy
                   + 14*rax*raz*rby*cxx - 7*ray*ray*rbx*czy - 7*ray*ray*rby*czx
                   - 14*ray*raz*rbx*cyy - 14*ray*raz*rby*cyx + 2*rax*cxx*czy
                   + 2*rax*cxy*czx - 2*ray*cyx*czy - 2*ray*cyy*czx + 2*raz*cxx*cxy
                   - 2*raz*cyx*cyy)/20;

    if (level >= 1) {
        result.s1[0] = rt5*(126*rax*raz*rbx*rby + 14*rax*rbx*czy
                          + 14*rax*rby*czx + 14*raz*rbx*cxy + 14*raz*rby*cxx + 2*cxx*czy
                          + 2*cxy*czx)/20;
        result.s1[1] = rt5*(-126*ray*raz*rbx*rby - 14*ray*rbx*czy
                          - 14*ray*rby*czx - 14*raz*rbx*cyy - 14*raz*rby*cyx - 2*cyx*czy
                          - 2*cyy*czx)/20;
        result.s1[2] = rt5*(63*rax*rax*rbx*rby - 63*ray*ray*rbx*rby
                          + 14*rax*rbx*cxy + 14*rax*rby*cxx - 14*ray*rbx*cyy - 14*ray*rby*cyx
                          + 2*cxx*cxy - 2*cyx*cyy)/20;
        result.s1[3] = rt5*(63*rax*rax*raz*rby - 63*ray*ray*raz*rby
                          + 7*rax*rax*czy + 14*rax*raz*cxy - 7*ray*ray*czy - 14*ray*raz*cyy)/20;
        result.s1[4] = rt5*(63*rax*rax*raz*rbx - 63*ray*ray*raz*rbx
                          + 7*rax*rax*czx + 14*rax*raz*cxx - 7*ray*ray*czx - 14*ray*raz*cyx)/20;
        result.s1[6] = rt5*(14*rax*raz*rby + 2*rax*czy + 2*raz*cxy)/20;
        result.s1[9] = rt5*(-14*ray*raz*rby - 2*ray*czy - 2*raz*cyy)/20;
        result.s1[12] = rt5*(7*rax*rax*rby - 7*ray*ray*rby + 2*rax*cxy
                          - 2*ray*cyy)/20;
        result.s1[7] = rt5*(14*rax*raz*rbx + 2*rax*czx + 2*raz*cxx)/20;
        result.s1[10] = rt5*(-14*ray*raz*rbx - 2*ray*czx - 2*raz*cyx)/20;
        result.s1[13] = rt5*(7*rax*rax*rbx - 7*ray*ray*rbx + 2*rax*cxx
                           - 2*ray*cyx)/20;

        if (level >= 2) {
            result.s2[0] = rt5*(126*rbx*rby*raz + 14*rbx*czy + 14*rby*czx)/20;
            result.s2[2] = rt5*(-126*rbx*rby*raz - 14*rbx*czy - 14*rby*czx)/20;
            result.s2[3] = rt5*(126*rbx*rby*rax + 14*rbx*cxy + 14*rby*cxx)/20;
            result.s2[4] = rt5*(-126*rbx*rby*ray - 14*rbx*cyy - 14*rby*cyx)/20;
            result.s2[6] = rt5*(126*rax*raz*rby + 14*rax*czy + 14*raz*cxy)/20;
            result.s2[7] = rt5*(-126*ray*raz*rby - 14*ray*czy - 14*raz*cyy)/20;
            result.s2[8] = rt5*(63*rax*rax*rby - 63*ray*ray*rby + 14*rax*cxy
                              - 14*ray*cyy)/20;
            result.s2[10] = rt5*(126*rax*raz*rbx + 14*rax*czx + 14*raz*cxx)/20;
            result.s2[11] = rt5*(-126*ray*raz*rbx - 14*ray*czx - 14*raz*cyx)/20;
            result.s2[12] = rt5*(63*rax*rax*rbx - 63*ray*ray*rbx + 14*rax*cxx
                               - 14*ray*cyx)/20;
            result.s2[13] = rt5*(63*rax*rax*raz - 63*ray*ray*raz)/20;
            result.s2[21] = rt5*(14*raz*rby + 2*czy)/20;
            result.s2[23] = rt5*(14*rax*rby + 2*cxy)/20;
            result.s2[25] = 7.0/10.0*rt5*rax*raz;
            result.s2[29] = rt5*(-14*raz*rby - 2*czy)/20;
            result.s2[30] = rt5*(-14*ray*rby - 2*cyy)/20;
            result.s2[32] = -7.0/10.0*rt5*ray*raz;
            result.s2[36] = rt5*(14*rax*rby + 2*cxy)/20;
            result.s2[37] = rt5*(-14*ray*rby - 2*cyy)/20;
            result.s2[40] = rt5*(7*rax*rax - 7*ray*ray)/20;
            result.s2[45] = rt5*(14*raz*rbx + 2*czx)/20;
            result.s2[47] = rt5*(14*rax*rbx + 2*cxx)/20;
            result.s2[48] = 7.0/10.0*rt5*rax*raz;
            result.s2[51] = rt5*raz/10;
            result.s2[53] = rt5*rax/10;
            result.s2[56] = rt5*(-14*raz*rbx - 2*czx)/20;
            result.s2[57] = rt5*(-14*rbx*ray - 2*cyx)/20;
            result.s2[58] = -7.0/10.0*rt5*ray*raz;
            result.s2[62] = -rt5*raz/10;
            result.s2[63] = -rt5*ray/10;
            result.s2[66] = rt5*(14*rax*rbx + 2*cxx)/20;
            result.s2[67] = rt5*(-14*rbx*ray - 2*cyx)/20;
            result.s2[69] = rt5*(7*rax*rax - 7*ray*ray)/20;
            result.s2[72] = rt5*rax/10;
            result.s2[73] = -rt5*ray/10;
        }
    }
}

/**
 * Octopole-32s × Quadrupole-20 kernel
 * Orient case 249
 */
void octopole_32s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rbz = sf.rbz();
    double cxz = sf.cxz();
    double cyz = sf.cyz();
    double czz = sf.czz();

    result.s0 = rt15*(63*rax*ray*raz*rbz*rbz-7*rax*ray*raz
        +14*rax*ray*rbz*czz+14*rax*raz*rbz*cyz+14*ray*raz*rbz*cxz
        +2*rax*cyz*czz+2*ray*cxz*czz+2*raz*cxz*cyz)/20;

    if (level >= 1) {
        result.s1[0] = rt15*(63*rbz*rbz*ray*raz-7*ray*raz
            +14*rbz*ray*czz+14*rbz*raz*cyz+2*cyz*czz)/20;
        result.s1[1] = rt15*(63*rbz*rbz*rax*raz-7*rax*raz
            +14*rbz*rax*czz+14*rbz*raz*cxz+2*cxz*czz)/20;
        result.s1[2] = rt15*(63*rbz*rbz*rax*ray-7*rax*ray
            +14*rbz*rax*cyz+14*rbz*ray*cxz+2*cxz*cyz)/20;
        result.s1[5] = rt15*(126*rax*ray*raz*rbz+14*rax*ray*czz
            +14*rax*raz*cyz+14*ray*raz*cxz)/20;
        result.s1[8] = rt15*(14*ray*raz*rbz+2*ray*czz+2*raz*cyz)/20;
        result.s1[11] = rt15*(14*rax*raz*rbz+2*rax*czz+2*raz*cxz)/20;
        result.s1[14] = rt15*(14*rax*ray*rbz+2*rax*cyz+2*ray*cxz)/20;
        if (level >= 2) {
            result.s2[1] = rt15*(63*rbz*rbz*raz-7*raz+14*rbz*czz)/20;
            result.s2[3] = rt15*(63*rbz*rbz*ray-7*ray+14*rbz*cyz)/20;
            result.s2[4] = rt15*(63*rbz*rbz*rax-7*rax+14*rbz*cxz)/20;
            result.s2[15] = rt15*(126*ray*raz*rbz+14*ray*czz
                +14*raz*cyz)/20;
            result.s2[16] = rt15*(126*rax*raz*rbz+14*rax*czz
                +14*raz*cxz)/20;
            result.s2[17] = rt15*(126*rax*ray*rbz+14*rax*cyz
                +14*ray*cxz)/20;
            result.s2[20] = 63.0/10.0*rt15*rax*ray*raz;
            result.s2[79] = rt15*(14*raz*rbz+2*czz)/20;
            result.s2[80] = rt15*(14*rbz*ray+2*cyz)/20;
            result.s2[83] = 7.0/10.0*rt15*ray*raz;
            result.s2[91] = rt15*(14*raz*rbz+2*czz)/20;
            result.s2[93] = rt15*(14*rbz*rax+2*cxz)/20;
            result.s2[96] = 7.0/10.0*rt15*rax*raz;
            result.s2[103] = rt15*raz/10;
            result.s2[105] = rt15*(14*rbz*ray+2*cyz)/20;
            result.s2[106] = rt15*(14*rbz*rax+2*cxz)/20;
            result.s2[110] = 7.0/10.0*rt15*rax*ray;
            result.s2[117] = rt15*ray/10;
            result.s2[118] = rt15*rax/10;
        }
    }
}

/**
 * Octopole-32s × Quadrupole-21c kernel
 * Orient case 250
 */
void octopole_32s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rbz = sf.rbz();
    double cxx = sf.cxx();
    double cyx = sf.cyx();
    double czx = sf.czx();
    double cxz = sf.cxz();
    double cyz = sf.cyz();
    double czz = sf.czz();

    result.s0 = rt5*(63*rax*ray*raz*rbx*rbz+7*rax*ray*rbx*czz
        +7*rax*ray*rbz*czx+7*rax*raz*rbx*cyz+7*rax*raz*rbz*cyx
        +7*ray*raz*rbx*cxz+7*ray*raz*rbz*cxx+rax*cyx*czz+rax*cyz*czx
        +ray*cxx*czz+ray*cxz*czx+raz*cxx*cyz+raz*cxz*cyx)/10;

    if (level >= 1) {
        result.s1[0] = rt5*(63*rbx*rbz*ray*raz+7*rbx*ray*czz
            +7*rbz*ray*czx+7*rbx*raz*cyz+7*rbz*raz*cyx+cyx*czz
            +czx*cyz)/10;
        result.s1[1] = rt5*(63*rax*raz*rbx*rbz+7*rax*rbx*czz
            +7*rax*rbz*czx+7*raz*rbx*cxz+7*raz*rbz*cxx+cxx*czz
            +cxz*czx)/10;
        result.s1[2] = rt5*(63*rbx*rbz*rax*ray+7*rbx*rax*cyz
            +7*rbz*rax*cyx+7*rbx*ray*cxz+7*rbz*ray*cxx+cxx*cyz
            +cyx*cxz)/10;
        result.s1[3] = rt5*(63*rax*ray*raz*rbz+7*rax*ray*czz
            +7*rax*raz*cyz+7*ray*raz*cxz)/10;
        result.s1[5] = rt5*(63*rax*ray*raz*rbx+7*rax*ray*czx
            +7*rax*raz*cyx+7*ray*raz*cxx)/10;
        result.s1[6] = rt5*(7*ray*raz*rbz+ray*czz+raz*cyz)/10;
        result.s1[9] = rt5*(7*rax*raz*rbz+rax*czz+raz*cxz)/10;
        result.s1[12] = rt5*(7*rax*ray*rbz+rax*cyz+ray*cxz)/10;
        result.s1[8] = rt5*(7*ray*raz*rbx+ray*czx+raz*cyx)/10;
        result.s1[11] = rt5*(7*rax*raz*rbx+rax*czx+raz*cxx)/10;
        result.s1[14] = rt5*(7*rax*ray*rbx+rax*cyx+ray*cxx)/10;
        if (level >= 2) {
            result.s2[1] = rt5*(63*rbx*rbz*raz+7*rbx*czz+7*rbz*czx)/10;
            result.s2[3] = rt5*(63*rbx*rbz*ray+7*rbx*cyz+7*rbz*cyx)/10;
            result.s2[4] = rt5*(63*rbx*rbz*rax+7*rbx*cxz+7*rbz*cxx)/10;
            result.s2[6] = rt5*(63*ray*raz*rbz+7*ray*czz+7*raz*cyz)/10;
            result.s2[7] = rt5*(63*rax*raz*rbz+7*rax*czz+7*raz*cxz)/10;
            result.s2[8] = rt5*(63*rax*ray*rbz+7*rax*cyz+7*ray*cxz)/10;
            result.s2[15] = rt5*(63*ray*raz*rbx+7*ray*czx+7*raz*cyx)/10;
            result.s2[16] = rt5*(63*rax*raz*rbx+7*rax*czx+7*raz*cxx)/10;
            result.s2[17] = rt5*(63*rax*ray*rbx+7*rax*cyx+7*ray*cxx)/10;
            result.s2[18] = 63.0/10.0*rt5*rax*ray*raz;
            result.s2[22] = rt5*(7*raz*rbz+czz)/10;
            result.s2[23] = rt5*(7*rbz*ray+cyz)/10;
            result.s2[26] = 7.0/10.0*rt5*ray*raz;
            result.s2[28] = rt5*(7*raz*rbz+czz)/10;
            result.s2[30] = rt5*(7*rbz*rax+cxz)/10;
            result.s2[33] = 7.0/10.0*rt5*rax*raz;
            result.s2[36] = rt5*(7*rbz*ray+cyz)/10;
            result.s2[37] = rt5*(7*rbz*rax+cxz)/10;
            result.s2[41] = 7.0/10.0*rt5*rax*ray;
            result.s2[79] = rt5*(7*raz*rbx+czx)/10;
            result.s2[80] = rt5*(7*rbx*ray+cyx)/10;
            result.s2[81] = 7.0/10.0*rt5*ray*raz;
            result.s2[85] = rt5*raz/10;
            result.s2[86] = rt5*ray/10;
            result.s2[91] = rt5*(7*raz*rbx+czx)/10;
            result.s2[93] = rt5*(7*rax*rbx+cxx)/10;
            result.s2[94] = 7.0/10.0*rt5*rax*raz;
            result.s2[97] = rt5*raz/10;
            result.s2[99] = rt5*rax/10;
            result.s2[105] = rt5*(7*rbx*ray+cyx)/10;
            result.s2[106] = rt5*(7*rax*rbx+cxx)/10;
            result.s2[108] = 7.0/10.0*rt5*rax*ray;
            result.s2[111] = rt5*ray/10;
            result.s2[112] = rt5*rax/10;
        }
    }
}

/**
 * Octopole-32s × Quadrupole-21s kernel
 * Orient case 251
 */
void octopole_32s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rby = sf.rby();
    double rbz = sf.rbz();
    double cxy = sf.cxy();
    double cyy = sf.cyy();
    double czy = sf.czy();
    double cxz = sf.cxz();
    double cyz = sf.cyz();
    double czz = sf.czz();

    result.s0 = rt5*(63*rax*ray*raz*rby*rbz+7*rax*ray*rby*czz
        +7*rax*ray*rbz*czy+7*rax*raz*rby*cyz+7*rax*raz*rbz*cyy
        +7*ray*raz*rby*cxz+7*ray*raz*rbz*cxy+rax*cyy*czz+rax*cyz*czy
        +ray*cxy*czz+ray*cxz*czy+raz*cxy*cyz+raz*cxz*cyy)/10;

    if (level >= 1) {
        result.s1[0] = rt5*(63*ray*raz*rby*rbz+7*ray*rby*czz
            +7*ray*rbz*czy+7*raz*rby*cyz+7*raz*rbz*cyy+cyy*czz
            +cyz*czy)/10;
        result.s1[1] = rt5*(63*rax*raz*rby*rbz+7*rax*rby*czz
            +7*rax*rbz*czy+7*raz*rby*cxz+7*raz*rbz*cxy+cxy*czz
            +cxz*czy)/10;
        result.s1[2] = rt5*(63*rby*rbz*rax*ray+7*rby*rax*cyz
            +7*rbz*rax*cyy+7*rby*ray*cxz+7*rbz*ray*cxy+cxy*cyz
            +cyy*cxz)/10;
        result.s1[4] = rt5*(63*rax*ray*raz*rbz+7*rax*ray*czz
            +7*rax*raz*cyz+7*ray*raz*cxz)/10;
        result.s1[5] = rt5*(63*rax*ray*raz*rby+7*rax*ray*czy
            +7*rax*raz*cyy+7*ray*raz*cxy)/10;
        result.s1[7] = rt5*(7*ray*raz*rbz+ray*czz+raz*cyz)/10;
        result.s1[10] = rt5*(7*rax*raz*rbz+rax*czz+raz*cxz)/10;
        result.s1[13] = rt5*(7*rax*ray*rbz+rax*cyz+ray*cxz)/10;
        result.s1[8] = rt5*(7*ray*raz*rby+ray*czy+raz*cyy)/10;
        result.s1[11] = rt5*(7*rax*raz*rby+rax*czy+raz*cxy)/10;
        result.s1[14] = rt5*(7*rax*ray*rby+rax*cyy+ray*cxy)/10;
        if (level >= 2) {
            result.s2[1] = rt5*(63*rby*rbz*raz+7*rby*czz+7*rbz*czy)/10;
            result.s2[3] = rt5*(63*rby*rbz*ray+7*rby*cyz+7*rbz*cyy)/10;
            result.s2[4] = rt5*(63*rby*rbz*rax+7*rby*cxz+7*rbz*cxy)/10;
            result.s2[10] = rt5*(63*ray*raz*rbz+7*ray*czz+7*raz*cyz)/10;
            result.s2[11] = rt5*(63*rax*raz*rbz+7*rax*czz+7*raz*cxz)/10;
            result.s2[12] = rt5*(63*rax*ray*rbz+7*rax*cyz+7*ray*cxz)/10;
            result.s2[15] = rt5*(63*ray*raz*rby+7*ray*czy+7*raz*cyy)/10;
            result.s2[16] = rt5*(63*rax*raz*rby+7*rax*czy+7*raz*cxy)/10;
            result.s2[17] = rt5*(63*rax*ray*rby+7*rax*cyy+7*ray*cxy)/10;
            result.s2[19] = 63.0/10.0*rt5*rax*ray*raz;
            result.s2[46] = rt5*(7*raz*rbz+czz)/10;
            result.s2[47] = rt5*(7*rbz*ray+cyz)/10;
            result.s2[50] = 7.0/10.0*rt5*ray*raz;
            result.s2[55] = rt5*(7*raz*rbz+czz)/10;
            result.s2[57] = rt5*(7*rbz*rax+cxz)/10;
            result.s2[60] = 7.0/10.0*rt5*rax*raz;
            result.s2[66] = rt5*(7*rbz*ray+cyz)/10;
            result.s2[67] = rt5*(7*rbz*rax+cxz)/10;
            result.s2[71] = 7.0/10.0*rt5*rax*ray;
            result.s2[79] = rt5*(7*raz*rby+czy)/10;
            result.s2[80] = rt5*(7*ray*rby+cyy)/10;
            result.s2[82] = 7.0/10.0*rt5*ray*raz;
            result.s2[88] = rt5*raz/10;
            result.s2[89] = rt5*ray/10;
            result.s2[91] = rt5*(7*raz*rby+czy)/10;
            result.s2[93] = rt5*(7*rax*rby+cxy)/10;
            result.s2[95] = 7.0/10.0*rt5*rax*raz;
            result.s2[100] = rt5*raz/10;
            result.s2[102] = rt5*rax/10;
            result.s2[105] = rt5*(7*ray*rby+cyy)/10;
            result.s2[106] = rt5*(7*rax*rby+cxy)/10;
            result.s2[109] = 7.0/10.0*rt5*rax*ray;
            result.s2[114] = rt5*ray/10;
            result.s2[115] = rt5*rax/10;
        }
    }
}

/**
 * Octopole-32s × Quadrupole-22c kernel
 * Orient case 252
 */
void octopole_32s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double cxx = sf.cxx();
    double cxy = sf.cxy();
    double cyx = sf.cyx();
    double cyy = sf.cyy();
    double czx = sf.czx();
    double czy = sf.czy();

    result.s0 = rt5*(63*rax*ray*raz*rbx*rbx-63*rax*ray*raz*rby*rby
        +14*rax*ray*rbx*czx-14*rax*ray*rby*czy+14*rax*raz*rbx*cyx
        -14*rax*raz*rby*cyy+14*ray*raz*rbx*cxx-14*ray*raz*rby*cxy
        +2*rax*cyx*czx-2*rax*cyy*czy+2*ray*cxx*czx-2*ray*cxy*czy
        +2*raz*cxx*cyx-2*raz*cxy*cyy)/20;

    if (level >= 1) {
        result.s1[0] = rt5*(63*ray*raz*rbx*rbx-63*ray*raz*rby*rby
            +14*ray*rbx*czx-14*ray*rby*czy+14*raz*rbx*cyx-14*raz*rby*cyy
            +2*cyx*czx-2*cyy*czy)/20;
        result.s1[1] = rt5*(63*rax*raz*rbx*rbx-63*rax*raz*rby*rby
            +14*rax*rbx*czx-14*rax*rby*czy+14*raz*rbx*cxx-14*raz*rby*cxy
            +2*cxx*czx-2*cxy*czy)/20;
        result.s1[2] = rt5*(63*rbx*rbx*rax*ray-63*rby*rby*rax*ray
            +14*rbx*rax*cyx-14*rby*rax*cyy+14*rbx*ray*cxx-14*rby*ray*cxy
            +2*cxx*cyx-2*cxy*cyy)/20;
        result.s1[3] = rt5*(126*rax*ray*raz*rbx+14*rax*ray*czx
            +14*rax*raz*cyx+14*ray*raz*cxx)/20;
        result.s1[4] = rt5*(-126*rax*ray*raz*rby-14*rax*ray*czy
            -14*rax*raz*cyy-14*ray*raz*cxy)/20;
        result.s1[6] = rt5*(14*ray*raz*rbx+2*ray*czx+2*raz*cyx)/20;
        result.s1[9] = rt5*(14*rax*raz*rbx+2*rax*czx+2*raz*cxx)/20;
        result.s1[12] = rt5*(14*rax*ray*rbx+2*rax*cyx+2*ray*cxx)/20;
        result.s1[7] = rt5*(-14*ray*raz*rby-2*ray*czy-2*raz*cyy)/20;
        result.s1[10] = rt5*(-14*rax*raz*rby-2*rax*czy-2*raz*cxy)/20;
        result.s1[13] = rt5*(-14*rax*ray*rby-2*rax*cyy-2*ray*cxy)/20;
        if (level >= 2) {
            result.s2[1] = rt5*(63*rbx*rbx*raz-63*rby*rby*raz+14*rbx*czx
                -14*rby*czy)/20;
            result.s2[3] = rt5*(63*rbx*rbx*ray-63*rby*rby*ray+14*rbx*cyx
                -14*rby*cyy)/20;
            result.s2[4] = rt5*(63*rbx*rbx*rax-63*rby*rby*rax+14*rbx*cxx
                -14*rby*cxy)/20;
            result.s2[6] = rt5*(126*ray*raz*rbx+14*ray*czx+14*raz*cyx)/20;
            result.s2[7] = rt5*(126*rax*raz*rbx+14*rax*czx+14*raz*cxx)/20;
            result.s2[8] = rt5*(126*rax*ray*rbx+14*rax*cyx+14*ray*cxx)/20;
            result.s2[9] = 63.0/10.0*rt5*rax*ray*raz;
            result.s2[10] = rt5*(-126*ray*raz*rby-14*ray*czy
                -14*raz*cyy)/20;
            result.s2[11] = rt5*(-126*rax*raz*rby-14*rax*czy
                -14*raz*cxy)/20;
            result.s2[12] = rt5*(-126*rax*ray*rby-14*rax*cyy
                -14*ray*cxy)/20;
            result.s2[14] = -63.0/10.0*rt5*rax*ray*raz;
            result.s2[22] = rt5*(14*raz*rbx+2*czx)/20;
            result.s2[23] = rt5*(14*rbx*ray+2*cyx)/20;
            result.s2[24] = 7.0/10.0*rt5*ray*raz;
            result.s2[28] = rt5*(14*raz*rbx+2*czx)/20;
            result.s2[30] = rt5*(14*rax*rbx+2*cxx)/20;
            result.s2[31] = 7.0/10.0*rt5*rax*raz;
            result.s2[34] = rt5*raz/10;
            result.s2[36] = rt5*(14*rbx*ray+2*cyx)/20;
            result.s2[37] = rt5*(14*rax*rbx+2*cxx)/20;
            result.s2[39] = 7.0/10.0*rt5*rax*ray;
            result.s2[42] = rt5*ray/10;
            result.s2[43] = rt5*rax/10;
            result.s2[46] = rt5*(-14*raz*rby-2*czy)/20;
            result.s2[47] = rt5*(-14*ray*rby-2*cyy)/20;
            result.s2[49] = -7.0/10.0*rt5*ray*raz;
            result.s2[55] = rt5*(-14*raz*rby-2*czy)/20;
            result.s2[57] = rt5*(-14*rax*rby-2*cxy)/20;
            result.s2[59] = -7.0/10.0*rt5*rax*raz;
            result.s2[64] = -rt5*raz/10;
            result.s2[66] = rt5*(-14*ray*rby-2*cyy)/20;
            result.s2[67] = rt5*(-14*rax*rby-2*cxy)/20;
            result.s2[70] = -7.0/10.0*rt5*rax*ray;
            result.s2[75] = -rt5*ray/10;
            result.s2[76] = -rt5*rax/10;
        }
    }
}

/**
 * Octopole-32s × Quadrupole-22s kernel
 * Orient case 253
 */
void octopole_32s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax = sf.rax();
    double ray = sf.ray();
    double raz = sf.raz();
    double rbx = sf.rbx();
    double rby = sf.rby();
    double cxx = sf.cxx();
    double cxy = sf.cxy();
    double cyx = sf.cyx();
    double cyy = sf.cyy();
    double czx = sf.czx();
    double czy = sf.czy();

    result.s0 = rt5*(63*rax*ray*raz*rbx*rby+7*rax*ray*rbx*czy
        +7*rax*ray*rby*czx+7*rax*raz*rbx*cyy+7*rax*raz*rby*cyx
        +7*ray*raz*rbx*cxy+7*ray*raz*rby*cxx+rax*cyx*czy+rax*cyy*czx
        +ray*cxx*czy+ray*cxy*czx+raz*cxx*cyy+raz*cxy*cyx)/10;

    if (level >= 1) {
        result.s1[0] = rt5*(63*ray*raz*rbx*rby+7*ray*rbx*czy
            +7*ray*rby*czx+7*raz*rbx*cyy+7*raz*rby*cyx+cyx*czy
            +cyy*czx)/10;
        result.s1[1] = rt5*(63*rax*raz*rbx*rby+7*rax*rbx*czy
            +7*rax*rby*czx+7*raz*rbx*cxy+7*raz*rby*cxx+cxx*czy
            +cxy*czx)/10;
        result.s1[2] = rt5*(63*rax*ray*rbx*rby+7*rax*rbx*cyy
            +7*rax*rby*cyx+7*ray*rbx*cxy+7*ray*rby*cxx+cxx*cyy
            +cxy*cyx)/10;
        result.s1[3] = rt5*(63*rax*ray*raz*rby+7*rax*ray*czy
            +7*rax*raz*cyy+7*ray*raz*cxy)/10;
        result.s1[4] = rt5*(63*rax*ray*raz*rbx+7*rax*ray*czx
            +7*rax*raz*cyx+7*ray*raz*cxx)/10;
        result.s1[6] = rt5*(7*ray*raz*rby+ray*czy+raz*cyy)/10;
        result.s1[9] = rt5*(7*rax*raz*rby+rax*czy+raz*cxy)/10;
        result.s1[12] = rt5*(7*rax*ray*rby+rax*cyy+ray*cxy)/10;
        result.s1[7] = rt5*(7*ray*raz*rbx+ray*czx+raz*cyx)/10;
        result.s1[10] = rt5*(7*rax*raz*rbx+rax*czx+raz*cxx)/10;
        result.s1[13] = rt5*(7*rax*ray*rbx+rax*cyx+ray*cxx)/10;
        if (level >= 2) {
            result.s2[1] = rt5*(63*rbx*rby*raz+7*rbx*czy+7*rby*czx)/10;
            result.s2[3] = rt5*(63*rbx*rby*ray+7*rbx*cyy+7*rby*cyx)/10;
            result.s2[4] = rt5*(63*rbx*rby*rax+7*rbx*cxy+7*rby*cxx)/10;
            result.s2[6] = rt5*(63*ray*raz*rby+7*ray*czy+7*raz*cyy)/10;
            result.s2[7] = rt5*(63*rax*raz*rby+7*rax*czy+7*raz*cxy)/10;
            result.s2[8] = rt5*(63*rax*ray*rby+7*rax*cyy+7*ray*cxy)/10;
            result.s2[10] = rt5*(63*ray*raz*rbx+7*ray*czx+7*raz*cyx)/10;
            result.s2[11] = rt5*(63*rax*raz*rbx+7*rax*czx+7*raz*cxx)/10;
            result.s2[12] = rt5*(63*rax*ray*rbx+7*rax*cyx+7*ray*cxx)/10;
            result.s2[13] = 63.0/10.0*rt5*rax*ray*raz;
            result.s2[22] = rt5*(7*raz*rby+czy)/10;
            result.s2[23] = rt5*(7*ray*rby+cyy)/10;
            result.s2[25] = 7.0/10.0*rt5*ray*raz;
            result.s2[28] = rt5*(7*raz*rby+czy)/10;
            result.s2[30] = rt5*(7*rax*rby+cxy)/10;
            result.s2[32] = 7.0/10.0*rt5*rax*raz;
            result.s2[36] = rt5*(7*ray*rby+cyy)/10;
            result.s2[37] = rt5*(7*rax*rby+cxy)/10;
            result.s2[40] = 7.0/10.0*rt5*rax*ray;
            result.s2[46] = rt5*(7*raz*rbx+czx)/10;
            result.s2[47] = rt5*(7*rbx*ray+cyx)/10;
            result.s2[48] = 7.0/10.0*rt5*ray*raz;
            result.s2[52] = rt5*raz/10;
            result.s2[53] = rt5*ray/10;
            result.s2[55] = rt5*(7*raz*rbx+czx)/10;
            result.s2[57] = rt5*(7*rax*rbx+cxx)/10;
            result.s2[58] = 7.0/10.0*rt5*rax*raz;
            result.s2[61] = rt5*raz/10;
            result.s2[63] = rt5*rax/10;
            result.s2[66] = rt5*(7*rbx*ray+cyx)/10;
            result.s2[67] = rt5*(7*rax*rbx+cxx)/10;
            result.s2[69] = 7.0/10.0*rt5*rax*ray;
            result.s2[72] = rt5*ray/10;
            result.s2[73] = rt5*rax/10;
        }
    }
}

/**
 * Octopole-33c × Quadrupole-20 kernel
 * Orient case 254: Q33c × Q20
 */
void octopole_33c_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt10 * (63 * rax_val * rax_val * rax_val * rbz_val * rbz_val -
                        189 * rax_val * ray_val * ray_val * rbz_val * rbz_val -
                        7 * rax_val * rax_val * rax_val +
                        21 * rax_val * ray_val * ray_val +
                        42 * rax_val * rax_val * rbz_val * cxz_val -
                        84 * rax_val * ray_val * rbz_val * cyz_val -
                        42 * ray_val * ray_val * rbz_val * cxz_val +
                        6 * rax_val * cxz_val * cxz_val -
                        6 * rax_val * cyz_val * cyz_val -
                        12 * ray_val * cxz_val * cyz_val) / 80;

    if (level >= 1) {
        result.s1[0] = rt10 * (189 * rbz_val * rbz_val * rax_val * rax_val -
                               189 * rbz_val * rbz_val * ray_val * ray_val -
                               21 * rax_val * rax_val +
                               21 * ray_val * ray_val +
                               84 * rbz_val * rax_val * cxz_val -
                               84 * rbz_val * ray_val * cyz_val +
                               6 * cxz_val * cxz_val -
                               6 * cyz_val * cyz_val) / 80;
        result.s1[1] = rt10 * (-378 * rbz_val * rbz_val * rax_val * ray_val +
                               42 * rax_val * ray_val -
                               84 * rbz_val * rax_val * cyz_val -
                               84 * rbz_val * ray_val * cxz_val -
                               12 * cxz_val * cyz_val) / 80;
        result.s1[5] = rt10 * (126 * rax_val * rax_val * rax_val * rbz_val -
                               378 * rax_val * ray_val * ray_val * rbz_val +
                               42 * rax_val * rax_val * cxz_val -
                               84 * rax_val * ray_val * cyz_val -
                               42 * ray_val * ray_val * cxz_val) / 80;
        result.s1[8] = rt10 * (42 * rax_val * rax_val * rbz_val -
                                42 * ray_val * ray_val * rbz_val +
                                12 * rax_val * cxz_val -
                                12 * ray_val * cyz_val) / 80;
        result.s1[11] = rt10 * (-84 * rax_val * ray_val * rbz_val -
                                12 * rax_val * cyz_val -
                                12 * ray_val * cxz_val) / 80;

        if (level >= 2) {
            result.s2[0] = rt10 * (378 * rbz_val * rbz_val * rax_val -
                                   42 * rax_val +
                                   84 * rbz_val * cxz_val) / 80;
            result.s2[1] = rt10 * (-378 * rbz_val * rbz_val * ray_val +
                                   42 * ray_val -
                                   84 * rbz_val * cyz_val) / 80;
            result.s2[2] = rt10 * (-378 * rbz_val * rbz_val * rax_val +
                                   42 * rax_val -
                                   84 * rbz_val * cxz_val) / 80;
            result.s2[15] = rt10 * (378 * rax_val * rax_val * rbz_val -
                                    378 * ray_val * ray_val * rbz_val +
                                    84 * rax_val * cxz_val -
                                    84 * ray_val * cyz_val) / 80;
            result.s2[16] = rt10 * (-756 * rax_val * ray_val * rbz_val -
                                    84 * rax_val * cyz_val -
                                    84 * ray_val * cxz_val) / 80;
            result.s2[20] = rt10 * (126 * rax_val * rax_val * rax_val -
                                    378 * rax_val * ray_val * ray_val) / 80;
            result.s2[78] = rt10 * (84 * rbz_val * rax_val +
                                    12 * cxz_val) / 80;
            result.s2[79] = rt10 * (-84 * rbz_val * ray_val -
                                    12 * cyz_val) / 80;
            result.s2[83] = rt10 * (42 * rax_val * rax_val -
                                    42 * ray_val * ray_val) / 80;
            result.s2[90] = 3.0 / 20.0 * rt10 * rax_val;
            result.s2[91] = rt10 * (-84 * rbz_val * ray_val -
                                    12 * cyz_val) / 80;
            result.s2[92] = rt10 * (-84 * rbz_val * rax_val -
                                    12 * cxz_val) / 80;
            result.s2[96] = -21.0 / 20.0 * rt10 * rax_val * ray_val;
            result.s2[103] = -3.0 / 20.0 * rt10 * ray_val;
            result.s2[104] = -3.0 / 20.0 * rt10 * rax_val;
        }
    }
}

/**
 * Octopole-33c × Quadrupole-21c kernel
 * Orient case 255: Q33c × Q21c
 */
void octopole_33c_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt30 * (21 * rax_val * rax_val * rax_val * rbx_val * rbz_val -
                        63 * rax_val * ray_val * ray_val * rbx_val * rbz_val +
                        7 * rax_val * rax_val * rbx_val * cxz_val +
                        7 * rax_val * rax_val * rbz_val * cxx_val -
                        14 * rax_val * ray_val * rbx_val * cyz_val -
                        14 * rax_val * ray_val * rbz_val * cyx_val -
                        7 * ray_val * ray_val * rbx_val * cxz_val -
                        7 * ray_val * ray_val * rbz_val * cxx_val +
                        2 * rax_val * cxx_val * cxz_val -
                        2 * rax_val * cyx_val * cyz_val -
                        2 * ray_val * cxx_val * cyz_val -
                        2 * ray_val * cxz_val * cyx_val) / 40;

    if (level >= 1) {
        result.s1[0] = rt30 * (63 * rbx_val * rbz_val * rax_val * rax_val -
                               63 * rbx_val * rbz_val * ray_val * ray_val +
                               14 * rbx_val * rax_val * cxz_val +
                               14 * rbz_val * rax_val * cxx_val -
                               14 * rbx_val * ray_val * cyz_val -
                               14 * rbz_val * ray_val * cyx_val +
                               2 * cxx_val * cxz_val -
                               2 * cyx_val * cyz_val) / 40;
        result.s1[1] = rt30 * (-126 * rbx_val * rbz_val * rax_val * ray_val -
                               14 * rbx_val * rax_val * cyz_val -
                               14 * rbz_val * rax_val * cyx_val -
                               14 * rbx_val * ray_val * cxz_val -
                               14 * rbz_val * ray_val * cxx_val -
                               2 * cxx_val * cyz_val -
                               2 * cyx_val * cxz_val) / 40;
        result.s1[3] = rt30 * (21 * rax_val * rax_val * rax_val * rbz_val -
                               63 * rax_val * ray_val * ray_val * rbz_val +
                               7 * rax_val * rax_val * cxz_val -
                               14 * rax_val * ray_val * cyz_val -
                               7 * ray_val * ray_val * cxz_val) / 40;
        result.s1[5] = rt30 * (21 * rax_val * rax_val * rax_val * rbx_val -
                               63 * rax_val * ray_val * ray_val * rbx_val +
                               7 * rax_val * rax_val * cxx_val -
                               14 * rax_val * ray_val * cyx_val -
                               7 * ray_val * ray_val * cxx_val) / 40;
        result.s1[6] = rt30 * (7 * rax_val * rax_val * rbz_val -
                               7 * ray_val * ray_val * rbz_val +
                               2 * rax_val * cxz_val -
                               2 * ray_val * cyz_val) / 40;
        result.s1[9] = rt30 * (-14 * rax_val * ray_val * rbz_val -
                               2 * rax_val * cyz_val -
                               2 * ray_val * cxz_val) / 40;
        result.s1[8] = rt30 * (7 * rax_val * rax_val * rbx_val -
                                7 * ray_val * ray_val * rbx_val +
                                2 * rax_val * cxx_val -
                                2 * ray_val * cyx_val) / 40;
        result.s1[11] = rt30 * (-14 * rax_val * ray_val * rbx_val -
                                2 * rax_val * cyx_val -
                                2 * ray_val * cxx_val) / 40;

        if (level >= 2) {
            result.s2[0] = rt30 * (126 * rbx_val * rbz_val * rax_val +
                                   14 * rbx_val * cxz_val +
                                   14 * rbz_val * cxx_val) / 40;
            result.s2[1] = rt30 * (-126 * rbx_val * rbz_val * ray_val -
                                   14 * rbx_val * cyz_val -
                                   14 * rbz_val * cyx_val) / 40;
            result.s2[2] = rt30 * (-126 * rbx_val * rbz_val * rax_val -
                                   14 * rbx_val * cxz_val -
                                   14 * rbz_val * cxx_val) / 40;
            result.s2[6] = rt30 * (63 * rax_val * rax_val * rbz_val -
                                   63 * ray_val * ray_val * rbz_val +
                                   14 * rax_val * cxz_val -
                                   14 * ray_val * cyz_val) / 40;
            result.s2[7] = rt30 * (-126 * rax_val * ray_val * rbz_val -
                                   14 * rax_val * cyz_val -
                                   14 * ray_val * cxz_val) / 40;
            result.s2[15] = rt30 * (63 * rax_val * rax_val * rbx_val -
                                    63 * ray_val * ray_val * rbx_val +
                                    14 * rax_val * cxx_val -
                                    14 * ray_val * cyx_val) / 40;
            result.s2[16] = rt30 * (-126 * rax_val * ray_val * rbx_val -
                                    14 * rax_val * cyx_val -
                                    14 * ray_val * cxx_val) / 40;
            result.s2[18] = rt30 * (21 * rax_val * rax_val * rax_val -
                                    63 * rax_val * ray_val * ray_val) / 40;
            result.s2[21] = rt30 * (14 * rbz_val * rax_val +
                                    2 * cxz_val) / 40;
            result.s2[22] = rt30 * (-14 * rbz_val * ray_val -
                                    2 * cyz_val) / 40;
            result.s2[26] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[28] = rt30 * (-14 * rbz_val * ray_val -
                                    2 * cyz_val) / 40;
            result.s2[29] = rt30 * (-14 * rbz_val * rax_val -
                                    2 * cxz_val) / 40;
            result.s2[33] = -7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[78] = rt30 * (14 * rax_val * rbx_val +
                                    2 * cxx_val) / 40;
            result.s2[79] = rt30 * (-14 * rbx_val * ray_val -
                                    2 * cyx_val) / 40;
            result.s2[81] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[84] = rt30 * rax_val / 20;
            result.s2[85] = -rt30 * ray_val / 20;
            result.s2[91] = rt30 * (-14 * rbx_val * ray_val -
                                    2 * cyx_val) / 40;
            result.s2[92] = rt30 * (-14 * rax_val * rbx_val -
                                    2 * cxx_val) / 40;
            result.s2[94] = -7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[97] = -rt30 * ray_val / 20;
            result.s2[98] = -rt30 * rax_val / 20;
        }
    }
}

/**
 * Octopole-33c × Quadrupole-21s kernel
 * Orient case 256: Q33c × Q21s
 */
void octopole_33c_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt30 * (21 * rax_val * rax_val * rax_val * rby_val * rbz_val -
                        63 * rax_val * ray_val * ray_val * rby_val * rbz_val +
                        7 * rax_val * rax_val * rby_val * cxz_val +
                        7 * rax_val * rax_val * rbz_val * cxy_val -
                        14 * rax_val * ray_val * rby_val * cyz_val -
                        14 * rax_val * ray_val * rbz_val * cyy_val -
                        7 * ray_val * ray_val * rby_val * cxz_val -
                        7 * ray_val * ray_val * rbz_val * cxy_val +
                        2 * rax_val * cxy_val * cxz_val -
                        2 * rax_val * cyy_val * cyz_val -
                        2 * ray_val * cxy_val * cyz_val -
                        2 * ray_val * cxz_val * cyy_val) / 40;

    if (level >= 1) {
        result.s1[0] = rt30 * (63 * rby_val * rbz_val * rax_val * rax_val -
                               63 * rby_val * rbz_val * ray_val * ray_val +
                               14 * rby_val * rax_val * cxz_val +
                               14 * rbz_val * rax_val * cxy_val -
                               14 * rby_val * ray_val * cyz_val -
                               14 * rbz_val * ray_val * cyy_val +
                               2 * cxy_val * cxz_val -
                               2 * cyy_val * cyz_val) / 40;
        result.s1[1] = rt30 * (-126 * rby_val * rbz_val * rax_val * ray_val -
                               14 * rby_val * rax_val * cyz_val -
                               14 * rbz_val * rax_val * cyy_val -
                               14 * rby_val * ray_val * cxz_val -
                               14 * rbz_val * ray_val * cxy_val -
                               2 * cxy_val * cyz_val -
                               2 * cyy_val * cxz_val) / 40;
        result.s1[4] = rt30 * (21 * rax_val * rax_val * rax_val * rbz_val -
                               63 * rax_val * ray_val * ray_val * rbz_val +
                               7 * rax_val * rax_val * cxz_val -
                               14 * rax_val * ray_val * cyz_val -
                               7 * ray_val * ray_val * cxz_val) / 40;
        result.s1[5] = rt30 * (21 * rax_val * rax_val * rax_val * rby_val -
                               63 * rax_val * ray_val * ray_val * rby_val +
                               7 * rax_val * rax_val * cxy_val -
                               14 * rax_val * ray_val * cyy_val -
                               7 * ray_val * ray_val * cxy_val) / 40;
        result.s1[7] = rt30 * (7 * rax_val * rax_val * rbz_val -
                               7 * ray_val * ray_val * rbz_val +
                               2 * rax_val * cxz_val -
                               2 * ray_val * cyz_val) / 40;
        result.s1[10] = rt30 * (-14 * rax_val * ray_val * rbz_val -
                                2 * rax_val * cyz_val -
                                2 * ray_val * cxz_val) / 40;
        result.s1[8] = rt30 * (7 * rax_val * rax_val * rby_val -
                                7 * ray_val * ray_val * rby_val +
                                2 * rax_val * cxy_val -
                                2 * ray_val * cyy_val) / 40;
        result.s1[11] = rt30 * (-14 * rax_val * ray_val * rby_val -
                                2 * rax_val * cyy_val -
                                2 * ray_val * cxy_val) / 40;

        if (level >= 2) {
            result.s2[0] = rt30 * (126 * rby_val * rbz_val * rax_val +
                                   14 * rby_val * cxz_val +
                                   14 * rbz_val * cxy_val) / 40;
            result.s2[1] = rt30 * (-126 * rby_val * rbz_val * ray_val -
                                   14 * rby_val * cyz_val -
                                   14 * rbz_val * cyy_val) / 40;
            result.s2[2] = rt30 * (-126 * rby_val * rbz_val * rax_val -
                                   14 * rby_val * cxz_val -
                                   14 * rbz_val * cxy_val) / 40;
            result.s2[10] = rt30 * (63 * rax_val * rax_val * rbz_val -
                                    63 * ray_val * ray_val * rbz_val +
                                    14 * rax_val * cxz_val -
                                    14 * ray_val * cyz_val) / 40;
            result.s2[11] = rt30 * (-126 * rax_val * ray_val * rbz_val -
                                    14 * rax_val * cyz_val -
                                    14 * ray_val * cxz_val) / 40;
            result.s2[15] = rt30 * (63 * rax_val * rax_val * rby_val -
                                    63 * ray_val * ray_val * rby_val +
                                    14 * rax_val * cxy_val -
                                    14 * ray_val * cyy_val) / 40;
            result.s2[16] = rt30 * (-126 * rax_val * ray_val * rby_val -
                                    14 * rax_val * cyy_val -
                                    14 * ray_val * cxy_val) / 40;
            result.s2[19] = rt30 * (21 * rax_val * rax_val * rax_val -
                                    63 * rax_val * ray_val * ray_val) / 40;
            result.s2[45] = rt30 * (14 * rbz_val * rax_val +
                                    2 * cxz_val) / 40;
            result.s2[46] = rt30 * (-14 * rbz_val * ray_val -
                                    2 * cyz_val) / 40;
            result.s2[50] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[55] = rt30 * (-14 * rbz_val * ray_val -
                                    2 * cyz_val) / 40;
            result.s2[56] = rt30 * (-14 * rbz_val * rax_val -
                                    2 * cxz_val) / 40;
            result.s2[60] = -7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[78] = rt30 * (14 * rax_val * rby_val +
                                    2 * cxy_val) / 40;
            result.s2[79] = rt30 * (-14 * ray_val * rby_val -
                                    2 * cyy_val) / 40;
            result.s2[82] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[87] = rt30 * rax_val / 20;
            result.s2[88] = -rt30 * ray_val / 20;
            result.s2[91] = rt30 * (-14 * ray_val * rby_val -
                                    2 * cyy_val) / 40;
            result.s2[92] = rt30 * (-14 * rax_val * rby_val -
                                    2 * cxy_val) / 40;
            result.s2[95] = -7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[100] = -rt30 * ray_val / 20;
            result.s2[101] = -rt30 * rax_val / 20;
        }
    }
}

/**
 * Octopole-33c × Quadrupole-22c kernel
 * Orient case 257: Q33c × Q22c
 */
void octopole_33c_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt30 * (21 * rax_val * rax_val * rax_val * rbx_val * rbx_val -
                        21 * rax_val * rax_val * rax_val * rby_val * rby_val -
                        63 * rax_val * ray_val * ray_val * rbx_val * rbx_val +
                        63 * rax_val * ray_val * ray_val * rby_val * rby_val +
                        14 * rax_val * rax_val * rbx_val * cxx_val -
                        14 * rax_val * rax_val * rby_val * cxy_val -
                        28 * rax_val * ray_val * rbx_val * cyx_val +
                        28 * rax_val * ray_val * rby_val * cyy_val -
                        14 * ray_val * ray_val * rbx_val * cxx_val +
                        14 * ray_val * ray_val * rby_val * cxy_val +
                        2 * rax_val * cxx_val * cxx_val -
                        2 * rax_val * cxy_val * cxy_val -
                        2 * rax_val * cyx_val * cyx_val +
                        2 * rax_val * cyy_val * cyy_val -
                        4 * ray_val * cxx_val * cyx_val +
                        4 * ray_val * cxy_val * cyy_val) / 80;

    if (level >= 1) {
        result.s1[0] = rt30 * (63 * rax_val * rax_val * rbx_val * rbx_val -
                               63 * rax_val * rax_val * rby_val * rby_val -
                               63 * ray_val * ray_val * rbx_val * rbx_val +
                               63 * ray_val * ray_val * rby_val * rby_val +
                               28 * rax_val * rbx_val * cxx_val -
                               28 * rax_val * rby_val * cxy_val -
                               28 * ray_val * rbx_val * cyx_val +
                               28 * ray_val * rby_val * cyy_val +
                               2 * cxx_val * cxx_val -
                               2 * cxy_val * cxy_val -
                               2 * cyx_val * cyx_val +
                               2 * cyy_val * cyy_val) / 80;
        result.s1[1] = rt30 * (-126 * rbx_val * rbx_val * rax_val * ray_val +
                               126 * rby_val * rby_val * rax_val * ray_val -
                               28 * rbx_val * rax_val * cyx_val +
                               28 * rby_val * rax_val * cyy_val -
                               28 * rbx_val * ray_val * cxx_val +
                               28 * rby_val * ray_val * cxy_val -
                               4 * cxx_val * cyx_val +
                               4 * cxy_val * cyy_val) / 80;
        result.s1[3] = rt30 * (42 * rax_val * rax_val * rax_val * rbx_val -
                               126 * rax_val * ray_val * ray_val * rbx_val +
                               14 * rax_val * rax_val * cxx_val -
                               28 * rax_val * ray_val * cyx_val -
                               14 * ray_val * ray_val * cxx_val) / 80;
        result.s1[4] = rt30 * (-42 * rax_val * rax_val * rax_val * rby_val +
                               126 * rax_val * ray_val * ray_val * rby_val -
                               14 * rax_val * rax_val * cxy_val +
                               28 * rax_val * ray_val * cyy_val +
                               14 * ray_val * ray_val * cxy_val) / 80;
        result.s1[6] = rt30 * (14 * rax_val * rax_val * rbx_val -
                               14 * ray_val * ray_val * rbx_val +
                               4 * rax_val * cxx_val -
                               4 * ray_val * cyx_val) / 80;
        result.s1[9] = rt30 * (-28 * rax_val * ray_val * rbx_val -
                               4 * rax_val * cyx_val -
                               4 * ray_val * cxx_val) / 80;
        result.s1[7] = rt30 * (-14 * rax_val * rax_val * rby_val +
                               14 * ray_val * ray_val * rby_val -
                               4 * rax_val * cxy_val +
                               4 * ray_val * cyy_val) / 80;
        result.s1[10] = rt30 * (28 * rax_val * ray_val * rby_val +
                                4 * rax_val * cyy_val +
                                4 * ray_val * cxy_val) / 80;

        if (level >= 2) {
            result.s2[0] = rt30 * (126 * rbx_val * rbx_val * rax_val -
                                   126 * rby_val * rby_val * rax_val +
                                   28 * rbx_val * cxx_val -
                                   28 * rby_val * cxy_val) / 80;
            result.s2[1] = rt30 * (-126 * rbx_val * rbx_val * ray_val +
                                   126 * rby_val * rby_val * ray_val -
                                   28 * rbx_val * cyx_val +
                                   28 * rby_val * cyy_val) / 80;
            result.s2[2] = rt30 * (-126 * rbx_val * rbx_val * rax_val +
                                   126 * rby_val * rby_val * rax_val -
                                   28 * rbx_val * cxx_val +
                                   28 * rby_val * cxy_val) / 80;
            result.s2[6] = rt30 * (126 * rax_val * rax_val * rbx_val -
                                   126 * ray_val * ray_val * rbx_val +
                                   28 * rax_val * cxx_val -
                                   28 * ray_val * cyx_val) / 80;
            result.s2[7] = rt30 * (-252 * rax_val * ray_val * rbx_val -
                                   28 * rax_val * cyx_val -
                                   28 * ray_val * cxx_val) / 80;
            result.s2[9] = rt30 * (42 * rax_val * rax_val * rax_val -
                                   126 * rax_val * ray_val * ray_val) / 80;
            result.s2[10] = rt30 * (-126 * rax_val * rax_val * rby_val +
                                    126 * ray_val * ray_val * rby_val -
                                    28 * rax_val * cxy_val +
                                    28 * ray_val * cyy_val) / 80;
            result.s2[11] = rt30 * (252 * rax_val * ray_val * rby_val +
                                    28 * rax_val * cyy_val +
                                    28 * ray_val * cxy_val) / 80;
            result.s2[14] = rt30 * (-42 * rax_val * rax_val * rax_val +
                                    126 * rax_val * ray_val * ray_val) / 80;
            result.s2[21] = rt30 * (28 * rax_val * rbx_val +
                                    4 * cxx_val) / 80;
            result.s2[22] = rt30 * (-28 * rbx_val * ray_val -
                                    4 * cyx_val) / 80;
            result.s2[24] = rt30 * (14 * rax_val * rax_val -
                                    14 * ray_val * ray_val) / 80;
            result.s2[27] = rt30 * rax_val / 20;
            result.s2[28] = rt30 * (-28 * rbx_val * ray_val -
                                    4 * cyx_val) / 80;
            result.s2[29] = rt30 * (-28 * rax_val * rbx_val -
                                    4 * cxx_val) / 80;
            result.s2[31] = -7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[34] = -rt30 * ray_val / 20;
            result.s2[35] = -rt30 * rax_val / 20;
            result.s2[45] = rt30 * (-28 * rax_val * rby_val -
                                    4 * cxy_val) / 80;
            result.s2[46] = rt30 * (28 * ray_val * rby_val +
                                    4 * cyy_val) / 80;
            result.s2[49] = rt30 * (-14 * rax_val * rax_val +
                                    14 * ray_val * ray_val) / 80;
            result.s2[54] = -rt30 * rax_val / 20;
            result.s2[55] = rt30 * (28 * ray_val * rby_val +
                                    4 * cyy_val) / 80;
            result.s2[56] = rt30 * (28 * rax_val * rby_val +
                                    4 * cxy_val) / 80;
            result.s2[59] = 7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[64] = rt30 * ray_val / 20;
            result.s2[65] = rt30 * rax_val / 20;
        }
    }
}

/**
 * Octopole-33c × Quadrupole-22s kernel
 * Orient case 258: Q33c × Q22s
 */
void octopole_33c_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt30 * (21 * rax_val * rax_val * rax_val * rbx_val * rby_val -
                        63 * rax_val * ray_val * ray_val * rbx_val * rby_val +
                        7 * rax_val * rax_val * rbx_val * cxy_val +
                        7 * rax_val * rax_val * rby_val * cxx_val -
                        14 * rax_val * ray_val * rbx_val * cyy_val -
                        14 * rax_val * ray_val * rby_val * cyx_val -
                        7 * ray_val * ray_val * rbx_val * cxy_val -
                        7 * ray_val * ray_val * rby_val * cxx_val +
                        2 * rax_val * cxx_val * cxy_val -
                        2 * rax_val * cyx_val * cyy_val -
                        2 * ray_val * cxx_val * cyy_val -
                        2 * ray_val * cxy_val * cyx_val) / 40;

    if (level >= 1) {
        result.s1[0] = rt30 * (63 * rax_val * rax_val * rbx_val * rby_val -
                               63 * ray_val * ray_val * rbx_val * rby_val +
                               14 * rax_val * rbx_val * cxy_val +
                               14 * rax_val * rby_val * cxx_val -
                               14 * ray_val * rbx_val * cyy_val -
                               14 * ray_val * rby_val * cyx_val +
                               2 * cxx_val * cxy_val -
                               2 * cyx_val * cyy_val) / 40;
        result.s1[1] = rt30 * (-126 * rax_val * ray_val * rbx_val * rby_val -
                               14 * rax_val * rbx_val * cyy_val -
                               14 * rax_val * rby_val * cyx_val -
                               14 * ray_val * rbx_val * cxy_val -
                               14 * ray_val * rby_val * cxx_val -
                               2 * cxx_val * cyy_val -
                               2 * cxy_val * cyx_val) / 40;
        result.s1[3] = rt30 * (21 * rax_val * rax_val * rax_val * rby_val -
                               63 * rax_val * ray_val * ray_val * rby_val +
                               7 * rax_val * rax_val * cxy_val -
                               14 * rax_val * ray_val * cyy_val -
                               7 * ray_val * ray_val * cxy_val) / 40;
        result.s1[4] = rt30 * (21 * rax_val * rax_val * rax_val * rbx_val -
                               63 * rax_val * ray_val * ray_val * rbx_val +
                               7 * rax_val * rax_val * cxx_val -
                               14 * rax_val * ray_val * cyx_val -
                               7 * ray_val * ray_val * cxx_val) / 40;
        result.s1[6] = rt30 * (7 * rax_val * rax_val * rby_val -
                               7 * ray_val * ray_val * rby_val +
                               2 * rax_val * cxy_val -
                               2 * ray_val * cyy_val) / 40;
        result.s1[9] = rt30 * (-14 * rax_val * ray_val * rby_val -
                               2 * rax_val * cyy_val -
                               2 * ray_val * cxy_val) / 40;
        result.s1[7] = rt30 * (7 * rax_val * rax_val * rbx_val -
                               7 * ray_val * ray_val * rbx_val +
                               2 * rax_val * cxx_val -
                               2 * ray_val * cyx_val) / 40;
        result.s1[10] = rt30 * (-14 * rax_val * ray_val * rbx_val -
                                2 * rax_val * cyx_val -
                                2 * ray_val * cxx_val) / 40;

        if (level >= 2) {
            result.s2[0] = rt30 * (126 * rbx_val * rby_val * rax_val +
                                   14 * rbx_val * cxy_val +
                                   14 * rby_val * cxx_val) / 40;
            result.s2[1] = rt30 * (-126 * rbx_val * rby_val * ray_val -
                                   14 * rbx_val * cyy_val -
                                   14 * rby_val * cyx_val) / 40;
            result.s2[2] = rt30 * (-126 * rbx_val * rby_val * rax_val -
                                   14 * rbx_val * cxy_val -
                                   14 * rby_val * cxx_val) / 40;
            result.s2[6] = rt30 * (63 * rax_val * rax_val * rby_val -
                                   63 * ray_val * ray_val * rby_val +
                                   14 * rax_val * cxy_val -
                                   14 * ray_val * cyy_val) / 40;
            result.s2[7] = rt30 * (-126 * rax_val * ray_val * rby_val -
                                   14 * rax_val * cyy_val -
                                   14 * ray_val * cxy_val) / 40;
            result.s2[10] = rt30 * (63 * rax_val * rax_val * rbx_val -
                                    63 * ray_val * ray_val * rbx_val +
                                    14 * rax_val * cxx_val -
                                    14 * ray_val * cyx_val) / 40;
            result.s2[11] = rt30 * (-126 * rax_val * ray_val * rbx_val -
                                    14 * rax_val * cyx_val -
                                    14 * ray_val * cxx_val) / 40;
            result.s2[13] = rt30 * (21 * rax_val * rax_val * rax_val -
                                    63 * rax_val * ray_val * ray_val) / 40;
            result.s2[21] = rt30 * (14 * rax_val * rby_val +
                                    2 * cxy_val) / 40;
            result.s2[22] = rt30 * (-14 * ray_val * rby_val -
                                    2 * cyy_val) / 40;
            result.s2[25] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[28] = rt30 * (-14 * ray_val * rby_val -
                                    2 * cyy_val) / 40;
            result.s2[29] = rt30 * (-14 * rax_val * rby_val -
                                    2 * cxy_val) / 40;
            result.s2[32] = -7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[45] = rt30 * (14 * rax_val * rbx_val +
                                    2 * cxx_val) / 40;
            result.s2[46] = rt30 * (-14 * rbx_val * ray_val -
                                    2 * cyx_val) / 40;
            result.s2[48] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[51] = rt30 * rax_val / 20;
            result.s2[52] = -rt30 * ray_val / 20;
            result.s2[55] = rt30 * (-14 * rbx_val * ray_val -
                                    2 * cyx_val) / 40;
            result.s2[56] = rt30 * (-14 * rax_val * rbx_val -
                                    2 * cxx_val) / 40;
            result.s2[58] = -7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[61] = -rt30 * ray_val / 20;
            result.s2[62] = -rt30 * rax_val / 20;
        }
    }
}

/**
 * Octopole-33s × Quadrupole-20 kernel
 * Orient case 259: Q33s × Q20
 */
void octopole_33s_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt10 * (189 * rax_val * rax_val * ray_val * rbz_val * rbz_val -
                        63 * ray_val * ray_val * ray_val * rbz_val * rbz_val -
                        21 * rax_val * rax_val * ray_val +
                        7 * ray_val * ray_val * ray_val +
                        42 * rax_val * rax_val * rbz_val * cyz_val +
                        84 * rax_val * ray_val * rbz_val * cxz_val -
                        42 * ray_val * ray_val * rbz_val * cyz_val +
                        12 * rax_val * cxz_val * cyz_val +
                        6 * ray_val * cxz_val * cxz_val -
                        6 * ray_val * cyz_val * cyz_val) / 80;

    if (level >= 1) {
        result.s1[0] = rt10 * (378 * rbz_val * rbz_val * rax_val * ray_val -
                               42 * rax_val * ray_val +
                               84 * rbz_val * rax_val * cyz_val +
                               84 * rbz_val * ray_val * cxz_val +
                               12 * cxz_val * cyz_val) / 80;
        result.s1[1] = rt10 * (189 * rbz_val * rbz_val * rax_val * rax_val -
                               189 * rbz_val * rbz_val * ray_val * ray_val -
                               21 * rax_val * rax_val +
                               21 * ray_val * ray_val +
                               84 * rbz_val * rax_val * cxz_val -
                               84 * rbz_val * ray_val * cyz_val +
                               6 * cxz_val * cxz_val -
                               6 * cyz_val * cyz_val) / 80;
        result.s1[5] = rt10 * (378 * rax_val * rax_val * ray_val * rbz_val -
                               126 * ray_val * ray_val * ray_val * rbz_val +
                               42 * rax_val * rax_val * cyz_val +
                               84 * rax_val * ray_val * cxz_val -
                               42 * ray_val * ray_val * cyz_val) / 80;
        result.s1[8] = rt10 * (84 * rax_val * ray_val * rbz_val +
                                12 * rax_val * cyz_val +
                                12 * ray_val * cxz_val) / 80;
        result.s1[11] = rt10 * (42 * rax_val * rax_val * rbz_val -
                                42 * ray_val * ray_val * rbz_val +
                                12 * rax_val * cxz_val -
                                12 * ray_val * cyz_val) / 80;

        if (level >= 2) {
            result.s2[0] = rt10 * (378 * rbz_val * rbz_val * ray_val -
                                   42 * ray_val +
                                   84 * rbz_val * cyz_val) / 80;
            result.s2[1] = rt10 * (378 * rbz_val * rbz_val * rax_val -
                                   42 * rax_val +
                                   84 * rbz_val * cxz_val) / 80;
            result.s2[2] = rt10 * (-378 * rbz_val * rbz_val * ray_val +
                                   42 * ray_val -
                                   84 * rbz_val * cyz_val) / 80;
            result.s2[15] = rt10 * (756 * rax_val * ray_val * rbz_val +
                                    84 * rax_val * cyz_val +
                                    84 * ray_val * cxz_val) / 80;
            result.s2[16] = rt10 * (378 * rax_val * rax_val * rbz_val -
                                    378 * ray_val * ray_val * rbz_val +
                                    84 * rax_val * cxz_val -
                                    84 * ray_val * cyz_val) / 80;
            result.s2[20] = rt10 * (378 * rax_val * rax_val * ray_val -
                                    126 * ray_val * ray_val * ray_val) / 80;
            result.s2[78] = rt10 * (84 * rbz_val * ray_val +
                                    12 * cyz_val) / 80;
            result.s2[79] = rt10 * (84 * rbz_val * rax_val +
                                    12 * cxz_val) / 80;
            result.s2[83] = 21.0 / 20.0 * rt10 * rax_val * ray_val;
            result.s2[90] = 3.0 / 20.0 * rt10 * ray_val;
            result.s2[91] = rt10 * (84 * rbz_val * rax_val +
                                    12 * cxz_val) / 80;
            result.s2[92] = rt10 * (-84 * rbz_val * ray_val -
                                    12 * cyz_val) / 80;
            result.s2[96] = rt10 * (42 * rax_val * rax_val -
                                    42 * ray_val * ray_val) / 80;
            result.s2[103] = 3.0 / 20.0 * rt10 * rax_val;
            result.s2[104] = -3.0 / 20.0 * rt10 * ray_val;
        }
    }
}

/**
 * Octopole-33s × Quadrupole-21c kernel
 * Orient case 260: Q33s × Q21c
 */
void octopole_33s_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt30 * (63 * rax_val * rax_val * ray_val * rbx_val * rbz_val -
                        21 * ray_val * ray_val * ray_val * rbx_val * rbz_val +
                        7 * rax_val * rax_val * rbx_val * cyz_val +
                        7 * rax_val * rax_val * rbz_val * cyx_val +
                        14 * rax_val * ray_val * rbx_val * cxz_val +
                        14 * rax_val * ray_val * rbz_val * cxx_val -
                        7 * ray_val * ray_val * rbx_val * cyz_val -
                        7 * ray_val * ray_val * rbz_val * cyx_val +
                        2 * rax_val * cxx_val * cyz_val +
                        2 * rax_val * cxz_val * cyx_val +
                        2 * ray_val * cxx_val * cxz_val -
                        2 * ray_val * cyx_val * cyz_val) / 40;

    if (level >= 1) {
        result.s1[0] = rt30 * (126 * rbx_val * rbz_val * rax_val * ray_val +
                               14 * rbx_val * rax_val * cyz_val +
                               14 * rbz_val * rax_val * cyx_val +
                               14 * rbx_val * ray_val * cxz_val +
                               14 * rbz_val * ray_val * cxx_val +
                               2 * cxx_val * cyz_val +
                               2 * cyx_val * cxz_val) / 40;
        result.s1[1] = rt30 * (63 * rbx_val * rbz_val * rax_val * rax_val -
                               63 * rbx_val * rbz_val * ray_val * ray_val +
                               14 * rbx_val * rax_val * cxz_val +
                               14 * rbz_val * rax_val * cxx_val -
                               14 * rbx_val * ray_val * cyz_val -
                               14 * rbz_val * ray_val * cyx_val +
                               2 * cxx_val * cxz_val -
                               2 * cyx_val * cyz_val) / 40;
        result.s1[3] = rt30 * (63 * rax_val * rax_val * ray_val * rbz_val -
                               21 * ray_val * ray_val * ray_val * rbz_val +
                               7 * rax_val * rax_val * cyz_val +
                               14 * rax_val * ray_val * cxz_val -
                               7 * ray_val * ray_val * cyz_val) / 40;
        result.s1[5] = rt30 * (63 * rax_val * rax_val * ray_val * rbx_val -
                               21 * ray_val * ray_val * ray_val * rbx_val +
                               7 * rax_val * rax_val * cyx_val +
                               14 * rax_val * ray_val * cxx_val -
                               7 * ray_val * ray_val * cyx_val) / 40;
        result.s1[6] = rt30 * (14 * rax_val * ray_val * rbz_val +
                               2 * rax_val * cyz_val +
                               2 * ray_val * cxz_val) / 40;
        result.s1[9] = rt30 * (7 * rax_val * rax_val * rbz_val -
                               7 * ray_val * ray_val * rbz_val +
                               2 * rax_val * cxz_val -
                               2 * ray_val * cyz_val) / 40;
        result.s1[8] = rt30 * (14 * rax_val * ray_val * rbx_val +
                                2 * rax_val * cyx_val +
                                2 * ray_val * cxx_val) / 40;
        result.s1[11] = rt30 * (7 * rax_val * rax_val * rbx_val -
                                7 * ray_val * ray_val * rbx_val +
                                2 * rax_val * cxx_val -
                                2 * ray_val * cyx_val) / 40;

        if (level >= 2) {
            result.s2[0] = rt30 * (126 * rbx_val * rbz_val * ray_val +
                                   14 * rbx_val * cyz_val +
                                   14 * rbz_val * cyx_val) / 40;
            result.s2[1] = rt30 * (126 * rbx_val * rbz_val * rax_val +
                                   14 * rbx_val * cxz_val +
                                   14 * rbz_val * cxx_val) / 40;
            result.s2[2] = rt30 * (-126 * rbx_val * rbz_val * ray_val -
                                   14 * rbx_val * cyz_val -
                                   14 * rbz_val * cyx_val) / 40;
            result.s2[6] = rt30 * (126 * rax_val * ray_val * rbz_val +
                                   14 * rax_val * cyz_val +
                                   14 * ray_val * cxz_val) / 40;
            result.s2[7] = rt30 * (63 * rax_val * rax_val * rbz_val -
                                   63 * ray_val * ray_val * rbz_val +
                                   14 * rax_val * cxz_val -
                                   14 * ray_val * cyz_val) / 40;
            result.s2[15] = rt30 * (126 * rax_val * ray_val * rbx_val +
                                    14 * rax_val * cyx_val +
                                    14 * ray_val * cxx_val) / 40;
            result.s2[16] = rt30 * (63 * rax_val * rax_val * rbx_val -
                                    63 * ray_val * ray_val * rbx_val +
                                    14 * rax_val * cxx_val -
                                    14 * ray_val * cyx_val) / 40;
            result.s2[18] = rt30 * (63 * rax_val * rax_val * ray_val -
                                    21 * ray_val * ray_val * ray_val) / 40;
            result.s2[21] = rt30 * (14 * rbz_val * ray_val +
                                    2 * cyz_val) / 40;
            result.s2[22] = rt30 * (14 * rbz_val * rax_val +
                                    2 * cxz_val) / 40;
            result.s2[26] = 7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[28] = rt30 * (14 * rbz_val * rax_val +
                                    2 * cxz_val) / 40;
            result.s2[29] = rt30 * (-14 * rbz_val * ray_val -
                                    2 * cyz_val) / 40;
            result.s2[33] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[78] = rt30 * (14 * rbx_val * ray_val +
                                    2 * cyx_val) / 40;
            result.s2[79] = rt30 * (14 * rax_val * rbx_val +
                                    2 * cxx_val) / 40;
            result.s2[81] = 7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[84] = rt30 * ray_val / 20;
            result.s2[85] = rt30 * rax_val / 20;
            result.s2[91] = rt30 * (14 * rax_val * rbx_val +
                                    2 * cxx_val) / 40;
            result.s2[92] = rt30 * (-14 * rbx_val * ray_val -
                                    2 * cyx_val) / 40;
            result.s2[94] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[97] = rt30 * rax_val / 20;
            result.s2[98] = -rt30 * ray_val / 20;
        }
    }
}

/**
 * Octopole-33s × Quadrupole-21s kernel
 * Orient case 261: Q33s × Q21s
 */
void octopole_33s_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt30 * (63 * rax_val * rax_val * ray_val * rby_val * rbz_val -
                        21 * ray_val * ray_val * ray_val * rby_val * rbz_val +
                        7 * rax_val * rax_val * rby_val * cyz_val +
                        7 * rax_val * rax_val * rbz_val * cyy_val +
                        14 * rax_val * ray_val * rby_val * cxz_val +
                        14 * rax_val * ray_val * rbz_val * cxy_val -
                        7 * ray_val * ray_val * rby_val * cyz_val -
                        7 * ray_val * ray_val * rbz_val * cyy_val +
                        2 * rax_val * cxy_val * cyz_val +
                        2 * rax_val * cxz_val * cyy_val +
                        2 * ray_val * cxy_val * cxz_val -
                        2 * ray_val * cyy_val * cyz_val) / 40;

    if (level >= 1) {
        result.s1[0] = rt30 * (126 * rby_val * rbz_val * rax_val * ray_val +
                               14 * rby_val * rax_val * cyz_val +
                               14 * rbz_val * rax_val * cyy_val +
                               14 * rby_val * ray_val * cxz_val +
                               14 * rbz_val * ray_val * cxy_val +
                               2 * cxy_val * cyz_val +
                               2 * cyy_val * cxz_val) / 40;
        result.s1[1] = rt30 * (63 * rby_val * rbz_val * rax_val * rax_val -
                               63 * rby_val * rbz_val * ray_val * ray_val +
                               14 * rby_val * rax_val * cxz_val +
                               14 * rbz_val * rax_val * cxy_val -
                               14 * rby_val * ray_val * cyz_val -
                               14 * rbz_val * ray_val * cyy_val +
                               2 * cxy_val * cxz_val -
                               2 * cyy_val * cyz_val) / 40;
        result.s1[4] = rt30 * (63 * rax_val * rax_val * ray_val * rbz_val -
                               21 * ray_val * ray_val * ray_val * rbz_val +
                               7 * rax_val * rax_val * cyz_val +
                               14 * rax_val * ray_val * cxz_val -
                               7 * ray_val * ray_val * cyz_val) / 40;
        result.s1[5] = rt30 * (63 * rax_val * rax_val * ray_val * rby_val -
                               21 * ray_val * ray_val * ray_val * rby_val +
                               7 * rax_val * rax_val * cyy_val +
                               14 * rax_val * ray_val * cxy_val -
                               7 * ray_val * ray_val * cyy_val) / 40;
        result.s1[7] = rt30 * (14 * rax_val * ray_val * rbz_val +
                               2 * rax_val * cyz_val +
                               2 * ray_val * cxz_val) / 40;
        result.s1[10] = rt30 * (7 * rax_val * rax_val * rbz_val -
                                7 * ray_val * ray_val * rbz_val +
                                2 * rax_val * cxz_val -
                                2 * ray_val * cyz_val) / 40;
        result.s1[8] = rt30 * (14 * rax_val * ray_val * rby_val +
                                2 * rax_val * cyy_val +
                                2 * ray_val * cxy_val) / 40;
        result.s1[11] = rt30 * (7 * rax_val * rax_val * rby_val -
                                7 * ray_val * ray_val * rby_val +
                                2 * rax_val * cxy_val -
                                2 * ray_val * cyy_val) / 40;

        if (level >= 2) {
            result.s2[0] = rt30 * (126 * rby_val * rbz_val * ray_val +
                                   14 * rby_val * cyz_val +
                                   14 * rbz_val * cyy_val) / 40;
            result.s2[1] = rt30 * (126 * rby_val * rbz_val * rax_val +
                                   14 * rby_val * cxz_val +
                                   14 * rbz_val * cxy_val) / 40;
            result.s2[2] = rt30 * (-126 * rby_val * rbz_val * ray_val -
                                   14 * rby_val * cyz_val -
                                   14 * rbz_val * cyy_val) / 40;
            result.s2[10] = rt30 * (126 * rax_val * ray_val * rbz_val +
                                    14 * rax_val * cyz_val +
                                    14 * ray_val * cxz_val) / 40;
            result.s2[11] = rt30 * (63 * rax_val * rax_val * rbz_val -
                                    63 * ray_val * ray_val * rbz_val +
                                    14 * rax_val * cxz_val -
                                    14 * ray_val * cyz_val) / 40;
            result.s2[15] = rt30 * (126 * rax_val * ray_val * rby_val +
                                    14 * rax_val * cyy_val +
                                    14 * ray_val * cxy_val) / 40;
            result.s2[16] = rt30 * (63 * rax_val * rax_val * rby_val -
                                    63 * ray_val * ray_val * rby_val +
                                    14 * rax_val * cxy_val -
                                    14 * ray_val * cyy_val) / 40;
            result.s2[19] = rt30 * (63 * rax_val * rax_val * ray_val -
                                    21 * ray_val * ray_val * ray_val) / 40;
            result.s2[45] = rt30 * (14 * rbz_val * ray_val +
                                    2 * cyz_val) / 40;
            result.s2[46] = rt30 * (14 * rbz_val * rax_val +
                                    2 * cxz_val) / 40;
            result.s2[50] = 7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[55] = rt30 * (14 * rbz_val * rax_val +
                                    2 * cxz_val) / 40;
            result.s2[56] = rt30 * (-14 * rbz_val * ray_val -
                                    2 * cyz_val) / 40;
            result.s2[60] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[78] = rt30 * (14 * ray_val * rby_val +
                                    2 * cyy_val) / 40;
            result.s2[79] = rt30 * (14 * rax_val * rby_val +
                                    2 * cxy_val) / 40;
            result.s2[82] = 7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[87] = rt30 * ray_val / 20;
            result.s2[88] = rt30 * rax_val / 20;
            result.s2[91] = rt30 * (14 * rax_val * rby_val +
                                    2 * cxy_val) / 40;
            result.s2[92] = rt30 * (-14 * ray_val * rby_val -
                                    2 * cyy_val) / 40;
            result.s2[95] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[100] = rt30 * rax_val / 20;
            result.s2[101] = -rt30 * ray_val / 20;
        }
    }
}

/**
 * Octopole-33s × Quadrupole-22c kernel
 * Orient case 262: Q33s × Q22c
 */
void octopole_33s_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt30 * (63 * rax_val * rax_val * ray_val * rbx_val * rbx_val -
                        63 * rax_val * rax_val * ray_val * rby_val * rby_val -
                        21 * ray_val * ray_val * ray_val * rbx_val * rbx_val +
                        21 * ray_val * ray_val * ray_val * rby_val * rby_val +
                        14 * rax_val * rax_val * rbx_val * cyx_val -
                        14 * rax_val * rax_val * rby_val * cyy_val +
                        28 * rax_val * ray_val * rbx_val * cxx_val -
                        28 * rax_val * ray_val * rby_val * cxy_val -
                        14 * ray_val * ray_val * rbx_val * cyx_val +
                        14 * ray_val * ray_val * rby_val * cyy_val +
                        4 * rax_val * cxx_val * cyx_val -
                        4 * rax_val * cxy_val * cyy_val +
                        2 * ray_val * cxx_val * cxx_val -
                        2 * ray_val * cxy_val * cxy_val -
                        2 * ray_val * cyx_val * cyx_val +
                        2 * ray_val * cyy_val * cyy_val) / 80;

    if (level >= 1) {
        result.s1[0] = rt30 * (126 * rbx_val * rbx_val * rax_val * ray_val -
                               126 * rby_val * rby_val * rax_val * ray_val +
                               28 * rbx_val * rax_val * cyx_val -
                               28 * rby_val * rax_val * cyy_val +
                               28 * rbx_val * ray_val * cxx_val -
                               28 * rby_val * ray_val * cxy_val +
                               4 * cxx_val * cyx_val -
                               4 * cxy_val * cyy_val) / 80;
        result.s1[1] = rt30 * (63 * rax_val * rax_val * rbx_val * rbx_val -
                               63 * rax_val * rax_val * rby_val * rby_val -
                               63 * ray_val * ray_val * rbx_val * rbx_val +
                               63 * ray_val * ray_val * rby_val * rby_val +
                               28 * rax_val * rbx_val * cxx_val -
                               28 * rax_val * rby_val * cxy_val -
                               28 * ray_val * rbx_val * cyx_val +
                               28 * ray_val * rby_val * cyy_val +
                               2 * cxx_val * cxx_val -
                               2 * cxy_val * cxy_val -
                               2 * cyx_val * cyx_val +
                               2 * cyy_val * cyy_val) / 80;
        result.s1[3] = rt30 * (126 * rax_val * rax_val * ray_val * rbx_val -
                               42 * ray_val * ray_val * ray_val * rbx_val +
                               14 * rax_val * rax_val * cyx_val +
                               28 * rax_val * ray_val * cxx_val -
                               14 * ray_val * ray_val * cyx_val) / 80;
        result.s1[4] = rt30 * (-126 * rax_val * rax_val * ray_val * rby_val +
                               42 * ray_val * ray_val * ray_val * rby_val -
                               14 * rax_val * rax_val * cyy_val -
                               28 * rax_val * ray_val * cxy_val +
                               14 * ray_val * ray_val * cyy_val) / 80;
        result.s1[6] = rt30 * (28 * rax_val * ray_val * rbx_val +
                               4 * rax_val * cyx_val +
                               4 * ray_val * cxx_val) / 80;
        result.s1[9] = rt30 * (14 * rax_val * rax_val * rbx_val -
                               14 * ray_val * ray_val * rbx_val +
                               4 * rax_val * cxx_val -
                               4 * ray_val * cyx_val) / 80;
        result.s1[7] = rt30 * (-28 * rax_val * ray_val * rby_val -
                               4 * rax_val * cyy_val -
                               4 * ray_val * cxy_val) / 80;
        result.s1[10] = rt30 * (-14 * rax_val * rax_val * rby_val +
                                14 * ray_val * ray_val * rby_val -
                                4 * rax_val * cxy_val +
                                4 * ray_val * cyy_val) / 80;

        if (level >= 2) {
            result.s2[0] = rt30 * (126 * rbx_val * rbx_val * ray_val -
                                   126 * rby_val * rby_val * ray_val +
                                   28 * rbx_val * cyx_val -
                                   28 * rby_val * cyy_val) / 80;
            result.s2[1] = rt30 * (126 * rbx_val * rbx_val * rax_val -
                                   126 * rby_val * rby_val * rax_val +
                                   28 * rbx_val * cxx_val -
                                   28 * rby_val * cxy_val) / 80;
            result.s2[2] = rt30 * (-126 * rbx_val * rbx_val * ray_val +
                                   126 * rby_val * rby_val * ray_val -
                                   28 * rbx_val * cyx_val +
                                   28 * rby_val * cyy_val) / 80;
            result.s2[6] = rt30 * (252 * rax_val * ray_val * rbx_val +
                                   28 * rax_val * cyx_val +
                                   28 * ray_val * cxx_val) / 80;
            result.s2[7] = rt30 * (126 * rax_val * rax_val * rbx_val -
                                   126 * ray_val * ray_val * rbx_val +
                                   28 * rax_val * cxx_val -
                                   28 * ray_val * cyx_val) / 80;
            result.s2[9] = rt30 * (126 * rax_val * rax_val * ray_val -
                                   42 * ray_val * ray_val * ray_val) / 80;
            result.s2[10] = rt30 * (-252 * rax_val * ray_val * rby_val -
                                    28 * rax_val * cyy_val -
                                    28 * ray_val * cxy_val) / 80;
            result.s2[11] = rt30 * (-126 * rax_val * rax_val * rby_val +
                                    126 * ray_val * ray_val * rby_val -
                                    28 * rax_val * cxy_val +
                                    28 * ray_val * cyy_val) / 80;
            result.s2[14] = rt30 * (-126 * rax_val * rax_val * ray_val +
                                    42 * ray_val * ray_val * ray_val) / 80;
            result.s2[21] = rt30 * (28 * rbx_val * ray_val +
                                    4 * cyx_val) / 80;
            result.s2[22] = rt30 * (28 * rax_val * rbx_val +
                                    4 * cxx_val) / 80;
            result.s2[24] = 7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[27] = rt30 * ray_val / 20;
            result.s2[28] = rt30 * (28 * rax_val * rbx_val +
                                    4 * cxx_val) / 80;
            result.s2[29] = rt30 * (-28 * rbx_val * ray_val -
                                    4 * cyx_val) / 80;
            result.s2[31] = rt30 * (14 * rax_val * rax_val -
                                    14 * ray_val * ray_val) / 80;
            result.s2[34] = rt30 * rax_val / 20;
            result.s2[35] = -rt30 * ray_val / 20;
            result.s2[45] = rt30 * (-28 * ray_val * rby_val -
                                    4 * cyy_val) / 80;
            result.s2[46] = rt30 * (-28 * rax_val * rby_val -
                                    4 * cxy_val) / 80;
            result.s2[49] = -7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[54] = -rt30 * ray_val / 20;
            result.s2[55] = rt30 * (-28 * rax_val * rby_val -
                                    4 * cxy_val) / 80;
            result.s2[56] = rt30 * (28 * ray_val * rby_val +
                                    4 * cyy_val) / 80;
            result.s2[59] = rt30 * (-14 * rax_val * rax_val +
                                    14 * ray_val * ray_val) / 80;
            result.s2[64] = -rt30 * rax_val / 20;
            result.s2[65] = rt30 * ray_val / 20;
        }
    }
}

/**
 * Octopole-33s × Quadrupole-22s kernel
 * Orient case 263: Q33s × Q22s
 */
void octopole_33s_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cxy_val = sf.cxy(), cxz_val = sf.cxz();
    double cyx_val = sf.cyx(), cyy_val = sf.cyy(), cyz_val = sf.cyz();
    double czx_val = sf.czx(), czy_val = sf.czy(), czz_val = sf.czz();

    result.s0 = rt30 * (63 * rax_val * rax_val * ray_val * rbx_val * rby_val -
                        21 * ray_val * ray_val * ray_val * rbx_val * rby_val +
                        7 * rax_val * rax_val * rbx_val * cyy_val +
                        7 * rax_val * rax_val * rby_val * cyx_val +
                        14 * rax_val * ray_val * rbx_val * cxy_val +
                        14 * rax_val * ray_val * rby_val * cxx_val -
                        7 * ray_val * ray_val * rbx_val * cyy_val -
                        7 * ray_val * ray_val * rby_val * cyx_val +
                        2 * rax_val * cxx_val * cyy_val +
                        2 * rax_val * cxy_val * cyx_val +
                        2 * ray_val * cxx_val * cxy_val -
                        2 * ray_val * cyx_val * cyy_val) / 40;

    if (level >= 1) {
        result.s1[0] = rt30 * (126 * rax_val * ray_val * rbx_val * rby_val +
                               14 * rax_val * rbx_val * cyy_val +
                               14 * rax_val * rby_val * cyx_val +
                               14 * ray_val * rbx_val * cxy_val +
                               14 * ray_val * rby_val * cxx_val +
                               2 * cxx_val * cyy_val +
                               2 * cxy_val * cyx_val) / 40;
        result.s1[1] = rt30 * (63 * rax_val * rax_val * rbx_val * rby_val -
                               63 * ray_val * ray_val * rbx_val * rby_val +
                               14 * rax_val * rbx_val * cxy_val +
                               14 * rax_val * rby_val * cxx_val -
                               14 * ray_val * rbx_val * cyy_val -
                               14 * ray_val * rby_val * cyx_val +
                               2 * cxx_val * cxy_val -
                               2 * cyx_val * cyy_val) / 40;
        result.s1[3] = rt30 * (63 * rax_val * rax_val * ray_val * rby_val -
                               21 * ray_val * ray_val * ray_val * rby_val +
                               7 * rax_val * rax_val * cyy_val +
                               14 * rax_val * ray_val * cxy_val -
                               7 * ray_val * ray_val * cyy_val) / 40;
        result.s1[4] = rt30 * (63 * rax_val * rax_val * ray_val * rbx_val -
                               21 * ray_val * ray_val * ray_val * rbx_val +
                               7 * rax_val * rax_val * cyx_val +
                               14 * rax_val * ray_val * cxx_val -
                               7 * ray_val * ray_val * cyx_val) / 40;
        result.s1[6] = rt30 * (14 * rax_val * ray_val * rby_val +
                               2 * rax_val * cyy_val +
                               2 * ray_val * cxy_val) / 40;
        result.s1[9] = rt30 * (7 * rax_val * rax_val * rby_val -
                               7 * ray_val * ray_val * rby_val +
                               2 * rax_val * cxy_val -
                               2 * ray_val * cyy_val) / 40;
        result.s1[7] = rt30 * (14 * rax_val * ray_val * rbx_val +
                               2 * rax_val * cyx_val +
                               2 * ray_val * cxx_val) / 40;
        result.s1[10] = rt30 * (7 * rax_val * rax_val * rbx_val -
                                7 * ray_val * ray_val * rbx_val +
                                2 * rax_val * cxx_val -
                                2 * ray_val * cyx_val) / 40;

        if (level >= 2) {
            result.s2[0] = rt30 * (126 * rbx_val * rby_val * ray_val +
                                   14 * rbx_val * cyy_val +
                                   14 * rby_val * cyx_val) / 40;
            result.s2[1] = rt30 * (126 * rbx_val * rby_val * rax_val +
                                   14 * rbx_val * cxy_val +
                                   14 * rby_val * cxx_val) / 40;
            result.s2[2] = rt30 * (-126 * rbx_val * rby_val * ray_val -
                                   14 * rbx_val * cyy_val -
                                   14 * rby_val * cyx_val) / 40;
            result.s2[6] = rt30 * (126 * rax_val * ray_val * rby_val +
                                   14 * rax_val * cyy_val +
                                   14 * ray_val * cxy_val) / 40;
            result.s2[7] = rt30 * (63 * rax_val * rax_val * rby_val -
                                   63 * ray_val * ray_val * rby_val +
                                   14 * rax_val * cxy_val -
                                   14 * ray_val * cyy_val) / 40;
            result.s2[10] = rt30 * (126 * rax_val * ray_val * rbx_val +
                                    14 * rax_val * cyx_val +
                                    14 * ray_val * cxx_val) / 40;
            result.s2[11] = rt30 * (63 * rax_val * rax_val * rbx_val -
                                    63 * ray_val * ray_val * rbx_val +
                                    14 * rax_val * cxx_val -
                                    14 * ray_val * cyx_val) / 40;
            result.s2[13] = rt30 * (63 * rax_val * rax_val * ray_val -
                                    21 * ray_val * ray_val * ray_val) / 40;
            result.s2[21] = rt30 * (14 * ray_val * rby_val +
                                    2 * cyy_val) / 40;
            result.s2[22] = rt30 * (14 * rax_val * rby_val +
                                    2 * cxy_val) / 40;
            result.s2[25] = 7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[28] = rt30 * (14 * rax_val * rby_val +
                                    2 * cxy_val) / 40;
            result.s2[29] = rt30 * (-14 * ray_val * rby_val -
                                    2 * cyy_val) / 40;
            result.s2[32] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[45] = rt30 * (14 * rbx_val * ray_val +
                                    2 * cyx_val) / 40;
            result.s2[46] = rt30 * (14 * rax_val * rbx_val +
                                    2 * cxx_val) / 40;
            result.s2[48] = 7.0 / 20.0 * rt30 * rax_val * ray_val;
            result.s2[51] = rt30 * ray_val / 20;
            result.s2[52] = rt30 * rax_val / 20;
            result.s2[55] = rt30 * (14 * rax_val * rbx_val +
                                    2 * cxx_val) / 40;
            result.s2[56] = rt30 * (-14 * rbx_val * ray_val -
                                    2 * cyx_val) / 40;
            result.s2[58] = rt30 * (7 * rax_val * rax_val -
                                    7 * ray_val * ray_val) / 40;
            result.s2[61] = rt30 * rax_val / 20;
            result.s2[62] = -rt30 * ray_val / 20;
        }
    }
}

// ============================================================================
// HEXADECAPOLE-DIPOLE KERNELS (Orient cases 264-287)
// Hexadecapole @ A (uses rax, ray, raz), Dipole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Hexadecapole-40 × Dipole-z kernel
 * Orient case 264: Q40 × Q10
 */
void hexadecapole_40_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czz_val = sf.czz();

    result.s0 = 63.0/8.0*raz_val*raz_val*raz_val*raz_val*rbz_val - 21.0/4.0*raz_val*raz_val*rbz_val +
                7.0/2.0*raz_val*raz_val*raz_val*czz_val + 3.0/8.0*rbz_val - 3.0/2.0*raz_val*czz_val;

    if (level >= 1) {
        result.s1[2] = 63.0/2.0*raz_val*raz_val*raz_val*rbz_val - 21.0/2.0*raz_val*rbz_val +
                       21.0/2.0*raz_val*raz_val*czz_val - 3.0/2.0*czz_val;
        result.s1[5] = 63.0/8.0*raz_val*raz_val*raz_val*raz_val - 21.0/4.0*raz_val*raz_val + 3.0/8.0;
        result.s1[14] = 7.0/2.0*raz_val*raz_val*raz_val - 3.0/2.0*raz_val;

        if (level >= 2) {
            result.s2[5] = 189.0/2.0*raz_val*raz_val*rbz_val - 21.0/2.0*rbz_val + 21.0*raz_val*czz_val;
            result.s2[17] = 63.0/2.0*raz_val*raz_val*raz_val - 21.0/2.0*raz_val;
            result.s2[107] = 21.0/2.0*raz_val*raz_val - 3.0/2.0;
        }
    }
}

/**
 * Hexadecapole-40 × Dipole-x kernel
 * Orient case 265: Q40 × Q11c
 */
void hexadecapole_40_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czx_val = sf.czx();

    result.s0 = 63.0/8.0*raz_val*raz_val*raz_val*raz_val*rbx_val - 21.0/4.0*raz_val*raz_val*rbx_val +
                7.0/2.0*raz_val*raz_val*raz_val*czx_val + 3.0/8.0*rbx_val - 3.0/2.0*raz_val*czx_val;

    if (level >= 1) {
        result.s1[2] = 63.0/2.0*raz_val*raz_val*raz_val*rbx_val - 21.0/2.0*raz_val*rbx_val +
                       21.0/2.0*raz_val*raz_val*czx_val - 3.0/2.0*czx_val;
        result.s1[3] = 63.0/8.0*raz_val*raz_val*raz_val*raz_val - 21.0/4.0*raz_val*raz_val + 3.0/8.0;
        result.s1[12] = 7.0/2.0*raz_val*raz_val*raz_val - 3.0/2.0*raz_val;

        if (level >= 2) {
            result.s2[5] = 189.0/2.0*raz_val*raz_val*rbx_val - 21.0/2.0*rbx_val + 21.0*raz_val*czx_val;
            result.s2[8] = 63.0/2.0*raz_val*raz_val*raz_val - 21.0/2.0*raz_val;
            result.s2[38] = 21.0/2.0*raz_val*raz_val - 3.0/2.0;
        }
    }
}

/**
 * Hexadecapole-40 × Dipole-y kernel
 * Orient case 266: Q40 × Q11s
 */
void hexadecapole_40_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czy_val = sf.czy();

    result.s0 = 63.0/8.0*raz_val*raz_val*raz_val*raz_val*rby_val - 21.0/4.0*raz_val*raz_val*rby_val +
                7.0/2.0*raz_val*raz_val*raz_val*czy_val + 3.0/8.0*rby_val - 3.0/2.0*raz_val*czy_val;

    if (level >= 1) {
        result.s1[2] = 63.0/2.0*raz_val*raz_val*raz_val*rby_val - 21.0/2.0*raz_val*rby_val +
                       21.0/2.0*raz_val*raz_val*czy_val - 3.0/2.0*czy_val;
        result.s1[4] = 63.0/8.0*raz_val*raz_val*raz_val*raz_val - 21.0/4.0*raz_val*raz_val + 3.0/8.0;
        result.s1[13] = 7.0/2.0*raz_val*raz_val*raz_val - 3.0/2.0*raz_val;

        if (level >= 2) {
            result.s2[5] = 189.0/2.0*raz_val*raz_val*rby_val - 21.0/2.0*rby_val + 21.0*raz_val*czy_val;
            result.s2[12] = 63.0/2.0*raz_val*raz_val*raz_val - 21.0/2.0*raz_val;
            result.s2[68] = 21.0/2.0*raz_val*raz_val - 3.0/2.0;
        }
    }
}

/**
 * Hexadecapole-41c × Dipole-z kernel
 * Orient case 267: Q41c × Q10
 */
void hexadecapole_41c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czz_val = sf.czz(), cxz_val = sf.cxz();

    result.s0 = rt10*(63*rax_val*raz_val*raz_val*raz_val*rbz_val - 21*rax_val*raz_val*rbz_val +
                      21*rax_val*raz_val*raz_val*czz_val + 7*raz_val*raz_val*raz_val*cxz_val -
                      3*rax_val*czz_val - 3*raz_val*cxz_val)/20;

    if (level >= 1) {
        result.s1[0] = rt10*(63*raz_val*raz_val*raz_val*rbz_val - 21*raz_val*rbz_val +
                             21*raz_val*raz_val*czz_val - 3*czz_val)/20;
        result.s1[2] = rt10*(189*rax_val*raz_val*raz_val*rbz_val - 21*rbz_val*rax_val +
                             42*rax_val*raz_val*czz_val + 21*raz_val*raz_val*cxz_val - 3*cxz_val)/20;
        result.s1[5] = rt10*(63*rax_val*raz_val*raz_val*raz_val - 21*rax_val*raz_val)/20;
        result.s1[8] = rt10*(7*raz_val*raz_val*raz_val - 3*raz_val)/20;
        result.s1[14] = rt10*(21*rax_val*raz_val*raz_val - 3*rax_val)/20;

        if (level >= 2) {
            result.s2[3] = rt10*(189*raz_val*raz_val*rbz_val - 21*rbz_val + 42*raz_val*czz_val)/20;
            result.s2[5] = rt10*(378*rax_val*raz_val*rbz_val + 42*rax_val*czz_val + 42*raz_val*cxz_val)/20;
            result.s2[15] = rt10*(63*raz_val*raz_val*raz_val - 21*raz_val)/20;
            result.s2[17] = rt10*(189*rax_val*raz_val*raz_val - 21*rax_val)/20;
            result.s2[80] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[105] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[107] = 21.0/10.0*rt10*rax_val*raz_val;
        }
    }
}

/**
 * Hexadecapole-41c × Dipole-x kernel
 * Orient case 268: Q41c × Q11c
 */
void hexadecapole_41c_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czx_val = sf.czx(), cxx_val = sf.cxx();

    result.s0 = rt10*(63*rax_val*raz_val*raz_val*raz_val*rbx_val - 21*rax_val*raz_val*rbx_val +
                      21*rax_val*raz_val*raz_val*czx_val + 7*raz_val*raz_val*raz_val*cxx_val -
                      3*rax_val*czx_val - 3*raz_val*cxx_val)/20;

    if (level >= 1) {
        result.s1[0] = rt10*(63*raz_val*raz_val*raz_val*rbx_val - 21*raz_val*rbx_val +
                             21*raz_val*raz_val*czx_val - 3*czx_val)/20;
        result.s1[2] = rt10*(189*rax_val*raz_val*raz_val*rbx_val - 21*rax_val*rbx_val +
                             42*rax_val*raz_val*czx_val + 21*raz_val*raz_val*cxx_val - 3*cxx_val)/20;
        result.s1[3] = rt10*(63*rax_val*raz_val*raz_val*raz_val - 21*rax_val*raz_val)/20;
        result.s1[6] = rt10*(7*raz_val*raz_val*raz_val - 3*raz_val)/20;
        result.s1[12] = rt10*(21*rax_val*raz_val*raz_val - 3*rax_val)/20;

        if (level >= 2) {
            result.s2[3] = rt10*(189*raz_val*raz_val*rbx_val - 21*rbx_val + 42*raz_val*czx_val)/20;
            result.s2[5] = rt10*(378*rax_val*raz_val*rbx_val + 42*rax_val*czx_val + 42*raz_val*cxx_val)/20;
            result.s2[6] = rt10*(63*raz_val*raz_val*raz_val - 21*raz_val)/20;
            result.s2[8] = rt10*(189*rax_val*raz_val*raz_val - 21*rax_val)/20;
            result.s2[23] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[36] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[38] = 21.0/10.0*rt10*rax_val*raz_val;
        }
    }
}

/**
 * Hexadecapole-41c × Dipole-y kernel
 * Orient case 269: Q41c × Q11s
 */
void hexadecapole_41c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czy_val = sf.czy(), cxy_val = sf.cxy();

    result.s0 = rt10*(63*rax_val*raz_val*raz_val*raz_val*rby_val - 21*rax_val*raz_val*rby_val +
                      21*rax_val*raz_val*raz_val*czy_val + 7*raz_val*raz_val*raz_val*cxy_val -
                      3*rax_val*czy_val - 3*raz_val*cxy_val)/20;

    if (level >= 1) {
        result.s1[0] = rt10*(63*raz_val*raz_val*raz_val*rby_val - 21*raz_val*rby_val +
                             21*raz_val*raz_val*czy_val - 3*czy_val)/20;
        result.s1[2] = rt10*(189*rax_val*raz_val*raz_val*rby_val - 21*rax_val*rby_val +
                             42*rax_val*raz_val*czy_val + 21*raz_val*raz_val*cxy_val - 3*cxy_val)/20;
        result.s1[4] = rt10*(63*rax_val*raz_val*raz_val*raz_val - 21*rax_val*raz_val)/20;
        result.s1[7] = rt10*(7*raz_val*raz_val*raz_val - 3*raz_val)/20;
        result.s1[13] = rt10*(21*rax_val*raz_val*raz_val - 3*rax_val)/20;

        if (level >= 2) {
            result.s2[3] = rt10*(189*raz_val*raz_val*rby_val - 21*rby_val + 42*raz_val*czy_val)/20;
            result.s2[5] = rt10*(378*rax_val*raz_val*rby_val + 42*rax_val*czy_val + 42*raz_val*cxy_val)/20;
            result.s2[10] = rt10*(63*raz_val*raz_val*raz_val - 21*raz_val)/20;
            result.s2[12] = rt10*(189*rax_val*raz_val*raz_val - 21*rax_val)/20;
            result.s2[47] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[66] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[68] = 21.0/10.0*rt10*rax_val*raz_val;
        }
    }
}

/**
 * Hexadecapole-41s × Dipole-z kernel
 * Orient case 270: Q41s × Q10
 */
void hexadecapole_41s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czz_val = sf.czz(), cyz_val = sf.cyz();

    result.s0 = rt10*(63*ray_val*raz_val*raz_val*raz_val*rbz_val - 21*ray_val*raz_val*rbz_val +
                      21*ray_val*raz_val*raz_val*czz_val + 7*raz_val*raz_val*raz_val*cyz_val -
                      3*ray_val*czz_val - 3*raz_val*cyz_val)/20;

    if (level >= 1) {
        result.s1[1] = rt10*(63*raz_val*raz_val*raz_val*rbz_val - 21*raz_val*rbz_val +
                             21*raz_val*raz_val*czz_val - 3*czz_val)/20;
        result.s1[2] = rt10*(189*ray_val*raz_val*raz_val*rbz_val - 21*rbz_val*ray_val +
                             42*ray_val*raz_val*czz_val + 21*raz_val*raz_val*cyz_val - 3*cyz_val)/20;
        result.s1[5] = rt10*(63*ray_val*raz_val*raz_val*raz_val - 21*ray_val*raz_val)/20;
        result.s1[11] = rt10*(7*raz_val*raz_val*raz_val - 3*raz_val)/20;
        result.s1[14] = rt10*(21*ray_val*raz_val*raz_val - 3*ray_val)/20;

        if (level >= 2) {
            result.s2[4] = rt10*(189*raz_val*raz_val*rbz_val - 21*rbz_val + 42*raz_val*czz_val)/20;
            result.s2[5] = rt10*(378*ray_val*raz_val*rbz_val + 42*ray_val*czz_val + 42*raz_val*cyz_val)/20;
            result.s2[16] = rt10*(63*raz_val*raz_val*raz_val - 21*raz_val)/20;
            result.s2[17] = rt10*(189*ray_val*raz_val*raz_val - 21*ray_val)/20;
            result.s2[93] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[106] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[107] = 21.0/10.0*rt10*ray_val*raz_val;
        }
    }
}

/**
 * Hexadecapole-41s × Dipole-x kernel
 * Orient case 271: Q41s × Q11c
 */
void hexadecapole_41s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czx_val = sf.czx(), cyx_val = sf.cyx();

    result.s0 = rt10*(63*ray_val*raz_val*raz_val*raz_val*rbx_val - 21*ray_val*raz_val*rbx_val +
                      21*ray_val*raz_val*raz_val*czx_val + 7*raz_val*raz_val*raz_val*cyx_val -
                      3*ray_val*czx_val - 3*raz_val*cyx_val)/20;

    if (level >= 1) {
        result.s1[1] = rt10*(63*raz_val*raz_val*raz_val*rbx_val - 21*raz_val*rbx_val +
                             21*raz_val*raz_val*czx_val - 3*czx_val)/20;
        result.s1[2] = rt10*(189*ray_val*raz_val*raz_val*rbx_val - 21*rbx_val*ray_val +
                             42*ray_val*raz_val*czx_val + 21*raz_val*raz_val*cyx_val - 3*cyx_val)/20;
        result.s1[3] = rt10*(63*ray_val*raz_val*raz_val*raz_val - 21*ray_val*raz_val)/20;
        result.s1[9] = rt10*(7*raz_val*raz_val*raz_val - 3*raz_val)/20;
        result.s1[12] = rt10*(21*ray_val*raz_val*raz_val - 3*ray_val)/20;

        if (level >= 2) {
            result.s2[4] = rt10*(189*raz_val*raz_val*rbx_val - 21*rbx_val + 42*raz_val*czx_val)/20;
            result.s2[5] = rt10*(378*ray_val*raz_val*rbx_val + 42*ray_val*czx_val + 42*raz_val*cyx_val)/20;
            result.s2[7] = rt10*(63*raz_val*raz_val*raz_val - 21*raz_val)/20;
            result.s2[8] = rt10*(189*ray_val*raz_val*raz_val - 21*ray_val)/20;
            result.s2[30] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[37] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[38] = 21.0/10.0*rt10*ray_val*raz_val;
        }
    }
}

/**
 * Hexadecapole-41s × Dipole-y kernel
 * Orient case 272: Q41s × Q11s
 */
void hexadecapole_41s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czy_val = sf.czy(), cyy_val = sf.cyy();

    result.s0 = rt10*(63*ray_val*raz_val*raz_val*raz_val*rby_val - 21*ray_val*raz_val*rby_val +
                      21*ray_val*raz_val*raz_val*czy_val + 7*raz_val*raz_val*raz_val*cyy_val -
                      3*ray_val*czy_val - 3*raz_val*cyy_val)/20;

    if (level >= 1) {
        result.s1[1] = rt10*(63*raz_val*raz_val*raz_val*rby_val - 21*raz_val*rby_val +
                             21*raz_val*raz_val*czy_val - 3*czy_val)/20;
        result.s1[2] = rt10*(189*ray_val*raz_val*raz_val*rby_val - 21*ray_val*rby_val +
                             42*ray_val*raz_val*czy_val + 21*raz_val*raz_val*cyy_val - 3*cyy_val)/20;
        result.s1[4] = rt10*(63*ray_val*raz_val*raz_val*raz_val - 21*ray_val*raz_val)/20;
        result.s1[10] = rt10*(7*raz_val*raz_val*raz_val - 3*raz_val)/20;
        result.s1[13] = rt10*(21*ray_val*raz_val*raz_val - 3*ray_val)/20;

        if (level >= 2) {
            result.s2[4] = rt10*(189*raz_val*raz_val*rby_val - 21*rby_val + 42*raz_val*czy_val)/20;
            result.s2[5] = rt10*(378*ray_val*raz_val*rby_val + 42*ray_val*czy_val + 42*raz_val*cyy_val)/20;
            result.s2[11] = rt10*(63*raz_val*raz_val*raz_val - 21*raz_val)/20;
            result.s2[12] = rt10*(189*ray_val*raz_val*raz_val - 21*ray_val)/20;
            result.s2[57] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[67] = rt10*(21*raz_val*raz_val - 3)/20;
            result.s2[68] = 21.0/10.0*rt10*ray_val*raz_val;
        }
    }
}

/**
 * Hexadecapole-42c × Dipole-z kernel
 * Orient case 273: Q42c × Q10
 */
void hexadecapole_42c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czz_val = sf.czz(), cxz_val = sf.cxz(), cyz_val = sf.cyz();

    result.s0 = rt5*(63*rax_val*rax_val*raz_val*raz_val*rbz_val - 63*ray_val*ray_val*raz_val*raz_val*rbz_val -
                     7*rax_val*rax_val*rbz_val + 7*ray_val*ray_val*rbz_val +
                     14*rax_val*rax_val*raz_val*czz_val + 14*rax_val*raz_val*raz_val*cxz_val -
                     14*ray_val*ray_val*raz_val*czz_val - 14*ray_val*raz_val*raz_val*cyz_val -
                     2*rax_val*cxz_val + 2*ray_val*cyz_val)/20;

    if (level >= 1) {
        result.s1[0] = rt5*(126*rax_val*raz_val*raz_val*rbz_val - 14*rbz_val*rax_val +
                            28*rax_val*raz_val*czz_val + 14*raz_val*raz_val*cxz_val - 2*cxz_val)/20;
        result.s1[1] = rt5*(-126*ray_val*raz_val*raz_val*rbz_val + 14*rbz_val*ray_val -
                            28*ray_val*raz_val*czz_val - 14*raz_val*raz_val*cyz_val + 2*cyz_val)/20;
        result.s1[2] = rt5*(126*rax_val*rax_val*raz_val*rbz_val - 126*ray_val*ray_val*raz_val*rbz_val +
                            14*rax_val*rax_val*czz_val + 28*rax_val*raz_val*cxz_val -
                            14*ray_val*ray_val*czz_val - 28*ray_val*raz_val*cyz_val)/20;
        result.s1[5] = rt5*(63*rax_val*rax_val*raz_val*raz_val - 63*ray_val*ray_val*raz_val*raz_val -
                            7*rax_val*rax_val + 7*ray_val*ray_val)/20;
        result.s1[8] = rt5*(14*rax_val*raz_val*raz_val - 2*rax_val)/20;
        result.s1[11] = rt5*(-14*ray_val*raz_val*raz_val + 2*ray_val)/20;
        result.s1[14] = rt5*(14*rax_val*rax_val*raz_val - 14*ray_val*ray_val*raz_val)/20;

        if (level >= 2) {
            result.s2[0] = rt5*(126*raz_val*raz_val*rbz_val - 14*rbz_val + 28*raz_val*czz_val)/20;
            result.s2[2] = rt5*(-126*raz_val*raz_val*rbz_val + 14*rbz_val - 28*raz_val*czz_val)/20;
            result.s2[3] = rt5*(252*rax_val*raz_val*rbz_val + 28*rax_val*czz_val + 28*raz_val*cxz_val)/20;
            result.s2[4] = rt5*(-252*ray_val*raz_val*rbz_val - 28*ray_val*czz_val - 28*raz_val*cyz_val)/20;
            result.s2[5] = rt5*(126*rax_val*rax_val*rbz_val - 126*ray_val*ray_val*rbz_val +
                                28*rax_val*cxz_val - 28*ray_val*cyz_val)/20;
            result.s2[15] = rt5*(126*rax_val*raz_val*raz_val - 14*rax_val)/20;
            result.s2[16] = rt5*(-126*ray_val*raz_val*raz_val + 14*ray_val)/20;
            result.s2[17] = rt5*(126*rax_val*rax_val*raz_val - 126*ray_val*ray_val*raz_val)/20;
            result.s2[78] = rt5*(14*raz_val*raz_val - 2)/20;
            result.s2[80] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[92] = rt5*(-14*raz_val*raz_val + 2)/20;
            result.s2[93] = -7.0/5.0*rt5*ray_val*raz_val;
            result.s2[105] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[106] = -7.0/5.0*rt5*ray_val*raz_val;
            result.s2[107] = rt5*(14*rax_val*rax_val - 14*ray_val*ray_val)/20;
        }
    }
}

/**
 * Hexadecapole-42c × Dipole-x kernel
 * Orient case 274: Q42c × Q11c
 */
void hexadecapole_42c_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czx_val = sf.czx(), cxx_val = sf.cxx(), cyx_val = sf.cyx();

    result.s0 = rt5*(63*rax_val*rax_val*raz_val*raz_val*rbx_val - 63*ray_val*ray_val*raz_val*raz_val*rbx_val -
                     7*rax_val*rax_val*rbx_val + 7*ray_val*ray_val*rbx_val +
                     14*rax_val*rax_val*raz_val*czx_val + 14*rax_val*raz_val*raz_val*cxx_val -
                     14*ray_val*ray_val*raz_val*czx_val - 14*ray_val*raz_val*raz_val*cyx_val -
                     2*rax_val*cxx_val + 2*ray_val*cyx_val)/20;

    if (level >= 1) {
        result.s1[0] = rt5*(126*rax_val*raz_val*raz_val*rbx_val - 14*rax_val*rbx_val +
                            28*rax_val*raz_val*czx_val + 14*raz_val*raz_val*cxx_val - 2*cxx_val)/20;
        result.s1[1] = rt5*(-126*ray_val*raz_val*raz_val*rbx_val + 14*rbx_val*ray_val -
                            28*ray_val*raz_val*czx_val - 14*raz_val*raz_val*cyx_val + 2*cyx_val)/20;
        result.s1[2] = rt5*(126*rax_val*rax_val*raz_val*rbx_val - 126*ray_val*ray_val*raz_val*rbx_val +
                            14*rax_val*rax_val*czx_val + 28*rax_val*raz_val*cxx_val -
                            14*ray_val*ray_val*czx_val - 28*ray_val*raz_val*cyx_val)/20;
        result.s1[3] = rt5*(63*rax_val*rax_val*raz_val*raz_val - 63*ray_val*ray_val*raz_val*raz_val -
                            7*rax_val*rax_val + 7*ray_val*ray_val)/20;
        result.s1[6] = rt5*(14*rax_val*raz_val*raz_val - 2*rax_val)/20;
        result.s1[9] = rt5*(-14*ray_val*raz_val*raz_val + 2*ray_val)/20;
        result.s1[12] = rt5*(14*rax_val*rax_val*raz_val - 14*ray_val*ray_val*raz_val)/20;

        if (level >= 2) {
            result.s2[0] = rt5*(126*raz_val*raz_val*rbx_val - 14*rbx_val + 28*raz_val*czx_val)/20;
            result.s2[2] = rt5*(-126*raz_val*raz_val*rbx_val + 14*rbx_val - 28*raz_val*czx_val)/20;
            result.s2[3] = rt5*(252*rax_val*raz_val*rbx_val + 28*rax_val*czx_val + 28*raz_val*cxx_val)/20;
            result.s2[4] = rt5*(-252*ray_val*raz_val*rbx_val - 28*ray_val*czx_val - 28*raz_val*cyx_val)/20;
            result.s2[5] = rt5*(126*rax_val*rax_val*rbx_val - 126*ray_val*ray_val*rbx_val +
                                28*rax_val*cxx_val - 28*ray_val*cyx_val)/20;
            result.s2[6] = rt5*(126*rax_val*raz_val*raz_val - 14*rax_val)/20;
            result.s2[7] = rt5*(-126*ray_val*raz_val*raz_val + 14*ray_val)/20;
            result.s2[8] = rt5*(126*rax_val*rax_val*raz_val - 126*ray_val*ray_val*raz_val)/20;
            result.s2[21] = rt5*(14*raz_val*raz_val - 2)/20;
            result.s2[23] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[29] = rt5*(-14*raz_val*raz_val + 2)/20;
            result.s2[30] = -7.0/5.0*rt5*ray_val*raz_val;
            result.s2[36] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[37] = -7.0/5.0*rt5*ray_val*raz_val;
            result.s2[38] = rt5*(14*rax_val*rax_val - 14*ray_val*ray_val)/20;
        }
    }
}

/**
 * Hexadecapole-42c × Dipole-y kernel
 * Orient case 275: Q42c × Q11s
 */
void hexadecapole_42c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czy_val = sf.czy(), cxy_val = sf.cxy(), cyy_val = sf.cyy();

    result.s0 = rt5*(63*rax_val*rax_val*raz_val*raz_val*rby_val - 63*ray_val*ray_val*raz_val*raz_val*rby_val -
                     7*rax_val*rax_val*rby_val + 7*ray_val*ray_val*rby_val +
                     14*rax_val*rax_val*raz_val*czy_val + 14*rax_val*raz_val*raz_val*cxy_val -
                     14*ray_val*ray_val*raz_val*czy_val - 14*ray_val*raz_val*raz_val*cyy_val -
                     2*rax_val*cxy_val + 2*ray_val*cyy_val)/20;

    if (level >= 1) {
        result.s1[0] = rt5*(126*rax_val*raz_val*raz_val*rby_val - 14*rax_val*rby_val +
                            28*rax_val*raz_val*czy_val + 14*raz_val*raz_val*cxy_val - 2*cxy_val)/20;
        result.s1[1] = rt5*(-126*ray_val*raz_val*raz_val*rby_val + 14*ray_val*rby_val -
                            28*ray_val*raz_val*czy_val - 14*raz_val*raz_val*cyy_val + 2*cyy_val)/20;
        result.s1[2] = rt5*(126*rax_val*rax_val*raz_val*rby_val - 126*ray_val*ray_val*raz_val*rby_val +
                            14*rax_val*rax_val*czy_val + 28*rax_val*raz_val*cxy_val -
                            14*ray_val*ray_val*czy_val - 28*ray_val*raz_val*cyy_val)/20;
        result.s1[4] = rt5*(63*rax_val*rax_val*raz_val*raz_val - 63*ray_val*ray_val*raz_val*raz_val -
                            7*rax_val*rax_val + 7*ray_val*ray_val)/20;
        result.s1[7] = rt5*(14*rax_val*raz_val*raz_val - 2*rax_val)/20;
        result.s1[10] = rt5*(-14*ray_val*raz_val*raz_val + 2*ray_val)/20;
        result.s1[13] = rt5*(14*rax_val*rax_val*raz_val - 14*ray_val*ray_val*raz_val)/20;

        if (level >= 2) {
            result.s2[0] = rt5*(126*raz_val*raz_val*rby_val - 14*rby_val + 28*raz_val*czy_val)/20;
            result.s2[2] = rt5*(-126*raz_val*raz_val*rby_val + 14*rby_val - 28*raz_val*czy_val)/20;
            result.s2[3] = rt5*(252*rax_val*raz_val*rby_val + 28*rax_val*czy_val + 28*raz_val*cxy_val)/20;
            result.s2[4] = rt5*(-252*ray_val*raz_val*rby_val - 28*ray_val*czy_val - 28*raz_val*cyy_val)/20;
            result.s2[5] = rt5*(126*rax_val*rax_val*rby_val - 126*ray_val*ray_val*rby_val +
                                28*rax_val*cxy_val - 28*ray_val*cyy_val)/20;
            result.s2[10] = rt5*(126*rax_val*raz_val*raz_val - 14*rax_val)/20;
            result.s2[11] = rt5*(-126*ray_val*raz_val*raz_val + 14*ray_val)/20;
            result.s2[12] = rt5*(126*rax_val*rax_val*raz_val - 126*ray_val*ray_val*raz_val)/20;
            result.s2[45] = rt5*(14*raz_val*raz_val - 2)/20;
            result.s2[47] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[56] = rt5*(-14*raz_val*raz_val + 2)/20;
            result.s2[57] = -7.0/5.0*rt5*ray_val*raz_val;
            result.s2[66] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[67] = -7.0/5.0*rt5*ray_val*raz_val;
            result.s2[68] = rt5*(14*rax_val*rax_val - 14*ray_val*ray_val)/20;
        }
    }
}

/**
 * Hexadecapole-42s × Dipole-z kernel
 * Orient case 276: Q42s × Q10
 */
void hexadecapole_42s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czz_val = sf.czz(), cxz_val = sf.cxz(), cyz_val = sf.cyz();

    result.s0 = rt5*(63*rax_val*ray_val*raz_val*raz_val*rbz_val - 7*rax_val*ray_val*rbz_val +
                     14*rax_val*ray_val*raz_val*czz_val + 7*rax_val*raz_val*raz_val*cyz_val +
                     7*ray_val*raz_val*raz_val*cxz_val - rax_val*cyz_val - ray_val*cxz_val)/10;

    if (level >= 1) {
        result.s1[0] = rt5*(63*ray_val*raz_val*raz_val*rbz_val - 7*rbz_val*ray_val +
                            14*ray_val*raz_val*czz_val + 7*raz_val*raz_val*cyz_val - cyz_val)/10;
        result.s1[1] = rt5*(63*rax_val*raz_val*raz_val*rbz_val - 7*rbz_val*rax_val +
                            14*rax_val*raz_val*czz_val + 7*raz_val*raz_val*cxz_val - cxz_val)/10;
        result.s1[2] = rt5*(126*rax_val*ray_val*raz_val*rbz_val + 14*rax_val*ray_val*czz_val +
                            14*rax_val*raz_val*cyz_val + 14*ray_val*raz_val*cxz_val)/10;
        result.s1[5] = rt5*(63*rax_val*ray_val*raz_val*raz_val - 7*rax_val*ray_val)/10;
        result.s1[8] = rt5*(7*ray_val*raz_val*raz_val - ray_val)/10;
        result.s1[11] = rt5*(7*rax_val*raz_val*raz_val - rax_val)/10;
        result.s1[14] = 7.0/5.0*rt5*rax_val*ray_val*raz_val;

        if (level >= 2) {
            result.s2[1] = rt5*(63*raz_val*raz_val*rbz_val - 7*rbz_val + 14*raz_val*czz_val)/10;
            result.s2[3] = rt5*(126*ray_val*raz_val*rbz_val + 14*ray_val*czz_val + 14*raz_val*cyz_val)/10;
            result.s2[4] = rt5*(126*rax_val*raz_val*rbz_val + 14*rax_val*czz_val + 14*raz_val*cxz_val)/10;
            result.s2[5] = rt5*(126*rax_val*ray_val*rbz_val + 14*rax_val*cyz_val + 14*ray_val*cxz_val)/10;
            result.s2[15] = rt5*(63*ray_val*raz_val*raz_val - 7*ray_val)/10;
            result.s2[16] = rt5*(63*rax_val*raz_val*raz_val - 7*rax_val)/10;
            result.s2[17] = 63.0/5.0*rt5*rax_val*ray_val*raz_val;
            result.s2[79] = rt5*(7*raz_val*raz_val - 1)/10;
            result.s2[80] = 7.0/5.0*rt5*ray_val*raz_val;
            result.s2[91] = rt5*(7*raz_val*raz_val - 1)/10;
            result.s2[93] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[105] = 7.0/5.0*rt5*ray_val*raz_val;
            result.s2[106] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[107] = 7.0/5.0*rt5*rax_val*ray_val;
        }
    }
}

/**
 * Hexadecapole-42s × Dipole-x kernel
 * Orient case 277: Q42s × Q11c
 */
void hexadecapole_42s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czx_val = sf.czx(), cxx_val = sf.cxx(), cyx_val = sf.cyx();

    result.s0 = rt5*(63*rax_val*ray_val*raz_val*raz_val*rbx_val - 7*rax_val*ray_val*rbx_val +
                     14*rax_val*ray_val*raz_val*czx_val + 7*rax_val*raz_val*raz_val*cyx_val +
                     7*ray_val*raz_val*raz_val*cxx_val - rax_val*cyx_val - ray_val*cxx_val)/10;

    if (level >= 1) {
        result.s1[0] = rt5*(63*ray_val*raz_val*raz_val*rbx_val - 7*rbx_val*ray_val +
                            14*ray_val*raz_val*czx_val + 7*raz_val*raz_val*cyx_val - cyx_val)/10;
        result.s1[1] = rt5*(63*rax_val*raz_val*raz_val*rbx_val - 7*rax_val*rbx_val +
                            14*rax_val*raz_val*czx_val + 7*raz_val*raz_val*cxx_val - cxx_val)/10;
        result.s1[2] = rt5*(126*rax_val*ray_val*raz_val*rbx_val + 14*rax_val*ray_val*czx_val +
                            14*rax_val*raz_val*cyx_val + 14*ray_val*raz_val*cxx_val)/10;
        result.s1[3] = rt5*(63*rax_val*ray_val*raz_val*raz_val - 7*rax_val*ray_val)/10;
        result.s1[6] = rt5*(7*ray_val*raz_val*raz_val - ray_val)/10;
        result.s1[9] = rt5*(7*rax_val*raz_val*raz_val - rax_val)/10;
        result.s1[12] = 7.0/5.0*rt5*rax_val*ray_val*raz_val;

        if (level >= 2) {
            result.s2[1] = rt5*(63*raz_val*raz_val*rbx_val - 7*rbx_val + 14*raz_val*czx_val)/10;
            result.s2[3] = rt5*(126*ray_val*raz_val*rbx_val + 14*ray_val*czx_val + 14*raz_val*cyx_val)/10;
            result.s2[4] = rt5*(126*rax_val*raz_val*rbx_val + 14*rax_val*czx_val + 14*raz_val*cxx_val)/10;
            result.s2[5] = rt5*(126*rax_val*ray_val*rbx_val + 14*rax_val*cyx_val + 14*ray_val*cxx_val)/10;
            result.s2[6] = rt5*(63*ray_val*raz_val*raz_val - 7*ray_val)/10;
            result.s2[7] = rt5*(63*rax_val*raz_val*raz_val - 7*rax_val)/10;
            result.s2[8] = 63.0/5.0*rt5*rax_val*ray_val*raz_val;
            result.s2[22] = rt5*(7*raz_val*raz_val - 1)/10;
            result.s2[23] = 7.0/5.0*rt5*ray_val*raz_val;
            result.s2[28] = rt5*(7*raz_val*raz_val - 1)/10;
            result.s2[30] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[36] = 7.0/5.0*rt5*ray_val*raz_val;
            result.s2[37] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[38] = 7.0/5.0*rt5*rax_val*ray_val;
        }
    }
}

/**
 * Hexadecapole-42s × Dipole-y kernel
 * Orient case 278: Q42s × Q11s
 */
void hexadecapole_42s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czy_val = sf.czy(), cxy_val = sf.cxy(), cyy_val = sf.cyy();

    result.s0 = rt5*(63*rax_val*ray_val*raz_val*raz_val*rby_val - 7*rax_val*ray_val*rby_val +
                     14*rax_val*ray_val*raz_val*czy_val + 7*rax_val*raz_val*raz_val*cyy_val +
                     7*ray_val*raz_val*raz_val*cxy_val - rax_val*cyy_val - ray_val*cxy_val)/10;

    if (level >= 1) {
        result.s1[0] = rt5*(63*ray_val*raz_val*raz_val*rby_val - 7*ray_val*rby_val +
                            14*ray_val*raz_val*czy_val + 7*raz_val*raz_val*cyy_val - cyy_val)/10;
        result.s1[1] = rt5*(63*rax_val*raz_val*raz_val*rby_val - 7*rax_val*rby_val +
                            14*rax_val*raz_val*czy_val + 7*raz_val*raz_val*cxy_val - cxy_val)/10;
        result.s1[2] = rt5*(126*rax_val*ray_val*raz_val*rby_val + 14*rax_val*ray_val*czy_val +
                            14*rax_val*raz_val*cyy_val + 14*ray_val*raz_val*cxy_val)/10;
        result.s1[4] = rt5*(63*rax_val*ray_val*raz_val*raz_val - 7*rax_val*ray_val)/10;
        result.s1[7] = rt5*(7*ray_val*raz_val*raz_val - ray_val)/10;
        result.s1[10] = rt5*(7*rax_val*raz_val*raz_val - rax_val)/10;
        result.s1[13] = 7.0/5.0*rt5*rax_val*ray_val*raz_val;

        if (level >= 2) {
            result.s2[1] = rt5*(63*raz_val*raz_val*rby_val - 7*rby_val + 14*raz_val*czy_val)/10;
            result.s2[3] = rt5*(126*ray_val*raz_val*rby_val + 14*ray_val*czy_val + 14*raz_val*cyy_val)/10;
            result.s2[4] = rt5*(126*rax_val*raz_val*rby_val + 14*rax_val*czy_val + 14*raz_val*cxy_val)/10;
            result.s2[5] = rt5*(126*rax_val*ray_val*rby_val + 14*rax_val*cyy_val + 14*ray_val*cxy_val)/10;
            result.s2[10] = rt5*(63*ray_val*raz_val*raz_val - 7*ray_val)/10;
            result.s2[11] = rt5*(63*rax_val*raz_val*raz_val - 7*rax_val)/10;
            result.s2[12] = 63.0/5.0*rt5*rax_val*ray_val*raz_val;
            result.s2[46] = rt5*(7*raz_val*raz_val - 1)/10;
            result.s2[47] = 7.0/5.0*rt5*ray_val*raz_val;
            result.s2[55] = rt5*(7*raz_val*raz_val - 1)/10;
            result.s2[57] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[66] = 7.0/5.0*rt5*ray_val*raz_val;
            result.s2[67] = 7.0/5.0*rt5*rax_val*raz_val;
            result.s2[68] = 7.0/5.0*rt5*rax_val*ray_val;
        }
    }
}

/**
 * Hexadecapole-43c × Dipole-z kernel
 * Orient case 279: Q43c × Q10
 */
void hexadecapole_43c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czz_val = sf.czz(), cxz_val = sf.cxz(), cyz_val = sf.cyz();

    result.s0 = rt70*(9*rax_val*rax_val*rax_val*raz_val*rbz_val - 27*rax_val*ray_val*ray_val*raz_val*rbz_val +
                      rax_val*rax_val*rax_val*czz_val + 3*rax_val*rax_val*raz_val*cxz_val -
                      3*rax_val*ray_val*ray_val*czz_val - 6*rax_val*ray_val*raz_val*cyz_val -
                      3*ray_val*ray_val*raz_val*cxz_val)/20;

    if (level >= 1) {
        result.s1[0] = rt70*(27*rax_val*rax_val*raz_val*rbz_val - 27*ray_val*ray_val*raz_val*rbz_val +
                             3*rax_val*rax_val*czz_val + 6*rax_val*raz_val*cxz_val -
                             3*ray_val*ray_val*czz_val - 6*ray_val*raz_val*cyz_val)/20;
        result.s1[1] = rt70*(-54*rax_val*ray_val*raz_val*rbz_val - 6*rax_val*ray_val*czz_val -
                             6*rax_val*raz_val*cyz_val - 6*ray_val*raz_val*cxz_val)/20;
        result.s1[2] = rt70*(9*rax_val*rax_val*rax_val*rbz_val - 27*rax_val*ray_val*ray_val*rbz_val +
                             3*rax_val*rax_val*cxz_val - 6*rax_val*ray_val*cyz_val - 3*ray_val*ray_val*cxz_val)/20;
        result.s1[5] = rt70*(9*rax_val*rax_val*rax_val*raz_val - 27*rax_val*ray_val*ray_val*raz_val)/20;
        result.s1[8] = rt70*(3*rax_val*rax_val*raz_val - 3*ray_val*ray_val*raz_val)/20;
        result.s1[11] = -3.0/10.0*rt70*rax_val*ray_val*raz_val;
        result.s1[14] = rt70*(rax_val*rax_val*rax_val - 3*rax_val*ray_val*ray_val)/20;

        if (level >= 2) {
            result.s2[0] = rt70*(54*rax_val*raz_val*rbz_val + 6*rax_val*czz_val + 6*raz_val*cxz_val)/20;
            result.s2[1] = rt70*(-54*ray_val*raz_val*rbz_val - 6*ray_val*czz_val - 6*raz_val*cyz_val)/20;
            result.s2[2] = rt70*(-54*rax_val*raz_val*rbz_val - 6*rax_val*czz_val - 6*raz_val*cxz_val)/20;
            result.s2[3] = rt70*(27*rax_val*rax_val*rbz_val - 27*ray_val*ray_val*rbz_val +
                                 6*rax_val*cxz_val - 6*ray_val*cyz_val)/20;
            result.s2[4] = rt70*(-54*rax_val*ray_val*rbz_val - 6*rax_val*cyz_val - 6*ray_val*cxz_val)/20;
            result.s2[15] = rt70*(27*rax_val*rax_val*raz_val - 27*ray_val*ray_val*raz_val)/20;
            result.s2[16] = -27.0/10.0*rt70*rax_val*ray_val*raz_val;
            result.s2[17] = rt70*(9*rax_val*rax_val*rax_val - 27*rax_val*ray_val*ray_val)/20;
            result.s2[78] = 3.0/10.0*rt70*rax_val*raz_val;
            result.s2[79] = -3.0/10.0*rt70*ray_val*raz_val;
            result.s2[80] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
            result.s2[91] = -3.0/10.0*rt70*ray_val*raz_val;
            result.s2[92] = -3.0/10.0*rt70*rax_val*raz_val;
            result.s2[93] = -3.0/10.0*rt70*rax_val*ray_val;
            result.s2[105] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
            result.s2[106] = -3.0/10.0*rt70*rax_val*ray_val;
        }
    }
}

/**
 * Hexadecapole-43c × Dipole-x kernel
 * Orient case 280: Q43c × Q11c
 */
void hexadecapole_43c_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czx_val = sf.czx(), cxx_val = sf.cxx(), cyx_val = sf.cyx();

    result.s0 = rt70*(9*rax_val*rax_val*rax_val*raz_val*rbx_val - 27*rax_val*ray_val*ray_val*raz_val*rbx_val +
                      rax_val*rax_val*rax_val*czx_val + 3*rax_val*rax_val*raz_val*cxx_val -
                      3*rax_val*ray_val*ray_val*czx_val - 6*rax_val*ray_val*raz_val*cyx_val -
                      3*ray_val*ray_val*raz_val*cxx_val)/20;

    if (level >= 1) {
        result.s1[0] = rt70*(27*rax_val*rax_val*raz_val*rbx_val - 27*ray_val*ray_val*raz_val*rbx_val +
                             3*rax_val*rax_val*czx_val + 6*rax_val*raz_val*cxx_val -
                             3*ray_val*ray_val*czx_val - 6*ray_val*raz_val*cyx_val)/20;
        result.s1[1] = rt70*(-54*rax_val*ray_val*raz_val*rbx_val - 6*rax_val*ray_val*czx_val -
                             6*rax_val*raz_val*cyx_val - 6*ray_val*raz_val*cxx_val)/20;
        result.s1[2] = rt70*(9*rax_val*rax_val*rax_val*rbx_val - 27*rax_val*ray_val*ray_val*rbx_val +
                             3*rax_val*rax_val*cxx_val - 6*rax_val*ray_val*cyx_val - 3*ray_val*ray_val*cxx_val)/20;
        result.s1[3] = rt70*(9*rax_val*rax_val*rax_val*raz_val - 27*rax_val*ray_val*ray_val*raz_val)/20;
        result.s1[6] = rt70*(3*rax_val*rax_val*raz_val - 3*ray_val*ray_val*raz_val)/20;
        result.s1[9] = -3.0/10.0*rt70*rax_val*ray_val*raz_val;
        result.s1[12] = rt70*(rax_val*rax_val*rax_val - 3*rax_val*ray_val*ray_val)/20;

        if (level >= 2) {
            result.s2[0] = rt70*(54*rax_val*raz_val*rbx_val + 6*rax_val*czx_val + 6*raz_val*cxx_val)/20;
            result.s2[1] = rt70*(-54*ray_val*raz_val*rbx_val - 6*ray_val*czx_val - 6*raz_val*cyx_val)/20;
            result.s2[2] = rt70*(-54*rax_val*raz_val*rbx_val - 6*rax_val*czx_val - 6*raz_val*cxx_val)/20;
            result.s2[3] = rt70*(27*rax_val*rax_val*rbx_val - 27*ray_val*ray_val*rbx_val +
                                 6*rax_val*cxx_val - 6*ray_val*cyx_val)/20;
            result.s2[4] = rt70*(-54*rax_val*ray_val*rbx_val - 6*rax_val*cyx_val - 6*ray_val*cxx_val)/20;
            result.s2[6] = rt70*(27*rax_val*rax_val*raz_val - 27*ray_val*ray_val*raz_val)/20;
            result.s2[7] = -27.0/10.0*rt70*rax_val*ray_val*raz_val;
            result.s2[8] = rt70*(9*rax_val*rax_val*rax_val - 27*rax_val*ray_val*ray_val)/20;
            result.s2[21] = 3.0/10.0*rt70*rax_val*raz_val;
            result.s2[22] = -3.0/10.0*rt70*ray_val*raz_val;
            result.s2[23] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
            result.s2[28] = -3.0/10.0*rt70*ray_val*raz_val;
            result.s2[29] = -3.0/10.0*rt70*rax_val*raz_val;
            result.s2[30] = -3.0/10.0*rt70*rax_val*ray_val;
            result.s2[36] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
            result.s2[37] = -3.0/10.0*rt70*rax_val*ray_val;
        }
    }
}

/**
 * Hexadecapole-43c × Dipole-y kernel
 * Orient case 281: Q43c × Q11s
 */
void hexadecapole_43c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czy_val = sf.czy(), cxy_val = sf.cxy(), cyy_val = sf.cyy();

    result.s0 = rt70*(9*rax_val*rax_val*rax_val*raz_val*rby_val - 27*rax_val*ray_val*ray_val*raz_val*rby_val +
                      rax_val*rax_val*rax_val*czy_val + 3*rax_val*rax_val*raz_val*cxy_val -
                      3*rax_val*ray_val*ray_val*czy_val - 6*rax_val*ray_val*raz_val*cyy_val -
                      3*ray_val*ray_val*raz_val*cxy_val)/20;

    if (level >= 1) {
        result.s1[0] = rt70*(27*rax_val*rax_val*raz_val*rby_val - 27*ray_val*ray_val*raz_val*rby_val +
                             3*rax_val*rax_val*czy_val + 6*rax_val*raz_val*cxy_val -
                             3*ray_val*ray_val*czy_val - 6*ray_val*raz_val*cyy_val)/20;
        result.s1[1] = rt70*(-54*rax_val*ray_val*raz_val*rby_val - 6*rax_val*ray_val*czy_val -
                             6*rax_val*raz_val*cyy_val - 6*ray_val*raz_val*cxy_val)/20;
        result.s1[2] = rt70*(9*rax_val*rax_val*rax_val*rby_val - 27*rax_val*ray_val*ray_val*rby_val +
                             3*rax_val*rax_val*cxy_val - 6*rax_val*ray_val*cyy_val - 3*ray_val*ray_val*cxy_val)/20;
        result.s1[4] = rt70*(9*rax_val*rax_val*rax_val*raz_val - 27*rax_val*ray_val*ray_val*raz_val)/20;
        result.s1[7] = rt70*(3*rax_val*rax_val*raz_val - 3*ray_val*ray_val*raz_val)/20;
        result.s1[10] = -3.0/10.0*rt70*rax_val*ray_val*raz_val;
        result.s1[13] = rt70*(rax_val*rax_val*rax_val - 3*rax_val*ray_val*ray_val)/20;

        if (level >= 2) {
            result.s2[0] = rt70*(54*rax_val*raz_val*rby_val + 6*rax_val*czy_val + 6*raz_val*cxy_val)/20;
            result.s2[1] = rt70*(-54*ray_val*raz_val*rby_val - 6*ray_val*czy_val - 6*raz_val*cyy_val)/20;
            result.s2[2] = rt70*(-54*rax_val*raz_val*rby_val - 6*rax_val*czy_val - 6*raz_val*cxy_val)/20;
            result.s2[3] = rt70*(27*rax_val*rax_val*rby_val - 27*ray_val*ray_val*rby_val +
                                 6*rax_val*cxy_val - 6*ray_val*cyy_val)/20;
            result.s2[4] = rt70*(-54*rax_val*ray_val*rby_val - 6*rax_val*cyy_val - 6*ray_val*cxy_val)/20;
            result.s2[10] = rt70*(27*rax_val*rax_val*raz_val - 27*ray_val*ray_val*raz_val)/20;
            result.s2[11] = -27.0/10.0*rt70*rax_val*ray_val*raz_val;
            result.s2[12] = rt70*(9*rax_val*rax_val*rax_val - 27*rax_val*ray_val*ray_val)/20;
            result.s2[45] = 3.0/10.0*rt70*rax_val*raz_val;
            result.s2[46] = -3.0/10.0*rt70*ray_val*raz_val;
            result.s2[47] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
            result.s2[55] = -3.0/10.0*rt70*ray_val*raz_val;
            result.s2[56] = -3.0/10.0*rt70*rax_val*raz_val;
            result.s2[57] = -3.0/10.0*rt70*rax_val*ray_val;
            result.s2[66] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
            result.s2[67] = -3.0/10.0*rt70*rax_val*ray_val;
        }
    }
}

/**
 * Hexadecapole-43s × Dipole-z kernel
 * Orient case 282: Q43s × Q10
 */
void hexadecapole_43s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czz_val = sf.czz(), cxz_val = sf.cxz(), cyz_val = sf.cyz();

    result.s0 = rt70*(27*rax_val*rax_val*ray_val*raz_val*rbz_val - 9*ray_val*ray_val*ray_val*raz_val*rbz_val +
                      3*rax_val*rax_val*ray_val*czz_val + 3*rax_val*rax_val*raz_val*cyz_val +
                      6*rax_val*ray_val*raz_val*cxz_val - ray_val*ray_val*ray_val*czz_val -
                      3*ray_val*ray_val*raz_val*cyz_val)/20;

    if (level >= 1) {
        result.s1[0] = rt70*(54*rax_val*ray_val*raz_val*rbz_val + 6*rax_val*ray_val*czz_val +
                             6*rax_val*raz_val*cyz_val + 6*ray_val*raz_val*cxz_val)/20;
        result.s1[1] = rt70*(27*rax_val*rax_val*raz_val*rbz_val - 27*ray_val*ray_val*raz_val*rbz_val +
                             3*rax_val*rax_val*czz_val + 6*rax_val*raz_val*cxz_val -
                             3*ray_val*ray_val*czz_val - 6*ray_val*raz_val*cyz_val)/20;
        result.s1[2] = rt70*(27*rax_val*rax_val*ray_val*rbz_val - 9*ray_val*ray_val*ray_val*rbz_val +
                             3*rax_val*rax_val*cyz_val + 6*rax_val*ray_val*cxz_val - 3*ray_val*ray_val*cyz_val)/20;
        result.s1[5] = rt70*(27*rax_val*rax_val*ray_val*raz_val - 9*ray_val*ray_val*ray_val*raz_val)/20;
        result.s1[8] = 3.0/10.0*rt70*rax_val*ray_val*raz_val;
        result.s1[11] = rt70*(3*rax_val*rax_val*raz_val - 3*ray_val*ray_val*raz_val)/20;
        result.s1[14] = rt70*(3*rax_val*rax_val*ray_val - ray_val*ray_val*ray_val)/20;

        if (level >= 2) {
            result.s2[0] = rt70*(54*ray_val*raz_val*rbz_val + 6*ray_val*czz_val + 6*raz_val*cyz_val)/20;
            result.s2[1] = rt70*(54*rax_val*raz_val*rbz_val + 6*rax_val*czz_val + 6*raz_val*cxz_val)/20;
            result.s2[2] = rt70*(-54*ray_val*raz_val*rbz_val - 6*ray_val*czz_val - 6*raz_val*cyz_val)/20;
            result.s2[3] = rt70*(54*rax_val*ray_val*rbz_val + 6*rax_val*cyz_val + 6*ray_val*cxz_val)/20;
            result.s2[4] = rt70*(27*rax_val*rax_val*rbz_val - 27*ray_val*ray_val*rbz_val +
                                 6*rax_val*cxz_val - 6*ray_val*cyz_val)/20;
            result.s2[15] = 27.0/10.0*rt70*rax_val*ray_val*raz_val;
            result.s2[16] = rt70*(27*rax_val*rax_val*raz_val - 27*ray_val*ray_val*raz_val)/20;
            result.s2[17] = rt70*(27*rax_val*rax_val*ray_val - 9*ray_val*ray_val*ray_val)/20;
            result.s2[78] = 3.0/10.0*rt70*ray_val*raz_val;
            result.s2[79] = 3.0/10.0*rt70*rax_val*raz_val;
            result.s2[80] = 3.0/10.0*rt70*rax_val*ray_val;
            result.s2[91] = 3.0/10.0*rt70*rax_val*raz_val;
            result.s2[92] = -3.0/10.0*rt70*ray_val*raz_val;
            result.s2[93] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
            result.s2[105] = 3.0/10.0*rt70*rax_val*ray_val;
            result.s2[106] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
        }
    }
}

/**
 * Hexadecapole-43s × Dipole-x kernel
 * Orient case 283: Q43s × Q11c
 */
void hexadecapole_43s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czx_val = sf.czx(), cxx_val = sf.cxx(), cyx_val = sf.cyx();

    result.s0 = rt70*(27*rax_val*rax_val*ray_val*raz_val*rbx_val - 9*ray_val*ray_val*ray_val*raz_val*rbx_val +
                      3*rax_val*rax_val*ray_val*czx_val + 3*rax_val*rax_val*raz_val*cyx_val +
                      6*rax_val*ray_val*raz_val*cxx_val - ray_val*ray_val*ray_val*czx_val -
                      3*ray_val*ray_val*raz_val*cyx_val)/20;

    if (level >= 1) {
        result.s1[0] = rt70*(54*rax_val*ray_val*raz_val*rbx_val + 6*rax_val*ray_val*czx_val +
                             6*rax_val*raz_val*cyx_val + 6*ray_val*raz_val*cxx_val)/20;
        result.s1[1] = rt70*(27*rax_val*rax_val*raz_val*rbx_val - 27*ray_val*ray_val*raz_val*rbx_val +
                             3*rax_val*rax_val*czx_val + 6*rax_val*raz_val*cxx_val -
                             3*ray_val*ray_val*czx_val - 6*ray_val*raz_val*cyx_val)/20;
        result.s1[2] = rt70*(27*rax_val*rax_val*ray_val*rbx_val - 9*ray_val*ray_val*ray_val*rbx_val +
                             3*rax_val*rax_val*cyx_val + 6*rax_val*ray_val*cxx_val - 3*ray_val*ray_val*cyx_val)/20;
        result.s1[3] = rt70*(27*rax_val*rax_val*ray_val*raz_val - 9*ray_val*ray_val*ray_val*raz_val)/20;
        result.s1[6] = 3.0/10.0*rt70*rax_val*ray_val*raz_val;
        result.s1[9] = rt70*(3*rax_val*rax_val*raz_val - 3*ray_val*ray_val*raz_val)/20;
        result.s1[12] = rt70*(3*rax_val*rax_val*ray_val - ray_val*ray_val*ray_val)/20;

        if (level >= 2) {
            result.s2[0] = rt70*(54*ray_val*raz_val*rbx_val + 6*ray_val*czx_val + 6*raz_val*cyx_val)/20;
            result.s2[1] = rt70*(54*rax_val*raz_val*rbx_val + 6*rax_val*czx_val + 6*raz_val*cxx_val)/20;
            result.s2[2] = rt70*(-54*ray_val*raz_val*rbx_val - 6*ray_val*czx_val - 6*raz_val*cyx_val)/20;
            result.s2[3] = rt70*(54*rax_val*ray_val*rbx_val + 6*rax_val*cyx_val + 6*ray_val*cxx_val)/20;
            result.s2[4] = rt70*(27*rax_val*rax_val*rbx_val - 27*ray_val*ray_val*rbx_val +
                                 6*rax_val*cxx_val - 6*ray_val*cyx_val)/20;
            result.s2[6] = 27.0/10.0*rt70*rax_val*ray_val*raz_val;
            result.s2[7] = rt70*(27*rax_val*rax_val*raz_val - 27*ray_val*ray_val*raz_val)/20;
            result.s2[8] = rt70*(27*rax_val*rax_val*ray_val - 9*ray_val*ray_val*ray_val)/20;
            result.s2[21] = 3.0/10.0*rt70*ray_val*raz_val;
            result.s2[22] = 3.0/10.0*rt70*rax_val*raz_val;
            result.s2[23] = 3.0/10.0*rt70*rax_val*ray_val;
            result.s2[28] = 3.0/10.0*rt70*rax_val*raz_val;
            result.s2[29] = -3.0/10.0*rt70*ray_val*raz_val;
            result.s2[30] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
            result.s2[36] = 3.0/10.0*rt70*rax_val*ray_val;
            result.s2[37] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
        }
    }
}

/**
 * Hexadecapole-43s × Dipole-y kernel
 * Orient case 284: Q43s × Q11s
 */
void hexadecapole_43s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double czy_val = sf.czy(), cxy_val = sf.cxy(), cyy_val = sf.cyy();

    result.s0 = rt70*(27*rax_val*rax_val*ray_val*raz_val*rby_val - 9*ray_val*ray_val*ray_val*raz_val*rby_val +
                      3*rax_val*rax_val*ray_val*czy_val + 3*rax_val*rax_val*raz_val*cyy_val +
                      6*rax_val*ray_val*raz_val*cxy_val - ray_val*ray_val*ray_val*czy_val -
                      3*ray_val*ray_val*raz_val*cyy_val)/20;

    if (level >= 1) {
        result.s1[0] = rt70*(54*rax_val*ray_val*raz_val*rby_val + 6*rax_val*ray_val*czy_val +
                             6*rax_val*raz_val*cyy_val + 6*ray_val*raz_val*cxy_val)/20;
        result.s1[1] = rt70*(27*rax_val*rax_val*raz_val*rby_val - 27*ray_val*ray_val*raz_val*rby_val +
                             3*rax_val*rax_val*czy_val + 6*rax_val*raz_val*cxy_val -
                             3*ray_val*ray_val*czy_val - 6*ray_val*raz_val*cyy_val)/20;
        result.s1[2] = rt70*(27*rax_val*rax_val*ray_val*rby_val - 9*ray_val*ray_val*ray_val*rby_val +
                             3*rax_val*rax_val*cyy_val + 6*rax_val*ray_val*cxy_val - 3*ray_val*ray_val*cyy_val)/20;
        result.s1[4] = rt70*(27*rax_val*rax_val*ray_val*raz_val - 9*ray_val*ray_val*ray_val*raz_val)/20;
        result.s1[7] = 3.0/10.0*rt70*rax_val*ray_val*raz_val;
        result.s1[10] = rt70*(3*rax_val*rax_val*raz_val - 3*ray_val*ray_val*raz_val)/20;
        result.s1[13] = rt70*(3*rax_val*rax_val*ray_val - ray_val*ray_val*ray_val)/20;

        if (level >= 2) {
            result.s2[0] = rt70*(54*ray_val*raz_val*rby_val + 6*ray_val*czy_val + 6*raz_val*cyy_val)/20;
            result.s2[1] = rt70*(54*rax_val*raz_val*rby_val + 6*rax_val*czy_val + 6*raz_val*cxy_val)/20;
            result.s2[2] = rt70*(-54*ray_val*raz_val*rby_val - 6*ray_val*czy_val - 6*raz_val*cyy_val)/20;
            result.s2[3] = rt70*(54*rax_val*ray_val*rby_val + 6*rax_val*cyy_val + 6*ray_val*cxy_val)/20;
            result.s2[4] = rt70*(27*rax_val*rax_val*rby_val - 27*ray_val*ray_val*rby_val +
                                 6*rax_val*cxy_val - 6*ray_val*cyy_val)/20;
            result.s2[10] = 27.0/10.0*rt70*rax_val*ray_val*raz_val;
            result.s2[11] = rt70*(27*rax_val*rax_val*raz_val - 27*ray_val*ray_val*raz_val)/20;
            result.s2[12] = rt70*(27*rax_val*rax_val*ray_val - 9*ray_val*ray_val*ray_val)/20;
            result.s2[45] = 3.0/10.0*rt70*ray_val*raz_val;
            result.s2[46] = 3.0/10.0*rt70*rax_val*raz_val;
            result.s2[47] = 3.0/10.0*rt70*rax_val*ray_val;
            result.s2[55] = 3.0/10.0*rt70*rax_val*raz_val;
            result.s2[56] = -3.0/10.0*rt70*ray_val*raz_val;
            result.s2[57] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
            result.s2[66] = 3.0/10.0*rt70*rax_val*ray_val;
            result.s2[67] = rt70*(3*rax_val*rax_val - 3*ray_val*ray_val)/20;
        }
    }
}

/**
 * Hexadecapole-44c × Dipole-z kernel
 * Orient case 285: Q44c × Q10
 */
void hexadecapole_44c_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxz_val = sf.cxz(), cyz_val = sf.cyz();

    result.s0 = rt35*(9*rax_val*rax_val*rax_val*rax_val*rbz_val - 54*rax_val*rax_val*ray_val*ray_val*rbz_val +
                      9*ray_val*ray_val*ray_val*ray_val*rbz_val + 4*rax_val*rax_val*rax_val*cxz_val -
                      12*rax_val*rax_val*ray_val*cyz_val - 12*rax_val*ray_val*ray_val*cxz_val +
                      4*ray_val*ray_val*ray_val*cyz_val)/40;

    if (level >= 1) {
        result.s1[0] = rt35*(36*rax_val*rax_val*rax_val*rbz_val - 108*rax_val*ray_val*ray_val*rbz_val +
                             12*rax_val*rax_val*cxz_val - 24*rax_val*ray_val*cyz_val - 12*ray_val*ray_val*cxz_val)/40;
        result.s1[1] = rt35*(-108*rax_val*rax_val*ray_val*rbz_val + 36*ray_val*ray_val*ray_val*rbz_val -
                             12*rax_val*rax_val*cyz_val - 24*rax_val*ray_val*cxz_val + 12*ray_val*ray_val*cyz_val)/40;
        result.s1[5] = rt35*(9*rax_val*rax_val*rax_val*rax_val - 54*rax_val*rax_val*ray_val*ray_val +
                             9*ray_val*ray_val*ray_val*ray_val)/40;
        result.s1[8] = rt35*(4*rax_val*rax_val*rax_val - 12*rax_val*ray_val*ray_val)/40;
        result.s1[11] = rt35*(-12*rax_val*rax_val*ray_val + 4*ray_val*ray_val*ray_val)/40;

        if (level >= 2) {
            result.s2[0] = rt35*(108*rax_val*rax_val*rbz_val - 108*ray_val*ray_val*rbz_val +
                                 24*rax_val*cxz_val - 24*ray_val*cyz_val)/40;
            result.s2[1] = rt35*(-216*rax_val*ray_val*rbz_val - 24*rax_val*cyz_val - 24*ray_val*cxz_val)/40;
            result.s2[2] = rt35*(-108*rax_val*rax_val*rbz_val + 108*ray_val*ray_val*rbz_val -
                                 24*rax_val*cxz_val + 24*ray_val*cyz_val)/40;
            result.s2[15] = rt35*(36*rax_val*rax_val*rax_val - 108*rax_val*ray_val*ray_val)/40;
            result.s2[16] = rt35*(-108*rax_val*rax_val*ray_val + 36*ray_val*ray_val*ray_val)/40;
            result.s2[78] = rt35*(12*rax_val*rax_val - 12*ray_val*ray_val)/40;
            result.s2[79] = -3.0/5.0*rt35*rax_val*ray_val;
            result.s2[91] = -3.0/5.0*rt35*rax_val*ray_val;
            result.s2[92] = rt35*(-12*rax_val*rax_val + 12*ray_val*ray_val)/40;
        }
    }
}

/**
 * Hexadecapole-44c × Dipole-x kernel
 * Orient case 286: Q44c × Q11c
 */
void hexadecapole_44c_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cyx_val = sf.cyx();

    result.s0 = rt35*(9*rax_val*rax_val*rax_val*rax_val*rbx_val - 54*rax_val*rax_val*ray_val*ray_val*rbx_val +
                      9*ray_val*ray_val*ray_val*ray_val*rbx_val + 4*rax_val*rax_val*rax_val*cxx_val -
                      12*rax_val*rax_val*ray_val*cyx_val - 12*rax_val*ray_val*ray_val*cxx_val +
                      4*ray_val*ray_val*ray_val*cyx_val)/40;

    if (level >= 1) {
        result.s1[0] = rt35*(36*rax_val*rax_val*rax_val*rbx_val - 108*rax_val*ray_val*ray_val*rbx_val +
                             12*rax_val*rax_val*cxx_val - 24*rax_val*ray_val*cyx_val - 12*ray_val*ray_val*cxx_val)/40;
        result.s1[1] = rt35*(-108*rax_val*rax_val*ray_val*rbx_val + 36*ray_val*ray_val*ray_val*rbx_val -
                             12*rax_val*rax_val*cyx_val - 24*rax_val*ray_val*cxx_val + 12*ray_val*ray_val*cyx_val)/40;
        result.s1[3] = rt35*(9*rax_val*rax_val*rax_val*rax_val - 54*rax_val*rax_val*ray_val*ray_val +
                             9*ray_val*ray_val*ray_val*ray_val)/40;
        result.s1[6] = rt35*(4*rax_val*rax_val*rax_val - 12*rax_val*ray_val*ray_val)/40;
        result.s1[9] = rt35*(-12*rax_val*rax_val*ray_val + 4*ray_val*ray_val*ray_val)/40;

        if (level >= 2) {
            result.s2[0] = rt35*(108*rax_val*rax_val*rbx_val - 108*ray_val*ray_val*rbx_val +
                                 24*rax_val*cxx_val - 24*ray_val*cyx_val)/40;
            result.s2[1] = rt35*(-216*rax_val*ray_val*rbx_val - 24*rax_val*cyx_val - 24*ray_val*cxx_val)/40;
            result.s2[2] = rt35*(-108*rax_val*rax_val*rbx_val + 108*ray_val*ray_val*rbx_val -
                                 24*rax_val*cxx_val + 24*ray_val*cyx_val)/40;
            result.s2[6] = rt35*(36*rax_val*rax_val*rax_val - 108*rax_val*ray_val*ray_val)/40;
            result.s2[7] = rt35*(-108*rax_val*rax_val*ray_val + 36*ray_val*ray_val*ray_val)/40;
            result.s2[21] = rt35*(12*rax_val*rax_val - 12*ray_val*ray_val)/40;
            result.s2[22] = -3.0/5.0*rt35*rax_val*ray_val;
            result.s2[28] = -3.0/5.0*rt35*rax_val*ray_val;
            result.s2[29] = rt35*(-12*rax_val*rax_val + 12*ray_val*ray_val)/40;
        }
    }
}

/**
 * Hexadecapole-44c × Dipole-y kernel
 * Orient case 287: Q44c × Q11s
 */
void hexadecapole_44c_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxy_val = sf.cxy(), cyy_val = sf.cyy();

    result.s0 = rt35*(9*rax_val*rax_val*rax_val*rax_val*rby_val - 54*rax_val*rax_val*ray_val*ray_val*rby_val +
                      9*ray_val*ray_val*ray_val*ray_val*rby_val + 4*rax_val*rax_val*rax_val*cxy_val -
                      12*rax_val*rax_val*ray_val*cyy_val - 12*rax_val*ray_val*ray_val*cxy_val +
                      4*ray_val*ray_val*ray_val*cyy_val)/40;

    if (level >= 1) {
        result.s1[0] = rt35*(36*rax_val*rax_val*rax_val*rby_val - 108*rax_val*ray_val*ray_val*rby_val +
                             12*rax_val*rax_val*cxy_val - 24*rax_val*ray_val*cyy_val - 12*ray_val*ray_val*cxy_val)/40;
        result.s1[1] = rt35*(-108*rax_val*rax_val*ray_val*rby_val + 36*ray_val*ray_val*ray_val*rby_val -
                             12*rax_val*rax_val*cyy_val - 24*rax_val*ray_val*cxy_val + 12*ray_val*ray_val*cyy_val)/40;
        result.s1[4] = rt35*(9*rax_val*rax_val*rax_val*rax_val - 54*rax_val*rax_val*ray_val*ray_val +
                             9*ray_val*ray_val*ray_val*ray_val)/40;
        result.s1[7] = rt35*(4*rax_val*rax_val*rax_val - 12*rax_val*ray_val*ray_val)/40;
        result.s1[10] = rt35*(-12*rax_val*rax_val*ray_val + 4*ray_val*ray_val*ray_val)/40;

        if (level >= 2) {
            result.s2[0] = rt35*(108*rax_val*rax_val*rby_val - 108*ray_val*ray_val*rby_val +
                                 24*rax_val*cxy_val - 24*ray_val*cyy_val)/40;
            result.s2[1] = rt35*(-216*rax_val*ray_val*rby_val - 24*rax_val*cyy_val - 24*ray_val*cxy_val)/40;
            result.s2[2] = rt35*(-108*rax_val*rax_val*rby_val + 108*ray_val*ray_val*rby_val -
                                 24*rax_val*cxy_val + 24*ray_val*cyy_val)/40;
            result.s2[10] = rt35*(36*rax_val*rax_val*rax_val - 108*rax_val*ray_val*ray_val)/40;
            result.s2[11] = rt35*(-108*rax_val*rax_val*ray_val + 36*ray_val*ray_val*ray_val)/40;
            result.s2[45] = rt35*(12*rax_val*rax_val - 12*ray_val*ray_val)/40;
            result.s2[46] = -3.0/5.0*rt35*rax_val*ray_val;
            result.s2[55] = -3.0/5.0*rt35*rax_val*ray_val;
            result.s2[56] = rt35*(-12*rax_val*rax_val + 12*ray_val*ray_val)/40;
        }
    }
}

/**
 * Hexadecapole-44s × Dipole-z kernel
 * Orient case 288: Q44s × Q10
 */
void hexadecapole_44s_dipole_10(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxz_val = sf.cxz(), cyz_val = sf.cyz();

    result.s0 = rt35*(9*rax_val*rax_val*rax_val*ray_val*rbz_val - 9*rax_val*ray_val*ray_val*ray_val*rbz_val +
                      rax_val*rax_val*rax_val*cyz_val + 3*rax_val*rax_val*ray_val*cxz_val -
                      3*rax_val*ray_val*ray_val*cyz_val - ray_val*ray_val*ray_val*cxz_val)/10;

    if (level >= 1) {
        result.s1[0] = rt35*(27*rax_val*rax_val*ray_val*rbz_val - 9*ray_val*ray_val*ray_val*rbz_val +
                             3*rax_val*rax_val*cyz_val + 6*rax_val*ray_val*cxz_val - 3*ray_val*ray_val*cyz_val)/10;
        result.s1[1] = rt35*(9*rax_val*rax_val*rax_val*rbz_val - 27*rax_val*ray_val*ray_val*rbz_val +
                             3*rax_val*rax_val*cxz_val - 6*rax_val*ray_val*cyz_val - 3*ray_val*ray_val*cxz_val)/10;
        result.s1[5] = rt35*(9*rax_val*rax_val*rax_val*ray_val - 9*rax_val*ray_val*ray_val*ray_val)/10;
        result.s1[8] = rt35*(3*rax_val*rax_val*ray_val - ray_val*ray_val*ray_val)/10;
        result.s1[11] = rt35*(rax_val*rax_val*rax_val - 3*rax_val*ray_val*ray_val)/10;

        if (level >= 2) {
            result.s2[0] = rt35*(54*rax_val*ray_val*rbz_val + 6*rax_val*cyz_val + 6*ray_val*cxz_val)/10;
            result.s2[1] = rt35*(27*rax_val*rax_val*rbz_val - 27*ray_val*ray_val*rbz_val +
                                 6*rax_val*cxz_val - 6*ray_val*cyz_val)/10;
            result.s2[2] = rt35*(-54*rax_val*ray_val*rbz_val - 6*rax_val*cyz_val - 6*ray_val*cxz_val)/10;
            result.s2[15] = rt35*(27*rax_val*rax_val*ray_val - 9*ray_val*ray_val*ray_val)/10;
            result.s2[16] = rt35*(9*rax_val*rax_val*rax_val - 27*rax_val*ray_val*ray_val)/10;
            result.s2[78] = 3.0/5.0*rt35*rax_val*ray_val;
            result.s2[79] = rt35*(3*rax_val*rax_val - 3*ray_val*ray_val)/10;
            result.s2[91] = rt35*(3*rax_val*rax_val - 3*ray_val*ray_val)/10;
            result.s2[92] = -3.0/5.0*rt35*rax_val*ray_val;
        }
    }
}

/**
 * Hexadecapole-44s × Dipole-x kernel
 * Orient case 289: Q44s × Q11c
 */
void hexadecapole_44s_dipole_11c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxx_val = sf.cxx(), cyx_val = sf.cyx();

    result.s0 = rt35*(9*rax_val*rax_val*rax_val*ray_val*rbx_val - 9*rax_val*ray_val*ray_val*ray_val*rbx_val +
                      rax_val*rax_val*rax_val*cyx_val + 3*rax_val*rax_val*ray_val*cxx_val -
                      3*rax_val*ray_val*ray_val*cyx_val - ray_val*ray_val*ray_val*cxx_val)/10;

    if (level >= 1) {
        result.s1[0] = rt35*(27*rax_val*rax_val*ray_val*rbx_val - 9*ray_val*ray_val*ray_val*rbx_val +
                             3*rax_val*rax_val*cyx_val + 6*rax_val*ray_val*cxx_val - 3*ray_val*ray_val*cyx_val)/10;
        result.s1[1] = rt35*(9*rax_val*rax_val*rax_val*rbx_val - 27*rax_val*ray_val*ray_val*rbx_val +
                             3*rax_val*rax_val*cxx_val - 6*rax_val*ray_val*cyx_val - 3*ray_val*ray_val*cxx_val)/10;
        result.s1[3] = rt35*(9*rax_val*rax_val*rax_val*ray_val - 9*rax_val*ray_val*ray_val*ray_val)/10;
        result.s1[6] = rt35*(3*rax_val*rax_val*ray_val - ray_val*ray_val*ray_val)/10;
        result.s1[9] = rt35*(rax_val*rax_val*rax_val - 3*rax_val*ray_val*ray_val)/10;

        if (level >= 2) {
            result.s2[0] = rt35*(54*rax_val*ray_val*rbx_val + 6*rax_val*cyx_val + 6*ray_val*cxx_val)/10;
            result.s2[1] = rt35*(27*rax_val*rax_val*rbx_val - 27*ray_val*ray_val*rbx_val +
                                 6*rax_val*cxx_val - 6*ray_val*cyx_val)/10;
            result.s2[2] = rt35*(-54*rax_val*ray_val*rbx_val - 6*rax_val*cyx_val - 6*ray_val*cxx_val)/10;
            result.s2[6] = rt35*(27*rax_val*rax_val*ray_val - 9*ray_val*ray_val*ray_val)/10;
            result.s2[7] = rt35*(9*rax_val*rax_val*rax_val - 27*rax_val*ray_val*ray_val)/10;
            result.s2[21] = 3.0/5.0*rt35*rax_val*ray_val;
            result.s2[22] = rt35*(3*rax_val*rax_val - 3*ray_val*ray_val)/10;
            result.s2[28] = rt35*(3*rax_val*rax_val - 3*ray_val*ray_val)/10;
            result.s2[29] = -3.0/5.0*rt35*rax_val*ray_val;
        }
    }
}

/**
 * Hexadecapole-44s × Dipole-y kernel
 * Orient case 290: Q44s × Q11s
 */
void hexadecapole_44s_dipole_11s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    double rax_val = sf.rax(), ray_val = sf.ray(), raz_val = sf.raz();
    double rbx_val = sf.rbx(), rby_val = sf.rby(), rbz_val = sf.rbz();
    double cxy_val = sf.cxy(), cyy_val = sf.cyy();

    result.s0 = rt35*(9*rax_val*rax_val*rax_val*ray_val*rby_val - 9*rax_val*ray_val*ray_val*ray_val*rby_val +
                      rax_val*rax_val*rax_val*cyy_val + 3*rax_val*rax_val*ray_val*cxy_val -
                      3*rax_val*ray_val*ray_val*cyy_val - ray_val*ray_val*ray_val*cxy_val)/10;

    if (level >= 1) {
        result.s1[0] = rt35*(27*rax_val*rax_val*ray_val*rby_val - 9*ray_val*ray_val*ray_val*rby_val +
                             3*rax_val*rax_val*cyy_val + 6*rax_val*ray_val*cxy_val - 3*ray_val*ray_val*cyy_val)/10;
        result.s1[1] = rt35*(9*rax_val*rax_val*rax_val*rby_val - 27*rax_val*ray_val*ray_val*rby_val +
                             3*rax_val*rax_val*cxy_val - 6*rax_val*ray_val*cyy_val - 3*ray_val*ray_val*cxy_val)/10;
        result.s1[4] = rt35*(9*rax_val*rax_val*rax_val*ray_val - 9*rax_val*ray_val*ray_val*ray_val)/10;
        result.s1[7] = rt35*(3*rax_val*rax_val*ray_val - ray_val*ray_val*ray_val)/10;
        result.s1[10] = rt35*(rax_val*rax_val*rax_val - 3*rax_val*ray_val*ray_val)/10;

        if (level >= 2) {
            result.s2[0] = rt35*(54*rax_val*ray_val*rby_val + 6*rax_val*cyy_val + 6*ray_val*cxy_val)/10;
            result.s2[1] = rt35*(27*rax_val*rax_val*rby_val - 27*ray_val*ray_val*rby_val +
                                 6*rax_val*cxy_val - 6*ray_val*cyy_val)/10;
            result.s2[2] = rt35*(-54*rax_val*ray_val*rby_val - 6*rax_val*cyy_val - 6*ray_val*cxy_val)/10;
            result.s2[10] = rt35*(27*rax_val*rax_val*ray_val - 9*ray_val*ray_val*ray_val)/10;
            result.s2[11] = rt35*(9*rax_val*rax_val*rax_val - 27*rax_val*ray_val*ray_val)/10;
            result.s2[45] = 3.0/5.0*rt35*rax_val*ray_val;
            result.s2[46] = rt35*(3*rax_val*rax_val - 3*ray_val*ray_val)/10;
            result.s2[55] = rt35*(3*rax_val*rax_val - 3*ray_val*ray_val)/10;
            result.s2[56] = -3.0/5.0*rt35*rax_val*ray_val;
        }
    }
}

} // namespace occ::mults::kernels
