#include <occ/mults/sfunctions.h>

namespace occ::mults::kernels {

// ============================================================================
// DIPOLE-QUADRUPOLE KERNELS (Orient cases 34-48)
// Dipole @ A (uses rax, ray, raz), Quadrupole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Dipole-z × Quadrupole-20 kernel
 * Orient case 34: Q10 × Q20
 * Formula: S0 = 5/2*rbz²*raz - raz/2 + rbz*czz
 */
void dipole_z_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = 2.5 * sf.rbz() * sf.rbz() * sf.raz() - 0.5 * sf.raz() + sf.rbz() * sf.czz();

    if (level >= 1) {
        result.s1[2] = 2.5 * sf.rbz() * sf.rbz() - 0.5; // d/d(raz)
        result.s1[5] = 5.0 * sf.raz() * sf.rbz() + sf.czz(); // d/d(rbz) [includes czz term from rbz*czz]
        result.s1[14] = sf.rbz();                       // d/d(czz)
        if (level >= 2) {
            result.s2[17] = 5.0 * sf.rbz(); // d²/d(raz)d(rbz)
            result.s2[20] = 5.0 * sf.raz(); // d²/d(rbz)²
        }
    }
}

/**
 * Dipole-z × Quadrupole-21c kernel
 * Orient case 35: Q10 × Q21c
 * Formula: S0 = rt3*(5*rbx*rbz*raz + rbx*czz + rbz*czx)/3
 */
void dipole_z_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rbx() * sf.rbz() * sf.raz() + sf.rbx() * sf.czz() + sf.rbz() * sf.czx()) / 3.0;

    if (level >= 1) {
        result.s1[2] = rt3 * 5.0 / 3.0 * sf.rbx() * sf.rbz(); // d/d(raz)
        result.s1[3] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rbz(); // d/d(rbx)
        result.s1[5] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rbx(); // d/d(rbz)
        result.s1[12] = rt3 / 3.0 * sf.rbz();                  // d/d(czx)
        result.s1[14] = rt3 / 3.0 * sf.rbx();                 // d/d(czz)
        if (level >= 2) {
            result.s2[8] = rt3 * 5.0 / 3.0 * sf.rbz();  // d²/d(raz)d(rbx)
            result.s2[17] = rt3 * 5.0 / 3.0 * sf.rbx(); // d²/d(raz)d(rbz)
            result.s2[18] = rt3 * 5.0 / 3.0 * sf.raz(); // d²/d(rbx)d(rbz)
        }
    }
}

/**
 * Dipole-z × Quadrupole-21s kernel
 * Orient case 36: Q10 × Q21s
 * Formula: S0 = rt3*(5*rby*rbz*raz + rby*czz + rbz*czy)/3
 */
void dipole_z_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rby() * sf.rbz() * sf.raz() + sf.rby() * sf.czz() + sf.rbz() * sf.czy()) / 3.0;

    if (level >= 1) {
        result.s1[2] = rt3 * 5.0 / 3.0 * sf.rby() * sf.rbz(); // d/d(raz)
        result.s1[4] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rbz(); // d/d(rby)
        result.s1[5] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rby(); // d/d(rbz)
        result.s1[13] = rt3 / 3.0 * sf.rbz();                 // d/d(czy)
        result.s1[14] = rt3 / 3.0 * sf.rby();                 // d/d(czz)
        if (level >= 2) {
            result.s2[12] = rt3 * 5.0 / 3.0 * sf.rbz(); // d²/d(raz)d(rby)
            result.s2[17] = rt3 * 5.0 / 3.0 * sf.rby(); // d²/d(raz)d(rbz)
            result.s2[19] = rt3 * 5.0 / 3.0 * sf.raz(); // d²/d(rby)d(rbz)
        }
    }
}

/**
 * Dipole-z × Quadrupole-22c kernel
 * Orient case 37: Q10 × Q22c
 * Formula: S0 = rt3*(5*rbx²*raz - 5*rby²*raz + 2*rbx*czx - 2*rby*czy)/6
 */
void dipole_z_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rbx() * sf.rbx() * sf.raz() - 5.0 * sf.rby() * sf.rby() * sf.raz() + 2.0 * sf.rbx() * sf.czx() - 2.0 * sf.rby() * sf.czy()) / 6.0;

    if (level >= 1) {
        result.s1[2] = rt3 * (5.0 * sf.rbx() * sf.rbx() - 5.0 * sf.rby() * sf.rby()) / 6.0; // d/d(raz)
        result.s1[3] = rt3 * 10.0 / 6.0 * sf.raz() * sf.rbx();             // d/d(rbx)
        result.s1[4] = rt3 * (-10.0) / 6.0 * sf.raz() * sf.rby();          // d/d(rby)
        result.s1[12] = rt3 * 2.0 / 6.0 * sf.rbx();                         // d/d(czx)
        result.s1[13] = rt3 * (-2.0) / 6.0 * sf.rby();                     // d/d(czy)
        if (level >= 2) {
            result.s2[8] = rt3 * 5.0 / 3.0 * sf.rbx();     // d²/d(raz)d(rbx)
            result.s2[9] = rt3 * 5.0 / 3.0 * sf.raz();     // d²/d(rbx)²
            result.s2[12] = rt3 * (-5.0) / 3.0 * sf.rby(); // d²/d(raz)d(rby)
            result.s2[14] = rt3 * (-5.0) / 3.0 * sf.raz(); // d²/d(rby)²
        }
    }
}

/**
 * Dipole-z × Quadrupole-22s kernel
 * Orient case 38: Q10 × Q22s
 * Formula: S0 = rt3*(5*rbx*rby*raz + rbx*czy + rby*czx)/3
 */
void dipole_z_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rbx() * sf.rby() * sf.raz() + sf.rbx() * sf.czy() + sf.rby() * sf.czx()) / 3.0;

    if (level >= 1) {
        result.s1[2] = rt3 * 5.0 / 3.0 * sf.rbx() * sf.rby(); // d/d(raz)
        result.s1[3] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rby(); // d/d(rbx)
        result.s1[4] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rbx(); // d/d(rby)
        result.s1[12] = rt3 / 3.0 * sf.rby();                  // d/d(czx)
        result.s1[13] = rt3 / 3.0 * sf.rbx();                 // d/d(czy)
        if (level >= 2) {
            result.s2[8] = rt3 * 5.0 / 3.0 * sf.rby();  // d²/d(raz)d(rbx)
            result.s2[12] = rt3 * 5.0 / 3.0 * sf.rbx(); // d²/d(raz)d(rby)
            result.s2[13] = rt3 * 5.0 / 3.0 * sf.raz(); // d²/d(rbx)d(rby)
        }
    }
}

/**
 * Dipole-x × Quadrupole-20 kernel
 * Orient case 39: Q11c × Q20
 * Formula: S0 = 5/2*rbz²*rax - rax/2 + rbz*cxz
 */
void dipole_x_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = 2.5 * sf.rbz() * sf.rbz() * sf.rax() - 0.5 * sf.rax() + sf.rbz() * sf.cxz();

    if (level >= 1) {
        result.s1[0] = 2.5 * sf.rbz() * sf.rbz() - 0.5; // d/d(rax)
        result.s1[5] = 5.0 * sf.rbz() * sf.rax();       // d/d(rbz)
        result.s1[8] = sf.rbz();                       // d/d(cxz)
        if (level >= 2) {
            result.s2[15] = 5.0 * sf.rbz(); // d²/d(rax)d(rbz)
            result.s2[20] = 5.0 * sf.rax(); // d²/d(rbz)²
        }
    }
}

/**
 * Dipole-x × Quadrupole-21c kernel
 * Orient case 40: Q11c × Q21c
 * Formula: S0 = rt3*(5*rbx*rbz*rax + rbx*cxz + rbz*cxx)/3
 */
void dipole_x_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rbx() * sf.rbz() * sf.rax() + sf.rbx() * sf.cxz() + sf.rbz() * sf.cxx()) / 3.0;

    if (level >= 1) {
        result.s1[0] = rt3 * 5.0 / 3.0 * sf.rbx() * sf.rbz(); // d/d(rax)
        result.s1[3] = rt3 * (5.0 * sf.rbz() * sf.rax() + sf.cxz()) / 3.0; // d/d(rbx)
        result.s1[5] = rt3 * (5.0 * sf.rax() * sf.rbx() + sf.cxx()) / 3.0; // d/d(rbz)
        result.s1[6] = rt3 / 3.0 * sf.rbz();                  // d/d(cxx)
        result.s1[8] = rt3 / 3.0 * sf.rbx();                 // d/d(cxz)
        if (level >= 2) {
            result.s2[6] = rt3 * 5.0 / 3.0 * sf.rbz();  // d²/d(rax)d(rbx)
            result.s2[15] = rt3 * 5.0 / 3.0 * sf.rbx(); // d²/d(rax)d(rbz)
            result.s2[18] = rt3 * 5.0 / 3.0 * sf.rax(); // d²/d(rbx)d(rbz)
        }
    }
}

/**
 * Dipole-x × Quadrupole-21s kernel
 * Orient case 41: Q11c × Q21s
 * Formula: S0 = rt3*(5*rby*rbz*rax + rby*cxz + rbz*cxy)/3
 */
void dipole_x_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rby() * sf.rbz() * sf.rax() + sf.rby() * sf.cxz() + sf.rbz() * sf.cxy()) / 3.0;

    if (level >= 1) {
        result.s1[0] = rt3 * 5.0 / 3.0 * sf.rby() * sf.rbz(); // d/d(rax)
        result.s1[4] = rt3 * 5.0 / 3.0 * sf.rbz() * sf.rax(); // d/d(rby)
        result.s1[5] = rt3 * 5.0 / 3.0 * sf.rax() * sf.rby(); // d/d(rbz)
        result.s1[7] = rt3 / 3.0 * sf.rbz();                  // d/d(cxy)
        result.s1[8] = rt3 / 3.0 * sf.rby();                 // d/d(cxz)
        if (level >= 2) {
            result.s2[10] = rt3 * 5.0 / 3.0 * sf.rbz(); // d²/d(rax)d(rby)
            result.s2[15] = rt3 * 5.0 / 3.0 * sf.rby(); // d²/d(rax)d(rbz)
            result.s2[19] = rt3 * 5.0 / 3.0 * sf.rax(); // d²/d(rby)d(rbz)
        }
    }
}

/**
 * Dipole-x × Quadrupole-22c kernel
 * Orient case 42: Q11c × Q22c
 * Formula: S0 = rt3*(5*rbx²*rax - 5*rby²*rax + 2*rbx*cxx - 2*rby*cxy)/6
 */
void dipole_x_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rbx() * sf.rbx() * sf.rax() - 5.0 * sf.rby() * sf.rby() * sf.rax() + 2.0 * sf.rbx() * sf.cxx() - 2.0 * sf.rby() * sf.cxy()) / 6.0;
    // Debug output disabled
    // fmt::print("DEBUG case42: dip={}, quad={}, rax={}, rbx={}, rby={}, cxx={}, cxy={}, s0={}\n",
    //           dip_comp, quad_comp, rax(), rbx(), rby(), cxx(), cxy(), result.s0);
    if (level >= 1) {
        result.s1[0] = rt3 * (5.0 * sf.rbx() * sf.rbx() - 5.0 * sf.rby() * sf.rby()) / 6.0; // d/d(rax)
        result.s1[3] = rt3 * (10.0 * sf.rax() * sf.rbx() + 2.0 * sf.cxx()) / 6.0; // d/d(rbx)
        result.s1[4] = rt3 * (-10.0 * sf.rax() * sf.rby()) / 6.0; // d/d(rby)
        result.s1[6] = rt3 * 2.0 / 6.0 * sf.rbx();                // d/d(cxx)
        result.s1[7] = rt3 * (-2.0) / 6.0 * sf.rby();             // d/d(cxy)
        if (level >= 2) {
            result.s2[6] = rt3 * 5.0 / 3.0 * sf.rbx();     // d²/d(rax)d(rbx)
            result.s2[9] = rt3 * 5.0 / 3.0 * sf.rax();     // d²/d(rbx)²
            result.s2[10] = rt3 * (-5.0) / 3.0 * sf.rby(); // d²/d(rax)d(rby)
            result.s2[14] = rt3 * (-5.0) / 3.0 * sf.rax(); // d²/d(rby)²
        }
    }
}

/**
 * Dipole-x × Quadrupole-22s kernel
 * Orient case 43: Q11c × Q22s
 * Formula: S0 = rt3*(5*rbx*rby*rax + rbx*cxy + rby*cxx)/3
 */
void dipole_x_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rbx() * sf.rby() * sf.rax() + sf.rbx() * sf.cxy() + sf.rby() * sf.cxx()) / 3.0;

    if (level >= 1) {
        result.s1[0] = rt3 * 5.0 / 3.0 * sf.rbx() * sf.rby();   // d/d(rax)
        result.s1[3] = rt3 * (5.0 * sf.rax() * sf.rby()) / 3.0; // d/d(rbx)
        result.s1[4] = rt3 * (5.0 * sf.rax() * sf.rbx() + sf.cxx()) / 3.0; // d/d(rby)
        result.s1[6] = rt3 / 3.0 * sf.rby();                    // d/d(cxx)
        result.s1[7] = rt3 / 3.0 * sf.rbx();                    // d/d(cxy)
        if (level >= 2) {
            result.s2[6] = rt3 * 5.0 / 3.0 * sf.rby();  // d²/d(rax)d(rbx)
            result.s2[10] = rt3 * 5.0 / 3.0 * sf.rbx(); // d²/d(rax)d(rby)
            result.s2[13] = rt3 * 5.0 / 3.0 * sf.rax(); // d²/d(rbx)d(rby)
        }
    }
}

/**
 * Dipole-y × Quadrupole-20 kernel
 * Orient case 44: Q11s × Q20
 * Formula: S0 = 5/2*rbz²*ray - ray/2 + rbz*cyz
 */
void dipole_y_quadrupole_20(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = 2.5 * sf.rbz() * sf.rbz() * sf.ray() - 0.5 * sf.ray() + sf.rbz() * sf.cyz();

    if (level >= 1) {
        result.s1[1] = 2.5 * sf.rbz() * sf.rbz() - 0.5; // d/d(ray)
        result.s1[5] = 5.0 * sf.rbz() * sf.ray();       // d/d(rbz)
        result.s1[11] = sf.rbz();                       // d/d(cyz)
        if (level >= 2) {
            result.s2[16] = 5.0 * sf.rbz(); // d²/d(ray)d(rbz)
            result.s2[20] = 5.0 * sf.ray(); // d²/d(rbz)²
        }
    }
}

/**
 * Dipole-y × Quadrupole-21c kernel
 * Orient case 45: Q11s × Q21c
 * Formula: S0 = rt3*(5*rbx*rbz*ray + rbx*cyz + rbz*cyx)/3
 */
void dipole_y_quadrupole_21c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rbx() * sf.rbz() * sf.ray() + sf.rbx() * sf.cyz() + sf.rbz() * sf.cyx()) / 3.0;

    if (level >= 1) {
        result.s1[1] = rt3 * 5.0 / 3.0 * sf.rbx() * sf.rbz(); // d/d(ray)
        result.s1[3] = rt3 * 5.0 / 3.0 * sf.rbz() * sf.ray(); // d/d(rbx)
        result.s1[5] = rt3 * 5.0 / 3.0 * sf.rbx() * sf.ray(); // d/d(rbz)
        result.s1[9] = rt3 / 3.0 * sf.rbz();                  // d/d(cyx)
        result.s1[11] = rt3 / 3.0 * sf.rbx();                 // d/d(cyz)
        if (level >= 2) {
            result.s2[7] = rt3 * 5.0 / 3.0 * sf.rbz();  // d²/d(ray)d(rbx)
            result.s2[16] = rt3 * 5.0 / 3.0 * sf.rbx(); // d²/d(ray)d(rbz)
            result.s2[18] = rt3 * 5.0 / 3.0 * sf.ray(); // d²/d(rbx)d(rbz)
        }
    }
}

/**
 * Dipole-y × Quadrupole-21s kernel
 * Orient case 46: Q11s × Q21s
 * Formula: S0 = rt3*(5*rby*rbz*ray + rby*cyz + rbz*cyy)/3
 */
void dipole_y_quadrupole_21s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rby() * sf.rbz() * sf.ray() + sf.rby() * sf.cyz() + sf.rbz() * sf.cyy()) / 3.0;

    if (level >= 1) {
        result.s1[1] = rt3 * 5.0 / 3.0 * sf.rby() * sf.rbz(); // d/d(ray)
        result.s1[4] = rt3 * (5.0 * sf.rbz() * sf.ray() + sf.cyy()) / 3.0; // d/d(rby)
        result.s1[5] = rt3 * (5.0 * sf.ray() * sf.rby() + sf.cyy()) / 3.0; // d/d(rbz)
        result.s1[10] = rt3 / 3.0 * sf.rbz();                 // d/d(cyy)
        result.s1[11] = rt3 / 3.0 * sf.rby();                 // d/d(cyz)
        if (level >= 2) {
            result.s2[11] = rt3 * 5.0 / 3.0 * sf.rbz(); // d²/d(ray)d(rby)
            result.s2[16] = rt3 * 5.0 / 3.0 * sf.rby(); // d²/d(ray)d(rbz)
            result.s2[19] = rt3 * 5.0 / 3.0 * sf.ray(); // d²/d(rby)d(rbz)
        }
    }
}

/**
 * Dipole-y × Quadrupole-22c kernel
 * Orient case 47: Q11s × Q22c
 * Formula: S0 = rt3*(5*rbx²*ray - 5*rby²*ray + 2*rbx*cyx - 2*rby*cyy)/6
 */
void dipole_y_quadrupole_22c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rbx() * sf.rbx() * sf.ray() - 5.0 * sf.rby() * sf.rby() * sf.ray() + 2.0 * sf.rbx() * sf.cyx() - 2.0 * sf.rby() * sf.cyy()) / 6.0;

    if (level >= 1) {
        result.s1[1] = rt3 * (5.0 * sf.rbx() * sf.rbx() - 5.0 * sf.rby() * sf.rby()) / 6.0; // d/d(ray)
        result.s1[3] = rt3 * (10.0 * sf.rbx() * sf.ray()) / 6.0;           // d/d(rbx)
        result.s1[4] = rt3 * (-10.0 * sf.ray() * sf.rby() - 2.0 * sf.cyy()) / 6.0; // d/d(rby)
        result.s1[9] = rt3 * 2.0 / 6.0 * sf.rbx();                      // d/d(cyx)
        result.s1[10] = rt3 * (-2.0) / 6.0 * sf.rby();                  // d/d(cyy)
        if (level >= 2) {
            result.s2[7] = rt3 * 5.0 / 3.0 * sf.rbx();     // d²/d(ray)d(rbx)
            result.s2[9] = rt3 * 5.0 / 3.0 * sf.ray();     // d²/d(rbx)²
            result.s2[11] = rt3 * (-5.0) / 3.0 * sf.rby(); // d²/d(ray)d(rby)
            result.s2[14] = rt3 * (-5.0) / 3.0 * sf.ray(); // d²/d(rby)²
        }
    }
}

/**
 * Dipole-y × Quadrupole-22s kernel
 * Orient case 48: Q11s × Q22s
 * Formula: S0 = rt3*(5*rbx*rby*ray + rbx*cyy + rby*cyx)/3
 */
void dipole_y_quadrupole_22s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rbx() * sf.rby() * sf.ray() + sf.rbx() * sf.cyy() + sf.rby() * sf.cyx()) / 3.0;

    if (level >= 1) {
        result.s1[1] = rt3 * 5.0 / 3.0 * sf.rbx() * sf.rby(); // d/d(ray)
        result.s1[3] = rt3 * (5.0 * sf.ray() * sf.rby() + sf.cyy()) / 3.0; // d/d(rbx)
        result.s1[4] = rt3 * (5.0 * sf.rbx() * sf.ray()) / 3.0; // d/d(rby)
        result.s1[9] = rt3 / 3.0 * sf.rby();                    // d/d(cyx)
        result.s1[10] = rt3 / 3.0 * sf.rbx();                   // d/d(cyy)
        if (level >= 2) {
            result.s2[7] = rt3 * 5.0 / 3.0 * sf.rby();  // d²/d(ray)d(rbx)
            result.s2[11] = rt3 * 5.0 / 3.0 * sf.rbx(); // d²/d(ray)d(rby)
            result.s2[13] = rt3 * 5.0 / 3.0 * sf.ray(); // d²/d(rbx)d(rby)
        }
    }
}

// ============================================================================
// QUADRUPOLE-DIPOLE KERNELS (Orient cases 49-63)
// Quadrupole @ A (uses rax, ray, raz), Dipole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Quadrupole-20 × Dipole-z kernel
 * Orient case 49: Q20 × Q10
 * Formula: S0 = 5/2*raz²*rbz - rbz/2 + raz*czz
 */
void quadrupole_20_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = 2.5 * sf.raz() * sf.raz() * sf.rbz() - 0.5 * sf.rbz() + sf.raz() * sf.czz();

    if (level >= 1) {
        result.s1[2] = 5.0 * sf.raz() * sf.rbz();       // d/d(raz)
        result.s1[5] = 2.5 * sf.raz() * sf.raz() - 0.5; // d/d(rbz)
        result.s1[14] = sf.raz();                       // d/d(czz)
        if (level >= 2) {
            result.s2[8] = 5.0 * sf.rbz();  // d²/d(raz)²
            result.s2[17] = 5.0 * sf.raz(); // d²/d(raz)d(rbz)
        }
    }
}

/**
 * Quadrupole-20 × Dipole-x kernel
 * Orient case 50: Q20 × Q11c
 * Formula: S0 = 5/2*raz²*rbx - rbx/2 + raz*czx
 */
void quadrupole_20_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = 2.5 * sf.raz() * sf.raz() * sf.rbx() - 0.5 * sf.rbx() + sf.raz() * sf.czx();

    if (level >= 1) {
        result.s1[2] = 5.0 * sf.raz() * sf.rbx();       // d/d(raz)
        result.s1[3] = 2.5 * sf.raz() * sf.raz() - 0.5; // d/d(rbx)
        result.s1[12] = sf.raz();                        // d/d(czx)
        if (level >= 2) {
            result.s2[8] = 5.0 * sf.rbx(); // d²/d(raz)²
            result.s2[8] = 5.0 * sf.raz(); // d²/d(raz)d(rbx)
        }
    }
}

/**
 * Quadrupole-20 × Dipole-y kernel
 * Orient case 51: Q20 × Q11s
 * Formula: S0 = 5/2*raz²*rby - rby/2 + raz*czy
 */
void quadrupole_20_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    result.s0 = 2.5 * sf.raz() * sf.raz() * sf.rby() - 0.5 * sf.rby() + sf.raz() * sf.czy();

    if (level >= 1) {
        result.s1[2] = 5.0 * sf.raz() * sf.rby();       // d/d(raz)
        result.s1[4] = 2.5 * sf.raz() * sf.raz() - 0.5; // d/d(rby)
        result.s1[13] = sf.raz();                       // d/d(czy)
        if (level >= 2) {
            result.s2[8] = 5.0 * sf.rby();  // d²/d(raz)²
            result.s2[12] = 5.0 * sf.raz(); // d²/d(raz)d(rby)
        }
    }
}

/**
 * Quadrupole-21c × Dipole-z kernel
 * Orient case 52: Q21c × Q10
 * Formula: S0 = rt3*(5*rax*raz*rbz + rax*czz + raz*cxz)/3
 */
void quadrupole_21c_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rax() * sf.raz() * sf.rbz() + sf.rax() * sf.czz() + sf.raz() * sf.cxz()) / 3.0;

    if (level >= 1) {
        result.s1[0] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rbz(); // d/d(rax)
        result.s1[2] = rt3 * 5.0 / 3.0 * sf.rbz() * sf.rax(); // d/d(raz)
        result.s1[5] = rt3 * 5.0 / 3.0 * sf.rax() * sf.raz(); // d/d(rbz)
        result.s1[8] = rt3 / 3.0 * sf.raz();                 // d/d(cxz)
        result.s1[14] = rt3 / 3.0 * sf.rax();                 // d/d(czz)
        if (level >= 2) {
            result.s2[6] = rt3 * 5.0 / 3.0 * sf.rbz();  // d²/d(rax)d(raz)
            result.s2[15] = rt3 * 5.0 / 3.0 * sf.raz(); // d²/d(rax)d(rbz)
            result.s2[17] = rt3 * 5.0 / 3.0 * sf.rax(); // d²/d(raz)d(rbz)
        }
    }
}

/**
 * Quadrupole-21c × Dipole-x kernel
 * Orient case 53: Q21c × Q11c
 * Formula: S0 = rt3*(5*rax*raz*rbx + rax*czx + raz*cxx)/3
 */
void quadrupole_21c_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rax() * sf.raz() * sf.rbx() + sf.rax() * sf.czx() + sf.raz() * sf.cxx()) / 3.0;

    if (level >= 1) {
        result.s1[0] = rt3 * (5.0 * sf.raz() * sf.rbx() + sf.czx()) / 3.0; // d/d(rax)
        result.s1[2] = rt3 * (5.0 * sf.rax() * sf.rbx() + sf.cxx()) / 3.0; // d/d(raz)
        result.s1[3] = rt3 * 5.0 / 3.0 * sf.rax() * sf.raz(); // d/d(rbx)
        result.s1[6] = rt3 / 3.0 * sf.raz();                  // d/d(cxx)
        result.s1[12] = rt3 / 3.0 * sf.rax();                  // d/d(czx)
        if (level >= 2) {
            result.s2[6] = rt3 * 5.0 / 3.0 * sf.rbx(); // d²/d(rax)d(raz)
            result.s2[6] = rt3 * 5.0 / 3.0 * sf.raz(); // d²/d(rax)d(rbx)
            result.s2[8] = rt3 * 5.0 / 3.0 * sf.rax(); // d²/d(raz)d(rbx)
        }
    }
}

/**
 * Quadrupole-21c × Dipole-y kernel
 * Orient case 54: Q21c × Q11s
 * Formula: S0 = rt3*(5*rax*raz*rby + rax*czy + raz*cxy)/3
 */
void quadrupole_21c_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rax() * sf.raz() * sf.rby() + sf.rax() * sf.czy() + sf.raz() * sf.cxy()) / 3.0;

    if (level >= 1) {
        result.s1[0] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rby(); // d/d(rax)
        result.s1[2] = rt3 * 5.0 / 3.0 * sf.rax() * sf.rby(); // d/d(raz)
        result.s1[4] = rt3 * 5.0 / 3.0 * sf.rax() * sf.raz(); // d/d(rby)
        result.s1[7] = rt3 / 3.0 * sf.raz();                  // d/d(cxy)
        result.s1[13] = rt3 / 3.0 * sf.rax();                 // d/d(czy)
        if (level >= 2) {
            result.s2[6] = rt3 * 5.0 / 3.0 * sf.rby();  // d²/d(rax)d(raz)
            result.s2[10] = rt3 * 5.0 / 3.0 * sf.raz(); // d²/d(rax)d(rby)
            result.s2[12] = rt3 * 5.0 / 3.0 * sf.rax(); // d²/d(raz)d(rby)
        }
    }
}

/**
 * Quadrupole-21s × Dipole-z kernel
 * Orient case 55: Q21s × Q10
 * Formula: S0 = rt3*(5*ray*raz*rbz + ray*czz + raz*cyz)/3
 */
void quadrupole_21s_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.ray() * sf.raz() * sf.rbz() + sf.ray() * sf.czz() + sf.raz() * sf.cyz()) / 3.0;

    if (level >= 1) {
        result.s1[1] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rbz(); // d/d(ray)
        result.s1[2] = rt3 * 5.0 / 3.0 * sf.rbz() * sf.ray(); // d/d(raz)
        result.s1[5] = rt3 * 5.0 / 3.0 * sf.ray() * sf.raz(); // d/d(rbz)
        result.s1[11] = rt3 / 3.0 * sf.raz();                 // d/d(cyz)
        result.s1[14] = rt3 / 3.0 * sf.ray();                 // d/d(czz)
        if (level >= 2) {
            result.s2[7] = rt3 * 5.0 / 3.0 * sf.rbz();  // d²/d(ray)d(raz)
            result.s2[16] = rt3 * 5.0 / 3.0 * sf.raz(); // d²/d(ray)d(rbz)
            result.s2[17] = rt3 * 5.0 / 3.0 * sf.ray(); // d²/d(raz)d(rbz)
        }
    }
}

/**
 * Quadrupole-21s × Dipole-x kernel
 * Orient case 56: Q21s × Q11c
 * Formula: S0 = rt3*(5*ray*raz*rbx + ray*czx + raz*cyx)/3
 */
void quadrupole_21s_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.ray() * sf.raz() * sf.rbx() + sf.ray() * sf.czx() + sf.raz() * sf.cyx()) / 3.0;

    if (level >= 1) {
        result.s1[1] = rt3 * 5.0 / 3.0 * sf.raz() * sf.rbx(); // d/d(ray)
        result.s1[2] = rt3 * 5.0 / 3.0 * sf.rbx() * sf.ray(); // d/d(raz)
        result.s1[3] = rt3 * 5.0 / 3.0 * sf.ray() * sf.raz(); // d/d(rbx)
        result.s1[9] = rt3 / 3.0 * sf.raz();                  // d/d(cyx)
        result.s1[12] = rt3 / 3.0 * sf.ray();                  // d/d(czx)
        if (level >= 2) {
            result.s2[7] = rt3 * 5.0 / 3.0 * sf.rbx(); // d²/d(ray)d(raz)
            result.s2[7] = rt3 * 5.0 / 3.0 * sf.raz(); // d²/d(ray)d(rbx)
            result.s2[8] = rt3 * 5.0 / 3.0 * sf.ray(); // d²/d(raz)d(rbx)
        }
    }
}

/**
 * Quadrupole-21s × Dipole-y kernel
 * Orient case 57: Q21s × Q11s
 * Formula: S0 = rt3*(5*ray*raz*rby + ray*czy + raz*cyy)/3
 */
void quadrupole_21s_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.ray() * sf.raz() * sf.rby() + sf.ray() * sf.czy() + sf.raz() * sf.cyy()) / 3.0;

    if (level >= 1) {
        result.s1[1] = rt3 * (5.0 * sf.raz() * sf.rby() + sf.czy()) / 3.0; // d/d(ray)
        result.s1[2] = rt3 * (5.0 * sf.ray() * sf.rby() + sf.cyy()) / 3.0; // d/d(raz)
        result.s1[4] = rt3 * 5.0 / 3.0 * sf.ray() * sf.raz(); // d/d(rby)
        result.s1[10] = rt3 / 3.0 * sf.raz();                 // d/d(cyy)
        result.s1[11] = rt3 / 3.0 * sf.ray();                 // d/d(czy)
        if (level >= 2) {
            result.s2[7] = rt3 * 5.0 / 3.0 * sf.rby();  // d²/d(ray)d(raz)
            result.s2[11] = rt3 * 5.0 / 3.0 * sf.raz(); // d²/d(ray)d(rby)
            result.s2[12] = rt3 * 5.0 / 3.0 * sf.ray(); // d²/d(raz)d(rby)
        }
    }
}

/**
 * Quadrupole-22c × Dipole-z kernel
 * Orient case 58: Q22c × Q10
 * Formula: S0 = rt3*(5*rax²*rbz - 5*ray²*rbz + 2*rax*cxz - 2*ray*cyz)/6
 */
void quadrupole_22c_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rax() * sf.rax() * sf.rbz() - 5.0 * sf.ray() * sf.ray() * sf.rbz() + 2.0 * sf.rax() * sf.cxz() - 2.0 * sf.ray() * sf.cyz()) / 6.0;

    if (level >= 1) {
        result.s1[0] = rt3 * 10.0 / 6.0 * sf.rbz() * sf.rax();    // d/d(rax)
        result.s1[1] = rt3 * (-10.0) / 6.0 * sf.rbz() * sf.ray(); // d/d(ray)
        result.s1[5] = rt3 * (5.0 * sf.rax() * sf.rax() - 5.0 * sf.ray() * sf.ray()) / 6.0; // d/d(rbz)
        result.s1[8] = rt3 * 2.0 / 6.0 * sf.rax();               // d/d(cxz)
        result.s1[11] = rt3 * (-2.0) / 6.0 * sf.ray();            // d/d(cyz)
        if (level >= 2) {
            result.s2[0] = rt3 * 5.0 / 3.0 * sf.rbz();     // d²/d(rax)²
            result.s2[4] = rt3 * (-5.0) / 3.0 * sf.rbz();  // d²/d(ray)²
            result.s2[15] = rt3 * 5.0 / 3.0 * sf.rax();    // d²/d(rax)d(rbz)
            result.s2[16] = rt3 * (-5.0) / 3.0 * sf.ray(); // d²/d(ray)d(rbz)
        }
    }
}

/**
 * Quadrupole-22c × Dipole-x kernel
 * Orient case 59: Q22c × Q11c
 * Formula: S0 = rt3*(5*rax²*rbx - 5*ray²*rbx + 2*rax*cxx - 2*ray*cyx)/6
 */
void quadrupole_22c_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rax() * sf.rax() * sf.rbx() - 5.0 * sf.ray() * sf.ray() * sf.rbx() + 2.0 * sf.rax() * sf.cxx() - 2.0 * sf.ray() * sf.cyx()) / 6.0;

    if (level >= 1) {
        result.s1[0] = rt3 * (10.0 * sf.rax() * sf.rbx() + 2.0 * sf.cxx()) / 6.0; // d/d(rax)
        result.s1[1] = rt3 * (-10.0 * sf.rbx() * sf.ray()) / 6.0;      // d/d(ray)
        result.s1[3] = rt3 * (5.0 * sf.rax() * sf.rax() - 5.0 * sf.ray() * sf.ray()) / 6.0; // d/d(rbx)
        result.s1[6] = rt3 * 2.0 / 6.0 * sf.rax();                     // d/d(cxx)
        result.s1[9] = rt3 * (-2.0) / 6.0 * sf.ray();                  // d/d(cyx)
        if (level >= 2) {
            result.s2[0] = rt3 * 5.0 / 3.0 * sf.rbx();    // d²/d(rax)²
            result.s2[4] = rt3 * (-5.0) / 3.0 * sf.rbx(); // d²/d(ray)²
            result.s2[6] = rt3 * 5.0 / 3.0 * sf.rax();    // d²/d(rax)d(rbx)
            result.s2[7] = rt3 * (-5.0) / 3.0 * sf.ray(); // d²/d(ray)d(rbx)
        }
    }
}

/**
 * Quadrupole-22c × Dipole-y kernel
 * Orient case 60: Q22c × Q11s
 * Formula: S0 = rt3*(5*rax²*rby - 5*ray²*rby + 2*rax*cxy - 2*ray*cyy)/6
 */
void quadrupole_22c_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rax() * sf.rax() * sf.rby() - 5.0 * sf.ray() * sf.ray() * sf.rby() + 2.0 * sf.rax() * sf.cxy() - 2.0 * sf.ray() * sf.cyy()) / 6.0;

    if (level >= 1) {
        result.s1[0] = rt3 * 10.0 / 6.0 * sf.rby() * sf.rax(); // d/d(rax)
        result.s1[1] = rt3 * (-10.0 * sf.rby() * sf.ray() - 2.0 * sf.cyy()) / 6.0; // d/d(ray)
        result.s1[4] = rt3 * (5.0 * sf.rax() * sf.rax() - 5.0 * sf.ray() * sf.ray()) / 6.0; // d/d(rby)
        result.s1[7] = rt3 * 2.0 / 6.0 * sf.rax();                      // d/d(cxy)
        result.s1[10] = rt3 * (-2.0) / 6.0 * sf.ray();                  // d/d(cyy)
        if (level >= 2) {
            result.s2[0] = rt3 * 5.0 / 3.0 * sf.rby();     // d²/d(rax)²
            result.s2[4] = rt3 * (-5.0) / 3.0 * sf.rby();  // d²/d(ray)²
            result.s2[10] = rt3 * 5.0 / 3.0 * sf.rax();    // d²/d(rax)d(rby)
            result.s2[11] = rt3 * (-5.0) / 3.0 * sf.ray(); // d²/d(ray)d(rby)
        }
    }
}

/**
 * Quadrupole-22s × Dipole-z kernel
 * Orient case 61: Q22s × Q10
 * Formula: S0 = rt3*(5*rax*ray*rbz + rax*cyz + ray*cxz)/3
 */
void quadrupole_22s_dipole_z(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rax() * sf.ray() * sf.rbz() + sf.rax() * sf.cyz() + sf.ray() * sf.cxz()) / 3.0;

    if (level >= 1) {
        result.s1[0] = rt3 * 5.0 / 3.0 * sf.ray() * sf.rbz(); // d/d(rax)
        result.s1[1] = rt3 * 5.0 / 3.0 * sf.rax() * sf.rbz(); // d/d(ray)
        result.s1[5] = rt3 * 5.0 / 3.0 * sf.rax() * sf.ray(); // d/d(rbz)
        result.s1[8] = rt3 / 3.0 * sf.ray();                 // d/d(cxz)
        result.s1[11] = rt3 / 3.0 * sf.rax();                 // d/d(cyz)
        if (level >= 2) {
            result.s2[3] = rt3 * 5.0 / 3.0 * sf.rbz();  // d²/d(rax)d(ray)
            result.s2[15] = rt3 * 5.0 / 3.0 * sf.ray(); // d²/d(rax)d(rbz)
            result.s2[16] = rt3 * 5.0 / 3.0 * sf.rax(); // d²/d(ray)d(rbz)
        }
    }
}

/**
 * Quadrupole-22s × Dipole-x kernel
 * Orient case 62: Q22s × Q11c
 * Formula: S0 = rt3*(5*rax*ray*rbx + rax*cyx + ray*cxx)/3
 */
void quadrupole_22s_dipole_x(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rax() * sf.ray() * sf.rbx() + sf.rax() * sf.cyx() + sf.ray() * sf.cxx()) / 3.0;

    if (level >= 1) {
        result.s1[0] = rt3 * (5.0 * sf.ray() * sf.rbx() + sf.cyx()) / 3.0; // d/d(rax)
        result.s1[1] = rt3 * (5.0 * sf.rax() * sf.rbx() + sf.cxx()) / 3.0; // d/d(ray)
        result.s1[3] = rt3 * 5.0 / 3.0 * sf.rax() * sf.ray(); // d/d(rbx)
        result.s1[6] = rt3 / 3.0 * sf.ray();                  // d/d(cxx)
        result.s1[9] = rt3 / 3.0 * sf.rax();                  // d/d(cyx)
        if (level >= 2) {
            result.s2[3] = rt3 * 5.0 / 3.0 * sf.rbx(); // d²/d(rax)d(ray)
            result.s2[6] = rt3 * 5.0 / 3.0 * sf.ray(); // d²/d(rax)d(rbx)
            result.s2[7] = rt3 * 5.0 / 3.0 * sf.rax(); // d²/d(ray)d(rbx)
        }
    }
}

/**
 * Quadrupole-22s × Dipole-y kernel
 * Orient case 63: Q22s × Q11s
 * Formula: S0 = rt3*(5*rax*ray*rby + rax*cyy + ray*cxy)/3
 */
void quadrupole_22s_dipole_y(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {

    constexpr double rt3 = 1.7320508075688772935;
    result.s0 = rt3 * (5.0 * sf.rax() * sf.ray() * sf.rby() + sf.rax() * sf.cyy() + sf.ray() * sf.cxy()) / 3.0;

    if (level >= 1) {
        result.s1[0] = rt3 * (5.0 * sf.ray() * sf.rby() + sf.cyy()) / 3.0; // d/d(rax)
        result.s1[1] = rt3 * (5.0 * sf.rax() * sf.rby() + sf.cxy()) / 3.0; // d/d(ray)
        result.s1[4] = rt3 * 5.0 / 3.0 * sf.rax() * sf.ray();   // d/d(rby)
        result.s1[7] = rt3 / 3.0 * sf.ray();                    // d/d(cxy)
        result.s1[10] = rt3 / 3.0 * sf.rax();                   // d/d(cyy)
        if (level >= 2) {
            result.s2[3] = rt3 * 5.0 / 3.0 * sf.rby();  // d²/d(rax)d(ray)
            result.s2[10] = rt3 * 5.0 / 3.0 * sf.ray(); // d²/d(rax)d(rby)
            result.s2[11] = rt3 * 5.0 / 3.0 * sf.rax(); // d²/d(ray)d(rby)
        }
    }
}

// ============================================================================
// CHARGE-OCTOPOLE KERNELS
// Charge @ A, Octopole @ B (uses rbx, rby, rbz)
// ============================================================================

/**
 * Charge × Octopole-30 kernel
 * Formula: S0 = 5z³/2 - 3z/2
 */
void charge_octopole_30(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    result.s0 = 2.5 * sf.rbz() * sf.rbz() * sf.rbz() - 1.5 * sf.rbz();
    if (level >= 1) {
        result.s1[5] = 7.5 * sf.rbz() * sf.rbz() - 1.5; // d/d(rbz)
        if (level >= 2) {
            result.s2[20] = 15.0 * sf.rbz(); // d²/d(rbz)², normalized
        }
    }
}

/**
 * Charge × Octopole-31c kernel
 * Formula: S0 = √6(5xz²-x)/4
 */
void charge_octopole_31c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    constexpr double rt6 = 2.4494897427831780981;
    result.s0 = rt6 * (5.0 * sf.rbx() * sf.rbz() * sf.rbz() - sf.rbx()) / 4.0;
    if (level >= 1) {
        result.s1[3] = rt6 * (5.0 * sf.rbz() * sf.rbz() - 1.0) / 4.0; // d/d(rbx)
        result.s1[5] = 2.5 * rt6 * sf.rbx() * sf.rbz();               // d/d(rbz)
        if (level >= 2) {
            result.s2[18] = 2.5 * rt6 * sf.rbz(); // d²/d(rbx)d(rbz), normalized
            result.s2[20] = 2.5 * rt6 * sf.rbx(); // d²/d(rbz)², normalized
        }
    }
}

/**
 * Charge × Octopole-31s kernel
 * Formula: S0 = √6(5yz²-y)/4
 */
void charge_octopole_31s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    constexpr double rt6 = 2.4494897427831780981;
    result.s0 = rt6 * (5.0 * sf.rby() * sf.rbz() * sf.rbz() - sf.rby()) / 4.0;
    if (level >= 1) {
        result.s1[4] = rt6 * (5.0 * sf.rbz() * sf.rbz() - 1.0) / 4.0; // d/d(rby)
        result.s1[5] = 2.5 * rt6 * sf.rby() * sf.rbz();               // d/d(rbz)
        if (level >= 2) {
            result.s2[19] = 2.5 * rt6 * sf.rbz(); // d²/d(rby)d(rbz), normalized
            result.s2[20] = 2.5 * rt6 * sf.rby(); // d²/d(rbz)², normalized
        }
    }
}

/**
 * Charge × Octopole-32c kernel
 * Formula: S0 = √15(x²z-y²z)/2
 */
void charge_octopole_32c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    constexpr double rt15 = 3.872983346207416885;
    result.s0 = rt15 * (sf.rbx() * sf.rbx() * sf.rbz() - sf.rby() * sf.rby() * sf.rbz()) / 2.0;
    if (level >= 1) {
        result.s1[3] = rt15 * sf.rbx() * sf.rbz();  // d/d(rbx), normalized
        result.s1[4] = -rt15 * sf.rby() * sf.rbz(); // d/d(rby), normalized
        result.s1[5] = rt15 * (sf.rbx() * sf.rbx() - sf.rby() * sf.rby()) /
                       2.0; // d/d(rbz), normalized
        if (level >= 2) {
            result.s2[9] = rt15 * sf.rbz();   // d²/d(rbx)², normalized
            result.s2[14] = -rt15 * sf.rbz(); // d²/d(rby)², normalized
            result.s2[18] = rt15 * sf.rbx();  // d²/d(rbx)d(rbz), normalized
            result.s2[19] = -rt15 * sf.rby(); // d²/d(rby)d(rbz), normalized
        }
    }
}

/**
 * Charge × Octopole-32s kernel
 * Formula: S0 = √15xyz
 */
void charge_octopole_32s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    constexpr double rt15 = 3.872983346207416885;
    result.s0 = rt15 * sf.rbx() * sf.rby() * sf.rbz();
    if (level >= 1) {
        result.s1[3] = rt15 * sf.rby() * sf.rbz(); // d/d(rbx), normalized
        result.s1[4] = rt15 * sf.rbx() * sf.rbz(); // d/d(rby), normalized
        result.s1[5] = rt15 * sf.rbx() * sf.rby(); // d/d(rbz), normalized
        if (level >= 2) {
            result.s2[13] = rt15 * sf.rbz(); // d²/d(rbx)d(rby), normalized
            result.s2[18] = rt15 * sf.rby(); // d²/d(rbx)d(rbz), normalized
            result.s2[19] = rt15 * sf.rbx(); // d²/d(rby)d(rbz), normalized
        }
    }
}

/**
 * Charge × Octopole-33c kernel
 * Formula: S0 = √10(x³-3xy²)/4
 */
void charge_octopole_33c(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    constexpr double rt10 = 3.1622776601683793320;
    result.s0 =
        rt10 * (sf.rbx() * sf.rbx() * sf.rbx() - 3.0 * sf.rbx() * sf.rby() * sf.rby()) / 4.0;
    if (level >= 1) {
        result.s1[3] = rt10 * (3.0 * sf.rbx() * sf.rbx() - 3.0 * sf.rby() * sf.rby()) /
                       4.0;                               // d/d(rbx), normalized
        result.s1[4] = -1.5 * rt10 * sf.rbx() * sf.rby(); // d/d(rby), normalized
        if (level >= 2) {
            result.s2[9] = 1.5 * rt10 * sf.rbx();   // d²/d(rbx)², normalized
            result.s2[13] = -1.5 * rt10 * sf.rby(); // d²/d(rbx)d(rby), normalized
            result.s2[14] = -1.5 * rt10 * sf.rbx(); // d²/d(rby)², normalized
        }
    }
}

/**
 * Charge × Octopole-33s kernel
 * Formula: S0 = √10(3x²y-y³)/4
 */
void charge_octopole_33s(const SFunctions& sf, int level, SFunctions::SFunctionResult& result) {
    constexpr double rt10 = 3.1622776601683793320;
    result.s0 =
        rt10 * (3.0 * sf.rbx() * sf.rbx() * sf.rby() - sf.rby() * sf.rby() * sf.rby()) / 4.0;
    if (level >= 1) {
        result.s1[3] = 1.5 * rt10 * sf.rbx() * sf.rby(); // d/d(rbx), normalized
        result.s1[4] = rt10 * (3.0 * sf.rbx() * sf.rbx() - 3.0 * sf.rby() * sf.rby()) /
                       4.0; // d/d(rby), normalized
        if (level >= 2) {
            result.s2[9] = 1.5 * rt10 * sf.rby();   // d²/d(rbx)², normalized
            result.s2[13] = 1.5 * rt10 * sf.rbx();  // d²/d(rbx)d(rby), normalized
            result.s2[14] = -1.5 * rt10 * sf.rby(); // d²/d(rby)², normalized
        }
    }
}

} // namespace occ::mults::kernels
