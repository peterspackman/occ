/**
 * @file rotation_validation_test.cpp
 * @brief Comprehensive tests for multipole rotation and body-to-lab frame transformations
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/rotation.h>
#include <occ/mults/esp.h>
#include <occ/mults/cartesian_multipole.h>
#include <occ/mults/cartesian_rotation.h>
#include <fmt/core.h>

using namespace occ;
using namespace occ::mults;
using namespace occ::dma;
using Approx = Catch::Approx;

TEST_CASE("Body-to-lab rotation: Simple dipole examples", "[mults][rotation][validation]") {

    SECTION("Dipole along z rotated 90 deg around y becomes dipole along x") {
        Mult dipole_body(1);
        dipole_body.Q00() = 0.0;
        dipole_body.Q10() = 1.0;  // z component
        dipole_body.Q11c() = 0.0; // x component
        dipole_body.Q11s() = 0.0; // y component

        // 90 degree rotation around y-axis (beta = 90 deg in ZYZ convention)
        double beta = M_PI / 2.0;
        Mat3 R = rotation_utils::euler_to_rotation(0.0, beta, 0.0);

        Mult dipole_lab = rotated_multipole(dipole_body, R);

        // After rotation: z -> x
        REQUIRE(dipole_lab.Q00() == Approx(0.0));
        REQUIRE(dipole_lab.Q10() == Approx(0.0).margin(1e-12));  // z component now zero
        REQUIRE(dipole_lab.Q11c() == Approx(1.0));  // x component now 1
        REQUIRE(dipole_lab.Q11s() == Approx(0.0).margin(1e-12));  // y component still zero
    }

    SECTION("Dipole along x rotated 90 deg around z becomes dipole along y") {
        Mult dipole_body(1);
        dipole_body.Q00() = 0.0;
        dipole_body.Q10() = 0.0;  // z component
        dipole_body.Q11c() = 1.0; // x component
        dipole_body.Q11s() = 0.0; // y component

        // 90 degree rotation around z-axis (alpha = 90 deg)
        double alpha = M_PI / 2.0;
        Mat3 R = rotation_utils::euler_to_rotation(alpha, 0.0, 0.0);

        Mult dipole_lab = rotated_multipole(dipole_body, R);

        // After rotation: x -> y
        REQUIRE(dipole_lab.Q00() == Approx(0.0));
        REQUIRE(dipole_lab.Q10() == Approx(0.0).margin(1e-12));  // z component still zero
        REQUIRE(dipole_lab.Q11c() == Approx(0.0).margin(1e-12));  // x component now zero
        REQUIRE(dipole_lab.Q11s() == Approx(1.0));  // y component now 1
    }
}

TEST_CASE("Body-to-lab rotation: Interaction energy invariance", "[mults][rotation][validation]") {

    SECTION("Same physical configuration gives same energy regardless of frame") {
        // Configuration: Two dipoles along x-axis, separated along y-axis
        // Setup 1: Specify directly in lab frame
        Mult d1_lab(1);
        d1_lab.Q00() = 0.0;
        d1_lab.Q10() = 0.0;
        d1_lab.Q11c() = 1.0;  // x-direction
        d1_lab.Q11s() = 0.0;

        Mult d2_lab(1);
        d2_lab.Q00() = 0.0;
        d2_lab.Q10() = 0.0;
        d2_lab.Q11c() = 1.0;  // x-direction
        d2_lab.Q11s() = 0.0;

        Vec3 pos1(0, 0, 0);
        Vec3 pos2(0, 5, 0);  // 5 bohr along y

        MultipoleESP esp(2);
        double energy_lab = esp.compute_interaction_energy(d1_lab, pos1, d2_lab, pos2);

        // Setup 2: Specify in body frame (z-axis) and rotate to lab frame
        Mult d1_body(1);
        d1_body.Q00() = 0.0;
        d1_body.Q10() = 1.0;  // z-direction in body frame
        d1_body.Q11c() = 0.0;
        d1_body.Q11s() = 0.0;

        Mult d2_body(1);
        d2_body.Q00() = 0.0;
        d2_body.Q10() = 1.0;  // z-direction in body frame
        d2_body.Q11c() = 0.0;
        d2_body.Q11s() = 0.0;

        // Rotate both by 90 deg around y to point along x
        double beta = M_PI / 2.0;
        Mat3 R = rotation_utils::euler_to_rotation(0.0, beta, 0.0);

        Mult d1_rotated = rotated_multipole(d1_body, R);
        Mult d2_rotated = rotated_multipole(d2_body, R);

        double energy_rotated = esp.compute_interaction_energy(d1_rotated, pos1, d2_rotated, pos2);

        // Should get the same energy
        REQUIRE(energy_rotated == Approx(energy_lab).epsilon(1e-10));

        // Expected value: E = mu1·mu2/r³ = 1.0 / 5³ = 0.008 au
        REQUIRE(energy_lab == Approx(0.008).epsilon(1e-10));
    }
}

TEST_CASE("Body-to-lab rotation: Complex water multipoles", "[mults][rotation][validation]") {

    SECTION("Water oxygen multipoles maintain magnitude under rotation") {
        // Water oxygen multipoles (from orient_water_esp.txt)
        Mult water_body(2);
        water_body.Q00() = -0.330960;
        water_body.Q10() = 0.0;
        water_body.Q11c() = -0.297907;
        water_body.Q11s() = 0.0;
        water_body.Q20() = 0.117935;
        water_body.Q21c() = 0.0;
        water_body.Q21s() = 0.0;
        water_body.Q22c() = 0.673922;
        water_body.Q22s() = 0.0;

        // Arbitrary rotation
        Mat3 R = rotation_utils::euler_to_rotation(M_PI/6, M_PI/4, M_PI/3);

        Mult water_lab = rotated_multipole(water_body, R);

        // Monopole should be unchanged
        REQUIRE(water_lab.Q00() == Approx(water_body.Q00()));

        // Total dipole magnitude should be conserved
        double dipole_mag_body = std::sqrt(
            water_body.Q10() * water_body.Q10() +
            water_body.Q11c() * water_body.Q11c() +
            water_body.Q11s() * water_body.Q11s()
        );

        double dipole_mag_lab = std::sqrt(
            water_lab.Q10() * water_lab.Q10() +
            water_lab.Q11c() * water_lab.Q11c() +
            water_lab.Q11s() * water_lab.Q11s()
        );

        REQUIRE(dipole_mag_lab == Approx(dipole_mag_body).epsilon(1e-10));
    }
}

TEST_CASE("Body-to-lab rotation: Rotation composition", "[mults][rotation][validation]") {

    SECTION("Sequential rotations match composed rotation") {
        Mult dipole(1);
        dipole.Q00() = 0.0;
        dipole.Q10() = 1.0;
        dipole.Q11c() = 0.5;
        dipole.Q11s() = 0.3;

        // Two rotations
        Mat3 R1 = rotation_utils::euler_to_rotation(M_PI/7, M_PI/5, M_PI/9);
        Mat3 R2 = rotation_utils::euler_to_rotation(M_PI/11, M_PI/13, M_PI/17);

        // Apply sequentially
        Mult step1 = rotated_multipole(dipole, R1);
        Mult step2 = rotated_multipole(step1, R2);

        // Apply composed
        Mat3 R_composed = R2 * R1;
        Mult composed = rotated_multipole(dipole, R_composed);

        // Should match
        REQUIRE(step2.Q00() == Approx(composed.Q00()).epsilon(1e-10));
        REQUIRE(step2.Q10() == Approx(composed.Q10()).epsilon(1e-10));
        REQUIRE(step2.Q11c() == Approx(composed.Q11c()).epsilon(1e-10));
        REQUIRE(step2.Q11s() == Approx(composed.Q11s()).epsilon(1e-10));
    }
}

// -------------------------------------------------------------------
// Cartesian rotation validation: Wigner D vs Cartesian rotation kernel
// -------------------------------------------------------------------

TEST_CASE("Cartesian rotation vs Wigner D: all ranks",
          "[mults][rotation][cartesian]") {

    // Test strategy:
    //   Path A: spherical → Wigner D rotate → spherical_to_cartesian
    //   Path B: spherical → spherical_to_cartesian → rotate_cartesian_multipole
    // Both should produce identical Cartesian multipoles.

    // Multiple rotation matrices to test
    struct RotCase {
        const char *name;
        double alpha, beta, gamma;
    };
    RotCase rotations[] = {
        {"90 deg about z",     M_PI / 2,  0,        0},
        {"90 deg about y",     0,         M_PI / 2, 0},
        {"45 deg all axes",    M_PI / 4,  M_PI / 4, M_PI / 4},
        {"arbitrary 1",        0.7,       0.4,      -0.3},
        {"arbitrary 2",        -0.5,      0.8,      1.2},
        {"near-gimbal",        1.0,       0.01,     2.0},
        {"large angles",       2.5,       1.3,      -1.8},
    };

    SECTION("Rank 1: dipole") {
        Mult body(4);
        body.Q10() = 0.234;
        body.Q11c() = -0.15;
        body.Q11s() = 0.42;

        for (auto &rc : rotations) {
            CAPTURE(rc.name);
            Mat3 R = rotation_utils::euler_to_rotation(rc.alpha, rc.beta, rc.gamma);

            // Path A: Wigner D → Cartesian
            Mult lab_sph = rotated_multipole(body, R);
            CartesianMultipole<4> cart_wigner;
            spherical_to_cartesian<4>(lab_sph, cart_wigner);

            // Path B: Cartesian → rotate
            CartesianMultipole<4> cart_body;
            spherical_to_cartesian<4>(body, cart_body);
            CartesianMultipole<4> cart_rotated;
            rotate_cartesian_multipole<4>(cart_body, R, cart_rotated);

            for (int i = 0; i < CartesianMultipole<4>::size; ++i) {
                REQUIRE(cart_rotated.data[i] ==
                        Approx(cart_wigner.data[i]).margin(1e-12));
            }
        }
    }

    SECTION("Rank 2: quadrupole") {
        Mult body(4);
        body.Q20() = -0.123;
        body.Q21c() = 0.05;
        body.Q21s() = -0.07;
        body.Q22c() = 0.08;
        body.Q22s() = 0.03;

        for (auto &rc : rotations) {
            CAPTURE(rc.name);
            Mat3 R = rotation_utils::euler_to_rotation(rc.alpha, rc.beta, rc.gamma);

            Mult lab_sph = rotated_multipole(body, R);
            CartesianMultipole<4> cart_wigner;
            spherical_to_cartesian<4>(lab_sph, cart_wigner);

            CartesianMultipole<4> cart_body;
            spherical_to_cartesian<4>(body, cart_body);
            CartesianMultipole<4> cart_rotated;
            rotate_cartesian_multipole<4>(cart_body, R, cart_rotated);

            for (int i = 0; i < CartesianMultipole<4>::size; ++i) {
                REQUIRE(cart_rotated.data[i] ==
                        Approx(cart_wigner.data[i]).margin(1e-12));
            }
        }
    }

    SECTION("Rank 3: octupole") {
        Mult body(4);
        body.Q30() = 0.05;
        body.Q31c() = -0.03;
        body.Q31s() = 0.02;
        body.Q32c() = 0.04;
        body.Q32s() = -0.01;
        body.Q33c() = 0.06;
        body.Q33s() = -0.02;

        for (auto &rc : rotations) {
            CAPTURE(rc.name);
            Mat3 R = rotation_utils::euler_to_rotation(rc.alpha, rc.beta, rc.gamma);

            Mult lab_sph = rotated_multipole(body, R);
            CartesianMultipole<4> cart_wigner;
            spherical_to_cartesian<4>(lab_sph, cart_wigner);

            CartesianMultipole<4> cart_body;
            spherical_to_cartesian<4>(body, cart_body);
            CartesianMultipole<4> cart_rotated;
            rotate_cartesian_multipole<4>(cart_body, R, cart_rotated);

            for (int i = 0; i < CartesianMultipole<4>::size; ++i) {
                REQUIRE(cart_rotated.data[i] ==
                        Approx(cart_wigner.data[i]).margin(1e-12));
            }
        }
    }

    SECTION("Rank 4: hexadecapole") {
        Mult body(4);
        body.Q40() = 0.01;
        body.Q41c() = -0.02;
        body.Q41s() = 0.015;
        body.Q42c() = 0.03;
        body.Q42s() = -0.005;
        body.Q43c() = 0.008;
        body.Q43s() = -0.012;
        body.Q44c() = 0.004;
        body.Q44s() = 0.006;

        for (auto &rc : rotations) {
            CAPTURE(rc.name);
            Mat3 R = rotation_utils::euler_to_rotation(rc.alpha, rc.beta, rc.gamma);

            Mult lab_sph = rotated_multipole(body, R);
            CartesianMultipole<4> cart_wigner;
            spherical_to_cartesian<4>(lab_sph, cart_wigner);

            CartesianMultipole<4> cart_body;
            spherical_to_cartesian<4>(body, cart_body);
            CartesianMultipole<4> cart_rotated;
            rotate_cartesian_multipole<4>(cart_body, R, cart_rotated);

            for (int i = 0; i < CartesianMultipole<4>::size; ++i) {
                REQUIRE(cart_rotated.data[i] ==
                        Approx(cart_wigner.data[i]).margin(1e-12));
            }
        }
    }

    SECTION("All ranks combined") {
        Mult body(4);
        body.Q00() = -0.669;
        body.Q10() = 0.234;
        body.Q11c() = -0.15;
        body.Q11s() = 0.42;
        body.Q20() = -0.123;
        body.Q21c() = 0.05;
        body.Q22c() = 0.08;
        body.Q30() = 0.05;
        body.Q31c() = -0.03;
        body.Q32c() = 0.04;
        body.Q33c() = 0.06;
        body.Q40() = 0.01;
        body.Q41c() = -0.02;
        body.Q42c() = 0.03;
        body.Q43c() = 0.008;
        body.Q44c() = 0.004;

        for (auto &rc : rotations) {
            CAPTURE(rc.name);
            Mat3 R = rotation_utils::euler_to_rotation(rc.alpha, rc.beta, rc.gamma);

            Mult lab_sph = rotated_multipole(body, R);
            CartesianMultipole<4> cart_wigner;
            spherical_to_cartesian<4>(lab_sph, cart_wigner);

            CartesianMultipole<4> cart_body;
            spherical_to_cartesian<4>(body, cart_body);
            CartesianMultipole<4> cart_rotated;
            rotate_cartesian_multipole<4>(cart_body, R, cart_rotated);

            for (int i = 0; i < CartesianMultipole<4>::size; ++i) {
                REQUIRE(cart_rotated.data[i] ==
                        Approx(cart_wigner.data[i]).margin(1e-12));
            }
        }
    }
}

TEST_CASE("Cartesian rotation roundtrip",
          "[mults][rotation][cartesian]") {

    // Rotate body → lab, then rotate lab → body (using R^T), should recover original
    Mult body(4);
    body.Q00() = -0.669;
    body.Q10() = 0.234;
    body.Q11c() = -0.15;
    body.Q20() = -0.123;
    body.Q22c() = 0.08;
    body.Q30() = 0.05;
    body.Q31c() = -0.03;
    body.Q33c() = 0.06;
    body.Q40() = 0.01;
    body.Q42c() = 0.03;
    body.Q44c() = 0.004;

    CartesianMultipole<4> cart_body;
    spherical_to_cartesian<4>(body, cart_body);

    Mat3 R = rotation_utils::euler_to_rotation(0.7, 0.4, -0.3);

    CartesianMultipole<4> cart_lab;
    rotate_cartesian_multipole<4>(cart_body, R, cart_lab);

    CartesianMultipole<4> cart_recovered;
    rotate_cartesian_multipole<4>(cart_lab, R.transpose(), cart_recovered);

    for (int i = 0; i < CartesianMultipole<4>::size; ++i) {
        REQUIRE(cart_recovered.data[i] ==
                Approx(cart_body.data[i]).margin(1e-12));
    }
}
