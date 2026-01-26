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

        MultipoleESP esp(1);
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
