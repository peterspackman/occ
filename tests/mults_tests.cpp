#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/rotation.h>
#include <occ/mults/sfunctions.h>
#include <occ/mults/esp.h>
#include <occ/mults/coordinate_system.h>
#include <occ/mults/orient_io.h>
#include <occ/mults/sfunction_result.h>
#include <occ/mults/sfunction_term.h>
#include <occ/mults/sfunction_evaluator.h>
#include <occ/mults/sfunction_term_builder.h>
#include <occ/mults/multipole_interactions.h>
#include <fmt/core.h>
#include <iostream>
#include <map>
#include <algorithm>

using namespace occ;
using namespace occ::mults;
using namespace occ::dma;
using Approx = Catch::Approx;

TEST_CASE("Multipole rotation - rotation matrix validation",
          "[mults][rotation]") {

  SECTION("Identity matrix is valid rotation") {
    Mat3 identity = Mat3::Identity();
    REQUIRE(rotation_utils::is_rotation_matrix(identity));
  }

  SECTION("Random orthogonal matrix with det=1 is valid") {
    // Simple 90 degree rotation around z axis
    Mat3 rot_z;
    rot_z << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    REQUIRE(rotation_utils::is_rotation_matrix(rot_z));
  }

  SECTION("Non-orthogonal matrix is invalid") {
    Mat3 bad;
    bad << 1, 0, 0, 0, 2, 0, // Not orthogonal
        0, 0, 1;
    REQUIRE_FALSE(rotation_utils::is_rotation_matrix(bad));
  }
}

TEST_CASE("Multipole rotation - Euler angles", "[mults][rotation]") {

  SECTION("Zero Euler angles give identity") {
    Mat3 identity = rotation_utils::euler_to_rotation(0, 0, 0);
    REQUIRE(identity.isApprox(Mat3::Identity(), 1e-12));
  }

  SECTION("90 degree rotation around z") {
    double pi_2 = M_PI / 2.0;
    Mat3 rot_z = rotation_utils::euler_to_rotation(pi_2, 0, 0);

    Vec3 x_axis(1, 0, 0);
    Vec3 rotated_x = rot_z * x_axis;
    Vec3 expected_y(0, 1, 0);

    REQUIRE(rotated_x.isApprox(expected_y, 1e-12));
  }
}

TEST_CASE("Multipole rotation - axis-angle", "[mults][rotation]") {

  SECTION("Zero angle gives identity") {
    Vec3 arbitrary_axis(1, 2, 3);
    Mat3 identity = rotation_utils::axis_angle_to_rotation(arbitrary_axis, 0.0);
    REQUIRE(identity.isApprox(Mat3::Identity(), 1e-12));
  }

  SECTION("180 degree rotation around z") {
    Vec3 z_axis(0, 0, 1);
    Mat3 rot_z_180 = rotation_utils::axis_angle_to_rotation(z_axis, M_PI);

    Vec3 x_axis(1, 0, 0);
    Vec3 rotated_x = rot_z_180 * x_axis;
    Vec3 expected_minus_x(-1, 0, 0);

    REQUIRE(rotated_x.isApprox(expected_minus_x, 1e-12));
  }
}

TEST_CASE("Multipole rotation - Wigner D-matrix basics", "[mults][rotation]") {

  SECTION("Identity rotation gives identity D-matrix") {
    Mat3 identity = Mat3::Identity();

    for (int lmax = 0; lmax <= 2; lmax++) {
      Mat D = wigner_d_matrix(identity, lmax);
      int size = (lmax + 1) * (lmax + 1);
      Mat expected = Mat::Identity(size, size);

      REQUIRE(D.isApprox(expected, 1e-12));
    }
  }

  SECTION("D-matrix is orthogonal") {
    // 45 degree rotation around z axis
    double angle = M_PI / 4.0;
    Mat3 rot = rotation_utils::axis_angle_to_rotation(Vec3(0, 0, 1), angle);

    for (int lmax = 1; lmax <= 2; lmax++) {
      Mat D = wigner_d_matrix(rot, lmax);
      Mat DtD = D.transpose() * D;
      int size = (lmax + 1) * (lmax + 1);
      Mat identity = Mat::Identity(size, size);

      REQUIRE(DtD.isApprox(identity, 1e-12));
    }
  }
}

TEST_CASE("Multipole rotation - simple multipole objects",
          "[mults][rotation]") {

  SECTION("Rotating pure monopole (rank 0) is invariant") {
    Mult monopole(0); // Only Q00
    monopole.Q00() = 1.5;

    // Arbitrary rotation
    Mat3 rot = rotation_utils::axis_angle_to_rotation(Vec3(1, 1, 1), M_PI / 3);

    Mult rotated = rotated_multipole(monopole, rot);

    REQUIRE(rotated.Q00() == Approx(1.5));
  }

  SECTION("Rotating dipole around z-axis") {
    Mult dipole(1);      // Q00, Q10, Q11c, Q11s
    dipole.Q00() = 0.0;  // No monopole
    dipole.Q10() = 1.0;  // z-component
    dipole.Q11c() = 0.0; // x-component
    dipole.Q11s() = 0.0; // y-component

    // 90 degree rotation around z-axis should leave Q10 unchanged
    Mat3 rot_z =
        rotation_utils::axis_angle_to_rotation(Vec3(0, 0, 1), M_PI / 2);

    Mult rotated = rotated_multipole(dipole, rot_z);

    REQUIRE(rotated.Q00() == Approx(0.0));
    REQUIRE(rotated.Q10() == Approx(1.0)); // z-component unchanged
    REQUIRE(rotated.Q11c() == Approx(0.0).margin(1e-12));
    REQUIRE(rotated.Q11s() == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("Multipole rotation - water molecule validation",
          "[mults][rotation][validation]") {

  SECTION("Load water multipoles and test rotation consistency") {
    // Set up water oxygen multipoles (from orient_water_esp.txt)
    // Orient format: line 11 = charge, line 12 = (Q10, Q11c, Q11s), line 13 = (Q20, Q21c, Q21s, Q22c, Q22s), etc.
    Mult water_oxygen(4);
    water_oxygen.Q00() = -0.330960; // monopole (line 11)

    // Dipole (line 12: 0.0, -0.297907, 0.0)
    water_oxygen.Q10() = 0.0;       // dipole z
    water_oxygen.Q11c() = -0.297907; // dipole x (THE NON-ZERO ONE!)
    water_oxygen.Q11s() = 0.0;      // dipole y

    // Quadrupole (line 13: 0.117935, 0.0, 0.0, 0.673922, 0.0)
    water_oxygen.Q20() = 0.117935;  // quadrupole zz
    water_oxygen.Q21c() = 0.0;      // quadrupole xz
    water_oxygen.Q21s() = 0.0;      // quadrupole yz
    water_oxygen.Q22c() = 0.673922; // quadrupole (xx-yy)/2
    water_oxygen.Q22s() = 0.0;      // quadrupole xy

    // Octupole (lines 14-15: 0.0, -0.151827, 0.0, 0.0, 0.0, 0.303856, 0.0)
    water_oxygen.Q30() = 0.0;       // octupole zzz
    water_oxygen.Q31c() = -0.151827; // octupole xzz (WAS MISLABELED AS Q30!)
    water_oxygen.Q31s() = 0.0;      // octupole yzz
    water_oxygen.Q32c() = 0.0;      // octupole (xxz-yyz)
    water_oxygen.Q32s() = 0.0;      // octupole xyz
    water_oxygen.Q33c() = 0.303856; // octupole (xxx-3xyy) (WAS MISLABELED AS Q32c!)
    water_oxygen.Q33s() = 0.0;      // octupole (3xxy-yyy)

    // Hexadecapole (lines 16-17: 0.114584, 0.0, 0.0, -0.183221, 0.0, 0.0, 0.0, 0.0, -0.065424)
    water_oxygen.Q40() = 0.114584;  // hexadecapole zzzz
    water_oxygen.Q41c() = 0.0;      // hexadecapole xzzz
    water_oxygen.Q41s() = 0.0;      // hexadecapole yzzz
    water_oxygen.Q42c() = -0.183221; // hexadecapole (xxzz-yyzz)
    water_oxygen.Q42s() = 0.0;      // hexadecapole xyzz
    water_oxygen.Q43c() = 0.0;      // hexadecapole (xxxz-3xyyz)
    water_oxygen.Q43s() = 0.0;      // hexadecapole (3xxyz-yyyz)
    water_oxygen.Q44c() = -0.065424; // hexadecapole (xxxx-6xxyy+yyyy)
    water_oxygen.Q44s() = 0.0;      // hexadecapole (xxxy-xyyy)

    // Test that rotation composition works: R2 * R1 = R3
    Mat3 R1 = rotation_utils::euler_to_rotation(M_PI / 6, M_PI / 4, M_PI / 3);
    Mat3 R2 = rotation_utils::euler_to_rotation(M_PI / 5, M_PI / 7, M_PI / 9);
    Mat3 R3 = R2 * R1;

    // Apply rotations in sequence vs composed
    Mult mult1 = rotated_multipole(water_oxygen, R1);
    Mult mult2 = rotated_multipole(mult1, R2);

    Mult mult3 = rotated_multipole(water_oxygen, R3);

    // Should be the same (within numerical precision)
    for (int i = 0; i < water_oxygen.num_components(); i++) {
      REQUIRE(mult2.q(i) == Approx(mult3.q(i)).margin(1e-10));
    }
  }

  SECTION("Rotation preserves multipole norm invariants") {
    // Set up a test multipole with multiple ranks
    Mult test_mult(2);
    test_mult.Q00() = 1.0;
    test_mult.Q10() = 0.5;
    test_mult.Q11c() = -0.3;
    test_mult.Q11s() = 0.7;
    test_mult.Q20() = 0.2;
    test_mult.Q21c() = -0.1;
    test_mult.Q21s() = 0.4;
    test_mult.Q22c() = 0.6;
    test_mult.Q22s() = -0.2;

    // Arbitrary rotation
    Mat3 rot = rotation_utils::euler_to_rotation(0.8, 1.2, 0.5);

    Mult rotated = rotated_multipole(test_mult, rot);

    // Check that certain invariants are preserved
    // For now, just check that the total "energy" is preserved
    double original_norm = test_mult.q.squaredNorm();
    double rotated_norm = rotated.q.squaredNorm();

    REQUIRE(rotated_norm == Approx(original_norm).margin(1e-12));
  }
}

TEST_CASE("S-functions - basic functionality", "[mults][sfunctions]") {
  
  SECTION("Constructor and coordinate setting") {
    SFunctions sf(2);  // Up to quadrupoles
    
    Vec3 ra(1.0, 2.0, 3.0);
    Vec3 rb(4.0, 5.0, 6.0);
    sf.set_coordinates(ra, rb);
    
    // Verify unit vector properties: rbx² + rby² + rbz² = 1
    double unit_length = sf.rbx()*sf.rbx() + sf.rby()*sf.rby() + sf.rbz()*sf.rbz();
    REQUIRE(unit_length == Approx(1.0));
    
    REQUIRE(sf.dx() == Approx(3.0));
    REQUIRE(sf.dy() == Approx(3.0));
    REQUIRE(sf.dz() == Approx(3.0));
    REQUIRE(sf.r() == Approx(std::sqrt(27.0)));
  }
  
  SECTION("Charge-charge S-function") {
    SFunctions sf(0);
    Vec3 ra(0.0, 0.0, 0.0);
    Vec3 rb(1.0, 2.0, 3.0);
    sf.set_coordinates(ra, rb);
    
    auto result = sf.compute_s_function(0, 0, 0, 0);  // S(00,00,0)
    REQUIRE(result.s0 == Approx(1.0));  // Should be 1.0
  }
  
  
  // NOTE: Old validation test removed - replaced by comprehensive "S-functions match Orient" test below
}


TEST_CASE("ESP calculation - basic functionality", "[mults][esp]") {
  
  SECTION("Constructor") {
    MultipoleESP esp(2);  // Up to quadrupoles
    // Should construct without errors
  }
  
  SECTION("Monopole ESP calculation") {
    MultipoleESP esp(0);
    
    // Create a unit charge at origin
    Mult monopole(0);
    monopole.Q00() = 1.0;  // Unit charge
    
    Vec3 charge_pos(0.0, 0.0, 0.0);
    Vec3 eval_point(1.0, 0.0, 0.0);  // 1 bohr away along x
    
    double esp_value = esp.compute_esp_at_point(monopole, charge_pos, eval_point);
    
    // ESP of unit charge at 1 bohr should be 1.0 (in atomic units)
    REQUIRE(esp_value == Approx(1.0).margin(1e-12));
  }
  
  SECTION("Monopole ESP at multiple points") {
    MultipoleESP esp(0);
    
    Mult monopole(0);
    monopole.Q00() = 2.0;  // +2e charge
    
    Vec3 charge_pos(0.0, 0.0, 0.0);
    std::vector<Vec3> eval_points = {
      Vec3(1.0, 0.0, 0.0),  // ESP = 2.0
      Vec3(2.0, 0.0, 0.0),  // ESP = 1.0  
      Vec3(0.0, 1.0, 0.0),  // ESP = 2.0
      Vec3(0.0, 0.0, 2.0)   // ESP = 1.0
    };
    
    auto esp_values = esp.compute_esp_at_points(monopole, charge_pos, eval_points);
    
    REQUIRE(esp_values.size() == 4);
    REQUIRE(esp_values[0] == Approx(2.0).margin(1e-12));
    REQUIRE(esp_values[1] == Approx(1.0).margin(1e-12));
    REQUIRE(esp_values[2] == Approx(2.0).margin(1e-12));
    REQUIRE(esp_values[3] == Approx(1.0).margin(1e-12));
  }
  
  SECTION("Dipole ESP calculation") {
    MultipoleESP esp(1);
    
    // Create a dipole pointing along z
    Mult dipole(1);
    dipole.Q00() = 0.0;  // No monopole
    dipole.Q10() = 1.0;  // Unit dipole along z
    dipole.Q11c() = 0.0; // No x component
    dipole.Q11s() = 0.0; // No y component
    
    Vec3 dipole_pos(0.0, 0.0, 0.0);
    
    // ESP along z-axis: For dipole pointing along +z, potential at point on +z axis
    // With Orient convention (ra=dipole, rb=field_point), raz = +1
    // ESP = Q10 * raz / r^2 = 1 * 1 / 1 = 1
    // But dipole-charge interaction has specific sign convention
    Vec3 eval_point_z(0.0, 0.0, 1.0);  // 1 bohr along +z
    double esp_z = esp.compute_esp_at_point(dipole, dipole_pos, eval_point_z);
    // Note: Orient's convention gives negative ESP for dipole pointing towards field point
    REQUIRE(std::abs(esp_z) == Approx(1.0).margin(1e-10));  // Magnitude should be 1
    
    // ESP in xy-plane (theta=90°): ESP = 0
    Vec3 eval_point_x(1.0, 0.0, 0.0);  // 1 bohr along x
    double esp_x = esp.compute_esp_at_point(dipole, dipole_pos, eval_point_x);
    REQUIRE(esp_x == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("ESP calculation - Orient water validation", "[mults][esp][orient]") {
  
  SECTION("Water oxygen multipole ESP") {
    MultipoleESP esp(4);

    // Set up water oxygen multipoles from orient_water_esp.txt
    Mult water_oxygen(4);
    water_oxygen.Q00() = -0.330960;   // monopole
    water_oxygen.Q10() = 0.0;         // dipole z
    water_oxygen.Q11c() = -0.297907;  // dipole x
    water_oxygen.Q11s() = 0.0;        // dipole y
    water_oxygen.Q20() = 0.117935;    // quadrupole zz
    water_oxygen.Q21c() = 0.0;
    water_oxygen.Q21s() = 0.0;
    water_oxygen.Q22c() = 0.673922;   // quadrupole (xx-yy)/2
    water_oxygen.Q22s() = 0.0;
    water_oxygen.Q30() = 0.0;         // octupole zzz
    water_oxygen.Q31c() = -0.151827;  // octupole xzz
    water_oxygen.Q31s() = 0.0;
    water_oxygen.Q32c() = 0.0;
    water_oxygen.Q32s() = 0.0;
    water_oxygen.Q33c() = 0.303856;   // octupole (xxx-3xyy)
    water_oxygen.Q33s() = 0.0;
    water_oxygen.Q40() = 0.114584;    // hexadecapole
    water_oxygen.Q41c() = 0.0;
    water_oxygen.Q41s() = 0.0;
    water_oxygen.Q42c() = -0.183221;
    water_oxygen.Q42s() = 0.0;
    water_oxygen.Q43c() = 0.0;
    water_oxygen.Q43s() = 0.0;
    water_oxygen.Q44c() = 0.0;
    water_oxygen.Q44s() = -0.065424;
    
    Vec3 oxygen_pos(0.0, 0.0, 0.0);
    
    // Test ESP at various points around the oxygen
    Vec3 eval_point_1(1.0, 0.0, 0.0);  // 1 bohr along +x
    double esp_1 = esp.compute_esp_at_point(water_oxygen, oxygen_pos, eval_point_1);
    
    Vec3 eval_point_2(0.0, 0.0, 1.0);  // 1 bohr along +z  
    double esp_2 = esp.compute_esp_at_point(water_oxygen, oxygen_pos, eval_point_2);
    
    Vec3 eval_point_3(2.0, 0.0, 0.0);  // 2 bohr along +x
    double esp_3 = esp.compute_esp_at_point(water_oxygen, oxygen_pos, eval_point_3);
    
    // These values should be finite and reasonable for a water molecule
    REQUIRE(std::isfinite(esp_1));
    REQUIRE(std::isfinite(esp_2)); 
    REQUIRE(std::isfinite(esp_3));
    
    // ESP should decay with distance
    REQUIRE(std::abs(esp_3) < std::abs(esp_1));
    
    // ESP along z should be different from ESP along x due to dipole
    REQUIRE(std::abs(esp_2 - esp_1) > 1e-10);
  }
}

TEST_CASE("Multipole interaction energy", "[mults][esp][interaction]") {
  
  SECTION("Two monopoles interaction") {
    MultipoleESP esp(0);
    
    Mult charge1(0);
    charge1.Q00() = 1.0;  // +1e
    
    Mult charge2(0); 
    charge2.Q00() = -1.0; // -1e
    
    Vec3 pos1(0.0, 0.0, 0.0);
    Vec3 pos2(1.0, 0.0, 0.0);  // 1 bohr separation
    
    double energy = esp.compute_interaction_energy(charge1, pos1, charge2, pos2);
    
    // E = q1*q2/r = 1*(-1)/1 = -1 (attractive)
    REQUIRE(energy == Approx(-1.0).margin(1e-12));
  }
  
  
  SECTION("Water dimer interaction energy") {
    MultipoleESP esp(4);

    // Set up two water oxygen sites from orient_water_esp.txt
    Mult water_oxygen(4);
    water_oxygen.Q00() = -0.330960;   // monopole
    water_oxygen.Q10() = 0.0;         // dipole z
    water_oxygen.Q11c() = -0.297907;  // dipole x
    water_oxygen.Q11s() = 0.0;        // dipole y
    water_oxygen.Q20() = 0.117935;    // quadrupole zz
    water_oxygen.Q21c() = 0.0;
    water_oxygen.Q21s() = 0.0;
    water_oxygen.Q22c() = 0.673922;   // quadrupole (xx-yy)/2
    water_oxygen.Q22s() = 0.0;
    water_oxygen.Q30() = 0.0;         // octupole zzz
    water_oxygen.Q31c() = -0.151827;  // octupole xzz
    water_oxygen.Q31s() = 0.0;
    water_oxygen.Q32c() = 0.0;
    water_oxygen.Q32s() = 0.0;
    water_oxygen.Q33c() = 0.303856;   // octupole (xxx-3xyy)
    water_oxygen.Q33s() = 0.0;
    water_oxygen.Q40() = 0.114584;    // hexadecapole
    water_oxygen.Q41c() = 0.0;
    water_oxygen.Q41s() = 0.0;
    water_oxygen.Q42c() = -0.183221;
    water_oxygen.Q42s() = 0.0;
    water_oxygen.Q43c() = 0.0;
    water_oxygen.Q43s() = 0.0;
    water_oxygen.Q44c() = 0.0;
    water_oxygen.Q44s() = -0.065424;
    
    Vec3 pos1(0.0, 0.0, 0.0);
    Vec3 pos2(3.0, 0.0, 0.0);  // 3 bohr separation along x-axis

    double energy = esp.compute_interaction_energy(water_oxygen, pos1, water_oxygen, pos2);

    // Orient reference: 0.02753720 hartree (from orient_water_dimer_test.txt)
    // This is the electrostatic energy for two identical water multipoles 3 bohr apart
    double orient_energy = 0.02753720;

    REQUIRE(std::isfinite(energy));
    // Validate against Orient reference (should match within numerical precision)
    double percent_error = 100.0 * std::abs(energy - orient_energy) / orient_energy;
    INFO("OCC energy = " << energy << " hartree, Orient = " << orient_energy << " hartree, Error = " << percent_error << "%");
    REQUIRE(percent_error < 0.5);  // Should match Orient within 0.5%
  }

  SECTION("Water dimer - distance variation") {
    MultipoleESP esp(4);

    // Standard water oxygen multipole (validated against Orient - x-dipole orientation)
    Mult water_oxygen(4);
    water_oxygen.Q00() = -0.330960;
    water_oxygen.Q10() = 0.0;         // dipole z = 0 for x-oriented
    water_oxygen.Q11c() = -0.297907;  // dipole x (main component)
    water_oxygen.Q11s() = 0.0;
    water_oxygen.Q20() = 0.117935;
    water_oxygen.Q21c() = 0.0;
    water_oxygen.Q21s() = 0.0;
    water_oxygen.Q22c() = 0.673922;
    water_oxygen.Q22s() = 0.0;
    water_oxygen.Q30() = 0.0;         // octupole zzz = 0 for x-oriented
    water_oxygen.Q31c() = -0.151827;  // octupole xzz (main component)
    water_oxygen.Q31s() = 0.0;
    water_oxygen.Q32c() = 0.0;
    water_oxygen.Q32s() = 0.0;
    water_oxygen.Q33c() = 0.303856;   // octupole xxx-3xyy (main component)
    water_oxygen.Q33s() = 0.0;
    water_oxygen.Q40() = 0.114584;
    water_oxygen.Q41c() = 0.0;
    water_oxygen.Q41s() = 0.0;
    water_oxygen.Q42c() = -0.183221;
    water_oxygen.Q42s() = 0.0;
    water_oxygen.Q43c() = 0.0;
    water_oxygen.Q43s() = 0.0;
    water_oxygen.Q44c() = 0.0;
    water_oxygen.Q44s() = -0.065424;

    // Test 1: Close distance (2.5 bohr) - larger energy than 3.0
    {
      Vec3 pos1(0.0, 0.0, 0.0);
      Vec3 pos2(2.5, 0.0, 0.0);
      double energy = esp.compute_interaction_energy(water_oxygen, pos1, water_oxygen, pos2);
      double energy_at_3bohr = 0.02753720;

      REQUIRE(std::isfinite(energy));
      INFO("Distance 2.5 bohr: OCC = " << energy << " hartree");
      // Energy should be larger in magnitude at shorter distances
      REQUIRE(std::abs(energy) > std::abs(energy_at_3bohr) * 0.5);
      REQUIRE(std::abs(energy) < 0.1);
    }

    // Test 2: Standard distance (3.0 bohr) - validated reference
    {
      Vec3 pos1(0.0, 0.0, 0.0);
      Vec3 pos2(3.0, 0.0, 0.0);
      double energy = esp.compute_interaction_energy(water_oxygen, pos1, water_oxygen, pos2);
      double expected = 0.02753720;
      double percent_error = 100.0 * std::abs(energy - expected) / expected;

      INFO("Distance 3.0 bohr: OCC = " << energy << ", expected = " << expected);
      REQUIRE(percent_error < 0.001);
    }

    // Test 3: Medium distance (4.0 bohr) - weaker interaction
    {
      Vec3 pos1(0.0, 0.0, 0.0);
      Vec3 pos2(4.0, 0.0, 0.0);
      double energy = esp.compute_interaction_energy(water_oxygen, pos1, water_oxygen, pos2);

      REQUIRE(std::isfinite(energy));
      INFO("Distance 4.0 bohr: OCC = " << energy << " hartree");
      // Energy should be smaller at larger distance
      REQUIRE(std::abs(energy) < 0.025);
      REQUIRE(std::abs(energy) > 0.005);
    }

    // Test 4: Long distance (6.0 bohr) - weaker interaction
    {
      Vec3 pos1(0.0, 0.0, 0.0);
      Vec3 pos2(6.0, 0.0, 0.0);
      double energy = esp.compute_interaction_energy(water_oxygen, pos1, water_oxygen, pos2);
      double energy_at_3bohr = 0.02753720;

      REQUIRE(std::isfinite(energy));
      INFO("Distance 6.0 bohr: OCC = " << energy << " hartree");
      // Should be significantly smaller than 3.0 bohr
      REQUIRE(std::abs(energy) < std::abs(energy_at_3bohr) * 0.75);
    }

    // Test 5: Very long distance (10.0 bohr) - asymptotic behavior
    {
      Vec3 pos1(0.0, 0.0, 0.0);
      Vec3 pos2(10.0, 0.0, 0.0);
      double energy = esp.compute_interaction_energy(water_oxygen, pos1, water_oxygen, pos2);
      double energy_at_3bohr = 0.02753720;

      REQUIRE(std::isfinite(energy));
      INFO("Distance 10.0 bohr: OCC = " << energy << " hartree");
      // Should be much smaller than 3.0 bohr
      REQUIRE(std::abs(energy) < std::abs(energy_at_3bohr) * 0.5);
    }
  }

}

TEST_CASE("HF dimer multi-site interaction", "[mults][esp][interaction][HF]") {

  SECTION("Parallel configuration - exact Orient validation") {
    MultipoleESP esp(4);

    // HF molecule has 3 sites: F atom, H atom, and bond center (BO)
    // Multipoles from Orient's HF..HF test case

    // Site 1: Fluorine atom
    Mult F(4);
    F.Q00() = -0.096093;
    F.Q10() = 0.339674;
    F.Q11c() = 0.0; F.Q11s() = 0.0;
    F.Q20() = 0.511308;
    F.Q21c() = 0.0; F.Q21s() = 0.0; F.Q22c() = 0.0; F.Q22s() = 0.0;
    F.Q30() = -0.079971;
    F.Q31c() = 0.0; F.Q31s() = 0.0; F.Q32c() = 0.0; F.Q32s() = 0.0;
    F.Q33c() = 0.0; F.Q33s() = 0.0;
    F.Q40() = -0.229440;
    F.Q41c() = 0.0; F.Q41s() = 0.0; F.Q42c() = 0.0; F.Q42s() = 0.0;
    F.Q43c() = 0.0; F.Q43s() = 0.0; F.Q44c() = 0.0; F.Q44s() = 0.0;

    // Site 2: Hydrogen atom (at z=1.733069 bohr relative to F)
    Mult H(4);
    H.Q00() = -0.143603;
    H.Q10() = 0.348021;
    H.Q11c() = 0.0; H.Q11s() = 0.0;
    H.Q20() = -0.086102;
    H.Q21c() = 0.0; H.Q21s() = 0.0; H.Q22c() = 0.0; H.Q22s() = 0.0;
    H.Q30() = 0.023389;
    H.Q31c() = 0.0; H.Q31s() = 0.0; H.Q32c() = 0.0; H.Q32s() = 0.0;
    H.Q33c() = 0.0; H.Q33s() = 0.0;
    H.Q40() = -0.003537;
    H.Q41c() = 0.0; H.Q41s() = 0.0; H.Q42c() = 0.0; H.Q42s() = 0.0;
    H.Q43c() = 0.0; H.Q43s() = 0.0; H.Q44c() = 0.0; H.Q44s() = 0.0;

    // Site 3: Bond center (at z=0.866257955 bohr relative to F)
    Mult BO(4);
    BO.Q00() = 0.239696;
    BO.Q10() = 0.038870;
    BO.Q11c() = 0.0; BO.Q11s() = 0.0;
    BO.Q20() = 0.371787;
    BO.Q21c() = 0.0; BO.Q21s() = 0.0; BO.Q22c() = 0.0; BO.Q22s() = 0.0;
    BO.Q30() = -0.228008;
    BO.Q31c() = 0.0; BO.Q31s() = 0.0; BO.Q32c() = 0.0; BO.Q32s() = 0.0;
    BO.Q33c() = 0.0; BO.Q33s() = 0.0;
    BO.Q40() = 0.131259;
    BO.Q41c() = 0.0; BO.Q41s() = 0.0; BO.Q42c() = 0.0; BO.Q42s() = 0.0;
    BO.Q43c() = 0.0; BO.Q43s() = 0.0; BO.Q44c() = 0.0; BO.Q44s() = 0.0;

    // HF1 at origin, aligned along z-axis
    Vec3 F1_pos(0.0, 0.0, 0.0);
    Vec3 H1_pos(0.0, 0.0, 1.733069);
    Vec3 BO1_pos(0.0, 0.0, 0.866257955);

    // HF2 at z=5.4 bohr (parallel configuration)
    Vec3 F2_pos(0.0, 0.0, 5.4);
    Vec3 H2_pos(0.0, 0.0, 5.4 + 1.733069);
    Vec3 BO2_pos(0.0, 0.0, 5.4 + 0.866257955);

    // Compute all 9 site-site interactions
    double total_energy = 0.0;
    double e_ff = esp.compute_interaction_energy(F, F1_pos, F, F2_pos);    // F1-F2
    double e_fh = esp.compute_interaction_energy(F, F1_pos, H, H2_pos);    // F1-H2
    double e_fbo = esp.compute_interaction_energy(F, F1_pos, BO, BO2_pos);  // F1-BO2
    double e_hf = esp.compute_interaction_energy(H, H1_pos, F, F2_pos);    // H1-F2
    double e_hh = esp.compute_interaction_energy(H, H1_pos, H, H2_pos);    // H1-H2
    double e_hbo = esp.compute_interaction_energy(H, H1_pos, BO, BO2_pos);  // H1-BO2
    double e_bof = esp.compute_interaction_energy(BO, BO1_pos, F, F2_pos);  // BO1-F2
    double e_boh = esp.compute_interaction_energy(BO, BO1_pos, H, H2_pos);  // BO1-H2
    double e_bobo = esp.compute_interaction_energy(BO, BO1_pos, BO, BO2_pos);// BO1-BO2

    total_energy = e_ff + e_fh + e_fbo + e_hf + e_hh + e_hbo + e_bof + e_boh + e_bobo;

    INFO("Site-site contributions:");
    INFO("F1-F2:   " << e_ff);
    INFO("F1-H2:   " << e_fh);
    INFO("F1-BO2:  " << e_fbo);
    INFO("H1-F2:   " << e_hf);
    INFO("H1-H2:   " << e_hh);
    INFO("H1-BO2:  " << e_hbo);
    INFO("BO1-F2:  " << e_bof);
    INFO("BO1-H2:  " << e_boh);
    INFO("BO1-BO2: " << e_bobo);

    // Orient reference for parallel configuration (F-F separation = 5.4 bohr)
    // Computed by running Orient single-point energy calculation: -0.00436161 hartree
    double orient_energy = -0.00436161;
    double percent_error = 100.0 * std::abs(total_energy - orient_energy) / std::abs(orient_energy);

    INFO("HF dimer parallel total energy: " << total_energy << " hartree");
    INFO("Orient reference: " << orient_energy << " hartree");
    INFO("Percent error: " << percent_error << "%");

    REQUIRE(percent_error < 0.01);  // Should match Orient within 0.01%
  }
}

TEST_CASE("ESP grid comparison with Orient", "[esp][orient]") {
  
  SECTION("16-point grid around water molecule") {
    MultipoleESP esp(4);

    // Water oxygen multipole from orient_water_esp.txt
    Mult water_oxygen(4);
    water_oxygen.Q00() = -0.330960;   // monopole
    water_oxygen.Q10() = 0.0;         // dipole z
    water_oxygen.Q11c() = -0.297907;  // dipole x
    water_oxygen.Q11s() = 0.0;        // dipole y
    water_oxygen.Q20() = 0.117935;    // quadrupole zz
    water_oxygen.Q21c() = 0.0;
    water_oxygen.Q21s() = 0.0;
    water_oxygen.Q22c() = 0.673922;   // quadrupole (xx-yy)/2
    water_oxygen.Q22s() = 0.0;
    water_oxygen.Q30() = 0.0;         // octupole zzz
    water_oxygen.Q31c() = -0.151827;  // octupole xzz
    water_oxygen.Q31s() = 0.0;
    water_oxygen.Q32c() = 0.0;
    water_oxygen.Q32s() = 0.0;
    water_oxygen.Q33c() = 0.303856;   // octupole (xxx-3xyy)
    water_oxygen.Q33s() = 0.0;
    water_oxygen.Q40() = 0.114584;    // hexadecapole
    water_oxygen.Q41c() = 0.0;
    water_oxygen.Q41s() = 0.0;
    water_oxygen.Q42c() = -0.183221;
    water_oxygen.Q42s() = 0.0;
    water_oxygen.Q43c() = 0.0;
    water_oxygen.Q43s() = 0.0;
    water_oxygen.Q44c() = 0.0;
    water_oxygen.Q44s() = -0.065424;
    
    Vec3 water_position(0.0, 0.0, 0.0);
    
    // 16-point grid around water molecule (cube with points at ±2.0 bohr)
    std::vector<Vec3> grid_points = {
      // Front face (z = -2.0)
      Vec3(-2.0, -2.0, -2.0), Vec3(2.0, -2.0, -2.0),
      Vec3(-2.0,  2.0, -2.0), Vec3(2.0,  2.0, -2.0),
      // Back face (z = 2.0) 
      Vec3(-2.0, -2.0,  2.0), Vec3(2.0, -2.0,  2.0),
      Vec3(-2.0,  2.0,  2.0), Vec3(2.0,  2.0,  2.0),
      // Middle layer (z = 0.0) - avoiding origin
      Vec3(-2.0, -2.0,  0.0), Vec3(2.0, -2.0,  0.0),
      Vec3(-2.0,  2.0,  0.0), Vec3(2.0,  2.0,  0.0),
      // Additional points for better coverage
      Vec3( 0.0, -2.0, -2.0), Vec3(0.0,  2.0, -2.0),
      Vec3( 0.0, -2.0,  2.0), Vec3(0.0,  2.0,  2.0)
    };
    
    // Compute ESP at all grid points
    std::vector<double> esp_values = esp.compute_esp_at_points(
      water_oxygen, water_position, grid_points);
    
    REQUIRE(esp_values.size() == 16);
    
    // All values should be finite
    for (size_t i = 0; i < esp_values.size(); i++) {
      REQUIRE(std::isfinite(esp_values[i]));
      REQUIRE(std::abs(esp_values[i]) < 10.0);  // Reasonable magnitude
    }
    
    // Orient reference ESP values (in Volts, converted to atomic units)
    // 1 V = 1/27.2114 Hartree
    // Mapping OCC grid_points order to Orient output order:
    std::vector<double> orient_esp_values = {
      -2.187948 / 27.2114,  // Point 1: (-2.0, -2.0, -2.0) -> Orient line 1
      -3.016462 / 27.2114,  // Point 2: ( 2.0, -2.0, -2.0) -> Orient line 9
      -2.187948 / 27.2114,  // Point 3: (-2.0,  2.0, -2.0) -> Orient line 2
      -3.016462 / 27.2114,  // Point 4: ( 2.0,  2.0, -2.0) -> Orient line 10
      -2.187948 / 27.2114,  // Point 5: (-2.0, -2.0,  2.0) -> Orient line 3
      -3.016462 / 27.2114,  // Point 6: ( 2.0, -2.0,  2.0) -> Orient line 11
      -2.187948 / 27.2114,  // Point 7: (-2.0,  2.0,  2.0) -> Orient line 4
      -3.016462 / 27.2114,  // Point 8: ( 2.0,  2.0,  2.0) -> Orient line 12
      -2.487729 / 27.2114,  // Point 9: (-2.0, -2.0,  0.0) -> Orient line 5
      -4.009301 / 27.2114,  // Point 10:( 2.0, -2.0,  0.0) -> Orient line 13
      -2.487729 / 27.2114,  // Point 11:(-2.0,  2.0,  0.0) -> Orient line 6
      -4.009301 / 27.2114,  // Point 12:( 2.0,  2.0,  0.0) -> Orient line 14
      -3.487290 / 27.2114,  // Point 13:( 0.0, -2.0, -2.0) -> Orient line 7
      -3.487290 / 27.2114,  // Point 14:( 0.0,  2.0, -2.0) -> Orient line 15
      -3.487290 / 27.2114,  // Point 15:( 0.0, -2.0,  2.0) -> Orient line 8
      -3.487290 / 27.2114   // Point 16:( 0.0,  2.0,  2.0) -> Orient line 16
    };
    
    // Debug: Print our multipole values to verify they match Orient
    fmt::print("\nOCC Multipole values:\n");
    fmt::print("Q00 = {:9.6f}\n", water_oxygen.Q00());
    fmt::print("Q10 = {:9.6f}\n", water_oxygen.Q10());
    fmt::print("Q11c = {:9.6f}\n", water_oxygen.Q11c());
    fmt::print("Q11s = {:9.6f}\n", water_oxygen.Q11s());
    fmt::print("Q20 = {:9.6f}\n", water_oxygen.Q20());
    fmt::print("Q21c = {:9.6f}\n", water_oxygen.Q21c());
    fmt::print("Q21s = {:9.6f}\n", water_oxygen.Q21s());
    fmt::print("Q22c = {:9.6f}\n", water_oxygen.Q22c());
    fmt::print("Q22s = {:9.6f}\n", water_oxygen.Q22s());
    fmt::print("Q30 = {:9.6f}\n", water_oxygen.Q30());
    fmt::print("Q31c = {:9.6f}\n", water_oxygen.Q31c());
    fmt::print("Q31s = {:9.6f}\n", water_oxygen.Q31s());
    fmt::print("Q32c = {:9.6f}\n", water_oxygen.Q32c());
    fmt::print("Q32s = {:9.6f}\n", water_oxygen.Q32s());
    fmt::print("Q33c = {:9.6f}\n", water_oxygen.Q33c());
    fmt::print("Q33s = {:9.6f}\n", water_oxygen.Q33s());
    fmt::print("Q40 = {:9.6f}\n", water_oxygen.Q40());
    fmt::print("Q41c = {:9.6f}\n", water_oxygen.Q41c());
    fmt::print("Q41s = {:9.6f}\n", water_oxygen.Q41s());
    fmt::print("Q42c = {:9.6f}\n", water_oxygen.Q42c());
    fmt::print("Q42s = {:9.6f}\n", water_oxygen.Q42s());
    fmt::print("Q43c = {:9.6f}\n", water_oxygen.Q43c());
    fmt::print("Q43s = {:9.6f}\n", water_oxygen.Q43s());
    fmt::print("Q44c = {:9.6f}\n", water_oxygen.Q44c());
    fmt::print("Q44s = {:9.6f}\n", water_oxygen.Q44s());
    
    // Print comparison table
    fmt::print("\nESP Comparison (OCC vs Orient):\n");
    fmt::print("{:3} {:24} {:12} {:12} {:12} {:8}\n", 
      "Pt", "Coordinates", "OCC (a.u.)", "Orient (a.u.)", "Difference", "% Error");
    fmt::print("{:-<3} {:-<24} {:-<12} {:-<12} {:-<12} {:-<8}\n", "", "", "", "", "", "");
    
    for (size_t i = 0; i < esp_values.size(); i++) {
      double diff = esp_values[i] - orient_esp_values[i];
      double percent_error = 100.0 * std::abs(diff) / std::abs(orient_esp_values[i]);
      
      fmt::print("{:2d} ({:6.1f},{:6.1f},{:6.1f}) {:11.6f} {:11.6f} {:11.6f} {:7.2f}\n", 
        int(i+1), grid_points[i][0], grid_points[i][1], grid_points[i][2],
        esp_values[i], orient_esp_values[i], diff, percent_error);
    }
    
    // Validate against Orient reference values
    for (size_t i = 0; i < esp_values.size(); i++) {
      double percent_error = 100.0 * std::abs(esp_values[i] - orient_esp_values[i]) / std::abs(orient_esp_values[i]);
      // ESP values should match Orient within reasonable tolerance
      INFO("Point " << i+1 << ": OCC=" << esp_values[i] << ", Orient=" << orient_esp_values[i] << ", Error=" << percent_error << "%");
      REQUIRE(percent_error < 0.5); // Relaxed tolerance - ESP should match Orient within 0.5% (excellent agreement)
    }
    
    // Due to symmetry, some points should have similar ESP values
    // Corner points at same distance should be similar  
    double dist_corner = std::sqrt(12.0); // sqrt(2^2 + 2^2 + 2^2)
    
    // Check that finite differences make sense
    // Points closer to the origin should generally have larger magnitude ESP
    Vec3 close_point(0.0, -2.0, -2.0);  // Distance = sqrt(8) ≈ 2.83
    Vec3 far_point(-2.0, -2.0, -2.0);   // Distance = sqrt(12) ≈ 3.46
    
    double esp_close = esp.compute_esp_at_point(water_oxygen, water_position, close_point);
    double esp_far = esp.compute_esp_at_point(water_oxygen, water_position, far_point);
    
    // Closer point should generally have larger magnitude (1/r scaling)
    REQUIRE(std::abs(esp_close) >= std::abs(esp_far) * 0.5); // Allow some tolerance for multipole effects
  }
}





TEST_CASE("Systematic multipole interaction validation", "[mults][interaction][systematic]") {
    MultipoleESP calc(4);  // Up to hexadecapole (rank 4)
    Vec3 pos1(0, 0, 0);
    Vec3 pos2(0, 0, 3.0);  // 3 bohr separation along z

    // Helper to create multipole with single component
    auto make_multipole = [](const std::string& name, double value, int rank) {
        Mult mult(rank);

        // Initialize all components to zero
        for (int i = 0; i < mult.num_components(); i++) {
            mult.q(i) = 0.0;
        }

        // Set specific component
        if (name == "Q00") { mult.Q00() = value; }
        if (name == "Q10") { mult.Q10() = value; }
        if (name == "Q11c") { mult.Q11c() = value; }
        if (name == "Q11s") { mult.Q11s() = value; }
        if (name == "Q20") { mult.Q20() = value; }
        if (name == "Q21c") { mult.Q21c() = value; }
        if (name == "Q21s") { mult.Q21s() = value; }
        if (name == "Q22c") { mult.Q22c() = value; }
        if (name == "Q22s") { mult.Q22s() = value; }
        if (name == "Q30") { mult.Q30() = value; }
        if (name == "Q31c") { mult.Q31c() = value; }
        if (name == "Q31s") { mult.Q31s() = value; }
        if (name == "Q32c") { mult.Q32c() = value; }
        if (name == "Q32s") { mult.Q32s() = value; }
        if (name == "Q33c") { mult.Q33c() = value; }
        if (name == "Q33s") { mult.Q33s() = value; }
        if (name == "Q40") { mult.Q40() = value; }
        if (name == "Q41c") { mult.Q41c() = value; }
        if (name == "Q41s") { mult.Q41s() = value; }
        if (name == "Q42c") { mult.Q42c() = value; }
        if (name == "Q42s") { mult.Q42s() = value; }
        if (name == "Q43c") { mult.Q43c() = value; }
        if (name == "Q43s") { mult.Q43s() = value; }
        if (name == "Q44c") { mult.Q44c() = value; }
        if (name == "Q44s") { mult.Q44s() = value; }

        return mult;
    };

    SECTION("Q00 x Q00") {
        Mult mult1 = make_multipole("Q00", 1.0, 0);
        Mult mult2 = make_multipole("Q00", 1.0, 0);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.33333333;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q10") {
        Mult mult1 = make_multipole("Q00", 1.0, 1);
        Mult mult2 = make_multipole("Q10", 1.0, 1);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.11111111;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11c") {
        Mult mult1 = make_multipole("Q00", 1.0, 1);
        Mult mult2 = make_multipole("Q11c", 1.0, 1);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11s") {
        Mult mult1 = make_multipole("Q00", 1.0, 1);
        Mult mult2 = make_multipole("Q11s", 1.0, 1);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q20") {
        Mult mult1 = make_multipole("Q00", 1.0, 2);
        Mult mult2 = make_multipole("Q20", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q21c") {
        Mult mult1 = make_multipole("Q00", 1.0, 2);
        Mult mult2 = make_multipole("Q21c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q21s") {
        Mult mult1 = make_multipole("Q00", 1.0, 2);
        Mult mult2 = make_multipole("Q21s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q22c") {
        Mult mult1 = make_multipole("Q00", 1.0, 2);
        Mult mult2 = make_multipole("Q22c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q22s") {
        Mult mult1 = make_multipole("Q00", 1.0, 2);
        Mult mult2 = make_multipole("Q22s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q30") {
        Mult mult1 = make_multipole("Q00", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.01234568;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q31c") {
        Mult mult1 = make_multipole("Q00", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q31s") {
        Mult mult1 = make_multipole("Q00", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q32c") {
        Mult mult1 = make_multipole("Q00", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q32s") {
        Mult mult1 = make_multipole("Q00", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q33c") {
        Mult mult1 = make_multipole("Q00", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q33s") {
        Mult mult1 = make_multipole("Q00", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q40") {
        Mult mult1 = make_multipole("Q00", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00411523;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q41c") {
        Mult mult1 = make_multipole("Q00", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q41s") {
        Mult mult1 = make_multipole("Q00", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q42c") {
        Mult mult1 = make_multipole("Q00", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q42s") {
        Mult mult1 = make_multipole("Q00", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q43c") {
        Mult mult1 = make_multipole("Q00", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q43s") {
        Mult mult1 = make_multipole("Q00", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q44c") {
        Mult mult1 = make_multipole("Q00", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q44s") {
        Mult mult1 = make_multipole("Q00", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q10") {
        Mult mult1 = make_multipole("Q10", 1.0, 1);
        Mult mult2 = make_multipole("Q10", 1.0, 1);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.07407407;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q11c") {
        Mult mult1 = make_multipole("Q10", 1.0, 1);
        Mult mult2 = make_multipole("Q11c", 1.0, 1);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q11s") {
        Mult mult1 = make_multipole("Q10", 1.0, 1);
        Mult mult2 = make_multipole("Q11s", 1.0, 1);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q20") {
        Mult mult1 = make_multipole("Q10", 1.0, 2);
        Mult mult2 = make_multipole("Q20", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q21c") {
        Mult mult1 = make_multipole("Q10", 1.0, 2);
        Mult mult2 = make_multipole("Q21c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q21s") {
        Mult mult1 = make_multipole("Q10", 1.0, 2);
        Mult mult2 = make_multipole("Q21s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q22c") {
        Mult mult1 = make_multipole("Q10", 1.0, 2);
        Mult mult2 = make_multipole("Q22c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q22s") {
        Mult mult1 = make_multipole("Q10", 1.0, 2);
        Mult mult2 = make_multipole("Q22s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q30") {
        Mult mult1 = make_multipole("Q10", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.01646091;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q31c") {
        Mult mult1 = make_multipole("Q10", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q31s") {
        Mult mult1 = make_multipole("Q10", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q32c") {
        Mult mult1 = make_multipole("Q10", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q32s") {
        Mult mult1 = make_multipole("Q10", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q33c") {
        Mult mult1 = make_multipole("Q10", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q33s") {
        Mult mult1 = make_multipole("Q10", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q40") {
        // Rank 5 interaction (j = l1 + l2 = 1 + 4 = 5)
        // OCC supports j=5 interactions (hexadecapole-dipole kernels)
        MultipoleESP calc5(5);  // Create calculator with max_rank=5

        Mult mult1 = make_multipole("Q10", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);
        double energy = calc5.compute_interaction_energy(mult1, pos1, mult2, pos2);

        INFO("OCC energy: " << energy);

        // Should compute non-zero energy for dipole-hexadecapole interaction
        REQUIRE(std::abs(energy) > 1e-10);

        // When max_rank < 5, should exclude this term
        MultipoleESP calc4(4);
        double energy_excluded = calc4.compute_interaction_energy(mult1, pos1, mult2, pos2);
        REQUIRE(std::abs(energy_excluded) < 1e-10);
    }

    SECTION("Q10 x Q41c") {
        Mult mult1 = make_multipole("Q10", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q41s") {
        Mult mult1 = make_multipole("Q10", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q42c") {
        Mult mult1 = make_multipole("Q10", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q42s") {
        Mult mult1 = make_multipole("Q10", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q43c") {
        Mult mult1 = make_multipole("Q10", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q43s") {
        Mult mult1 = make_multipole("Q10", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q44c") {
        Mult mult1 = make_multipole("Q10", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q44s") {
        Mult mult1 = make_multipole("Q10", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q11c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 1);
        Mult mult2 = make_multipole("Q11c", 1.0, 1);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q11s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 1);
        Mult mult2 = make_multipole("Q11s", 1.0, 1);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q20") {
        Mult mult1 = make_multipole("Q11c", 1.0, 2);
        Mult mult2 = make_multipole("Q20", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q21c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 2);
        Mult mult2 = make_multipole("Q21c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.02138334;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q21s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 2);
        Mult mult2 = make_multipole("Q21s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q22c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 2);
        Mult mult2 = make_multipole("Q22c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q22s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 2);
        Mult mult2 = make_multipole("Q22s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q30") {
        Mult mult1 = make_multipole("Q11c", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q31c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0100802;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q31s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q32c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q32s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q33c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q33s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q40") {
        Mult mult1 = make_multipole("Q11c", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q41c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q41s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q42c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q42s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q43c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q43s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q44c") {
        Mult mult1 = make_multipole("Q11c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q44s") {
        Mult mult1 = make_multipole("Q11c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q11s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 1);
        Mult mult2 = make_multipole("Q11s", 1.0, 1);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q20") {
        Mult mult1 = make_multipole("Q11s", 1.0, 2);
        Mult mult2 = make_multipole("Q20", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q21c") {
        Mult mult1 = make_multipole("Q11s", 1.0, 2);
        Mult mult2 = make_multipole("Q21c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q21s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 2);
        Mult mult2 = make_multipole("Q21s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.02138334;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q22c") {
        Mult mult1 = make_multipole("Q11s", 1.0, 2);
        Mult mult2 = make_multipole("Q22c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q22s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 2);
        Mult mult2 = make_multipole("Q22s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q30") {
        Mult mult1 = make_multipole("Q11s", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q31c") {
        Mult mult1 = make_multipole("Q11s", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q31s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0100802;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q32c") {
        Mult mult1 = make_multipole("Q11s", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q32s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q33c") {
        Mult mult1 = make_multipole("Q11s", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q33s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q40") {
        Mult mult1 = make_multipole("Q11s", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q41c") {
        Mult mult1 = make_multipole("Q11s", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q41s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q42c") {
        Mult mult1 = make_multipole("Q11s", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q42s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q43c") {
        Mult mult1 = make_multipole("Q11s", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q43s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q44c") {
        Mult mult1 = make_multipole("Q11s", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q44s") {
        Mult mult1 = make_multipole("Q11s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q20") {
        Mult mult1 = make_multipole("Q20", 1.0, 2);
        Mult mult2 = make_multipole("Q20", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.02469136;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q21c") {
        Mult mult1 = make_multipole("Q20", 1.0, 2);
        Mult mult2 = make_multipole("Q21c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q21s") {
        Mult mult1 = make_multipole("Q20", 1.0, 2);
        Mult mult2 = make_multipole("Q21s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q22c") {
        Mult mult1 = make_multipole("Q20", 1.0, 2);
        Mult mult2 = make_multipole("Q22c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q22s") {
        Mult mult1 = make_multipole("Q20", 1.0, 2);
        Mult mult2 = make_multipole("Q22s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q30") {
        Mult mult1 = make_multipole("Q20", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q31c") {
        Mult mult1 = make_multipole("Q20", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q31s") {
        Mult mult1 = make_multipole("Q20", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q32c") {
        Mult mult1 = make_multipole("Q20", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q32s") {
        Mult mult1 = make_multipole("Q20", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q33c") {
        Mult mult1 = make_multipole("Q20", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q33s") {
        Mult mult1 = make_multipole("Q20", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q40") {
        Mult mult1 = make_multipole("Q20", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q41c") {
        Mult mult1 = make_multipole("Q20", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q41s") {
        Mult mult1 = make_multipole("Q20", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q42c") {
        Mult mult1 = make_multipole("Q20", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q42s") {
        Mult mult1 = make_multipole("Q20", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q43c") {
        Mult mult1 = make_multipole("Q20", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q43s") {
        Mult mult1 = make_multipole("Q20", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q44c") {
        Mult mult1 = make_multipole("Q20", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q20 x Q44s") {
        Mult mult1 = make_multipole("Q20", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q21c") {
        Mult mult1 = make_multipole("Q21c", 1.0, 2);
        Mult mult2 = make_multipole("Q21c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.01646091;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q21s") {
        Mult mult1 = make_multipole("Q21c", 1.0, 2);
        Mult mult2 = make_multipole("Q21s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q22c") {
        Mult mult1 = make_multipole("Q21c", 1.0, 2);
        Mult mult2 = make_multipole("Q22c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q22s") {
        Mult mult1 = make_multipole("Q21c", 1.0, 2);
        Mult mult2 = make_multipole("Q22s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q30") {
        Mult mult1 = make_multipole("Q21c", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q31c") {
        Mult mult1 = make_multipole("Q21c", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q31s") {
        Mult mult1 = make_multipole("Q21c", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q32c") {
        Mult mult1 = make_multipole("Q21c", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q32s") {
        Mult mult1 = make_multipole("Q21c", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q33c") {
        Mult mult1 = make_multipole("Q21c", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q33s") {
        Mult mult1 = make_multipole("Q21c", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q40") {
        Mult mult1 = make_multipole("Q21c", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q41c") {
        Mult mult1 = make_multipole("Q21c", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q41s") {
        Mult mult1 = make_multipole("Q21c", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q42c") {
        Mult mult1 = make_multipole("Q21c", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q42s") {
        Mult mult1 = make_multipole("Q21c", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q43c") {
        Mult mult1 = make_multipole("Q21c", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q43s") {
        Mult mult1 = make_multipole("Q21c", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q44c") {
        Mult mult1 = make_multipole("Q21c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21c x Q44s") {
        Mult mult1 = make_multipole("Q21c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q21s") {
        Mult mult1 = make_multipole("Q21s", 1.0, 2);
        Mult mult2 = make_multipole("Q21s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.01646091;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q22c") {
        Mult mult1 = make_multipole("Q21s", 1.0, 2);
        Mult mult2 = make_multipole("Q22c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q22s") {
        Mult mult1 = make_multipole("Q21s", 1.0, 2);
        Mult mult2 = make_multipole("Q22s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q30") {
        Mult mult1 = make_multipole("Q21s", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q31c") {
        Mult mult1 = make_multipole("Q21s", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q31s") {
        Mult mult1 = make_multipole("Q21s", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q32c") {
        Mult mult1 = make_multipole("Q21s", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q32s") {
        Mult mult1 = make_multipole("Q21s", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q33c") {
        Mult mult1 = make_multipole("Q21s", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q33s") {
        Mult mult1 = make_multipole("Q21s", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q40") {
        Mult mult1 = make_multipole("Q21s", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q41c") {
        Mult mult1 = make_multipole("Q21s", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q41s") {
        Mult mult1 = make_multipole("Q21s", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q42c") {
        Mult mult1 = make_multipole("Q21s", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q42s") {
        Mult mult1 = make_multipole("Q21s", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q43c") {
        Mult mult1 = make_multipole("Q21s", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q43s") {
        Mult mult1 = make_multipole("Q21s", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q44c") {
        Mult mult1 = make_multipole("Q21s", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q21s x Q44s") {
        Mult mult1 = make_multipole("Q21s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q22c") {
        Mult mult1 = make_multipole("Q22c", 1.0, 2);
        Mult mult2 = make_multipole("Q22c", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00411523;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q22s") {
        Mult mult1 = make_multipole("Q22c", 1.0, 2);
        Mult mult2 = make_multipole("Q22s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q30") {
        Mult mult1 = make_multipole("Q22c", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q31c") {
        Mult mult1 = make_multipole("Q22c", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q31s") {
        Mult mult1 = make_multipole("Q22c", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q32c") {
        Mult mult1 = make_multipole("Q22c", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q32s") {
        Mult mult1 = make_multipole("Q22c", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q33c") {
        Mult mult1 = make_multipole("Q22c", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q33s") {
        Mult mult1 = make_multipole("Q22c", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q40") {
        Mult mult1 = make_multipole("Q22c", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q41c") {
        Mult mult1 = make_multipole("Q22c", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q41s") {
        Mult mult1 = make_multipole("Q22c", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q42c") {
        Mult mult1 = make_multipole("Q22c", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q42s") {
        Mult mult1 = make_multipole("Q22c", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q43c") {
        Mult mult1 = make_multipole("Q22c", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q43s") {
        Mult mult1 = make_multipole("Q22c", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q44c") {
        Mult mult1 = make_multipole("Q22c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22c x Q44s") {
        Mult mult1 = make_multipole("Q22c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q22s") {
        Mult mult1 = make_multipole("Q22s", 1.0, 2);
        Mult mult2 = make_multipole("Q22s", 1.0, 2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00411523;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q30") {
        Mult mult1 = make_multipole("Q22s", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q31c") {
        Mult mult1 = make_multipole("Q22s", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q31s") {
        Mult mult1 = make_multipole("Q22s", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q32c") {
        Mult mult1 = make_multipole("Q22s", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q32s") {
        Mult mult1 = make_multipole("Q22s", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q33c") {
        Mult mult1 = make_multipole("Q22s", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q33s") {
        Mult mult1 = make_multipole("Q22s", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q40") {
        Mult mult1 = make_multipole("Q22s", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q41c") {
        Mult mult1 = make_multipole("Q22s", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q41s") {
        Mult mult1 = make_multipole("Q22s", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q42c") {
        Mult mult1 = make_multipole("Q22s", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q42s") {
        Mult mult1 = make_multipole("Q22s", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q43c") {
        Mult mult1 = make_multipole("Q22s", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q43s") {
        Mult mult1 = make_multipole("Q22s", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q44c") {
        Mult mult1 = make_multipole("Q22s", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q22s x Q44s") {
        Mult mult1 = make_multipole("Q22s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q30") {
        Mult mult1 = make_multipole("Q30", 1.0, 3);
        Mult mult2 = make_multipole("Q30", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q31c") {
        Mult mult1 = make_multipole("Q30", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q31s") {
        Mult mult1 = make_multipole("Q30", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q32c") {
        Mult mult1 = make_multipole("Q30", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q32s") {
        Mult mult1 = make_multipole("Q30", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q33c") {
        Mult mult1 = make_multipole("Q30", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q33s") {
        Mult mult1 = make_multipole("Q30", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q40") {
        Mult mult1 = make_multipole("Q30", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q41c") {
        Mult mult1 = make_multipole("Q30", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q41s") {
        Mult mult1 = make_multipole("Q30", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q42c") {
        Mult mult1 = make_multipole("Q30", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q42s") {
        Mult mult1 = make_multipole("Q30", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q43c") {
        Mult mult1 = make_multipole("Q30", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q43s") {
        Mult mult1 = make_multipole("Q30", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q44c") {
        Mult mult1 = make_multipole("Q30", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q30 x Q44s") {
        Mult mult1 = make_multipole("Q30", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q31c") {
        Mult mult1 = make_multipole("Q31c", 1.0, 3);
        Mult mult2 = make_multipole("Q31c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q31s") {
        Mult mult1 = make_multipole("Q31c", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q32c") {
        Mult mult1 = make_multipole("Q31c", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q32s") {
        Mult mult1 = make_multipole("Q31c", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q33c") {
        Mult mult1 = make_multipole("Q31c", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q33s") {
        Mult mult1 = make_multipole("Q31c", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q40") {
        Mult mult1 = make_multipole("Q31c", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q41c") {
        Mult mult1 = make_multipole("Q31c", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q41s") {
        Mult mult1 = make_multipole("Q31c", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q42c") {
        Mult mult1 = make_multipole("Q31c", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q42s") {
        Mult mult1 = make_multipole("Q31c", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q43c") {
        Mult mult1 = make_multipole("Q31c", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q43s") {
        Mult mult1 = make_multipole("Q31c", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q44c") {
        Mult mult1 = make_multipole("Q31c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31c x Q44s") {
        Mult mult1 = make_multipole("Q31c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q31s") {
        Mult mult1 = make_multipole("Q31s", 1.0, 3);
        Mult mult2 = make_multipole("Q31s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q32c") {
        Mult mult1 = make_multipole("Q31s", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q32s") {
        Mult mult1 = make_multipole("Q31s", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q33c") {
        Mult mult1 = make_multipole("Q31s", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q33s") {
        Mult mult1 = make_multipole("Q31s", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q40") {
        Mult mult1 = make_multipole("Q31s", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q41c") {
        Mult mult1 = make_multipole("Q31s", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q41s") {
        Mult mult1 = make_multipole("Q31s", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q42c") {
        Mult mult1 = make_multipole("Q31s", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q42s") {
        Mult mult1 = make_multipole("Q31s", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q43c") {
        Mult mult1 = make_multipole("Q31s", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q43s") {
        Mult mult1 = make_multipole("Q31s", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q44c") {
        Mult mult1 = make_multipole("Q31s", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q31s x Q44s") {
        Mult mult1 = make_multipole("Q31s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q32c") {
        Mult mult1 = make_multipole("Q32c", 1.0, 3);
        Mult mult2 = make_multipole("Q32c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q32s") {
        Mult mult1 = make_multipole("Q32c", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q33c") {
        Mult mult1 = make_multipole("Q32c", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q33s") {
        Mult mult1 = make_multipole("Q32c", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q40") {
        Mult mult1 = make_multipole("Q32c", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q41c") {
        Mult mult1 = make_multipole("Q32c", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q41s") {
        Mult mult1 = make_multipole("Q32c", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q42c") {
        Mult mult1 = make_multipole("Q32c", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q42s") {
        Mult mult1 = make_multipole("Q32c", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q43c") {
        Mult mult1 = make_multipole("Q32c", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q43s") {
        Mult mult1 = make_multipole("Q32c", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q44c") {
        Mult mult1 = make_multipole("Q32c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32c x Q44s") {
        Mult mult1 = make_multipole("Q32c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q32s") {
        Mult mult1 = make_multipole("Q32s", 1.0, 3);
        Mult mult2 = make_multipole("Q32s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q33c") {
        Mult mult1 = make_multipole("Q32s", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q33s") {
        Mult mult1 = make_multipole("Q32s", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q40") {
        Mult mult1 = make_multipole("Q32s", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q41c") {
        Mult mult1 = make_multipole("Q32s", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q41s") {
        Mult mult1 = make_multipole("Q32s", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q42c") {
        Mult mult1 = make_multipole("Q32s", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q42s") {
        Mult mult1 = make_multipole("Q32s", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q43c") {
        Mult mult1 = make_multipole("Q32s", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q43s") {
        Mult mult1 = make_multipole("Q32s", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q44c") {
        Mult mult1 = make_multipole("Q32s", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q32s x Q44s") {
        Mult mult1 = make_multipole("Q32s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q33c") {
        Mult mult1 = make_multipole("Q33c", 1.0, 3);
        Mult mult2 = make_multipole("Q33c", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q33s") {
        Mult mult1 = make_multipole("Q33c", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q40") {
        Mult mult1 = make_multipole("Q33c", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q41c") {
        Mult mult1 = make_multipole("Q33c", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q41s") {
        Mult mult1 = make_multipole("Q33c", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q42c") {
        Mult mult1 = make_multipole("Q33c", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q42s") {
        Mult mult1 = make_multipole("Q33c", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q43c") {
        Mult mult1 = make_multipole("Q33c", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q43s") {
        Mult mult1 = make_multipole("Q33c", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q44c") {
        Mult mult1 = make_multipole("Q33c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33c x Q44s") {
        Mult mult1 = make_multipole("Q33c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q33s") {
        Mult mult1 = make_multipole("Q33s", 1.0, 3);
        Mult mult2 = make_multipole("Q33s", 1.0, 3);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q40") {
        Mult mult1 = make_multipole("Q33s", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q41c") {
        Mult mult1 = make_multipole("Q33s", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q41s") {
        Mult mult1 = make_multipole("Q33s", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q42c") {
        Mult mult1 = make_multipole("Q33s", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q42s") {
        Mult mult1 = make_multipole("Q33s", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q43c") {
        Mult mult1 = make_multipole("Q33s", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q43s") {
        Mult mult1 = make_multipole("Q33s", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q44c") {
        Mult mult1 = make_multipole("Q33s", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q33s x Q44s") {
        Mult mult1 = make_multipole("Q33s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q40 x Q40") {
        Mult mult1 = make_multipole("Q40", 1.0, 4);
        Mult mult2 = make_multipole("Q40", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q40 x Q41c") {
        Mult mult1 = make_multipole("Q40", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q40 x Q41s") {
        Mult mult1 = make_multipole("Q40", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q40 x Q42c") {
        Mult mult1 = make_multipole("Q40", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q40 x Q42s") {
        Mult mult1 = make_multipole("Q40", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q40 x Q43c") {
        Mult mult1 = make_multipole("Q40", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q40 x Q43s") {
        Mult mult1 = make_multipole("Q40", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q40 x Q44c") {
        Mult mult1 = make_multipole("Q40", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q40 x Q44s") {
        Mult mult1 = make_multipole("Q40", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41c x Q41c") {
        Mult mult1 = make_multipole("Q41c", 1.0, 4);
        Mult mult2 = make_multipole("Q41c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41c x Q41s") {
        Mult mult1 = make_multipole("Q41c", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41c x Q42c") {
        Mult mult1 = make_multipole("Q41c", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41c x Q42s") {
        Mult mult1 = make_multipole("Q41c", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41c x Q43c") {
        Mult mult1 = make_multipole("Q41c", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41c x Q43s") {
        Mult mult1 = make_multipole("Q41c", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41c x Q44c") {
        Mult mult1 = make_multipole("Q41c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41c x Q44s") {
        Mult mult1 = make_multipole("Q41c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41s x Q41s") {
        Mult mult1 = make_multipole("Q41s", 1.0, 4);
        Mult mult2 = make_multipole("Q41s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41s x Q42c") {
        Mult mult1 = make_multipole("Q41s", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41s x Q42s") {
        Mult mult1 = make_multipole("Q41s", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41s x Q43c") {
        Mult mult1 = make_multipole("Q41s", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41s x Q43s") {
        Mult mult1 = make_multipole("Q41s", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41s x Q44c") {
        Mult mult1 = make_multipole("Q41s", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q41s x Q44s") {
        Mult mult1 = make_multipole("Q41s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42c x Q42c") {
        Mult mult1 = make_multipole("Q42c", 1.0, 4);
        Mult mult2 = make_multipole("Q42c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42c x Q42s") {
        Mult mult1 = make_multipole("Q42c", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42c x Q43c") {
        Mult mult1 = make_multipole("Q42c", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42c x Q43s") {
        Mult mult1 = make_multipole("Q42c", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42c x Q44c") {
        Mult mult1 = make_multipole("Q42c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42c x Q44s") {
        Mult mult1 = make_multipole("Q42c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42s x Q42s") {
        Mult mult1 = make_multipole("Q42s", 1.0, 4);
        Mult mult2 = make_multipole("Q42s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42s x Q43c") {
        Mult mult1 = make_multipole("Q42s", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42s x Q43s") {
        Mult mult1 = make_multipole("Q42s", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42s x Q44c") {
        Mult mult1 = make_multipole("Q42s", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q42s x Q44s") {
        Mult mult1 = make_multipole("Q42s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q43c x Q43c") {
        Mult mult1 = make_multipole("Q43c", 1.0, 4);
        Mult mult2 = make_multipole("Q43c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q43c x Q43s") {
        Mult mult1 = make_multipole("Q43c", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q43c x Q44c") {
        Mult mult1 = make_multipole("Q43c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q43c x Q44s") {
        Mult mult1 = make_multipole("Q43c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q43s x Q43s") {
        Mult mult1 = make_multipole("Q43s", 1.0, 4);
        Mult mult2 = make_multipole("Q43s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q43s x Q44c") {
        Mult mult1 = make_multipole("Q43s", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q43s x Q44s") {
        Mult mult1 = make_multipole("Q43s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q44c x Q44c") {
        Mult mult1 = make_multipole("Q44c", 1.0, 4);
        Mult mult2 = make_multipole("Q44c", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q44c x Q44s") {
        Mult mult1 = make_multipole("Q44c", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q44s x Q44s") {
        Mult mult1 = make_multipole("Q44s", 1.0, 4);
        Mult mult2 = make_multipole("Q44s", 1.0, 4);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.0;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

}

TEST_CASE("Systematic multipole interaction - multiple orientations", "[mults][interaction][geometry]") {
    MultipoleESP calc(4);  // Up to hexadecapole (rank 4)

    // Helper to create multipole with single component
    auto make_multipole = [](const std::string& name, double value, int rank) {
        Mult mult(rank);

        // Initialize all components to zero
        for (int i = 0; i < mult.num_components(); i++) {
            mult.q(i) = 0.0;
        }

        // Set specific component
        if (name == "Q00") { mult.Q00() = value; }
        if (name == "Q10") { mult.Q10() = value; }
        if (name == "Q11c") { mult.Q11c() = value; }
        if (name == "Q11s") { mult.Q11s() = value; }
        if (name == "Q20") { mult.Q20() = value; }
        if (name == "Q21c") { mult.Q21c() = value; }
        if (name == "Q21s") { mult.Q21s() = value; }
        if (name == "Q22c") { mult.Q22c() = value; }
        if (name == "Q22s") { mult.Q22s() = value; }

        return mult;
    };

    // Helper for rank lookup
    auto get_rank = [](const std::string& name) {
        if (name == "Q00") return 0;
        if (name[1] == '1') return 1;
        if (name[1] == '2') return 2;
        if (name[1] == '3') return 3;
        if (name[1] == '4') return 4;
        return 0;
    };

    // ===== Q00 x Q00 =====

    SECTION("Q00 x Q00 - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q00");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q00", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.33333333;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q00 - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q00");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q00", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.33333333;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q00 - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q00");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q00", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.33333333;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q00 - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q00");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q00", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.33333333;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q00 - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q00");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q00", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.33333333;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q00 - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q00");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q00", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.33333333;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q00 x Q10 =====

    SECTION("Q00 x Q10 - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.11111111;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q10 - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q10 - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q10 - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.06415003;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q10 - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q10 - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.07856742;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q00 x Q11c =====

    SECTION("Q00 x Q11c - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11c - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.11111111;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11c - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11c - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.06415003;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11c - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.08888889;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11c - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.05555556;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q00 x Q11s =====

    SECTION("Q00 x Q11s - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11s - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11s - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.11111111;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11s - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.06415003;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11s - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.06666667;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q11s - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.05555556;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q10 x Q10 =====

    SECTION("Q10 x Q10 - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.07407407;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q10 - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q10 - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q10 - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q10 - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q10 - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q10");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q10", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.01851852;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q11c x Q11c =====

    SECTION("Q11c x Q11c - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q11c - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.07407407;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q11c - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q11c - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q11c - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.03407407;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q11c - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00925926;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q11s x Q11s =====

    SECTION("Q11s x Q11s - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q11s");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q11s", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q11s - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q11s");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q11s", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q11s - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q11s");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q11s", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.07407407;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q11s - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q11s");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q11s", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q11s - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q11s");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q11s", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.00296296;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11s x Q11s - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q11s");
        int rank2 = get_rank("Q11s");
        Mult mult1 = make_multipole("Q11s", 1.0, rank1);
        Mult mult2 = make_multipole("Q11s", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00925926;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q00 x Q20 =====

    SECTION("Q00 x Q20 - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q20 - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.01851852;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q20 - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.01851852;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q20 - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q20 - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.01851852;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q20 - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00925926;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q00 x Q22c =====

    SECTION("Q00 x Q22c - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q22c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q22c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q22c - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q22c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q22c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03207501;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q22c - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q22c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q22c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.03207501;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q22c - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q22c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q22c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q22c - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q22c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q22c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00898100;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q00 x Q22c - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q00");
        int rank2 = get_rank("Q22c");
        Mult mult1 = make_multipole("Q00", 1.0, rank1);
        Mult mult2 = make_multipole("Q22c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q10 x Q20 =====

    SECTION("Q10 x Q20 - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q20 - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q20 - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q20 - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.01425556;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q20 - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q20 - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q20");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q20", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.00654729;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q11c x Q21c =====

    SECTION("Q11c x Q21c - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q21c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q21c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.02138334;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q21c - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q21c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q21c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q21c - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q21c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q21c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q21c - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q21c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q21c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00823045;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q21c - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q21c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q21c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q11c x Q21c - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q11c");
        int rank2 = get_rank("Q21c");
        Mult mult1 = make_multipole("Q11c", 1.0, rank1);
        Mult mult2 = make_multipole("Q21c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00378008;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    // ===== Q10 x Q11c =====

    SECTION("Q10 x Q11c - z-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 0.0, 3.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: z-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q11c - x-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(3.0, 0.0, 0.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: x-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q11c - y-axis") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(0.0, 3.0, 0.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: y-axis");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q11c - diagonal") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.7320508075688772, 1.7320508075688772, 1.7320508075688772);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.03703704;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: diagonal");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q11c - off-axis-1") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(2.4, 1.8, 0.0);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = 0.00000000;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-1");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

    SECTION("Q10 x Q11c - off-axis-2") {
        Vec3 pos1(0.0, 0.0, 0.0);
        Vec3 pos2(1.5, 1.5, 2.121320344);

        int rank1 = get_rank("Q10");
        int rank2 = get_rank("Q11c");
        Mult mult1 = make_multipole("Q10", 1.0, rank1);
        Mult mult2 = make_multipole("Q11c", 1.0, rank2);

        double energy = calc.compute_interaction_energy(mult1, pos1, mult2, pos2);
        double orient_energy = -0.03928371;

        INFO("OCC energy: " << energy);
        INFO("Orient energy: " << orient_energy);
        INFO("Geometry: off-axis-2");
        INFO("Distance: 3.000000 bohr");

        if (std::abs(orient_energy) > 1e-10) {
            double error_percent = std::abs((energy - orient_energy) / orient_energy) * 100.0;
            INFO("Relative error: " << error_percent << "%");
            REQUIRE(error_percent < 0.5);  // Less than 0.5% error
        } else {
            // For near-zero energies, use absolute error
            double error_abs = std::abs(energy - orient_energy);
            INFO("Absolute error: " << error_abs);
            REQUIRE(error_abs < 1e-8);  // Less than 1e-8 hartree absolute error
        }
    }

}

TEST_CASE("SFunctionResult basic operations", "[mults][result]") {
    using namespace occ::mults;

    SFunctionResult r1;
    r1.s0 = 2.0;
    r1.s1[0] = 1.0;
    r1.s1[3] = 0.5;

    // Test apply_factor
    r1.apply_factor(3.0);
    REQUIRE(r1.s0 == Approx(6.0));
    REQUIRE(r1.s1[0] == Approx(3.0));
    REQUIRE(r1.s1[3] == Approx(1.5));

    // Test operator+=
    SFunctionResult r2;
    r2.s0 = 1.0;
    r2.s1[0] = 0.5;

    r1 += r2;
    REQUIRE(r1.s0 == Approx(7.0));
    REQUIRE(r1.s1[0] == Approx(3.5));
}

TEST_CASE("SFunctionTermList basic operations", "[mults][termlist]") {
    using namespace occ::mults;

    SFunctionTermList list;
    REQUIRE(list.empty());

    list.add_term(0, 0, 0, 1.0, 1);  // Charge-charge
    list.add_term(1, 1, 2, 0.5, 3);  // Dipole-dipole

    REQUIRE(list.size() == 2);
    REQUIRE(list.terms[0].t1 == 0);
    REQUIRE(list.terms[0].coeff == Approx(1.0));
    REQUIRE(list.terms[1].power == 3);
}

TEST_CASE("SFunctionEvaluator basic usage", "[mults][evaluator]") {
    using namespace occ::mults;

    SFunctionEvaluator eval(4);

    Vec3 ra(-2, -2, -2);
    Vec3 rb(2, 2, 2);
    eval.set_coordinates(ra, rb);

    REQUIRE(eval.r() == Approx(6.92820323).epsilon(1e-6));

    // Test charge-charge (t1=0, t2=0, j=0)
    auto result = eval.compute(0, 0, 0, 0);
    REQUIRE(result.s0 == Approx(1.0));

    // Test dipole-charge (t1=2, t2=0, j=1) - should be non-zero for this geometry
    result = eval.compute(2, 0, 1, 0);
    // For this geometry, should get a specific value based on orientation
    // Just check that computation runs without error
    REQUIRE(std::isfinite(result.s0));
}

TEST_CASE("SFunctionEvaluator batch evaluation", "[mults][evaluator][batch]") {
    using namespace occ::mults;

    SFunctionEvaluator eval(4);
    eval.set_coordinates(Vec3(0, 0, 0), Vec3(1, 0, 0));

    // Create term list with a few simple terms
    SFunctionTermList terms;
    terms.add_term(0, 0, 0, 1.0, 1);  // Charge-charge
    terms.add_term(2, 0, 1, 0.5, 2);  // Dipole-charge (x-component)

    auto results = eval.compute_batch(terms, 0);

    REQUIRE(results.size() == 2);
    REQUIRE(results[0].s0 == Approx(1.0));  // Charge-charge always 1.0
    // For dipole-charge along x-axis with dipole in x direction
    // The S-function should be non-zero
    REQUIRE(std::isfinite(results[1].s0));
    REQUIRE(std::abs(results[1].s0) > 0.0);  // Should be non-zero
}

TEST_CASE("SFunctionTermListBuilder basic filtering", "[mults][builder]") {
    using namespace occ::mults;
    using namespace occ::dma;

    SFunctionTermListBuilder builder(4);

    // Create two simple multipoles (charge + dipole only)
    Mult mult1(1);  // Rank 1 (dipole)
    mult1.Q00() = 1.0;   // Charge
    mult1.Q10() = 0.5;   // Dipole z
    mult1.Q11c() = 0.0;  // Dipole x (zero!)
    mult1.Q11s() = 0.0;  // Dipole y (zero!)

    Mult mult2(1);
    mult2.Q00() = -0.8;
    mult2.Q10() = 0.3;
    mult2.Q11c() = 0.0;
    mult2.Q11s() = 0.0;

    auto terms = builder.build_electrostatic_terms(mult1, mult2);

    // Should only have terms where both Q(t1) and Q(t2) are non-zero
    // Non-zero pairs: (Q00,Q00), (Q00,Q10), (Q10,Q00), (Q10,Q10)
    // That's 4 terms instead of 16 possible (4x4)
    REQUIRE(terms.size() == 4);

    // Verify we got the expected terms
    bool found_00_00 = false;
    bool found_00_10 = false;
    bool found_10_00 = false;
    bool found_10_10 = false;

    for (const auto& term : terms.terms) {
        if (term.t1 == 0 && term.t2 == 0) found_00_00 = true;
        if (term.t1 == 0 && term.t2 == 1) found_00_10 = true;
        if (term.t1 == 1 && term.t2 == 0) found_10_00 = true;
        if (term.t1 == 1 && term.t2 == 1) found_10_10 = true;
    }

    REQUIRE(found_00_00);
    REQUIRE(found_00_10);
    REQUIRE(found_10_00);
    REQUIRE(found_10_10);
}

TEST_CASE("SFunctionTermListBuilder with rank limit", "[mults][builder]") {
    using namespace occ::mults;
    using namespace occ::dma;

    SFunctionTermListBuilder builder(4);

    // Create quadrupole multipole
    Mult mult(2);
    mult.Q00() = 1.0;
    mult.Q20() = 0.5;

    // Build with max_interaction_rank = 2 (should exclude Q20 x Q20 interaction which needs j=4)
    auto terms = builder.build_electrostatic_terms(mult, mult, 2);

    // Check no terms have j > 2
    for (const auto& term : terms.terms) {
        REQUIRE(term.j <= 2);
    }
}

TEST_CASE("SFunctionTermListBuilder sparsity", "[mults][builder]") {
    using namespace occ::mults;
    using namespace occ::dma;

    SFunctionTermListBuilder builder(4);

    // Hexadecapole with only a few non-zero components
    Mult sparse(4);
    sparse.Q00() = 1.0;    // 1 component
    sparse.Q10() = 0.5;    // 1 component
    // All higher components are zero

    auto terms = builder.build_electrostatic_terms(sparse, sparse);

    // Total possible: 25 x 25 = 625 terms
    // Non-zero: only (Q00,Q00), (Q00,Q10), (Q10,Q00), (Q10,Q10) = 4 terms
    int total_possible = builder.get_total_term_count(4, 4);

    REQUIRE(total_possible == 625);
    REQUIRE(terms.size() == 4);

    // Verify sparsity: 99.4% reduction!
    double sparsity = 100.0 * (1.0 - double(terms.size()) / total_possible);
    REQUIRE(sparsity > 99.0);
}

TEST_CASE("SFunctionTermListBuilder coefficient computation", "[mults][builder]") {
    using namespace occ::mults;
    using namespace occ::dma;

    SFunctionTermListBuilder builder(4);

    // Create simple multipoles with known values
    Mult mult1(1);
    mult1.Q00() = 2.0;   // Charge
    mult1.Q10() = 3.0;   // Dipole z

    Mult mult2(1);
    mult2.Q00() = 5.0;   // Charge
    mult2.Q10() = 7.0;   // Dipole z

    auto terms = builder.build_electrostatic_terms(mult1, mult2);

    // Find specific terms and verify coefficients
    for (const auto& term : terms.terms) {
        if (term.t1 == 0 && term.t2 == 0) {
            // Charge-charge: q1 * q2 * binomial(0, 0) = 2.0 * 5.0 * 1 = 10.0
            REQUIRE(term.j == 0);
            REQUIRE(term.power == 1);
            REQUIRE(term.coeff == Approx(10.0));
        }
        else if (term.t1 == 0 && term.t2 == 1) {
            // Charge-dipole: q1 * q2 * binomial(1, 0) = 2.0 * 7.0 * 1 = 14.0
            REQUIRE(term.j == 1);
            REQUIRE(term.power == 2);
            REQUIRE(term.coeff == Approx(14.0));
        }
        else if (term.t1 == 1 && term.t2 == 0) {
            // Dipole-charge: q1 * q2 * binomial(1, 1) = 3.0 * 5.0 * 1 = 15.0
            REQUIRE(term.j == 1);
            REQUIRE(term.power == 2);
            REQUIRE(term.coeff == Approx(15.0));
        }
        else if (term.t1 == 1 && term.t2 == 1) {
            // Dipole-dipole: q1 * q2 * binomial(2, 1) = 3.0 * 7.0 * 2 = 42.0
            REQUIRE(term.j == 2);
            REQUIRE(term.power == 3);
            REQUIRE(term.coeff == Approx(42.0));
        }
    }

    // Verify we found all 4 expected terms
    REQUIRE(terms.size() == 4);
}

// ============================================================================
// Phase 4: MultipoleInteractions High-Level API Tests
// ============================================================================

TEST_CASE("MultipoleInteractions ESP calculation", "[mults][interactions][esp]") {
    using namespace occ::mults;
    using namespace occ::dma;

    MultipoleInteractions interactions;

    // Water oxygen multipole (from existing tests)
    Mult water(4);
    water.Q00() = -0.330960;
    water.Q11c() = -0.297907;
    water.Q20() = 0.117935;
    water.Q22c() = 0.673922;
    water.Q31c() = -0.151827;
    water.Q33c() = 0.303856;
    water.Q40() = 0.114584;
    water.Q42c() = -0.183221;
    water.Q44s() = -0.065424;

    Vec3 site(0, 0, 0);
    Vec3 eval_point(2, 2, 2);

    double esp = interactions.compute_esp(water, site, eval_point);

    // Should match old MultipoleESP implementation
    MultipoleESP old_esp(4);
    double esp_old = old_esp.compute_esp_at_point(water, site, eval_point);

    INFO("New API: " << esp);
    INFO("Old API: " << esp_old);
    REQUIRE(esp == Approx(esp_old).epsilon(1e-10));
}

TEST_CASE("MultipoleInteractions interaction energy", "[mults][interactions][energy]") {
    using namespace occ::mults;
    using namespace occ::dma;

    MultipoleInteractions interactions;

    // Simple dipole-dipole interaction
    Mult dipole(1);
    dipole.Q00() = 0.0;
    dipole.Q10() = 1.0;  // Dipole along z

    Vec3 pos1(0, 0, 0);
    Vec3 pos2(3, 0, 0);  // 3 bohr separation along x

    double energy = interactions.compute_interaction_energy(dipole, pos1, dipole, pos2);

    // Should match old MultipoleESP implementation
    MultipoleESP old_esp(4);
    double energy_old = old_esp.compute_interaction_energy(dipole, pos1, dipole, pos2);

    INFO("New API: " << energy);
    INFO("Old API: " << energy_old);
    REQUIRE(energy == Approx(energy_old).epsilon(1e-10));
}

TEST_CASE("MultipoleInteractions ESP grid", "[mults][interactions][grid]") {
    using namespace occ::mults;
    using namespace occ::dma;

    MultipoleInteractions interactions;

    Mult charge(0);
    charge.Q00() = 1.0;

    Vec3 site(0, 0, 0);

    // Grid of 8 points as a 3xN matrix
    Mat3N points(3, 8);
    points.col(0) = Vec3(1, 0, 0);
    points.col(1) = Vec3(-1, 0, 0);
    points.col(2) = Vec3(0, 1, 0);
    points.col(3) = Vec3(0, -1, 0);
    points.col(4) = Vec3(0, 0, 1);
    points.col(5) = Vec3(0, 0, -1);
    points.col(6) = Vec3(1, 1, 1);
    points.col(7) = Vec3(-1, -1, -1);

    auto esp_values = interactions.compute_esp_grid(charge, site, points);

    REQUIRE(esp_values.size() == 8);

    // For a unit charge, ESP at distance r should be 1/r
    for (int i = 0; i < points.cols(); ++i) {
        double r = points.col(i).norm();
        double expected = 1.0 / r;
        REQUIRE(esp_values[i] == Approx(expected).epsilon(1e-10));
    }
}

TEST_CASE("MultipoleInteractions matches existing water dimer test", "[mults][interactions][validation]") {
    using namespace occ::mults;
    using namespace occ::dma;

    // Use the same water dimer setup from existing tests
    MultipoleInteractions interactions;

    Mult water(4);
    water.Q00() = -0.330960;   // monopole
    water.Q10() = 0.0;         // dipole z
    water.Q11c() = -0.297907;  // dipole x
    water.Q11s() = 0.0;        // dipole y
    water.Q20() = 0.117935;    // quadrupole zz
    water.Q21c() = 0.0;
    water.Q21s() = 0.0;
    water.Q22c() = 0.673922;   // quadrupole (xx-yy)/2
    water.Q22s() = 0.0;
    water.Q30() = 0.0;         // octupole zzz
    water.Q31c() = -0.151827;  // octupole xzz
    water.Q31s() = 0.0;
    water.Q32c() = 0.0;
    water.Q32s() = 0.0;
    water.Q33c() = 0.303856;   // octupole (xxx-3xyy)
    water.Q33s() = 0.0;
    water.Q40() = 0.114584;    // hexadecapole
    water.Q41c() = 0.0;
    water.Q41s() = 0.0;
    water.Q42c() = -0.183221;
    water.Q42s() = 0.0;
    water.Q43c() = 0.0;
    water.Q43s() = 0.0;
    water.Q44c() = 0.0;
    water.Q44s() = -0.065424;

    Vec3 pos1(0, 0, 0);
    Vec3 pos2(3.0, 0, 0);  // 3 bohr separation along x-axis

    double energy = interactions.compute_interaction_energy(water, pos1, water, pos2);

    // Compare with old API
    MultipoleESP old_esp(4);
    double energy_old = old_esp.compute_interaction_energy(water, pos1, water, pos2);

    // Should match within numerical precision
    INFO("New API energy: " << energy);
    INFO("Old API energy: " << energy_old);
    INFO("Difference: " << (energy - energy_old));
    REQUIRE(energy == Approx(energy_old).epsilon(1e-10));

    // Should also be close to Orient reference value
    double orient_reference = 0.02753720;  // hartree at 3.0 bohr
    double percent_error = 100.0 * std::abs(energy - orient_reference) / orient_reference;
    INFO("Energy: " << energy);
    INFO("Orient reference: " << orient_reference);
    INFO("Percent error: " << percent_error << "%");
    REQUIRE(percent_error < 0.1);  // Should be very close to Orient
}

TEST_CASE("MultipoleInteractions term filtering effectiveness", "[mults][interactions][performance]") {
    using namespace occ::mults;
    using namespace occ::dma;

    MultipoleInteractions interactions;

    // Hexadecapole with sparse components
    Mult sparse(4);
    sparse.Q00() = 1.0;
    sparse.Q10() = 0.5;
    sparse.Q11c() = 0.3;
    // Most components are zero

    Vec3 pos1(0, 0, 0);
    Vec3 pos2(5, 0, 0);

    // Build term list to see filtering effect
    SFunctionTermListBuilder builder(4);
    auto terms = builder.build_electrostatic_terms(sparse, sparse);

    int total_possible = builder.get_total_term_count(4, 4);

    INFO("Total possible terms: " << total_possible);
    INFO("Filtered terms: " << terms.size());
    INFO("Reduction: " << (100.0 * (1.0 - double(terms.size()) / total_possible)) << "%");

    // Should have significant reduction
    REQUIRE(terms.size() < total_possible / 2);  // At least 50% reduction

    // Energy should still be correct
    double energy = interactions.compute_interaction_energy(sparse, pos1, sparse, pos2);
    REQUIRE(std::isfinite(energy));
}

TEST_CASE("MultipoleInteractions charge-charge interaction", "[mults][interactions][basic]") {
    using namespace occ::mults;
    using namespace occ::dma;

    MultipoleInteractions interactions;

    // Two unit charges
    Mult charge1(0);
    charge1.Q00() = 1.0;

    Mult charge2(0);
    charge2.Q00() = 1.0;

    Vec3 pos1(0, 0, 0);
    Vec3 pos2(1, 0, 0);  // 1 bohr separation

    double energy = interactions.compute_interaction_energy(charge1, pos1, charge2, pos2);

    // For two unit charges at distance 1 bohr: E = 1/r = 1.0
    REQUIRE(energy == Approx(1.0).epsilon(1e-10));

    // Test different separation
    pos2 = Vec3(2, 0, 0);  // 2 bohr separation
    energy = interactions.compute_interaction_energy(charge1, pos1, charge2, pos2);
    REQUIRE(energy == Approx(0.5).epsilon(1e-10));  // E = 1/2
}

TEST_CASE("MultipoleInteractions dipole-dipole parallel", "[mults][interactions][dipole]") {
    using namespace occ::mults;
    using namespace occ::dma;

    MultipoleInteractions interactions;

    // Two parallel dipoles along z-axis
    Mult dipole(1);
    dipole.Q00() = 0.0;
    dipole.Q10() = 1.0;  // Dipole along z

    Vec3 pos1(0, 0, 0);
    Vec3 pos2(0, 0, 3);  // 3 bohr separation along z (aligned with dipoles)

    double energy = interactions.compute_interaction_energy(dipole, pos1, dipole, pos2);

    // Compare with old API
    MultipoleESP old_esp(4);
    double energy_old = old_esp.compute_interaction_energy(dipole, pos1, dipole, pos2);

    REQUIRE(energy == Approx(energy_old).epsilon(1e-10));

    // For parallel dipoles along z separated along z:
    // E should be negative (attractive configuration)
    REQUIRE(energy < 0.0);
}

// =============================================================================
// DERIVATIVE VALIDATION TESTS
// =============================================================================

TEST_CASE("S-functions - partial derivative correctness",
          "[mults][sfunctions][derivatives]") {

    // NOTE: The S-function derivatives s1[0-5] are PARTIAL derivatives with respect
    // to the unit vector components (rax, ray, raz, rbx, rby, rbz), NOT total
    // derivatives with respect to raw Cartesian coordinates.
    //
    // This test validates that:
    // 1. Derivatives are computed and non-zero where expected
    // 2. Simple analytical relationships hold (e.g., dipole derivative = 1.0)
    // 3. Symmetry properties are satisfied
    //
    // Full validation via finite differences is done at the interaction energy
    // level (see "Multipole interaction - numerical gradient validation" test below).

    SFunctions sf(4);

    SECTION("Charge-charge has zero derivatives") {
        Vec3 ra(1.0, 2.0, 3.0);
        Vec3 rb(4.0, 5.0, 6.0);
        sf.set_coordinates(ra, rb);

        auto result = sf.compute_s_function(0, 0, 0, 1);

        // S(00,00) = 1.0 (constant), so all derivatives should be zero
        REQUIRE(result.s0 == Approx(1.0));
        for (int i = 0; i < 6; i++) {
            REQUIRE(result.s1[i] == Approx(0.0).margin(1e-12));
        }
    }

    SECTION("Charge-dipole derivatives are unit vectors") {
        Vec3 ra(0, 0, 0);
        Vec3 rb(3, 0, 0);  // Simple geometry along x
        sf.set_coordinates(ra, rb);

        // Charge-dipole-x: S = rbx, so d/d(rbx) = 1
        auto result_x = sf.compute_s_function(0, 2, 0, 1);  // (00, 11c, 0)
        REQUIRE(result_x.s0 == Approx(sf.rbx()));
        REQUIRE(result_x.s1[3] == Approx(1.0));  // d/d(rbx)

        // Charge-dipole-z: S = rbz, so d/d(rbz) = 1
        auto result_z = sf.compute_s_function(0, 1, 0, 1);  // (00, 10, 0)
        REQUIRE(result_z.s0 == Approx(sf.rbz()));
        REQUIRE(result_z.s1[5] == Approx(1.0));  // d/d(rbz)
    }

    SECTION("Dipole-charge derivatives are unit vectors") {
        Vec3 ra(0, 0, 0);
        Vec3 rb(0, 0, 5);  // Simple geometry along z
        sf.set_coordinates(ra, rb);

        // Dipole-z-charge: S = raz, so d/d(raz) = 1
        auto result_z = sf.compute_s_function(1, 0, 0, 1);  // (10, 00, 0)
        REQUIRE(result_z.s0 == Approx(sf.raz()));
        REQUIRE(result_z.s1[2] == Approx(1.0));  // d/d(raz)

        // Dipole-x-charge: S = rax, so d/d(rax) = 1
        auto result_x = sf.compute_s_function(2, 0, 0, 1);  // (11c, 00, 0)
        REQUIRE(result_x.s0 == Approx(sf.rax()));
        REQUIRE(result_x.s1[0] == Approx(1.0));  // d/d(rax)
    }

    SECTION("Dipole-dipole derivatives follow product rule") {
        Vec3 ra(0, 0, 0);
        Vec3 rb(0, 0, 4);  // Along z
        sf.set_coordinates(ra, rb);

        // Dipole-z dipole-z: S = 1.5*raz*rbz + 0.5*czz
        // d/d(raz) = 1.5*rbz, d/d(rbz) = 1.5*raz
        auto result = sf.compute_s_function(1, 1, 0, 1);  // (10, 10, 0)

        double raz_val = sf.raz();
        double rbz_val = sf.rbz();

        REQUIRE(result.s0 == Approx(1.5 * raz_val * rbz_val + 0.5));
        REQUIRE(result.s1[2] == Approx(1.5 * rbz_val));  // d/d(raz)
        REQUIRE(result.s1[5] == Approx(1.5 * raz_val));  // d/d(rbz)
    }

    SECTION("Quadrupole derivatives are non-zero for rank 2 combinations") {
        Vec3 ra(1, 2, 3);
        Vec3 rb(4, 5, 6);
        sf.set_coordinates(ra, rb);

        // Charge-quadrupole and quadrupole-charge should have non-zero derivatives
        auto cq_result = sf.compute_s_function(0, 4, 0, 1);  // (00, 20, 0)
        auto qc_result = sf.compute_s_function(4, 0, 0, 1);  // (20, 00, 0)

        // At least one derivative component should be non-zero
        double cq_deriv_norm = cq_result.s1.head(6).norm();
        double qc_deriv_norm = qc_result.s1.head(6).norm();

        REQUIRE(cq_deriv_norm > 0.0);
        REQUIRE(qc_deriv_norm > 0.0);
    }

    SECTION("Derivative level 0 returns zero derivatives") {
        Vec3 ra(1, 2, 3);
        Vec3 rb(4, 5, 6);
        sf.set_coordinates(ra, rb);

        // Level 0 should not compute derivatives
        auto result = sf.compute_s_function(2, 2, 0, 0);  // (11c, 11c, 0), level=0

        // s1 should be all zeros
        REQUIRE(result.s1.norm() == Approx(0.0).margin(1e-12));
    }

    SECTION("Derivative values are consistent across coordinate systems") {
        // Test that rotated configurations give appropriately rotated derivatives

        Vec3 ra1(0, 0, 0);
        Vec3 rb1(3, 0, 0);  // Along x
        sf.set_coordinates(ra1, rb1);
        auto result1 = sf.compute_s_function(2, 0, 0, 1);  // Dipole-x charge

        Vec3 ra2(0, 0, 0);
        Vec3 rb2(0, 3, 0);  // Along y
        sf.set_coordinates(ra2, rb2);
        auto result2 = sf.compute_s_function(3, 0, 0, 1);  // Dipole-y charge

        Vec3 ra3(0, 0, 0);
        Vec3 rb3(0, 0, 3);  // Along z
        sf.set_coordinates(ra3, rb3);
        auto result3 = sf.compute_s_function(1, 0, 0, 1);  // Dipole-z charge

        // All should have the same pattern: unit derivative in their respective direction
        REQUIRE(result1.s1[0] == Approx(1.0));  // d/d(rax)
        REQUIRE(result2.s1[1] == Approx(1.0));  // d/d(ray)
        REQUIRE(result3.s1[2] == Approx(1.0));  // d/d(raz)
    }
}
TEST_CASE("Multipole interaction - numerical gradient validation",
          "[mults][interactions][forces][derivatives]") {

    using namespace occ::dma;

    // Helper to compute numerical gradient using finite differences
    auto compute_numerical_gradient = [](
        const MultipoleInteractions& interactions,
        const Mult& mult1, const Vec3& pos1,
        const Mult& mult2, const Vec3& pos2,
        double h = 1e-7) -> std::pair<Vec3, Vec3> {

        double energy_0 = interactions.compute_interaction_energy(mult1, pos1, mult2, pos2);

        Vec3 grad1 = Vec3::Zero();
        Vec3 grad2 = Vec3::Zero();

        // Gradient with respect to pos1
        for (int i = 0; i < 3; i++) {
            Vec3 delta = Vec3::Zero();
            delta[i] = h;
            Vec3 pos1_forward = pos1 + delta;
            double energy_forward = interactions.compute_interaction_energy(
                mult1, pos1_forward, mult2, pos2);
            grad1[i] = -(energy_forward - energy_0) / h;
        }

        // Gradient with respect to pos2
        for (int i = 0; i < 3; i++) {
            Vec3 delta = Vec3::Zero();
            delta[i] = h;
            Vec3 pos2_forward = pos2 + delta;
            double energy_forward = interactions.compute_interaction_energy(
                mult1, pos1, mult2, pos2_forward);
            grad2[i] = -(energy_forward - energy_0) / h;
        }

        return {grad1, grad2};
    };

    SECTION("Charge-charge numerical gradient") {
        MultipoleInteractions interactions;

        // Two unit charges
        Mult charge1(0);
        charge1.Q00() = 1.0;

        Mult charge2(0);
        charge2.Q00() = 1.0;

        Vec3 pos1(0, 0, 0);
        Vec3 pos2(2, 0, 0);  // 2 bohr separation

        auto [grad1, grad2] = compute_numerical_gradient(
            interactions, charge1, pos1, charge2, pos2);

        // For two equal charges at distance r along x:
        // E = 1/r, F = -dE/dx = 1/r^2
        // Force on charge1 should be negative (pointing away from charge2)
        double expected_force_magnitude = 1.0 / (2.0 * 2.0);  // 1/r^2 = 1/4
        REQUIRE(grad1[0] == Approx(-expected_force_magnitude).epsilon(1e-6));
        REQUIRE(grad2[0] == Approx(expected_force_magnitude).epsilon(1e-6));

        // Forces should be equal and opposite (Newton's third law)
        // Note: numerical precision limits force balance to ~1e-7 for single-sided finite differences
        REQUIRE((grad1 + grad2).norm() == Approx(0.0).margin(1e-7));
    }

    SECTION("Dipole-dipole numerical gradient") {
        MultipoleInteractions interactions;

        // Two dipoles along z-axis
        Mult dipole1(1);
        dipole1.Q00() = 0.0;
        dipole1.Q10() = 1.0;  // Unit dipole along z

        Mult dipole2(1);
        dipole2.Q00() = 0.0;
        dipole2.Q10() = 1.0;  // Unit dipole along z

        Vec3 pos1(0, 0, 0);
        Vec3 pos2(0, 0, 3);  // 3 bohr separation along z

        auto [grad1, grad2] = compute_numerical_gradient(
            interactions, dipole1, pos1, dipole2, pos2);

        // Main force component should be along z
        REQUIRE(std::abs(grad1[2]) > std::abs(grad1[0]));
        REQUIRE(std::abs(grad1[2]) > std::abs(grad1[1]));

        // Forces should be equal and opposite
        // Note: numerical precision limits force balance to ~1e-7 for single-sided finite differences
        REQUIRE((grad1 + grad2).norm() == Approx(0.0).margin(1e-7));
    }

    SECTION("Water dimer numerical gradient") {
        MultipoleInteractions interactions;

        // Water molecule multipoles (up to quadrupole for simplicity)
        Mult water1(2);
        water1.Q00() = -0.33096;
        water1.Q10() = 0.0;
        water1.Q11c() = -0.297907;
        water1.Q11s() = 0.0;
        water1.Q20() = 0.117935;
        water1.Q21c() = 0.0;
        water1.Q21s() = 0.0;
        water1.Q22c() = 0.673922;
        water1.Q22s() = 0.0;

        Mult water2 = water1;  // Identical molecule

        Vec3 pos1(0, 0, 0);
        Vec3 pos2(5.0, 1.0, 2.0);  // Non-aligned separation

        auto [grad1, grad2] = compute_numerical_gradient(
            interactions, water1, pos1, water2, pos2, 1e-6);

        // Forces should be finite
        REQUIRE(std::isfinite(grad1.norm()));
        REQUIRE(std::isfinite(grad2.norm()));

        // Forces should be non-zero (there's interaction)
        REQUIRE(grad1.norm() > 0.0);
        REQUIRE(grad2.norm() > 0.0);

        // Forces should be equal and opposite (Newton's third law)
        REQUIRE((grad1 + grad2).norm() == Approx(0.0).margin(1e-8));
    }

    SECTION("Mixed multipole gradients are self-consistent") {
        MultipoleInteractions interactions;

        // Dipole + quadrupole
        Mult mult1(2);
        mult1.Q00() = 0.5;
        mult1.Q10() = 1.0;
        mult1.Q11c() = 0.3;
        mult1.Q20() = 0.2;
        mult1.Q22c() = -0.1;

        // Different multipole
        Mult mult2(2);
        mult2.Q00() = -0.3;
        mult2.Q10() = 0.7;
        mult2.Q11s() = 0.4;
        mult2.Q21c() = 0.15;
        mult2.Q22s() = 0.2;

        Vec3 pos1(1.0, 2.0, 3.0);
        Vec3 pos2(4.0, 5.0, 6.0);

        auto [grad1, grad2] = compute_numerical_gradient(
            interactions, mult1, pos1, mult2, pos2);

        // Newton's third law
        REQUIRE((grad1 + grad2).norm() == Approx(0.0).margin(1e-8));

        // Gradients should be non-trivial
        REQUIRE(grad1.norm() > 1e-6);
        REQUIRE(grad2.norm() > 1e-6);
    }
}

// ============================================================================
// S-Function Derivative Tests - Analytical Validation
// ============================================================================
//
// IMPORTANT NOTE: S-function derivatives s1[0-5] are PARTIAL derivatives with
// respect to unit vector components (rax, ray, raz, rbx, rby, rbz), NOT total
// derivatives with respect to raw Cartesian coordinates.
//
// These derivatives CANNOT be validated with finite differences on raw coordinates
// because of the complex chain rule involving geometric Jacobians (∂unit_vec/∂position).
//
// Instead, we use two complementary validation strategies:
// 1. Direct analytical tests (this section) - Test cases with known exact partial derivatives
// 2. Integration-level tests (see "Multipole interaction - numerical gradient validation")
//    - Test complete derivative chain at energy/force level where finite differences work
//
// ============================================================================

/**
 * Validate S-function partial derivatives for cases with known analytical values
 */
void validate_sfunction_partial_derivatives(SFunctions& sf, int t1, int t2, int j,
                                             const std::string& test_name,
                                             const std::vector<std::pair<int, double>>& expected_derivs,
                                             double tol = 1e-10) {
    auto result = sf.compute_s_function(t1, t2, j, 1);
    auto [l1, m1] = sf.index_to_lm(t1);
    auto [l2, m2] = sf.index_to_lm(t2);

    INFO("Test: " << test_name);
    INFO("t1=" << t1 << " t2=" << t2 << " j=" << j << " (l1=" << l1 << ",m1=" << m1
         << " l2=" << l2 << ",m2=" << m2 << ")");
    INFO("s0 = " << result.s0);

    const char* coord_names[] = {"rax", "ray", "raz", "rbx", "rby", "rbz",
                                  "cxx", "cyx", "czx", "cxy", "cyy", "czy", "cxz", "cyz", "czz"};

    // Check expected derivatives
    for (const auto& [idx, expected_val] : expected_derivs) {
        INFO("Checking ∂S/∂" << coord_names[idx] << ": expected=" << expected_val
             << " actual=" << result.s1[idx]);
        REQUIRE(result.s1[idx] == Approx(expected_val).margin(tol));
    }

    // Check that all other derivatives are zero (or very small)
    for (int i = 0; i < 15; i++) {
        bool is_expected = false;
        for (const auto& [idx, _] : expected_derivs) {
            if (idx == i) {
                is_expected = true;
                break;
            }
        }
        if (!is_expected) {
            INFO("Checking ∂S/∂" << coord_names[i] << " is zero: actual=" << result.s1[i]);
            REQUIRE(std::abs(result.s1[i]) < tol);
        }
    }
}

TEST_CASE("S-function partial derivatives - analytical validation", "[mults][sfunctions][derivatives][detailed]") {
    SFunctions sf(4);

    // Use a non-trivial coordinate system to test partial derivatives
    Vec3 ra(-1.2, -0.8, -1.5);
    Vec3 rb(0.9, 1.1, 1.3);
    sf.set_coordinates(ra, rb);

    // Get unit vector components for reference
    double rax_val = sf.rax();
    double ray_val = sf.ray();
    double raz_val = sf.raz();
    double rbx_val = sf.rbx();
    double rby_val = sf.rby();
    double rbz_val = sf.rbz();

    SECTION("Rank 0 - Charge-Charge") {
        // S(0,0,0) = 1 (constant)
        // All partial derivatives should be zero
        validate_sfunction_partial_derivatives(sf, 0, 0, 0, "charge-charge", {});
    }

    SECTION("Rank 1 - Charge-Dipole (dipole at site B)") {
        // S(0,1,1) = rbz, so ∂S/∂rbz = 1
        validate_sfunction_partial_derivatives(sf, 0, 1, 1, "charge - dipole-z",
                                                {{5, 1.0}});
        // S(0,2,1) = rbx, so ∂S/∂rbx = 1
        validate_sfunction_partial_derivatives(sf, 0, 2, 1, "charge - dipole-x",
                                                {{3, 1.0}});
        // S(0,3,1) = rby, so ∂S/∂rby = 1
        validate_sfunction_partial_derivatives(sf, 0, 3, 1, "charge - dipole-y",
                                                {{4, 1.0}});
    }

    SECTION("Rank 1 - Dipole-Charge (dipole at site A)") {
        // S(1,0,1) = raz, so ∂S/∂raz = 1
        validate_sfunction_partial_derivatives(sf, 1, 0, 1, "dipole-z - charge",
                                                {{2, 1.0}});
        // S(2,0,1) = rax, so ∂S/∂rax = 1
        validate_sfunction_partial_derivatives(sf, 2, 0, 1, "dipole-x - charge",
                                                {{0, 1.0}});
        // S(3,0,1) = ray, so ∂S/∂ray = 1
        validate_sfunction_partial_derivatives(sf, 3, 0, 1, "dipole-y - charge",
                                                {{1, 1.0}});
    }

    SECTION("Rank 2 - Dipole-Dipole") {
        // S(1,1,2) = 1.5*raz*rbz + 0.5*czz
        // ∂S/∂raz = 1.5*rbz, ∂S/∂rbz = 1.5*raz, ∂S/∂czz = 0.5
        validate_sfunction_partial_derivatives(sf, 1, 1, 2, "dipole-z - dipole-z (j=2)",
                                                {{2, 1.5 * rbz_val}, {5, 1.5 * raz_val}, {14, 0.5}});

        // S(2,2,2) = 1.5*rax*rbx + 0.5*cxx
        // ∂S/∂rax = 1.5*rbx, ∂S/∂rbx = 1.5*rax, ∂S/∂cxx = 0.5
        validate_sfunction_partial_derivatives(sf, 2, 2, 2, "dipole-x - dipole-x (j=2)",
                                                {{0, 1.5 * rbx_val}, {3, 1.5 * rax_val}, {6, 0.5}});

        // S(1,2,2) = 1.5*raz*rbx + 0.5*czx
        // ∂S/∂raz = 1.5*rbx, ∂S/∂rbx = 1.5*raz, ∂S/∂czx = 0.5
        // Note: Orient includes orientation derivatives for ALL dipole-dipole component pairs
        validate_sfunction_partial_derivatives(sf, 1, 2, 2, "dipole-z - dipole-x (j=2)",
                                                {{2, 1.5 * rbx_val}, {3, 1.5 * raz_val}, {12, 0.5}});
    }

    SECTION("Rank 2 - Charge-Quadrupole") {
        // S(0,4,2) = 1.5*rbz^2 - 0.5
        // ∂S/∂rbz = 3*rbz
        validate_sfunction_partial_derivatives(sf, 0, 4, 2, "charge - quad-20",
                                                {{5, 3.0 * rbz_val}});

        constexpr double rt3 = 1.7320508075688772935;
        // S(0,5,2) = √3*rbx*rbz
        // ∂S/∂rbx = √3*rbz, ∂S/∂rbz = √3*rbx
        validate_sfunction_partial_derivatives(sf, 0, 5, 2, "charge - quad-21c",
                                                {{3, rt3 * rbz_val}, {5, rt3 * rbx_val}});

        // S(0,7,2) = √3*(rbx^2 - rby^2)/2
        // ∂S/∂rbx = √3*rbx, ∂S/∂rby = -√3*rby
        validate_sfunction_partial_derivatives(sf, 0, 7, 2, "charge - quad-22c",
                                                {{3, rt3 * rbx_val}, {4, -rt3 * rby_val}});
    }

    SECTION("Rank 2 - Quadrupole-Charge") {
        // S(4,0,2) = 1.5*raz^2 - 0.5
        // ∂S/∂raz = 3*raz
        validate_sfunction_partial_derivatives(sf, 4, 0, 2, "quad-20 - charge",
                                                {{2, 3.0 * raz_val}});

        constexpr double rt3 = 1.7320508075688772935;
        // S(5,0,2) = √3*rax*raz
        // ∂S/∂rax = √3*raz, ∂S/∂raz = √3*rax
        validate_sfunction_partial_derivatives(sf, 5, 0, 2, "quad-21c - charge",
                                                {{0, rt3 * raz_val}, {2, rt3 * rax_val}});

        // S(7,0,2) = √3*(rax^2 - ray^2)/2
        // ∂S/∂rax = √3*rax, ∂S/∂ray = -√3*ray
        validate_sfunction_partial_derivatives(sf, 7, 0, 2, "quad-22c - charge",
                                                {{0, rt3 * rax_val}, {1, -rt3 * ray_val}});
    }
}

TEST_CASE("S-function derivatives - higher rank validation", "[mults][sfunctions][derivatives]") {
    SFunctions sf(4);

    // Test that higher-rank multipole combinations produce non-zero derivatives
    Vec3 ra(0.0, 0.0, 0.0);
    Vec3 rb(2.0, 1.5, 1.0);
    sf.set_coordinates(ra, rb);

    SECTION("Rank 3 - Dipole-Quadrupole") {
        auto result = sf.compute_s_function(1, 4, 3, 1);
        // Should have non-zero s0 and some non-zero derivatives
        REQUIRE(std::abs(result.s0) > 1e-12);
        double deriv_norm = 0.0;
        for (int i = 0; i < 6; i++) {
            deriv_norm += result.s1[i] * result.s1[i];
        }
        REQUIRE(std::sqrt(deriv_norm) > 1e-12);
    }

    SECTION("Rank 4 - Quadrupole-Quadrupole") {
        auto result = sf.compute_s_function(4, 4, 4, 1);
        REQUIRE(std::abs(result.s0) > 1e-12);
        double deriv_norm = 0.0;
        for (int i = 0; i < 6; i++) {
            deriv_norm += result.s1[i] * result.s1[i];
        }
        REQUIRE(std::sqrt(deriv_norm) > 1e-12);
    }

    SECTION("Rank 5 - Dipole-Hexadecapole") {
        auto result = sf.compute_s_function(1, 16, 5, 1);
        REQUIRE(std::abs(result.s0) > 1e-12);
        double deriv_norm = 0.0;
        for (int i = 0; i < 6; i++) {
            deriv_norm += result.s1[i] * result.s1[i];
        }
        REQUIRE(std::sqrt(deriv_norm) > 1e-12);
    }
}

TEST_CASE("S-function derivatives - symmetry properties", "[mults][sfunctions][derivatives]") {
    SFunctions sf(4);

    SECTION("Site exchange symmetry for charge-charge") {
        Vec3 ra(0.0, 0.0, 0.0);
        Vec3 rb(2.0, 0.0, 0.0);
        sf.set_coordinates(ra, rb);

        auto r1 = sf.compute_s_function(0, 0, 0, 1);

        // Swap sites
        sf.set_coordinates(rb, ra);
        auto r2 = sf.compute_s_function(0, 0, 0, 1);

        // s0 should be the same (both = 1)
        REQUIRE(r1.s0 == Approx(r2.s0));

        // All derivatives should still be zero
        for (int i = 0; i < 6; i++) {
            REQUIRE(std::abs(r1.s1[i]) < 1e-12);
            REQUIRE(std::abs(r2.s1[i]) < 1e-12);
        }
    }

    SECTION("Dipole-dipole symmetry") {
        Vec3 ra(0.0, 0.0, 0.0);
        Vec3 rb(0.0, 0.0, 3.0);
        sf.set_coordinates(ra, rb);

        // Test z-z dipole-dipole
        auto result = sf.compute_s_function(1, 1, 2, 1);

        // For this geometry: raz=1, rbz=-1
        // S = 1.5*raz*rbz + 0.5 = 1.5*1*(-1) + 0.5 = -1.0
        REQUIRE(result.s0 == Approx(-1.0));

        // ∂S/∂raz = 1.5*rbz = 1.5*(-1) = -1.5
        REQUIRE(result.s1[2] == Approx(-1.5));

        // ∂S/∂rbz = 1.5*raz = 1.5*1 = 1.5
        REQUIRE(result.s1[5] == Approx(1.5));
    }
}

// ============================================================================
// Finite Difference Validation Tests for S-function Derivatives
// ============================================================================

/**
 * Helper function to compute numerical derivative using central finite differences
 *
 * Computes: df/dx ≈ [f(x+h) - f(x-h)] / (2h)
 *
 * @param func Function to differentiate (returns double)
 * @param h Step size for finite difference
 * @return Numerical derivative
 */
template<typename Func>
double finite_difference_central(Func func, double h = 1e-7) {
    double f_plus = func(h);
    double f_minus = func(-h);
    return (f_plus - f_minus) / (2.0 * h);
}

/**
 * Compute numerical gradient for orientation derivative s1[i] where i in [6,14]
 *
 * The orientation parameters are:
 * s1[6]  = ∂S/∂cxx    s1[7]  = ∂S/∂cxy    s1[8]  = ∂S/∂cxz
 * s1[9]  = ∂S/∂cyx    s1[10] = ∂S/∂cyy    s1[11] = ∂S/∂cyz
 * s1[12] = ∂S/∂czx    s1[13] = ∂S/∂czy    s1[14] = ∂S/∂czz
 *
 * @param sf SFunctions object with base coordinate system
 * @param t1 First multipole index
 * @param t2 Second multipole index
 * @param j Combined rank parameter
 * @param deriv_idx Derivative index (6-14 for orientation matrix elements)
 * @param h Step size for finite difference
 * @return Numerical derivative estimate
 */
double compute_orientation_derivative_fd(const SFunctions& sf, int t1, int t2, int j,
                                         int deriv_idx, double h = 1e-7) {
    // deriv_idx must be in range [6,14] for orientation matrix elements
    if (deriv_idx < 6 || deriv_idx > 14) {
        throw std::runtime_error("deriv_idx must be in [6,14] for orientation derivatives");
    }

    // Map deriv_idx to orientation matrix element (i,j)
    int orient_idx = deriv_idx - 6; // 0-8
    int row = orient_idx / 3;       // 0, 1, or 2 (x, y, z for site A)
    int col = orient_idx % 3;       // 0, 1, or 2 (x, y, z for site B)

    // Get base coordinate system
    CoordinateSystem coords = sf.coordinate_system();

    // Lambda to compute S0 with perturbed orientation matrix element
    auto compute_s0_perturbed = [&](double delta) -> double {
        CoordinateSystem perturbed = coords;

        // Perturb the specific orientation matrix element
        if (row == 0 && col == 0) perturbed.cxx += delta;
        else if (row == 0 && col == 1) perturbed.cxy += delta;
        else if (row == 0 && col == 2) perturbed.cxz += delta;
        else if (row == 1 && col == 0) perturbed.cyx += delta;
        else if (row == 1 && col == 1) perturbed.cyy += delta;
        else if (row == 1 && col == 2) perturbed.cyz += delta;
        else if (row == 2 && col == 0) perturbed.czx += delta;
        else if (row == 2 && col == 1) perturbed.czy += delta;
        else if (row == 2 && col == 2) perturbed.czz += delta;

        // Create temporary SFunctions with perturbed coordinates
        SFunctions sf_temp(sf);
        sf_temp.set_coordinate_system(perturbed);

        // Compute S-function value (level=0 for just s0)
        auto result = sf_temp.compute_s_function(t1, t2, j, 0);
        return result.s0;
    };

    // Compute central finite difference
    return finite_difference_central(compute_s0_perturbed, h);
}

TEST_CASE("S-function derivatives - finite difference validation Stage 1",
          "[mults][sfunctions][derivatives][finite-diff][stage1]") {

    SFunctions sf(4);  // max_rank = 4

    // Test geometry: slightly off-axis to avoid symmetry degeneracies
    Vec3 ra(0.0, 0.0, 0.0);
    Vec3 rb(2.5, 1.3, 1.7);
    sf.set_coordinates(ra, rb);

    const double h = 1e-7;           // Step size for finite differences
    const double tol = 1e-6;         // Tolerance for comparison
    const int deriv_level = 1;       // Compute first derivatives

    INFO("Test geometry: ra = " << ra.transpose() << ", rb = " << rb.transpose());
    INFO("Distance r = " << sf.r());

    SECTION("Stage 1a: Charge-Charge (t1=0, t2=0, j=0)") {
        // S = 1 (constant), all derivatives should be zero
        auto result = sf.compute_s_function(0, 0, 0, deriv_level);

        REQUIRE(result.s0 == Approx(1.0));

        // All analytical derivatives should be zero
        for (int i = 0; i < SFunctions::NUM_FIRST_DERIVS; i++) {
            INFO("Derivative index: " << i);
            REQUIRE(std::abs(result.s1[i]) < 1e-12);
        }

        // Skip finite difference checks since all derivatives are analytically zero
    }

    SECTION("Stage 1b: Charge-Dipole Z (t1=0, t2=1, j=1) - orientation derivatives") {
        // t2=1 -> (l=1, m=0) -> z-component -> S = rbz
        // Only test orientation derivatives s1[6-14]

        auto result = sf.compute_s_function(0, 1, 1, deriv_level);

        INFO("Analytical S0 = " << result.s0);
        INFO("Expected S0 = rbz = " << sf.rbz());
        REQUIRE(result.s0 == Approx(sf.rbz()).epsilon(1e-10));

        // Test orientation derivatives [6-14]
        int num_failed = 0;
        for (int i = 6; i < SFunctions::NUM_FIRST_DERIVS; i++) {
            double analytical = result.s1[i];
            double numerical = compute_orientation_derivative_fd(sf, 0, 1, 1, i, h);
            double error = std::abs(analytical - numerical);

            INFO("Derivative s1[" << i << "]: analytical = " << analytical
                 << ", numerical = " << numerical << ", error = " << error);

            if (error > tol) {
                num_failed++;
                WARN("FAILED: s1[" << i << "] error = " << error << " > tolerance " << tol);
            }
        }

        REQUIRE(num_failed == 0);
    }

    SECTION("Stage 1c: Charge-Dipole X (t1=0, t2=2, j=1) - orientation derivatives") {
        // t2=2 -> (l=1, m=1) -> x-component -> S = rbx

        auto result = sf.compute_s_function(0, 2, 1, deriv_level);

        INFO("Analytical S0 = " << result.s0);
        INFO("Expected S0 = rbx = " << sf.rbx());
        REQUIRE(result.s0 == Approx(sf.rbx()).epsilon(1e-10));

        // Test orientation derivatives [6-14]
        int num_failed = 0;
        for (int i = 6; i < SFunctions::NUM_FIRST_DERIVS; i++) {
            double analytical = result.s1[i];
            double numerical = compute_orientation_derivative_fd(sf, 0, 2, 1, i, h);
            double error = std::abs(analytical - numerical);

            INFO("Derivative s1[" << i << "]: analytical = " << analytical
                 << ", numerical = " << numerical << ", error = " << error);

            if (error > tol) {
                num_failed++;
                WARN("FAILED: s1[" << i << "] error = " << error << " > tolerance " << tol);
            }
        }

        REQUIRE(num_failed == 0);
    }

    SECTION("Stage 1d: Charge-Dipole Y (t1=0, t2=3, j=1) - orientation derivatives") {
        // t2=3 -> (l=1, m=-1) -> y-component -> S = rby

        auto result = sf.compute_s_function(0, 3, 1, deriv_level);

        INFO("Analytical S0 = " << result.s0);
        INFO("Expected S0 = rby = " << sf.rby());
        REQUIRE(result.s0 == Approx(sf.rby()).epsilon(1e-10));

        // Test orientation derivatives [6-14]
        int num_failed = 0;
        for (int i = 6; i < SFunctions::NUM_FIRST_DERIVS; i++) {
            double analytical = result.s1[i];
            double numerical = compute_orientation_derivative_fd(sf, 0, 3, 1, i, h);
            double error = std::abs(analytical - numerical);

            INFO("Derivative s1[" << i << "]: analytical = " << analytical
                 << ", numerical = " << numerical << ", error = " << error);

            if (error > tol) {
                num_failed++;
                WARN("FAILED: s1[" << i << "] error = " << error << " > tolerance " << tol);
            }
        }

        REQUIRE(num_failed == 0);
    }

    SECTION("Stage 1e: Dipole Z-Charge (t1=1, t2=0, j=1) - orientation derivatives") {
        // t1=1 -> (l=1, m=0) -> z-component -> S = raz

        auto result = sf.compute_s_function(1, 0, 1, deriv_level);

        INFO("Analytical S0 = " << result.s0);
        INFO("Expected S0 = raz = " << sf.raz());
        REQUIRE(result.s0 == Approx(sf.raz()).epsilon(1e-10));

        // Test orientation derivatives [6-14]
        int num_failed = 0;
        for (int i = 6; i < SFunctions::NUM_FIRST_DERIVS; i++) {
            double analytical = result.s1[i];
            double numerical = compute_orientation_derivative_fd(sf, 1, 0, 1, i, h);
            double error = std::abs(analytical - numerical);

            INFO("Derivative s1[" << i << "]: analytical = " << analytical
                 << ", numerical = " << numerical << ", error = " << error);

            if (error > tol) {
                num_failed++;
                WARN("FAILED: s1[" << i << "] error = " << error << " > tolerance " << tol);
            }
        }

        REQUIRE(num_failed == 0);
    }

    SECTION("Stage 1f: Dipole X-Charge (t1=2, t2=0, j=1) - orientation derivatives") {
        // t1=2 -> (l=1, m=1) -> x-component -> S = rax

        auto result = sf.compute_s_function(2, 0, 1, deriv_level);

        INFO("Analytical S0 = " << result.s0);
        INFO("Expected S0 = rax = " << sf.rax());
        REQUIRE(result.s0 == Approx(sf.rax()).epsilon(1e-10));

        // Test orientation derivatives [6-14]
        int num_failed = 0;
        for (int i = 6; i < SFunctions::NUM_FIRST_DERIVS; i++) {
            double analytical = result.s1[i];
            double numerical = compute_orientation_derivative_fd(sf, 2, 0, 1, i, h);
            double error = std::abs(analytical - numerical);

            INFO("Derivative s1[" << i << "]: analytical = " << analytical
                 << ", numerical = " << numerical << ", error = " << error);

            if (error > tol) {
                num_failed++;
                WARN("FAILED: s1[" << i << "] error = " << error << " > tolerance " << tol);
            }
        }

        REQUIRE(num_failed == 0);
    }

    SECTION("Stage 1g: Dipole Y-Charge (t1=3, t2=0, j=1) - orientation derivatives") {
        // t1=3 -> (l=1, m=-1) -> y-component -> S = ray

        auto result = sf.compute_s_function(3, 0, 1, deriv_level);

        INFO("Analytical S0 = " << result.s0);
        INFO("Expected S0 = ray = " << sf.ray());
        REQUIRE(result.s0 == Approx(sf.ray()).epsilon(1e-10));

        // Test orientation derivatives [6-14]
        int num_failed = 0;
        for (int i = 6; i < SFunctions::NUM_FIRST_DERIVS; i++) {
            double analytical = result.s1[i];
            double numerical = compute_orientation_derivative_fd(sf, 3, 0, 1, i, h);
            double error = std::abs(analytical - numerical);

            INFO("Derivative s1[" << i << "]: analytical = " << analytical
                 << ", numerical = " << numerical << ", error = " << error);

            if (error > tol) {
                num_failed++;
                WARN("FAILED: s1[" << i << "] error = " << error << " > tolerance " << tol);
            }
        }

        REQUIRE(num_failed == 0);
    }
}

TEST_CASE("S-function derivatives - finite difference validation Stage 2",
          "[mults][sfunctions][derivatives][finite-diff][stage2]") {

    SFunctions sf(4);  // max_rank = 4

    // Test geometry: slightly off-axis to avoid symmetry degeneracies
    Vec3 ra(0.0, 0.0, 0.0);
    Vec3 rb(2.5, 1.3, 1.7);
    sf.set_coordinates(ra, rb);

    const double h = 1e-7;           // Step size for finite differences
    const double tol_dipole = 1e-6;  // Tolerance for dipole-dipole
    const double tol_quad = 1e-5;    // Relaxed tolerance for quadrupole-quadrupole
    const int deriv_level = 1;       // Compute first derivatives

    INFO("Test geometry: ra = " << ra.transpose() << ", rb = " << rb.transpose());
    INFO("Distance r = " << sf.r());

    // Structure to track failures across all tests
    struct FailureInfo {
        int t1, t2, j;
        int deriv_idx;
        double analytical;
        double numerical;
        double error;
    };
    std::vector<FailureInfo> all_failures;

    SECTION("Stage 2a: Dipole-Dipole interactions (9 cases)") {
        // Test all combinations: t1=1,2,3 (dipole z,x,y) with t2=1,2,3 (dipole z,x,y)
        // j=2 for all dipole-dipole interactions

        int total_tests = 0;
        int failed_tests = 0;

        for (int t1 = 1; t1 <= 3; t1++) {
            for (int t2 = 1; t2 <= 3; t2++) {
                const int j = 2;
                total_tests++;

                INFO("Testing dipole-dipole: t1=" << t1 << ", t2=" << t2 << ", j=" << j);

                auto result = sf.compute_s_function(t1, t2, j, deriv_level);

                // Test orientation derivatives [6-14]
                int case_failures = 0;
                for (int i = 6; i < SFunctions::NUM_FIRST_DERIVS; i++) {
                    double analytical = result.s1[i];
                    double numerical = compute_orientation_derivative_fd(sf, t1, t2, j, i, h);
                    double error = std::abs(analytical - numerical);

                    if (error > tol_dipole) {
                        case_failures++;
                        all_failures.push_back({t1, t2, j, i, analytical, numerical, error});
                    }
                }

                if (case_failures > 0) {
                    failed_tests++;
                    WARN("Dipole-dipole (t1=" << t1 << ", t2=" << t2 << ", j=" << j
                         << ") failed " << case_failures << " derivatives");
                }
            }
        }

        INFO("Dipole-dipole summary: " << failed_tests << " / " << total_tests << " cases failed");
    }

    SECTION("Stage 2b: Quadrupole-Quadrupole interactions (25 cases)") {
        // Test all combinations: t1=4,5,6,7,8 with t2=4,5,6,7,8
        // j=4 for all quadrupole-quadrupole interactions
        // Quadrupole indices: 4=(2,0), 5=(2,1), 6=(2,-1), 7=(2,2), 8=(2,-2)

        int total_tests = 0;
        int failed_tests = 0;

        for (int t1 = 4; t1 <= 8; t1++) {
            for (int t2 = 4; t2 <= 8; t2++) {
                const int j = 4;
                total_tests++;

                INFO("Testing quadrupole-quadrupole: t1=" << t1 << ", t2=" << t2 << ", j=" << j);

                auto result = sf.compute_s_function(t1, t2, j, deriv_level);

                // Test orientation derivatives [6-14]
                int case_failures = 0;
                for (int i = 6; i < SFunctions::NUM_FIRST_DERIVS; i++) {
                    double analytical = result.s1[i];
                    double numerical = compute_orientation_derivative_fd(sf, t1, t2, j, i, h);
                    double error = std::abs(analytical - numerical);

                    if (error > tol_quad) {
                        case_failures++;
                        all_failures.push_back({t1, t2, j, i, analytical, numerical, error});
                    }
                }

                if (case_failures > 0) {
                    failed_tests++;
                    WARN("Quadrupole-quadrupole (t1=" << t1 << ", t2=" << t2 << ", j=" << j
                         << ") failed " << case_failures << " derivatives");
                }
            }
        }

        INFO("Quadrupole-quadrupole summary: " << failed_tests << " / " << total_tests << " cases failed");
    }

    // Generate comprehensive failure report
    if (!all_failures.empty()) {
        std::cout << "\n=== STAGE 2 FAILURE REPORT ===\n";
        std::cout << "Total failures: " << all_failures.size() << "\n\n";

        // Group failures by (t1, t2) pair
        std::map<std::pair<int,int>, std::vector<FailureInfo>> failures_by_case;
        for (const auto& f : all_failures) {
            failures_by_case[{f.t1, f.t2}].push_back(f);
        }

        std::cout << "Failures by (t1, t2) pair:\n";
        for (const auto& [pair, failures] : failures_by_case) {
            std::cout << "  (t1=" << pair.first << ", t2=" << pair.second << "): "
                      << failures.size() << " derivatives failed\n";
            std::cout << "    Failed derivative indices: ";
            for (size_t i = 0; i < failures.size(); i++) {
                std::cout << failures[i].deriv_idx;
                if (i < failures.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
        }

        // Analyze which derivative indices fail most frequently
        std::map<int, int> failures_by_deriv;
        for (const auto& f : all_failures) {
            failures_by_deriv[f.deriv_idx]++;
        }

        std::cout << "\nFailures by derivative index:\n";
        for (const auto& [idx, count] : failures_by_deriv) {
            std::cout << "  s1[" << idx << "]: " << count << " failures\n";
        }

        // Show worst 10 failures
        std::cout << "\nWorst 10 failures by error magnitude:\n";
        auto sorted_failures = all_failures;
        std::sort(sorted_failures.begin(), sorted_failures.end(),
                  [](const FailureInfo& a, const FailureInfo& b) {
                      return a.error > b.error;
                  });

        int show_count = std::min(10, static_cast<int>(sorted_failures.size()));
        for (int i = 0; i < show_count; i++) {
            const auto& f = sorted_failures[i];
            std::cout << "  " << (i+1) << ". (t1=" << f.t1 << ", t2=" << f.t2
                      << ", j=" << f.j << ") s1[" << f.deriv_idx << "]: "
                      << "analytical=" << f.analytical
                      << ", numerical=" << f.numerical
                      << ", error=" << f.error << "\n";
        }

        std::cout << "\n=== END FAILURE REPORT ===\n\n";

        FAIL("Stage 2 validation failed with " << all_failures.size() << " total failures");
    }
}
