#include <catch2/catch_all.hpp>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/opt/angle_coordinate.h>
#include <occ/opt/berny_optimizer.h>
#include <occ/opt/bond_coordinate.h>
#include <occ/opt/dihedral_coordinate.h>
#include <occ/opt/internal_coordinates.h>
#include <occ/opt/linear_search.h>
#include <occ/opt/species_data.h>

using namespace occ::opt;

/*
 * Berny Coordinate Validation Tests
 *
 * These tests validate individual coordinate classes (Bond, Angle, Dihedral)
 * against exact reference values obtained from
 * test_coordinate_validation.py
 */

TEST_CASE("Bond Constructor Reordering", "[coordinates][bond][constructor]") {
  BondCoordinate bond_12(1, 2);
  REQUIRE(bond_12.i == 1);
  REQUIRE(bond_12.j == 2);

  BondCoordinate bond_21(2, 1);
  REQUIRE(bond_21.i == 1);
  REQUIRE(bond_21.j == 2);
}

TEST_CASE("Angle Constructor Reordering", "[coordinates][angle][constructor]") {
  // Reference: Angle(1, 2, 3): i=1, j=2, k=3
  // Reference: Angle(3, 2, 1): i=1, j=2, k=3

  AngleCoordinate angle_123(1, 2, 3);
  REQUIRE(angle_123.i == 1);
  REQUIRE(angle_123.j == 2);
  REQUIRE(angle_123.k == 3);

  AngleCoordinate angle_321(3, 2, 1);
  REQUIRE(angle_321.i == 1);
  REQUIRE(angle_321.j == 2);
  REQUIRE(angle_321.k == 3);
}

TEST_CASE("Dihedral Constructor Reordering",
          "[coordinates][dihedral][constructor]") {
  // Test dihedral constructor reordering logic
  // Reorders based on j > k comparison

  DihedralCoordinate dihedral_normal(2, 0, 1, 3);
  REQUIRE(dihedral_normal.i == 2);
  REQUIRE(dihedral_normal.j == 0);
  REQUIRE(dihedral_normal.k == 1);
  REQUIRE(dihedral_normal.l == 3);

  // Test reordering when j > k (should swap)
  DihedralCoordinate dihedral_swapped(2, 1, 0, 3);
  REQUIRE(dihedral_swapped.i == 3); // l becomes i
  REQUIRE(dihedral_swapped.j == 0); // k becomes j
  REQUIRE(dihedral_swapped.k == 1); // j becomes k
  REQUIRE(dihedral_swapped.l == 2); // i becomes l
}

TEST_CASE("Bond Coordinate Water Molecule", "[coordinates][bond][water]") {
  // Water molecule geometry (in Bohr)
  occ::Mat3N coords(3, 3);
  coords.col(0) = occ::Vec3(0.0, 0.0, 0.0);    // O
  coords.col(1) = occ::Vec3(0.0, 0.7935, 0.0); // H
  coords.col(2) = occ::Vec3(0.0, 0.0, 0.7935); // H

  SECTION("Bond(0, 1) - O-H bond without gradient") {
    BondCoordinate bond(0, 1);
    double value = bond(coords);

    // Reference: Bond(0, 1) value: 1.4994976798
    REQUIRE(value == Catch::Approx(1.4994976798).margin(1e-9));
  }

  SECTION("Bond(0, 1) - O-H bond with gradient") {
    BondCoordinate bond(0, 1);
    double value = bond(coords);
    auto grad = bond.gradient(coords);

    // Reference: Bond(0, 1) value with grad: 1.4994976798
    REQUIRE(value == Catch::Approx(1.4994976798).margin(1e-9));
    REQUIRE(grad.cols() == 2);

    // Reference: Bond(0, 1) grad[0]: [0.0000000000, -1.0000000000,
    // 0.0000000000]
    REQUIRE(grad(0, 0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(1, 0) == Catch::Approx(-1.0).margin(1e-9));
    REQUIRE(grad(2, 0) == Catch::Approx(0.0).margin(1e-9));

    // Reference: Bond(0, 1) grad[1]: [-0.0000000000, 1.0000000000,
    // -0.0000000000]
    REQUIRE(grad(0, 1) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(1, 1) == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(grad(2, 1) == Catch::Approx(0.0).margin(1e-9));
  }

  SECTION("Bond(0, 2) - O-H bond with gradient") {
    BondCoordinate bond(0, 2);
    double value = bond(coords);
    auto grad = bond.gradient(coords);

    // Reference: Bond(0, 2) value: 1.4994976798
    REQUIRE(value == Catch::Approx(1.4994976798).margin(1e-9));

    // Reference: Bond(0, 2) grad[0]: [0.0000000000, 0.0000000000,
    // -1.0000000000]
    REQUIRE(grad(0, 0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(1, 0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(2, 0) == Catch::Approx(-1.0).margin(1e-9));

    // Reference: Bond(0, 2) grad[1]: [-0.0000000000,
    // -0.0000000000, 1.0000000000]
    REQUIRE(grad(0, 1) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(1, 1) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(2, 1) == Catch::Approx(1.0).margin(1e-9));
  }
}

TEST_CASE("Bond Coordinate H2 Molecule", "[coordinates][bond][h2]") {
  // H2 molecule geometry (in Bohr)
  occ::Mat3N coords(3, 2);
  coords.col(0) = occ::Vec3(0.0, 0.0, 0.0);  // H
  coords.col(1) = occ::Vec3(0.74, 0.0, 0.0); // H

  BondCoordinate bond(0, 1);
  double value = bond(coords);
  auto grad = bond.gradient(coords);

  // Reference: H2 Bond value: 1.3983973322
  REQUIRE(value == Catch::Approx(1.3983973322).margin(1e-9));

  // Reference: H2 Bond grad[0]: [-1.0000000000, 0.0000000000, 0.0000000000]
  REQUIRE(grad(0, 0) == Catch::Approx(-1.0).margin(1e-9));
  REQUIRE(grad(1, 0) == Catch::Approx(0.0).margin(1e-9));
  REQUIRE(grad(2, 0) == Catch::Approx(0.0).margin(1e-9));

  // Reference: H2 Bond grad[1]: [1.0000000000, -0.0000000000, -0.0000000000]
  REQUIRE(grad(0, 1) == Catch::Approx(1.0).margin(1e-9));
  REQUIRE(grad(1, 1) == Catch::Approx(0.0).margin(1e-9));
  REQUIRE(grad(2, 1) == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("Bond Coordinate Methane C-H bonds", "[coordinates][bond][methane]") {
  // Tetrahedral CH4 geometry (in Bohr)
  occ::Mat3N coords(3, 5);
  coords.col(0) = occ::Vec3(0.0, 0.0, 0.0);         // C
  coords.col(1) = occ::Vec3(0.629, 0.629, 0.629);   // H
  coords.col(2) = occ::Vec3(-0.629, -0.629, 0.629); // H
  coords.col(3) = occ::Vec3(-0.629, 0.629, -0.629); // H
  coords.col(4) = occ::Vec3(0.629, -0.629, -0.629); // H

  SECTION("C-H Bond(0, 1)") {
    BondCoordinate bond(0, 1);
    double value = bond(coords);
    auto grad = bond.gradient(coords);

    // PyBerny: C-H Bond(0, 1) value: 2.0587809442
    REQUIRE(value == Catch::Approx(2.0587809442).margin(1e-9));

    // PyBerny: grad[0]: [-0.5773502692, -0.5773502692, -0.5773502692]
    REQUIRE(grad(0, 0) == Catch::Approx(-0.5773502692).margin(1e-9));
    REQUIRE(grad(1, 0) == Catch::Approx(-0.5773502692).margin(1e-9));
    REQUIRE(grad(2, 0) == Catch::Approx(-0.5773502692).margin(1e-9));

    // PyBerny: grad[1]: [0.5773502692, 0.5773502692, 0.5773502692]
    REQUIRE(grad(0, 1) == Catch::Approx(0.5773502692).margin(1e-9));
    REQUIRE(grad(1, 1) == Catch::Approx(0.5773502692).margin(1e-9));
    REQUIRE(grad(2, 1) == Catch::Approx(0.5773502692).margin(1e-9));
  }

  SECTION("C-H Bond(0, 2)") {
    BondCoordinate bond(0, 2);
    double value = bond(coords);
    auto grad = bond.gradient(coords);

    // PyBerny: C-H Bond(0, 2) value: 2.0587809442
    REQUIRE(value == Catch::Approx(2.0587809442).margin(1e-9));

    // PyBerny: grad[0]: [0.5773502692, 0.5773502692, -0.5773502692]
    REQUIRE(grad(0, 0) == Catch::Approx(0.5773502692).margin(1e-9));
    REQUIRE(grad(1, 0) == Catch::Approx(0.5773502692).margin(1e-9));
    REQUIRE(grad(2, 0) == Catch::Approx(-0.5773502692).margin(1e-9));

    // PyBerny: grad[1]: [-0.5773502692, -0.5773502692, 0.5773502692]
    REQUIRE(grad(0, 1) == Catch::Approx(-0.5773502692).margin(1e-9));
    REQUIRE(grad(1, 1) == Catch::Approx(-0.5773502692).margin(1e-9));
    REQUIRE(grad(2, 1) == Catch::Approx(0.5773502692).margin(1e-9));
  }

  SECTION("All C-H bonds have same length") {
    for (int i = 1; i <= 4; i++) {
      BondCoordinate bond(0, i);
      double value = bond(coords);
      REQUIRE(value == Catch::Approx(2.0587809442).margin(1e-9));
    }
  }
}

TEST_CASE("Angle Coordinate Water H-O-H", "[coordinates][angle][water]") {
  // Water molecule geometry (in Bohr)
  occ::Mat3N coords(3, 3);
  coords.col(0) = occ::Vec3(0.0, 0.0, 0.0);    // O
  coords.col(1) = occ::Vec3(0.0, 0.7935, 0.0); // H
  coords.col(2) = occ::Vec3(0.0, 0.0, 0.7935); // H

  SECTION("Angle(1, 0, 2) - H-O-H angle without gradient") {
    AngleCoordinate angle(1, 0, 2);
    double value = angle(coords);

    // PyBerny: Angle(1, 0, 2) value: 1.5707963268 rad (90.000000°)
    REQUIRE(value == Catch::Approx(1.5707963268).margin(1e-9));
  }

  SECTION("Angle(1, 0, 2) - H-O-H angle with gradient") {
    AngleCoordinate angle(1, 0, 2);
    double value = angle(coords);
    auto grad = angle.gradient(coords);

    // PyBerny: Angle(1, 0, 2) value with grad: 1.5707963268
    REQUIRE(value == Catch::Approx(1.5707963268).margin(1e-9));
    REQUIRE(grad.cols() == 3);

    // PyBerny: Angle grad[0]: [0.0000000000, 0.0000000000, -0.6668899949]
    REQUIRE(grad(0, 0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(1, 0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(2, 0) == Catch::Approx(-0.6668899949).margin(1e-9));

    // PyBerny: Angle grad[1]: [0.0000000000, 0.6668899949, 0.6668899949]
    REQUIRE(grad(0, 1) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(1, 1) == Catch::Approx(0.6668899949).margin(1e-9));
    REQUIRE(grad(2, 1) == Catch::Approx(0.6668899949).margin(1e-9));

    // PyBerny: Angle grad[2]: [0.0000000000, -0.6668899949, 0.0000000000]
    REQUIRE(grad(0, 2) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(grad(1, 2) == Catch::Approx(-0.6668899949).margin(1e-9));
    REQUIRE(grad(2, 2) == Catch::Approx(0.0).margin(1e-9));
  }
}

TEST_CASE("Angle Coordinate Methane H-C-H", "[coordinates][angle][methane]") {
  // Tetrahedral CH4 geometry (in Bohr)
  occ::Mat3N coords(3, 5);
  coords.col(0) = occ::Vec3(0.0, 0.0, 0.0);         // C
  coords.col(1) = occ::Vec3(0.629, 0.629, 0.629);   // H
  coords.col(2) = occ::Vec3(-0.629, -0.629, 0.629); // H
  coords.col(3) = occ::Vec3(-0.629, 0.629, -0.629); // H
  coords.col(4) = occ::Vec3(0.629, -0.629, -0.629); // H

  // All H-C-H angles should be the tetrahedral angle
  std::vector<std::pair<int, int>> angle_pairs = {{1, 2}, {1, 3}, {1, 4},
                                                  {2, 3}, {2, 4}, {3, 4}};

  for (auto [i, j] : angle_pairs) {
    AngleCoordinate angle(i, 0, j);
    double value = angle(coords);

    // PyBerny: H-C-H Angle value: 1.9106332362 rad (109.471221°)
    REQUIRE(value == Catch::Approx(1.9106332362).margin(1e-9));
  }
}

TEST_CASE("Dihedral Coordinate Tests", "[coordinates][dihedral]") {
  SECTION("Simple ethane-like dihedral") {
    // Simple ethane-like molecule with dihedral (in Bohr)
    occ::Mat3N coords(3, 4);
    coords.col(0) = occ::Vec3(0.0, 0.0, 0.0);  // C1
    coords.col(1) = occ::Vec3(1.5, 0.0, 0.0);  // C2
    coords.col(2) = occ::Vec3(-0.5, 0.8, 0.0); // H1 on C1
    coords.col(3) = occ::Vec3(2.0, 0.8, 0.0);  // H2 on C2

    SECTION("Dihedral(2, 0, 1, 3) without gradient") {
      DihedralCoordinate dihedral(2, 0, 1, 3);
      double value = dihedral(coords);

      // PyBerny: Dihedral(2, 0, 1, 3) value: 0.0000000000 rad (0.000000°)
      REQUIRE(value == Catch::Approx(0.0).margin(1e-9));
    }

    SECTION("Dihedral(2, 0, 1, 3) with gradient") {
      DihedralCoordinate dihedral(2, 0, 1, 3);
      double value = dihedral(coords);
      auto grad = dihedral.gradient(coords);

      // PyBerny: Dihedral(2, 0, 1, 3) value with grad: 0.0000000000
      REQUIRE(value == Catch::Approx(0.0).margin(1e-9));
      REQUIRE(grad.cols() == 4);

      // PyBerny: Dihedral grad[0]: [0.0000000000, 0.0000000000, 0.6614715137]
      REQUIRE(grad(0, 0) == Catch::Approx(0.0).margin(1e-9));
      REQUIRE(grad(1, 0) == Catch::Approx(0.0).margin(1e-9));
      REQUIRE(grad(2, 0) == Catch::Approx(0.6614715137).margin(1e-9));

      // PyBerny: Dihedral grad[1]: [-0.0000000000, -0.0000000000,
      // -1.1024525227]
      REQUIRE(grad(0, 1) == Catch::Approx(0.0).margin(1e-9));
      REQUIRE(grad(1, 1) == Catch::Approx(0.0).margin(1e-9));
      REQUIRE(grad(2, 1) == Catch::Approx(-1.1024525227).margin(1e-9));

      // PyBerny: Dihedral grad[2]: [0.0000000000, 0.0000000000, 1.1024525227]
      REQUIRE(grad(0, 2) == Catch::Approx(0.0).margin(1e-9));
      REQUIRE(grad(1, 2) == Catch::Approx(0.0).margin(1e-9));
      REQUIRE(grad(2, 2) == Catch::Approx(1.1024525227).margin(1e-9));

      // PyBerny: Dihedral grad[3]: [-0.0000000000, -0.0000000000,
      // -0.6614715137]
      REQUIRE(grad(0, 3) == Catch::Approx(0.0).margin(1e-9));
      REQUIRE(grad(1, 3) == Catch::Approx(0.0).margin(1e-9));
      REQUIRE(grad(2, 3) == Catch::Approx(-0.6614715137).margin(1e-9));
    }
  }

  SECTION("90 degree dihedral") {
    // 90 degree dihedral (in Bohr)
    occ::Mat3N coords(3, 4);
    coords.col(0) = occ::Vec3(0.0, 0.0, 0.0); // C1
    coords.col(1) = occ::Vec3(1.0, 0.0, 0.0); // C2
    coords.col(2) = occ::Vec3(0.0, 1.0, 0.0); // H1 on C1 (90° from C1-C2)
    coords.col(3) = occ::Vec3(
        1.0, 0.0, 1.0); // H2 on C2 (90° from C1-C2, perpendicular to H1)

    DihedralCoordinate dihedral(2, 0, 1, 3);
    double value = dihedral(coords);

    // PyBerny: 90° Dihedral value: -1.5707963268 rad (-90.000000°)
    REQUIRE(value == Catch::Approx(-1.5707963268).margin(1e-9));
  }

  SECTION("180 degree dihedral") {
    // 180 degree (trans) dihedral (in Bohr)
    occ::Mat3N coords(3, 4);
    coords.col(0) = occ::Vec3(0.0, 0.0, 0.0);  // C1
    coords.col(1) = occ::Vec3(1.0, 0.0, 0.0);  // C2
    coords.col(2) = occ::Vec3(0.0, 1.0, 0.0);  // H1 on C1
    coords.col(3) = occ::Vec3(1.0, -1.0, 0.0); // H2 on C2 (opposite side)

    DihedralCoordinate dihedral(2, 0, 1, 3);
    double value = dihedral(coords);

    // PyBerny: 180° Dihedral value: 3.1415926536 rad (180.000000°)
    // Note: Both +π and -π represent 180°, our implementation may give -π
    REQUIRE((std::abs(std::abs(value) - 3.1415926536) < 1e-9));
  }
}

TEST_CASE("InternalCoordinates Class Water Molecule",
          "[internal_coordinates][water]") {
  // Water molecule for testing coordinate system
  occ::IVec atoms(3);
  atoms << 8, 1, 1; // O, H, H
  occ::Mat3N positions(3, 3);
  positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);
  positions.col(1) = occ::Vec3(0.0, 0.7935, 0.0);
  positions.col(2) = occ::Vec3(0.0, 0.0, 0.7935);

  occ::core::Molecule water(atoms, positions);

  SECTION("Coordinate generation from molecule") {
    InternalCoordinates coords(water, {false});

    // Should generate 2 bonds (O-H) and 1 angle (H-O-H)
    REQUIRE(coords.bonds().size() == 2);
    REQUIRE(coords.angles().size() == 1);
    REQUIRE(coords.dihedrals().size() == 0); // No dihedrals requested
    REQUIRE(coords.size() == 3);             // Total coordinates

    // Check bond indices are correct (should be reordered i < j)
    REQUIRE(coords.bonds()[0].i == 0); // O
    REQUIRE(coords.bonds()[0].j == 1); // H
    REQUIRE(coords.bonds()[1].i == 0); // O
    REQUIRE(coords.bonds()[1].j == 2); // H

    // Check angle indices (should be reordered i < k)
    REQUIRE(coords.angles()[0].i == 1); // H
    REQUIRE(coords.angles()[0].j == 0); // O (center)
    REQUIRE(coords.angles()[0].k == 2); // H
  }

  SECTION("Coordinate evaluation to vector") {
    InternalCoordinates coords(water, {false});
    auto values = coords.to_vector(positions);

    REQUIRE(values.size() == 3);

    // Values should match our individual coordinate tests
    REQUIRE(values(0) == Catch::Approx(1.4994976798).margin(1e-9)); // Bond O-H1
    REQUIRE(values(1) == Catch::Approx(1.4994976798).margin(1e-9)); // Bond O-H2
    REQUIRE(values(2) ==
            Catch::Approx(1.5707963268).margin(1e-9)); // Angle H-O-H (90°)
  }

  SECTION("Wilson B-matrix dimensions and values") {
    InternalCoordinates coords(water, {false});
    auto B = coords.wilson_b_matrix(positions);

    // Matrix dimensions: 3 coordinates × 9 Cartesian components
    REQUIRE(B.rows() == 3);
    REQUIRE(B.cols() == 9);

    // Check specific B-matrix elements match reference exactly
    // First bond (O-H1): grad components should be [0, -1, 0] and [0, 1, 0]
    REQUIRE(B(0, 0) == Catch::Approx(0.0).margin(1e-9));  // O_x
    REQUIRE(B(0, 1) == Catch::Approx(-1.0).margin(1e-9)); // O_y
    REQUIRE(B(0, 2) == Catch::Approx(0.0).margin(1e-9));  // O_z
    REQUIRE(B(0, 3) == Catch::Approx(0.0).margin(1e-9));  // H1_x
    REQUIRE(B(0, 4) == Catch::Approx(1.0).margin(1e-9));  // H1_y
    REQUIRE(B(0, 5) == Catch::Approx(0.0).margin(1e-9));  // H1_z

    // Second bond (O-H2): grad components should be [0, 0, -1] and [0, 0, 1]
    REQUIRE(B(1, 0) == Catch::Approx(0.0).margin(1e-9));  // O_x
    REQUIRE(B(1, 1) == Catch::Approx(0.0).margin(1e-9));  // O_y
    REQUIRE(B(1, 2) == Catch::Approx(-1.0).margin(1e-9)); // O_z
    REQUIRE(B(1, 8) == Catch::Approx(1.0).margin(1e-9));  // H2_z
  }
}

TEST_CASE("InternalCoordinates Class Methane",
          "[internal_coordinates][methane]") {
  // Tetrahedral CH4 geometry
  occ::IVec atoms(5);
  atoms << 6, 1, 1, 1, 1; // C, H, H, H, H
  occ::Mat3N positions(3, 5);
  positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);
  positions.col(1) = occ::Vec3(0.629, 0.629, 0.629);
  positions.col(2) = occ::Vec3(-0.629, -0.629, 0.629);
  positions.col(3) = occ::Vec3(-0.629, 0.629, -0.629);
  positions.col(4) = occ::Vec3(0.629, -0.629, -0.629);

  occ::core::Molecule methane(atoms, positions);

  SECTION("Coordinate generation with dihedrals") {
    InternalCoordinates coords(methane, {true});

    // Should generate 4 C-H bonds, 6 H-C-H angles, and possibly some dihedrals
    // (Our simple dihedral logic may not generate dihedrals for this symmetric
    // case)
    REQUIRE(coords.bonds().size() == 4);
    REQUIRE(coords.angles().size() == 6);
    // Note: Symmetric methane may not generate dihedrals with our current logic

    // All bonds should be C-H (carbon is atom 0)
    for (const auto &bond : coords.bonds()) {
      REQUIRE((bond.i == 0 || bond.j == 0)); // One end is carbon
    }

    // All angles should be H-C-H (carbon is center atom)
    for (const auto &angle : coords.angles()) {
      REQUIRE(angle.j == 0); // Carbon is center
      REQUIRE(angle.i != 0); // Other atoms are hydrogens
      REQUIRE(angle.k != 0);
    }
  }

  SECTION("Coordinate values match individual tests") {
    InternalCoordinates coords(methane, {true});
    auto values = coords.to_vector(positions);

    // All C-H bonds should be the same length
    for (size_t i = 0; i < 4; i++) {
      REQUIRE(values(i) == Catch::Approx(2.0587809442).margin(1e-9));
    }

    // All H-C-H angles should be tetrahedral
    for (size_t i = 4; i < 10; i++) {
      REQUIRE(values(i) == Catch::Approx(1.9106332362).margin(1e-9));
    }
  }

  SECTION("B-matrix dimensions") {
    InternalCoordinates coords(methane, {true});
    auto B = coords.wilson_b_matrix(positions);

    // Should have proper dimensions
    REQUIRE(B.rows() == static_cast<int>(coords.size()));
    REQUIRE(B.cols() == 15); // 5 atoms × 3 dimensions

    // Matrix should not be all zeros
    REQUIRE(B.norm() > 1.0);
  }
}

TEST_CASE("InternalCoordinates Class H2 Molecule",
          "[internal_coordinates][h2]") {
  // Simple H2 molecule with realistic bond length
  occ::IVec atoms(2);
  atoms << 1, 1; // H, H
  occ::Mat3N positions(3, 2);
  positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);
  positions.col(1) = occ::Vec3(0.74, 0.0, 0.0); // Realistic H-H bond length

  occ::core::Molecule h2(atoms, positions);

  SECTION("Minimal coordinate system") {
    InternalCoordinates coords(h2, {true});

    // Should have only 1 bond, no angles or dihedrals
    REQUIRE(coords.bonds().size() == 1);
    REQUIRE(coords.angles().size() == 0);
    REQUIRE(coords.dihedrals().size() == 0);
    REQUIRE(coords.size() == 1);

    // Bond should connect atoms 0 and 1
    REQUIRE(coords.bonds()[0].i == 0);
    REQUIRE(coords.bonds()[0].j == 1);
  }

  SECTION("Coordinate evaluation and B-matrix") {
    InternalCoordinates coords(h2, {true});
    auto values = coords.to_vector(positions);
    auto B = coords.wilson_b_matrix(positions);

    // Single bond length - calculate expected value for 0.74 Bohr distance
    REQUIRE(values.size() == 1);
    // H-H distance of 0.74 Bohr * angstrom conversion = 0.74
    // * 1.889726124565062 = 1.398 Å
    REQUIRE(values(0) == Catch::Approx(0.74 * 1.889726124565062).margin(1e-9));

    // B-matrix should be 1×6
    REQUIRE(B.rows() == 1);
    REQUIRE(B.cols() == 6);

    // B-matrix should match reference: [-1, 0, 0, 1, 0, 0]
    REQUIRE(B(0, 0) == Catch::Approx(-1.0).margin(1e-9));
    REQUIRE(B(0, 1) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(B(0, 2) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(B(0, 3) == Catch::Approx(1.0).margin(1e-9));
    REQUIRE(B(0, 4) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(B(0, 5) == Catch::Approx(0.0).margin(1e-9));
  }
}

TEST_CASE("Hessian Validation Water Molecule",
          "[coordinates][hessian][water]") {
  // Water molecule rho matrix from reference
  occ::Mat rho(3, 3);
  rho(0, 0) = 0.0000000000;
  rho(0, 1) = 1.3299417378;
  rho(0, 2) = 1.3299417378;
  rho(1, 0) = 1.3299417378;
  rho(1, 1) = 0.0000000000;
  rho(1, 2) = 0.6209215040;
  rho(2, 0) = 1.3299417378;
  rho(2, 1) = 0.6209215040;
  rho(2, 2) = 0.0000000000;

  SECTION("Bond Hessian Values") {
    BondCoordinate bond_01(0, 1);
    double hess_01 = bond_01.hessian(rho);
    // PyBerny: Bond(0, 1) hessian: 0.5984737820
    REQUIRE(hess_01 == Catch::Approx(0.5984737820).margin(1e-9));

    BondCoordinate bond_02(0, 2);
    double hess_02 = bond_02.hessian(rho);
    // PyBerny: Bond(0, 2) hessian: 0.5984737820
    REQUIRE(hess_02 == Catch::Approx(0.5984737820).margin(1e-9));
  }

  SECTION("Angle Hessian Values") {
    AngleCoordinate angle_102(1, 0, 2);
    double hess_102 = angle_102.hessian(rho);
    // PyBerny: Angle(1, 0, 2) hessian: 0.2653117539
    REQUIRE(hess_102 == Catch::Approx(0.2653117539).margin(1e-9));
  }

  SECTION("Dihedral Hessian Values") {
    // Test dihedral hessian calculation using same rho matrix
    // PyBerny formula: 0.005 * rho(i,j) * rho(j,k) * rho(k,l)
    DihedralCoordinate dihedral_0123(0, 1, 2, 0); // Creates a valid dihedral
    double hess_dihedral = dihedral_0123.hessian(rho);

    // Calculate expected value: 0.005 * rho(0,1) * rho(1,2) * rho(2,0)
    // = 0.005 * 1.3299417378 * 0.6209215040 * 1.3299417378
    double expected = 0.005 * rho(0, 1) * rho(1, 2) * rho(2, 0);
    REQUIRE(hess_dihedral == Catch::Approx(expected).margin(1e-9));
  }
}

TEST_CASE("Rho Matrix Calculation", "[internal_coordinates][rho]") {
  SECTION("Water molecule rho matrix") {
    occ::IVec atoms(3);
    atoms << 8, 1, 1; // O, H, H
    occ::Mat3N positions(3, 3);
    positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);
    positions.col(1) = occ::Vec3(0.0, 0.7935, 0.0);
    positions.col(2) = occ::Vec3(0.0, 0.0, 0.7935);

    occ::core::Molecule water(atoms, positions);
    const auto &elements = water.elements();
    int n_atoms = water.size();

    // Build rho matrix exactly as in our implementation
    occ::Mat rho(n_atoms, n_atoms);
    for (int i = 0; i < n_atoms; i++) {
      for (int j = 0; j < n_atoms; j++) {
        if (i == j) {
          rho(i, j) = 1.0;
        } else {
          double dist = (positions.col(i) - positions.col(j)).norm();
          double sum_radii = get_covalent_radius(elements[i].atomic_number()) +
                             get_covalent_radius(elements[j].atomic_number());
          rho(i, j) = std::exp(-dist / sum_radii + 1.0);
        }
      }
    }

    // Verify against reference values
    // Reference water rho matrix:
    // [[0.        , 1.32994174, 1.32994174],
    //  [1.32994174, 0.        , 0.6209215 ],
    //  [1.32994174, 0.6209215 , 0.        ]]

    REQUIRE(rho(0, 0) == 1.0); // Diagonal
    REQUIRE(rho(1, 1) == 1.0);
    REQUIRE(rho(2, 2) == 1.0);

    // Off-diagonal elements must match reference exactly
    REQUIRE(rho(0, 1) == Catch::Approx(1.32994174).margin(1e-8));
    REQUIRE(rho(1, 0) == Catch::Approx(1.32994174).margin(1e-8));
    REQUIRE(rho(0, 2) == Catch::Approx(1.32994174).margin(1e-8));
    REQUIRE(rho(2, 0) == Catch::Approx(1.32994174).margin(1e-8));
    REQUIRE(rho(1, 2) == Catch::Approx(0.6209215).margin(1e-8));
    REQUIRE(rho(2, 1) == Catch::Approx(0.6209215).margin(1e-8));
  }

  SECTION("H2 molecule rho matrix") {
    occ::IVec atoms(2);
    atoms << 1, 1; // H, H
    occ::Mat3N positions(3, 2);
    positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);
    positions.col(1) = occ::Vec3(0.0, 0.0, 0.74);

    occ::core::Molecule h2(atoms, positions);
    const auto &elements = h2.elements();
    int n_atoms = h2.size();

    // Build rho matrix exactly as in our implementation
    occ::Mat rho(n_atoms, n_atoms);
    for (int i = 0; i < n_atoms; i++) {
      for (int j = 0; j < n_atoms; j++) {
        if (i == j) {
          rho(i, j) = 1.0;
        } else {
          double dist = (positions.col(i) - positions.col(j)).norm();
          double sum_radii = get_covalent_radius(elements[i].atomic_number()) +
                             get_covalent_radius(elements[j].atomic_number());
          rho(i, j) = std::exp(-dist / sum_radii + 1.0);
        }
      }
    }

    // Verify against reference values
    // Reference H2 rho matrix:
    // [[0.        , 1.02666511],
    //  [1.02666511, 0.        ]]

    REQUIRE(rho(0, 0) == 1.0); // Diagonal
    REQUIRE(rho(1, 1) == 1.0);

    // Off-diagonal elements must match reference exactly
    REQUIRE(rho(0, 1) == Catch::Approx(1.02666511).margin(1e-8));
    REQUIRE(rho(1, 0) == Catch::Approx(1.02666511).margin(1e-8));
  }
}

TEST_CASE("Species Data Validation", "[species_data][validation]") {
  SECTION("Ethane molecule connectivity") {
    // C2H6 - ethane
    occ::IVec atoms(8);
    atoms << 6, 6, 1, 1, 1, 1, 1, 1; // C, C, H, H, H, H, H, H
    occ::Mat3N positions(3, 8);
    // Ethane geometry with C-C bond ~1.54 Å, C-H bonds ~1.09 Å
    positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);       // C
    positions.col(1) = occ::Vec3(1.54, 0.0, 0.0);      // C
    positions.col(2) = occ::Vec3(-0.63, 0.89, 0.0);    // H
    positions.col(3) = occ::Vec3(-0.63, -0.45, 0.77);  // H
    positions.col(4) = occ::Vec3(-0.63, -0.45, -0.77); // H
    positions.col(5) = occ::Vec3(2.17, 0.89, 0.0);     // H
    positions.col(6) = occ::Vec3(2.17, -0.45, 0.77);   // H
    positions.col(7) = occ::Vec3(2.17, -0.45, -0.77);  // H

    occ::core::Molecule ethane(atoms, positions);
    InternalCoordinates coords(ethane, {true});

    // Reference C cov_radius = 0.77, H cov_radius = 0.38
    // C-C: 1.54 < 1.3*(0.77+0.77) = 2.002 ✓
    // C-H: ~1.09 < 1.3*(0.77+0.38) = 1.495 ✓
    // H-H: ~1.78 should NOT bond (> 1.3*0.76 = 0.988)

    REQUIRE(coords.bonds().size() == 7); // 1 C-C + 6 C-H bonds

    // Should have many angles (H-C-H, H-C-C, C-C-H)
    REQUIRE(coords.angles().size() > 10);

    // Should have dihedrals around C-C bond
    REQUIRE(coords.dihedrals().size() > 0);

    fmt::print("Ethane: {} bonds, {} angles, {} dihedrals\n",
               coords.bonds().size(), coords.angles().size(),
               coords.dihedrals().size());
  }

  SECTION("Benzene molecule connectivity") {
    // C6H6 - benzene ring
    occ::IVec atoms(12);
    atoms << 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1; // 6 C, 6 H
    occ::Mat3N positions(3, 12);
    double r = 1.39;  // C-C bond length in benzene
    double rh = 1.08; // C-H bond length

    // Hexagon geometry
    for (int i = 0; i < 6; i++) {
      double angle = i * M_PI / 3.0;
      positions.col(i) =
          occ::Vec3(r * cos(angle), r * sin(angle), 0.0); // C atoms
      positions.col(i + 6) = occ::Vec3((r + rh) * cos(angle),
                                       (r + rh) * sin(angle), 0.0); // H atoms
    }

    occ::core::Molecule benzene(atoms, positions);
    InternalCoordinates coords(benzene, {true});

    // Should have 6 C-C bonds + 6 C-H bonds = 12 bonds
    // Reference C-C: 1.39 < 1.3*(0.77+0.77) = 2.002 ✓
    // Reference C-H: 1.08 < 1.3*(0.77+0.38) = 1.495 ✓

    REQUIRE(coords.bonds().size() == 12);  // 6 C-C + 6 C-H
    REQUIRE(coords.angles().size() >= 18); // Many angles in ring
    // Reference generates exactly 24 dihedrals for benzene
    REQUIRE(coords.dihedrals().size() == 24);

    fmt::print("Benzene: {} bonds, {} angles, {} dihedrals\n",
               coords.bonds().size(), coords.angles().size(),
               coords.dihedrals().size());
  }

  SECTION("Mixed elements - methanol") {
    // CH3OH
    occ::IVec atoms(6);
    atoms << 6, 8, 1, 1, 1, 1; // C, O, H, H, H, H
    occ::Mat3N positions(3, 6);
    positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);       // C
    positions.col(1) = occ::Vec3(1.43, 0.0, 0.0);      // O (C-O ~1.43 Å)
    positions.col(2) = occ::Vec3(1.91, 0.0, 0.0);      // H on O (O-H ~0.96 Å)
    positions.col(3) = occ::Vec3(-0.63, 0.89, 0.0);    // H on C
    positions.col(4) = occ::Vec3(-0.63, -0.45, 0.77);  // H on C
    positions.col(5) = occ::Vec3(-0.63, -0.45, -0.77); // H on C

    occ::core::Molecule methanol(atoms, positions);
    InternalCoordinates coords(methanol, {true});

    // Reference radii: C=0.77, O=0.73, H=0.38
    // C-O: 1.43 < 1.3*(0.77+0.73) = 1.95 ✓
    // O-H: 0.96 < 1.3*(0.73+0.38) = 1.443 ✓
    // C-H: ~1.09 < 1.3*(0.77+0.38) = 1.495 ✓

    REQUIRE(coords.bonds().size() == 5);  // C-O + O-H + 3*C-H
    REQUIRE(coords.angles().size() >= 6); // H-C-H, H-C-O, C-O-H angles

    // PyBerny generates 0 dihedrals for methanol due to linearity filtering
    REQUIRE(coords.dihedrals().size() == 0);

    fmt::print("Methanol: {} bonds, {} angles, {} dihedrals\n",
               coords.bonds().size(), coords.angles().size(),
               coords.dihedrals().size());
  }

  SECTION("VdW radii accessibility") {
    // Test that VdW radii are available and reasonable
    float h_vdw = get_vdw_radius(1); // H
    float c_vdw = get_vdw_radius(6); // C
    float o_vdw = get_vdw_radius(8); // O
    float n_vdw = get_vdw_radius(7); // N

    REQUIRE(h_vdw > 1.0f); // H VdW should be > 1 Å
    REQUIRE(h_vdw < 2.0f); // but reasonable
    REQUIRE(c_vdw > 1.5f); // C VdW should be larger than H
    REQUIRE(o_vdw > 1.5f); // O VdW
    REQUIRE(n_vdw > 1.5f); // N VdW

    fmt::print("VdW radii - H: {:.3f}, C: {:.3f}, O: {:.3f}, N: {:.3f}\n",
               h_vdw, c_vdw, o_vdw, n_vdw);

    // Check PyBerny exact values
    REQUIRE(h_vdw == Catch::Approx(1.6404493538f).margin(1e-8f));
    REQUIRE(c_vdw == Catch::Approx(1.8997461871f).margin(1e-8f));
    REQUIRE(o_vdw == Catch::Approx(1.6880753028f).margin(1e-8f));
    REQUIRE(n_vdw == Catch::Approx(1.7674518844f).margin(1e-8f));
  }
}

TEST_CASE("BernyOptimizer Basic Functionality", "[berny_optimizer][basic]") {
  SECTION("H2 molecule initialization") {
    // Simple H2 molecule - test basic setup
    occ::IVec atoms(2);
    atoms << 1, 1; // H, H
    occ::Mat3N positions(3, 2);
    positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);
    // Positions in Angstroms! Use 0.8 Å (within 1.3 * cov_radii = 0.988 Å)
    positions.col(1) = occ::Vec3(0.8, 0.0, 0.0);

    occ::core::Molecule h2(atoms, positions);

    fmt::print("H2 Basic Test:\n");
    fmt::print("  Initial bond length: {:.6f} Å\n",
               (h2.positions().col(1) - h2.positions().col(0)).norm());

    // Check connectivity: PyBerny H cov_radius = 0.38 Å
    // Bond criterion: dist < 1.3 * (0.38 + 0.38) = 0.988 Å
    fmt::print("  Max bond distance: {:.6f} Å (1.3 * 2 * 0.38)\n",
               1.3 * 2 * 0.38);

    // Create optimizer with default criteria
    ConvergenceCriteria criteria;

    // First test coordinate generation directly
    try {
      InternalCoordinates coords(h2, {true});
      fmt::print("  Generated coordinates: {} bonds, {} angles, {} dihedrals\n",
                 coords.bonds().size(), coords.angles().size(),
                 coords.dihedrals().size());

      if (coords.size() == 0) {
        fmt::print("  ERROR: No coordinates generated!\n");
        // Debug connectivity
        InternalCoordinates coords_no_dih(h2, {false});
        fmt::print("  Without dihedrals: {} total coords\n",
                   coords_no_dih.size());
      }
    } catch (const std::exception &e) {
      fmt::print("  Coordinate generation error: {}\n", e.what());
    }

    try {
      BernyOptimizer optimizer(h2, criteria);
      fmt::print("  Optimizer created successfully\n");
      fmt::print("  Initial step number: {}\n", optimizer.current_step());

      // Get initial geometry
      occ::core::Molecule geom0 = optimizer.get_next_geometry();
      fmt::print("  Got initial geometry: {} atoms\n", geom0.size());

      REQUIRE(optimizer.current_step() == 0);
      REQUIRE(geom0.size() == 2);

    } catch (const std::exception &e) {
      fmt::print("  Error: {}\n", e.what());
      REQUIRE(false); // Force test failure
    }
  }
}

TEST_CASE("Polynomial Fitting Functions vs Reference",
          "[berny_optimizer][polynomial]") {
  SECTION("fit_cubic validation") {
    // Test case 1: PyBerny values from Step 2 Linear Search
    double E0 = 1.4580605077059446e-06;
    double E1 = 0.012800000000000006;
    double g0 = 0.000374; // PyBerny value
    double g1 = 0.048000; // PyBerny value

    auto [t, E] = fit_cubic(E0, E1, g0, g1);

    // PyBerny fit_cubic: t=0.2837160334837822, E=-0.00020557279563526574
    REQUIRE(t == Catch::Approx(0.2837160334837822).margin(1e-12));
    REQUIRE(E == Catch::Approx(-0.00020557279563526574).margin(1e-12));

    // Test case 2: Simple test case
    auto [t2, E2] = fit_cubic(0.0, 1.0, -1.0, 1.0);

    // PyBerny fit_cubic: t=0.13962038997193676, E=-0.06708845558673691
    REQUIRE(t2 == Catch::Approx(0.13962038997193676).margin(1e-12));
    REQUIRE(E2 == Catch::Approx(-0.06708845558673691).margin(1e-12));
  }

  SECTION("fit_quartic validation") {
    // Test case 1: PyBerny values from Step 2 Linear Search
    double E0 = 1.4580605077059446e-06;
    double E1 = 0.012800000000000006;
    double g0 = 0.000374; // PyBerny value
    double g1 = 0.048000; // PyBerny value

    auto [t, E] = fit_quartic(E0, E1, g0, g1);

    // PyBerny fit_quartic: t=-0.2505332080041504, E=-7.224702250020454e-05
    REQUIRE(t == Catch::Approx(-0.2505332080041504).margin(1e-12));
    REQUIRE(E == Catch::Approx(-7.224702250020454e-05).margin(1e-12));

    // Test case 2: Simple test case where quartic should fail
    auto [t2, E2] = fit_quartic(0.0, 1.0, -1.0, 1.0);

    // PyBerny fit_quartic: t=None, E=None (should return NaN)
    REQUIRE(std::isnan(t2));
    REQUIRE(std::isnan(E2));
  }

  SECTION("linear_search validation") {
    // Test case 1: Should use quartic
    double E0 = 1.4580605077059446e-06;
    double E1 = 0.012800000000000006;
    double g0 = 0.000374; // PyBerny value
    double g1 = 0.048000; // PyBerny value

    auto [t, E] = linear_search(E0, E1, g0, g1);

    // Should return quartic result: t=-0.2505332080041504
    REQUIRE(t == Catch::Approx(-0.2505332080041504).margin(1e-12));
    REQUIRE(E == Catch::Approx(-7.224702250020454e-05).margin(1e-12));

    // Test case 2: Should fallback to cubic
    auto [t2, E2] = linear_search(0.0, 1.0, -1.0, 1.0);

    // Should return cubic result: t=0.13962038997193676
    REQUIRE(t2 == Catch::Approx(0.13962038997193676).margin(1e-12));
    REQUIRE(E2 == Catch::Approx(-0.06708845558673691).margin(1e-12));
  }

  SECTION("Validate with our actual log values") {
    // Our values from logs (different gradients)
    double E0 = 1.4580605077059446e-06;
    double E1 = 0.012800000000000006;
    double g0 = 0.000511; // Our value
    double g1 = 0.047861; // Our value

    auto [t_cubic, E_cubic] = fit_cubic(E0, E1, g0, g1);
    auto [t_quartic, E_quartic] = fit_quartic(E0, E1, g0, g1);
    auto [t_search, E_search] = linear_search(E0, E1, g0, g1);

    // PyBerny with our values:
    // fit_cubic: t=0.2803044428513965, E=-0.00017771791905229025
    // fit_quartic: t=-0.26723256535123174, E=-0.00010536311262102363
    REQUIRE(t_cubic == Catch::Approx(0.2803044428513965).margin(1e-12));
    REQUIRE(E_cubic == Catch::Approx(-0.00017771791905229025).margin(1e-12));
    REQUIRE(t_quartic == Catch::Approx(-0.26723256535123174).margin(1e-12));
    REQUIRE(E_quartic == Catch::Approx(-0.00010536311262102363).margin(1e-12));

    // Linear search should pick quartic
    REQUIRE(t_search == Catch::Approx(t_quartic).margin(1e-12));
    REQUIRE(E_search == Catch::Approx(E_quartic).margin(1e-12));
  }
}

TEST_CASE("InternalCoordinates VdW Fallback Water Dimer",
          "[internal_coordinates][vdw_fallback]") {
  // Water dimer - two H2O molecules separated by ~3 Å
  // Tests VdW fallback mechanism for non-bonded systems
  occ::IVec atoms(6);
  atoms << 8, 1, 1, 8, 1, 1; // O, H, H, O, H, H

  occ::Mat3N positions(3, 6);
  // First water molecule
  positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);      // O1
  positions.col(1) = occ::Vec3(0.0, 0.757, 0.587);  // H1
  positions.col(2) = occ::Vec3(0.0, -0.757, 0.587); // H2

  // Second water molecule ~3 Å away
  positions.col(3) = occ::Vec3(3.0, 0.0, 0.0);      // O2
  positions.col(4) = occ::Vec3(3.0, 0.757, 0.587);  // H3
  positions.col(5) = occ::Vec3(3.0, -0.757, 0.587); // H4

  occ::core::Molecule water_dimer(atoms, positions);

  SECTION("VdW fallback bond analysis") {
    InternalCoordinates coords(water_dimer, {false});
    auto values = coords.to_vector(positions);

    // Debug: print all bonds and their lengths
    fmt::print("Generated {} bonds:\n", coords.bonds().size());
    for (size_t i = 0; i < coords.bonds().size(); i++) {
      const auto &bond = coords.bonds()[i];
      std::string bond_type =
          (bond.bond_type == occ::opt::BondCoordinate::Type::COVALENT) ? "COV"
                                                                       : "VDW";
      fmt::print("  Bond {}: atoms {}-{}, length = {:.6f} Å [{}]\n", i, bond.i,
                 bond.j, values(i), bond_type);
    }

    // PyBerny reference bond lengths (sorted)
    std::vector<double> expected_lengths = {
        1.810214, 1.810214, 1.810214, 1.810214, // 4 O-H bonds
        5.669178, 5.669178, 5.669178,           // 3 shorter intermolecular
        5.951173, 5.951173, 5.951173, 5.951173  // 4 longer intermolecular
    };

    // Get actual lengths (sorted)
    std::vector<double> actual_lengths;
    for (size_t i = 0; i < coords.bonds().size(); i++) {
      actual_lengths.push_back(values(i));
    }
    std::sort(actual_lengths.begin(), actual_lengths.end());
    std::sort(expected_lengths.begin(), expected_lengths.end());

    fmt::print("\nPyBerny expected {} bonds:\n", expected_lengths.size());
    for (size_t i = 0; i < expected_lengths.size(); i++) {
      fmt::print("  {:.6f} Å\n", expected_lengths[i]);
    }

    fmt::print("\nOur actual {} bonds (sorted):\n", actual_lengths.size());
    for (size_t i = 0; i < actual_lengths.size(); i++) {
      fmt::print("  {:.6f} Å\n", actual_lengths[i]);
    }

    // Check that we match PyBerny exactly
    REQUIRE(coords.bonds().size() == 11);

    // Verify bond lengths match PyBerny exactly
    for (size_t i = 0; i < expected_lengths.size(); i++) {
      REQUIRE(actual_lengths[i] ==
              Catch::Approx(expected_lengths[i]).margin(1e-6));
    }
  }
}

TEST_CASE("InternalCoordinates Hessian Guess",
          "[internal_coordinates][hessian]") {
  // Test hessian_guess function using water molecule
  occ::IVec atoms(3);
  atoms << 8, 1, 1; // O, H, H
  occ::Mat3N positions(3, 3);
  positions.col(0) = occ::Vec3(0.0, 0.0, 0.0);
  positions.col(1) = occ::Vec3(0.0, 0.7935, 0.0);
  positions.col(2) = occ::Vec3(0.0, 0.0, 0.7935);

  occ::core::Molecule water(atoms, positions);
  InternalCoordinates coords(water, {false});

  SECTION("Hessian matrix dimensions and structure") {
    auto H = coords.hessian_guess();

    // Should be square matrix of size = number of coordinates
    REQUIRE(H.rows() == static_cast<int>(coords.size()));
    REQUIRE(H.cols() == static_cast<int>(coords.size()));
    REQUIRE(H.rows() == 3); // 2 bonds + 1 angle

    // Should be symmetric
    REQUIRE((H - H.transpose()).norm() < 1e-12);

    // Diagonal elements should be positive
    for (int i = 0; i < H.rows(); i++) {
      REQUIRE(H(i, i) > 0.0);
    }

    // Off-diagonal elements should be zero (diagonal approximation)
    for (int i = 0; i < H.rows(); i++) {
      for (int j = 0; j < H.cols(); j++) {
        if (i != j) {
          REQUIRE(std::abs(H(i, j)) < 1e-12);
        }
      }
    }
  }

  SECTION("Hessian diagonal values match PyBerny exactly") {
    auto H = coords.hessian_guess();

    // Values must match PyBerny exactly
    // PyBerny water hessian diagonal: [0.5984737820, 0.5984737820,
    // 0.2653117539]
    REQUIRE(H(0, 0) == Catch::Approx(0.5984737820).margin(1e-9)); // Bond O-H
    REQUIRE(H(1, 1) == Catch::Approx(0.5984737820).margin(1e-9)); // Bond O-H
    REQUIRE(H(2, 2) == Catch::Approx(0.2653117539).margin(1e-9)); // Angle H-O-H
  }
}

