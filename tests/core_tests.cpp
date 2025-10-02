#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/ostream.h>
#include <occ/core/dimer.h>
#include <occ/core/eem.h>
#include <occ/core/eeq.h>
#include <occ/core/elastic_tensor.h>
#include <occ/core/element.h>
#include <occ/core/graph.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/multipole.h>
#include <occ/core/numpy.h>
#include <occ/core/optimize.h>
#include <occ/core/point_group.h>
#include <occ/core/quasirandom.h>
#include <occ/core/table.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>

/* Dimer tests */
using occ::core::Dimer;
using occ::core::MolecularPointGroup;
using occ::core::Molecule;
using occ::core::Multipole;
using occ::core::SymOp;
using occ::util::all_close;

using Catch::Approx;
using Catch::Matchers::WithinAbs;

inline Molecule water_molecule() {
  occ::Mat3N pos(3, 3);
  occ::IVec nums(3);
  nums << 8, 1, 1;
  pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
      0.07959806, 0.01878821, -0.02171049, 0.00986248;

  return Molecule(nums, pos);
}

inline Molecule oh_molecule() {
  occ::Mat3N pos(3, 2);
  occ::IVec nums(2);
  nums << 8, 1;
  pos << -1.32695761, -1.93166418, -0.10593856, 1.60017351, 0.01878821,
      -0.02171049;

  return Molecule(nums, pos);
}

TEST_CASE("Dimer transform", "[dimer]") {
  occ::Mat3N pos(3, 3), pos2(3, 3);
  occ::IVec nums(3);
  nums << 8, 1, 1;
  pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
      0.07959806, 0.01878821, -0.02171049, 0.00986248;
  pos2 = pos;
  pos2.topRows(1).array() *= -1;

  Molecule m(nums, pos), m2(nums, pos2);

  Dimer dim(m, m2);

  fmt::print("m1\n{}\n", occ::format_matrix(m.positions()));
  fmt::print("m2\n{}\n", occ::format_matrix(m2.positions()));

  auto transform = dim.symmetry_relation().value();

  fmt::print("Transform matrix:\n{}\n", occ::format_matrix(transform));
  m.transform(transform, Molecule::Origin::Centroid);
  fmt::print("m1\n{}\n", occ::format_matrix(m.positions()));
  fmt::print("m2\n{}\n", occ::format_matrix(m2.positions()));
  REQUIRE(all_close(m.positions(), m2.positions(), 1e-5, 1e-5));
}

TEST_CASE("Dimer separations", "[dimer]") {
  occ::Mat3N pos(3, 3), pos2(3, 3);
  occ::IVec nums(3);
  nums << 8, 1, 1;
  pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
      0.07959806, 0.01878821, -0.02171049, 0.00986248;
  pos2 = pos;
  pos2.topRows(1).array() *= -1;

  Molecule m(nums, pos), m2(nums, pos2);

  Dimer dim(m, m2);

  REQUIRE(dim.nearest_distance() == Approx(0.8605988136));
  REQUIRE(dim.centroid_distance() == Approx(1.8479851333));
  REQUIRE(dim.center_of_mass_distance() == Approx(2.5186418514));
}

// Element tests
TEST_CASE("Element constructor", "[element]") {
  using occ::core::Element;
  REQUIRE(Element("H").symbol() == "H");
  REQUIRE(Element("He").symbol() == "He");
  REQUIRE(Element("He1").symbol() == "He");
  REQUIRE(Element(6).name() == "carbon");
  REQUIRE(Element("Ne") > Element("H"));
  REQUIRE(Element("NA").symbol() == "N");
  REQUIRE(Element("Na").symbol() == "Na");
}

// Molecule

TEST_CASE("Molecule constructor", "[molecule]") {
  occ::Mat3N pos(3, 2);
  occ::IVec nums(2);
  nums << 1, 1;
  pos << -1.0, 1.0, 0.0, 0.0, 1.0, 1.0;
  Molecule m(nums, pos);
  REQUIRE(m.size() == 2);
}

TEST_CASE("Molecule atom properties", "[molecule]") {
  occ::Mat3N pos(3, 3);
  occ::IVec nums(3);
  nums << 8, 1, 1;
  pos << -1.32695761, -0.10593856, 0.01878821, -1.93166418, 1.60017351,
      -0.02171049, 0.48664409, 0.07959806, 0.00986248;
  Molecule m(nums, pos);

  auto masses = m.atomic_masses();
  occ::Vec3 expected_masses = {15.994, 1.00794, 1.00794};

  fmt::print("Atomic masses:\n{}\n", occ::format_matrix(masses));
  REQUIRE(all_close(masses, expected_masses, 1e-3, 1e-3));
}

TEST_CASE("Molecule centroids", "[molecule]") {
  occ::Mat3N pos(3, 3);
  occ::IVec nums(3);
  nums << 8, 1, 1;
  pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
      0.07959806, 0.01878821, -0.02171049, 0.00986248;
  Molecule m(nums, pos);

  occ::Vec3 expected_centroid = {-0.92399257, 0.524611, 0.0023134};
  occ::Vec3 calc_centroid = m.centroid();
  fmt::print("Calculated centroid:\n{}\n", occ::format_matrix(calc_centroid));
  REQUIRE(all_close(expected_centroid, calc_centroid, 1e-05, 1e-05));

  occ::Vec3 expected_com = {-1.25932093, -0.000102380208, 0.0160229578};
  occ::Vec3 calc_com = m.center_of_mass();
  fmt::print("Calculated center of mass:\n{}\n", occ::format_matrix(calc_com));
  REQUIRE(all_close(expected_com, calc_com, 1e-05, 1e-05));
}

TEST_CASE("Molecule rotation & translation", "[molecule]") {
  occ::Mat3N pos(3, 3);
  occ::IVec nums(3);
  nums << 8, 1, 1;
  pos << -1.32695761, -1.93166418, 0.48664409, -0.10593856, 1.60017351,
      0.07959806, 0.01878821, -0.02171049, 0.00986248;
  Molecule m(nums, pos);

  Eigen::Affine3d rotation360;
  rotation360 = Eigen::AngleAxis<double>(M_PI * 2, occ::Vec3(0, 1, 0));

  m.rotate(rotation360);
  REQUIRE(all_close(pos, m.positions()));

  Eigen::Affine3d rotation180;
  rotation180 = Eigen::AngleAxis<double>(M_PI, occ::Vec3(1, 0, 0));
  auto expected_pos = pos;
  expected_pos.bottomRows(2).array() *= -1;
  m.rotate(rotation180);
  fmt::print("Rot:\n{}\n", occ::format_matrix(rotation180.linear()));
  fmt::print("Expected:\n{}\nFound:\n{}\n", occ::format_matrix(expected_pos),
             occ::format_matrix(m.positions()));
  REQUIRE(all_close(expected_pos, m.positions()));
}

TEST_CASE("Molecule permute", "[molecule]") {
  // Create a simple molecule with 4 atoms
  occ::Mat3N pos(3, 4);
  occ::IVec nums(4);
  nums << 6, 1, 1, 8;        // C, H, H, O
  pos << 0.0, 1.0, 2.0, 3.0, // x
      0.0, 0.0, 0.0, 0.0,    // y
      0.0, 0.0, 0.0, 0.0;    // z

  Molecule m(nums, pos);

  // Set some additional properties to test they're permuted correctly
  occ::IVec asym_idx(4);
  asym_idx << 0, 1, 1, 2;
  m.set_asymmetric_unit_idx(asym_idx);

  // Add some bonds
  m.add_bond(0, 1); // C-H
  m.add_bond(0, 2); // C-H
  m.add_bond(0, 3); // C-O

  SECTION("Identity permutation") {
    std::vector<int> perm = {0, 1, 2, 3};
    Molecule permuted = m.permute(perm);

    REQUIRE(all_close(m.positions(), permuted.positions()));
    REQUIRE(m.atomic_numbers() == permuted.atomic_numbers());
    REQUIRE(permuted.asymmetric_unit_idx() == asym_idx);
  }

  SECTION("Swap hydrogens") {
    std::vector<int> perm = {0, 2, 1, 3}; // Swap H atoms
    Molecule permuted = m.permute(perm);

    // Check positions are swapped
    REQUIRE(permuted.positions()(0, 1) == Approx(2.0));
    REQUIRE(permuted.positions()(0, 2) == Approx(1.0));

    // Check atomic numbers are swapped
    REQUIRE(permuted.atomic_numbers()(1) == 1);
    REQUIRE(permuted.atomic_numbers()(2) == 1);

    // Check asymmetric indices are swapped
    REQUIRE(permuted.asymmetric_unit_idx()(1) == 1);
    REQUIRE(permuted.asymmetric_unit_idx()(2) == 1);

    // Check bonds are updated correctly
    auto bonds = permuted.bonds();
    REQUIRE(bonds.size() == 3);
    // Bonds should now be (0,2), (0,1), (0,3) due to the swap
    bool found_0_1 = false, found_0_2 = false, found_0_3 = false;
    for (const auto &bond : bonds) {
      if ((bond.first == 0 && bond.second == 1) ||
          (bond.first == 1 && bond.second == 0))
        found_0_1 = true;
      if ((bond.first == 0 && bond.second == 2) ||
          (bond.first == 2 && bond.second == 0))
        found_0_2 = true;
      if ((bond.first == 0 && bond.second == 3) ||
          (bond.first == 3 && bond.second == 0))
        found_0_3 = true;
    }
    REQUIRE(found_0_1);
    REQUIRE(found_0_2);
    REQUIRE(found_0_3);
  }

  SECTION("Reverse order") {
    std::vector<int> perm = {3, 2, 1, 0};
    Molecule permuted = m.permute(perm);

    // Check positions are reversed
    REQUIRE(permuted.positions()(0, 0) == Approx(3.0));
    REQUIRE(permuted.positions()(0, 1) == Approx(2.0));
    REQUIRE(permuted.positions()(0, 2) == Approx(1.0));
    REQUIRE(permuted.positions()(0, 3) == Approx(0.0));

    // Check atomic numbers are reversed
    REQUIRE(permuted.atomic_numbers()(0) == 8); // O
    REQUIRE(permuted.atomic_numbers()(1) == 1); // H
    REQUIRE(permuted.atomic_numbers()(2) == 1); // H
    REQUIRE(permuted.atomic_numbers()(3) == 6); // C
  }

  SECTION("Invalid permutation size") {
    std::vector<int> perm = {0, 1, 2}; // Too short
    REQUIRE_THROWS(m.permute(perm));
  }
}

// Multipole

TEST_CASE("Multipole constructor", "[multipole]") {
  auto c = Multipole<0>{};
  auto d = Multipole<1>{};
  auto q = Multipole<2>{};
  auto o = Multipole<3>{};
  fmt::print("Charge\n{}\n", c.to_string());
  fmt::print("Dipole\n{}\n", d.to_string());
  fmt::print("Quadrupole\n{}\n", q.to_string());
  fmt::print("Octupole\n{}\n", o.to_string());
  REQUIRE(c.charge() == 0.0);
}

TEST_CASE("Multipole addition", "[multipole]") {
  auto o = Multipole<3>{{1.0,  0.0, 0.0, 0.5, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
                         10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}};
  auto q = Multipole<2>{{1.0, 0.0, 0.0, 0.5, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0}};
  auto sum_oq = o + q;
  fmt::print("Result\n{}\n", sum_oq.to_string());
  for (unsigned int i = 0; i < o.num_components; i++) {
    if (i < q.num_components)
      REQUIRE(sum_oq.components[i] == (o.components[i] + q.components[i]));
    else
      REQUIRE(sum_oq.components[i] == o.components[i]);
  }
}

// Numpy IO

TEST_CASE("Write eigen array to numpy .npy file", "[numpy]") {
  Eigen::MatrixXd i = Eigen::MatrixXd::Identity(6, 6);
  i(0, 4) = 4;
  const std::string filename = "identity.npy";
  occ::core::numpy::save_npy(filename, i);
  auto arr = occ::core::numpy::load_npy(filename);
  Eigen::Map<Eigen::MatrixXd, 0> m(arr.data<double>(), arr.shape[0],
                                   arr.shape[1]);
  REQUIRE(all_close(i, m));
  std::remove(filename.c_str());
}

// Optimize

double sx(double x) { return std::sin(x); }

double ax(double x) { return std::abs(x - 4); }

TEST_CASE("Line search") {
  using occ::core::opt::LineSearch;

  auto info = [](const std::string &desc, int num_iter, double x, double y) {
    occ::log::info("Found minimum of {} in {} evaluations: ({:.3f}, {:.3f})",
                   desc, num_iter, x, y);
  };

  SECTION("quadratic") {
    for (double a = -9.5; a < 10.0; a += 1.0) {
      auto x2 = [&a](double x) { return (x - a) * (x - a); };
      LineSearch b(x2);
      double xmin = b.xmin();
      double fmin = b.f_xmin();
      REQUIRE_THAT(xmin, WithinAbs(a, 1e-6));
      info(fmt::format("(x - {:.2f})^2", a), b.num_calls(), xmin, fmin);
    }
  }

  SECTION("gaussian") {
    for (double a = 0.5; a < 10.0; a += 1.0) {
      auto gx = [&a](double x) { return -std::exp(-a * x * x); };
      LineSearch b(gx);
      double xmin = b.xmin();
      double fmin = b.f_xmin();
      REQUIRE_THAT(xmin, WithinAbs(0, 1e-6));
      REQUIRE_THAT(fmin, WithinAbs(-1, 1e-6));
      info(fmt::format("- exp(-{:.2f} x^2)", a), b.num_calls(), xmin, fmin);
    }
  }

  SECTION("abs") {
    for (double a = -9.5; a < 10.0; a += 1.0) {
      auto absx = [&a](double x) { return std::abs(x - a); };
      LineSearch b(absx);
      double xmin = b.xmin();
      double fmin = b.f_xmin();
      REQUIRE_THAT(xmin, WithinAbs(a, 1e-6));
      REQUIRE_THAT(fmin, WithinAbs(0, 1e-6));
      info(fmt::format("abs(x - {:.2f})", a), b.num_calls(), xmin, fmin);
    }
  }
}

// Molecular Point Group

TEST_CASE("Point group Symop constructors", "[point_group]") {
  occ::Vec3 axis(0.0, 1.0, 0.0);
  double angle = 90.0;
  occ::Vec3 rotvec = angle * axis;
  auto s = SymOp::from_axis_angle(axis, angle);
  fmt::print("Transformation:\n{}\n", occ::format_matrix(s.transformation));
  auto s2 = SymOp::from_rotation_vector(rotvec);
  fmt::print("Transformation:\n{}\n", occ::format_matrix(s.transformation));
}

TEST_CASE("PointGroup determination water = C2v", "[point_group]") {
  occ::Mat3N pos(3, 3);
  occ::IVec nums(3);
  nums << 8, 1, 1;
  pos << -0.7021961, -0.0560603, 0.0099423, -1.0221932, 0.8467758, -0.0114887,
      0.2575211, 0.0421215, 0.0052190;

  Molecule m(nums, pos.transpose());

  MolecularPointGroup pg(m);

  fmt::print("Water group: {}\n", pg.point_group_string());
  REQUIRE(pg.point_group() == occ::core::PointGroup::C2v);
  REQUIRE(pg.symmetry_number() == 2);
}

TEST_CASE("PointGroup determination O2 = Dooh", "[point_group]") {
  occ::Mat3N pos(3, 2);
  occ::IVec nums(2);
  nums << 8, 8;
  pos << -0.616, 0.616, 0, 0, 0, 0;

  Molecule m(nums, pos);

  MolecularPointGroup pg(m);

  fmt::print("Oxygen group: {}\n", pg.point_group_string());
  for (const auto &sym : pg.symops()) {
    fmt::print("symop:\n{}\n", occ::format_matrix(sym.transformation));
  }
  for (const auto &sym : pg.rotational_symmetries()) {
    fmt::print("rotational symmetry: {}\n", sym.second);
  }

  REQUIRE(pg.point_group() == occ::core::PointGroup::Dooh);
  REQUIRE(pg.symmetry_number() == 2);
}

TEST_CASE("PointGroup determination BF3 = D3h", "[point_group]") {
  occ::Mat3N pos(3, 4);
  occ::IVec nums(4);
  nums << 5, 9, 9, 9;
  pos << 0.0, 0.0000, 0.8121, -0.8121, 0.0, -0.9377, 0.4689, 0.4689, 0.0,
      0.0000, 0.0000, 0.0000;

  Molecule m(nums, pos);

  MolecularPointGroup pg(m);

  fmt::print("BF3 group: {}\n", pg.point_group_string());
  for (const auto &sym : pg.symops()) {
    fmt::print("symop:\n{}\n", occ::format_matrix(sym.transformation));
  }
  for (const auto &sym : pg.rotational_symmetries()) {
    fmt::print("rotational symmetry: {}\n", sym.second);
  }
  REQUIRE(pg.point_group() == occ::core::PointGroup::D3h);
  REQUIRE(pg.symmetry_number() == 6);
}

TEST_CASE("PointGroup determination benzene = D6h", "[point_group]") {
  occ::Mat3N pos(3, 12);
  occ::IVec nums(12);
  nums << 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1;
  pos << -0.71308, -0.00904, 1.39904, 2.10308, 1.39904, -0.00904, -1.73551,
      -0.52025, 1.91025, 3.12551, 1.91025, -0.52025, 1.20378, -0.01566,
      -0.01566, 1.20378, 2.42321, 2.42321, 1.20378, -0.90111, -0.90111, 1.20378,
      3.30866, 3.30866, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0;

  Molecule m(nums, pos);

  MolecularPointGroup pg(m);

  fmt::print("Benzene group: {}\n", pg.point_group_string());
  REQUIRE(pg.point_group() == occ::core::PointGroup::D6h);
}

TEST_CASE("PointGroup determination methane = Td", "[point_group]") {
  occ::Mat3N pos(3, 5);
  occ::IVec nums(5);
  nums << 6, 1, 1, 1, 1;
  pos << 0.0, 0.00, 1.026719, -0.513360, -0.513360, 0.0, 0.00, 0.000000,
      -0.889165, 0.889165, 0.0, 1.08, -0.363000, -0.363000, -0.363000;

  Molecule m(nums, pos);

  MolecularPointGroup pg(m);

  fmt::print("CH4 group: {}\n", pg.point_group_string());
  REQUIRE(pg.point_group() == occ::core::PointGroup::Td);
}

TEST_CASE("PointGroup determination cube of atoms = Oh", "[point_group]") {
  occ::Mat3N pos(3, 8);
  occ::IVec nums(8);

  nums.setConstant(6);
  pos << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
      0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;

  Molecule m(nums, pos);

  MolecularPointGroup pg(m);
  fmt::print("Cube group: {}\n", pg.point_group_string());
  for (const auto &sym : pg.symops()) {
    fmt::print("symop:\n{}\n", occ::format_matrix(sym.transformation));
  }
  for (const auto &sym : pg.rotational_symmetries()) {
    fmt::print("rotational symmetry: {}\n", sym.second);
  }

  REQUIRE(pg.point_group() == occ::core::PointGroup::Oh);
}

TEST_CASE("PointGroup determination CHFBrCl = C1", "[point_group]") {
  occ::Mat3N pos(3, 5);
  occ::IVec nums(5);

  nums << 6, 1, 9, 35, 17;
  pos << 0.000000, 0.000000, 1.026719, -0.513360, -0.513360, 0.000000, 0.000000,
      0.000000, -0.889165, 0.889165, 0.000000, 1.080000, -0.363000, -0.363000,
      -0.363000;

  Molecule m(nums, pos);

  MolecularPointGroup pg(m);
  fmt::print("CHFBrCl group: {}\n", pg.point_group_string());
  REQUIRE(pg.point_group() == occ::core::PointGroup::C1);
}

// Table TODO remove or update it

TEST_CASE("Table constructor", "[table]") {
  using occ::io::Table;

  Table t({"test", "columns"});
  std::vector<int> test{1, 2, 3};
  std::vector<std::string> columns{"this", "is", "a", "test"};
  t.set_column("test", test);
  t.set_column("columns", columns);
  t.print();
}

TEST_CASE("Table Eigen matrix print", "[table]") {
  using occ::io::Table;
  Eigen::MatrixXd r = Eigen::MatrixXd::Random(4, 3);
  Table t;

  t.set_column("random", r);
  t.print();
}

// Utils

TEST_CASE("Basic usage of all_close helper", "[util]") {
  using occ::util::all_close;

  occ::Mat x = occ::Mat::Identity(3, 3);
  occ::Mat x2 = occ::Mat::Identity(3, 3) * 2;
  REQUIRE(all_close(x, x));
  REQUIRE(!all_close(x, x2));
  REQUIRE(all_close(x, x2 * 0.5));
}

TEST_CASE("Basic usage of is_close helper", "[util]") {
  using occ::util::is_close;
  double x1 = 1e-6;
  double x2 = 2e-6;
  REQUIRE(!is_close(x1, x2));
  REQUIRE(is_close(x1, x2 / 2));
  REQUIRE(is_close(1e-17, 1e-18));
}

TEST_CASE("Basic usage of is_even, is_odd helpers", "[util]") {
  using occ::util::is_even, occ::util::is_odd;
  REQUIRE(is_even(2));
  REQUIRE(is_odd(1));
  REQUIRE(is_even(0));
  REQUIRE(is_odd(1312321413));
  REQUIRE(!is_odd(1312421412));
  REQUIRE(!is_even(1312421411));
}

TEST_CASE("Basic cases of smallest_common_factor", "[util]") {
  using occ::util::smallest_common_factor;
  REQUIRE(smallest_common_factor(1, 3) == 1);
  REQUIRE(smallest_common_factor(-1, 3) == 1);
  REQUIRE(smallest_common_factor(-2, 4) == 2);
  REQUIRE(smallest_common_factor(-4, 12) == 4);
  REQUIRE(smallest_common_factor(12, 12) == 12);
  REQUIRE(smallest_common_factor(0, 12) == 12);
}

TEST_CASE("Basic cases of human_readable_size", "[util]") {
  using occ::util::human_readable_size;

  REQUIRE(human_readable_size(1024, "B") == "1.00KiB");
  REQUIRE(human_readable_size(1024 * 1024, "B") == "1.00MiB");
  REQUIRE(human_readable_size(1024 * 1024 * 1024, "B") == "1.00GiB");
  REQUIRE(human_readable_size(0, "B") == "0.00B");
}

TEST_CASE("Depth first & Breadth first graph traversal", "[graph]") {
  using Graph = occ::core::graph::Graph<int, int>;
  using vertex_desc = Graph::VertexDescriptor;
  Graph graph;

  std::vector<vertex_desc> descriptors;

  for (int i = 0; i < 10; i++) {
    descriptors.push_back(graph.add_vertex(i));
  }

  auto e1 = graph.add_edge(descriptors[0], descriptors[1], 1, true);
  auto e2 = graph.add_edge(descriptors[1], descriptors[2], 1, true);
  auto e3 = graph.add_edge(descriptors[2], descriptors[3], 1, true);
  auto e4 = graph.add_edge(descriptors[4], descriptors[5], 1, true);
  auto e5 = graph.add_edge(descriptors[4], descriptors[6], 1, true);
  auto e6 = graph.add_edge(descriptors[4], descriptors[7], 1, true);

  auto f = [](const vertex_desc &v) { fmt::print("vertex = {}\n", v); };

  fmt::print("Depth first:\n");
  graph.depth_first_traversal(descriptors[0], f);
  fmt::print("Breadth first:\n");
  graph.breadth_first_traversal(descriptors[0], f);
  auto components = graph.connected_components();
  REQUIRE(components[descriptors[0]] == components[descriptors[3]]);
  REQUIRE(components[descriptors[0]] != components[descriptors[4]]);
  REQUIRE(components[descriptors[4]] == components[descriptors[6]]);
  REQUIRE(components[descriptors[4]] == components[descriptors[5]]);
}

TEST_CASE("Quasirandom KGF", "[quasirandom]") {

  occ::Mat pts = occ::core::quasirandom_kgf(3, 5, 10);
  occ::Mat expected(3, 5);
  expected << 0.5108976473578082, 0.3300701607539729, 0.1492426741501376,
      0.9684151875463005, 0.7875877009424652, 0.8814796737416799,
      0.5525232804454685, 0.2235668871492571, 0.8946104938530475,
      0.5656541005568361, 0.5467052569216708, 0.0964057348236409,
      0.646106212725611, 0.19580669062758105, 0.7455071685295511;

  REQUIRE(occ::util::all_close(pts, expected));
}

TEST_CASE("EEM water", "[charge]") {
  occ::Mat3N pos(3, 3);
  occ::IVec nums(3);
  nums << 8, 1, 1;
  pos << -0.7021961, -1.0221932, 0.2575211, -0.0560603, 0.8467758, 0.0421215,
      0.0099423, -0.0114887, 0.0052190;

  auto q = occ::core::charges::eem_partial_charges(nums, pos, 0.0);
  fmt::print("EEM water charges:\n{}\n", occ::format_matrix(q));

  occ::Vec expected_q(3);
  expected_q << -0.637207, 0.319851, 0.317356;
  REQUIRE(occ::util::all_close(expected_q, q, 1e-5, 1e-5));
}

TEST_CASE("EEQ water", "[charge]") {
  occ::Mat3N pos(3, 3);
  occ::IVec nums(3);
  nums << 8, 1, 1;
  pos << -0.7021961, -1.0221932, 0.2575211, -0.0560603, 0.8467758, 0.0421215,
      0.0099423, -0.0114887, 0.0052190;

  auto cn = occ::core::charges::eeq_coordination_numbers(nums, pos);
  fmt::print("EEQ water coordination numbers:\n{}\n", occ::format_matrix(cn));
  occ::Vec expected(3);
  expected << 1.989382, 0.995133, 0.994268;
  REQUIRE(occ::util::all_close(cn, expected, 1e-5, 1e-5));

  auto q = occ::core::charges::eeq_partial_charges(nums, pos, 0.0);
  fmt::print("EEQ water charges:\n{}\n", occ::format_matrix(q));

  occ::Vec expected_q(3);
  expected_q << -0.592456, 0.297289, 0.295167;
  REQUIRE(occ::util::all_close(expected_q, q, 1e-5, 1e-5));
}

TEST_CASE("Elastic tensor", "[elastic_tensor]") {
  using occ::core::ElasticTensor;

  Eigen::Matrix<double, 6, 6> tensor;
  tensor << 48.137, 11.411, 12.783, 0.000, -3.654, 0.000, 11.411, 34.968,
      14.749, 0.000, -0.094, 0.000, 12.783, 14.749, 26.015, 0.000, -4.528,
      0.000, 0.000, 0.000, 0.000, 14.545, 0.000, 0.006, -3.654, -0.094, -4.528,
      0.000, 10.771, 0.000, 0.000, 0.000, 0.000, 0.006, 0.000, 11.947;

  ElasticTensor elastic(tensor);

  SECTION("Basics") {
    REQUIRE(occ::util::all_close(tensor, elastic.voigt_c(), 1e-6, 1e-6));
    REQUIRE(
        occ::util::all_close(tensor.inverse(), elastic.voigt_s(), 1e-6, 1e-6));
    fmt::print("Voigt S\n{}\n", occ::format_matrix(elastic.voigt_s()));
  }

  SECTION("Voigt Averages") {
    const auto avg = ElasticTensor::AveragingScheme::Voigt;
    REQUIRE(elastic.average_bulk_modulus(avg) == Approx(20.778).epsilon(1e-3));
    REQUIRE(elastic.average_youngs_modulus(avg) ==
            Approx(30.465).epsilon(1e-3));
    REQUIRE(elastic.average_shear_modulus(avg) == Approx(12.131).epsilon(1e-3));
    REQUIRE(elastic.average_poisson_ratio(avg) ==
            Approx(0.25564).epsilon(1e-4));
  }

  SECTION("Reuss Averages") {
    const auto avg = ElasticTensor::AveragingScheme::Reuss;
    REQUIRE(elastic.average_bulk_modulus(avg) == Approx(19.000).epsilon(1e-3));
    REQUIRE(elastic.average_youngs_modulus(avg) ==
            Approx(27.087).epsilon(1e-3));
    REQUIRE(elastic.average_shear_modulus(avg) == Approx(10.728).epsilon(1e-3));
    REQUIRE(elastic.average_poisson_ratio(avg) ==
            Approx(0.26239).epsilon(1e-4));
  }

  SECTION("Hill Averages") {
    const auto avg = ElasticTensor::AveragingScheme::Hill;
    REQUIRE(elastic.average_bulk_modulus(avg) == Approx(19.889).epsilon(1e-3));
    REQUIRE(elastic.average_youngs_modulus(avg) ==
            Approx(28.777).epsilon(1e-3));
    REQUIRE(elastic.average_shear_modulus(avg) == Approx(11.43).epsilon(1e-3));
    REQUIRE(elastic.average_poisson_ratio(avg) ==
            Approx(0.25886).epsilon(1e-4));
  }

  SECTION("Young's Modulus") {
    occ::Vec3 dmin(0.3540, 0.0, 0.9352);
    occ::Vec3 dmax(0.9885, 0.0000, -0.1511);
    double ymin = elastic.youngs_modulus(dmin);
    REQUIRE(ymin == Approx(14.751).epsilon(1e-2));
    double ymax = elastic.youngs_modulus(dmax);
    REQUIRE(ymax == Approx(41.961).epsilon(1e-2));
  }

  SECTION("Linear Compressibility") {
    occ::Vec3 dmin(0.9295, -0.0000, -0.3688);
    occ::Vec3 dmax(0.3688, -0.0000, 0.9295);
    double lcmin = elastic.linear_compressibility(dmin);
    REQUIRE(lcmin == Approx(8.2545).epsilon(1e-2));
    double lcmax = elastic.linear_compressibility(dmax);
    REQUIRE(lcmax == Approx(31.357).epsilon(1e-2));
  }

  SECTION("Shear modulus") {
    occ::Vec3 d1min(-0.2277, 0.7071, -0.6694);
    occ::Vec3 d2min(-0.2276, -0.7071, -0.6695);

    occ::Vec3 d1max(0.7352, 0.6348, 0.2378);
    occ::Vec3 d2max(-0.6612, 0.5945, 0.4575);
    double smin = elastic.shear_modulus(d1min, d2min);
    REQUIRE(smin == Approx(6.5183).epsilon(1e-2));
    double smax = elastic.shear_modulus(d1max, d2max);
    REQUIRE(smax == Approx(15.505).epsilon(1e-2));
  }

  SECTION("Poisson's ratio") {
    occ::Vec3 d1min(0.5593, 0.6044, 0.5674);
    occ::Vec3 d2min(0.0525, 0.6572, -0.7519);

    occ::Vec3 d1max(0.0, 1.0, -0.0);
    occ::Vec3 d2max(-0.2611, -0.0000, -0.9653);
    double vmin = elastic.poisson_ratio(d1min, d2min);
    REQUIRE(vmin == Approx(0.067042).epsilon(1e-2));
    double vmax = elastic.poisson_ratio(d1max, d2max);
    REQUIRE(vmax == Approx(0.59507).epsilon(1e-2));
  }
}

TEST_CASE("Elastic tensor - ELATE comparison", "[elastic_tensor]") {
  using occ::core::ElasticTensor;

  // Test tensor from user with ELATE reference values
  Eigen::Matrix<double, 6, 6> tensor;
  tensor << 228.38, 85.741, 81.503, 0.000, -0.737, 0.000, 85.741, 217.47,
      94.201, 0.000, -20.213, 0.000, 81.503, 94.201, 178.81, 0.000, -9.472,
      0.000, 0.000, 0.000, 0.000, 35.094, 0.000, -17.851, -0.737, -20.213,
      -9.472, 0.000, 37.778, 0.000, 0.000, 0.000, 0.000, -17.851, 0.000, 42.708;

  ElasticTensor elastic(tensor);

  SECTION("Voigt Averages - ELATE comparison") {
    const auto avg = ElasticTensor::AveragingScheme::Voigt;
    double bulk = elastic.average_bulk_modulus(avg);
    double youngs = elastic.average_youngs_modulus(avg);
    double shear = elastic.average_shear_modulus(avg);
    double poisson = elastic.average_poisson_ratio(avg);

    fmt::print("Voigt averages (OCC vs ELATE reference):\n");
    fmt::print("Bulk modulus: {:.3f} GPa (ELATE: 127.51 GPa)\n", bulk);
    fmt::print("Young's modulus: {:.3f} GPa (ELATE: 126.36 GPa)\n", youngs);
    fmt::print("Shear modulus: {:.3f} GPa (ELATE: 47.331 GPa)\n", shear);
    fmt::print("Poisson's ratio: {:.5f} (ELATE: 0.33483)\n", poisson);

    // Test against ELATE reference values
    REQUIRE(bulk == Approx(127.51).epsilon(1e-2));
    REQUIRE(youngs == Approx(126.36).epsilon(1e-2));
    REQUIRE(shear == Approx(47.331).epsilon(1e-2));
    REQUIRE(poisson == Approx(0.33483).epsilon(1e-4));
  }

  SECTION("Reuss Averages - ELATE comparison") {
    const auto avg = ElasticTensor::AveragingScheme::Reuss;
    double bulk = elastic.average_bulk_modulus(avg);
    double youngs = elastic.average_youngs_modulus(avg);
    double shear = elastic.average_shear_modulus(avg);
    double poisson = elastic.average_poisson_ratio(avg);

    fmt::print("Reuss averages (OCC vs ELATE reference):\n");
    fmt::print("Bulk modulus: {:.3f} GPa (ELATE: 123.55 GPa)\n", bulk);
    fmt::print("Young's modulus: {:.3f} GPa (ELATE: 105.5 GPa)\n", youngs);
    fmt::print("Shear modulus: {:.3f} GPa (ELATE: 38.853 GPa)\n", shear);
    fmt::print("Poisson's ratio: {:.5f} (ELATE: 0.35768)\n", poisson);

    // Test against ELATE reference values
    REQUIRE(bulk == Approx(123.55).epsilon(1e-2));
    REQUIRE(youngs == Approx(105.5).epsilon(1e-2));
    REQUIRE(shear == Approx(38.853).epsilon(1e-2));
    REQUIRE(poisson == Approx(0.35768).epsilon(1e-4));
  }

  SECTION("Hill Averages - ELATE comparison") {
    const auto avg = ElasticTensor::AveragingScheme::Hill;
    double bulk = elastic.average_bulk_modulus(avg);
    double youngs = elastic.average_youngs_modulus(avg);
    double shear = elastic.average_shear_modulus(avg);
    double poisson = elastic.average_poisson_ratio(avg);

    fmt::print("Hill averages (OCC vs ELATE reference):\n");
    fmt::print("Bulk modulus: {:.3f} GPa (ELATE: 125.53 GPa)\n", bulk);
    fmt::print("Young's modulus: {:.3f} GPa (ELATE: 116 GPa)\n", youngs);
    fmt::print("Shear modulus: {:.3f} GPa (ELATE: 43.092 GPa)\n", shear);
    fmt::print("Poisson's ratio: {:.5f} (ELATE: 0.34598)\n", poisson);

    // Test against ELATE reference values
    REQUIRE(bulk == Approx(125.53).epsilon(1e-2));
    REQUIRE(youngs == Approx(116.0).epsilon(1e-2));
    REQUIRE(shear == Approx(43.092).epsilon(1e-2));
    REQUIRE(poisson == Approx(0.34598).epsilon(1e-4));
  }

  SECTION("Matrix validation") {
    // Verify the tensor was loaded correctly
    REQUIRE(occ::util::all_close(tensor, elastic.voigt_c(), 1e-6, 1e-6));

    // Print the tensor for verification
    fmt::print("Input tensor:\n{}\n", occ::format_matrix(tensor));
    fmt::print("Compliance matrix:\n{}\n",
               occ::format_matrix(elastic.voigt_s()));
  }
}

TEST_CASE("Elastic tensor rotation invariance", "[elastic_tensor][rotation]") {
  using occ::core::ElasticTensor;

  // Use the same test tensor as ELATE comparison
  Eigen::Matrix<double, 6, 6> tensor;
  tensor << 228.38, 85.741, 81.503, 0.000, -0.737, 0.000, 85.741, 217.47,
      94.201, 0.000, -20.213, 0.000, 81.503, 94.201, 178.81, 0.000, -9.472,
      0.000, 0.000, 0.000, 0.000, 35.094, 0.000, -17.851, -0.737, -20.213,
      -9.472, 0.000, 37.778, 0.000, 0.000, 0.000, 0.000, -17.851, 0.000, 42.708;

  ElasticTensor elastic(tensor);

  // Get original properties that should be invariant
  double original_bulk_voigt = elastic.average_bulk_modulus(ElasticTensor::AveragingScheme::Voigt);
  double original_bulk_reuss = elastic.average_bulk_modulus(ElasticTensor::AveragingScheme::Reuss);
  double original_shear_voigt = elastic.average_shear_modulus(ElasticTensor::AveragingScheme::Voigt);
  double original_shear_reuss = elastic.average_shear_modulus(ElasticTensor::AveragingScheme::Reuss);
  double original_youngs_voigt = elastic.average_youngs_modulus(ElasticTensor::AveragingScheme::Voigt);
  double original_youngs_reuss = elastic.average_youngs_modulus(ElasticTensor::AveragingScheme::Reuss);

  // Test some directional properties
  occ::Vec3 test_dir(1.0, 0.0, 0.0);
  double original_E_x = elastic.youngs_modulus(test_dir);
  double original_lc_x = elastic.linear_compressibility(test_dir);

  SECTION("Rotation around z-axis by 45 degrees") {
    double angle = M_PI / 4.0;
    Eigen::Matrix3d R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    auto rotated_c = elastic.rotate_voigt_stiffness(R);
    ElasticTensor rotated(rotated_c);

    // Averaged properties should be invariant
    REQUIRE(rotated.average_bulk_modulus(ElasticTensor::AveragingScheme::Voigt) ==
            Approx(original_bulk_voigt).epsilon(1e-6));
    REQUIRE(rotated.average_bulk_modulus(ElasticTensor::AveragingScheme::Reuss) ==
            Approx(original_bulk_reuss).epsilon(1e-6));
    REQUIRE(rotated.average_shear_modulus(ElasticTensor::AveragingScheme::Voigt) ==
            Approx(original_shear_voigt).epsilon(1e-6));
    REQUIRE(rotated.average_shear_modulus(ElasticTensor::AveragingScheme::Reuss) ==
            Approx(original_shear_reuss).epsilon(1e-6));
    REQUIRE(rotated.average_youngs_modulus(ElasticTensor::AveragingScheme::Voigt) ==
            Approx(original_youngs_voigt).epsilon(1e-6));
    REQUIRE(rotated.average_youngs_modulus(ElasticTensor::AveragingScheme::Reuss) ==
            Approx(original_youngs_reuss).epsilon(1e-6));

    // Test that directional properties transform correctly
    // Direction (1,0,0) rotated by 45° around z becomes (cos(45°), sin(45°), 0)
    occ::Vec3 rotated_dir = R * test_dir;
    double rotated_E_x = rotated.youngs_modulus(test_dir);
    double expected_E = elastic.youngs_modulus(rotated_dir);
    REQUIRE(rotated_E_x == Approx(expected_E).epsilon(1e-8));
  }

  SECTION("Rotation around x-axis by 90 degrees") {
    double angle = M_PI / 2.0;
    Eigen::Matrix3d R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();

    auto rotated_c = elastic.rotate_voigt_stiffness(R);
    ElasticTensor rotated(rotated_c);

    REQUIRE(rotated.average_bulk_modulus(ElasticTensor::AveragingScheme::Voigt) ==
            Approx(original_bulk_voigt).epsilon(1e-6));
    REQUIRE(rotated.average_shear_modulus(ElasticTensor::AveragingScheme::Voigt) ==
            Approx(original_shear_voigt).epsilon(1e-6));
    REQUIRE(rotated.average_youngs_modulus(ElasticTensor::AveragingScheme::Voigt) ==
            Approx(original_youngs_voigt).epsilon(1e-6));
  }

  SECTION("Arbitrary rotation") {
    // Arbitrary axis rotation
    Eigen::Vector3d axis(1.0, 2.0, 3.0);
    axis.normalize();
    double angle = 0.7;

    Eigen::Matrix3d R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();

    auto rotated_c = elastic.rotate_voigt_stiffness(R);
    ElasticTensor rotated(rotated_c);

    // Averaged properties should be invariant
    REQUIRE(rotated.average_bulk_modulus(ElasticTensor::AveragingScheme::Voigt) ==
            Approx(original_bulk_voigt).epsilon(1e-8));
    REQUIRE(rotated.average_shear_modulus(ElasticTensor::AveragingScheme::Voigt) ==
            Approx(original_shear_voigt).epsilon(1e-8));
    REQUIRE(rotated.average_youngs_modulus(ElasticTensor::AveragingScheme::Reuss) ==
            Approx(original_youngs_reuss).epsilon(1e-8));
  }

  SECTION("Double rotation returns to original") {
    double angle = M_PI / 3.0;
    Eigen::Matrix3d R = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    // Rotate, then rotate back
    auto rotated_c = elastic.rotate_voigt_stiffness(R);
    auto double_rotated_c = ElasticTensor(rotated_c).rotate_voigt_stiffness(R.transpose());

    // Should get back the original
    REQUIRE(occ::util::all_close(double_rotated_c, elastic.voigt_c(), 1e-8, 1e-8));
  }
}

TEST_CASE("Molecule label generation", "[molecule]") {
  SECTION("single molecule gets label 1A") {
    std::vector<occ::core::Molecule> molecules{water_molecule()};
    occ::core::label_molecules_by_chemical_formula(molecules);
    REQUIRE(molecules[0].name() == "1A");
  }

  SECTION("identical formulae get same number with incrementing letters") {
    std::vector<occ::core::Molecule> molecules{
        water_molecule(), water_molecule(), water_molecule()};
    occ::core::label_molecules_by_chemical_formula(molecules);
    REQUIRE(molecules[0].name() == "1A");
    REQUIRE(molecules[1].name() == "1B");
    REQUIRE(molecules[2].name() == "1C");
  }

  SECTION(
      "different molecules get different numbers with incrementing letters") {
    std::vector<occ::core::Molecule> molecules{water_molecule(), oh_molecule(),
                                               water_molecule(), oh_molecule()};

    occ::core::label_molecules_by_chemical_formula(molecules);
    REQUIRE(molecules[0].name() == "1A");
    REQUIRE(molecules[1].name() == "2A");
    REQUIRE(molecules[2].name() == "1B");
    REQUIRE(molecules[3].name() == "2B");
  }

  SECTION("more than 26 instances get repeat letters") {
    std::vector<occ::core::Molecule> molecules;
    for (int i = 0; i < 54; i++) {
      molecules.push_back(water_molecule());
    }

    occ::core::label_molecules_by_chemical_formula(molecules);
    REQUIRE(molecules[0].name() == "1A");
    REQUIRE(molecules[1].name() == "1B");
    REQUIRE(molecules[12].name() == "1M");
    REQUIRE(molecules[25].name() == "1Z");
    REQUIRE(molecules[26].name() == "1AA");
    REQUIRE(molecules[52].name() == "1BA");
  }
}
