#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_mapping_table.h>
#include <occ/crystal/muldin.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/surface.h>
#include <occ/crystal/symmetryoperation.h>

using occ::format_matrix;
using occ::IVec3;
using occ::Mat3N;
using occ::MatN3;
using occ::Vec3;
using occ::crystal::AsymmetricUnit;
using occ::crystal::Crystal;
using occ::crystal::HKL;
using occ::crystal::SpaceGroup;
using occ::crystal::Surface;
using occ::crystal::SymmetryOperation;
using occ::crystal::UnitCell;
using occ::units::radians;
using occ::util::all_close;

// Crystal

auto ice_ii_asym() {
  const std::vector<std::string> labels = {
      "O1",  "H1",  "H2",  "O2",  "H3",  "H4",  "O3",  "H5",  "H6",
      "O4",  "H7",  "H8",  "O5",  "H9",  "H10", "O6",  "H11", "H12",
      "O7",  "H13", "H14", "O8",  "H15", "H16", "O9",  "H17", "H18",
      "O10", "H19", "H20", "O11", "H21", "H22", "O12", "H23", "H24",
  };

  occ::IVec nums(labels.size());
  occ::Mat positions(labels.size(), 3);
  for (size_t i = 0; i < labels.size(); i++) {
    nums(i) = occ::core::Element(labels[i]).atomic_number();
  }

  positions << 0.273328954083, 0.026479033257, 0.855073668062, 0.152000330304,
      0.043488909374, 0.793595454907, 0.420775085827, 0.191165194485,
      0.996362203192, 0.144924657237, 0.726669877048, 0.973520141937,
      0.206402797363, 0.847998481439, 0.956510183901, 0.003636687868,
      0.579223433079, 0.808833958746, 0.026477491142, 0.855072949204,
      0.273328854276, 0.043487719387, 0.793594459529, 0.152000553858,
      0.191163388489, 0.996362120061, 0.420774953988, 0.726670757782,
      0.973520932681, 0.144926297633, 0.847999275418, 0.956510882297,
      0.206404294889, 0.579224602173, 0.808834869258, 0.003637530197,
      0.855073412561, 0.273329597478, 0.026478702027, 0.793594909621,
      0.152000771295, 0.043488316376, 0.996362312075, 0.420775512814,
      0.191164826329, 0.973520717390, 0.144925628579, 0.726671054509,
      0.956510600982, 0.206403626100, 0.847999547813, 0.808834607385,
      0.003637609551, 0.579224562315, 0.477029330652, 0.749805220756,
      0.331717174202, 0.402360172390, 0.720795433576, 0.401054786853,
      0.368036378343, 0.742284933413, 0.207434128329, 0.668282055550,
      0.522969467265, 0.250193622013, 0.598945169999, 0.597639203188,
      0.279204514235, 0.792565160978, 0.631962548905, 0.257714022497,
      0.749805496250, 0.331717033025, 0.477029827575, 0.720795009402,
      0.401054437437, 0.402360618546, 0.742284706875, 0.207433751728,
      0.368036342085, 0.522969071341, 0.250193392512, 0.668282780114,
      0.597638176364, 0.279203622225, 0.598945231951, 0.631962932785,
      0.257715003205, 0.792566578018, 0.331715381178, 0.477028907327,
      0.749804544234, 0.401053887354, 0.402360576463, 0.720795552111,
      0.207432480540, 0.368035542438, 0.742284142147, 0.250193225247,
      0.668282913065, 0.522970147212, 0.279203658434, 0.598945325854,
      0.597639149965, 0.257715011998, 0.792566781760, 0.631964289620;

  return AsymmetricUnit(positions.transpose(), nums, labels);
}

auto acetic_asym() {

  const std::vector<std::string> labels = {"C1", "C2", "H1", "H2",
                                           "H3", "H4", "O1", "O2"};
  occ::IVec nums(labels.size());
  occ::Mat positions(labels.size(), 3);
  for (size_t i = 0; i < labels.size(); i++) {
    nums(i) = occ::core::Element(labels[i]).atomic_number();
  }
  positions << 0.16510, 0.28580, 0.17090, 0.08940, 0.37620, 0.34810, 0.18200,
      0.05100, -0.11600, 0.12800, 0.51000, 0.49100, 0.03300, 0.54000, 0.27900,
      0.05300, 0.16800, 0.42100, 0.12870, 0.10750, 0.00000, 0.25290, 0.37030,
      0.17690;
  return AsymmetricUnit(positions.transpose(), nums, labels);
}

void print_asymmetric_unit(const AsymmetricUnit &asym) {
  for (size_t i = 0; i < asym.size(); i++) {
    fmt::print("{:<6s} {:>3d}   {:.6f} {:.6f} {:.6f}\n", asym.labels[i],
               asym.atomic_numbers(i), asym.positions(0, i),
               asym.positions(1, i), asym.positions(2, i));
  }
}

TEST_CASE("AsymmetricUnit constructor", "[crystal]") {

  AsymmetricUnit asym = ice_ii_asym();
  print_asymmetric_unit(asym);
  REQUIRE(asym.labels.size() == 36);
  std::vector<std::string> old_labels = asym.labels;
  asym.generate_default_labels();
  print_asymmetric_unit(asym);
  REQUIRE(old_labels == asym.labels);
}

TEST_CASE("UnitCell constructor", "[crystal]") {
  UnitCell ice = occ::crystal::rhombohedral_cell(7.78, radians(113.1));
  REQUIRE(ice.a() == Catch::Approx(7.78));
}

TEST_CASE("Symmetry unique and unit cell molecules (ice-II crystal)",
          "[crystal]") {
  AsymmetricUnit asym = ice_ii_asym();
  SpaceGroup sg(1);
  UnitCell cell = occ::crystal::rhombohedral_cell(7.78, radians(113.1));
  Crystal ice_ii(asym, sg, cell);
  REQUIRE(ice_ii.symmetry_operations().size() == 1);
  fmt::print("Unit cell molecules:\n");
  for (const auto &mol : ice_ii.unit_cell_molecules()) {
    fmt::print("{}\n", mol.name());
  }
  REQUIRE(ice_ii.symmetry_unique_molecules().size() == 12);
  fmt::print("Asymmetric unit molecules:\n");
  for (const auto &mol : ice_ii.symmetry_unique_molecules()) {
    fmt::print("{}\n", mol.name());
  }
}

SymmetryOperation dimer_symop(const occ::core::Dimer &dimer,
                              const Crystal &crystal) {
  const auto &a = dimer.a();
  const auto &b = dimer.b();

  int sa_int = a.asymmetric_unit_symop()(0);
  int sb_int = b.asymmetric_unit_symop()(0);

  SymmetryOperation symop_a(sa_int);
  SymmetryOperation symop_b(sb_int);

  auto symop_ab = symop_b * symop_a.inverted();
  occ::Vec3 c_a =
      symop_ab(crystal.to_fractional(a.positions())).rowwise().mean();
  occ::Vec3 v_ab = crystal.to_fractional(b.centroid()) - c_a;

  symop_ab = symop_ab.translated(v_ab);
  return symop_ab;
}

TEST_CASE("Symmetry unique and unit cell molecules (acetic acid crystal)",
          "[crystal]") {
  AsymmetricUnit asym = acetic_asym();
  SpaceGroup sg(33);
  UnitCell cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);

  Crystal acetic(asym, sg, cell);
  REQUIRE(acetic.unit_cell_molecules().size() == 4);
  fmt::print("Unit cell molecules:\n");
  for (const auto &mol : acetic.unit_cell_molecules()) {
    fmt::print("{}\n", mol.name());
  }
  REQUIRE(acetic.symmetry_unique_molecules().size() == 1);
  fmt::print("Asymmetric unit molecules:\n");
  for (const auto &mol : acetic.symmetry_unique_molecules()) {
    fmt::print("{}\n", mol.name());
  }

  auto crystal_dimers = acetic.symmetry_unique_dimers(3.8);
  const auto &dimers = crystal_dimers.unique_dimers;
  REQUIRE(dimers.size() == 7);
  fmt::print("Dimers\n");
  for (const auto &dimer : dimers) {
    auto s_ab = dimer_symop(dimer, acetic);
    fmt::print("R = {:.3f}, symop = {}\n", dimer.nearest_distance(),
               s_ab.to_string());
  }

  const auto &mol_neighbors = crystal_dimers.molecule_neighbors;
  for (size_t i = 0; i < mol_neighbors.size(); i++) {
    const auto &n = mol_neighbors[i];
    fmt::print("Neighbors for molecule {}\n", i);
    for (const auto &[dimer, unique_index] : n) {
      auto s_ab = dimer_symop(dimer, acetic);
      fmt::print("R = {:.3f}, symop = {}, unique_idx = {}\n",
                 dimer.nearest_distance(), s_ab.to_string(), unique_index);
    }
  }
}

TEST_CASE("Supercell construction (acetic acid crystal)", "[crystal]") {
  AsymmetricUnit asym = acetic_asym();
  SpaceGroup sg(33);
  UnitCell cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);

  Crystal acetic(asym, sg, cell);
  REQUIRE(acetic.unit_cell_molecules().size() == 4);
  fmt::print("Unit cell molecules:\n");
  for (const auto &mol : acetic.unit_cell_molecules()) {
    fmt::print("{}\n", mol.name());
  }
  Crystal acetic_p1 = Crystal::create_primitive_supercell(acetic, {1, 1, 1});
  REQUIRE(acetic.unit_cell_molecules().size() ==
          acetic_p1.symmetry_unique_molecules().size());
  fmt::print("Unique P1 molecules:\n");
  for (const auto &mol : acetic_p1.symmetry_unique_molecules()) {
    fmt::print("{}\n", mol.name());
  }
}

// SpaceGroup

TEST_CASE("SpaceGroup constructor", "[space_group]") {
  SpaceGroup sg14c = SpaceGroup("P21/c");
  REQUIRE(sg14c.number() == 14);
  SpaceGroup sg14a = SpaceGroup("P21/a");
  REQUIRE(sg14a.number() == 14);
  REQUIRE(sg14a.symmetry_operations().size() == 4);
}

TEST_CASE("Spacegroup symmetry operations are correct (P21/c)",
          "[space_group]") {
  MatN3 coords(8, 3);
  coords << 1.650999999999999968e-01, 2.857999999999999985e-01,
      1.708999999999999964e-01, 8.939999999999999336e-02,
      3.761999999999999789e-01, 3.481000000000000205e-01,
      1.819999999999999951e-01, 5.099999999999999672e-02,
      -1.160000000000000059e-01, 1.280000000000000027e-01,
      5.100000000000000089e-01, 4.909999999999999920e-01,
      3.300000000000000155e-02, 5.400000000000000355e-01,
      2.790000000000000258e-01, 5.299999999999999850e-02,
      1.680000000000000104e-01, 4.209999999999999853e-01,
      1.287000000000000088e-01, 1.074999999999999983e-01,
      0.000000000000000000e+00, 2.529000000000000137e-01,
      3.703000000000000180e-01, 1.769000000000000017e-01;

  REQUIRE(coords(0, 2) == 1.708999999999999964e-01);

  MatN3 coords_expected(32, 3);
  coords_expected << 1.650999999999999968e-01, 2.857999999999999985e-01,
      1.708999999999999964e-01, 8.939999999999999336e-02,
      3.761999999999999789e-01, 3.481000000000000205e-01,
      1.819999999999999951e-01, 5.099999999999999672e-02,
      -1.160000000000000059e-01, 1.280000000000000027e-01,
      5.100000000000000089e-01, 4.909999999999999920e-01,
      3.300000000000000155e-02, 5.400000000000000355e-01,
      2.790000000000000258e-01, 5.299999999999999850e-02,
      1.680000000000000104e-01, 4.209999999999999853e-01,
      1.287000000000000088e-01, 1.074999999999999983e-01,
      0.000000000000000000e+00, 2.529000000000000137e-01,
      3.703000000000000180e-01, 1.769000000000000017e-01,
      -1.650999999999999968e-01, 7.858000000000000540e-01,
      3.291000000000000036e-01, -8.939999999999999336e-02,
      8.761999999999999789e-01, 1.518999999999999795e-01,
      -1.819999999999999951e-01, 5.510000000000000453e-01,
      6.159999999999999920e-01, -1.280000000000000027e-01,
      1.010000000000000009e+00, 9.000000000000007994e-03,
      -3.300000000000000155e-02, 1.040000000000000036e+00,
      2.209999999999999742e-01, -5.299999999999999850e-02,
      6.680000000000000382e-01, 7.900000000000001465e-02,
      -1.287000000000000088e-01, 6.075000000000000400e-01,
      5.000000000000000000e-01, -2.529000000000000137e-01,
      8.703000000000000735e-01, 3.230999999999999983e-01,
      -1.650999999999999968e-01, -2.857999999999999985e-01,
      -1.708999999999999964e-01, -8.939999999999999336e-02,
      -3.761999999999999789e-01, -3.481000000000000205e-01,
      -1.819999999999999951e-01, -5.099999999999999672e-02,
      1.160000000000000059e-01, -1.280000000000000027e-01,
      -5.100000000000000089e-01, -4.909999999999999920e-01,
      -3.300000000000000155e-02, -5.400000000000000355e-01,
      -2.790000000000000258e-01, -5.299999999999999850e-02,
      -1.680000000000000104e-01, -4.209999999999999853e-01,
      -1.287000000000000088e-01, -1.074999999999999983e-01,
      0.000000000000000000e+00, -2.529000000000000137e-01,
      -3.703000000000000180e-01, -1.769000000000000017e-01,
      1.650999999999999968e-01, 2.142000000000000015e-01,
      6.709000000000000519e-01, 8.939999999999999336e-02,
      1.238000000000000211e-01, 8.481000000000000760e-01,
      1.819999999999999951e-01, 4.490000000000000102e-01,
      3.840000000000000080e-01, 1.280000000000000027e-01,
      -1.000000000000000888e-02, 9.909999999999999920e-01,
      3.300000000000000155e-02, -4.000000000000003553e-02,
      7.790000000000000258e-01, 5.299999999999999850e-02,
      3.319999999999999618e-01, 9.210000000000000409e-01,
      1.287000000000000088e-01, 3.925000000000000155e-01,
      5.000000000000000000e-01, 2.529000000000000137e-01,
      1.296999999999999820e-01, 6.769000000000000572e-01;

  SpaceGroup sg14("P21/c");
  Mat3N asym(coords.transpose());
  auto [symops, expanded] = sg14.apply_all_symmetry_operations(asym);
  std::cout << "Value - expected\n" << expanded.transpose() - coords_expected;
  REQUIRE(all_close(expanded.transpose(), coords_expected, 1e-3, 1e-3));
}

// Surface

auto acetic_acid_crystal() {
  const std::vector<std::string> labels = {"C1", "C2", "H1", "H2",
                                           "H3", "H4", "O1", "O2"};
  occ::IVec nums(labels.size());
  occ::Mat positions(labels.size(), 3);
  for (size_t i = 0; i < labels.size(); i++) {
    nums(i) = occ::core::Element(labels[i]).atomic_number();
  }
  positions << 0.16510, 0.28580, 0.17090, 0.08940, 0.37620, 0.34810, 0.18200,
      0.05100, -0.11600, 0.12800, 0.51000, 0.49100, 0.03300, 0.54000, 0.27900,
      0.05300, 0.16800, 0.42100, 0.12870, 0.10750, 0.00000, 0.25290, 0.37030,
      0.17690;
  AsymmetricUnit asym = AsymmetricUnit(positions.transpose(), nums, labels);
  SpaceGroup sg(33);
  UnitCell cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);
  return Crystal(asym, sg, cell);
}

TEST_CASE("Crystal Surface construction", "[crystal, surface]") {
  Crystal a = acetic_acid_crystal();
  HKL m010{0, 1, 0};
  Surface surf010(m010, a);
  surf010.print();
  REQUIRE(surf010.depth_vector().norm() == Catch::Approx(surf010.depth()));
  REQUIRE(surf010.area() == Catch::Approx(76.5325));
  HKL m011{0, 1, 1};
  Surface surf011(m011, a);
  surf011.print();
  REQUIRE(surf011.depth_vector().norm() == Catch::Approx(surf011.depth()));
  REQUIRE(surf011.area() == Catch::Approx(93.9958));
  HKL m321{3, 2, 1};
  Surface surf321(m321, a);
  surf321.print();
  REQUIRE(surf321.depth_vector().norm() == Catch::Approx(surf321.depth()));
  REQUIRE(surf321.area() == Catch::Approx(177.2256));
}

TEST_CASE("Crystal surface generation (dhkl order)", "[crystal, surface]") {
  Crystal a = acetic_acid_crystal();
  occ::timing::StopWatch sw;
  sw.start();
  occ::crystal::CrystalSurfaceGenerationParameters params;
  params.d_min = 0.1;
  params.unique = false;
  auto surfaces = occ::crystal::generate_surfaces(a, params);
  sw.stop();
  size_t n = 0;
  fmt::print("Top 10 surfaces\n");
  for (const auto &surf : surfaces) {
    surf.print();
    fmt::print("Absent: {}\n",
               Surface::check_systematic_absence(a, surf.hkl()));
    n++;
    if (n > 10)
      break;
  }
  fmt::print("Generation took {} s\n", sw.read());
}

TEST_CASE("Crystal surface molecules", "[crystal, surface]") {
  Crystal a = acetic_acid_crystal();
  HKL m011{0, 1, 1};
  Surface surf011(m011, a);
  surf011.print();
  auto mols = a.unit_cell_molecules();
  auto translations = surf011.find_molecule_cell_translations(mols, -10.0);
  auto dimers = a.unit_cell_dimers(12.0);
  auto counts = surf011.count_crystal_dimers_cut_by_surface(dimers);
  for (int i = 0; i < counts.above.size(); i++) {
    const auto &neighbor_counts = counts.above[i];
    for (int j = 0; j < neighbor_counts.size(); j++) {
      if (neighbor_counts[j] > 0) {
        fmt::print("{} {}: {}\n", i, j, neighbor_counts[j]);
      }
    }
  }
}

// SymmetryOperation

TEST_CASE("SymmetryOperation constructor", "[symmetry_operation]") {
  REQUIRE(SymmetryOperation("x,y,z").is_identity());
  REQUIRE(SymmetryOperation(16484).is_identity());
}

TEST_CASE("SymmetryOperation Seitz matrix", "[symmetry_operation]") {
  auto id = SymmetryOperation(16484);
  REQUIRE(all_close(id.seitz(), Eigen::Matrix4d::Identity()));
  REQUIRE(all_close(id.rotation(), id.inverted().rotation()));
}

TEST_CASE("MULDIN 3d", "[transformations]") {
  occ::Vec3 hkl(1.0, 1.0, 0.0);
  occ::Mat3 result = occ::crystal::muldin(hkl);
  fmt::print("hkl\n{}\n", format_matrix(hkl));
  fmt::print("Result\n{}\n", format_matrix(result));
  {
    occ::Vec3 inp(1.0, 1.0, 0.0);
    occ::Mat3 result;
    result << 0, 1, 1, 0, 0, 1, 1, 0, 0;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(1.0, 0.0, 0.0);
    occ::Mat3 result;
    result << 0, 0, 1, 1, 0, 0, 0, 1, 0;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(0.0, 0.0, 1.0);
    occ::Mat3 result;
    result << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {

    occ::Vec3 inp(-5.0, -1.0, 1.0);
    occ::Mat3 result;
    result << -1, 0, -5, 0, -1, -1, 0, 0, 1;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(2.0, 0.0, -3.0);
    occ::Mat3 result;
    result << 0, 1, 2, 1, 0, 0, 0, -1, -3;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(1.0, 1.0, -1.0);
    occ::Mat3 result;
    result << 0, 1, 1, 1, 0, 1, 0, 0, -1;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(-3.0, -2.0, 0.0);
    occ::Mat3 result;
    result << 0, -2, -3, 0, -1, -2, 1, 0, 0;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(0.0, -1.0, -3.0);
    occ::Mat3 result;
    result << 1, 0, 0, 0, -1, -1, 0, -2, -3;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(-1.0, -2.0, 0.0);
    occ::Mat3 result;
    result << 0, -1, -1, 0, -1, -2, 1, 0, 0;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(-3.0, -4.0, -3.0);
    occ::Mat3 result;
    result << -2, -1, -3, -3, 0, -4, -2, 0, -3;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(1.0, -5.0, -2.0);
    occ::Mat3 result;
    result << 1, 0, 1, -1, -1, -5, -1, 0, -2;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(-5.0, 0.0, 3.0);
    occ::Mat3 result;
    result << 0, -2, -5, 1, 0, 0, 0, 1, 3;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
  {
    occ::Vec3 inp(-1.0, 4.0, -5.0);
    occ::Mat3 result;
    result << -1, -1, -1, 3, 4, 4, -3, -4, -5;
    REQUIRE(all_close(result, occ::crystal::muldin(inp)));
  }
}

TEST_CASE("MULDIN surface transform", "[transformations]") {
  occ::Mat3N hkl_examples(3, 10);
  hkl_examples << 1, 0, 0, 2, 3, 4, 5, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0,
      0, 1, 0, 1, 3, 2, 0, 1, 1;

  occ::crystal::UnitCell cell(12.93, 12.3, 16.1, occ::units::radians(90),
                              occ::units::radians(115.9),
                              occ::units::radians(90));

  double volume = cell.volume();
  occ::Vec3 zeros(0, 0, 0);

  for (int i = 0; i < hkl_examples.cols(); i++) {
    occ::Mat3 S = occ::crystal::muldin(hkl_examples.col(i));
    occ::Mat3 transform = S.inverse().transpose();
    occ::Mat3 D = cell.direct() * transform;
    occ::Vec3 normal_vector =
        (cell.reciprocal() * hkl_examples.col(i)).normalized();
    occ::Vec3 dps = normal_vector.transpose() * D;
    REQUIRE_THAT(dps(0), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(dps(1), Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(D.determinant(), Catch::Matchers::WithinAbs(volume, 1e-6));
  }
}

TEST_CASE("SymmetryOperation ADP rotation", "[crystal][symmetry]") {
  using occ::Vec6;

  SECTION("Identity operation") {
    SymmetryOperation identity("x,y,z");
    Vec6 adp(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Vec6 rotated_adp = identity.rotate_adp(adp);
    INFO(fmt::format("rotated : {}\n", format_matrix(rotated_adp)));
    INFO(fmt::format("expected: {}\n", format_matrix(adp)));
    REQUIRE(rotated_adp.isApprox(adp));
  }

  SECTION("90-degree rotation around z-axis") {
    SymmetryOperation rot_z("-y,x,z");
    Vec6 adp(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Vec6 expected_adp(2.0, 1.0, 3.0, -0.1, -0.3, 0.2);
    Vec6 rotated_adp = rot_z.rotate_adp(adp);
    INFO(fmt::format("rotated : {}\n", format_matrix(rotated_adp)));
    INFO(fmt::format("expected: {}\n", format_matrix(expected_adp)));
    REQUIRE(rotated_adp.isApprox(expected_adp, 1e-10));
  }

  SECTION("Inversion") {
    SymmetryOperation inversion("-x,-y,-z");
    Vec6 adp(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Vec6 rotated_adp = inversion.rotate_adp(adp);

    INFO(fmt::format("rotated : {}\n", format_matrix(rotated_adp)));
    INFO(fmt::format("expected: {}\n", format_matrix(adp)));
    REQUIRE(rotated_adp.isApprox(adp));
  }

  SECTION("Mirror plane perpendicular to x-axis") {
    SymmetryOperation mirror_x("-x,y,z");
    Vec6 adp(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Vec6 expected_adp(1.0, 2.0, 3.0, -0.1, -0.2, 0.3);
    Vec6 rotated_adp = mirror_x.rotate_adp(adp);

    INFO(fmt::format("rotated : {}\n", format_matrix(rotated_adp)));
    INFO(fmt::format("expected: {}\n", format_matrix(expected_adp)));
    REQUIRE(rotated_adp.isApprox(expected_adp, 1e-10));
  }

  SECTION("Rotation with translation") {
    SymmetryOperation rot_trans("-y,x,z+1/2");
    Vec6 adp(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Vec6 expected_adp(2.0, 1.0, 3.0, -0.1, -0.3, 0.2);
    Vec6 rotated_adp = rot_trans.rotate_adp(adp);

    INFO(fmt::format("rotated : {}\n", format_matrix(rotated_adp)));
    INFO(fmt::format("expected: {}\n", format_matrix(expected_adp)));
    REQUIRE(rotated_adp.isApprox(expected_adp, 1e-10));
  }

  SECTION("General rotation") {
    SymmetryOperation general_rot("z,x,y");
    Vec6 adp(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Vec6 expected_adp(3.0, 1.0, 2.0, 0.2, 0.3, 0.1);
    Vec6 rotated_adp = general_rot.rotate_adp(adp);

    INFO(fmt::format("rotated : {}\n", format_matrix(rotated_adp)));
    INFO(fmt::format("expected: {}\n", format_matrix(expected_adp)));
    REQUIRE(rotated_adp.isApprox(expected_adp, 1e-10));
  }

  SECTION("Pure translation") {
    SymmetryOperation translation("x+1/2,y+1/4,z+1/3");
    Vec6 adp(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
    Vec6 rotated_adp = translation.rotate_adp(adp);
    INFO(fmt::format("rotated : {}\n", format_matrix(rotated_adp)));
    INFO(fmt::format("expected: {}\n", format_matrix(adp)));
    REQUIRE(rotated_adp.isApprox(adp));
  }
}

TEST_CASE("UnitCell ADP conversion", "[crystal][unitcell]") {
  using Catch::Approx;
  using occ::Mat3;
  using occ::Mat6N;
  using occ::units::radians;

  SECTION("Triclinic unit cell") {
    // Sample of data from CIF of 5FCINH doi 10.1021/acs.cgd.4c00401
    UnitCell cell(3.6833, 9.6484, 16.3512, radians(76.068), radians(88.936),
                  radians(81.425));

    Mat6N frac_adps(6, 4);
    Mat6N expected(6, 4);
    // clang-format off
    frac_adps << 0.0355,   0.0324,   0.0339,   0.02357,
                 0.01611,  0.01700,  0.02178,  0.01559,
                 0.0212,   0.01354,  0.01362,  0.01127,
                -0.0130,  -0.01223, -0.01379, -0.00727,
                 0.0034,   0.00180,  0.00597,  0.00137,
                -0.00430, -0.00492, -0.00708, -0.00323;

    expected << 0.03274,  0.02976,  0.03107,  0.02224,
                0.01621,  0.01636,  0.02034,  0.01559,
                0.02120,  0.01354,  0.01362,  0.01127,
               -0.01027, -0.00979, -0.00973, -0.00487,
                0.00318,  0.00132,  0.00520,  0.00110,
                0.00083, -0.00171, -0.00392, -0.00053;
    // clang-format on

    INFO(fmt::format("original\n{}\n", format_matrix(frac_adps)));

    Mat6N cart_adps = cell.to_cartesian_adp(frac_adps);
    INFO(fmt::format("expected cartesian\n{}\n", format_matrix(expected)));
    INFO(fmt::format("cartesian\n{}\n", format_matrix(cart_adps)));

    Mat6N back_to_frac = cell.to_fractional_adp(cart_adps);
    INFO(fmt::format("back\n{}\n", format_matrix(back_to_frac)));

    // we're only giving 3 decimal places in the frac adps, and 4 in the
    // expected so the tolerance must be a bit coarse
    REQUIRE(expected.isApprox(cart_adps, 1e-3));
    REQUIRE(back_to_frac.isApprox(frac_adps, 1e-10));
  }
}

TEST_CASE("Atom mapping table", "[crystal][unitcell]") {
  using occ::crystal::SiteMappingTable;
  Crystal crystal = acetic_acid_crystal();
  SiteMappingTable atom_table = SiteMappingTable::build_atom_table(crystal);
  const auto &atoms = crystal.unit_cell_atoms();
  const auto &symops = crystal.symmetry_operations();

  SECTION("Table Size") {
    for (size_t i = 0; i < atom_table.size(); ++i) {
      fmt::print("Atom {}:\n", i);
      auto symops = atom_table.get_symmetry_operations(i);
      for (const auto &[symop, offset] : symops) {
        auto target = atom_table.get_target(i, symop, offset);
        if (target) {
          fmt::print("  {} ({} {} {}) -> Atom {}\n",
                     SymmetryOperation(symop).to_string(), offset.h, offset.k,
                     offset.l, *target);
        }
      }
    }
    CHECK(atom_table.size() == atoms.size());
  }

  SECTION("Identity Mapping") {
    for (size_t i = 0; i < atoms.size(); ++i) {
      auto target =
          atom_table.get_target(i, 16484, {0, 0, 0}); // 16484 is identity
      REQUIRE(target.has_value());
      CHECK(*target == i);
    }
  }

  SECTION("Symmetry Operations") {
    for (size_t i = 0; i < atoms.size(); ++i) {
      auto symops_with_offsets = atom_table.get_symmetry_operations(i);
      CHECK(symops_with_offsets.size() == symops.size());
    }
  }

  SECTION("Neighbor Consistency") {
    for (size_t i = 0; i < atoms.size(); ++i) {
      auto neighbors = atom_table.get_neighbors(i);
      auto symops_with_offsets = atom_table.get_symmetry_operations(i);

      CHECK(neighbors.size() == symops_with_offsets.size());

      for (const auto &[symop_int, offset] : symops_with_offsets) {
        auto target = atom_table.get_target(i, symop_int, offset);
        REQUIRE(target.has_value());
        CHECK_THAT(neighbors, Catch::Matchers::VectorContains(*target));
      }
    }
  }

  SECTION("Edge Consistency") {
    for (size_t i = 0; i < atoms.size(); ++i) {
      auto neighbors = atom_table.get_neighbors(i);
      for (const auto &neighbor : neighbors) {
        auto edges = atom_table.get_edges(i, neighbor);
        CHECK(!edges.empty());
        for (const auto &edge : edges) {
          CHECK(edge.source == i);
          CHECK(edge.target == neighbor);
          auto target = atom_table.get_target(i, edge.symop, edge.offset);
          REQUIRE(target.has_value());
          CHECK(*target == neighbor);
        }
      }
    }
  }

  SECTION("Atom Type Preservation") {
    for (size_t i = 0; i < atoms.size(); ++i) {
      auto symops_with_offsets = atom_table.get_symmetry_operations(i);
      for (const auto &[symop_int, offset] : symops_with_offsets) {
        auto target = atom_table.get_target(i, symop_int, offset);
        REQUIRE(target.has_value());
        CHECK(atoms.atomic_numbers(i) == atoms.atomic_numbers(*target));
      }
    }
  }

  SECTION("Atom position consistency") {
    for (size_t i = 0; i < atoms.size(); ++i) {
      auto symops_with_offsets = atom_table.get_symmetry_operations(i);
      for (const auto &[symop_int, offset] : symops_with_offsets) {
        auto target = atom_table.get_target(i, symop_int, offset);
        REQUIRE(target.has_value());
        occ::Vec3 pos = offset.vector() + atoms.frac_pos.col(*target);
        occ::Vec3 pos_symop =
            SymmetryOperation(symop_int).apply(atoms.frac_pos.col(i));
        REQUIRE(pos.isApprox(pos_symop));
        CHECK(atoms.atomic_numbers(i) == atoms.atomic_numbers(*target));
      }
    }
  }
}

using occ::crystal::DimerIndex;
using occ::crystal::HKL;
using occ::crystal::SiteMappingTable;

TEST_CASE("Molecule mapping table", "[crystal][unitcell]") {
  using occ::crystal::SiteMappingTable;
  Crystal crystal = acetic_acid_crystal();
  SiteMappingTable molecule_table =
      SiteMappingTable::build_molecule_table(crystal);
  const auto &molecules = crystal.unit_cell_molecules();
  const auto &symops = crystal.symmetry_operations();

  SECTION("Table Size") {
    // Print out the molecule mapping table
    for (size_t i = 0; i < molecule_table.size(); ++i) {
      fmt::print("Molecule {}:\n", i);
      auto symops = molecule_table.get_symmetry_operations(i);
      for (const auto &[symop, offset] : symops) {
        auto target = molecule_table.get_target(i, symop, offset);
        if (target) {
          fmt::print("  {} ({} {} {}) -> Molecule {}\n",
                     SymmetryOperation(symop).to_string(), offset.h, offset.k,
                     offset.l, *target);
        }
      }
    }
    CHECK(molecule_table.size() == molecules.size());
  }

  SECTION("Identity Mapping") {
    for (size_t i = 0; i < molecules.size(); ++i) {
      auto target =
          molecule_table.get_target(i, 16484, {0, 0, 0}); // 16484 is identity
      REQUIRE(target.has_value());
      CHECK(*target == i);
    }
  }

  SECTION("Symmetry Operations") {
    for (size_t i = 0; i < molecules.size(); ++i) {
      auto symops_with_offsets = molecule_table.get_symmetry_operations(i);
      CHECK(symops_with_offsets.size() == symops.size());
    }
  }

  SECTION("Neighbor Consistency") {
    for (size_t i = 0; i < molecules.size(); ++i) {
      auto neighbors = molecule_table.get_neighbors(i);
      auto symops_with_offsets = molecule_table.get_symmetry_operations(i);
      CHECK(neighbors.size() == symops_with_offsets.size());
      for (const auto &[symop_int, offset] : symops_with_offsets) {
        auto target = molecule_table.get_target(i, symop_int, offset);
        REQUIRE(target.has_value());
        CHECK_THAT(neighbors, Catch::Matchers::VectorContains(*target));
      }
    }
  }

  SECTION("Edge Consistency") {
    for (size_t i = 0; i < molecules.size(); ++i) {
      auto neighbors = molecule_table.get_neighbors(i);
      for (const auto &neighbor : neighbors) {
        auto edges = molecule_table.get_edges(i, neighbor);
        CHECK(!edges.empty());
        for (const auto &edge : edges) {
          CHECK(edge.source == i);
          CHECK(edge.target == neighbor);
          auto target = molecule_table.get_target(i, edge.symop, edge.offset);
          REQUIRE(target.has_value());
          CHECK(*target == neighbor);
        }
      }
    }
  }

  SECTION("Molecule Type Preservation") {
    for (size_t i = 0; i < molecules.size(); ++i) {
      auto symops_with_offsets = molecule_table.get_symmetry_operations(i);
      for (const auto &[symop_int, offset] : symops_with_offsets) {
        auto target = molecule_table.get_target(i, symop_int, offset);
        REQUIRE(target.has_value());
        CHECK(molecules[i].is_equivalent_to(molecules[*target]));
      }
    }
  }

  SECTION("Centroid Consistency") {
    const double tolerance = 1e-6;
    for (size_t i = 0; i < molecules.size(); ++i) {
      auto symops_with_offsets = molecule_table.get_symmetry_operations(i);
      for (const auto &[symop_int, offset] : symops_with_offsets) {
        auto target = molecule_table.get_target(i, symop_int, offset);
        REQUIRE(target.has_value());

        Vec3 centroid_i = crystal.to_fractional(molecules[i].centroid());
        Vec3 centroid_target =
            crystal.to_fractional(molecules[*target].centroid());

        Vec3 transformed_centroid =
            SymmetryOperation(symop_int).apply(centroid_i);
        transformed_centroid += Vec3(offset.h, offset.k, offset.l);

        // Wrap the transformed centroid back to the unit cell
        for (int j = 0; j < 3; ++j) {
          transformed_centroid[j] -= std::floor(transformed_centroid[j]);
        }

        CHECK((transformed_centroid - centroid_target).norm() < tolerance);
      }
    }
  }
}

using occ::crystal::DimerIndex;
using occ::crystal::DimerMappingTable;
using occ::crystal::SiteIndex;

DimerIndex make_dimer(int a_offset, int a_h, int a_k, int a_l, int b_offset,
                      int b_h, int b_k, int b_l) {
  return DimerIndex{SiteIndex{a_offset, HKL{a_h, a_k, a_l}},
                    SiteIndex{b_offset, HKL{b_h, b_k, b_l}}};
}

TEST_CASE("DimerMappingTable construction and basic properties",
          "[crystal][dimer]") {
  Crystal crystal = acetic_acid_crystal();
  auto dimers = crystal.unit_cell_dimers(3.8); // Assuming this method exists

  SECTION("Construction without inversion") {
    DimerMappingTable table =
        DimerMappingTable::build_dimer_table(crystal, dimers, false);

    REQUIRE(table.unique_dimers().size() == 56);
    REQUIRE(table.symmetry_unique_dimers().size() == 14);
  }

  SECTION("Construction with inversion") {
    DimerMappingTable table =
        DimerMappingTable::build_dimer_table(crystal, dimers, true);

    REQUIRE(table.unique_dimers().size() == 28);
    REQUIRE(table.symmetry_unique_dimers().size() == 7);
  }
}

TEST_CASE("DimerMappingTable symmetry_unique_dimer", "[crystal][dimer]") {
  Crystal crystal = acetic_acid_crystal();
  auto dimers = crystal.unit_cell_dimers(3.8);
  DimerMappingTable table =
      DimerMappingTable::build_dimer_table(crystal, dimers, true);

  SECTION("Known dimer") {
    DimerIndex test_dimer = make_dimer(0, 0, 0, 0, 1, -1, 0, -1);
    DimerIndex expected_unique = make_dimer(0, 0, 0, 0, 1, -1, 0, -1);

    REQUIRE(table.symmetry_unique_dimer(test_dimer) == expected_unique);
  }

  SECTION("Symmetry-related dimer") {
    DimerIndex test_dimer = make_dimer(1, 0, 0, 0, 0, 1, 0, 1);
    DimerIndex expected_unique = make_dimer(0, 0, 0, 0, 1, -1, 0, -1);

    REQUIRE(table.symmetry_unique_dimer(test_dimer) == expected_unique);
  }

  SECTION("Unknown dimer") {
    DimerIndex test_dimer = make_dimer(0, 0, 0, 0, 3, 2, 2, 2);

    REQUIRE(table.symmetry_unique_dimer(test_dimer) == test_dimer);
  }
}

TEST_CASE("DimerMappingTable symmetry_related_dimers", "[crystal][dimer]") {
  Crystal crystal = acetic_acid_crystal();
  auto dimers = crystal.unit_cell_dimers(3.8);
  DimerMappingTable table =
      DimerMappingTable::build_dimer_table(crystal, dimers, false);

  SECTION("Known dimer") {
    DimerIndex test_dimer = make_dimer(0, 0, 0, 0, 1, -1, 0, -1);
    auto related_dimers = table.symmetry_related_dimers(test_dimer);

    REQUIRE(related_dimers.size() == 4);
    REQUIRE(std::find(related_dimers.begin(), related_dimers.end(),
                      make_dimer(0, 0, 0, 0, 1, -1, 0, -1)) !=
            related_dimers.end());
    REQUIRE(std::find(related_dimers.begin(), related_dimers.end(),
                      make_dimer(1, 0, 0, 0, 0, 1, 0, 0)) !=
            related_dimers.end());
  }

  SECTION("Unknown dimer") {
    DimerIndex test_dimer = make_dimer(0, 0, 0, 0, 3, 2, 2, 2);
    auto related_dimers = table.symmetry_related_dimers(test_dimer);

    REQUIRE(related_dimers.size() == 1);
    REQUIRE(related_dimers[0] == test_dimer);
  }
}

TEST_CASE("DimerMappingTable with inversion", "[crystal][dimer]") {
  Crystal crystal = acetic_acid_crystal();
  auto dimers = crystal.unit_cell_dimers(3.8);
  DimerMappingTable table =
      DimerMappingTable::build_dimer_table(crystal, dimers, true);

  SECTION("Inverted dimer") {
    DimerIndex dimer = make_dimer(0, 0, 0, 0, 1, -1, 0, -1);
    DimerIndex inverted_dimer = make_dimer(1, 0, 0, 0, 0, 1, 0, 1);

    REQUIRE(table.symmetry_unique_dimer(dimer) ==
            table.symmetry_unique_dimer(inverted_dimer));
  }

  SECTION("Symmetry-related dimers with inversion") {
    DimerIndex dimer = make_dimer(0, 0, 0, 0, 1, -1, 0, -1);
    auto related_dimers = table.symmetry_related_dimers(dimer);

    REQUIRE(related_dimers.size() == 4);
    REQUIRE(std::find(related_dimers.begin(), related_dimers.end(),
                      make_dimer(0, 0, 0, 0, 1, -1, 0, -1)) !=
            related_dimers.end());
  }
}

TEST_CASE("DimerMappingTable consistency", "[crystal][dimer]") {
  Crystal crystal = acetic_acid_crystal();
  auto dimers = crystal.unit_cell_dimers(3.8);
  DimerMappingTable table_no_inv =
      DimerMappingTable::build_dimer_table(crystal, dimers, false);
  DimerMappingTable table_inv =
      DimerMappingTable::build_dimer_table(crystal, dimers, true);

  SECTION("Symmetry-unique dimers are consistent") {
    REQUIRE(table_no_inv.symmetry_unique_dimers().size() ==
            2 * table_inv.symmetry_unique_dimers().size());

    for (const auto &dimer : table_inv.symmetry_unique_dimers()) {
      const auto &sym = table_no_inv.symmetry_unique_dimers();
      REQUIRE(std::find(sym.begin(), sym.end(), dimer) != sym.end());
    }
  }

  SECTION("All unique dimers map to symmetry-unique dimers") {
    for (const auto &dimer : table_no_inv.unique_dimers()) {
      const auto &sym = table_no_inv.symmetry_unique_dimers();
      REQUIRE(std::find(sym.begin(), sym.end(),
                        table_no_inv.symmetry_unique_dimer(dimer)) !=
              sym.end());
    }

    for (const auto &dimer : table_inv.unique_dimers()) {
      const auto &sym = table_no_inv.symmetry_unique_dimers();
      REQUIRE(std::find(sym.begin(), sym.end(),
                        table_inv.symmetry_unique_dimer(dimer)) != sym.end());
    }
  }
}

auto ibuprofen_asym() {
  const std::vector<std::string> labels = {
      "O1",  "O2",  "C1",  "C2",  "C3",  "C4",  "C5",  "C6",  "C7",
      "C8",  "C9",  "C10", "C11", "C12", "C13", "H1",  "H2",  "H3",
      "H4",  "H5",  "H6",  "H7",  "H8",  "H9",  "H10", "H11", "H12",
      "H13", "H14", "H15", "H16", "H17", "H18"};

  occ::IVec nums(labels.size());
  occ::Mat positions(labels.size(), 3);
  for (size_t i = 0; i < labels.size(); i++) {
    nums(i) = occ::core::Element(labels[i]).atomic_number();
  }

  positions << 0.3792, 0.4968, 0.4148, // O1
      0.4969, 0.3117, 0.4375,          // O2
      0.4162, 0.3483, 0.3962,          // C1
      0.3501, 0.2207, 0.3217,          // C2
      0.3996, 0.1089, 0.2383,          // C3
      0.3038, 0.1238, 0.4203,          // C4
      0.2173, 0.1726, 0.4466,          // C5
      0.1760, 0.0902, 0.5385,          // C6
      0.2203, -0.0484, 0.6065,         // C7
      0.3079, -0.0965, 0.5821,         // C8
      0.3486, -0.0127, 0.4879,         // C9
      0.1741, -0.1385, 0.7045,         // C10
      0.0970, -0.2639, 0.6467,         // C11
      0.0414, -0.3210, 0.7486,         // C12
      0.1373, -0.4152, 0.5836,         // C13
      0.4256, 0.5679, 0.4704,          // H1
      0.2952, 0.2914, 0.2604,          // H2
      0.4572, 0.0401, 0.2957,          // H3
      0.4308, 0.1831, 0.1693,          // H4
      0.3509, 0.0167, 0.1870,          // H5
      0.1819, 0.2809, 0.3955,          // H6
      0.1070, 0.1277, 0.5567,          // H7
      0.3465, -0.1990, 0.6368,         // H8
      0.4178, -0.0546, 0.4737,         // H9
      0.2261, -0.2096, 0.7703,         // H10
      0.1460, -0.0447, 0.7637,         // H11
      0.0511, -0.1938, 0.5723,         // H12
      0.0874, -0.3850, 0.8263,         // H13
      0.0095, -0.2138, 0.7903,         // H14
      -0.0156, -0.4021, 0.7030,        // H15
      0.1805, -0.3744, 0.5143,         // H16
      0.0808, -0.4953, 0.5352,         // H17
      0.1822, -0.4898, 0.6561;         // H18

  return AsymmetricUnit(positions.transpose(), nums, labels);
}

TEST_CASE("Ibuprofen unit cell molecules basic properties",
          "[crystal][ibuprofen]") {
  AsymmetricUnit asym = ibuprofen_asym();
  SpaceGroup sg(14); // P21/c
  UnitCell cell = occ::crystal::monoclinic_cell(14.397, 7.818, 10.506,
                                                occ::units::radians(99.7));
  Crystal ibuprofen(asym, sg, cell);

  SECTION("Basic molecule count") {
    const auto &uc_mols = ibuprofen.unit_cell_molecules();
    REQUIRE(uc_mols.size() == 4); // P21/c should give 4 molecules in unit cell
    REQUIRE(ibuprofen.symmetry_unique_molecules().size() ==
            1); // One unique molecule
  }

  SECTION("Molecule sizes") {
    const auto &uc_mols = ibuprofen.unit_cell_molecules();
    for (const auto &mol : uc_mols) {
      REQUIRE(mol.size() == 33); // Each ibuprofen has 33 atoms
    }
  }
}

TEST_CASE("Ibuprofen gamma point molecule centering", "[crystal][ibuprofen]") {
  AsymmetricUnit asym = ibuprofen_asym();
  SpaceGroup sg(14);
  UnitCell cell = occ::crystal::monoclinic_cell(14.397, 7.818, 10.506,
                                                occ::units::radians(99.7));

  SECTION("Default behavior - no gamma point centering") {
    Crystal ibuprofen(asym, sg, cell);
    ibuprofen.set_gamma_point_unit_cell_molecules(false);
    const auto &uc_mols = ibuprofen.unit_cell_molecules();

    // Check centroids are preserved without enforced centering
    std::vector<Vec3> centroids;
    for (const auto &mol : uc_mols) {
      Vec3 frac_centroid = ibuprofen.to_fractional(mol.centroid());
      centroids.push_back(frac_centroid);
    }

    // Original centroids should be related by symmetry operations
    const auto &symops = ibuprofen.symmetry_operations();
    REQUIRE(centroids.size() == 4);
  }

  SECTION("Gamma point centering enabled") {
    Crystal ibuprofen(asym, sg, cell);
    ibuprofen.set_gamma_point_unit_cell_molecules(true);
    const auto &uc_mols = ibuprofen.unit_cell_molecules();

    // All molecule centroids should be in [0,1) range
    for (const auto &mol : uc_mols) {
      Vec3 frac_centroid = ibuprofen.to_fractional(mol.centroid());
      INFO("Fractional centroid: " << frac_centroid.transpose());

      for (int i = 0; i < 3; i++) {
        REQUIRE(frac_centroid[i] >= 0.0);
        REQUIRE(frac_centroid[i] < 1.0);
      }
    }
  }
}

TEST_CASE("Ibuprofen molecule connectivity preservation",
          "[crystal][ibuprofen]") {
  AsymmetricUnit asym = ibuprofen_asym();
  SpaceGroup sg(14);
  UnitCell cell = occ::crystal::monoclinic_cell(14.397, 7.818, 10.506,
                                                occ::units::radians(99.7));

  SECTION("Connectivity with and without gamma point centering") {
    Crystal ibuprofen1(asym, sg, cell);
    // Get molecules without centering
    ibuprofen1.set_gamma_point_unit_cell_molecules(false);
    const auto mols_without = ibuprofen1.unit_cell_molecules();

    // Get molecules with centering
    Crystal ibuprofen2(asym, sg, cell);
    ibuprofen2.set_gamma_point_unit_cell_molecules(true);
    const auto mols_with = ibuprofen2.unit_cell_molecules();

    REQUIRE(mols_without.size() == mols_with.size());

    // Check that molecule graphs are preserved
    for (size_t i = 0; i < mols_without.size(); i++) {
      const auto &mol1 = mols_without[i];
      const auto &mol2 = mols_with[i];

      // Check atom types match
      REQUIRE(mol1.atomic_numbers() == mol2.atomic_numbers());

      Mat3N pos1 = mol1.positions();
      Mat3N pos2 = mol2.positions();

      for (int j = 0; j < pos1.cols(); j++) {
        for (int k = j + 1; k < pos1.cols(); k++) {
          double d1 = (pos1.col(j) - pos1.col(k)).norm();
          double d2 = (pos2.col(j) - pos2.col(k)).norm();
          REQUIRE(d1 == Catch::Approx(d2).margin(1e-10));
        }
      }
    }
  }
}

TEST_CASE("Ibuprofen dimer generation", "[crystal][ibuprofen]") {
  AsymmetricUnit asym = ibuprofen_asym();
  SpaceGroup sg(14);
  UnitCell cell = occ::crystal::monoclinic_cell(14.397, 7.818, 10.506,
                                                occ::units::radians(99.7));

  SECTION("Dimer count consistency") {
    // Get dimers with and without gamma point centering
    Crystal ibuprofen1(asym, sg, cell);
    ibuprofen1.set_gamma_point_unit_cell_molecules(false);
    auto dimers_without = ibuprofen1.symmetry_unique_dimers(5.0);

    Crystal ibuprofen2(asym, sg, cell);
    ibuprofen2.set_gamma_point_unit_cell_molecules(true);
    auto dimers_with = ibuprofen2.symmetry_unique_dimers(5.0);

    // Number of unique dimers should be same
    REQUIRE(dimers_without.unique_dimers.size() ==
            dimers_with.unique_dimers.size());

    // Check dimer distances match
    for (size_t i = 0; i < dimers_without.unique_dimers.size(); i++) {
      const auto &d1 = dimers_without.unique_dimers[i];
      const auto &d2 = dimers_with.unique_dimers[i];
      REQUIRE(d1.nearest_distance() ==
              Catch::Approx(d2.nearest_distance()).margin(1e-10));
    }
  }
}

TEST_CASE("Ibuprofen dimer generation debug", "[crystal][ibuprofen]") {
  AsymmetricUnit asym = ibuprofen_asym();
  SpaceGroup sg(14);
  UnitCell cell = occ::crystal::monoclinic_cell(14.397, 7.818, 10.506,
                                                occ::units::radians(99.7));

  SECTION("Compare dimer properties") {
    Crystal ibuprofen1(asym, sg, cell);
    ibuprofen1.set_gamma_point_unit_cell_molecules(false);
    auto dimers_without = ibuprofen1.symmetry_unique_dimers(5.0);

    Crystal ibuprofen2(asym, sg, cell);
    ibuprofen2.set_gamma_point_unit_cell_molecules(true);
    auto dimers_with = ibuprofen2.symmetry_unique_dimers(5.0);

    fmt::print("\nWithout gamma point centering ({} dimers):\n",
               dimers_without.unique_dimers.size());
    for (size_t i = 0; i < dimers_without.unique_dimers.size(); i++) {
      const auto &d = dimers_without.unique_dimers[i];
      fmt::print("Dimer {}: dist={:.4f}, symop={}\n", i, d.nearest_distance(),
                 ibuprofen1.dimer_symmetry_string(d));
    }

    fmt::print("\nWith gamma point centering ({} dimers):\n",
               dimers_with.unique_dimers.size());
    for (size_t i = 0; i < dimers_with.unique_dimers.size(); i++) {
      const auto &d = dimers_with.unique_dimers[i];
      fmt::print("Dimer {}: dist={:.4f}, symop={}\n", i, d.nearest_distance(),
                 ibuprofen2.dimer_symmetry_string(d));
    }

    // Print molecule centroids for comparison
    fmt::print("\nMolecule centroids without centering:\n");
    for (const auto &mol : ibuprofen1.unit_cell_molecules()) {
      Vec3 frac_centroid = ibuprofen1.to_fractional(mol.centroid());
      fmt::print("Molecule {}: ({:.4f}, {:.4f}, {:.4f})\n",
                 mol.unit_cell_molecule_idx(), frac_centroid[0],
                 frac_centroid[1], frac_centroid[2]);
    }

    fmt::print("\nMolecule centroids with centering:\n");
    for (const auto &mol : ibuprofen2.unit_cell_molecules()) {
      Vec3 frac_centroid = ibuprofen2.to_fractional(mol.centroid());
      fmt::print("Molecule {}: ({:.4f}, {:.4f}, {:.4f})\n",
                 mol.unit_cell_molecule_idx(), frac_centroid[0],
                 frac_centroid[1], frac_centroid[2]);
    }

    // Compare specific dimer properties
    std::vector<std::pair<double, std::string>> dimer_props_without;
    std::vector<std::pair<double, std::string>> dimer_props_with;

    for (const auto &d : dimers_without.unique_dimers) {
      dimer_props_without.push_back(
          {d.nearest_distance(), ibuprofen1.dimer_symmetry_string(d)});
    }

    for (const auto &d : dimers_with.unique_dimers) {
      dimer_props_with.push_back(
          {d.nearest_distance(), ibuprofen2.dimer_symmetry_string(d)});
    }

    // Sort both sets by distance and symmetry string for comparison
    auto sort_fn = [](const auto &a, const auto &b) {
      if (std::abs(a.first - b.first) < 1e-6) {
        return a.second < b.second;
      }
      return a.first < b.first;
    };

    std::sort(dimer_props_without.begin(), dimer_props_without.end(), sort_fn);
    std::sort(dimer_props_with.begin(), dimer_props_with.end(), sort_fn);

    fmt::print("\nSorted unique properties:\n");
    fmt::print("Without centering:\n");
    for (const auto &[dist, symop] : dimer_props_without) {
      fmt::print("{:.4f} {}\n", dist, symop);
    }
    fmt::print("\nWith centering:\n");
    for (const auto &[dist, symop] : dimer_props_with) {
      fmt::print("{:.4f} {}\n", dist, symop);
    }

    REQUIRE(dimers_without.unique_dimers.size() ==
            dimers_with.unique_dimers.size());
  }
}

TEST_CASE("Cell shift computation", "[crystal]") {
  SECTION("simple positive differences") {
    Vec3 uc_center(1.7, 0.2, 0.8);
    Vec3 asym_center(0.0, 0.0, 0.0);
    SymmetryOperation identity("x,y,z");
    IVec3 shift = Crystal::compute_cell_shift(uc_center, asym_center, identity);
    REQUIRE(shift(0) == 2);
    REQUIRE(shift(1) == 0);
    REQUIRE(shift(2) == 1);
  }

  SECTION("simple negative differences") {
    Vec3 uc_center(-1.7, -0.2, -0.8);
    Vec3 asym_center(0.0, 0.0, 0.0);
    SymmetryOperation identity("x,y,z");
    IVec3 shift = Crystal::compute_cell_shift(uc_center, asym_center, identity);
    REQUIRE(shift(0) == -2);
    REQUIRE(shift(1) == 0);
    REQUIRE(shift(2) == -1);
  }

  SECTION("mixed positive/negative with symmetry") {
    Crystal crystal = acetic_acid_crystal();
    const auto &uc_mols = crystal.unit_cell_molecules();
    const auto &asym_mols = crystal.symmetry_unique_molecules();
    const auto &uc_mol = uc_mols[2];
    Vec3 uc_center = crystal.to_fractional(uc_mol.centroid());
    Vec3 asym_center = crystal.to_fractional(
        asym_mols[uc_mol.asymmetric_molecule_idx()].centroid());
    SymmetryOperation symop(uc_mol.asymmetric_unit_symop()(0));
    IVec3 shift = Crystal::compute_cell_shift(uc_center, asym_center, symop);
    REQUIRE(shift == uc_mol.cell_shift());
  }
}
