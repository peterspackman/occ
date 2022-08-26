#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/surface.h>
#include <occ/crystal/symmetryoperation.h>

using occ::Mat3N;
using occ::MatN3;
using occ::crystal::AsymmetricUnit;
using occ::crystal::Crystal;
using occ::crystal::MillerIndex;
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
}

TEST_CASE("ice_ii molecules", "[crystal]") {
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

TEST_CASE("acetic molecules", "[crystal]") {
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
        size_t j = 0;
        for (const auto &dimer : n) {
            auto s_ab = dimer_symop(dimer, acetic);
            fmt::print("R = {:.3f}, symop = {}, unique_idx = {}\n",
                       dimer.nearest_distance(), s_ab.to_string(),
                       crystal_dimers.unique_dimer_idx[i][j]);
            j++;
        }
    }
}

TEST_CASE("acetic supercell", "[crystal]") {
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

TEST_CASE("Spacegroup symops", "[space_group]") {
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

TEST_CASE("Surface constructor", "[crystal, surface]") {
    Crystal a = acetic_acid_crystal();
    MillerIndex m{0, 1, 0};
    Surface surf(m, a);
    surf.print();
}

TEST_CASE("Surface generation", "[crystal, surface]") {
    Crystal a = acetic_acid_crystal();
    occ::timing::StopWatch sw;
    sw.start();
    auto surfaces = occ::crystal::generate_surfaces(a, 0.1);
    sw.stop();
    size_t n = 0;
    fmt::print("Top 10 surfaces\n");
    for (const auto &surf : surfaces) {
        surf.print();
        n++;
        if (n > 10)
            break;
    }
    fmt::print("Generation took {} s\n", sw.read());
}

// SymmetryOperation

TEST_CASE("SymmetryOperation constructor", "[symmetry_operation]") {
    REQUIRE(SymmetryOperation("x,y,z").is_identity());
    REQUIRE(SymmetryOperation(16484).is_identity());
}

TEST_CASE("SymmetryOperation seitz", "[symmetry_operation]") {
    auto id = SymmetryOperation(16484);
    REQUIRE(all_close(id.seitz(), Eigen::Matrix4d::Identity()));
    REQUIRE(all_close(id.rotation(), id.inverted().rotation()));
}