#include <occ/core/point_group.h>
#include <occ/core/util.h>
#include <fmt/ostream.h>
#include "catch.hpp"

using occ::chem::Molecule;
using occ::core::MolecularPointGroup;
using occ::core::SymOp;
using occ::util::all_close;

TEST_CASE("Symop constructors", "[point_group]") {
    occ::Vec3 axis(0.0, 1.0, 0.0);
    double angle = 90.0;
    occ::Vec3 rotvec = angle * axis;
    auto s = SymOp::from_axis_angle(axis, angle);
    fmt::print("Transformation:\n{}\n", s.transformation);
    auto s2 = SymOp::from_rotation_vector(rotvec);
    fmt::print("Transformation:\n{}\n", s.transformation);
}

TEST_CASE("Water: C2v", "[point_group]")
{
    occ::Mat3N pos(3, 3);
    occ::IVec nums(3);
    nums << 8, 1, 1;
    pos << -0.7021961, -0.0560603,  0.0099423,
           -1.0221932,  0.8467758, -0.0114887,
            0.2575211,  0.0421215,  0.0052190;

    Molecule m(nums, pos.transpose());

    MolecularPointGroup pg(m);

    fmt::print("Water group: {}\n", pg.point_group_string());
    REQUIRE(pg.point_group() == occ::core::PointGroup::C2v);
    REQUIRE(pg.symmetry_number() == 2);
}

TEST_CASE("Oxygen: Dooh", "[point_group]")
{
    occ::Mat3N pos(3, 2);
    occ::IVec nums(2);
    nums << 8, 8;
    pos << -0.616, 0.616,
            0, 0,
            0, 0;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);

    fmt::print("Oxygen group: {}\n", pg.point_group_string());
    for(const auto& sym: pg.symops()) {
        fmt::print("symop:\n{}\n", sym.transformation);
    }
    for(const auto& sym: pg.rotational_symmetries()) {
        fmt::print("rotational symmetry: {}\n", sym.second);
    }

    REQUIRE(pg.point_group() == occ::core::PointGroup::Dooh);
    REQUIRE(pg.symmetry_number() == 2);
}

TEST_CASE("BF3: D3h", "[point_group]")
{
    occ::Mat3N pos(3, 4);
    occ::IVec nums(4);
    nums << 5, 9, 9, 9;
    pos <<  0.0,  0.0000,  0.8121, -0.8121,
            0.0, -0.9377,  0.4689,  0.4689,
            0.0,  0.0000,  0.0000,  0.0000;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);

    fmt::print("BF3 group: {}\n", pg.point_group_string());
    for(const auto& sym: pg.symops()) {
        fmt::print("symop:\n{}\n", sym.transformation);
    }
    for(const auto& sym: pg.rotational_symmetries()) {
        fmt::print("rotational symmetry: {}\n", sym.second);
    }
    REQUIRE(pg.point_group() == occ::core::PointGroup::D3h);
    REQUIRE(pg.symmetry_number() == 6);
}

TEST_CASE("Benzene: D6h", "[point_group]")
{
    occ::Mat3N pos(3, 12);
    occ::IVec nums(12);
    nums << 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1;
    pos << -0.71308, -0.00904, 1.39904, 2.10308, 1.39904, -0.00904, -1.73551, -0.52025, 1.91025, 3.12551, 1.91025, -0.52025,
            1.20378, -0.01566, -0.01566, 1.20378, 2.42321, 2.42321, 1.20378, -0.90111, -0.90111, 1.20378, 3.30866, 3.30866,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);

    fmt::print("Benzene group: {}\n", pg.point_group_string());
    REQUIRE(pg.point_group() == occ::core::PointGroup::D6h);
}

TEST_CASE("CH4: Td", "[point_group]")
{
    occ::Mat3N pos(3, 5);
    occ::IVec nums(5);
    nums << 6, 1, 1, 1, 1;
    pos <<  0.0,  0.00,  1.026719, -0.513360, -0.513360,
            0.0,  0.00,  0.000000, -0.889165,  0.889165,
            0.0,  1.08, -0.363000, -0.363000, -0.363000;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);

    fmt::print("CH4 group: {}\n", pg.point_group_string());
    REQUIRE(pg.point_group() == occ::core::PointGroup::Td);

}

TEST_CASE("Cube: Oh", "[point_group]")
{
    occ::Mat3N pos(3, 8);
    occ::IVec nums(8);

    nums.setConstant(6);
    pos << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
           0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
           0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);
    fmt::print("Cube group: {}\n", pg.point_group_string());
    for(const auto& sym: pg.symops()) {
        fmt::print("symop:\n{}\n", sym.transformation);
    }
    for(const auto& sym: pg.rotational_symmetries()) {
        fmt::print("rotational symmetry: {}\n", sym.second);
    }

    REQUIRE(pg.point_group() == occ::core::PointGroup::Oh);
}

TEST_CASE("CHFBrCl: C1", "[point_group]")
{
    occ::Mat3N pos(3, 5);
    occ::IVec nums(5);

    nums << 6, 1, 9, 35, 17;
    pos <<  0.000000, 0.000000,  1.026719, -0.513360, -0.513360,
            0.000000, 0.000000,  0.000000, -0.889165,  0.889165,
            0.000000, 1.080000, -0.363000, -0.363000, -0.363000;

    Molecule m(nums, pos);

    MolecularPointGroup pg(m);
    fmt::print("CHFBrCl group: {}\n", pg.point_group_string());
    REQUIRE(pg.point_group() == occ::core::PointGroup::C1);
}
