#include <tonto/core/dimer.h>
#include <tonto/core/util.h>
#include <fmt/ostream.h>
#include "catch.hpp"

using tonto::chem::Dimer;
using tonto::chem::Molecule;
using tonto::util::all_close;

TEST_CASE("Dimer constructor", "[dimer]")
{
    tonto::Mat3N pos(3, 3), pos2(3, 3);
    tonto::IVec nums(3);
    nums << 8, 1, 1;
    pos << -1.32695761, -1.93166418, 0.48664409,
           -0.10593856,  1.60017351, 0.07959806,
            0.01878821, -0.02171049, 0.00986248;

    Molecule m(nums, pos);

    auto masses = m.atomic_masses();
    tonto::Vec3 expected_masses = {15.994, 1.00794, 1.00794};

    fmt::print("Atomic masses:\n{}\n\n", masses);
    REQUIRE(all_close(masses, expected_masses, 1e-3, 1e-3));

}


TEST_CASE("Dimer transform", "[dimer]")
{
    tonto::Mat3N pos(3, 3), pos2(3, 3);
    tonto::IVec nums(3);
    nums << 8, 1, 1;
    pos << -1.32695761, -1.93166418, 0.48664409,
           -0.10593856,  1.60017351, 0.07959806,
            0.01878821, -0.02171049, 0.00986248;
    pos2 = pos;
    pos2.topRows(1).array() *= -1;

    Molecule m(nums, pos), m2(nums, pos2);

    Dimer dim(m, m2);

    fmt::print("m1\n{}\n", m.positions());
    fmt::print("m2\n{}\n", m2.positions());

    auto transform = dim.symmetry_relation().value();

    fmt::print("Transform matrix:\n{}\n", transform);
    m.transform(transform, Molecule::Origin::Centroid);
    fmt::print("m1\n{}\n", m.positions());
    fmt::print("m2\n{}\n", m2.positions());
    REQUIRE(all_close(m.positions(), m2.positions(), 1e-5, 1e-5));
}

TEST_CASE("Dimer separations", "[dimer]")
{
    tonto::Mat3N pos(3, 3), pos2(3, 3);
    tonto::IVec nums(3);
    nums << 8, 1, 1;
    pos << -1.32695761, -1.93166418, 0.48664409,
           -0.10593856,  1.60017351, 0.07959806,
            0.01878821, -0.02171049, 0.00986248;
    pos2 = pos;
    pos2.topRows(1).array() *= -1;

    Molecule m(nums, pos), m2(nums, pos2);

    Dimer dim(m, m2);

    REQUIRE(dim.nearest_distance() == Approx(0.8605988136));
    REQUIRE(dim.centroid_distance() == Approx(1.8479851333));
    REQUIRE(dim.center_of_mass_distance() == Approx(2.5186418514));
}
