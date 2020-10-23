#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "lebedev.h"
#include "dft_grid.h"
#include <fmt/ostream.h>

TEST_CASE("lebedev", "[grid]")
{
    auto grid = tonto::grid::lebedev(110);
    fmt::print("grid:\n{}\n", grid);
    REQUIRE(grid.rows() == 110);
    REQUIRE(grid.cols() == 4);

    BENCHMARK("Lebedev 86") {
        return tonto::grid::lebedev(86);
    };

    BENCHMARK("Lebedev 590") {
        return tonto::grid::lebedev(590);
    };

    BENCHMARK("Lebedev 5810") {
        return tonto::grid::lebedev(5810);
    };
}

TEST_CASE("radial grid", "[grid]")
{
    auto radial = tonto::dft::generate_becke_radial_grid(10);
    fmt::print("Radial grid:\n{}\n", radial.points);

    BENCHMARK("Becke radial 10") {
        return tonto::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 50") {
        return tonto::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 80") {
        return tonto::dft::generate_becke_radial_grid(10);
    };
}

TEST_CASE("atom grid", "[grid]")
{
    auto atom = tonto::dft::generate_atom_grid(1);
    fmt::print("Atom grid:\n{}\n", atom.points.transpose());

    BENCHMARK("Atom Carbon") {
        return tonto::dft::generate_atom_grid(6);
    };

    BENCHMARK("Atom Carbon 590") {
        return tonto::dft::generate_atom_grid(6, 590);
    };

}
