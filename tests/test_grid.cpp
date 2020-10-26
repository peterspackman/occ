#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "lebedev.h"
#include "dft_grid.h"
#include <fmt/ostream.h>
#include <libint2/atom.h>

TEST_CASE("lebedev", "[grid]")
{
    auto grid = tonto::grid::lebedev(110);
    fmt::print("grid:\n{}\n", grid);
    REQUIRE(grid.rows() == 110);
    REQUIRE(grid.cols() == 4);

    /*
    BENCHMARK("Lebedev 86") {
        return tonto::grid::lebedev(86);
    };

    BENCHMARK("Lebedev 590") {
        return tonto::grid::lebedev(590);
    };

    BENCHMARK("Lebedev 5810") {
        return tonto::grid::lebedev(5810);
    };
    */
}

TEST_CASE("radial grid", "[grid]")
{
    auto radial = tonto::dft::generate_becke_radial_grid(10);
    fmt::print("Radial grid:\n{}\n", radial.points);

    /*
    BENCHMARK("Becke radial 10") {
        return tonto::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 50") {
        return tonto::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 80") {
        return tonto::dft::generate_becke_radial_grid(10);
    };
    */
}

TEST_CASE("atom grid", "[grid]")
{
    auto atom = tonto::dft::generate_atom_grid(1);
    fmt::print("Atom grid:\n{}\n", atom.points.transpose());
/*
    BENCHMARK("Atom Carbon") {
        return tonto::dft::generate_atom_grid(6);
    };

    BENCHMARK("Atom Carbon 590") {
        return tonto::dft::generate_atom_grid(6, 590);
    };
*/
    std::vector<libint2::Atom> atoms {
        {6, -1.0478252000, -1.4216736000, 0.0000000000},
        {6, -1.4545034000, -0.8554459000, 1.2062048000},
        {6, -1.4545034000, -0.8554459000, -1.2062048000},
        {6, -2.2667970000, 0.2771610000, 1.2069539000},
        {7, -2.6714781000, 0.8450211000, 0.0000000000},
        {6, -2.2667970000, 0.2771610000, -1.2069539000},
        {1, -1.1338534000, -1.2920593000, -2.1423150000},
        {1, -2.5824943000, 0.7163066000, -2.1437977000},
        {1, -3.3030422000, 1.7232700000, 0.0000000000},
        {1, -2.5824943000, 0.7163066000, 2.1437977000},
        {1, -1.1338534000, -1.2920593000, 2.1423150000},
        {1, -0.4060253000, -2.2919049000, 0.0000000000}
    };
    tonto::dft::MolecularGrid mgrid(atoms);

    auto grid = mgrid.generate_partitioned_atom_grid(0);
    fmt::print("{}\n{}\n", grid.points, grid.atomic_number);
}

