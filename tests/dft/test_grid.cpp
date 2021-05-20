#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include <occ/dft/lebedev.h>
#include <occ/dft/grid.h>
#include <fmt/ostream.h>
#include <occ/qm/basisset.h>
#include <occ/core/util.h>

using occ::util::all_close;

TEST_CASE("lebedev", "[grid]")
{
    auto grid = occ::grid::lebedev(110);
    fmt::print("grid:\n{}\n", grid);
    REQUIRE(grid.rows() == 110);
    REQUIRE(grid.cols() == 4);

    /*
    BENCHMARK("Lebedev 86") {
        return occ::grid::lebedev(86);
    };0

    BENCHMARK("Lebedev 590") {
        return occ::grid::lebedev(590);
    };

    BENCHMARK("Lebedev 5810") {
        return occ::grid::lebedev(5810);
    };
    */
}

TEST_CASE("Becke radial grid", "[radial]")
{
    auto radial = occ::dft::generate_becke_radial_grid(3, 0.6614041435977716);
    occ::Vec3 expected_pts{9.21217133, 0.66140414, 0.04748668};
    occ::Vec3 expected_weights{77.17570606, 1.3852416, 0.39782349};
    fmt::print("Becke radial grid:\n{} == {}\n{} == {}\n", radial.points.transpose(), expected_pts.transpose(), radial.weights.transpose(), expected_weights.transpose());

    REQUIRE(all_close(radial.points, expected_pts, 1e-5));
    REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
    /*
    BENCHMARK("Becke radial 10") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 50") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 80") {
        return occ::dft::generate_becke_radial_grid(10);
    };
    */
}

TEST_CASE("Gauss-Chebyshev radial grid", "[radial]")
{
    auto radial = occ::dft::generate_gauss_chebyshev_radial_grid(3);
    occ::Vec3 expected_pts{8.66025404e-01, 6.123234e-17, -8.66025404e-01};
    occ::Vec3 expected_weights{1.04719755, 1.04719755, 1.04719755};
    fmt::print("Gauss-Chebyshev radial grid:\n{} == {}\n{} == {}\n", radial.points.transpose(), expected_pts.transpose(), radial.weights.transpose(), expected_weights.transpose());

    REQUIRE(all_close(radial.points, expected_pts, 1e-5));
    REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
}


TEST_CASE("Mura-Knowles radial grid", "[radial]")
{
    auto radial = occ::dft::generate_mura_knowles_radial_grid(3, 1);
    occ::Vec3 expected_pts{0.02412997, 0.69436324, 4.49497829};
    occ::Vec3 expected_weights{0.14511628, 1.48571429, 8.57142857};
    fmt::print("Mura-Knowles radial grid:\n{} == {}\n{} == {}\n", radial.points.transpose(), expected_pts.transpose(), radial.weights.transpose(), expected_weights.transpose());

    REQUIRE(all_close(radial.points, expected_pts, 1e-5));
    REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
    /*
    BENCHMARK("Becke radial 10") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 50") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 80") {
        return occ::dft::generate_becke_radial_grid(10);
    };
    */
}

TEST_CASE("Treutler-Alrichs radial grid", "[radial]")
{
    auto radial = occ::dft::generate_treutler_alrichs_radial_grid(3);
    occ::Vec3 expected_pts{0.10934791, 1, 3.82014324};
    occ::Vec3 expected_weights{0.34905607, 1.60432893, 4.51614622};
    fmt::print("Treutler-Alrichs radial grid:\n{} == {}\n{} == {}\n", radial.points.transpose(), expected_pts.transpose(), radial.weights.transpose(), expected_weights.transpose());

    REQUIRE(all_close(radial.points, expected_pts, 1e-5));
    REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
    /*
    BENCHMARK("Becke radial 10") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 50") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 80") {
        return occ::dft::generate_becke_radial_grid(10);
    };
    */
}

TEST_CASE("atom grid", "[grid]")
{
    auto atom = occ::dft::generate_atom_grid(1, 302, 50);
   // fmt::print("Atom grid:\n{}\n", atom.points.transpose());
    fmt::print("Sum weights\n{}\n", atom.weights.array().sum());
    fmt::print("Shape: {} {}\n", atom.points.rows(), atom.points.cols());
    fmt::print("Max weight\n{}\n", atom.weights.maxCoeff());

    /*
    BENCHMARK("Atom Carbon") {
        return occ::dft::generate_atom_grid(6);
    };

    BENCHMARK("Atom Carbon 590") {
        return occ::dft::generate_atom_grid(6, 590);
    };
*/
    std::vector<libint2::Atom> atoms {
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.39839733}
    };
    occ::qm::BasisSet basis("sto-3g", atoms);
    occ::dft::MolecularGrid mgrid(basis, atoms);

    auto grid = mgrid.generate_partitioned_atom_grid(0);
    //fmt::print("{}\n{}\n", grid.points, grid.atomic_number);
}
