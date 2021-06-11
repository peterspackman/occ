#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include <occ/qm/basisset.h>
#include <occ/gto/gto.h>
#include <occ/dft/grid.h>
#include <occ/core/util.h>
#include <fmt/ostream.h>

using occ::gto::GTOValues;
using occ::qm::BasisSet;
using occ::util::all_close;


TEST_CASE("evaluate_basis", "[gto]")
{
    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    BasisSet basis("3-21G", atoms);
    basis.set_pure(false);
    occ::dft::MolecularGrid mgrid(basis, atoms);
    std::vector<occ::dft::AtomGrid> grids{
        mgrid.generate_partitioned_atom_grid(0),
        mgrid.generate_partitioned_atom_grid(1),
        mgrid.generate_partitioned_atom_grid(2)
    };
    constexpr size_t block_size = 128;

    auto current_values = occ::gto::evaluate_basis_on_grid<0>(basis, atoms, grids[0].points);

    auto new_values = occ::gto::evaluate_basis_gau2grid<0>(basis, atoms, grids[0].points);

    fmt::print("OLD\n{}\n", current_values.phi.block(0, 0, 3, current_values.phi.cols()));
    fmt::print("NEW\n{}\n", new_values.phi.block(0, 0, 3, current_values.phi.cols()));



    BENCHMARK("mine") {
        GTOValues<1> values(basis.nbf(), grids[0].points.cols());
        for(const auto &grid: grids) {
            values.set_zero();
            values = occ::gto::evaluate_basis_on_grid<1>(basis, atoms, grid.points);
        }
        return 0;
    };


    BENCHMARK("gau2grid") {
        GTOValues<1> values(basis.nbf(), grids[0].points.cols());
        for(const auto &grid: grids) {
            values.set_zero();
            values = occ::gto::evaluate_basis_gau2grid<1>(basis, atoms, grid.points);
        }
        return 0;
    };
}
