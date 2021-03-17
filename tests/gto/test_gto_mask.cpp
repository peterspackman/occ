#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include <tonto/qm/basisset.h>
#include <tonto/gto/gto.h>
#include <tonto/dft/grid.h>
#include <tonto/core/util.h>
#include <fmt/ostream.h>

using tonto::gto::GTOValues;
using tonto::qm::BasisSet;
using tonto::util::all_close;



template<size_t max_derivative, size_t block_size=128>
void evaluate_basis_on_grid(GTOValues<max_derivative> &gto_values,
                            tonto::dft::AtomGrid &grid,
                            const BasisSet &basis,
                            const std::vector<libint2::Atom> &atoms)
{
    const size_t nbf = basis.nbf();
    const size_t npts = grid.num_points();
    const size_t natoms = atoms.size();
    const auto& mask = grid.shell_mask;
    const size_t num_blocks = mask.rows();
    auto shell2bf = basis.shell2bf();
    auto atom2shell = basis.atom2shell(atoms);

    for(size_t i = 0; i < natoms; i++)
    {
        const auto& atom = atoms[i];
        for(const auto& shell_idx: atom2shell[i]) {
            const auto& shell = basis[shell_idx];
            size_t bf = shell2bf[shell_idx];
            const auto& dists = grid.atom_distances[i];

            for(size_t block = 0; block < num_blocks; block++)
            {
                if(!mask(block, shell_idx)) continue;
                size_t lower = block * block_size;
                size_t N = std::min(block_size, npts - (block * block_size));
                tonto::gto::impl::add_shell_contribution_block<max_derivative>(bf, shell, dists, gto_values, lower, N);
            }
        }
    }
}




TEST_CASE("evaluate_basis", "[gto]")
{
    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    BasisSet basis("3-21G", atoms);
    basis.set_pure(false);
    tonto::dft::MolecularGrid mgrid(basis, atoms);
    std::vector<tonto::dft::AtomGrid> grids{
        mgrid.generate_partitioned_atom_grid(0),
        mgrid.generate_partitioned_atom_grid(1),
        mgrid.generate_partitioned_atom_grid(2)
    };
    constexpr size_t block_size = 128;

    for(auto& grid: grids)
    {
        grid.compute_distances(atoms);
        grid.compute_basis_screen<block_size>(basis, atoms);
    }

    GTOValues<1> values_new(basis.nbf(), grids[0].points.cols());
    values_new.set_zero();
    evaluate_basis_on_grid<1, block_size>(values_new, grids[0], basis, atoms);
    auto values_old = tonto::gto::evaluate_basis_on_grid<1>(basis, atoms, grids[0].points);


    REQUIRE(all_close(values_new.phi, values_old.phi));
    REQUIRE(all_close(values_new.phi_x, values_old.phi_x));
    REQUIRE(all_close(values_new.phi_y, values_old.phi_y));
    REQUIRE(all_close(values_new.phi_z, values_old.phi_z));
/*
    BENCHMARK("Screen basis") {
        for(const auto &grid: grids) {
            auto mask = screen_basis(basis, atoms, grids[0].points);
            auto values = evaluate_basis_on_grid<1>(basis, atoms, grid.points, mask);
        }
        return 0;
    };

    std::vector<tonto::MaskMat> masks = {
        screen_basis(basis, atoms, grids[0].points),
        screen_basis(basis, atoms, grids[1].points),
        screen_basis(basis, atoms, grids[2].points)
    };

*/

    BENCHMARK("Prescreen") {
        GTOValues<1> values(basis.nbf(), grids[0].points.cols());
        for(size_t n = 0; n < grids.size(); n++) {
            values.set_zero();
            evaluate_basis_on_grid<1, block_size>(values, grids[0], basis, atoms);
            values.set_zero();
        }
        return 0;
    };

    BENCHMARK("Current") {
        GTOValues<1> values(basis.nbf(), grids[0].points.cols());
        for(const auto &grid: grids) {
            values.set_zero();
            values = tonto::gto::evaluate_basis_on_grid<1>(basis, atoms, grid.points);
            values.set_zero();
        }
        return 0;
    };
}
