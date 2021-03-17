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


template<size_t max_derivative>
void add_shell_contribution(size_t bf, const libint2::Shell &shell, const Eigen::Ref<const tonto::Mat>& dists,
                    GTOValues<max_derivative>& result, size_t offset, size_t N)
{
    size_t n_pt = dists.cols();
    size_t n_prim = shell.nprim();
    constexpr size_t LMAX{5};
    for(const auto& contraction: shell.contr) {
        switch (contraction.l) {
        case 0: {
            for(size_t pt = offset; pt < offset + N; pt++) {
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double cc_exp_r2 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    cc_exp_r2 += cexp;
                    if constexpr (max_derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;

                result.phi(pt, bf) += cc_exp_r2;
                if constexpr ( max_derivative > 0)
                {
                    result.phi_x(pt, bf) += g1 * x;
                    result.phi_y(pt, bf) += g1 * y;
                    result.phi_z(pt, bf) += g1 * z;
                }
            }
            break;
        }
        case 1: {
            for (size_t pt = offset; pt <  offset + N; pt++) {
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double g0 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    g0 += cexp;
                    if constexpr (max_derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;
                double g1x = g1 * x;
                double g1y = g1 * y;
                double g1z = g1 * z;
                result.phi(pt, bf) += x * g0;
                result.phi(pt, bf + 1) += y * g0;
                result.phi(pt, bf + 2) += z * g0;
                if constexpr (max_derivative > 0) {
                    result.phi_x(pt, bf) += g0 + x*g1x;
                    result.phi_y(pt, bf) += x * g1y;
                    result.phi_z(pt, bf) += x * g1z;
                    result.phi_x(pt, bf + 1) += y*g1x;
                    result.phi_y(pt, bf + 1) += g0 + y * g1y;
                    result.phi_z(pt, bf + 1) += y * g1z;
                    result.phi_x(pt, bf + 2) += z * g1x;
                    result.phi_y(pt, bf + 2) += z * g1y;
                    result.phi_z(pt, bf + 2) += g0 + z * g1z;
                }
            }
            break;
        }
        default: {
            std::array<double, LMAX> bx, by, bz, gxb, gyb, gzb;
            bx[0] = 1.0; by[0] = 1.0; bz[0] = 1.0;
            for (size_t pt = offset; pt < offset + N; pt++) {
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double g0 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    g0 += cexp;
                    if constexpr (max_derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;
                double g1x = g1 * x;
                double g1y = g1 * y;
                double g1z = g1 * z;
                double g1xx = g1x * x;
                double g1yy = g1y * y;
                double g1zz = g1z * z;
                double bxb = x, byb = y, bzb = z;
                bx[1] = x; by[1] = y; bz[1] = z;
                gxb[0] = g1x; gyb[0] = g1y; gzb[0] = g1z;
                gxb[1] = g0 + g1xx; gyb[1] = g0 + g1yy; gzb[1] = g0 + g1zz;

                for(size_t b = 2; b <= contraction.l; b++) {
                    gxb[b] = (b * g0 + g1xx) * bxb;
                    gyb[b] = (b * g0 + g1yy) * byb;
                    gzb[b] = (b * g0 + g1zz) * bzb;
                    bxb *= x; byb *= y; bzb *= z;
                    bx[b] = bxb; by[b] = byb; bz[b] = bzb;
                }
                int L, M, N;
                size_t offset = 0;
                FOR_CART(L, M, N, contraction.l)
                    bxb = bx[L]; byb = by[M]; bzb = bz[N];
                    double by_bz = byb * bzb;
                    result.phi(pt, bf + offset) += bxb * by_bz * g0;
                    if constexpr (max_derivative > 0) {
                        result.phi_x(pt, bf + offset) += gxb[L] * by_bz;
                        result.phi_y(pt, bf + offset) += bxb * gyb[M] * bzb;
                        result.phi_z(pt, bf + offset) += bxb * byb * gzb[N];
                    }
                    offset++;
                END_FOR_CART
            }
        }
        }
    }
}


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
                add_shell_contribution<max_derivative>(
                    bf, shell, dists, gto_values, lower, N);
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
    BasisSet basis("def2-tzvp", atoms);
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
