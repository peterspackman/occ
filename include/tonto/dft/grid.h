#pragma once
#include <vector>
#include <tonto/core/linear_algebra.h>
#include <tonto/qm/basisset.h>

namespace tonto::dft {

using tonto::Mat3N;
using tonto::MatRM;
using tonto::MatN4;
using tonto::Vec;
using tonto::IVec;
using tonto::MatRM;
using tonto::qm::BasisSet;

struct AtomGrid
{
    AtomGrid() {}
    AtomGrid(size_t num_points) : points(3, num_points), weights(num_points) {}
    inline size_t num_points() const { return weights.size(); }
    uint_fast8_t atomic_number;
    Mat3N points;
    Vec weights;
    std::vector<tonto::Mat4N> atom_distances;
    tonto::MaskMat shell_mask;

    void compute_distances(const std::vector<libint2::Atom> &atoms)
    {
        atom_distances.clear();
        const size_t natoms = atoms.size();
        const size_t npts = points.cols();
        atom_distances.reserve(natoms);
        for(size_t i = 0; i < natoms; i++)
        {
            const auto& atom = atoms[i];
            tonto::Mat dists(4, npts);
            tonto::Vec3 xyz(atom.x, atom.y, atom.z);
            dists.block(0, 0, 3, npts) = points.block(0, 0, 3, npts).colwise() - xyz;
            dists.row(3) = dists.block(0, 0, 3, npts).colwise().squaredNorm();
            atom_distances.push_back(dists);
        }
    }

    template<size_t block_size=128>
    void compute_basis_screen(const BasisSet &basis, const std::vector<libint2::Atom> &atoms)
    {
        if(atom_distances.size() < 1) return;

        constexpr auto EXPCUTOFF{50};
        const size_t nshells = basis.size();
        const size_t npts = points.cols();
        const size_t natoms = atoms.size();
        auto shell2bf = basis.shell2bf();
        auto atom2shell = basis.atom2shell(atoms);

        const size_t num_blocks = (npts + block_size - 1) / block_size;
        shell_mask = tonto::MaskMat(num_blocks, nshells);

        for(size_t i = 0; i < natoms; i++)
        {
            const auto &dists = atom_distances[i];

            for(size_t shell_idx: atom2shell[i]) {
                const auto& shell = basis[shell_idx];
                size_t bf = shell2bf[shell_idx];
                for(size_t block = 0; block < num_blocks; block++)
                {
                    size_t lower = block * block_size;
                    size_t N = std::min(block_size, npts - (block * block_size));
                    for(size_t pt = lower; pt < lower + N; pt++) {
                        for(size_t prim = 0; prim < shell.nprim(); prim++) {
                            if((shell.alpha[prim] * dists(3, pt) - shell.max_ln_coeff[prim]) < EXPCUTOFF) {
                                shell_mask(block, shell_idx) = true;
                                goto next_block;
                            }
                        }
                    }
                    shell_mask(block, shell_idx) = false;
    next_block:;
                }
            }
        }
    }
};

struct RadialGrid
{
    RadialGrid() {}
    RadialGrid(size_t num_points) : points(num_points), weights(num_points) {}
    inline size_t num_points() const { return weights.size(); }
    Vec points;
    Vec weights;
};

IVec prune_nwchem_scheme(size_t nuclear_charge, size_t max_angular, size_t num_radial, const tonto::Vec& radii);
IVec prune_numgrid_scheme(size_t nuclear_charge, size_t max_angular, size_t min_angular, const tonto::Vec& radii);
RadialGrid generate_becke_radial_grid(size_t num_points, double rm = 1.0);
RadialGrid generate_mura_knowles_radial_grid(size_t num_points, size_t charge);
RadialGrid generate_treutler_alrichs_radial_grid(size_t num_points);
RadialGrid generate_gauss_chebyshev_radial_grid(size_t num_points);
RadialGrid generate_lmg_radial_grid(double radial_precision, double alpha_max, const tonto::Vec& alpha_min);
AtomGrid generate_atom_grid(size_t atomic_number, size_t max_angular_points = 590, size_t radial_points = 82);

class MolecularGrid
{
public:
    MolecularGrid(const BasisSet&, const std::vector<libint2::Atom>&);
    void set_angular_points(size_t n);
    void set_radial_points(size_t n);
    void set_max_angular_points(size_t n) {m_max_angular = n; }
    void set_min_angular_points(size_t n) {m_min_angular = n; }
    void set_radial_precision(double p) {m_radial_precision = p; }
    const auto n_atoms() const { return m_atomic_numbers.size(); }
    AtomGrid generate_partitioned_atom_grid(size_t atom_idx) const;
    AtomGrid generate_lmg_atom_grid(size_t atomic_number);
private:
    tonto::IVec m_atomic_numbers;
    Mat3N m_positions;
    Mat m_dists;
    std::vector<AtomGrid> m_unique_atom_grids;
    size_t m_max_angular{302};
    size_t m_min_angular{50};
    size_t m_radial_points{65};
    double m_radial_precision{1e-12};
    std::vector<std::pair<size_t, size_t>> m_grid_atom_blocks;
    Mat3N m_points;
    Vec m_weights;
    IVec m_l_max;
    Vec m_alpha_max;
    MatRM m_alpha_min;
};
}
