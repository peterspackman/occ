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
};

struct RadialGrid
{
    RadialGrid() {}
    RadialGrid(size_t num_points) : points(num_points), weights(num_points) {}
    inline size_t num_points() const { return weights.size(); }
    Vec points;
    Vec weights;
};

tonto::IVec prune_nwchem_scheme(size_t nuclear_charge, size_t max_angular, size_t num_radial, const tonto::Vec& radii);
tonto::IVec prune_numgrid_scheme(size_t nuclear_charge, size_t max_angular, size_t min_angular, const tonto::Vec& radii);
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
