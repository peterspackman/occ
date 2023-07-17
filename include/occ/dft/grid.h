#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/io/grid_settings.h>
#include <occ/qm/shell.h>
#include <vector>

namespace occ::dft {

using occ::io::BeckeGridSettings;
using occ::qm::AOBasis;

struct AtomGrid {
    AtomGrid() {}
    AtomGrid(size_t num_points) : points(3, num_points), weights(num_points) {}
    inline size_t num_points() const { return weights.size(); }
    uint_fast8_t atomic_number;
    Mat3N points;
    Vec weights;
};

struct RadialGrid {
    RadialGrid() {}
    RadialGrid(size_t num_points) : points(num_points), weights(num_points) {}
    inline size_t num_points() const { return weights.size(); }
    Vec points;
    Vec weights;
};

IVec prune_nwchem_scheme(size_t nuclear_charge, size_t max_angular,
                         size_t num_radial, const occ::Vec &radii);
IVec prune_numgrid_scheme(size_t nuclear_charge, size_t max_angular,
                          size_t min_angular, const occ::Vec &radii);
RadialGrid generate_becke_radial_grid(size_t num_points, double rm = 1.0);
RadialGrid generate_mura_knowles_radial_grid(size_t num_points, size_t charge);
RadialGrid generate_treutler_alrichs_radial_grid(size_t num_points);
RadialGrid generate_gauss_chebyshev_radial_grid(size_t num_points);
RadialGrid generate_lmg_radial_grid(double radial_precision, double alpha_max,
                                    const occ::Vec &alpha_min);
AtomGrid generate_atom_grid(size_t atomic_number,
                            size_t max_angular_points = 302,
                            size_t radial_points = 50);

class MolecularGrid {
  public:
    MolecularGrid(const AOBasis &, const BeckeGridSettings &settings = {});
    const auto n_atoms() const { return m_atomic_numbers.size(); }
    AtomGrid generate_partitioned_atom_grid(size_t atom_idx) const;
    AtomGrid generate_lmg_atom_grid(size_t atomic_number);

    inline const auto &settings() const { return m_settings; }

  private:
    void ensure_settings();

    occ::IVec m_atomic_numbers;
    Mat3N m_positions;
    Mat m_dists;
    std::vector<AtomGrid> m_unique_atom_grids;
    BeckeGridSettings m_settings;
    std::vector<std::pair<size_t, size_t>> m_grid_atom_blocks;
    Mat3N m_points;
    Vec m_weights;
    IVec m_l_max;
    Vec m_alpha_max;
    Mat m_alpha_min;
};
} // namespace occ::dft
