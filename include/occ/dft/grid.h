#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/io/grid_settings.h>
#include <occ/qm/shell.h>
#include <vector>
namespace occ::dft {
using occ::io::BeckeGridSettings;
using occ::qm::AOBasis;
enum class PartitionFunction {
  Becke,
  StratmannScuseria,
};
Mat calculate_atomic_grid_weights(PartitionFunction func,
                                  const Mat &grid_points,
                                  const Mat &atomic_positions,
                                  const IVec &atomic_numbers,
                                  const Mat &interatomic_distances);
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

class MolecularGridPoints {
public:
  MolecularGridPoints() = default;

  inline MolecularGridPoints(
      const Mat3N &points, const Vec &weights,
      const std::vector<std::pair<size_t, size_t>> &atom_blocks)
      : m_points(points), m_weights(weights), m_atom_blocks(atom_blocks) {}

  inline MolecularGridPoints(const std::vector<AtomGrid> &atom_grids) {
    initialize_from_atom_grids(atom_grids);
  }

  inline void
  initialize_from_atom_grids(const std::vector<AtomGrid> &atom_grids) {
    size_t total_points = 0;
    for (const auto &grid : atom_grids) {
      total_points += grid.num_points();
    }

    m_points.resize(3, total_points);
    m_weights.resize(total_points);
    m_atom_blocks.clear();

    size_t offset = 0;
    for (size_t i = 0; i < atom_grids.size(); i++) {
      const auto &grid = atom_grids[i];
      size_t npt = grid.num_points();

      if (npt > 0) {
        m_points.middleCols(offset, npt) = grid.points;
        m_weights.segment(offset, npt) = grid.weights;
        m_atom_blocks.push_back({offset, npt});
        offset += npt;
      } else {
        m_atom_blocks.push_back({offset, 0});
      }
    }
  }

  inline size_t num_points() const { return m_weights.size(); }

  inline size_t num_atoms() const { return m_atom_blocks.size(); }

  inline const Mat3N &points() const { return m_points; }

  inline const Vec &weights() const { return m_weights; }

  inline const std::vector<std::pair<size_t, size_t>> &atom_blocks() const {
    return m_atom_blocks;
  }
  inline Eigen::Ref<const Mat3N> points_for_atom(size_t atom_idx) const {
    if (atom_idx >= m_atom_blocks.size()) {
      throw std::out_of_range("Atom index out of range");
    }

    const auto &[offset, size] = m_atom_blocks[atom_idx];
    return m_points.middleCols(offset, size);
  }

  inline Eigen::Ref<const Vec> weights_for_atom(size_t atom_idx) const {
    if (atom_idx >= m_atom_blocks.size()) {
      throw std::out_of_range("Atom index out of range");
    }

    const auto &[offset, size] = m_atom_blocks[atom_idx];
    if (size == 0) {
      return Vec(0);
    }

    return m_weights.segment(offset, size);
  }

  inline AtomGrid get_atom_grid(size_t atom_idx,
                                uint_fast8_t atomic_number) const {
    if (atom_idx >= m_atom_blocks.size()) {
      throw std::out_of_range("Atom index out of range");
    }

    const auto &[offset, size] = m_atom_blocks[atom_idx];
    AtomGrid grid(size);

    if (size > 0) {
      grid.points = m_points.middleCols(offset, size);
      grid.weights = m_weights.segment(offset, size);
    }

    grid.atomic_number = atomic_number;
    return grid;
  }

private:
  Mat3N m_points;
  Vec m_weights;
  std::vector<std::pair<size_t, size_t>> m_atom_blocks;
};

class MolecularGrid {
public:
  MolecularGrid(const AOBasis &, const BeckeGridSettings &settings = {});
  const auto n_atoms() const { return m_atomic_numbers.size(); }
  AtomGrid generate_partitioned_atom_grid(size_t atom_idx) const;
  AtomGrid generate_lmg_atom_grid(size_t atomic_number);
  inline const auto &settings() const { return m_settings; }

  void populate_molecular_grid_points();
  inline const MolecularGridPoints &get_molecular_grid_points() const {
    return m_grid_points;
  };

private:
  void ensure_settings();
  occ::IVec m_atomic_numbers;
  Mat3N m_positions;
  Mat m_dists;
  std::vector<AtomGrid> m_unique_atom_grids;
  BeckeGridSettings m_settings;
  mutable MolecularGridPoints m_grid_points;
  mutable bool m_grid_initialized{false};
  IVec m_l_max;
  Vec m_alpha_max;
  Mat m_alpha_min;
};
} // namespace occ::dft
