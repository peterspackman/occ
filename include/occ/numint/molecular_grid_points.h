#pragma once
#include <occ/numint/grid_types.h>
#include <occ/dft/spatial_grid_hierarchy.h>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace occ::dft {

/**
 * @brief Class for storing and managing molecular grid points
 *
 * This class stores the combined grid points from multiple atom-centered grids,
 * along with their weights and organization by atoms.
 */
class MolecularGridPoints {
public:
  /**
   * @brief Default constructor
   */
  MolecularGridPoints() = default;

  /**
   * @brief Construct from points, weights, and atom blocks
   *
   * @param points Matrix of grid points (3 x N)
   * @param weights Vector of weights (N)
   * @param atom_blocks Vector of (offset, size) pairs for each atom
   */
  MolecularGridPoints(
      const Mat3N &points, const Vec &weights,
      const std::vector<std::pair<size_t, size_t>> &atom_blocks);

  /**
   * @brief Construct from a vector of atom grids
   *
   * @param atom_grids Vector of atom-centered grids
   */
  MolecularGridPoints(const std::vector<AtomGrid> &atom_grids);

  /**
   * @brief Initialize from a vector of atom grids
   *
   * @param atom_grids Vector of atom-centered grids
   */
  void initialize_from_atom_grids(const std::vector<AtomGrid> &atom_grids);

  /**
   * @brief Get the total number of grid points
   *
   * @return size_t Total number of grid points
   */
  size_t num_points() const;

  /**
   * @brief Get the number of atoms
   *
   * @return size_t Number of atoms
   */
  size_t num_atoms() const;

  /**
   * @brief Get all grid points
   *
   * @return const Mat3N& Matrix of all grid points (3 x N)
   */
  const Mat3N &points() const;

  /**
   * @brief Get all weights
   *
   * @return const Vec& Vector of all weights (N)
   */
  const Vec &weights() const;

  /**
   * @brief Get atom blocks info
   *
   * Each pair contains the offset and size of the grid points for each atom.
   *
   * @return const std::vector<std::pair<size_t, size_t>>& Vector of (offset,
   * size) pairs
   */
  const std::vector<std::pair<size_t, size_t>> &atom_blocks() const;

  /**
   * @brief Get points for a specific atom
   *
   * @param atom_idx Index of the atom
   * @return Eigen::Ref<const Mat3N> Reference to the atom's grid points
   * @throws std::out_of_range if the atom index is invalid
   */
  Eigen::Ref<const Mat3N> points_for_atom(size_t atom_idx) const;

  /**
   * @brief Get weights for a specific atom
   *
   * @param atom_idx Index of the atom
   * @return Eigen::Ref<const Vec> Reference to the atom's weights
   * @throws std::out_of_range if the atom index is invalid
   */
  Eigen::Ref<const Vec> weights_for_atom(size_t atom_idx) const;

  /**
   * @brief Extract an atom grid for a specific atom
   *
   * @param atom_idx Index of the atom
   * @param atomic_number Atomic number to assign to the grid
   * @return AtomGrid A new AtomGrid object for the specified atom
   * @throws std::out_of_range if the atom index is invalid
   */
  AtomGrid get_atom_grid(size_t atom_idx, uint_fast8_t atomic_number) const;

  /**
   * @brief Get spatial hierarchy for efficient batch processing
   *
   * Hierarchy is built lazily on first call and cached for subsequent calls.
   * The hierarchy organizes grid points spatially using Morton ordering and
   * provides batch access with precomputed bounding spheres.
   *
   * @param settings Configuration for hierarchy construction
   * @return const SpatialGridHierarchy& Reference to the cached hierarchy
   */
  const SpatialGridHierarchy& get_hierarchy(
      const SpatialHierarchySettings& settings = {}) const;

  /**
   * @brief Check if hierarchy has been built
   *
   * @return bool True if hierarchy is cached
   */
  bool has_hierarchy() const { return m_hierarchy.has_value(); }

  /**
   * @brief Clear cached hierarchy
   *
   * Call this if points are modified or you want to rebuild with different settings.
   */
  void clear_hierarchy() { m_hierarchy.reset(); }

private:
  Mat3N m_points; ///< All grid points (3 x N)
  Vec m_weights;  ///< All weights (N)
  std::vector<std::pair<size_t, size_t>>
      m_atom_blocks; ///< (offset, size) for each atom
  mutable std::optional<SpatialGridHierarchy> m_hierarchy; ///< Cached spatial hierarchy
};

} // namespace occ::dft
