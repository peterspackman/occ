#pragma once
#include <cstdint>
#include <occ/core/linear_algebra.h>
#include <occ/io/grid_settings.h>
#include <vector>

namespace occ::dft {

/**
 * @brief Enumeration of available partition functions for atomic weight
 * calculation
 */
enum class PartitionFunction {
  Becke,             ///< Becke's original partition function
  StratmannScuseria, ///< Stratmann-Scuseria partition function
};

struct PartitionMethod {
  PartitionFunction partition_function{PartitionFunction::Becke};
  bool treutler_alrichs_radii_adjustment{true};
};

/**
 * @brief Enumeration of available radial grid generation methods
 */
enum class RadialGridMethod {
  LMG,             ///< Lindh-Malmqvist-Gagliardi method
  TreutlerAlrichs, ///< Treutler-Alrichs method
  MuraKnowles,     ///< Mura-Knowles method
  Becke,           ///< Becke's transformation method
  GaussChebyshev,  ///< Gauss-Chebyshev method
  EulerMaclaurin   ///< Gauss-Chebyshev method
};

using occ::io::GridSettings;
using occ::io::PruningScheme;

/**
 * @brief Structure to hold radial grid data
 *
 * Contains the radial points and corresponding weights for numerical
 * integration
 */
struct RadialGrid {
  /**
   * @brief Default constructor
   */
  RadialGrid() = default;

  /**
   * @brief Constructor that pre-allocates memory
   * @param num_points Number of grid points to allocate
   */
  RadialGrid(size_t num_points);

  /**
   * @brief Gets the number of points in the grid
   * @return The number of grid points
   */
  inline size_t num_points() const { return weights.size(); }

  Vec points;  ///< Radial points (distances from the center)
  Vec weights; ///< Integration weights for each point
};

/**
 * @brief Structure to hold atom-centered grid data
 *
 * Contains the 3D points, weights, and atomic number for a grid centered at an
 * atom
 */
struct AtomGrid {
  /**
   * @brief Default constructor
   */
  AtomGrid() = default;

  /**
   * @brief Constructor that pre-allocates memory
   * @param num_points Number of grid points to allocate
   */
  AtomGrid(size_t num_points);

  /**
   * @brief Gets the number of points in the grid
   * @return The number of grid points
   */
  inline size_t num_points() const { return weights.size(); }

  uint_fast8_t atomic_number = 0; ///< Atomic number of the center atom
  Mat3N points;                   ///< 3D coordinates of grid points
  Vec weights;                    ///< Integration weights for each point
};

} // namespace occ::dft
