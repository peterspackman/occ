#pragma once
#include <array>
#include <string>

namespace occ::io {

/**
 * @brief Enumeration of available angular pruning schemes
 */
enum class PruningScheme {
  None,   ///< No pruning (uniform angular points)
  NWChem, ///< NWChem pruning scheme
  NumGrid ///< NumGrid pruning scheme
};

/**
 * @brief Standard grid quality levels (similar to Q-Chem's SG system)
 */
enum class GridQuality {
  Coarse,    ///< Like SG-0: ~23 radial, ~170 angular points - fastest, basic
             ///< accuracy
  Standard,  ///< Like SG-1: ~50 radial, ~194 angular points - standard for
             ///< LDA/GGA
  Fine,      ///< Like SG-2: ~75 radial, ~302 angular points - meta-GGAs, B95/B97
  VeryFine   ///< Like SG-3: ~99 radial, ~590 angular points - Minnesota
             ///< functionals, highest accuracy
};

/**
 * @brief COSX/SGX grid levels matching ORCA's GRIDX scheme
 *
 * ORCA uses 3 grid levels for COSX with specific IntAcc values and
 * 5-region angular pruning. These correspond to GRIDX 1, 2, 3 in ORCA.
 */
enum class COSXGridLevel {
  Grid1, ///< GRIDX 1: IntAcc=3.816, max 50 Lebedev points
  Grid2, ///< GRIDX 2: IntAcc=4.020, max 110 Lebedev points
  Grid3  ///< GRIDX 3: IntAcc=4.338, max 194 Lebedev points
};

/**
 * @brief Extended settings for Becke grid generation
 *
 */
struct GridSettings {
  /**
   * @brief Create GridSettings from GridQuality enum
   *
   * @param quality Grid quality level
   * @return GridSettings configured for the specified quality
   */
  static GridSettings from_grid_quality(GridQuality quality);

  /**
   * @brief Create GridSettings for SGX/COSX seminumerical exchange
   *
   * Uses coarser grids appropriate for seminumerical exchange:
   * - 50 angular points (ORCA default for COSX)
   * - Lower radial precision (1e-5)
   * - No pruning for uniform coverage
   *
   * @param angular_points Number of angular Lebedev grid points (default 50)
   * @return GridSettings configured for SGX/COSX
   * @deprecated Use for_cosx(COSXGridLevel) instead for ORCA-compatible grids
   */
  static GridSettings for_sgx(size_t angular_points = 50);

  /**
   * @brief Create GridSettings for COSX matching ORCA's GRIDX scheme
   *
   * Creates grids matching ORCA's COSX grid levels with:
   * - IntAcc-based radial point count: nr = (15 * IntAcc - 40) + b * ROW
   * - 5-region angular pruning with Lebedev grids
   * - Gauss-Chebyshev quadrature with M3 mapping
   *
   * @param level COSX grid level (Grid1, Grid2, or Grid3)
   * @return GridSettings configured for the specified COSX level
   */
  static GridSettings for_cosx(COSXGridLevel level);

  size_t max_angular_points = 302; ///< Maximum number of angular points
  size_t min_angular_points = 110; ///< Minimum number of angular points
  size_t radial_points = 50; ///< Number of radial points (for some methods)
  double radial_precision =
      1e-12; ///< Precision for radial grid (for LMG method)
  double int_acc = 0.0; ///< ORCA-style IntAcc parameter (0 = not used)
  bool reduced_first_row_element_grid =
      true; ///< Whether to use reduced grid for H and He
  bool treutler_alrichs_adjustment = true;
  PruningScheme pruning_scheme =
      PruningScheme::NWChem; ///< Pruning scheme to use

  /// @brief ORCA-style 5-region angular grid levels [region1..region5]
  /// If non-empty, overrides min/max_angular_points with per-region Lebedev counts
  std::array<size_t, 5> angular_regions = {0, 0, 0, 0, 0};

  inline bool operator==(const GridSettings &rhs) const = default;
  inline bool operator!=(const GridSettings &rhs) const = default;

  /// @brief Check if ORCA-style angular regions are enabled
  inline bool has_angular_regions() const { return angular_regions[0] > 0; }

  /// @brief Get IntAcc description string
  std::string int_acc_string() const;
};

/**
 * @brief Get grid settings for a specific quality level
 *
 * @param quality Grid quality level
 * @return GridSettings configured for the specified quality
 */
GridSettings get_grid_settings(GridQuality quality);

/**
 * @brief Convert GridQuality enum to string
 *
 * @param quality Grid quality level
 * @return std::string String representation of the quality level
 */
std::string grid_quality_to_string(GridQuality quality);

/**
 * @brief Convert string to GridQuality enum
 *
 * @param str String representation (case-insensitive)
 * @return GridQuality Grid quality level
 * @throws std::invalid_argument if string is not recognized
 */
GridQuality grid_quality_from_string(const std::string &str);

/**
 * @brief Convert COSXGridLevel enum to string
 *
 * @param level COSX grid level
 * @return std::string String representation (e.g., "GRIDX 1")
 */
std::string cosx_grid_level_to_string(COSXGridLevel level);

/**
 * @brief Get IntAcc value for a COSX grid level
 *
 * Returns the ORCA IntAcc parameter for the given COSX grid level
 *
 * @param level COSX grid level
 * @return double IntAcc value
 */
double cosx_grid_int_acc(COSXGridLevel level);

/**
 * @brief Calculate number of radial points using ORCA's formula
 *
 * Uses: nr = (15 * IntAcc - 40) + b * ROW
 * where ROW is the periodic table row of the element
 *
 * @param int_acc IntAcc parameter
 * @param atomic_number Atomic number of the element
 * @param b Row multiplier (default 5.0)
 * @return size_t Number of radial points
 */
size_t calculate_radial_points_orca(double int_acc, size_t atomic_number, double b = 5.0);

/**
 * @brief Print ORCA-style grid generation summary
 *
 * Prints a summary similar to ORCA's "COSX GRID GENERATION" output:
 * - IntAcc, angular grid level, pruning method
 * - Total grid points, batches, average points per batch/atom
 *
 * @param settings Grid settings used
 * @param level COSX grid level (Grid1, Grid2, Grid3)
 * @param num_points Total number of grid points
 * @param num_batches Number of batches
 * @param num_atoms Number of atoms
 */
void print_cosx_grid_summary(const GridSettings& settings, COSXGridLevel level,
                             size_t num_points, size_t num_batches, size_t num_atoms);

} // namespace occ::io
