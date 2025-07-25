#pragma once
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
  Coarse,   ///< Like SG-0: ~23 radial, ~170 angular points - fastest, basic
            ///< accuracy
  Standard, ///< Like SG-1: ~50 radial, ~194 angular points - standard for
            ///< LDA/GGA
  Fine,     ///< Like SG-2: ~75 radial, ~302 angular points - meta-GGAs, B95/B97
  VeryFine  ///< Like SG-3: ~99 radial, ~590 angular points - Minnesota
            ///< functionals, highest accuracy
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

  size_t max_angular_points = 302; ///< Maximum number of angular points
  size_t min_angular_points = 110; ///< Minimum number of angular points
  size_t radial_points = 50; ///< Number of radial points (for some methods)
  double radial_precision =
      1e-12; ///< Precision for radial grid (for LMG method)
  bool reduced_first_row_element_grid =
      true; ///< Whether to use reduced grid for H and He
  bool treutler_alrichs_adjustment = true;
  PruningScheme pruning_scheme =
      PruningScheme::NWChem; ///< Pruning scheme to use
  inline bool operator==(const GridSettings &rhs) const = default;
  inline bool operator!=(const GridSettings &rhs) const = default;
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

} // namespace occ::io
