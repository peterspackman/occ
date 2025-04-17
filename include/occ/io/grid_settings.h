#pragma once

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
 * @brief Extended settings for Becke grid generation
 *
 */
struct GridSettings {
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

} // namespace occ::io
