#include <occ/numint/grid_types.h>

namespace occ::numint {

RadialGrid::RadialGrid(size_t num_points)
    : points(num_points), weights(num_points) {
  // Memory allocation only, no initialization
}

AtomGrid::AtomGrid(size_t num_points)
    : points(3, num_points), weights(num_points) {
  // Memory allocation only, no initialization
}

} // namespace occ::numint
