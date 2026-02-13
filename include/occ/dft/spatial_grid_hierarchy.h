#pragma once
// Forwarding header for backward compatibility
// Spatial grid hierarchy has moved to occ::qm
#include <occ/qm/spatial_grid_hierarchy.h>

namespace occ::dft {
    // Forward all symbols from occ::qm for backward compatibility
    using occ::qm::GridBoundingSphere;
    using occ::qm::GridBatchLeaf;
    using occ::qm::SpatialHierarchySettings;
    using occ::qm::SpatialGridHierarchy;
}
