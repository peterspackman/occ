#include <occ/qm/spatial_grid_hierarchy.h>
#include <algorithm>
#include <numeric>

namespace occ::qm {

namespace {

// Simple Morton encoding for 21-bit coordinates
// Spreads bits of x into positions 0,3,6,9,...
inline uint64_t dilate_3d(uint64_t x) {
    x &= 0x1fffff;  // Limit to 21 bits
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8))  & 0x100f00f00f00f00f;
    x = (x | (x << 4))  & 0x10c30c30c30c30c3;
    x = (x | (x << 2))  & 0x1249249249249249;
    return x;
}

// Encode 3D coordinates into Morton code
inline uint64_t morton_encode(uint64_t x, uint64_t y, uint64_t z) {
    return dilate_3d(x) | (dilate_3d(y) << 1) | (dilate_3d(z) << 2);
}

} // anonymous namespace

SpatialGridHierarchy::SpatialGridHierarchy(
    const Mat3N& points, const Vec& weights,
    const SpatialHierarchySettings& settings)
    : m_settings(settings) {
    build_hierarchy(points, weights);
}

void SpatialGridHierarchy::build_hierarchy(const Mat3N& points, const Vec& weights) {
    sort_by_morton_codes(points, weights);
    build_leaves();
}

void SpatialGridHierarchy::sort_by_morton_codes(const Mat3N& points, const Vec& weights) {
    const size_t n = points.cols();
    if (n == 0) {
        m_sorted_points.resize(3, 0);
        m_sorted_weights.resize(0);
        return;
    }

    // Compute bounding box for normalization
    Vec3 min_pt = points.rowwise().minCoeff();
    Vec3 max_pt = points.rowwise().maxCoeff();
    Vec3 extent = max_pt - min_pt;

    // Handle degenerate cases
    for (int i = 0; i < 3; ++i) {
        if (extent(i) < 1e-10) extent(i) = 1.0;
    }

    // Compute Morton codes for all points
    std::vector<std::pair<uint64_t, size_t>> morton_indices(n);

    for (size_t i = 0; i < n; ++i) {
        // Normalize to [0, 1] range
        Vec3 normalized = (points.col(i) - min_pt).array() / extent.array();

        // Clamp to valid range
        normalized = normalized.cwiseMax(0.0).cwiseMin(1.0);

        // Convert to Morton code (21 bits per dimension for 63-bit total)
        constexpr uint64_t scale = (1ULL << 21) - 1;
        uint64_t x = static_cast<uint64_t>(normalized(0) * scale);
        uint64_t y = static_cast<uint64_t>(normalized(1) * scale);
        uint64_t z = static_cast<uint64_t>(normalized(2) * scale);

        // Use the morton encoding
        morton_indices[i] = {morton_encode(x, y, z), i};
    }

    // Sort by Morton code
    std::sort(morton_indices.begin(), morton_indices.end());

    // Build permutation and reorder points/weights
    m_permutation.resize(n);
    m_sorted_points.resize(3, n);
    m_sorted_weights.resize(n);

    for (size_t i = 0; i < n; ++i) {
        size_t orig_idx = morton_indices[i].second;
        m_permutation[i] = orig_idx;
        m_sorted_points.col(i) = points.col(orig_idx);
        m_sorted_weights(i) = weights(orig_idx);
    }
}

void SpatialGridHierarchy::build_leaves() {
    const size_t n = m_sorted_points.cols();
    if (n == 0) return;

    m_leaves.clear();

    size_t offset = 0;
    while (offset < n) {
        size_t remaining = n - offset;
        size_t count;

        if (remaining <= m_settings.max_leaf_size) {
            // Take all remaining points
            count = remaining;
        } else if (remaining < 2 * m_settings.min_leaf_size) {
            // Split evenly to avoid tiny last leaf
            count = remaining / 2;
        } else {
            // Use target leaf size
            count = std::min(m_settings.target_leaf_size, remaining);
        }

        GridBatchLeaf leaf;
        leaf.offset = offset;
        leaf.count = count;
        leaf.bounds = compute_bounding_sphere(m_sorted_points, offset, count);

        m_leaves.push_back(leaf);
        offset += count;
    }
}

GridBoundingSphere SpatialGridHierarchy::compute_bounding_sphere(
    const Mat3N& points, size_t offset, size_t count) {

    GridBoundingSphere sphere;

    if (count == 0) return sphere;

    // Get the block of points
    auto block = points.middleCols(offset, count);

    // Compute centroid
    sphere.center = block.rowwise().mean();

    // Compute radius as max distance from center
    sphere.radius = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double dist = (block.col(i) - sphere.center).norm();
        sphere.radius = std::max(sphere.radius, dist);
    }

    return sphere;
}

} // namespace occ::qm
