#pragma once
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::qm {

/// Bounding sphere for a group of grid points
struct GridBoundingSphere {
    Vec3 center{0.0, 0.0, 0.0};
    double radius{0.0};

    /// Check if a point at given distance from center could be within sphere
    bool may_overlap(double distance_to_center, double point_extent) const {
        return distance_to_center - point_extent < radius;
    }
};

/// Leaf node containing grid point batch
struct GridBatchLeaf {
    size_t offset{0};           // Start index in sorted points array
    size_t count{0};            // Number of points in this batch
    GridBoundingSphere bounds;  // Precomputed bounding sphere
};

/// Configuration for hierarchy construction
struct SpatialHierarchySettings {
    size_t target_leaf_size{128};
    size_t min_leaf_size{32};
    size_t max_leaf_size{256};
};

/// Spatial hierarchy for molecular grid points using Morton ordering
class SpatialGridHierarchy {
public:
    /// Construct hierarchy from points and weights
    SpatialGridHierarchy(const Mat3N& points, const Vec& weights,
                         const SpatialHierarchySettings& settings = {});

    /// Get sorted points (Morton-ordered)
    const Mat3N& sorted_points() const { return m_sorted_points; }
    const Vec& sorted_weights() const { return m_sorted_weights; }

    /// Access leaves
    const std::vector<GridBatchLeaf>& leaves() const { return m_leaves; }
    size_t num_leaves() const { return m_leaves.size(); }

    /// Get points for a specific leaf (returns a block reference)
    auto leaf_points(size_t leaf_idx) const {
        const auto& leaf = m_leaves[leaf_idx];
        return m_sorted_points.middleCols(leaf.offset, leaf.count);
    }

    /// Get weights for a specific leaf
    auto leaf_weights(size_t leaf_idx) const {
        const auto& leaf = m_leaves[leaf_idx];
        return m_sorted_weights.segment(leaf.offset, leaf.count);
    }

    /// Get bounding sphere for a leaf
    const GridBoundingSphere& leaf_bounds(size_t leaf_idx) const {
        return m_leaves[leaf_idx].bounds;
    }

    /// Get original index mapping (sorted_idx -> original_idx)
    const std::vector<size_t>& permutation() const { return m_permutation; }

private:
    void build_hierarchy(const Mat3N& points, const Vec& weights);
    void sort_by_morton_codes(const Mat3N& points, const Vec& weights);
    void build_leaves();
    static GridBoundingSphere compute_bounding_sphere(
        const Mat3N& points, size_t offset, size_t count);

    Mat3N m_sorted_points;
    Vec m_sorted_weights;
    std::vector<size_t> m_permutation;
    std::vector<GridBatchLeaf> m_leaves;
    SpatialHierarchySettings m_settings;
};

} // namespace occ::qm
