#include <occ/descriptors/pdd_amd.h>
#include <occ/descriptors/fast_neighbors.h>
#include <numeric>
#include <unordered_set>

namespace occ::descriptors {

PointwiseDistanceDistribution::PointwiseDistanceDistribution(const crystal::Crystal &crystal, int k, const PointwiseDistanceDistributionConfig &config)
    : m_k(k) {
    calculate_pdd(crystal, config);
}

void PointwiseDistanceDistribution::calculate_pdd(const crystal::Crystal &crystal, const PointwiseDistanceDistributionConfig &config) {
    const auto &asym_unit = crystal.asymmetric_unit();
    const size_t num_asym_atoms = asym_unit.size();
    
    // Validate inputs
    if (num_asym_atoms == 0) {
        throw std::invalid_argument("Crystal has no atoms in asymmetric unit");
    }
    if (m_k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    
    // Step 1: Calculate k-nearest neighbor distances for each asymmetric unit atom
    // This is the core computational step that finds the crystallographic environment
    Mat neighbor_distances = calculate_neighbor_distances(crystal);
    
    // Step 2: Initialize weights and groupings
    // Each asymmetric unit atom starts with equal weight (1/N) and its own group
    Vec weights = Vec::Ones(num_asym_atoms) / static_cast<double>(num_asym_atoms);
    
    // Initialize groups (each atom starts in its own group)
    // Each column represents one environment, each row one potential atom index
    m_groups.resize(num_asym_atoms, num_asym_atoms);
    m_groups.setConstant(-1); // Fill with -1 to indicate unused slots
    for (size_t i = 0; i < num_asym_atoms; ++i) {
        m_groups(0, i) = static_cast<int>(i); // Each atom in its own group initially
    }
    
    // Step 3: Merge chemically equivalent environments if requested
    if (config.collapse) {
        std::vector<int> group_labels;
        std::tie(weights, neighbor_distances, group_labels) = 
            merge_similar_rows(weights, neighbor_distances, config.collapse_tol);
        
        // Update groupings based on the merging
        if (group_labels.size() == num_asym_atoms && 
            neighbor_distances.rows() < static_cast<Eigen::Index>(num_asym_atoms)) {
            
            // Count how many atoms are in each new group
            int num_new_groups = neighbor_distances.rows();
            std::vector<int> group_counts(num_new_groups, 0);
            for (size_t old_idx = 0; old_idx < num_asym_atoms; ++old_idx) {
                int new_idx = group_labels[old_idx];
                group_counts[new_idx]++;
            }
            
            // Resize groups matrix and fill with merged groups
            int max_group_size = *std::max_element(group_counts.begin(), group_counts.end());
            m_groups.resize(max_group_size, num_new_groups);
            m_groups.setConstant(-1);
            
            std::vector<int> group_fill_counts(num_new_groups, 0);
            for (size_t old_idx = 0; old_idx < num_asym_atoms; ++old_idx) {
                int new_idx = group_labels[old_idx];
                int row_idx = group_fill_counts[new_idx]++;
                m_groups(row_idx, new_idx) = static_cast<int>(old_idx);
            }
        }
    }
    
    // Step 4: Lexicographically sort environments if requested
    if (config.lexsort) {
        // Create sorting indices based on lexicographic order of distance rows
        std::vector<size_t> indices(neighbor_distances.rows());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) {
            // Lexsort like Python: np.lexsort(dists.T[::-1])
            // Sort by last column first, then second-to-last, etc.
            for (int col = neighbor_distances.cols() - 1; col >= 0; --col) {
                if (std::abs(neighbor_distances(i, col) - neighbor_distances(j, col)) > 1e-10) {
                    return neighbor_distances(i, col) < neighbor_distances(j, col);
                }
            }
            return false;
        });
        
        // Apply the sorting permutation
        Mat sorted_distances(neighbor_distances.rows(), neighbor_distances.cols());
        Vec sorted_weights(weights.size());
        Eigen::MatrixXi sorted_groups(m_groups.rows(), m_groups.cols());
        
        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_distances.row(i) = neighbor_distances.row(indices[i]);
            sorted_weights(i) = weights(indices[i]);
            sorted_groups.col(i) = m_groups.col(indices[i]);
        }
        
        neighbor_distances = sorted_distances;
        weights = sorted_weights;
        m_groups = sorted_groups;
    }
    
    // Step 5: Store weights and distances separately
    m_weights = weights;
    m_distances = neighbor_distances.transpose(); // Transpose so environments are columns
    
    // Clean up groups if not requested
    if (!config.return_groups) {
        m_groups.resize(0, 0);
    }
}

Mat PointwiseDistanceDistribution::calculate_neighbor_distances(const crystal::Crystal &crystal) const {
    const auto &asym_unit = crystal.asymmetric_unit();
    const size_t num_asym_atoms = asym_unit.size();
    
    // Use zero-allocation fast neighbor finder
    Mat neighbor_distances(num_asym_atoms, m_k);
    
    // Query points are the asymmetric unit atoms (converted to Cartesian)
    Mat3N query_points = crystal.to_cartesian(asym_unit.positions);
    
    // Template parameter 256 should handle most reasonable k values
    if (m_k <= 256) {
        find_multiple_k_nearest<256>(crystal, query_points, m_k, neighbor_distances);
    } else {
        // Fallback for very large k - use larger template parameter
        find_multiple_k_nearest<1024>(crystal, query_points, m_k, neighbor_distances);
    }
    
    return neighbor_distances;
}

crystal::CrystalAtomRegion PointwiseDistanceDistribution::create_neighbor_slab(const crystal::Crystal &crystal, 
                                                   double max_distance) const {
    // Estimate how many unit cells we need in each direction
    const auto &unit_cell = crystal.unit_cell();
    Vec3 cell_lengths(unit_cell.a(), unit_cell.b(), unit_cell.c());
    
    // Calculate buffer in terms of unit cells
    IVec3 buffer;
    for (int i = 0; i < 3; ++i) {
        buffer(i) = std::max(1, static_cast<int>(std::ceil(max_distance / cell_lengths(i))));
    }
    
    // Create slab with sufficient buffer
    crystal::HKL lower{-buffer(0), -buffer(1), -buffer(2)};
    crystal::HKL upper{buffer(0), buffer(1), buffer(2)};
    
    return crystal.slab(lower, upper);
}

std::tuple<Vec, Mat, std::vector<int>> 
PointwiseDistanceDistribution::merge_similar_rows(const Vec &weights, const Mat &distances, double tolerance) const {
    const int n = distances.rows();
    const int k = distances.cols();
    
    std::vector<int> group_labels(n);
    std::unordered_set<int> processed;
    int current_group = 0;
    
    // Find groups of similar rows using Chebyshev distance
    for (int i = 0; i < n; ++i) {
        if (processed.count(i)) continue;
        
        group_labels[i] = current_group;
        processed.insert(i);
        
        // Find all rows similar to row i
        for (int j = i + 1; j < n; ++j) {
            if (processed.count(j)) continue;
            
            // Check Chebyshev distance (max absolute difference)
            double max_diff = 0.0;
            for (int col = 0; col < k; ++col) {
                max_diff = std::max(max_diff, std::abs(distances(i, col) - distances(j, col)));
            }
            
            if (max_diff <= tolerance) {
                group_labels[j] = current_group;
                processed.insert(j);
            }
        }
        
        current_group++;
    }
    
    // If no merging occurred, return original data
    if (current_group == n) {
        return std::make_tuple(weights, distances, group_labels);
    }
    
    // Merge groups
    Vec merged_weights = Vec::Zero(current_group);
    Mat merged_distances = Mat::Zero(current_group, k);
    std::vector<int> group_counts(current_group, 0);
    
    // Accumulate weights and distances
    for (int i = 0; i < n; ++i) {
        int group = group_labels[i];
        merged_weights(group) += weights(i);
        merged_distances.row(group) += distances.row(i);
        group_counts[group]++;
    }
    
    // Average distances within each group
    for (int g = 0; g < current_group; ++g) {
        if (group_counts[g] > 1) {
            merged_distances.row(g) /= group_counts[g];
        }
    }
    
    return std::make_tuple(merged_weights, merged_distances, group_labels);
}

Vec PointwiseDistanceDistribution::average_minimum_distance() const {
    if (m_weights.size() == 0 || m_distances.rows() == 0) {
        return Vec::Zero(m_k);
    }
    
    // Calculate weighted average of each row (each row is a distance rank)
    Vec result(m_k);
    for (int i = 0; i < m_k; ++i) {
        result(i) = m_weights.dot(m_distances.row(i));
    }
    
    return result;
}

} // namespace occ::descriptors