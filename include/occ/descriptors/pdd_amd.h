#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/crystal/crystal.h>
#include <vector>

namespace occ::descriptors {

/**
 * \brief Configuration for Pointwise Distance Distribution calculations
 */
struct PointwiseDistanceDistributionConfig {
    bool lexsort = true;           ///< Lexicographically sort rows
    bool collapse = true;          ///< Merge similar rows within tolerance
    double collapse_tol = 1e-4;    ///< Tolerance for merging rows (Chebyshev distance)
    bool return_groups = false;    ///< Return grouping information
};

/**
 * \brief Pointwise Distance Distribution for crystals
 *
 * Based on the work by Widdowson et al., providing geometry-based 
 * crystallographic descriptors independent of unit cell choice.
 *
 * The PDD is a matrix where each row corresponds to a unique chemical environment
 * and contains the k nearest neighbor distances along with a weight indicating
 * the relative abundance of that environment.
 *
 * References:
 * - Widdowson, D. & Kurlin, V. (2022). Average Minimum Distances of periodic point sets.
 * - https://github.com/dwiddo/average-minimum-distance
 */
class PointwiseDistanceDistribution {
public:
    /**
     * \brief Construct PDD from crystal structure
     *
     * \param crystal The crystal structure to analyze
     * \param k Number of nearest neighbors to consider
     * \param config Configuration options for the calculation
     */
    PointwiseDistanceDistribution(const crystal::Crystal &crystal, int k, 
                                  const PointwiseDistanceDistributionConfig &config = PointwiseDistanceDistributionConfig{});
    
    /**
     * \brief Get the weights for each environment
     */
    const Vec& weights() const { return m_weights; }
    
    /**
     * \brief Get the distance matrix (environments as columns)
     */
    const Mat& distances() const { return m_distances; }
    
    /**
     * \brief Calculate Average Minimum Distance from this PDD
     * 
     * Computes the weighted average of each distance column, providing
     * a k-dimensional vector representing the average k-nearest neighbor
     * distances across all chemical environments.
     */
    Vec average_minimum_distance() const;
    
    /**
     * \brief Get the full PDD matrix (weights + distances) - assembled on demand
     */
    Mat matrix() const {
        Mat result(m_k + 1, m_weights.size());
        result.row(0) = m_weights.transpose();
        result.bottomRows(m_k) = m_distances;
        return result;
    }
    
    /**
     * \brief Number of unique chemical environments
     */
    size_t size() const { return m_weights.size(); }
    
    /**
     * \brief Number of neighbors considered
     */
    int k() const { return m_k; }
    
    /**
     * \brief Get grouping information if available
     * 
     * Returns a matrix where each column corresponds to a chemical environment
     * and contains the indices of asymmetric unit atoms in that environment.
     * Unused entries are filled with -1.
     */
    const Eigen::MatrixXi& groups() const { return m_groups; }
    
private:
    Vec m_weights;                               ///< Weights for each environment
    Mat m_distances;                             ///< Distance matrix (k x num_environments, environments as columns)
    Eigen::MatrixXi m_groups;                    ///< Which asymmetric unit atoms correspond to each column (cols=environments, rows=atom indices, -1 for unused)
    int m_k;                                     ///< Number of neighbors

    void calculate_pdd(const crystal::Crystal &crystal, const PointwiseDistanceDistributionConfig &config);
    Mat calculate_neighbor_distances(const crystal::Crystal &crystal) const;
    std::tuple<Vec, Mat, std::vector<int>> merge_similar_rows(const Vec &weights, 
                                                             const Mat &distances,
                                                             double tolerance) const;
    crystal::CrystalAtomRegion create_neighbor_slab(const crystal::Crystal &crystal,
                                                   double max_distance) const;
};

// Type aliases for backward compatibility
using PDDConfig = PointwiseDistanceDistributionConfig;
using PDD = PointwiseDistanceDistribution;

} // namespace occ::descriptors