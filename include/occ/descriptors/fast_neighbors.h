#pragma once
#include <algorithm>
#include <array>
#include <fmt/core.h>
#include <occ/core/linear_algebra.h>
#include <occ/crystal/crystal.h>
#include <occ/descriptors/sorted_k_distances.h>

namespace occ::descriptors {

/**
 * \brief Faster neighbor finding using incremental shell approach
 *
 * Try to avoid as much dynamic allocations in the hot path as possible
 */
template <int MaxK = 256> class FastNeighborFinder {
private:
  SortedKDistances<MaxK> m_sq_distances;

public:
  /**
   * \brief Find k nearest neighbors
   */
  Eigen::Ref<const Eigen::ArrayXd>
  find_k_nearest(const crystal::Crystal &crystal,
                 Eigen::Ref<const Vec3> query_pos, int k) {
    static_assert(MaxK >= 4, "MaxK must be at least 4");
    if (k > MaxK) {
      throw std::invalid_argument("k exceeds MaxK template parameter");
    }

    const auto &unit_cell = crystal.unit_cell();

    // Get the full unit cell atoms (all symmetry-equivalent positions)
    auto uc_atoms = crystal.unit_cell_atoms();
    Mat3N motif_cart = uc_atoms.cart_pos;
    const size_t num_motif = motif_cart.cols();

    // Block-vectorized distance calculation - reusable blocks
    constexpr int BLOCK_SIZE = 4;
    Eigen::Matrix<double, 3, BLOCK_SIZE> block_motif;
    Eigen::Matrix<double, 1, BLOCK_SIZE> block_sq_distances;

    // Step 1: Process initial layer (origin lattice point)
    m_sq_distances.clear();

    size_t motif_idx = 0;

    // Process full blocks
    for (; motif_idx + BLOCK_SIZE <= num_motif; motif_idx += BLOCK_SIZE) {
      block_motif = motif_cart.middleCols<BLOCK_SIZE>(motif_idx);
      block_sq_distances =
          (block_motif.colwise() - query_pos).colwise().squaredNorm();
      
      // Filter out self-distances using Eigen select
      block_sq_distances = (block_sq_distances.array() > 1e-6).select(
          block_sq_distances, std::numeric_limits<double>::infinity());
      
      m_sq_distances.try_insert_batch4(block_sq_distances.data());
    }

    // Process remaining atoms individually
    for (; motif_idx < num_motif; ++motif_idx) {
      Vec3 candidate = motif_cart.col(motif_idx);
      double sq_dist = (query_pos - candidate).squaredNorm();
      if (sq_dist > 1e-6) { // Skip self
        m_sq_distances.try_insert(sq_dist);
      }
    }

    // Calculate motif diameter (same as AMD library) - vectorized
    double motif_diam = 0.0;
    for (size_t i = 0; i < num_motif; ++i) {
      // Vectorized distance calculation from atom i to all atoms j > i
      if (i + 1 < num_motif) {
        auto remaining_atoms = motif_cart.rightCols(num_motif - i - 1);
        Eigen::RowVectorXd distances =
            (remaining_atoms.colwise() - motif_cart.col(i)).colwise().norm();
        motif_diam = std::max(motif_diam, distances.maxCoeff());
      }
    }

    // Distances are already sorted by SortedKDistances

    // Step 2: Layer-by-layer expansion with adaptive bounds
    int shell_radius = 1;
    const int max_shell_radius = 50;

    // Continue exploring shells until no improvements are found
    // This matches the AMD library behavior
    int shells_without_improvement = 0;
    const int max_shells_without_improvement =
        3; // Stop after 3 shells with no improvements

    while ((static_cast<int>(m_sq_distances.size()) < k ||
            shells_without_improvement < max_shells_without_improvement) &&
           shell_radius < max_shell_radius) {
      // Calculate bound like AMD library: bound = (sqrt(max_sqd) +
      // motif_diam)^2
      double max_sq_dist =
          m_sq_distances.empty() ? 1e10 : m_sq_distances.back();
      double bound_sq = (std::sqrt(max_sq_dist) + motif_diam) *
                        (std::sqrt(max_sq_dist) + motif_diam);

      // Remember the worst distance before processing this layer
      double worst_before =
          m_sq_distances.empty()
              ? 1e20
              : (m_sq_distances.size() >= static_cast<size_t>(k)
                     ? m_sq_distances.back()
                     : 1e20);

      bool found_improvements = false;

      generate_shell_lattice<3>(shell_radius, [&](int h, int k_lat, int l) {
        // Skip origin (already processed in initial layer)
        if (h == 0 && k_lat == 0 && l == 0)
          return;

        Vec3 lattice_cart = unit_cell.to_cartesian(
            Vec3(static_cast<double>(h), static_cast<double>(k_lat),
                 static_cast<double>(l)));

        // More generous pruning: check if ANY part of the unit cell could
        // contain neighbors
        double max_cell_extent =
            unit_cell.lengths().maxCoeff(); // Conservative upper bound
        double lattice_dist = lattice_cart.norm();

        // Only prune if even the closest possible neighbor in this cell is too
        // far
        double min_possible_dist =
            std::max(0.0, lattice_dist - max_cell_extent);
        if (min_possible_dist * min_possible_dist >= bound_sq)
          return;

        // Add all motif points translated by this lattice vector - block
        // vectorized
        size_t motif_idx = 0;
        // Process full blocks
        for (; motif_idx + BLOCK_SIZE <= num_motif; motif_idx += BLOCK_SIZE) {
          block_motif = motif_cart.middleCols<BLOCK_SIZE>(motif_idx);
          // Translate block by lattice vector
          block_motif.colwise() += lattice_cart;
          // Compute squared distances for the block
          block_sq_distances =
              (block_motif.colwise() - query_pos).colwise().squaredNorm();

          // Filter out self-distances using Eigen select
          block_sq_distances = (block_sq_distances.array() > 1e-6).select(
              block_sq_distances, std::numeric_limits<double>::infinity());
          
          int inserted_count = m_sq_distances.try_insert_batch4(block_sq_distances.data());
          if (inserted_count > 0) {
            found_improvements = true;
          }
        }

        // Process remaining atoms individually
        for (; motif_idx < num_motif; ++motif_idx) {
          Vec3 candidate = motif_cart.col(motif_idx) + lattice_cart;
          double sq_dist = (query_pos - candidate).squaredNorm();
          if (sq_dist > 1e-6) { // Skip self and very close points
            if (m_sq_distances.try_insert(sq_dist)) {
              found_improvements = true;
            }
          }
        }
      });

      if (!found_improvements) {
        // No candidates in this shell
        shells_without_improvement++;
        shell_radius++;
        continue;
      }

      // Check if we found any improvements
      double worst_after =
          m_sq_distances.empty() ? 1e20 : m_sq_distances.back();
      if (worst_after < worst_before ||
          static_cast<int>(m_sq_distances.size()) < k) {
        // We found improvements or filled empty slots
        shells_without_improvement = 0;
      } else {
        shells_without_improvement++;
      }

      shell_radius++;
    }

    // Ensure we have exactly k distances
    if (static_cast<int>(m_sq_distances.size()) < k) {
      throw std::runtime_error(
          fmt::format("Failed to find {} neighbors, only found {} neighbors", k,
                      static_cast<int>(m_sq_distances.size())));
    }

    return m_sq_distances.as_array(k);
  }

private:
  /**
   * \brief Generate lattice points in spherical shells (like AMD library)
   *
   * Generates integer lattice points (h,k,l) such that:
   * (radius-1)² < h² + k² + l² <= radius²
   *
   * This matches the AMD library's approach of using spherical shells
   */
  template <int Dim, typename Func>
  void generate_shell_lattice(int radius, Func &&func) {
    if (radius == 0) {
      func(0, 0, 0);
      return;
    }

    int radius_sq = radius * radius;
    int prev_radius_sq = (radius - 1) * (radius - 1);

    // Generate all points in the spherical shell
    for (int h = -radius; h <= radius; ++h) {
      for (int k = -radius; k <= radius; ++k) {
        for (int l = -radius; l <= radius; ++l) {
          int dist_sq = h * h + k * k + l * l;
          // Include points where (radius-1)² < dist² <= radius²
          if (dist_sq > prev_radius_sq && dist_sq <= radius_sq) {
            func(h, k, l);
          }
        }
      }
    }
  }

  /**
   * \brief Calculate minimum possible distance for a shell
   */
  double calculate_min_shell_distance_sq(int shell_radius,
                                         const crystal::UnitCell &unit_cell) {
    double min_length = unit_cell.lengths().minCoeff();
    return (shell_radius * min_length) * (shell_radius * min_length);
  }
};

/**
 * \brief Convenient wrapper for multiple query points
 */
template <int MaxK = 256>
void find_multiple_k_nearest(const crystal::Crystal &crystal,
                             const Mat3N &query_points, int k,
                             Mat &out_distances) {
  FastNeighborFinder<MaxK> finder;

  const int num_queries = query_points.cols();
  out_distances.resize(num_queries, k);

  for (int i = 0; i < num_queries; ++i) {
    // Get distances directly into a RowVector and assign to matrix row
    out_distances.row(i) =
        finder.find_k_nearest(crystal, query_points.col(i), k).sqrt();
  }
}

} // namespace occ::descriptors
