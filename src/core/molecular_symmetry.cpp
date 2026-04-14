#include <ankerl/unordered_dense.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <occ/core/molecular_symmetry.h>

namespace occ::core {

namespace {

// Solve square linear assignment (min-cost) with O(n^3) Hungarian algorithm.
// Returns assignment a_to_b where a_to_b[i] = j.
std::vector<int> hungarian_min_assignment(const std::vector<std::vector<double>> &cost) {
  const int n = static_cast<int>(cost.size());
  if (n == 0) return {};
  for (const auto &row : cost) {
    if (static_cast<int>(row.size()) != n) {
      throw std::runtime_error("hungarian_min_assignment requires square cost matrix");
    }
  }

  std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
  std::vector<int> p(n + 1, 0), way(n + 1, 0);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
    std::vector<char> used(n + 1, false);
    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = std::numeric_limits<double>::infinity();
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[j]) continue;
        const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> a_to_b(n, -1);
  for (int j = 1; j <= n; ++j) {
    if (p[j] > 0) {
      a_to_b[p[j] - 1] = j - 1;
    }
  }
  return a_to_b;
}

} // namespace

namespace {

// Greedy nearest-unused-neighbor matching. O(n^2) per group. Can produce a
// suboptimal assignment when two atoms compete for the same nearest partner.
std::vector<int> greedy_nn_min_assignment(
    const std::vector<std::vector<double>> &cost) {
  const size_t n_rows = cost.size();
  const size_t n_cols = n_rows == 0 ? 0 : cost.front().size();
  std::vector<int> mapping(n_rows, -1);
  std::vector<char> used(n_cols, false);
  for (size_t i = 0; i < n_rows; ++i) {
    double best = std::numeric_limits<double>::infinity();
    int best_j = -1;
    for (size_t j = 0; j < n_cols; ++j) {
      if (used[j]) continue;
      if (cost[i][j] < best) {
        best = cost[i][j];
        best_j = static_cast<int>(j);
      }
    }
    if (best_j >= 0) {
      mapping[i] = best_j;
      used[best_j] = true;
    }
  }
  return mapping;
}

} // namespace

SymmetryMappingResult try_transformation_with_grouped_permutations(
    const IVec &labels_A, const Mat3N &positions_A, const IVec &labels_B,
    const Mat3N &positions_B, const Mat3 &transformation,
    double rmsd_threshold, AtomMatchingMethod method) {

  // Apply transformation to positions B
  Mat3N transformed_positions = transformation * positions_B;

  // Calculate centroid translation to align with positions A
  Vec3 centroid_A = positions_A.rowwise().mean();
  Vec3 centroid_transformed = transformed_positions.rowwise().mean();
  Vec3 translation = centroid_A - centroid_transformed;
  transformed_positions.colwise() += translation;

  // Group atoms by labels
  ankerl::unordered_dense::map<int, std::vector<int>> groups_A, groups_B;
  for (int i = 0; i < labels_A.size(); i++) {
    groups_A[labels_A(i)].push_back(i);
  }
  for (int i = 0; i < labels_B.size(); i++) {
    groups_B[labels_B(i)].push_back(i);
  }

  // Check that groups match
  if (groups_A.size() != groups_B.size()) {
    return {false, {}, std::numeric_limits<double>::infinity()};
  }

  // Initialize result permutation
  std::vector<int> result_permutation(labels_A.size(), -1);
  double total_rmsd = 0.0;
  bool all_groups_matched = true;

  // Try permutations for each group
  for (const auto &[label, indices_A] : groups_A) {
    if (groups_B.find(label) == groups_B.end()) {
      all_groups_matched = false;
      break;
    }

    auto indices_B = groups_B[label];
    if (indices_A.size() != indices_B.size()) {
      all_groups_matched = false;
      break;
    }

    // Get positions for this group
    Mat3N group_positions_A(3, indices_A.size());
    Mat3N group_transformed_B(3, indices_B.size());

    for (size_t i = 0; i < indices_A.size(); i++) {
      group_positions_A.col(i) = positions_A.col(indices_A[i]);
      group_transformed_B.col(i) = transformed_positions.col(indices_B[i]);
    }

    std::vector<std::vector<double>> cost(indices_A.size(),
                                          std::vector<double>(indices_B.size(), 0.0));
    for (size_t i = 0; i < indices_A.size(); ++i) {
      for (size_t j = 0; j < indices_B.size(); ++j) {
        cost[i][j] =
            (group_positions_A.col(i) - group_transformed_B.col(j)).squaredNorm();
      }
    }
    const std::vector<int> group_mapping =
        (method == AtomMatchingMethod::Hungarian)
            ? hungarian_min_assignment(cost)
            : greedy_nn_min_assignment(cost);

    double group_diff_norm_sq = 0.0;
    for (size_t i = 0; i < indices_A.size(); ++i) {
      const int j = group_mapping[i];
      if (j < 0 || j >= static_cast<int>(indices_B.size())) {
        all_groups_matched = false;
        break;
      }
      group_diff_norm_sq += cost[i][j];
    }
    if (!all_groups_matched) break;

    const double group_rmsd =
        std::sqrt(group_diff_norm_sq / std::max<size_t>(1, indices_A.size()));

    // Check this label-group match quality against per-atom RMSD threshold.
    if (group_rmsd > rmsd_threshold) {
      all_groups_matched = false;
      break;
    }

    // Store the mapping for this group
    // group_mapping[i] tells us that position i in group A matches position
    // group_mapping[i] in group B For molecule::permute, we need to know which
    // atom from A goes to each position So result_permutation[i] tells us which
    // atom from A should be at position i to match B
    for (size_t i = 0; i < indices_A.size(); i++) {
      // indices_A[i] in reference matches indices_B[group_mapping[i]] in
      // transformed So to reorder reference to match transformed, position
      // indices_B[group_mapping[i]] should get atom indices_A[i]
      int ref_pos = indices_A[i];
      int transformed_pos = indices_B[group_mapping[i]];
      result_permutation[transformed_pos] = ref_pos;
    }

    total_rmsd += group_rmsd * group_rmsd;
  }

  if (!all_groups_matched) {
    return {false, {}, std::numeric_limits<double>::infinity()};
  }

  // Calculate final RMSD
  total_rmsd = std::sqrt(total_rmsd / labels_A.size());

  return {true, result_permutation, total_rmsd};
}

} // namespace occ::core
