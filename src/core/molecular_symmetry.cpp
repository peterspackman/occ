#include <ankerl/unordered_dense.h>
#include <cmath>
#include <occ/core/molecular_symmetry.h>

namespace occ::core {

SymmetryMappingResult try_transformation_with_grouped_permutations(
    const IVec &labels_A, const Mat3N &positions_A, const IVec &labels_B,
    const Mat3N &positions_B, const Mat3 &transformation,
    double rmsd_threshold) {

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

    // Use nearest neighbor matching instead of trying all permutations
    std::vector<int> group_mapping(indices_A.size(), -1);
    std::vector<bool> used_B(indices_B.size(), false);
    double group_diff_norm_sq = 0.0;

    // For each point in reference group A, find nearest unused point in
    // transformed group B
    for (size_t i = 0; i < indices_A.size(); i++) {
      double min_dist_sq = std::numeric_limits<double>::infinity();
      int best_j = -1;

      for (size_t j = 0; j < indices_B.size(); j++) {
        if (used_B[j])
          continue;

        double dist_sq = (group_positions_A.col(i) - group_transformed_B.col(j))
                             .squaredNorm();
        if (dist_sq < min_dist_sq) {
          min_dist_sq = dist_sq;
          best_j = j;
        }
      }

      if (best_j >= 0) {
        group_mapping[i] = best_j;
        used_B[best_j] = true;
        group_diff_norm_sq += min_dist_sq;
      }
    }

    double best_group_rmsd = std::sqrt(group_diff_norm_sq);

    // Check if this group has a good match (scale threshold by group size)
    if (best_group_rmsd > rmsd_threshold * indices_A.size()) {
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

    total_rmsd += best_group_rmsd * best_group_rmsd;
  }

  if (!all_groups_matched) {
    return {false, {}, std::numeric_limits<double>::infinity()};
  }

  // Calculate final RMSD
  total_rmsd = std::sqrt(total_rmsd / labels_A.size());

  return {true, result_permutation, total_rmsd};
}

} // namespace occ::core