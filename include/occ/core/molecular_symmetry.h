#pragma once
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::core {

/**
 * Result of attempting a transformation with grouped permutations
 */
struct SymmetryMappingResult {
  bool success; ///< Whether a valid mapping was found
  std::vector<int>
      permutation; ///< Permutation mapping transformed atoms to reference atoms
  double rmsd;     ///< Final RMSD of the best match
};

/**
 * Attempts to find a transformation mapping from one set of atoms to another
 * using grouped permutations based on provided labels.
 *
 * This function applies the given transformation matrix to positions B and then
 * finds permutations within atom groups to match reference positions A.
 * Groups are formed by the provided labels (e.g., asymmetric unit indices or
 * atomic numbers). The transformed positions are automatically aligned by
 * centroid.
 *
 * @param labels_A The labels for reference atoms (A)
 * @param positions_A The reference positions (A) to match against
 * @param labels_B The labels for atoms to transform (B)
 * @param positions_B The positions (B) to transform and permute
 * @param transformation The transformation matrix to apply to positions_B
 * @param rmsd_threshold The RMSD threshold for considering a match (default:
 * 1e-3)
 * @return SymmetryMappingResult containing success status, permutation, and
 * RMSD
 */
SymmetryMappingResult try_transformation_with_grouped_permutations(
    const IVec &labels_A, const Mat3N &positions_A, const IVec &labels_B,
    const Mat3N &positions_B, const Mat3 &transformation,
    double rmsd_threshold = 1e-3);

} // namespace occ::core