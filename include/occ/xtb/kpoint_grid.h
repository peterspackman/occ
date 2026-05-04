#pragma once
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::xtb {

// A k-point with weight. `k` is in 1/Bohr (Cartesian). Weights sum to 1.
struct KPoint {
  Vec3 k;
  double weight;
};

// Build a Γ-centered uniform Monkhorst-Pack mesh of n1×n2×n3 k-points in the
// first Brillouin zone:
//   k_{j1,j2,j3} = (j1/n1) b1 + (j2/n2) b2 + (j3/n3) b3,  j_i = 0..n_i-1
// All weights = 1/(n1·n2·n3). No symmetry reduction yet.
//
// `reciprocal_bohr` columns are b1, b2, b3 (1/Bohr).
std::vector<KPoint> monkhorst_pack_grid(const Mat3 &reciprocal_bohr, int n1,
                                         int n2, int n3);

} // namespace occ::xtb
