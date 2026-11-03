#pragma once
// Grid batches and pair screening shared by the seminumerical exchange build
// and its derivatives.
#include <algorithm>
#include <occ/core/linear_algebra.h>
#include <occ/qm/seminumerical_exchange.h>
#include <utility>
#include <vector>

namespace occ::qm::cosx::detail {

/// Consecutive points [start, start + count) of one atom's grid; `global` is
/// the first point's offset in the concatenated grid (GridBatches order).
struct AtomBatch {
  int atom;
  Eigen::Index start, count;
  size_t global;
};

/// Every atom's grid in batches of at most `block` points, so each batch has
/// one parent atom. `blocks`, if given, receives each atom's (offset, count)
/// in the concatenated grid.
inline std::vector<AtomBatch>
atom_batches(const std::vector<occ::numint::AtomGrid> &grids,
             Eigen::Index block,
             std::vector<std::pair<size_t, size_t>> *blocks = nullptr) {
  std::vector<AtomBatch> batches;
  size_t total = 0;
  for (int a = 0; a < static_cast<int>(grids.size()); a++) {
    const Eigen::Index n = grids[a].points.cols();
    if (blocks)
      blocks->push_back({total, static_cast<size_t>(n)});
    for (Eigen::Index s = 0; s < n; s += block)
      batches.push_back({a, s, std::min(block, n - s), total + s});
    total += n;
  }
  return batches;
}

/// Atomic positions, one column per atom.
inline Mat3N atom_positions(const std::vector<occ::core::Atom> &atoms) {
  Mat3N positions(3, atoms.size());
  for (size_t a = 0; a < atoms.size(); a++)
    positions.col(a) = atoms[a].position();
  return positions;
}

/// Point coordinates in structure-of-arrays form for the ESP kernels, padded
/// to npad points with copies of the last.
inline void soa_coordinates(Eigen::Ref<const Mat3N> pts, int npad,
                            std::vector<double> &x, std::vector<double> &y,
                            std::vector<double> &z) {
  const Eigen::Index npt = pts.cols();
  x.resize(npad);
  y.resize(npad);
  z.resize(npad);
  for (int i = 0; i < npad; ++i) {
    const Eigen::Index c = std::min<Eigen::Index>(i, npt - 1);
    x[i] = pts(0, c);
    y[i] = pts(1, c);
    z[i] = pts(2, c);
  }
}

/// A shell pair's bound on its ESP integrals over a batch of points: the
/// charge bound, tightened to q1 / (R - spread) when the batch's bounding
/// sphere is a distance R - spread > 0 beyond the pair's spread (far = {q1,
/// spread}, ESPEvaluator::pair_far_estimate). The 1/R tail is never cut.
inline double pair_charge_bound(double charge, const Vec3 &pair_centre,
                                const double *far, const GridBatchInfo &batch) {
  const double R =
      (pair_centre - batch.center).norm() - batch.radius - far[1];
  return R > 0.0 ? std::min(charge, far[0] / R) : charge;
}

} // namespace occ::qm::cosx::detail
