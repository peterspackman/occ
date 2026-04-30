#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::core {

// Uniform-grid spatial index over groups of atom positions with per-group
// cutoff radii (squared thresholds). Used by promolecule and void density
// evaluators to skip the full slab scan on every sample.
//
// Template parameter `Group` must expose:
//   - `g.positions` of type FMat3N (3 x N cartesian, bohr)
//   - `g.threshold` (float, squared cutoff in bohr^2)
//
// The bin grid covers the bounding box of all atoms; bin size is chosen as
// (max_cutoff_radius / subdivision). Larger `subdivision` ⇒ smaller bins,
// fewer atoms per bin, but a larger (2s+1)^3 neighbourhood walk per query.
// `subdivision = 2` (5^3 walk) is a reasonable default for moderately-dense
// slabs; experiment for very sparse or very dense systems.
class AtomCellList {
public:
  struct Entry {
    Eigen::Vector3f position;
    float threshold;       // squared cutoff radius (bohr^2)
    uint32_t interpolator; // index into the groups array
  };

  template <typename Group>
  void build(const std::vector<Group> &groups, int subdivision = 2) {
    m_entries.clear();
    m_bins.clear();

    if (groups.empty())
      return;

    // 1) Flatten entries and find max cutoff + atom bbox.
    float max_thresh = 0.0f;
    Eigen::Vector3f lo = Eigen::Vector3f::Constant(
        std::numeric_limits<float>::max());
    Eigen::Vector3f hi = Eigen::Vector3f::Constant(
        std::numeric_limits<float>::lowest());

    for (uint32_t gi = 0; gi < groups.size(); gi++) {
      const auto &g = groups[gi];
      max_thresh = std::max(max_thresh, g.threshold);
      for (int i = 0; i < g.positions.cols(); i++) {
        Entry e;
        e.position = g.positions.col(i);
        e.threshold = g.threshold;
        e.interpolator = gi;
        m_entries.push_back(e);
        lo = lo.cwiseMin(e.position);
        hi = hi.cwiseMax(e.position);
      }
    }
    if (m_entries.empty())
      return;

    // 2) Bin size = cutoff_radius / subdivision.
    m_search_radius = std::max(1, subdivision);
    float cutoff = std::sqrt(max_thresh);
    m_bin_size = cutoff / static_cast<float>(m_search_radius);
    if (m_bin_size <= 0.0f)
      m_bin_size = 1.0f;
    m_inv_bin_size = 1.0f / m_bin_size;

    // 3) Pad bbox by one cutoff in each direction so boundary queries see
    // all neighbours.
    float pad = cutoff;
    m_origin = lo.array() - pad;
    Eigen::Vector3f extent = (hi - lo).array() + 2.0f * pad;
    for (int a = 0; a < 3; a++) {
      m_dims(a) = std::max(1, static_cast<int>(std::ceil(
                                  extent(a) * m_inv_bin_size)));
    }
    m_bins.assign(static_cast<size_t>(m_dims(0)) * m_dims(1) * m_dims(2),
                  std::vector<uint32_t>{});

    // 4) Scatter entries into bins.
    for (uint32_t i = 0; i < m_entries.size(); i++) {
      Eigen::Vector3i b = bin_index(m_entries[i].position);
      m_bins[linear_index(b)].push_back(i);
    }
  }

  // Visit every atom potentially within cutoff of `p`. Visitor signature is
  // `v(uint32_t group_idx, float r_sq, const Eigen::Vector3f &atom_minus_p)`.
  // Only atoms whose squared distance is <= their threshold are reported.
  template <typename Visitor>
  inline void for_each_close(const Eigen::Vector3f &p, Visitor &&v) const {
    if (m_entries.empty())
      return;

    Eigen::Vector3i bin = bin_index_clamped(p);
    const int r = m_search_radius;
    int x0 = std::max(0, bin(0) - r);
    int x1 = std::min(m_dims(0) - 1, bin(0) + r);
    int y0 = std::max(0, bin(1) - r);
    int y1 = std::min(m_dims(1) - 1, bin(1) + r);
    int z0 = std::max(0, bin(2) - r);
    int z1 = std::min(m_dims(2) - 1, bin(2) + r);

    for (int z = z0; z <= z1; z++) {
      for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
          const auto &bin_entries =
              m_bins[(static_cast<size_t>(z) * m_dims(1) + y) * m_dims(0) + x];
          for (uint32_t idx : bin_entries) {
            const Entry &e = m_entries[idx];
            Eigen::Vector3f delta = e.position - p;
            float r_sq = delta.squaredNorm();
            if (r_sq > e.threshold)
              continue;
            v(e.interpolator, r_sq, delta);
          }
        }
      }
    }
  }

  inline bool empty() const { return m_entries.empty(); }
  inline size_t num_atoms() const { return m_entries.size(); }
  inline Eigen::Vector3i dims() const { return m_dims; }
  inline float bin_size() const { return m_bin_size; }

private:
  inline Eigen::Vector3i bin_index(const Eigen::Vector3f &p) const {
    return ((p - m_origin) * m_inv_bin_size).array().floor().cast<int>();
  }

  inline Eigen::Vector3i bin_index_clamped(const Eigen::Vector3f &p) const {
    Eigen::Vector3i b = bin_index(p);
    for (int a = 0; a < 3; a++)
      b(a) = std::clamp(b(a), 0, m_dims(a) - 1);
    return b;
  }

  inline size_t linear_index(const Eigen::Vector3i &b) const {
    return (static_cast<size_t>(b(2)) * m_dims(1) + b(1)) * m_dims(0) + b(0);
  }

  Eigen::Vector3f m_origin{Eigen::Vector3f::Zero()};
  float m_bin_size{1.0f};
  float m_inv_bin_size{1.0f};
  int m_search_radius{1};
  Eigen::Vector3i m_dims{Eigen::Vector3i::Zero()};
  std::vector<Entry> m_entries;
  std::vector<std::vector<uint32_t>> m_bins;
};

} // namespace occ::core
