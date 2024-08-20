#pragma once
#include <fmt/core.h>
#include <occ/core/graph.h>
#include <occ/crystal/site_index.h>
#include <occ/crystal/unitcell.h>
#include <utility>
#include <vector>

namespace occ::core {
class Dimer;
}

namespace occ::crystal {

class Crystal;
class SiteMappingTable;
class CrystalDimers;

struct DimerIndex {
  SiteIndex a;
  SiteIndex b;

  inline HKL hkl_difference() const { return b.hkl - a.hkl; }

  inline bool operator==(const DimerIndex &other) const {
    return a == other.a && b == other.b;
  }

  inline bool operator<(const DimerIndex &other) const {
    if (a.offset != other.a.offset)
      return a.offset < other.a.offset;
    if (b.offset != other.b.offset)
      return b.offset < other.b.offset;
    if (a.hkl != other.a.hkl)
      return a.hkl < other.a.hkl;
    return b.hkl < other.b.hkl;
  }
};

struct DimerIndexHash {
  using is_avalanching = void;
  [[nodiscard]] auto
  operator()(DimerIndex const &idx) const noexcept -> uint64_t {
    static_assert(std::has_unique_object_representations_v<DimerIndex>);
    return ankerl::unordered_dense::detail::wyhash::hash(&idx, sizeof(idx));
  }
};

struct DimerMappingEdge {
  size_t source{0}, target{0};
  int symop{16484};
  HKL offset;
};

struct DimerMappingVertex {
  size_t index{0};
};

using DimerMappingGraph =
    core::graph::Graph<DimerMappingVertex, DimerMappingEdge>;

class DimerMappingTable {
public:
  static DimerMappingTable build_dimer_table(const Crystal &crystal,
                                             const CrystalDimers &dimers,
                                             bool consider_inversion);

  DimerIndex symmetry_unique_dimer(const DimerIndex &dimer) const;
  std::vector<DimerIndex>
  symmetry_related_dimers(const DimerIndex &dimer) const;

  inline const auto &unique_dimers() const { return m_unique_dimers; }
  inline const auto &symmetry_unique_dimers() const {
    return m_symmetry_unique_dimers;
  }
  inline const auto &symmetry_unique_dimer_map() const {
    return m_symmetry_unique_dimer_map;
  }

  std::pair<Vec3, Vec3> dimer_positions(const core::Dimer &) const;
  DimerIndex dimer_index(const core::Dimer &) const;
  DimerIndex dimer_index(const Vec3 &, const Vec3 &) const;

  static DimerIndex normalized_dimer_index(const DimerIndex &);
  DimerIndex canonical_dimer_index(const DimerIndex &) const;

private:
  UnitCell m_cell;
  Mat3N m_centroids;
  std::vector<DimerIndex> m_unique_dimers;
  std::vector<DimerIndex> m_symmetry_unique_dimers;
  std::vector<int> m_asym_crystal_dimer_indices;
  ankerl::unordered_dense::map<DimerIndex, DimerIndex, DimerIndexHash>
      m_unique_dimer_map;
  ankerl::unordered_dense::map<DimerIndex, DimerIndex, DimerIndexHash>
      m_symmetry_unique_dimer_map;
  ankerl::unordered_dense::map<DimerIndex, std::vector<DimerIndex>,
                               DimerIndexHash>
      m_symmetry_related_dimers;
  bool m_consider_inversion{false};
};

} // namespace occ::crystal

template <>
struct fmt::formatter<occ::crystal::DimerIndex> : nested_formatter<int> {
  auto format(const occ::crystal::DimerIndex &,
              format_context &ctx) const -> format_context::iterator;
};
