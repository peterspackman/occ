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

/**
 * \brief Represents a pair of sites in a crystal structure
 *
 * DimerIndex stores two SiteIndex objects representing the endpoints of a
 * dimer, along with functionality to compare and analyze the relative
 * positions.
 */
struct DimerIndex {
  SiteIndex a; ///< First site of the dimer
  SiteIndex b; ///< Second site of the dimer

  /**
   * \brief Calculates the HKL difference between the two sites
   * \return HKL vector representing the difference between sites
   */
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
  [[nodiscard]] auto operator()(DimerIndex const &idx) const noexcept
      -> uint64_t {
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

/**
 * \brief Maps and manages symmetry relationships between dimers in a crystal
 *
 * DimerMappingTable provides functionality to analyze and track symmetry
 * relationships between dimers in a crystal structure, including mapping
 * between unique and symmetry-related dimers.
 */
class DimerMappingTable {
public:
  /**
   * \brief Constructs a dimer mapping table for a crystal structure
   * \param crystal The crystal structure to analyze
   * \param dimers The set of crystal dimers to process
   * \param consider_inversion Whether to include inversion symmetry
   * \return A new DimerMappingTable instance
   */
  static DimerMappingTable build_dimer_table(const Crystal &crystal,
                                             const CrystalDimers &dimers,
                                             bool consider_inversion);

  /**
   * \brief Finds the symmetry-unique representative for a given dimer
   * \param dimer The dimer index to analyze
   * \return DimerIndex representing the symmetry-unique form
   */
  DimerIndex symmetry_unique_dimer(const DimerIndex &dimer) const;
  /**
   * \brief Gets all symmetry-related forms of a given dimer
   * \param dimer The dimer index to analyze
   * \return Vector of symmetry-related dimer indices
   */
  std::vector<DimerIndex>
  symmetry_related_dimers(const DimerIndex &dimer) const;

  /**
   * \brief Gets all unique dimers (unique up to translation)
   * \return Vector of unique dimer indices
   */
  inline const auto &unique_dimers() const { return m_unique_dimers; }

  /**
   * \brief Gets all symmetry-unique dimers (including crystal symmetry)
   * \return Vector of symmetry-unique dimer indices
   */
  inline const auto &symmetry_unique_dimers() const {
    return m_symmetry_unique_dimers;
  }

  inline const auto &symmetry_unique_dimer_map() const {
    return m_symmetry_unique_dimer_map;
  }

  /**
   * \brief Gets a pair of positions for the given dimer (fractional)
   * \param the dimer object in question
   * \return pair of fractional coordinates for each half of the dimer
   */
  std::pair<Vec3, Vec3> dimer_positions(const core::Dimer &) const;

  /**
   * \brief Gets the dimer index for the given dimer
   * \param the dimer object in question
   * \return DimerIndex representing the provided dimer
   */
  DimerIndex dimer_index(const core::Dimer &) const;

  /**
   * \brief Gets the dimer index for the given dimer specified by positions
   * \param The two positions of dimer end points (fractional)
   * \return DimerIndex representing the provided dimer
   */
  DimerIndex dimer_index(const Vec3 &, const Vec3 &) const;

  /**
   * \brief Gets the Normalized dimer index (starting from 000 cell)
   * \param The DimerIndex to normalize
   * \return DimerIndex representing the provided dimer
   */
  static DimerIndex normalized_dimer_index(const DimerIndex &);

  /**
   * \brief Gets the Normalized dimer index (a,b sorted and starting from 000
   * cell)
   * \param The DimerIndex to canonicalize
   * \return DimerIndex representing the provided dimer
   */
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
  auto format(const occ::crystal::DimerIndex &, format_context &ctx) const
      -> format_context::iterator;
};
