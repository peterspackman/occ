#include <occ/core/log.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_mapping_table.h>
#include <occ/crystal/site_mapping_table.h>

namespace occ::crystal {

DimerIndex
DimerMappingTable::symmetry_unique_dimer(const DimerIndex &dimer) const {
  auto it = m_symmetry_unique_dimer_map.find(dimer);
  if (it != m_symmetry_unique_dimer_map.end()) {
    return it->second;
  }
  // return dimer if not found
  return dimer;
}

std::vector<DimerIndex>
DimerMappingTable::symmetry_related_dimers(const DimerIndex &dimer) const {
  DimerIndex symmetry_unique = symmetry_unique_dimer(dimer);
  auto it = m_symmetry_related_dimers.find(symmetry_unique);
  if (it != m_symmetry_related_dimers.end()) {
    return it->second;
  }
  // return just dimer If not found
  return {dimer};
}

inline Vec3 clean_small_values(const Vec3 &v, double epsilon = 1e-14) {
  return v.unaryExpr(
      [epsilon](double x) { return (std::abs(x) < epsilon) ? 0.0 : x; });
}

inline Vec3 wrap_to_unit_cell(const Vec3 &v) {
  return Vec3(v.array() - v.array().floor());
}

inline SiteIndex find_matching_position(const Mat3N &positions,
                                        const Vec3 &point,
                                        double tolerance = 1e-5) {
  Vec3 cleaned_point = clean_small_values(point);

  Vec3 wrapped_point = wrap_to_unit_cell(cleaned_point);
  IVec3 cell_offset =
      (cleaned_point.array() - wrapped_point.array()).round().cast<int>();

  int matching_index = -1;
  double min_distance = std::numeric_limits<double>::max();

  for (int i = 0; i < positions.cols(); i++) {
    Vec3 diff = wrapped_point - clean_small_values(positions.col(i));
    diff = diff.array() - diff.array().round(); 
    double d = diff.squaredNorm();

    if (d < min_distance) {
      min_distance = d;
      matching_index = i;
    }
  }

  if (min_distance < tolerance * tolerance) {
    return SiteIndex{matching_index,
                     HKL{cell_offset(0), cell_offset(1), cell_offset(2)}};
  }

  occ::log::warn("Could not find matching position for:");
  occ::log::warn("original point: {}", point.transpose());
  occ::log::warn("cleaned point: {}", cleaned_point.transpose());
  occ::log::warn("wrapped_point: {}", wrapped_point.transpose());
  occ::log::warn("cell_offset: {}", cell_offset.transpose());
  occ::log::warn("min_distance: {}", std::sqrt(min_distance));
  return SiteIndex{-1, HKL{cell_offset(0), cell_offset(1), cell_offset(2)}};
}

std::pair<Vec3, Vec3>
DimerMappingTable::dimer_positions(const core::Dimer &dimer) const {
  Vec3 a_pos = m_cell.to_fractional(dimer.a().centroid());
  Vec3 b_pos = m_cell.to_fractional(dimer.b().centroid());
  return {a_pos, b_pos};
}

DimerIndex DimerMappingTable::dimer_index(const core::Dimer &dimer) const {
  auto [a_pos, b_pos] = dimer_positions(dimer);
  SiteIndex a = find_matching_position(m_centroids, a_pos);
  SiteIndex b = find_matching_position(m_centroids, b_pos);
  if (a == b) {
    occ::log::warn("Matching positions for dimer with");
    occ::log::warn("Dimer.a.positions:\n{}\n",
                   dimer.a().positions().transpose());
    occ::log::warn("Dimer.b.positions:\n{}\n",
                   dimer.b().positions().transpose());
    occ::log::warn("a_pos = {}", a_pos.transpose());
    occ::log::warn("b_pos = {}", b_pos.transpose());
    occ::log::warn("{}", DimerIndex{a, b});
  }
  return DimerIndex{a, b};
}

DimerIndex DimerMappingTable::dimer_index(const Vec3 &a_pos,
                                          const Vec3 &b_pos) const {
  SiteIndex a = find_matching_position(m_centroids, a_pos);
  SiteIndex b = find_matching_position(m_centroids, b_pos);
  return DimerIndex{a, b};
}

DimerIndex DimerMappingTable::normalized_dimer_index(const DimerIndex &idx) {
  DimerIndex normalized = idx;
  normalized.b.hkl -= idx.a.hkl;
  normalized.a.hkl = HKL{0, 0, 0};
  return normalized;
}

DimerIndex
DimerMappingTable::canonical_dimer_index(const DimerIndex &idx) const {
  DimerIndex normalized = normalized_dimer_index(idx);
  if (m_consider_inversion) {
    DimerIndex inverted = normalized_dimer_index(DimerIndex{idx.b, idx.a});
    return (normalized < inverted) ? normalized : inverted;
  }
  return normalized;
}

DimerMappingTable
DimerMappingTable::build_dimer_table(const Crystal &crystal,
                                     const CrystalDimers &dimers,
                                     bool consider_inversion) {
  DimerMappingTable table;
  table.m_consider_inversion = consider_inversion;
  table.m_cell = crystal.unit_cell();

  const auto &uc_mols = crystal.unit_cell_molecules();
  table.m_centroids = Mat3N(3, uc_mols.size());
  for (int i = 0; i < uc_mols.size(); i++) {
    table.m_centroids.col(i) = crystal.to_fractional(uc_mols[i].centroid());
  }
  occ::log::trace("Dimer mapping molecule centroids:\n{}\n", table.m_centroids);

  const auto &symops = crystal.symmetry_operations();
  ankerl::unordered_dense::set<DimerIndex, DimerIndexHash> unique_dimers_set;

  for (const auto &mol_dimers : dimers.molecule_neighbors) {
    for (const auto &[dimer, asym_idx] : mol_dimers) {

      auto [a_pos, b_pos] = table.dimer_positions(dimer);
      DimerIndex ab = table.dimer_index(dimer);
      DimerIndex norm_ab = table.normalized_dimer_index(ab);
      DimerIndex canonical_ab = table.canonical_dimer_index(ab);

      if (unique_dimers_set.insert(canonical_ab).second) {
        table.m_unique_dimers.push_back(canonical_ab);
        table.m_unique_dimer_map[canonical_ab] = canonical_ab;
        table.m_symmetry_unique_dimer_map[canonical_ab] = canonical_ab;

        std::vector<DimerIndex> related_dimers;
        for (const auto &symop : symops) {
          Vec3 ta = symop.apply(a_pos);
          Vec3 tb = symop.apply(b_pos);
          DimerIndex symmetry_ab = table.dimer_index(ta, tb);
          DimerIndex canonical_symmetry_ab =
              table.canonical_dimer_index(symmetry_ab);

          if (unique_dimers_set.insert(canonical_symmetry_ab).second) {
            table.m_unique_dimers.push_back(canonical_symmetry_ab);
            table.m_symmetry_unique_dimer_map[canonical_symmetry_ab] =
                canonical_ab;
          }
          related_dimers.push_back(canonical_symmetry_ab);
          table.m_unique_dimer_map[canonical_symmetry_ab] =
              canonical_symmetry_ab;
        }
        table.m_symmetry_related_dimers[canonical_ab] = related_dimers;
      }

      // Always map both ab and normalized ab to the canonical dimer
      table.m_unique_dimer_map[ab] = canonical_ab;
      table.m_unique_dimer_map[norm_ab] = canonical_ab;
      table.m_symmetry_unique_dimer_map[ab] =
          table.m_symmetry_unique_dimer_map[canonical_ab];
      table.m_symmetry_unique_dimer_map[norm_ab] =
          table.m_symmetry_unique_dimer_map[canonical_ab];

      occ::log::trace("Canonical: {} {} -> {} {}", canonical_ab.a.offset,
                      canonical_ab.a.hkl, canonical_ab.b.offset,
                      canonical_ab.b.hkl);
    }
  }

  // Populate m_symmetry_unique_dimers
  for (const auto &dimer : table.m_unique_dimers) {
    if (table.m_symmetry_unique_dimer_map[dimer] == dimer) {
      table.m_symmetry_unique_dimers.push_back(dimer);
    }
  }

  occ::log::trace("Dimmer mapping table has {} unique dimers",
                  table.m_unique_dimers.size());
  for (const auto &dimer : table.m_unique_dimers) {
    occ::log::trace("U {} {} -> {} {}", dimer.a.offset, dimer.a.hkl,
                    dimer.b.offset, dimer.b.hkl);
  }

  occ::log::trace("Dimer mapping table has {} symmetry-unique dimers",
                  table.m_symmetry_unique_dimers.size());
  for (const auto &dimer : table.m_symmetry_unique_dimers) {
    occ::log::trace("SU {} {} -> {} {}", dimer.a.offset, dimer.a.hkl,
                    dimer.b.offset, dimer.b.hkl);
  }

  return table;
}

} // namespace occ::crystal

auto fmt::formatter<occ::crystal::DimerIndex>::format(
    const occ::crystal::DimerIndex &idx,
    format_context &ctx) const -> decltype(ctx.out()) {
  return fmt::format_to(ctx.out(), "DimerIndex [{} {} -> {} {}]", idx.a.offset,
                        idx.a.hkl, idx.b.offset, idx.b.hkl);
}
