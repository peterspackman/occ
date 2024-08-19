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
  // If not found, return the original dimer
  return dimer;
}

std::vector<DimerIndex>
DimerMappingTable::symmetry_related_dimers(const DimerIndex &dimer) const {
  DimerIndex symmetry_unique = symmetry_unique_dimer(dimer);
  auto it = m_symmetry_related_dimers.find(symmetry_unique);
  if (it != m_symmetry_related_dimers.end()) {
    return it->second;
  }
  // If not found, return a vector containing only the original dimer
  return {dimer};
}

inline SiteIndex find_matching_position(const Mat3N &positions,
                                        const Vec3 &point,
                                        double tolerance = 1e-5) {
  IVec3 offset = point.array().floor().cast<int>();
  Vec3 p000 = point - offset.cast<double>();
  HKL hkl{offset(0), offset(1), offset(2)};
  for (int i = 0; i < positions.cols(); i++) {
    double d = (positions.col(i) - p000).squaredNorm();
    if (d < tolerance) {
      return SiteIndex{i, hkl};
    }
  }
  occ::log::trace("point: {}", point.transpose());
  occ::log::trace("offset: {}", offset.transpose());
  occ::log::trace("p000: {}", p000.transpose());
  return SiteIndex{-1, hkl};
}

DimerMappingTable
DimerMappingTable::build_dimer_table(const Crystal &crystal,
                                     const CrystalDimers &dimers,
                                     bool consider_inversion) {
  DimerMappingTable table;
  table.m_consider_inversion = consider_inversion;
  const auto &uc_mols = crystal.unit_cell_molecules();
  Mat3N molecule_centroids(3, uc_mols.size());
  for (int i = 0; i < uc_mols.size(); i++) {
    molecule_centroids.col(i) = crystal.to_fractional(uc_mols[i].centroid());
  }
  occ::log::debug("Molecule centroids:\n{}\n", molecule_centroids.transpose());

  const auto get_dimer_index = [&](const Vec3 &a_pos, const Vec3 &b_pos) {
    SiteIndex a = find_matching_position(molecule_centroids, a_pos);
    SiteIndex b = find_matching_position(molecule_centroids, b_pos);
    return DimerIndex{a, b};
  };

  auto normalize_dimer = [](const DimerIndex &dimer) {
    DimerIndex normalized = dimer;
    normalized.b.hkl -= dimer.a.hkl;
    normalized.a.hkl = HKL{0, 0, 0};
    return normalized;
  };

  auto get_canonical_dimer = [consider_inversion,
                              normalize_dimer](const DimerIndex &dimer) {
    DimerIndex normalized = normalize_dimer(dimer);
    if (consider_inversion) {
      DimerIndex inverted = normalize_dimer(DimerIndex{dimer.b, dimer.a});
      return (normalized < inverted) ? normalized : inverted;
    }
    return normalized;
  };

  const auto &symops = crystal.symmetry_operations();
  ankerl::unordered_dense::set<DimerIndex, DimerIndexHash> unique_dimers_set;

  for (const auto &mol_dimers : dimers.molecule_neighbors) {
    for (const auto &[dimer, asym_idx] : mol_dimers) {
      Vec3 a_pos = crystal.to_fractional(dimer.a().centroid());
      Vec3 b_pos = crystal.to_fractional(dimer.b().centroid());
      DimerIndex ab = get_dimer_index(a_pos, b_pos);
      DimerIndex canonical_ab = get_canonical_dimer(ab);

      if (unique_dimers_set.insert(canonical_ab).second) {
        table.m_unique_dimers.push_back(canonical_ab);
        table.m_unique_dimer_map[canonical_ab] = canonical_ab;
        table.m_symmetry_unique_dimer_map[canonical_ab] = canonical_ab;

        std::vector<DimerIndex> related_dimers;
        for (const auto &symop : symops) {
          Vec3 ta = symop.apply(a_pos);
          Vec3 tb = symop.apply(b_pos);
          DimerIndex symmetry_ab = get_dimer_index(ta, tb);
          DimerIndex canonical_symmetry_ab = get_canonical_dimer(symmetry_ab);

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
      table.m_unique_dimer_map[normalize_dimer(ab)] = canonical_ab;
      table.m_symmetry_unique_dimer_map[ab] =
          table.m_symmetry_unique_dimer_map[canonical_ab];
      table.m_symmetry_unique_dimer_map[normalize_dimer(ab)] =
          table.m_symmetry_unique_dimer_map[canonical_ab];

      occ::log::trace("Dimer AB: {} {} -> {} {} (Canonical: {} {} -> {} {})",
                      ab.a.offset, ab.a.hkl, ab.b.offset, ab.b.hkl,
                      canonical_ab.a.offset, canonical_ab.a.hkl,
                      canonical_ab.b.offset, canonical_ab.b.hkl);
    }
  }

  // Populate m_symmetry_unique_dimers
  for (const auto &dimer : table.m_unique_dimers) {
    if (table.m_symmetry_unique_dimer_map[dimer] == dimer) {
      table.m_symmetry_unique_dimers.push_back(dimer);
    }
  }

  occ::log::debug("Have {} unique dimers", table.m_unique_dimers.size());
  for (const auto &dimer : table.m_unique_dimers) {
    occ::log::debug("U {} {} -> {} {}", dimer.a.offset, dimer.a.hkl,
                    dimer.b.offset, dimer.b.hkl);
  }

  occ::log::debug("Have {} symmetry-unique dimers",
                  table.m_symmetry_unique_dimers.size());
  for (const auto &dimer : table.m_symmetry_unique_dimers) {
    occ::log::debug("SU {} {} -> {} {}", dimer.a.offset, dimer.a.hkl,
                    dimer.b.offset, dimer.b.hkl);
  }

  return table;
}

} // namespace occ::crystal
