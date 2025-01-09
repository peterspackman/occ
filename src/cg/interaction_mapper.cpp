#include <fmt/os.h>
#include <occ/cg/interaction_mapper.h>
#include <occ/core/log.h>

namespace occ::cg {

inline void write_dimer(const std::string &filename, const Dimer &dimer) {

  using occ::core::Element;
  auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                               fmt::file::CREATE);
  const auto &pos = dimer.positions();
  const auto &nums = dimer.atomic_numbers();
  output.print("{}\n", nums.rows());
  output.print("\n");
  for (size_t i = 0; i < nums.rows(); i++) {
    output.print("{:5s} {:12.5f} {:12.5f} {:12.5f}\n",
                 Element(nums(i)).symbol(), pos(0, i), pos(1, i), pos(2, i));
  }
}

InteractionMapper::InteractionMapper(const Crystal &crystal,
                                     const CrystalDimers &dimers,
                                     CrystalDimers &uc_dimers,
                                     bool consider_inversion)
    : m_crystal(crystal), m_dimers(dimers), m_uc_dimers(uc_dimers),
      m_mapping_table(DimerMappingTable::build_dimer_table(
          crystal, uc_dimers, consider_inversion)) {}

std::vector<double> InteractionMapper::map_interactions(
    const std::vector<double> &solution_terms,
    const std::vector<DimerResults> &interaction_energies_vec) {
  const auto &uc_molecules = m_crystal.unit_cell_molecules();
  std::vector<double> solution_terms_uc(m_uc_dimers.molecule_neighbors.size());

  for (size_t i = 0; i < m_uc_dimers.molecule_neighbors.size(); i++) {
    map_molecule_interactions(i, uc_molecules[i], solution_terms,
                              interaction_energies_vec, solution_terms_uc);
  }

  return solution_terms_uc;
}

void InteractionMapper::map_molecule_interactions(
    size_t mol_idx, const Molecule &mol,
    const std::vector<double> &solution_terms,
    const std::vector<DimerResults> &interaction_energies_vec,
    std::vector<double> &solution_terms_uc) {
  size_t asym_idx = mol.asymmetric_molecule_idx();
  solution_terms_uc[mol_idx] = solution_terms[asym_idx];

  auto &unit_cell_neighbors = m_uc_dimers.molecule_neighbors[mol_idx];
  const auto &asymmetric_neighbors = m_dimers.molecule_neighbors[asym_idx];
  const auto &interaction_energies = interaction_energies_vec[asym_idx];

  log_neighbor_info(mol_idx, unit_cell_neighbors.size(),
                    asymmetric_neighbors.size(), interaction_energies.size());

  map_neighbor_interactions(mol_idx, unit_cell_neighbors, asymmetric_neighbors,
                            interaction_energies);
}

void InteractionMapper::map_neighbor_interactions(
    size_t mol_idx,
    std::vector<CrystalDimers::SymmetryRelatedDimer> &unit_cell_neighbors,
    const std::vector<CrystalDimers::SymmetryRelatedDimer>
        &asymmetric_neighbors,
    const DimerResults &interaction_energies) {
  for (size_t j = 0; j < unit_cell_neighbors.size(); j++) {
    auto &[dimer, unique_idx] = unit_cell_neighbors[j];
    map_single_dimer(mol_idx, j, dimer, asymmetric_neighbors,
                     interaction_energies);
  }
}

void InteractionMapper::map_single_dimer(
    size_t mol_idx, size_t neighbor_idx, Dimer &dimer,
    const std::vector<CrystalDimers::SymmetryRelatedDimer>
        &asymmetric_neighbors,
    const DimerResults &interaction_energies) {
  const auto dimer_index =
      m_mapping_table.canonical_dimer_index(m_mapping_table.dimer_index(dimer));
  const auto &related = m_mapping_table.symmetry_related_dimers(dimer_index);

  auto related_set = build_related_set(related);
  size_t interaction_idx = find_matching_interaction(
      dimer, dimer_index, asymmetric_neighbors, related_set);

  if (interaction_idx >= interaction_energies.size()) {
    throw std::runtime_error("Matching interaction index exceeds number of "
                             "known interactions energies");
  }

  update_dimer_properties(dimer, interaction_idx, asymmetric_neighbors,
                          interaction_energies);
  log_dimer_info(neighbor_idx, dimer);
}

ankerl::unordered_dense::set<DimerIndex, DimerIndexHash>
InteractionMapper::build_related_set(
    const std::vector<DimerIndex> &related) const {
  return ankerl::unordered_dense::set<DimerIndex, DimerIndexHash>(
      related.begin(), related.end());
}

size_t InteractionMapper::find_matching_interaction(
    const Dimer &dimer, const DimerIndex &dimer_index,
    const std::vector<CrystalDimers::SymmetryRelatedDimer>
        &asymmetric_neighbors,
    const ankerl::unordered_dense::set<DimerIndex, DimerIndexHash> &related_set)
    const {

  for (size_t idx = 0; idx < asymmetric_neighbors.size(); idx++) {
    const auto &d_a = asymmetric_neighbors[idx].dimer;
    auto idx_asym =
        m_mapping_table.canonical_dimer_index(m_mapping_table.dimer_index(d_a));

    if (related_set.contains(idx_asym)) {
      return idx;
    }
  }

  handle_unmatched_dimer(dimer, dimer_index);
  return asymmetric_neighbors.size();
}

void InteractionMapper::handle_unmatched_dimer(
    const Dimer &dimer, const DimerIndex &dimer_index) const {
  auto idx = m_mapping_table.dimer_index(dimer);
  auto sidx = m_mapping_table.symmetry_unique_dimer(idx);
  write_dimer("unmatched_dimer.xyz", dimer);
  throw std::runtime_error(
      fmt::format("No matching interaction found for dimer = {}, wrote "
                  "'unmatched_dimer.xyz' file\n",
                  dimer_index));
}

void InteractionMapper::update_dimer_properties(
    Dimer &dimer, size_t interaction_id,
    const std::vector<CrystalDimers::SymmetryRelatedDimer> &asym_dimers,
    const DimerResults &energies) const {

  dimer.set_property(
      "asymmetric_dimer",
      fmt::format("dimer_{}", asym_dimers[interaction_id].unique_index));
  dimer.set_interaction_energies(energies[interaction_id].energy_components);
  dimer.set_interaction_id(interaction_id);
}

void InteractionMapper::log_neighbor_info(size_t mol_idx,
                                          size_t uc_neighbors_size,
                                          size_t asym_neighbors_size,
                                          size_t energies_size) const {
  occ::log::debug("Num asym neighbors = {}, num interaction energies = {}",
                  asym_neighbors_size, energies_size);
  occ::log::debug("Neighbors for unit cell molecule {} ({})", mol_idx,
                  uc_neighbors_size);
  occ::log::debug("{:<7s} {:>7s} {:>10s} {:>7s} {:>7s}", "N", "b", "Tb",
                  "E_int", "R");
}

void InteractionMapper::log_dimer_info(size_t neighbor_idx,
                                       const Dimer &dimer) const {
  const auto &uc_mols = m_crystal.unit_cell_molecules();
  auto idx_b = dimer.b().unit_cell_molecule_idx();
  auto shift_b = dimer.b().cell_shift() - uc_mols[idx_b].cell_shift();
  double rc = dimer.centroid_distance();

  occ::log::debug("{:<7d} {:>7d} {:>10s} {:7.2f} {:7.3f}", neighbor_idx, idx_b,
                  fmt::format("{},{},{}", shift_b[0], shift_b[1], shift_b[2]),
                  dimer.interaction_energy(), rc);
}

} // namespace occ::cg
