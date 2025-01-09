#pragma once
#include <occ/cg/result_types.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_mapping_table.h>

namespace occ::cg {

using Crystal = occ::crystal::Crystal;
using CrystalDimers = occ::crystal::CrystalDimers;
using DimerMappingTable = occ::crystal::DimerMappingTable;
using DimerIndex = occ::crystal::DimerIndex;
using DimerIndexHash = occ::crystal::DimerIndexHash;
using Molecule = occ::core::Molecule;
using Dimer = occ::core::Dimer;

class InteractionMapper {
public:
  InteractionMapper(const Crystal &crystal, const CrystalDimers &dimers,
                    CrystalDimers &uc_dimers, bool consider_inversion = false);

  std::vector<double>
  map_interactions(const std::vector<double> &solution_terms,
                   const std::vector<DimerResults> &interaction_energies_vec);

private:
  void map_molecule_interactions(
      size_t mol_idx, const Molecule &mol,
      const std::vector<double> &solution_terms,
      const std::vector<DimerResults> &interaction_energies_vec,
      std::vector<double> &solution_terms_uc);

  void map_neighbor_interactions(
      size_t mol_idx,
      std::vector<CrystalDimers::SymmetryRelatedDimer> &unit_cell_neighbors,
      const std::vector<CrystalDimers::SymmetryRelatedDimer>
          &asymmetric_neighbors,
      const DimerResults &interaction_energies);

  void map_single_dimer(size_t mol_idx, size_t neighbor_idx, Dimer &dimer,
                        const std::vector<CrystalDimers::SymmetryRelatedDimer>
                            &asymmetric_neighbors,
                        const DimerResults &interaction_energies);

  ankerl::unordered_dense::set<DimerIndex, DimerIndexHash>
  build_related_set(const std::vector<DimerIndex> &related) const;

  size_t find_matching_interaction(
      const Dimer &dimer, const DimerIndex &dimer_index,
      const std::vector<CrystalDimers::SymmetryRelatedDimer>
          &asymmetric_neighbors,
      const ankerl::unordered_dense::set<DimerIndex, DimerIndexHash>
          &related_set) const;

  void handle_unmatched_dimer(const Dimer &dimer,
                              const DimerIndex &dimer_index) const;

  void update_dimer_properties(
      Dimer &dimer, size_t interaction_id,
      const std::vector<CrystalDimers::SymmetryRelatedDimer> &asym_dimers,
      const DimerResults &energies) const;

  void log_neighbor_info(size_t mol_idx, size_t uc_neighbors_size,
                         size_t asym_neighbors_size,
                         size_t energies_size) const;

  void log_dimer_info(size_t neighbor_idx, const Dimer &dimer) const;

private:
  const Crystal &m_crystal;
  const CrystalDimers &m_dimers;
  CrystalDimers &m_uc_dimers;
  DimerMappingTable m_mapping_table;
};

} // namespace occ::cg
