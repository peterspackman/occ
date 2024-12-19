#pragma once
#include <occ/cg/crystal_growth_energies.h>
#include <occ/cg/distance_partition.h>
#include <occ/cg/smd_solvation.h>
#include <occ/cg/solvation_contribution.h>
#include <occ/cg/solvent_surface.h>
#include <occ/core/dimer.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/point_group.h>
#include <occ/core/timings.h>
#include <occ/crystal/crystal.h>
#include <occ/driver/single_point.h>
#include <occ/interaction/pair_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/io/wavefunction_json.h>
#include <occ/qm/wavefunction.h>
#include <string>
#include <vector>

namespace occ::driver {

enum class WavefunctionChoice { GasPhase, Solvated };
using occ::core::Molecule;
using occ::qm::Wavefunction;
using WavefunctionList = std::vector<Wavefunction>;
using SolventNeighborContributionList = std::vector<cg::SolvationContribution>;

struct AssignedEnergy {
  bool is_nn{true};
  double energy{0.0};
};

std::vector<AssignedEnergy> assign_interaction_terms_to_nearest_neighbours(
    const crystal::CrystalDimers::MoleculeNeighbors &neighbors,
    const std::vector<double> &dimer_energies, double cg_radius);

std::vector<occ::Vec3>
calculate_net_dipole(const WavefunctionList &wavefunctions,
                     const crystal::CrystalDimers &crystal_dimers);

class CEModelCrystalGrowthCalculator {
public:
  CEModelCrystalGrowthCalculator(const crystal::Crystal &crystal,
                                 const std::string &solvent);
  void set_basename(const std::string &basename) { m_basename = basename; }
  void set_wavefunction_choice(WavefunctionChoice choice) {
    m_wfn_choice = choice;
  }

  void set_output_verbosity(bool output) { m_output = output; }

  void set_energy_model(const std::string &model) { m_model = model; }

  inline auto &gas_phase_wavefunctions() { return m_gas_phase_wavefunctions; }
  inline auto &solvated_wavefunctions() { return m_solvated_wavefunctions; }
  inline auto &inner_wavefunctions() {
    switch (m_wfn_choice) {
    case WavefunctionChoice::Solvated:
      return solvated_wavefunctions();
    default:
      return gas_phase_wavefunctions();
    }
  }

  inline auto &outer_wavefunctions() {
    switch (m_wfn_choice) {
    case WavefunctionChoice::Solvated:
      return solvated_wavefunctions();
    default:
      return gas_phase_wavefunctions();
    }
  }
  inline auto &solvated_surface_properties() {
    return m_solvated_surface_properties;
  }

  void dipole_correction();

  inline auto &crystal() { return m_crystal; }
  inline const auto &name() { return m_basename; }
  inline const auto &solvent() { return m_solvent; }
  inline const auto &molecules() { return m_molecules; }

  inline auto &nearest_dimers() { return m_nearest_dimers; }
  inline auto &full_dimers() { return m_full_dimers; }
  inline auto &dimer_energies() { return m_dimer_energies; }

  inline auto &solution_terms() { return m_solution_terms; }
  inline auto &interaction_energies() { return m_interaction_energies; }
  inline auto &crystal_interaction_energies() {
    return m_crystal_interaction_energies;
  }

  inline void set_use_wolf_sum(bool value) { m_use_wolf_sum = value; }
  inline void set_use_crystal_polarization(bool value) {
    m_use_crystal_polarization = value;
  }

  void init_monomer_energies();

  void converge_lattice_energy(double inner_radius, double outer_radius);

  void set_molecule_charges(const std::vector<int> &charges);

  std::tuple<cg::EnergyTotal, std::vector<cg::CrystalGrowthDimer>>
  process_neighbors_for_symmetry_unique_molecule(int i,
                                                 const std::string &molname);

  cg::CrystalGrowthResult evaluate_molecular_surroundings();
  const auto &lattice_energies() const { return m_lattice_energies; }

private:
  bool m_output{true};
  bool m_use_wolf_sum{false};
  bool m_use_crystal_polarization{false};
  WavefunctionChoice m_wfn_choice{WavefunctionChoice::GasPhase};
  crystal::Crystal m_crystal;
  std::vector<occ::core::Molecule> m_molecules;
  std::vector<double> m_lattice_energies;
  std::string m_solvent;
  std::string m_model;
  std::string m_basename;
  WavefunctionList m_gas_phase_wavefunctions;
  WavefunctionList m_solvated_wavefunctions;
  std::vector<cg::SMDSolventSurfaces> m_solvated_surface_properties;
  crystal::CrystalDimers m_full_dimers;
  cg::PairEnergies m_dimer_energies;
  crystal::CrystalDimers m_nearest_dimers;
  std::vector<SolventNeighborContributionList> m_solvation_breakdowns;
  std::vector<cg::Energies> m_interaction_energies;
  std::vector<cg::Energies> m_crystal_interaction_energies;
  std::vector<double> m_solution_terms;
  double m_inner_radius{0.0}, m_outer_radius{0.0};
};

class XTBCrystalGrowthCalculator {
public:
  XTBCrystalGrowthCalculator(const crystal::Crystal &crystal,
                             const std::string &solvent);

  // do nothing
  inline void set_wavefunction_choice(WavefunctionChoice choice) {}
  inline void set_energy_model(const std::string &model) { m_model = model; }
  inline void set_use_crystal_polarization(bool value) {}
  void set_output_verbosity(bool output) { m_output = output; }

  inline auto &crystal() { return m_crystal; }
  inline const auto &name() { return m_basename; }
  inline const auto &solvent() { return m_solvent; }
  inline const auto &molecules() { return m_molecules; }

  inline auto &nearest_dimers() { return m_nearest_dimers; }
  inline auto &full_dimers() { return m_full_dimers; }
  inline auto &dimer_energies() { return m_dimer_energies; }

  inline auto &solution_terms() { return m_solution_terms; }
  inline auto &interaction_energies() { return m_interaction_energies; }
  inline auto &crystal_interaction_energies() {
    return m_crystal_interaction_energies;
  }

  inline void set_use_wolf_sum(bool value) { m_use_wolf_sum = value; }

  inline void set_basename(const std::string &basename) {
    m_basename = basename;
  }
  inline void set_solvation_model(const std::string &model) {
    m_solvation_model = model;
  }

  void set_molecule_charges(const std::vector<int> &charges);

  void converge_lattice_energy(double inner_radius, double outer_radius);

  occ::cg::CrystalGrowthResult evaluate_molecular_surroundings();

  void init_monomer_energies();

private:
  std::tuple<occ::cg::EnergyTotal, std::vector<occ::cg::CrystalGrowthDimer>>
  process_neighbors_for_symmetry_unique_molecule(int i,
                                                 const std::string &molname);

  const auto &lattice_energies() const { return m_lattice_energies; }

  crystal::Crystal m_crystal;
  std::vector<occ::core::Molecule> m_molecules;
  std::vector<double> m_lattice_energies;
  std::string m_solvent;
  std::string m_basename;
  std::string m_model{"gfn2-xtb"};
  std::string m_solvation_model{"cpcmx"};
  bool m_output{true};
  std::vector<double> m_gas_phase_energies;
  std::vector<occ::Vec> m_partial_charges;
  std::vector<double> m_solvated_energies;
  crystal::CrystalDimers m_full_dimers;
  std::vector<double> m_dimer_energies;
  std::vector<double> m_solvated_dimer_energies;
  crystal::CrystalDimers m_nearest_dimers;
  std::vector<cg::Energies> m_interaction_energies;
  std::vector<cg::Energies> m_crystal_interaction_energies;
  std::vector<double> m_solution_terms;
  double m_inner_radius{0.0}, m_outer_radius{0.0};
  bool m_use_wolf_sum{false};
};

} // namespace occ::driver
