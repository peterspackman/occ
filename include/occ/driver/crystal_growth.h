#pragma once
#include <occ/cg/distance_partition.h>
#include <occ/cg/result_types.h>
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

struct CrystalGrowthCalculatorOptions {
  std::string solvent{"water"};
  std::string basename{"calculation"};
  std::string energy_model{"ce-b3lyp"};

  bool use_asymmetric_partition{true};
  bool use_wolf_sum{false};
  bool use_crystal_polarization{false};
  bool write_debug_output_files{true};
  double inner_radius{3.8};
  double outer_radius{3.8};
  std::string xtb_solvation_model{"cpcmx"};
  WavefunctionChoice wavefunction_choice{WavefunctionChoice::GasPhase};
};

class CrystalGrowthCalculator {
public:
  explicit CrystalGrowthCalculator(
      const crystal::Crystal &crystal,
      const CrystalGrowthCalculatorOptions &options);

  inline void set_options(const CrystalGrowthCalculatorOptions &opts) {
    m_options = opts;
  }

  inline const auto &options() const { return m_options; }
  inline auto &options() { return m_options; }

  inline auto &gas_phase_wavefunctions() { return m_gas_phase_wavefunctions; }
  inline auto &solvated_wavefunctions() { return m_solvated_wavefunctions; }

  inline auto &inner_wavefunctions() {
    switch (options().wavefunction_choice) {
    case WavefunctionChoice::Solvated:
      return solvated_wavefunctions();
    default:
      return gas_phase_wavefunctions();
    }
  }

  inline auto &outer_wavefunctions() {
    switch (options().wavefunction_choice) {
    case WavefunctionChoice::Solvated:
      return solvated_wavefunctions();
    default:
      return gas_phase_wavefunctions();
    }
  }

  inline auto &solvated_surface_properties() {
    return m_solvated_surface_properties;
  }

  inline auto &crystal() { return m_crystal; }
  inline const auto &molecules() { return m_molecules; }

  inline auto &nearest_dimers() { return m_nearest_dimers; }
  inline auto &full_dimers() { return m_full_dimers; }
  inline auto &dimer_energies() { return m_dimer_energies; }

  inline auto &solution_terms() { return m_solution_terms; }
  inline auto &interaction_energies() { return m_interaction_energies; }
  inline auto &crystal_interaction_energies() {
    return m_crystal_interaction_energies;
  }

  void set_molecule_charges(const std::vector<int> &charges);
  virtual void init_monomer_energies() = 0;
  virtual void converge_lattice_energy() = 0;

  virtual cg::MoleculeResult process_neighbors_for_symmetry_unique_molecule(
      int i, const std::string &molname) = 0;

  virtual cg::CrystalGrowthResult evaluate_molecular_surroundings() = 0;
  const auto &lattice_energies() const { return m_lattice_energies; }

protected:
  crystal::Crystal m_crystal;
  std::vector<occ::core::Molecule> m_molecules;
  std::vector<double> m_lattice_energies;
  WavefunctionList m_gas_phase_wavefunctions;
  WavefunctionList m_solvated_wavefunctions;
  std::vector<cg::SMDSolventSurfaces> m_solvated_surface_properties;
  crystal::CrystalDimers m_full_dimers;
  cg::PairEnergies m_dimer_energies;
  crystal::CrystalDimers m_nearest_dimers;
  std::vector<SolventNeighborContributionList> m_solvation_breakdowns;
  std::vector<cg::DimerResults> m_interaction_energies;
  std::vector<cg::DimerResults> m_crystal_interaction_energies;
  std::vector<double> m_solution_terms;

private:
  CrystalGrowthCalculatorOptions m_options;
};

class CEModelCrystalGrowthCalculator : public CrystalGrowthCalculator {
public:
  explicit CEModelCrystalGrowthCalculator(
      const crystal::Crystal &crystal,
      const CrystalGrowthCalculatorOptions &options);

  void init_monomer_energies() override;
  void converge_lattice_energy() override;

  cg::CrystalGrowthResult evaluate_molecular_surroundings() override;

  cg::MoleculeResult process_neighbors_for_symmetry_unique_molecule(
      int i, const std::string &molname) override;
};

class XTBCrystalGrowthCalculator : public CrystalGrowthCalculator {
public:
  explicit XTBCrystalGrowthCalculator(
      const crystal::Crystal &crystal,
      const CrystalGrowthCalculatorOptions &options);

  void init_monomer_energies() override;
  void converge_lattice_energy() override;

  cg::CrystalGrowthResult evaluate_molecular_surroundings() override;

  cg::MoleculeResult process_neighbors_for_symmetry_unique_molecule(
      int i, const std::string &molname) override;

  std::vector<double> m_gas_phase_energies;
  std::vector<occ::Vec> m_partial_charges;
  std::vector<double> m_solvated_energies;
  std::vector<double> m_dimer_energies;
  std::vector<double> m_solvated_dimer_energies;
};

} // namespace occ::driver
