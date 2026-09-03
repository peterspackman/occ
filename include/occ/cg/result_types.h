#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/cg/morphology_types.h>
#include <occ/core/dimer.h>
#include <occ/interaction/pairinteraction.h>
#include <string>
#include <vector>

namespace occ::cg {

using PairEnergies = std::vector<occ::interaction::CEEnergyComponents>;
using CGEnergyComponents = ankerl::unordered_dense::map<std::string, double>;
using CGEnergies = std::vector<CGEnergyComponents>;

namespace components {

constexpr const char *total = "Total";
constexpr const char *crystal_total = "Crystal Total";
constexpr const char *crystal_nn = "Crystal (redistributed)";
constexpr const char *solvation_ab = "Solvation (A->B)";
constexpr const char *solvation_ba = "Solvation (B->A)";
constexpr const char *solvation_total = "Solvation Total";
constexpr const char *coulomb = "Coulomb";
constexpr const char *polarization = "Polarization";
constexpr const char *dispersion = "Dispersion";
constexpr const char *repulsion = "Repulsion";
constexpr const char *exchange = "Exchange";

/// Descriptors are merged into the dimer's interaction map under this prefix
/// so surface-level code can tell them apart from energies.
constexpr const char *descriptor_prefix = "descriptor:";

} // namespace components

struct DimerSolventTerm {
  double ab{0.0};
  double ba{0.0};
  double total{0.0};
};

struct EnergyTotal {
  double crystal_energy{0.0};
  double interaction_energy{0.0};
  double solution_term{0.0};
};

struct DimerResult {
  occ::core::Dimer dimer;
  bool is_nearest_neighbor{false};
  int unique_idx{0};

  CGEnergyComponents energy_components{{components::total, 0.0},
                                       {components::crystal_total, 0.0},
                                       {components::solvation_total, 0.0}};

  /// Non-energy per-contact quantities carried by the solvation model, in
  /// whatever units it documents: the σ model emits `reorganisation` in
  /// kJ/mol and `hbond_area` in Å², plus a `<cavity>_area` in Bohr².
  CGEnergyComponents descriptors{};

  double total_energy() const;

  void set_energy_component(const std::string &key, double value);
  double energy_component(const std::string &key) const;

  bool has_energy_component(const std::string &key) const;
};

using DimerResults = std::vector<DimerResult>;

struct MoleculeResult {
  CGEnergyComponents energy_components{{components::total, 0.0},
                                       {components::crystal_total, 0.0},
                                       {components::solvation_total, 0.0}};

  std::vector<DimerResult> dimer_results;
  bool has_inversion_symmetry{true};

  /// Descriptors summed over this molecule's nearest-neighbour contacts.
  CGEnergyComponents descriptors{};

  cg::EnergyTotal total;

  double total_energy() const;
  void add_dimer_result(const DimerResult &dimer);

  void set_energy_component(const std::string &key, double value);
  double energy_component(const std::string &key) const;

  bool has_energy_component(const std::string &key) const;
};

struct CrystalGrowthResult {
  std::vector<MoleculeResult> molecule_results;
  /// Particle size/shape-dependent energies; empty unless morphology was
  /// requested (CGConfig::compute_morphology).
  MorphologyResult morphology;
};

} // namespace occ::cg
