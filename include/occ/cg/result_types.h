#pragma once
#include <ankerl/unordered_dense.h>
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

  cg::EnergyTotal total;

  double total_energy() const;
  void add_dimer_result(const DimerResult &dimer);

  void set_energy_component(const std::string &key, double value);
  double energy_component(const std::string &key) const;

  bool has_energy_component(const std::string &key) const;
};

struct CrystalGrowthResult {
  std::vector<MoleculeResult> molecule_results;
};

} // namespace occ::cg
