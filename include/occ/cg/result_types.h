#pragma once
#include <ankerl/unordered_dense.h>
#include <string>
#include <vector>

namespace occ::cg {
using CGEnergyComponents = ankerl::unordered_dense::map<std::string, double>;

namespace components {

constexpr const char *total = "Total";
constexpr const char *crystal_total = "Crystal Total";
constexpr const char *crystal_nn = "Crystal (reassigned)";
constexpr const char *solvation_ab = "Solvation (A->B)";
constexpr const char *solvation_ba = "Solvation (B->A)";
constexpr const char *solvation_total = "Solvation Total";
constexpr const char *coulomb = "Coulomb";
constexpr const char *polarization = "Polarization";
constexpr const char *dispersion = "Dispersion";
constexpr const char *repulsion = "Repulsion";
constexpr const char *exchange = "Exchange";

} // namespace components

struct DimerResult {
  bool is_nearest_neighbor{false};
  size_t unique_idx{0};

  CGEnergyComponents energy_components{{components::total, 0.0},
                                       {components::crystal_total, 0.0},
                                       {components::solvation_total, 0.0}};

  double total_energy() const;

  void set_energy_component(const std::string &key, double value);
  double energy_component(const std::string &key) const;

  bool has_energy_component(const std::string &key) const;
};

struct MoleculeResult {
  CGEnergyComponents energy_components{{components::total, 0.0},
                                       {components::crystal_total, 0.0},
                                       {components::solvation_total, 0.0}};

  std::vector<DimerResult> dimer_results;
  bool has_inversion_symmetry{true};

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
