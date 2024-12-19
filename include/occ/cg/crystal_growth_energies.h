#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/core/dimer.h>
#include <occ/interaction/pairinteraction.h>
#include <vector>

namespace occ::cg {

using PairEnergies = std::vector<occ::interaction::CEEnergyComponents>;
using EnergyComponents = ankerl::unordered_dense::map<std::string, double>;
using Energies = std::vector<EnergyComponents>;

struct EnergyTotal {
  double crystal_energy{0.0};
  double interaction_energy{0.0};
  double solution_term{0.0};
};

struct DimerSolventTerm {
  double ab{0.0};
  double ba{0.0};
  double total{0.0};
};

struct CrystalGrowthDimer {
  core::Dimer dimer;
  int unique_dimer_index{-1};
  double interaction_energy{0.0};
  DimerSolventTerm solvent_term{};
  double crystal_contribution{0.0};
  bool nearest_neighbor{false};
};

struct CrystalGrowthResult {
  std::vector<std::vector<CrystalGrowthDimer>> pair_energies;
  std::vector<cg::EnergyTotal> total_energies;
};

} // namespace occ::cg
