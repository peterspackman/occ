#pragma once
#include <CLI/App.hpp>
#include <occ/cg/crystal_growth_energies.h>
#include <occ/core/dimer.h>
#include <occ/interaction/pair_energy.h>

namespace occ::main {

struct CGConfig {
  interaction::LatticeConvergenceSettings lattice_settings;
  std::string solvent{"water"};
  std::string charge_string{""};
  std::string wavefunction_choice{"gas"};
  double cg_radius{3.8};
  int max_facets{0};
  bool write_dump_files{false};
  bool spherical{false};
  bool write_kmcpp_file{false};
  bool use_xtb{false};
  std::string xtb_solvation_model{"cpcmx"};
  bool list_solvents{false};
  bool crystal_is_atomic{false};
};

CLI::App *add_cg_subcommand(CLI::App &app);

occ::cg::CrystalGrowthResult run_cg(CGConfig const &);
void run_cg_subcommand(CGConfig const &);

} // namespace occ::main
