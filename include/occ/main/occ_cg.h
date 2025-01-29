#pragma once
#include <CLI/App.hpp>
#include <occ/cg/result_types.h>
#include <occ/core/dimer.h>
#include <occ/interaction/lattice_convergence_settings.h>

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
  bool asymmetric_solvent_contribution{false};
  bool gamma_point_molecules{true};
  std::string xtb_solvation_model{"cpcmx"};
  bool list_solvents{false};
  bool crystal_is_atomic{false};
};

CLI::App *add_cg_subcommand(CLI::App &app);

occ::cg::CrystalGrowthResult run_cg(CGConfig const &);
void run_cg_subcommand(CGConfig const &);

} // namespace occ::main
