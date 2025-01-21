#pragma once
#include <CLI/App.hpp>
#include <occ/interaction/lattice_convergence_settings.h>

namespace occ::main {

CLI::App *add_elat_subcommand(CLI::App &app);
void run_elat_subcommand(interaction::LatticeConvergenceSettings const &);

} // namespace occ::main
