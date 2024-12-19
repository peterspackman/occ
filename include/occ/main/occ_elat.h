#pragma once
#include <CLI/App.hpp>
#include <occ/interaction/pair_energy.h>

namespace occ::main {

CLI::App *add_elat_subcommand(CLI::App &app);
void run_elat_subcommand(interaction::LatticeConvergenceSettings const &);

} // namespace occ::main
