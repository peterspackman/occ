#pragma once
#include <CLI/App.hpp>
#include <occ/main/pair_energy.h>

namespace occ::main {

CLI::App *add_elat_subcommand(CLI::App &app);
void run_elat_subcommand(LatticeConvergenceSettings const &);

} // namespace occ::main
