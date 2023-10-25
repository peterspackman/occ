#pragma once
#include <CLI/App.hpp>
#include <occ/io/occ_input.h>

namespace occ::main {

CLI::App *add_scf_subcommand(CLI::App &app);
void run_scf_subcommand(occ::io::OccInput &config);

} // namespace occ::main
