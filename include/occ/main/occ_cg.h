#pragma once
#include <CLI/App.hpp>
#include <occ/driver/cg_runner.h>

namespace occ::main {

// CGConfig and run_cg() now live in occ_driver (occ/driver/cg_runner.h) so the
// bindings can drive a crystal-growth calculation without depending on occ_main
// (which would create an occ_main <-> bindings cycle). occ_main keeps only the
// CLI subcommand wiring.
CLI::App *add_cg_subcommand(CLI::App &app);
void run_cg_subcommand(occ::driver::CGConfig const &);

} // namespace occ::main
