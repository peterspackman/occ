#pragma once
#include "occ/qm/wavefunction.h"
#include <CLI/App.hpp>
#include <occ/io/occ_input.h>

namespace occ::main {

CLI::App *add_scf_subcommand(CLI::App &app);
// driver_from_cli: config.driver was given on the command line, so it wins
// over the driver named in the input file.
void run_scf_subcommand(occ::io::OccInput config, bool driver_from_cli = false);
void read_input_file(const std::string &filename, io::OccInput &config);
occ::qm::Wavefunction run_scf_external(occ::io::OccInput config, bool write_wfn = false);

} // namespace occ::main
