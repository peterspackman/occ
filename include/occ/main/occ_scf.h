#pragma once
#include "occ/qm/wavefunction.h"
#include <CLI/App.hpp>
#include <occ/io/occ_input.h>

namespace occ::main {

CLI::App *add_scf_subcommand(CLI::App &app);
void run_scf_subcommand(occ::io::OccInput config);
void read_input_file(const std::string &filename, io::OccInput &config);
occ::qm::Wavefunction run_scf_external(occ::io::OccInput config, bool write_wfn = false);

} // namespace occ::main
