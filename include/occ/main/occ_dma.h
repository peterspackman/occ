#pragma once
#include <CLI/App.hpp>
#include <occ/dma/dma.h>

namespace occ::main {

struct DMAConfig {
  std::string wavefunction_filename;
  std::string punch_filename;
  dma::DMASettings settings;
};

CLI::App *add_dma_subcommand(CLI::App &app);
void run_dma_subcommand(DMAConfig const &);

} // namespace occ::main
