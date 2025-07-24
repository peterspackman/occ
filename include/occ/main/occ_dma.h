#pragma once
#include <CLI/App.hpp>
#include <occ/driver/dma_driver.h>

namespace occ::main {

// Use the driver's DMAConfig
using occ::driver::DMAConfig;

CLI::App *add_dma_subcommand(CLI::App &app);
void run_dma_subcommand(DMAConfig const &);

} // namespace occ::main
