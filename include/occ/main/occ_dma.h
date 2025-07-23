#pragma once
#include <CLI/App.hpp>
#include <occ/dma/dma.h>
#include <ankerl/unordered_dense.h>

namespace occ::main {

struct DMAConfig {
  // Input/output files
  std::string wavefunction_filename;
  std::string punch_filename{"dma.punch"};  // Default output filename
  
  // Basic DMA settings
  dma::DMASettings settings;
  
  // Atom-specific settings
  ankerl::unordered_dense::map<std::string, double> atom_radii;     // Element symbol -> radius in Angstrom
  ankerl::unordered_dense::map<std::string, int> atom_limits;       // Element symbol -> max rank
  
  // Output options
  bool verbose{false};
  bool write_punch{true};  // Write punch file by default
  
  // Additional site specifications (future enhancement)
  // std::vector<std::array<double, 3>> extra_sites;  // For bond midpoints etc.
};

CLI::App *add_dma_subcommand(CLI::App &app);
void run_dma_subcommand(DMAConfig const &);

} // namespace occ::main
