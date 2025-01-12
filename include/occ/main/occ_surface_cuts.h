#pragma once

#include <CLI/App.hpp>
#include <vector>

namespace occ::main {

struct SurfaceCutsConfig {
  std::string filename{""};
  double dmin{0.1};
  size_t count{10};
};

CLI::App *add_surface_cuts_subcommand(CLI::App &app);
void run_surface_cuts_subcommand(SurfaceCutsConfig);
} // namespace occ::main
