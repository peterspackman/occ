#pragma once
#include <CLI/App.hpp>
#include <string>

namespace occ::main {

struct SigmaConfig {
  std::string geometry_filename{""};
  std::string output_filename{""};
  std::string method{"b3lyp"};
  std::string basis{"def2-tzvp"};
  std::string model{"cosmo-sac-2010"};
  std::string solvent{""};
  double probe_radius{0.0};
  double temperature{298.15};
  int angular_points{590};
  bool cartesian{false};
};

CLI::App *add_sigma_subcommand(CLI::App &app);
void run_sigma_subcommand(SigmaConfig const &);

} // namespace occ::main
