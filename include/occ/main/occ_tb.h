#pragma once
#include <CLI/App.hpp>
#include <string>
#include <vector>

namespace occ::main {

struct TbConfig {
  std::string filename;        // .xyz / .cif / .gen / .com
  double charge{0.0};
  bool include_multipoles{true};
  bool include_dispersion{true};
  bool multipole_ewald{true};
  std::vector<int> kpoints{1, 1, 1};
  bool print_charges{true};
};

CLI::App *add_tb_subcommand(CLI::App &app);
void run_tb_subcommand(const TbConfig &config);

} // namespace occ::main
