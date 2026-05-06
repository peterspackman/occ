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
  std::vector<int> kpoints{1, 1, 1};
  bool print_charges{true};
  // Crystal-only: after the periodic SCC, also run a molecular SCC for each
  // symmetry-unique molecule and report the lattice binding energy
  // (E_crystal − Σ E_mol_i) per molecule, in kJ/mol.
  bool lattice_energy{false};
};

CLI::App *add_tb_subcommand(CLI::App &app);
void run_tb_subcommand(const TbConfig &config);

} // namespace occ::main
