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
  // Crystal-only: after the periodic SCC, also run a molecular SCC for each
  // symmetry-unique molecule and report the lattice binding energy
  // (E_crystal − Σ E_mol_i) per molecule, in kJ/mol.
  bool lattice_energy{false};
  // Run a geometry optimization (Berny/internal-coords). Molecular only.
  // Uses the analytical gradient. Writes <input>_opt.xyz on convergence and
  // <input>_trj.xyz with the trajectory.
  bool optimize{false};
  // Compute the numerical Hessian and report vibrational frequencies.
  // Molecular only. Costs `6N` analytical-gradient evaluations.
  bool frequencies{false};
  // Step size (Bohr) for the FD Hessian.
  double freq_step_bohr{0.005};
  // Project translations and rotations out of the mass-weighted Hessian
  // before diagonalisation (ORCA-style PROJECTTR).
  bool freq_project_tr_rot{true};
};

CLI::App *add_tb_subcommand(CLI::App &app);
void run_tb_subcommand(const TbConfig &config);

} // namespace occ::main
