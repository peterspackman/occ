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
  /// Geometry of the solvent molecule. When given, the openCOSMO-RS
  /// solvation free energy is assembled against a conductor cavity computed
  /// for it, rather than against a stored σ-profile.
  std::string solvent_geometry{""};
  /// Where to write this molecule's openCOSMO-RS segment ensemble, so it can
  /// be used as a cached solvent later.
  std::string segments_filename{""};
  /// Use the openCOSMO-RS kernel against the stored ensemble for `--solvent`
  /// rather than the COSMO-SAC σ-potential.
  bool opencosmors{false};
  /// Solute liquid molar volume, Å³ per molecule, for the reference-state
  /// term. Non-positive drops that term.
  double solvent_volume_liquid{0.0};
  int num_rings{0};
  double probe_radius{0.0};
  double temperature{298.15};
  int angular_points{590};
  bool unconstrained_charge{false};
  bool cartesian{false};
};

CLI::App *add_sigma_subcommand(CLI::App &app);
void run_sigma_subcommand(SigmaConfig const &);

} // namespace occ::main
