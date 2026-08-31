#pragma once
#include <CLI/App.hpp>
#include <string>

namespace occ::main {

struct CosmoRSConfig {
  std::string geometry_filename{""};
  std::string method{"b3lyp"};
  std::string basis{"def2-tzvp"};
  /// Named solvent, resolved against the shipped `.rsseg` ensembles.
  std::string solvent{""};
  /// Geometry of the solvent molecule, as an alternative to a named solvent:
  /// its conductor cavity is computed here rather than loaded.
  std::string solvent_geometry{""};
  /// Where to write this molecule's segment ensemble, so it can be reused as
  /// a cached solvent later.
  std::string segments_filename{""};
  /// Liquid-phase volume per solute molecule, Angstrom^3, for the
  /// reference-state term. Non-positive drops that term.
  double liquid_volume{0.0};
  /// Rings in the solute, for the ring correction. Negative counts them from
  /// the bond graph.
  int num_rings{-1};
  double probe_radius{0.0};
  double temperature{298.15};
  int angular_points{590};
  bool unconstrained_charge{false};
  bool cartesian{false};
  bool list_solvents{false};
};

CLI::App *add_cosmors_subcommand(CLI::App &app);
void run_cosmors_subcommand(CosmoRSConfig const &);

} // namespace occ::main
