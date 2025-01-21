#pragma once
#include <occ/interaction/pair_energy.h>

namespace occ::interaction {

struct LatticeConvergenceSettings {
  double min_radius{3.8};       // angstroms
  double max_radius{30.0};      // angstroms
  double radius_increment{3.8}; // angstroms
  double energy_tolerance{1.0}; // kj/mol
  bool wolf_sum{false};
  bool crystal_field_polarization{false};
  std::string model_name{"ce-b3lyp"};
  std::string crystal_filename{""};
  std::string output_json_filename{""};
  bool spherical_basis{false};
  std::string charge_string;
};

} // namespace occ::interaction
