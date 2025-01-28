#pragma once
#include <CLI/App.hpp>
#include <occ/isosurface/orbital_index.h>
#include <occ/isosurface/surface_types.h>
#include <vector>

namespace occ::main {

struct IsosurfaceConfig {

  std::string geometry_filename{""};
  std::string environment_filename{""};
  size_t max_depth{4};
  double separation{0.2};
  std::vector<double> isovalues{0.02};
  double background_density{0.0};
  bool use_hashed_mc{false};
  std::string wavefunction_filename{""};
  std::vector<double> wfn_rotation{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  std::vector<double> wfn_translation{0.0, 0.0, 0.0};
  int orbital_index{0};
  bool binary_output{true};
  std::string kind{"promolecule_density"};
  std::string output_template{"surface{}.ply"};
  std::vector<std::string> additional_properties{};
  std::vector<occ::isosurface::OrbitalIndex> orbital_indices{};
  std::string orbitals_input{"homo"};

  std::vector<isosurface::PropertyKind> surface_properties() const;
  isosurface::SurfaceKind surface_type() const;

  bool requires_crystal() const;
  bool requires_environment() const;
  bool requires_wavefunction() const;
  bool have_environment_file() const;

  // New helper methods
  std::string
  format_output_filename(size_t index,
                         std::optional<std::string> label = std::nullopt) const;
  bool has_multiple_outputs() const;
};

CLI::App *add_isosurface_subcommand(CLI::App &app);
void run_isosurface_subcommand(IsosurfaceConfig);
} // namespace occ::main
