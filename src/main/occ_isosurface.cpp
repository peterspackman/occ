#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/eeq.h>
#include <occ/core/kdtree.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/numpy.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/geometry/marching_cubes.h>
#include <occ/io/cifparser.h>
#include <occ/io/obj.h>
#include <occ/io/ply.h>
#include <occ/io/tinyply.h>
#include <occ/io/xyz.h>
#include <occ/isosurface/curvature.h>
#include <occ/isosurface/isosurface.h>
#include <occ/main/occ_isosurface.h>

using occ::FVec;
using occ::IVec;
using occ::Mat3N;
using occ::Vec3;
using occ::core::Molecule;
using occ::isosurface::Isosurface;
using occ::isosurface::IsosurfaceGenerationParameters;
using occ::isosurface::OrbitalIndex;
using occ::isosurface::PropertyKind;
using occ::isosurface::SurfaceKind;
using occ::qm::Wavefunction;

namespace occ::main {

std::vector<IsosurfaceGenerationParameters>
generate_parameter_combinations(const IsosurfaceConfig &config) {
  std::vector<IsosurfaceGenerationParameters> params;

  IsosurfaceGenerationParameters base;
  base.surface_kind = config.surface_type();
  base.separation = config.separation;
  base.property_orbital_indices = config.orbital_indices;
  base.properties = config.surface_properties();

  if (config.surface_type() == SurfaceKind::Orbital) {
    // Generate all combinations of orbitals and isovalues
    for (double isovalue : config.isovalues) {
      for (const auto &orbital : config.orbital_indices) {
        auto p = base;
        p.isovalue = isovalue;
        p.surface_orbital_index = orbital;
        params.push_back(p);
      }
    }
  } else {
    // Just use isovalues
    for (double isovalue : config.isovalues) {
      auto p = base;
      p.isovalue = isovalue;
      params.push_back(p);
    }
  }
  return params;
}

std::string
format_output_filename(const IsosurfaceGenerationParameters &params,
                       const IsosurfaceConfig &config,
                       std::optional<std::string> label = std::nullopt) {
  if (label) {
    return fmt::format(fmt::runtime(config.output_template), *label);
  } else if (!config.has_multiple_outputs() && !label) {
    return fmt::format(fmt::runtime(config.output_template), "");
  } else {
    std::string generated_label;
    if (params.surface_kind == SurfaceKind::Orbital) {
      generated_label = fmt::format("{}_{}", params.isovalue,
                                    (params.surface_orbital_index).format());
    } else {
      generated_label = fmt::format("{}", params.isovalue);
    }
    return fmt::format(fmt::runtime(config.output_template), generated_label);
  }
}

SurfaceKind IsosurfaceConfig::surface_type() const {
  return isosurface::surface_from_string(kind);
}

std::vector<PropertyKind> IsosurfaceConfig::surface_properties() const {

  std::vector<PropertyKind> properties =
      isosurface::default_properties(have_environment_file());

  ankerl::unordered_dense::set<PropertyKind> result(properties.begin(),
                                                    properties.end());

  for (const auto &p : additional_properties) {
    result.insert(isosurface::property_from_string(p));
  }
  return std::vector<PropertyKind>(result.begin(), result.end());
}

bool IsosurfaceConfig::have_environment_file() const {
  return !environment_filename.empty();
}

bool IsosurfaceConfig::requires_crystal() const {
  return surface_type() == SurfaceKind::CrystalVoid;
}

bool IsosurfaceConfig::requires_environment() const {
  if (isosurface::surface_requires_environment(surface_type()))
    return true;

  for (const auto &prop : surface_properties()) {
    if (isosurface::property_requires_environment(prop))
      return true;
  }
  return false;
}

bool IsosurfaceConfig::requires_wavefunction() const {
  if (isosurface::surface_requires_wavefunction(surface_type()))
    return true;

  for (const auto &prop : surface_properties()) {
    if (isosurface::property_requires_wavefunction(prop))
      return true;
  }
  return false;
}

bool IsosurfaceConfig::has_multiple_outputs() const {
  return isovalues.size() > 1 || ((surface_type() == SurfaceKind::Orbital) &&
                                  (orbital_indices.size() > 1));
}

void ensure_isosurface_configuration_valid(const IsosurfaceConfig &config,
                                           bool have_wavefunction,
                                           bool have_crystal) {
  if (config.requires_wavefunction() && !have_wavefunction) {
    throw std::runtime_error(
        "Surface, or surface properties require a wavefunction");
  }
  if (config.requires_environment() && !config.have_environment_file()) {
    throw std::runtime_error(
        "Surface, or surface properties require an environment");
  }
  if (config.requires_crystal() && !have_crystal) {
    throw std::runtime_error(
        "Surface, or surface properties requires a crystal structure");
  }
}

CLI::App *add_isosurface_subcommand(CLI::App &app) {
  CLI::App *iso =
      app.add_subcommand("isosurface", "compute molecular isosurfaces");
  auto config = std::make_shared<IsosurfaceConfig>();

  iso->add_option("geometry", config->geometry_filename,
                  "input geometry file (xyz)")
      ->required();

  iso->add_option("environment", config->environment_filename,
                  "environment geometry file (xyz)");

  iso->add_option("--kind", config->kind, "surface kind");

  iso->add_flag("--binary,!--ascii", config->binary_output,
                "Write binary/ascii file format (default binary)");

  iso->add_option("--wavefunction,-w", config->wavefunction_filename,
                  "Wavefunction filename for geometry");
  iso->add_option("--wfn-rotation,--wfn_rotation", config->wfn_rotation,
                  "Rotation for supplied wavefunction (row major order)")
      ->expected(9);

  iso->add_option("--wfn-translation,--wfn_translation",
                  config->wfn_translation,
                  "Translation for wavefunction (Angstrom)")
      ->expected(3);

  iso->add_option("--properties,--additional_properties",
                  config->additional_properties,
                  "Additional properties to compute");

  iso->add_option("--max-depth", config->max_depth, "Maximum voxel depth");
  iso->add_option("--separation", config->separation,
                  "targt voxel separation (Angstrom)");

  iso->add_option("--isovalue", config->isovalues,
                  "target isovalue(s)")
      ->expected(-1); // Allow multiple isovalues

  iso->add_option("--orbitals", config->orbitals_input,
                  "orbital indices (for orbital surfaces)");

  iso->add_option("-o,--output-template", config->output_template,
                  "template for output files (use {} for index placement)");
  iso->add_option("--background-density", config->background_density,
                  "add background density to close surface");

  iso->fallthrough();
  iso->callback([config]() { run_isosurface_subcommand(*config); });
  return iso;
}

Wavefunction load_wfn(const IsosurfaceConfig &config) {
  Wavefunction wfn;
  if (Wavefunction::is_likely_wavefunction_filename(
          config.wavefunction_filename)) {
    occ::log::info("Loading wavefunction data from '{}'",
                   config.wavefunction_filename);
    wfn = Wavefunction::load(config.wavefunction_filename);
  } else if (Wavefunction::is_likely_wavefunction_filename(
                 config.geometry_filename)) {
    occ::log::info("Loading wavefunction data from geometry file '{}'",
                   config.geometry_filename);
    wfn = Wavefunction::load(config.geometry_filename);
  }
  if (wfn.atoms.size() > 0) {
    occ::log::info("Loaded wavefunction, applying transformation:");
    occ::Mat3 rotation = Eigen::Map<const Mat3RM>(config.wfn_rotation.data());
    occ::log::info("Rotation\n{}", format_matrix(rotation));
    occ::Vec3 translation =
        Eigen::Map<const Vec3>(config.wfn_translation.data()) *
        occ::units::ANGSTROM_TO_BOHR;
    occ::log::info("Translation (Bohr) [{:.5f}, {:.5f}, {:.5f}]",
                   translation(0), translation(1), translation(2));
    wfn.apply_transformation(rotation, translation);
  }
  return wfn;
}

void run_isosurface_subcommand(IsosurfaceConfig config) {

  struct Geometries {
    Molecule interior;
    Molecule exterior;
    Wavefunction wavefunction;
    std::optional<crystal::Crystal> crystal;

    inline bool have_wavefunction() const {
      return wavefunction.atoms.size() > 0;
    }
  };

  bool use_wfn_mol = false;
  config.separation *= occ::units::ANGSTROM_TO_BOHR;

  const auto properties_to_compute = config.surface_properties();
  if (properties_to_compute.size() > 0) {
    occ::log::debug("Properties to compute:");
    for (const auto &prop : properties_to_compute) {
      occ::log::debug("  {}", isosurface::property_to_string(prop));
    }
  }

  if (Wavefunction::is_likely_wavefunction_filename(config.geometry_filename)) {
    config.wavefunction_filename = config.geometry_filename;
    use_wfn_mol = true;
  }

  Geometries geometry;

  geometry.wavefunction = load_wfn(config);

  bool have_crystal =
      occ::io::CifParser::is_likely_cif_filename(config.geometry_filename);
  ensure_isosurface_configuration_valid(config, geometry.have_wavefunction(),
                                        have_crystal);

  if (use_wfn_mol) {
    geometry.interior = Molecule(geometry.wavefunction.atoms);
  } else if (!have_crystal) {
    geometry.interior =
        occ::io::molecule_from_xyz_file(config.geometry_filename);
  }

  if (!config.environment_filename.empty()) {
    geometry.exterior =
        occ::io::molecule_from_xyz_file(config.environment_filename);
  }
  if (have_crystal) {
    occ::io::CifParser parser;
    geometry.crystal = parser.parse_crystal_from_file(config.geometry_filename);
  }

  const auto surface_type = config.surface_type();
  occ::log::info("Isosurface kind: {}",
                 isosurface::surface_to_string(surface_type));

  // Ensure we have at least one isovalue
  if (config.isovalues.empty()) {
    config.isovalues.push_back(0.02);
  }

  if (surface_type == SurfaceKind::Orbital) {
    config.orbital_indices =
        isosurface::parse_orbital_descriptions(config.orbitals_input);
  }

  auto parameter_combinations =
      occ::main::generate_parameter_combinations(config);

  // Loop over all isovalues
  for (const auto &params : parameter_combinations) {

    const double isovalue = params.isovalue;
    if (params.surface_kind == SurfaceKind::Orbital) {
      occ::log::info("Processing surface isovalue = {} (orbital = {})",
                     isovalue, params.surface_orbital_index.format());
    } else {
      occ::log::info("Processing surface isovalue = {}", isovalue);
    }

    isosurface::IsosurfaceCalculator calculator;
    calculator.set_parameters(params);

    if (geometry.interior.size() > 0) {
      calculator.set_molecule(geometry.interior);
    }

    if (geometry.exterior.size() > 0) {
      calculator.set_environment(geometry.exterior);
    }

    if (geometry.have_wavefunction()) {
      calculator.set_wavefunction(geometry.wavefunction);
    }

    if (geometry.crystal) {
      calculator.set_crystal(geometry.crystal.value());
    }

    if (!calculator.validate()) {
      throw std::runtime_error("Isosurface parameters invalid: " +
                               calculator.error_message());
    }
    calculator.compute();

    isosurface::Isosurface result = calculator.isosurface();

    // Generate output filename based on index
    std::string output_filename = format_output_filename(params, config);
    occ::log::info("Writing surface to {}", output_filename);
    occ::io::write_ply_mesh(output_filename, result, config.binary_output);
  }
}

} // namespace occ::main
