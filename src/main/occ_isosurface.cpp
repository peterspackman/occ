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
using occ::io::IsosurfaceMesh;
using occ::io::VertexProperties;
using occ::qm::Wavefunction;
namespace iso = occ::isosurface;

namespace occ::main {

std::string to_string(IsosurfaceConfig::Property prop) {
  switch (prop) {
  case IsosurfaceConfig::Property::Dnorm:
    return "dnorm";
  case IsosurfaceConfig::Property::Dint_norm:
    return "di_norm";
  case IsosurfaceConfig::Property::Dint:
    return "di";
  case IsosurfaceConfig::Property::Dext_norm:
    return "de_norm";
  case IsosurfaceConfig::Property::Dext:
    return "de";
  case IsosurfaceConfig::Property::FragmentPatch:
    return "fragment_patch";
  case IsosurfaceConfig::Property::ShapeIndex:
    return "shape_index";
  case IsosurfaceConfig::Property::Curvedness:
    return "curvedness";
  case IsosurfaceConfig::Property::PromoleculeDensity:
    return "promolecule_density";
  case IsosurfaceConfig::Property::EEQ_ESP:
    return "eeq_esp";
  case IsosurfaceConfig::Property::ESP:
    return "esp";
  case IsosurfaceConfig::Property::ElectronDensity:
    return "electron_density";
  case IsosurfaceConfig::Property::DeformationDensity:
    return "deformation_density";
  case IsosurfaceConfig::Property::Orbital:
    return "orbital_density";
  case IsosurfaceConfig::Property::SpinDensity:
    return "spin_density";
  default:
    return "unknown_property";
  }
}

std::string to_string(IsosurfaceConfig::Surface surface) {
  switch (surface) {
  case IsosurfaceConfig::Surface::PromoleculeDensity:
    return "promolecule_density";
  case IsosurfaceConfig::Surface::Hirshfeld:
    return "hirshfeld";
  case IsosurfaceConfig::Surface::EEQ_ESP:
    return "eeq_esp";
  case IsosurfaceConfig::Surface::ESP:
    return "esp";
  case IsosurfaceConfig::Surface::ElectronDensity:
    return "electron_density";
  case IsosurfaceConfig::Surface::DeformationDensity:
    return "deformation_density";
  case IsosurfaceConfig::Surface::Orbital:
    return "orbital_density";
  case IsosurfaceConfig::Surface::SpinDensity:
    return "spin_density";
  case IsosurfaceConfig::Surface::CrystalVoid:
    return "void";
  default:
    return "unknown_surface";
  }
}

struct ParameterCombination {
  double isovalue{0.0};
  std::optional<OrbitalIndex> orbital_index;
};

std::vector<ParameterCombination>
generate_parameter_combinations(const IsosurfaceConfig &config) {
  std::vector<ParameterCombination> params;

  if (config.surface_type() == IsosurfaceConfig::Surface::Orbital) {
    // Generate all combinations of orbitals and isovalues
    for (double isovalue : config.isovalues) {
      for (const auto &orbital : config.orbital_indices) {
        params.push_back({isovalue, orbital});
      }
    }
  } else {
    // Just use isovalues
    for (double isovalue : config.isovalues) {
      params.push_back({isovalue});
    }
  }
  return params;
}

void parse_orbital_descriptions(std::vector<OrbitalIndex> &indices,
                                const std::string &input) {
  std::vector<std::string> orbital_specs;
  std::stringstream ss(input);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    if (!token.empty()) {
      orbital_specs.push_back(token);
    }
  }

  indices.clear();
  for (const auto &spec : orbital_specs) {
    std::string spec_lower = spec;
    std::transform(spec_lower.begin(), spec_lower.end(), spec_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Remove all spaces
    spec_lower.erase(
        std::remove_if(spec_lower.begin(), spec_lower.end(), ::isspace),
        spec_lower.end());

    OrbitalIndex idx;

    // Check if it's a pure number
    if (std::all_of(spec_lower.begin(), spec_lower.end(),
                    [](char c) { return std::isdigit(c) || c == '-'; })) {
      try {
        int num = std::stoi(spec_lower);
        if (num == 0) {
          throw std::runtime_error("Orbital indices must be non-zero");
        }
        idx.reference = OrbitalIndex::Reference::Absolute;
        idx.offset = num - 1; // Convert to 0-based
      } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("Invalid orbital index: {}", spec));
      }
    } else {
      bool is_homo = spec_lower.find("homo") == 0;
      bool is_lumo = spec_lower.find("lumo") == 0;

      if (!is_homo && !is_lumo) {
        throw std::runtime_error(
            fmt::format("Invalid orbital specification: {}", spec));
      }

      idx.reference = is_homo ? OrbitalIndex::Reference::HOMO
                              : OrbitalIndex::Reference::LUMO;

      // Just parse everything after homo/lumo as the offset
      std::string offset_str = spec_lower.substr(4);
      try {
        idx.offset = offset_str.empty() ? 0 : std::stoi(offset_str);
      } catch (const std::exception &e) {
        throw std::runtime_error(
            fmt::format("Invalid orbital offset in: {}", spec));
      }
    }
    indices.push_back(idx);
  }

  if (indices.empty()) {
    throw std::runtime_error("No valid orbital specifications provided");
  }
}

std::string
format_output_filename(const ParameterCombination &params,
                       const IsosurfaceConfig &config,
                       std::optional<std::string> label = std::nullopt) {
  if (label) {
    return fmt::format(fmt::runtime(config.output_template), *label);
  } else if (!config.has_multiple_outputs() && !label) {
    return fmt::format(fmt::runtime(config.output_template), "");
  } else {
    std::string generated_label;
    if (params.orbital_index) {
      generated_label = fmt::format("{}_{}", params.isovalue,
                                    (*params.orbital_index).format());
    } else {
      generated_label = fmt::format("{}", params.isovalue);
    }
    return fmt::format(fmt::runtime(config.output_template), generated_label);
  }
}

int OrbitalIndex::resolve(int num_alpha, int num_beta) const {
  int homo = num_alpha - 1;
  switch (reference) {
  case Reference::Absolute:
    return offset;
  case Reference::HOMO:
    return homo + offset;
  case Reference::LUMO:
    return (homo + 1) + offset;
  }
  throw std::runtime_error("Invalid orbital reference type");
}

// Format for output
std::string OrbitalIndex::format() const {
  switch (reference) {
  case Reference::Absolute:
    return std::to_string(offset); // Convert to 1-based for display
  case Reference::HOMO:
    return offset == 0 ? "HOMO" : fmt::format("HOMO{:+d}", offset);
  case Reference::LUMO:
    return offset == 0 ? "LUMO" : fmt::format("LUMO{:+d}", offset);
  }
  throw std::runtime_error("Invalid orbital reference type");
}

} // namespace occ::main

template <typename F>
IsosurfaceMesh as_mesh(const F &func, const std::vector<float> &vertices,
                       const std::vector<uint32_t> &indices,
                       const std::vector<float> &normals,
                       const std::vector<float> &curvature) {

  IsosurfaceMesh result;

  func.remap_vertices(vertices, result.vertices);
  result.normals.resize(vertices.size());
  result.faces.resize(indices.size());
  result.gaussian_curvature.reserve(curvature.size() / 2);
  result.mean_curvature.reserve(curvature.size() / 2);

  for (size_t i = 0; i < normals.size(); i += 3) {
    Eigen::Vector3f normal =
        Eigen::Vector3f(normals[i], normals[i + 1], normals[i + 2]);
    result.normals[i] = normal(0);
    result.normals[i + 1] = normal(1);
    result.normals[i + 2] = normal(2);
  }

  for (size_t i = 0; i < curvature.size(); i += 2) {
    result.mean_curvature.push_back(curvature[i]);
    result.gaussian_curvature.push_back(curvature[i + 1]);
  }

  // winding is backward for some reason out of the marching cubes.
  for (size_t i = 0; i < indices.size(); i += 3) {
    result.faces[i] = indices[i];
    result.faces[i + 1] = indices[i + 1];
    result.faces[i + 2] = indices[i + 2];
  }

  return result;
}

template <typename F>
IsosurfaceMesh extract_surface(F &func, float isovalue, bool flip = false) {
  occ::timing::StopWatch sw;
  auto cubes = func.cubes_per_side();
  occ::log::info("Marching cubes voxels: {}x{}x{}", cubes(0), cubes(1),
                 cubes(2));
  auto mc = occ::geometry::mc::MarchingCubes(cubes(0), cubes(1), cubes(2));
  mc.set_origin_and_side_lengths(func.origin(), func.side_length());
  mc.isovalue = isovalue;
  mc.flip_normals = (isovalue < 0.0) || flip;
  if (mc.flip_normals)
    occ::log::info("Negative isovalue provided, will flip normals");

  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<float> curvature;
  std::vector<uint32_t> faces;
  sw.start();
  mc.extract_with_curvature(func, vertices, faces, normals, curvature);
  sw.stop();
  occ::log::debug("Required {} function calls ", func.num_calls());
  occ::log::info("Surface extraction took {:.5f} s", sw.read());

  occ::log::info("Surface has {} vertices, {} faces", vertices.size() / 3,
                 faces.size() / 3);
  if (vertices.size() < 3) {
    throw std::runtime_error(
        "Invalid isosurface encountered, not enough vertices?");
  }
  return as_mesh(func, vertices, faces, normals, curvature);
}

VertexProperties
compute_atom_surface_properties(const Molecule &m1, const Molecule &m2,
                                Eigen::Ref<const Eigen::Matrix3Xf> vertices) {
  const size_t N = vertices.cols();
  constexpr size_t num_results = 6;
  FVec di_norm(N), dnorm(N);
  VertexProperties properties;
  int nthreads = occ::parallel::get_num_threads();

  if (m1.size() > 0) {
    Eigen::Matrix3Xf inside = m1.positions().cast<float>();
    Eigen::VectorXf vdw_inside = m1.vdw_radii().cast<float>();

    occ::core::KDTree<float> interior_tree(inside.rows(), inside,
                                           occ::core::max_leaf);
    interior_tree.index->buildIndex();

    FVec di(N);
    IVec di_idx(N), di_norm_idx(N);

    auto fill_interior_properties = [&](int thread_id) {
      std::vector<size_t> indices(num_results);
      std::vector<float> dist_sq(num_results);
      std::vector<float> dist_norm(num_results);

      for (int i = 0; i < vertices.cols(); i++) {
        if (i % nthreads != thread_id)
          continue;

        Eigen::Vector3f v = vertices.col(i);
        float dist_inside_norm = std::numeric_limits<float>::max();
        nanoflann::KNNResultSet<float> results(num_results);
        results.init(&indices[0], &dist_sq[0]);
        bool populated = interior_tree.index->findNeighbors(
            results, v.data(), nanoflann::SearchParams());
        if (!populated)
          continue;
        di(i) = std::sqrt(dist_sq[0]);
        di_idx(i) = indices[0];

        size_t inside_idx = 0;
        for (int idx = 0; idx < results.size(); idx++) {

          float vdw = vdw_inside(indices[idx]);
          float dnorm = (std::sqrt(dist_sq[idx]) - vdw) / vdw;

          if (dnorm < dist_inside_norm) {
            inside_idx = indices[idx];
            dist_inside_norm = dnorm;
          }
        }
        di_norm(i) = dist_inside_norm;
        di_norm_idx(i) = inside_idx;
      }
    };

    occ::timing::start(occ::timing::category::isosurface_properties);
    occ::parallel::parallel_do(fill_interior_properties);
    occ::timing::stop(occ::timing::category::isosurface_properties);

    properties.add_property("di", di);
    properties.add_property("di_idx", di_idx);
    properties.add_property("di_norm", di_norm);
    properties.add_property("di_norm_idx", di_norm_idx);
  }

  if (m2.size() > 0) {
    FVec de(N), de_norm(N);
    IVec de_idx(N), de_norm_idx(N);
    Eigen::Matrix3Xf outside = m2.positions().cast<float>();
    Eigen::VectorXf vdw_outside = m2.vdw_radii().cast<float>();
    occ::core::KDTree<float> exterior_tree(outside.rows(), outside,
                                           occ::core::max_leaf);
    exterior_tree.index->buildIndex();
    auto fill_exterior_properties = [&](int thread_id) {
      std::vector<size_t> indices(num_results);
      std::vector<float> dist_sq(num_results);
      std::vector<float> dist_norm(num_results);

      for (int i = 0; i < vertices.cols(); i++) {
        if (i % nthreads != thread_id)
          continue;

        Eigen::Vector3f v = vertices.col(i);
        float dist_outside_norm = std::numeric_limits<float>::max();
        nanoflann::KNNResultSet<float> results(num_results);
        results.init(&indices[0], &dist_sq[0]);
        bool populated = exterior_tree.index->findNeighbors(
            results, v.data(), nanoflann::SearchParams());
        if (!populated)
          continue;
        de(i) = std::sqrt(dist_sq[0]);
        de_idx(i) = indices[0];

        size_t outside_idx = 0;
        for (int idx = 0; idx < results.size(); idx++) {

          float vdw = vdw_outside(indices[idx]);
          float dnorm = (std::sqrt(dist_sq[idx]) - vdw) / vdw;

          if (dnorm < dist_outside_norm) {
            outside_idx = indices[idx];
            dist_outside_norm = dnorm;
          }
        }
        de_norm(i) = dist_outside_norm;
        de_norm_idx(i) = outside_idx;

        if (m1.size() > 0) {
          dnorm(i) = de_norm(i) + di_norm(i);
        }
      }
    };
    occ::timing::start(occ::timing::category::isosurface_properties);
    occ::parallel::parallel_do(fill_exterior_properties);
    occ::timing::stop(occ::timing::category::isosurface_properties);

    properties.add_property("de_idx", de_idx);
    properties.add_property("de", de);
    properties.add_property("de_norm", de_norm);
    properties.add_property("de_norm_idx", de_norm_idx);
    if (m1.size() > 0) {
      properties.add_property("dnorm", dnorm);
    }
  }

  return properties;
}

namespace occ::main {

IsosurfaceConfig::Surface IsosurfaceConfig::surface_type() const {
  std::vector<IsosurfaceConfig::Surface> surfaces{
      IsosurfaceConfig::Surface::PromoleculeDensity,
      IsosurfaceConfig::Surface::Hirshfeld,
      IsosurfaceConfig::Surface::EEQ_ESP,
      IsosurfaceConfig::Surface::ElectronDensity,
      IsosurfaceConfig::Surface::ESP,
      IsosurfaceConfig::Surface::SpinDensity,
      IsosurfaceConfig::Surface::DeformationDensity,
      IsosurfaceConfig::Surface::Orbital,
      IsosurfaceConfig::Surface::CrystalVoid};

  ankerl::unordered_dense::map<std::string, IsosurfaceConfig::Surface>
      name2surface{
          // ESP
          {"electric_potential", IsosurfaceConfig::Surface::ESP},
          {"electric potential", IsosurfaceConfig::Surface::ESP},
          // Electron density
          {"electron_density", IsosurfaceConfig::Surface::ElectronDensity},
          {"electron density", IsosurfaceConfig::Surface::ElectronDensity},
          {"rho", IsosurfaceConfig::Surface::ElectronDensity},
          {"density", IsosurfaceConfig::Surface::ElectronDensity},
          // Molecular orbitals
          {"orbital", IsosurfaceConfig::Surface::Orbital},
          {"mo", IsosurfaceConfig::Surface::Orbital},
          // Promolecule
          {"promol", IsosurfaceConfig::Surface::PromoleculeDensity},
          {"pro", IsosurfaceConfig::Surface::PromoleculeDensity},
          // Hirshfeld
          {"stockholder weight", IsosurfaceConfig::Surface::Hirshfeld},
          {"hs", IsosurfaceConfig::Surface::Hirshfeld},
          {"stockholder_weight", IsosurfaceConfig::Surface::Hirshfeld},
          // void
          {"crystal_void", IsosurfaceConfig::Surface::CrystalVoid},
      };
  for (const auto &s : surfaces) {
    name2surface.insert({to_string(s), s});
  }

  auto s = occ::util::to_lower_copy(kind);
  auto loc = name2surface.find(s);
  if (loc != name2surface.end()) {
    return loc->second;
  }
  throw std::runtime_error(fmt::format("Unknown surface type: {}", kind));
}

std::vector<IsosurfaceConfig::Property>
IsosurfaceConfig::surface_properties() const {
  std::vector<IsosurfaceConfig::Property> properties{
      IsosurfaceConfig::Property::Dnorm,
      IsosurfaceConfig::Property::Dint_norm,
      IsosurfaceConfig::Property::Dext_norm,
      IsosurfaceConfig::Property::Dint,
      IsosurfaceConfig::Property::Dext,
      IsosurfaceConfig::Property::FragmentPatch,
      IsosurfaceConfig::Property::ShapeIndex,
      IsosurfaceConfig::Property::Curvedness,
      IsosurfaceConfig::Property::EEQ_ESP,
      IsosurfaceConfig::Property::PromoleculeDensity,
      IsosurfaceConfig::Property::ESP,
      IsosurfaceConfig::Property::ElectronDensity,
      IsosurfaceConfig::Property::SpinDensity,
      IsosurfaceConfig::Property::DeformationDensity,
      IsosurfaceConfig::Property::Orbital};

  ankerl::unordered_dense::set<IsosurfaceConfig::Property> result{
      IsosurfaceConfig::Property::Dint,
      IsosurfaceConfig::Property::Dint_norm,
      IsosurfaceConfig::Property::ShapeIndex,
      IsosurfaceConfig::Property::Curvedness,
      IsosurfaceConfig::Property::EEQ_ESP,
  };

  if (have_environment_file()) {
    result.insert(IsosurfaceConfig::Property::Dext);
    result.insert(IsosurfaceConfig::Property::Dext_norm);
    result.insert(IsosurfaceConfig::Property::Dnorm);
    result.insert(IsosurfaceConfig::Property::FragmentPatch);
  }

  ankerl::unordered_dense::map<std::string, IsosurfaceConfig::Property>
      name2prop{
          // ESP
          {"electric_potential", IsosurfaceConfig::Property::ESP},
          {"electric potential", IsosurfaceConfig::Property::ESP},
          // Electron density
          {"electron density", IsosurfaceConfig::Property::ElectronDensity},
          {"rho", IsosurfaceConfig::Property::ElectronDensity},
          {"orbital", IsosurfaceConfig::Property::Orbital},
          {"density", IsosurfaceConfig::Property::ElectronDensity},
          {"eeq_esp", IsosurfaceConfig::Property::EEQ_ESP},
          {"esp", IsosurfaceConfig::Property::ESP},
      };

  for (const auto &p : properties) {
    name2prop.insert({to_string(p), p});
  }

  for (const auto &p : additional_properties) {
    auto s = occ::util::to_lower_copy(p);
    auto loc = name2prop.find(s);
    if (loc != name2prop.end()) {
      result.insert(loc->second);
    } else {
      occ::log::warn("Unknown property: {}, ignoring", p);
    }
  }
  return std::vector<IsosurfaceConfig::Property>(result.begin(), result.end());
}

bool IsosurfaceConfig::have_environment_file() const {
  return !environment_filename.empty();
}

bool IsosurfaceConfig::requires_crystal() const {
  return surface_type() == IsosurfaceConfig::Surface::CrystalVoid;
}

bool IsosurfaceConfig::requires_environment() const {
  auto s = surface_type();
  switch (s) {
  case IsosurfaceConfig::Surface::Hirshfeld:
    return true;
  default:
    break;
  }

  for (const auto &prop : surface_properties()) {
    switch (prop) {
    case IsosurfaceConfig::Property::Dext:
      return true;
    case IsosurfaceConfig::Property::Dext_norm:
      return true;
    case IsosurfaceConfig::Property::Dnorm:
      return true;
    case IsosurfaceConfig::Property::FragmentPatch:
      return true;
    default:
      break;
    }
  }
  return false;
}

bool IsosurfaceConfig::requires_wavefunction() const {
  auto s = surface_type();
  switch (s) {
  case IsosurfaceConfig::Surface::ESP:
    return true;
  case IsosurfaceConfig::Surface::ElectronDensity:
    return true;
  case IsosurfaceConfig::Surface::DeformationDensity:
    return true;
  case IsosurfaceConfig::Surface::Orbital:
    return true;
  case IsosurfaceConfig::Surface::SpinDensity:
    return true;
  default:
    break;
  }

  for (const auto &prop : surface_properties()) {
    switch (prop) {
    case IsosurfaceConfig::Property::ESP:
      return true;
    case IsosurfaceConfig::Property::ElectronDensity:
      return true;
    case IsosurfaceConfig::Property::DeformationDensity:
      return true;
    case IsosurfaceConfig::Property::Orbital:
      return true;
    case IsosurfaceConfig::Property::SpinDensity:
      return true;
    default:
      break;
    }
  }
  return false;
}

bool IsosurfaceConfig::has_multiple_outputs() const {
  return isovalues.size() > 1 ||
         ((surface_type() == IsosurfaceConfig::Surface::Orbital) &&
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

FVec compute_surface_property(IsosurfaceConfig::Property prop,
                              Eigen::Ref<const Eigen::Matrix3Xf> vertices,
                              const Molecule &m1, const Molecule &m2,
                              const IsosurfaceConfig &config,
                              const Wavefunction &wfn = {}) {

  FVec result = FVec::Zero(vertices.cols());

  switch (prop) {
  case IsosurfaceConfig::Property::PromoleculeDensity: {
    auto func = iso::BatchFunctor<slater::PromoleculeDensity>(m1);
    func.batch(vertices * occ::units::ANGSTROM_TO_BOHR, result);
    occ::log::info("Min {} Max {} Mean {}", result.minCoeff(),
                   result.maxCoeff(), result.mean());
    occ::log::info("Computed Promoecule Density for {} vertices",
                   func.num_calls());
    break;
  }
  case IsosurfaceConfig::Property::ElectronDensity: {
    auto func = iso::ElectronDensityFunctor(wfn);
    func.batch(vertices * occ::units::ANGSTROM_TO_BOHR, result);
    occ::log::info("Min {} Max {} Mean {}", result.minCoeff(),
                   result.maxCoeff(), result.mean());
    occ::log::info("Computed Electron Density for {} vertices",
                   func.num_calls());
    break;
  }
  case IsosurfaceConfig::Property::Orbital: {
    // Handle orbital property
    if (config.orbital_indices.empty()) {
      throw std::runtime_error(
          "No orbital indices specified for orbital property");
    }

    auto func = iso::ElectronDensityFunctor(wfn, -1);
    int prev_calls = 0;
    for (const auto &orbital_index : config.orbital_indices) {
      func.set_orbital_index(
          orbital_index.resolve(wfn.mo.n_alpha, wfn.mo.n_beta));
      func.batch(vertices * occ::units::ANGSTROM_TO_BOHR, result);
      occ::log::info("Computed Orbital {} Density for {} vertices",
                     orbital_index.format(), func.num_calls() - prev_calls);
      occ::log::info("Min {} Max {} Mean {}", result.minCoeff(),
                     result.maxCoeff(), result.mean());
      prev_calls = func.num_calls();
    }
    break;
  }
  case IsosurfaceConfig::Property::ESP: {
    auto func = iso::ElectricPotentialFunctor(wfn);
    func.batch(vertices * occ::units::ANGSTROM_TO_BOHR, result);
    occ::log::info("Min {} Max {} Mean {}", result.minCoeff(),
                   result.maxCoeff(), result.mean());
    occ::log::info("Computed ESP (QM) for {} vertices", func.num_calls());
    break;
  }
  case IsosurfaceConfig::Property::EEQ_ESP: {
    auto m = m1;
    auto q = occ::core::charges::eeq_partial_charges(m.atomic_numbers(),
                                                     m.positions(), m.charge());
    occ::log::info("Molecule partial charges (EEQ)");
    for (int i = 0; i < q.rows(); i++) {
      occ::log::info("Atom {}: {:12.5f}", i, q(i));
    }
    m.set_partial_charges(q);
    auto func = iso::ElectricPotentialFunctorPC(m);
    func.batch(vertices, result);
    occ::log::info("Min {} Max {} Mean {}", result.minCoeff(),
                   result.maxCoeff(), result.mean());
    occ::log::info("Computed EEQ ESP for {} vertices", func.num_calls());
    break;
  }
  default:
    break;
  }

  return result;
}

void run_isosurface_subcommand(IsosurfaceConfig config) {
  IsosurfaceMesh mesh;
  VertexProperties properties;
  bool use_wfn_mol = false;
  config.separation *= occ::units::ANGSTROM_TO_BOHR;

  const auto properties_to_compute = config.surface_properties();
  if (properties_to_compute.size() > 0) {
    occ::log::info("Properties to compute:");
    for (const auto &prop : properties_to_compute) {
      occ::log::info("{}", to_string(prop));
    }
  }

  if (Wavefunction::is_likely_wavefunction_filename(config.geometry_filename)) {
    config.wavefunction_filename = config.geometry_filename;
    use_wfn_mol = true;
  }

  Wavefunction wfn = load_wfn(config);
  bool have_wfn = wfn.atoms.size() > 0;

  Molecule m1, m2;

  if (use_wfn_mol) {
    m1 = Molecule(wfn.atoms);
  }

  bool have_crystal =
      occ::io::CifParser::is_likely_cif_filename(config.geometry_filename);
  ensure_isosurface_configuration_valid(config, have_wfn, have_crystal);

  if (!config.environment_filename.empty()) {
    m2 = occ::io::molecule_from_xyz_file(config.environment_filename);
  }

  const auto surface_type = config.surface_type();
  occ::log::info("Isosurface kind: {}", to_string(surface_type));

  // Ensure we have at least one isovalue
  if (config.isovalues.empty()) {
    config.isovalues.push_back(0.02);
  }

  // In run_isosurface_subcommand, where we resolve the indices:

  if (surface_type == IsosurfaceConfig::Surface::Orbital) {
    parse_orbital_descriptions(config.orbital_indices, config.orbitals_input);
  }

  auto parameter_combinations =
      occ::main::generate_parameter_combinations(config);

  // Loop over all isovalues
  for (const auto &params : parameter_combinations) {

    const double isovalue = params.isovalue;
    if (params.orbital_index) {
      occ::log::info("Processing surface isovalue = {} (orbital = {})",
                     isovalue, (*params.orbital_index).format());
    } else {
      occ::log::info("Processing surface isovalue = {}", isovalue);
    }

    IsosurfaceMesh mesh;
    VertexProperties properties;

    switch (config.surface_type()) {
    case IsosurfaceConfig::Surface::ESP: {
      auto func = iso::MCElectricPotentialFunctor(wfn, config.separation);
      mesh = extract_surface(func, isovalue);
      break;
    }
    case IsosurfaceConfig::Surface::ElectronDensity: {
      auto func = iso::MCElectronDensityFunctor(wfn, config.separation);
      mesh = extract_surface(func, isovalue);
      break;
    }
    case IsosurfaceConfig::Surface::CrystalVoid: {
      occ::io::CifParser parser;
      auto crystal = parser.parse_crystal(config.geometry_filename).value();
      auto func = iso::VoidSurfaceFunctor(crystal, config.separation);
      if (m2.size() == 0) {
        m2 = func.molecule();
      }
      mesh = extract_surface(func, isovalue, true);
      break;
    }
    case IsosurfaceConfig::Surface::Hirshfeld: {
      if (!use_wfn_mol) {
        m1 = occ::io::molecule_from_xyz_file(config.geometry_filename);
      }

      occ::log::info("Interior region has {} atoms", m1.size());
      occ::log::info("Exterior region has {} atoms", m2.size());

      auto func = iso::StockholderWeightFunctor(m1, m2, config.separation);
      func.set_background_density(config.background_density);
      mesh = extract_surface(func, 0.5f); // Hirshfeld always uses 0.5
      break;
    }
    case IsosurfaceConfig::Surface::PromoleculeDensity: {
      if (!use_wfn_mol) {
        m1 = occ::io::molecule_from_xyz_file(config.geometry_filename);
      }
      occ::log::info("Interior region has {} atoms", m1.size());
      auto func = iso::MCPromoleculeDensityFunctor(m1, config.separation);
      func.set_isovalue(isovalue);
      mesh = extract_surface(func, isovalue);
      break;
    }
    case IsosurfaceConfig::Surface::DeformationDensity: {
      if (!use_wfn_mol) {
        m1 = occ::io::molecule_from_xyz_file(config.geometry_filename);
      }
      occ::log::info("Interior region has {} atoms", m1.size());
      auto func = iso::MCDeformationDensityFunctor(m1, wfn, config.separation);
      func.set_isovalue(isovalue);
      mesh = extract_surface(func, isovalue);
      break;
    }
    case IsosurfaceConfig::Surface::Orbital: {

      if (!params.orbital_index) {
        throw std::runtime_error(
            "No orbital index specified for orbital property");
      }
      auto func = iso::MCElectronDensityFunctor(
          wfn, config.separation,
          (*params.orbital_index).resolve(wfn.mo.n_alpha, wfn.mo.n_beta));
      mesh = extract_surface(func, isovalue);
      break;
    }

    default: {
      throw std::runtime_error("Not implemented");
      break;
    }
    }

    Eigen::Map<const FMat3N> verts(mesh.vertices.data(), 3,
                                   mesh.vertices.size() / 3);
    Eigen::Map<const FMat3N> normals(mesh.normals.data(), 3,
                                     mesh.normals.size() / 3);
    Eigen::Map<const Eigen::Matrix<uint32_t, 3, Eigen::Dynamic>> faces(
        mesh.faces.data(), 3, mesh.faces.size() / 3);

    occ::log::info("Computing surface curvature properties");
    auto c = occ::isosurface::calculate_curvature(mesh.mean_curvature,
                                                  mesh.gaussian_curvature);

    occ::log::info("Computing atom internal/external neighbor properties");
    properties = compute_atom_surface_properties(m1, m2, verts);

    properties.add_property("shape_index", c.shape_index);
    properties.add_property("curvedness", c.curvedness);
    properties.add_property("gaussian_curvature", c.gaussian);
    properties.add_property("mean_curvature", c.mean);
    properties.add_property("k1", c.k1);
    properties.add_property("k2", c.k2);

    for (const auto &prop : properties_to_compute) {
      const auto s = to_string(prop);
      if (properties.fprops.contains(s))
        continue;
      if (properties.iprops.contains(s))
        continue;
      occ::log::info("Need to compute: {}", s);
      properties.add_property(
          s, compute_surface_property(prop, verts, m1, m2, config, wfn));
    }

    Eigen::Vector3f lower_left = verts.rowwise().minCoeff();
    Eigen::Vector3f upper_right = verts.rowwise().maxCoeff();
    occ::log::info("Lower corner of mesh: [{:.3f} {:.3f} {:.3f}]",
                   lower_left(0), lower_left(1), lower_left(2));
    occ::log::info("Upper corner of mesh: [{:.3f} {:.3f} {:.3f}]",
                   upper_right(0), upper_right(1), upper_right(2));

    // Generate output filename based on index
    std::string output_filename = format_output_filename(params, config);
    occ::log::info("Writing surface to {}", output_filename);
    occ::io::write_ply_mesh(output_filename, mesh, properties,
                            config.binary_output);
  }
}

} // namespace occ::main
