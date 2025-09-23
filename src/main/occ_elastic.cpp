#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <fmt/os.h>
#include <occ/core/elastic_tensor.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/surface.h>
#include <occ/geometry/icosphere_mesh.h>
#include <occ/io/cifparser.h>
#include <occ/io/core_json.h>
#include <occ/io/crystal_json.h>
#include <occ/io/ply.h>
#include <occ/isosurface/isosurface.h>
#include <occ/main/occ_elastic.h>

using occ::core::ElasticTensor;
using occ::crystal::Crystal;
using occ::geometry::IcosphereMesh;
using occ::isosurface::Isosurface;

enum class ElasticProperty {
  YoungsModulus,
  LinearCompressibility,
  ShearModulus,
  PoissonRatio
};

inline ElasticTensor read_tensor(const std::string &filename) {
  occ::Mat6 tensor;
  std::ifstream file(filename);
  if (file.is_open()) {
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        file >> tensor(i, j);
      }
    }
    file.close();
  } else
    throw std::runtime_error("Unable to open elastic tensor file");

  return ElasticTensor(tensor);
}

template <typename F>
void print_averages(const std::string &name, const std::string &units, F f) {
  occ::log::info("");
  occ::log::info("{} ({})", name, units);
  occ::log::info("{:<10s} {:>12s}", "Scheme", "Value");
  occ::log::info("{:-<24s}", "");
  for (const auto &[k, v] :
       {std::pair{"Voigt", ElasticTensor::AveragingScheme::Voigt},
        {"Reuss", ElasticTensor::AveragingScheme::Reuss},
        {"Hill", ElasticTensor::AveragingScheme::Hill}}) {
    occ::log::info("{:<10s} {:12.3f}", k, f(v));
  }
}

void print_averaged_properties(const ElasticTensor &tensor,
                               const Crystal *crystal = nullptr) {
  print_averages("Young's modulus", "GPa", [&](auto scheme) {
    return tensor.average_youngs_modulus(scheme);
  });

  print_averages("Bulk modulus", "GPa", [&](auto scheme) {
    return tensor.average_bulk_modulus(scheme);
  });

  print_averages("Shear modulus", "GPa", [&](auto scheme) {
    return tensor.average_shear_modulus(scheme);
  });

  print_averages("Poisson's ratio", "-", [&](auto scheme) {
    return tensor.average_poisson_ratio(scheme);
  });

  // Print acoustic velocities if crystal density is available
  if (crystal) {
    double density = crystal->density();
    occ::log::info("");
    occ::log::info("Acoustic velocities (density: {:.3f} g/cm³)", density);
    occ::log::info("{:<10s} {:>10s} {:>10s}", "Scheme", "V_s (m/s)",
                   "V_p (m/s)");
    occ::log::info("{:-<32s}", "");

    for (const auto &[k, v] :
         {std::pair{"Voigt", ElasticTensor::AveragingScheme::Voigt},
          {"Reuss", ElasticTensor::AveragingScheme::Reuss},
          {"Hill", ElasticTensor::AveragingScheme::Hill}}) {
      double K = tensor.average_bulk_modulus(v);
      double G = tensor.average_shear_modulus(v);
      double v_s = tensor.transverse_acoustic_velocity(K, G, density);
      double v_p = tensor.longitudinal_acoustic_velocity(K, G, density);
      occ::log::info("{:<10s} {:>10.0f} {:>10.0f}", k, v_s, v_p);
    }
  }
}

inline occ::isosurface::IsosurfaceProperties
compute_mesh_properties(const IcosphereMesh &icosphere,
                        const ElasticTensor &tensor) {

  occ::isosurface::IsosurfaceProperties result;
  const auto &verts = icosphere.vertices();
  {
    occ::FVec ym(verts.cols());
    occ::FVec ym_reduced(verts.cols());
    for (int i = 0; i < verts.cols(); i++) {
      ym(i) = static_cast<float>(tensor.youngs_modulus(verts.col(i)));
      ym_reduced(i) =
          static_cast<float>(tensor.reduced_youngs_modulus(verts.col(i)));
    }
    result.add("youngs_modulus", ym);
    result.add("youngs_modulus_reduced", ym_reduced);
  }

  {
    occ::FVec lc(verts.cols());
    for (int i = 0; i < verts.cols(); i++) {
      lc(i) = static_cast<float>(tensor.linear_compressibility(verts.col(i)));
    }
    result.add("linear_compressibility", lc);
  }

  {
    occ::FVec shear_min(verts.cols());
    occ::FVec shear_max(verts.cols());
    for (int i = 0; i < verts.cols(); i++) {
      auto [l, u] = tensor.shear_modulus_minmax(verts.col(i));
      shear_min(i) = static_cast<float>(l);
      shear_max(i) = static_cast<float>(u);
    }
    result.add("shear_modulus_min", shear_min);
    result.add("shear_modulus_max", shear_max);
  }

  {
    occ::FVec p_min(verts.cols());
    occ::FVec p_max(verts.cols());
    occ::FVec p_iso(verts.cols());
    for (int i = 0; i < verts.cols(); i++) {
      auto [l, u] = tensor.poisson_ratio_minmax(verts.col(i));
      p_min(i) = static_cast<float>(l);
      p_max(i) = static_cast<float>(u);
      p_iso(i) = static_cast<float>(
          tensor.average_poisson_ratio_direction(verts.col(i)));
    }
    result.add("poissons_ratio_min", p_min);
    result.add("poissons_ratio_max", p_max);
    result.add("poissons_ratio_iso", p_iso);
  }
  return result;
}

inline void write_meshes(const ElasticTensor &tensor, int subdivisions,
                         const std::string &basename) {

  IcosphereMesh icosphere(subdivisions);

  auto props = compute_mesh_properties(icosphere, tensor);

  occ::log::info("");
  occ::log::info("Writing surface meshes:");
  for (const auto &[name, vals] : props.properties) {
    std::string filename = fmt::format("{}_{}.ply", basename, name);
    occ::Mat3N v = icosphere.vertices();
    const auto &fvals = std::get<occ::FVec>(vals);
    for (int i = 0; i < v.cols(); i++) {
      v.col(i) *= static_cast<double>(fvals(i));
    }
    Isosurface mesh;
    mesh.vertices = v.cast<float>();
    mesh.faces = icosphere.faces();
    mesh.normals = icosphere.vertices().cast<float>();
    occ::log::info("  {} -> {}", name, filename);
    occ::io::write_ply_mesh(filename, mesh, true);
  }
  occ::log::info("");
}

namespace occ::main {

CLI::App *add_elastic_subcommand(CLI::App &app) {

  CLI::App *elastic = app.add_subcommand(
      "elastic", "compute elastic tensor properties or meshes");
  auto config = std::make_shared<ElasticSettings>();

  elastic
      ->add_option("--tensor", config->tensor_filename,
                   "input tensor filename (txt)")
      ->required();
  elastic->add_option("--crystal", config->crystal_filename,
                      "input crystal structure (CIF) for face analysis");
  elastic->add_option("--max-surfaces", config->max_surfaces,
                      "maximum number of crystal surfaces to analyze");
  elastic->add_option("--subdivisions", config->subdivisions,
                      "icosphere mesh subdivisions (resolution) for surfaces");
  elastic->add_option("--json", config->output_json_filename,
                      "JSON filename for output");
  elastic->add_option("--scale", config->scale, "maximum extent for property");
  elastic->fallthrough();
  elastic->callback([config]() { run_elastic_subcommand(*config); });
  return elastic;
}

void run_elastic_subcommand(const ElasticSettings &settings) {
  ElasticTensor tensor = read_tensor(settings.tensor_filename);
  occ::log::info("Loaded tensor from {}", settings.tensor_filename);
  occ::log::info("");
  occ::log::info("Voigt C matrix (GPa)");
  occ::log::info("{}", format_matrix(tensor.voigt_c()));
  occ::log::info("");
  occ::log::info("Voigt S matrix (GPa^-1)");
  occ::log::info("{}", format_matrix(tensor.voigt_s()));

  occ::Vec6 e = tensor.eigenvalues();

  occ::log::info("");
  occ::log::info("Eigenvalues of Voigt C (GPa)");
  occ::log::info("{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f}", e(0), e(1),
                 e(2), e(3), e(4), e(5));

  // Crystal loading and acoustic velocity calculation
  Crystal *crystal_ptr = nullptr;
  std::unique_ptr<Crystal> crystal_storage;

  if (!settings.crystal_filename.empty()) {
    occ::io::CifParser parser;
    auto crystal_result =
        parser.parse_crystal_from_file(settings.crystal_filename);
    if (crystal_result.has_value()) {
      crystal_storage = std::make_unique<Crystal>(crystal_result.value());
      crystal_ptr = crystal_storage.get();
      occ::log::info("Loaded crystal structure from {}",
                     settings.crystal_filename);
    } else {
      occ::log::error("Failed to load crystal structure from {}",
                      settings.crystal_filename);
    }
  }

  print_averaged_properties(tensor, crystal_ptr);
  write_meshes(tensor, settings.subdivisions, "elastic");

  // Crystal face analysis if crystal was loaded
  if (crystal_ptr) {
    // Compute properties along crystal faces
    compute_crystal_face_properties(tensor, *crystal_ptr, settings);
  }
}

void compute_crystal_face_properties(const ElasticTensor &tensor,
                                     const Crystal &crystal,
                                     const ElasticSettings &settings) {
  occ::log::info("");
  occ::log::info("Crystal Face Properties");
  occ::log::info("");

  // Generate crystal surfaces with parameters similar to crystal surface energy
  // calculation
  occ::crystal::CrystalSurfaceGenerationParameters params;
  params.d_min = 0.1;
  params.unique = true;
  auto surfaces = occ::crystal::generate_surfaces(crystal, params);

  if (surfaces.empty()) {
    occ::log::warn("No crystal surfaces generated");
    return;
  }

  // Follow the same logic as occ cg - just take the first N surfaces as
  // generated
  int num_surfaces =
      std::min(static_cast<int>(surfaces.size()), settings.max_surfaces);

  occ::log::info("Analyzing {} of {} crystallographic surfaces", num_surfaces,
                 surfaces.size());
  occ::log::info("");

  // Simplified table header
  occ::log::info("{:<17} {:>20} {:>8} {:>8} {:>8} {:>8}", "Surface", "Normal",
                 "E", "v_avg", "E_R", "d");
  occ::log::info("{:<17} {:>20} {:>8} {:>8} {:>8} {:>8}", "", "", "(GPa)", "",
                 "(GPa)", "(Å)");
  occ::log::info("{:-<73}", "");

  for (int i = 0; i < num_surfaces; ++i) {
    const auto &surface = surfaces[i];
    const auto &hkl = surface.hkl();
    std::string surface_name = fmt::format("({} {} {})", hkl.h, hkl.k, hkl.l);

    // Get the surface normal vector
    occ::Vec3 normal = surface.normal_vector();
    std::string normal_str = fmt::format("[{: .2f} {: .2f} {: .2f}]",
                                         normal.x(), normal.y(), normal.z());

    // Compute elastic properties along this surface normal
    double E = tensor.youngs_modulus(normal);
    double v_avg = tensor.average_poisson_ratio_direction(normal);
    double E_red = tensor.reduced_youngs_modulus(normal);
    double reciprocal_d = surface.d();     // This is in Å⁻¹
    double d_spacing = 1.0 / reciprocal_d; // Convert to Å

    occ::log::info("{:<17} {:>20} {:>8.1f} {:>8.3f} {:>8.1f} {:>8.1f}",
                   surface_name, normal_str, E, v_avg, E_red, d_spacing);
  }
}

} // namespace occ::main
