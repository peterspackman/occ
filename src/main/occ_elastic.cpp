#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <fmt/os.h>
#include <occ/core/elastic_tensor.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/geometry/icosphere_mesh.h>
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

template <typename F> void print_averages(const std::string &name, F f) {
  occ::log::info("{:-<40s}", name + " ");
  for (const auto &[k, v] :
       {std::pair{"Voigt", ElasticTensor::AveragingScheme::Voigt},
        {"Reuss", ElasticTensor::AveragingScheme::Reuss},
        {"Hill", ElasticTensor::AveragingScheme::Hill}}) {
    occ::log::info("{:<10s} {:12.6f}", k, f(v));
  }
}

void print_averaged_properties(const ElasticTensor &tensor) {
  print_averages("Young's modulus", [&](auto scheme) {
    return tensor.average_youngs_modulus(scheme);
  });

  print_averages("Bulk modulus", [&](auto scheme) {
    return tensor.average_bulk_modulus(scheme);
  });

  print_averages("Shear modulus", [&](auto scheme) {
    return tensor.average_shear_modulus(scheme);
  });

  print_averages("Poisson's ratio", [&](auto scheme) {
    return tensor.average_poisson_ratio(scheme);
  });
}

inline occ::isosurface::IsosurfaceProperties
compute_mesh_properties(const IcosphereMesh &icosphere,
                        const ElasticTensor &tensor) {

  occ::isosurface::IsosurfaceProperties result;
  const auto &verts = icosphere.vertices();
  {
    occ::FVec ym(verts.cols());
    for (int i = 0; i < verts.cols(); i++) {
      ym(i) = static_cast<float>(tensor.youngs_modulus(verts.col(i)));
    }
    result.add("youngs_modulus", ym);
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
    for (int i = 0; i < verts.cols(); i++) {
      auto [l, u] = tensor.poisson_ratio_minmax(verts.col(i));
      p_min(i) = static_cast<float>(l);
      p_max(i) = static_cast<float>(u);
    }
    result.add("poissons_ratio_min", p_min);
    result.add("poissons_ratio_max", p_max);
  }
  return result;
}

inline void write_meshes(const ElasticTensor &tensor, int subdivisions,
                         const std::string &basename) {

  IcosphereMesh icosphere(subdivisions);

  auto props = compute_mesh_properties(icosphere, tensor);

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
    occ::log::info("Writing {} surface to {}", name, filename);
    occ::io::write_ply_mesh(filename, mesh, true);
  }
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
  occ::log::info("{:-<40s}\n{}", "Voigt C matrix (GPa) ",
                 format_matrix(tensor.voigt_c()));
  occ::log::info("{:-<40s}\n{}", "Voigt S matrix (GPa^-1) ",
                 format_matrix(tensor.voigt_s()));

  Eigen::SelfAdjointEigenSolver<occ::Mat6> solv(tensor.voigt_c());
  occ::Vec6 e = solv.eigenvalues();

  occ::log::info("{:-<40s}", "Eigenvalues of Voigt C (GPa) ");
  occ::log::info("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f} {:12.5f}", e(0),
                 e(1), e(2), e(3), e(4), e(5));

  print_averaged_properties(tensor);
  write_meshes(tensor, settings.subdivisions, "elastic");
}

} // namespace occ::main
