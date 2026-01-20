#include "isosurface_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/molecule.h>
#include <occ/io/cube.h>
#include <occ/io/isosurface_json.h>
#include <occ/isosurface/isosurface.h>
#include <occ/isosurface/surface_types.h>
#include <occ/qm/wavefunction.h>

using namespace emscripten;
using namespace occ;

void register_isosurface_bindings() {
  // Register SurfaceKind enum
  enum_<isosurface::SurfaceKind>("SurfaceKind")
      .value("PromoleculeDensity", isosurface::SurfaceKind::PromoleculeDensity)
      .value("Hirshfeld", isosurface::SurfaceKind::Hirshfeld)
      .value("EEQ_ESP", isosurface::SurfaceKind::EEQ_ESP)
      .value("ElectronDensity", isosurface::SurfaceKind::ElectronDensity)
      .value("ESP", isosurface::SurfaceKind::ESP)
      .value("SpinDensity", isosurface::SurfaceKind::SpinDensity)
      .value("DeformationDensity", isosurface::SurfaceKind::DeformationDensity)
      .value("Orbital", isosurface::SurfaceKind::Orbital)
      .value("CrystalVoid", isosurface::SurfaceKind::CrystalVoid)
      .value("VolumeGrid", isosurface::SurfaceKind::VolumeGrid)
      .value("SoftVoronoi", isosurface::SurfaceKind::SoftVoronoi)
      .value("VDWLogSumExp", isosurface::SurfaceKind::VDWLogSumExp)
      .value("HSRinv", isosurface::SurfaceKind::HSRinv)
      .value("HSExp", isosurface::SurfaceKind::HSExp);

  // Register PropertyKind enum
  enum_<isosurface::PropertyKind>("PropertyKind")
      .value("Dnorm", isosurface::PropertyKind::Dnorm)
      .value("Dint_norm", isosurface::PropertyKind::Dint_norm)
      .value("Dext_norm", isosurface::PropertyKind::Dext_norm)
      .value("Dint", isosurface::PropertyKind::Dint)
      .value("Dext", isosurface::PropertyKind::Dext)
      .value("FragmentPatch", isosurface::PropertyKind::FragmentPatch)
      .value("ShapeIndex", isosurface::PropertyKind::ShapeIndex)
      .value("Curvedness", isosurface::PropertyKind::Curvedness)
      .value("EEQ_ESP", isosurface::PropertyKind::EEQ_ESP)
      .value("PromoleculeDensity", isosurface::PropertyKind::PromoleculeDensity)
      .value("ESP", isosurface::PropertyKind::ESP)
      .value("ElectronDensity", isosurface::PropertyKind::ElectronDensity)
      .value("SpinDensity", isosurface::PropertyKind::SpinDensity)
      .value("DeformationDensity", isosurface::PropertyKind::DeformationDensity)
      .value("Orbital", isosurface::PropertyKind::Orbital)
      .value("GaussianCurvature", isosurface::PropertyKind::GaussianCurvature)
      .value("MeanCurvature", isosurface::PropertyKind::MeanCurvature)
      .value("CurvatureK1", isosurface::PropertyKind::CurvatureK1)
      .value("CurvatureK2", isosurface::PropertyKind::CurvatureK2);

  // Register IsosurfaceProperties class
  class_<isosurface::IsosurfaceProperties>("IsosurfaceProperties")
      .constructor<>()
      .function("hasProperty", &isosurface::IsosurfaceProperties::has_property)
      .function("count", &isosurface::IsosurfaceProperties::count)
      .function("merge", &isosurface::IsosurfaceProperties::merge)
      .function(
          "addFloat",
          optional_override([](isosurface::IsosurfaceProperties &self,
                               const std::string &name, const val &jsArray) {
            const int length = jsArray["length"].as<int>();
            FVec vec(length);
            for (int i = 0; i < length; ++i) {
              vec(i) = jsArray[i].as<float>();
            }
            self.add<FVec>(name, vec);
          }))
      .function("addInt", optional_override(
                              [](isosurface::IsosurfaceProperties &self,
                                 const std::string &name, const val &jsArray) {
                                const int length = jsArray["length"].as<int>();
                                IVec vec(length);
                                for (int i = 0; i < length; ++i) {
                                  vec(i) = jsArray[i].as<int>();
                                }
                                self.add<IVec>(name, vec);
                              }))
      .function(
          "getFloat",
          optional_override([](const isosurface::IsosurfaceProperties &self,
                               const std::string &name) -> val {
            if (auto ptr = self.get<FVec>(name)) {
              const auto &vec = *ptr;
              val result = val::global("Float32Array").new_(vec.size());
              for (int i = 0; i < vec.size(); ++i) {
                result.set(i, vec(i));
              }
              return result;
            }
            return val::null();
          }))
      .function(
          "getInt",
          optional_override([](const isosurface::IsosurfaceProperties &self,
                               const std::string &name) -> val {
            if (auto ptr = self.get<IVec>(name)) {
              const auto &vec = *ptr;
              val result = val::global("Int32Array").new_(vec.size());
              for (int i = 0; i < vec.size(); ++i) {
                result.set(i, vec(i));
              }
              return result;
            }
            return val::null();
          }));

  // Register IsosurfaceGenerationParameters class
  class_<isosurface::IsosurfaceGenerationParameters>(
      "IsosurfaceGenerationParameters")
      .constructor<>()
      .property("isovalue",
                &isosurface::IsosurfaceGenerationParameters::isovalue)
      .property("separation",
                &isosurface::IsosurfaceGenerationParameters::separation)
      .property("backgroundDensity",
                &isosurface::IsosurfaceGenerationParameters::background_density)
      .property("flipNormals",
                &isosurface::IsosurfaceGenerationParameters::flip_normals)
      .property("binaryOutput",
                &isosurface::IsosurfaceGenerationParameters::binary_output)
      .property("surfaceKind",
                &isosurface::IsosurfaceGenerationParameters::surface_kind)
      .property("properties",
                &isosurface::IsosurfaceGenerationParameters::properties);

  // Register Isosurface class
  class_<isosurface::Isosurface>("Isosurface")
      .constructor<>()
      .function("volume", &isosurface::Isosurface::volume)
      .function("surfaceArea", &isosurface::Isosurface::surface_area)
      .function("getVertices",
                optional_override([](const isosurface::Isosurface &surf) {
                  size_t numVertices = surf.vertices.cols();
                  val vertices =
                      val::global("Float32Array").new_(3 * numVertices);
                  for (size_t i = 0; i < numVertices; ++i) {
                    vertices.set(i * 3 + 0, surf.vertices(0, i));
                    vertices.set(i * 3 + 1, surf.vertices(1, i));
                    vertices.set(i * 3 + 2, surf.vertices(2, i));
                  }
                  return vertices;
                }))
      .function("getFaces",
                optional_override([](const isosurface::Isosurface &surf) {
                  size_t numFaces = surf.faces.cols();
                  val faces = val::global("Uint32Array").new_(3 * numFaces);
                  for (size_t i = 0; i < numFaces; ++i) {
                    faces.set(i * 3 + 0, surf.faces(0, i));
                    faces.set(i * 3 + 1, surf.faces(1, i));
                    faces.set(i * 3 + 2, surf.faces(2, i));
                  }
                  return faces;
                }))
      .function("getNormals",
                optional_override([](const isosurface::Isosurface &surf) {
                  size_t numVertices = surf.vertices.cols();
                  val normals =
                      val::global("Float32Array").new_(3 * numVertices);
                  for (size_t i = 0; i < numVertices; ++i) {
                    normals.set(i * 3 + 0, surf.normals(0, i));
                    normals.set(i * 3 + 1, surf.normals(1, i));
                    normals.set(i * 3 + 2, surf.normals(2, i));
                  }
                  return normals;
                }))
      .property("properties", &isosurface::Isosurface::properties);

  // Register IsosurfaceCalculator class
  class_<isosurface::IsosurfaceCalculator>("IsosurfaceCalculator")
      .constructor<>()
      .function("setMolecule", &isosurface::IsosurfaceCalculator::set_molecule)
      .function("setEnvironment",
                &isosurface::IsosurfaceCalculator::set_environment)
      .function("setWavefunction",
                &isosurface::IsosurfaceCalculator::set_wavefunction)
      .function("setParameters",
                &isosurface::IsosurfaceCalculator::set_parameters)
      .function("validate", &isosurface::IsosurfaceCalculator::validate)
      .function("compute", &isosurface::IsosurfaceCalculator::compute)
      .function("computeSurfaceProperty",
                &isosurface::IsosurfaceCalculator::compute_surface_property)
      .function("isosurface", &isosurface::IsosurfaceCalculator::isosurface,
                allow_raw_pointers())
      .function("requiresCrystal",
                &isosurface::IsosurfaceCalculator::requires_crystal)
      .function("requiresWavefunction",
                &isosurface::IsosurfaceCalculator::requires_wavefunction)
      .function("requiresEnvironment",
                &isosurface::IsosurfaceCalculator::requires_environment)
      .function("haveCrystal", &isosurface::IsosurfaceCalculator::have_crystal)
      .function("haveWavefunction",
                &isosurface::IsosurfaceCalculator::have_wavefunction)
      .function("haveEnvironment",
                &isosurface::IsosurfaceCalculator::have_environment)
      .function("errorMessage",
                &isosurface::IsosurfaceCalculator::error_message);
  // Helper function to generate an isosurface mesh with promolecule density
  function("generatePromoleculeDensityIsosurface",
           optional_override([](const core::Molecule &mol, double isovalue,
                                double separation) {
             isosurface::IsosurfaceCalculator calc;
             calc.set_molecule(mol);

             isosurface::IsosurfaceGenerationParameters params;
             params.isovalue = isovalue;
             params.separation = separation;
             params.surface_kind = isosurface::SurfaceKind::PromoleculeDensity;

             calc.set_parameters(params);

             if (!calc.validate()) {
               throw std::runtime_error(
                   "Failed to validate isosurface calculation: " +
                   calc.error_message());
             }

             calc.compute();
             const auto &surf = calc.isosurface();

             // Return mesh data as a JavaScript object
             val result = val::object();

             // Get vertices
             size_t numVertices = surf.vertices.cols();
             val vertices = val::global("Float32Array").new_(3 * numVertices);
             for (size_t i = 0; i < numVertices; ++i) {
               vertices.set(i * 3 + 0, surf.vertices(0, i));
               vertices.set(i * 3 + 1, surf.vertices(1, i));
               vertices.set(i * 3 + 2, surf.vertices(2, i));
             }

             // Get faces
             size_t numFaces = surf.faces.cols();
             val faces = val::global("Uint32Array").new_(3 * numFaces);
             for (size_t i = 0; i < numFaces; ++i) {
               faces.set(i * 3 + 0, surf.faces(0, i));
               faces.set(i * 3 + 1, surf.faces(1, i));
               faces.set(i * 3 + 2, surf.faces(2, i));
             }

             // Get normals
             val normals = val::global("Float32Array").new_(3 * numVertices);
             for (size_t i = 0; i < numVertices; ++i) {
               normals.set(i * 3 + 0, surf.normals(0, i));
               normals.set(i * 3 + 1, surf.normals(1, i));
               normals.set(i * 3 + 2, surf.normals(2, i));
             }

             result.set("vertices", vertices);
             result.set("faces", faces);
             result.set("normals", normals);
             result.set("numVertices", numVertices);
             result.set("numFaces", numFaces);
             result.set("volume", surf.volume());
             result.set("surfaceArea", surf.surface_area());

             return result;
           }));

  // Helper function to generate an isosurface mesh with electron density
  function("generateElectronDensityIsosurface",
           optional_override([](const qm::Wavefunction &wfn, double isovalue,
                                double separation) {
             isosurface::IsosurfaceCalculator calc;
             calc.set_wavefunction(wfn);

             // Extract molecule from wavefunction atoms
             std::vector<int> atomic_numbers;
             Mat3N positions(3, wfn.atoms.size());
             for (size_t i = 0; i < wfn.atoms.size(); ++i) {
               atomic_numbers.push_back(wfn.atoms[i].atomic_number);
               positions.col(i) = wfn.atoms[i].position();
             }
             core::Molecule mol(
                 IVec::Map(atomic_numbers.data(), atomic_numbers.size()),
                 positions);
             calc.set_molecule(mol);

             isosurface::IsosurfaceGenerationParameters params;
             params.isovalue = isovalue;
             params.separation = separation;
             params.surface_kind = isosurface::SurfaceKind::ElectronDensity;

             calc.set_parameters(params);

             if (!calc.validate()) {
               throw std::runtime_error(
                   "Failed to validate isosurface calculation: " +
                   calc.error_message());
             }

             calc.compute();
             const auto &surf = calc.isosurface();

             // Return mesh data as a JavaScript object
             val result = val::object();

             // Get vertices
             size_t numVertices = surf.vertices.cols();
             val vertices = val::global("Float32Array").new_(3 * numVertices);
             for (size_t i = 0; i < numVertices; ++i) {
               vertices.set(i * 3 + 0, surf.vertices(0, i));
               vertices.set(i * 3 + 1, surf.vertices(1, i));
               vertices.set(i * 3 + 2, surf.vertices(2, i));
             }

             // Get faces
             size_t numFaces = surf.faces.cols();
             val faces = val::global("Uint32Array").new_(3 * numFaces);
             for (size_t i = 0; i < numFaces; ++i) {
               faces.set(i * 3 + 0, surf.faces(0, i));
               faces.set(i * 3 + 1, surf.faces(1, i));
               faces.set(i * 3 + 2, surf.faces(2, i));
             }

             // Get normals
             val normals = val::global("Float32Array").new_(3 * numVertices);
             for (size_t i = 0; i < numVertices; ++i) {
               normals.set(i * 3 + 0, surf.normals(0, i));
               normals.set(i * 3 + 1, surf.normals(1, i));
               normals.set(i * 3 + 2, surf.normals(2, i));
             }

             result.set("vertices", vertices);
             result.set("faces", faces);
             result.set("normals", normals);
             result.set("numVertices", numVertices);
             result.set("numFaces", numFaces);
             result.set("volume", surf.volume());
             result.set("surfaceArea", surf.surface_area());

             return result;
           }));

  // Utility functions for isosurface string conversion
  function("surfaceToString",
           optional_override([](isosurface::SurfaceKind kind) {
             return std::string(isosurface::surface_to_string(kind));
           }));

  function("propertyToString",
           optional_override([](isosurface::PropertyKind kind) {
             return std::string(isosurface::property_to_string(kind));
           }));

  function("surfaceFromString", &isosurface::surface_from_string);
  function("propertyFromString", &isosurface::property_from_string);

  function("surfaceRequiresWavefunction",
           &isosurface::surface_requires_wavefunction);
  function("propertyRequiresWavefunction",
           &isosurface::property_requires_wavefunction);
  function("surfaceRequiresEnvironment",
           &isosurface::surface_requires_environment);
  function("propertyRequiresEnvironment",
           &isosurface::property_requires_environment);

  // Export isosurface as JSON string
  function("isosurfaceToJSON",
           optional_override([](const isosurface::Isosurface &surf) {
             return io::isosurface_to_json_string(surf);
           }));
}