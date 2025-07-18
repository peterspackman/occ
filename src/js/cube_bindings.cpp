#include "cube_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <memory>
#include <occ/core/atom.h>
#include <occ/core/molecule.h>
#include <occ/io/cube.h>
#include <occ/isosurface/electron_density.h>
#include <occ/qm/wavefunction.h>
#include <sstream>

using namespace emscripten;
using namespace occ;

void register_cube_bindings() {
  // Register Cube class
  class_<io::Cube>("Cube")
      .constructor<>()
      .property("name", &io::Cube::name)
      .property("description", &io::Cube::description)
      .function("setOrigin", optional_override(
                                 [](io::Cube &cube, double x, double y,
                                    double z) { cube.origin = Vec3(x, y, z); }))
      .function("getOrigin", optional_override([](const io::Cube &cube) {
                  val result = val::array();
                  result.set(0, cube.origin(0));
                  result.set(1, cube.origin(1));
                  result.set(2, cube.origin(2));
                  return result;
                }))
      .function("setBasis",
                optional_override([](io::Cube &cube, const val &basisArray) {
                  // Expect a 3x3 matrix as a flat array of 9 elements
                  if (basisArray["length"].as<int>() != 9) {
                    throw std::runtime_error(
                        "Basis matrix must be 3x3 (9 elements)");
                  }
                  for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                      cube.basis(i, j) = basisArray[i * 3 + j].as<double>();
                    }
                  }
                }))
      .function("getBasis", optional_override([](const io::Cube &cube) {
                  val result = val::global("Float64Array").new_(9);
                  for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                      result.set(i * 3 + j, cube.basis(i, j));
                    }
                  }
                  return result;
                }))
      .function("setSteps",
                optional_override([](io::Cube &cube, int nx, int ny, int nz) {
                  cube.steps = IVec3(nx, ny, nz);
                }))
      .function("getSteps", optional_override([](const io::Cube &cube) {
                  val result = val::array();
                  result.set(0, cube.steps(0));
                  result.set(1, cube.steps(1));
                  result.set(2, cube.steps(2));
                  return result;
                }))
      .function("addAtom",
                optional_override([](io::Cube &cube, int atomicNumber, double x,
                                     double y, double z) {
                  cube.atoms.push_back({atomicNumber, x, y, z});
                }))
      .function("setMolecule", optional_override([](io::Cube &cube,
                                                    const core::Molecule &mol) {
                  cube.atoms.clear();
                  for (size_t i = 0; i < mol.size(); ++i) {
                    cube.atoms.push_back(mol.atoms()[i]);
                  }
                }))
      .function("centerMolecule", &io::Cube::center_molecule)
      .function("getData", optional_override([](const io::Cube &cube) {
                  size_t totalSize =
                      cube.steps(0) * cube.steps(1) * cube.steps(2);
                  val result = val::global("Float32Array").new_(totalSize);
                  const float *data = cube.data();
                  for (size_t i = 0; i < totalSize; ++i) {
                    result.set(i, data[i]);
                  }
                  return result;
                }))
      .function("setData",
                optional_override([](io::Cube &cube, const val &dataArray) {
                  size_t totalSize =
                      cube.steps(0) * cube.steps(1) * cube.steps(2);
                  if (dataArray["length"].as<size_t>() != totalSize) {
                    throw std::runtime_error(
                        "Data array size must match grid dimensions");
                  }
                  float *data = cube.data();
                  for (size_t i = 0; i < totalSize; ++i) {
                    data[i] = dataArray[i].as<float>();
                  }
                }))
      .function(
          "fillFromElectronDensity",
          optional_override([](io::Cube &cube, const qm::Wavefunction &wfn) {
            // Create electron density functor and fill the cube
            isosurface::ElectronDensityFunctor func(wfn);
            auto batch_func = [&func](const Mat3N &points, Vec &result) {
              // Convert double precision points to float for the functor
              FMat3N float_points = points.cast<float>();
              FVec float_result(result.size());
              func.batch(float_points, float_result);
              result = float_result.cast<double>();
            };
            cube.fill_data_from_function(batch_func);
          }))
      .function("save", optional_override(
                            [](io::Cube &cube, const std::string &filename) {
                              cube.save(filename);
                            }))
      .function("toString", optional_override([](io::Cube &cube) {
                  std::ostringstream oss;
                  cube.save(oss);
                  return oss.str();
                }));

  // Static methods for loading - using shared_ptr to avoid copy issues
  function("loadCube", optional_override([](const std::string &filename) {
             auto cube = std::make_shared<io::Cube>(io::Cube::load(filename));
             return cube;
           }),
           allow_raw_pointers());

  function("loadCubeFromString",
           optional_override([](const std::string &cubeData) {
             std::istringstream iss(cubeData);
             auto cube = std::make_shared<io::Cube>(io::Cube::load(iss));
             return cube;
           }),
           allow_raw_pointers());
}