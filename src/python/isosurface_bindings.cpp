#include "isosurface_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/io/ply.h>
#include <occ/isosurface/isosurface.h>
#include <occ/isosurface/volume_calculator.h>
#include <occ/isosurface/volume_data.h>

using namespace occ::isosurface;

using occ::FVec;
using occ::IVec;

nb::module_ register_isosurface_bindings(nb::module_ &m) {
  using namespace nb::literals;

  nb::enum_<SurfaceKind>(m, "SurfaceKind")
      .value("PromoleculeDensity", SurfaceKind::PromoleculeDensity)
      .value("Hirshfeld", SurfaceKind::Hirshfeld)
      .value("EEQ_ESP", SurfaceKind::EEQ_ESP)
      .value("ElectronDensity", SurfaceKind::ElectronDensity)
      .value("ESP", SurfaceKind::ESP)
      .value("SpinDensity", SurfaceKind::SpinDensity)
      .value("DeformationDensity", SurfaceKind::DeformationDensity)
      .value("Orbital", SurfaceKind::Orbital)
      .value("CrystalVoid", SurfaceKind::CrystalVoid);

  nb::enum_<PropertyKind>(m, "PropertyKind")
      .value("Dnorm", PropertyKind::Dnorm)
      .value("Dint_norm", PropertyKind::Dint_norm)
      .value("Dext_norm", PropertyKind::Dext_norm)
      .value("Dint", PropertyKind::Dint)
      .value("Dext", PropertyKind::Dext)
      .value("FragmentPatch", PropertyKind::FragmentPatch)
      .value("ShapeIndex", PropertyKind::ShapeIndex)
      .value("Curvedness", PropertyKind::Curvedness)
      .value("EEQ_ESP", PropertyKind::EEQ_ESP)
      .value("PromoleculeDensity", PropertyKind::PromoleculeDensity)
      .value("ESP", PropertyKind::ESP)
      .value("ElectronDensity", PropertyKind::ElectronDensity)
      .value("SpinDensity", PropertyKind::SpinDensity)
      .value("DeformationDensity", PropertyKind::DeformationDensity)
      .value("Orbital", PropertyKind::Orbital);

  nb::class_<IsosurfaceProperties>(m, "IsosurfaceProperties")
      .def(nb::init<>())
      .def(
          "add",
          [](IsosurfaceProperties &self, const std::string &name,
             const Eigen::VectorXf &values) { self.add<FVec>(name, values); },
          "Add float property")
      .def(
          "add",
          [](IsosurfaceProperties &self, const std::string &name,
             const Eigen::VectorXi &values) { self.add<IVec>(name, values); },
          "Add integer property")
      .def("has_property", &IsosurfaceProperties::has_property)
      .def(
          "get_float",
          [](const IsosurfaceProperties &self,
             const std::string &name) -> nb::object {
            if (auto ptr = self.get<FVec>(name)) {
              return nb::cast(*ptr);
            }
            return nb::none();
          },
          "Get float property")
      .def(
          "get_int",
          [](const IsosurfaceProperties &self,
             const std::string &name) -> nb::object {
            if (auto ptr = self.get<IVec>(name)) {
              return nb::cast(*ptr);
            }
            return nb::none();
          },
          "Get integer property")
      .def("merge", &IsosurfaceProperties::merge)
      .def("count", &IsosurfaceProperties::count);

  nb::class_<Isosurface>(m, "Isosurface")
      .def(nb::init<>())
      .def_rw("vertices", &Isosurface::vertices)
      .def_rw("faces", &Isosurface::faces)
      .def_rw("normals", &Isosurface::normals)
      .def_rw("gaussian_curvature", &Isosurface::gaussian_curvature)
      .def_rw("mean_curvature", &Isosurface::mean_curvature)
      .def_rw("properties", &Isosurface::properties)
      .def(
          "save",
          [](const Isosurface &iso, const std::string &filename, bool binary) {
            occ::io::write_ply_mesh(filename, iso, binary);
          },
          "filename"_a, "binary"_a = true);

  nb::class_<IsosurfaceGenerationParameters>(m,
                                             "IsosurfaceGenerationParameters")
      .def(nb::init<>())
      .def_rw("isovalue", &IsosurfaceGenerationParameters::isovalue)
      .def_rw("separation", &IsosurfaceGenerationParameters::separation)
      .def_rw("background_density",
              &IsosurfaceGenerationParameters::background_density)
      .def_rw("surface_orbital_index",
              &IsosurfaceGenerationParameters::surface_orbital_index)
      .def_rw("property_orbital_indices",
              &IsosurfaceGenerationParameters::property_orbital_indices)
      .def_rw("flip_normals", &IsosurfaceGenerationParameters::flip_normals)
      .def_rw("binary_output", &IsosurfaceGenerationParameters::binary_output)
      .def_rw("surface_kind", &IsosurfaceGenerationParameters::surface_kind)
      .def_rw("properties", &IsosurfaceGenerationParameters::properties);

  nb::class_<IsosurfaceCalculator>(m, "IsosurfaceCalculator")
      .def(nb::init<>())
      .def("set_molecule", &IsosurfaceCalculator::set_molecule)
      .def("set_environment", &IsosurfaceCalculator::set_environment)
      .def("set_wavefunction", &IsosurfaceCalculator::set_wavefunction)
      .def("set_crystal", &IsosurfaceCalculator::set_crystal)
      .def("set_parameters", &IsosurfaceCalculator::set_parameters)
      .def("validate", &IsosurfaceCalculator::validate)
      .def("compute", &IsosurfaceCalculator::compute)
      .def("compute_surface_property",
           &IsosurfaceCalculator::compute_surface_property)
      .def("isosurface", &IsosurfaceCalculator::isosurface)
      .def("requires_crystal", &IsosurfaceCalculator::requires_crystal)
      .def("requires_wavefunction",
           &IsosurfaceCalculator::requires_wavefunction)
      .def("requires_environment", &IsosurfaceCalculator::requires_environment)
      .def("have_crystal", &IsosurfaceCalculator::have_crystal)
      .def("have_wavefunction", &IsosurfaceCalculator::have_wavefunction)
      .def("have_environment", &IsosurfaceCalculator::have_environment)
      .def("error_message", &IsosurfaceCalculator::error_message);

  nb::class_<ElectronDensityFunctor>(m, "ElectronDensityFunctor")
      .def(nb::init<const occ::qm::Wavefunction &, int>(), "wavefunction"_a,
           "mo_index"_a = 1)
      .def_prop_rw("orbital_index", &ElectronDensityFunctor::orbital_index,
                   &ElectronDensityFunctor::set_orbital_index)
      .def("__call__",
           [](const ElectronDensityFunctor &f, const occ::FMat3N &pts) -> FVec {
             FVec result(pts.cols());
             f.batch(pts, result);
             return result;
           })
      .def("num_calls", &ElectronDensityFunctor::num_calls);

  nb::class_<ElectricPotentialFunctor>(m, "ElectricPotentialFunctor")
      .def(nb::init<const occ::qm::Wavefunction &>(), "wavefunction"_a)
      .def("__call__",
           [](const ElectricPotentialFunctor &f,
              const occ::FMat3N &pts) -> FVec {
             FVec result(pts.cols());
             f.batch(pts, result);
             return result;
           })
      .def("num_calls", &ElectricPotentialFunctor::num_calls);

  nb::class_<ElectricPotentialFunctorPC>(m, "ElectricPotentialFunctorPC")
      .def(nb::init<const occ::core::Molecule &>(), "molecule"_a)
      .def("__call__",
           [](const ElectricPotentialFunctorPC &f,
              const occ::FMat3N &pts) -> FVec {
             FVec result(pts.cols());
             f.batch(pts, result);
             return result;
           })
      .def("num_calls", &ElectricPotentialFunctorPC::num_calls);

  // Volume generation bindings
  nb::enum_<VolumePropertyKind>(m, "VolumePropertyKind")
      .value("ElectronDensity", VolumePropertyKind::ElectronDensity)
      .value("ElectronDensityAlpha", VolumePropertyKind::ElectronDensityAlpha)
      .value("ElectronDensityBeta", VolumePropertyKind::ElectronDensityBeta)
      .value("ElectricPotential", VolumePropertyKind::ElectricPotential)
      .value("EEQ_ESP", VolumePropertyKind::EEQ_ESP)
      .value("PromoleculeDensity", VolumePropertyKind::PromoleculeDensity)
      .value("DeformationDensity", VolumePropertyKind::DeformationDensity)
      .value("XCDensity", VolumePropertyKind::XCDensity)
      .value("CrystalVoid", VolumePropertyKind::CrystalVoid);

  nb::enum_<SpinConstraint>(m, "SpinConstraint")
      .value("Total", SpinConstraint::Total)
      .value("Alpha", SpinConstraint::Alpha)
      .value("Beta", SpinConstraint::Beta);

  nb::class_<VolumeGenerationParameters>(m, "VolumeGenerationParameters")
      .def(nb::init<>())
      .def_rw("property", &VolumeGenerationParameters::property)
      .def_rw("spin", &VolumeGenerationParameters::spin)
      .def_rw("functional", &VolumeGenerationParameters::functional)
      .def_rw("mo_number", &VolumeGenerationParameters::mo_number)
      .def_rw("steps", &VolumeGenerationParameters::steps)
      .def_rw("da", &VolumeGenerationParameters::da)
      .def_rw("db", &VolumeGenerationParameters::db)
      .def_rw("dc", &VolumeGenerationParameters::dc)
      .def_rw("origin", &VolumeGenerationParameters::origin)
      .def_rw("adaptive_bounds", &VolumeGenerationParameters::adaptive_bounds)
      .def_rw("value_threshold", &VolumeGenerationParameters::value_threshold)
      .def_rw("buffer_distance", &VolumeGenerationParameters::buffer_distance)
      .def_rw("crystal_filename", &VolumeGenerationParameters::crystal_filename);

  nb::class_<VolumeData>(m, "VolumeData")
      .def_ro("name", &VolumeData::name)
      .def_ro("property", &VolumeData::property)
      .def("nx", &VolumeData::nx)
      .def("ny", &VolumeData::ny)
      .def("nz", &VolumeData::nz)
      .def("total_points", &VolumeData::total_points)
      .def_ro("origin", &VolumeData::origin)
      .def_ro("basis", &VolumeData::basis)
      .def_ro("steps", &VolumeData::steps)
      .def("get_data", [](const VolumeData &v) {
        // Return flattened data - Python can reshape as needed
        std::vector<double> result;
        result.reserve(v.total_points());
        
        for (int i = 0; i < v.nx(); i++) {
            for (int j = 0; j < v.ny(); j++) {
                for (int k = 0; k < v.nz(); k++) {
                    result.push_back(v.data(i, j, k));
                }
            }
        }
        
        return result;
      }, "Get volume data as flattened vector (use reshape((nx, ny, nz)) in Python)");

  nb::class_<VolumeCalculator>(m, "VolumeCalculator")
      .def(nb::init<>())
      .def("set_wavefunction", &VolumeCalculator::set_wavefunction)
      .def("set_molecule", &VolumeCalculator::set_molecule)
      .def("compute_volume", &VolumeCalculator::compute_volume)
      .def("volume_as_cube_string", &VolumeCalculator::volume_as_cube_string)
      .def_static("compute_density_volume", &VolumeCalculator::compute_density_volume)
      .def_static("compute_mo_volume", &VolumeCalculator::compute_mo_volume);

  // Convenience functions for common volume types
  m.def("generate_electron_density_cube", 
        [](const occ::qm::Wavefunction &wfn, int nx, int ny, int nz) {
          VolumeCalculator calc;
          calc.set_wavefunction(wfn);
          
          VolumeGenerationParameters params;
          params.property = VolumePropertyKind::ElectronDensity;
          params.steps = {nx, ny, nz};
          
          auto volume = calc.compute_volume(params);
          return calc.volume_as_cube_string(volume);
        }, "wavefunction"_a, "nx"_a, "ny"_a, "nz"_a,
        "Generate electron density cube file as string");

  m.def("generate_mo_cube",
        [](const occ::qm::Wavefunction &wfn, int mo_index, int nx, int ny, int nz) {
          VolumeCalculator calc;
          calc.set_wavefunction(wfn);
          
          VolumeGenerationParameters params;
          params.property = VolumePropertyKind::ElectronDensity;
          params.mo_number = mo_index;
          params.steps = {nx, ny, nz};
          
          auto volume = calc.compute_volume(params);
          return calc.volume_as_cube_string(volume);
        }, "wavefunction"_a, "mo_index"_a, "nx"_a, "ny"_a, "nz"_a,
        "Generate molecular orbital cube file as string");

  m.def("generate_esp_cube",
        [](const occ::qm::Wavefunction &wfn, int nx, int ny, int nz) {
          VolumeCalculator calc;
          calc.set_wavefunction(wfn);
          
          VolumeGenerationParameters params;
          params.property = VolumePropertyKind::ElectricPotential;
          params.steps = {nx, ny, nz};
          
          auto volume = calc.compute_volume(params);
          return calc.volume_as_cube_string(volume);
        }, "wavefunction"_a, "nx"_a, "ny"_a, "nz"_a,
        "Generate electrostatic potential cube file as string");

  return m;
}
