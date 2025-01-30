#include "isosurface_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/io/ply.h>
#include <occ/isosurface/isosurface.h>

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

  return m;
}
