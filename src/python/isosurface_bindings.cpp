#include "isosurface_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/isosurface/isosurface.h>

using namespace occ::isosurface;

nb::module_ register_isosurface_bindings(nb::module_ &parent) {
  auto m = parent.def_submodule("isosurface",
                                "Isosurface evaluation functionality.");

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
  return m;
}
