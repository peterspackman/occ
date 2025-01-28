#pragma once
#include <string>
#include <vector>

namespace occ::isosurface {

enum class SurfaceKind {
  PromoleculeDensity,
  Hirshfeld,
  EEQ_ESP,
  ElectronDensity,
  ESP,
  SpinDensity,
  DeformationDensity,
  Orbital,
  CrystalVoid
};

enum class PropertyKind {
  Dnorm,
  Dint_norm,
  Dext_norm,
  Dint,
  Dext,
  FragmentPatch,
  ShapeIndex,
  Curvedness,
  EEQ_ESP,
  PromoleculeDensity,
  ESP,
  ElectronDensity,
  SpinDensity,
  DeformationDensity,
  Orbital,
  GaussianCurvature,
  MeanCurvature,
  CurvatureK1,
  CurvatureK2,
};

constexpr inline const char *property_to_string(PropertyKind prop) {
  switch (prop) {
  case PropertyKind::Dnorm:
    return "dnorm";
  case PropertyKind::Dint_norm:
    return "di_norm";
  case PropertyKind::Dint:
    return "di";
  case PropertyKind::Dext_norm:
    return "de_norm";
  case PropertyKind::Dext:
    return "de";
  case PropertyKind::FragmentPatch:
    return "fragment_patch";
  case PropertyKind::ShapeIndex:
    return "shape_index";
  case PropertyKind::Curvedness:
    return "curvedness";
  case PropertyKind::PromoleculeDensity:
    return "promolecule_density";
  case PropertyKind::EEQ_ESP:
    return "eeq_esp";
  case PropertyKind::ESP:
    return "esp";
  case PropertyKind::ElectronDensity:
    return "electron_density";
  case PropertyKind::DeformationDensity:
    return "deformation_density";
  case PropertyKind::Orbital:
    return "orbital_density";
  case PropertyKind::SpinDensity:
    return "spin_density";
  case PropertyKind::GaussianCurvature:
    return "gaussian_curvature";
  case PropertyKind::MeanCurvature:
    return "mean_curvature";
  case PropertyKind::CurvatureK1:
    return "k1";
  case PropertyKind::CurvatureK2:
    return "k2";
  default:
    return "unknown_property";
  }
}

constexpr inline const char *surface_to_string(SurfaceKind surface) {
  switch (surface) {
  case SurfaceKind::PromoleculeDensity:
    return "promolecule_density";
  case SurfaceKind::Hirshfeld:
    return "hirshfeld";
  case SurfaceKind::EEQ_ESP:
    return "eeq_esp";
  case SurfaceKind::ESP:
    return "esp";
  case SurfaceKind::ElectronDensity:
    return "electron_density";
  case SurfaceKind::DeformationDensity:
    return "deformation_density";
  case SurfaceKind::Orbital:
    return "orbital_density";
  case SurfaceKind::SpinDensity:
    return "spin_density";
  case SurfaceKind::CrystalVoid:
    return "void";
  default:
    return "unknown_surface";
  }
}

constexpr inline bool surface_requires_wavefunction(SurfaceKind kind) {
  switch (kind) {
  case SurfaceKind::ESP:
    return true;
  case SurfaceKind::ElectronDensity:
    return true;
  case SurfaceKind::DeformationDensity:
    return true;
  case SurfaceKind::Orbital:
    return true;
  case SurfaceKind::SpinDensity:
    return true;
  default:
    return false;
  }
}

constexpr inline bool property_requires_wavefunction(PropertyKind kind) {
  switch (kind) {
  case PropertyKind::ESP:
    return true;
  case PropertyKind::ElectronDensity:
    return true;
  case PropertyKind::DeformationDensity:
    return true;
  case PropertyKind::Orbital:
    return true;
  case PropertyKind::SpinDensity:
    return true;
  default:
    return false;
  }
}

constexpr inline bool surface_requires_environment(SurfaceKind kind) {
  switch (kind) {
  case SurfaceKind::Hirshfeld:
    return true;
  default:
    return false;
  }
}

constexpr inline bool property_requires_environment(PropertyKind kind) {
  switch (kind) {
  case PropertyKind::Dext:
    return true;
  case PropertyKind::Dext_norm:
    return true;
  case PropertyKind::Dnorm:
    return true;
  case PropertyKind::FragmentPatch:
    return true;
  default:
    return false;
  }
}

SurfaceKind surface_from_string(const std::string &);
const std::vector<SurfaceKind> &available_surface_types();

const std::vector<PropertyKind> &default_properties(bool have_environment);
PropertyKind property_from_string(const std::string &);

std::vector<PropertyKind> surface_properties_to_compute(
    const std::vector<PropertyKind> &additional_properties,
    bool have_environment);

} // namespace occ::isosurface
