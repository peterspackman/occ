#include <ankerl/unordered_dense.h>
#include <fmt/format.h>
#include <occ/core/util.h>
#include <occ/isosurface/surface_types.h>

namespace occ::isosurface {

namespace {
const std::vector<SurfaceKind> surface_types{SurfaceKind::PromoleculeDensity,
                                             SurfaceKind::Hirshfeld,
                                             SurfaceKind::EEQ_ESP,
                                             SurfaceKind::ElectronDensity,
                                             SurfaceKind::ESP,
                                             SurfaceKind::SpinDensity,
                                             SurfaceKind::DeformationDensity,
                                             SurfaceKind::Orbital,
                                             SurfaceKind::CrystalVoid};

const ankerl::unordered_dense::map<std::string, SurfaceKind> name_to_surface{
    // ESP
    {"esp", SurfaceKind::ESP},
    {"electric_potential", SurfaceKind::ESP},
    {"electric potential", SurfaceKind::ESP},
    // EEQ ESP
    {"eeq_esp", SurfaceKind::EEQ_ESP},
    {"electric_potential_eeq", SurfaceKind::EEQ_ESP},
    {"electric potential eeq", SurfaceKind::EEQ_ESP},
    // Electron density
    {"electron_density", SurfaceKind::ElectronDensity},
    {"electron density", SurfaceKind::ElectronDensity},
    {"rho", SurfaceKind::ElectronDensity},
    {"density", SurfaceKind::ElectronDensity},
    // Deformation density
    {"deformation_density", SurfaceKind::DeformationDensity},
    {"deformation density", SurfaceKind::DeformationDensity},
    {"def_rho", SurfaceKind::DeformationDensity},
    {"def rho", SurfaceKind::DeformationDensity},
    {"def density", SurfaceKind::DeformationDensity},
    // Hirshfeld
    {"hirshfeld", SurfaceKind::Hirshfeld},
    {"hs", SurfaceKind::Hirshfeld},
    {"stockholder_weight", SurfaceKind::Hirshfeld},
    {"stockholder weight", SurfaceKind::Hirshfeld},
    // Orbital
    {"orbital_density", SurfaceKind::Orbital},
    {"orbital density", SurfaceKind::Orbital},
    {"orbital", SurfaceKind::Orbital},
    {"mo", SurfaceKind::Orbital},
    // Spin density
    {"spin_density", SurfaceKind::SpinDensity},
    {"spin density", SurfaceKind::SpinDensity},
    {"spin", SurfaceKind::SpinDensity},
    // Promolecule Density
    {"promol", SurfaceKind::PromoleculeDensity},
    {"promolecule_density", SurfaceKind::PromoleculeDensity},
    {"promolecule density", SurfaceKind::PromoleculeDensity},
    {"pro", SurfaceKind::PromoleculeDensity},
    // Void
    {"crystal_void", SurfaceKind::CrystalVoid},
    {"void", SurfaceKind::CrystalVoid},
};

std::vector<PropertyKind> property_types{
    PropertyKind::Dnorm,       PropertyKind::Dint_norm,
    PropertyKind::Dext_norm,   PropertyKind::Dint,
    PropertyKind::Dext,        PropertyKind::FragmentPatch,
    PropertyKind::ShapeIndex,  PropertyKind::Curvedness,
    PropertyKind::EEQ_ESP,     PropertyKind::PromoleculeDensity,
    PropertyKind::ESP,         PropertyKind::ElectronDensity,
    PropertyKind::SpinDensity, PropertyKind::DeformationDensity,
    PropertyKind::Orbital};

ankerl::unordered_dense::map<std::string, PropertyKind> name_to_property{
    {"esp", PropertyKind::ESP},
    {"electric_potential", PropertyKind::ESP},
    {"electric potential", PropertyKind::ESP},
    // EEQ ESP
    {"eeq_esp", PropertyKind::EEQ_ESP},
    {"electric_potential_eeq", PropertyKind::EEQ_ESP},
    {"electric potential eeq", PropertyKind::EEQ_ESP},
    // Electron density
    {"electron_density", PropertyKind::ElectronDensity},
    {"electron density", PropertyKind::ElectronDensity},
    {"rho", PropertyKind::ElectronDensity},
    {"density", PropertyKind::ElectronDensity},
    // Deformation density
    {"deformation_density", PropertyKind::DeformationDensity},
    {"deformation density", PropertyKind::DeformationDensity},
    {"def_rho", PropertyKind::DeformationDensity},
    {"def rho", PropertyKind::DeformationDensity},
    {"def density", PropertyKind::DeformationDensity},
    // Orbital
    {"orbital_density", PropertyKind::Orbital},
    {"orbital density", PropertyKind::Orbital},
    {"orbital", PropertyKind::Orbital},
    {"mo", PropertyKind::Orbital},
    // Spin density
    {"spin_density", PropertyKind::SpinDensity},
    {"spin density", PropertyKind::SpinDensity},
    {"spin", PropertyKind::SpinDensity},
    // Promolecule Density
    {"promol", PropertyKind::PromoleculeDensity},
    {"promolecule_density", PropertyKind::PromoleculeDensity},
    {"promolecule density", PropertyKind::PromoleculeDensity},
    {"pro", PropertyKind::PromoleculeDensity},
    // Dnorm, di, de etc.
    {"dnorm", PropertyKind::Dnorm},
    {"d_norm", PropertyKind::Dnorm},
    {"d norm", PropertyKind::Dnorm},

    {"di", PropertyKind::Dint},
    {"d_i", PropertyKind::Dint},

    {"de", PropertyKind::Dint},
    {"d_e", PropertyKind::Dint},

    {"di_norm", PropertyKind::Dint_norm},
    {"d_i_norm", PropertyKind::Dint_norm},
    {"di norm", PropertyKind::Dint_norm},
    {"d_i norm", PropertyKind::Dint_norm},

    {"de_norm", PropertyKind::Dext_norm},
    {"d_e_norm", PropertyKind::Dext_norm},
    {"de norm", PropertyKind::Dext_norm},
    {"d_e norm", PropertyKind::Dext_norm},

    {"de_norm", PropertyKind::Dext_norm},
    {"de norm", PropertyKind::Dext_norm},

    {"fragment_patch", PropertyKind::FragmentPatch},
    {"fragment patch", PropertyKind::FragmentPatch},
    {"frag", PropertyKind::FragmentPatch},

    {"shape_index", PropertyKind::ShapeIndex},
    {"shape index", PropertyKind::ShapeIndex},

    {"curvedness", PropertyKind::Curvedness},

    {"gaussian_curvature", PropertyKind::GaussianCurvature},
    {"gaussian curvature", PropertyKind::GaussianCurvature},

    {"mean_curvature", PropertyKind::MeanCurvature},
    {"mean curvature", PropertyKind::MeanCurvature},

    {"k1", PropertyKind::CurvatureK1},
    {"k2", PropertyKind::CurvatureK2},

};

std::vector<PropertyKind> default_properties_with_environment{
    PropertyKind::Dint,         PropertyKind::Dint_norm,
    PropertyKind::ShapeIndex,   PropertyKind::Curvedness,
    PropertyKind::EEQ_ESP,      PropertyKind::Dext,
    PropertyKind::Dext_norm,    PropertyKind::Dnorm,
    PropertyKind::FragmentPatch};

std::vector<PropertyKind> default_properties_no_environment{
    PropertyKind::Dint, PropertyKind::Dint_norm, PropertyKind::ShapeIndex,
    PropertyKind::Curvedness, PropertyKind::EEQ_ESP};

} // anonymous namespace

SurfaceKind surface_from_string(const std::string &surface_name) {
  auto lowercase_name = occ::util::to_lower_copy(surface_name);
  auto it = name_to_surface.find(lowercase_name);
  if (it != name_to_surface.end()) {
    return it->second;
  }
  throw std::runtime_error(
      fmt::format("Unknown surface type: {}", surface_name));
}

const std::vector<SurfaceKind> &available_surface_types() {
  return surface_types;
}

const std::vector<PropertyKind> &default_properties(bool have_environment) {

  if (have_environment)
    return default_properties_with_environment;
  else
    return default_properties_no_environment;
}

PropertyKind property_from_string(const std::string &name) {
  auto lowercase_name = occ::util::to_lower_copy(name);
  auto it = name_to_property.find(lowercase_name);
  if (it != name_to_property.end()) {
    return it->second;
  }
  throw std::runtime_error(fmt::format("Unknown property kind: {}", name));
}

} // namespace occ::isosurface
