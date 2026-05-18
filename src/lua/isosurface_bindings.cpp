#include "isosurface_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/io/ply.h>
#include <occ/isosurface/isosurface.h>
#include <occ/isosurface/volume_calculator.h>
#include <occ/isosurface/volume_data.h>

namespace sol {
template <>
struct is_automagical<occ::isosurface::Isosurface> : std::false_type {};
template <>
struct is_automagical<occ::isosurface::IsosurfaceCalculator>
    : std::false_type {};
template <>
struct is_automagical<occ::isosurface::IsosurfaceProperties>
    : std::false_type {};
template <>
struct is_automagical<occ::isosurface::VolumeCalculator> : std::false_type {};
template <>
struct is_automagical<occ::isosurface::VolumeData> : std::false_type {};
template <>
struct is_automagical<occ::isosurface::ElectronDensityFunctor>
    : std::false_type {};
template <>
struct is_automagical<occ::isosurface::ElectricPotentialFunctor>
    : std::false_type {};
template <>
struct is_automagical<occ::isosurface::ElectricPotentialFunctorPC>
    : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using namespace occ::isosurface;

void register_isosurface_bindings(sol::state_view, sol::table &m) {
  m.new_enum<SurfaceKind>(
      "SurfaceKind",
      {{"PromoleculeDensity", SurfaceKind::PromoleculeDensity},
       {"Hirshfeld", SurfaceKind::Hirshfeld},
       {"EEQ_ESP", SurfaceKind::EEQ_ESP},
       {"ElectronDensity", SurfaceKind::ElectronDensity},
       {"ESP", SurfaceKind::ESP},
       {"SpinDensity", SurfaceKind::SpinDensity},
       {"DeformationDensity", SurfaceKind::DeformationDensity},
       {"Orbital", SurfaceKind::Orbital},
       {"CrystalVoid", SurfaceKind::CrystalVoid}});

  m.new_enum<PropertyKind>(
      "PropertyKind", {{"Dnorm", PropertyKind::Dnorm},
                        {"Dint_norm", PropertyKind::Dint_norm},
                        {"Dext_norm", PropertyKind::Dext_norm},
                        {"Dint", PropertyKind::Dint},
                        {"Dext", PropertyKind::Dext},
                        {"FragmentPatch", PropertyKind::FragmentPatch},
                        {"ShapeIndex", PropertyKind::ShapeIndex},
                        {"Curvedness", PropertyKind::Curvedness},
                        {"EEQ_ESP", PropertyKind::EEQ_ESP},
                        {"PromoleculeDensity", PropertyKind::PromoleculeDensity},
                        {"ESP", PropertyKind::ESP},
                        {"ElectronDensity", PropertyKind::ElectronDensity},
                        {"SpinDensity", PropertyKind::SpinDensity},
                        {"DeformationDensity", PropertyKind::DeformationDensity},
                        {"Orbital", PropertyKind::Orbital}});

  m.new_usertype<IsosurfaceProperties>(
      "IsosurfaceProperties",
      sol::call_constructor,
      sol::factories([]() { return IsosurfaceProperties{}; }),
      "has_property", &IsosurfaceProperties::has_property,
      "count", &IsosurfaceProperties::count);

  m.new_usertype<Isosurface>(
      "Isosurface",
      sol::call_constructor, sol::factories([]() { return Isosurface{}; }),
      "save",
      [](const Isosurface &iso, const std::string &filename,
         sol::optional<bool> binary) {
        occ::io::write_ply_mesh(filename, iso, binary.value_or(true));
      });

  m.new_usertype<IsosurfaceGenerationParameters>(
      "IsosurfaceGenerationParameters",
      sol::call_constructor,
      sol::factories([]() { return IsosurfaceGenerationParameters{}; }),
      "isovalue", &IsosurfaceGenerationParameters::isovalue,
      "separation", &IsosurfaceGenerationParameters::separation,
      "background_density",
      &IsosurfaceGenerationParameters::background_density,
      "surface_orbital_index",
      &IsosurfaceGenerationParameters::surface_orbital_index,
      "flip_normals", &IsosurfaceGenerationParameters::flip_normals,
      "binary_output", &IsosurfaceGenerationParameters::binary_output,
      "surface_kind", &IsosurfaceGenerationParameters::surface_kind);

  m.new_usertype<IsosurfaceCalculator>(
      "IsosurfaceCalculator",
      sol::call_constructor,
      sol::factories([]() { return IsosurfaceCalculator{}; }),
      "set_molecule", &IsosurfaceCalculator::set_molecule,
      "set_environment", &IsosurfaceCalculator::set_environment,
      "set_wavefunction", &IsosurfaceCalculator::set_wavefunction,
      "set_crystal", &IsosurfaceCalculator::set_crystal,
      "set_parameters", &IsosurfaceCalculator::set_parameters,
      "validate", &IsosurfaceCalculator::validate,
      "compute", &IsosurfaceCalculator::compute,
      "isosurface", &IsosurfaceCalculator::isosurface,
      "requires_crystal", &IsosurfaceCalculator::requires_crystal,
      "requires_wavefunction", &IsosurfaceCalculator::requires_wavefunction,
      "requires_environment", &IsosurfaceCalculator::requires_environment,
      "have_crystal", &IsosurfaceCalculator::have_crystal,
      "have_wavefunction", &IsosurfaceCalculator::have_wavefunction,
      "have_environment", &IsosurfaceCalculator::have_environment,
      "error_message", &IsosurfaceCalculator::error_message);

  m.new_enum<VolumePropertyKind>(
      "VolumePropertyKind",
      {{"ElectronDensity", VolumePropertyKind::ElectronDensity},
       {"ElectronDensityAlpha", VolumePropertyKind::ElectronDensityAlpha},
       {"ElectronDensityBeta", VolumePropertyKind::ElectronDensityBeta},
       {"ElectricPotential", VolumePropertyKind::ElectricPotential},
       {"EEQ_ESP", VolumePropertyKind::EEQ_ESP},
       {"PromoleculeDensity", VolumePropertyKind::PromoleculeDensity},
       {"DeformationDensity", VolumePropertyKind::DeformationDensity},
       {"XCDensity", VolumePropertyKind::XCDensity},
       {"CrystalVoid", VolumePropertyKind::CrystalVoid}});

  m.new_enum<SpinConstraint>("SpinConstraint",
                              {{"Total", SpinConstraint::Total},
                               {"Alpha", SpinConstraint::Alpha},
                               {"Beta", SpinConstraint::Beta}});

  m.new_usertype<VolumeGenerationParameters>(
      "VolumeGenerationParameters",
      sol::call_constructor,
      sol::factories([]() { return VolumeGenerationParameters{}; }),
      "property", &VolumeGenerationParameters::property,
      "spin", &VolumeGenerationParameters::spin,
      "functional", &VolumeGenerationParameters::functional,
      "mo_number", &VolumeGenerationParameters::mo_number,
      "value_threshold", &VolumeGenerationParameters::value_threshold,
      "buffer_distance", &VolumeGenerationParameters::buffer_distance,
      "crystal_filename", &VolumeGenerationParameters::crystal_filename,
      "adaptive_bounds", &VolumeGenerationParameters::adaptive_bounds);

  m.new_usertype<VolumeData>(
      "VolumeData", sol::no_constructor,
      "name", sol::readonly(&VolumeData::name),
      "property", sol::readonly(&VolumeData::property),
      "nx", &VolumeData::nx,
      "ny", &VolumeData::ny,
      "nz", &VolumeData::nz,
      "total_points", &VolumeData::total_points,
      // Flattened (k-fastest) data for downstream reshape.
      "get_data",
      [](const VolumeData &v, sol::this_state s) {
        sol::state_view lua(s);
        sol::table out =
            lua.create_table(static_cast<int>(v.total_points()), 0);
        int idx = 1;
        for (int i = 0; i < v.nx(); ++i) {
          for (int j = 0; j < v.ny(); ++j) {
            for (int k = 0; k < v.nz(); ++k) {
              out[idx++] = v.data(i, j, k);
            }
          }
        }
        return out;
      });

  m.new_usertype<VolumeCalculator>(
      "VolumeCalculator",
      sol::call_constructor,
      sol::factories([]() { return VolumeCalculator{}; }),
      "set_wavefunction", &VolumeCalculator::set_wavefunction,
      "set_molecule", &VolumeCalculator::set_molecule,
      "compute_volume", &VolumeCalculator::compute_volume,
      "volume_as_cube_string", &VolumeCalculator::volume_as_cube_string);

  // Convenience generators
  m.set_function("generate_electron_density_cube",
                 [](const occ::qm::Wavefunction &wfn, int nx, int ny, int nz) {
                   VolumeCalculator calc;
                   calc.set_wavefunction(wfn);
                   VolumeGenerationParameters params;
                   params.property = VolumePropertyKind::ElectronDensity;
                   params.steps = {nx, ny, nz};
                   return calc.volume_as_cube_string(
                       calc.compute_volume(params));
                 });
  m.set_function("generate_esp_cube",
                 [](const occ::qm::Wavefunction &wfn, int nx, int ny, int nz) {
                   VolumeCalculator calc;
                   calc.set_wavefunction(wfn);
                   VolumeGenerationParameters params;
                   params.property = VolumePropertyKind::ElectricPotential;
                   params.steps = {nx, ny, nz};
                   return calc.volume_as_cube_string(
                       calc.compute_volume(params));
                 });
}

} // namespace occ::lua_bindings
