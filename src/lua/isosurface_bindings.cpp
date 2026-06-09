#include "isosurface_bindings.h"
#include "eigen_conv.h"
#include "enum_stacks.h"
#include <fmt/core.h>
#include <occ/isosurface/ply.h>
#include <occ/isosurface/isosurface.h>
#include <occ/isosurface/volume_calculator.h>
#include <occ/isosurface/volume_data.h>

namespace occ::lua_bindings {

using namespace occ::isosurface;
namespace lb = luabridge;

void register_isosurface_bindings(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      // Enums round-trip through luabridge::Stack<E> (see enum_stacks.h);
      // value lists are defined once in enum_defs.h.
      OCC_LUA_ENUM_NAMESPACE("SurfaceKind", OCC_ENUM_SurfaceKind)
      OCC_LUA_ENUM_NAMESPACE("PropertyKind", OCC_ENUM_PropertyKind)
      OCC_LUA_ENUM_NAMESPACE("OrbitalReference", OCC_ENUM_OrbitalReference)

      .beginClass<IsosurfaceProperties>("IsosurfaceProperties")
      .addConstructor<void (*)()>()
      .addFunction("has_property", &IsosurfaceProperties::has_property)
      .addProperty("count", &IsosurfaceProperties::count)
      .endClass()

      .beginClass<Isosurface>("Isosurface")
      .addConstructor<void (*)()>()
      // sol::optional<bool> binary default = true; split into two named
      // functions. Use a LuaRef for the optional `binary` arg so the
      // default fires when omitted (a plain `bool` would silently
      // decode-as-false and write ASCII).
      .addFunction(
          "save",
          +[](const Isosurface *iso, const std::string &filename,
              const lb::LuaRef &binary) {
            const bool b = binary.isNil() ? true : binary.unsafe_cast<bool>();
            occ::io::write_ply_mesh(filename, *iso, b);
          })
      .endClass()

      .beginClass<IsosurfaceGenerationParameters>(
          "IsosurfaceGenerationParameters")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("isovalue",
                            &IsosurfaceGenerationParameters::isovalue)
      .addPropertyReadWrite("separation",
                            &IsosurfaceGenerationParameters::separation)
      .addPropertyReadWrite("background_density",
                            &IsosurfaceGenerationParameters::background_density)
      // surface_orbital_index is OrbitalIndex (a struct with an inner
      // enum); we expose its scalar fields through getters/setters so
      // Lua can write `p.surface_orbital_offset = 1`.
      .addProperty(
          "surface_orbital_offset",
          +[](const IsosurfaceGenerationParameters *p) {
            return p->surface_orbital_index.offset;
          },
          +[](IsosurfaceGenerationParameters *p, int v) {
            p->surface_orbital_index.offset = v;
          })
      .addProperty(
          "surface_orbital_reference",
          +[](const IsosurfaceGenerationParameters *p) {
            return p->surface_orbital_index.reference;
          },
          +[](IsosurfaceGenerationParameters *p, OrbitalIndex::Reference r) {
            p->surface_orbital_index.reference = r;
          })
      .addPropertyReadWrite("flip_normals",
                            &IsosurfaceGenerationParameters::flip_normals)
      .addPropertyReadWrite("binary_output",
                            &IsosurfaceGenerationParameters::binary_output)
      .addPropertyReadWrite("surface_kind",
                            &IsosurfaceGenerationParameters::surface_kind)
      .endClass()

      .beginClass<IsosurfaceCalculator>("IsosurfaceCalculator")
      .addConstructor<void (*)()>()
      .addFunction("set_molecule", &IsosurfaceCalculator::set_molecule)
      .addFunction("set_environment", &IsosurfaceCalculator::set_environment)
      .addFunction("set_wavefunction", &IsosurfaceCalculator::set_wavefunction)
      .addFunction("set_crystal", &IsosurfaceCalculator::set_crystal)
      .addFunction("set_parameters", &IsosurfaceCalculator::set_parameters)
      .addFunction("validate", &IsosurfaceCalculator::validate)
      .addFunction("compute", &IsosurfaceCalculator::compute)
      .addProperty("isosurface", &IsosurfaceCalculator::isosurface)
      .addProperty("requires_crystal", &IsosurfaceCalculator::requires_crystal)
      .addProperty("requires_wavefunction",
                   &IsosurfaceCalculator::requires_wavefunction)
      .addProperty("requires_environment",
                   &IsosurfaceCalculator::requires_environment)
      .addProperty("have_crystal", &IsosurfaceCalculator::have_crystal)
      .addProperty("have_wavefunction",
                   &IsosurfaceCalculator::have_wavefunction)
      .addProperty("have_environment", &IsosurfaceCalculator::have_environment)
      .addProperty("error_message", &IsosurfaceCalculator::error_message)
      .endClass()

      OCC_LUA_ENUM_NAMESPACE("VolumePropertyKind", OCC_ENUM_VolumePropertyKind)
      OCC_LUA_ENUM_NAMESPACE("SpinComponent", OCC_ENUM_SpinComponent)

      .beginClass<VolumeGenerationParameters>("VolumeGenerationParameters")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("property", &VolumeGenerationParameters::property)
      .addPropertyReadWrite("spin", &VolumeGenerationParameters::spin)
      .addPropertyReadWrite("functional",
                            &VolumeGenerationParameters::functional)
      .addPropertyReadWrite("mo_number", &VolumeGenerationParameters::mo_number)
      .addPropertyReadWrite("value_threshold",
                            &VolumeGenerationParameters::value_threshold)
      .addPropertyReadWrite("buffer_distance",
                            &VolumeGenerationParameters::buffer_distance)
      .addPropertyReadWrite("crystal_filename",
                            &VolumeGenerationParameters::crystal_filename)
      .addPropertyReadWrite("adaptive_bounds",
                            &VolumeGenerationParameters::adaptive_bounds)
      .endClass()

      .beginClass<VolumeData>("VolumeData")
      .addProperty("name", &VolumeData::name)
      .addProperty("property", &VolumeData::property)
      .addProperty("nx", &VolumeData::nx)
      .addProperty("ny", &VolumeData::ny)
      .addProperty("nz", &VolumeData::nz)
      .addProperty("total_points", &VolumeData::total_points)
      // Flattened (k-fastest) data for downstream reshape.
      .addFunction(
          "get_data",
          +[](const VolumeData *v, lua_State *S) {
            lb::LuaRef out = lb::newTable(S);
            int idx = 1;
            for (int i = 0; i < v->nx(); ++i) {
              for (int j = 0; j < v->ny(); ++j) {
                for (int k = 0; k < v->nz(); ++k) {
                  out[idx++] = v->data(i, j, k);
                }
              }
            }
            return out;
          })
      .endClass()

      .beginClass<VolumeCalculator>("VolumeCalculator")
      .addConstructor<void (*)()>()
      .addFunction("set_wavefunction", &VolumeCalculator::set_wavefunction)
      .addFunction("set_molecule", &VolumeCalculator::set_molecule)
      .addFunction("compute_volume", &VolumeCalculator::compute_volume)
      .addFunction("volume_as_cube_string",
                   &VolumeCalculator::volume_as_cube_string)
      .endClass()

      // Convenience generators
      .addFunction(
          "generate_electron_density_cube",
          +[](const occ::qm::Wavefunction &wfn, int nx, int ny, int nz) {
            VolumeCalculator calc;
            calc.set_wavefunction(wfn);
            VolumeGenerationParameters params;
            params.property = VolumePropertyKind::ElectronDensity;
            params.steps = {nx, ny, nz};
            return calc.volume_as_cube_string(calc.compute_volume(params));
          })
      .addFunction(
          "generate_esp_cube",
          +[](const occ::qm::Wavefunction &wfn, int nx, int ny, int nz) {
            VolumeCalculator calc;
            calc.set_wavefunction(wfn);
            VolumeGenerationParameters params;
            params.property = VolumePropertyKind::ElectricPotential;
            params.steps = {nx, ny, nz};
            return calc.volume_as_cube_string(calc.compute_volume(params));
          })

      .endNamespace();
}

} // namespace occ::lua_bindings
