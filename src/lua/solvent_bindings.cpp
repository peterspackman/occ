#include "solvent_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/core/units.h>
#include <occ/driver/cosmors_driver.h>

namespace occ::lua_bindings {

using occ::driver::CosmoRSSolvation;
using occ::driver::CosmoRSSolvationSettings;
using occ::solvent::cosmors::SolvationEnergy;
namespace lb = luabridge;

void register_solvent_bindings(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      .beginClass<CosmoRSSolvationSettings>("CosmoRSSettings")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("method", &CosmoRSSolvationSettings::method)
      .addPropertyReadWrite("basis", &CosmoRSSolvationSettings::basis)
      .addPropertyReadWrite("pure_spherical",
                            &CosmoRSSolvationSettings::pure_spherical)
      .addPropertyReadWrite("probe_radius",
                            &CosmoRSSolvationSettings::probe_radius_angs)
      .addPropertyReadWrite("angular_points",
                            &CosmoRSSolvationSettings::angular_points)
      .addPropertyReadWrite("constrain_charge",
                            &CosmoRSSolvationSettings::constrain_charge)
      .addPropertyReadWrite("temperature",
                            &CosmoRSSolvationSettings::temperature)
      .addPropertyReadWrite("liquid_volume",
                            &CosmoRSSolvationSettings::liquid_volume)
      .addPropertyReadWrite("num_rings", &CosmoRSSolvationSettings::num_rings)
      .endClass()

      .beginClass<SolvationEnergy>("CosmoRSEnergy")
      .addProperty("dielectric", &SolvationEnergy::dielectric)
      .addProperty("residual", &SolvationEnergy::residual)
      .addProperty("combinatorial", &SolvationEnergy::combinatorial)
      .addProperty("vdw", &SolvationEnergy::vdw)
      .addProperty("ring", &SolvationEnergy::ring)
      .addProperty("reference_state", &SolvationEnergy::reference_state)
      .addProperty("eta", &SolvationEnergy::eta)
      .addFunction("total", &SolvationEnergy::total)
      .addFunction(
          "__tostring",
          +[](const SolvationEnergy *e) {
            return fmt::format("<CosmoRSEnergy total={:.3f} kJ/mol>",
                               e->total() * occ::units::AU_TO_KJ_PER_MOL);
          })
      .endClass()

      .beginClass<CosmoRSSolvation>("CosmoRSSolvationResult")
      .addProperty("energy", &CosmoRSSolvation::energy)
      .addProperty("cavity_area", &CosmoRSSolvation::cavity_area)
      .addProperty("cavity_volume", &CosmoRSSolvation::cavity_volume)
      .addProperty("num_rings", &CosmoRSSolvation::num_rings)
      .addProperty("gas", &CosmoRSSolvation::gas)
      .addProperty("conductor", &CosmoRSSolvation::conductor)
      .addFunction("total", &CosmoRSSolvation::total)
      .endClass()

      .addFunction(
          "cosmo_rs_solvation_free_energy",
          +[](const occ::core::Molecule &solute, const std::string &solvent,
              const CosmoRSSolvationSettings &settings) {
            return occ::driver::cosmors_solvation_free_energy(solute, solvent,
                                                              settings);
          })
      .addFunction(
          "cosmo_rs_solvation_free_energy_with_solvent_geometry",
          +[](const occ::core::Molecule &solute,
              const occ::core::Molecule &solvent,
              const CosmoRSSolvationSettings &settings) {
            return occ::driver::cosmors_solvation_free_energy(solute, solvent,
                                                              settings);
          })
      .addFunction("available_cosmo_rs_solvents",
                   &occ::driver::available_cosmors_solvents)
      .endNamespace();
}

} // namespace occ::lua_bindings
