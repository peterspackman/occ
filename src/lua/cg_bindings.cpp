#include "cg_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/cg/interaction_mapper.h>
#include <occ/interaction/pair_energy.h>
#include <occ/main/occ_cg.h>

namespace occ::lua_bindings {

using occ::cg::CrystalGrowthResult;
using occ::cg::DimerResult;
using occ::cg::DimerSolventTerm;
using occ::cg::InteractionMapper;
using occ::cg::MoleculeResult;
using occ::interaction::LatticeConvergenceSettings;
using occ::main::CGConfig;
namespace lb = luabridge;

namespace {

// Convert a small string->double map to a Lua table. Used for
// `energy_components`; LuaBridge3 doesn't auto-convert ankerl maps.
template <typename Map> lb::LuaRef map_to_table(lua_State *L, const Map &map) {
  lb::LuaRef t = lb::newTable(L);
  for (const auto &[k, v] : map) {
    t[k] = v;
  }
  return t;
}

} // namespace

void register_cg_bindings(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      .beginClass<LatticeConvergenceSettings>("LatticeConvergenceSettings")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("min_radius",
                            &LatticeConvergenceSettings::min_radius)
      .addPropertyReadWrite("max_radius",
                            &LatticeConvergenceSettings::max_radius)
      .addPropertyReadWrite("radius_increment",
                            &LatticeConvergenceSettings::radius_increment)
      .addPropertyReadWrite("energy_tolerance",
                            &LatticeConvergenceSettings::energy_tolerance)
      .addPropertyReadWrite("wolf_sum", &LatticeConvergenceSettings::wolf_sum)
      .addPropertyReadWrite(
          "crystal_field_polarization",
          &LatticeConvergenceSettings::crystal_field_polarization)
      .addPropertyReadWrite("model_name",
                            &LatticeConvergenceSettings::model_name)
      .addPropertyReadWrite("crystal_filename",
                            &LatticeConvergenceSettings::crystal_filename)
      .addPropertyReadWrite("output_json_filename",
                            &LatticeConvergenceSettings::output_json_filename)
      .addPropertyReadWrite("write_all_pairs",
                            &LatticeConvergenceSettings::write_all_pairs)
      .addPropertyReadWrite("spherical_basis",
                            &LatticeConvergenceSettings::spherical_basis)
      .addPropertyReadWrite("charge_string",
                            &LatticeConvergenceSettings::charge_string)
      .addPropertyReadWrite("multiplicity_string",
                            &LatticeConvergenceSettings::multiplicity_string)
      .addPropertyReadWrite("external_command",
                            &LatticeConvergenceSettings::external_command)
      .addPropertyReadWrite("normalize_hydrogens",
                            &LatticeConvergenceSettings::normalize_hydrogens)
      .addPropertyReadWrite("run_elastic_fitting",
                            &LatticeConvergenceSettings::run_elastic_fitting)
      .addPropertyReadWrite("elastic_output_file",
                            &LatticeConvergenceSettings::elastic_output_file)
      .endClass()

      .beginClass<CGConfig>("CrystalGrowthConfig")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("lattice_settings", &CGConfig::lattice_settings)
      .addPropertyReadWrite("cg_radius", &CGConfig::cg_radius)
      .addPropertyReadWrite("solvent", &CGConfig::solvent)
      .addPropertyReadWrite("charge_string", &CGConfig::charge_string)
      .addPropertyReadWrite("wavefunction_choice",
                            &CGConfig::wavefunction_choice)
      // Keep `num_surface_energies` for the existing alias to max_facets
      // (matches the CLI's --surface-energies); also bind the C++ name
      // for callers using the model identifier.
      .addPropertyReadWrite("max_facets", &CGConfig::max_facets)
      .addPropertyReadWrite("num_surface_energies", &CGConfig::max_facets)
      .addPropertyReadWrite("write_dump_files", &CGConfig::write_dump_files)
      .addPropertyReadWrite("spherical", &CGConfig::spherical)
      .addPropertyReadWrite("write_kmcpp_file", &CGConfig::write_kmcpp_file)
      .addPropertyReadWrite("use_xtb", &CGConfig::use_xtb)
      .addPropertyReadWrite("dry_run", &CGConfig::dry_run)
      .addPropertyReadWrite("asymmetric_solvent_contribution",
                            &CGConfig::asymmetric_solvent_contribution)
      .addPropertyReadWrite("gamma_point_molecules",
                            &CGConfig::gamma_point_molecules)
      .addPropertyReadWrite("xtb_solvation_model",
                            &CGConfig::xtb_solvation_model)
      .addPropertyReadWrite("list_solvents", &CGConfig::list_solvents)
      .addPropertyReadWrite("crystal_is_atomic",
                            &CGConfig::crystal_is_atomic)
      .endClass()

      .beginClass<DimerSolventTerm>("DimerSolventTerm")
      .addProperty("ab", &DimerSolventTerm::ab)
      .addProperty("ba", &DimerSolventTerm::ba)
      .addProperty("total", &DimerSolventTerm::total)
      .endClass()

      .beginClass<DimerResult>("DimerResult")
      .addConstructor<void (*)(occ::core::Dimer &, bool, int)>()
      .addProperty(
          "dimer",
          +[](const DimerResult *d) -> const occ::core::Dimer & {
            return d->dimer;
          })
      .addProperty("unique_idx", &DimerResult::unique_idx)
      .addFunction("set_energy_component", &DimerResult::set_energy_component)
      .addProperty("total_energy", &DimerResult::total_energy)
      .addFunction("energy_component", &DimerResult::energy_component)
      .addFunction(
          "energy_components",
          +[](const DimerResult *d, lua_State *S) {
            return map_to_table(S, d->energy_components);
          })
      .addProperty("is_nearest_neighbor", &DimerResult::is_nearest_neighbor)
      .endClass()

      .beginClass<MoleculeResult>("MoleculeResult")
      .addFunction(
          "dimer_results",
          +[](const MoleculeResult *r, lua_State *S) {
            lb::LuaRef t = lb::newTable(S);
            for (size_t i = 0; i < r->dimer_results.size(); ++i) {
              t[static_cast<int>(i + 1)] = r->dimer_results[i];
            }
            return t;
          })
      .addProperty("total", &MoleculeResult::total)
      .addProperty("has_inversion_symmetry",
                   &MoleculeResult::has_inversion_symmetry)
      .addProperty("total_energy", &MoleculeResult::total_energy)
      .addFunction(
          "energy_components",
          +[](const MoleculeResult *r, lua_State *S) {
            return map_to_table(S, r->energy_components);
          })
      .addFunction("energy_component", &MoleculeResult::energy_component)
      .endClass()

      .beginClass<CrystalGrowthResult>("CrystalGrowthResult")
      .addFunction(
          "molecule_results",
          +[](const CrystalGrowthResult *r, lua_State *S) {
            lb::LuaRef t = lb::newTable(S);
            for (size_t i = 0; i < r->molecule_results.size(); ++i) {
              t[static_cast<int>(i + 1)] = r->molecule_results[i];
            }
            return t;
          })
      .endClass()

      .beginClass<occ::cg::EnergyTotal>("CrystalGrowthEnergyTotal")
      .addProperty("crystal", &occ::cg::EnergyTotal::crystal_energy)
      .addProperty("int_", &occ::cg::EnergyTotal::interaction_energy)
      .addProperty("solution", &occ::cg::EnergyTotal::solution_term)
      .addFunction(
          "__tostring",
          +[](const occ::cg::EnergyTotal *t) {
            return fmt::format("(crys={:.6f}, int={:.6f}, sol={:.6f})",
                               t->crystal_energy, t->interaction_energy,
                               t->solution_term);
          })
      .endClass()

      .beginClass<InteractionMapper>("InteractionMapper")
      .addConstructor<void (*)(const occ::crystal::Crystal &,
                               const occ::crystal::CrystalDimers &,
                               occ::crystal::CrystalDimers &, bool)>()
      .addFunction("map_interactions", &InteractionMapper::map_interactions)
      .endClass()

      .addFunction(
          "calculate_crystal_growth_energies",
          +[](const occ::main::CGConfig &config) {
            return occ::main::run_cg(config);
          })

      .endNamespace();
}

} // namespace occ::lua_bindings
