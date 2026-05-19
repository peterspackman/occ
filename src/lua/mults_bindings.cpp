#include "mults_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/crystal/crystal.h>
#include <occ/io/structure_format.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_energy_setup.h>
#include <occ/mults/crystal_optimizer.h>
#include <occ/mults/multipole_source.h>
#include <occ/mults/rigid_molecule.h>

namespace occ::lua_bindings {

using namespace occ::mults;
namespace lb = luabridge;

namespace {
// Convert vector<Vec3> → Lua table of 3-element tables, used for
// forces/torques etc.
lb::LuaRef vec3_list_to_table(lua_State *L, const std::vector<Vec3> &xs) {
  lb::LuaRef out = lb::newTable(L);
  for (size_t i = 0; i < xs.size(); ++i) {
    out[static_cast<int>(i + 1)] = vec_to_table(L, xs[i]);
  }
  return out;
}
} // namespace

void register_mults_bindings(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      // Enums — LuaBridge3 doesn't have a dedicated enum binding;
      // expose the values as a plain Lua table.
      .beginNamespace("ForceFieldType")
      .addProperty(
          "None_", +[]() { return static_cast<int>(ForceFieldType::None); })
      .addProperty(
          "LennardJones",
          +[]() { return static_cast<int>(ForceFieldType::LennardJones); })
      .addProperty(
          "BuckinghamDE",
          +[]() { return static_cast<int>(ForceFieldType::BuckinghamDE); })
      .addProperty(
          "Custom", +[]() { return static_cast<int>(ForceFieldType::Custom); })
      .endNamespace()

      .beginNamespace("OptimizationMethod")
      .addProperty(
          "MSTMIN",
          +[]() { return static_cast<int>(OptimizationMethod::MSTMIN); })
      .addProperty(
          "LBFGS",
          +[]() { return static_cast<int>(OptimizationMethod::LBFGS); })
      .addProperty(
          "TrustRegion",
          +[]() { return static_cast<int>(OptimizationMethod::TrustRegion); })
      .endNamespace()

      .beginClass<MoleculeState>("MoleculeState")
      .addConstructor<void (*)()>()
      .addProperty(
          "get_position",
          +[](const MoleculeState *s) -> occ::Vec3 { return s->position; })
      .addFunction(
          "set_position",
          +[](MoleculeState *s, const lb::LuaRef &t) {
            s->position = table_to_vec3(t);
          })
      .addProperty(
          "get_angle_axis",
          +[](const MoleculeState *s) -> occ::Vec3 { return s->angle_axis; })
      .addFunction(
          "set_angle_axis",
          +[](MoleculeState *s, const lb::LuaRef &t) {
            s->angle_axis = table_to_vec3(t);
          })
      .addPropertyReadWrite("parity", &MoleculeState::parity)
      .addProperty(
          "rotation_matrix",
          +[](const MoleculeState *s) -> occ::Mat3 {
            return s->rotation_matrix();
          })
      .addFunction(
          "__tostring",
          +[](const MoleculeState *s) {
            return fmt::format("<MoleculeState pos=({:.3f},{:.3f},{:.3f}) "
                               "parity={}>",
                               s->position.x(), s->position.y(),
                               s->position.z(), s->parity);
          })
      .endClass()

      .beginClass<CrystalEnergyResult>("CrystalEnergyResult")
      .addProperty("total_energy", &CrystalEnergyResult::total_energy)
      .addProperty("electrostatic_energy",
                   &CrystalEnergyResult::electrostatic_energy)
      .addProperty("repulsion_dispersion",
                   &CrystalEnergyResult::repulsion_dispersion)
      .addFunction(
          "forces",
          +[](const CrystalEnergyResult *r, lua_State *S) {
            return vec3_list_to_table(S, r->forces);
          })
      .addFunction(
          "torques",
          +[](const CrystalEnergyResult *r, lua_State *S) {
            return vec3_list_to_table(S, r->torques);
          })
      .addFunction(
          "__tostring",
          +[](const CrystalEnergyResult *r) {
            return fmt::format("<CrystalEnergyResult E={:.4f} elec={:.4f} "
                               "sr={:.4f}>",
                               r->total_energy, r->electrostatic_energy,
                               r->repulsion_dispersion);
          })
      .endClass()

      .beginClass<CrystalEnergySetup>("CrystalEnergySetup")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("cutoff_radius", &CrystalEnergySetup::cutoff_radius)
      .addPropertyReadWrite("use_ewald", &CrystalEnergySetup::use_ewald)
      .addPropertyReadWrite("ewald_accuracy",
                            &CrystalEnergySetup::ewald_accuracy)
      .addPropertyReadWrite("max_interaction_order",
                            &CrystalEnergySetup::max_interaction_order)
      .endClass()

      .beginClass<CrystalEnergy>("CrystalEnergy")
      .addConstructor<void (*)(CrystalEnergySetup)>()
      .addFunction("compute", &CrystalEnergy::compute)
      .addFunction("compute_energy", &CrystalEnergy::compute_energy)
      .addFunction(
          "initial_states",
          +[](const CrystalEnergy *e, lua_State *S) {
            lb::LuaRef out = lb::newTable(S);
            auto v = e->initial_states();
            for (size_t i = 0; i < v.size(); ++i) {
              out[static_cast<int>(i + 1)] = v[i];
            }
            return out;
          })
      .addProperty("num_molecules", &CrystalEnergy::num_molecules)
      .addProperty("num_sites", &CrystalEnergy::num_sites)
      .addFunction(
          "__tostring",
          +[](const CrystalEnergy *e) {
            return fmt::format("<CrystalEnergy mols={} sites={} pairs={}>",
                               e->num_molecules(), e->num_sites(),
                               e->neighbor_pairs().size());
          })
      .endClass()

      .beginClass<CrystalOptimizerSettings>("CrystalOptimizerSettings")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("method", &CrystalOptimizerSettings::method)
      .addPropertyReadWrite("gradient_tolerance",
                            &CrystalOptimizerSettings::gradient_tolerance)
      .addPropertyReadWrite("energy_tolerance",
                            &CrystalOptimizerSettings::energy_tolerance)
      .addPropertyReadWrite("max_iterations",
                            &CrystalOptimizerSettings::max_iterations)
      .addPropertyReadWrite("neighbor_radius",
                            &CrystalOptimizerSettings::neighbor_radius)
      .addPropertyReadWrite("force_field",
                            &CrystalOptimizerSettings::force_field)
      .addPropertyReadWrite("optimize_cell",
                            &CrystalOptimizerSettings::optimize_cell)
      .addPropertyReadWrite("use_ewald", &CrystalOptimizerSettings::use_ewald)
      .addPropertyReadWrite("max_interaction_order",
                            &CrystalOptimizerSettings::max_interaction_order)
      .addPropertyReadWrite("external_pressure_gpa",
                            &CrystalOptimizerSettings::external_pressure_gpa)
      .endClass()

      .beginClass<CrystalOptimizerResult>("CrystalOptimizerResult")
      .addProperty("final_energy", &CrystalOptimizerResult::final_energy)
      .addProperty("electrostatic_energy",
                   &CrystalOptimizerResult::electrostatic_energy)
      .addProperty("repulsion_dispersion_energy",
                   &CrystalOptimizerResult::repulsion_dispersion_energy)
      .addProperty("initial_energy", &CrystalOptimizerResult::initial_energy)
      .addProperty("iterations", &CrystalOptimizerResult::iterations)
      .addProperty("function_evaluations",
                   &CrystalOptimizerResult::function_evaluations)
      .addProperty("converged", &CrystalOptimizerResult::converged)
      .addProperty("termination_reason",
                   &CrystalOptimizerResult::termination_reason)
      .addFunction(
          "final_states",
          +[](const CrystalOptimizerResult *r, lua_State *S) {
            lb::LuaRef out = lb::newTable(S);
            for (size_t i = 0; i < r->final_states.size(); ++i) {
              out[static_cast<int>(i + 1)] = r->final_states[i];
            }
            return out;
          })
      .addFunction(
          "__tostring",
          +[](const CrystalOptimizerResult *r) {
            return fmt::format(
                "<CrystalOptimizerResult E={:.4f} converged={} iter={}>",
                r->final_energy, r->converged, r->iterations);
          })
      .endClass()

      .beginClass<CrystalOptimizer>("CrystalOptimizer")
      // Two-arg constructor matches the sol2 factory pair; for
      // settings-default callers we expose a `with_default_settings`
      // helper rather than overloading.
      .addConstructor<void (*)(CrystalEnergySetup,
                               const CrystalOptimizerSettings &)>()
      .addStaticFunction(
          "with_default_settings",
          +[](CrystalEnergySetup setup) {
            return new CrystalOptimizer(std::move(setup),
                                        CrystalOptimizerSettings{});
          })
      .addFunction(
          "optimize", +[](CrystalOptimizer *o) { return o->optimize(); })
      .addFunction(
          "states",
          +[](const CrystalOptimizer *o, lua_State *S) {
            lb::LuaRef out = lb::newTable(S);
            auto sts = o->states();
            for (size_t i = 0; i < sts.size(); ++i) {
              out[static_cast<int>(i + 1)] = sts[i];
            }
            return out;
          })
      .addFunction(
          "__tostring",
          +[](const CrystalOptimizer *o) {
            return fmt::format("<CrystalOptimizer mols={} params={}>",
                               o->energy_calculator().num_molecules(),
                               o->num_parameters());
          })
      .endClass()

      .beginClass<MultipoleConfig>("MultipoleConfig")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("method", &MultipoleConfig::method)
      .addPropertyReadWrite("basis_set", &MultipoleConfig::basis_set)
      .addPropertyReadWrite("basename", &MultipoleConfig::basename)
      .addPropertyReadWrite("max_rank", &MultipoleConfig::max_rank)
      .endClass()

      // `from_crystal` runs an SCF + DMA and mutates the crystal in
      // place (caches data on the asymmetric unit), so it takes a
      // non-const Crystal. sol2's `optional` default-arg pattern is
      // expressed via two function names here.
      .addFunction(
          "from_crystal",
          +[](occ::crystal::Crystal *c, const MultipoleConfig &cfg) {
            return from_crystal(*c, cfg);
          })
      .addFunction(
          "from_crystal_default",
          +[](occ::crystal::Crystal *c) {
            return from_crystal(*c, MultipoleConfig{});
          })

      .addFunction(
          "compute_crystal_energy",
          +[](const std::string &json_path) {
            auto si = occ::io::read_structure_json(json_path);
            auto setup = from_structure_input(si);
            CrystalEnergy calc(std::move(setup));
            auto states = calc.initial_states();
            return calc.compute(states);
          })

      .endNamespace();
}

} // namespace occ::lua_bindings
