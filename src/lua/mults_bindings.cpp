#include "mults_bindings.h"
#include "eigen_conv.h"
#include "enum_stacks.h"
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

      // Enums round-trip through luabridge::Stack<E> (see enum_stacks.h),
      // so values pushed here decay to lua_Integer and the binding side
      // decodes them back to the enum on the way in. Value lists live in
      // enum_defs.h (shared with the Stack specializations).
      OCC_LUA_ENUM_NAMESPACE("ForceFieldType", OCC_ENUM_ForceFieldType)
      OCC_LUA_ENUM_NAMESPACE("OptimizationMethod", OCC_ENUM_OptimizationMethod)

      .beginClass<MoleculeState>("MoleculeState")
      .addConstructor<void (*)()>()
      .addProperty(
          "position",
          +[](const MoleculeState *s) -> occ::Vec3 { return s->position; },
          +[](MoleculeState *s, const lb::LuaRef &t) {
            s->position = table_to_vec3(t);
          })
      .addProperty(
          "angle_axis",
          +[](const MoleculeState *s) -> occ::Vec3 { return s->angle_axis; },
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
      .addPropertyReadWrite("force_field", &CrystalEnergySetup::force_field)
      .addPropertyReadWrite("cutoff_radius", &CrystalEnergySetup::cutoff_radius)
      .addPropertyReadWrite("use_ewald", &CrystalEnergySetup::use_ewald)
      .addPropertyReadWrite("ewald_accuracy",
                            &CrystalEnergySetup::ewald_accuracy)
      .addPropertyReadWrite("ewald_eta", &CrystalEnergySetup::ewald_eta)
      .addPropertyReadWrite("ewald_kmax", &CrystalEnergySetup::ewald_kmax)
      .addPropertyReadWrite("max_interaction_order",
                            &CrystalEnergySetup::max_interaction_order)
      .addPropertyReadWrite("taper_on", &CrystalEnergySetup::taper_on)
      .addPropertyReadWrite("taper_off", &CrystalEnergySetup::taper_off)
      .addPropertyReadWrite("taper_order", &CrystalEnergySetup::taper_order)
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
      // Engine / DOF
      .addPropertyReadWrite("use_cartesian_engine",
                            &CrystalOptimizerSettings::use_cartesian_engine)
      .addPropertyReadWrite("use_symmetry",
                            &CrystalOptimizerSettings::use_symmetry)
      .addPropertyReadWrite("constrain_cell_strain_by_lattice",
                            &CrystalOptimizerSettings::constrain_cell_strain_by_lattice)
      .addPropertyReadWrite("fix_first_translation",
                            &CrystalOptimizerSettings::fix_first_translation)
      .addPropertyReadWrite("fix_first_rotation",
                            &CrystalOptimizerSettings::fix_first_rotation)
      // L-BFGS
      .addPropertyReadWrite("lbfgs_memory",
                            &CrystalOptimizerSettings::lbfgs_memory)
      // MSTMIN tuning
      .addPropertyReadWrite("max_displacement",
                            &CrystalOptimizerSettings::max_displacement)
      .addPropertyReadWrite("mst_step_tolerance",
                            &CrystalOptimizerSettings::mst_step_tolerance)
      .addPropertyReadWrite("mst_rotation_scale",
                            &CrystalOptimizerSettings::mst_rotation_scale)
      .addPropertyReadWrite("mst_cell_scale",
                            &CrystalOptimizerSettings::mst_cell_scale)
      .addPropertyReadWrite("max_hessian_updates",
                            &CrystalOptimizerSettings::max_hessian_updates)
      .addPropertyReadWrite("mst_max_line_search",
                            &CrystalOptimizerSettings::mst_max_line_search)
      .addPropertyReadWrite("mst_max_line_search_restarts",
                            &CrystalOptimizerSettings::mst_max_line_search_restarts)
      .addPropertyReadWrite("mst_max_function_evaluations",
                            &CrystalOptimizerSettings::mst_max_function_evaluations)
      .addPropertyReadWrite("mst_line_search_report_interval",
                            &CrystalOptimizerSettings::mst_line_search_report_interval)
      // Trust region
      .addPropertyReadWrite("trust_region_radius",
                            &CrystalOptimizerSettings::trust_region_radius)
      .addPropertyReadWrite("hessian_update_interval",
                            &CrystalOptimizerSettings::hessian_update_interval)
      .addPropertyReadWrite("require_exact_hessian",
                            &CrystalOptimizerSettings::require_exact_hessian)
      // I/O
      .addPropertyReadWrite("trajectory_file",
                            &CrystalOptimizerSettings::trajectory_file)
      // Ewald
      .addPropertyReadWrite("ewald_accuracy",
                            &CrystalOptimizerSettings::ewald_accuracy)
      .addPropertyReadWrite("ewald_eta", &CrystalOptimizerSettings::ewald_eta)
      .addPropertyReadWrite("ewald_kmax",
                            &CrystalOptimizerSettings::ewald_kmax)
      // Adaptive neighbor list
      .addPropertyReadWrite("adaptive_neighbor_rebuild",
                            &CrystalOptimizerSettings::adaptive_neighbor_rebuild)
      .addPropertyReadWrite("neighbor_rebuild_displacement",
                            &CrystalOptimizerSettings::neighbor_rebuild_displacement)
      .addPropertyReadWrite("neighbor_rebuild_rotation",
                            &CrystalOptimizerSettings::neighbor_rebuild_rotation)
      .addPropertyReadWrite("neighbor_rebuild_cell_strain",
                            &CrystalOptimizerSettings::neighbor_rebuild_cell_strain)
      .addPropertyReadWrite("neighbor_rebuild_interval",
                            &CrystalOptimizerSettings::neighbor_rebuild_interval)
      .addPropertyReadWrite("freeze_neighbors_during_linesearch",
                            &CrystalOptimizerSettings::freeze_neighbors_during_linesearch)
      .endClass()

      .beginClass<CrystalOptimizerResult>("CrystalOptimizerResult")
      .addProperty("final_energy", &CrystalOptimizerResult::final_energy)
      .addProperty("electrostatic_energy",
                   &CrystalOptimizerResult::electrostatic_energy)
      .addProperty("repulsion_dispersion_energy",
                   &CrystalOptimizerResult::repulsion_dispersion_energy)
      .addProperty("pressure_volume_energy",
                   &CrystalOptimizerResult::pressure_volume_energy)
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
