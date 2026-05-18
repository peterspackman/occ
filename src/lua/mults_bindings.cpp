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

namespace sol {
template <>
struct is_automagical<occ::mults::CrystalEnergy> : std::false_type {};
template <>
struct is_automagical<occ::mults::CrystalEnergySetup> : std::false_type {};
template <>
struct is_automagical<occ::mults::CrystalEnergyResult> : std::false_type {};
template <>
struct is_automagical<occ::mults::CrystalOptimizer> : std::false_type {};
template <>
struct is_automagical<occ::mults::CrystalOptimizerResult> : std::false_type {};
template <>
struct is_automagical<occ::mults::RigidMolecule> : std::false_type {};
template <>
struct is_automagical<occ::mults::MoleculeState> : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using namespace occ::mults;

void register_mults_bindings(sol::state_view, sol::table &m) {
  m.new_enum<ForceFieldType>(
      "ForceFieldType", {{"None_", ForceFieldType::None},
                          {"LennardJones", ForceFieldType::LennardJones},
                          {"BuckinghamDE", ForceFieldType::BuckinghamDE},
                          {"Custom", ForceFieldType::Custom}});

  m.new_enum<OptimizationMethod>(
      "OptimizationMethod",
      {{"MSTMIN", OptimizationMethod::MSTMIN},
       {"LBFGS", OptimizationMethod::LBFGS},
       {"TrustRegion", OptimizationMethod::TrustRegion}});

  m.new_usertype<MoleculeState>(
      "MoleculeState",
      sol::call_constructor, sol::factories([]() { return MoleculeState{}; }),
      "position",
      sol::property(
          [](const MoleculeState &s, sol::this_state st) {
            return vec_to_table(st, s.position);
          },
          [](MoleculeState &s, const sol::table &t) {
            s.position = table_to_vec3(t);
          }),
      "angle_axis",
      sol::property(
          [](const MoleculeState &s, sol::this_state st) {
            return vec_to_table(st, s.angle_axis);
          },
          [](MoleculeState &s, const sol::table &t) {
            s.angle_axis = table_to_vec3(t);
          }),
      "parity", &MoleculeState::parity,
      "rotation_matrix",
      [](const MoleculeState &s, sol::this_state st) {
        return mat_to_table(st, s.rotation_matrix());
      },
      sol::meta_function::to_string, [](const MoleculeState &s) {
        return fmt::format(
            "<MoleculeState pos=({:.3f},{:.3f},{:.3f}) parity={}>",
            s.position.x(), s.position.y(), s.position.z(), s.parity);
      });

  m.new_usertype<CrystalEnergyResult>(
      "CrystalEnergyResult", sol::no_constructor,
      "total_energy", sol::readonly(&CrystalEnergyResult::total_energy),
      "electrostatic_energy",
      sol::readonly(&CrystalEnergyResult::electrostatic_energy),
      "repulsion_dispersion",
      sol::readonly(&CrystalEnergyResult::repulsion_dispersion),
      // forces/torques are vector<Vec3>; convert each Vec3 element into a
      // 3-table and return the whole thing as a nested Lua table.
      "forces",
      sol::readonly_property(
          [](const CrystalEnergyResult &r, sol::this_state s) {
            sol::state_view lua(s);
            sol::table out =
                lua.create_table(static_cast<int>(r.forces.size()), 0);
            for (size_t i = 0; i < r.forces.size(); ++i) {
              out[i + 1] = vec_to_table(s, r.forces[i]);
            }
            return out;
          }),
      "torques",
      sol::readonly_property(
          [](const CrystalEnergyResult &r, sol::this_state s) {
            sol::state_view lua(s);
            sol::table out =
                lua.create_table(static_cast<int>(r.torques.size()), 0);
            for (size_t i = 0; i < r.torques.size(); ++i) {
              out[i + 1] = vec_to_table(s, r.torques[i]);
            }
            return out;
          }),
      sol::meta_function::to_string, [](const CrystalEnergyResult &r) {
        return fmt::format(
            "<CrystalEnergyResult E={:.4f} elec={:.4f} sr={:.4f}>",
            r.total_energy, r.electrostatic_energy, r.repulsion_dispersion);
      });

  m.new_usertype<CrystalEnergySetup>(
      "CrystalEnergySetup",
      sol::call_constructor,
      sol::factories([]() { return CrystalEnergySetup{}; }),
      "cutoff_radius", &CrystalEnergySetup::cutoff_radius,
      "use_ewald", &CrystalEnergySetup::use_ewald,
      "ewald_accuracy", &CrystalEnergySetup::ewald_accuracy,
      "max_interaction_order", &CrystalEnergySetup::max_interaction_order);

  m.new_usertype<CrystalEnergy>(
      "CrystalEnergy",
      sol::call_constructor, sol::constructors<CrystalEnergy(CrystalEnergySetup)>(),
      "compute", &CrystalEnergy::compute,
      "compute_energy", &CrystalEnergy::compute_energy,
      "initial_states",
      [](const CrystalEnergy &e) { return sol::as_table(e.initial_states()); },
      "num_molecules", &CrystalEnergy::num_molecules,
      "num_sites", &CrystalEnergy::num_sites,
      sol::meta_function::to_string, [](const CrystalEnergy &e) {
        return fmt::format("<CrystalEnergy mols={} sites={} pairs={}>",
                           e.num_molecules(), e.num_sites(),
                           e.neighbor_pairs().size());
      });

  m.new_usertype<CrystalOptimizerSettings>(
      "CrystalOptimizerSettings",
      sol::call_constructor,
      sol::factories([]() { return CrystalOptimizerSettings{}; }),
      "method", &CrystalOptimizerSettings::method,
      "gradient_tolerance", &CrystalOptimizerSettings::gradient_tolerance,
      "energy_tolerance", &CrystalOptimizerSettings::energy_tolerance,
      "max_iterations", &CrystalOptimizerSettings::max_iterations,
      "neighbor_radius", &CrystalOptimizerSettings::neighbor_radius,
      "force_field", &CrystalOptimizerSettings::force_field,
      "optimize_cell", &CrystalOptimizerSettings::optimize_cell,
      "use_ewald", &CrystalOptimizerSettings::use_ewald,
      "max_interaction_order", &CrystalOptimizerSettings::max_interaction_order,
      "external_pressure_gpa",
      &CrystalOptimizerSettings::external_pressure_gpa);

  m.new_usertype<CrystalOptimizerResult>(
      "CrystalOptimizerResult", sol::no_constructor,
      "final_energy", sol::readonly(&CrystalOptimizerResult::final_energy),
      "electrostatic_energy",
      sol::readonly(&CrystalOptimizerResult::electrostatic_energy),
      "repulsion_dispersion_energy",
      sol::readonly(&CrystalOptimizerResult::repulsion_dispersion_energy),
      "initial_energy",
      sol::readonly(&CrystalOptimizerResult::initial_energy),
      "iterations", sol::readonly(&CrystalOptimizerResult::iterations),
      "function_evaluations",
      sol::readonly(&CrystalOptimizerResult::function_evaluations),
      "converged", sol::readonly(&CrystalOptimizerResult::converged),
      "termination_reason",
      sol::readonly(&CrystalOptimizerResult::termination_reason),
      "final_states",
      sol::readonly_property([](const CrystalOptimizerResult &r) {
        return sol::as_table(r.final_states);
      }),
      sol::meta_function::to_string, [](const CrystalOptimizerResult &r) {
        return fmt::format(
            "<CrystalOptimizerResult E={:.4f} converged={} iter={}>",
            r.final_energy, r.converged, r.iterations);
      });

  m.new_usertype<CrystalOptimizer>(
      "CrystalOptimizer",
      sol::call_constructor,
      sol::factories(
          [](CrystalEnergySetup setup) {
            return CrystalOptimizer(std::move(setup),
                                     CrystalOptimizerSettings{});
          },
          [](CrystalEnergySetup setup, const CrystalOptimizerSettings &s) {
            return CrystalOptimizer(std::move(setup), s);
          }),
      "optimize", [](CrystalOptimizer &o) { return o.optimize(); },
      "states", [](const CrystalOptimizer &o) {
        return sol::as_table(o.states());
      },
      sol::meta_function::to_string, [](const CrystalOptimizer &o) {
        return fmt::format("<CrystalOptimizer mols={} params={}>",
                           o.energy_calculator().num_molecules(),
                           o.num_parameters());
      });

  m.new_usertype<MultipoleConfig>(
      "MultipoleConfig",
      sol::call_constructor,
      sol::factories([]() { return MultipoleConfig{}; }),
      "method", &MultipoleConfig::method,
      "basis_set", &MultipoleConfig::basis_set,
      "basename", &MultipoleConfig::basename,
      "max_rank", &MultipoleConfig::max_rank);

  // `from_crystal` runs an SCF + DMA and mutates the crystal in places
  // (caches data on the asymmetric unit), so it takes a non-const Crystal.
  m.set_function("from_crystal",
                 [](occ::crystal::Crystal &c,
                    sol::optional<MultipoleConfig> cfg) {
                   return from_crystal(c, cfg.value_or(MultipoleConfig{}));
                 });

  // High-level convenience: load JSON → compute energy in one call.
  m.set_function("compute_crystal_energy", [](const std::string &json_path) {
    auto si = occ::io::read_structure_json(json_path);
    auto setup = from_structure_input(si);
    CrystalEnergy calc(std::move(setup));
    auto states = calc.initial_states();
    return calc.compute(states);
  });
}

} // namespace occ::lua_bindings
