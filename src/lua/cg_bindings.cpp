#include "cg_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/cg/interaction_mapper.h>
#include <occ/interaction/pair_energy.h>
#include <occ/main/occ_cg.h>

namespace sol {
template <>
struct is_automagical<occ::cg::DimerResult> : std::false_type {};
template <>
struct is_automagical<occ::cg::MoleculeResult> : std::false_type {};
template <>
struct is_automagical<occ::cg::CrystalGrowthResult> : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using occ::cg::CrystalGrowthResult;
using occ::cg::DimerResult;
using occ::cg::DimerSolventTerm;
using occ::cg::InteractionMapper;
using occ::cg::MoleculeResult;
using occ::interaction::LatticeConvergenceSettings;
using occ::main::CGConfig;

namespace {

// Convert a small string->double map to a Lua table. Used for
// `energy_components`; sol2 doesn't auto-convert ankerl maps.
template <typename Map>
sol::table map_to_table(sol::this_state s, const Map &map) {
  sol::state_view lua(s);
  sol::table t = lua.create_table(0, static_cast<int>(map.size()));
  for (const auto &[k, v] : map) {
    t[k] = v;
  }
  return t;
}

} // namespace

void register_cg_bindings(sol::state_view, sol::table &m) {
  m.new_usertype<LatticeConvergenceSettings>(
      "LatticeConvergenceSettings",
      sol::call_constructor,
      sol::factories([]() { return LatticeConvergenceSettings{}; }),
      "min_radius", &LatticeConvergenceSettings::min_radius,
      "max_radius", &LatticeConvergenceSettings::max_radius,
      "radius_increment", &LatticeConvergenceSettings::radius_increment,
      "energy_tolerance", &LatticeConvergenceSettings::energy_tolerance,
      "wolf_sum", &LatticeConvergenceSettings::wolf_sum,
      "crystal_field_polarization",
      &LatticeConvergenceSettings::crystal_field_polarization,
      "model_name", &LatticeConvergenceSettings::model_name,
      "crystal_filename", &LatticeConvergenceSettings::crystal_filename,
      "output_json_filename",
      &LatticeConvergenceSettings::output_json_filename);

  m.new_usertype<CGConfig>(
      "CrystalGrowthConfig",
      sol::call_constructor, sol::factories([]() { return CGConfig{}; }),
      "lattice_settings", &CGConfig::lattice_settings,
      "cg_radius", &CGConfig::cg_radius,
      "solvent", &CGConfig::solvent,
      "wavefunction_choice", &CGConfig::wavefunction_choice,
      "num_surface_energies", &CGConfig::max_facets);

  m.new_usertype<DimerSolventTerm>(
      "DimerSolventTerm", sol::no_constructor,
      "ab", &DimerSolventTerm::ab,
      "ba", &DimerSolventTerm::ba,
      "total", &DimerSolventTerm::total);

  m.new_usertype<DimerResult>(
      "DimerResult",
      sol::call_constructor,
      sol::factories([](occ::core::Dimer &d, bool is_nn, int idx) {
        return DimerResult(d, is_nn, idx);
      }),
      "dimer",
      sol::readonly_property(
          [](const DimerResult &d) -> const occ::core::Dimer & {
            return d.dimer;
          }),
      "unique_idx", sol::readonly(&DimerResult::unique_idx),
      "set_energy_component", &DimerResult::set_energy_component,
      "total_energy", &DimerResult::total_energy,
      "energy_component", &DimerResult::energy_component,
      "energy_components",
      [](const DimerResult &d, sol::this_state s) {
        return map_to_table(s, d.energy_components);
      },
      "is_nearest_neighbor", sol::readonly(&DimerResult::is_nearest_neighbor));

  m.new_usertype<MoleculeResult>(
      "MoleculeResult", sol::no_constructor,
      "dimer_results",
      sol::readonly_property([](const MoleculeResult &r) {
        return sol::as_table(r.dimer_results);
      }),
      "total", sol::readonly(&MoleculeResult::total),
      "has_inversion_symmetry",
      sol::readonly(&MoleculeResult::has_inversion_symmetry),
      "total_energy", &MoleculeResult::total_energy,
      "energy_components",
      [](const MoleculeResult &r, sol::this_state s) {
        return map_to_table(s, r.energy_components);
      },
      "energy_component", &MoleculeResult::energy_component);

  m.new_usertype<CrystalGrowthResult>(
      "CrystalGrowthResult", sol::no_constructor,
      "molecule_results",
      sol::readonly_property([](const CrystalGrowthResult &r) {
        return sol::as_table(r.molecule_results);
      }));

  m.new_usertype<occ::cg::EnergyTotal>(
      "CrystalGrowthEnergyTotal", sol::no_constructor,
      "crystal", sol::readonly(&occ::cg::EnergyTotal::crystal_energy),
      "int_", sol::readonly(&occ::cg::EnergyTotal::interaction_energy),
      "solution", sol::readonly(&occ::cg::EnergyTotal::solution_term),
      sol::meta_function::to_string, [](const occ::cg::EnergyTotal &t) {
        return fmt::format("(crys={:.6f}, int={:.6f}, sol={:.6f})",
                           t.crystal_energy, t.interaction_energy,
                           t.solution_term);
      });

  m.new_usertype<InteractionMapper>(
      "InteractionMapper",
      sol::call_constructor,
      sol::constructors<InteractionMapper(
          const occ::crystal::Crystal &,
          const occ::crystal::CrystalDimers &,
          occ::crystal::CrystalDimers &, bool)>(),
      "map_interactions", &InteractionMapper::map_interactions);

  m.set_function("calculate_crystal_growth_energies",
                 [](const occ::main::CGConfig &config) {
                   return occ::main::run_cg(config);
                 });
}

} // namespace occ::lua_bindings
