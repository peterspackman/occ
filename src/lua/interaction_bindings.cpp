#include "interaction_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/coulomb.h>
#include <occ/interaction/pair_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/wavefunction_transform.h>
#include <occ/interaction/wolf.h>

namespace sol {
template <>
struct is_automagical<occ::interaction::CEEnergyComponents>
    : std::false_type {};
template <>
struct is_automagical<occ::interaction::CEModelInteraction>
    : std::false_type {};
template <>
struct is_automagical<occ::interaction::CEEnergyModel> : std::false_type {};
template <>
struct is_automagical<occ::interaction::transform::WavefunctionTransformer>
    : std::false_type {};
template <>
struct is_automagical<occ::interaction::transform::TransformResult>
    : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using namespace occ::interaction;
using transform::TransformResult;
using transform::WavefunctionTransformer;

void register_interaction_bindings(sol::state_view, sol::table &m) {
  // Bind the parameterization look-up as a free function — `nb::class_`
  // with a single factory method is overkill.
  m.set_function("ce_model_from_string", &ce_model_from_string);

  m.new_usertype<CEEnergyComponents>(
      "CEEnergyComponents",
      sol::call_constructor,
      sol::factories([]() { return CEEnergyComponents{}; }),
      "coulomb", sol::readonly(&CEEnergyComponents::coulomb),
      "exchange", sol::readonly(&CEEnergyComponents::exchange),
      "repulsion", sol::readonly(&CEEnergyComponents::repulsion),
      "polarization", sol::readonly(&CEEnergyComponents::polarization),
      "dispersion", sol::readonly(&CEEnergyComponents::dispersion),
      "total", sol::readonly(&CEEnergyComponents::total),
      "exchange_repulsion",
      sol::readonly(&CEEnergyComponents::exchange_repulsion),
      "coulomb_kjmol", &CEEnergyComponents::coulomb_kjmol,
      "exchange_repulsion_kjmol",
      &CEEnergyComponents::exchange_repulsion_kjmol,
      "polarization_kjmol", &CEEnergyComponents::polarization_kjmol,
      "dispersion_kjmol", &CEEnergyComponents::dispersion_kjmol,
      "repulsion_kjmol", &CEEnergyComponents::repulsion_kjmol,
      "exchange_kjmol", &CEEnergyComponents::exchange_kjmol,
      "total_kjmol", &CEEnergyComponents::total_kjmol,
      sol::meta_function::addition, &CEEnergyComponents::operator+,
      sol::meta_function::subtraction, &CEEnergyComponents::operator-);

  m.new_usertype<CEModelInteraction>(
      "CEModelInteraction",
      sol::call_constructor,
      sol::constructors<CEModelInteraction(const CEParameterizedModel &)>(),
      // operator() is overloaded — bind the dimer-energy entry point as
      // a `compute` method with explicit name.
      "compute", &CEModelInteraction::operator());

  m.new_usertype<TransformResult>(
      "TransformResult", sol::no_constructor,
      "rotation",
      sol::readonly_property([](const TransformResult &r, sol::this_state s) {
        return r.rotation;
      }),
      "translation",
      sol::readonly_property([](const TransformResult &r, sol::this_state s) {
        return r.translation;
      }),
      "wfn", sol::readonly(&TransformResult::wfn),
      "rmsd", sol::readonly(&TransformResult::rmsd));

  m.new_usertype<WavefunctionTransformer>(
      "WavefunctionTransformer", sol::no_constructor,
      "calculate_transform",
      &WavefunctionTransformer::calculate_transform);

  m.new_usertype<WolfParameters>(
      "WolfParameters",
      sol::call_constructor,
      sol::factories([]() { return WolfParameters{}; }),
      "cutoff", &WolfParameters::cutoff,
      "alpha", &WolfParameters::alpha,
      sol::meta_function::to_string, [](const WolfParameters &p) {
        return fmt::format("<WolfParameters cutoff={:.2f} alpha={:.4f}>",
                           p.cutoff, p.alpha);
      });

  m.new_usertype<WolfCouplingTerm>(
      "WolfCouplingTerm", sol::no_constructor,
      "neighbor_a", sol::readonly(&WolfCouplingTerm::neighbor_a),
      "neighbor_b", sol::readonly(&WolfCouplingTerm::neighbor_b),
      "coupling_energy", sol::readonly(&WolfCouplingTerm::coupling_energy));

  m.new_usertype<WolfCouplingResult>(
      "WolfCouplingResult", sol::no_constructor,
      "coupling_terms",
      sol::readonly_property([](const WolfCouplingResult &r) {
        return sol::as_table(r.coupling_terms);
      }),
      "total_coupling", sol::readonly(&WolfCouplingResult::total_coupling));

  // -- Free functions ----------------------------------------------------
  // All Eigen-bearing free functions go through table_to_* / vec_to_table
  // wrappers; they're verbose but mechanical.
  m.set_function(
      "wolf_pair_energy",
      [](const sol::table &charges_a, const sol::table &positions_a,
         const sol::table &charges_b, const sol::table &positions_b,
         const WolfParameters &params) {
        return wolf_pair_energy(table_to_vecx(charges_a),
                                 table_to_mat3n(positions_a),
                                 table_to_vecx(charges_b),
                                 table_to_mat3n(positions_b), params);
      });

  m.set_function(
      "coulomb_energy",
      [](const sol::table &charges, const sol::table &positions) {
        return coulomb_energy(table_to_vecx(charges),
                               table_to_mat3n(positions));
      });

  m.set_function(
      "coulomb_pair_energy",
      [](const sol::table &charges_a, const sol::table &positions_a,
         const sol::table &charges_b, const sol::table &positions_b) {
        return coulomb_pair_energy(table_to_vecx(charges_a),
                                    table_to_mat3n(positions_a),
                                    table_to_vecx(charges_b),
                                    table_to_mat3n(positions_b));
      });

  m.set_function(
      "coulomb_efield",
      [](const sol::table &charges, const sol::table &positions,
         const sol::table &point, sol::this_state s) {
        return vec_to_table(
            s, coulomb_efield(table_to_vecx(charges),
                                table_to_mat3n(positions),
                                table_to_vec3(point)));
      });

  m.new_usertype<CEEnergyModel>(
      "CEEnergyModel",
      sol::call_constructor,
      sol::factories(
          [](const occ::crystal::Crystal &c,
             const std::vector<occ::qm::Wavefunction> &wfns_a) {
            return CEEnergyModel(c, wfns_a, {});
          },
          [](const occ::crystal::Crystal &c,
             const std::vector<occ::qm::Wavefunction> &wfns_a,
             const std::vector<occ::qm::Wavefunction> &wfns_b) {
            return CEEnergyModel(c, wfns_a, wfns_b);
          }),
      "set_model_name", &CEEnergyModel::set_model_name,
      "compute_energy", &CEEnergyModel::compute_energy,
      "coulomb_scale_factor", &CEEnergyModel::coulomb_scale_factor,
      "polarization_scale_factor", &CEEnergyModel::polarization_scale_factor,
      sol::meta_function::to_string,
      [](const CEEnergyModel &) { return std::string{"<CEEnergyModel>"}; });
}

} // namespace occ::lua_bindings
