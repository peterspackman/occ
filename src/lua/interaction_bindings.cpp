#include "interaction_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/coulomb.h>
#include <occ/interaction/pair_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/wavefunction_transform.h>
#include <occ/interaction/wolf.h>

namespace occ::lua_bindings {

using namespace occ::interaction;
using transform::TransformResult;
using transform::WavefunctionTransformer;
namespace lb = luabridge;

void register_interaction_bindings(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      .beginClass<CEParameterizedModel>("CEParameterizedModel")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("coulomb", &CEParameterizedModel::coulomb)
      .addPropertyReadWrite("exchange", &CEParameterizedModel::exchange)
      .addPropertyReadWrite("repulsion", &CEParameterizedModel::repulsion)
      .addPropertyReadWrite("polarization", &CEParameterizedModel::polarization)
      .addPropertyReadWrite("dispersion", &CEParameterizedModel::dispersion)
      .addPropertyReadWrite("name", &CEParameterizedModel::name)
      .addPropertyReadWrite("method", &CEParameterizedModel::method)
      .addPropertyReadWrite("basis", &CEParameterizedModel::basis)
      .addPropertyReadWrite("xdm", &CEParameterizedModel::xdm)
      .addPropertyReadWrite("xdm_a1", &CEParameterizedModel::xdm_a1)
      .addPropertyReadWrite("xdm_a2", &CEParameterizedModel::xdm_a2)
      .endClass()

      .addFunction("ce_model_from_string", &ce_model_from_string)

      .beginClass<CEEnergyComponents>("CEEnergyComponents")
      .addConstructor<void (*)()>()
      .addProperty("coulomb", &CEEnergyComponents::coulomb)
      .addProperty("exchange", &CEEnergyComponents::exchange)
      .addProperty("repulsion", &CEEnergyComponents::repulsion)
      .addProperty("polarization", &CEEnergyComponents::polarization)
      .addProperty("dispersion", &CEEnergyComponents::dispersion)
      .addProperty("total", &CEEnergyComponents::total)
      .addProperty("exchange_repulsion",
                   &CEEnergyComponents::exchange_repulsion)
      .addProperty("coulomb_kjmol", &CEEnergyComponents::coulomb_kjmol)
      .addProperty("exchange_repulsion_kjmol",
                   &CEEnergyComponents::exchange_repulsion_kjmol)
      .addProperty("polarization_kjmol",
                   &CEEnergyComponents::polarization_kjmol)
      .addProperty("dispersion_kjmol", &CEEnergyComponents::dispersion_kjmol)
      .addProperty("repulsion_kjmol", &CEEnergyComponents::repulsion_kjmol)
      .addProperty("exchange_kjmol", &CEEnergyComponents::exchange_kjmol)
      .addProperty("total_kjmol", &CEEnergyComponents::total_kjmol)
      // operator+/- are non-const upstream (likely oversight); take
      // mutable refs here rather than reinterpret_cast'ing.
      .addFunction(
          "__add", +[](CEEnergyComponents *a,
                       const CEEnergyComponents &b) { return *a + b; })
      .addFunction(
          "__sub", +[](CEEnergyComponents *a,
                       const CEEnergyComponents &b) { return *a - b; })
      .endClass()

      .beginClass<CEModelInteraction>("CEModelInteraction")
      .addConstructor<void (*)(const CEParameterizedModel &)>()
      .addFunction(
          "compute", +[](CEModelInteraction *self, qm::Wavefunction &a,
                         qm::Wavefunction &b) { return (*self)(a, b); })
      .endClass()

      .beginClass<TransformResult>("TransformResult")
      .addProperty(
          "rotation",
          +[](const TransformResult *r) -> occ::Mat3 { return r->rotation; })
      .addProperty(
          "translation",
          +[](const TransformResult *r) -> occ::Vec3 { return r->translation; })
      .addProperty("wfn", &TransformResult::wfn)
      .addProperty("rmsd", &TransformResult::rmsd)
      .endClass()

      .beginClass<WavefunctionTransformer>("WavefunctionTransformer")
      .addStaticFunction("calculate_transform",
                         &WavefunctionTransformer::calculate_transform)
      .endClass()

      .beginClass<WolfParameters>("WolfParameters")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("cutoff", &WolfParameters::cutoff)
      .addPropertyReadWrite("alpha", &WolfParameters::alpha)
      .addFunction(
          "__tostring",
          +[](const WolfParameters *p) {
            return fmt::format("<WolfParameters cutoff={:.2f} alpha={:.4f}>",
                               p->cutoff, p->alpha);
          })
      .endClass()

      .beginClass<WolfCouplingTerm>("WolfCouplingTerm")
      .addProperty("neighbor_a", &WolfCouplingTerm::neighbor_a)
      .addProperty("neighbor_b", &WolfCouplingTerm::neighbor_b)
      .addProperty("coupling_energy", &WolfCouplingTerm::coupling_energy)
      .endClass()

      .beginClass<WolfCouplingResult>("WolfCouplingResult")
      .addFunction(
          "coupling_terms",
          +[](const WolfCouplingResult *r, lua_State *S) {
            lb::LuaRef t = lb::newTable(S);
            for (size_t i = 0; i < r->coupling_terms.size(); ++i) {
              t[static_cast<int>(i + 1)] = r->coupling_terms[i];
            }
            return t;
          })
      .addProperty("total_coupling", &WolfCouplingResult::total_coupling)
      .endClass()

      .addFunction(
          "wolf_pair_energy",
          +[](const lb::LuaRef &charges_a, const lb::LuaRef &positions_a,
              const lb::LuaRef &charges_b, const lb::LuaRef &positions_b,
              const WolfParameters &params) {
            return wolf_pair_energy(
                table_to_vecx(charges_a), table_to_mat3n(positions_a),
                table_to_vecx(charges_b), table_to_mat3n(positions_b), params);
          })

      .addFunction(
          "coulomb_energy",
          +[](const lb::LuaRef &charges, const lb::LuaRef &positions) {
            return coulomb_energy(table_to_vecx(charges),
                                  table_to_mat3n(positions));
          })

      .addFunction(
          "coulomb_pair_energy",
          +[](const lb::LuaRef &charges_a, const lb::LuaRef &positions_a,
              const lb::LuaRef &charges_b, const lb::LuaRef &positions_b) {
            return coulomb_pair_energy(
                table_to_vecx(charges_a), table_to_mat3n(positions_a),
                table_to_vecx(charges_b), table_to_mat3n(positions_b));
          })

      // lua_State* is recovered via charges.state() — LuaBridge3's
      // first-arg lua_State* injection misbehaves when followed by
      // LuaRef args (see [[feedback-luabridge-quirks]]).
      .addFunction(
          "coulomb_efield",
          +[](const lb::LuaRef &charges, const lb::LuaRef &positions,
              const lb::LuaRef &point) {
            return vec_to_table(charges.state(),
                                coulomb_efield(table_to_vecx(charges),
                                               table_to_mat3n(positions),
                                               table_to_vec3(point)));
          })

      .beginClass<CEEnergyModel>("CEEnergyModel")
      // Two construction shapes — LuaBridge3 doesn't auto-overload
      // constructors, so expose them as static factory functions.
      .addStaticFunction(
          "new",
          +[](const occ::crystal::Crystal &c,
              const std::vector<occ::qm::Wavefunction> &wfns_a) {
            return new CEEnergyModel(c, wfns_a, {});
          })
      .addStaticFunction(
          "new_with_b",
          +[](const occ::crystal::Crystal &c,
              const std::vector<occ::qm::Wavefunction> &wfns_a,
              const std::vector<occ::qm::Wavefunction> &wfns_b) {
            return new CEEnergyModel(c, wfns_a, wfns_b);
          })
      .addFunction("set_model_name", &CEEnergyModel::set_model_name)
      .addFunction("compute_energy", &CEEnergyModel::compute_energy)
      .addProperty("coulomb_scale_factor", &CEEnergyModel::coulomb_scale_factor)
      .addProperty("polarization_scale_factor",
                   &CEEnergyModel::polarization_scale_factor)
      .addFunction(
          "__tostring",
          +[](const CEEnergyModel *) { return std::string{"<CEEnergyModel>"}; })
      .endClass()

      .endNamespace();
}

} // namespace occ::lua_bindings
