#include "dma_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/dma/dma.h>
#include <occ/dma/linear_multipole_calculator.h>
#include <occ/dma/mult.h>
#include <occ/dma/multipole_calculator.h>
#include <occ/driver/dma_driver.h>
#include <occ/qm/wavefunction.h>

namespace occ::lua_bindings {

using namespace occ::dma;
namespace lb = luabridge;

void register_dma_bindings(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      .beginClass<Mult>("Mult")
      // Two construction shapes — split into a default constructor and
      // a static factory for the rank-typed variant.
      .addConstructor<void (*)()>()
      .addStaticFunction(
          "new_with_rank", +[](int max_rank) { return new Mult(max_rank); })
      .addPropertyReadWrite("max_rank", &Mult::max_rank)
      // q is a Vec — expose as method-style getter/setter going through
      // Lua tables. Property getters can't reliably accept lua_State*.
      .addProperty(
          "get_q", +[](const Mult *mp) -> occ::Vec { return mp->q; })
      .addFunction(
          "set_q",
          +[](Mult *mp, const lb::LuaRef &t) { mp->q = table_to_vecx(t); })
      .addProperty("num_components", &Mult::num_components)
      .addFunction("to_string", &Mult::to_string)
      .addFunction(
          "get_multipole", +[](const Mult *mp, int l,
                               int mm) { return mp->get_multipole(l, mm); })
      .addFunction(
          "get_component",
          +[](const Mult *mp, const std::string &name) {
            return mp->get_component(name);
          })
      .addFunction(
          "__tostring",
          +[](const Mult *mp) {
            return fmt::format("<Mult max_rank={} components={}>", mp->max_rank,
                               mp->num_components());
          })
      .endClass()

      .addFunction("component_name_to_lm", &Mult::component_name_to_lm)

      .beginClass<DMASettings>("DMASettings")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("max_rank", &DMASettings::max_rank)
      .addPropertyReadWrite("big_exponent", &DMASettings::big_exponent)
      .addPropertyReadWrite("include_nuclei", &DMASettings::include_nuclei)
      .addFunction(
          "__tostring",
          +[](const DMASettings *s) {
            return fmt::format("<DMASettings max_rank={} big_exponent={:.2f} "
                               "include_nuclei={}>",
                               s->max_rank, s->big_exponent,
                               s->include_nuclei ? "true" : "false");
          })
      .endClass()

      .beginClass<DMAResult>("DMAResult")
      .addConstructor<void (*)()>()
      .addProperty("max_rank", &DMAResult::max_rank)
      .addFunction(
          "multipoles",
          +[](const DMAResult *r, lua_State *S) {
            lb::LuaRef t = lb::newTable(S);
            for (size_t i = 0; i < r->multipoles.size(); ++i) {
              t[static_cast<int>(i + 1)] = r->multipoles[i];
            }
            return t;
          })
      .addFunction(
          "__tostring",
          +[](const DMAResult *r) {
            return fmt::format("<DMAResult max_rank={} num_sites={}>",
                               r->max_rank, r->multipoles.size());
          })
      .endClass()

      .beginClass<DMASites>("DMASites")
      .addConstructor<void (*)()>()
      .addProperty("size", &DMASites::size)
      .addProperty("num_atoms", &DMASites::num_atoms)
      .addFunction(
          "atoms",
          +[](const DMASites *s, lua_State *S) {
            lb::LuaRef t = lb::newTable(S);
            for (size_t i = 0; i < s->atoms.size(); ++i) {
              t[static_cast<int>(i + 1)] = s->atoms[i];
            }
            return t;
          })
      .addFunction(
          "name",
          +[](const DMASites *s, lua_State *S) {
            lb::LuaRef t = lb::newTable(S);
            for (size_t i = 0; i < s->name.size(); ++i) {
              t[static_cast<int>(i + 1)] = s->name[i];
            }
            return t;
          })
      .addProperty(
          "positions",
          +[](const DMASites *s) -> occ::Mat3N { return s->positions; })
      .addProperty(
          "atom_indices",
          +[](const DMASites *s) -> occ::IVec { return s->atom_indices; })
      .addProperty(
          "radii", +[](const DMASites *s) -> occ::Vec { return s->radii; })
      .addProperty(
          "limits", +[](const DMASites *s) -> occ::IVec { return s->limits; })
      .addFunction(
          "__tostring",
          +[](const DMASites *s) {
            return fmt::format("<DMASites num_sites={} num_atoms={}>",
                               s->size(), s->num_atoms());
          })
      .endClass()

      .beginClass<DMACalculator>("DMACalculator")
      .addConstructor<void (*)(const occ::qm::Wavefunction &)>()
      .addFunction("update_settings", &DMACalculator::update_settings)
      .addProperty("settings", &DMACalculator::settings)
      .addFunction("set_radius_for_element",
                   &DMACalculator::set_radius_for_element)
      .addFunction("set_limit_for_element",
                   &DMACalculator::set_limit_for_element)
      .addProperty("sites", &DMACalculator::sites)
      .addFunction("compute_multipoles", &DMACalculator::compute_multipoles)
      .addFunction("compute_total_multipoles",
                   &DMACalculator::compute_total_multipoles)
      .addFunction(
          "__tostring",
          +[](const DMACalculator *c) {
            return fmt::format("<DMACalculator num_sites={} max_rank={}>",
                               c->sites().size(), c->settings().max_rank);
          })
      .endClass()

      .beginClass<LinearDMASettings>("LinearDMASettings")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("max_rank", &LinearDMASettings::max_rank)
      .addPropertyReadWrite("include_nuclei",
                            &LinearDMASettings::include_nuclei)
      .addPropertyReadWrite("use_slices", &LinearDMASettings::use_slices)
      .addPropertyReadWrite("tolerance", &LinearDMASettings::tolerance)
      .addPropertyReadWrite("default_radius",
                            &LinearDMASettings::default_radius)
      .addPropertyReadWrite("hydrogen_radius",
                            &LinearDMASettings::hydrogen_radius)
      .endClass()

      .beginClass<LinearMultipoleCalculator>("LinearMultipoleCalculator")
      // Two construction shapes — default settings vs explicit.
      .addConstructor<void (*)(const occ::qm::Wavefunction &,
                               const LinearDMASettings &)>()
      .addStaticFunction(
          "new_default_settings",
          +[](const occ::qm::Wavefunction &w) {
            return new LinearMultipoleCalculator(w, LinearDMASettings{});
          })
      .addFunction("calculate", &LinearMultipoleCalculator::calculate)
      .endClass()

      .beginClass<occ::driver::DMAConfig>("DMAConfig")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("wavefunction_filename",
                            &occ::driver::DMAConfig::wavefunction_filename)
      .addPropertyReadWrite("punch_filename",
                            &occ::driver::DMAConfig::punch_filename)
      .addPropertyReadWrite("settings", &occ::driver::DMAConfig::settings)
      .addPropertyReadWrite("write_punch", &occ::driver::DMAConfig::write_punch)
      .endClass()

      .beginClass<occ::driver::DMADriver::DMAOutput>("DMAOutput")
      .addProperty("result", &occ::driver::DMADriver::DMAOutput::result)
      .addProperty("sites", &occ::driver::DMADriver::DMAOutput::sites)
      .endClass()

      .beginClass<occ::driver::DMADriver>("DMADriver")
      // Default + config-init constructors.
      .addConstructor<void (*)()>()
      .addStaticFunction(
          "new_with_config",
          +[](const occ::driver::DMAConfig &c) {
            return new occ::driver::DMADriver(c);
          })
      .addFunction("set_config", &occ::driver::DMADriver::set_config)
      .addProperty("config", &occ::driver::DMADriver::config)
      // sol::overload split: parameterless vs wavefunction-driven run.
      .addFunction(
          "run", +[](occ::driver::DMADriver *d) { return d->run(); })
      .addFunction(
          "run_with_wavefunction",
          +[](occ::driver::DMADriver *d, const occ::qm::Wavefunction &w) {
            return d->run(w);
          })
      .endClass()

      .addFunction("dma_generate_punch_file",
                   &occ::driver::DMADriver::generate_punch_file)
      .addFunction("dma_write_punch_file",
                   &occ::driver::DMADriver::write_punch_file)

      .endNamespace();
}

} // namespace occ::lua_bindings
