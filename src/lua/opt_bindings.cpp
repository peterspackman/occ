#include "opt_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/core/molecule.h>
#include <occ/opt/geometry_optimizer.h>

namespace occ::lua_bindings {

using namespace occ::opt;
namespace lb = luabridge;

void register_opt_bindings(lua_State *L) {
  // Python uses a submodule (`occpy.opt.X`); mirror that as a nested
  // namespace (`occ.opt.X`).
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .beginNamespace("opt")

      .beginClass<ConvergenceCriteria>("ConvergenceCriteria")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("gradient_max", &ConvergenceCriteria::gradient_max)
      .addPropertyReadWrite("gradient_rms", &ConvergenceCriteria::gradient_rms)
      .addPropertyReadWrite("step_max", &ConvergenceCriteria::step_max)
      .addPropertyReadWrite("step_rms", &ConvergenceCriteria::step_rms)
      .addPropertyReadWrite("energy_change",
                            &ConvergenceCriteria::energy_change)
      .addPropertyReadWrite("use_energy_criterion",
                            &ConvergenceCriteria::use_energy_criterion)
      .addPropertyReadWrite("max_iterations",
                            &ConvergenceCriteria::max_iterations)
      .endClass()

      .beginClass<GeometryOptimizer>("GeometryOptimizer")
      // Two construction shapes: molecule only, or with criteria.
      .addConstructor<void (*)(const occ::core::Molecule &,
                               const ConvergenceCriteria &)>()
      .addStaticFunction(
          "new_default",
          +[](const occ::core::Molecule &m) { return new GeometryOptimizer(m); })
      .addFunction("step", &GeometryOptimizer::step)
      // Accept either a Mat3N userdata (direct from compute_gradient) or a
      // nested Lua table.
      .addFunction(
          "update",
          +[](GeometryOptimizer *o, double energy, const lb::LuaRef &gradient) {
            if (gradient.isUserdata())
              o->update(energy, gradient.unsafe_cast<const occ::Mat3N &>());
            else
              o->update(energy, table_to_mat3n(gradient));
          })
      .addFunction(
          "get_next_geometry",
          +[](const GeometryOptimizer *o) { return o->get_next_geometry(); })
      .addProperty("is_converged", &GeometryOptimizer::is_converged)
      .addProperty("current_step", &GeometryOptimizer::current_step)
      .addProperty("current_energy", &GeometryOptimizer::current_energy)
      .addProperty("trust_radius", &GeometryOptimizer::trust_radius)
      .addFunction(
          "__tostring",
          +[](const GeometryOptimizer *o) {
            return fmt::format("<GeometryOptimizer: step={}, converged={}>",
                               o->current_step(), o->is_converged());
          })
      .endClass()

      .endNamespace()
      .endNamespace();
}

} // namespace occ::lua_bindings
