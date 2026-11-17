#include "opt_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/core/molecule.h>
#include <occ/opt/geometry_optimizer.h>
#include <occ/opt/neb.h>

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

      .beginClass<OptimizerSettings>("OptimizerSettings")
      .addConstructor<void (*)()>()
      .addStaticFunction("saddle_point", +[](int mode) {
        return OptimizerSettings::saddle_point(mode);
      })
      .addPropertyReadWrite("trust", &OptimizerSettings::trust)
      .addPropertyReadWrite("trust_min", &OptimizerSettings::trust_min)
      .addPropertyReadWrite("trust_max", &OptimizerSettings::trust_max)
      .addPropertyReadWrite("project_net_force_torque",
                            &OptimizerSettings::project_net_force_torque)
      .addPropertyReadWrite("transition_state",
                            &OptimizerSettings::transition_state)
      .addPropertyReadWrite("ts_mode", &OptimizerSettings::ts_mode)
      .endClass()

      .beginClass<GeometryOptimizer>("GeometryOptimizer")
      // Three construction shapes: molecule only, with criteria, or with
      // criteria and settings.
      .addConstructor<void (*)(const occ::core::Molecule &,
                               const ConvergenceCriteria &)>()
      .addStaticFunction(
          "new_default",
          +[](const occ::core::Molecule &m) { return new GeometryOptimizer(m); })
      .addStaticFunction(
          "new_with_settings",
          +[](const occ::core::Molecule &m, const ConvergenceCriteria &c,
              const OptimizerSettings &s) {
            return new GeometryOptimizer(m, c, s);
          })
      // a Matrix userdata or a nested table (3N x 3N, Hartree/Bohr^2)
      .addFunction("set_initial_hessian",
                   +[](GeometryOptimizer *o, const lb::LuaRef &h) {
                     o->set_initial_hessian(
                         table_to_eigen_matrix<occ::Mat>(h));
                   })
      .addFunction("set_followed_mode",
                   +[](GeometryOptimizer *o, const lb::LuaRef &mode) {
                     o->set_followed_mode(table_to_vecx(mode));
                   })
      .addProperty("negative_modes", &GeometryOptimizer::negative_modes)
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

      .beginClass<NebSettings>("NebSettings")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("images", &NebSettings::images)
      .addPropertyReadWrite("spring", &NebSettings::spring)
      .addPropertyReadWrite("spring_max", &NebSettings::spring_max)
      .addPropertyReadWrite("idpp", &NebSettings::idpp)
      .addPropertyReadWrite("climbing_image", &NebSettings::climbing_image)
      .addPropertyReadWrite("climb_threshold", &NebSettings::climb_threshold)
      .addPropertyReadWrite("climbing_force_max",
                            &NebSettings::climbing_force_max)
      .addPropertyReadWrite("force_max", &NebSettings::force_max)
      .addPropertyReadWrite("max_step", &NebSettings::max_step)
      .endClass()

      .beginClass<NudgedElasticBand>("NudgedElasticBand")
      .addConstructor<void (*)(const occ::core::Molecule &,
                               const occ::core::Molecule &,
                               const NebSettings &)>()
      .addFunction("images", &NudgedElasticBand::images)
      .addFunction("path", &NudgedElasticBand::path)
      .addFunction("set_endpoint_energies",
                   &NudgedElasticBand::set_endpoint_energies)
      // energies: table of numbers; gradients: table of Mat3N userdata or
      // nested tables, one per image
      .addFunction(
          "update",
          +[](NudgedElasticBand *b, const lb::LuaRef &energies,
              const lb::LuaRef &gradients) {
            std::vector<double> e;
            std::vector<occ::Mat3N> g;
            for (int i = 1; i <= energies.length(); i++)
              e.push_back(energies[i].unsafe_cast<double>());
            for (int i = 1; i <= gradients.length(); i++) {
              const lb::LuaRef gi = gradients[i];
              g.push_back(gi.isUserdata()
                              ? gi.unsafe_cast<const occ::Mat3N &>()
                              : table_to_mat3n(gi));
            }
            b->update(e, g);
          })
      .addFunction("step", &NudgedElasticBand::step)
      .addFunction("tangent", &NudgedElasticBand::tangent)
      .addFunction("energies",
                   +[](const NudgedElasticBand *b) { return b->energies(); })
      .addProperty("is_converged", &NudgedElasticBand::is_converged)
      .addProperty("climbing", &NudgedElasticBand::climbing)
      .addProperty("highest_image", &NudgedElasticBand::highest_image)
      .addProperty("max_force", &NudgedElasticBand::max_force)
      .addProperty("current_step", &NudgedElasticBand::current_step)
      .endClass()

      .endNamespace()
      .endNamespace();
}

} // namespace occ::lua_bindings
