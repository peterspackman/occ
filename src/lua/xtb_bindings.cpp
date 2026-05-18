#include "xtb_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/core/dimer.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/qm/wavefunction.h>
#include <occ/xtb/xtb_calculator.h>
#include <occ/xtb/xtb_result.h>

namespace occ::lua_bindings {

using occ::xtb::XtbCalculator;
using occ::xtb::XtbResult;

void register_xtb_bindings(sol::state_view lua, sol::table &occ_module) {
  // ----- XtbResult --------------------------------------------------------
  // Read-only view of the cached SCC result. Matrices are exposed as Lua
  // tables (row-major, 1-indexed) so they're easy to inspect from a script
  // without a separate matrix userdata type.
  //
  // sol::this_state is sol2's "magic" parameter: at call time, sol2 injects
  // the active lua_State* so we don't need to capture a sol::state_view
  // (which would dangle once register_xtb_bindings returns).
  occ_module.new_usertype<XtbResult>(
      "XtbResult", sol::no_constructor,
      "scc_energy", sol::readonly(&XtbResult::scc_energy),
      "repulsion_energy", sol::readonly(&XtbResult::repulsion_energy),
      "dispersion_energy", sol::readonly(&XtbResult::dispersion_energy),
      "total_energy", sol::readonly(&XtbResult::total_energy),
      "n_iterations", sol::readonly(&XtbResult::n_iterations),
      "converged", sol::readonly(&XtbResult::converged),
      // Vector / matrix fields → Eigen userdata (Vector, Matrix). No
      // copy at the boundary; mutation through the userdata feeds back
      // into the result object.
      "shell_charges",
      sol::readonly_property([](const XtbResult &r) -> const occ::Vec & {
        return r.shell_charges;
      }),
      "atomic_charges",
      sol::readonly_property([](const XtbResult &r) -> const occ::Vec & {
        return r.atomic_charges;
      }),
      "orbital_energies",
      sol::readonly_property([](const XtbResult &r) -> const occ::Vec & {
        return r.orbital_energies;
      }),
      "orbital_occupations",
      sol::readonly_property([](const XtbResult &r) -> const occ::Vec & {
        return r.orbital_occupations;
      }),
      "density_matrix",
      sol::readonly_property([](const XtbResult &r) -> const occ::Mat & {
        return r.density_matrix;
      }),
      "overlap_matrix",
      sol::readonly_property([](const XtbResult &r) -> const occ::Mat & {
        return r.overlap_matrix;
      }),
      "orbital_coefficients",
      sol::readonly_property([](const XtbResult &r) -> const occ::Mat & {
        return r.orbital_coefficients;
      }),
      sol::meta_function::to_string, [](const XtbResult &r) {
        return fmt::format(
            "<XtbResult E={:.10f} Ha (scc={:.6f} rep={:.6f} disp={:.6f}) "
            "n_iter={} converged={}>",
            r.total_energy, r.scc_energy, r.repulsion_energy,
            r.dispersion_energy, r.n_iterations, r.converged);
      });

  // ----- XtbMethod enum ---------------------------------------------------
  occ_module.new_enum<XtbCalculator::Method>(
      "XtbMethod", {{"GFN2", XtbCalculator::Method::GFN2}});

  // ----- XtbCalculator ----------------------------------------------------
  // XtbCalculator is move-only (owns a unique_ptr<Gfn2Engine>). sol::factories
  // wraps the constructors as free functions; the right factory is picked by
  // the argument's userdata type.
  //
  // sol::call_constructor exposes them under `()` so Lua callers can write
  // `occ.XtbCalculator(mol)` to match the Python / JS APIs. We also expose
  // `.new(mol)` as an alternate.
  auto factories = sol::factories(
      [](const occ::core::Molecule &mol) { return XtbCalculator(mol); },
      [](const occ::core::Dimer &dimer) { return XtbCalculator(dimer); },
      [](const occ::crystal::Crystal &crystal) {
        return XtbCalculator(crystal);
      });

  occ_module.new_usertype<XtbCalculator>(
      "XtbCalculator",
      sol::call_constructor, factories,
      "new", factories,

      // Identity / topology
      "method", sol::readonly_property(&XtbCalculator::method),
      "method_name", sol::readonly_property(&XtbCalculator::method_name),
      "backend_name", sol::readonly_property(&XtbCalculator::backend_name),
      "is_periodic", sol::readonly_property(&XtbCalculator::is_periodic),
      "num_atoms", sol::readonly_property(&XtbCalculator::num_atoms),
      // Eigen matrices/vectors are now first-class userdata — return them
      // directly; sol2 pushes them via the registered usertype. Callers
      // get `pos[1][2]` access (1-indexed), `pos:rows()` / `pos:cols()`,
      // `print(pos)` pretty repr, and in-place mutation.
      "atomic_numbers",
      sol::readonly_property([](const XtbCalculator &c) {
        return c.atomic_numbers();
      }),
      "positions",
      sol::readonly_property([](const XtbCalculator &c) { return c.positions(); }),
      "lattice",
      sol::readonly_property(
          [](const XtbCalculator &c, sol::this_state s) -> sol::object {
            if (!c.is_periodic()) return sol::lua_nil;
            return sol::make_object(s, c.lattice());
          }),

      // Configuration (read/write properties — accessor + mutator pairs)
      "charge",
      sol::property(&XtbCalculator::charge, &XtbCalculator::set_charge),
      "num_unpaired_electrons",
      sol::property(&XtbCalculator::num_unpaired_electrons,
                    &XtbCalculator::set_num_unpaired_electrons),
      "max_iterations",
      sol::property(&XtbCalculator::max_iterations,
                    &XtbCalculator::set_max_iterations),
      "temperature",
      sol::property(&XtbCalculator::temperature,
                    &XtbCalculator::set_temperature),
      "mixer_damping",
      sol::property(&XtbCalculator::mixer_damping,
                    &XtbCalculator::set_mixer_damping),
      "include_multipoles",
      sol::property(&XtbCalculator::include_multipoles,
                    &XtbCalculator::set_include_multipoles),
      "include_dispersion",
      sol::property(&XtbCalculator::include_dispersion,
                    &XtbCalculator::set_include_dispersion),
      "kpoints",
      sol::property(
          [](const XtbCalculator &c, sol::this_state s) {
            sol::state_view lua(s);
            auto k = c.kpoints();
            sol::table t = lua.create_table(3, 0);
            t[1] = k[0];
            t[2] = k[1];
            t[3] = k[2];
            return t;
          },
          [](XtbCalculator &c, const sol::table &t) {
            c.set_kpoints(t.get<int>(1), t.get<int>(2), t.get<int>(3));
          }),
      "set_solvent", &XtbCalculator::set_solvent,

      // Geometry update — typed Eigen parameters. Lua callers pass a
      // Mat3N userdata (built via `occ.Mat3N({{xs},{ys},{zs}})` or
      // obtained from another molecule's `.positions`).
      "update_structure",
      sol::overload(
          [](XtbCalculator &c, const occ::Mat3N &positions) {
            c.update_structure(positions);
          },
          [](XtbCalculator &c, const occ::Mat3N &positions,
             const occ::Mat3 &lattice) {
            c.update_structure(positions, lattice);
          }),

      // Run + result access. Returning by value keeps the userdata in
      // Lua's hands instead of pinning the XtbCalculator's internal cache.
      "single_point",
      [](XtbCalculator &c) -> XtbResult { return c.single_point(); },
      "single_point_energy", &XtbCalculator::single_point_energy,
      "last_result",
      sol::readonly_property(
          [](const XtbCalculator &c) -> XtbResult { return c.last_result(); }),

      // Derived quantities — all Eigen returns now go straight back as
      // typed userdata.
      "charges", [](const XtbCalculator &c) { return c.charges(); },
      "bond_orders",
      [](const XtbCalculator &c) { return c.bond_orders(); },
      "total_energy", &XtbCalculator::total_energy,
      "scc_energy", &XtbCalculator::scc_energy,
      "repulsion_energy", &XtbCalculator::repulsion_energy,
      "dispersion_energy", &XtbCalculator::dispersion_energy,

      // Gradient. Returns a Mat3N userdata (3 rows = x/y/z,
      // N cols = atoms). Read with `g[1][j]`.
      "gradient", [](XtbCalculator &c) { return c.gradient(); },
      "gradient_numerical",
      [](XtbCalculator &c, double step) { return c.gradient_numerical(step); },
      "energy_and_gradient",
      [](XtbCalculator &c, bool numerical, double step, sol::this_state s) {
        sol::state_view lua(s);
        auto [e, g] = c.compute_energy_and_gradient(numerical, step);
        sol::table out = lua.create_table(0, 2);
        out["energy"] = e;
        // sol::table_proxy::operator= can't infer the push path for
        // usertype-registered Eigen matrices on its own — wrap with
        // sol::make_object so the right pusher kicks in.
        out["gradient"] = sol::make_object(s, g);
        return out;
      },

      // Hessian / vibrations
      "hessian",
      [](XtbCalculator &c, double step) {
        return c.compute_hessian_numerical(step);
      },

      // Conversion
      "to_molecule", &XtbCalculator::to_molecule,
      "to_crystal", &XtbCalculator::to_crystal,
      "to_wavefunction", &XtbCalculator::to_wavefunction,

      "print_summary", &XtbCalculator::print_summary,

      sol::meta_function::to_string, [](const XtbCalculator &c) {
        return fmt::format(
            "<XtbCalculator method={} backend={} atoms={} periodic={}>",
            c.method_name(), c.backend_name(), c.num_atoms(),
            c.is_periodic());
      });
}

} // namespace occ::lua_bindings
