#include "xtb_bindings.h"
#include "eigen_conv.h"
#include "enum_stacks.h"
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
namespace lb = luabridge;

void register_xtb_bindings(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      // ----- XtbResult ----------------------------------------------------
      // Read-only view of the cached SCC result. Eigen matrices/vectors are
      // exposed as 1-indexed Lua tables (per eigen_conv.h convention).
      .beginClass<XtbResult>("XtbResult")
      .addProperty("scc_energy", &XtbResult::scc_energy)
      .addProperty("repulsion_energy", &XtbResult::repulsion_energy)
      .addProperty("dispersion_energy", &XtbResult::dispersion_energy)
      .addProperty("total_energy", &XtbResult::total_energy)
      .addProperty("n_iterations", &XtbResult::n_iterations)
      .addProperty("converged", &XtbResult::converged)
      // Eigen-typed fields exposed as properties. Return-by-value
      // (Vec / Mat) puts the userdata on the mutable path so the
      // row-proxy indexing works on the caller side.
      .addProperty(
          "shell_charges",
          +[](const XtbResult *r) -> Vec { return r->shell_charges; })
      .addProperty(
          "atomic_charges",
          +[](const XtbResult *r) -> Vec { return r->atomic_charges; })
      .addProperty(
          "orbital_energies",
          +[](const XtbResult *r) -> Vec { return r->orbital_energies; })
      .addProperty(
          "orbital_occupations",
          +[](const XtbResult *r) -> Vec { return r->orbital_occupations; })
      .addProperty(
          "density_matrix",
          +[](const XtbResult *r) -> Mat { return r->density_matrix; })
      .addProperty(
          "overlap_matrix",
          +[](const XtbResult *r) -> Mat { return r->overlap_matrix; })
      .addProperty(
          "orbital_coefficients",
          +[](const XtbResult *r) -> Mat { return r->orbital_coefficients; })
      .addFunction(
          "__tostring",
          +[](const XtbResult *r) {
            return fmt::format(
                "<XtbResult E={:.10f} Ha (scc={:.6f} rep={:.6f} disp={:.6f}) "
                "n_iter={} converged={}>",
                r->total_energy, r->scc_energy, r->repulsion_energy,
                r->dispersion_energy, r->n_iterations, r->converged);
          })
      .endClass()

      // ----- XtbMethod enum -----------------------------------------------
      OCC_LUA_ENUM_NAMESPACE("XtbMethod", OCC_ENUM_XtbMethod)

      // ----- XtbCalculator ------------------------------------------------
      // XtbCalculator is move-only (owns a unique_ptr<Gfn2Engine>). Three
      // construction shapes — pick molecule-init as canonical and expose
      // dimer/crystal init as static factories.
      .beginClass<XtbCalculator>("XtbCalculator")
      .addConstructor<void (*)(const occ::core::Molecule &)>()
      .addStaticFunction(
          "new_from_dimer",
          +[](const occ::core::Dimer &dimer) {
            return new XtbCalculator(dimer);
          })
      .addStaticFunction(
          "new_from_crystal",
          +[](const occ::crystal::Crystal &crystal) {
            return new XtbCalculator(crystal);
          })

      // Identity / topology
      .addProperty("method", &XtbCalculator::method)
      .addProperty("method_name", &XtbCalculator::method_name)
      .addProperty("backend_name", &XtbCalculator::backend_name)
      .addProperty("is_periodic", &XtbCalculator::is_periodic)
      .addProperty("num_atoms", &XtbCalculator::num_atoms)
      // Return Eigen types as non-const copies so callers get the
      // mutable-userdata path (LuaBridge3 only installs the
      // row-proxy __index on the non-const metatable).
      .addProperty(
          "atomic_numbers",
          +[](const XtbCalculator *c) -> IVec { return c->atomic_numbers(); })
      .addProperty(
          "positions",
          +[](const XtbCalculator *c) -> Mat3N { return c->positions(); })
      .addFunction(
          "lattice",
          +[](const XtbCalculator *c, lua_State *S) -> lb::LuaRef {
            if (!c->is_periodic())
              return lb::LuaRef(S);
            return mat_to_table(S, c->lattice());
          })

      // Configuration — read/write properties
      .addProperty("charge", &XtbCalculator::charge, &XtbCalculator::set_charge)
      .addProperty("num_unpaired_electrons",
                   &XtbCalculator::num_unpaired_electrons,
                   &XtbCalculator::set_num_unpaired_electrons)
      .addProperty("max_iterations", &XtbCalculator::max_iterations,
                   &XtbCalculator::set_max_iterations)
      .addProperty("temperature", &XtbCalculator::temperature,
                   &XtbCalculator::set_temperature)
      .addProperty("mixer_damping", &XtbCalculator::mixer_damping,
                   &XtbCalculator::set_mixer_damping)
      .addProperty("include_multipoles", &XtbCalculator::include_multipoles,
                   &XtbCalculator::set_include_multipoles)
      .addProperty("include_dispersion", &XtbCalculator::include_dispersion,
                   &XtbCalculator::set_include_dispersion)
      .addFunction(
          "get_kpoints",
          +[](const XtbCalculator *c, lua_State *S) {
            lb::LuaRef t = lb::newTable(S);
            auto k = c->kpoints();
            t[1] = k[0];
            t[2] = k[1];
            t[3] = k[2];
            return t;
          })
      .addFunction(
          "set_kpoints",
          +[](XtbCalculator *c, const lb::LuaRef &t) {
            c->set_kpoints(static_cast<int>(lua_get_num(t, 1)),
                           static_cast<int>(lua_get_num(t, 2)),
                           static_cast<int>(lua_get_num(t, 3)));
          })
      .addFunction("set_solvent", &XtbCalculator::set_solvent)

      // Geometry update — sol::overload split; accept tables.
      .addFunction(
          "update_structure",
          +[](XtbCalculator *c, const lb::LuaRef &positions) {
            c->update_structure(table_to_mat3n(positions));
          })
      .addFunction(
          "update_structure_with_lattice",
          +[](XtbCalculator *c, const lb::LuaRef &positions,
              const lb::LuaRef &lattice) {
            c->update_structure(table_to_mat3n(positions),
                                table_to_mat3(lattice));
          })

      // Run + result access
      .addFunction(
          "single_point",
          +[](XtbCalculator *c) -> XtbResult { return c->single_point(); })
      .addFunction("single_point_energy", &XtbCalculator::single_point_energy)
      .addFunction(
          "last_result",
          +[](const XtbCalculator *c) -> XtbResult { return c->last_result(); })

      // Derived quantities — Eigen returns as userdata.
      .addProperty(
          "charges",
          +[](const XtbCalculator *c) -> Vec { return c->charges(); })
      .addProperty(
          "bond_orders",
          +[](const XtbCalculator *c) -> Mat { return c->bond_orders(); })
      .addProperty("total_energy", &XtbCalculator::total_energy)
      .addProperty("scc_energy", &XtbCalculator::scc_energy)
      .addProperty("repulsion_energy", &XtbCalculator::repulsion_energy)
      .addProperty("dispersion_energy", &XtbCalculator::dispersion_energy)

      // Gradient — Mat3N as registered userdata so callers can use
      // the row-proxy idiom `g[1][j]` for x of atom j.
      .addFunction(
          "gradient", +[](XtbCalculator *c) -> Mat3N { return c->gradient(); })
      .addFunction(
          "gradient_numerical",
          +[](XtbCalculator *c, double step) -> Mat3N {
            return c->gradient_numerical(step);
          })
      .addFunction(
          "energy_and_gradient",
          +[](XtbCalculator *c, bool numerical, double step, lua_State *S) {
            auto [e, g] = c->compute_energy_and_gradient(numerical, step);
            lb::LuaRef out = lb::newTable(S);
            out["energy"] = e;
            out["gradient"] = Mat3N(g);
            return out;
          })

      // Hessian / vibrations
      .addFunction(
          "hessian",
          +[](XtbCalculator *c, double step, lua_State *S) {
            return mat_to_table(S, c->compute_hessian_numerical(step));
          })

      // Conversion
      .addFunction("to_molecule", &XtbCalculator::to_molecule)
      .addFunction("to_crystal", &XtbCalculator::to_crystal)
      .addFunction("to_wavefunction", &XtbCalculator::to_wavefunction)

      .addFunction("print_summary", &XtbCalculator::print_summary)

      .addFunction(
          "__tostring",
          +[](const XtbCalculator *c) {
            return fmt::format(
                "<XtbCalculator method={} backend={} atoms={} periodic={}>",
                c->method_name(), c->backend_name(), c->num_atoms(),
                c->is_periodic());
          })
      .endClass()

      .endNamespace();
}

} // namespace occ::lua_bindings
