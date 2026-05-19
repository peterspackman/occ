#include "xtb_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/dimer.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/qm/wavefunction.h>
#include <occ/xtb/xtb_calculator.h>
#include <occ/xtb/xtb_result.h>

using namespace emscripten;
using occ::xtb::XtbCalculator;
using occ::xtb::XtbResult;

void register_xtb_bindings() {
  // ----- XtbResult --------------------------------------------------------
  // Mirrors the Python binding: read-only properties for every field.
  class_<XtbResult>("XtbResult")
      .property("sccEnergy", &XtbResult::scc_energy)
      .property("repulsionEnergy", &XtbResult::repulsion_energy)
      .property("dispersionEnergy", &XtbResult::dispersion_energy)
      .property("totalEnergy", &XtbResult::total_energy)
      .property("shellCharges", &XtbResult::shell_charges)
      .property("atomicCharges", &XtbResult::atomic_charges)
      .property("orbitalEnergies", &XtbResult::orbital_energies)
      .property("orbitalOccupations", &XtbResult::orbital_occupations)
      .property("densityMatrix", &XtbResult::density_matrix)
      .property("overlapMatrix", &XtbResult::overlap_matrix)
      .property("orbitalCoefficients", &XtbResult::orbital_coefficients)
      .property("nIterations", &XtbResult::n_iterations)
      .property("converged", &XtbResult::converged)
      .function("toString", optional_override([](const XtbResult &r) {
                  return std::string("<XtbResult E=") +
                         std::to_string(r.total_energy) + " Ha n_iter=" +
                         std::to_string(r.n_iterations) +
                         (r.converged ? " converged>" : " NOT converged>");
                }));

  // ----- XtbMethod enum ---------------------------------------------------
  enum_<XtbCalculator::Method>("XtbMethod")
      .value("GFN2", XtbCalculator::Method::GFN2);

  // ----- XtbCalculator ----------------------------------------------------
  // Mirrors the Python binding 1:1 (camelCase JS conventions). Accessor /
  // mutator pairs are bound as `.property` where the underlying setter
  // takes a single value, otherwise as `.function` getter/setter pairs.
  class_<XtbCalculator>("XtbCalculator")
      // Embind can't overload constructors by argument *type* — only by arity.
      // Expose explicit factories instead so JS callers say
      // `XtbCalculator.fromMolecule(mol)`, `XtbCalculator.fromCrystal(c)`,
      // `XtbCalculator.fromDimer(d)`.
      // XtbCalculator is move-only (owns a unique_ptr<Gfn2Engine>); embind
      // can't return move-only types by value, so factories return raw
      // pointers and JS takes ownership.
      .class_function(
          "fromMolecule",
          optional_override([](const occ::core::Molecule &mol) {
            return new XtbCalculator(mol);
          }),
          allow_raw_pointers())
      .class_function(
          "fromDimer",
          optional_override([](const occ::core::Dimer &dimer) {
            return new XtbCalculator(dimer);
          }),
          allow_raw_pointers())
      .class_function(
          "fromCrystal",
          optional_override([](const occ::crystal::Crystal &crystal) {
            return new XtbCalculator(crystal);
          }),
          allow_raw_pointers())

      // Identity / topology
      .function("method", &XtbCalculator::method)
      .function("methodName", &XtbCalculator::method_name)
      .function("backendName", &XtbCalculator::backend_name)
      .function("isPeriodic", &XtbCalculator::is_periodic)
      .function("numAtoms", &XtbCalculator::num_atoms)
      .function("atomicNumbers", optional_override([](const XtbCalculator &c) {
                  return c.atomic_numbers();
                }))
      .function("positions", optional_override([](const XtbCalculator &c) {
                  return c.positions();
                }))
      .function("lattice", optional_override([](const XtbCalculator &c) {
                  return c.lattice();
                }))

      // Configuration (getter / setter pairs)
      .property("charge", &XtbCalculator::charge, &XtbCalculator::set_charge)
      .property("numUnpairedElectrons",
                &XtbCalculator::num_unpaired_electrons,
                &XtbCalculator::set_num_unpaired_electrons)
      .property("maxIterations", &XtbCalculator::max_iterations,
                &XtbCalculator::set_max_iterations)
      .property("temperature", &XtbCalculator::temperature,
                &XtbCalculator::set_temperature)
      .property("mixerDamping", &XtbCalculator::mixer_damping,
                &XtbCalculator::set_mixer_damping)
      .property("includeMultipoles", &XtbCalculator::include_multipoles,
                &XtbCalculator::set_include_multipoles)
      .property("includeDispersion", &XtbCalculator::include_dispersion,
                &XtbCalculator::set_include_dispersion)
      .function("setKpoints", &XtbCalculator::set_kpoints)
      .function("kpoints", optional_override([](const XtbCalculator &c) {
                  auto k = c.kpoints();
                  // Return as a plain JS array for ergonomics.
                  emscripten::val arr = emscripten::val::array();
                  arr.call<void>("push", k[0]);
                  arr.call<void>("push", k[1]);
                  arr.call<void>("push", k[2]);
                  return arr;
                }))
      .function("setSolvent", &XtbCalculator::set_solvent)

      // Geometry update
      .function("updateStructure",
                optional_override([](XtbCalculator &c, const occ::Mat3N &p) {
                  c.update_structure(p);
                }))
      .function("updateStructureWithLattice",
                optional_override([](XtbCalculator &c, const occ::Mat3N &p,
                                      const occ::Mat3 &lat) {
                  c.update_structure(p, lat);
                }))

      // Run + result access
      .function("singlePoint",
                optional_override([](XtbCalculator &c) {
                  // Return a copy so the JS side owns the result and isn't
                  // tied to the calculator's internal cache lifetime.
                  return c.single_point();
                }))
      .function("singlePointEnergy", &XtbCalculator::single_point_energy)
      .function("lastResult", optional_override([](const XtbCalculator &c) {
                  return c.last_result();
                }))

      // Derived quantities
      .function("charges", &XtbCalculator::charges)
      .function("bondOrders", &XtbCalculator::bond_orders)
      .function("totalEnergy", &XtbCalculator::total_energy)
      .function("sccEnergy", &XtbCalculator::scc_energy)
      .function("repulsionEnergy", &XtbCalculator::repulsion_energy)
      .function("dispersionEnergy", &XtbCalculator::dispersion_energy)

      // Gradient
      .function("gradient", &XtbCalculator::gradient)
      .function("gradientNumerical",
                optional_override([](XtbCalculator &c, double step) {
                  return c.gradient_numerical(step);
                }))
      .function("energyAndGradient",
                optional_override([](XtbCalculator &c, bool numerical,
                                      double step) {
                  auto [e, g] = c.compute_energy_and_gradient(numerical, step);
                  // Return as a plain {energy, gradient} JS object so the
                  // caller doesn't have to destructure a C++ pair.
                  emscripten::val out = emscripten::val::object();
                  out.set("energy", e);
                  out.set("gradient", g);
                  return out;
                }))

      // Hessian / vibrations
      .function("hessian",
                optional_override([](XtbCalculator &c, double step) {
                  return c.compute_hessian_numerical(step);
                }))
      .function("vibrationalModes",
                optional_override([](XtbCalculator &c, double step,
                                      bool project_tr_rot) {
                  return c.compute_vibrational_modes(step, project_tr_rot);
                }))

      // Conversion
      .function("toMolecule", &XtbCalculator::to_molecule)
      .function("toCrystal", &XtbCalculator::to_crystal)
      .function("toWavefunction", &XtbCalculator::to_wavefunction)

      .function("printSummary", &XtbCalculator::print_summary)
      .function("toString", optional_override([](const XtbCalculator &c) {
                  return std::string("<XtbCalculator method=") +
                         c.method_name() + " backend=" + c.backend_name() +
                         " atoms=" + std::to_string(c.num_atoms()) +
                         (c.is_periodic() ? " periodic>" : ">");
                }));
}
