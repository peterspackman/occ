#include "xtb_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <occ/core/dimer.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/qm/wavefunction.h>
#include <occ/xtb/xtb_calculator.h>
#include <occ/xtb/xtb_result.h>

using namespace nb::literals;
using occ::xtb::XtbCalculator;
using occ::xtb::XtbResult;

nb::module_ register_xtb_bindings(nb::module_ &parent) {
  // Match the rest of the occpy bindings — bind names directly into the
  // parent module rather than creating an `occpy.xtb` submodule, so the
  // surface stays consistent (`occpy.XtbCalculator`, `occpy.XtbResult`).
  auto &m = parent;

  // ----- XtbResult --------------------------------------------------------

  nb::class_<XtbResult>(m, "XtbResult",
                        "Result of a single GFN2-xTB SCC. All energies in "
                        "Hartree; matrices use the GFN2 valence basis.")
      .def_ro("scc_energy", &XtbResult::scc_energy,
              "Electronic + isotropic-Coulomb + multipole AES energy.")
      .def_ro("repulsion_energy", &XtbResult::repulsion_energy,
              "Closed-form repulsion energy.")
      .def_ro("dispersion_energy", &XtbResult::dispersion_energy,
              "D4 dispersion energy (zero if dispersion is disabled).")
      .def_ro("total_energy", &XtbResult::total_energy,
              "scc + repulsion + dispersion.")
      .def_ro("shell_charges", &XtbResult::shell_charges,
              "Per-shell partial charges q_shell = ref_occ − Mulliken_pop.")
      .def_ro("atomic_charges", &XtbResult::atomic_charges,
              "Per-atom partial charges (Mulliken).")
      .def_ro("orbital_energies", &XtbResult::orbital_energies,
              "Orbital energies ε at Γ (Hartree).")
      .def_ro("orbital_occupations", &XtbResult::orbital_occupations,
              "Per-orbital occupations (0..2 for closed shell).")
      .def_ro("density_matrix", &XtbResult::density_matrix,
              "Density matrix P at Γ (closed-shell, summed over both spins).")
      .def_ro("overlap_matrix", &XtbResult::overlap_matrix,
              "Overlap matrix S at Γ.")
      .def_ro("orbital_coefficients", &XtbResult::orbital_coefficients,
              "Orbital coefficients C at Γ.")
      .def_ro("n_iterations", &XtbResult::n_iterations,
              "Number of SCC iterations.")
      .def_ro("converged", &XtbResult::converged,
              "True if the SCC converged within thresholds.")
      .def("__repr__", [](const XtbResult &r) {
        return fmt::format(
            "<XtbResult E={:.10f} Ha (scc={:.6f} rep={:.6f} disp={:.6f}) "
            "n_iter={} converged={}>",
            r.total_energy, r.scc_energy, r.repulsion_energy,
            r.dispersion_energy, r.n_iterations, r.converged);
      });

  // ----- XtbCalculator ----------------------------------------------------

  // Renamed from `Method` (which is too generic at top level) to `XtbMethod`.
  nb::enum_<XtbCalculator::Method>(m, "XtbMethod")
      .value("GFN2", XtbCalculator::Method::GFN2);

  nb::class_<XtbCalculator>(
      m, "XtbCalculator",
      "GFN2-xTB calculator. Constructed from an "
      ":class:`occ.core.Molecule`, :class:`occ.core.Dimer`, or "
      ":class:`occ.crystal.Crystal`. Configure via the property "
      "setters and call :meth:`single_point` to run an SCC.\n\n"
      "Example::\n\n"
      "    calc = XtbCalculator(mol)\n"
      "    calc.charge = 0\n"
      "    result = calc.single_point()\n"
      "    energy, gradient = calc.energy_and_gradient()")
      .def(nb::init<const occ::core::Molecule &>(), "molecule"_a,
           "Construct from an isolated molecule.")
      .def(nb::init<const occ::core::Dimer &>(), "dimer"_a,
           "Construct from a dimer (atoms = union of monomers).")
      .def(nb::init<const occ::crystal::Crystal &>(), "crystal"_a,
           "Construct from a 3D periodic crystal (Γ-only by default; "
           "set ``kpoints`` for k-sampling).")

      // Identity / topology
      .def_prop_ro("method", &XtbCalculator::method,
                   "Tight-binding method enum (currently always GFN2).")
      .def_prop_ro("method_name", &XtbCalculator::method_name,
                   "Method name as a string.")
      .def_prop_ro("backend_name", &XtbCalculator::backend_name,
                   "Backend name as a string ('Native').")
      .def_prop_ro("is_periodic", &XtbCalculator::is_periodic,
                   "True if constructed from a Crystal.")
      .def_prop_ro("num_atoms", &XtbCalculator::num_atoms,
                   "Number of atoms in the (central) cell.")
      .def_prop_ro("atomic_numbers", &XtbCalculator::atomic_numbers,
                   "Atomic numbers (length = num_atoms).")
      .def_prop_ro("positions", &XtbCalculator::positions,
                   "Cartesian positions in Bohr (3 × N).")
      .def_prop_ro(
          "lattice",
          [](const XtbCalculator &c) -> nb::object {
            if (!c.is_periodic()) return nb::none();
            return nb::cast(c.lattice());
          },
          "Lattice vectors as columns in Bohr (3 × 3); None for molecular.")

      // Configuration (property pairs)
      .def_prop_rw("charge", &XtbCalculator::charge, &XtbCalculator::set_charge,
                   "Net charge in electrons.")
      .def_prop_rw("num_unpaired_electrons",
                   &XtbCalculator::num_unpaired_electrons,
                   &XtbCalculator::set_num_unpaired_electrons,
                   "Number of unpaired electrons (only 0 supported).")
      .def_prop_rw("max_iterations", &XtbCalculator::max_iterations,
                   &XtbCalculator::set_max_iterations,
                   "Maximum SCC iterations.")
      .def_prop_rw("temperature", &XtbCalculator::temperature,
                   &XtbCalculator::set_temperature,
                   "Electronic temperature in K (Fermi smearing).")
      .def_prop_rw("mixer_damping", &XtbCalculator::mixer_damping,
                   &XtbCalculator::set_mixer_damping,
                   "SCC mixer damping factor (weight on previous iter).")
      .def_prop_rw("include_multipoles", &XtbCalculator::include_multipoles,
                   &XtbCalculator::set_include_multipoles,
                   "Toggle CAMM multipole AES + on-site polarisation.")
      .def_prop_rw("include_dispersion", &XtbCalculator::include_dispersion,
                   &XtbCalculator::set_include_dispersion,
                   "Toggle native D4 dispersion.")
      .def_prop_rw(
          "kpoints",
          [](const XtbCalculator &c) {
            auto k = c.kpoints();
            return std::array<int, 3>{k[0], k[1], k[2]};
          },
          [](XtbCalculator &c, const std::array<int, 3> &k) {
            c.set_kpoints(k[0], k[1], k[2]);
          },
          "Monkhorst-Pack k-grid as (n1, n2, n3); (1, 1, 1) means Γ-only.")

      .def("set_solvent", &XtbCalculator::set_solvent, "name"_a,
           "Enable an implicit-solvent model. Returns False (not yet "
           "implemented in the native backend).")

      // Geometry update
      .def(
          "update_structure",
          [](XtbCalculator &c, const occ::Mat3N &positions) {
            c.update_structure(positions);
          },
          "positions"_a,
          "Update Cartesian positions in Bohr (3 × N). Atomic numbers and "
          "shell layout are kept fixed.")
      .def(
          "update_structure",
          [](XtbCalculator &c, const occ::Mat3N &positions,
             const occ::Mat3 &lattice_bohr) {
            c.update_structure(positions, lattice_bohr);
          },
          "positions"_a, "lattice"_a,
          "Update positions and lattice vectors simultaneously (periodic "
          "only).")

      // Run + results
      .def("single_point", &XtbCalculator::single_point,
           nb::rv_policy::reference_internal,
           "Run an SCC at the current geometry and return the cached "
           ":class:`XtbResult`.")
      .def("single_point_energy", &XtbCalculator::single_point_energy,
           "Convenience wrapper returning just the total energy in Hartree.")
      .def_prop_ro("last_result", &XtbCalculator::last_result,
                   nb::rv_policy::reference_internal,
                   "The most recent SCC result (call ``single_point`` first).")

      // Derived quantities
      .def_prop_ro("charges", &XtbCalculator::charges,
                   "Per-atom Mulliken charges (length = num_atoms).")
      .def_prop_ro("bond_orders", &XtbCalculator::bond_orders,
                   "Wiberg bond-order matrix (N × N).")
      .def_prop_ro("total_energy", &XtbCalculator::total_energy)
      .def_prop_ro("scc_energy", &XtbCalculator::scc_energy)
      .def_prop_ro("repulsion_energy", &XtbCalculator::repulsion_energy)
      .def_prop_ro("dispersion_energy", &XtbCalculator::dispersion_energy)

      // Gradient
      .def("gradient", &XtbCalculator::gradient,
           "Analytical nuclear gradient in Hartree/Bohr (3 × N). Runs a "
           "charge-only SCC internally so the (energy, gradient) pair is "
           "self-consistent.")
      .def("gradient_numerical", &XtbCalculator::gradient_numerical,
           "step_bohr"_a = 1e-3,
           "Five-point central-difference gradient. Slow (6N SCC evals); "
           "useful as an oracle to validate the analytical version.")
      .def("energy_and_gradient", &XtbCalculator::compute_energy_and_gradient,
           "numerical"_a = false, "step_bohr"_a = 1e-3,
           "(energy, gradient) pair. Analytical by default.")

      // Hessian / vibrations
      .def("hessian", &XtbCalculator::compute_hessian_numerical,
           "step_bohr"_a = 0.005,
           "Numerical 3N×3N Hessian (Hartree/Bohr²) via central differences "
           "of the analytical gradient. Costs 6N gradient evaluations.")
      .def("vibrational_modes",
           &XtbCalculator::compute_vibrational_modes,
           "step_bohr"_a = 0.005, "project_tr_rot"_a = false,
           "Build the Hessian and run a normal-mode analysis. Returns a "
           ":class:`occ.core.VibrationalModes` with frequencies (cm⁻¹), "
           "normal modes, and the mass-weighted Hessian.")

      // Conversion
      .def("to_molecule", &XtbCalculator::to_molecule,
           "Snapshot atoms / positions as an :class:`occ.core.Molecule`.")
      .def("to_crystal", &XtbCalculator::to_crystal,
           "Snapshot atoms / lattice as an :class:`occ.crystal.Crystal` "
           "(periodic only; raises otherwise).")
      .def("to_wavefunction", &XtbCalculator::to_wavefunction,
           "Convert the converged SCC state into a "
           ":class:`occ.qm.Wavefunction` for downstream analysis.")

      .def("print_summary", &XtbCalculator::print_summary,
           "Log a results summary (energy decomposition, charges, "
           "HOMO/LUMO/gap) at INFO level.")

      .def("__repr__", [](const XtbCalculator &c) {
        return fmt::format(
            "<XtbCalculator method={} backend={} atoms={} periodic={}>",
            c.method_name(), c.backend_name(), c.num_atoms(),
            c.is_periodic());
      });

  return m;
}
