#include "solvent_bindings.h"
#include <fmt/core.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/core/units.h>
#include <occ/driver/cosmors_driver.h>

using namespace nb::literals;
using occ::driver::CosmoRSSolvation;
using occ::driver::CosmoRSSolvationSettings;
using occ::solvent::cosmors::SolvationEnergy;

nb::module_ register_solvent_bindings(nb::module_ &m) {

  nb::class_<CosmoRSSolvationSettings>(
      m, "CosmoRSSettings",
      "Settings for an openCOSMO-RS solvation free energy")
      .def(nb::init<>())
      .def_rw("method", &CosmoRSSolvationSettings::method, "DFT functional")
      .def_rw("basis", &CosmoRSSolvationSettings::basis, "basis set")
      .def_rw("pure_spherical", &CosmoRSSolvationSettings::pure_spherical,
              "spherical (5d) rather than cartesian (6d) basis functions")
      .def_rw("probe_radius", &CosmoRSSolvationSettings::probe_radius_angs,
              "solvent probe radius used to build the cavity, Angstrom")
      .def_rw("angular_points", &CosmoRSSolvationSettings::angular_points,
              "Lebedev order per atom for the cavity")
      .def_rw("constrain_charge", &CosmoRSSolvationSettings::constrain_charge,
              "constrain the surface charge to -q")
      .def_rw("temperature", &CosmoRSSolvationSettings::temperature, "K")
      .def_rw("liquid_volume", &CosmoRSSolvationSettings::liquid_volume,
              "liquid-phase volume per solute molecule (Angstrom^3) for the "
              "reference-state term; non-positive drops that term")
      .def_rw("num_rings", &CosmoRSSolvationSettings::num_rings,
              "rings in the solute; negative counts them from the bond graph");

  nb::class_<SolvationEnergy>(
      m, "CosmoRSEnergy",
      "The terms of an openCOSMO-RS solvation free energy, each in Hartree")
      .def_ro("dielectric", &SolvationEnergy::dielectric,
              "gas to ideal conductor")
      .def_ro("residual", &SolvationEnergy::residual, "RT ln(gamma_res)")
      .def_ro("combinatorial", &SolvationEnergy::combinatorial,
              "RT ln(gamma_comb)")
      .def_ro("vdw", &SolvationEnergy::vdw,
              "van der Waals surface term, -sum_a tau_a A_a")
      .def_ro("ring", &SolvationEnergy::ring, "-omega_ring n_ring")
      .def_ro("reference_state", &SolvationEnergy::reference_state,
              "-RT ln(v_gas/v_liquid)")
      .def_ro("eta", &SolvationEnergy::eta, "fitted intercept")
      .def("total", &SolvationEnergy::total, "sum of every term, Hartree")
      .def("__repr__", [](const SolvationEnergy &e) {
        return fmt::format("<CosmoRSEnergy total={:.3f} kJ/mol>",
                           e.total() * occ::units::AU_TO_KJ_PER_MOL);
      });

  nb::class_<CosmoRSSolvation>(m, "CosmoRSSolvationResult")
      .def_ro("energy", &CosmoRSSolvation::energy, "the term breakdown")
      .def_ro("cavity_area", &CosmoRSSolvation::cavity_area, "Angstrom^2")
      .def_ro("cavity_volume", &CosmoRSSolvation::cavity_volume, "Angstrom^3")
      .def_ro("num_rings", &CosmoRSSolvation::num_rings,
              "rings used, whether given or counted")
      .def_ro("gas", &CosmoRSSolvation::gas, "gas-phase wavefunction")
      .def_ro("conductor", &CosmoRSSolvation::conductor,
              "ideal-conductor wavefunction")
      .def("total", &CosmoRSSolvation::total,
           "the solvation free energy, Hartree");

  m.def(
      "cosmo_rs_solvation_free_energy",
      [](const occ::core::Molecule &solute, const std::string &solvent,
         const CosmoRSSolvationSettings &settings) {
        return occ::driver::cosmors_solvation_free_energy(solute, solvent,
                                                          settings);
      },
      "solute"_a, "solvent"_a, "settings"_a = CosmoRSSolvationSettings{},
      "openCOSMO-RS solvation free energy of a molecule in a named solvent");

  m.def(
      "cosmo_rs_solvation_free_energy",
      [](const occ::core::Molecule &solute, const occ::core::Molecule &solvent,
         const CosmoRSSolvationSettings &settings) {
        return occ::driver::cosmors_solvation_free_energy(solute, solvent,
                                                          settings);
      },
      "solute"_a, "solvent"_a, "settings"_a = CosmoRSSolvationSettings{},
      "The same, computing the solvent's cavity from its geometry rather than "
      "loading a cached ensemble");

  m.def("available_cosmo_rs_solvents",
        &occ::driver::available_cosmors_solvents,
        "solvent names with a cached segment ensemble, sorted");

  return m;
}
