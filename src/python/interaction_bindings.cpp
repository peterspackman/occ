#include "interaction_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <occ/interaction/pair_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/wavefunction_transform.h>
#include <occ/interaction/wolf.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/coulomb.h>

using namespace nb::literals;
using namespace occ::interaction;
using transform::TransformResult;
using transform::WavefunctionTransformer;

nb::module_ register_interaction_bindings(nb::module_ &m) {

  nb::class_<CEParameterizedModel>(m, "CEParameterizedModel")
      .def("ce_model_from_string",
           [](const std::string &s) { return ce_model_from_string(s); });

  nb::class_<CEEnergyComponents>(m, "CEEnergyComponents")
      // Raw energy values (Hartree)
      .def_ro("coulomb", &CEEnergyComponents::coulomb, "Coulomb energy (Hartree)")
      .def_ro("exchange", &CEEnergyComponents::exchange, "Exchange energy (Hartree)")
      .def_ro("repulsion", &CEEnergyComponents::repulsion, "Repulsion energy (Hartree)")
      .def_ro("polarization", &CEEnergyComponents::polarization, "Polarization energy (Hartree)")
      .def_ro("dispersion", &CEEnergyComponents::dispersion, "Dispersion energy (Hartree)")
      .def_ro("total", &CEEnergyComponents::total, "Total energy (Hartree)")
      .def_ro("exchange_repulsion", &CEEnergyComponents::exchange_repulsion, "Exchange-repulsion energy (Hartree)")
      // Energy values in kJ/mol
      .def("coulomb_kjmol", &CEEnergyComponents::coulomb_kjmol)
      .def("exchange_repulsion_kjmol",
           &CEEnergyComponents::exchange_repulsion_kjmol)
      .def("polarization_kjmol", &CEEnergyComponents::polarization_kjmol)
      .def("dispersion_kjmol", &CEEnergyComponents::dispersion_kjmol)
      .def("repulsion_kjmol", &CEEnergyComponents::repulsion_kjmol)
      .def("exchange_kjmol", &CEEnergyComponents::exchange_kjmol)
      .def("total_kjmol", &CEEnergyComponents::total_kjmol)
      // Operators
      .def("__add__", &CEEnergyComponents::operator+)
      .def("__sub__", &CEEnergyComponents::operator-)
      .def("__iadd__", &CEEnergyComponents::operator+=)
      .def("__isub__", &CEEnergyComponents::operator-=);

  nb::class_<CEModelInteraction>(m, "CEModelInteraction")
      .def(nb::init<const CEParameterizedModel &>())
      .def("__call__", &CEModelInteraction::operator());

  nb::class_<TransformResult>(m, "TransformResult")
      .def_ro("rotation", &TransformResult::rotation)
      .def_ro("translation", &TransformResult::translation)
      .def_ro("wfn", &TransformResult::wfn)
      .def_ro("rmsd", &TransformResult::rmsd);

  nb::class_<WavefunctionTransformer>(m, "WavefunctionTransformer")
      .def("calculate_transform",
           &WavefunctionTransformer::calculate_transform);

  // Wolf sum and utilities
  nb::class_<WolfParameters>(m, "WolfParameters")
      .def(nb::init<>())
      .def_rw("cutoff", &WolfParameters::cutoff,
              "Wolf sum cutoff radius (Bohr)")
      .def_rw("alpha", &WolfParameters::alpha,
              "Wolf sum convergence parameter (1/Bohr)")
      .def("__repr__", [](const WolfParameters &p) {
        return fmt::format("<WolfParameters cutoff={:.2f} alpha={:.4f}>",
                           p.cutoff, p.alpha);
      });

  nb::class_<WolfCouplingTerm>(m, "WolfCouplingTerm")
      .def_ro("neighbor_a", &WolfCouplingTerm::neighbor_a,
              "Index of first neighbor")
      .def_ro("neighbor_b", &WolfCouplingTerm::neighbor_b,
              "Index of second neighbor")
      .def_ro("coupling_energy", &WolfCouplingTerm::coupling_energy,
              "Coupling energy in Hartree");

  nb::class_<WolfCouplingResult>(m, "WolfCouplingResult")
      .def_ro("coupling_terms", &WolfCouplingResult::coupling_terms,
              "List of coupling terms")
      .def_ro("total_coupling", &WolfCouplingResult::total_coupling,
              "Total coupling energy in Hartree");

  m.def("wolf_pair_energy", &wolf_pair_energy,
        "charges_a"_a, "positions_a"_a, "charges_b"_a, "positions_b"_a,
        "params"_a,
        "Compute Wolf sum pairwise interaction energy between two sets of charges");

  m.def("wolf_electric_field", &wolf_electric_field,
        "charges"_a, "source_positions"_a, "target_positions"_a, "params"_a,
        "Compute Wolf sum electric field at target positions from source charges");

  m.def("compute_wolf_coupling_terms", &compute_wolf_coupling_terms,
        "electric_fields_per_neighbor"_a, "polarizabilities"_a,
        "Compute many-body polarization coupling terms from electric fields");

  // Classical Coulomb utilities
  m.def("coulomb_energy", &coulomb_energy,
        "charges"_a, "positions"_a,
        "Compute classical Coulomb self-energy for a set of charges");

  m.def("coulomb_pair_energy", &coulomb_pair_energy,
        "charges_a"_a, "positions_a"_a, "charges_b"_a, "positions_b"_a,
        "Compute classical Coulomb interaction energy between two sets of charges");

  m.def("coulomb_efield", &coulomb_efield,
        "charges"_a, "positions"_a, "point"_a,
        "Compute classical Coulomb electric field at a point from a set of charges");

  m.def("coulomb_pair_efield", &coulomb_pair_efield,
        "charges_a"_a, "positions_a"_a, "charges_b"_a, "positions_b"_a,
        "Compute classical Coulomb electric field from pair: returns (E_a, E_b)");

  m.def("coulomb_interaction_energy_asym_charges",
        &coulomb_interaction_energy_asym_charges,
        "dimer"_a, "asymmetric_charges"_a,
        "Compute classical Coulomb interaction energy for dimer using asymmetric unit charges");

  m.def("coulomb_efield_asym_charges", &coulomb_efield_asym_charges,
        "dimer"_a, "asymmetric_charges"_a,
        "Compute classical Coulomb electric field for dimer using asymmetric unit charges");

  m.def("coulomb_self_energy_asym_charges", &coulomb_self_energy_asym_charges,
        "molecule"_a, "asymmetric_charges"_a,
        "Compute classical Coulomb self-energy for molecule using asymmetric unit charges");

  // CE Energy Model
  nb::class_<CEEnergyModel>(m, "CEEnergyModel")
      .def(nb::init<const occ::crystal::Crystal &,
                    const std::vector<occ::qm::Wavefunction> &,
                    const std::vector<occ::qm::Wavefunction> &>(),
           "crystal"_a, "wavefunctions_a"_a, "wavefunctions_b"_a = std::vector<occ::qm::Wavefunction>{},
           "Create a CE energy model for a crystal with embedded wavefunctions")
      .def("set_model_name", &CEEnergyModel::set_model_name, "model_name"_a,
           "Set the CE model name (e.g., 'ce-b3lyp', 'ce-hf')")
      .def("compute_energy", &CEEnergyModel::compute_energy, "dimer"_a,
           "Compute CE energy components for a dimer")
      .def("compute_electric_field", &CEEnergyModel::compute_electric_field,
           "dimer"_a,
           "Compute electric field from a dimer")
      .def("partial_charges", &CEEnergyModel::partial_charges,
           "Get partial charges for all unique molecules")
      .def("coulomb_scale_factor", &CEEnergyModel::coulomb_scale_factor,
           "Get Coulomb scaling factor for this model")
      .def("polarization_scale_factor", &CEEnergyModel::polarization_scale_factor,
           "Get polarization scaling factor for this model")
      .def("compute_total_electric_field_from_neighbors",
           &CEEnergyModel::compute_total_electric_field_from_neighbors,
           "target_molecule"_a, "neighbor_dimers"_a,
           "Compute total electric field at target molecule from all neighbors")
      .def("compute_crystal_field_polarization_energy",
           &CEEnergyModel::compute_crystal_field_polarization_energy,
           "molecule"_a, "crystal_field"_a,
           "Compute polarization energy from crystal field")
      .def("get_polarizabilities", &CEEnergyModel::get_polarizabilities,
           "molecule"_a,
           "Get atomic polarizabilities for a molecule")
      .def("__repr__", [](const CEEnergyModel &model) {
        return fmt::format("<CEEnergyModel>");
      });

  return m;
}
