#include "mults_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/io/structure_format.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_energy_setup.h>
#include <occ/mults/rigid_molecule.h>

using namespace nb::literals;
using namespace occ::mults;

nb::module_ register_mults_bindings(nb::module_ &m) {

    // ========================================================================
    // Structure format types (IO)
    // ========================================================================

    nb::class_<occ::io::SiteMultipoles>(m, "SiteMultipoles")
        .def(nb::init<>())
        .def_rw("charge", &occ::io::SiteMultipoles::charge)
        .def_rw("dipole", &occ::io::SiteMultipoles::dipole)
        .def_rw("quadrupole", &occ::io::SiteMultipoles::quadrupole)
        .def_rw("octupole", &occ::io::SiteMultipoles::octupole)
        .def_rw("hexadecapole", &occ::io::SiteMultipoles::hexadecapole)
        .def("max_rank", &occ::io::SiteMultipoles::max_rank)
        .def("to_flat", &occ::io::SiteMultipoles::to_flat)
        .def_static("from_flat", &occ::io::SiteMultipoles::from_flat, "flat"_a);

    nb::class_<occ::io::MoleculeSite>(m, "MoleculeSite")
        .def(nb::init<>())
        .def_rw("label", &occ::io::MoleculeSite::label)
        .def_rw("element", &occ::io::MoleculeSite::element)
        .def_rw("type", &occ::io::MoleculeSite::type)
        .def_rw("position", &occ::io::MoleculeSite::position)
        .def_rw("multipoles", &occ::io::MoleculeSite::multipoles);

    nb::class_<occ::io::MoleculeType>(m, "MoleculeType")
        .def(nb::init<>())
        .def_rw("name", &occ::io::MoleculeType::name)
        .def_rw("sites", &occ::io::MoleculeType::sites);

    nb::class_<occ::io::BuckinghamPair>(m, "BuckinghamPair")
        .def(nb::init<>())
        .def_rw("types", &occ::io::BuckinghamPair::types)
        .def_rw("elements", &occ::io::BuckinghamPair::elements)
        .def_rw("A", &occ::io::BuckinghamPair::A)
        .def_rw("rho", &occ::io::BuckinghamPair::rho)
        .def_rw("C6", &occ::io::BuckinghamPair::C6);

    nb::class_<occ::io::Potentials>(m, "Potentials")
        .def(nb::init<>())
        .def_rw("cutoff", &occ::io::Potentials::cutoff)
        .def_rw("buckingham", &occ::io::Potentials::buckingham);

    nb::class_<occ::io::Settings>(m, "Settings")
        .def(nb::init<>())
        .def_rw("ewald_accuracy", &occ::io::Settings::ewald_accuracy)
        .def_rw("max_interaction_order", &occ::io::Settings::max_interaction_order)
        .def_rw("max_iterations", &occ::io::Settings::max_iterations)
        .def_rw("gradient_tolerance", &occ::io::Settings::gradient_tolerance)
        .def_rw("neighbor_radius", &occ::io::Settings::neighbor_radius)
        .def_rw("pressure_gpa", &occ::io::Settings::pressure_gpa)
        .def_rw("use_ewald", &occ::io::Settings::use_ewald);

    nb::class_<occ::io::IndependentMolecule>(m, "IndependentMolecule")
        .def(nb::init<>())
        .def_rw("type", &occ::io::IndependentMolecule::type)
        .def_rw("translation", &occ::io::IndependentMolecule::translation)
        .def_rw("orientation", &occ::io::IndependentMolecule::orientation)
        .def_rw("parity", &occ::io::IndependentMolecule::parity);

    nb::class_<occ::io::Basis>(m, "Basis")
        .def(nb::init<>())
        .def_rw("molecule_types", &occ::io::Basis::molecule_types)
        .def_rw("potentials", &occ::io::Basis::potentials)
        .def_rw("settings", &occ::io::Basis::settings);

    nb::class_<occ::io::CrystalData>(m, "CrystalData")
        .def(nb::init<>())
        .def_rw("a", &occ::io::CrystalData::a)
        .def_rw("b", &occ::io::CrystalData::b)
        .def_rw("c", &occ::io::CrystalData::c)
        .def_rw("alpha", &occ::io::CrystalData::alpha)
        .def_rw("beta", &occ::io::CrystalData::beta)
        .def_rw("gamma", &occ::io::CrystalData::gamma)
        .def_rw("space_group", &occ::io::CrystalData::space_group)
        .def_rw("molecules", &occ::io::CrystalData::molecules);

    nb::class_<occ::io::ReferenceEnergies>(m, "ReferenceEnergies")
        .def(nb::init<>())
        .def_rw("total", &occ::io::ReferenceEnergies::total)
        .def_rw("components", &occ::io::ReferenceEnergies::components);

    nb::class_<occ::io::StructureInput>(m, "StructureInput")
        .def(nb::init<>())
        .def_rw("title", &occ::io::StructureInput::title)
        .def_rw("basis", &occ::io::StructureInput::basis)
        .def_rw("crystal", &occ::io::StructureInput::crystal)
        .def_rw("reference", &occ::io::StructureInput::reference)
        .def("has_crystal", &occ::io::StructureInput::has_crystal)
        .def("molecule_types", &occ::io::StructureInput::molecule_types,
             nb::rv_policy::reference_internal)
        .def("potentials", &occ::io::StructureInput::potentials,
             nb::rv_policy::reference_internal)
        .def("settings", &occ::io::StructureInput::settings,
             nb::rv_policy::reference_internal)
        .def("__repr__", [](const occ::io::StructureInput &si) {
            return fmt::format("<StructureInput title='{}' types={} mols={} crystal={}>",
                               si.title, si.molecule_types().size(),
                               si.crystal.molecules.size(),
                               si.has_crystal() ? "yes" : "no");
        });

    // File I/O
    m.def("read_structure_json", &occ::io::read_structure_json, "path"_a,
          "Read a StructureInput from a JSON file");
    m.def("write_structure_json", &occ::io::write_structure_json,
          "path"_a, "input"_a,
          "Write a StructureInput to a JSON file");
    m.def("write_basis_json", &occ::io::write_basis_json,
          "path"_a, "basis"_a, "title"_a = "",
          "Write a Basis to a JSON file (no crystal data)");
    m.def("is_structure_format", &occ::io::is_structure_format, "path"_a,
          "Check if a JSON file is a structure format file");

    // ========================================================================
    // MoleculeState
    // ========================================================================

    nb::class_<MoleculeState>(m, "MoleculeState")
        .def(nb::init<>())
        .def_rw("position", &MoleculeState::position)
        .def_rw("angle_axis", &MoleculeState::angle_axis)
        .def_rw("parity", &MoleculeState::parity)
        .def("rotation_matrix", &MoleculeState::rotation_matrix)
        .def("__repr__", [](const MoleculeState &s) {
            return fmt::format("<MoleculeState pos=({:.3f},{:.3f},{:.3f}) parity={}>",
                               s.position.x(), s.position.y(), s.position.z(),
                               s.parity);
        });

    // ========================================================================
    // CrystalEnergyResult
    // ========================================================================

    nb::class_<CrystalEnergyResult>(m, "CrystalEnergyResult")
        .def_ro("total_energy", &CrystalEnergyResult::total_energy)
        .def_ro("electrostatic_energy", &CrystalEnergyResult::electrostatic_energy)
        .def_ro("repulsion_dispersion", &CrystalEnergyResult::repulsion_dispersion)
        .def_ro("forces", &CrystalEnergyResult::forces)
        .def_ro("torques", &CrystalEnergyResult::torques)
        .def_ro("strain_gradient", &CrystalEnergyResult::strain_gradient)
        .def("__repr__", [](const CrystalEnergyResult &r) {
            return fmt::format("<CrystalEnergyResult E={:.4f} elec={:.4f} sr={:.4f}>",
                               r.total_energy, r.electrostatic_energy,
                               r.repulsion_dispersion);
        });

    // ========================================================================
    // CrystalEnergySetup + builders
    // ========================================================================

    nb::class_<CrystalEnergySetup>(m, "CrystalEnergySetup")
        .def(nb::init<>())
        .def_rw("cutoff_radius", &CrystalEnergySetup::cutoff_radius)
        .def_rw("use_ewald", &CrystalEnergySetup::use_ewald)
        .def_rw("ewald_accuracy", &CrystalEnergySetup::ewald_accuracy)
        .def_rw("max_interaction_order", &CrystalEnergySetup::max_interaction_order)
        .def("__repr__", [](const CrystalEnergySetup &s) {
            return fmt::format("<CrystalEnergySetup mols={} ewald={}>",
                               s.molecules.size(),
                               s.use_ewald ? "on" : "off");
        });

    m.def("from_structure_input", &from_structure_input, "si"_a,
          "Build CrystalEnergySetup from a StructureInput");

    m.def("to_structure_input", &to_structure_input,
          "setup"_a, "title"_a = "",
          "Convert CrystalEnergySetup to StructureInput for serialization");

    // ========================================================================
    // CrystalEnergy
    // ========================================================================

    nb::class_<CrystalEnergy>(m, "CrystalEnergy")
        .def(nb::init<CrystalEnergySetup>(), "setup"_a)
        .def("compute", &CrystalEnergy::compute, "states"_a,
             "Compute energy, forces and torques for given molecular states")
        .def("compute_energy", &CrystalEnergy::compute_energy, "states"_a,
             "Compute energy only (faster, for line search)")
        .def("initial_states", &CrystalEnergy::initial_states,
             "Get initial molecular states")
        .def("num_molecules", &CrystalEnergy::num_molecules)
        .def("num_sites", &CrystalEnergy::num_sites)
        .def("__repr__", [](const CrystalEnergy &e) {
            return fmt::format("<CrystalEnergy mols={} sites={} pairs={}>",
                               e.num_molecules(), e.num_sites(),
                               e.neighbor_pairs().size());
        });

    // ========================================================================
    // Convenience: load JSON -> compute energy in one call
    // ========================================================================

    m.def("compute_crystal_energy",
          [](const std::string &json_path) {
              auto si = occ::io::read_structure_json(json_path);
              auto setup = from_structure_input(si);
              CrystalEnergy calc(std::move(setup));
              auto states = calc.initial_states();
              return calc.compute(states);
          },
          "json_path"_a,
          "Load a structure JSON and compute crystal energy");

    return m;
}
