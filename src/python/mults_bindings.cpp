#include "mults_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/dma/mult.h>
#include <occ/dma/dma.h>
#include <occ/io/structure_format.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_energy_setup.h>
#include <occ/mults/crystal_optimizer.h>
#include <occ/mults/multipole_source.h>
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
    m.def("write_force_field_json", &occ::io::write_force_field_json,
          "path"_a, "basis"_a, "title"_a = "",
          "Write a force-field definition (molecule types, multipoles, pair "
          "potentials, settings) to a JSON file. No crystal data.");
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

    // ========================================================================
    // ForceFieldType enum
    // ========================================================================

    nb::enum_<ForceFieldType>(m, "ForceFieldType")
        .value("None", ForceFieldType::None)
        .value("LennardJones", ForceFieldType::LennardJones)
        .value("BuckinghamDE", ForceFieldType::BuckinghamDE)
        .value("Custom", ForceFieldType::Custom);

    // ========================================================================
    // RigidMolecule
    // ========================================================================

    nb::class_<RigidMolecule::Site>(m, "RigidMoleculeSite")
        .def(nb::init<>())
        .def_rw("position", &RigidMolecule::Site::position)
        .def_rw("multipole", &RigidMolecule::Site::multipole)
        .def_rw("atom_index", &RigidMolecule::Site::atom_index)
        .def_rw("short_range_type", &RigidMolecule::Site::short_range_type);

    nb::class_<RigidMolecule::Atom>(m, "RigidMoleculeAtom")
        .def(nb::init<>())
        .def_rw("atomic_number", &RigidMolecule::Atom::atomic_number)
        .def_rw("position", &RigidMolecule::Atom::position);

    nb::class_<RigidMolecule>(m, "RigidMolecule")
        .def(nb::init<>())
        .def_rw("sites", &RigidMolecule::sites)
        .def_rw("atoms", &RigidMolecule::atoms)
        .def_rw("com", &RigidMolecule::com)
        .def_rw("angle_axis", &RigidMolecule::angle_axis)
        .def_rw("parity", &RigidMolecule::parity)
        .def("rotation_matrix", &RigidMolecule::rotation_matrix)
        .def("__repr__", [](const RigidMolecule &rm) {
            return fmt::format("<RigidMolecule sites={} atoms={} com=({:.3f},{:.3f},{:.3f})>",
                               rm.sites.size(), rm.atoms.size(),
                               rm.com.x(), rm.com.y(), rm.com.z());
        });

    // ========================================================================
    // CrystalOptimizer
    // ========================================================================

    nb::enum_<OptimizationMethod>(m, "OptimizationMethod")
        .value("MSTMIN", OptimizationMethod::MSTMIN)
        .value("LBFGS", OptimizationMethod::LBFGS)
        .value("TrustRegion", OptimizationMethod::TrustRegion);

    nb::class_<CrystalOptimizerSettings>(m, "CrystalOptimizerSettings")
        .def(nb::init<>())
        .def_rw("method", &CrystalOptimizerSettings::method)
        .def_rw("gradient_tolerance", &CrystalOptimizerSettings::gradient_tolerance)
        .def_rw("energy_tolerance", &CrystalOptimizerSettings::energy_tolerance)
        .def_rw("max_iterations", &CrystalOptimizerSettings::max_iterations)
        .def_rw("neighbor_radius", &CrystalOptimizerSettings::neighbor_radius)
        .def_rw("force_field", &CrystalOptimizerSettings::force_field)
        .def_rw("optimize_cell", &CrystalOptimizerSettings::optimize_cell)
        .def_rw("use_ewald", &CrystalOptimizerSettings::use_ewald)
        .def_rw("max_interaction_order", &CrystalOptimizerSettings::max_interaction_order)
        .def_rw("external_pressure_gpa", &CrystalOptimizerSettings::external_pressure_gpa)
        .def("__repr__", [](const CrystalOptimizerSettings &s) {
            return fmt::format("<CrystalOptimizerSettings method={} gtol={:.1e} max_iter={}>",
                               static_cast<int>(s.method), s.gradient_tolerance,
                               s.max_iterations);
        });

    nb::class_<CrystalOptimizerResult>(m, "CrystalOptimizerResult")
        .def_ro("final_energy", &CrystalOptimizerResult::final_energy)
        .def_ro("electrostatic_energy", &CrystalOptimizerResult::electrostatic_energy)
        .def_ro("repulsion_dispersion_energy", &CrystalOptimizerResult::repulsion_dispersion_energy)
        .def_ro("initial_energy", &CrystalOptimizerResult::initial_energy)
        .def_ro("iterations", &CrystalOptimizerResult::iterations)
        .def_ro("function_evaluations", &CrystalOptimizerResult::function_evaluations)
        .def_ro("converged", &CrystalOptimizerResult::converged)
        .def_ro("termination_reason", &CrystalOptimizerResult::termination_reason)
        .def_ro("final_states", &CrystalOptimizerResult::final_states)
        .def("__repr__", [](const CrystalOptimizerResult &r) {
            return fmt::format("<CrystalOptimizerResult E={:.4f} converged={} iter={}>",
                               r.final_energy, r.converged, r.iterations);
        });

    nb::class_<CrystalOptimizer>(m, "CrystalOptimizer")
        .def(nb::init<CrystalEnergySetup, const CrystalOptimizerSettings &>(),
             "setup"_a, "settings"_a = CrystalOptimizerSettings{})
        .def("optimize", nb::overload_cast<>(&CrystalOptimizer::optimize),
             "Run crystal structure optimization")
        .def("energy_calculator",
             [](CrystalOptimizer &o) -> CrystalEnergy& { return o.energy_calculator(); },
             nb::rv_policy::reference_internal,
             "Access the energy calculator")
        .def("states", &CrystalOptimizer::states,
             "Get current molecular states")
        .def("settings",
             [](const CrystalOptimizer &o) -> const CrystalOptimizerSettings& {
                 return o.settings();
             },
             nb::rv_policy::reference_internal)
        .def("__repr__", [](const CrystalOptimizer &o) {
            return fmt::format("<CrystalOptimizer mols={} params={}>",
                               o.energy_calculator().num_molecules(),
                               o.num_parameters());
        });

    // ========================================================================
    // MultipoleConfig + from_crystal (full CIF pipeline)
    // ========================================================================

    nb::class_<MultipoleConfig>(m, "MultipoleConfig")
        .def(nb::init<>())
        .def_rw("method", &MultipoleConfig::method)
        .def_rw("basis_set", &MultipoleConfig::basis_set)
        .def_rw("basename", &MultipoleConfig::basename)
        .def_rw("max_rank", &MultipoleConfig::max_rank)
        .def("__repr__", [](const MultipoleConfig &c) {
            return fmt::format("<MultipoleConfig method='{}' basis='{}' rank={}>",
                               c.method, c.basis_set, c.max_rank);
        });

    m.def("from_crystal", &from_crystal,
          "crystal"_a, "config"_a = MultipoleConfig{},
          "Build CrystalEnergySetup from Crystal (runs SCF + DMA)");

    // ========================================================================
    // Convenience functions
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
