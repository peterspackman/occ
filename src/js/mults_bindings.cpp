#include "mults_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/io/structure_format.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_energy_setup.h>
#include <occ/mults/crystal_optimizer.h>
#include <occ/mults/rigid_molecule.h>

using namespace emscripten;
using namespace occ::mults;

namespace {

// Helper: Vec3 -> JS array
val vec3_to_array(const occ::Vec3 &v) {
    val a = val::array();
    a.set(0, v(0));
    a.set(1, v(1));
    a.set(2, v(2));
    return a;
}

// Helper: std::array<double,3> -> JS array
val arr3_to_array(const std::array<double, 3> &a) {
    val r = val::array();
    r.set(0, a[0]);
    r.set(1, a[1]);
    r.set(2, a[2]);
    return r;
}

} // namespace

void register_mults_bindings() {

    // ========================================================================
    // Structure format types
    // ========================================================================

    class_<occ::io::SiteMultipoles>("SiteMultipoles")
        .constructor<>()
        .property("charge", &occ::io::SiteMultipoles::charge)
        .function("maxRank", &occ::io::SiteMultipoles::max_rank)
        .function("toFlat", &occ::io::SiteMultipoles::to_flat);

    class_<occ::io::MoleculeSite>("MoleculeSite")
        .constructor<>()
        .property("label", &occ::io::MoleculeSite::label)
        .property("element", &occ::io::MoleculeSite::element)
        .property("type", &occ::io::MoleculeSite::type)
        .function("position", optional_override(
            [](const occ::io::MoleculeSite &s) { return arr3_to_array(s.position); }))
        .property("multipoles", &occ::io::MoleculeSite::multipoles);

    class_<occ::io::MoleculeType>("MoleculeType")
        .constructor<>()
        .property("name", &occ::io::MoleculeType::name);

    register_vector<occ::io::MoleculeSite>("VectorMoleculeSite");
    register_vector<occ::io::MoleculeType>("VectorMoleculeType");
    register_vector<occ::io::IndependentMolecule>("VectorIndependentMolecule");
    register_vector<occ::io::BuckinghamPair>("VectorBuckinghamPair");

    class_<occ::io::BuckinghamPair>("BuckinghamPair")
        .constructor<>()
        .property("A", &occ::io::BuckinghamPair::A)
        .property("rho", &occ::io::BuckinghamPair::rho)
        .property("C6", &occ::io::BuckinghamPair::C6);

    class_<occ::io::Potentials>("Potentials")
        .constructor<>()
        .property("cutoff", &occ::io::Potentials::cutoff);

    class_<occ::io::Settings>("Settings")
        .constructor<>()
        .property("ewald_accuracy", &occ::io::Settings::ewald_accuracy)
        .property("use_ewald", &occ::io::Settings::use_ewald)
        .property("pressure_gpa", &occ::io::Settings::pressure_gpa);

    class_<occ::io::IndependentMolecule>("IndependentMolecule")
        .constructor<>()
        .property("type", &occ::io::IndependentMolecule::type)
        .property("parity", &occ::io::IndependentMolecule::parity)
        .function("translation", optional_override(
            [](const occ::io::IndependentMolecule &m) {
                return arr3_to_array(m.translation);
            }))
        .function("orientation", optional_override(
            [](const occ::io::IndependentMolecule &m) {
                return arr3_to_array(m.orientation);
            }));

    class_<occ::io::Basis>("Basis")
        .constructor<>()
        .property("potentials", &occ::io::Basis::potentials)
        .property("settings", &occ::io::Basis::settings);

    class_<occ::io::CrystalData>("CrystalData")
        .constructor<>()
        .property("a", &occ::io::CrystalData::a)
        .property("b", &occ::io::CrystalData::b)
        .property("c", &occ::io::CrystalData::c)
        .property("alpha", &occ::io::CrystalData::alpha)
        .property("beta", &occ::io::CrystalData::beta)
        .property("gamma", &occ::io::CrystalData::gamma)
        .property("space_group", &occ::io::CrystalData::space_group);

    class_<occ::io::ReferenceEnergies>("ReferenceEnergies")
        .constructor<>()
        .property("total", &occ::io::ReferenceEnergies::total);

    class_<occ::io::StructureInput>("StructureInput")
        .constructor<>()
        .property("title", &occ::io::StructureInput::title)
        .property("basis", &occ::io::StructureInput::basis)
        .property("crystal", &occ::io::StructureInput::crystal)
        .property("reference", &occ::io::StructureInput::reference)
        .function("hasCrystal", &occ::io::StructureInput::has_crystal);

    // File I/O
    function("readStructureJson", &occ::io::read_structure_json);
    function("writeStructureJson", &occ::io::write_structure_json);
    function("isStructureFormat", &occ::io::is_structure_format);

    // ========================================================================
    // Enums
    // ========================================================================

    enum_<ForceFieldType>("ForceFieldType")
        .value("None", ForceFieldType::None)
        .value("LennardJones", ForceFieldType::LennardJones)
        .value("BuckinghamDE", ForceFieldType::BuckinghamDE)
        .value("Custom", ForceFieldType::Custom);

    enum_<OptimizationMethod>("OptimizationMethod")
        .value("MSTMIN", OptimizationMethod::MSTMIN)
        .value("LBFGS", OptimizationMethod::LBFGS)
        .value("TrustRegion", OptimizationMethod::TrustRegion);

    // ========================================================================
    // MoleculeState
    // ========================================================================

    register_vector<MoleculeState>("VectorMoleculeState");

    class_<MoleculeState>("MoleculeState")
        .constructor<>()
        .property("parity", &MoleculeState::parity)
        .function("position", optional_override(
            [](const MoleculeState &s) { return vec3_to_array(s.position); }))
        .function("angleAxis", optional_override(
            [](const MoleculeState &s) { return vec3_to_array(s.angle_axis); }));

    // ========================================================================
    // CrystalEnergyResult
    // ========================================================================

    class_<CrystalEnergyResult>("CrystalEnergyResult")
        .property("totalEnergy", &CrystalEnergyResult::total_energy)
        .property("electrostaticEnergy", &CrystalEnergyResult::electrostatic_energy)
        .property("repulsionDispersion", &CrystalEnergyResult::repulsion_dispersion);

    // ========================================================================
    // CrystalEnergySetup + builders
    // ========================================================================

    class_<CrystalEnergySetup>("CrystalEnergySetup")
        .constructor<>()
        .property("cutoffRadius", &CrystalEnergySetup::cutoff_radius)
        .property("useEwald", &CrystalEnergySetup::use_ewald)
        .property("ewaldAccuracy", &CrystalEnergySetup::ewald_accuracy)
        .property("maxInteractionOrder", &CrystalEnergySetup::max_interaction_order);

    function("fromStructureInput", &from_structure_input);

    // ========================================================================
    // CrystalEnergy
    // ========================================================================

    class_<CrystalEnergy>("CrystalEnergy")
        .constructor<CrystalEnergySetup>()
        .function("compute", &CrystalEnergy::compute)
        .function("computeEnergy", &CrystalEnergy::compute_energy)
        .function("initialStates", &CrystalEnergy::initial_states)
        .function("numMolecules", &CrystalEnergy::num_molecules)
        .function("numSites", &CrystalEnergy::num_sites);

    // ========================================================================
    // CrystalOptimizer
    // ========================================================================

    class_<CrystalOptimizerSettings>("CrystalOptimizerSettings")
        .constructor<>()
        .property("method", &CrystalOptimizerSettings::method)
        .property("gradientTolerance", &CrystalOptimizerSettings::gradient_tolerance)
        .property("energyTolerance", &CrystalOptimizerSettings::energy_tolerance)
        .property("maxIterations", &CrystalOptimizerSettings::max_iterations)
        .property("forceField", &CrystalOptimizerSettings::force_field)
        .property("optimizeCell", &CrystalOptimizerSettings::optimize_cell)
        .property("useEwald", &CrystalOptimizerSettings::use_ewald)
        .property("externalPressureGpa", &CrystalOptimizerSettings::external_pressure_gpa);

    class_<CrystalOptimizerResult>("CrystalOptimizerResult")
        .property("finalEnergy", &CrystalOptimizerResult::final_energy)
        .property("electrostaticEnergy", &CrystalOptimizerResult::electrostatic_energy)
        .property("repulsionDispersionEnergy", &CrystalOptimizerResult::repulsion_dispersion_energy)
        .property("initialEnergy", &CrystalOptimizerResult::initial_energy)
        .property("iterations", &CrystalOptimizerResult::iterations)
        .property("converged", &CrystalOptimizerResult::converged)
        .property("terminationReason", &CrystalOptimizerResult::termination_reason)
        .property("finalStates", &CrystalOptimizerResult::final_states);

    class_<CrystalOptimizer>("CrystalOptimizer")
        .constructor<CrystalEnergySetup, CrystalOptimizerSettings>()
        .function("optimize", select_overload<CrystalOptimizerResult()>(
            &CrystalOptimizer::optimize))
        .function("numParameters", &CrystalOptimizer::num_parameters);

    // ========================================================================
    // Convenience
    // ========================================================================

    function("computeCrystalEnergy",
        optional_override([](const std::string &json_path) {
            auto si = occ::io::read_structure_json(json_path);
            auto setup = from_structure_input(si);
            CrystalEnergy calc(std::move(setup));
            auto states = calc.initial_states();
            return calc.compute(states);
        }));
}
