#include "qm_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/element.h>
#include <occ/core/atom.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/qm/chelpg.h>
#include <occ/qm/expectation.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/scf.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>

using namespace emscripten;
using namespace occ::qm;
using namespace occ::core;

void register_qm_bindings() {
    // SpinorbitalKind enum
    enum_<SpinorbitalKind>("SpinorbitalKind")
        .value("Restricted", SpinorbitalKind::Restricted)
        .value("Unrestricted", SpinorbitalKind::Unrestricted)
        .value("General", SpinorbitalKind::General);

    // Shell class binding
    class_<Shell>("Shell")
        .constructor<occ::core::PointCharge, double>()
        .property("origin", &Shell::origin)
        .property("exponents", &Shell::exponents)
        .property("contractionCoefficients", &Shell::contraction_coefficients)
        .function("numContractions", &Shell::num_contractions)
        .function("numPrimitives", &Shell::num_primitives)
        .function("norm", &Shell::norm)
        .function("toString", optional_override([](const Shell& s) {
            return std::string("<Shell l=") + std::to_string(s.l) + " [" + 
                   std::to_string(s.origin(0)) + ", " + std::to_string(s.origin(1)) + 
                   ", " + std::to_string(s.origin(2)) + "]>";
        }));

    // AOBasis class binding
    class_<AOBasis>("AOBasis")
        .class_function("load", optional_override([](const std::vector<Atom>& atoms, const std::string& name) {
            return AOBasis::load(atoms, name);
        }))
        .function("shells", &AOBasis::shells)
        .function("setPure", &AOBasis::set_pure)
        .function("size", &AOBasis::size)
        .function("nbf", &AOBasis::nbf)
        .function("atoms", &AOBasis::atoms)
        .function("firstBf", &AOBasis::first_bf)
        .function("bfToShell", &AOBasis::bf_to_shell)
        .function("bfToAtom", &AOBasis::bf_to_atom)
        .function("shellToAtom", &AOBasis::shell_to_atom)
        .function("atomToShell", &AOBasis::atom_to_shell)
        .function("lMax", &AOBasis::l_max)
        .function("name", &AOBasis::name)
        .function("evaluate", optional_override([](const AOBasis& basis, const occ::Mat3N& points, int maxDerivative) {
            if (maxDerivative > 2 || maxDerivative < 0)
                throw std::runtime_error("Invalid max derivative (must be 0, 1, 2)");
            return occ::gto::evaluate_basis(basis, points, maxDerivative);
        }))
        .function("toString", optional_override([](const AOBasis& basis) {
            return std::string("<AOBasis (") + basis.name() + ") nsh=" + 
                   std::to_string(basis.nsh()) + " nbf=" + std::to_string(basis.nbf()) + 
                   " natoms=" + std::to_string(basis.atoms().size()) + ">";
        }));

    // MolecularOrbitals class binding
    class_<MolecularOrbitals>("MolecularOrbitals")
        .constructor<>()
        .property("kind", &MolecularOrbitals::kind)
        .property("numAlpha", &MolecularOrbitals::n_alpha)
        .property("numBeta", &MolecularOrbitals::n_beta)
        .property("numAo", &MolecularOrbitals::n_ao)
        .property("orbitalCoeffs", &MolecularOrbitals::C)
        .property("occupiedOrbitalCoeffs", &MolecularOrbitals::Cocc)
        .property("densityMatrix", &MolecularOrbitals::D)
        .property("orbitalEnergies", &MolecularOrbitals::energies)
        .function("expectationValue", optional_override([](const MolecularOrbitals& mo, const occ::Mat& op) {
            return 2 * occ::qm::expectation(mo.kind, mo.D, op);
        }))
        .function("toString", optional_override([](const MolecularOrbitals& mo) {
            return std::string("<MolecularOrbitals kind=") + spinorbital_kind_to_string(mo.kind) + 
                   " nao=" + std::to_string(mo.n_ao) + " nalpha=" + std::to_string(mo.n_alpha) + 
                   " nbeta=" + std::to_string(mo.n_beta) + ">";
        }));

    // Wavefunction class binding
    class_<Wavefunction>("Wavefunction")
        .property("molecularOrbitals", &Wavefunction::mo)
        .property("atoms", &Wavefunction::atoms)
        .property("basis", &Wavefunction::basis)
        .function("mullikenCharges", &Wavefunction::mulliken_charges)
        .function("multiplicity", &Wavefunction::multiplicity)
        .function("rotate", &Wavefunction::apply_rotation)
        .function("translate", &Wavefunction::apply_translation)
        .function("transform", &Wavefunction::apply_transformation)
        .function("charge", &Wavefunction::charge)
        .class_function("load", &Wavefunction::load)
        .function("save", optional_override([](Wavefunction& wfn, const std::string& filename) {
            wfn.save(filename);
        }))
        .function("electronDensity", optional_override([](const Wavefunction& wfn, const occ::Mat3N& points, int derivatives) {
            return occ::density::evaluate_density_on_grid(wfn, points, derivatives);
        }))
        .function("chelpgCharges", optional_override([](const Wavefunction& wfn) { 
            return chelpg_charges(wfn); 
        }))
        .function("toFchk", optional_override([](Wavefunction& wfn, const std::string& filename) {
            auto writer = occ::io::FchkWriter(filename);
            wfn.save(writer);
            writer.write();
        }))
        .class_function("fromFchk", optional_override([](const std::string& filename) {
            auto reader = occ::io::FchkReader(filename);
            return Wavefunction(reader);
        }))
        .class_function("fromMolden", optional_override([](const std::string& filename) {
            auto reader = occ::io::MoldenReader(filename);
            return Wavefunction(reader);
        }))
        .function("toString", optional_override([](const Wavefunction& wfn) {
            std::string formula = "molecule"; // Simplified for now
            return std::string("<Wavefunction ") + formula + " " + wfn.method + "/" + 
                   wfn.basis.name() + " kind=" + spinorbital_kind_to_string(wfn.mo.kind) + 
                   " nbf=" + std::to_string(wfn.basis.nbf()) + " charge=" + std::to_string(wfn.charge()) + ">";
        }));

    // SCFConvergenceSettings class binding
    class_<SCFConvergenceSettings>("SCFConvergenceSettings")
        .constructor<>()
        .property("energyThreshold", &SCFConvergenceSettings::energy_threshold)
        .property("commutatorThreshold", &SCFConvergenceSettings::commutator_threshold)
        .property("incrementalFockThreshold", &SCFConvergenceSettings::incremental_fock_threshold)
        .function("energyConverged", &SCFConvergenceSettings::energy_converged)
        .function("commutatorConverged", &SCFConvergenceSettings::commutator_converged)
        .function("energyAndCommutatorConverged", &SCFConvergenceSettings::energy_and_commutator_converged)
        .function("startIncrementalFock", &SCFConvergenceSettings::start_incremental_fock);

    // HartreeFock class binding
    class_<HartreeFock>("HartreeFock")
        .constructor<const AOBasis&>()
        .function("pointChargeInteractionEnergy", &HartreeFock::nuclear_point_charge_interaction_energy)
        .function("wolfPointChargeInteractionEnergy", &HartreeFock::wolf_point_charge_interaction_energy)
        .function("pointChargeInteractionMatrix", &HartreeFock::compute_point_charge_interaction_matrix)
        .function("wolfInteractionMatrix", &HartreeFock::compute_wolf_interaction_matrix)
        .function("nuclearAttractionMatrix", &HartreeFock::compute_nuclear_attraction_matrix)
        .function("nuclearElectricFieldContribution", &HartreeFock::nuclear_electric_field_contribution)
        .function("electronicElectricFieldContribution", &HartreeFock::electronic_electric_field_contribution)
        .function("nuclearElectricPotentialContribution", &HartreeFock::nuclear_electric_potential_contribution)
        .function("electronicElectricPotentialContribution", &HartreeFock::electronic_electric_potential_contribution)
        .function("setDensityFittingBasis", &HartreeFock::set_density_fitting_basis)
        .function("kineticMatrix", &HartreeFock::compute_kinetic_matrix)
        .function("overlapMatrix", &HartreeFock::compute_overlap_matrix)
        .function("overlapMatrixForBasis", &HartreeFock::compute_overlap_matrix_for_basis)
        .function("nuclearRepulsion", &HartreeFock::nuclear_repulsion_energy)
        .function("setPrecision", &HartreeFock::set_precision)
        .function("coulombMatrix", optional_override([](const HartreeFock& hf, const MolecularOrbitals& mo) {
            return hf.compute_J(mo);
        }))
        .function("coulombAndExchangeMatrices", optional_override([](const HartreeFock& hf, const MolecularOrbitals& mo) {
            return hf.compute_JK(mo);
        }))
        .function("fockMatrix", optional_override([](const HartreeFock& hf, const MolecularOrbitals& mo) {
            return hf.compute_fock(mo);
        }))
        .function("toString", optional_override([](const HartreeFock& hf) {
            return std::string("<HartreeFock (") + hf.aobasis().name() + ", " + 
                   std::to_string(hf.atoms().size()) + " atoms)>";
        }));

    // SCF<HartreeFock> class binding
    using HF = SCF<HartreeFock>;
    class_<HF>("HF")
        .constructor<HartreeFock&>()
        .constructor<HartreeFock&, SpinorbitalKind>()
        .property("convergenceSettings", &HF::convergence_settings)
        .function("setChargeMultiplicity", &HF::set_charge_multiplicity)
        .function("setInitialGuess", &HF::set_initial_guess_from_wfn)
        .function("getScfKind", optional_override([](const HF& hf) {
            return std::string(hf.scf_kind());
        }))
        .function("run", &HF::compute_scf_energy)
        .function("computeScfEnergy", &HF::compute_scf_energy)
        .function("wavefunction", &HF::wavefunction)
        .function("toString", optional_override([](const HF& hf) {
            return std::string("<SCF(HF) (") + hf.m_procedure.aobasis().name() + ", " + 
                   std::to_string(hf.m_procedure.atoms().size()) + " atoms)>";
        }));

    // JKPair and JKTriple classes
    class_<JKPair>("JKPair")
        .constructor<>()
        .property("J", &JKPair::J)
        .property("K", &JKPair::K);

    class_<JKTriple>("JKTriple")
        .constructor<>()
        .property("J", &JKTriple::J)
        .property("K", &JKTriple::K);

    // Integral operators enum
    enum_<cint::Operator>("Operator")
        .value("Overlap", cint::Operator::overlap)
        .value("Nuclear", cint::Operator::nuclear)
        .value("Kinetic", cint::Operator::kinetic)
        .value("Coulomb", cint::Operator::coulomb)
        .value("Dipole", cint::Operator::dipole)
        .value("Quadrupole", cint::Operator::quadrupole)
        .value("Octapole", cint::Operator::octapole)
        .value("Hexadecapole", cint::Operator::hexadecapole)
        .value("Rinv", cint::Operator::rinv);

    // IntegralEngine class binding
    class_<IntegralEngine>("IntegralEngine")
        .constructor<const AOBasis&>()
        .constructor<const std::vector<occ::core::Atom>&, const std::vector<Shell>&>()
        .function("schwarz", &IntegralEngine::schwarz)
        .function("setPrecision", &IntegralEngine::set_precision)
        .function("setRangeSeparatedOmega", &IntegralEngine::set_range_separated_omega)
        .function("rangeSeparatedOmega", &IntegralEngine::range_separated_omega)
        .function("isSpherical", &IntegralEngine::is_spherical)
        .function("haveAuxiliaryBasis", &IntegralEngine::have_auxiliary_basis)
        .function("setAuxiliaryBasis", optional_override([](IntegralEngine& engine, const std::vector<Shell>& basis, bool dummy) {
            engine.set_auxiliary_basis(basis, dummy);
        }))
        .function("clearAuxiliaryBasis", &IntegralEngine::clear_auxiliary_basis)
        .function("oneElectronOperator", &IntegralEngine::one_electron_operator)
        .function("coulomb", &IntegralEngine::coulomb)
        .function("coulombAndExchange", &IntegralEngine::coulomb_and_exchange)
        .function("fockOperator", &IntegralEngine::fock_operator)
        .function("pointChargePotential", &IntegralEngine::point_charge_potential)
        .function("electricPotential", &IntegralEngine::electric_potential)
        .function("multipole", &IntegralEngine::multipole)
        .function("nbf", &IntegralEngine::nbf)
        .function("nsh", &IntegralEngine::nsh)
        .function("aobasis", &IntegralEngine::aobasis)
        .function("auxbasis", &IntegralEngine::auxbasis)
        .function("nbfAux", &IntegralEngine::nbf_aux)
        .function("nshAux", &IntegralEngine::nsh_aux)
        .function("toString", optional_override([](const IntegralEngine& engine) {
            return std::string("<IntegralEngine nbf=") + std::to_string(engine.nbf()) + 
                   " nsh=" + std::to_string(engine.nsh()) + " spherical=" + 
                   (engine.is_spherical() ? "true" : "false") + ">";
        }));
}