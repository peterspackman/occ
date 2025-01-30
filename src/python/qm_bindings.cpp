#include "qm_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <occ/core/element.h>
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

using namespace nb::literals;
using occ::Mat;
using namespace occ::qm;

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;
constexpr auto G = occ::qm::SpinorbitalKind::General;

inline std::string chemical_formula_from_atoms(const std::vector<Atom> &atoms) {
  using occ::core::Element;
  std::vector<Element> elements;
  elements.reserve(atoms.size());
  for (const auto &atom : atoms) {
    elements.push_back(Element(atom.atomic_number));
  }
  return occ::core::chemical_formula(elements);
}
nb::module_ register_qm_bindings(nb::module_ &m) {

  nb::enum_<SpinorbitalKind>(m, "SpinorbitalKind")
      .value("Restricted", SpinorbitalKind::Restricted)
      .value("Unrestricted", SpinorbitalKind::Unrestricted)
      .value("General", SpinorbitalKind::General);

  nb::class_<Shell>(m, "Shell")
      .def(nb::init<occ::core::PointCharge, double>())
      .def_ro("origin", &Shell::origin, "shell position/origin (Bohr)")
      .def_ro("exponents", &Shell::exponents,
              "array of exponents for primitives in this shell")
      .def_ro("contraction_coefficients", &Shell::contraction_coefficients,
              "array of contraction coefficients for in this shell")
      .def("num_contractions", &Shell::num_contractions,
           "number of contractions")
      .def("num_primitives", &Shell::num_primitives,
           "number of primitive gaussians")
      .def("norm", &Shell::norm, "norm of the shell")
      .def("__repr__", [](const Shell &s) {
        return fmt::format("<Shell l={} [{:.5f}, {:.5f}, {:.5f}]>", s.l,
                           s.origin(0), s.origin(1), s.origin(2));
      });

  nb::class_<AOBasis>(m, "AOBasis")
      .def_static("load", &AOBasis::load)
      .def("shells", &AOBasis::shells)
      .def("set_pure", &AOBasis::set_pure)
      .def("size", &AOBasis::size)
      .def("nbf", &AOBasis::nbf)
      .def("atoms", &AOBasis::atoms)
      .def("first_bf", &AOBasis::first_bf)
      .def("bf_to_shell", &AOBasis::bf_to_shell)
      .def("bf_to_atom", &AOBasis::bf_to_atom)
      .def("shell_to_atom", &AOBasis::shell_to_atom)
      .def("atom_to_shell", &AOBasis::atom_to_shell)
      .def("l_max", &AOBasis::l_max)
      .def("name", &AOBasis::name)
      .def(
          "evaluate",
          [](const AOBasis &basis, const occ::Mat3N &points,
             int max_derivative = 0) {
            if (max_derivative > 2 || max_derivative < 0)
              throw std::runtime_error(
                  "Invalid max derivative (must be 0, 1, 2)");
            return occ::gto::evaluate_basis(basis, points, max_derivative);
          },
          "points"_a, "derivatives"_a = 0)
      .def("__repr__", [](const AOBasis &basis) {
        return fmt::format("<AOBasis ({}) nsh={} nbf={} natoms={}>",
                           basis.name(), basis.nsh(), basis.nbf(),
                           basis.atoms().size());
      });

  nb::class_<MolecularOrbitals>(m, "MolecularOrbitals")
      .def(nb::init<>())
      .def_rw("kind", &MolecularOrbitals::kind)
      .def_rw("num_alpha", &MolecularOrbitals::n_alpha)
      .def_rw("num_beta", &MolecularOrbitals::n_beta)
      .def_rw("num_ao", &MolecularOrbitals::n_ao)
      .def_rw("orbital_coeffs", &MolecularOrbitals::C)
      .def_rw("occupied_orbital_coeffs", &MolecularOrbitals::Cocc)
      .def_rw("density_matrix", &MolecularOrbitals::D)
      .def_rw("orbital_energies", &MolecularOrbitals::energies)
      .def("expectation_value",
           [](const MolecularOrbitals &mo, const Mat &op) {
             return 2 * occ::qm::expectation(mo.kind, mo.D, op);
           })
      .def("__repr__", [](const MolecularOrbitals &mo) {
        return fmt::format(
            "<MolecularOrbitals kind={} nao={} nalpha={} nbeta={}>",
            spinorbital_kind_to_string(mo.kind), mo.n_ao, mo.n_alpha,
            mo.n_beta);
      });

  nb::class_<Wavefunction>(m, "Wavefunction")
      .def_rw("molecular_orbitals", &Wavefunction::mo)
      .def_ro("atoms", &Wavefunction::atoms)
      .def("mulliken_charges", &Wavefunction::mulliken_charges)
      .def("multiplicity", &Wavefunction::multiplicity)
      .def("rotate", &Wavefunction::apply_rotation)
      .def("translate", &Wavefunction::apply_translation)
      .def("transform", &Wavefunction::apply_transformation)
      .def("charge", &Wavefunction::charge)
      .def_static("load", &Wavefunction::load)
      .def("save", nb::overload_cast<const std::string &>(&Wavefunction::save))
      .def_ro("basis", &Wavefunction::basis)
      .def(
          "electron_density",
          [](const Wavefunction &wfn, const occ::Mat3N &points,
             int derivatives = 0) {
            return occ::density::evaluate_density_on_grid(wfn, points,
                                                          derivatives);
          },
          "points"_a, "derivatives"_a = 0)
      .def("chelpg_charges",
           [](const Wavefunction &wfn) { return chelpg_charges(wfn); })
      .def("to_fchk",
           [](Wavefunction &wfn, const std::string &filename) {
             auto writer = occ::io::FchkWriter(filename);
             wfn.save(writer);
             writer.write();
           })
      .def_static("from_fchk",
                  [](const std::string &filename) {
                    auto reader = occ::io::FchkReader(filename);
                    Wavefunction wfn(reader);
                    return wfn;
                  })
      .def_static("from_molden",
                  [](const std::string &filename) {
                    auto reader = occ::io::MoldenReader(filename);
                    Wavefunction wfn(reader);
                    return wfn;
                  })
      .def("__repr__", [](const Wavefunction &wfn) {
        return fmt::format("<Wavefunction {} {}/{} kind={} nbf={} "
                           "charge={}>",
                           chemical_formula_from_atoms(wfn.atoms), wfn.method,
                           wfn.basis.name(),
                           spinorbital_kind_to_string(wfn.mo.kind),
                           wfn.basis.nbf(), wfn.charge());
      });

  using HF = SCF<HartreeFock>;

  nb::class_<SCFConvergenceSettings>(m, "SCFConvergenceSettings")
      .def(nb::init<>())
      .def_rw("energy_threshold", &SCFConvergenceSettings::energy_threshold)
      .def_rw("commutator_threshold",
              &SCFConvergenceSettings::commutator_threshold)
      .def_rw("incremental_fock_threshold",
              &SCFConvergenceSettings::incremental_fock_threshold)
      .def("energy_converged", &SCFConvergenceSettings::energy_converged)
      .def("commutator_converged",
           &SCFConvergenceSettings::commutator_converged)
      .def("energy_and_commutator_converged",
           &SCFConvergenceSettings::energy_and_commutator_converged)
      .def("start_incremental_fock",
           &SCFConvergenceSettings::start_incremental_fock);

  nb::class_<HF>(m, "HF")
      .def(nb::init<HartreeFock &>())
      .def(nb::init<HartreeFock &, SpinorbitalKind>())
      .def_rw("convergence_settings", &HF::convergence_settings)
      .def("set_charge_multiplicity", &HF::set_charge_multiplicity)
      .def("set_initial_guess", &HF::set_initial_guess_from_wfn)
      .def("scf_kind", &HF::scf_kind)
      .def("run", &HF::compute_scf_energy)
      .def("compute_scf_energy", &HF::compute_scf_energy)
      .def("wavefunction", &HF::wavefunction)
      .def("__repr__", [](const HF &hf) {
        return fmt::format("<SCF(HF) ({}, {} atoms)>",
                           hf.m_procedure.aobasis().name(),
                           hf.m_procedure.atoms().size());
      });

  nb::class_<HartreeFock>(m, "HartreeFock")
      .def(nb::init<const AOBasis &>())
      .def("point_charge_interaction_energy",
           &HartreeFock::nuclear_point_charge_interaction_energy)
      .def("wolf_point_charge_interaction_energy",
           &HartreeFock::wolf_point_charge_interaction_energy)
      .def("point_charge_interaction_matrix",
           &HartreeFock::compute_point_charge_interaction_matrix,
           "point_charges"_a, "alpha"_a = 1e16)
      .def("wolf_interaction_matrix",
           &HartreeFock::compute_wolf_interaction_matrix)
      .def("nuclear_attraction_matrix",
           &HartreeFock::compute_nuclear_attraction_matrix)
      .def("nuclear_attraction_matrix",
           &HartreeFock::compute_nuclear_attraction_matrix)
      .def("nuclear_electric_field_contribution",
           &HartreeFock::nuclear_electric_field_contribution)
      .def("electronic_electric_field_contribution",
           &HartreeFock::electronic_electric_field_contribution)
      .def("nuclear_electric_potential_contribution",
           &HartreeFock::nuclear_electric_potential_contribution,
           nb::rv_policy::move)
      .def("electronic_electric_potential_contribution",
           &HartreeFock::electronic_electric_potential_contribution,
           nb::rv_policy::move)
      .def("set_density_fitting_basis", &HartreeFock::set_density_fitting_basis)
      .def("kinetic_matrix", &HartreeFock::compute_kinetic_matrix)
      .def("overlap_matrix", &HartreeFock::compute_overlap_matrix)
      .def("overlap_matrix_for_basis",
           &HartreeFock::compute_overlap_matrix_for_basis)
      .def("nuclear_repulsion", &HartreeFock::nuclear_repulsion_energy)
      .def(
          "scf",
          [](HartreeFock &hf,
             SpinorbitalKind kind = SpinorbitalKind::Restricted) {
            return HF(hf, kind);
          },
          "unrestricted"_a = SpinorbitalKind::Restricted)
      .def("set_precision", &HartreeFock::set_precision)
      .def("coulomb_matrix",
           [](const HartreeFock &hf, const MolecularOrbitals &mo) {
             return hf.compute_J(mo);
           })
      .def("coulomb_and_exchange_matrices",
           [](const HartreeFock &hf, const MolecularOrbitals &mo) {
             return hf.compute_JK(mo);
           })
      .def("fock_matrix",
           [](const HartreeFock &hf, const MolecularOrbitals &mo) {
             return hf.compute_fock(mo);
           })
      .def("__repr__", [](const HartreeFock &hf) {
        return fmt::format("<HartreeFock ({}, {} atoms)>", hf.aobasis().name(),
                           hf.atoms().size());
      });

  nb::class_<JKPair>(m, "JKPair")
      .def(nb::init<>())
      .def_rw("J", &JKPair::J)
      .def_rw("K", &JKPair::K);

  nb::class_<JKTriple>(m, "JKTriple")
      .def(nb::init<>())
      .def_rw("J", &JKTriple::J)
      .def_rw("K", &JKTriple::K);

  nb::enum_<cint::Operator>(m, "Operator")
      .value("Overlap", cint::Operator::overlap)
      .value("Nuclear", cint::Operator::nuclear)
      .value("Kinetic", cint::Operator::kinetic)
      .value("Coulomb", cint::Operator::coulomb)
      .value("Dipole", cint::Operator::dipole)
      .value("Quadrupole", cint::Operator::quadrupole)
      .value("Octapole", cint::Operator::octapole)
      .value("Hexadecapole", cint::Operator::hexadecapole)
      .value("Rinv", cint::Operator::rinv);

  nb::class_<IntegralEngine>(m, "IntegralEngine")
      .def(nb::init<const AOBasis &>())
      .def(nb::init<const std::vector<occ::core::Atom> &,
                    const std::vector<Shell> &>())
      .def("schwarz", &IntegralEngine::schwarz)
      .def("set_precision", &IntegralEngine::set_precision)
      .def("set_range_separated_omega",
           &IntegralEngine::set_range_separated_omega)
      .def("range_separated_omega", &IntegralEngine::range_separated_omega)
      .def("is_spherical", &IntegralEngine::is_spherical)
      .def("have_auxiliary_basis", &IntegralEngine::have_auxiliary_basis)
      .def("set_auxiliary_basis",
           nb::overload_cast<const std::vector<Shell> &, bool>(
               &IntegralEngine::set_auxiliary_basis),
           "basis"_a, "dummy"_a = false)
      .def("clear_auxiliary_basis", &IntegralEngine::clear_auxiliary_basis)
      .def("one_electron_operator", &IntegralEngine::one_electron_operator,
           "operator"_a, "use_shellpair_list"_a = true)
      .def("coulomb", &IntegralEngine::coulomb, "kind"_a, "mo"_a,
           "schwarz"_a = Mat())
      .def("coulomb_and_exchange", &IntegralEngine::coulomb_and_exchange,
           "kind"_a, "mo"_a, "schwarz"_a = Mat())
      .def("fock_operator", &IntegralEngine::fock_operator, "kind"_a, "mo"_a,
           "schwarz"_a = Mat())
      .def("point_charge_potential", &IntegralEngine::point_charge_potential,
           "charges"_a, "alpha"_a = 1e16)
      .def("electric_potential", &IntegralEngine::electric_potential)
      .def("multipole", &IntegralEngine::multipole, "order"_a, "mo"_a,
           "origin"_a = occ::Vec3::Zero())
      .def("nbf", &IntegralEngine::nbf)
      .def("nsh", &IntegralEngine::nsh)
      .def("aobasis", &IntegralEngine::aobasis)
      .def("auxbasis", &IntegralEngine::auxbasis)
      .def("nbf_aux", &IntegralEngine::nbf_aux)
      .def("nsh_aux", &IntegralEngine::nsh_aux)
      .def("one_electron_operator_grad",
           &IntegralEngine::one_electron_operator_grad, "operator"_a,
           "use_shellpair_list"_a = true)
      .def("fock_operator_grad", &IntegralEngine::fock_operator_grad, "kind"_a,
           "mo"_a, "schwarz"_a = Mat())
      .def("coulomb_grad", &IntegralEngine::coulomb_grad, "kind"_a, "mo"_a,
           "schwarz"_a = Mat())
      .def("coulomb_exchange_grad", &IntegralEngine::coulomb_exchange_grad,
           "kind"_a, "mo"_a, "schwarz"_a = Mat())
      .def("fock_operator_mixed_basis",
           &IntegralEngine::fock_operator_mixed_basis, "density_matrix"_a,
           "density_basis"_a, "is_shell_diagonal"_a)
      .def("coulomb_list", &IntegralEngine::coulomb_list, "kind"_a,
           "molecular_orbitals"_a, "schwarz"_a = Mat())
      .def("coulomb_and_exchange_list",
           &IntegralEngine::coulomb_and_exchange_list, "kind"_a,
           "molecular_orbitals"_a, "schwarz"_a = Mat())
      .def("effective_core_potential",
           &IntegralEngine::effective_core_potential,
           "use_shellpair_list"_a = true)
      .def("have_effective_core_potentials",
           &IntegralEngine::have_effective_core_potentials)
      .def("set_effective_core_potentials",
           &IntegralEngine::set_effective_core_potentials, "ecp_shells"_a,
           "ecp_electrons"_a)
      .def("rinv_operator_atom_center",
           &IntegralEngine::rinv_operator_atom_center, "atom_index"_a,
           "use_shellpair_list"_a = true)
      .def("rinv_operator_grad_atom", &IntegralEngine::rinv_operator_grad_atom,
           "atom_index"_a, "use_shellpair_list"_a = true)

      .def("wolf_point_charge_potential",
           &IntegralEngine::wolf_point_charge_potential, "charges"_a,
           "partial_charges"_a, "alpha"_a, "cutoff_radius"_a)

      .def("__repr__", [](const IntegralEngine &engine) {
        return fmt::format("<IntegralEngine nbf={} nsh={} spherical={}>",
                           engine.nbf(), engine.nsh(),
                           engine.is_spherical() ? "true" : "false");
      });

  nb::enum_<OrbitalSmearing::Kind>(m, "OrbitalSmearingKind")
      .value("None", OrbitalSmearing::Kind::None)
      .value("Fermi", OrbitalSmearing::Kind::Fermi)
      .value("Gaussian", OrbitalSmearing::Kind::Gaussian)
      .value("Linear", OrbitalSmearing::Kind::Linear);

  nb::class_<OrbitalSmearing>(m, "OrbitalSmearing")
      .def(nb::init<>())
      .def_rw("kind", &OrbitalSmearing::kind)
      .def_rw("mu", &OrbitalSmearing::mu)
      .def_rw("fermi_level", &OrbitalSmearing::fermi_level)
      .def_rw("sigma", &OrbitalSmearing::sigma)
      .def_rw("entropy", &OrbitalSmearing::entropy)
      .def("smear_orbitals", &OrbitalSmearing::smear_orbitals)
      .def("calculate_entropy", &OrbitalSmearing::calculate_entropy)
      .def("ec_entropy", &OrbitalSmearing::ec_entropy)
      .def("calculate_fermi_occupations",
           &OrbitalSmearing::calculate_fermi_occupations)
      .def("calculate_gaussian_occupations",
           &OrbitalSmearing::calculate_gaussian_occupations)
      .def("calculate_linear_occupations",
           &OrbitalSmearing::calculate_linear_occupations)
      .def("__repr__", [](const OrbitalSmearing &os) {
        const char *kind_str;
        switch (os.kind) {
        case OrbitalSmearing::Kind::None:
          kind_str = "None";
          break;
        case OrbitalSmearing::Kind::Fermi:
          kind_str = "Fermi";
          break;
        case OrbitalSmearing::Kind::Gaussian:
          kind_str = "Gaussian";
          break;
        case OrbitalSmearing::Kind::Linear:
          kind_str = "Linear";
          break;
        }
        return fmt::format("<OrbitalSmearing kind={} sigma={:.6f} mu={:.6f}>",
                           kind_str, os.sigma, os.mu);
      });

  nb::enum_<IntegralEngineDF::Policy>(m, "IntegralEngineDFPolicy")
      .value("Choose", IntegralEngineDF::Policy::Choose)
      .value("Direct", IntegralEngineDF::Policy::Direct)
      .value("Stored", IntegralEngineDF::Policy::Stored);

  nb::class_<IntegralEngineDF>(m, "IntegralEngineDF")
      .def(nb::init<const std::vector<occ::core::Atom> &,
                    const std::vector<Shell> &, const std::vector<Shell> &>())
      .def("exchange", &IntegralEngineDF::exchange)
      .def("coulomb", &IntegralEngineDF::coulomb)
      .def("coulomb_and_exchange", &IntegralEngineDF::coulomb_and_exchange)
      .def("fock_operator", &IntegralEngineDF::fock_operator)
      .def("set_integral_policy", &IntegralEngineDF::set_integral_policy)
      .def("set_range_separated_omega",
           &IntegralEngineDF::set_range_separated_omega)
      .def("set_precision", &IntegralEngineDF::set_precision)
      .def("__repr__", [](const IntegralEngineDF &engine) {
        const char *policy_str;
        switch (engine.integral_policy()) {
        case IntegralEngineDF::Policy::Choose:
          policy_str = "Choose";
          break;
        case IntegralEngineDF::Policy::Direct:
          policy_str = "Direct";
          break;
        case IntegralEngineDF::Policy::Stored:
          policy_str = "Stored";
          break;
        }
        return fmt::format("<IntegralEngineDF policy={} precision={:.2e}>",
                           policy_str, engine.precision());
      });

  using occ::gto::GTOValues;
  nb::class_<GTOValues>(m, "GTOValues")
      .def(nb::init<>())
      .def("reserve", &GTOValues::reserve)
      .def("set_zero", &GTOValues::set_zero)
      .def_rw("phi", &GTOValues::phi)
      .def_rw("phi_x", &GTOValues::phi_x)
      .def_rw("phi_y", &GTOValues::phi_y)
      .def_rw("phi_z", &GTOValues::phi_z)
      .def_rw("phi_xx", &GTOValues::phi_xx)
      .def_rw("phi_xy", &GTOValues::phi_xy)
      .def_rw("phi_xz", &GTOValues::phi_xz)
      .def_rw("phi_yy", &GTOValues::phi_yy)
      .def_rw("phi_yz", &GTOValues::phi_yz)
      .def_rw("phi_zz", &GTOValues::phi_zz);

  return m;
}
