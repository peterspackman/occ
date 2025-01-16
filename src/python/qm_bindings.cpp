#include "qm_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/qm/expectation.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>

using namespace nb::literals;
using occ::Mat;
using occ::qm::AOBasis;
using occ::qm::HartreeFock;
using occ::qm::MolecularOrbitals;
using occ::qm::SCF;
using occ::qm::Shell;
using occ::qm::Wavefunction;

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;
constexpr auto G = occ::qm::SpinorbitalKind::General;

nb::module_ register_qm_bindings(nb::module_ &parent) {
  auto m =
      parent.def_submodule("qm", "Quantum mechanics functionality for OCC");

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
      .def("__repr__", [](const AOBasis &basis) {
        return fmt::format("<AOBasis ({}) nsh={} nbf={} natoms={}>",
                           basis.name(), basis.nsh(), basis.nbf(),
                           basis.atoms().size());
      });

  nb::class_<MolecularOrbitals>(m, "MolecularOrbitals")
      .def_rw("num_alpha", &MolecularOrbitals::n_alpha)
      .def_rw("num_beta", &MolecularOrbitals::n_beta)
      .def_rw("num_ao", &MolecularOrbitals::n_ao)
      .def_rw("orbital_coeffs", &MolecularOrbitals::C)
      .def_rw("occupied_orbital_coeffs", &MolecularOrbitals::Cocc)
      .def_rw("density_matrix", &MolecularOrbitals::D)
      .def_rw("orbital_energies", &MolecularOrbitals::energies)
      .def("expectation_value", [](const MolecularOrbitals &mo, const Mat &op) {
        return 2 * occ::qm::expectation(mo.kind, mo.D, op);
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
      .def_static("from_molden", [](const std::string &filename) {
        auto reader = occ::io::MoldenReader(filename);
        Wavefunction wfn(reader);
        return wfn;
      });

  using HF = SCF<HartreeFock>;

  nb::class_<HF>(m, "HF")
      .def(nb::init<HartreeFock &>())
      .def("set_charge_multiplicity", &HF::set_charge_multiplicity)
      .def("set_initial_guess", &HF::set_initial_guess_from_wfn)
      .def("scf_kind", &HF::scf_kind)
      .def("run", &HF::compute_scf_energy)
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
      .def("set_density_fitting_basis", &HartreeFock::set_density_fitting_basis)
      .def("kinetic_matrix", &HartreeFock::compute_kinetic_matrix)
      .def("overlap_matrix", &HartreeFock::compute_overlap_matrix)
      .def("overlap_matrix_for_basis",
           &HartreeFock::compute_overlap_matrix_for_basis)
      .def("nuclear_repulsion", &HartreeFock::nuclear_repulsion_energy)
      .def(
          "scf",
          [](HartreeFock &hf, bool unrestricted = false) {
            if (unrestricted)
              return HF(hf, U);
            else
              return HF(hf, R);
          },
          "unrestricted"_a = false)
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

  return m;
}
