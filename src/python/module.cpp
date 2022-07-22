#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/crystal/crystal.h>
#include <occ/dft/dft.h>
#include <occ/io/cifparser.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/xyz.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using occ::IVec;
using occ::Mat3N;
using occ::Vec3;
using occ::core::Atom;
using occ::core::Element;
using occ::core::Molecule;
using occ::crystal::AsymmetricUnit;
using occ::crystal::Crystal;
using occ::crystal::SpaceGroup;
using occ::crystal::SymmetryOperation;
using occ::crystal::UnitCell;
using occ::dft::DFT;
using occ::hf::HartreeFock;
using occ::qm::AOBasis;
using occ::qm::MolecularOrbitals;
using occ::qm::Shell;
using occ::qm::Wavefunction;
using occ::scf::SCF;

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;
constexpr auto G = occ::qm::SpinorbitalKind::General;

PYBIND11_MODULE(_occ, m) {
    py::class_<Element>(m, "Element")
        .def(py::init<const std::string &>())
        .def("symbol", &Element::symbol)
        .def("mass", &Element::mass)
        .def("name", &Element::name)
        .def("vdw", &Element::vdw)
        .def("cov", &Element::cov)
        .def("atomic_number", &Element::n)
        .def("__repr__", [](const Element &a) {
            return "<occ.Element '" + a.symbol() + "'>";
        });

    py::class_<Atom>(m, "Atom")
        .def_readwrite("atomic_number", &Atom::atomic_number)
        .def_property("position", &Atom::position, &Atom::set_position)
        .def("__repr__", [](const Atom &a) {
            return fmt::format("<occ.Atom {} [{:.5f}, {:.5f}, {:.5f}>",
                               a.atomic_number, a.x, a.y, a.z);
        });

    py::class_<Shell>(m, "Shell")
        .def_readonly("origin", &Shell::origin)
        .def_readonly("exponents", &Shell::exponents)
        .def_readonly("contraction_coefficients",
                      &Shell::contraction_coefficients)
        .def_readonly("contraction_coefficients",
                      &Shell::contraction_coefficients)
        .def("num_contractions", &Shell::num_contractions)
        .def("num_primitives", &Shell::num_primitives)
        .def("norm", &Shell::norm)
        .def("__repr__", [](const Shell &s) {
            return fmt::format("<occ.Shell l={} [{:.5f}, {:.5f}, {:.5f}]>", s.l,
                               s.origin(0), s.origin(1), s.origin(2));
        });

    py::class_<AOBasis>(m, "AOBasis")
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
            return fmt::format("<occ.AOBasis nsh={} nbf={} natoms={}>",
                               basis.nsh(), basis.nbf(), basis.atoms().size());
        });

    py::class_<MolecularOrbitals>(m, "MolecularOrbitals")
        .def_readwrite("num_alpha", &MolecularOrbitals::n_alpha)
        .def_readwrite("num_beta", &MolecularOrbitals::n_beta)
        .def_readwrite("num_ao", &MolecularOrbitals::n_ao)
        .def_readwrite("orbital_coeffs", &MolecularOrbitals::C)
        .def_readwrite("occupied_orbital_coeffs", &MolecularOrbitals::Cocc)
        .def_readwrite("density_matrix", &MolecularOrbitals::D)
        .def_readwrite("orbital_energies", &MolecularOrbitals::energies);

    py::class_<Wavefunction>(m, "Wavefunction")
        .def_readwrite("molecular_orbitals", &Wavefunction::mo)
        .def_readonly("atoms", &Wavefunction::atoms)
        .def("mulliken_charges", &Wavefunction::mulliken_charges)
        .def("multiplicity", &Wavefunction::multiplicity)
        .def("rotate", &Wavefunction::apply_rotation)
        .def("translate", &Wavefunction::apply_translation)
        .def("transform", &Wavefunction::apply_transformation)
        .def("charge", &Wavefunction::charge)
        .def_readonly("basis", &Wavefunction::basis)
        .def("to_fchk",
             [](Wavefunction &wfn, const std::string &filename) {
                 auto writer = occ::io::FchkWriter(filename);
                 wfn.save(writer);
                 writer.write();
             })
        .def_static("from_fchk", [](const std::string &filename) {
            auto reader = occ::io::FchkReader(filename);
            Wavefunction wfn(reader);
            return wfn;
        });

    using RHF = SCF<HartreeFock, R>;
    using UHF = SCF<HartreeFock, U>;
    using GHF = SCF<HartreeFock, G>;

    py::class_<RHF>(m, "RHF")
        .def(py::init<HartreeFock &>())
        .def("set_charge_multiplicity", &RHF::set_charge_multiplicity)
        .def("set_initial_guess", &RHF::set_initial_guess_from_wfn)
        .def("scf_kind", &RHF::scf_kind)
        .def("run", &RHF::compute_scf_energy)
        .def("wavefunction", &RHF::wavefunction);

    py::class_<UHF>(m, "UHF")
        .def(py::init<HartreeFock &>())
        .def("set_charge_multiplicity", &UHF::set_charge_multiplicity)
        .def("set_initial_guess", &UHF::set_initial_guess_from_wfn)
        .def("scf_kind", &UHF::scf_kind)
        .def("run", &UHF::compute_scf_energy)
        .def("wavefunction", &UHF::wavefunction);

    py::class_<GHF>(m, "GHF")
        .def(py::init<HartreeFock &>())
        .def("set_charge_multiplicity", &GHF::set_charge_multiplicity)
        .def("set_initial_guess", &GHF::set_initial_guess_from_wfn)
        .def("scf_kind", &GHF::scf_kind)
        .def("run", &GHF::compute_scf_energy)
        .def("wavefunction", &GHF::wavefunction);

    using RKS = SCF<DFT, R>;
    using UKS = SCF<DFT, U>;

    py::class_<RKS>(m, "RKS")
        .def(py::init<DFT &>())
        .def("set_charge_multiplicity", &RKS::set_charge_multiplicity)
        .def("set_initial_guess", &RKS::set_initial_guess_from_wfn)
        .def("scf_kind", &RKS::scf_kind)
        .def("run", &RKS::compute_scf_energy)
        .def("wavefunction", &RKS::wavefunction);

    py::class_<UKS>(m, "UKS")
        .def(py::init<DFT &>())
        .def("set_charge_multiplicity", &UKS::set_charge_multiplicity)
        .def("set_initial_guess", &UKS::set_initial_guess_from_wfn)
        .def("scf_kind", &UKS::scf_kind)
        .def("run", &UKS::compute_scf_energy)
        .def("wavefunction", &UKS::wavefunction);

    py::class_<HartreeFock>(m, "HartreeFock")
        .def(py::init<const AOBasis &>())
        .def("nuclear_attraction_matrix",
             &HartreeFock::compute_nuclear_attraction_matrix)
        .def("kinetic_matrix", &HartreeFock::compute_kinetic_matrix)
        .def("overlap_matrix", &HartreeFock::compute_overlap_matrix)
        .def("nuclear_repulsion", &HartreeFock::nuclear_repulsion_energy)
        .def("rhf", [](HartreeFock &hf) { return RHF(hf); })
        .def("uhf", [](HartreeFock &hf) { return UHF(hf); })
        .def("ghf", [](HartreeFock &hf) { return GHF(hf); });

    py::class_<DFT>(m, "DFT")
        .def(py::init<const std::string &, const AOBasis &>())
        .def("nuclear_attraction_matrix",
             &DFT::compute_nuclear_attraction_matrix)
        .def("kinetic_matrix", &DFT::compute_kinetic_matrix)
        .def("overlap_matrix", &DFT::compute_overlap_matrix)
        .def("nuclear_repulsion", &DFT::nuclear_repulsion_energy)
        .def("rks", [](DFT &dft) { return RKS(dft); })
        .def("uks", [](DFT &dft) { return UKS(dft); });

    py::class_<Molecule>(m, "Molecule")
        .def(py::init<const IVec &, const Mat3N &>())
        .def("__len__", &Molecule::size)
        .def("elements", &Molecule::elements)
        .def("positions", &Molecule::positions)
        .def_property("name", &Molecule::name, &Molecule::set_name)
        .def("atomic_numbers", &Molecule::atomic_numbers)
        .def("vdw_radii", &Molecule::vdw_radii)
        .def("molar_mass", &Molecule::molar_mass)
        .def("atoms", &Molecule::atoms)
        .def("center_of_mass", &Molecule::center_of_mass)
        .def_static("from_xyz_file",
                    [](const std::string &filename) {
                        return occ::io::molecule_from_xyz_file(filename);
                    })
        .def_static("from_xyz_string",
                    [](const std::string &contents) {
                        return occ::io::molecule_from_xyz_string(contents);
                    })
        .def("__repr__", [](const Molecule &mol) {
            auto com = mol.center_of_mass();
            return fmt::format(
                "<occ.Molecule {}, center=[{:.5f}, {:.5f}, {:.5f}]>",
                mol.name(), com.x(), com.y(), com.z());
        });

    // occ::crystal

    py::class_<Crystal>(m, "Crystal")
        .def("symmetry_unique_molecules", &Crystal::symmetry_unique_molecules)
        .def("symmetry_unique_dimers", &Crystal::symmetry_unique_dimers)
        .def("unit_cell", &Crystal::unit_cell)
        .def("asymmetric_unit",
             py::overload_cast<>(&Crystal::asymmetric_unit, py::const_))
        .def_static("from_cif_file", [](const std::string &filename) {
            occ::io::CifParser parser;
            return parser.parse_crystal(filename).value();
        });

    py::class_<AsymmetricUnit>(m, "AsymmetricUnit")
        .def_readwrite("positions", &AsymmetricUnit::positions)
        .def_readwrite("atomic_numbers", &AsymmetricUnit::atomic_numbers)
        .def_readwrite("occupations", &AsymmetricUnit::occupations)
        .def_readwrite("charges", &AsymmetricUnit::charges)
        .def_readwrite("labels", &AsymmetricUnit::labels)
        .def("__len__", &AsymmetricUnit::size)
        .def("__repr__", [](const AsymmetricUnit &asym) {
            return fmt::format("<occ.AsymmetricUnit {}>",
                               asym.chemical_formula());
        });

    py::class_<UnitCell>(m, "UnitCell")
        .def_property("a", &UnitCell::a, &UnitCell::set_a)
        .def_property("b", &UnitCell::b, &UnitCell::set_b)
        .def_property("c", &UnitCell::c, &UnitCell::set_c)
        .def_property("alpha", &UnitCell::alpha, &UnitCell::set_alpha)
        .def_property("beta", &UnitCell::beta, &UnitCell::set_beta)
        .def_property("gamma", &UnitCell::gamma, &UnitCell::set_gamma)
        .def("lengths", &UnitCell::lengths)
        .def("to_fractional", &UnitCell::to_fractional)
        .def("to_cartesian", &UnitCell::to_cartesian)
        .def("cell_type", &UnitCell::cell_type)
        .def("__repr__", [](const UnitCell &uc) {
            return fmt::format("<occ.UnitCell {} ({:.5f}, {:.5f}, {:.5f})>",
                               uc.cell_type(), uc.a(), uc.b(), uc.c());
        });

    m.def("set_num_threads", [](int n) { occ::parallel::set_num_threads(n); });
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
