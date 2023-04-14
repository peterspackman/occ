#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/crystal/crystal.h>
#include <occ/dft/dft.h>
#include <occ/io/cifparser.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/moldenreader.h>
#include <occ/io/xyz.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;
using namespace nb::literals;

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
using occ::qm::AOBasis;
using occ::qm::HartreeFock;
using occ::qm::MolecularOrbitals;
using occ::qm::Shell;
using occ::qm::Wavefunction;
using occ::scf::SCF;

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;
constexpr auto G = occ::qm::SpinorbitalKind::General;

NB_MODULE(_occ, m) {
    nb::class_<Element>(m, "Element")
        .def(nb::init<const std::string &>())
        .def("symbol", &Element::symbol)
        .def("mass", &Element::mass)
        .def("name", &Element::name)
        .def("van_der_waals_radius", &Element::van_der_waals_radius)
        .def("covalent_radius", &Element::covalent_radius)
        .def("atomic_number", &Element::atomic_number)
        .def("__repr__", [](const Element &a) {
            return "<occ.Element '" + a.symbol() + "'>";
        });

    nb::class_<Atom>(m, "Atom")
        .def_rw("atomic_number", &Atom::atomic_number)
        .def_prop_rw("position", &Atom::position, &Atom::set_position)
        .def("__repr__", [](const Atom &a) {
            return fmt::format("<occ.Atom {} [{:.5f}, {:.5f}, {:.5f}>",
                               a.atomic_number, a.x, a.y, a.z);
        });

    nb::class_<Shell>(m, "Shell")
        .def_ro("origin", &Shell::origin)
        .def_ro("exponents", &Shell::exponents)
        .def_ro("contraction_coefficients", &Shell::contraction_coefficients)
        .def_ro("contraction_coefficients", &Shell::contraction_coefficients)
        .def("num_contractions", &Shell::num_contractions)
        .def("num_primitives", &Shell::num_primitives)
        .def("norm", &Shell::norm)
        .def("__repr__", [](const Shell &s) {
            return fmt::format("<occ.Shell l={} [{:.5f}, {:.5f}, {:.5f}]>", s.l,
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
            return fmt::format("<occ.AOBasis nsh={} nbf={} natoms={}>",
                               basis.nsh(), basis.nbf(), basis.atoms().size());
        });

    nb::class_<MolecularOrbitals>(m, "MolecularOrbitals")
        .def_rw("num_alpha", &MolecularOrbitals::n_alpha)
        .def_rw("num_beta", &MolecularOrbitals::n_beta)
        .def_rw("num_ao", &MolecularOrbitals::n_ao)
        .def_rw("orbital_coeffs", &MolecularOrbitals::C)
        .def_rw("occupied_orbital_coeffs", &MolecularOrbitals::Cocc)
        .def_rw("density_matrix", &MolecularOrbitals::D)
        .def_rw("orbital_energies", &MolecularOrbitals::energies);

    nb::class_<Wavefunction>(m, "Wavefunction")
        .def_rw("molecular_orbitals", &Wavefunction::mo)
        .def_ro("atoms", &Wavefunction::atoms)
        .def("mulliken_charges", &Wavefunction::mulliken_charges)
        .def("multiplicity", &Wavefunction::multiplicity)
        .def("rotate", &Wavefunction::apply_rotation)
        .def("translate", &Wavefunction::apply_translation)
        .def("transform", &Wavefunction::apply_transformation)
        .def("charge", &Wavefunction::charge)
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
        .def("wavefunction", &HF::wavefunction);

    using KS = SCF<DFT>;

    nb::class_<KS>(m, "KS")
        .def(nb::init<DFT &>())
        .def("set_charge_multiplicity", &KS::set_charge_multiplicity)
        .def("set_initial_guess", &KS::set_initial_guess_from_wfn)
        .def("scf_kind", &KS::scf_kind)
        .def("run", &KS::compute_scf_energy)
        .def("wavefunction", &KS::wavefunction);

    nb::class_<HartreeFock>(m, "HartreeFock")
        .def(nb::init<const AOBasis &>())
        .def("nuclear_attraction_matrix",
             &HartreeFock::compute_nuclear_attraction_matrix)
        .def("set_density_fitting_basis",
             &HartreeFock::set_density_fitting_basis)
        .def("kinetic_matrix", &HartreeFock::compute_kinetic_matrix)
        .def("overlap_matrix", &HartreeFock::compute_overlap_matrix)
        .def("nuclear_repulsion", &HartreeFock::nuclear_repulsion_energy)
        .def("scf", [](HartreeFock &hf) { return HF(hf); })
        .def("coulomb_matrix", &HartreeFock::compute_J, nb::arg("mo"),
             nb::arg("precision") = std::numeric_limits<double>::epsilon(),
             nb::arg("Schwarz") = occ::Mat())
        .def("coulomb_and_exchange_matrices", &HartreeFock::compute_JK,
             nb::arg("mo"),
             nb::arg("precision") = std::numeric_limits<double>::epsilon(),
             nb::arg("Schwarz") = occ::Mat())
        .def("fock_matrix", &HartreeFock::compute_fock, nb::arg("mo"),
             nb::arg("precision") = std::numeric_limits<double>::epsilon(),
             nb::arg("Schwarz") = occ::Mat());

    using occ::dft::AtomGridSettings;
    nb::class_<AtomGridSettings>(m, "AtomGridSettings")
        .def(nb::init<>())
        .def_rw("max_angular_points", &AtomGridSettings::max_angular_points)
        .def_rw("min_angular_points", &AtomGridSettings::min_angular_points)
        .def_rw("radial_points", &AtomGridSettings::radial_points)
        .def_rw("radial_precision", &AtomGridSettings::radial_precision);

    nb::class_<DFT>(m, "DFT")
        .def(nb::init<const std::string &, const AOBasis &,
                      const AtomGridSettings &>())
        .def("nuclear_attraction_matrix",
             &DFT::compute_nuclear_attraction_matrix)
        .def("kinetic_matrix", &DFT::compute_kinetic_matrix)
        .def("overlap_matrix", &DFT::compute_overlap_matrix)
        .def("nuclear_repulsion", &DFT::nuclear_repulsion_energy)
        .def("set_method", &DFT::set_method, nb::arg("method_string"),
             nb::arg("unrestricted") = false)
        .def("set_unrestricted", &DFT::set_unrestricted)
        .def("fock_matrix", &DFT::compute_fock, nb::arg("mo"),
             nb::arg("precision") = std::numeric_limits<double>::epsilon(),
             nb::arg("Schwarz") = occ::Mat())
        .def("scf", [](DFT &dft) { return KS(dft); });

    nb::class_<Molecule>(m, "Molecule")
        .def(nb::init<const IVec &, const Mat3N &>())
        .def("__len__", &Molecule::size)
        .def("elements", &Molecule::elements)
        .def("positions", &Molecule::positions)
        .def_prop_rw("name", &Molecule::name, &Molecule::set_name)
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

    nb::class_<Crystal>(m, "Crystal")
        .def("symmetry_unique_molecules", &Crystal::symmetry_unique_molecules)
        .def("symmetry_unique_dimers", &Crystal::symmetry_unique_dimers)
        .def("unit_cell", &Crystal::unit_cell)
        .def("asymmetric_unit",
             nb::overload_cast<>(&Crystal::asymmetric_unit, nb::const_))
        .def_static("from_cif_file", [](const std::string &filename) {
            occ::io::CifParser parser;
            return parser.parse_crystal(filename).value();
        });

    nb::class_<AsymmetricUnit>(m, "AsymmetricUnit")
        .def_rw("positions", &AsymmetricUnit::positions)
        .def_rw("atomic_numbers", &AsymmetricUnit::atomic_numbers)
        .def_rw("occupations", &AsymmetricUnit::occupations)
        .def_rw("charges", &AsymmetricUnit::charges)
        .def_rw("labels", &AsymmetricUnit::labels)
        .def("__len__", &AsymmetricUnit::size)
        .def("__repr__", [](const AsymmetricUnit &asym) {
            return fmt::format("<occ.AsymmetricUnit {}>",
                               asym.chemical_formula());
        });

    nb::class_<UnitCell>(m, "UnitCell")
        .def_prop_rw("a", &UnitCell::a, &UnitCell::set_a)
        .def_prop_rw("b", &UnitCell::b, &UnitCell::set_b)
        .def_prop_rw("c", &UnitCell::c, &UnitCell::set_c)
        .def_prop_rw("alpha", &UnitCell::alpha, &UnitCell::set_alpha)
        .def_prop_rw("beta", &UnitCell::beta, &UnitCell::set_beta)
        .def_prop_rw("gamma", &UnitCell::gamma, &UnitCell::set_gamma)
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
