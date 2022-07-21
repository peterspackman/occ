#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
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
using occ::hf::HartreeFock;
using occ::qm::AOBasis;
using occ::qm::Shell;
using occ::scf::SCF;

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;
constexpr auto G = occ::qm::SpinorbitalKind::General;

PYBIND11_MODULE(_occpy, m) {
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

    py::class_<Shell>(m, "Shell");

    py::class_<AOBasis>(m, "AOBasis")
        .def_static("load", &AOBasis::load)
        .def("set_pure", &AOBasis::set_pure);

    using RHF = SCF<HartreeFock, R>;
    py::class_<RHF>(m, "RHF")
        .def("set_charge_multiplicity", &RHF::set_charge_multiplicity)
        .def("run", &RHF::compute_scf_energy);

    using UHF = SCF<HartreeFock, U>;
    py::class_<UHF>(m, "UHF")
        .def("set_charge_multiplicity", &UHF::set_charge_multiplicity)
        .def("run", &UHF::compute_scf_energy);

    using GHF = SCF<HartreeFock, G>;
    py::class_<GHF>(m, "GHF")
        .def("set_charge_multiplicity", &GHF::set_charge_multiplicity)
        .def("run", &GHF::compute_scf_energy);

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

    py::class_<Molecule>(m, "Molecule")
        .def(py::init<const IVec &, const Mat3N &>())
        .def("__len__", &Molecule::size)
        .def("elements", &Molecule::elements)
        .def("positions", &Molecule::positions)
        .def_property("name", &Molecule::name, &Molecule::set_name)
        .def("atomic_numbers", &Molecule::atomic_numbers)
        .def("vdw_radii", &Molecule::vdw_radii)
        .def("molar_mass", &Molecule::molar_mass)
        .def("atoms", &Molecule::atoms);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
