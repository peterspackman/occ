#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;
using occ::IVec;
using occ::Mat3N;
using occ::core::Element;
using occ::core::Molecule;

PYBIND11_MODULE(_occpy, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: scikit_build_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

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

    py::class_<Molecule>(m, "Molecule")
        .def(py::init<const IVec &, const Mat3N &>())
        .def("__len__", &Molecule::size)
        .def("elements", &Molecule::elements)
        .def("positions", &Molecule::positions)
        .def_property("name", &Molecule::name, &Molecule::set_name)
        .def("atomic_numbers", &Molecule::atomic_numbers)
        .def("vdw_radii", &Molecule::vdw_radii)
        .def("molar_mass", &Molecule::molar_mass);

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def(
        "subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
