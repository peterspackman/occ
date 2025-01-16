#include "core_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/core/dimer.h>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <occ/core/point_charge.h>
#include <occ/io/xyz.h>

using occ::IVec;
using occ::Mat;
using occ::Mat3N;
using occ::Vec3;
using occ::core::Atom;
using occ::core::Dimer;
using occ::core::Element;
using occ::core::Molecule;
using occ::core::PointCharge;

nb::module_ register_core_bindings(nb::module_ &parent) {
  using namespace nb::literals;

  auto m = parent.def_submodule("core", "Core functionality for OCC");

  nb::class_<Element>(m, "Element")
      .def(nb::init<const std::string &>())
      .def("symbol", &Element::symbol,
           "The symbol of the element e.g. H, He ...")
      .def("mass", &Element::mass,
           "Mass number of the element e.g. 12.01 for C")
      .def("name", &Element::name,
           "The name of the element e.g hydrogen, helium...")
      .def("van_der_waals_radius", &Element::van_der_waals_radius,
           "Bondi van der Waals radius for element")
      .def("covalent_radius", &Element::covalent_radius,
           "Covalent radius for element")
      .def("atomic_number", &Element::atomic_number,
           "Atomic number e.g. 1, 2 ...")
      .def("__repr__",
           [](const Element &a) { return "<Element '" + a.symbol() + "'>"; });

  nb::class_<Atom>(m, "Atom")
      .def(nb::init<int, double, double, double>())
      .def_rw("atomic_number", &Atom::atomic_number,
              "Atomic number for corresponding element")
      .def_prop_rw("position", &Atom::position, &Atom::set_position,
                   "Cartesian position of the atom (Bohr)")
      .def("__repr__", [](const Atom &a) {
        return fmt::format("<Atom {} [{:.5f}, {:.5f}, {:.5f}>", a.atomic_number,
                           a.x, a.y, a.z);
      });

  nb::class_<PointCharge>(m, "PointCharge")
      .def(nb::init<double, double, double, double>())
      .def(nb::init<double, std::array<double, 3>>())
      .def(nb::init<double, const occ::Vec3 &>())
      .def_prop_rw("charge", &PointCharge::charge, &PointCharge::set_charge)
      .def_prop_rw("position", &PointCharge::position,
                   &PointCharge::set_position)
      .def("__repr__", [](const PointCharge &pc) {
        const auto &pos = pc.position();
        return fmt::format("<PointCharge q={:.5f} [{:.5f}, {:.5f}, {:.5f}]>",
                           pc.charge(), pos.x(), pos.y(), pos.z());
      });

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
        return fmt::format("<Molecule {} @[{:.5f}, {:.5f}, {:.5f}]>",
                           mol.name(), com.x(), com.y(), com.z());
      });

  nb::class_<Dimer>(m, "Dimer")
      .def(nb::init<const Molecule &, const Molecule &>())
      .def(nb::init<const std::vector<Atom> &, const std::vector<Atom>>())
      .def_prop_ro("a", &Dimer::a)
      .def_prop_ro("b", &Dimer::b)
      .def_prop_rw("name", &Dimer::name, &Dimer::set_name);

  return m;
}
