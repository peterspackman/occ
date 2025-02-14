#include "core_bindings.h"
#include "eigen_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <fmt/core.h>
#include <occ/core/dimer.h>
#include <occ/core/eem.h>
#include <occ/core/eeq.h>
#include <occ/core/elastic_tensor.h>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <occ/core/multipole.h>
#include <occ/core/point_charge.h>
#include <occ/core/point_group.h>
#include <occ/core/quasirandom.h>
#include <occ/io/xyz.h>

using namespace emscripten;
using occ::IVec;
using occ::Mat;
using occ::Mat3;
using occ::Mat3N;
using occ::Mat4;
using occ::Vec3;
using namespace occ::core;
using namespace occ;

namespace occ {
namespace wasm {
namespace detail {

void register_element() {
  class_<Element>("Element")
      .constructor<int>()
      .property("symbol", &Element::symbol)
      .property("mass", &Element::mass)
      .property("name", &Element::name)
      .property("van_der_waals_radius", &Element::van_der_waals_radius)
      .property("covalent_radius", &Element::covalent_radius)
      .property("atomic_number", &Element::atomic_number)
      .function("toString", optional_override([](const Element &self) {
                  return "<Element '" + self.symbol() + "'>";
                }));
}

void register_atom() {
  class_<Atom>("Atom")
      .constructor<int, double, double, double>()
      .property("atomic_number", &Atom::atomic_number)
      .property("position", select_overload<Vec3() const>(&Atom::position),
                select_overload<void(const Vec3 &)>(&Atom::set_position))
      .function("toString", optional_override([](const Atom &self) {
                  return fmt::format("<Atom {} [{:.5f}, {:.5f}, {:.5f}>",
                                     self.atomic_number, self.x, self.y,
                                     self.z);
                }));
}

void register_point_charge() {
  class_<PointCharge>("PointCharge")
      .constructor<double, double, double, double>()
      .property("charge", &PointCharge::charge, &PointCharge::set_charge)
      .property("position", &PointCharge::position, &PointCharge::set_position)
      .function("toString", optional_override([](const PointCharge &self) {
                  const auto &pos = self.position();
                  return fmt::format(
                      "<PointCharge q={:.5f} [{:.5f}, {:.5f}, {:.5f}]>",
                      self.charge(), pos.x(), pos.y(), pos.z());
                }));
}

void register_molecule() {
  enum_<Molecule::Origin>("Origin")
      .value("CARTESIAN", Molecule::Origin::Cartesian)
      .value("CENTROID", Molecule::Origin::Centroid)
      .value("CENTEROFMASS", Molecule::Origin::CenterOfMass);

  class_<Molecule>("Molecule")
      .constructor<const IVec &, const Mat3N &>()
      .function("size", &Molecule::size)
      .function("elements", &Molecule::elements)
      .property("positions", &Molecule::positions)
      .property("name", &Molecule::name, &Molecule::set_name)
      .function("partial_charges", &Molecule::partial_charges)
      .function("esp_partial_charges", &Molecule::esp_partial_charges)
      .function("atomic_masses", &Molecule::atomic_masses)
      .function("atomic_numbers", &Molecule::atomic_numbers)
      .function("vdw_radii", &Molecule::vdw_radii)
      .function("molar_mass", &Molecule::molar_mass)
      .function("atoms", &Molecule::atoms)
      .function("center_of_mass", &Molecule::center_of_mass)
      .function("centroid", &Molecule::centroid)
      .function("rotate", select_overload<void(const Mat3 &, Molecule::Origin)>(
                              &Molecule::rotate))
      .function("rotate", select_overload<void(const Mat3 &, const Vec3 &)>(
                              &Molecule::rotate))
      .function("transform",
                select_overload<void(const Mat4 &, Molecule::Origin)>(
                    &Molecule::transform))
      .function("transform", select_overload<void(const Mat4 &, const Vec3 &)>(
                                 &Molecule::transform))
      .function("translate", &Molecule::translate)
      .function("rotated",
                select_overload<Molecule(const Mat3 &, Molecule::Origin) const>(
                    &Molecule::rotated))
      .function("rotated",
                select_overload<Molecule(const Mat3 &, const Vec3 &) const>(
                    &Molecule::rotated))
      .function("transformed",
                select_overload<Molecule(const Mat4 &, Molecule::Origin) const>(
                    &Molecule::transformed))
      .function("transformed",
                select_overload<Molecule(const Mat4 &, const Vec3 &) const>(
                    &Molecule::transformed))
      .function("translated", &Molecule::translated)
      .function("from_xyz_string",
                optional_override([](const std::string &contents) {
                  return occ::io::molecule_from_xyz_string(contents);
                }))
      .function("translational_free_energy",
                &Molecule::translational_free_energy)
      .function("rotational_free_energy", &Molecule::rotational_free_energy)
      .function("toString", optional_override([](const Molecule &self) {
                  auto com = self.center_of_mass();
                  return fmt::format("<Molecule {} @[{:.5f}, {:.5f}, {:.5f}]>",
                                     self.name(), com.x(), com.y(), com.z());
                }));
}

void register_point_group() {
  enum_<PointGroup>("PointGroup")
      .value("C1", PointGroup::C1)
      .value("Ci", PointGroup::Ci)
      .value("Cs", PointGroup::Cs)
      // ... add all other point groups
      .value("Ih", PointGroup::Ih);

  enum_<MirrorType>("MirrorType")
      .value("None", MirrorType::None)
      .value("H", MirrorType::H)
      .value("D", MirrorType::D)
      .value("V", MirrorType::V);

  class_<MolecularPointGroup>("MolecularPointGroup")
      .constructor<const Molecule &>()
      .property("description",
                optional_override([](const MolecularPointGroup &self) {
                  return std::string(self.description());
                }))
      .property("point_group_string",
                optional_override([](const MolecularPointGroup &self) {
                  return std::string(self.point_group_string());
                }))
      .property("point_group", &MolecularPointGroup::point_group)
      .property("symops", &MolecularPointGroup::symops)
      .property("rotational_symmetries",
                &MolecularPointGroup::rotational_symmetries)
      .property("symmetry_number", &MolecularPointGroup::symmetry_number)
      .function("toString",
                optional_override([](const MolecularPointGroup &self) {
                  return fmt::format("<MolecularPointGroup '{}'>",
                                     self.point_group_string());
                }));
}

template <int L> void register_multipole(const std::string &name) {
  class_<Multipole<L>>(name.data())
      .constructor<>()
      .property("num_components", optional_override([](const Multipole<L> &m) {
                  return m.components.size();
                }))
      .property("charge", &Multipole<L>::charge)
      .property("components", &Multipole<L>::components)
      .function("to_string", &Multipole<L>::to_string);
}

void register_multipoles() {
  register_multipole<0>("Monopole");
  register_multipole<1>("Dipole");
  register_multipole<2>("Quadrupole");
  register_multipole<3>("Octupole");
  register_multipole<4>("Hexadecapole");

  // Register free functions for multipole operations
  function("create_multipole", optional_override([](int order) -> val {
             switch (order) {
             case 0:
               return val(Multipole<0>());
             case 1:
               return val(Multipole<1>());
             case 2:
               return val(Multipole<2>());
             case 3:
               return val(Multipole<3>());
             case 4:
               return val(Multipole<4>());
             default:
               throw std::runtime_error(
                   fmt::format("Unsupported multipole order: {}", order));
             }
           }));
}

} // namespace detail

void register_core_bindings() {
  register_matrix<double, 3, 1>("Vec3");
  register_matrix<double, Eigen::Dynamic, 1>("Vec");
  register_matrix<int, Eigen::Dynamic, 1>("IVec");
  register_matrix<double, 3, 3>("Mat3");
  register_matrix<double, Eigen::Dynamic, Eigen::Dynamic>("Mat");
  register_matrix<double, 3, Eigen::Dynamic>("Mat3N");

  detail::register_element();
  detail::register_atom();
  detail::register_point_charge();
  detail::register_molecule();
  detail::register_point_group();
  detail::register_multipoles();

  // Register free functions
  function("eem_partial_charges", &occ::core::charges::eem_partial_charges);
  function("eeq_partial_charges", &occ::core::charges::eeq_partial_charges);
  function("eeq_coordination_numbers",
           &occ::core::charges::eeq_coordination_numbers);
  function("quasirandom_kgf", &occ::core::quasirandom_kgf);
  function("dihedral_group", &occ::core::dihedral_group);
  function("cyclic_group", &occ::core::cyclic_group);
}

} // namespace wasm
} // namespace occ
