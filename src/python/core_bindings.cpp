#include "core_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
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

using occ::IVec;
using occ::Mat;
using occ::Mat3;
using occ::Mat3N;
using occ::Mat4;
using occ::Vec3;
using namespace occ::core;
using namespace occ;

nb::module_ register_core_bindings(nb::module_ &m) {
  using namespace nb::literals;

  nb::class_<Element>(m, "Element")
      .def(nb::init<const std::string &>())
      .def(nb::init<int>())
      .def_prop_ro("symbol", &Element::symbol,
                   "The symbol of the element e.g. H, He ...")
      .def_prop_ro("mass", &Element::mass,
                   "Mass number of the element e.g. 12.01 for C")
      .def_prop_ro("name", &Element::name,
                   "The name of the element e.g hydrogen, helium...")
      .def_prop_ro("van_der_waals_radius", &Element::van_der_waals_radius,
                   "Bondi van der Waals radius for element")
      .def_prop_ro("covalent_radius", &Element::covalent_radius,
                   "Covalent radius for element")
      .def_prop_ro("atomic_number", &Element::atomic_number,
                   "Atomic number e.g. 1, 2 ...")
      .def("__gt__", &Element::operator>)
      .def("__lt__", &Element::operator<)
      .def("__eq__", &Element::operator==)
      .def("__ne__", &Element::operator!=)
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

  nb::enum_<Molecule::Origin>(m, "Origin")
      .value("CARTESIAN", Molecule::Origin::Cartesian)
      .value("CENTROID", Molecule::Origin::Centroid)
      .value("CENTEROFMASS", Molecule::Origin::CenterOfMass);

  nb::class_<Molecule>(m, "Molecule")
      .def(nb::init<const IVec &, const Mat3N &>())
      .def("__len__", &Molecule::size)
      .def("elements", &Molecule::elements)
      .def_prop_ro("positions", &Molecule::positions)
      .def_prop_rw("name", &Molecule::name, &Molecule::set_name)
      .def_prop_rw("partial_charges", &Molecule::partial_charges,
                   &Molecule::set_partial_charges)
      .def("esp_partial_charges", &Molecule::esp_partial_charges)
      .def("atomic_masses", &Molecule::atomic_masses)
      .def("atomic_numbers", &Molecule::atomic_numbers)
      .def("vdw_radii", &Molecule::vdw_radii)
      .def("molar_mass", &Molecule::molar_mass)
      .def("atoms", &Molecule::atoms)
      .def("center_of_mass", &Molecule::center_of_mass)
      .def("centroid", &Molecule::centroid)
      .def("rotate",
           nb::overload_cast<const Eigen::Affine3d &, Molecule::Origin>(
               &Molecule::rotate),
           "Rotate molecule in-place about origin", "rotation"_a,
           "origin"_a = Molecule::Origin::Cartesian)
      .def("rotate",
           nb::overload_cast<const Mat3 &, Molecule::Origin>(&Molecule::rotate),
           "Rotate molecule in-place about origin", "rotation"_a,
           "origin"_a = Molecule::Origin::Cartesian)
      .def("rotate",
           nb::overload_cast<const Mat3 &, const Vec3 &>(&Molecule::rotate),
           "Rotate molecule in-place about point", "rotation"_a, "point"_a)

      .def("transform",
           nb::overload_cast<const Mat4 &, Molecule::Origin>(
               &Molecule::transform),
           "Transform molecule in-place about origin", "transform"_a,
           "origin"_a = Molecule::Origin::Cartesian)
      .def("transform",
           nb::overload_cast<const Mat4 &, const Vec3 &>(&Molecule::transform),
           "Transform molecule in-place about point", "transform"_a, "point"_a)
      .def("translate", &Molecule::translate, "Translate molecule in-place",
           "translation"_a)
      .def("rotated",
           nb::overload_cast<const Eigen::Affine3d &, Molecule::Origin>(
               &Molecule::rotated, nb::const_),
           "Return rotated copy about origin", "rotation"_a,
           "origin"_a = Molecule::Origin::Cartesian)
      .def("rotated",
           nb::overload_cast<const Mat3 &, Molecule::Origin>(&Molecule::rotated,
                                                             nb::const_),
           "Return rotated copy about origin", "rotation"_a,
           "origin"_a = Molecule::Origin::Cartesian)
      .def("rotated",
           nb::overload_cast<const Mat3 &, const Vec3 &>(&Molecule::rotated,
                                                         nb::const_),
           "Return rotated copy about point", "rotation"_a, "point"_a)
      .def("transformed",
           nb::overload_cast<const Mat4 &, Molecule::Origin>(
               &Molecule::transformed, nb::const_),
           "Return transformed copy about origin", "transform"_a,
           "origin"_a = Molecule::Origin::Cartesian)
      .def("transformed",
           nb::overload_cast<const Mat4 &, const Vec3 &>(&Molecule::transformed,
                                                         nb::const_),
           "Return transformed copy about point", "transform"_a, "point"_a)
      .def("translated", &Molecule::translated, "Return translated copy",
           "translation"_a)
      .def(
          "centered",
          [](const Molecule &mol,
             Molecule::Origin origin = Molecule::Origin::Centroid) {
            Vec3 center;
            switch (origin) {
            case Molecule::Origin::Centroid:
              center = mol.centroid();
              break;
            case Molecule::Origin::CenterOfMass:
              center = mol.center_of_mass();
              break;
            default:
              center = Vec3::Zero();
            }
            return mol.translated(-center);
          },
          "Return copy centered at origin",
          "origin"_a = Molecule::Origin::Centroid)
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
      .def_prop_ro("nearest_distance", &Dimer::nearest_distance)
      .def_prop_ro("center_of_mass_distance", &Dimer::center_of_mass_distance)
      .def_prop_ro("centroid_distance", &Dimer::centroid_distance)
      .def("symmetry_relation", &Dimer::symmetry_relation)
      .def_prop_rw("name", &Dimer::name, &Dimer::set_name);

  using occ::Mat6;
  using occ::core::ElasticTensor;

  nb::enum_<ElasticTensor::AveragingScheme>(m, "AveragingScheme")
      .value("VOIGT", ElasticTensor::AveragingScheme::Voigt)
      .value("REUSS", ElasticTensor::AveragingScheme::Reuss)
      .value("HILL", ElasticTensor::AveragingScheme::Hill)
      .value("NUMERICAL", ElasticTensor::AveragingScheme::Numerical);

  using ElasticTensorArray =
      nb::ndarray<double, nb::numpy, nb::shape<3, 3, 3, 3>, nb::c_contig>;
  nb::class_<ElasticTensor>(m, "ElasticTensor")
      .def(nb::init<Eigen::Ref<const Mat6>>())
      .def_prop_ro(
          "tensor",
          [](ElasticTensor &self) { return ElasticTensorArray(self.data()); },
          nb::rv_policy::reference_internal)
      .def("youngs_modulus",
           [](const ElasticTensor &self, const Vec3 &d) {
             return self.youngs_modulus(d);
           })
      .def("linear_compressibility",
           [](const ElasticTensor &self, const Vec3 &d) {
             return self.linear_compressibility(d);
           })
      .def("shear_modulus",
           [](const ElasticTensor &self, const Vec3 &d1, const Vec3 &d2) {
             return self.shear_modulus(d1, d2);
           })
      .def("poisson_ratio",
           [](const ElasticTensor &self, const Vec3 &d1, const Vec3 &d2) {
             return self.poisson_ratio(d1, d2);
           })
      .def(
          "youngs_modulus_vec",
          [](const ElasticTensor &self,
             nb::ndarray<double, nb::ndim<2>> directions) {
            size_t n = directions.shape(0);
            std::vector<double> results(n);
            for (size_t i = 0; i < n; i++) {
              Eigen::Vector3d dir(directions(i, 0), directions(i, 1),
                                  directions(i, 2));
              results[i] = self.youngs_modulus(dir);
            }
            return results;
          },
          "Compute Young's modulus for multiple directions")
      .def("shear_modulus_minmax", &ElasticTensor::shear_modulus_minmax)
      .def("poisson_ratio_minmax", &ElasticTensor::poisson_ratio_minmax)

      // Average properties
      .def("average_bulk_modulus", &ElasticTensor::average_bulk_modulus,
           "avg"_a = ElasticTensor::AveragingScheme::Hill)
      .def("average_shear_modulus", &ElasticTensor::average_shear_modulus,
           "avg"_a = ElasticTensor::AveragingScheme::Hill)
      .def("average_youngs_modulus", &ElasticTensor::average_youngs_modulus,
           "avg"_a = ElasticTensor::AveragingScheme::Hill)
      .def("average_poisson_ratio", &ElasticTensor::average_poisson_ratio,
           "avg"_a = ElasticTensor::AveragingScheme::Hill)
      .def_prop_ro("voigt_s", &ElasticTensor::voigt_s)
      .def_prop_ro("voigt_c", &ElasticTensor::voigt_c)
      .def("component", nb::overload_cast<int, int, int, int>(
                            &ElasticTensor::component, nb::const_))

      // Convenience methods for plotting
      .def(
          "compute_directional_properties",
          [](const ElasticTensor &self, int n_theta, int n_phi) {
            // Create arrays for spherical coordinates
            std::vector<double> theta(n_theta), phi(n_phi);
            std::vector<std::vector<double>> youngs(n_theta,
                                                    std::vector<double>(n_phi));
            std::vector<std::vector<double>> linear_comp(
                n_theta, std::vector<double>(n_phi));

            for (int i = 0; i < n_theta; i++) {
              theta[i] = M_PI * i / (n_theta - 1);
              for (int j = 0; j < n_phi; j++) {
                phi[j] = 2 * M_PI * j / (n_phi - 1);

                // Convert to Cartesian coordinates
                Eigen::Vector3d dir;
                dir << sin(theta[i]) * cos(phi[j]), sin(theta[i]) * sin(phi[j]),
                    cos(theta[i]);

                youngs[i][j] = self.youngs_modulus(dir);
                linear_comp[i][j] = self.linear_compressibility(dir);
              }
            }

            return std::make_tuple(theta, phi, youngs, linear_comp);
          },
          "n_theta"_a = 50, "n_phi"_a = 50);

  using occ::core::MirrorType;
  using occ::core::PointGroup;
  nb::enum_<PointGroup>(m, "PointGroup")
      .value("C1", PointGroup::C1)
      .value("Ci", PointGroup::Ci)
      .value("Cs", PointGroup::Cs)
      .value("C2", PointGroup::C2)
      .value("C3", PointGroup::C3)
      .value("C4", PointGroup::C4)
      .value("C5", PointGroup::C5)
      .value("C6", PointGroup::C6)
      .value("C8", PointGroup::C8)
      .value("Coov", PointGroup::Coov)
      .value("Dooh", PointGroup::Dooh)
      .value("C2v", PointGroup::C2v)
      .value("C3v", PointGroup::C3v)
      .value("C4v", PointGroup::C4v)
      .value("C5v", PointGroup::C5v)
      .value("C6v", PointGroup::C6v)
      .value("C2h", PointGroup::C2h)
      .value("C3h", PointGroup::C3h)
      .value("C4h", PointGroup::C4h)
      .value("C5h", PointGroup::C5h)
      .value("C6h", PointGroup::C6h)
      .value("D2", PointGroup::D2)
      .value("D3", PointGroup::D3)
      .value("D4", PointGroup::D4)
      .value("D5", PointGroup::D5)
      .value("D6", PointGroup::D6)
      .value("D7", PointGroup::D7)
      .value("D8", PointGroup::D8)
      .value("D2h", PointGroup::D2h)
      .value("D3h", PointGroup::D3h)
      .value("D4h", PointGroup::D4h)
      .value("D5h", PointGroup::D5h)
      .value("D6h", PointGroup::D6h)
      .value("D7h", PointGroup::D7h)
      .value("D8h", PointGroup::D8h)
      .value("D2d", PointGroup::D2d)
      .value("D3d", PointGroup::D3d)
      .value("D4d", PointGroup::D4d)
      .value("D5d", PointGroup::D5d)
      .value("D6d", PointGroup::D6d)
      .value("D7d", PointGroup::D7d)
      .value("D8d", PointGroup::D8d)
      .value("S4", PointGroup::S4)
      .value("S6", PointGroup::S6)
      .value("S8", PointGroup::S8)
      .value("T", PointGroup::T)
      .value("Td", PointGroup::Td)
      .value("Th", PointGroup::Th)
      .value("O", PointGroup::O)
      .value("Oh", PointGroup::Oh)
      .value("I", PointGroup::I)
      .value("Ih", PointGroup::Ih);

  nb::enum_<MirrorType>(m, "MirrorType")
      .value("None", MirrorType::None)
      .value("H", MirrorType::H)
      .value("D", MirrorType::D)
      .value("V", MirrorType::V);

  nb::class_<SymOp>(m, "SymOp")
      .def(nb::init<>())
      .def(nb::init<const Mat3 &, const Vec3 &>())
      .def(nb::init<const Mat4 &>())
      .def("apply", &SymOp::apply)
      .def_prop_ro("rotation", &SymOp::rotation)
      .def_prop_ro("translation", &SymOp::translation)
      .def_rw("transformation", &SymOp::transformation)
      .def_static("from_rotation_vector", &SymOp::from_rotation_vector)
      .def_static("from_axis_angle", &SymOp::from_axis_angle)
      .def_static("reflection", &SymOp::reflection)
      .def_static("rotoreflection", &SymOp::rotoreflection)
      .def_static("inversion", &SymOp::inversion)
      .def_static("identity", &SymOp::identity);

  nb::class_<MolecularPointGroup>(m, "MolecularPointGroup")
      .def(nb::init<const Molecule &>())
      .def_prop_ro("description", &MolecularPointGroup::description)
      .def_prop_ro("point_group_string",
                   &MolecularPointGroup::point_group_string)
      .def_prop_ro("point_group", &MolecularPointGroup::point_group)
      .def_prop_ro("symops", &MolecularPointGroup::symops)
      .def_prop_ro("rotational_symmetries",
                   &MolecularPointGroup::rotational_symmetries)
      .def_prop_ro("symmetry_number", &MolecularPointGroup::symmetry_number)
      .def("__repr__", [](const MolecularPointGroup &pg) {
        return fmt::format("<MolecularPointGroup '{}'>",
                           pg.point_group_string());
      });

  m.def("dihedral_group", &occ::core::dihedral_group, "order"_a,
        "mirror_type"_a);
  m.def("cyclic_group", &occ::core::cyclic_group, "order"_a, "mirror_type"_a);

  using occ::core::Multipole;
  // Monopole (L=0)
  nb::class_<Multipole<0>>(m, "Monopole")
      .def(nb::init<>())
      .def_prop_ro("num_components",
                   [](const Multipole<0> &m) { return m.components.size(); })
      .def_prop_ro("charge", &Multipole<0>::charge)
      .def_rw("components", &Multipole<0>::components)
      .def("to_string", &Multipole<0>::to_string)
      .def("__add__", &Multipole<0>::operator+ <0>)
      .def("__add__", &Multipole<0>::operator+ <1>)
      .def("__add__", &Multipole<0>::operator+ <2>)
      .def("__add__", &Multipole<0>::operator+ <3>)
      .def("__add__", &Multipole<0>::operator+ <4>)
      .def("__repr__", [](const Multipole<0> &mp) {
        return fmt::format("<Monopole q={:.6f}>", mp.charge());
      });

  // Dipole (L=1)
  nb::class_<Multipole<1>>(m, "Dipole")
      .def(nb::init<>())
      .def_prop_ro("num_components",
                   [](const Multipole<1> &m) { return m.components.size(); })
      .def_prop_ro("charge", &Multipole<1>::charge)
      .def_prop_ro("dipole", &Multipole<1>::dipole)
      .def_rw("components", &Multipole<1>::components)
      .def("to_string", &Multipole<1>::to_string)
      .def("__add__", &Multipole<1>::operator+ <0>)
      .def("__add__", &Multipole<1>::operator+ <1>)
      .def("__add__", &Multipole<1>::operator+ <2>)
      .def("__add__", &Multipole<1>::operator+ <3>)
      .def("__add__", &Multipole<1>::operator+ <4>)
      .def("__repr__", [](const Multipole<1> &mp) {
        auto d = mp.dipole();
        return fmt::format("<Dipole q={:.6f} μ=[{:.6f}, {:.6f}, {:.6f}]>",
                           mp.charge(), d[0], d[1], d[2]);
      });

  // Quadrupole (L=2)
  nb::class_<Multipole<2>>(m, "Quadrupole")
      .def(nb::init<>())
      .def_prop_ro("num_components",
                   [](const Multipole<2> &m) { return m.components.size(); })
      .def_prop_ro("charge", &Multipole<2>::charge)
      .def_prop_ro("dipole", &Multipole<2>::dipole)
      .def_prop_ro("quadrupole", &Multipole<2>::quadrupole)
      .def_rw("components", &Multipole<2>::components)
      .def("to_string", &Multipole<2>::to_string)
      .def("__add__", &Multipole<2>::operator+ <0>)
      .def("__add__", &Multipole<2>::operator+ <1>)
      .def("__add__", &Multipole<2>::operator+ <2>)
      .def("__add__", &Multipole<2>::operator+ <3>)
      .def("__add__", &Multipole<2>::operator+ <4>)
      .def("__repr__", [](const Multipole<2> &mp) {
        return fmt::format("<Quadrupole q={:.6f}>", mp.charge());
      });

  // Octupole (L=3)
  nb::class_<Multipole<3>>(m, "Octupole")
      .def(nb::init<>())
      .def_prop_ro("num_components",
                   [](const Multipole<3> &m) { return m.components.size(); })
      .def_prop_ro("charge", &Multipole<3>::charge)
      .def_prop_ro("dipole", &Multipole<3>::dipole)
      .def_prop_ro("quadrupole", &Multipole<3>::quadrupole)
      .def_prop_ro("octupole", &Multipole<3>::octupole)
      .def_rw("components", &Multipole<3>::components)
      .def("to_string", &Multipole<3>::to_string)
      .def("__add__", &Multipole<3>::operator+ <0>)
      .def("__add__", &Multipole<3>::operator+ <1>)
      .def("__add__", &Multipole<3>::operator+ <2>)
      .def("__add__", &Multipole<3>::operator+ <3>)
      .def("__add__", &Multipole<3>::operator+ <4>)
      .def("__repr__", [](const Multipole<3> &mp) {
        return fmt::format("<Octupole q={:.6f}>", mp.charge());
      });

  // Hexadecapole (L=4)
  nb::class_<Multipole<4>>(m, "Hexadecapole")
      .def(nb::init<>())
      .def_prop_ro("num_components",
                   [](const Multipole<4> &m) { return m.components.size(); })
      .def_prop_ro("charge", &Multipole<4>::charge)
      .def_prop_ro("dipole", &Multipole<4>::dipole)
      .def_prop_ro("quadrupole", &Multipole<4>::quadrupole)
      .def_prop_ro("octupole", &Multipole<4>::octupole)
      .def_prop_ro("hexadecapole", &Multipole<4>::hexadecapole)
      .def_rw("components", &Multipole<4>::components)
      .def("to_string", &Multipole<4>::to_string)
      .def("__add__", &Multipole<4>::operator+ <0>)
      .def("__add__", &Multipole<4>::operator+ <1>)
      .def("__add__", &Multipole<4>::operator+ <2>)
      .def("__add__", &Multipole<4>::operator+ <3>)
      .def("__add__", &Multipole<4>::operator+ <4>)
      .def("__repr__", [](const Multipole<4> &mp) {
        return fmt::format("<Hexadecapole q={:.6f}>", mp.charge());
      });

  // Add this after all the Multipole class bindings

  m.def(
      "Multipole",
      [](int order) -> nb::object {
        switch (order) {
        case 0:
          return nb::cast(Multipole<0>());
        case 1:
          return nb::cast(Multipole<1>());
        case 2:
          return nb::cast(Multipole<2>());
        case 3:
          return nb::cast(Multipole<3>());
        case 4:
          return nb::cast(Multipole<4>());
        default:
          throw std::runtime_error(
              fmt::format("Unsupported multipole order: {}", order));
        }
      },
      "Create a multipole of specified order", "order"_a);

  m.def(
      "Multipole",
      [](int order, const std::vector<double> &components) -> nb::object {
        auto init_multipole = [&components, &order](auto &&mp) {
          if (components.size() != mp.num_components) {
            throw std::runtime_error(
                fmt::format("Expected {} components for order {}, got {}",
                            mp.num_components, order, components.size()));
          }
          std::copy(components.begin(), components.end(),
                    mp.components.begin());
          return mp;
        };

        switch (order) {
        case 0:
          return nb::cast(init_multipole(Multipole<0>()));
        case 1:
          return nb::cast(init_multipole(Multipole<1>()));
        case 2:
          return nb::cast(init_multipole(Multipole<2>()));
        case 3:
          return nb::cast(init_multipole(Multipole<3>()));
        case 4:
          return nb::cast(init_multipole(Multipole<4>()));
        default:
          throw std::runtime_error(
              fmt::format("Unsupported multipole order: {}", order));
        }
      },
      "Create a multipole of specified order with components", "order"_a,
      "components"_a);

  nb::class_<MatTriple>(m, "MatTriple")
      .def(nb::init<>())
      .def_rw("x", &MatTriple::x)
      .def_rw("y", &MatTriple::y)
      .def_rw("z", &MatTriple::z)
      .def("scale_by", &MatTriple::scale_by)
      .def("symmetrize", &MatTriple::symmetrize)
      .def("__add__", &MatTriple::operator+)
      .def("__sub__", &MatTriple::operator-)
      .def("__repr__", [](const MatTriple &mt) {
        return fmt::format("<MatTriple ({}x{})>", mt.x.rows(), mt.x.cols());
      });

  m.def("eem_partial_charges", &occ::core::charges::eem_partial_charges,
        "atomic_numbers"_a, "positions"_a, "_charge"_a = 0.0);

  m.def("eeq_partial_charges", &occ::core::charges::eeq_partial_charges,
        "atomic_numbers"_a, "positions"_a, "_charge"_a = 0.0);

  m.def("eeq_coordination_numbers",
        &occ::core::charges::eeq_coordination_numbers, "atomic_numbers"_a,
        "positions"_a);

  m.def("quasirandom_kgf", &occ::core::quasirandom_kgf, "ndims"_a, "count"_a,
        "seed"_a = 0);

  return m;
}
