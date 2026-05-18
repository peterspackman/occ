#include "core_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/core/dimer.h>
#include <occ/core/eem.h>
#include <occ/core/eeq.h>
#include <occ/core/elastic_tensor.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/multipole.h>
#include <occ/core/point_charge.h>
#include <occ/core/point_group.h>
#include <occ/core/quasirandom.h>
#include <occ/crystal/crystal.h>
#include <occ/io/xyz.h>
#include <spdlog/spdlog.h>

namespace occ::lua_bindings {

using occ::IVec;
using occ::Mat;
using occ::Mat3;
using occ::Mat3N;
using occ::Mat4;
using occ::Mat6;
using occ::Vec;
using occ::Vec3;
using namespace occ::core;
using occ::crystal::Crystal;

namespace {

// -- Element / Atom / PointCharge -----------------------------------------

void register_atomic_types(sol::table &m) {
  m.new_usertype<Element>(
      "Element",
      sol::call_constructor,
      sol::factories([](const std::string &symbol) { return Element(symbol); },
                     [](int atomic_number) { return Element(atomic_number); }),
      "symbol", sol::readonly_property(&Element::symbol),
      "mass", sol::readonly_property(&Element::mass),
      "name", sol::readonly_property(&Element::name),
      "van_der_waals_radius",
      sol::readonly_property(&Element::van_der_waals_radius),
      "covalent_radius", sol::readonly_property(&Element::covalent_radius),
      "atomic_number", sol::readonly_property(&Element::atomic_number),
      sol::meta_function::less_than,
      [](const Element &a, const Element &b) { return a < b; },
      sol::meta_function::equal_to,
      [](const Element &a, const Element &b) { return a == b; },
      sol::meta_function::to_string, [](const Element &a) {
        return fmt::format("<Element '{}'>", a.symbol());
      });

  m.new_usertype<Atom>(
      "Atom",
      sol::call_constructor,
      sol::constructors<Atom(int, double, double, double)>(),
      "atomic_number", &Atom::atomic_number,
      "x", &Atom::x,
      "y", &Atom::y,
      "z", &Atom::z,
      "position",
      sol::property(
          [](const Atom &a, sol::this_state s) {
            return Vec3(a.x, a.y, a.z);
          },
          [](Atom &a, const sol::table &t) {
            a.set_position(table_to_vec3(t));
          }),
      sol::meta_function::to_string, [](const Atom &a) {
        return fmt::format("<Atom {} [{:.5f}, {:.5f}, {:.5f}]>",
                           a.atomic_number, a.x, a.y, a.z);
      });

  m.new_usertype<PointCharge>(
      "PointCharge",
      sol::call_constructor,
      sol::factories(
          [](double q, double x, double y, double z) {
            return PointCharge(q, x, y, z);
          },
          [](double q, const sol::table &pos) {
            return PointCharge(q, table_to_vec3(pos));
          }),
      "charge", sol::property(&PointCharge::charge, &PointCharge::set_charge),
      "position",
      sol::property(
          [](const PointCharge &pc, sol::this_state s) {
            return pc.position();
          },
          [](PointCharge &pc, const sol::table &t) {
            pc.set_position(table_to_vec3(t));
          }),
      sol::meta_function::to_string, [](const PointCharge &pc) {
        const auto &p = pc.position();
        return fmt::format("<PointCharge q={:.5f} [{:.5f}, {:.5f}, {:.5f}]>",
                           pc.charge(), p.x(), p.y(), p.z());
      });
}

// -- Molecule / Dimer -----------------------------------------------------

void register_molecule_and_dimer(sol::table &m) {
  m.new_enum<Molecule::Origin>(
      "Origin", {{"CARTESIAN", Molecule::Origin::Cartesian},
                 {"CENTROID", Molecule::Origin::Centroid},
                 {"CENTEROFMASS", Molecule::Origin::CenterOfMass}});

  m.new_usertype<Molecule>(
      "Molecule",
      sol::call_constructor,
      sol::factories(
          // Construct from atomic-number list + 3xN positions (rows are
          // accepted in either orientation by table_to_mat3n).
          [](const sol::table &atomic_numbers, const sol::table &positions) {
            const int n = static_cast<int>(atomic_numbers.size());
            IVec z(n);
            for (int i = 0; i < n; ++i) z(i) = atomic_numbers.get<int>(i + 1);
            return Molecule(z, table_to_mat3n(positions));
          }),
      sol::meta_function::length, &Molecule::size,
      "size", &Molecule::size,
      "elements", &Molecule::elements,
      "positions",
      sol::readonly_property([](const Molecule &mol, sol::this_state s) {
        return mol.positions();
      }),
      "name", sol::property(&Molecule::name, &Molecule::set_name),
      "partial_charges",
      sol::property(
          [](const Molecule &mol, sol::this_state s) {
            return mol.partial_charges();
          },
          [](Molecule &mol, const sol::table &t) {
            mol.set_partial_charges(table_to_vecx(t));
          }),
      "esp_partial_charges",
      [](const Molecule &mol, const Vec &charges, sol::this_state s) {
        return mol.esp_partial_charges(charges);
      },
      "atomic_masses",
      [](const Molecule &mol, sol::this_state s) {
        return mol.atomic_masses();
      },
      "atomic_numbers",
      [](const Molecule &mol, sol::this_state s) {
        return mol.atomic_numbers();
      },
      "vdw_radii",
      [](const Molecule &mol, sol::this_state s) {
        return mol.vdw_radii();
      },
      "molar_mass", &Molecule::molar_mass,
      "atoms", &Molecule::atoms,
      "bonds", &Molecule::bonds,
      "center_of_mass",
      [](const Molecule &mol, sol::this_state s) {
        return mol.center_of_mass();
      },
      "centroid",
      [](const Molecule &mol, sol::this_state s) {
        return mol.centroid();
      },
      "unit_cell_molecule_idx", &Molecule::unit_cell_molecule_idx,
      "asymmetric_molecule_idx", &Molecule::asymmetric_molecule_idx,
      "asymmetric_unit_idx",
      [](const Molecule &mol, sol::this_state s) {
        return mol.asymmetric_unit_idx();
      },
      "is_equivalent_to", &Molecule::is_equivalent_to,
      // -- In-place transforms (return self for chaining? — kept void to
      // match the Python binding)
      "rotate",
      sol::overload(
          [](Molecule &mol, const sol::table &rotation,
             Molecule::Origin origin) {
            mol.rotate(table_to_mat3(rotation), origin);
          },
          [](Molecule &mol, const sol::table &rotation,
             const sol::table &point) {
            mol.rotate(table_to_mat3(rotation), table_to_vec3(point));
          }),
      "transform",
      sol::overload(
          [](Molecule &mol, const sol::table &t, Molecule::Origin origin) {
            mol.transform(table_to_mat4(t), origin);
          },
          [](Molecule &mol, const sol::table &t, const sol::table &point) {
            mol.transform(table_to_mat4(t), table_to_vec3(point));
          }),
      "translate",
      [](Molecule &mol, const sol::table &translation) {
        mol.translate(table_to_vec3(translation));
      },
      // -- Copy-returning variants
      "rotated",
      sol::overload(
          [](const Molecule &mol, const sol::table &rotation,
             Molecule::Origin origin) {
            return mol.rotated(table_to_mat3(rotation), origin);
          },
          [](const Molecule &mol, const sol::table &rotation,
             const sol::table &point) {
            return mol.rotated(table_to_mat3(rotation), table_to_vec3(point));
          }),
      "transformed",
      sol::overload(
          [](const Molecule &mol, const sol::table &t,
             Molecule::Origin origin) {
            return mol.transformed(table_to_mat4(t), origin);
          },
          [](const Molecule &mol, const sol::table &t,
             const sol::table &point) {
            return mol.transformed(table_to_mat4(t), table_to_vec3(point));
          }),
      "translated",
      [](const Molecule &mol, const sol::table &translation) {
        return mol.translated(table_to_vec3(translation));
      },
      "centered",
      [](const Molecule &mol, sol::optional<Molecule::Origin> origin) {
        Vec3 center;
        switch (origin.value_or(Molecule::Origin::Centroid)) {
        case Molecule::Origin::Centroid: center = mol.centroid(); break;
        case Molecule::Origin::CenterOfMass:
          center = mol.center_of_mass();
          break;
        default: center = Vec3::Zero();
        }
        return mol.translated(-center);
      },
      "translational_free_energy", &Molecule::translational_free_energy,
      "rotational_free_energy", &Molecule::rotational_free_energy,
      sol::meta_function::to_string, [](const Molecule &mol) {
        auto com = mol.center_of_mass();
        return fmt::format("<Molecule {} @[{:.5f}, {:.5f}, {:.5f}]>",
                           mol.name(), com.x(), com.y(), com.z());
      });

  // Static factories on Molecule mirror the Python `from_xyz_file` /
  // `from_xyz_string`. Lua doesn't have a static-method idiom; expose
  // them as plain free functions in occ.molecule_from_xyz_*.
  m.set_function("molecule_from_xyz_file", [](const std::string &filename) {
    return occ::io::molecule_from_xyz_file(filename);
  });
  m.set_function("molecule_from_xyz_string", [](const std::string &contents) {
    return occ::io::molecule_from_xyz_string(contents);
  });

  m.new_usertype<Dimer>(
      "Dimer",
      sol::call_constructor,
      sol::factories(
          [](const Molecule &a, const Molecule &b) { return Dimer(a, b); },
          [](const std::vector<Atom> &a, const std::vector<Atom> &b) {
            return Dimer(a, b);
          }),
      "a", sol::readonly_property(&Dimer::a),
      "b", sol::readonly_property(&Dimer::b),
      "nearest_distance", sol::readonly_property(&Dimer::nearest_distance),
      "center_of_mass_distance",
      sol::readonly_property(&Dimer::center_of_mass_distance),
      "centroid_distance", sol::readonly_property(&Dimer::centroid_distance),
      // optional<Mat4> needs an explicit wrapper — sol2's default container
      // dispatch tries to iterate Eigen matrices and fails to compile.
      "symmetry_relation",
      [](const Dimer &d, sol::this_state s) -> sol::object {
        auto rel = d.symmetry_relation();
        if (!rel) return sol::nil;
        return sol::make_object(s, mat_to_table(s, *rel));
      },
      "name", sol::property(&Dimer::name, &Dimer::set_name),
      "v_ab_com", [](const Dimer &d, sol::this_state s) {
        return d.v_ab_com();
      });
}

// -- Elastic tensor -------------------------------------------------------

void register_elastic_tensor(sol::table &m) {
  m.new_enum<ElasticTensor::AveragingScheme>(
      "AveragingScheme", {{"VOIGT", ElasticTensor::AveragingScheme::Voigt},
                          {"REUSS", ElasticTensor::AveragingScheme::Reuss},
                          {"HILL", ElasticTensor::AveragingScheme::Hill},
                          {"NUMERICAL",
                           ElasticTensor::AveragingScheme::Numerical}});

  m.new_usertype<ElasticTensor>(
      "ElasticTensor",
      sol::call_constructor,
      sol::factories([](const sol::table &voigt_c) {
        // Accept a 6x6 row-major Voigt table.
        Mat6 c;
        for (int i = 0; i < 6; ++i) {
          sol::table row = voigt_c.get<sol::table>(i + 1);
          for (int j = 0; j < 6; ++j) c(i, j) = row.get<double>(j + 1);
        }
        return ElasticTensor(c);
      }),
      "youngs_modulus",
      [](const ElasticTensor &self, const sol::table &d) {
        return self.youngs_modulus(table_to_vec3(d));
      },
      "linear_compressibility",
      [](const ElasticTensor &self, const sol::table &d) {
        return self.linear_compressibility(table_to_vec3(d));
      },
      "shear_modulus",
      [](const ElasticTensor &self, const sol::table &d1,
         const sol::table &d2) {
        return self.shear_modulus(table_to_vec3(d1), table_to_vec3(d2));
      },
      "poisson_ratio",
      [](const ElasticTensor &self, const sol::table &d1,
         const sol::table &d2) {
        return self.poisson_ratio(table_to_vec3(d1), table_to_vec3(d2));
      },
      "shear_modulus_minmax",
      [](const ElasticTensor &self, const sol::table &d, sol::this_state s) {
        auto pair = self.shear_modulus_minmax(table_to_vec3(d));
        sol::state_view lua(s);
        sol::table t = lua.create_table(2, 0);
        t[1] = pair.first;
        t[2] = pair.second;
        return t;
      },
      "poisson_ratio_minmax",
      [](const ElasticTensor &self, const sol::table &d, sol::this_state s) {
        auto pair = self.poisson_ratio_minmax(table_to_vec3(d));
        sol::state_view lua(s);
        sol::table t = lua.create_table(2, 0);
        t[1] = pair.first;
        t[2] = pair.second;
        return t;
      },
      "average_bulk_modulus",
      [](const ElasticTensor &self,
         sol::optional<ElasticTensor::AveragingScheme> scheme) {
        return self.average_bulk_modulus(
            scheme.value_or(ElasticTensor::AveragingScheme::Hill));
      },
      "average_shear_modulus",
      [](const ElasticTensor &self,
         sol::optional<ElasticTensor::AveragingScheme> scheme) {
        return self.average_shear_modulus(
            scheme.value_or(ElasticTensor::AveragingScheme::Hill));
      },
      "average_youngs_modulus",
      [](const ElasticTensor &self,
         sol::optional<ElasticTensor::AveragingScheme> scheme) {
        return self.average_youngs_modulus(
            scheme.value_or(ElasticTensor::AveragingScheme::Hill));
      },
      "average_poisson_ratio",
      [](const ElasticTensor &self,
         sol::optional<ElasticTensor::AveragingScheme> scheme) {
        return self.average_poisson_ratio(
            scheme.value_or(ElasticTensor::AveragingScheme::Hill));
      },
      "average_poisson_ratio_direction",
      [](const ElasticTensor &self, const sol::table &direction,
         sol::optional<int> num_samples) {
        return self.average_poisson_ratio_direction(
            table_to_vec3(direction), num_samples.value_or(360));
      },
      "reduced_youngs_modulus",
      [](const ElasticTensor &self, const sol::table &direction,
         sol::optional<int> num_samples) {
        return self.reduced_youngs_modulus(table_to_vec3(direction),
                                           num_samples.value_or(360));
      },
      "transverse_acoustic_velocity",
      &ElasticTensor::transverse_acoustic_velocity,
      "longitudinal_acoustic_velocity",
      &ElasticTensor::longitudinal_acoustic_velocity,
      "voigt_s",
      [](const ElasticTensor &self, sol::this_state s) {
        return self.voigt_s();
      },
      "voigt_c",
      [](const ElasticTensor &self, sol::this_state s) {
        return self.voigt_c();
      },
      "component",
      [](const ElasticTensor &self, int i, int j, int k, int l) {
        return self.component(i, j, k, l);
      },
      "eigenvalues",
      [](const ElasticTensor &self, sol::this_state s) {
        return self.eigenvalues();
      },
      "voigt_rotation_matrix",
      [](const ElasticTensor &self, const sol::table &rotation,
         sol::this_state s) {
        return mat_to_table(s,
                            self.voigt_rotation_matrix(table_to_mat3(rotation)));
      },
      "rotate_voigt_stiffness",
      [](const ElasticTensor &self, const sol::table &rotation,
         sol::this_state s) {
        return mat_to_table(s,
                            self.rotate_voigt_stiffness(table_to_mat3(rotation)));
      },
      "rotate_voigt_compliance",
      [](const ElasticTensor &self, const sol::table &rotation,
         sol::this_state s) {
        return mat_to_table(
            s, self.rotate_voigt_compliance(table_to_mat3(rotation)));
      });
}

// -- Point groups / symmetry ----------------------------------------------

void register_point_groups(sol::table &m) {
  m.new_enum<PointGroup>(
      "PointGroup",
      {{"C1", PointGroup::C1},     {"Ci", PointGroup::Ci},
       {"Cs", PointGroup::Cs},     {"C2", PointGroup::C2},
       {"C3", PointGroup::C3},     {"C4", PointGroup::C4},
       {"C5", PointGroup::C5},     {"C6", PointGroup::C6},
       {"C8", PointGroup::C8},     {"Coov", PointGroup::Coov},
       {"Dooh", PointGroup::Dooh}, {"C2v", PointGroup::C2v},
       {"C3v", PointGroup::C3v},   {"C4v", PointGroup::C4v},
       {"C5v", PointGroup::C5v},   {"C6v", PointGroup::C6v},
       {"C2h", PointGroup::C2h},   {"C3h", PointGroup::C3h},
       {"C4h", PointGroup::C4h},   {"C5h", PointGroup::C5h},
       {"C6h", PointGroup::C6h},   {"D2", PointGroup::D2},
       {"D3", PointGroup::D3},     {"D4", PointGroup::D4},
       {"D5", PointGroup::D5},     {"D6", PointGroup::D6},
       {"D7", PointGroup::D7},     {"D8", PointGroup::D8},
       {"D2h", PointGroup::D2h},   {"D3h", PointGroup::D3h},
       {"D4h", PointGroup::D4h},   {"D5h", PointGroup::D5h},
       {"D6h", PointGroup::D6h},   {"D7h", PointGroup::D7h},
       {"D8h", PointGroup::D8h},   {"D2d", PointGroup::D2d},
       {"D3d", PointGroup::D3d},   {"D4d", PointGroup::D4d},
       {"D5d", PointGroup::D5d},   {"D6d", PointGroup::D6d},
       {"D7d", PointGroup::D7d},   {"D8d", PointGroup::D8d},
       {"S4", PointGroup::S4},     {"S6", PointGroup::S6},
       {"S8", PointGroup::S8},     {"T", PointGroup::T},
       {"Td", PointGroup::Td},     {"Th", PointGroup::Th},
       {"O", PointGroup::O},       {"Oh", PointGroup::Oh},
       {"I", PointGroup::I},       {"Ih", PointGroup::Ih}});

  m.new_enum<MirrorType>(
      "MirrorType",
      {{"None_", MirrorType::None},  // `none` is reserved in some Lua dialects
       {"H", MirrorType::H},
       {"D", MirrorType::D},
       {"V", MirrorType::V}});

  m.new_usertype<SymOp>(
      "SymOp",
      sol::call_constructor,
      sol::factories([]() { return SymOp(); },
                     [](const sol::table &rotation, const sol::table &trans) {
                       return SymOp(table_to_mat3(rotation),
                                    table_to_vec3(trans));
                     },
                     [](const sol::table &transformation) {
                       return SymOp(table_to_mat4(transformation));
                     }),
      "apply",
      [](const SymOp &op, const sol::table &positions, sol::this_state s) {
        return op.apply(table_to_mat3n(positions));
      },
      // SymOp::rotation/translation return Eigen Block *expressions*
      // (because of the `auto` return type in the C++ header), not
      // concrete Mat3/Vec3 — force materialization here so sol2 has a
      // registered usertype to push.
      "rotation",
      sol::readonly_property([](const SymOp &op) -> occ::Mat3 {
        return op.rotation();
      }),
      "translation",
      sol::readonly_property([](const SymOp &op) -> occ::Vec3 {
        return op.translation();
      }),
      "transformation",
      sol::property(
          [](const SymOp &op, sol::this_state s) {
            return op.transformation;
          },
          [](SymOp &op, const sol::table &t) {
            op.transformation = table_to_mat4(t);
          }));
  // Free helpers; sol2 doesn't model Python's `def_static`.
  m.set_function(
      "symop_from_rotation_vector",
      [](const sol::table &axis, double angle) {
        return SymOp::from_rotation_vector(table_to_vec3(axis));
      });
  m.set_function("symop_from_axis_angle",
                 [](const sol::table &axis, double angle) {
                   return SymOp::from_axis_angle(table_to_vec3(axis), angle);
                 });
  m.set_function("symop_reflection", [](const sol::table &normal) {
    return SymOp::reflection(table_to_vec3(normal));
  });
  m.set_function("symop_rotoreflection",
                 [](const sol::table &axis, double angle) {
                   return SymOp::rotoreflection(table_to_vec3(axis), angle);
                 });
  m.set_function("symop_inversion", []() { return SymOp::inversion(); });
  m.set_function("symop_identity", []() { return SymOp::identity(); });

  m.new_usertype<MolecularPointGroup>(
      "MolecularPointGroup",
      sol::call_constructor,
      sol::constructors<MolecularPointGroup(const Molecule &)>(),
      "description", sol::readonly_property(&MolecularPointGroup::description),
      "point_group_string",
      sol::readonly_property(&MolecularPointGroup::point_group_string),
      "point_group", sol::readonly_property(&MolecularPointGroup::point_group),
      "symops", sol::readonly_property(&MolecularPointGroup::symops),
      "rotational_symmetries",
      sol::readonly_property([](const MolecularPointGroup &pg) {
        return sol::as_table(pg.rotational_symmetries());
      }),
      "symmetry_number",
      sol::readonly_property(&MolecularPointGroup::symmetry_number),
      sol::meta_function::to_string, [](const MolecularPointGroup &pg) {
        return fmt::format("<MolecularPointGroup '{}'>",
                           pg.point_group_string());
      });

  m.set_function("dihedral_group", &occ::core::dihedral_group);
  m.set_function("cyclic_group", &occ::core::cyclic_group);
}

// -- Multipoles -----------------------------------------------------------

// Minimal binding for each rank — accessors + components, no operator+
// chain (which would need 25 overloads). Lua callers can sum the
// `components` tables directly if they need it.
template <int L>
void register_multipole(sol::table &m, const std::string &name) {
  using MP = Multipole<L>;
  auto t = m.new_usertype<MP>(
      name,
      sol::call_constructor,
      sol::factories([]() { return MP{}; },
                     [](const sol::table &components) {
                       MP mp{};
                       const size_t n = std::min(
                           static_cast<size_t>(components.size()),
                           mp.components.size());
                       for (size_t i = 0; i < n; ++i) {
                         mp.components[i] = components.get<double>(i + 1);
                       }
                       return mp;
                     }),
      "num_components",
      sol::readonly_property([](const MP &mp) { return mp.components.size(); }),
      "charge", sol::readonly_property(&MP::charge),
      "components",
      sol::property(
          [](const MP &mp, sol::this_state s) {
            sol::state_view lua(s);
            sol::table t = lua.create_table(
                static_cast<int>(mp.components.size()), 0);
            for (size_t i = 0; i < mp.components.size(); ++i) {
              t[i + 1] = mp.components[i];
            }
            return t;
          },
          [](MP &mp, const sol::table &components) {
            const size_t n = std::min(
                static_cast<size_t>(components.size()),
                mp.components.size());
            for (size_t i = 0; i < n; ++i) {
              mp.components[i] = components.get<double>(i + 1);
            }
          }),
      "to_string", &MP::to_string,
      sol::meta_function::to_string, [name](const MP &mp) {
        return fmt::format("<{} q={:.6f}>", name, mp.charge());
      });
  if constexpr (L >= 1) {
    t["dipole"] = sol::readonly_property([](const MP &mp, sol::this_state s) {
      return Vec3(mp.dipole().data());
    });
  }
}

void register_multipoles(sol::table &m) {
  register_multipole<0>(m, "Monopole");
  register_multipole<1>(m, "Dipole");
  register_multipole<2>(m, "Quadrupole");
  register_multipole<3>(m, "Octupole");
  register_multipole<4>(m, "Hexadecapole");

  // Polymorphic factory mirroring the Python `Multipole(order, components?)`
  // wrapper. Returns the rank-specific userdata.
  m.set_function(
      "Multipole",
      sol::overload(
          [](int order, sol::this_state s) -> sol::object {
            sol::state_view lua(s);
            switch (order) {
            case 0: return sol::make_object(lua, Multipole<0>{});
            case 1: return sol::make_object(lua, Multipole<1>{});
            case 2: return sol::make_object(lua, Multipole<2>{});
            case 3: return sol::make_object(lua, Multipole<3>{});
            case 4: return sol::make_object(lua, Multipole<4>{});
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported multipole order: {}", order));
            }
          },
          [](int order, const sol::table &components,
             sol::this_state s) -> sol::object {
            sol::state_view lua(s);
            auto fill = [&](auto &&mp) {
              const size_t n = std::min(static_cast<size_t>(components.size()),
                                         mp.components.size());
              for (size_t i = 0; i < n; ++i) {
                mp.components[i] = components.get<double>(i + 1);
              }
              return mp;
            };
            switch (order) {
            case 0: return sol::make_object(lua, fill(Multipole<0>{}));
            case 1: return sol::make_object(lua, fill(Multipole<1>{}));
            case 2: return sol::make_object(lua, fill(Multipole<2>{}));
            case 3: return sol::make_object(lua, fill(Multipole<3>{}));
            case 4: return sol::make_object(lua, fill(Multipole<4>{}));
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported multipole order: {}", order));
            }
          }));
}

// -- MatTriple + free functions -------------------------------------------

void register_misc(sol::table &m) {
  m.new_usertype<MatTriple>(
      "MatTriple",
      sol::call_constructor,
      sol::factories([]() { return MatTriple{}; }),
      "x",
      sol::property(
          [](const MatTriple &mt, sol::this_state s) {
            return mt.x;
          },
          // Setting MatTriple.x/y/z from a table is unusual; expose only
          // the getter for now. Direct mutation via the C++ binding would
          // be required for write access.
          [](MatTriple &mt, sol::object) {
            throw std::runtime_error(
                "MatTriple components are read-only from Lua");
          }),
      "y",
      sol::readonly_property([](const MatTriple &mt, sol::this_state s) {
        return mt.y;
      }),
      "z",
      sol::readonly_property([](const MatTriple &mt, sol::this_state s) {
        return mt.z;
      }),
      "scale_by", &MatTriple::scale_by,
      "symmetrize", &MatTriple::symmetrize,
      sol::meta_function::addition, &MatTriple::operator+,
      sol::meta_function::subtraction, &MatTriple::operator-,
      sol::meta_function::to_string, [](const MatTriple &mt) {
        return fmt::format("<MatTriple ({}x{})>", mt.x.rows(), mt.x.cols());
      });

  m.set_function(
      "eem_partial_charges",
      [](const sol::table &atomic_numbers, const sol::table &positions,
         sol::optional<double> charge, sol::this_state s) {
        const int n = static_cast<int>(atomic_numbers.size());
        IVec z(n);
        for (int i = 0; i < n; ++i) z(i) = atomic_numbers.get<int>(i + 1);
        return vec_to_table(s, occ::core::charges::eem_partial_charges(
                                    z, table_to_mat3n(positions),
                                    charge.value_or(0.0)));
      });

  m.set_function(
      "eeq_partial_charges",
      [](const sol::table &atomic_numbers, const sol::table &positions,
         sol::optional<double> charge, sol::this_state s) {
        const int n = static_cast<int>(atomic_numbers.size());
        IVec z(n);
        for (int i = 0; i < n; ++i) z(i) = atomic_numbers.get<int>(i + 1);
        return vec_to_table(s, occ::core::charges::eeq_partial_charges(
                                    z, table_to_mat3n(positions),
                                    charge.value_or(0.0)));
      });

  m.set_function(
      "eeq_coordination_numbers",
      [](const sol::table &atomic_numbers, const sol::table &positions,
         sol::this_state s) {
        const int n = static_cast<int>(atomic_numbers.size());
        IVec z(n);
        for (int i = 0; i < n; ++i) z(i) = atomic_numbers.get<int>(i + 1);
        return vec_to_table(
            s, occ::core::charges::eeq_coordination_numbers(
                   z, table_to_mat3n(positions)));
      });

  m.set_function(
      "quasirandom_kgf",
      [](int ndims, int count, sol::optional<int> seed, sol::this_state s) {
        return mat_to_table(
            s, occ::core::quasirandom_kgf(ndims, count, seed.value_or(0)));
      });
}

// -- Logging --------------------------------------------------------------

void register_logging(sol::table &m) {
  m.new_enum<spdlog::level::level_enum>(
      "LogLevel",
      {{"TRACE", spdlog::level::trace}, {"DEBUG", spdlog::level::debug},
       {"INFO", spdlog::level::info},   {"WARN", spdlog::level::warn},
       {"ERROR", spdlog::level::err},   {"CRITICAL", spdlog::level::critical},
       {"OFF", spdlog::level::off}});

  // set_log_level already registered in occ_module.cpp; the other helpers
  // round out the surface.
  m.set_function("clear_log_buffer", &occ::log::clear_log_buffer);
  m.set_function("set_log_buffering", &occ::log::set_log_buffering);
  m.set_function("get_buffered_logs",
                 [](sol::this_state s) {
                   sol::state_view lua(s);
                   auto logs = occ::log::get_buffered_logs();
                   sol::table out =
                       lua.create_table(static_cast<int>(logs.size()), 0);
                   int i = 1;
                   for (const auto &[level, message] : logs) {
                     sol::table entry = lua.create_table(0, 2);
                     entry["level"] = static_cast<int>(level);
                     entry["message"] = message;
                     out[i++] = entry;
                   }
                   return out;
                 });

  m.set_function("log_trace", [](const std::string &msg) {
    occ::log::trace(msg);
  });
  m.set_function("log_debug", [](const std::string &msg) {
    occ::log::debug(msg);
  });
  m.set_function("log_info", [](const std::string &msg) {
    occ::log::info(msg);
  });
  m.set_function("log_warn", [](const std::string &msg) {
    occ::log::warn(msg);
  });
  m.set_function("log_error", [](const std::string &msg) {
    occ::log::error(msg);
  });
  m.set_function("log_critical", [](const std::string &msg) {
    occ::log::critical(msg);
  });
}

} // namespace

void register_core_bindings(sol::state_view, sol::table &occ_module) {
  register_atomic_types(occ_module);
  register_molecule_and_dimer(occ_module);
  register_elastic_tensor(occ_module);
  register_point_groups(occ_module);
  register_multipoles(occ_module);
  register_misc(occ_module);
  register_logging(occ_module);
}

} // namespace occ::lua_bindings
