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
namespace lb = luabridge;

namespace {

// -- Element / Atom / PointCharge -----------------------------------------

void register_atomic_types(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .beginClass<Element>("Element")
      // Two overloaded constructors so `occ.Element("C")` and
      // `occ.Element(1)` both work — type-only form, LuaBridge3
      // dispatches on arity and the type checkers.
      .addConstructor<void (*)(const std::string &), void (*)(int)>()
      .addProperty("symbol", &Element::symbol)
      .addProperty("mass", &Element::mass)
      .addProperty("name", &Element::name)
      .addProperty("van_der_waals_radius", &Element::van_der_waals_radius)
      .addProperty("covalent_radius", &Element::covalent_radius)
      .addProperty("atomic_number", &Element::atomic_number)
      .addFunction(
          "__lt", +[](const Element *a, const Element &b) { return *a < b; })
      .addFunction(
          "__eq", +[](const Element *a, const Element &b) { return *a == b; })
      .addFunction(
          "__tostring",
          +[](const Element *a) {
            return fmt::format("<Element '{}'>", a->symbol());
          })
      .endClass()

      .beginClass<Atom>("Atom")
      .addConstructor<void (*)(int, double, double, double)>()
      .addProperty("atomic_number", &Atom::atomic_number)
      .addProperty("x", &Atom::x)
      .addProperty("y", &Atom::y)
      .addProperty("z", &Atom::z)
      // sol::property with custom getter/setter (and getter needs
      // lua_State for tab-building) → paired get_/set_ functions.
      .addProperty(
          "get_position",
          +[](const Atom *a) -> Vec3 { return Vec3(a->x, a->y, a->z); })
      .addFunction(
          "set_position",
          +[](Atom *a, const lb::LuaRef &t) {
            a->set_position(table_to_vec3(t));
          })
      .addFunction(
          "__tostring",
          +[](const Atom *a) {
            return fmt::format("<Atom {} [{:.5f}, {:.5f}, {:.5f}]>",
                               a->atomic_number, a->x, a->y, a->z);
          })
      .endClass()

      .beginClass<PointCharge>("PointCharge")
      // Two factory shapes (q,x,y,z) and (q, pos-table). Pick the
      // explicit-coords form as canonical; expose the table form as
      // static factory.
      .addConstructor<void (*)(double, double, double, double)>()
      .addStaticFunction(
          "from_position",
          +[](double q, const lb::LuaRef &pos) {
            return new PointCharge(q, table_to_vec3(pos));
          })
      .addProperty("charge", &PointCharge::charge, &PointCharge::set_charge)
      .addProperty(
          "get_position",
          +[](const PointCharge *pc) -> Vec3 { return pc->position(); })
      .addFunction(
          "set_position",
          +[](PointCharge *pc, const lb::LuaRef &t) {
            pc->set_position(table_to_vec3(t));
          })
      .addFunction(
          "__tostring",
          +[](const PointCharge *pc) {
            const auto &p = pc->position();
            return fmt::format(
                "<PointCharge q={:.5f} [{:.5f}, {:.5f}, {:.5f}]>", pc->charge(),
                p.x(), p.y(), p.z());
          })
      .endClass()
      .endNamespace();
}

// -- Molecule / Dimer -----------------------------------------------------

void register_molecule_and_dimer(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      .beginNamespace("Origin")
      .addProperty(
          "CARTESIAN",
          +[]() { return static_cast<int>(Molecule::Origin::Cartesian); })
      .addProperty(
          "CENTROID",
          +[]() { return static_cast<int>(Molecule::Origin::Centroid); })
      .addProperty(
          "CENTEROFMASS",
          +[]() { return static_cast<int>(Molecule::Origin::CenterOfMass); })
      .endNamespace()

      .beginClass<Molecule>("Molecule")
      // Construct from atomic-number list + 3xN positions table —
      // `occ.Molecule({Z}, {{x},{y},{z}})`.
      .addConstructor<Molecule *(*)(void *, const lb::LuaRef &,
                                    const lb::LuaRef &)>(
          +[](void *p, const lb::LuaRef &atomic_numbers,
              const lb::LuaRef &positions) {
            const int n = atomic_numbers.length();
            IVec z(n);
            for (int i = 0; i < n; ++i) {
              z(i) = static_cast<int>(lua_get_num(atomic_numbers, i + 1));
            }
            return new (p) Molecule(z, table_to_mat3n(positions));
          })
      .addFunction("__len", &Molecule::size)
      .addProperty("size", &Molecule::size)
      // elements() returns const std::vector<Element>& — properties
      // tripping over the const ref binding to non-trivially-copyable
      // userdata, leave as a method.
      .addFunction("elements", &Molecule::elements)
      .addProperty(
          "positions",
          +[](const Molecule *mol) -> Mat3N { return mol->positions(); })
      .addProperty("name", &Molecule::name, &Molecule::set_name)
      .addProperty(
          "get_partial_charges",
          +[](const Molecule *mol) -> Vec { return mol->partial_charges(); })
      .addFunction(
          "set_partial_charges",
          +[](Molecule *mol, const lb::LuaRef &t) {
            mol->set_partial_charges(table_to_vecx(t));
          })
      .addFunction(
          "esp_partial_charges",
          +[](const Molecule *mol, lua_State *S, const Vec &charges) {
            return vec_to_table(S, mol->esp_partial_charges(charges));
          })
      .addProperty(
          "atomic_masses",
          +[](const Molecule *mol) -> Vec { return mol->atomic_masses(); })
      .addProperty(
          "atomic_numbers",
          +[](const Molecule *mol) -> IVec { return mol->atomic_numbers(); })
      .addProperty(
          "vdw_radii",
          +[](const Molecule *mol) -> Vec { return mol->vdw_radii(); })
      .addProperty("molar_mass", &Molecule::molar_mass)
      // atoms()/bonds() return const std::vector<...>& — stay as methods
      // for the same reason as elements().
      .addFunction("atoms", &Molecule::atoms)
      .addFunction("bonds", &Molecule::bonds)
      .addProperty(
          "center_of_mass",
          +[](const Molecule *mol) -> Vec3 { return mol->center_of_mass(); })
      .addProperty(
          "centroid",
          +[](const Molecule *mol) -> Vec3 { return mol->centroid(); })
      .addProperty("unit_cell_molecule_idx", &Molecule::unit_cell_molecule_idx)
      .addProperty("asymmetric_molecule_idx",
                   &Molecule::asymmetric_molecule_idx)
      .addProperty(
          "asymmetric_unit_idx",
          +[](const Molecule *mol) -> IVec {
            return mol->asymmetric_unit_idx();
          })
      .addFunction("is_equivalent_to", &Molecule::is_equivalent_to)
      // In-place transforms. sol::overload split into uniquely-named.
      .addFunction(
          "rotate",
          +[](Molecule *mol, const lb::LuaRef &rotation, int origin) {
            mol->rotate(table_to_mat3(rotation),
                        static_cast<Molecule::Origin>(origin));
          })
      .addFunction(
          "rotate_about_point",
          +[](Molecule *mol, const lb::LuaRef &rotation,
              const lb::LuaRef &point) {
            mol->rotate(table_to_mat3(rotation), table_to_vec3(point));
          })
      .addFunction(
          "transform",
          +[](Molecule *mol, const lb::LuaRef &t, int origin) {
            mol->transform(table_to_mat4(t),
                           static_cast<Molecule::Origin>(origin));
          })
      .addFunction(
          "transform_about_point",
          +[](Molecule *mol, const lb::LuaRef &t, const lb::LuaRef &point) {
            mol->transform(table_to_mat4(t), table_to_vec3(point));
          })
      .addFunction(
          "translate",
          +[](Molecule *mol, const lb::LuaRef &translation) {
            mol->translate(table_to_vec3(translation));
          })
      // Copy-returning variants.
      .addFunction(
          "rotated",
          +[](const Molecule *mol, const lb::LuaRef &rotation, int origin) {
            return mol->rotated(table_to_mat3(rotation),
                                static_cast<Molecule::Origin>(origin));
          })
      .addFunction(
          "rotated_about_point",
          +[](const Molecule *mol, const lb::LuaRef &rotation,
              const lb::LuaRef &point) {
            return mol->rotated(table_to_mat3(rotation), table_to_vec3(point));
          })
      .addFunction(
          "transformed",
          +[](const Molecule *mol, const lb::LuaRef &t, int origin) {
            return mol->transformed(table_to_mat4(t),
                                    static_cast<Molecule::Origin>(origin));
          })
      .addFunction(
          "transformed_about_point",
          +[](const Molecule *mol, const lb::LuaRef &t,
              const lb::LuaRef &point) {
            return mol->transformed(table_to_mat4(t), table_to_vec3(point));
          })
      .addFunction(
          "translated",
          +[](const Molecule *mol, const lb::LuaRef &translation) {
            return mol->translated(table_to_vec3(translation));
          })
      // sol::optional<Origin> → split.
      .addFunction(
          "centered",
          +[](const Molecule *mol, int origin) {
            Vec3 center;
            switch (static_cast<Molecule::Origin>(origin)) {
            case Molecule::Origin::Centroid:
              center = mol->centroid();
              break;
            case Molecule::Origin::CenterOfMass:
              center = mol->center_of_mass();
              break;
            default:
              center = Vec3::Zero();
            }
            return mol->translated(-center);
          })
      .addFunction(
          "centered_default",
          +[](const Molecule *mol) {
            return mol->translated(-mol->centroid());
          })
      .addFunction("translational_free_energy",
                   &Molecule::translational_free_energy)
      .addFunction("rotational_free_energy", &Molecule::rotational_free_energy)
      .addFunction(
          "__tostring",
          +[](const Molecule *mol) {
            auto com = mol->center_of_mass();
            return fmt::format("<Molecule {} @[{:.5f}, {:.5f}, {:.5f}]>",
                               mol->name(), com.x(), com.y(), com.z());
          })
      .endClass()

      // Static factories on Molecule mirror the Python `from_xyz_file` /
      // `from_xyz_string`. Expose as plain free functions.
      .addFunction(
          "molecule_from_xyz_file",
          +[](const std::string &filename) {
            return occ::io::molecule_from_xyz_file(filename);
          })
      .addFunction(
          "molecule_from_xyz_string",
          +[](const std::string &contents) {
            return occ::io::molecule_from_xyz_string(contents);
          })

      .beginClass<Dimer>("Dimer")
      // Two factory shapes (Molecule, Molecule) and (atoms, atoms).
      // Pick Molecule pair as canonical; atoms-pair as static.
      .addConstructor<void (*)(const Molecule &, const Molecule &)>()
      .addStaticFunction(
          "from_atom_lists",
          +[](const std::vector<Atom> &a, const std::vector<Atom> &b) {
            return new Dimer(a, b);
          })
      .addProperty("a", &Dimer::a)
      .addProperty("b", &Dimer::b)
      .addProperty("nearest_distance", &Dimer::nearest_distance)
      .addProperty("center_of_mass_distance", &Dimer::center_of_mass_distance)
      .addProperty("centroid_distance", &Dimer::centroid_distance)
      // optional<Mat4>: returns nil if absent, else table-of-tables.
      // LuaBridge3: build dynamically via a LuaRef return.
      .addFunction(
          "symmetry_relation",
          +[](const Dimer *d, lua_State *S) {
            auto rel = d->symmetry_relation();
            if (!rel)
              return lb::LuaRef(S); // nil
            return mat_to_table(S, *rel);
          })
      .addProperty("name", &Dimer::name, &Dimer::set_name)
      .addProperty(
          "v_ab_com", +[](const Dimer *d) -> Vec3 { return d->v_ab_com(); })
      .endClass()
      .endNamespace();
}

// -- Elastic tensor -------------------------------------------------------

void register_elastic_tensor(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .beginNamespace("AveragingScheme")
      .addProperty(
          "VOIGT",
          +[]() {
            return static_cast<int>(ElasticTensor::AveragingScheme::Voigt);
          })
      .addProperty(
          "REUSS",
          +[]() {
            return static_cast<int>(ElasticTensor::AveragingScheme::Reuss);
          })
      .addProperty(
          "HILL",
          +[]() {
            return static_cast<int>(ElasticTensor::AveragingScheme::Hill);
          })
      .addProperty(
          "NUMERICAL",
          +[]() {
            return static_cast<int>(ElasticTensor::AveragingScheme::Numerical);
          })
      .endNamespace()

      .beginClass<ElasticTensor>("ElasticTensor")
      // Direct construction from a 6x6 row-major Voigt table —
      // `occ.ElasticTensor(c)` matches the existing example style.
      .addConstructor<ElasticTensor *(*)(void *, const lb::LuaRef &)>(
          +[](void *p, const lb::LuaRef &voigt_c) {
            Mat6 c;
            for (int i = 0; i < 6; ++i) {
              lb::LuaRef row = lua_get_table(voigt_c, i + 1);
              for (int j = 0; j < 6; ++j)
                c(i, j) = lua_get_num(row, j + 1);
            }
            return new (p) ElasticTensor(c);
          })
      .addFunction(
          "youngs_modulus",
          +[](const ElasticTensor *self, const lb::LuaRef &d) {
            return self->youngs_modulus(table_to_vec3(d));
          })
      .addFunction(
          "linear_compressibility",
          +[](const ElasticTensor *self, const lb::LuaRef &d) {
            return self->linear_compressibility(table_to_vec3(d));
          })
      .addFunction(
          "shear_modulus",
          +[](const ElasticTensor *self, const lb::LuaRef &d1,
              const lb::LuaRef &d2) {
            return self->shear_modulus(table_to_vec3(d1), table_to_vec3(d2));
          })
      .addFunction(
          "poisson_ratio",
          +[](const ElasticTensor *self, const lb::LuaRef &d1,
              const lb::LuaRef &d2) {
            return self->poisson_ratio(table_to_vec3(d1), table_to_vec3(d2));
          })
      .addFunction(
          "shear_modulus_minmax",
          +[](const ElasticTensor *self, lua_State *S, const lb::LuaRef &d) {
            auto pair = self->shear_modulus_minmax(table_to_vec3(d));
            lb::LuaRef t = lb::newTable(S);
            t[1] = pair.first;
            t[2] = pair.second;
            return t;
          })
      .addFunction(
          "poisson_ratio_minmax",
          +[](const ElasticTensor *self, lua_State *S, const lb::LuaRef &d) {
            auto pair = self->poisson_ratio_minmax(table_to_vec3(d));
            lb::LuaRef t = lb::newTable(S);
            t[1] = pair.first;
            t[2] = pair.second;
            return t;
          })
      // sol::optional<AveragingScheme> → split for each average_*.
      .addFunction(
          "average_bulk_modulus",
          +[](const ElasticTensor *self, int scheme) {
            return self->average_bulk_modulus(
                static_cast<ElasticTensor::AveragingScheme>(scheme));
          })
      .addFunction(
          "average_bulk_modulus_default",
          +[](const ElasticTensor *self) {
            return self->average_bulk_modulus(
                ElasticTensor::AveragingScheme::Hill);
          })
      .addFunction(
          "average_shear_modulus",
          +[](const ElasticTensor *self, int scheme) {
            return self->average_shear_modulus(
                static_cast<ElasticTensor::AveragingScheme>(scheme));
          })
      .addFunction(
          "average_shear_modulus_default",
          +[](const ElasticTensor *self) {
            return self->average_shear_modulus(
                ElasticTensor::AveragingScheme::Hill);
          })
      .addFunction(
          "average_youngs_modulus",
          +[](const ElasticTensor *self, int scheme) {
            return self->average_youngs_modulus(
                static_cast<ElasticTensor::AveragingScheme>(scheme));
          })
      .addFunction(
          "average_youngs_modulus_default",
          +[](const ElasticTensor *self) {
            return self->average_youngs_modulus(
                ElasticTensor::AveragingScheme::Hill);
          })
      .addFunction(
          "average_poisson_ratio",
          +[](const ElasticTensor *self, int scheme) {
            return self->average_poisson_ratio(
                static_cast<ElasticTensor::AveragingScheme>(scheme));
          })
      .addFunction(
          "average_poisson_ratio_default",
          +[](const ElasticTensor *self) {
            return self->average_poisson_ratio(
                ElasticTensor::AveragingScheme::Hill);
          })
      // sol::optional<int> num_samples → split.
      .addFunction(
          "average_poisson_ratio_direction",
          +[](const ElasticTensor *self, const lb::LuaRef &direction,
              int num_samples) {
            return self->average_poisson_ratio_direction(
                table_to_vec3(direction), num_samples);
          })
      .addFunction(
          "average_poisson_ratio_direction_default",
          +[](const ElasticTensor *self, const lb::LuaRef &direction) {
            return self->average_poisson_ratio_direction(
                table_to_vec3(direction), 360);
          })
      .addFunction(
          "reduced_youngs_modulus",
          +[](const ElasticTensor *self, const lb::LuaRef &direction,
              int num_samples) {
            return self->reduced_youngs_modulus(table_to_vec3(direction),
                                                num_samples);
          })
      .addFunction(
          "reduced_youngs_modulus_default",
          +[](const ElasticTensor *self, const lb::LuaRef &direction) {
            return self->reduced_youngs_modulus(table_to_vec3(direction), 360);
          })
      .addFunction("transverse_acoustic_velocity",
                   &ElasticTensor::transverse_acoustic_velocity)
      .addFunction("longitudinal_acoustic_velocity",
                   &ElasticTensor::longitudinal_acoustic_velocity)
      .addProperty(
          "voigt_s",
          +[](const ElasticTensor *self) -> Mat6 { return self->voigt_s(); })
      .addProperty(
          "voigt_c",
          +[](const ElasticTensor *self) -> Mat6 { return self->voigt_c(); })
      .addFunction(
          "component", +[](const ElasticTensor *self, int i, int j, int k,
                           int l) { return self->component(i, j, k, l); })
      .addProperty(
          "eigenvalues",
          +[](const ElasticTensor *self) -> Vec { return self->eigenvalues(); })
      .addFunction(
          "voigt_rotation_matrix",
          +[](const ElasticTensor *self, lua_State *S,
              const lb::LuaRef &rotation) {
            return mat_to_table(
                S, self->voigt_rotation_matrix(table_to_mat3(rotation)));
          })
      .addFunction(
          "rotate_voigt_stiffness",
          +[](const ElasticTensor *self, lua_State *S,
              const lb::LuaRef &rotation) {
            return mat_to_table(
                S, self->rotate_voigt_stiffness(table_to_mat3(rotation)));
          })
      .addFunction(
          "rotate_voigt_compliance",
          +[](const ElasticTensor *self, lua_State *S,
              const lb::LuaRef &rotation) {
            return mat_to_table(
                S, self->rotate_voigt_compliance(table_to_mat3(rotation)));
          })
      .endClass()
      .endNamespace();
}

// -- Point groups / symmetry ----------------------------------------------

void register_point_groups(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .beginNamespace("PointGroup")
      .addProperty(
          "C1", +[]() { return static_cast<int>(::occ::core::PointGroup::C1); })
      .addProperty(
          "Ci", +[]() { return static_cast<int>(::occ::core::PointGroup::Ci); })
      .addProperty(
          "Cs", +[]() { return static_cast<int>(::occ::core::PointGroup::Cs); })
      .addProperty(
          "C2", +[]() { return static_cast<int>(::occ::core::PointGroup::C2); })
      .addProperty(
          "C3", +[]() { return static_cast<int>(::occ::core::PointGroup::C3); })
      .addProperty(
          "C4", +[]() { return static_cast<int>(::occ::core::PointGroup::C4); })
      .addProperty(
          "C5", +[]() { return static_cast<int>(::occ::core::PointGroup::C5); })
      .addProperty(
          "C6", +[]() { return static_cast<int>(::occ::core::PointGroup::C6); })
      .addProperty(
          "C8", +[]() { return static_cast<int>(::occ::core::PointGroup::C8); })
      .addProperty(
          "Coov",
          +[]() { return static_cast<int>(::occ::core::PointGroup::Coov); })
      .addProperty(
          "Dooh",
          +[]() { return static_cast<int>(::occ::core::PointGroup::Dooh); })
      .addProperty(
          "C2v",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C2v); })
      .addProperty(
          "C3v",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C3v); })
      .addProperty(
          "C4v",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C4v); })
      .addProperty(
          "C5v",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C5v); })
      .addProperty(
          "C6v",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C6v); })
      .addProperty(
          "C2h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C2h); })
      .addProperty(
          "C3h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C3h); })
      .addProperty(
          "C4h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C4h); })
      .addProperty(
          "C5h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C5h); })
      .addProperty(
          "C6h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::C6h); })
      .addProperty(
          "D2", +[]() { return static_cast<int>(::occ::core::PointGroup::D2); })
      .addProperty(
          "D3", +[]() { return static_cast<int>(::occ::core::PointGroup::D3); })
      .addProperty(
          "D4", +[]() { return static_cast<int>(::occ::core::PointGroup::D4); })
      .addProperty(
          "D5", +[]() { return static_cast<int>(::occ::core::PointGroup::D5); })
      .addProperty(
          "D6", +[]() { return static_cast<int>(::occ::core::PointGroup::D6); })
      .addProperty(
          "D7", +[]() { return static_cast<int>(::occ::core::PointGroup::D7); })
      .addProperty(
          "D8", +[]() { return static_cast<int>(::occ::core::PointGroup::D8); })
      .addProperty(
          "D2h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D2h); })
      .addProperty(
          "D3h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D3h); })
      .addProperty(
          "D4h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D4h); })
      .addProperty(
          "D5h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D5h); })
      .addProperty(
          "D6h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D6h); })
      .addProperty(
          "D7h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D7h); })
      .addProperty(
          "D8h",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D8h); })
      .addProperty(
          "D2d",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D2d); })
      .addProperty(
          "D3d",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D3d); })
      .addProperty(
          "D4d",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D4d); })
      .addProperty(
          "D5d",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D5d); })
      .addProperty(
          "D6d",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D6d); })
      .addProperty(
          "D7d",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D7d); })
      .addProperty(
          "D8d",
          +[]() { return static_cast<int>(::occ::core::PointGroup::D8d); })
      .addProperty(
          "S4", +[]() { return static_cast<int>(::occ::core::PointGroup::S4); })
      .addProperty(
          "S6", +[]() { return static_cast<int>(::occ::core::PointGroup::S6); })
      .addProperty(
          "S8", +[]() { return static_cast<int>(::occ::core::PointGroup::S8); })
      .addProperty(
          "T", +[]() { return static_cast<int>(::occ::core::PointGroup::T); })
      .addProperty(
          "Td", +[]() { return static_cast<int>(::occ::core::PointGroup::Td); })
      .addProperty(
          "Th", +[]() { return static_cast<int>(::occ::core::PointGroup::Th); })
      .addProperty(
          "O", +[]() { return static_cast<int>(::occ::core::PointGroup::O); })
      .addProperty(
          "Oh", +[]() { return static_cast<int>(::occ::core::PointGroup::Oh); })
      .addProperty(
          "I", +[]() { return static_cast<int>(::occ::core::PointGroup::I); })
      .addProperty(
          "Ih", +[]() { return static_cast<int>(::occ::core::PointGroup::Ih); })
      .endNamespace()

      .beginNamespace("MirrorType")
      .addProperty(
          "None_", +[]() { return static_cast<int>(MirrorType::None); })
      .addProperty(
          "H", +[]() { return static_cast<int>(MirrorType::H); })
      .addProperty(
          "D", +[]() { return static_cast<int>(MirrorType::D); })
      .addProperty(
          "V", +[]() { return static_cast<int>(MirrorType::V); })
      .endNamespace()

      .beginClass<SymOp>("SymOp")
      // Three factory shapes: default, (rotation, translation), (Mat4).
      // Pick default as canonical, expose other two as static.
      .addConstructor<void (*)()>()
      .addStaticFunction(
          "from_rotation_translation",
          +[](const lb::LuaRef &rotation, const lb::LuaRef &trans) {
            return new SymOp(table_to_mat3(rotation), table_to_vec3(trans));
          })
      .addStaticFunction(
          "from_transformation",
          +[](const lb::LuaRef &transformation) {
            return new SymOp(table_to_mat4(transformation));
          })
      .addFunction(
          "apply",
          +[](const SymOp *op, lua_State *S, const lb::LuaRef &positions) {
            return mat_to_table(S, op->apply(table_to_mat3n(positions)));
          })
      // SymOp::rotation/translation return Eigen Block *expressions*;
      // materialize via explicit return type.
      .addProperty(
          "rotation", +[](const SymOp *op) -> Mat3 { return op->rotation(); })
      .addProperty(
          "translation",
          +[](const SymOp *op) -> Vec3 { return op->translation(); })
      .addProperty(
          "get_transformation",
          +[](const SymOp *op) -> Mat4 { return op->transformation; })
      .addFunction(
          "set_transformation",
          +[](SymOp *op, const lb::LuaRef &t) {
            op->transformation = table_to_mat4(t);
          })
      .endClass()

      // Free helpers; LuaBridge3 doesn't model Python's `def_static` either.
      .addFunction(
          "symop_from_rotation_vector",
          +[](const lb::LuaRef &axis, double /*angle*/) {
            return SymOp::from_rotation_vector(table_to_vec3(axis));
          })
      .addFunction(
          "symop_from_axis_angle",
          +[](const lb::LuaRef &axis, double angle) {
            return SymOp::from_axis_angle(table_to_vec3(axis), angle);
          })
      .addFunction(
          "symop_reflection",
          +[](const lb::LuaRef &normal) {
            return SymOp::reflection(table_to_vec3(normal));
          })
      .addFunction(
          "symop_rotoreflection",
          +[](const lb::LuaRef &axis, double angle) {
            return SymOp::rotoreflection(table_to_vec3(axis), angle);
          })
      .addFunction(
          "symop_inversion", +[]() { return SymOp::inversion(); })
      .addFunction(
          "symop_identity", +[]() { return SymOp::identity(); })

      .beginClass<MolecularPointGroup>("MolecularPointGroup")
      .addConstructor<void (*)(const Molecule &)>()
      .addProperty("description", &MolecularPointGroup::description)
      .addProperty("point_group_string",
                   &MolecularPointGroup::point_group_string)
      // Returns const PointGroup by value — LuaBridge3 can't build a
      // const userdata, so wrap in a lambda that drops the cv-qual.
      .addProperty(
          "point_group",
          +[](const MolecularPointGroup *pg) {
            return static_cast<int>(pg->point_group());
          })
      .addProperty("symops", &MolecularPointGroup::symops)
      .addFunction(
          "rotational_symmetries",
          +[](const MolecularPointGroup *pg, lua_State *S) {
            lb::LuaRef t = lb::newTable(S);
            const auto &v = pg->rotational_symmetries();
            for (size_t i = 0; i < v.size(); ++i) {
              t[static_cast<int>(i + 1)] = v[i];
            }
            return t;
          })
      .addProperty("symmetry_number", &MolecularPointGroup::symmetry_number)
      .addFunction(
          "__tostring",
          +[](const MolecularPointGroup *pg) {
            return fmt::format("<MolecularPointGroup '{}'>",
                               pg->point_group_string());
          })
      .endClass()

      .addFunction("dihedral_group", &occ::core::dihedral_group)
      .addFunction("cyclic_group", &occ::core::cyclic_group)
      .endNamespace();
}

// -- Multipoles -----------------------------------------------------------

// Minimal binding for each rank — accessors + components.
//
// LuaBridge3's `Namespace::Class<T>` is move-only (non-copyable), so we
// can't grab it with `auto cls = chain` and split the chain later for
// the conditional `if constexpr (L >= 1)` dipole binding. Instead we
// register the dipole on a SECOND beginClass<MP> chain — LuaBridge3
// re-opens the existing class table and the extra method is appended.
template <int L>
void register_multipole(lua_State *Ls, const std::string &name) {
  using MP = Multipole<L>;
  lb::getGlobalNamespace(Ls)
      .beginNamespace("occ")
      .template beginClass<MP>(name.c_str())
      .template addConstructor<void (*)()>()
      .addStaticFunction(
          "from_components",
          +[](const lb::LuaRef &components) {
            MP *mp = new MP{};
            const size_t n = std::min(static_cast<size_t>(components.length()),
                                      mp->components.size());
            for (size_t i = 0; i < n; ++i) {
              mp->components[i] =
                  lua_get_num(components, static_cast<int>(i + 1));
            }
            return mp;
          })
      .addProperty(
          "num_components", +[](const MP *mp) { return mp->components.size(); })
      .addProperty("charge", &MP::charge)
      .addFunction(
          "get_components",
          +[](const MP *mp, lua_State *S) {
            lb::LuaRef t = lb::newTable(S);
            for (size_t i = 0; i < mp->components.size(); ++i) {
              t[static_cast<int>(i + 1)] = mp->components[i];
            }
            return t;
          })
      .addFunction(
          "set_components",
          +[](MP *mp, const lb::LuaRef &components) {
            const size_t n = std::min(static_cast<size_t>(components.length()),
                                      mp->components.size());
            for (size_t i = 0; i < n; ++i) {
              mp->components[i] =
                  lua_get_num(components, static_cast<int>(i + 1));
            }
          })
      .addFunction("to_string", &MP::to_string)
      // Capturing lambda — drop the unary `+` so LuaBridge3 routes
      // it through std::function rather than a plain function pointer.
      .addFunction("__tostring",
                   [name](const MP *mp) {
                     return fmt::format("<{} q={:.6f}>", name, mp->charge());
                   })
      .endClass()
      .endNamespace();

  if constexpr (L >= 1) {
    lb::getGlobalNamespace(Ls)
        .beginNamespace("occ")
        .template beginClass<MP>(name.c_str())
        .addProperty(
            "dipole",
            +[](const MP *mp) -> Vec3 { return Vec3(mp->dipole().data()); })
        .endClass()
        .endNamespace();
  }
}

void register_multipoles(lua_State *L) {
  register_multipole<0>(L, "Monopole");
  register_multipole<1>(L, "Dipole");
  register_multipole<2>(L, "Quadrupole");
  register_multipole<3>(L, "Octupole");
  register_multipole<4>(L, "Hexadecapole");

  // Polymorphic factory mirroring the Python `Multipole(order, components?)`
  // wrapper. sol2 used `sol::object` + `sol::make_object`; LuaBridge3's
  // equivalent is `lb::LuaRef::newValue(L, value)`. Split components-taking
  // form into a separate name since LuaBridge3 doesn't auto-overload.
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .addFunction(
          "Multipole",
          +[](lua_State *S, int order) {
            switch (order) {
            case 0:
              return lb::LuaRef(S, Multipole<0>{});
            case 1:
              return lb::LuaRef(S, Multipole<1>{});
            case 2:
              return lb::LuaRef(S, Multipole<2>{});
            case 3:
              return lb::LuaRef(S, Multipole<3>{});
            case 4:
              return lb::LuaRef(S, Multipole<4>{});
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported multipole order: {}", order));
            }
          })
      .addFunction(
          "Multipole_with_components",
          +[](lua_State *S, int order, const lb::LuaRef &components) {
            auto fill = [&](auto mp) {
              const size_t n =
                  std::min(static_cast<size_t>(components.length()),
                           mp.components.size());
              for (size_t i = 0; i < n; ++i) {
                mp.components[i] =
                    lua_get_num(components, static_cast<int>(i + 1));
              }
              return mp;
            };
            switch (order) {
            case 0:
              return lb::LuaRef(S, fill(Multipole<0>{}));
            case 1:
              return lb::LuaRef(S, fill(Multipole<1>{}));
            case 2:
              return lb::LuaRef(S, fill(Multipole<2>{}));
            case 3:
              return lb::LuaRef(S, fill(Multipole<3>{}));
            case 4:
              return lb::LuaRef(S, fill(Multipole<4>{}));
            default:
              throw std::runtime_error(
                  fmt::format("Unsupported multipole order: {}", order));
            }
          })
      .endNamespace();
}

// -- MatTriple + free functions -------------------------------------------

void register_misc(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .beginClass<MatTriple>("MatTriple")
      .addConstructor<void (*)()>()
      .addProperty(
          "get_x", +[](const MatTriple *mt) -> Mat { return mt->x; })
      .addProperty(
          "get_y", +[](const MatTriple *mt) -> Mat { return mt->y; })
      .addProperty(
          "get_z", +[](const MatTriple *mt) -> Mat { return mt->z; })
      .addFunction("scale_by", &MatTriple::scale_by)
      .addFunction("symmetrize", &MatTriple::symmetrize)
      // operator+/- may be non-const upstream — take mutable self ptr.
      .addFunction(
          "__add", +[](MatTriple *a, const MatTriple &b) { return *a + b; })
      .addFunction(
          "__sub", +[](MatTriple *a, const MatTriple &b) { return *a - b; })
      .addFunction(
          "__tostring",
          +[](const MatTriple *mt) {
            return fmt::format("<MatTriple ({}x{})>", mt->x.rows(),
                               mt->x.cols());
          })
      .endClass()

      // Optional `charge` arg as `LuaRef` so a 2-arg call also works
      // (defaults to 0). The lua_State* is recoverable from the
      // atomic_numbers LuaRef so we don't need it as a separate
      // parameter (LuaBridge3's auto-injection looks brittle when
      // followed by other LuaRef args).
      .addFunction(
          "eem_partial_charges",
          +[](const lb::LuaRef &atomic_numbers, const lb::LuaRef &positions,
              const lb::LuaRef &charge) {
            const int n = atomic_numbers.length();
            IVec z(n);
            for (int i = 0; i < n; ++i) {
              z(i) = static_cast<int>(lua_get_num(atomic_numbers, i + 1));
            }
            const double q =
                charge.isNumber() ? charge.unsafe_cast<double>() : 0.0;
            return vec_to_table(atomic_numbers.state(),
                                occ::core::charges::eem_partial_charges(
                                    z, table_to_mat3n(positions), q));
          })
      .addFunction(
          "eem_partial_charges_default",
          +[](lua_State *S, const lb::LuaRef &atomic_numbers,
              const lb::LuaRef &positions) {
            const int n = atomic_numbers.length();
            IVec z(n);
            for (int i = 0; i < n; ++i) {
              z(i) = static_cast<int>(lua_get_num(atomic_numbers, i + 1));
            }
            return vec_to_table(S, occ::core::charges::eem_partial_charges(
                                       z, table_to_mat3n(positions), 0.0));
          })

      .addFunction(
          "eeq_partial_charges",
          +[](const lb::LuaRef &atomic_numbers, const lb::LuaRef &positions,
              const lb::LuaRef &charge) {
            const int n = atomic_numbers.length();
            IVec z(n);
            for (int i = 0; i < n; ++i) {
              z(i) = static_cast<int>(lua_get_num(atomic_numbers, i + 1));
            }
            const double q =
                charge.isNumber() ? charge.unsafe_cast<double>() : 0.0;
            return vec_to_table(atomic_numbers.state(),
                                occ::core::charges::eeq_partial_charges(
                                    z, table_to_mat3n(positions), q));
          })

      .addFunction(
          "eeq_coordination_numbers",
          +[](const lb::LuaRef &atomic_numbers, const lb::LuaRef &positions) {
            const int n = atomic_numbers.length();
            IVec z(n);
            for (int i = 0; i < n; ++i) {
              z(i) = static_cast<int>(lua_get_num(atomic_numbers, i + 1));
            }
            return vec_to_table(atomic_numbers.state(),
                                occ::core::charges::eeq_coordination_numbers(
                                    z, table_to_mat3n(positions)));
          })

      // sol::optional<int> seed → split.
      .addFunction(
          "quasirandom_kgf",
          +[](lua_State *S, int ndims, int count, int seed) {
            return mat_to_table(S,
                                occ::core::quasirandom_kgf(ndims, count, seed));
          })
      .addFunction(
          "quasirandom_kgf_default",
          +[](lua_State *S, int ndims, int count) {
            return mat_to_table(S, occ::core::quasirandom_kgf(ndims, count, 0));
          })
      .endNamespace();
}

// -- Logging --------------------------------------------------------------

void register_logging(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
      .beginNamespace("LogLevel")
      .addProperty(
          "TRACE", +[]() { return static_cast<int>(spdlog::level::trace); })
      .addProperty(
          "DEBUG", +[]() { return static_cast<int>(spdlog::level::debug); })
      .addProperty(
          "INFO", +[]() { return static_cast<int>(spdlog::level::info); })
      .addProperty(
          "WARN", +[]() { return static_cast<int>(spdlog::level::warn); })
      .addProperty(
          "ERROR", +[]() { return static_cast<int>(spdlog::level::err); })
      .addProperty(
          "CRITICAL",
          +[]() { return static_cast<int>(spdlog::level::critical); })
      .addProperty(
          "OFF", +[]() { return static_cast<int>(spdlog::level::off); })
      .endNamespace()

      // set_log_level is already registered in occ_module.cpp.
      .addFunction("clear_log_buffer", &occ::log::clear_log_buffer)
      .addFunction("set_log_buffering", &occ::log::set_log_buffering)
      .addFunction(
          "get_buffered_logs",
          +[](lua_State *S) {
            auto logs = occ::log::get_buffered_logs();
            lb::LuaRef out = lb::newTable(S);
            int i = 1;
            for (const auto &[level, message] : logs) {
              lb::LuaRef entry = lb::newTable(S);
              entry["level"] = static_cast<int>(level);
              entry["message"] = message;
              out[i++] = entry;
            }
            return out;
          })

      .addFunction(
          "log_trace", +[](const std::string &msg) { occ::log::trace(msg); })
      .addFunction(
          "log_debug", +[](const std::string &msg) { occ::log::debug(msg); })
      .addFunction(
          "log_info", +[](const std::string &msg) { occ::log::info(msg); })
      .addFunction(
          "log_warn", +[](const std::string &msg) { occ::log::warn(msg); })
      .addFunction(
          "log_error", +[](const std::string &msg) { occ::log::error(msg); })
      .addFunction(
          "log_critical",
          +[](const std::string &msg) { occ::log::critical(msg); })
      .endNamespace();
}

} // namespace

void register_core_bindings(lua_State *L) {
  register_atomic_types(L);
  register_molecule_and_dimer(L);
  register_elastic_tensor(L);
  register_point_groups(L);
  register_multipoles(L);
  register_misc(L);
  register_logging(L);
}

} // namespace occ::lua_bindings
