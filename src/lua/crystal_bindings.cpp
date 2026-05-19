#include "crystal_bindings.h"
#include "eigen_conv.h"
#include <Eigen/LU>
#include <ankerl/unordered_dense.h>
#include <fmt/core.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_mapping_table.h>
#include <occ/crystal/surface.h>
#include <occ/io/cifparser.h>

namespace occ::lua_bindings {

using namespace occ::crystal;
using SymmetryRelatedDimer = occ::crystal::CrystalDimers::SymmetryRelatedDimer;
namespace lb = luabridge;

namespace {

// -- HKL ------------------------------------------------------------------

void register_hkl(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<HKL>("HKL")
          .addConstructor<void (*)(int, int, int)>()
          .addProperty("h", &HKL::h)
          .addProperty("k", &HKL::k)
          .addProperty("l", &HKL::l)
          .addFunction("d", &HKL::d)
          .addProperty("vector",
                       +[](const HKL *hkl) -> occ::Vec3 { return hkl->vector(); })
          .addFunction("__tostring",
                       +[](const HKL *hkl) {
                         return fmt::format("<HKL [{} {} {}]>", hkl->h, hkl->k,
                                            hkl->l);
                       })
        .endClass()
        .addFunction("hkl_floor",
                     +[](const lb::LuaRef &v) {
                       return HKL::floor(table_to_vec3(v));
                     })
        .addFunction("hkl_ceil",
                     +[](const lb::LuaRef &v) {
                       return HKL::ceil(table_to_vec3(v));
                     })
        .addFunction("hkl_maximum", +[]() { return HKL::maximum(); })
        .addFunction("hkl_minimum", +[]() { return HKL::minimum(); })
      .endNamespace();
}

// -- SymmetryOperation ----------------------------------------------------

void register_symmetry_operation(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<SymmetryOperationFormat>("SymmetryOperationFormat")
          .addConstructor<void (*)()>()
          .addPropertyReadWrite("fmt_string",
                                &SymmetryOperationFormat::fmt_string)
          .addPropertyReadWrite("delimiter",
                                &SymmetryOperationFormat::delimiter)
        .endClass()

        .beginClass<SymmetryOperation>("SymmetryOperation")
          // Single canonical constructor (int code); the other shapes
          // (seitz matrix table, string spec) live as static factories
          // since LuaBridge3 doesn't auto-overload.
          .addConstructor<void (*)(int)>()
          .addStaticFunction(
              "from_seitz",
              +[](const lb::LuaRef &seitz) {
                return new SymmetryOperation(table_to_mat4(seitz));
              })
          .addStaticFunction("from_string", +[](const std::string &s) {
            return new SymmetryOperation(s);
          })
          .addFunction("to_int", &SymmetryOperation::to_int)
          .addFunction("to_string",
                       +[](const SymmetryOperation *op) {
                         return op->to_string();
                       })
          .addFunction("to_string_with_format",
                       +[](const SymmetryOperation *op,
                           const SymmetryOperationFormat &fmt) {
                         return op->to_string(fmt);
                       })
          .addFunction("inverted", &SymmetryOperation::inverted)
          .addFunction("translated",
                       +[](const SymmetryOperation *op,
                           const lb::LuaRef &t) {
                         return op->translated(table_to_vec3(t));
                       })
          .addProperty("is_identity", &SymmetryOperation::is_identity)
          .addFunction("apply",
                       +[](const SymmetryOperation *op, lua_State *S,
                           const lb::LuaRef &positions) {
                         return mat_to_table(S, op->apply(table_to_mat3n(positions)));
                       })
          .addProperty("seitz",
                       +[](const SymmetryOperation *op) -> occ::Mat4 { return op->seitz(); })
          .addProperty("rotation",
                       +[](const SymmetryOperation *op) -> occ::Mat3 { return op->rotation(); })
          .addFunction("cartesian_rotation",
                       +[](const SymmetryOperation *op, lua_State *S,
                           const UnitCell &cell) {
                         return mat_to_table(S, op->cartesian_rotation(cell));
                       })
          .addProperty("translation",
                       +[](const SymmetryOperation *op) -> occ::Vec3 { return op->translation(); })
          .addProperty("has_translation", &SymmetryOperation::has_translation)
          .addFunction("__call",
                       +[](const SymmetryOperation *op, lua_State *S,
                           const lb::LuaRef &positions) {
                         return mat_to_table(S,
                                             (*op)(table_to_mat3n(positions)));
                       })
          .addFunction("__mul",
                       +[](const SymmetryOperation *a,
                           const SymmetryOperation &b) { return *a * b; })
          .addFunction("__eq",
                       +[](const SymmetryOperation *a,
                           const SymmetryOperation &b) { return *a == b; })
          .addFunction("__lt",
                       +[](const SymmetryOperation *a,
                           const SymmetryOperation &b) { return *a < b; })
          .addFunction("__le",
                       +[](const SymmetryOperation *a,
                           const SymmetryOperation &b) { return *a <= b; })
          .addFunction("__tostring",
                       +[](const SymmetryOperation *op) {
                         return fmt::format("<SymmetryOperation '{}'>",
                                            op->to_string());
                       })
        .endClass()
      .endNamespace();
}

// -- SpaceGroup -----------------------------------------------------------

void register_spacegroup(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<SpaceGroup>("SpaceGroup")
          .addConstructor<void (*)()>()
          .addStaticFunction("from_number", +[](int n) {
            return new SpaceGroup(n);
          })
          .addStaticFunction("from_symbol", +[](const std::string &s) {
            return new SpaceGroup(s);
          })
          .addStaticFunction(
              "from_symop_strings",
              +[](const std::vector<std::string> &symops) {
                return new SpaceGroup(symops);
              })
          .addStaticFunction(
              "from_symops",
              +[](const std::vector<SymmetryOperation> &symops) {
                return new SpaceGroup(symops);
              })
          .addProperty("number", &SpaceGroup::number)
          .addProperty("symbol", &SpaceGroup::symbol)
          .addProperty("short_name", &SpaceGroup::short_name)
          // Vector-returning getter — LuaBridge3 property-getter
          // specialization doesn't accept const std::vector<T>&; leave as
          // a method.
          .addFunction("symmetry_operations", &SpaceGroup::symmetry_operations)
          .addProperty("has_H_R_choice", &SpaceGroup::has_H_R_choice)
          .addFunction(
              "apply_all_symmetry_operations",
              +[](const SpaceGroup *sg, lua_State *S,
                  const lb::LuaRef &positions) {
                auto p =
                    sg->apply_all_symmetry_operations(table_to_mat3n(positions));
                lb::LuaRef out = lb::newTable(S);
                out["indices"] = vec_to_table(S, p.first);
                out["positions"] = mat_to_table(S, p.second);
                return out;
              })
          .addFunction(
              "apply_rotations",
              +[](const SpaceGroup *sg, lua_State *S,
                  const lb::LuaRef &positions) {
                auto p = sg->apply_rotations(table_to_mat3n(positions));
                lb::LuaRef out = lb::newTable(S);
                out["indices"] = vec_to_table(S, p.first);
                out["positions"] = mat_to_table(S, p.second);
                return out;
              })
          .addFunction("__tostring",
                       +[](const SpaceGroup *sg) {
                         return fmt::format(
                             "<SpaceGroup {} ({}), {} operations>", sg->symbol(),
                             sg->number(), sg->symmetry_operations().size());
                       })
        .endClass()
      .endNamespace();
}

// -- AsymmetricUnit -------------------------------------------------------

void register_asymmetric_unit(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<AsymmetricUnit>("AsymmetricUnit")
          .addConstructor<void (*)()>()
          .addStaticFunction(
              "from_positions_and_z",
              +[](const lb::LuaRef &positions, const lb::LuaRef &atomic_numbers) {
                const int n = atomic_numbers.length();
                IVec z(n);
                for (int i = 0; i < n; ++i) {
                  z(i) = static_cast<int>(lua_get_num(atomic_numbers, i + 1));
                }
                return new AsymmetricUnit(table_to_mat3n(positions), z);
              })
          .addStaticFunction(
              "from_positions_z_labels",
              +[](const lb::LuaRef &positions, const lb::LuaRef &atomic_numbers,
                  const std::vector<std::string> &labels) {
                const int n = atomic_numbers.length();
                IVec z(n);
                for (int i = 0; i < n; ++i) {
                  z(i) = static_cast<int>(lua_get_num(atomic_numbers, i + 1));
                }
                return new AsymmetricUnit(table_to_mat3n(positions), z, labels);
              })
          .addProperty("get_positions",
                       +[](const AsymmetricUnit *a) -> occ::Mat3N { return a->positions; })
          .addFunction("set_positions",
                       +[](AsymmetricUnit *a, const lb::LuaRef &t) {
                         a->positions = table_to_mat3n(t);
                       })
          .addProperty(
              "get_atomic_numbers",
              +[](const AsymmetricUnit *a) -> occ::IVec { return a->atomic_numbers; })
          .addFunction(
              "set_atomic_numbers",
              +[](AsymmetricUnit *a, const lb::LuaRef &t) {
                const int n = t.length();
                IVec z(n);
                for (int i = 0; i < n; ++i) {
                  z(i) = static_cast<int>(lua_get_num(t, i + 1));
                }
                a->atomic_numbers = z;
              })
          .addProperty("get_occupations",
                       +[](const AsymmetricUnit *a) -> occ::Vec { return a->occupations; })
          .addFunction("set_occupations",
                       +[](AsymmetricUnit *a, const lb::LuaRef &t) {
                         a->occupations = table_to_vecx(t);
                       })
          .addProperty("get_charges",
                       +[](const AsymmetricUnit *a) -> occ::Vec { return a->charges; })
          .addFunction("set_charges",
                       +[](AsymmetricUnit *a, const lb::LuaRef &t) {
                         a->charges = table_to_vecx(t);
                       })
          .addProperty("labels", &AsymmetricUnit::labels)
          .addProperty("size", &AsymmetricUnit::size)
          .addFunction("__len", &AsymmetricUnit::size)
          .addFunction("__tostring",
                       +[](const AsymmetricUnit *asym) {
                         return fmt::format("<AsymmetricUnit {}>",
                                            asym->chemical_formula());
                       })
        .endClass()
      .endNamespace();
}

// -- UnitCell -------------------------------------------------------------

void register_unitcell(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<UnitCell>("UnitCell")
          .addConstructor<void (*)(double, double, double, double, double,
                                    double)>()
          .addStaticFunction("default_cell",
                             +[]() { return new UnitCell(); })
          .addStaticFunction(
              "from_lengths_and_angles",
              +[](const lb::LuaRef &lengths, const lb::LuaRef &angles) {
                return new UnitCell(table_to_vec3(lengths),
                                     table_to_vec3(angles));
              })
          .addStaticFunction(
              "from_lattice_vectors",
              +[](const lb::LuaRef &lattice_vectors) {
                return new UnitCell(table_to_mat3(lattice_vectors));
              })
          // sol2 `sol::property(getter, setter)` ↦ LuaBridge3 `addProperty`
          // with member-function pointers.
          .addProperty("a", &UnitCell::a, &UnitCell::set_a)
          .addProperty("b", &UnitCell::b, &UnitCell::set_b)
          .addProperty("c", &UnitCell::c, &UnitCell::set_c)
          .addProperty("alpha", &UnitCell::alpha, &UnitCell::set_alpha)
          .addProperty("beta", &UnitCell::beta, &UnitCell::set_beta)
          .addProperty("gamma", &UnitCell::gamma, &UnitCell::set_gamma)
          .addProperty("volume", &UnitCell::volume)
          .addProperty("direct",
                       +[](const UnitCell *c) -> occ::Mat3 { return c->direct(); })
          .addProperty("reciprocal",
                       +[](const UnitCell *c) -> occ::Mat3 { return c->reciprocal(); })
          .addProperty("inverse",
                       +[](const UnitCell *c) -> occ::Mat3 { return c->inverse(); })
          .addProperty("lengths",
                       +[](const UnitCell *c) -> occ::Vec3 { return c->lengths(); })
          .addProperty("angles",
                       +[](const UnitCell *c) -> occ::Vec3 { return c->angles(); })
          .addProperty("is_cubic", &UnitCell::is_cubic)
          .addProperty("is_triclinic", &UnitCell::is_triclinic)
          .addProperty("is_monoclinic", &UnitCell::is_monoclinic)
          .addProperty("is_orthorhombic", &UnitCell::is_orthorhombic)
          .addProperty("is_tetragonal", &UnitCell::is_tetragonal)
          .addProperty("is_rhombohedral", &UnitCell::is_rhombohedral)
          .addProperty("is_hexagonal", &UnitCell::is_hexagonal)
          .addProperty("is_orthogonal", &UnitCell::is_orthogonal)
          .addProperty("cell_type", &UnitCell::cell_type)
          // to_cartesian / to_fractional / to_reciprocal return Eigen
          // Product expression templates — materialize via `-> Mat3N`.
          .addFunction(
              "to_cartesian",
              +[](const UnitCell *c, lua_State *S, const lb::LuaRef &frac) {
                occ::Mat3N out = c->to_cartesian(table_to_mat3n(frac));
                return mat_to_table(S, out);
              })
          .addFunction(
              "to_fractional",
              +[](const UnitCell *c, lua_State *S, const lb::LuaRef &cart) {
                occ::Mat3N out = c->to_fractional(table_to_mat3n(cart));
                return mat_to_table(S, out);
              })
          .addFunction(
              "to_reciprocal",
              +[](const UnitCell *c, lua_State *S, const lb::LuaRef &v) {
                occ::Mat3N out = c->to_reciprocal(table_to_mat3n(v));
                return mat_to_table(S, out);
              })
          .addProperty("a_vector",
                       +[](const UnitCell *c) -> occ::Vec3 { return c->a_vector(); })
          .addProperty("b_vector",
                       +[](const UnitCell *c) -> occ::Vec3 { return c->b_vector(); })
          .addProperty("c_vector",
                       +[](const UnitCell *c) -> occ::Vec3 { return c->c_vector(); })
          .addProperty("a_star_vector",
                       +[](const UnitCell *c) -> occ::Vec3 { return c->a_star_vector(); })
          .addProperty("b_star_vector",
                       +[](const UnitCell *c) -> occ::Vec3 { return c->b_star_vector(); })
          .addProperty("c_star_vector",
                       +[](const UnitCell *c) -> occ::Vec3 { return c->c_star_vector(); })
          .addFunction("hkl_limits", &UnitCell::hkl_limits)
          .addFunction("__tostring",
                       +[](const UnitCell *cell) {
                         return fmt::format(
                             "<UnitCell {} ({:.5f}, {:.5f}, {:.5f})>",
                             cell->cell_type(), cell->a(), cell->b(),
                             cell->c());
                       })
        .endClass()

        .addFunction("cubic_cell", &cubic_cell)
        .addFunction("rhombohedral_cell", &rhombohedral_cell)
        .addFunction("tetragonal_cell", &tetragonal_cell)
        .addFunction("hexagonal_cell", &hexagonal_cell)
        .addFunction("orthorhombic_cell", &orthorhombic_cell)
        .addFunction("monoclinic_cell", &monoclinic_cell)
        .addFunction("triclinic_cell", &triclinic_cell)
      .endNamespace();
}

// -- Crystal --------------------------------------------------------------

void register_crystal(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<Crystal>("Crystal")
          .addConstructor<void (*)(const AsymmetricUnit &, const SpaceGroup &,
                                    const UnitCell &)>()
          // The vector-returning getters fail at runtime when bound as
          // properties under LuaBridge3 (Stack<vector<T>&> trips up).
          // Keep them as methods.
          .addFunction("symmetry_unique_molecules",
                       &Crystal::symmetry_unique_molecules)
          .addFunction("symmetry_unique_dimers",
                       &Crystal::symmetry_unique_dimers)
          .addProperty("unit_cell", &Crystal::unit_cell)
          .addFunction("unit_cell_molecules", &Crystal::unit_cell_molecules)
          .addFunction("unit_cell_atoms", &Crystal::unit_cell_atoms)
          .addFunction("unit_cell_dimers", &Crystal::unit_cell_dimers)
          .addFunction("atom_surroundings",
                       +[](const Crystal *c, int asym_idx) {
                         return c->atom_surroundings(asym_idx);
                       })
          .addFunction("atom_surroundings_default",
                       +[](const Crystal *c) {
                         return c->atom_surroundings(0);
                       })
          .addFunction("dimer_symmetry_string",
                       &Crystal::dimer_symmetry_string)
          .addFunction(
              "normalize_hydrogen_bondlengths",
              +[](Crystal *self) {
                return self->normalize_hydrogen_bondlengths();
              })
          .addFunction(
              "normalize_hydrogen_bondlengths_custom",
              +[](Crystal *self, const lb::LuaRef &custom_lengths) {
                ankerl::unordered_dense::map<int, double> map;
                // LuaBridge3's LuaRef doesn't expose begin/end directly;
                // walk the table via luabridge::Iterator.
                for (lb::Iterator it(custom_lengths); !it.isNil(); ++it) {
                  int k = it.key().unsafe_cast<int>();
                  double v = it.value().unsafe_cast<double>();
                  map[k] = v;
                }
                return self->normalize_hydrogen_bondlengths(map);
              })
          .addFunction("asymmetric_unit_atom_surroundings",
                       &Crystal::asymmetric_unit_atom_surroundings)
          .addProperty("num_sites", &Crystal::num_sites)
          .addFunction("labels", &Crystal::labels)
          .addFunction(
              "to_fractional",
              +[](const Crystal *c, lua_State *S, const lb::LuaRef &cart) {
                return mat_to_table(S, c->to_fractional(table_to_mat3n(cart)));
              })
          .addFunction(
              "to_cartesian",
              +[](const Crystal *c, lua_State *S, const lb::LuaRef &frac) {
                return mat_to_table(S, c->to_cartesian(table_to_mat3n(frac)));
              })
          .addProperty("volume", &Crystal::volume)
          .addProperty("density", &Crystal::density)
          .addProperty("space_group", &Crystal::space_group)
          .addFunction("slab", &Crystal::slab)
          .addProperty("asymmetric_unit",
                       +[](const Crystal *c) -> const AsymmetricUnit & {
                         return c->asymmetric_unit();
                       })
          .addFunction("__tostring",
                       +[](const Crystal *c) {
                         return fmt::format(
                             "<Crystal {} {}>",
                             c->asymmetric_unit().chemical_formula(),
                             c->space_group().symbol());
                       })
        .endClass()

        .addFunction("crystal_create_primitive_supercell",
                     &Crystal::create_primitive_supercell)
        .addFunction("crystal_from_cif_file",
                     +[](const std::string &filename) {
                       occ::io::CifParser parser;
                       return parser.parse_crystal_from_file(filename).value();
                     })
        .addFunction("crystal_from_cif_string",
                     +[](const std::string &contents) {
                       occ::io::CifParser parser;
                       return parser.parse_crystal_from_string(contents).value();
                     })
      .endNamespace();
}

// -- CrystalAtomRegion / dimer helpers -----------------------------------

void register_crystal_region_and_dimers(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<CrystalAtomRegion>("CrystalAtomRegion")
          .addProperty("frac_pos",
                       +[](const CrystalAtomRegion *r) -> occ::Mat3N { return r->frac_pos; })
          .addProperty("cart_pos",
                       +[](const CrystalAtomRegion *r) -> occ::Mat3N { return r->cart_pos; })
          .addProperty("asym_idx",
                       +[](const CrystalAtomRegion *r) -> occ::IVec { return r->asym_idx; })
          .addProperty("atomic_numbers",
                       +[](const CrystalAtomRegion *r) -> occ::IVec { return r->atomic_numbers; })
          .addProperty("symop",
                       +[](const CrystalAtomRegion *r) -> occ::IVec { return r->symop; })
          .addProperty("size", &CrystalAtomRegion::size)
          .addFunction("__tostring",
                       +[](const CrystalAtomRegion *r) {
                         return fmt::format("<CrystalAtomRegion (n={})>",
                                            r->size());
                       })
        .endClass()

        .beginClass<SymmetryRelatedDimer>("SymmetryRelatedDimer")
          .addProperty("unique_index", &SymmetryRelatedDimer::unique_index)
          .addProperty("dimer", &SymmetryRelatedDimer::dimer)
        .endClass()

        .beginClass<CrystalDimers>("CrystalDimers")
          .addProperty("radius", &CrystalDimers::radius)
          .addFunction("unique_dimers",
                       +[](const CrystalDimers *c, lua_State *S) {
                         lb::LuaRef out = lb::newTable(S);
                         for (size_t i = 0; i < c->unique_dimers.size(); ++i) {
                           out[static_cast<int>(i + 1)] = c->unique_dimers[i];
                         }
                         return out;
                       })
          .addFunction("molecule_neighbors",
                       +[](const CrystalDimers *c, lua_State *S) {
                         lb::LuaRef out = lb::newTable(S);
                         const auto &mn = c->molecule_neighbors;
                         for (size_t i = 0; i < mn.size(); ++i) {
                           lb::LuaRef inner = lb::newTable(S);
                           for (size_t j = 0; j < mn[i].size(); ++j) {
                             inner[static_cast<int>(j + 1)] = mn[i][j];
                           }
                           out[static_cast<int>(i + 1)] = inner;
                         }
                         return out;
                       })
        .endClass()
      .endNamespace();
}

// -- SiteIndex / DimerIndex / DimerMappingTable --------------------------

void register_site_and_dimer_index(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<SiteIndex>("SiteIndex")
          .addProperty("offset", &SiteIndex::offset)
          .addProperty("hkl", &SiteIndex::hkl)
          .addFunction("__tostring",
                       +[](const SiteIndex *s) {
                         return fmt::format("<SiteIndex {} [{} {} {}]>",
                                            s->offset, s->hkl.h, s->hkl.k,
                                            s->hkl.l);
                       })
        .endClass()

        .beginClass<DimerIndex>("DimerIndex")
          .addConstructor<void (*)(SiteIndex, SiteIndex)>()
          .addProperty("a", &DimerIndex::a)
          .addProperty("b", &DimerIndex::b)
          .addFunction("hkl_difference", &DimerIndex::hkl_difference)
          .addFunction("__eq",
                       +[](const DimerIndex *a, const DimerIndex &b) {
                         return *a == b;
                       })
          .addFunction("__lt",
                       +[](const DimerIndex *a, const DimerIndex &b) {
                         return *a < b;
                       })
          .addFunction("__tostring",
                       +[](const DimerIndex *d) {
                         return fmt::format(
                             "<DimerIndex {} [{} {} {}] - {} [{} {} {}]>",
                             d->a.offset, d->a.hkl.h, d->a.hkl.k, d->a.hkl.l,
                             d->b.offset, d->b.hkl.h, d->b.hkl.k, d->b.hkl.l);
                       })
        .endClass()

        .beginClass<DimerMappingTable>("DimerMappingTable")
          .addFunction("symmetry_unique_dimer",
                       &DimerMappingTable::symmetry_unique_dimer)
          .addFunction("symmetry_related_dimers",
                       &DimerMappingTable::symmetry_related_dimers)
          .addFunction("unique_dimers", &DimerMappingTable::unique_dimers)
          .addFunction("symmetry_unique_dimers",
                       &DimerMappingTable::symmetry_unique_dimers)
          .addFunction("dimer_positions", &DimerMappingTable::dimer_positions)
          .addFunction(
              "dimer_index",
              +[](const DimerMappingTable *t, const occ::core::Dimer &d) {
                return t->dimer_index(d);
              })
          .addFunction(
              "dimer_index_from_positions",
              +[](const DimerMappingTable *t, const lb::LuaRef &a,
                  const lb::LuaRef &b) {
                return t->dimer_index(table_to_vec3(a), table_to_vec3(b));
              })
          .addFunction("canonical_dimer_index",
                       &DimerMappingTable::canonical_dimer_index)
          .addFunction("__tostring",
                       +[](const DimerMappingTable *t) {
                         return fmt::format(
                             "<DimerMappingTable n_unique={} n_sym_unique={}>",
                             t->unique_dimers().size(),
                             t->symmetry_unique_dimers().size());
                       })
        .endClass()

        .addFunction("dimer_mapping_table_build",
                     &DimerMappingTable::build_dimer_table)
        .addFunction("dimer_mapping_table_normalized_index",
                     &DimerMappingTable::normalized_dimer_index)
      .endNamespace();
}

// -- Surface / SurfaceCutResult ------------------------------------------

void register_surface(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")
        .beginClass<SurfaceCutResult>("SurfaceCutResult")
          .addConstructor<void (*)(const CrystalDimers &)>()
          .addProperty("molecules", &SurfaceCutResult::molecules)
          .addProperty("exyz", &SurfaceCutResult::exyz)
          .addProperty("above", &SurfaceCutResult::above)
          .addProperty("below", &SurfaceCutResult::below)
          .addProperty("slab", &SurfaceCutResult::slab)
          .addProperty("bulk", &SurfaceCutResult::bulk)
          .addProperty("depth_scale", &SurfaceCutResult::depth_scale)
          .addProperty("get_basis",
                       +[](const SurfaceCutResult *r) -> occ::Mat3 { return r->basis; })
          .addFunction("set_basis",
                       +[](SurfaceCutResult *r, const lb::LuaRef &t) {
                         r->basis = table_to_mat3(t);
                       })
          .addProperty("cut_offset", &SurfaceCutResult::cut_offset)
          .addFunction("total_above", &SurfaceCutResult::total_above)
          .addFunction("total_below", &SurfaceCutResult::total_below)
          .addFunction("total_slab", &SurfaceCutResult::total_slab)
          .addFunction("total_bulk", &SurfaceCutResult::total_bulk)
          .addFunction("unique_counts_above",
                       &SurfaceCutResult::unique_counts_above)
          .addFunction("__tostring",
                       +[](const SurfaceCutResult *r) {
                         return fmt::format(
                             "<SurfaceCutResult n_molecules={} "
                             "depth_scale={:.3f}>",
                             r->molecules.size(), r->depth_scale);
                       })
        .endClass()

        .beginClass<CrystalSurfaceGenerationParameters>(
            "CrystalSurfaceGenerationParameters")
          .addConstructor<void (*)()>()
          .addPropertyReadWrite("d_min",
                                &CrystalSurfaceGenerationParameters::d_min)
          .addPropertyReadWrite("d_max",
                                &CrystalSurfaceGenerationParameters::d_max)
          .addPropertyReadWrite("unique",
                                &CrystalSurfaceGenerationParameters::unique)
          .addPropertyReadWrite("reduced",
                                &CrystalSurfaceGenerationParameters::reduced)
          .addPropertyReadWrite(
              "systematic_absences_allowed",
              &CrystalSurfaceGenerationParameters::systematic_absences_allowed)
        .endClass()

        .beginClass<Surface>("Surface")
          .addConstructor<void (*)(const HKL &, const Crystal &)>()
          .addProperty("depth", &Surface::depth)
          .addProperty("d", &Surface::d)
          .addFunction("print", &Surface::print)
          .addProperty("normal_vector",
                       +[](const Surface *s) -> occ::Vec3 { return s->normal_vector(); })
          .addProperty("hkl", &Surface::hkl)
          .addProperty("depth_vector",
                       +[](const Surface *s) -> occ::Vec3 { return s->depth_vector(); })
          .addProperty("a_vector",
                       +[](const Surface *s) -> occ::Vec3 { return s->a_vector(); })
          .addProperty("b_vector",
                       +[](const Surface *s) -> occ::Vec3 { return s->b_vector(); })
          .addProperty("area", &Surface::area)
          .addProperty("dipole", &Surface::dipole)
          .addFunction("basis_matrix",
                       +[](const Surface *s, lua_State *S, double depth_scale) {
                         return mat_to_table(S, s->basis_matrix(depth_scale));
                       })
          .addProperty("basis_matrix_default",
                       +[](const Surface *s) -> occ::Mat3 { return s->basis_matrix(1.0); })
          .addFunction(
              "find_molecule_cell_translations",
              +[](Surface *s,
                  const std::vector<occ::core::Molecule> &mols, double depth,
                  double cut_offset) {
                return s->find_molecule_cell_translations(mols, depth,
                                                            cut_offset);
              })
          .addFunction(
              "find_molecule_cell_translations_default",
              +[](Surface *s,
                  const std::vector<occ::core::Molecule> &mols, double depth) {
                return s->find_molecule_cell_translations(mols, depth, 0.0);
              })
          .addFunction(
              "count_crystal_dimers_cut_by_surface",
              +[](Surface *s, const CrystalDimers &dimers, double cut_offset) {
                return s->count_crystal_dimers_cut_by_surface(dimers,
                                                                cut_offset);
              })
          .addFunction(
              "count_crystal_dimers_cut_by_surface_default",
              +[](Surface *s, const CrystalDimers &dimers) {
                return s->count_crystal_dimers_cut_by_surface(dimers, 0.0);
              })
          .addFunction(
              "possible_cuts",
              +[](const Surface *s, const lb::LuaRef &unique_positions,
                  double epsilon) {
                return s->possible_cuts(table_to_mat3n(unique_positions),
                                         epsilon);
              })
          .addFunction(
              "possible_cuts_default",
              +[](const Surface *s, const lb::LuaRef &unique_positions) {
                return s->possible_cuts(table_to_mat3n(unique_positions),
                                         1e-6);
              })
          .addFunction("__tostring",
                       +[](const Surface *surface) {
                         return fmt::format(
                             "<Surface hkl=[{} {} {}] d={:.3f}>",
                             surface->hkl().h, surface->hkl().k,
                             surface->hkl().l, surface->d());
                       })
        .endClass()

        .addFunction("surface_check_systematic_absence",
                     &Surface::check_systematic_absence)
        .addFunction("surface_faces_are_equivalent",
                     &Surface::faces_are_equivalent)
        .addFunction(
            "generate_surfaces",
            +[](const Crystal &crystal,
                const CrystalSurfaceGenerationParameters &params) {
              return generate_surfaces(crystal, params);
            })
        .addFunction("generate_surfaces_default",
                     +[](const Crystal &crystal) {
                       return generate_surfaces(
                           crystal, CrystalSurfaceGenerationParameters{});
                     })
      .endNamespace();
}

} // namespace

void register_crystal_bindings(lua_State *L) {
  register_hkl(L);
  register_symmetry_operation(L);
  register_spacegroup(L);
  register_asymmetric_unit(L);
  register_unitcell(L);
  register_crystal(L);
  register_crystal_region_and_dimers(L);
  register_site_and_dimer_index(L);
  register_surface(L);
}

} // namespace occ::lua_bindings
