#include <Eigen/LU>
#include "crystal_bindings.h"
#include "eigen_conv.h"
#include <ankerl/unordered_dense.h>
#include <fmt/core.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_mapping_table.h>
#include <occ/crystal/surface.h>
#include <occ/io/cifparser.h>

// SymmetryOperation has an `operator()(const Mat3N&) const` that sol2's
// automagic enrollment tries to bind unconditionally for any T with a
// deducible call-operator. The default binding then instantiates the
// container-conversion machinery for the Mat3N return type, which fails
// to compile because Eigen iterators don't satisfy sol2's `decltype(*it)`
// requirements ("cannot form a reference to void"). Opting out of the
// automagic block entirely skips the auto operator() detection — we
// register everything we care about (call, comparisons, to_string)
// explicitly below.
namespace sol {
template <>
struct is_automagical<occ::crystal::SymmetryOperation> : std::false_type {};
} // namespace sol

namespace occ::lua_bindings {

using namespace occ::crystal;
using SymmetryRelatedDimer = occ::crystal::CrystalDimers::SymmetryRelatedDimer;

namespace {

// -- HKL ------------------------------------------------------------------

void register_hkl(sol::table &m) {
  m.new_usertype<HKL>(
      "HKL",
      sol::call_constructor,
      sol::constructors<HKL(int, int, int)>(),
      "h", &HKL::h,
      "k", &HKL::k,
      "l", &HKL::l,
      "d", &HKL::d,
      "vector",
      [](const HKL &hkl, sol::this_state s) {
        return hkl.vector();
      },
      sol::meta_function::to_string, [](const HKL &hkl) {
        return fmt::format("<HKL [{} {} {}]>", hkl.h, hkl.k, hkl.l);
      });

  m.set_function("hkl_floor",
                 [](const sol::table &v) { return HKL::floor(table_to_vec3(v)); });
  m.set_function("hkl_ceil",
                 [](const sol::table &v) { return HKL::ceil(table_to_vec3(v)); });
  m.set_function("hkl_maximum", []() { return HKL::maximum(); });
  m.set_function("hkl_minimum", []() { return HKL::minimum(); });
}

// -- SymmetryOperation ----------------------------------------------------

void register_symmetry_operation(sol::table &m) {
  m.new_usertype<SymmetryOperationFormat>(
      "SymmetryOperationFormat",
      sol::call_constructor, sol::factories([]() {
        return SymmetryOperationFormat{};
      }),
      "fmt_string", &SymmetryOperationFormat::fmt_string,
      "delimiter", &SymmetryOperationFormat::delimiter);

  m.new_usertype<SymmetryOperation>(
      "SymmetryOperation",
      sol::call_constructor,
      sol::factories(
          [](const sol::table &seitz) {
            return SymmetryOperation(table_to_mat4(seitz));
          },
          [](const std::string &s) { return SymmetryOperation(s); },
          [](int code) { return SymmetryOperation(code); }),
      "to_int", &SymmetryOperation::to_int,
      "to_string",
      [](const SymmetryOperation &op,
         sol::optional<SymmetryOperationFormat> format) {
        return op.to_string(format.value_or(SymmetryOperationFormat{}));
      },
      "inverted", &SymmetryOperation::inverted,
      "translated",
      [](const SymmetryOperation &op, const sol::table &t) {
        return op.translated(table_to_vec3(t));
      },
      "is_identity", &SymmetryOperation::is_identity,
      "apply",
      [](const SymmetryOperation &op, const sol::table &positions,
         sol::this_state s) {
        return op.apply(table_to_mat3n(positions));
      },
      "seitz",
      [](const SymmetryOperation &op, sol::this_state s) {
        return op.seitz();
      },
      "rotation",
      [](const SymmetryOperation &op, sol::this_state s) {
        return op.rotation();
      },
      "cartesian_rotation",
      [](const SymmetryOperation &op, const UnitCell &cell,
         sol::this_state s) {
        return op.cartesian_rotation(cell);
      },
      "translation",
      [](const SymmetryOperation &op, sol::this_state s) {
        return op.translation();
      },
      "has_translation", &SymmetryOperation::has_translation,
      sol::meta_function::call,
      [](const SymmetryOperation &op, const sol::table &positions,
         sol::this_state s) {
        return op(table_to_mat3n(positions));
      },
      sol::meta_function::multiplication,
      [](const SymmetryOperation &a, const SymmetryOperation &b) {
        return a * b;
      },
      sol::meta_function::equal_to,
      [](const SymmetryOperation &a, const SymmetryOperation &b) {
        return a == b;
      },
      sol::meta_function::less_than,
      [](const SymmetryOperation &a, const SymmetryOperation &b) {
        return a < b;
      },
      sol::meta_function::less_than_or_equal_to,
      [](const SymmetryOperation &a, const SymmetryOperation &b) {
        return a <= b;
      },
      sol::meta_function::to_string, [](const SymmetryOperation &op) {
        return fmt::format("<SymmetryOperation '{}'>", op.to_string());
      });
}

// -- SpaceGroup -----------------------------------------------------------

void register_spacegroup(sol::table &m) {
  m.new_usertype<SpaceGroup>(
      "SpaceGroup",
      sol::call_constructor,
      sol::factories(
          []() { return SpaceGroup(); },
          [](int num) { return SpaceGroup(num); },
          [](const std::string &symbol) { return SpaceGroup(symbol); },
          [](const std::vector<std::string> &symops) {
            return SpaceGroup(symops);
          },
          [](const std::vector<SymmetryOperation> &symops) {
            return SpaceGroup(symops);
          }),
      "number", &SpaceGroup::number,
      "symbol", sol::readonly_property(&SpaceGroup::symbol),
      "short_name", sol::readonly_property(&SpaceGroup::short_name),
      "symmetry_operations",
      sol::readonly_property(&SpaceGroup::symmetry_operations),
      "has_H_R_choice", &SpaceGroup::has_H_R_choice,
      // Both return (symop_indices, transformed_positions); package as a
      // table with explicit keys so callers can destructure cleanly.
      "apply_all_symmetry_operations",
      [](const SpaceGroup &sg, const sol::table &positions,
         sol::this_state s) {
        sol::state_view lua(s);
        auto p = sg.apply_all_symmetry_operations(table_to_mat3n(positions));
        sol::table out = lua.create_table(0, 2);
        out["indices"] = vec_to_table(s, p.first);
        out["positions"] = mat_to_table(s, p.second);
        return out;
      },
      "apply_rotations",
      [](const SpaceGroup &sg, const sol::table &positions,
         sol::this_state s) {
        sol::state_view lua(s);
        auto p = sg.apply_rotations(table_to_mat3n(positions));
        sol::table out = lua.create_table(0, 2);
        out["indices"] = vec_to_table(s, p.first);
        out["positions"] = mat_to_table(s, p.second);
        return out;
      },
      sol::meta_function::to_string, [](const SpaceGroup &sg) {
        return fmt::format("<SpaceGroup {} ({}), {} operations>",
                           sg.symbol(), sg.number(),
                           sg.symmetry_operations().size());
      });
}

// -- AsymmetricUnit -------------------------------------------------------

void register_asymmetric_unit(sol::table &m) {
  m.new_usertype<AsymmetricUnit>(
      "AsymmetricUnit",
      sol::call_constructor,
      sol::factories(
          []() { return AsymmetricUnit(); },
          [](const sol::table &positions, const sol::table &atomic_numbers) {
            const int n = static_cast<int>(atomic_numbers.size());
            IVec z(n);
            for (int i = 0; i < n; ++i) z(i) = atomic_numbers.get<int>(i + 1);
            return AsymmetricUnit(table_to_mat3n(positions), z);
          },
          [](const sol::table &positions, const sol::table &atomic_numbers,
             const std::vector<std::string> &labels) {
            const int n = static_cast<int>(atomic_numbers.size());
            IVec z(n);
            for (int i = 0; i < n; ++i) z(i) = atomic_numbers.get<int>(i + 1);
            return AsymmetricUnit(table_to_mat3n(positions), z, labels);
          }),
      "positions",
      sol::property(
          [](const AsymmetricUnit &a, sol::this_state s) {
            return a.positions;
          },
          [](AsymmetricUnit &a, const sol::table &t) {
            a.positions = table_to_mat3n(t);
          }),
      "atomic_numbers",
      sol::property(
          [](const AsymmetricUnit &a, sol::this_state s) {
            return a.atomic_numbers;
          },
          [](AsymmetricUnit &a, const sol::table &t) {
            const int n = static_cast<int>(t.size());
            IVec z(n);
            for (int i = 0; i < n; ++i) z(i) = t.get<int>(i + 1);
            a.atomic_numbers = z;
          }),
      "occupations",
      sol::property(
          [](const AsymmetricUnit &a, sol::this_state s) {
            return a.occupations;
          },
          [](AsymmetricUnit &a, const sol::table &t) {
            a.occupations = table_to_vecx(t);
          }),
      "charges",
      sol::property(
          [](const AsymmetricUnit &a, sol::this_state s) {
            return a.charges;
          },
          [](AsymmetricUnit &a, const sol::table &t) {
            a.charges = table_to_vecx(t);
          }),
      "labels", &AsymmetricUnit::labels,
      sol::meta_function::length, &AsymmetricUnit::size,
      "size", &AsymmetricUnit::size,
      sol::meta_function::to_string, [](const AsymmetricUnit &asym) {
        return fmt::format("<AsymmetricUnit {}>", asym.chemical_formula());
      });
}

// -- UnitCell -------------------------------------------------------------

void register_unitcell(sol::table &m) {
  m.new_usertype<UnitCell>(
      "UnitCell",
      sol::call_constructor,
      sol::factories(
          []() { return UnitCell(); },
          [](double a, double b, double c, double alpha, double beta,
             double gamma) { return UnitCell(a, b, c, alpha, beta, gamma); },
          [](const sol::table &lengths, const sol::table &angles) {
            return UnitCell(table_to_vec3(lengths), table_to_vec3(angles));
          },
          [](const sol::table &lattice_vectors) {
            return UnitCell(table_to_mat3(lattice_vectors));
          }),
      "a", sol::property(&UnitCell::a, &UnitCell::set_a),
      "b", sol::property(&UnitCell::b, &UnitCell::set_b),
      "c", sol::property(&UnitCell::c, &UnitCell::set_c),
      "alpha", sol::property(&UnitCell::alpha, &UnitCell::set_alpha),
      "beta", sol::property(&UnitCell::beta, &UnitCell::set_beta),
      "gamma", sol::property(&UnitCell::gamma, &UnitCell::set_gamma),
      "volume", sol::readonly_property(&UnitCell::volume),
      "direct",
      sol::readonly_property([](const UnitCell &c, sol::this_state s) {
        return c.direct();
      }),
      "reciprocal",
      sol::readonly_property([](const UnitCell &c, sol::this_state s) {
        return c.reciprocal();
      }),
      "inverse",
      sol::readonly_property([](const UnitCell &c, sol::this_state s) {
        return c.inverse();
      }),
      "lengths",
      sol::readonly_property([](const UnitCell &c, sol::this_state s) {
        return c.lengths();
      }),
      "angles",
      sol::readonly_property([](const UnitCell &c, sol::this_state s) {
        return c.angles();
      }),
      "is_cubic", &UnitCell::is_cubic,
      "is_triclinic", &UnitCell::is_triclinic,
      "is_monoclinic", &UnitCell::is_monoclinic,
      "is_orthorhombic", &UnitCell::is_orthorhombic,
      "is_tetragonal", &UnitCell::is_tetragonal,
      "is_rhombohedral", &UnitCell::is_rhombohedral,
      "is_hexagonal", &UnitCell::is_hexagonal,
      "is_orthogonal", &UnitCell::is_orthogonal,
      "cell_type", &UnitCell::cell_type,
      // to_cartesian / to_fractional / to_reciprocal are `inline auto`
      // in occ — they return Eigen Product expressions, not concrete
      // Mat3N. Force materialization with an explicit `-> Mat3N`.
      "to_cartesian",
      [](const UnitCell &c, const occ::Mat3N &frac) -> occ::Mat3N {
        return c.to_cartesian(frac);
      },
      "to_fractional",
      [](const UnitCell &c, const occ::Mat3N &cart) -> occ::Mat3N {
        return c.to_fractional(cart);
      },
      "to_reciprocal",
      [](const UnitCell &c, const occ::Mat3N &v) -> occ::Mat3N {
        return c.to_reciprocal(v);
      },
      "a_vector",
      // UnitCell's *_vector / *_star_vector return Eigen Block / column
      // expressions, not concrete Vec3 — force materialization so sol2
      // pushes the registered Vec3 usertype.
      [](const UnitCell &c) -> occ::Vec3 { return c.a_vector(); },
      "b_vector",
      [](const UnitCell &c) -> occ::Vec3 { return c.b_vector(); },
      "c_vector",
      [](const UnitCell &c) -> occ::Vec3 { return c.c_vector(); },
      "a_star_vector",
      [](const UnitCell &c) -> occ::Vec3 { return c.a_star_vector(); },
      "b_star_vector",
      [](const UnitCell &c) -> occ::Vec3 { return c.b_star_vector(); },
      "c_star_vector",
      [](const UnitCell &c) -> occ::Vec3 { return c.c_star_vector(); },
      "hkl_limits",
      [](const UnitCell &c, double d_min) { return c.hkl_limits(d_min); },
      sol::meta_function::to_string, [](const UnitCell &cell) {
        return fmt::format("<UnitCell {} ({:.5f}, {:.5f}, {:.5f})>",
                           cell.cell_type(), cell.a(), cell.b(), cell.c());
      });

  // Standard cell shape helpers.
  m.set_function("cubic_cell", &cubic_cell);
  m.set_function("rhombohedral_cell", &rhombohedral_cell);
  m.set_function("tetragonal_cell", &tetragonal_cell);
  m.set_function("hexagonal_cell", &hexagonal_cell);
  m.set_function("orthorhombic_cell", &orthorhombic_cell);
  m.set_function("monoclinic_cell", &monoclinic_cell);
  m.set_function("triclinic_cell", &triclinic_cell);
}

// -- Crystal --------------------------------------------------------------

void register_crystal(sol::table &m) {
  // The placeholder usertype registered by occ_module.cpp for stage 1 is
  // replaced now that we have full coverage. sol2 will silently overwrite
  // the entry — `new_usertype` rebinds the metatable.
  m.new_usertype<Crystal>(
      "Crystal",
      sol::call_constructor,
      sol::factories(
          [](const AsymmetricUnit &asym, const SpaceGroup &sg,
             const UnitCell &cell) { return Crystal(asym, sg, cell); }),
      "symmetry_unique_molecules", &Crystal::symmetry_unique_molecules,
      "symmetry_unique_dimers", &Crystal::symmetry_unique_dimers,
      "unit_cell", &Crystal::unit_cell,
      "unit_cell_molecules", &Crystal::unit_cell_molecules,
      "unit_cell_atoms", &Crystal::unit_cell_atoms,
      "unit_cell_dimers", &Crystal::unit_cell_dimers,
      "atom_surroundings",
      [](const Crystal &c, sol::optional<int> asym_idx) {
        return c.atom_surroundings(asym_idx.value_or(0));
      },
      "dimer_symmetry_string", &Crystal::dimer_symmetry_string,
      "normalize_hydrogen_bondlengths",
      sol::overload(
          [](Crystal &self) { return self.normalize_hydrogen_bondlengths(); },
          [](Crystal &self, const sol::table &custom_lengths) {
            ankerl::unordered_dense::map<int, double> map;
            // Lua table: { [Z] = length, ... } — iterate pairs.
            custom_lengths.for_each(
                [&](sol::object key, sol::object value) {
                  map[key.as<int>()] = value.as<double>();
                });
            return self.normalize_hydrogen_bondlengths(map);
          }),
      "asymmetric_unit_atom_surroundings",
      &Crystal::asymmetric_unit_atom_surroundings,
      "num_sites", &Crystal::num_sites,
      "labels", &Crystal::labels,
      "to_fractional",
      [](const Crystal &c, const sol::table &cart, sol::this_state s) {
        return c.to_fractional(table_to_mat3n(cart));
      },
      "to_cartesian",
      [](const Crystal &c, const sol::table &frac, sol::this_state s) {
        return c.to_cartesian(table_to_mat3n(frac));
      },
      "volume", &Crystal::volume,
      "density", &Crystal::density,
      "space_group", &Crystal::space_group,
      "slab",
      [](const Crystal &c, const HKL &lower, const HKL &upper) {
        return c.slab(lower, upper);
      },
      "asymmetric_unit",
      [](const Crystal &c) -> const AsymmetricUnit & {
        return c.asymmetric_unit();
      },
      sol::meta_function::to_string, [](const Crystal &c) {
        return fmt::format("<Crystal {} {}>",
                           c.asymmetric_unit().chemical_formula(),
                           c.space_group().symbol());
      });

  m.set_function("crystal_create_primitive_supercell",
                 &Crystal::create_primitive_supercell);
  m.set_function("crystal_from_cif_file", [](const std::string &filename) {
    occ::io::CifParser parser;
    return parser.parse_crystal_from_file(filename).value();
  });
  m.set_function("crystal_from_cif_string", [](const std::string &contents) {
    occ::io::CifParser parser;
    return parser.parse_crystal_from_string(contents).value();
  });
}

// -- CrystalAtomRegion / dimer helpers -----------------------------------

void register_crystal_region_and_dimers(sol::table &m) {
  m.new_usertype<CrystalAtomRegion>(
      "CrystalAtomRegion", sol::no_constructor,
      "frac_pos",
      sol::readonly_property([](const CrystalAtomRegion &r, sol::this_state s) {
        return r.frac_pos;
      }),
      "cart_pos",
      sol::readonly_property([](const CrystalAtomRegion &r, sol::this_state s) {
        return r.cart_pos;
      }),
      "asym_idx",
      sol::readonly_property([](const CrystalAtomRegion &r, sol::this_state s) {
        return r.asym_idx;
      }),
      "atomic_numbers",
      sol::readonly_property([](const CrystalAtomRegion &r, sol::this_state s) {
        return r.atomic_numbers;
      }),
      "symop",
      sol::readonly_property([](const CrystalAtomRegion &r, sol::this_state s) {
        return r.symop;
      }),
      "size", &CrystalAtomRegion::size,
      sol::meta_function::to_string, [](const CrystalAtomRegion &r) {
        return fmt::format("<CrystalAtomRegion (n={})>", r.size());
      });

  m.new_usertype<SymmetryRelatedDimer>(
      "SymmetryRelatedDimer", sol::no_constructor,
      "unique_index", &SymmetryRelatedDimer::unique_index,
      "dimer", &SymmetryRelatedDimer::dimer);

  // `unique_dimers` and `molecule_neighbors` are STL containers; the
  // default sol2 container metatable wants `operator==` on the value
  // type for find/contains. SymmetryRelatedDimer doesn't define one.
  // `sol::as_table` short-circuits the container machinery and pushes
  // the vector as a plain Lua table of registered-usertype elements.
  m.new_usertype<CrystalDimers>(
      "CrystalDimers", sol::no_constructor,
      "radius", &CrystalDimers::radius,
      "unique_dimers",
      [](const CrystalDimers &c) { return sol::as_table(c.unique_dimers); },
      "molecule_neighbors", [](const CrystalDimers &c) {
        return sol::as_table(c.molecule_neighbors);
      });
}

// -- SiteIndex / DimerIndex / DimerMappingTable --------------------------

void register_site_and_dimer_index(sol::table &m) {
  m.new_usertype<SiteIndex>(
      "SiteIndex", sol::no_constructor,
      "offset", &SiteIndex::offset,
      "hkl", &SiteIndex::hkl,
      sol::meta_function::to_string, [](const SiteIndex &s) {
        return fmt::format("<SiteIndex {} [{} {} {}]>", s.offset, s.hkl.h,
                           s.hkl.k, s.hkl.l);
      });

  m.new_usertype<DimerIndex>(
      "DimerIndex",
      sol::call_constructor,
      sol::constructors<DimerIndex(SiteIndex, SiteIndex)>(),
      "a", &DimerIndex::a,
      "b", &DimerIndex::b,
      "hkl_difference", &DimerIndex::hkl_difference,
      sol::meta_function::equal_to,
      [](const DimerIndex &a, const DimerIndex &b) { return a == b; },
      sol::meta_function::less_than,
      [](const DimerIndex &a, const DimerIndex &b) { return a < b; },
      sol::meta_function::to_string, [](const DimerIndex &d) {
        return fmt::format("<DimerIndex {} [{} {} {}] - {} [{} {} {}]>",
                           d.a.offset, d.a.hkl.h, d.a.hkl.k, d.a.hkl.l,
                           d.b.offset, d.b.hkl.h, d.b.hkl.k, d.b.hkl.l);
      });

  m.new_usertype<DimerMappingTable>(
      "DimerMappingTable", sol::no_constructor,
      "symmetry_unique_dimer", &DimerMappingTable::symmetry_unique_dimer,
      "symmetry_related_dimers", &DimerMappingTable::symmetry_related_dimers,
      "unique_dimers",
      sol::readonly_property(&DimerMappingTable::unique_dimers),
      "symmetry_unique_dimers",
      sol::readonly_property(&DimerMappingTable::symmetry_unique_dimers),
      "dimer_positions", &DimerMappingTable::dimer_positions,
      "dimer_index",
      sol::overload(
          [](const DimerMappingTable &t, const occ::core::Dimer &d) {
            return t.dimer_index(d);
          },
          [](const DimerMappingTable &t, const sol::table &a,
             const sol::table &b) {
            return t.dimer_index(table_to_vec3(a), table_to_vec3(b));
          }),
      "canonical_dimer_index", &DimerMappingTable::canonical_dimer_index,
      sol::meta_function::to_string, [](const DimerMappingTable &t) {
        return fmt::format("<DimerMappingTable n_unique={} n_sym_unique={}>",
                           t.unique_dimers().size(),
                           t.symmetry_unique_dimers().size());
      });

  m.set_function("dimer_mapping_table_build",
                 &DimerMappingTable::build_dimer_table);
  m.set_function("dimer_mapping_table_normalized_index",
                 &DimerMappingTable::normalized_dimer_index);
}

// -- Surface / SurfaceCutResult ------------------------------------------

void register_surface(sol::table &m) {
  m.new_usertype<SurfaceCutResult>(
      "SurfaceCutResult",
      sol::call_constructor,
      sol::constructors<SurfaceCutResult(const CrystalDimers &)>(),
      "molecules", &SurfaceCutResult::molecules,
      "exyz", &SurfaceCutResult::exyz,
      "above", &SurfaceCutResult::above,
      "below", &SurfaceCutResult::below,
      "slab", &SurfaceCutResult::slab,
      "bulk", &SurfaceCutResult::bulk,
      "depth_scale", &SurfaceCutResult::depth_scale,
      "basis",
      sol::property(
          [](const SurfaceCutResult &r, sol::this_state s) {
            return r.basis;
          },
          [](SurfaceCutResult &r, const sol::table &t) {
            r.basis = table_to_mat3(t);
          }),
      "cut_offset", &SurfaceCutResult::cut_offset,
      "total_above", &SurfaceCutResult::total_above,
      "total_below", &SurfaceCutResult::total_below,
      "total_slab", &SurfaceCutResult::total_slab,
      "total_bulk", &SurfaceCutResult::total_bulk,
      "unique_counts_above", &SurfaceCutResult::unique_counts_above,
      sol::meta_function::to_string, [](const SurfaceCutResult &r) {
        return fmt::format(
            "<SurfaceCutResult n_molecules={} depth_scale={:.3f}>",
            r.molecules.size(), r.depth_scale);
      });

  m.new_usertype<CrystalSurfaceGenerationParameters>(
      "CrystalSurfaceGenerationParameters",
      sol::call_constructor,
      sol::factories([]() { return CrystalSurfaceGenerationParameters{}; }),
      "d_min", &CrystalSurfaceGenerationParameters::d_min,
      "d_max", &CrystalSurfaceGenerationParameters::d_max,
      "unique", &CrystalSurfaceGenerationParameters::unique,
      "reduced", &CrystalSurfaceGenerationParameters::reduced,
      "systematic_absences_allowed",
      &CrystalSurfaceGenerationParameters::systematic_absences_allowed);

  m.new_usertype<Surface>(
      "Surface",
      sol::call_constructor,
      sol::constructors<Surface(const HKL &, const Crystal &)>(),
      "depth", &Surface::depth,
      "d", &Surface::d,
      "print", &Surface::print,
      "normal_vector",
      [](const Surface &s, sol::this_state st) {
        return vec_to_table(st, s.normal_vector());
      },
      "hkl", sol::readonly_property(&Surface::hkl),
      "depth_vector",
      sol::readonly_property([](const Surface &s, sol::this_state st) {
        return vec_to_table(st, s.depth_vector());
      }),
      "a_vector",
      sol::readonly_property([](const Surface &s, sol::this_state st) {
        return vec_to_table(st, s.a_vector());
      }),
      "b_vector",
      sol::readonly_property([](const Surface &s, sol::this_state st) {
        return vec_to_table(st, s.b_vector());
      }),
      "area", &Surface::area,
      "dipole", &Surface::dipole,
      "basis_matrix",
      [](const Surface &s, sol::optional<double> depth_scale,
         sol::this_state st) {
        return mat_to_table(st, s.basis_matrix(depth_scale.value_or(1.0)));
      },
      "find_molecule_cell_translations",
      [](Surface &s, const std::vector<occ::core::Molecule> &mols,
         double depth, sol::optional<double> cut_offset) {
        return s.find_molecule_cell_translations(mols, depth,
                                                   cut_offset.value_or(0.0));
      },
      "count_crystal_dimers_cut_by_surface",
      [](Surface &s, const CrystalDimers &dimers,
         sol::optional<double> cut_offset) {
        return s.count_crystal_dimers_cut_by_surface(
            dimers, cut_offset.value_or(0.0));
      },
      "possible_cuts",
      [](const Surface &s, const sol::table &unique_positions,
         sol::optional<double> epsilon) {
        return s.possible_cuts(table_to_mat3n(unique_positions),
                                epsilon.value_or(1e-6));
      },
      sol::meta_function::to_string, [](const Surface &surface) {
        return fmt::format("<Surface hkl=[{} {} {}] d={:.3f}>",
                           surface.hkl().h, surface.hkl().k, surface.hkl().l,
                           surface.d());
      });

  m.set_function("surface_check_systematic_absence",
                 &Surface::check_systematic_absence);
  m.set_function("surface_faces_are_equivalent", &Surface::faces_are_equivalent);
  m.set_function("generate_surfaces",
                 [](const Crystal &crystal,
                    sol::optional<CrystalSurfaceGenerationParameters> params) {
                   return generate_surfaces(
                       crystal,
                       params.value_or(CrystalSurfaceGenerationParameters{}));
                 });
}

} // namespace

void register_crystal_bindings(sol::state_view, sol::table &occ_module) {
  register_hkl(occ_module);
  register_symmetry_operation(occ_module);
  register_spacegroup(occ_module);
  register_asymmetric_unit(occ_module);
  register_unitcell(occ_module);
  register_crystal(occ_module);
  register_crystal_region_and_dimers(occ_module);
  register_site_and_dimer_index(occ_module);
  register_surface(occ_module);
}

} // namespace occ::lua_bindings
