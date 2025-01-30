#include "crystal_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_mapping_table.h>
#include <occ/crystal/surface.h>
#include <occ/io/cifparser.h>

using namespace nb::literals;
using namespace occ::crystal;
using SymmetryRelatedDimer = occ::crystal::CrystalDimers::SymmetryRelatedDimer;

nb::module_ register_crystal_bindings(nb::module_ &m) {

  nb::class_<HKL>(m, "HKL")
      .def(nb::init<int, int, int>())
      .def("d", &HKL::d)
      .def_static("floor", &HKL::floor)
      .def_static("ceil", &HKL::ceil)
      .def_static("maximum", &HKL::maximum)
      .def_static("minimum", &HKL::minimum)
      .def("vector", &HKL::vector)
      .def_rw("h", &HKL::h)
      .def_rw("k", &HKL::k)
      .def_rw("l", &HKL::l)
      .def("__repr__", [](const HKL &hkl) {
        return fmt::format("<HKL [{} {} {}]>", hkl.h, hkl.k, hkl.l);
      });

  nb::class_<SymmetryOperationFormat>(m, "SymmetryOperationFormat")
      .def(nb::init<>())
      .def_rw("fmt_string", &SymmetryOperationFormat::fmt_string)
      .def_rw("delimiter", &SymmetryOperationFormat::delimiter);

  nb::class_<SymmetryOperation>(m, "SymmetryOperation")
      // Constructors
      .def(nb::init<const occ::Mat4 &>(), "Construct from 4x4 Seitz matrix")
      .def(nb::init<const std::string &>(),
           "Construct from string representation (e.g. 'x,y,z')")
      .def(nb::init<int>(), "Construct from integer representation")

      // Methods
      .def("to_int", &SymmetryOperation::to_int,
           "Get integer representation of the symmetry operation")
      .def("to_string", &SymmetryOperation::to_string,
           "Get string representation of the symmetry operation",
           "format"_a = SymmetryOperationFormat{})
      .def("inverted", &SymmetryOperation::inverted,
           "Get inverted copy of the symmetry operation")
      .def("translated", &SymmetryOperation::translated,
           "Get translated copy of the symmetry operation")
      .def("is_identity", &SymmetryOperation::is_identity,
           "Check if this is the identity operation")
      .def("apply", &SymmetryOperation::apply,
           "Apply transformation to coordinates")
      .def("seitz", &SymmetryOperation::seitz, "Get the 4x4 Seitz matrix")
      .def("rotation", &SymmetryOperation::rotation,
           "Get the 3x3 rotation matrix")
      .def("cartesian_rotation", &SymmetryOperation::cartesian_rotation,
           "Get rotation matrix in Cartesian coordinates")
      .def("rotate_adp", &SymmetryOperation::rotate_adp,
           "Rotate anisotropic displacement parameters")
      .def("translation", &SymmetryOperation::translation,
           "Get translation vector")
      .def("has_translation", &SymmetryOperation::has_translation,
           "Check if operation includes translation")

      // Operators
      .def("__call__", &SymmetryOperation::operator())
      .def("__eq__", &SymmetryOperation::operator==)
      .def("__lt__", &SymmetryOperation::operator<)
      .def("__gt__", &SymmetryOperation::operator>)
      .def("__le__", &SymmetryOperation::operator<=)
      .def("__ge__", &SymmetryOperation::operator>=)
      .def("__mul__", &SymmetryOperation::operator*,
           "Compose two symmetry operations")
      .def("__repr__", [](const SymmetryOperation &op) {
        return fmt::format("<SymmetryOperation '{}'>", op.to_string());
      });

  nb::class_<SpaceGroup>(m, "SpaceGroup")
      .def(nb::init<>(),
           "Constructs a space group with only translational symmetry")
      .def(nb::init<int>(),
           "Constructs a space group with the given space group number")
      .def(nb::init<const std::string &>(),
           "Constructs a space group with the given space group symbol")
      .def(nb::init<const std::vector<std::string> &>(),
           "Constructs a space group with the list of symmetry operations in "
           "string form")
      .def(nb::init<const std::vector<SymmetryOperation> &>(),
           "Constructs a space group with the list of symmetry operations")
      .def("number", &SpaceGroup::number, "Returns the space group number")
      .def_prop_ro("symbol", &SpaceGroup::symbol,
                   "Returns the Hermann-Mauguin (international tables) symbol")
      .def_prop_ro("short_name", &SpaceGroup::short_name,
                   "Returns the shortened Hermann-Mauguin symbol")
      .def_prop_ro("symmetry_operations", &SpaceGroup::symmetry_operations,
                   "Returns the list of symmetry operations")
      .def("has_H_R_choice", &SpaceGroup::has_H_R_choice,
           "Determines whether this space group has hexagonal/rhombohedral "
           "choice")
      .def("apply_all_symmetry_operations",
           &SpaceGroup::apply_all_symmetry_operations,
           "Apply all symmetry operations to fractional coordinates")
      .def("apply_rotations", &SpaceGroup::apply_rotations,
           "Apply rotation parts of symmetry operations to fractional "
           "coordinates")
      .def("__repr__", [](const SpaceGroup &sg) {
        return fmt::format("<SpaceGroup {} ({}), {} operations>", sg.symbol(),
                           sg.number(), sg.symmetry_operations().size());
      });

  nb::class_<Crystal>(m, "Crystal")
      .def(nb::init<const AsymmetricUnit &, const SpaceGroup &,
                    const UnitCell &>())
      .def("symmetry_unique_molecules", &Crystal::symmetry_unique_molecules)
      .def("symmetry_unique_dimers", &Crystal::symmetry_unique_dimers)
      .def("unit_cell", &Crystal::unit_cell)
      .def("unit_cell_molecules", &Crystal::unit_cell_molecules)
      .def("unit_cell_atoms", &Crystal::unit_cell_atoms)
      .def("unit_cell_dimers", &Crystal::unit_cell_dimers)
      .def("atom_surroundings", &Crystal::atom_surroundings)
      .def("dimer_symmetry_string", &Crystal::dimer_symmetry_string)
      .def("asymmetric_unit_atom_surroundings",
           &Crystal::asymmetric_unit_atom_surroundings)
      .def("num_sites", &Crystal::num_sites)
      .def("labels", &Crystal::labels)
      .def("to_fractional", &Crystal::to_fractional)
      .def("to_cartesian", &Crystal::to_cartesian)
      .def("volume", &Crystal::volume)
      .def("slab", &Crystal::slab)
      .def("asymmetric_unit",
           nb::overload_cast<>(&Crystal::asymmetric_unit, nb::const_))
      .def_static("create_primitive_supercell",
                  &Crystal::create_primitive_supercell)
      .def_static("from_cif_file",
                  [](const std::string &filename) {
                    occ::io::CifParser parser;
                    return parser.parse_crystal_from_file(filename).value();
                  })
      .def_static("from_cif_string",
                  [](const std::string &contents) {
                    occ::io::CifParser parser;
                    return parser.parse_crystal_from_string(contents).value();
                  })
      .def("__repr__", [](const Crystal &c) {
        return fmt::format("<Crystal {} {}>",
                           c.asymmetric_unit().chemical_formula(),
                           c.space_group().symbol());
      });

  nb::class_<CrystalAtomRegion>(m, "CrystalAtomRegion")
      .def_ro("frac_pos", &CrystalAtomRegion::frac_pos)
      .def_ro("cart_pos", &CrystalAtomRegion::cart_pos)
      .def_ro("asym_idx", &CrystalAtomRegion::asym_idx)
      .def_ro("atomic_numbers", &CrystalAtomRegion::atomic_numbers)
      .def_ro("symop", &CrystalAtomRegion::symop)
      .def("size", &CrystalAtomRegion::size)
      .def("__repr__", [](const CrystalAtomRegion &region) {
        return fmt::format("<CrystalAtomRegion (n={})>", region.size());
      });

  nb::class_<SymmetryRelatedDimer>(m, "SymmetryRelatedDimer")
      .def_ro("unique_index", &SymmetryRelatedDimer::unique_index)
      .def_ro("dimer", &SymmetryRelatedDimer::dimer);

  nb::class_<CrystalDimers>(m, "CrystalDimers")
      .def_ro("radius", &CrystalDimers::radius)
      .def_ro("unique_dimers", &CrystalDimers::unique_dimers)
      .def_ro("molecule_neighbors", &CrystalDimers::molecule_neighbors);

  nb::class_<AsymmetricUnit>(m, "AsymmetricUnit")
      .def(nb::init<>())
      .def(nb::init<const Mat3N &, const IVec &>())
      .def(nb::init<const Mat3N &, const IVec &,
                    const std::vector<std::string> &>())
      .def_rw("positions", &AsymmetricUnit::positions)
      .def_rw("atomic_numbers", &AsymmetricUnit::atomic_numbers)
      .def_rw("occupations", &AsymmetricUnit::occupations)
      .def_rw("charges", &AsymmetricUnit::charges)
      .def_rw("labels", &AsymmetricUnit::labels)
      .def("__len__", &AsymmetricUnit::size)
      .def("__repr__", [](const AsymmetricUnit &asym) {
        return fmt::format("<AsymmetricUnit {}>", asym.chemical_formula());
      });

  nb::class_<UnitCell>(m, "UnitCell")
      .def(nb::init<>(), "Default constructor")
      .def(nb::init<double, double, double, double, double, double>(),
           "Construct with lengths (a,b,c) and angles (alpha,beta,gamma)")
      .def(nb::init<const occ::Vec3 &, const occ::Vec3 &>(),
           "Construct with lengths and angles vectors")
      .def(nb::init<const occ::Mat3 &>(),
           "Construct from lattice vectors matrix")
      .def_prop_rw("a", &UnitCell::a, &UnitCell::set_a,
                   "Length of a-axis in Angstroms")
      .def_prop_rw("b", &UnitCell::b, &UnitCell::set_b,
                   "Length of b-axis in Angstroms")
      .def_prop_rw("c", &UnitCell::c, &UnitCell::set_c,
                   "Length of c-axis in Angstroms")
      .def_prop_rw("alpha", &UnitCell::alpha, &UnitCell::set_alpha,
                   "Angle between b and c axes in radians")
      .def_prop_rw("beta", &UnitCell::beta, &UnitCell::set_beta,
                   "Angle between a and c axes in radians")
      .def_prop_rw("gamma", &UnitCell::gamma, &UnitCell::set_gamma,
                   "Angle between a and b axes in radians")
      .def_prop_ro("volume", &UnitCell::volume,
                   "Volume of the unit cell in cubic Angstroms")
      .def_prop_ro("direct", &UnitCell::direct,
                   "Direct matrix (columns are lattice vectors)")
      .def_prop_ro("reciprocal", &UnitCell::reciprocal,
                   "Reciprocal matrix (columns are reciprocal lattice vectors)")
      .def_prop_ro("inverse", &UnitCell::inverse,
                   "Inverse matrix (rows are reciprocal lattice vectors)")
      .def_prop_ro("lengths", &UnitCell::lengths, "Vector of lengths (a,b,c)")
      .def_prop_ro("angles", &UnitCell::angles,
                   "Vector of angles (alpha,beta,gamma)")
      .def("is_cubic", &UnitCell::is_cubic)
      .def("is_triclinic", &UnitCell::is_triclinic)
      .def("is_monoclinic", &UnitCell::is_monoclinic)
      .def("is_orthorhombic", &UnitCell::is_orthorhombic)
      .def("is_tetragonal", &UnitCell::is_tetragonal)
      .def("is_rhombohedral", &UnitCell::is_rhombohedral)
      .def("is_hexagonal", &UnitCell::is_hexagonal)
      .def("is_orthogonal", &UnitCell::is_orthogonal)
      .def("cell_type", &UnitCell::cell_type,
           "Get string representation of cell type")
      .def("to_cartesian", &UnitCell::to_cartesian,
           "Convert fractional to Cartesian coordinates")
      .def("to_fractional", &UnitCell::to_fractional,
           "Convert Cartesian to fractional coordinates")
      .def("to_reciprocal", &UnitCell::to_reciprocal,
           "Convert coordinates to reciprocal space")
      .def("to_cartesian_adp", &UnitCell::to_cartesian_adp,
           "Convert ADPs from fractional to Cartesian coordinates")
      .def("to_fractional_adp", &UnitCell::to_fractional_adp,
           "Convert ADPs from Cartesian to fractional coordinates")
      .def("hkl_limits", &UnitCell::hkl_limits,
           "Get HKL limits for given minimum d-spacing")
      .def("a_vector", &UnitCell::a_vector, "Get a lattice vector")
      .def("b_vector", &UnitCell::b_vector, "Get b lattice vector")
      .def("c_vector", &UnitCell::c_vector, "Get c lattice vector")
      .def("a_star_vector", &UnitCell::a_star_vector,
           "Get a* reciprocal vector")
      .def("b_star_vector", &UnitCell::b_star_vector,
           "Get b* reciprocal vector")
      .def("c_star_vector", &UnitCell::c_star_vector,
           "Get c* reciprocal vector")

      .def("__repr__", [](const UnitCell &cell) {
        return fmt::format("<UnitCell {} ({:.5f}, {:.5f}, {:.5f})>",
                           cell.cell_type(), cell.a(), cell.b(), cell.c());
      });

  m.def("cubic_cell", &cubic_cell, "length"_a, "Create cubic unit cell");
  m.def("rhombohedral_cell", &rhombohedral_cell, "length"_a, "angle"_a,
        "Create rhombohedral unit cell");
  m.def("tetragonal_cell", &tetragonal_cell, "a"_a, "c"_a,
        "Create tetragonal unit cell");
  m.def("hexagonal_cell", &hexagonal_cell, "a"_a, "c"_a,
        "Create hexagonal unit cell");
  m.def("orthorhombic_cell", &orthorhombic_cell, "a"_a, "b"_a, "c"_a,
        "Create orthorhombic unit cell");
  m.def("monoclinic_cell", &monoclinic_cell, "a"_a, "b"_a, "c"_a, "angle"_a,
        "Create monoclinic unit cell");
  m.def("triclinic_cell", &triclinic_cell, "a"_a, "b"_a, "c"_a, "alpha"_a,
        "beta"_a, "gamma"_a, "Create triclinic unit cell");

  nb::class_<SiteIndex>(m, "SiteIndex")
      .def_ro("offset", &SiteIndex::offset)
      .def_ro("hkl", &SiteIndex::offset)
      .def("__repr__", [](const SiteIndex &s) {
        return fmt::format("<SiteIndex {} [{} {} {}]>", s.offset, s.hkl.h,
                           s.hkl.k, s.hkl.l);
      });

  nb::class_<DimerIndex>(m, "DimerIndex")
      .def(nb::init<SiteIndex, SiteIndex>())
      .def_ro("a", &DimerIndex::a)
      .def_ro("b", &DimerIndex::b)
      .def("hkl_difference", &DimerIndex::hkl_difference)
      .def("__eq__", &DimerIndex::operator==)
      .def("__lt__", &DimerIndex::operator<)
      .def("__repr__", [](const DimerIndex &d) {
        return fmt::format("<DimerIndex {} [{} {} {}] - {} [{} {} {}]>",
                           d.a.offset, d.a.hkl.h, d.a.hkl.k, d.a.hkl.l,
                           d.b.offset, d.b.hkl.h, d.b.hkl.k, d.b.hkl.l);
      });

  nb::class_<DimerMappingTable>(m, "DimerMappingTable")
      .def_static("build_dimer_table", &DimerMappingTable::build_dimer_table,
                  "crystal"_a, "dimers"_a, "consider_inversion"_a)
      .def("symmetry_unique_dimer", &DimerMappingTable::symmetry_unique_dimer)
      .def("symmetry_related_dimers",
           &DimerMappingTable::symmetry_related_dimers)
      .def_prop_ro("unique_dimers", &DimerMappingTable::unique_dimers)
      .def_prop_ro("symmetry_unique_dimers",
                   &DimerMappingTable::symmetry_unique_dimers)
      .def("dimer_positions", &DimerMappingTable::dimer_positions)
      .def("dimer_index", nb::overload_cast<const occ::core::Dimer &>(
                              &DimerMappingTable::dimer_index, nb::const_))
      .def("dimer_index",
           nb::overload_cast<const occ::Vec3 &, const occ::Vec3 &>(
               &DimerMappingTable::dimer_index, nb::const_))
      .def_static("normalized_dimer_index",
                  &DimerMappingTable::normalized_dimer_index)
      .def("canonical_dimer_index", &DimerMappingTable::canonical_dimer_index)
      .def("__repr__", [](const DimerMappingTable &table) {
        return fmt::format("<DimerMappingTable n_unique={} n_sym_unique={}>",
                           table.unique_dimers().size(),
                           table.symmetry_unique_dimers().size());
      });

  nb::class_<SurfaceCutResult>(m, "SurfaceCutResult")
      .def(nb::init<const CrystalDimers &>())
      .def_rw("molecules", &SurfaceCutResult::molecules)
      .def_rw("exyz", &SurfaceCutResult::exyz)
      .def_rw("above", &SurfaceCutResult::above)
      .def_rw("below", &SurfaceCutResult::below)
      .def_rw("slab", &SurfaceCutResult::slab)
      .def_rw("bulk", &SurfaceCutResult::bulk)
      .def_rw("depth_scale", &SurfaceCutResult::depth_scale)
      .def_rw("basis", &SurfaceCutResult::basis)
      .def_rw("cut_offset", &SurfaceCutResult::cut_offset)
      .def("total_above", &SurfaceCutResult::total_above)
      .def("total_below", &SurfaceCutResult::total_below)
      .def("total_slab", &SurfaceCutResult::total_slab)
      .def("total_bulk", &SurfaceCutResult::total_bulk)
      .def("unique_counts_above", &SurfaceCutResult::unique_counts_above)
      .def("__repr__", [](const SurfaceCutResult &result) {
        return fmt::format(
            "<SurfaceCutResult n_molecules={} depth_scale={:.3f}>",
            result.molecules.size(), result.depth_scale);
      });

  nb::class_<CrystalSurfaceGenerationParameters>(
      m, "CrystalSurfaceGenerationParameters")
      .def(nb::init<>())
      .def_rw("d_min", &CrystalSurfaceGenerationParameters::d_min)
      .def_rw("d_max", &CrystalSurfaceGenerationParameters::d_max)
      .def_rw("unique", &CrystalSurfaceGenerationParameters::unique)
      .def_rw("reduced", &CrystalSurfaceGenerationParameters::reduced)
      .def_rw("systematic_absences_allowed",
              &CrystalSurfaceGenerationParameters::systematic_absences_allowed);

  nb::class_<Surface>(m, "Surface")
      .def(nb::init<const HKL &, const Crystal &>())
      .def("depth", &Surface::depth)
      .def("d", &Surface::d)
      .def("print", &Surface::print)
      .def("normal_vector", &Surface::normal_vector)
      .def_prop_ro("hkl", &Surface::hkl)
      .def_prop_ro("depth_vector", &Surface::depth_vector)
      .def_prop_ro("a_vector", &Surface::a_vector)
      .def_prop_ro("b_vector", &Surface::b_vector)
      .def("area", &Surface::area)
      .def("dipole", &Surface::dipole)
      .def("basis_matrix", &Surface::basis_matrix, "depth_scale"_a = 1.0)
      .def("find_molecule_cell_translations",
           &Surface::find_molecule_cell_translations, "unit_cell_mols"_a,
           "depth"_a, "cut_offset"_a = 0.0)
      .def("count_crystal_dimers_cut_by_surface",
           &Surface::count_crystal_dimers_cut_by_surface, "dimers"_a,
           "cut_offset"_a = 0.0)
      .def("possible_cuts", &Surface::possible_cuts, "unique_positions"_a,
           "epsilon"_a = 1e-6)
      .def_static("check_systematic_absence",
                  &Surface::check_systematic_absence)
      .def_static("faces_are_equivalent", &Surface::faces_are_equivalent)
      .def("__repr__", [](const Surface &surface) {
        return fmt::format("<Surface hkl=[{} {} {}] d={:.3f}>", surface.hkl().h,
                           surface.hkl().k, surface.hkl().l, surface.d());
      });

  // Add the free function for generating surfaces
  m.def("generate_surfaces", &generate_surfaces, "crystal"_a,
        "params"_a = CrystalSurfaceGenerationParameters{});

  return m;
}
