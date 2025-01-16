#include "crystal_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <occ/io/cifparser.h>

using namespace nb::literals;
using occ::crystal::AsymmetricUnit;
using occ::crystal::Crystal;
using occ::crystal::CrystalAtomRegion;
using occ::crystal::CrystalDimers;
using occ::crystal::HKL;
using occ::crystal::UnitCell;
using SymmetryRelatedDimer = occ::crystal::CrystalDimers::SymmetryRelatedDimer;

nb::module_ register_crystal_bindings(nb::module_ &parent) {
  auto m = parent.def_submodule("crystal",
                                "Crystal structure functionality for OCC");

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

  nb::class_<Crystal>(m, "Crystal")
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
      .def_prop_rw("a", &UnitCell::a, &UnitCell::set_a)
      .def_prop_rw("b", &UnitCell::b, &UnitCell::set_b)
      .def_prop_rw("c", &UnitCell::c, &UnitCell::set_c)
      .def_prop_rw("alpha", &UnitCell::alpha, &UnitCell::set_alpha)
      .def_prop_rw("beta", &UnitCell::beta, &UnitCell::set_beta)
      .def_prop_rw("gamma", &UnitCell::gamma, &UnitCell::set_gamma)
      .def("lengths", &UnitCell::lengths)
      .def("to_fractional", &UnitCell::to_fractional)
      .def("to_cartesian", &UnitCell::to_cartesian)
      .def("cell_type", &UnitCell::cell_type)
      .def("__repr__", [](const UnitCell &uc) {
        return fmt::format("<UnitCell {} ({:.5f}, {:.5f}, {:.5f})>",
                           uc.cell_type(), uc.a(), uc.b(), uc.c());
      });

  return m;
}
