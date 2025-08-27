#include "opt_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/core/molecule.h>
#include <occ/opt/angle_coordinate.h>
#include <occ/opt/berny_optimizer.h>
#include <occ/opt/bond_coordinate.h>
#include <occ/opt/dihedral_coordinate.h>
#include <occ/opt/internal_coordinates.h>
#include <occ/opt/optimization_state.h>

using namespace occ::opt;
using namespace occ;

nb::module_ register_opt_bindings(nb::module_ &m) {
  using namespace nb::literals;

  auto opt = m.def_submodule("opt", "Optimization module");

  // Bond type enum
  nb::enum_<BondCoordinate::Type>(opt, "BondCoordinateType")
      .value("COVALENT", BondCoordinate::Type::COVALENT)
      .value("VDW", BondCoordinate::Type::VDW);

  // Bond coordinate class
  nb::class_<BondCoordinate>(opt, "BondCoordinate")
      .def(nb::init<int, int, BondCoordinate::Type>(), "i"_a, "j"_a,
           "bond_type"_a = BondCoordinate::Type::COVALENT)
      .def_ro("i", &BondCoordinate::i, "First atom index")
      .def_ro("j", &BondCoordinate::j, "Second atom index")
      .def_ro("bond_type", &BondCoordinate::bond_type,
              "Bond type (COVALENT or VDW)")
      .def("__call__", &BondCoordinate::operator(), "coords"_a,
           "Calculate bond distance")
      .def("gradient", &BondCoordinate::gradient, "coords"_a,
           "Calculate gradient of bond distance")
      .def("__repr__", [](const BondCoordinate &b) {
        std::string type = (b.bond_type == BondCoordinate::Type::COVALENT)
                               ? "COVALENT"
                               : "VDW";
        return fmt::format("<BondCoordinate({}, {}, {})>", b.i, b.j, type);
      });

  // Angle coordinate class
  nb::class_<AngleCoordinate>(opt, "AngleCoordinate")
      .def(nb::init<int, int, int>(), "i"_a, "j"_a, "k"_a)
      .def_rw("i", &AngleCoordinate::i, "First atom index")
      .def_rw("j", &AngleCoordinate::j, "Central atom index")
      .def_rw("k", &AngleCoordinate::k, "Third atom index")
      .def("__call__", &AngleCoordinate::operator(), "coords"_a,
           "Calculate angle in radians")
      .def("gradient", &AngleCoordinate::gradient, "coords"_a,
           "Calculate gradient of angle")
      .def("__repr__", [](const AngleCoordinate &a) {
        return fmt::format("<AngleCoordinate({}, {}, {})>", a.i, a.j, a.k);
      });

  // Dihedral coordinate class
  nb::class_<DihedralCoordinate>(opt, "DihedralCoordinate")
      .def(nb::init<int, int, int, int>(), "i"_a, "j"_a, "k"_a, "l"_a)
      .def_rw("i", &DihedralCoordinate::i, "First atom index")
      .def_rw("j", &DihedralCoordinate::j, "Second atom index")
      .def_rw("k", &DihedralCoordinate::k, "Third atom index")
      .def_rw("l", &DihedralCoordinate::l, "Fourth atom index")
      .def("__call__", &DihedralCoordinate::operator(), "coords"_a,
           "Calculate dihedral angle in radians")
      .def("gradient", &DihedralCoordinate::gradient, "coords"_a,
           "Calculate gradient of dihedral angle")
      .def("__repr__", [](const DihedralCoordinate &d) {
        return fmt::format("<DihedralCoordinate({}, {}, {}, {})>", d.i, d.j,
                           d.k, d.l);
      });

  nb::class_<InternalCoordinates::Options>(opt, "InternalCoordinatesOptions")
      .def(nb::init<>())
      .def_rw("include_dihedrals",
              &InternalCoordinates::Options::include_dihedrals)
      .def_rw("superweak_dihedrals",
              &InternalCoordinates::Options::superweak_dihedrals);

  // InternalCoordinates structure
  nb::class_<InternalCoordinates>(opt, "InternalCoordinates")
      .def(nb::init<const occ::core::Molecule &,
                    const InternalCoordinates::Options &>(),
           "molecule"_a,
           "use_dihedrals"_a = InternalCoordinates::Options{true, false},
           "Build internal coordinates from molecular connectivity")
      .def("bonds", &InternalCoordinates::bonds, "List of bond coordinates")
      .def("angles", &InternalCoordinates::angles, "List of angle coordinates")
      .def("dihedrals", &InternalCoordinates::dihedrals,
           "List of dihedral coordinates")
      .def("weights", &InternalCoordinates::weights,
           "Weights for each internal coordinate")
      .def("size", &InternalCoordinates::size,
           "Total number of internal coordinates")
      .def("__len__", &InternalCoordinates::size)
      .def("__repr__",
           [](const InternalCoordinates &ic) {
             return fmt::format(
                 "<InternalCoordinates: {} bonds, {} angles, {} dihedrals>",
                 ic.bonds().size(), ic.angles().size(), ic.dihedrals().size());
           })
      .def("to_vector", &InternalCoordinates::to_vector, "positions"_a,
           "Get current values as a vector")
      .def("to_vector_with_template",
           &InternalCoordinates::to_vector_with_template, "positions"_a,
           "template_q"_a,
           "Get current values with discontinuity handling based on template")
      .def("wilson_b_matrix", &InternalCoordinates::wilson_b_matrix,
           "positions"_a, "Calculate Wilson B matrix");

  // Standalone transformation functions
  opt.def("transform_step_to_cartesian", &occ::opt::transform_step_to_cartesian,
          "internal_step"_a, "coords"_a, "positions"_a, "B_inv"_a,
          "Transform step from internal to Cartesian coordinates");

  // OptPoint structure
  nb::class_<OptPoint>(opt, "OptPoint")
      .def(nb::init<>())
      .def(nb::init<const occ::Vec &, double, const occ::Vec &>(), "q"_a,
           "energy"_a, "gradient"_a)
      .def_rw("q", &OptPoint::q, "Internal coordinates")
      .def_rw("E", &OptPoint::E, "Energy")
      .def_rw("g", &OptPoint::g, "Gradient in internal coordinates")
      .def("__repr__", [](const OptPoint &opt) {
        return fmt::format("<OptPoint: E={:.8f}, |q|={}, |g|={:.6f}>", opt.E,
                           opt.q.size(), opt.g.size() > 0 ? opt.g.norm() : 0.0);
      });

  // ConvergenceCriteria structure
  nb::class_<ConvergenceCriteria>(opt, "ConvergenceCriteria")
      .def(nb::init<>())
      .def_rw("gradient_max", &ConvergenceCriteria::gradient_max)
      .def_rw("gradient_rms", &ConvergenceCriteria::gradient_rms)
      .def_rw("step_max", &ConvergenceCriteria::step_max)
      .def_rw("step_rms", &ConvergenceCriteria::step_rms);

  // OptimizationState structure
  nb::class_<OptimizationState>(opt, "OptimizationState")
      .def_rw("positions", &OptimizationState::positions)
      .def_rw("energy", &OptimizationState::energy)
      .def_rw("gradient_cartesian", &OptimizationState::gradient_cartesian)
      .def_rw("current", &OptimizationState::current)
      .def_rw("best", &OptimizationState::best)
      .def_rw("previous", &OptimizationState::previous)
      .def_rw("interpolated", &OptimizationState::interpolated)
      .def_rw("predicted", &OptimizationState::predicted)
      .def_rw("future", &OptimizationState::future)
      .def_rw("hessian", &OptimizationState::hessian)
      .def_rw("trust_radius", &OptimizationState::trust_radius)
      .def_rw("first_step", &OptimizationState::first_step)
      .def_rw("step_number", &OptimizationState::step_number)
      .def_rw("converged", &OptimizationState::converged);

  // BernyOptimizer class
  nb::class_<BernyOptimizer>(opt, "BernyOptimizer")
      .def(nb::init<const occ::core::Molecule &>(), "molecule"_a)
      .def(nb::init<const occ::core::Molecule &, const ConvergenceCriteria &>(),
           "molecule"_a, "criteria"_a)
      .def("step", &BernyOptimizer::step,
           "Perform one optimization step, returns True if converged")
      .def("update", &BernyOptimizer::update, "energy"_a, "gradient"_a,
           "Update optimizer with energy and gradient")
      .def("get_next_geometry", &BernyOptimizer::get_next_geometry,
           "Get the next geometry for evaluation")
      .def("is_converged", &BernyOptimizer::is_converged,
           "Check if optimization has converged")
      .def("current_step", &BernyOptimizer::current_step,
           "Get current step number")
      .def("current_energy", &BernyOptimizer::current_energy,
           "Get current energy")
      .def("current_trust_radius", &BernyOptimizer::current_trust_radius,
           "Get current trust radius")
      .def("__repr__", [](const BernyOptimizer &opt) {
        return fmt::format("<BernyOptimizer: step={}, converged={}>",
                           opt.current_step(), opt.is_converged());
      });

  return opt;
}
