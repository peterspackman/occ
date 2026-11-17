#include "opt_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <occ/core/molecule.h>
#include <occ/opt/geometry_optimizer.h>

using namespace occ::opt;

nb::module_ register_opt_bindings(nb::module_ &m) {
  using namespace nb::literals;

  auto opt = m.def_submodule("opt", "Geometry optimization");

  nb::class_<ConvergenceCriteria>(opt, "ConvergenceCriteria")
      .def(nb::init<>())
      .def_rw("gradient_max", &ConvergenceCriteria::gradient_max,
              "Largest atomic gradient (Hartree/Bohr)")
      .def_rw("gradient_rms", &ConvergenceCriteria::gradient_rms,
              "RMS atomic gradient (Hartree/Bohr)")
      .def_rw("step_max", &ConvergenceCriteria::step_max,
              "Largest atomic displacement (Bohr)")
      .def_rw("step_rms", &ConvergenceCriteria::step_rms,
              "RMS atomic displacement (Bohr)")
      .def_rw("energy_change", &ConvergenceCriteria::energy_change,
              "Energy change (Hartree), if use_energy_criterion")
      .def_rw("use_energy_criterion",
              &ConvergenceCriteria::use_energy_criterion)
      .def_rw("max_iterations", &ConvergenceCriteria::max_iterations);

  nb::class_<GeometryOptimizer>(
      opt, "GeometryOptimizer",
      "Trust-region geometry optimizer in TRIC coordinates. Loop: "
      "update(energy, gradient) at get_next_geometry(), then step() until it "
      "returns True.")
      .def(nb::init<const occ::core::Molecule &, const ConvergenceCriteria &>(),
           "molecule"_a, "criteria"_a = ConvergenceCriteria{})
      .def("update", &GeometryOptimizer::update, "energy"_a, "gradient"_a,
           "Energy (Hartree) and gradient (Hartree/Bohr, 3 x N) at the "
           "geometry from get_next_geometry()")
      .def("step", &GeometryOptimizer::step,
           "Take a step; returns True when converged")
      .def("get_next_geometry", &GeometryOptimizer::get_next_geometry,
           "The geometry to evaluate next (the converged one once converged)")
      .def("set_initial_hessian", &GeometryOptimizer::set_initial_hessian,
           "hessian"_a, "Cartesian Hessian (Hartree/Bohr^2, 3N x 3N) to start "
           "from in place of the model")
      .def("is_converged", &GeometryOptimizer::is_converged)
      .def("current_step", &GeometryOptimizer::current_step)
      .def("current_energy", &GeometryOptimizer::current_energy,
           "Energy at the last accepted point")
      .def("trust_radius", &GeometryOptimizer::trust_radius,
           "Trust radius (RMS atomic displacement, Bohr)")
      .def("__repr__", [](const GeometryOptimizer &o) {
        return fmt::format("<GeometryOptimizer: step={}, converged={}>",
                           o.current_step(), o.is_converged());
      });

  return opt;
}
