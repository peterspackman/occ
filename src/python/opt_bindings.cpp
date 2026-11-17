#include "opt_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <occ/core/molecule.h>
#include <occ/opt/geometry_optimizer.h>
#include <occ/opt/neb.h>

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

  nb::class_<OptimizerSettings>(opt, "OptimizerSettings")
      .def(nb::init<>())
      .def_static("saddle_point", &OptimizerSettings::saddle_point,
                  "mode"_a = 0,
                  "Settings for a first-order saddle point search, starting "
                  "along internal mode `mode` from the lowest")
      .def_prop_rw(
          "cartesian",
          [](const OptimizerSettings &s) {
            return s.coordinates == OptimizerSettings::Coordinates::Cartesian;
          },
          [](OptimizerSettings &s, bool c) {
            s.coordinates = c ? OptimizerSettings::Coordinates::Cartesian
                              : OptimizerSettings::Coordinates::TRIC;
          },
          "Cartesian coordinates rather than TRIC")
      .def_rw("trust", &OptimizerSettings::trust,
              "Initial trust radius (RMS atomic displacement, Bohr)")
      .def_rw("trust_min", &OptimizerSettings::trust_min)
      .def_rw("trust_max", &OptimizerSettings::trust_max)
      .def_rw("project_net_force_torque",
              &OptimizerSettings::project_net_force_torque)
      .def_rw("transition_state", &OptimizerSettings::transition_state,
              "Search for a first-order saddle point")
      .def_rw("ts_mode", &OptimizerSettings::ts_mode);

  nb::class_<GeometryOptimizer>(
      opt, "GeometryOptimizer",
      "Trust-region geometry optimizer in TRIC coordinates, for minima or "
      "(OptimizerSettings.saddle_point()) first-order saddle points. Loop: "
      "update(energy, gradient) at get_next_geometry(), then step() until it "
      "returns True.")
      .def(nb::init<const occ::core::Molecule &, const ConvergenceCriteria &,
                    const OptimizerSettings &>(),
           "molecule"_a, "criteria"_a = ConvergenceCriteria{},
           "settings"_a = OptimizerSettings{})
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
      .def("set_followed_mode", &GeometryOptimizer::set_followed_mode,
           "mode"_a,
           "Saddle point search: start along the Hessian mode overlapping "
           "most with this Cartesian displacement (3N)")
      .def("followed_mode", &GeometryOptimizer::followed_mode)
      .def("negative_modes", &GeometryOptimizer::negative_modes,
           "Internal modes with negative curvature at the last step")
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

  nb::class_<NebSettings>(opt, "NebSettings")
      .def(nb::init<>())
      .def_rw("images", &NebSettings::images,
              "Images between the endpoints")
      .def_rw("spring", &NebSettings::spring,
              "Spring constant at and below the higher endpoint "
              "(Hartree/Bohr^2)")
      .def_rw("spring_max", &NebSettings::spring_max,
              "Spring constant at the highest image (Hartree/Bohr^2)")
      .def_rw("idpp", &NebSettings::idpp,
              "Interpolate by the image-dependent pair potential")
      .def_rw("climbing_image", &NebSettings::climbing_image)
      .def_rw("climb_threshold", &NebSettings::climb_threshold)
      .def_rw("climbing_force_max", &NebSettings::climbing_force_max)
      .def_rw("force_max", &NebSettings::force_max)
      .def_rw("max_step", &NebSettings::max_step)
      .def_rw("memory", &NebSettings::memory)
      .def_rw("project_net_force_torque",
              &NebSettings::project_net_force_torque);

  nb::class_<NudgedElasticBand>(
      opt, "NudgedElasticBand",
      "Climbing-image nudged elastic band. set_endpoint_energies(), then "
      "update(energies, gradients) at images(), then step() until it returns "
      "True.")
      .def(nb::init<const occ::core::Molecule &, const occ::core::Molecule &,
                    const NebSettings &>(),
           "reactant"_a, "product"_a, "settings"_a = NebSettings{})
      .def("images", &NudgedElasticBand::images,
           "The images between the endpoints, to evaluate")
      .def("path", &NudgedElasticBand::path, "All images, endpoints included")
      .def("set_endpoint_energies", &NudgedElasticBand::set_endpoint_energies,
           "reactant"_a, "product"_a)
      .def("update", &NudgedElasticBand::update, "energies"_a, "gradients"_a,
           "Energies (Hartree) and gradients (Hartree/Bohr, 3 x N) at images()")
      .def("step", &NudgedElasticBand::step,
           "Move the band; returns True when converged")
      .def("is_converged", &NudgedElasticBand::is_converged)
      .def("climbing", &NudgedElasticBand::climbing)
      .def("energies", &NudgedElasticBand::energies,
           "Path energies, endpoints included")
      .def("highest_image", &NudgedElasticBand::highest_image)
      .def("tangent", &NudgedElasticBand::tangent, "image"_a,
           "Unit tangent at path image `image` (3N)")
      .def("max_force", &NudgedElasticBand::max_force)
      .def("current_step", &NudgedElasticBand::current_step);

  return opt;
}
