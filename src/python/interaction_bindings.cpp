#include "interaction_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <occ/interaction/pair_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/wavefunction_transform.h>

using namespace nb::literals;
using namespace occ::interaction;
using transform::TransformResult;
using transform::WavefunctionTransformer;

nb::module_ register_interaction_bindings(nb::module_ &m) {

  nb::class_<CEParameterizedModel>(m, "CEParameterizedModel")
      .def("ce_model_from_string",
           [](const std::string &s) { return ce_model_from_string(s); });

  nb::class_<CEEnergyComponents>(m, "CEEnergyComponents")

      .def("coulomb_kjmol", &CEEnergyComponents::coulomb_kjmol)
      .def("exchange_repulsion_kjmol",
           &CEEnergyComponents::exchange_repulsion_kjmol)
      .def("polarization_kjmol", &CEEnergyComponents::polarization_kjmol)
      .def("dispersion_kjmol", &CEEnergyComponents::dispersion_kjmol)
      .def("repulsion_kjmol", &CEEnergyComponents::repulsion_kjmol)
      .def("exchange_kjmol", &CEEnergyComponents::exchange_kjmol)
      .def("total_kjmol", &CEEnergyComponents::total_kjmol)
      .def("__add__", &CEEnergyComponents::operator+)
      .def("__sub__", &CEEnergyComponents::operator-)
      .def("__iadd__", &CEEnergyComponents::operator+=)
      .def("__isub__", &CEEnergyComponents::operator-=);

  nb::class_<CEModelInteraction>(m, "CEModelInteraction")
      .def(nb::init<const CEParameterizedModel &>())
      .def("__call__", &CEModelInteraction::operator());

  nb::class_<TransformResult>(m, "TransformResult")
      .def_ro("rotation", &TransformResult::rotation)
      .def_ro("translation", &TransformResult::translation)
      .def_ro("wfn", &TransformResult::wfn)
      .def_ro("rmsd", &TransformResult::rmsd);

  nb::class_<WavefunctionTransformer>(m, "WavefunctionTransformer")
      .def("calculate_transform",
           &WavefunctionTransformer::calculate_transform);

  return m;
}
