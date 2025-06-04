#include "dma_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <occ/core/atom.h>
#include <occ/dma/dma.h>
#include <occ/dma/linear_multipole_calculator.h>
#include <occ/dma/linear_multipole_shifter.h>
#include <occ/dma/mult.h>
#include <occ/dma/multipole_calculator.h>
#include <occ/qm/wavefunction.h>

using namespace nb::literals;
using occ::Mat;
using occ::Mat3N;
using occ::Vec;
using occ::Vec3;
using namespace occ::dma;

nb::module_ register_dma_bindings(nb::module_ &m) {

  nb::class_<Mult>(m, "Mult")
      .def(nb::init<int>(), "max_rank"_a)
      .def(nb::init<>())
      .def_rw("max_rank", &Mult::max_rank, "maximum rank of multipole moments")
      .def_rw("q", &Mult::q, "multipole moment coefficients")
      .def("num_components", &Mult::num_components,
           "total number of multipole components")
      .def("to_string", &Mult::to_string, "lm"_a,
           "string representation of specific multipole component")
      // Level 0 (monopole)
      .def_prop_rw(
          "Q00", [](const Mult &m) -> double { return m.Q00(); },
          [](Mult &m, double val) { m.Q00() = val; }, "monopole moment Q00")
      .def_prop_rw(
          "charge", [](const Mult &m) -> double { return m.charge(); },
          [](Mult &m, double val) { m.charge() = val; },
          "charge (alias for Q00)")
      // Level 1 (dipole)
      .def_prop_rw(
          "Q10", [](const Mult &m) -> double { return m.Q10(); },
          [](Mult &m, double val) { m.Q10() = val; }, "dipole moment Q10")
      .def_prop_rw(
          "Q11c", [](const Mult &m) -> double { return m.Q11c(); },
          [](Mult &m, double val) { m.Q11c() = val; }, "dipole moment Q11c")
      .def_prop_rw(
          "Q11s", [](const Mult &m) -> double { return m.Q11s(); },
          [](Mult &m, double val) { m.Q11s() = val; }, "dipole moment Q11s")
      // Level 2 (quadrupole)
      .def_prop_rw(
          "Q20", [](const Mult &m) -> double { return m.Q20(); },
          [](Mult &m, double val) { m.Q20() = val; }, "quadrupole moment Q20")
      .def_prop_rw(
          "Q21c", [](const Mult &m) -> double { return m.Q21c(); },
          [](Mult &m, double val) { m.Q21c() = val; }, "quadrupole moment Q21c")
      .def_prop_rw(
          "Q21s", [](const Mult &m) -> double { return m.Q21s(); },
          [](Mult &m, double val) { m.Q21s() = val; }, "quadrupole moment Q21s")
      .def_prop_rw(
          "Q22c", [](const Mult &m) -> double { return m.Q22c(); },
          [](Mult &m, double val) { m.Q22c() = val; }, "quadrupole moment Q22c")
      .def_prop_rw(
          "Q22s", [](const Mult &m) -> double { return m.Q22s(); },
          [](Mult &m, double val) { m.Q22s() = val; }, "quadrupole moment Q22s")
      // Add commonly used higher multipoles (octupole - Q3X)
      .def_prop_rw(
          "Q30", [](const Mult &m) -> double { return m.Q30(); },
          [](Mult &m, double val) { m.Q30() = val; }, "octupole moment Q30")
      .def_prop_rw(
          "Q31c", [](const Mult &m) -> double { return m.Q31c(); },
          [](Mult &m, double val) { m.Q31c() = val; }, "octupole moment Q31c")
      .def_prop_rw(
          "Q31s", [](const Mult &m) -> double { return m.Q31s(); },
          [](Mult &m, double val) { m.Q31s() = val; }, "octupole moment Q31s")
      .def_prop_rw(
          "Q32c", [](const Mult &m) -> double { return m.Q32c(); },
          [](Mult &m, double val) { m.Q32c() = val; }, "octupole moment Q32c")
      .def_prop_rw(
          "Q32s", [](const Mult &m) -> double { return m.Q32s(); },
          [](Mult &m, double val) { m.Q32s() = val; }, "octupole moment Q32s")
      .def_prop_rw(
          "Q33c", [](const Mult &m) -> double { return m.Q33c(); },
          [](Mult &m, double val) { m.Q33c() = val; }, "octupole moment Q33c")
      .def_prop_rw(
          "Q33s", [](const Mult &m) -> double { return m.Q33s(); },
          [](Mult &m, double val) { m.Q33s() = val; }, "octupole moment Q33s")
      // Hexadecapole (Q4X)
      .def_prop_rw(
          "Q40", [](const Mult &m) -> double { return m.Q40(); },
          [](Mult &m, double val) { m.Q40() = val; }, "hexadecapole moment Q40")
      .def_prop_rw(
          "Q41c", [](const Mult &m) -> double { return m.Q41c(); },
          [](Mult &m, double val) { m.Q41c() = val; },
          "hexadecapole moment Q41c")
      .def_prop_rw(
          "Q41s", [](const Mult &m) -> double { return m.Q41s(); },
          [](Mult &m, double val) { m.Q41s() = val; },
          "hexadecapole moment Q41s")
      .def_prop_rw(
          "Q42c", [](const Mult &m) -> double { return m.Q42c(); },
          [](Mult &m, double val) { m.Q42c() = val; },
          "hexadecapole moment Q42c")
      .def_prop_rw(
          "Q42s", [](const Mult &m) -> double { return m.Q42s(); },
          [](Mult &m, double val) { m.Q42s() = val; },
          "hexadecapole moment Q42s")
      .def_prop_rw(
          "Q43c", [](const Mult &m) -> double { return m.Q43c(); },
          [](Mult &m, double val) { m.Q43c() = val; },
          "hexadecapole moment Q43c")
      .def_prop_rw(
          "Q43s", [](const Mult &m) -> double { return m.Q43s(); },
          [](Mult &m, double val) { m.Q43s() = val; },
          "hexadecapole moment Q43s")
      .def_prop_rw(
          "Q44c", [](const Mult &m) -> double { return m.Q44c(); },
          [](Mult &m, double val) { m.Q44c() = val; },
          "hexadecapole moment Q44c")
      .def_prop_rw(
          "Q44s", [](const Mult &m) -> double { return m.Q44s(); },
          [](Mult &m, double val) { m.Q44s() = val; },
          "hexadecapole moment Q44s")
      .def("__repr__", [](const Mult &mult) {
        return fmt::format("<Mult max_rank={} components={}>", mult.max_rank,
                           mult.num_components());
      });

  nb::class_<DMASettings>(m, "DMASettings")
      .def(nb::init<>())
      .def_rw("max_rank", &DMASettings::max_rank, "maximum multipole rank")
      .def_rw("big_exponent", &DMASettings::big_exponent,
              "large exponent threshold for analytical integration")
      .def_rw("include_nuclei", &DMASettings::include_nuclei,
              "include nuclear contributions to multipoles")
      .def("__repr__", [](const DMASettings &settings) {
        return fmt::format("<DMASettings max_rank={} big_exponent={:.2f} "
                           "include_nuclei={}>",
                           settings.max_rank, settings.big_exponent,
                           settings.include_nuclei ? "true" : "false");
      });

  nb::class_<DMAResult>(m, "DMAResult")
      .def(nb::init<>())
      .def_rw("max_rank", &DMAResult::max_rank, "maximum multipole rank")
      .def_rw("multipoles", &DMAResult::multipoles,
              "multipole moments for each site")
      .def("__repr__", [](const DMAResult &result) {
        return fmt::format("<DMAResult max_rank={} num_sites={}>",
                           result.max_rank, result.multipoles.size());
      });

  nb::class_<DMASites>(m, "DMASites")
      .def(nb::init<>())
      .def("size", &DMASites::size, "number of sites")
      .def("num_atoms", &DMASites::num_atoms, "number of atoms")
      .def_rw("atoms", &DMASites::atoms, "atom information")
      .def_rw("name", &DMASites::name, "site names")
      .def_rw("positions", &DMASites::positions, "site positions")
      .def_rw("atom_indices", &DMASites::atom_indices, "atom indices for sites")
      .def_rw("radii", &DMASites::radii, "site radii")
      .def_rw("limits", &DMASites::limits, "multipole rank limits per site")
      .def("__repr__", [](const DMASites &sites) {
        return fmt::format("<DMASites num_sites={} num_atoms={}>", sites.size(),
                           sites.num_atoms());
      });

  nb::class_<DMACalculator>(m, "DMACalculator")
      .def(nb::init<const occ::qm::Wavefunction &>(), "wavefunction"_a)
      .def("update_settings", &DMACalculator::update_settings, "settings"_a,
           "update DMA calculation settings")
      .def("settings", &DMACalculator::settings, "get current settings")
      .def("set_radius_for_element", &DMACalculator::set_radius_for_element,
           "atomic_number"_a, "radius_angs"_a,
           "set site radius for specific element")
      .def("set_limit_for_element", &DMACalculator::set_limit_for_element,
           "atomic_number"_a, "limit"_a,
           "set multipole rank limit for specific element")
      .def("sites", &DMACalculator::sites, "get DMA sites information")
      .def("compute_multipoles", &DMACalculator::compute_multipoles,
           "compute distributed multipole moments")
      .def("compute_total_multipoles", &DMACalculator::compute_total_multipoles,
           "result"_a, "compute total multipole moments from DMA result")
      .def("__repr__", [](const DMACalculator &calc) {
        const auto &sites = calc.sites();
        return fmt::format("<DMACalculator num_sites={} max_rank={}>",
                           sites.size(), calc.settings().max_rank);
      });

  nb::class_<LinearDMASettings>(m, "LinearDMASettings")
      .def(nb::init<>())
      .def_rw("max_rank", &LinearDMASettings::max_rank,
              "maximum multipole rank")
      .def_rw("include_nuclei", &LinearDMASettings::include_nuclei,
              "include nuclear contributions")
      .def_rw("use_slices", &LinearDMASettings::use_slices,
              "use slice-based integration")
      .def_rw("tolerance", &LinearDMASettings::tolerance,
              "numerical significance threshold")
      .def_rw("default_radius", &LinearDMASettings::default_radius,
              "default site radius in Angstrom")
      .def_rw("hydrogen_radius", &LinearDMASettings::hydrogen_radius,
              "hydrogen site radius in Angstrom")
      .def("__repr__", [](const LinearDMASettings &settings) {
        return fmt::format("<LinearDMASettings max_rank={} tolerance={:.2e} "
                           "use_slices={}>",
                           settings.max_rank, settings.tolerance,
                           settings.use_slices ? "true" : "false");
      });

  nb::class_<LinearMultipoleCalculator>(m, "LinearMultipoleCalculator")
      .def(nb::init<const occ::qm::Wavefunction &, const LinearDMASettings &>(),
           "wavefunction"_a, "settings"_a = LinearDMASettings{})
      .def("calculate", &LinearMultipoleCalculator::calculate,
           "calculate multipole moments for linear molecule")
      .def("__repr__", [](const LinearMultipoleCalculator &calc) {
        return fmt::format("<LinearMultipoleCalculator>");
      });

  return m;
}
