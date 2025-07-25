#include "dma_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <occ/core/atom.h>
#include <occ/core/units.h>
#include <occ/dma/dma.h>
#include <occ/dma/linear_multipole_calculator.h>
#include <occ/dma/linear_multipole_shifter.h>
#include <occ/dma/mult.h>
#include <occ/dma/multipole_calculator.h>
#include <occ/driver/dma_driver.h>
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
      // New accessor methods for any multipole component
      .def("get_multipole", 
           nb::overload_cast<int, int>(&Mult::get_multipole, nb::const_),
           "l"_a, "m"_a,
           "Get multipole component by quantum numbers (l, m)")
      .def("get_component", 
           nb::overload_cast<const std::string&>(&Mult::get_component, nb::const_),
           "name"_a,
           "Get multipole component by string name (e.g., 'Q21c', 'Q30')")
      .def_static("component_name_to_lm", &Mult::component_name_to_lm,
                  "name"_a,
                  "Convert component name to (l, m) quantum numbers")
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

  // DMA Driver bindings
  nb::class_<occ::driver::DMAConfig>(m, "DMAConfig")
      .def(nb::init<>())
      .def_rw("wavefunction_filename", &occ::driver::DMAConfig::wavefunction_filename,
              "path to wavefunction file")
      .def_rw("punch_filename", &occ::driver::DMAConfig::punch_filename,
              "path to punch file output (default: dma.punch)")
      .def_rw("settings", &occ::driver::DMAConfig::settings,
              "DMA calculation settings")
      .def_rw("atom_radii", &occ::driver::DMAConfig::atom_radii,
              "atom-specific radii (element symbol -> radius in Angstrom)")
      .def_rw("atom_limits", &occ::driver::DMAConfig::atom_limits,
              "atom-specific max ranks (element symbol -> max rank)")
      .def_rw("write_punch", &occ::driver::DMAConfig::write_punch,
              "whether to write punch file (default: True)")
      .def("__repr__", [](const occ::driver::DMAConfig &config) {
        return fmt::format("<DMAConfig wavefunction='{}' punch='{}' write_punch={}>",
                           config.wavefunction_filename, config.punch_filename,
                           config.write_punch ? "True" : "False");
      });

  nb::class_<occ::driver::DMADriver::DMAOutput>(m, "DMAOutput")
      .def_rw("result", &occ::driver::DMADriver::DMAOutput::result,
              "DMA calculation result")
      .def_rw("sites", &occ::driver::DMADriver::DMAOutput::sites,
              "DMA sites information")
      .def("__repr__", [](const occ::driver::DMADriver::DMAOutput &output) {
        return fmt::format("<DMAOutput num_sites={} max_rank={}>",
                           output.sites.size(), output.result.max_rank);
      });

  nb::class_<occ::driver::DMADriver>(m, "DMADriver")
      .def(nb::init<>())
      .def(nb::init<const occ::driver::DMAConfig &>(), "config"_a)
      .def("set_config", &occ::driver::DMADriver::set_config, "config"_a,
           "set DMA configuration")
      .def("config", &occ::driver::DMADriver::config, "get current configuration")
      .def("run", nb::overload_cast<>(&occ::driver::DMADriver::run),
           "run DMA calculation loading wavefunction from file")
      .def("run", nb::overload_cast<const occ::qm::Wavefunction &>(&occ::driver::DMADriver::run),
           "wavefunction"_a, "run DMA calculation with provided wavefunction")
      .def_static("generate_punch_file", &occ::driver::DMADriver::generate_punch_file,
                  "result"_a, "sites"_a, "generate punch file content as string")
      .def_static("write_punch_file", &occ::driver::DMADriver::write_punch_file,
                  "filename"_a, "result"_a, "sites"_a, "write punch file to disk")
      .def("__repr__", [](const occ::driver::DMADriver &driver) {
        return fmt::format("<DMADriver>");
      });

  // Convenience function for punch file generation
  m.def("generate_punch_file", &occ::driver::DMADriver::generate_punch_file,
        "result"_a, "sites"_a,
        "Generate GDMA-compatible punch file content from DMA results");

  m.def("write_punch_file", &occ::driver::DMADriver::write_punch_file,
        "filename"_a, "result"_a, "sites"_a,
        "Write GDMA-compatible punch file from DMA results");

  return m;
}
