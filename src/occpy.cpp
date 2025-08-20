#include "python/cg_bindings.h"
#include "python/core_bindings.h"
#include "python/crystal_bindings.h"
#include "python/dft_bindings.h"
#include "python/dma_bindings.h"
#include "python/interaction_bindings.h"
#include "python/isosurface_bindings.h"
#include "python/opt_bindings.h"
#include "python/qm_bindings.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/main/occ_cg.h>
#include <occ/qm/shell.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;

NB_MODULE(_occpy, m) {
  // Register core submodule
  auto core = register_core_bindings(m);
  auto cg = register_cg_bindings(m);
  auto crystal = register_crystal_bindings(m);
  auto qm = register_qm_bindings(m);
  auto dft = register_dft_bindings(m);
  auto dma = register_dma_bindings(m);
  auto iso = register_isosurface_bindings(m);
  auto intr = register_interaction_bindings(m);
  auto opt = register_opt_bindings(m);

  // LogLevel enum is already registered in core_bindings.cpp
  m.def("set_log_level", nb::overload_cast<int>(occ::log::set_log_level));
  m.def("set_log_level",
        nb::overload_cast<spdlog::level::level_enum>(occ::log::set_log_level));
  m.def("set_log_level",
        nb::overload_cast<const std::string &>(occ::log::set_log_level));

  m.def("set_log_file", occ::log::set_log_file);
  m.def("set_num_threads", [](int n) { occ::parallel::set_num_threads(n); });
  m.def("set_data_directory",
        [](const std::string &s) { occ::set_data_directory(s); });

  // Add the main calculation function
  m.def("calculate_crystal_growth_energies",
        [](const occ::main::CGConfig &config) {
          return occ::main::run_cg(config);
        });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "0.7.10";
#endif
}
