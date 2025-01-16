#include "python/cg_bindings.h"
#include "python/core_bindings.h"
#include "python/crystal_bindings.h"
#include "python/dft_bindings.h"
#include "python/qm_bindings.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
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
  auto crystal = register_crystal_bindings(m);
  auto qm = register_qm_bindings(m);
  auto dft = register_dft_bindings(m);

  m.def("setup_logging", [](int v) { occ::log::setup_logging(v); });
  m.def("set_num_threads", [](int n) { occ::parallel::set_num_threads(n); });
  m.def("set_data_directory",
        [](const std::string &s) { occ::qm::override_basis_set_directory(s); });

  // Add the main calculation function
  m.def("calculate_crystal_growth_energies",
        [](const occ::main::CGConfig &config) {
          return occ::main::run_cg(config);
        });

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "0.6.12";
#endif
}
