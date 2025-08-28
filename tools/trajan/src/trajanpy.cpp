#include "python/core_bindings.h"
#include <nanobind/nanobind.h>
#include <trajan/core/log.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;

NB_MODULE(_trajanpy, m) {
  auto core = register_core_bindings(m);

  nb::enum_<spdlog::level::level_enum>(m, "LogLevel")
      .value("TRACE", spdlog::level::level_enum::trace)
      .value("DEBUG", spdlog::level::level_enum::debug)
      .value("INFO", spdlog::level::level_enum::info)
      .value("WARN", spdlog::level::level_enum::warn)
      .value("ERROR", spdlog::level::level_enum::err)
      .value("CRITICAL", spdlog::level::level_enum::critical)
      .value("OFF", spdlog::level::level_enum::off);

  m.def("set_log_level", nb::overload_cast<int>(trajan::log::set_log_level));
  m.def("set_log_level", nb::overload_cast<spdlog::level::level_enum>(
                             trajan::log::set_log_level));
  m.def("set_log_level",
        nb::overload_cast<const std::string &>(trajan::log::set_log_level));

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "0.7.6";
#endif
}
