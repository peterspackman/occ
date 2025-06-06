#include "js/core_bindings.h"
#include "js/qm_bindings.h"
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>

using namespace emscripten;

EMSCRIPTEN_BINDINGS(occ) {
    // Register all binding modules
    register_core_bindings();
    register_qm_bindings();

    // Log level enum
    enum_<spdlog::level::level_enum>("LogLevel")
        .value("TRACE", spdlog::level::level_enum::trace)
        .value("DEBUG", spdlog::level::level_enum::debug)
        .value("INFO", spdlog::level::level_enum::info)
        .value("WARN", spdlog::level::level_enum::warn)
        .value("ERROR", spdlog::level::level_enum::err)
        .value("CRITICAL", spdlog::level::level_enum::critical)
        .value("OFF", spdlog::level::level_enum::off);

    // Global utility functions
    function("setLogLevel", select_overload<void(int)>(occ::log::set_log_level));
    function("setLogLevelEnum", select_overload<void(spdlog::level::level_enum)>(occ::log::set_log_level));
    function("setLogLevelString", select_overload<void(const std::string&)>(occ::log::set_log_level));
    function("setLogFile", &occ::log::set_log_file);
    function("setNumThreads", optional_override([](int n) { 
        occ::parallel::set_num_threads(n); 
    }));

    // Version information
    constant("version", std::string("0.7.6"));
}
