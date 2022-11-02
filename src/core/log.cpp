#include <occ/core/log.h>
#include <occ/core/util.h>

namespace occ::log {

void setup_logging(const std::string &verbosity) {
    auto level = occ::log::level::info;
    std::string level_lower = occ::util::to_lower_copy(verbosity);
    if (level_lower == "debug")
        level = occ::log::level::trace;
    else if (level_lower == "normal")
        level = occ::log::level::info;
    else if (level_lower == "verbose")
        level = occ::log::level::debug;
    else if (level_lower == "minimal")
        level = occ::log::level::warn;
    else if (level_lower == "silent")
        level = occ::log::level::critical;
    occ::log::set_level(level);
    spdlog::set_level(level);
    // store the last 32 debug messages in a buffer
    spdlog::enable_backtrace(32);
    spdlog::set_pattern("%v");
}

} // namespace occ::log
