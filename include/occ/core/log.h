#pragma once
#include <spdlog/spdlog.h>

namespace occ::log {
using spdlog::critical;
using spdlog::debug;
using spdlog::error;
using spdlog::info;
using spdlog::set_level;
using spdlog::trace;
using spdlog::warn;

namespace level {
using spdlog::level::critical;
using spdlog::level::debug;
using spdlog::level::err;
using spdlog::level::info;
using spdlog::level::trace;
using spdlog::level::warn;
} // namespace level

void setup_logging(const std::string &verbosity);
void setup_logging(int verbosity);

} // namespace occ::log
