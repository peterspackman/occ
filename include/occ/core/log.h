#pragma once
#include <spdlog/spdlog.h>

namespace occ::log {
using spdlog::critical;
using spdlog::debug;
using spdlog::error;
using spdlog::info;
using spdlog::set_level;
using spdlog::warn;

namespace level {
using spdlog::level::critical;
using spdlog::level::debug;
using spdlog::level::err;
using spdlog::level::info;
using spdlog::level::trace;
using spdlog::level::warn;
} // namespace level
} // namespace occ::log
