#pragma once
#include <spdlog/spdlog.h>
#include <string>

namespace occ::log {
using spdlog::critical;
using spdlog::debug;
using spdlog::error;
using spdlog::info;
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

void set_log_level(const std::string &verbosity);
void set_log_level(spdlog::level::level_enum level);
void set_log_level(int verbosity);

void set_log_file(const std::string &filename);

} // namespace occ::log
