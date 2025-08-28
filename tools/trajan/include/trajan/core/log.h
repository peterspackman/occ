#pragma once
#include <spdlog/spdlog.h>
#include <string>

namespace trajan::log {
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

// void setup_logging(const std::string &verbosity);
// void setup_logging(int verbosity);

void set_log_level(const std::string &verbosity);
void set_log_level(spdlog::level::level_enum level);
void set_log_level(int verbosity);

void set_log_file(const std::string &filename);

inline void flush() { spdlog::default_logger()->flush(); }

inline void flush_on(spdlog::level::level_enum level) {
  spdlog::flush_on(level);
}

inline void flush_every(std::chrono::seconds interval) {
  spdlog::flush_every(interval);
}

} // namespace trajan::log
