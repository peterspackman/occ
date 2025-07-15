#pragma once
#include <spdlog/spdlog.h>
#include <string>
#include <functional>
#include <vector>

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

inline void flush() { spdlog::default_logger()->flush(); }

inline void flush_on(spdlog::level::level_enum level) {
  spdlog::flush_on(level);
}

inline void flush_every(std::chrono::seconds interval) {
  spdlog::flush_every(interval);
}

// Callback type for log messages
using LogCallback = std::function<void(spdlog::level::level_enum level, const std::string& message)>;

// Register a callback to receive log messages
void register_log_callback(const LogCallback& callback);

// Clear all registered callbacks
void clear_log_callbacks();

// Get all log messages since the last clear (useful for buffering)
std::vector<std::pair<spdlog::level::level_enum, std::string>> get_buffered_logs();

// Clear the log buffer
void clear_log_buffer();

// Enable/disable buffering of log messages
void set_log_buffering(bool enable);

} // namespace occ::log
