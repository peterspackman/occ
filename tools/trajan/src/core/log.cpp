#include <memory>
#include <spdlog/sinks/basic_file_sink.h>
#include <trajan/core/log.h>
#include <trajan/core/util.h>

namespace trajan::log {

// void setup_logging(const std::string &verbosity) {
//   auto level = trajan::log::level::info;
//   std::string level_lower = trajan::util::to_lower_copy(verbosity);
//   if (level_lower == "debug")
//     level = trajan::log::level::trace;
//   else if (level_lower == "normal")
//     level = trajan::log::level::info;
//   else if (level_lower == "verbose")
//     level = trajan::log::level::debug;
//   else if (level_lower == "minimal")
//     level = trajan::log::level::warn;
//   else if (level_lower == "silent")
//     level = trajan::log::level::critical;
//   trajan::log::set_level(level);
//   spdlog::set_level(level);
//   // store the last 32 debug messages in a buffer
//   spdlog::enable_backtrace(32);
//   spdlog::set_pattern("%v");
// }
//
// void setup_logging(int verbosity) {
//   auto level = trajan::log::level::info;
//   switch (verbosity) {
//   case 4:
//     level = trajan::log::level::trace;
//     break;
//   case 3:
//     level = trajan::log::level::debug;
//     break;
//   case 1:
//     level = trajan::log::level::warn;
//     break;
//   case 0:
//     level = trajan::log::level::critical;
//     break;
//   default:
//     level = trajan::log::level::info;
//     break;
//   }
//   trajan::log::set_level(level);
//   spdlog::set_level(level);
//   // store the last 32 debug messages in a buffer
//   spdlog::enable_backtrace(32);
//   spdlog::set_pattern("%v");
// }

namespace {
std::shared_ptr<spdlog::logger> current_logger = spdlog::default_logger();

spdlog::level::level_enum verbosity_to_level(const std::string &verbosity) {
  std::string level_lower = trajan::util::to_lower_copy(verbosity);
  if (level_lower == "debug")
    return spdlog::level::trace;
  if (level_lower == "verbose")
    return spdlog::level::debug;
  if (level_lower == "minimal")
    return spdlog::level::warn;
  if (level_lower == "silent")
    return spdlog::level::critical;
  return spdlog::level::info; // default for "normal" and unknown values
}

spdlog::level::level_enum verbosity_to_level(int verbosity) {
  switch (verbosity) {
  case 4:
    return spdlog::level::trace;
  case 3:
    return spdlog::level::debug;
  case 1:
    return spdlog::level::warn;
  case 0:
    return spdlog::level::critical;
  default:
    return spdlog::level::info;
  }
}
} // namespace

void set_log_level(spdlog::level::level_enum level) {
  current_logger->set_level(level);
  spdlog::set_pattern("%v");
  spdlog::enable_backtrace(32);
}

void set_log_level(const std::string &verbosity) {
  auto level = verbosity_to_level(verbosity);
  current_logger->set_level(level);
  spdlog::set_pattern("%v");
  spdlog::enable_backtrace(32);
}

void set_log_level(int verbosity) {
  auto level = verbosity_to_level(verbosity);
  current_logger->set_level(level);
  spdlog::set_pattern("%v");
  spdlog::enable_backtrace(32);
}

void set_log_file(const std::string &filename) {
  try {
    auto file_logger = spdlog::basic_logger_mt("trajan_logger", filename,
                                               true); // true = truncate
    current_logger = file_logger;
    file_logger->set_level(current_logger->level());
    spdlog::set_default_logger(current_logger);
  } catch (const spdlog::spdlog_ex &ex) {
    spdlog::warn(
        "Failed to create file logger: {}. Using existing logger instead.",
        ex.what());
  }
  spdlog::set_pattern("%v");
  spdlog::enable_backtrace(32);
}

} // namespace trajan::log
