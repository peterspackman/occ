#include <memory>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/base_sink.h>
#include <mutex>

namespace occ::log {
namespace {
std::shared_ptr<spdlog::logger> current_logger = spdlog::default_logger();

// Custom sink that forwards messages to callbacks
template<typename Mutex>
class callback_sink : public spdlog::sinks::base_sink<Mutex>
{
public:
    void add_callback(const LogCallback& callback) {
        std::lock_guard<Mutex> lock(this->mutex_);
        callbacks_.push_back(callback);
    }

    void clear_callbacks() {
        std::lock_guard<Mutex> lock(this->mutex_);
        callbacks_.clear();
    }

    void set_buffering(bool enable) {
        std::lock_guard<Mutex> lock(this->mutex_);
        buffering_enabled_ = enable;
    }

    std::vector<std::pair<spdlog::level::level_enum, std::string>> get_buffer() {
        std::lock_guard<Mutex> lock(this->mutex_);
        return buffer_;
    }

    void clear_buffer() {
        std::lock_guard<Mutex> lock(this->mutex_);
        buffer_.clear();
    }

protected:
    void sink_it_(const spdlog::details::log_msg& msg) override {
        spdlog::memory_buf_t formatted;
        this->formatter_->format(msg, formatted);
        std::string formatted_msg = fmt::to_string(formatted);
        
        // Call all registered callbacks
        for (const auto& callback : callbacks_) {
            callback(msg.level, formatted_msg);
        }
        
        // Buffer the message if buffering is enabled
        if (buffering_enabled_) {
            buffer_.emplace_back(msg.level, formatted_msg);
        }
    }

    void flush_() override {}

private:
    std::vector<LogCallback> callbacks_;
    std::vector<std::pair<spdlog::level::level_enum, std::string>> buffer_;
    bool buffering_enabled_ = false;
};

using callback_sink_mt = callback_sink<std::mutex>;
using callback_sink_st = callback_sink<spdlog::details::null_mutex>;

// Global callback sink instance
std::shared_ptr<callback_sink_mt> callback_sink_instance;

void ensure_callback_sink() {
    if (!callback_sink_instance) {
        callback_sink_instance = std::make_shared<callback_sink_mt>();
        
        // Get the current logger's sinks
        auto sinks = current_logger->sinks();
        
        // Add our callback sink
        sinks.push_back(callback_sink_instance);
        
        // Create a new logger with the updated sinks
        auto new_logger = std::make_shared<spdlog::logger>("occ", sinks.begin(), sinks.end());
        new_logger->set_level(current_logger->level());
        
        // Replace the current logger
        current_logger = new_logger;
        spdlog::set_default_logger(current_logger);
    }
}

spdlog::level::level_enum verbosity_to_level(const std::string &verbosity) {
  std::string level_lower = occ::util::to_lower_copy(verbosity);
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
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true);
    
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(file_sink);
    
    // Preserve callback sink if it exists
    if (callback_sink_instance) {
      sinks.push_back(callback_sink_instance);
    }
    
    auto file_logger = std::make_shared<spdlog::logger>("occ_logger", sinks.begin(), sinks.end());
    file_logger->set_level(current_logger->level());
    
    current_logger = file_logger;
    spdlog::set_default_logger(current_logger);
  } catch (const spdlog::spdlog_ex &ex) {
    spdlog::warn(
        "Failed to create file logger: {}. Using existing logger instead.",
        ex.what());
  }
  spdlog::set_pattern("%v");
  spdlog::enable_backtrace(32);
}

void register_log_callback(const LogCallback& callback) {
  ensure_callback_sink();
  callback_sink_instance->add_callback(callback);
}

void clear_log_callbacks() {
  if (callback_sink_instance) {
    callback_sink_instance->clear_callbacks();
  }
}

std::vector<std::pair<spdlog::level::level_enum, std::string>> get_buffered_logs() {
  if (callback_sink_instance) {
    return callback_sink_instance->get_buffer();
  }
  return {};
}

void clear_log_buffer() {
  if (callback_sink_instance) {
    callback_sink_instance->clear_buffer();
  }
}

void set_log_buffering(bool enable) {
  ensure_callback_sink();
  callback_sink_instance->set_buffering(enable);
}

} // namespace occ::log
