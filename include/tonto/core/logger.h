#pragma once
#include <spdlog/spdlog.h>

namespace tonto::log {
    using spdlog::info;
    using spdlog::error;
    using spdlog::warn;
    using spdlog::critical;
    using spdlog::debug;
    using spdlog::set_level;

    namespace level {
        using spdlog::level::debug;
        using spdlog::level::info;
        using spdlog::level::warn;
    }
}
