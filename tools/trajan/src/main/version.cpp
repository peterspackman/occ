#include <Eigen/Core>
#include <fmt/core.h>
#include <trajan/core/log.h>
#include <trajan/main/version.h>

namespace trajan::main {

void print_header() {
  const int fmt_major = FMT_VERSION / 10000;
  const int fmt_minor = (FMT_VERSION % 10000) / 100;
  const int fmt_patch = (FMT_VERSION % 100);
  const auto eigen_version_string =
      fmt::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION,
                  EIGEN_MINOR_VERSION);
  const std::string fmt_version_string =
      fmt::format("{}.{}.{}", fmt_major, fmt_minor, fmt_patch);
  const std::string spdlog_version_string = fmt::format(
      "{}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);

  log::info(R"(
  
Trajectory Analysis (C++)

a program for molecular dynamics analysis.

copyright (C) 2024 -> Blake Armstrong

this version of trajan makes use of the following third party libraries:

CLI11                command line argument parser
eigen3               Linear Algebra (v {})
fmt                  String formatting (v {})
spdlog               Logging (v {})

)",
            eigen_version_string, fmt_version_string, spdlog_version_string);
}
} // namespace trajan::main
