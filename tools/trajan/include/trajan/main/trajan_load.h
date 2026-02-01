#pragma once

#include <CLI/CLI.hpp>
#include <filesystem>
#include <trajan/core/trajectory.h>
#include <vector>

namespace trajan::main {

using trajan::core::Trajectory;

struct LoadOpts {
  std::vector<std::filesystem::path> infiles;
  bool into_mem;
};

void run_load_subcommand(const LoadOpts &opts, Trajectory &traj);
CLI::App *add_load_subcommand(CLI::App &app, Trajectory &traj);

} // namespace trajan::main
