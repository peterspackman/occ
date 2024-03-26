#pragma once
#include <CLI/App.hpp>

namespace occ::main {

struct IsosurfaceConfig {
    std::string geometry_filename{""};
    std::string environment_filename{""};
    size_t max_depth{4};
    double separation{0.2};
    double isovalue{0.02};
    double background_density{0.0};
    bool use_hashed_mc{false};
    bool binary_output{true};
    std::string kind{"promol"};
    std::string output_filename{"surface.ply"};
};

CLI::App *add_isosurface_subcommand(CLI::App &app);
void run_isosurface_subcommand(IsosurfaceConfig const &);
} // namespace occ::main
