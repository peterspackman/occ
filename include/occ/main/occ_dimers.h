#pragma once
#include <CLI/App.hpp>

namespace occ::main {

struct DimerGenerationSettings {
    std::string crystal_filename;
    double max_radius{3.8};
    std::string output_json_filename{"dimers.json"};
    bool generate_xyz_files{false};
};

CLI::App *add_dimers_subcommand(CLI::App &app);
void run_dimers_subcommand(DimerGenerationSettings const &);

} // namespace occ::main
