#pragma once
#include <CLI/App.hpp>

namespace occ::main {

struct CubeConfig {
    std::string input_filename{""};
    std::string property{"density"};
    std::string points_filename{""};
    std::string output_filename{"out.cube"};

    std::string spin{"both"};
    int mo_number{-1};

    int divisions{10};
};

CLI::App *add_cube_subcommand(CLI::App &app);
void run_cube_subcommand(CubeConfig const &);

} // namespace occ::main
