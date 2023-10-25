#pragma once
#include <CLI/App.hpp>
#include <occ/io/occ_input.h>

namespace occ::main {

struct OccPairInput {
    std::string model_name{"ce-b3lyp"};
    std::string monomer_a;
    std::string monomer_b;
    std::vector<double> rotation_a{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> rotation_b{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<double> translation_a{0.0, 0.0, 0.0};
    std::vector<double> translation_b{0.0, 0.0, 0.0};
    std::string input_json_filename{""};
    std::string output_json_filename{""};
    std::string monomer_directory{"."};
};

CLI::App *add_pair_subcommand(CLI::App &app);
void run_pair_subcommand(OccPairInput const &);

} // namespace occ::main
