#pragma once
#include <CLI/App.hpp>
#include <string>

namespace occ::main {

struct RoptSettings {
    // Input/output
    std::string crystal_filename;
    std::string output_filename;
    std::string structure_json;
    std::string elastic_tensor_file;

    // DMA model (CIF path only)
    std::string model_name = "ce-b3lyp";
    std::string charge_string;
    std::string multiplicity_string;

    // Optimization
    std::string optimizer = "mstmin"; // "mstmin", "lbfgs", "trust-region"
    double neighbor_radius = 20.0;
    double gradient_tolerance = 1e-3;
    double energy_tolerance = 1e-5;
    int max_iterations = 200;
    int max_interaction_order = 4;

    // Cell optimization
    bool optimize_cell = false;
    bool compute_elastic_tensor = true;
    double external_pressure_gpa = 0.0;
    bool has_external_pressure = false;

    // Ewald
    bool use_ewald = true;

    // Trajectory
    bool write_trajectory = false;
};

CLI::App *add_ropt_subcommand(CLI::App &app);
void run_ropt_subcommand(const RoptSettings &settings);

} // namespace occ::main
