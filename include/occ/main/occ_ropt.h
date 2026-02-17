#pragma once
#include <CLI/App.hpp>

namespace occ::main {

struct RoptSettings {
    std::string crystal_filename;
    std::string output_filename;
    std::string model_name = "ce-b3lyp";
    std::string charge_string;
    std::string multiplicity_string;
    double neighbor_radius = 20.0;
    double gradient_tolerance = 1e-3;  // RMS gradient per DOF
    double energy_tolerance = 1e-5;
    int max_iterations = 200;
    bool use_cartesian_engine = true;
    bool fix_first_molecule = true;
    bool write_trajectory = false;
    bool normalize_hydrogens = false;
    bool spherical_basis = false;
    bool use_trust_region = true;  // Default to Trust Region Newton (second order)
    bool debug_pair_summary = false;
    bool debug_shell_histogram = false;
    bool debug_ewald = false;       // charge-only Ewald breakdown (energy-only)
    bool debug_charges = false;     // print per-molecule charge totals and site charges
    std::string multipole_json;     // load multipoles/potentials from DMACRYS JSON
    int max_interaction_order = 4;  // max multipole interaction order (lA+lB); -1 = no truncation
    bool use_ewald = true;          // enable Ewald-split electrostatics
    double ewald_accuracy = 1e-6;   // target accuracy for eta/cut selection
    double ewald_eta = 0.0;         // override Gaussian split (Angstrom^-1); 0 => auto
    int ewald_kmax = 0;             // override reciprocal cutoff (integer grid extent); 0 => auto
};

CLI::App *add_ropt_subcommand(CLI::App &app);
void run_ropt_subcommand(const RoptSettings &settings);

} // namespace occ::main
