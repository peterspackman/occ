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
    bool use_trust_region = false; // Use Trust Region Newton instead of default MSTMIN
    bool use_lbfgs = false;        // Use L-BFGS instead of default MSTMIN
    double mst_max_displacement = 0.05; // MSTMIN max component displacement per cycle
    double mst_step_tolerance = 1e-6;   // MSTMIN convergence on max |step component|
    double mst_rotation_scale = 0.2;    // MSTMIN internal scaling for rotational DOF
    double mst_cell_scale = 0.1;        // MSTMIN internal scaling for cell strain DOF
    int mst_max_line_search = 40;       // MSTMIN max trial line-search steps per cycle
    int mst_max_line_search_restarts = 2; // MSTMIN auto-restarts after failed line search
    int mst_max_evaluations = 4000;     // MSTMIN global objective-evaluation cap
    int mst_line_search_report_interval = 100; // Report line-search stalls every N trial evals
    double spli_min = 0.0;              // SPLI taper width: cutoff -> cutoff + spli_min
    double spli_max = 0.0;              // SPLI neighbor shell: cutoff -> cutoff + spli_max
    int spli_order = 3;                 // SPLI polynomial order (3=cubic, 5=quintic)
    bool free_cell_strain = false;      // if true, optimize all 6 Voigt strains (ignore lattice constraints)
    bool allow_approx_hessian = false;
    bool optimize_cell = false;
    bool compute_elastic_tensor = true;
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
    bool has_ewald_accuracy = false; // true if --ewald-acc was provided
    bool has_ewald_eta = false;      // true if --ewald-eta was provided
    bool has_ewald_kmax = false;     // true if --ewald-kmax was provided
    bool has_external_pressure = false; // true if --pressure was provided
    double external_pressure_gpa = 0.0; // external pressure (GPa), overrides DMACRYS PRES when set
};

CLI::App *add_ropt_subcommand(CLI::App &app);
void run_ropt_subcommand(const RoptSettings &settings);

} // namespace occ::main
