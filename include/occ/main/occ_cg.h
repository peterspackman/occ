#pragma once
#include <CLI/App.hpp>
#include <occ/core/dimer.h>
#include <occ/main/pair_energy.h>
#include <vector>

namespace occ::main {

struct CGConfig {
    LatticeConvergenceSettings lattice_settings;
    std::string solvent{"water"};
    std::string charge_string{""};
    std::string wavefunction_choice{"gas"};
    double cg_radius{3.8};
    int max_facets{0};
    bool write_dump_files{false};
    bool spherical{false};
    bool write_kmcpp_file{false};
    bool use_xtb{false};
    std::string xtb_solvation_model{"cpcmx"};
    bool list_solvents{false};
    bool crystal_is_atomic{false};
};

struct DimerSolventTerm {
    double ab{0.0};
    double ba{0.0};
    double total{0.0};
};

struct CGDimer {
    core::Dimer dimer;
    int unique_dimer_index{-1};
    double interaction_energy{0.0};
    DimerSolventTerm solvent_term{};
    double crystal_contribution{0.0};
    bool nearest_neighbor{false};
};

struct EnergyTotal {
    double crystal_energy{0.0};
    double interaction_energy{0.0};
    double solution_term{0.0};
};

struct CGResult {
    std::vector<std::vector<CGDimer>> pair_energies;
    std::vector<EnergyTotal> total_energies;
};

CLI::App *add_cg_subcommand(CLI::App &app);

CGResult run_cg(CGConfig const &);
void run_cg_subcommand(CGConfig const &);

} // namespace occ::main
