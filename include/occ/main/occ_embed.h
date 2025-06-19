#pragma once
#include <CLI/App.hpp>
#include <string>

namespace occ::main {

struct EmbedConfig {
    std::string cif_filename;
    std::string method_name = "hf";
    std::string basis_name = "3-21g";
    bool basis_spherical = true;
    std::string charge_scheme = "mulliken";  // mulliken, hirshfeld, chelpg
    bool use_wolf_sum = true;  // if false, use regular point charges
    bool atomic_mode = false;  // if true, treat each atom as separate "molecule"
    std::vector<int> net_charges;  // net charge for each molecule/atom
    std::vector<int> multiplicities;  // multiplicity for each molecule/atom
    double wolf_alpha = 0.2;
    double wolf_cutoff = 16.0;
    int max_embed_cycles = 10;
    double charge_convergence = 1e-4;
    double energy_convergence = 1e-6;
    std::string output_prefix = "embed";
};

CLI::App *add_embed_subcommand(CLI::App &app);
void run_embed_subcommand(const EmbedConfig &config);

} // namespace occ::main