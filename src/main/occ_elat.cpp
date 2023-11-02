#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/os.h>
#include <occ/core/kabsch.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/polarization.h>
#include <occ/io/cifparser.h>
#include <occ/main/monomer_wavefunctions.h>
#include <occ/main/occ_elat.h>
#include <occ/main/pair_energy.h>
#include <occ/qm/wavefunction.h>

namespace fs = std::filesystem;
using occ::core::Dimer;
using occ::core::Element;
using occ::core::Molecule;
using occ::crystal::Crystal;
using occ::crystal::SymmetryOperation;
using occ::interaction::CEEnergyComponents;
using occ::interaction::CEModelInteraction;
using occ::main::LatticeConvergenceSettings;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::units::AU_TO_KCAL_PER_MOL;
using occ::units::AU_TO_KJ_PER_MOL;
using occ::units::BOHR_TO_ANGSTROM;
using occ::util::all_close;

Crystal read_crystal(const std::string &filename) {
    occ::io::CifParser parser;
    return parser.parse_crystal(filename).value();
}

void calculate_lattice_energy(const LatticeConvergenceSettings settings) {
    std::string filename = settings.crystal_filename;
    std::string basename = fs::path(filename).stem().string();
    Crystal c = read_crystal(filename);
    occ::log::info("Energy model: {}", settings.model_name);
    occ::log::info("Loaded crystal from {}", filename);
    auto molecules = c.symmetry_unique_molecules();
    occ::log::info("Symmetry unique molecules in {}: {}", filename,
                   molecules.size());

    std::vector<Wavefunction> wfns;
    occ::log::info("Calculating symmetry unique dimers");
    occ::crystal::CrystalDimers crystal_dimers;
    std::vector<CEEnergyComponents> energies;
    occ::main::LatticeEnergyResult lattice_energy_result;
    if (settings.model_name == "xtb") {
        lattice_energy_result =
            converged_xtb_lattice_energies(c, basename, settings);
    } else {
        wfns = occ::main::calculate_wavefunctions(basename, molecules,
                                                  settings.model_name);
        occ::main::compute_monomer_energies(basename, wfns,
                                            settings.model_name);
        lattice_energy_result = occ::main::converged_lattice_energies(
            c, wfns, wfns, basename, settings);
    }

    const auto &dimers = lattice_energy_result.dimers.unique_dimers;
    if (dimers.size() < 1) {
        occ::log::error("No dimers found using neighbour radius {:.3f}",
                        settings.max_radius);
        exit(0);
    }

    const std::string row_fmt_string = "{:>7.3f} {:>7.3f} {:>20s} {: 8.3f} {: "
                                       "8.3f} {: 8.3f} {: 8.3f} {: 8.3f} {: "
                                       "8.3f}";
    size_t mol_idx{0};
    double etot{0.0}, elat{0.0};
    for (const auto &n : lattice_energy_result.dimers.molecule_neighbors) {

        occ::log::info("Neighbors for molecule {}", mol_idx);

        occ::log::info(
            "{:>7s} {:>7s} {:>20s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} "
            "{:>8s}",
            "Rn", "Rc", "Symop", "E_coul", "E_ex", "E_rep", "E_pol", "E_disp",
            "E_tot");
        occ::log::info("==================================================="
                       "================================");
        CEEnergyComponents molecule_total;

        size_t j = 0;
        for (const auto &[dimer, idx] : n) {
            auto s_ab = c.dimer_symmetry_string(dimer);
            double rn = dimer.nearest_distance();
            double rc = dimer.center_of_mass_distance();
            const auto &e = lattice_energy_result.energy_components[idx];
            if (!e.is_computed) {
                j++;
                continue;
            }
            double ecoul = e.coulomb_kjmol(), e_ex = e.exchange_kjmol(),
                   e_rep = e.repulsion_kjmol(), epol = e.polarization_kjmol(),
                   edisp = e.dispersion_kjmol(), etot_mol = e.total_kjmol();
            molecule_total = molecule_total + e;
            occ::log::info(row_fmt_string, rn, rc, s_ab, ecoul, e_ex, e_rep,
                           epol, edisp, etot_mol);
            j++;
        }
        occ::log::info("Molecule {} total: {:.3f} kJ/mol ({} pairs)\n", mol_idx,
                       molecule_total.total_kjmol(), j);
        etot += molecule_total.total_kjmol();
        mol_idx++;
    }
    occ::log::info("Final energy: {:.3f} kJ/mol", etot * 0.5);
    occ::log::info("Lattice energy: {:.3f} kJ/mol",
                   lattice_energy_result.lattice_energy);
}

namespace occ::main {

CLI::App *add_elat_subcommand(CLI::App &app) {

    CLI::App *pair =
        app.add_subcommand("elat", "compute crystal lattice energy");
    auto config = std::make_shared<LatticeConvergenceSettings>();

    pair->add_option("crystal", config->crystal_filename,
                     "input crystal structure (CIF)")
        ->required();
    pair->add_option("-m,--model", config->model_name, "Energy model");
    pair->add_option("--json", config->output_json_filename,
                     "JSON filename for output");
    pair->add_option("-r,--radius", config->max_radius,
                     "maximum radius (Angstroms) for neighbours");
    pair->add_option("--radius-increment", config->radius_increment,
                     "step size (Angstroms) direct space summation");
    pair->add_flag("-w,--wolf", config->wolf_sum,
                   "accelerate convergence using Wolf sum");
    pair->add_flag(
        "--crystal-polarization,--crystal_polarization",
        config->crystal_field_polarization,
        "calculate polarization term using full crystal electric field");
    pair->fallthrough();
    pair->callback([config]() { run_elat_subcommand(*config); });
    return pair;
}

void run_elat_subcommand(const LatticeConvergenceSettings &settings) {
    calculate_lattice_energy(settings);
}

} // namespace occ::main
