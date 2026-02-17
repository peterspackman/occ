#include <CLI/App.hpp>
#include <filesystem>
#include <memory>
#include <fmt/os.h>
#include <fmt/format.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/driver/dma_driver.h>
#include <occ/dma/mult.h>
#include <occ/io/cifparser.h>
#include <occ/io/cifwriter.h>
#include <occ/main/monomer_wavefunctions.h>
#include <occ/main/occ_ropt.h>
#include <occ/mults/crystal_optimizer.h>
#include <occ/mults/dmacrys_input.h>
#include <occ/mults/multipole_source.h>

namespace fs = std::filesystem;
using occ::crystal::Crystal;
using occ::mults::CrystalEnergy;
using occ::mults::CrystalOptimizer;
using occ::mults::CrystalOptimizerSettings;
using occ::mults::ForceFieldType;
using occ::mults::MultipoleSource;
using occ::mults::MoleculeState;
using occ::mults::OptimizationMethod;

namespace {

Crystal read_crystal(const std::string &filename) {
    occ::io::CifParser parser;
    return parser.parse_crystal_from_file(filename).value();
}

void set_charges_and_multiplicities(const std::string &charge_string,
                                    const std::string &multiplicity_string,
                                    std::vector<occ::core::Molecule> &molecules) {
    if (!charge_string.empty()) {
        std::vector<int> charges;
        auto tokens = occ::util::tokenize(charge_string, ",");
        for (const auto &token : tokens) {
            charges.push_back(std::stoi(token));
        }
        if (charges.size() != molecules.size()) {
            throw std::runtime_error(
                fmt::format("Require {} charges to be specified, found {}",
                            molecules.size(), charges.size()));
        }
        for (size_t i = 0; i < charges.size(); i++) {
            occ::log::info("Setting net charge for molecule {} = {}", i, charges[i]);
            molecules[i].set_charge(charges[i]);
        }
    }

    if (!multiplicity_string.empty()) {
        std::vector<int> multiplicities;
        auto tokens = occ::util::tokenize(multiplicity_string, ",");
        for (const auto &token : tokens) {
            multiplicities.push_back(std::stoi(token));
        }
        if (multiplicities.size() != molecules.size()) {
            throw std::runtime_error(
                fmt::format("Require {} multiplicities to be specified, found {}",
                            molecules.size(), multiplicities.size()));
        }
        for (size_t i = 0; i < multiplicities.size(); i++) {
            occ::log::info("Setting multiplicity for molecule {} = {}", i,
                           multiplicities[i]);
            molecules[i].set_multiplicity(multiplicities[i]);
        }
    }
}

std::vector<MultipoleSource> compute_multipoles_for_crystal(
    const std::string &basename,
    Crystal &crystal,
    const std::string &model_name,
    bool spherical_basis) {

    auto molecules = crystal.symmetry_unique_molecules();
    occ::log::info("Computing multipoles for {} unique molecules", molecules.size());

    // Compute wavefunctions
    auto wavefunctions = occ::main::calculate_wavefunctions(
        basename, molecules, model_name, spherical_basis);

    std::vector<MultipoleSource> sources;
    sources.reserve(molecules.size());

    for (size_t i = 0; i < molecules.size(); ++i) {
        occ::log::info("Computing DMA for molecule {}", i);

        // Setup DMA config
        occ::driver::DMAConfig dma_config;
        dma_config.settings.max_rank = 4;  // Up to hexadecapole
        dma_config.settings.big_exponent = 4.0;

        // Run DMA directly on wavefunction
        occ::driver::DMADriver driver(dma_config);
        auto output = driver.run(wavefunctions[i]);

        // Convert DMA result to MultipoleSource
        // Each DMA site becomes a body-frame site
        std::vector<MultipoleSource::BodySite> body_sites;

        const auto &mol = molecules[i];
        occ::Vec3 com = mol.center_of_mass();

        for (size_t site_idx = 0; site_idx < output.result.multipoles.size(); ++site_idx) {
            MultipoleSource::BodySite site;
            site.multipole = output.result.multipoles[site_idx];

            // Position relative to COM (convert from bohr to angstrom)
            occ::Vec3 pos = output.sites.positions.col(site_idx) * occ::units::BOHR_TO_ANGSTROM;
            site.offset = pos - com;

            body_sites.push_back(site);
        }

        sources.emplace_back(std::move(body_sites));

        // Set initial orientation from crystal
        sources.back().set_orientation(occ::Mat3::Identity(), com);
    }

    return sources;
}

} // anonymous namespace

namespace occ::main {

CLI::App *add_ropt_subcommand(CLI::App &app) {
    CLI::App *ropt = app.add_subcommand("ropt",
        "optimize rigid molecule crystal structure");

    auto config = std::make_shared<RoptSettings>();

    ropt->add_option("crystal", config->crystal_filename,
                     "input crystal structure (CIF)")
        ->required();

    ropt->add_option("-o,--output", config->output_filename,
                     "output CIF filename (default: <input>_opt.cif)");

    ropt->add_option("-m,--model", config->model_name,
                     "Energy model for DMA calculation (default: ce-b3lyp)");

    ropt->add_option("-r,--radius", config->neighbor_radius,
                     "neighbor radius in Angstroms (default: 20.0)");

    ropt->add_option("--gtol", config->gradient_tolerance,
                     "gradient convergence tolerance (default: 1e-4)");

    ropt->add_option("--etol", config->energy_tolerance,
                     "energy convergence tolerance (default: 1e-7)");

    ropt->add_option("--max-iter", config->max_iterations,
                     "maximum iterations (default: 200)");

    ropt->add_option("--charges", config->charge_string,
                     "molecular charges (comma-separated)");

    ropt->add_option("--multiplicities", config->multiplicity_string,
                     "spin multiplicities (comma-separated)");

    ropt->add_flag("--spherical", config->spherical_basis,
                   "use pure spherical basis sets");

    ropt->add_flag("--normalize-hbonds", config->normalize_hydrogens,
                   "normalize hydrogen bond lengths before optimization");

    ropt->add_flag("--s-functions", [config](int64_t) {
        config->use_cartesian_engine = false;
    }, "use S-function engine instead of Cartesian T-tensors");

    ropt->add_flag("--optimize-all", [config](int64_t) {
        config->fix_first_molecule = false;
    }, "optimize all molecules (don't fix first molecule)");

    ropt->add_flag("--trajectory", config->write_trajectory,
                   "write trajectory to XYZ file");

    ropt->add_flag("--debug-pairs", config->debug_pair_summary,
                   "print top attractive/repulsive pairs at start");
    ropt->add_flag("--debug-shells", config->debug_shell_histogram,
                   "print neighbor shell histogram at start");
    ropt->add_flag("--debug-ewald", config->debug_ewald,
                   "print charge-only Ewald energy breakdown at start");
    ropt->add_flag("--debug-charges", config->debug_charges,
                   "print per-molecule net charge and site charges after DMA");
    ropt->add_option("--multipole-json", config->multipole_json,
                     "load multipoles and potentials from DMACRYS JSON file");
    ropt->add_option("--max-order", config->max_interaction_order,
                     "max multipole interaction order lA+lB (default: 4, -1=no truncation)");

    ropt->add_flag("--no-ewald", [config](int64_t){ config->use_ewald = false; },
                   "disable Ewald electrostatics (use truncated real space)");
    ropt->add_option("--ewald-acc", config->ewald_accuracy,
                     "target accuracy for automatic Ewald eta/cutoffs (default 1e-6)");
    ropt->add_option("--ewald-eta", config->ewald_eta,
                     "override Ewald Gaussian split eta in Angstrom^-1 (0=auto)");
    ropt->add_option("--ewald-kmax", config->ewald_kmax,
                     "override reciprocal cutoff integer extent (0=auto)");

    ropt->add_flag("--lbfgs", [config](int64_t) {
        config->use_trust_region = false;
    }, "use L-BFGS optimizer instead of Trust Region Newton");

    ropt->fallthrough();
    ropt->callback([config]() { run_ropt_subcommand(*config); });

    return ropt;
}

/// Prepare inputs from DMACRYS JSON benchmark file.
struct PreparedInputs {
    Crystal crystal;
    std::vector<MultipoleSource> multipoles;
    std::map<std::pair<int,int>, mults::BuckinghamParams> custom_buck_params;
    bool use_custom_ff = false;
    std::unique_ptr<mults::DmacrysInput> dmacrys_input;  // for direct setup
};

PreparedInputs prepare_from_json(const std::string &json_path) {
    occ::log::info("Loading DMACRYS benchmark from {}", json_path);
    auto input = mults::read_dmacrys_json(json_path);

    occ::log::info("  Title: {} (source: {})", input.title, input.source);
    occ::log::info("  Space group: {}, Z = {}", input.crystal.space_group,
                   input.crystal.Z);
    occ::log::info("  {} multipole sites, rank up to {}",
                   input.molecule.sites.size(),
                   input.molecule.sites.empty()
                       ? 0
                       : input.molecule.sites[0].rank);
    occ::log::info("  {} Buckingham pairs", input.potentials.size());

    Crystal cryst = mults::build_crystal(input.crystal);
    auto multipoles = mults::build_multipole_sources(input, cryst);
    auto buck = mults::convert_buckingham_params(input.potentials);

    if (input.initial_ref.total_kJ_per_mol != 0.0) {
        occ::log::info("  DMACRYS reference (initial): {:.6f} kJ/mol",
                       input.initial_ref.total_kJ_per_mol);
        occ::log::info("    Rep-disp: {:.6f} eV/cell",
                       input.initial_ref.repulsion_dispersion_eV);
    }

    PreparedInputs result{std::move(cryst), multipoles,
                          std::move(buck), true};
    result.dmacrys_input = std::make_unique<mults::DmacrysInput>(std::move(input));
    return result;
}

PreparedInputs prepare_from_dma(const RoptSettings &settings,
                                const std::string &filename,
                                const std::string &basename) {
    occ::log::info("Loading crystal from {}", filename);
    Crystal cryst = read_crystal(filename);

    if (settings.normalize_hydrogens) {
        occ::log::info("Normalizing hydrogen bond lengths...");
        ankerl::unordered_dense::map<int, double> empty_map;
        int normalized = cryst.normalize_hydrogen_bondlengths(empty_map);
        occ::log::info("Normalized {} hydrogen bonds", normalized);
    }

    auto molecules = cryst.symmetry_unique_molecules();
    occ::log::info("Crystal has {} symmetry-unique molecules",
                   molecules.size());

    set_charges_and_multiplicities(settings.charge_string,
                                   settings.multiplicity_string, molecules);

    occ::log::info("Computing distributed multipoles using {} model",
                   settings.model_name);
    auto multipoles = compute_multipoles_for_crystal(
        basename, cryst, settings.model_name, settings.spherical_basis);

    return PreparedInputs{std::move(cryst), std::move(multipoles), {}, false};
}

void run_ropt_subcommand(const RoptSettings &settings) {
    std::string filename = settings.crystal_filename;
    std::string basename = fs::path(filename).stem().string();

    auto prepared = !settings.multipole_json.empty()
                        ? prepare_from_json(settings.multipole_json)
                        : prepare_from_dma(settings, filename, basename);

    if (settings.debug_charges) {
        for (size_t i = 0; i < prepared.multipoles.size(); ++i) {
            const auto cart = prepared.multipoles[i].cartesian();
            double total_q = 0.0;
            for (const auto& s : cart.sites) {
                double q = (s.rank >= 0) ? s.cart.data[0] : 0.0;
                total_q += q;
            }
            occ::log::info("Mol {} net charge (a.u.): {:.6f}", i, total_q);
            for (size_t si = 0; si < cart.sites.size(); ++si) {
                double q = (cart.sites[si].rank >= 0) ? cart.sites[si].cart.data[0] : 0.0;
                occ::log::info("  Site {:2d}: q={:+.6f}  rank={}", static_cast<int>(si), q, cart.sites[si].rank);
            }
        }
    }

    // Setup optimizer
    CrystalOptimizerSettings opt_settings;
    opt_settings.method = settings.use_trust_region ?
        OptimizationMethod::TrustRegion : OptimizationMethod::LBFGS;
    opt_settings.gradient_tolerance = settings.gradient_tolerance;
    opt_settings.energy_tolerance = settings.energy_tolerance;
    opt_settings.max_iterations = settings.max_iterations;
    opt_settings.neighbor_radius = settings.neighbor_radius;
    opt_settings.force_field = prepared.use_custom_ff
                                   ? ForceFieldType::Custom
                                   : ForceFieldType::BuckinghamDE;
    opt_settings.use_cartesian_engine = settings.use_cartesian_engine;
    opt_settings.max_interaction_order = settings.max_interaction_order;
    opt_settings.fix_first_translation = settings.fix_first_molecule;
    opt_settings.fix_first_rotation = false;  // Always allow rotation
    opt_settings.use_ewald = settings.use_ewald;
    opt_settings.ewald_accuracy = settings.ewald_accuracy;
    opt_settings.ewald_eta = settings.ewald_eta;
    opt_settings.ewald_kmax = settings.ewald_kmax;
    if (settings.write_trajectory) {
        opt_settings.trajectory_file = basename + "_traj.xyz";
    }

    occ::log::info("Setting up crystal optimizer...");
    occ::log::info("  Optimizer: {}",
                   opt_settings.method == OptimizationMethod::TrustRegion ?
                   "Trust Region Newton (2nd order)" : "L-BFGS (quasi-Newton)");
    occ::log::info("  Neighbor radius: {:.1f} Angstrom", opt_settings.neighbor_radius);
    occ::log::info("  Gradient tolerance: {:.1e}", opt_settings.gradient_tolerance);
    occ::log::info("  Energy tolerance: {:.1e}", opt_settings.energy_tolerance);
    occ::log::info("  Max iterations: {}", opt_settings.max_iterations);
    occ::log::info("  Fix first translation: {}", opt_settings.fix_first_translation);
    occ::log::info("  Fix first rotation: {}", opt_settings.fix_first_rotation);
    occ::log::info("  Engine: {}",
                   opt_settings.use_cartesian_engine ? "Cartesian T-tensor" : "S-functions");
    if (opt_settings.max_interaction_order >= 0) {
        occ::log::info("  Max interaction order: {} (lA+lB <= {})",
                       opt_settings.max_interaction_order,
                       opt_settings.max_interaction_order);
    } else {
        occ::log::info("  Max interaction order: unlimited");
    }

    // Keep a copy of multipoles for DMACRYS setup (before move)
    auto multipoles_copy = prepared.multipoles;

    CrystalOptimizer optimizer(prepared.crystal, std::move(prepared.multipoles),
                               opt_settings);

    // Apply custom Buckingham params if loading from JSON
    if (prepared.use_custom_ff) {
        for (const auto& [key, params] : prepared.custom_buck_params) {
            optimizer.energy_calculator().set_buckingham_params(
                key.first, key.second, params);
        }
    }

    // For DMACRYS JSON: bypass Crystal's molecule detection and set
    // geometry, states, and neighbor list directly.
    if (prepared.dmacrys_input) {
        mults::setup_crystal_energy_from_dmacrys(
            optimizer.energy_calculator(),
            *prepared.dmacrys_input,
            prepared.crystal,
            multipoles_copy);
    }

    if (settings.debug_shell_histogram) {
        auto bins = optimizer.energy_calculator().neighbor_shell_histogram();
        occ::log::info("Neighbor shell counts (<3,<6,<10,<15,>=15 Å): {} {} {} {} {}",
                       bins[0], bins[1], bins[2], bins[3], bins[4]);
    }

    if (settings.debug_pair_summary) {
        auto dbg = optimizer.energy_calculator().debug_pair_energies(optimizer.states());
        std::sort(dbg.begin(), dbg.end(), [](const auto& a, const auto& b){ return a.total < b.total;});
        int n = static_cast<int>(dbg.size());
        int report = std::min(5, n);
        occ::log::info("Most attractive pairs (energy kJ/mol):");
        for (int i = 0; i < report; ++i) {
            const auto& p = dbg[i];
            occ::log::info("  {:3d} {:3d} shift [{:2d} {:2d} {:2d}] w={:.2f} d={:.2f}  elec={:8.4f}  sr={:8.4f}  tot={:8.4f}",
                           p.mol_i, p.mol_j, p.cell_shift[0], p.cell_shift[1], p.cell_shift[2],
                           p.weight, p.com_distance, p.electrostatic, p.short_range, p.total);
        }
        occ::log::info("Most repulsive pairs (energy kJ/mol):");
        for (int i = 0; i < report; ++i) {
            const auto& p = dbg[n - 1 - i];
            occ::log::info("  {:3d} {:3d} shift [{:2d} {:2d} {:2d}] w={:.2f} d={:.2f}  elec={:8.4f}  sr={:8.4f}  tot={:8.4f}",
                           p.mol_i, p.mol_j, p.cell_shift[0], p.cell_shift[1], p.cell_shift[2],
                           p.weight, p.com_distance, p.electrostatic, p.short_range, p.total);
        }
    }

    if (settings.debug_ewald) {
        occ::log::info("Ewald diagnostics: use --log-level debug to see Ewald correction details");
    }

    // Run optimization
    occ::log::info("\nStarting optimization...\n");
    auto result = optimizer.optimize();

    // Report results
    occ::log::info("\n{:=<60s}", "");
    occ::log::info("Optimization completed");
    occ::log::info("{:=<60s}", "");
    occ::log::info("Converged: {}", result.converged ? "Yes" : "No");
    occ::log::info("Termination: {}", result.termination_reason);
    occ::log::info("Iterations: {}", result.iterations);
    occ::log::info("Function evaluations: {}", result.function_evaluations);
    occ::log::info("");
    occ::log::info("Initial energy:     {:12.4f} kJ/mol per molecule",
                   result.initial_energy);
    occ::log::info("Final energy:       {:12.4f} kJ/mol per molecule",
                   result.final_energy);
    occ::log::info("  Electrostatic:    {:12.4f} kJ/mol per molecule",
                   result.electrostatic_energy);
    occ::log::info("  Rep-dispersion:   {:12.4f} kJ/mol per molecule",
                   result.repulsion_dispersion_energy);
    occ::log::info("Energy change:      {:12.4f} kJ/mol per molecule",
                   result.final_energy - result.initial_energy);

    // Write output CIF
    std::string output_filename = settings.output_filename;
    if (output_filename.empty()) {
        output_filename = basename + "_opt.cif";
    }

    if (result.optimized_crystal.has_value()) {
        occ::log::info("\nWriting optimized structure to {}", output_filename);
        occ::io::CifWriter writer;
        writer.write(output_filename, result.optimized_crystal.value(),
                     basename + "_optimized");
    } else {
        occ::log::warn("Could not reconstruct optimized crystal structure");
    }

    occ::log::info("\nDone.");
}

} // namespace occ::main
