#include <CLI/App.hpp>
#include <filesystem>
#include <fmt/os.h>
#include <fmt/format.h>
#include <occ/core/log.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_labeller.h>
#include <occ/crystal/dimer_mapping_table.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/external_energy_model.h>
#include <occ/interaction/lattice_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/interaction_json.h>
#include <occ/interaction/xtb_energy_model.h>
#include <occ/io/cifparser.h>
#include <occ/io/cifwriter.h>
#include <occ/io/core_json.h>
#include <occ/io/crystal_json.h>
#include <occ/io/eigen_json.h>
#include <occ/main/monomer_wavefunctions.h>
#include <occ/main/occ_elat.h>
#include <occ/qm/wavefunction.h>
#include <occ/elastic_fit/elastic_fitting.h>

namespace fs = std::filesystem;
using occ::crystal::Crystal;
using occ::crystal::SymmetryDimerLabeller;
using occ::interaction::CEEnergyComponents;
using occ::interaction::CEEnergyModel;
using occ::interaction::ExternalEnergyModel;
using occ::interaction::LatticeConvergenceSettings;
using occ::interaction::LatticeEnergyCalculator;
using occ::interaction::LatticeEnergyResult;
using occ::interaction::XTBEnergyModel;
using occ::qm::Wavefunction;

inline Crystal read_crystal(const std::string &filename) {
  occ::io::CifParser parser;
  return parser.parse_crystal_from_file(filename).value();
}

inline void map_interactions_to_uc(const Crystal &crystal,
                                   const occ::crystal::CrystalDimers &dimers,
                                   occ::crystal::CrystalDimers &uc_dimers) {

  auto &uc_neighbors = uc_dimers.molecule_neighbors;
  const auto &mol_neighbors = dimers.molecule_neighbors;

  auto mapping = occ::crystal::DimerMappingTable::build_dimer_table(
      crystal, uc_dimers, false);

  // map interactions surrounding UC molecules to symmetry unique
  // interactions
  for (size_t i = 0; i < uc_neighbors.size(); i++) {
    const auto &m = crystal.unit_cell_molecules()[i];
    size_t asym_idx = m.asymmetric_molecule_idx();

    auto &unit_cell_neighbors = uc_neighbors[i];

    const auto &asymmetric_neighbors = mol_neighbors[asym_idx];

    int j = 0;
    for (auto &[dimer, unique_idx] : unit_cell_neighbors) {
      auto shift_b = dimer.b().cell_shift();
      auto idx_b = dimer.b().unit_cell_molecule_idx();
      const auto dimer_index =
          mapping.canonical_dimer_index(mapping.dimer_index(dimer));

      const auto &related = mapping.symmetry_related_dimers(dimer_index);
      ankerl::unordered_dense::set<occ::crystal::DimerIndex,
                                   occ::crystal::DimerIndexHash>
          related_set(related.begin(), related.end());
      occ::log::trace("Related dimers: {}", related_set.size());
      for (const auto &d : related_set) {
        occ::log::trace(" {}", d);
      }

      size_t idx{0};
      for (idx = 0; idx < asymmetric_neighbors.size(); idx++) {
        const auto &d_a = asymmetric_neighbors[idx].dimer;
        occ::log::trace("Candidate dimer: {} {}", d_a.a().name(),
                        d_a.b().name());
        auto idx_asym = mapping.canonical_dimer_index(mapping.dimer_index(d_a));
        occ::log::trace("Candidate: {}", idx_asym);
        if (related_set.contains(idx_asym)) {
          occ::log::trace("Given dimer:          {} ({})",
                          mapping.dimer_index(dimer), dimer_index);
          occ::log::trace("Found matching dimer: {} ({})",
                          mapping.dimer_index(d_a), idx_asym);
          break;
        }
      }
      if (idx >= asymmetric_neighbors.size()) {
        auto idx = mapping.dimer_index(dimer);
        auto sidx = mapping.symmetry_unique_dimer(idx);
        throw std::runtime_error(
            fmt::format("No matching interaction found for uc_mol "
                        "= {}, dimer = {})",
                        i, dimer_index));
      }
      occ::log::trace("Found match for uc dimer");
      double rn = dimer.nearest_distance();
      double rc = dimer.centroid_distance();

      dimer.set_interaction_energies(
          dimers.unique_dimers[idx].interaction_energies());
      dimer.set_interaction_id(idx);
    }
  }
}

inline void set_charges_and_multiplicities(const std::string &charge_string,
                                           const std::string &multiplicity_string,
                                           std::vector<occ::core::Molecule> &molecules) {
  // Handle charges
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
    for (int i = 0; i < charges.size(); i++) {
      occ::log::info("Setting net charge for molecule {} = {}", i, charges[i]);
      molecules[i].set_charge(charges[i]);
    }
  } else {
    occ::log::info("No charges provided, assuming neutral molecules");
  }

  // Handle multiplicities
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
    for (int i = 0; i < multiplicities.size(); i++) {
      occ::log::info("Setting multiplicity for molecule {} = {}", i, multiplicities[i]);
      molecules[i].set_multiplicity(multiplicities[i]);
    }
  } else {
    occ::log::info("No multiplicities provided, assuming singlet molecules");
  }
}

void calculate_lattice_energy(const LatticeConvergenceSettings settings) {
  std::string filename = settings.crystal_filename;
  std::string basename = fs::path(filename).stem().string();
  Crystal c = read_crystal(filename);
  
  if (settings.normalize_hydrogens) {
    try {
      occ::log::info("Starting hydrogen bond normalization...");
      ankerl::unordered_dense::map<int, double> empty_map;
      int normalized_bonds = c.normalize_hydrogen_bondlengths(empty_map);
      occ::log::info("Normalized {} hydrogen bonds", normalized_bonds);
      
      // Write normalized crystal to CIF file
      std::string normalized_filename = basename + "_norm.cif";
      occ::io::CifWriter writer;
      writer.write(normalized_filename, c, basename + "_normalized");
      occ::log::info("Wrote normalized crystal structure to {}", normalized_filename);
    } catch (const std::exception& e) {
      occ::log::error("Error during normalization: {}", e.what());
      throw;
    }
  }
  
  occ::log::info("Energy model: {}", settings.model_name);
  occ::log::info("Loaded crystal from {}", filename);
  auto molecules = c.symmetry_unique_molecules();
  set_charges_and_multiplicities(settings.charge_string, settings.multiplicity_string, molecules);
  occ::log::info("Symmetry unique molecules in {}: {}", filename,
                 molecules.size());

  std::vector<Wavefunction> wfns;
  occ::log::info("Calculating symmetry unique dimers");
  occ::crystal::CrystalDimers crystal_dimers;
  std::vector<CEEnergyComponents> energies;

  std::unique_ptr<occ::interaction::EnergyModelBase> energy_model;

  if (settings.model_name == "xtb") {
    energy_model = std::make_unique<XTBEnergyModel>(c);
  } else if (settings.model_name == "external") {
    if (settings.external_command.empty()) {
      throw std::runtime_error(
          "External command must be specified when using 'external' model");
    }
    energy_model =
        std::make_unique<ExternalEnergyModel>(c, settings.external_command);
  } else {
    wfns = occ::main::calculate_wavefunctions(
        basename, molecules, settings.model_name, settings.spherical_basis);
    occ::main::compute_monomer_energies(basename, wfns, settings.model_name);

    auto ce_model = std::make_unique<CEEnergyModel>(c, wfns, wfns);
    ce_model->set_model_name(settings.model_name);
    energy_model = std::move(ce_model);
  }

  LatticeEnergyCalculator calculator(std::move(energy_model), c, basename,
                                     settings);

  LatticeEnergyResult lattice_energy_result = calculator.compute();

  const auto &dimers = lattice_energy_result.dimers.unique_dimers;
  if (dimers.size() < 1) {
    occ::log::error("No dimers found using neighbour radius {:.3f}",
                    settings.max_radius);
    exit(0);
  }
  std::optional<occ::crystal::CrystalDimers> uc_dimers;
  if (settings.write_all_pairs || settings.run_elastic_fitting) {
    uc_dimers = c.unit_cell_dimers(settings.max_radius);
  }

  const std::string row_fmt_string = "{:>7.3f} {:>7.3f} {:>20s} {: 8.3f} {: "
                                     "8.3f} {: 8.3f} {: 8.3f} {: 8.3f} {: "
                                     "8.3f}";
  size_t mol_idx{0};
  double etot{0.0};
  for (const auto &n : lattice_energy_result.dimers.molecule_neighbors) {

    occ::log::info("Neighbors for molecule {}", mol_idx);

    occ::log::info("{:>7s} {:>7s} {:>20s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} "
                   "{:>8s}",
                   "Rn", "Rc", "Symop", "E_coul", "E_ex", "E_rep", "E_pol",
                   "E_disp", "E_tot");
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
      occ::log::info(fmt::runtime(row_fmt_string), rn, rc, s_ab, ecoul, e_ex,
                     e_rep, epol, edisp, etot_mol);
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

  // Create ElatResults and write using unified function
  occ::interaction::ElatResults elat_results{
    c,
    lattice_energy_result,
    basename,
    settings.model_name
  };

  std::string json_filename = fmt::format("{}_elat_results.json", basename);
  occ::interaction::write_elat_json(json_filename, elat_results);

  // Run elastic fitting if requested
  if (settings.run_elastic_fitting) {
    occ::log::info("Running elastic tensor fitting...");

    // Create default fitting settings
    occ::elastic_fit::FittingSettings fitting_settings;
    fitting_settings.potential_type = occ::elastic_fit::PotentialType::LJ;
    fitting_settings.include_positive = false;
    fitting_settings.max_to_zero = false;
    fitting_settings.scale_factor = 2.0;
    fitting_settings.temperature = 0.0;
    fitting_settings.gulp_scale = 0.01;
    fitting_settings.solver_type = occ::elastic_fit::LinearSolverType::SVD;
    fitting_settings.svd_threshold = 1e-12;
    fitting_settings.animate_phonons = false;
    fitting_settings.shrinking_factors = occ::IVec3(1, 1, 1);
    fitting_settings.shift = occ::Vec3(0.0, 0.0, 0.0);

    occ::log::info("Using Lennard-Jones potential with default settings for elastic fitting");

    occ::elastic_fit::ElasticFitter fitter(fitting_settings);
    // Pass the original elat_results and let fit_elastic_tensor handle unit cell mapping
    // fit_elastic_tensor will generate unit cell dimers and map energies as needed
    auto results = fitter.fit_elastic_tensor(elat_results);

    // Output results
    occ::log::info("Lattice energy from elastic fitting: {:.3f} kJ/(mole unit cells)", results.lattice_energy);
    occ::elastic_fit::ElasticFitter::print_elastic_tensor(results.elastic_tensor, "Elastic constant matrix: (Units=GPa)");

    occ::log::info("Elastic tensor fitting completed successfully");
  }
}

namespace occ::main {

CLI::App *add_elat_subcommand(CLI::App &app) {

  CLI::App *elat = app.add_subcommand("elat", "compute crystal lattice energy");
  auto config = std::make_shared<LatticeConvergenceSettings>();
  auto use_xtb = std::make_shared<bool>(false);

  elat->add_option("crystal", config->crystal_filename,
                   "input crystal structure (CIF)")
      ->required();
  elat->add_option("-m,--model", config->model_name, "Energy model");
  elat->add_option("--json", config->output_json_filename,
                   "JSON filename for output");
  elat->add_option("-r,--radius", config->max_radius,
                   "maximum radius (Angstroms) for neighbours");
  elat->add_option("--charges", config->charge_string, "system net charge");
  elat->add_option("--multiplicities", config->multiplicity_string, 
                   "spin multiplicities (comma-separated for each unique molecule)");
  elat->add_option("--radius-increment", config->radius_increment,
                   "step size (Angstroms) direct space summation");
  elat->add_option(
      "--convergence-threshold,--convergence_threshold",
      config->energy_tolerance,
      "energy convergence threshold (kJ/mol) for direct space summation");
  elat->add_flag("-w,--wolf", config->wolf_sum,
                 "accelerate convergence using Wolf sum");
  elat->add_flag("--spherical", config->spherical_basis,
                 "use pure spherical basis sets");
  elat->add_flag(
      "--crystal-polarization,--crystal_polarization",
      config->crystal_field_polarization,
      "calculate polarization term using full crystal electric field");
  elat->add_flag("--all-pairs", config->write_all_pairs,
                 "write all pairs of interactions");
  elat->add_flag("--xtb", *use_xtb, "use xtb for interaction energies");
  elat->add_option(
      "--external-command", config->external_command,
      "external command for energy calculations (for model=external)");
  elat->add_flag("--normalize-hbonds", config->normalize_hydrogens,
                 "normalize hydrogen bond lengths");

  // Elastic fitting options
  elat->add_flag("--elastic-fit", config->run_elastic_fitting,
                 "run elastic tensor fitting after lattice energy calculation");

  elat->fallthrough();
  elat->callback([config, use_xtb]() {
    if (*use_xtb) {
      config->model_name = "xtb";
    }
    if (!config->external_command.empty()) {
      config->model_name = "external";
    }
    run_elat_subcommand(*config);
  });
  return elat;
}

void run_elat_subcommand(const LatticeConvergenceSettings &settings) {
  calculate_lattice_energy(settings);
}

} // namespace occ::main
