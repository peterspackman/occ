#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/os.h>
#include <occ/core/kabsch.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_labeller.h>
#include <occ/crystal/dimer_mapping_table.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/external_energy_model.h>
#include <occ/interaction/lattice_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/polarization.h>
#include <occ/interaction/xtb_energy_model.h>
#include <occ/io/cifparser.h>
#include <occ/io/cifwriter.h>
#include <occ/io/core_json.h>
#include <occ/io/crystal_json.h>
#include <occ/io/eigen_json.h>
#include <occ/main/monomer_wavefunctions.h>
#include <occ/main/occ_elat.h>
#include <occ/qm/wavefunction.h>

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

inline void
write_elat_json(const std::string &basename, const std::string &model,
                const occ::crystal::Crystal &crystal,
                const occ::crystal::CrystalDimers &dimers,
                const std::optional<occ::crystal::CrystalDimers> &uc_dimers =
                    std::nullopt) {
  nlohmann::json j;
  j["result_type"] = "elat";
  j["title"] = basename;
  j["crystal"] = crystal;
  j["model"] = model;
  j["has_permutation_symmetry"] = true;

  const auto &uc_atoms = crystal.unit_cell_atoms();
  auto dimer_labeller = SymmetryDimerLabeller(crystal);
  dimer_labeller.connection = "-";
  dimer_labeller.format.fmt_string = "{}";

  j["pairs"] = {};
  for (const auto &mol_pairs : dimers.molecule_neighbors) {
    nlohmann::json m;
    for (const auto &[dimer, unique_idx] : mol_pairs) {
      const auto &unique_dimer = dimers.unique_dimers[unique_idx];
      if (unique_dimer.interaction_energy() == 0.0)
        continue;

      nlohmann::json d;
      nlohmann::json e;

      // Label generation
      auto label = dimer_labeller(dimer);
      d["Label"] = label;
      d["Unique Index"] = unique_idx;

      // Energy components
      const auto &energies = unique_dimer.interaction_energies();
      for (const auto &[k, v] : energies) {
        e[k] = v;
      }
      d["energies"] = e;

      // Nearest neighbor calculation based on distance threshold
      bool is_nearest = dimer.nearest_distance() <=
                        4.0; // You may want to make this threshold configurable
      d["Nearest Neighbor"] = is_nearest;

      // Unit cell atom offsets
      nlohmann::json offsets_a = {};
      {
        const auto &a = dimer.a();
        const auto &a_uc_idx = a.unit_cell_idx();
        const auto &a_uc_shift = a.unit_cell_shift();
        for (int i = 0; i < a_uc_idx.rows(); i++) {
          offsets_a.push_back(std::array<int, 4>{a_uc_idx(i), a_uc_shift(0, i),
                                                 a_uc_shift(1, i),
                                                 a_uc_shift(2, i)});
        }
      }

      nlohmann::json offsets_b = {};
      {
        const auto &b = dimer.b();
        const auto &b_uc_idx = b.unit_cell_idx();
        const auto &b_uc_shift = b.unit_cell_shift();
        for (int i = 0; i < b_uc_idx.rows(); i++) {
          offsets_b.push_back(std::array<int, 4>{b_uc_idx(i), b_uc_shift(0, i),
                                                 b_uc_shift(1, i),
                                                 b_uc_shift(2, i)});
        }
      }
      d["uc_atom_offsets"] = {offsets_a, offsets_b};

      m.push_back(d);
    }
    j["pairs"].push_back(m);
  }
  if (uc_dimers.has_value()) {
    j["all_pairs"] = {};
    const occ::crystal::CrystalDimers uc_dimers_value = uc_dimers.value();

    using Offset = std::tuple<int, int, int, int>;
    ankerl::unordered_dense::set<Offset> global_molecule_offsets;
    for (const auto &mol_pairs : uc_dimers_value.molecule_neighbors) {
      for (const auto &[dimer, unique_idx] : mol_pairs) {
        const auto &unique_dimer = dimers.unique_dimers[unique_idx];
        if (unique_dimer.interaction_energy() == 0.0)
          continue;

        const auto shift_a = dimer.a().cell_shift();
        Offset mol_a{dimer.a().unit_cell_molecule_idx(), shift_a[0], shift_a[1],
                     shift_a[2]};
        global_molecule_offsets.insert(mol_a);
        const auto shift_b = dimer.b().cell_shift();
        Offset mol_b{dimer.b().unit_cell_molecule_idx(), shift_b[0], shift_b[1],
                     shift_b[2]};
        global_molecule_offsets.insert(mol_b);
      }
    }
    ankerl::unordered_dense::map<Offset, int> map_molecules;
    ankerl::unordered_dense::map<int, int> map_mol_uc_idx;
    int counter = 0;
    for (const auto &offset : global_molecule_offsets) {
      map_molecules[offset] = counter;
      int mol_uc_idx = std::get<0>(offset);
      map_mol_uc_idx[counter] = mol_uc_idx;
      counter++;
    }

    for (const auto &mol_pairs : uc_dimers_value.molecule_neighbors) {
      nlohmann::json m;
      for (const auto &[dimer, unique_idx] : mol_pairs) {
        const auto &unique_dimer = dimers.unique_dimers[unique_idx];
        if (unique_dimer.interaction_energy() == 0.0)
          continue;

        nlohmann::json d;
        nlohmann::json e;

        // Label generation
        auto label = dimer_labeller(dimer);
        d["Label"] = label;
        d["Unique Index"] = unique_idx;

        // Energy components
        const auto &energies = unique_dimer.interaction_energies();
        for (const auto &[k, v] : energies) {
          e[k] = v;
        }
        d["energies"] = e;

        // Nearest neighbor calculation based on distance threshold
        bool is_nearest =
            dimer.nearest_distance() <=
            4.0; // You may want to make this threshold configurable
        d["Nearest Neighbor"] = is_nearest;

        double r = dimer.center_of_mass_distance();
        d["r"] = r;
        occ::Vec3 r_vec = dimer.v_ab_com();
        d["rvec"] = std::array<double, 3>({r_vec[0], r_vec[1], r_vec[2]});

        d["mass"] = std::tuple<double, double>{dimer.a().molar_mass(),
                                               dimer.b().molar_mass()};
        const auto shift_a = dimer.a().cell_shift();
        Offset mol_a{dimer.a().unit_cell_molecule_idx(), shift_a[0], shift_a[1],
                     shift_a[2]};
        const auto shift_b = dimer.b().cell_shift();
        Offset mol_b{dimer.b().unit_cell_molecule_idx(), shift_b[0], shift_b[1],
                     shift_b[2]};
        int idx_a = map_molecules[mol_a];
        int idx_b = map_molecules[mol_b];
        d["pair_indices"] = std::tuple<int, int>{idx_a, idx_b};
        d["pair_uc_indices"] =
            std::tuple<int, int>{map_mol_uc_idx[idx_a], map_mol_uc_idx[idx_b]};

        m.push_back(d);
      }
      j["all_pairs"].push_back(m);
    }
  }

  std::ofstream dest(fmt::format("{}_elat_results.json", basename));
  dest << j.dump(2);
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
  if (settings.write_all_pairs) {
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

  write_elat_json(basename, settings.model_name, c,
                  lattice_energy_result.dimers, uc_dimers);
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
