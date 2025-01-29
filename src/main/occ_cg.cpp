#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/os.h>
#include <fstream>
#include <occ/cg/distance_partition.h>
#include <occ/cg/interaction_mapper.h>
#include <occ/cg/smd_solvation.h>
#include <occ/core/kabsch.h>
#include <occ/core/log.h>
#include <occ/core/point_group.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_labeller.h>
#include <occ/crystal/dimer_mapping_table.h>
#include <occ/dft/dft.h>
#include <occ/driver/crystal_growth.h>
#include <occ/geometry/wulff.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pair_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/polarization.h>
#include <occ/io/core_json.h>
#include <occ/io/crystal_json.h>
#include <occ/io/crystalgrower.h>
#include <occ/io/eigen_json.h>
#include <occ/io/kmcpp.h>
#include <occ/io/load_geometry.h>
#include <occ/io/occ_input.h>
#include <occ/io/ply.h>
#include <occ/io/wavefunction_json.h>
#include <occ/io/xyz.h>
#include <occ/main/crystal_surface_energy.h>
#include <occ/main/occ_cg.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>

namespace fs = std::filesystem;
using occ::cg::CrystalGrowthResult;
using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;
using occ::driver::WavefunctionChoice;
using InteractionLabels = occ::io::crystalgrower::NetWriter::InteractionLabels;
using Options = occ::driver::CrystalGrowthCalculatorOptions;

inline void write_cg_structure_file(const std::string &filename,
                                    const Crystal &crystal,
                                    const CrystalDimers &uc_dimers) {
  occ::log::info("Writing crystalgrower structure file to '{}'", filename);
  occ::io::crystalgrower::StructureWriter cg_structure_writer(filename);
  cg_structure_writer.write(crystal, uc_dimers);
}

inline auto write_cg_net_file(const std::string &filename,
                              const Crystal &crystal,
                              const CrystalDimers &uc_dimers) {
  occ::log::info("Writing crystalgrower net file to '{}'", filename);
  occ::io::crystalgrower::NetWriter cg_net_writer(filename);
  cg_net_writer.write(crystal, uc_dimers);
  return cg_net_writer.interaction_labels();
}

inline void write_kmcpp_input_file(const std::string &filename,
                                   const Crystal &crystal,
                                   const CrystalDimers &uc_dimers,
                                   const std::vector<double> &solution_terms) {
  occ::log::info("Writing kmcpp structure file to '{}'", filename);
  occ::io::kmcpp::InputWriter kmcpp_structure_writer(filename);
  kmcpp_structure_writer.write(crystal, uc_dimers, solution_terms);
}

inline std::vector<double> map_unique_interactions_to_uc_molecules(
    const Crystal &crystal, const CrystalDimers &dimers,
    CrystalDimers &uc_dimers, const std::vector<double> &solution_terms,
    const std::vector<occ::cg::DimerResults> &interaction_energies_vec,
    bool inversion) {

  occ::cg::InteractionMapper mapper(crystal, dimers, uc_dimers, inversion);
  return mapper.map_interactions(solution_terms, interaction_energies_vec);
}

namespace occ::main {

CLI::App *add_cg_subcommand(CLI::App &app) {
  CLI::App *cg =
      app.add_subcommand("cg", "compute crystal growth free energies");
  auto config = std::make_shared<CGConfig>();
  cg->add_option("input", config->lattice_settings.crystal_filename,
                 "input CIF")
      ->required();
  cg->add_option("-r,--radius", config->lattice_settings.max_radius,
                 "maximum radius (Angstroms) for neighbours");
  cg->add_option("-m,--model", config->lattice_settings.model_name,
                 "energy model");
  cg->add_option(
      "--convergence-threshold,--convergence_threshold",
      config->lattice_settings.energy_tolerance,
      "energy convergence threshold (kJ/mol) for direct space summation");
  cg->add_option("-c,--cg-radius", config->cg_radius,
                 "maximum radius (Angstroms) for nearest neighbours in CG "
                 "file (must be <= radius)");
  cg->add_option("-s,--solvent", config->solvent, "solvent name");
  cg->add_option("--charges", config->charge_string, "system net charge");
  cg->add_option("-w,--wavefunction-choice", config->wavefunction_choice,
                 "Choice of wavefunctions");
  cg->add_flag("--write-kmcpp", config->write_kmcpp_file,
               "write out an input file for kmcpp program");
  cg->add_flag("--xtb", config->use_xtb, "use xtb for interaction energies");
  cg->add_option("--xtb-solvation-model,--xtb_solvation_model",
                 config->xtb_solvation_model,
                 "solvation model for use with xtb interaction energies");
  cg->add_flag("-d,--dump", config->write_dump_files, "Write dump files");
  cg->add_flag("--atomic", config->crystal_is_atomic,
               "Crystal is atomic (i.e. no bonds)");
  cg->add_flag(
      "--asymmetric-solvent-contribution,--asymmetric_solvent_contribution",
      config->asymmetric_solvent_contribution,
      "Crystal growth interactions will not have permutational symmetry (i.e. "
      "A->B != B->A) (default: false)");
  cg->add_flag(
      "--gamma-point-molecules,--gamma_point_molecules",
      config->gamma_point_molecules,
      "Enforce that the resulting unit cell molecules (e.g. in the net file) "
      "must have geometric centroids in the range [0,1) (default: true)");
  cg->add_option("--surface-energies", config->max_facets,
                 "Calculate surface energies and write .gmf morphology files");
  cg->add_flag("--list-available-solvents", config->list_solvents,
               "List available solvents and exit");
  cg->fallthrough();
  cg->callback([config]() { run_cg_subcommand(*config); });
  return cg;
}

inline Mat calculate_directional_correlation_matrix(
    const CrystalDimers::MoleculeNeighbors &neighbors) {
  Mat result(neighbors.size(), neighbors.size());
  for (size_t i = 0; i < neighbors.size(); i++) {
    const auto &[dimer_i, unique_i] = neighbors[i];
    occ::Vec3 vi = dimer_i.v_ab().normalized();
    for (size_t j = 0; j < neighbors.size(); j++) {
      const auto &[dimer_j, unique_j] = neighbors[j];
      occ::Vec3 vj = dimer_j.v_ab().normalized();
      result(i, j) = vi.dot(vj);
    }
  }
  return result;
}

inline void write_wulff(const std::string &filename,
                        const CrystalSurfaceEnergies &s) {
  const auto &sg = s.crystal.space_group();
  occ::Vec energies(s.facets.size());
  occ::Mat3N hkl(3, s.facets.size());
  for (size_t f = 0; f < s.facets.size(); f++) {
    const auto &facet = s.facets[f];
    hkl(0, f) = facet.hkl.h;
    hkl(1, f) = facet.hkl.k;
    hkl(2, f) = facet.hkl.l;
    energies(f) = facet.energy;
  }
  const auto &R = s.crystal.unit_cell().reciprocal();
  const auto &RI = s.crystal.unit_cell().direct().transpose();

  // convert to cartesian, then to fractional
  occ::Mat3N hkl_frac = s.crystal.to_fractional(RI * hkl);

  auto [symop_id, expanded_hkl_frac] = sg.apply_rotations(hkl_frac);
  occ::Vec expanded_energies =
      energies.replicate(sg.symmetry_operations().size(), 1);
  occ::Mat3N directions = s.crystal.unit_cell().to_cartesian(expanded_hkl_frac);
  directions.colwise().normalize();
  auto wulff = occ::geometry::WulffConstruction(directions, expanded_energies);
  occ::isosurface::Isosurface mesh;
  mesh.vertices = wulff.vertices().cast<float>();
  mesh.faces = wulff.triangles();
  occ::io::write_ply_mesh(filename, mesh, false);
}

inline void serialize_cg_dimers(nlohmann::json &j, const Crystal &crystal,
                                const CrystalGrowthResult &result,
                                const InteractionLabels &cg_labels) {
  j["totals_per_molecule"] = {};
  for (const auto &mol_result : result.molecule_results) {
    const auto &mol_total = mol_result.total;
    nlohmann::json e;
    e["crystal_energy"] = mol_total.crystal_energy;
    e["interaction_energy"] = mol_total.interaction_energy;
    e["solution_term"] = mol_total.solution_term;
    j["totals_per_molecule"].push_back(e);
  }

  const auto &uc_atoms = crystal.unit_cell_atoms();

  auto dimer_labeller = occ::crystal::SymmetryDimerLabeller(crystal);
  dimer_labeller.connection = "-";
  dimer_labeller.format.fmt_string = "{}";

  j["pairs"] = {};
  for (const auto &mol_result : result.molecule_results) {
    nlohmann::json m;
    for (const auto &dimer_result : mol_result.dimer_results) {
      const auto &dimer = dimer_result.dimer;
      nlohmann::json d;
      nlohmann::json e;
      auto label = dimer_labeller(dimer);
      std::string cg_id = fmt::format("U-{}", dimer_result.unique_idx);
      const auto kv = cg_labels.find(label);
      if (kv != cg_labels.end()) {
        cg_id = kv->second;
      }
      for (const auto &[k, v] : dimer_result.energy_components) {
        e[k] = v;
      }
      d["Nearest Neighbor"] = dimer_result.is_nearest_neighbor;
      d["Unique Index"] = dimer_result.unique_idx;
      d["Crystalgrower Identifier"] = cg_id;
      d["energies"] = e;

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
}

inline void
serialize_cg_results(nlohmann::json &j, const Options &opts,
                     const occ::crystal::Crystal &crystal,
                     const nlohmann::json &surface_energies_json = {}) {

  j["result_type"] = "cg";
  j["title"] = opts.basename;
  j["solvent"] = opts.solvent;
  j["model"] = fmt::format("crystalclear, solvent='{}'", opts.solvent);
  j["has_permutation_symmetry"] = !opts.use_asymmetric_partition;

  j["crystal"] = crystal;
}

template <typename Calculator>
inline void compute_and_serialize_surface_cuts(
    Calculator &calc, nlohmann::json &j, const Options &opts,
    const CrystalDimers &uc_dimers, const CrystalDimers &uc_dimers_vacuum,
    int max_facets) {

  occ::log::info("Crystal surface energies (solvated)");
  auto surface_energies = occ::main::calculate_crystal_surface_energies(
      fmt::format("{}_{}", opts.basename, opts.solvent), calc.crystal(),
      uc_dimers, max_facets, 1);

  occ::log::info("Crystal surface energies (vacuum)");
  auto vacuum_surface_energies = occ::main::calculate_crystal_surface_energies(
      fmt::format("{}_vacuum", opts.basename), calc.crystal(), uc_dimers_vacuum,
      max_facets, -1);

  j["surface_energies"] = surface_energies;
  write_wulff(fmt::format("{}_{}.ply", opts.basename, opts.solvent),
              surface_energies);
  write_wulff(fmt::format("{}_vacuum.ply", opts.basename),
              vacuum_surface_energies);

  // TODO refactor this
  nlohmann::json vacuum_energies;
  for (const auto &mol : calc.crystal_interaction_energies()) {
    nlohmann::json tmp;
    for (const auto &v : mol) {
      tmp.push_back(v.energy_components);
    }
    vacuum_energies.push_back(tmp);
  }
  nlohmann::json solvated_energies;
  for (const auto &mol : calc.interaction_energies()) {
    nlohmann::json tmp;
    for (const auto &v : mol) {
      tmp.push_back(v.energy_components);
    }
    solvated_energies.push_back(tmp);
  }
  j["vacuum"] = vacuum_energies;
  j["solvated"] = solvated_energies;

  occ::log::info("Appending surface energies to json output");
}

template <class Calculator>
CrystalGrowthResult run_cg_impl(CGConfig const &config) {
  std::string basename =
      fs::path(config.lattice_settings.crystal_filename).stem().string();
  Crystal c_symm =
      occ::io::load_crystal(config.lattice_settings.crystal_filename);

  c_symm.set_gamma_point_unit_cell_molecules(config.gamma_point_molecules);

  if (config.crystal_is_atomic) {
    c_symm.set_connectivity_criteria(false);
  }

  Options opts;
  opts.solvent = config.solvent;
  opts.basename = basename;
  opts.write_debug_output_files = config.write_dump_files;
  opts.energy_model = config.lattice_settings.model_name;
  opts.xtb_solvation_model = config.xtb_solvation_model;
  opts.use_asymmetric_partition = config.asymmetric_solvent_contribution;

  // just ensure this is true for further outputs as the xtb calculation is
  // always symmetric
  if (config.use_xtb)
    opts.use_asymmetric_partition = false;

  occ::log::info("Enforcing asymmetry via partitioning:   {}",
                 opts.use_asymmetric_partition);
  occ::log::info("Enforcing unit cell molecules in gamma: {}",
                 config.gamma_point_molecules);

  opts.wavefunction_choice =
      (config.wavefunction_choice == "gas" ? WavefunctionChoice::GasPhase
                                           : WavefunctionChoice::Solvated);
  opts.inner_radius = config.cg_radius;
  opts.outer_radius = config.lattice_settings.max_radius;

  // Setup calculator parameters
  std::vector<int> charges;
  if (!config.charge_string.empty()) {
    auto tokens = occ::util::tokenize(config.charge_string, ",");
    for (const auto &token : tokens) {
      charges.push_back(std::stoi(token));
    }
    opts.use_wolf_sum = true;
    opts.use_crystal_polarization = true;
  }

  auto calc = Calculator(c_symm, opts);

  if (charges.size() != 0) {
    calc.set_molecule_charges(charges);
  }

  calc.init_monomer_energies();
  calc.converge_lattice_energy();

  CrystalGrowthResult result = calc.evaluate_molecular_surroundings();

  auto uc_dimers = calc.crystal().unit_cell_dimers(config.cg_radius);
  auto uc_dimers_vacuum = uc_dimers;
  write_cg_structure_file(fmt::format("{}_cg.txt", basename), calc.crystal(),
                          uc_dimers);

  auto solution_terms_uc = map_unique_interactions_to_uc_molecules(
      calc.crystal(), calc.full_dimers(), uc_dimers, calc.solution_terms(),
      calc.interaction_energies(), !opts.use_asymmetric_partition);

  // TODO tidy this up, but for now just do the same thing for crystal
  // energies too so we get vacuum surface energies
  auto vacuum_terms_uc = map_unique_interactions_to_uc_molecules(
      calc.crystal(), calc.full_dimers(), uc_dimers_vacuum,
      calc.solution_terms(), calc.crystal_interaction_energies(),
      !opts.use_asymmetric_partition);

  if (config.write_kmcpp_file) {
    write_kmcpp_input_file(fmt::format("{}_kmcpp.json", basename),
                           calc.crystal(), uc_dimers, solution_terms_uc);
  }

  nlohmann::json surface_cuts_json;

  if (config.max_facets > 0) {
    compute_and_serialize_surface_cuts(calc, surface_cuts_json, opts, uc_dimers,
                                       uc_dimers_vacuum, config.max_facets);
  }

  auto cg_interaction_labels =
      write_cg_net_file(fmt::format("{}_{}_net.txt", basename, config.solvent),
                        calc.crystal(), uc_dimers);

  nlohmann::json results_json;
  serialize_cg_results(results_json, opts, calc.crystal());
  serialize_cg_dimers(results_json, calc.crystal(), result,
                      cg_interaction_labels);

  if (!surface_cuts_json.is_null()) {
    results_json["surface_cuts"] = surface_cuts_json;
  }

  std::ofstream dest(
      fmt::format("{}_{}_cg_results.json", opts.basename, opts.solvent));
  dest << results_json.dump(2);

  // calc.dipole_correction();
  return result;
}

CrystalGrowthResult run_cg(CGConfig const &config) {
  CrystalGrowthResult result;

  if (config.use_xtb) {
    result = run_cg_impl<driver::XTBCrystalGrowthCalculator>(config);
  } else {
    result = run_cg_impl<driver::CEModelCrystalGrowthCalculator>(config);
  }
  return result;
}

void run_cg_subcommand(CGConfig const &config) {
  if (config.list_solvents) {
    occ::solvent::list_available_solvents();
    return;
  }
  (void)run_cg(config);
}

} // namespace occ::main
