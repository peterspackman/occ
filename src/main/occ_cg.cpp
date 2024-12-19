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
using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;

using occ::driver::WavefunctionChoice;

void write_cg_structure_file(const std::string &filename,
                             const Crystal &crystal,
                             const CrystalDimers &uc_dimers) {
  occ::log::info("Writing crystalgrower structure file to '{}'", filename);
  occ::io::crystalgrower::StructureWriter cg_structure_writer(filename);
  cg_structure_writer.write(crystal, uc_dimers);
}

void write_cg_net_file(const std::string &filename, const Crystal &crystal,
                       const CrystalDimers &uc_dimers) {
  occ::log::info("Writing crystalgrower net file to '{}'", filename);
  occ::io::crystalgrower::NetWriter cg_net_writer(filename);
  cg_net_writer.write(crystal, uc_dimers);
}

void write_kmcpp_input_file(const std::string &filename, const Crystal &crystal,
                            const CrystalDimers &uc_dimers,
                            const std::vector<double> &solution_terms) {
  occ::log::info("Writing kmcpp structure file to '{}'", filename);
  occ::io::kmcpp::InputWriter kmcpp_structure_writer(filename);
  kmcpp_structure_writer.write(crystal, uc_dimers, solution_terms);
}

std::vector<double> map_unique_interactions_to_uc_molecules(
    const Crystal &crystal, const CrystalDimers &dimers,
    CrystalDimers &uc_dimers, const std::vector<double> &solution_terms,
    const std::vector<occ::cg::Energies> &interaction_energies_vec,
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
  occ::io::IsosurfaceMesh mesh =
      occ::io::mesh_from_vertices_faces(wulff.vertices(), wulff.triangles());
  occ::io::write_ply_mesh(filename, mesh, {}, false);
}

void write_cg_dimers(const std::string &basename, const std::string &solvent,
                     const occ::crystal::Crystal &crystal,
                     const occ::cg::CrystalGrowthResult &result) {
  nlohmann::json j;
  j["result_type"] = "cg";
  j["title"] = basename;
  j["solvent"] = solvent;
  j["model"] = fmt::format("crystalclear, solvent='{}'", solvent);
  j["has_perumutation_symmetry"] = false;

  j["crystal"] = crystal;

  j["totals_per_molecule"] = {};
  for (const auto &mol_total : result.total_energies) {
    nlohmann::json e;
    e["crystal_energy"] = mol_total.crystal_energy;
    e["interaction_energy"] = mol_total.interaction_energy;
    e["solution_term"] = mol_total.solution_term;
    j["totals_per_molecule"].push_back(e);
  }

  const auto &uc_atoms = crystal.unit_cell_atoms();

  j["pairs"] = {};
  for (const auto &mol_pairs : result.pair_energies) {
    nlohmann::json m;
    for (const auto &cg_dimer : mol_pairs) {
      nlohmann::json d;
      nlohmann::json e;
      e["unique_dimer_index"] = cg_dimer.unique_dimer_index;
      e["interaction_energy"] = cg_dimer.interaction_energy;
      e["crystal_contribution"] = cg_dimer.crystal_contribution;
      e["is_nearest_neighbor"] = cg_dimer.nearest_neighbor;
      e["solvent_ab"] = cg_dimer.solvent_term.ab;
      e["solvent_ba"] = cg_dimer.solvent_term.ba;
      e["solvent_total"] = cg_dimer.solvent_term.total;
      d["energies"] = e;

      nlohmann::json offsets_a = {};
      {
        const auto &a = cg_dimer.dimer.a();
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
        const auto &b = cg_dimer.dimer.b();
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
  std::ofstream dest(fmt::format("{}_{}_cg_results.json", basename, solvent));
  dest << j.dump(2);
}

void write_uc_json(const std::string &basename, const std::string &solvent,
                   const occ::crystal::Crystal &crystal,
                   const occ::crystal::CrystalDimers &dimers) {
  nlohmann::json j;
  j["result_type"] = "cg";
  j["title"] = basename;
  const auto &uc_molecules = crystal.unit_cell_molecules();
  j["unique_sites"] = uc_molecules.size();
  const auto &neighbors = dimers.molecule_neighbors;
  nlohmann::json molecules_json;
  molecules_json["kind"] = "atoms";
  j["lattice_vectors"] = crystal.unit_cell().direct();
  molecules_json["elements"] = {};
  molecules_json["positions"] = {};
  j["neighbor_offsets"] = {};
  size_t uc_idx_a = 0;
  j["neighbor_energies"] = {};
  j["neighbor_direction_correlations"] = {};
  for (const auto &mol : uc_molecules) {
    nlohmann::json molj = mol;
    molecules_json["elements"].push_back(molj["elements"]);
    molecules_json["positions"].push_back(molj["positions"]);
    j["neighbor_energies"].push_back(nlohmann::json::array({}));
    std::vector<std::vector<int>> shifts;
    for (const auto &[n, unique_idx] : neighbors[uc_idx_a]) {
      const auto uc_shift = n.b().cell_shift();
      const auto uc_idx_b = n.b().unit_cell_molecule_idx();
      shifts.push_back({uc_shift[0], uc_shift[1], uc_shift[2], uc_idx_b});
    }
    j["neighbor_offsets"][uc_idx_a] = shifts;
    const auto &neighbors_a = neighbors[uc_idx_a];
    for (const auto &[n, unique_idx] : neighbors_a) {
      nlohmann::json energies;
      for (const auto &[k, v] : n.interaction_energies()) {
        energies[k] = v;
      }
      j["neighbor_energies"][uc_idx_a].push_back(energies);
    }
    auto corr = calculate_directional_correlation_matrix(neighbors[uc_idx_a]);
    j["neighbor_direction_correlations"].push_back(corr);
    uc_idx_a++;
  }
  j["molecules"] = molecules_json;

  std::ofstream dest(
      fmt::format("{}_{}_uc_interactions.json", basename, solvent));
  dest << j.dump(2);
}

template <class Calculator>
occ::cg::CrystalGrowthResult run_cg_impl(CGConfig const &config) {
  std::string basename =
      fs::path(config.lattice_settings.crystal_filename).stem().string();
  Crystal c_symm =
      occ::io::load_crystal(config.lattice_settings.crystal_filename);

  if (config.crystal_is_atomic) {
    c_symm.set_connectivity_criteria(false);
  }

  auto calc = Calculator(c_symm, config.solvent);
  // Setup calculator parameters
  calc.set_basename(basename);
  calc.set_output_verbosity(config.write_dump_files);
  calc.set_energy_model(config.lattice_settings.model_name);
  if (!config.charge_string.empty()) {
    std::vector<int> charges;
    auto tokens = occ::util::tokenize(config.charge_string, ",");
    for (const auto &token : tokens) {
      charges.push_back(std::stoi(token));
    }
    calc.set_molecule_charges(charges);
    calc.set_use_wolf_sum(true);
    calc.set_use_crystal_polarization(true);
  }

  if constexpr (std::is_same<Calculator,
                             driver::XTBCrystalGrowthCalculator>::value) {
    occ::log::info("XTB solvation model: {}", config.xtb_solvation_model);
    calc.set_solvation_model(config.xtb_solvation_model);
  }

  calc.set_wavefunction_choice(config.wavefunction_choice == "gas"
                                   ? WavefunctionChoice::GasPhase
                                   : WavefunctionChoice::Solvated);

  calc.init_monomer_energies();
  calc.converge_lattice_energy(config.cg_radius,
                               config.lattice_settings.max_radius);

  occ::cg::CrystalGrowthResult result = calc.evaluate_molecular_surroundings();

  auto uc_dimers = calc.crystal().unit_cell_dimers(config.cg_radius);
  auto uc_dimers_vacuum = uc_dimers;
  write_cg_structure_file(fmt::format("{}_cg.txt", basename), calc.crystal(),
                          uc_dimers);

  auto solution_terms_uc = map_unique_interactions_to_uc_molecules(
      calc.crystal(), calc.full_dimers(), uc_dimers, calc.solution_terms(),
      calc.interaction_energies(), false);

  auto solution_terms_uc_throwaway = map_unique_interactions_to_uc_molecules(
      calc.crystal(), calc.full_dimers(), uc_dimers_vacuum,
      calc.solution_terms(), calc.crystal_interaction_energies(), true);

  if (config.write_kmcpp_file) {
    write_kmcpp_input_file(fmt::format("{}_kmcpp.json", basename),
                           calc.crystal(), uc_dimers, solution_terms_uc);
  }

  if (config.max_facets > 0) {
    occ::log::info("Crystal surface energies (solvated)");
    auto surface_energies = occ::main::calculate_crystal_surface_energies(
        fmt::format("{}_{}", basename, config.solvent), calc.crystal(),
        uc_dimers, config.max_facets, 1);
    occ::log::info("Crystal surface energies (vacuum)");
    auto vacuum_surface_energies =
        occ::main::calculate_crystal_surface_energies(
            fmt::format("{}_vacuum", basename), calc.crystal(),
            uc_dimers_vacuum, config.max_facets, -1);

    nlohmann::json j;
    j["surface_energies"] = surface_energies;
    write_wulff(fmt::format("{}_{}.ply", basename, config.solvent),
                surface_energies);
    write_wulff(fmt::format("{}_vacuum.ply", basename, config.solvent),
                vacuum_surface_energies);
    j["vacuum"] = calc.crystal_interaction_energies();
    j["solvated"] = calc.interaction_energies();
    std::string surf_energy_filename =
        fmt::format("{}_surface_energies.json", basename);
    occ::log::info("Writing surface energies to '{}'", surf_energy_filename);
    std::ofstream destination(surf_energy_filename);
    destination << j.dump(2);
  }

  write_cg_dimers(basename, config.solvent, calc.crystal(), result);
  write_uc_json(basename, config.solvent, calc.crystal(), uc_dimers);

  write_cg_net_file(fmt::format("{}_{}_net.txt", basename, config.solvent),
                    calc.crystal(), uc_dimers);

  // calc.dipole_correction();
  return result;
}

occ::cg::CrystalGrowthResult run_cg(CGConfig const &config) {
  occ::cg::CrystalGrowthResult result;
  std::string basename =
      fs::path(config.lattice_settings.crystal_filename).stem().string();
  Crystal c_symm =
      occ::io::load_crystal(config.lattice_settings.crystal_filename);

  if (config.crystal_is_atomic) {
    c_symm.set_connectivity_criteria(false);
  }

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
