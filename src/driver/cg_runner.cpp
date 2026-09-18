#include <algorithm>
#include <filesystem>
#include <fmt/os.h>
#include <fstream>
#include <occ/cg/interaction_mapper.h>
#include <occ/core/kabsch.h>
#include <occ/core/point_group.h>
#include <occ/crystal/dimer_labeller.h>
#include <occ/crystal/surface.h>
#include <occ/dft/dft.h>
#include <occ/driver/cg_pipeline.h>
#include <occ/driver/cg_runner.h>
#include <occ/driver/cg_solvation_model.h>
#include <occ/driver/crystal_growth.h>
#include <occ/driver/crystal_morphology.h>
#include <occ/driver/crystal_surface_energy.h>
#include <occ/geometry/wulff.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/polarization.h>
#include <occ/io/core_json.h>
#include <occ/io/crystal_json.h>
#include <occ/io/crystalgrower.h>
#include <occ/io/eigen_json.h>
#include <occ/io/kmcpp.h>
#include <occ/io/load_geometry.h>
#include <occ/io/occ_input.h>
#include <occ/io/xyz.h>
#include <occ/isosurface/ply.h>
#include <occ/qm/hf.h>
#include <occ/qm/io/wavefunction_json.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>
#include <optional>

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

namespace occ::driver {

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
  // Every distinct face image of every positive-energy cut; the Wulff
  // construction keeps the innermost plane of each face. Normals come from the
  // integer images, as in the morphology (see face_images).
  const Mat3 recip = s.crystal.unit_cell().reciprocal();
  std::vector<Vec3> normals;
  std::vector<double> energies;
  for (const auto &facet : s.facets) {
    if (facet.energy <= 0.0)
      continue;
    for (const auto &[image, symop] :
         occ::crystal::face_images(s.crystal, facet.hkl)) {
      normals.push_back((recip * Vec3(image.h, image.k, image.l)).normalized());
      energies.push_back(facet.energy);
    }
  }
  occ::Mat3N directions(3, normals.size());
  occ::Vec expanded_energies(normals.size());
  for (size_t j = 0; j < normals.size(); ++j) {
    directions.col(j) = normals[j];
    expanded_energies(j) = energies[j];
  }
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
    if (const auto &fe = mol_result.free_energy) {
      e["free_energy"] = {
          {"temperature", 298.0},
          {"lattice_energy", fe->lattice_energy},
          {"rotational_free_energy", fe->rotational_free_energy},
          {"translational_free_energy", fe->translational_free_energy},
          {"solvation_free_energy", fe->solvation_free_energy},
          {"dH_sublimation", fe->dH_sublimation},
          {"dS_sublimation", fe->dS_sublimation},
          {"dG_sublimation", fe->dG_sublimation},
          {"dG_solution", fe->dG_solution},
          {"equilibrium_constant", fe->equilibrium_constant},
          {"log_S", fe->log_S},
          {"solubility_g_per_L", fe->solubility_g_per_L},
          {"total_interaction_energy", fe->total_interaction_energy}};
    }
    if (!mol_result.descriptors.empty()) {
      nlohmann::json descriptors;
      for (const auto &[k, v] : mol_result.descriptors)
        descriptors[k] = v;
      e["descriptors"] = descriptors;
    }
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
      if (!dimer_result.descriptors.empty()) {
        nlohmann::json descriptors;
        for (const auto &[k, v] : dimer_result.descriptors)
          descriptors[k] = v;
        d["descriptors"] = descriptors;
      }

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
  j["energy_model"] = opts.energy_model;
  j["solvation_model"] = solvation_model_name(opts.solvation_model);
  j["has_permutation_symmetry"] = !opts.use_asymmetric_partition;

  j["crystal"] = crystal;
}

inline CrystalSurfaceEnergies compute_and_serialize_surface_cuts(
    CrystalGrowthCalculator &calc, nlohmann::json &j, const Options &opts,
    const CrystalDimers &uc_dimers, const CrystalDimers &uc_dimers_vacuum,
    int max_facets, double min_interplanar_spacing) {

  occ::log::info("Crystal surface energies (solvated)");
  auto surface_energies = calculate_crystal_surface_energies(
      fmt::format("{}_{}", opts.basename, opts.solvent_tag), calc.crystal(),
      uc_dimers, max_facets, 1, min_interplanar_spacing);

  occ::log::info("Crystal surface energies (vacuum)");
  auto vacuum_surface_energies = calculate_crystal_surface_energies(
      fmt::format("{}_vacuum", opts.basename), calc.crystal(), uc_dimers_vacuum,
      max_facets, -1, min_interplanar_spacing);

  j["surface_energies"] = surface_energies;
  write_wulff(fmt::format("{}_{}.ply", opts.basename, opts.solvent_tag),
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
  return surface_energies;
}

CGPreparation prepare_cg(CGConfig const &config) {
  std::string basename =
      fs::path(config.lattice_settings.crystal_filename).stem().string();
  Crystal c_symm =
      occ::io::load_crystal(config.lattice_settings.crystal_filename);

  c_symm.set_gamma_point_unit_cell_molecules(config.gamma_point_molecules);

  if (config.crystal_is_atomic) {
    c_symm.set_connectivity_criteria(false);
  }

  if (c_symm.asymmetric_unit().positions.cols() == 0) {
    throw std::runtime_error(fmt::format(
        "No atoms found in '{}' - the structure is empty. Check the input file "
        "(e.g. a CIF missing its _atom_site loop).",
        config.lattice_settings.crystal_filename));
  }

  Options opts;
  opts.solvent = config.solvent;
  opts.solvent_tag = SolventSpec::parse(config.solvent).filename_tag();
  opts.solvation_model = parse_solvation_model(config.solvation_model);
  opts.print_solvation_descriptors = config.print_solvation_descriptors;
  opts.temperature = config.temperature;
  opts.solvent_probe_radius = config.solvent_probe_radius;
  opts.basename = basename;
  opts.write_debug_output_files = config.write_dump_files;
  // --xtb with the default model name still runs xtb; say so in the output.
  opts.energy_model =
      config.use_xtb && !occ::interaction::model_name_implies_xtb(
                            config.lattice_settings.model_name)
          ? "xtb"
          : config.lattice_settings.model_name;
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

  return CGPreparation{std::move(c_symm), std::move(opts), std::move(charges)};
}

CrystalGrowthResult run_cg_pipeline(CrystalGrowthCalculator &calc,
                                    const Options &opts,
                                    CGConfig const &config) {
  const std::string &basename = opts.basename;

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

  const double d_min = config.min_interplanar_spacing;
  int n_facets = config.max_facets;
  if (d_min > 0.0) {
    if (n_facets > 0)
      occ::log::warn("--surface-d-min given, ignoring --surface-energies {}",
                     n_facets);
    n_facets = 0;
  } else {
    if (n_facets > 0)
      occ::log::warn(
          "--surface-energies is deprecated: a face count can split a Friedel "
          "pair, which skews the Wulff construction. Prefer --surface-d-min, "
          "which cuts on interplanar spacing");
    // --morphology needs surface energies; default the facet count if unset.
    if (config.compute_morphology && n_facets <= 0) {
      n_facets = 80;
      occ::log::info("--morphology: computing {} surface energies", n_facets);
    }
  }

  std::optional<CrystalSurfaceEnergies> surface_energies;
  if (n_facets > 0 || d_min > 0.0) {
    surface_energies = compute_and_serialize_surface_cuts(
        calc, surface_cuts_json, opts, uc_dimers, uc_dimers_vacuum, n_facets,
        d_min);
  }

  nlohmann::json morphology_json;
  if (!config.compute_morphology &&
      (!config.morphology_sizes.empty() || !config.morphology_shape.empty())) {
    occ::log::warn("--morphology-sizes and --morphology-shape have no effect "
                   "without --morphology");
  }
  if (config.compute_morphology && surface_energies) {
    occ::log::info("Computing particle size/shape-dependent energies");
    MorphologyOptions morphology_options;
    if (!config.morphology_sizes.empty()) {
      if (std::any_of(config.morphology_sizes.begin(),
                      config.morphology_sizes.end(),
                      [](int n) { return n <= 0; })) {
        throw std::invalid_argument("--morphology-sizes must all be positive");
      }
      morphology_options.sizes = config.morphology_sizes;
    }
    // A shape file may hold many habits. Everything the morphology needs --
    // pair energies, monomers, the surface enumeration -- is already done, so
    // the extra habits cost only the shape construction itself.
    std::vector<occ::driver::NamedShape> shapes;
    if (!config.morphology_shape.empty()) {
      std::ifstream shape_file(config.morphology_shape);
      if (!shape_file) {
        throw std::runtime_error(fmt::format(
            "Cannot read morphology shape file '{}'", config.morphology_shape));
      }
      shapes = read_morphology_shapes(shape_file);
      occ::log::info("Read {} shape(s) from {}", shapes.size(),
                     config.morphology_shape);
    } else {
      shapes.push_back({});   // the equilibrium (Wulff) shape
    }

    std::vector<MorphologyResult> morphologies;
    morphologies.reserve(shapes.size());
    for (const auto &shape : shapes) {
      morphology_options.user_shifts = shape.shifts;
      auto morphology =
          compute_crystal_morphology(calc.crystal(), uc_dimers,
                                     *surface_energies, result,
                                     morphology_options);
      morphology.name = shape.name;
      morphologies.push_back(std::move(morphology));
    }

    result.morphology = morphologies.front();
    if (morphologies.size() == 1) {
      to_json(morphology_json, result.morphology);
    } else {
      // Many habits: an array, so a scan is one run and one file. The singular
      // key stays a single object so existing readers keep working.
      morphology_json = nlohmann::json::array();
      for (const auto &morphology : morphologies) {
        nlohmann::json entry;
        to_json(entry, morphology);
        morphology_json.push_back(std::move(entry));
      }
    }
  }

  auto cg_interaction_labels = write_cg_net_file(
      fmt::format("{}_{}_net.txt", basename, opts.solvent_tag), calc.crystal(),
      uc_dimers);

  nlohmann::json results_json;
  serialize_cg_results(results_json, opts, calc.crystal());
  serialize_cg_dimers(results_json, calc.crystal(), result,
                      cg_interaction_labels);

  if (!surface_cuts_json.is_null()) {
    results_json["surface_cuts"] = surface_cuts_json;
  }
  if (!morphology_json.is_null()) {
    results_json[morphology_json.is_array() ? "morphologies" : "morphology"] =
        morphology_json;
  }

  std::ofstream dest(
      fmt::format("{}_{}_cg_results.json", opts.basename, opts.solvent_tag));
  dest << results_json.dump(2);

  return result;
}

// Thin wrapper: prepare, construct the concrete calculator, run the pipeline.
template <class Calculator>
CrystalGrowthResult run_cg_impl(CGConfig const &config) {
  auto prep = prepare_cg(config);
  Calculator calc(prep.crystal, prep.opts);
  if (!prep.charges.empty())
    calc.set_molecule_charges(prep.charges);
  return run_cg_pipeline(calc, prep.opts, config);
}

CrystalGrowthResult run_cg(CGConfig const &config) {
  CrystalGrowthResult result;

  const bool use_xtb_route =
      config.use_xtb || occ::interaction::model_name_implies_xtb(
                            config.lattice_settings.model_name);

  if (config.dry_run) {
    result = run_cg_impl<DummyCrystalGrowthCalculator>(config);
  } else if (use_xtb_route) {
    if (!config.use_xtb) {
      occ::log::info(
          "Model '{}' selected — routing through XTBCrystalGrowthCalculator "
          "(equivalent to --xtb).",
          config.lattice_settings.model_name);
    }
    result = run_cg_impl<XTBCrystalGrowthCalculator>(config);
  } else {
    result = run_cg_impl<CEModelCrystalGrowthCalculator>(config);
  }
  return result;
}

} // namespace occ::driver
