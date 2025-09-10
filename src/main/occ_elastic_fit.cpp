#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Validators.hpp>
#include <algorithm>
#include <fmt/os.h>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <occ/core/constants.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_labeller.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/io/crystal_json.h>
#include <occ/main/occ_elastic_fit.h>

#include <stdexcept>

using occ::crystal::Crystal;
using occ::main::EFSettings;
using occ::main::LinearSolverType;
using occ::main::LJ_AWrapper;

using occ::main::LJWrapper;
using occ::main::MorseWrapper;
using occ::main::PES;
using occ::main::PotentialType;

using occ::units::degrees;
using occ::units::EV_TO_KJ_PER_MOL;
using occ::units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
using occ::units::PI;

inline void print_vector(const occ::Vec &vec, int per_line) {
  std::string line;

  occ::Vec sorted_vec = vec;
  std::sort(sorted_vec.begin(), sorted_vec.end());

  for (size_t i = 0; i < sorted_vec.size(); ++i) {
    double val = sorted_vec(i);
    line += fmt::format("{:9.3f}", val);
    if ((i + 1) % per_line == 0 || i == vec.size() - 1) {
      spdlog::info("{}", line);
      line.clear();
    }
  }
}

inline void print_matrix_full(const occ::CMat &matrix, int precision = 6,
                              int width = 12) {
  for (int i = 0; i < matrix.rows(); ++i) {
    std::string row;
    for (int j = 0; j < matrix.cols(); ++j) {
      row += fmt::format("{:{}.{}f}", matrix(i, j).real(), width, precision);
    }
    occ::log::info("{}", row);
  }
  occ::log::info("");
}

inline void print_matrix_full(const occ::Mat6 &matrix, int precision = 6,
                              int width = 12) {
  for (int i = 0; i < 6; ++i) {
    std::string row;
    for (int j = 0; j < 6; ++j) {
      row += fmt::format("{:{}.{}f}", matrix(i, j), width, precision);
    }
    occ::log::info("{}", row);
  }
  occ::log::info("");
}

inline void print_matrix_upper_triangle(const occ::Mat6 &matrix,
                                        int precision = 3, int width = 9) {

  for (int i = 0; i < 6; ++i) {
    std::string row;
    for (int k = 0; k < i; ++k) {
      row += fmt::format("{:{}}", "", width);
    }
    for (int j = i; j < 6; ++j) {
      row += fmt::format("{:{}.{}f}", matrix(i, j), width, precision);
    }
    occ::log::info("{}", row);
  }
  occ::log::info("");
}

inline void print_matrix(const occ::Mat6 &matrix,
                         bool upper_triangle_only = false, int precision = 3,
                         int width = 9) {
  if (upper_triangle_only) {
    print_matrix_upper_triangle(matrix, precision, width);
  } else {
    print_matrix_full(matrix, precision, width);
  }
}


inline void save_matrix(const occ::Mat &matrix, const std::string &filename,
                        std::vector<std::string> comments = {},
                        bool upper_triangle_only = false, int width = 6) {
  std::ofstream file(filename);
  occ::log::info("Writing matrix to file {}", filename);
  for (const auto &comment : comments) {
    file << "# " << comment << std::endl;
  }
  file << std::fixed << std::setprecision(4);

  if (upper_triangle_only) {
    int count = 0;
    for (int i = 0; i < matrix.rows(); ++i) {
      for (int j = i; j < matrix.cols(); ++j) {
        file << std::setw(12) << matrix(i, j);
        count++;
        if (count % width == 0) {
          file << std::endl;
        } else {
          file << " ";
        }
      }
    }
    if (count % width != 0) {
      file << std::endl;
    }
    file.close();
    return;
  }

  int count = 0;
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      file << std::setw(12) << matrix(i, j);
      count++;
      if (count % width == 0) {
        file << std::endl;
      } else {
        file << " ";
      }
    }
  }
  if (count % width != 0) {
    file << std::endl;
  }

  file.close();
}

inline void build_pes_from_json(nlohmann::json j, PES &pes, EFSettings settings,
                                std::vector<std::string> &gulp_strings) {
  const auto &pairs = j["all_pairs"];
  double gs = settings.gulp_scale;
  pes.set_scale(settings.scale_factor);
  PotentialType potential_type = settings.potential_type;
  int discarded_count = 0;
  double discarded_total_energy = 0.0;

  double max_energy = 0.0;
  if (settings.max_to_zero) {
    if (settings.include_positive) {
      occ::log::warn("Can't include positive and set max to zero.");
    }
    for (size_t mol_idx = 0; mol_idx < pairs.size(); mol_idx++) {
      const auto &mol_pairs = pairs[mol_idx];
      for (const auto &pair : mol_pairs) {
        const auto &energies_json = pair["energies"];
        double total_energy = energies_json["Total"];
        if (total_energy > max_energy) {
          max_energy = total_energy;
        }
      }
    }
    occ::log::info("Shifting all pair energies down by {:.4f} kJ/mol",
                   max_energy);
    pes.set_shift(max_energy);
  }

  ankerl::unordered_dense::set<std::string> dedup_gulp_strings;

  for (size_t mol_idx = 0; mol_idx < pairs.size(); mol_idx++) {
    const auto &mol_pairs = pairs[mol_idx];
    for (const auto &pair : mol_pairs) {
      const auto r_arr = pair["rvec"];
      occ::Vec3 rvec(r_arr[0], r_arr[1], r_arr[2]);
      occ::Vec3 unit_vec = rvec.normalized();
      const std::pair<int, int> &pair_indices = pair["pair_indices"];
      const auto [p1, p2] = pair_indices;
      const std::pair<int, int> &uc_pair_indices = pair["pair_uc_indices"];
      const auto [uc_p1, uc_p2] = uc_pair_indices;

      const auto pair_masses = pair["mass"];
      const auto pair_mass =
          std::pair(static_cast<double>(pair_masses[0]),
                    static_cast<double>(pair_masses[1])); // kg / mole
      double m = std::sqrt(pair_mass.first * pair_mass.second);

      const auto &energies_json = pair["energies"];
      double total_energy = energies_json["Total"];
      total_energy -= max_energy;
      const double r0 = pair["r"];
      const double rl = r0 - gs, ru = r0 + gs;
      if (total_energy > 0.0) {
        if (!settings.include_positive) {
          occ::log::debug("Skipping pair with positive total energy {:.4f}",
                          total_energy);
          discarded_count++;
          discarded_total_energy += total_energy;
          continue;
        }
        const double eps = -1.0 * total_energy;
        auto potential = std::make_unique<LJ_AWrapper>(eps, r0, rvec);
        occ::log::debug(
            "Added LJ_A potential: {:30} between pair {:4} {:4} ({:4} {:4})",
            potential->to_string(), p1, p2, uc_p1, uc_p2);
        potential->set_pair_mass(pair_mass);
        pes.add_potential(std::move(potential));
        continue;
      }

      switch (potential_type) {
      case PotentialType::MORSE: {
        double D0 = -1.0 * total_energy;
        double h = std::pow(10, 13);
        double conversion_factor = 1.6605388e-24 * std::pow(h, 2) * 6.0221418;
        double k = m * conversion_factor; // kj/mol/angstrom^2
        double alpha = sqrt(k / (2 * abs(D0)));

        auto potential = std::make_unique<MorseWrapper>(D0, r0, alpha, rvec);
        potential->set_pair_indices(pair_indices);
        potential->set_uc_pair_indices(uc_pair_indices);
        potential->set_pair_mass(pair_mass);
        occ::log::debug("Added Morse potential: {} between pair {} {} ({} {})",
                        potential->to_string(), p1, p2, uc_p1, uc_p2);
        pes.add_potential(std::move(potential));
        break;
      }
      case PotentialType::LJ: {
        double eps = -1.0 * total_energy;
        auto potential = std::make_unique<LJWrapper>(eps, r0, rvec);
        potential->set_pair_indices(pair_indices);
        potential->set_uc_pair_indices(uc_pair_indices);
        potential->set_pair_mass(pair_mass);
        occ::log::debug(
            "Added LJ potential: {:30} between pair {:4} {:4} ({:4} {:4})",
            potential->to_string(), p1, p2, uc_p1, uc_p2);

        auto [smaller, larger] = std::minmax(uc_p1, uc_p2);
        const std::string gulp_str =
            fmt::format("X{} core X{} core {:12.5f} {:12.5f} {:12.5f} {:12.5f}",
                        smaller + 1, larger + 1, eps, r0, rl, ru);
        dedup_gulp_strings.insert(gulp_str);
        pes.add_potential(std::move(potential));
        break;
      }
      case PotentialType::LJ_A: {
        throw std::runtime_error("Should not have happened.");
      }
      }
    }
  }
  for (const auto &str : dedup_gulp_strings) {
    gulp_strings.push_back(str);
  }

  if (discarded_count > 0) {
    occ::log::warn("Discarded {} pairs with positive interaction energies "
                   "(total: {:.3f} kJ/mol)",
                   discarded_count, discarded_total_energy / 2.0);
  }
}

inline PotentialType
determine_potential_type(const std::string &user_preference) {
  if (!user_preference.empty()) {
    if (user_preference == "morse")
      return PotentialType::MORSE;
    if (user_preference == "lj")
      return PotentialType::LJ;
  }
  occ::log::debug("Unrecognised user preference '{}' for potential type",
                  user_preference);
  occ::log::debug("Options are 'morse' or 'lj'"); // TODO: make generic
  occ::log::info("Using default potential type (Lennard-Jones)");
  return PotentialType::LJ;
}


inline LinearSolverType
determine_solver_type(const std::string &user_preference) {
  if (!user_preference.empty()) {
    if (user_preference == "lu")
      return occ::main::LinearSolverType::LU;
    if (user_preference == "svd")
      return occ::main::LinearSolverType::SVD;
    if (user_preference == "qr")
      return occ::main::LinearSolverType::QR;
    if (user_preference == "ldlt")
      return occ::main::LinearSolverType::LDLT;
  }
  occ::log::debug("Unrecognised solver type '{}', using SVD decomposition",
                  user_preference);
  occ::log::debug("Options are 'lu', 'svd', 'qr', 'ldlt'");
  return occ::main::LinearSolverType::SVD;
}

inline occ::IVec3 determine_mp_shrinking_factors(
    const std::vector<size_t> &shrinking_factors_raw) {
  size_t n_sf = shrinking_factors_raw.size();
  if (n_sf == 1) {
    size_t sf = shrinking_factors_raw[0];
    return occ::IVec3(sf, sf, sf);
  } else if (n_sf == 3) {
    return occ::IVec3(shrinking_factors_raw[0], shrinking_factors_raw[1],
                      shrinking_factors_raw[2]);
  } else {
    throw std::runtime_error(fmt::format(
        "Raw shrinking factor input had size {} (should be 1 or 3)", n_sf));
  }
}

inline occ::Vec3 determine_mp_shifts(const std::vector<double> &shifts_raw) {
  size_t n_shifts = shifts_raw.size();
  if (n_shifts == 1) {
    double shift = shifts_raw[0];
    return occ::Vec3(shift, shift, shift);
  } else if (n_shifts == 3) {
    return occ::Vec3(shifts_raw[0], shifts_raw[1], shifts_raw[2]);
  } else {
    throw std::runtime_error(fmt::format(
        "Raw shifts input had size {} (should be 1 or 3)", n_shifts));
  }
}


inline void analyse_elat_results(const occ::main::EFSettings &settings) {
  std::string filename = settings.json_filename;
  occ::log::info("Reading elat results from: {}", filename);

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(fmt::format("Could not open file: {}", filename));
  }

  nlohmann::json j;
  file >> j;

  if (j["result_type"] != "elat") {
    throw std::runtime_error("Invalid JSON: not an elat result file");
  }

  if (!j.contains("all_pairs")) {
    throw std::runtime_error("Need 'all_pairs' in JSON output.");
  }

  occ::log::info("Title: {}", j["title"].get<std::string>());
  occ::log::info("Model: {}", j["model"].get<std::string>());


  Crystal crystal = j["crystal"];

  std::vector<std::string> gulp_strings;
  gulp_strings.push_back("conp prop phon noden hessian");
  gulp_strings.push_back("");
  gulp_strings.push_back("cell");

  const auto &uc = crystal.unit_cell();
  const auto &lengths = uc.lengths();
  const auto &angles = uc.angles();
  std::string cry_str =
      fmt::format("{:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}",
                  lengths[0], lengths[1], lengths[2], degrees(angles[0]),
                  degrees(angles[1]), degrees(angles[2]));
  gulp_strings.push_back(cry_str);
  gulp_strings.push_back("");
  gulp_strings.push_back("cart");
  const auto &uc_mols = crystal.unit_cell_molecules();
  for (const auto &mol : uc_mols) {
    const auto &com = mol.center_of_mass();
    const int mol_idx = mol.unit_cell_molecule_idx();
    std::string gulp_str = fmt::format("X{} core {:12.8f} {:12.8f} {:12.8f}",
                                       mol_idx + 1, com[0], com[1], com[2]);
    gulp_strings.push_back(gulp_str);
  }
  gulp_strings.push_back("");
  gulp_strings.push_back("element");
  for (const auto &mol : uc_mols) {
    double m = mol.molar_mass() * 1000;
    const int mol_idx = mol.unit_cell_molecule_idx();
    std::string gulp_str = fmt::format("mass X{} {:8.4f}", mol_idx + 1, m);
    gulp_strings.push_back(gulp_str);
  }
  gulp_strings.push_back("end");
  gulp_strings.push_back("");
  gulp_strings.push_back("space");
  gulp_strings.push_back("1");
  gulp_strings.push_back("");

  occ::main::EFSettings updated_settings = settings;

  updated_settings.potential_type =
      determine_potential_type(settings.potential_type_str);
  updated_settings.solver_type =
      determine_solver_type(settings.solver_type_str);
  updated_settings.shrinking_factors =
      determine_mp_shrinking_factors(settings.shrinking_factors_raw);
  updated_settings.shift = determine_mp_shifts(settings.shift_raw);

  std::string type_name;
  if (updated_settings.potential_type == PotentialType::MORSE) {
    type_name = "Morse";
    gulp_strings.push_back("morse inter kjmol");
  } else {
    type_name = "Lennard-Jones";
    gulp_strings.push_back("lennard epsilon kjmol");
  }
  occ::log::info("Using {} potential", type_name);

  PES pes(crystal);
  pes.set_temperature(settings.temperature);

  build_pes_from_json(j, pes, updated_settings, gulp_strings);

  gulp_strings.push_back("");
  gulp_strings.push_back("output drv file");

  if (!settings.gulp_file.empty()) {
    occ::log::info("Writing coarse-grained crystal to GULP input '{}'",
                   settings.gulp_file);
    occ::log::warn("The coarse-grained GULP input has not been thoroughly "
                   "tested. Use with caution.");
    std::ofstream gulp_file(settings.gulp_file);
    if (gulp_file.is_open()) {
      std::copy(gulp_strings.begin(), gulp_strings.end(),
                std::ostream_iterator<std::string>(gulp_file, "\n"));
    }
  }

  double elat = pes.lattice_energy(); // per mole of unit cells
  occ::Mat6 cij =
      pes.compute_elastic_tensor(crystal.volume(), updated_settings.solver_type,
                                 updated_settings.svd_threshold);

  if (settings.max_to_zero) {
    double og_elat = elat + pes.number_of_potentials() * pes.shift() / 2.0;
    occ::log::info("Unaltered lattice energy {:.3f} kJ/(mole unit cells)",
                   og_elat);
    occ::log::info("Shifted lattice energy {:.3f} kJ/(mole unit cells)", elat);
    occ::log::info("Shifted elastic constant matrix: (Units=GPa)");
    print_matrix(cij, true);
    cij *= og_elat / elat;
    occ::log::info("Scaled+shifted elastic constant matrix: (Units=GPa)");
    print_matrix(cij, true);
  } else {
    occ::log::info("Lattice energy {:.3f} kJ/(mole unit cells)", elat);
    occ::log::info("Elastic constant matrix: (Units=GPa)");
    print_matrix(cij, true);
  }
  save_matrix(cij, settings.output_file);

  pes.phonons(updated_settings.shrinking_factors, updated_settings.shift,
              updated_settings.animate_phonons);
}

namespace occ::main {

CLI::App *add_elastic_fit_subcommand(CLI::App &app) {
  CLI::App *elastic_fit = app.add_subcommand(
      "elastic_fit", "fit elastic tensor from ELAT JSON results");
  auto config = std::make_shared<EFSettings>();

  elastic_fit

      ->add_option("json_file", config->json_filename, "ELAT JSON results file")
      ->required()
      ->check(CLI::ExistingFile);


  elastic_fit->add_option("-o,--out", config->output_file,
                          "Output filename for elastic tensor");

  elastic_fit->add_option("-s,--scale", config->scale_factor,
                          "Factor to scale alpha by.");

  elastic_fit->add_option("-p,--potential", config->potential_type_str,
                          "Potential type to fit to. Either 'morse' or 'lj'.");

  elastic_fit->add_option("-g,--gulp-file", config->gulp_file,
                          "Write coarse grained crystal as a GULP input file.");

  elastic_fit->add_option(
      "--gulp-scale", config->gulp_scale,
      "Fraction of pair distance to set min and max cutoff for GULP.");

  elastic_fit->add_flag("--include-positive", config->include_positive,
                        "Whether or not to include positive "
                        "dimer energies when fitting the elastic tensor.");

  elastic_fit->add_flag("--max-to-zero", config->max_to_zero,
                        "Whether or not to shift all pair energies "
                        "such that the maximum is zero.");

  elastic_fit->add_option("--solver", config->solver_type_str,
                          "Linear solver type for elastic tensor calculation. "
                          "Options: 'lu', 'svd' (default), 'qr', 'ldlt'.");

  elastic_fit->add_option("-t,--temperature", config->temperature,
                          "Temperature in Kelvin for Uvib calculation.");

  elastic_fit->add_option(
      "--svd-threshold", config->svd_threshold,
      "SVD threshold for pseudoinverse (when using SVD solver).");

  elastic_fit->add_flag("--animate-phonons", config->animate_phonons,
                        "Animate the phonons and write them to XYZ files.");

  elastic_fit
      ->add_option("--mp-shrinking-factors", config->shrinking_factors_raw,
                   "Shrinking factors for Monkhorst-Pack for phonons (either 1 "
                   "or 3 numbers).")
      ->expected(1, 3);

  elastic_fit
      ->add_option(
          "--mp-shift", config->shift_raw,
          "Origin shift for Monkhorst-Pack for phonons (either 1 or 3 numbers)")
      ->expected(1, 3);

  elastic_fit->callback([config]() { run_elastic_fit_subcommand(*config); });

  return elastic_fit;
}

void run_elastic_fit_subcommand(const EFSettings &settings) {
  try {
    analyse_elat_results(settings);
  } catch (const std::exception &e) {
    occ::log::error("Error analysing ELAT results: {}", e.what());
    exit(1);
  }
}

} // namespace occ::main
