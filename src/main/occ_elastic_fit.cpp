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
#include <occ/crystal/dimer_mapping_table.h>
#include <occ/elastic_fit/elastic_fitting.h>
#include <occ/elastic_fit/elastic_fit_json.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/interaction_json.h>
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

// Old JSON parsing functions removed - functionality moved to structured data
// classes

// Convert EFSettings to FittingSettings
inline occ::elastic_fit::FittingSettings
convert_to_fitting_settings(const occ::main::EFSettings &ef_settings) {
  occ::elastic_fit::FittingSettings settings;
  settings.potential_type = ef_settings.potential_type;
  settings.include_positive = ef_settings.include_positive;
  settings.max_to_zero = ef_settings.max_to_zero;
  settings.scale_factor = ef_settings.scale_factor;
  settings.temperature = ef_settings.temperature;
  settings.gulp_scale = ef_settings.gulp_scale;
  settings.solver_type = ef_settings.solver_type;
  settings.svd_threshold = ef_settings.svd_threshold;
  settings.animate_phonons = ef_settings.animate_phonons;
  settings.save_debug_matrices = ef_settings.save_debug_matrices;
  settings.shrinking_factors = ef_settings.shrinking_factors;
  settings.shift = ef_settings.shift;
  return settings;
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
  // Set up fitting configuration first
  occ::main::EFSettings updated_settings = settings;
  updated_settings.potential_type =
      determine_potential_type(settings.potential_type_str);
  updated_settings.solver_type =
      determine_solver_type(settings.solver_type_str);
  updated_settings.shrinking_factors =
      determine_mp_shrinking_factors(settings.shrinking_factors_raw);
  updated_settings.shift = determine_mp_shifts(settings.shift_raw);

  occ::elastic_fit::FittingSettings fitting_settings =
      convert_to_fitting_settings(updated_settings);

  std::string type_name =
      (fitting_settings.potential_type == PotentialType::MORSE)
          ? "Morse"
          : "Lennard-Jones";
  occ::log::info("Using {} potential", type_name);

  occ::elastic_fit::ElasticFitter fitter(fitting_settings);
  occ::elastic_fit::FittingResults results;

  // Check if input is the new elastic_fit_pairs format
  {
    std::ifstream check_file(settings.json_filename);
    nlohmann::json check_json;
    check_file >> check_json;
    if (check_json.contains("format_type") &&
        check_json["format_type"] == "elastic_fit_pairs") {
      occ::log::info("Detected elastic_fit_pairs format");
      occ::elastic_fit::ElasticFitInput input =
          occ::elastic_fit::read_elastic_fit_json(settings.json_filename);
      occ::log::info("Loaded {} molecules and {} pairs",
                     input.molecules.size(), input.pairs.size());
      results = fitter.fit_elastic_tensor(input);
      goto output_results;
    }
  }

  {
    // Load elat format data
    occ::log::info("Detected elat format");
    occ::interaction::ElatResults elat_data =
        occ::interaction::read_elat_json(settings.json_filename);

    // Export to new format if requested
    if (!settings.export_pairs_file.empty()) {
      occ::elastic_fit::ElasticFitInput input =
          occ::elastic_fit::ElasticFitter::convert_elat_to_input(elat_data);
      occ::elastic_fit::write_elastic_fit_json(settings.export_pairs_file, input);
      occ::log::info("Exported {} molecules and {} pairs to '{}'",
                     input.molecules.size(), input.pairs.size(),
                     settings.export_pairs_file);
    }

    results = fitter.fit_elastic_tensor(elat_data);
  }

output_results:
  // Output results
  if (!settings.gulp_file.empty()) {
    occ::log::info("Writing coarse-grained crystal to GULP input '{}'",
                   settings.gulp_file);
    occ::log::warn("The coarse-grained GULP input has not been thoroughly "
                   "tested. Use with caution.");
    std::ofstream gulp_file(settings.gulp_file);
    if (gulp_file.is_open()) {
      std::copy(results.gulp_strings.begin(), results.gulp_strings.end(),
                std::ostream_iterator<std::string>(gulp_file, "\n"));
    }
  }

  if (settings.max_to_zero && results.energy_shift_applied > 0.0) {
    occ::elastic_fit::ElasticFitter::print_elastic_tensor(
        results.elastic_tensor, "Shifted elastic constant matrix: (Units=GPa)");
    occ::log::info("Final elastic constant matrix: (Units=GPa)");
  } else {
    occ::log::info("Lattice energy {:.3f} kJ/(mole unit cells)",
                   results.lattice_energy);
  }
  occ::elastic_fit::ElasticFitter::print_elastic_tensor(
      results.elastic_tensor, "Elastic constant matrix: (Units=GPa)");
  occ::elastic_fit::ElasticFitter::save_elastic_tensor(results.elastic_tensor,
                                                       settings.output_file);
}

namespace occ::main {

CLI::App *add_elastic_fit_subcommand(CLI::App &app) {
  CLI::App *elastic_fit = app.add_subcommand(
      "elastic_fit", "fit elastic tensor from pairwise interaction energies");
  elastic_fit->fallthrough();
  auto config = std::make_shared<EFSettings>();

  elastic_fit
      ->add_option("json_file", config->json_filename,
                   "Input JSON file (elat results or elastic_fit_pairs format)")
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

  elastic_fit->add_flag(
      "--save-debug-matrices", config->save_debug_matrices,
      "Save debug matrices (D_ij, D_ei, D_ee, etc.) to files.");

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

  elastic_fit->add_option(
      "--export-pairs", config->export_pairs_file,
      "Export minimal pairs JSON (for testing/external use)");

  elastic_fit->callback([config]() { run_elastic_fit_subcommand(*config); });

  return elastic_fit;
}

void run_elastic_fit_subcommand(const EFSettings &settings) {
  analyse_elat_results(settings);
}

} // namespace occ::main
