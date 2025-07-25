#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Validators.hpp>
#include <fmt/os.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/dimer_labeller.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/io/crystal_json.h>
#include <occ/main/occ_elastic_fit.h>

using occ::crystal::Crystal;
using occ::main::LJWrapper;
using occ::main::MorseWrapper;
using occ::main::PES;
using occ::main::PotentialType;

inline PES construct_pes_from_json(nlohmann::json j,
                                   PotentialType potential_type,
                                   double scale_factor = 1.0) {
  const auto &pairs = j["all_pairs"];
  PES pes(scale_factor);

  for (size_t mol_idx = 0; mol_idx < pairs.size(); mol_idx++) {
    const auto &mol_pairs = pairs[mol_idx];
    for (const auto &pair : mol_pairs) {
      const auto r_arr = pair["rvec"];
      occ::Vec3 rvec(r_arr[0], r_arr[1], r_arr[2]);
      occ::Vec3 unit_vec = rvec.normalized();

      const auto &energies_json = pair["energies"];
      double total_energy = energies_json["Total"];
      if (total_energy > 0.0) {
        occ::log::debug("Skipping pair with positive total energy {:.4f}",
                        total_energy);
        continue;
      }
      double r0 = pair["r"];

      switch (potential_type) {
      case PotentialType::MORSE: {
        double D0 = -1.0 * total_energy;
        double m = static_cast<double>(pair["mass"]); // kg / mole
        double h = std::pow(10, 13);
        double conversion_factor = 1.6605388e-24 * std::pow(h, 2) * 6.0221418;
        double k = m * conversion_factor; // kj/mol/angstrom^2
        double alpha = sqrt(k / (2 * abs(D0)));

        auto potential = std::make_unique<MorseWrapper>(D0, r0, alpha, rvec);
        occ::log::debug("Added Morse potential: {}", potential->to_string());
        pes.add_potential(std::move(potential));
        break;
      }
      case PotentialType::LJ: {
        double eps = -1.0 * total_energy;
        auto potential = std::make_unique<LJWrapper>(eps, r0, rvec);
        occ::log::debug("Added LJ potential: {}", potential->to_string());
        pes.add_potential(std::move(potential));
        break;
      }
      }
    }
  }

  return pes;
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

inline void save_matrix(const occ::Mat6 &matrix, const std::string &filename) {
  std::ofstream file(filename);
  occ::log::info("Writing matrix to file {}", filename);
  file << std::fixed << std::setprecision(6);
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      file << std::setw(10) << matrix(i, j);
      if (j < matrix.cols() - 1)
        file << " ";
    }
    file << "\n";
  }
  file.close();
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
  occ::log::info("Using default potential type");
  return PotentialType::LJ;
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

  PotentialType pot_type = determine_potential_type(settings.potential_type);

  const char *type_name =
      (pot_type == PotentialType::MORSE) ? "Morse" : "Lennard-Jones";
  occ::log::info("Using {} potential", type_name);

  Crystal crystal = j["crystal"];

  PES pes = construct_pes_from_json(j, pot_type, settings.scale_factor);

  double elat = pes.lattice_energy(); // per mole of unit cells
  occ::log::info("Lattice energy {:.3f} kJ/(mole unit cells)", elat);
  occ::Mat6 cij = pes.compute_voigt_elastic_tensor_analytical(crystal.volume());
  occ::log::info("Elastic Constant Matrix: (Units=GPa)");
  print_matrix(cij, true);
  save_matrix(cij, settings.output_file);
}

namespace occ::main {

CLI::App *add_elastic_fit_subcommand(CLI::App &app) {
  CLI::App *morse = app.add_subcommand(
      "elastic_fit", "fit elastic tensor from ELAT JSON results");
  auto config = std::make_shared<EFSettings>();

  morse
      ->add_option("json_file", config->json_filename, "ELAT JSON results file")
      ->required()
      ->check(CLI::ExistingFile);

  morse->add_option("-o,--out", config->output_file,
                    "Output filename for elastic tensor");

  morse->add_option("-s,--scale", config->scale_factor,
                    "Factor to scale alpha by.");

  morse->add_option("-p,--potential", config->potential_type,
                    "Potential type to fit to. Either 'morse' or 'lj'.");

  morse->callback([config]() { run_elastic_fit_subcommand(*config); });

  return morse;
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
