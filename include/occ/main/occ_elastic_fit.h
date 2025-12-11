#pragma once
#include <CLI/App.hpp>
#include <occ/elastic_fit/monkhorst_pack.h>
#include <occ/elastic_fit/pes.h>
#include <occ/elastic_fit/potentials.h>

namespace occ::main {

using occ::elastic_fit::LinearSolverType;
using occ::elastic_fit::LJ_AWrapper;
using occ::elastic_fit::LJWrapper;
using occ::elastic_fit::MonkhorstPack;
using occ::elastic_fit::MorseWrapper;
using occ::elastic_fit::PES;
using occ::elastic_fit::PotentialBase;
using occ::elastic_fit::PotentialType;

struct EFSettings {
  std::string json_filename;
  std::string output_file = "elastic_tensor.txt";
  std::string potential_type_str = "lj";
  PotentialType potential_type = PotentialType::LJ;
  bool include_positive = false;
  bool max_to_zero = false;
  double scale_factor = 2.0;
  double temperature = 0.0;
  double gulp_scale = 0.01;
  std::string gulp_file{""};
  LinearSolverType solver_type = LinearSolverType::SVD;
  std::string solver_type_str = "svd";
  double svd_threshold = 1e-12;
  bool animate_phonons = false;
  bool save_debug_matrices = false;
  std::vector<size_t> shrinking_factors_raw{1};
  occ::IVec3 shrinking_factors{1, 1, 1};
  std::vector<double> shift_raw{0.0};
  occ::Vec3 shift{0.0, 0.0, 0.0};
};

CLI::App *add_elastic_fit_subcommand(CLI::App &app);
void run_elastic_fit_subcommand(const EFSettings &settings);

} // namespace occ::main
