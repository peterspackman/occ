#pragma once
#include <CLI/App.hpp>
#include <occ/core/elastic_tensor.h>
#include <occ/crystal/crystal.h>

namespace occ::main {

struct ElasticSettings {
  std::string tensor_filename;
  double scale{1.0};
  std::string output_json_filename{"elastic_properties.json"};
  std::string basename;
  int subdivisions{5};
  std::string crystal_filename;
  int max_surfaces{20};
};

CLI::App *add_elastic_subcommand(CLI::App &app);
void run_elastic_subcommand(ElasticSettings const &);
void compute_crystal_face_properties(const occ::core::ElasticTensor &tensor,
                                     const occ::crystal::Crystal &crystal,
                                     const ElasticSettings &settings);

} // namespace occ::main
