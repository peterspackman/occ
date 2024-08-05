#pragma once
#include <CLI/App.hpp>

namespace occ::main {

struct ElasticSettings {
  std::string tensor_filename;
  double scale{1.0};
  std::string output_json_filename{"elastic_properties.json"};
  std::string basename;
  int subdivisions{5};
};

CLI::App *add_elastic_subcommand(CLI::App &app);
void run_elastic_subcommand(ElasticSettings const &);

} // namespace occ::main
