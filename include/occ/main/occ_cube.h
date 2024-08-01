#pragma once
#include <CLI/App.hpp>

namespace occ::main {

struct CubeConfig {
  std::string input_filename{""};
  std::string property{"electron_density"};
  std::string points_filename{""};
  std::string output_filename{"out.cube"};

  std::string spin{"both"};
  std::string functional{"blyp"};
  int mo_number{-1};

  std::vector<int> steps;
  std::vector<double> da;
  std::vector<double> db;
  std::vector<double> dc;
  std::vector<double> origin;
};

CLI::App *add_cube_subcommand(CLI::App &app);
void run_cube_subcommand(CubeConfig const &);

} // namespace occ::main
