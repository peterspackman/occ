#pragma once
#include <CLI/CLI.hpp>
#include <string>
#include <trajan/energy/xtb.h>
#include <trajan/io/file_handler.h>
#include <trajan/io/selection.h>

namespace trajan::main {

using XTBModel = energy::XTBModel;
namespace io = trajan::io;

struct OPTOpts : trajan::util::Opts {
  std::string outfile;
  std::string raw_sel1, raw_sel2;
  std::optional<io::SelectionCriteria> parsed_sel1, parsed_sel2;
  XTBModel::Type energy_model_type = XTBModel::Type::GFNFF;
  double epsilon = 1e-4;
  size_t max_iter = 100;
};

void run_opt_subcommand(OPTOpts const &opts);
CLI::App *add_opt_subcommand(CLI::App &app);

} // namespace trajan::main
