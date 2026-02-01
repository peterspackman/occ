#pragma once

#include <CLI/CLI.hpp>
#include <string>
#include <trajan/core/trajectory.h>
#include <trajan/io/file_handler.h>
#include <trajan/io/selection.h>

namespace trajan::main {

namespace io = trajan::io;
using trajan::core::Trajectory;

struct RDFOpts {
  std::string outfile{"gofr.out"};
  double rcut = 6.0;
  int nbins = 100;
  std::string raw_sel1, raw_sel2;
  std::vector<io::SelectionCriteria> parsed_sel1, parsed_sel2;
};

struct RDFResult {
  std::vector<double> r;
  std::vector<double> nofr;
  std::vector<double> gofr;

  RDFResult(size_t nbins) {
    r.resize(nbins, 0.0);
    nofr.resize(nbins, 0.0);
    gofr.resize(nbins, 0.0);
  }

  void normalise_by_count(size_t count);
};

void run_rdf_subcommand(RDFOpts const &opts, Trajectory &traj);
CLI::App *add_rdf_subcommand(CLI::App &app, Trajectory &traj);

} // namespace trajan::main
