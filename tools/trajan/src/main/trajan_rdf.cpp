#include <CLI/CLI.hpp>
#include <ankerl/unordered_dense.h>
#include <memory>
#include <stdexcept>
#include <trajan/core/log.h>
#include <trajan/core/neigh.h>
#include <trajan/core/trajectory.h>
#include <trajan/core/util.h>
#include <trajan/io/file_handler.h>
#include <trajan/io/selection.h>
#include <trajan/io/text.h>
#include <trajan/main/trajan_rdf.h>

namespace trajan::main {

namespace core = trajan::core;
namespace io = trajan::io;

void RDFResult::normalise_by_count(size_t count) {
  if (count == 0) {
    throw std::runtime_error("Count is 0.");
  }
  double d_count = count;
  for (size_t i = 0; i < gofr.size(); ++i) {
    nofr[i] /= d_count;
    gofr[i] /= d_count;
  }
}

void run_rdf_subcommand(const RDFOpts &opts) {

  RDFResult rdf(opts.nbins);
  double bin_width = opts.rcut / opts.nbins;
  double inv_bin_width = 1 / bin_width;
  double norm = 4.0 * trajan::units::PI / 3.0;
  for (size_t i = 0; i < opts.nbins; i++) {
    double ri = (i + 0.5) * bin_width;
    rdf.r[i] = ri;
  }

  core::Trajectory trajectory;

  trajectory.load_files(opts.infiles);

  size_t frame_count = 0;
  while (trajectory.next_frame()) {
    core::UnitCell uc = trajectory.unit_cell();
    // TODO: find a way to not create a new nl each time
    core::NeighbourList nl(uc, opts.rcut);
    std::vector<core::EntityType> sel1 =
        trajectory.get_entities(*opts.parsed_sel1);
    std::vector<core::EntityType> sel2 =
        trajectory.get_entities(*opts.parsed_sel2);
    // TODO: molecule origin input

    nl.update({sel1, sel2});

    std::fill(rdf.nofr.begin(), rdf.nofr.end(), 0.0);

    core::NeighbourCallback func = [&](const core::Entity &ent1,
                                       const core::Entity &ent2, double rsq) {
      double r = std::sqrt(rsq);
      size_t bin_idx = r * inv_bin_width;
      rdf.nofr[bin_idx]++;
    };
    nl.iterate_neighbours(func);
    double density_norm = sel1.size() * sel2.size() / uc.volume();
    for (size_t i = 0; i < opts.nbins; i++) {
      double ri = rdf.r[i];
      double shell_volume = norm * (std::pow(ri + bin_width / 2, 3) -
                                    std::pow(ri - bin_width / 2, 3));
      rdf.gofr[i] += rdf.nofr[i] / shell_volume / density_norm;
    }
    frame_count++;
  }

  rdf.normalise_by_count(frame_count);

  TextFileWriter outfile;
  outfile.open(opts.outfile);
  outfile.write_line("{:>16} {:>16}", "r", "gofr");
  std::string fmt_str = "{:>16.8f} {:>16.8f}";
  for (size_t i = 0; i < opts.nbins; i++) {
    outfile.write_line(fmt_str, rdf.r[i], rdf.gofr[i]);
  }
  outfile.close();
}

CLI::App *add_rdf_subcommand(CLI::App &app) {
  CLI::App *rdf =
      app.add_subcommand("rdf", "Radial Pair Distribution Function");
  auto opts = std::make_shared<RDFOpts>();
  rdf->add_option("--tr,--traj", opts->infiles, "Input trajectory file name")
      ->required()
      ->check(CLI::ExistingFile);
  rdf->add_option("--o,--out", opts->outfile, "Output file for RDF data")
      ->capture_default_str();
  rdf->add_option("--rc,--rcut", opts->rcut, "RDF cutoff")
      ->capture_default_str();
  rdf->add_option("--nb,--nbins", opts->nbins, "Number of bins for RDF")
      ->capture_default_str();
  std::string sel1 = "--s1,--sel1";
  rdf->add_option(sel1, opts->raw_sel1,
                  "First selection (prefix: i=atom indices, a=atom types, "
                  "j=molecule indices, m=molecule types)\n"
                  "Examples:\n"
                  "  i1,2,3-5    (atom indices 1,2,3,4,5)\n"
                  "  aC,N,O      (atom types C, N, O)\n"
                  "  j1,3-5      (molecule indices 1,3,4,5)\n"
                  "  mM1,M2      (molecule types M1,M2)")
      ->required()
      ->check(io::selection_validator(opts->parsed_sel1));

  rdf->add_option("--s2,--sel2", opts->raw_sel2,
                  fmt::format("Second selection (same format as {})", sel1))
      ->required()
      ->check(io::selection_validator(opts->parsed_sel2));

  opts->num_threads = app.get_option("--threads")->as<size_t>();

  rdf->callback([opts]() { run_rdf_subcommand(*opts); });
  return rdf;
}

} // namespace trajan::main
