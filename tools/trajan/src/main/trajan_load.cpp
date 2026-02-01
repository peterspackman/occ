#include <trajan/main/trajan_load.h>

namespace trajan::main {

void run_load_subcommand(const LoadOpts &opts, Trajectory &traj) {
  if (opts.into_mem) {
    traj.load_files_into_memory(opts.infiles);
    return;
  }
  traj.load_files(opts.infiles);
}

CLI::App *add_load_subcommand(CLI::App &app, Trajectory &traj) {
  CLI::App *load =
      app.add_subcommand("load", "Load trajectory data into program for "
                                 "analysis. Required by most subcommands.");
  auto opts = std::make_shared<LoadOpts>();
  load->add_option("files", opts->infiles, "Input trajectory file names")
      ->required()
      ->check(CLI::ExistingFile);
  load->add_flag("--into-mem", opts->into_mem,
                 "Whether or not to load files into memory. This is not "
                 "recommended for command-line use.");
  load->callback([opts, &traj]() { run_load_subcommand(*opts, traj); });
  return load;
}

} // namespace trajan::main
