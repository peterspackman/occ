#include <CLI/App.hpp>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/main/occ_cg.h>
#include <occ/main/occ_cube.h>
#include <occ/main/occ_describe.h>
#include <occ/main/occ_dimers.h>
#include <occ/main/occ_elastic.h>
#include <occ/main/occ_elat.h>
#include <occ/main/occ_isosurface.h>
#include <occ/main/occ_pair.h>
#include <occ/main/occ_scf.h>
#include <occ/main/occ_surface_cuts.h>

int main(int argc, char *argv[]) {
  occ::timing::start(occ::timing::category::global);
  occ::timing::start(occ::timing::category::io);
  occ::log::set_log_level(2);

  CLI::App app("occ - A program for quantum chemistry");
  app.allow_config_extras(CLI::config_extras_mode::error);
  app.set_config("--config", "occ_input.toml",
                 "Read configuration from an ini or TOML file", false);

  app.set_help_all_flag("--help-all", "Show help for all sub commands");

  auto *threads_option = app.add_flag_function(
      "--threads{1}",
      [](int num_threads) {
        occ::parallel::set_num_threads(std::max(1, num_threads));
      },
      "number of threads");
  threads_option->default_val(1);
  threads_option->run_callback_for_default();
  threads_option->force_callback();

  // logging verbosity
  auto *verbosity_option = app.add_flag_function(
      "--verbosity{2}",
      [](int verbosity) { occ::log::set_log_level(verbosity); },
      "logging verbosity {0=silent,1=minimal,2=normal,3=verbose,4=debug}");
  verbosity_option->default_val(2);
  verbosity_option->run_callback_for_default();
  verbosity_option->force_callback();

  // add all the subcommands here
  auto *cg = occ::main::add_cg_subcommand(app);
  auto *cube = occ::main::add_cube_subcommand(app);
  auto *describe = occ::main::add_describe_subcommand(app);
  auto *dimers = occ::main::add_dimers_subcommand(app);
  auto *elastic = occ::main::add_elastic_subcommand(app);
  auto *elat = occ::main::add_elat_subcommand(app);
  auto *iso = occ::main::add_isosurface_subcommand(app);
  auto *pair = occ::main::add_pair_subcommand(app);
  auto *scf = occ::main::add_scf_subcommand(app);
  auto *cuts = occ::main::add_surface_cuts_subcommand(app);

  // ensure we have a subcommand
  app.require_subcommand();

  constexpr auto *error_format = "exception:\n    {}\nterminating program.\n";
  try {
    CLI11_PARSE(app, argc, argv);
  } catch (const char *ex) {
    occ::log::error(error_format, ex);
    spdlog::dump_backtrace();
    return 1;
  } catch (std::string &ex) {
    occ::log::error(error_format, ex);
    spdlog::dump_backtrace();
    return 1;
  } catch (std::exception &ex) {
    occ::log::error(error_format, ex.what());
    spdlog::dump_backtrace();
    return 1;
  } catch (...) {
    occ::log::error("Exception:\n- Unknown...\n");
    spdlog::dump_backtrace();
    return 1;
  }

  occ::timing::stop(occ::timing::global);
  occ::timing::print_timings();
  occ::log::info("A job well done");
  // flush all FILE* streams before closing
  std::fflush(nullptr);
  return 0;
}
