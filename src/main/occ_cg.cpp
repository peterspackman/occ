#include <CLI/App.hpp>
#include <memory>
#include <occ/driver/cg_runner.h>
#include <occ/main/occ_cg.h>
#include <occ/solvent/solvation_correction.h>

namespace occ::main {

CLI::App *add_cg_subcommand(CLI::App &app) {
  CLI::App *cg =
      app.add_subcommand("cg", "compute crystal growth free energies");
  auto config = std::make_shared<occ::driver::CGConfig>();
  cg->add_option("input", config->lattice_settings.crystal_filename,
                 "input CIF")
      ->required();
  cg->add_option("-r,--radius", config->lattice_settings.max_radius,
                 "maximum radius (Angstroms) for neighbours");
  cg->add_option("-m,--model", config->lattice_settings.model_name,
                 "energy model");
  cg->add_option(
      "--convergence-threshold,--convergence_threshold",
      config->lattice_settings.energy_tolerance,
      "energy convergence threshold (kJ/mol) for direct space summation");
  cg->add_option("-c,--cg-radius", config->cg_radius,
                 "maximum radius (Angstroms) for nearest neighbours in CG "
                 "file (must be <= radius)");
  cg->add_option("-s,--solvent", config->solvent, "solvent name");
  cg->add_option("--charges", config->charge_string, "system net charge");
  cg->add_option("-w,--wavefunction-choice", config->wavefunction_choice,
                 "Choice of wavefunctions");
  cg->add_flag("--write-kmcpp", config->write_kmcpp_file,
               "write out an input file for kmcpp program");
  cg->add_flag("--xtb", config->use_xtb, "use xtb for interaction energies");
  cg->add_flag("--dry-run", config->dry_run,
               "don't calculate any interaction energies, but calculate a net "
               "and structure file");
  cg->add_option("--xtb-solvation-model,--xtb_solvation_model",
                 config->xtb_solvation_model,
                 "solvation model for use with xtb interaction energies");
  cg->add_flag("-d,--dump", config->write_dump_files, "Write dump files");
  cg->add_flag("--atomic", config->crystal_is_atomic,
               "Crystal is atomic (i.e. no bonds)");
  cg->add_flag(
      "--asymmetric-solvent-contribution,--asymmetric_solvent_contribution",
      config->asymmetric_solvent_contribution,
      "Crystal growth interactions will not have permutational symmetry (i.e. "
      "A->B != B->A) (default: false)");
  cg->add_flag(
      "--gamma-point-molecules,--gamma_point_molecules",
      config->gamma_point_molecules,
      "Enforce that the resulting unit cell molecules (e.g. in the net file) "
      "must have geometric centroids in the range [0,1) (default: true)");
  cg->add_option("--surface-energies", config->max_facets,
                 "Calculate surface energies and write .gmf morphology files");
  cg->add_flag("--list-available-solvents", config->list_solvents,
               "List available solvents and exit");
  cg->fallthrough();
  cg->callback([config]() { run_cg_subcommand(*config); });
  return cg;
}

void run_cg_subcommand(occ::driver::CGConfig const &config) {
  if (config.list_solvents) {
    occ::solvent::list_available_solvents();
    return;
  }
  (void)occ::driver::run_cg(config);
}

} // namespace occ::main
