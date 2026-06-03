#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/driver/geometry_optimization.h>
#include <occ/driver/single_point.h>
#include <occ/interaction/pair_energy.h>
#include <occ/io/cifparser.h>
#include <occ/io/crystal_json.h>
#include <occ/qm/io/fchkwriter.h>
#include <occ/qm/io/gaussian_input_file.h>
#include <occ/io/occ_input.h>
#include <occ/io/pc.h>
#include <occ/qm/io/qcschema.h>
#include <occ/qm/io/wavefunction_json.h>
#include <occ/io/xyz.h>
#include <occ/main/cli_validators.h>
#include <occ/main/occ_scf.h>
#include <occ/main/properties.h>
#include <occ/main/version.h>
#include <xc.h>

namespace occ::main {

namespace fs = std::filesystem;
using occ::io::OccInput;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;

void read_input_file(const std::string &filename, OccInput &config) {
  occ::timing::stop(occ::timing::category::io);
  auto path = fs::path(filename);
  std::string ext = path.extension().string();
  occ::log::debug("Attempting to read input from {}, file extension = {}",
                  filename, ext);
  if (!fs::exists(path))
    throw std::runtime_error("Input file does not exist.");
  if (ext == ".gjf" || ext == ".com") {
    occ::log::debug("Detected Gaussian input file");
    occ::io::GaussianInputFile g(filename);
    g.update_occ_input(config);
  } else if (ext == ".json") {
    occ::log::debug("Detected JSON input");
    occ::io::QCSchemaReader qcs(filename);
    qcs.update_occ_input(config);
  } else if (ext == ".xyz") {
    occ::log::debug("Detected xyz input");
    occ::io::XyzFileReader xyz(filename);
    xyz.update_occ_input(config);
  } else if (ext == ".cif") {
    occ::log::debug("Detected CIF input");
    occ::io::CifParser cif;
    auto crystal = cif.parse_crystal_from_file(filename);
    if (!crystal)
      throw std::runtime_error(fmt::format("Could not parse crystal: {}",
                                           cif.failure_description()));
    config.crystal.asymmetric_unit = (*crystal).asymmetric_unit();
    config.crystal.unit_cell = (*crystal).unit_cell();
    config.crystal.space_group = (*crystal).space_group();
    config.driver.driver = "crystal";
  } else {
    throw std::runtime_error("unknown file type");
  }
}

void write_output_files(const OccInput &config, Wavefunction &wfn) {
  for (const auto &format : config.output.formats) {
    fs::path path = config.filename;
    if (!config.solvent.solvent_name.empty()) {
      path.replace_extension(fmt::format(".solvated.owf.{}", format));
    } else {
      path.replace_extension(fmt::format(".owf.{}", format));
    }
    wfn.save(path.string());
  }
}

CLI::App *add_scf_subcommand(CLI::App &app) {
  auto config = std::make_shared<occ::io::OccInput>();

  CLI::App *scf =
      app.add_subcommand("scf", "Perform an SCF on a molecular geometry");
  scf->fallthrough();

  // Advanced / rarely-needed tuning flags are assigned to this group, which is
  // hidden from `occ scf --help` (an empty CLI11 group name suppresses an option
  // from help). They are still parsed, so existing scripts keep working, and
  // `occ scf --help-expert` reveals them.
  const std::string kHidden = "";

  // Named groups so `--help` reads as labelled sections rather than one flat
  // list. (Options left in the default group appear under "Options".)
  const std::string gDF = "Density fitting / acceleration";
  const std::string gMP2 = "MP2";
  const std::string gCC = "Coupled cluster";
  const std::string gSolv = "Solvation & dispersion";
  const std::string gOpt = "Geometry optimization & frequencies";

  // Reveal the hidden tuning options (move them into a visible group) and print
  // the full help. trigger_on_parse() runs the callback during parsing, so it
  // works without the otherwise-required input file.
  scf->add_flag_function(
         "--help-expert",
         [scf](std::int64_t) {
           for (const auto *opt : scf->get_options()) {
             if (opt->get_group().empty())
               const_cast<CLI::Option *>(opt)->group("Expert (advanced tuning)");
           }
           throw CLI::CallForHelp();
         },
         "show advanced / expert tuning options and exit")
      ->trigger_on_parse();

  CLI::Option *input_option =
      scf->add_option("input,--geometry-filename,--geometry_filename",
                      config->filename, "input file");
  input_option->check(CLI::ExistingFile);
  input_option->required();

  scf->add_option("method_name,--method", config->method.name, "method name");
  scf->add_option("basis_name,--basis", config->basis.name, "basis set name");
  // electronic
  scf->add_option("-c,--charge", config->electronic.charge,
                  "system net charge");

  scf->add_option("-o,--output", config->output.formats, "output formats");

  auto *multiplicity_option = scf->add_option(
      "--multiplicity", config->electronic.multiplicity, "system multiplicity");
  multiplicity_option->check(occ::main::validator::Multiplicity);

  scf->add_flag_function(
      "-u,--unrestricted",
      [config](int count) {
        config->electronic.spinorbital_kind = SpinorbitalKind::Unrestricted;
      },
      "use unrestricted SCF");

  scf->add_option("--driver", config->driver.driver, "override driver");

  scf->add_option("--basis-set-directory,--basis_set_directory",
                  config->basis.basis_set_directory,
                  "override basis set directory");

  scf->add_option("--integral-precision,--integral_precision",
                  config->method.integral_precision,
                  "cutoff for integral screening")
      ->group(kHidden);

  // SCF acceleration policy (density fitting / COSX selection)
  scf->add_option("--ri", config->method.ri_policy,
                  "SCF acceleration: auto (default; DF-J everywhere, DF-K below "
                  "and COSX above an nbf crossover for exact exchange) | none "
                  "(conventional) | jk (force DF) | cosx (force DF-J + COSX-K)")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, occ::io::RIPolicy>{
              {"auto", occ::io::RIPolicy::Auto},
              {"none", occ::io::RIPolicy::None},
              {"off", occ::io::RIPolicy::None},
              {"conventional", occ::io::RIPolicy::None},
              {"jk", occ::io::RIPolicy::JK},
              {"j", occ::io::RIPolicy::JK},
              {"ri", occ::io::RIPolicy::JK},
              {"cosx", occ::io::RIPolicy::COSX},
              {"rijcosx", occ::io::RIPolicy::COSX},
          },
          CLI::ignore_case))
      ->option_text("auto|none|jk|cosx")
      ->group(gDF);

  // Auxiliary basis sets. Defaults are chosen automatically from the orbital
  // basis (see share/basis/fitting_defaults.json) when these are left unset.
  scf->add_option("--aux,-d,--df-basis,--density_fitting_basis",
                  config->basis.df_name,
                  "SCF (J/K) auxiliary basis; 'auto' generates one. Default: "
                  "matched to the orbital basis.")
      ->group(gDF);
  scf->add_option("--corr-aux,--ri-basis", config->basis.ri_basis,
                  "correlation (RI/C) auxiliary basis for MP2/CCSD. Default: "
                  "matched to the orbital basis.")
      ->group(gDF);

  // --- hidden tuning knobs (still parsed) ---
  scf->add_flag("--direct-df,--use-direct-df-kernels,--use_direct_df_kernels",
                config->method.use_direct_df_kernels,
                "use direct density fitting kernels (recompute integrals each iteration)")
      ->group(kHidden);
  scf->add_flag("--split-ri-j,--split_ri_j", config->method.use_split_ri_j,
                "use Split-RI-J for Coulomb matrix (Neese 2003)")
      ->group(kHidden);
  scf->add_flag("--cosx", config->method.use_cosx,
                "force COSX seminumerical exchange (see --ri cosx)")
      ->group(kHidden);
  scf->add_option("--cosx-grid", config->method.cosx_grid_level,
                  "COSX grid level: 1=Grid1 (fast), 2=Grid2, 3=Grid3 (accurate)")
      ->transform(CLI::CheckedTransformer(
          std::map<std::string, occ::io::COSXGridLevel>{
              {"1", occ::io::COSXGridLevel::Grid1},
              {"2", occ::io::COSXGridLevel::Grid2},
              {"3", occ::io::COSXGridLevel::Grid3},
              {"grid1", occ::io::COSXGridLevel::Grid1},
              {"grid2", occ::io::COSXGridLevel::Grid2},
              {"grid3", occ::io::COSXGridLevel::Grid3},
          },
          CLI::ignore_case))
      ->group(kHidden);
  scf->add_option("--cosx-screen-threshold", config->method.cosx.screen_threshold,
                  "COSX shell screening threshold (default 1e-4)")
      ->group(kHidden);
  scf->add_option("--cosx-margin", config->method.cosx.margin,
                  "COSX geometric margin in Bohr (default 1.0)")
      ->group(kHidden);
  scf->add_option("--cosx-f-threshold", config->method.cosx.f_threshold,
                  "COSX F-intermediate threshold (default 1e-10)")
      ->group(kHidden);
  scf->add_option("--df-auto-threshold", config->basis.df_auto_threshold,
                  "Cholesky threshold for the 'auto' auxiliary basis (default: 1e-4)")
      ->group(kHidden);

  // dft grid (hidden)
  scf->add_option("--xc-screen-threshold,--xc_screen_threshold",
                  config->method.dft_xc_screening_threshold,
                  "DFT XC per-batch shell screening tolerance (default 1e-10, "
                  "larger = more screening/faster, <=0 disables)")
      ->group(kHidden);
  scf->add_option("--dft-grid-max-angular,--dft_grid_max_angular",
                  config->method.dft_grid.max_angular_points,
                  "maximum angular grid points for DFT integration")
      ->group(kHidden);
  scf->add_option("--dft-grid-min-angular,--dft_grid_min_angular",
                  config->method.dft_grid.min_angular_points,
                  "minimum angular grid points for DFT integration")
      ->group(kHidden);
  scf->add_option("--dft-grid-radial-precision,--dft_grid_radial_precision",
                  config->method.dft_grid.radial_precision,
                  "radial precision for DFT integration")
      ->group(kHidden);
  scf->add_option(
         "--dft-grid-reduce-light-elements,--dft_grid_reduce_light_elements",
         config->method.dft_grid.reduced_first_row_element_grid,
         "use a reduced radial grid for first-row elements")
      ->group(kHidden);
  scf->add_option("--mp2-backend", config->method.mp2_backend,
                  "MP2 integral backend: auto (RI if --ri-basis else "
                  "conventional) | ri/df (RI-MP2) | thc (LS-THC-MP2). The method "
                  "name may also request it, e.g. ri-mp2 / thc-mp2. RI/THC use "
                  "--ri-basis, else a matched correlation aux is chosen.")
      ->group(gMP2);
  scf->add_option("--mp2-spin-scaling", config->method.mp2_spin_scaling,
                  "MP2 spin-component scaling: none | scs | sos")
      ->group(gMP2);
  scf->add_option("--mp2-max-memory", config->method.mp2_max_memory_gb,
                  "MP2 memory budget in GiB (controls occupied blocking and "
                  "whether the dense 3-center store is used; default: 1.0)")
      ->group(gMP2);
  scf->add_option("--mp2-thc-c", config->method.mp2_thc_c_isdf,
                  "THC interpolation rank = c * nbf for --mp2-backend thc "
                  "(default: 6)")
      ->group(kHidden);
  scf->add_option("--mp2-thc-method", config->method.mp2_thc_method,
                  "MP2 THC ISDF point selector: cholesky | qr (default: cholesky)")
      ->group(kHidden);
  scf->add_option("--mp2-laplace-points", config->method.mp2_laplace_points,
                  "Laplace quadrature points for the THC-MP2 denominator "
                  "(default: 14)")
      ->group(kHidden);
  scf->add_option("--ccsd-backend", config->method.ccsd_backend,
                  "CCSD(T) integral backend: exact | df (= ri) | thc. The method "
                  "name may also request it, e.g. ri-ccsd(t) / thc-ccsd(t). "
                  "df/thc use --ri-basis, else a matched correlation aux is "
                  "chosen.")
      ->group(gCC);
  scf->add_option("--ccsd-frozen-core", config->method.ccsd_frozen_core,
                  "CCSD(T) frozen core orbitals: -1 auto (chemical core, "
                  "default), 0 all-electron, N freeze N lowest")
      ->group(gCC);
  scf->add_option("--ccsd-max-memory", config->method.ccsd_max_memory_gb,
                  "CCSD(T) integral-build memory budget in GiB (default: 1.0)")
      ->group(gCC);
  scf->add_option("--ccsd-thc-c", config->method.ccsd_thc_c_isdf,
                  "THC interpolation rank = c * nbf for --ccsd-backend thc "
                  "(default: 6 = sub-mHa sweet spot; cost ~ c^2)")
      ->group(kHidden);
  scf->add_option("--ccsd-thc-method", config->method.ccsd_thc_method,
                  "THC ISDF point selector: cholesky | qr (default: cholesky)")
      ->group(kHidden);
  scf->add_option("--ccsd-thc-grid-angular", config->method.ccsd_thc_grid_angular,
                  "THC candidate-grid max angular points (default: 110; raise "
                  "for a finer ISDF grid)")
      ->group(kHidden);
  scf->add_option("--ccsd-thc-grid-radial", config->method.ccsd_thc_grid_radial,
                  "THC candidate-grid radial precision (default: 1e-7; lower = "
                  "more radial points)")
      ->group(kHidden);
  scf->add_flag("--spherical", config->basis.spherical,
                "use spherical basis sets");

  scf->add_option("--orbital-smearing-sigma,--orbital_smearing_sigma",
                  config->method.orbital_smearing_sigma,
                  "Orbital smearing sigma")
      ->group(kHidden);
  // point charges
  CLI::Option *pc_option = scf->add_option(
      "--point-charges,--point_charge_file",
      config->geometry.point_charge_filename, "file listing point charges");
  pc_option->check(CLI::ExistingFile);

  // Solvation
  scf->add_option("-s,--solvent,--solvent_name", config->solvent.solvent_name,
                  "Solvent name")
      ->group(gSolv);
  scf->add_option("-f,--solvent-file,--solvent_file",
                  config->solvent.output_surface_filename,
                  "file to write solvent surface")
      ->group(gSolv);
  scf->add_flag("--solvent-radii-scaling,--solvent_radii_scaling,--draco",
                config->solvent.radii_scaling, "use DRACO for radii scaling")
      ->group(gSolv);
  // XDM
  scf->add_flag("--xdm", config->dispersion.evaluate_correction,
                "use XDM dispersion correction")
      ->group(gSolv);
  scf->add_option("--xdm-a1,--xdm_a1", config->dispersion.xdm_a1,
                  "a1 parameter for XDM")
      ->group(kHidden);
  scf->add_option("--xdm-a2,--xdm_a2", config->dispersion.xdm_a2,
                  "a2 parameter for XDM")
      ->group(kHidden);

  scf->add_option("--chelpg,", config->chelpg_filename,
                  "Filename for CHELPG charges");

  // Optimization convergence criteria
  scf->add_option("--opt-gradient-max,--opt_gradient_max",
                  config->optimization.gradient_max,
                  "Maximum gradient component for convergence (Ha/Angstrom)")
      ->group(gOpt);
  scf->add_option("--opt-gradient-rms,--opt_gradient_rms",
                  config->optimization.gradient_rms,
                  "RMS gradient for convergence (Ha/Angstrom)")
      ->group(gOpt);
  scf->add_option("--opt-step-max,--opt_step_max",
                  config->optimization.step_max,
                  "Maximum displacement for convergence (Angstrom)")
      ->group(gOpt);
  scf->add_option("--opt-step-rms,--opt_step_rms",
                  config->optimization.step_rms,
                  "RMS displacement for convergence (Angstrom)")
      ->group(gOpt);
  scf->add_option("--opt-energy-change,--opt_energy_change",
                  config->optimization.energy_change,
                  "Energy change threshold for convergence (Hartree)")
      ->group(gOpt);
  scf->add_flag("--opt-use-energy,--opt_use_energy",
                config->optimization.use_energy_criterion,
                "Use energy change as convergence criterion")
      ->group(gOpt);
  scf->add_option("--opt-max-iterations,--opt_max_iterations",
                  config->optimization.max_iterations,
                  "Maximum number of optimization steps")
      ->group(gOpt);
  scf->add_option("--opt-gradient-precision,--opt_gradient_precision",
                  config->optimization.gradient_integral_precision,
                  "Final gradient integral precision")
      ->group(kHidden);
  scf->add_option("--opt-early-gradient-precision,--opt_early_gradient_precision",
                  config->optimization.early_gradient_integral_precision,
                  "Looser gradient integral precision for early steps")
      ->group(kHidden);
  scf->add_option("--opt-tight-threshold,--opt_tight_threshold",
                  config->optimization.tight_gradient_threshold,
                  "Energy change threshold to switch to tight gradient precision (Hartree)")
      ->group(kHidden);
  scf->add_flag("--opt-write-wavefunctions,--opt_write_wavefunctions",
                config->optimization.write_wavefunction_steps,
                "Write wavefunction at each optimization step")
      ->group(gOpt);
  scf->add_flag("--frequencies,--freq",
                config->optimization.compute_frequencies,
                "Compute vibrational frequencies after geometry optimization")
      ->group(gOpt);

  scf->callback([config]() { run_scf_subcommand(*config); });
  return scf;
}

void run_scf_subcommand(occ::io::OccInput config) {

  occ::main::print_header();

  occ::timing::start(occ::timing::category::io);

  config.name = config.filename;
  // read input file first so we can override with command line settings
  read_input_file(config.filename, config);
  if (config.filename.empty()) {
    config.filename = config.name;
  }
  occ::timing::stop(occ::timing::category::io);

  if (!config.geometry.point_charge_filename.empty()) {
    occ::io::PointChargeFileReader pc(config.geometry.point_charge_filename);
    pc.update_occ_input(config);
  }

  occ::log::info("Driver: {}", config.driver.driver);
  if (config.driver.driver == "opt") {
    if (config.optimization.compute_frequencies) {
      auto [wfn, vib_modes] = driver::geometry_optimization_with_frequencies(config, true);
      write_output_files(config, wfn);
    } else {
      Wavefunction wfn = driver::geometry_optimization(config);
      write_output_files(config, wfn);
    }
  } else {
    // store solvent name so we can do an unsolvated calculation first
    std::string stored_solvent_name = config.solvent.solvent_name;
    config.solvent.solvent_name = "";

    Wavefunction wfn = occ::driver::single_point(config);
    write_output_files(config, wfn);
    if (wfn.method != "GFN2-xTB") {
      // GFN2 already prints its own native summary inside the driver; the
      // standard wavefunction properties printer (multipoles via HF integrals,
      // Mulliken via orbital partition) is misleading for a tight-binding
      // method, so skip it.
      occ::main::calculate_dispersion(config, wfn);
      occ::main::calculate_properties(config, wfn);
    }

    config.solvent.solvent_name = stored_solvent_name;

    if (!config.solvent.solvent_name.empty()) {
      Wavefunction wfn2 = occ::driver::single_point(config, wfn);
      double esolv = wfn2.energy.total - wfn.energy.total;

      occ::log::info("{:-<72s}", "Solvation free energy (SMD)  ");
      occ::log::info("dG(solv)    = {:20.12f} Hartree", esolv);
      occ::log::info("            = {:20.12f} kJ/mol",
                     esolv * occ::units::AU_TO_KJ_PER_MOL);
      occ::log::info("            = {:20.12f} kcal/mol",
                     esolv * occ::units::AU_TO_KCAL_PER_MOL);
      write_output_files(config, wfn2);
    }
  }
}

} // namespace occ::main
