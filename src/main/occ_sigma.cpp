#include <filesystem>
#include <fmt/os.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/driver/sigma_driver.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_sigma.h>
#include <occ/qm/scf.h>
#include <occ/solvent/sigma_io.h>

namespace occ::main {

namespace {

occ::solvent::sigma::Model parse_model(const std::string &name) {
  auto lowered = occ::util::to_lower_copy(name);
  if (lowered == "cosmo-sac-2002" || lowered == "2002")
    return occ::solvent::sigma::Model::CosmoSac2002;
  if (lowered == "cosmo-sac-2010" || lowered == "2010")
    return occ::solvent::sigma::Model::CosmoSac2010;
  throw std::runtime_error(
      fmt::format("unknown sigma model '{}' (cosmo-sac-2002, cosmo-sac-2010)",
                  name));
}

} // namespace

CLI::App *add_sigma_subcommand(CLI::App &app) {
  CLI::App *sigma = app.add_subcommand(
      "sigma", "compute a COSMO sigma profile and sigma potential");
  auto config = std::make_shared<SigmaConfig>();

  sigma->add_option("geometry", config->geometry_filename,
                    "input geometry file (xyz)")
      ->required();
  sigma->add_option("-o,--output", config->output_filename,
                    "output .sigma file (default: <geometry>.sigma)");
  sigma->add_option("--method", config->method, "DFT functional");
  sigma->add_option("--basis", config->basis, "basis set");
  sigma->add_option("--model", config->model,
                    "parameterisation (cosmo-sac-2002, cosmo-sac-2010)");
  sigma->add_option("--probe", config->probe_radius,
                    "solvent probe radius used to build the cavity (Angstrom)");
  sigma->add_option("--solvent", config->solvent,
                    "also report the sigma potential of this profile treated "
                    "as a pure solvent");
  sigma->add_option("--temperature", config->temperature,
                    "temperature for the sigma potential (K)");
  sigma->add_option("--angular-points", config->angular_points,
                    "Lebedev order per atom for the cavity");
  sigma->add_flag("--unconstrained-charge", config->unconstrained_charge,
                  "do not constrain the surface charge to -q");
  sigma->add_flag("--cartesian", config->cartesian,
                  "use cartesian (6d) basis functions");

  sigma->fallthrough();
  sigma->callback([config]() { run_sigma_subcommand(*config); });
  return sigma;
}

void run_sigma_subcommand(SigmaConfig const &config) {
  auto molecule = occ::io::molecule_from_xyz_file(config.geometry_filename);
  const auto model = parse_model(config.model);
  const auto params = occ::solvent::sigma::Parameters::for_model(model);

  occ::gto::AOBasis basis =
      occ::gto::AOBasis::load(molecule.atoms(), config.basis);
  basis.set_pure(!config.cartesian);

  occ::dft::DFT gas(config.method, basis);
  occ::qm::SCF<occ::dft::DFT> gas_scf(gas);
  gas_scf.compute_scf_energy();

  occ::driver::SigmaProfileSettings settings;
  settings.method = config.method;
  settings.basis = config.basis;
  settings.pure_spherical = !config.cartesian;
  settings.probe_radius_angs = config.probe_radius;
  settings.angular_points = config.angular_points;
  settings.constrain_charge = !config.unconstrained_charge;
  settings.model = model;

  auto result =
      occ::driver::conductor_profile(gas_scf.wavefunction(), settings);

  occ::solvent::sigma::Grid grid;
  auto profile = occ::solvent::sigma::bin_segments(result.segments, grid,
                                                   params.hbond_split());

  occ::log::info("cavity area          {:12.5f} Angstrom^2", result.cavity_area);
  occ::log::info("cavity volume        {:12.5f} Angstrom^3",
                 result.cavity_volume);
  occ::log::info("segments             {:12d}", result.segments.size());
  occ::log::info("screening charge     {:12.5f} e", result.screening_charge);
  occ::log::info("conductor stabilisation {:9.5f} Hartree",
                 result.energy_conductor - result.energy_gas);

  std::string path = config.output_filename;
  if (path.empty())
    path = std::filesystem::path(config.geometry_filename).stem().string() +
           ".sigma";
  occ::solvent::sigma::write_sigma_profile(
      path, molecule.name(), profile, params, result.cavity_area,
      result.cavity_volume);
  occ::log::info("wrote sigma profile to {}", path);

  if (config.solvent.empty())
    return;

  auto kernel =
      occ::solvent::sigma::build_kernel(grid, params, config.temperature);
  occ::solvent::sigma::PotentialOptions options;
  options.temperature = config.temperature;
  auto potential =
      occ::solvent::sigma::solve_sigma_potential(profile, kernel, options);
  occ::log::info("sigma potential converged in {} iterations (residual mu "
                 "{:.2e}, variance {:.2e})",
                 potential.iterations, potential.residual_mu,
                 potential.residual_variance);

  Vec centers = grid.centers();
  occ::log::info("{:>9} {:>10} {:>10} {:>10} {:>9}", "sigma", "p(sigma)", "mu",
                 "variance", "p_HB");
  for (int i = 0; i < grid.n; i++) {
    if (profile.total()(i) <= 0.0)
      continue;
    occ::log::info("{:+9.4f} {:10.4f} {:+10.4f} {:10.4f} {:9.4f}", centers(i),
                   profile.total()(i), potential.mu.row(i).mean(),
                   potential.variance.row(i).mean(),
                   potential.hbond_probability.row(i).mean());
  }
}

} // namespace occ::main
