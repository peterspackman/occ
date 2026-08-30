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
#include <occ/solvent/opencosmors.h>
#include <occ/solvent/opencosmors_io.h>
#include <occ/solvent/sigma_io.h>
#include <occ/solvent/sigma_solvation.h>

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
  sigma->add_option("--write-segments", config->segments_filename,
                    "write this molecule's openCOSMO-RS segment ensemble "
                    "(.rsseg) so it can be reused as a cached solvent");
  sigma->add_flag("--opencosmors", config->opencosmors,
                  "use the openCOSMO-RS kernel against the stored .rsseg "
                  "ensemble for --solvent");
  sigma->add_option("--solvent-geometry", config->solvent_geometry,
                    "solvent geometry; assembles the openCOSMO-RS solvation "
                    "free energy against a conductor cavity computed for it");
  sigma->add_option("--liquid-volume", config->solvent_volume_liquid,
                    "solute liquid molar volume (Angstrom^3 per molecule) for "
                    "the openCOSMO-RS reference-state term");
  sigma->add_option("--rings", config->num_rings,
                    "number of rings in the solute, for the openCOSMO-RS ring "
                    "correction");
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

  const auto dispersion = occ::solvent::sigma::dispersion_parameters(
      molecule.atomic_numbers(), molecule.positions() *
                                     occ::units::ANGSTROM_TO_BOHR);

  occ::log::info("cavity area          {:12.5f} Angstrom^2", result.cavity_area);
  occ::log::info("cavity volume        {:12.5f} Angstrom^3",
                 result.cavity_volume);
  occ::log::info("segments             {:12d}", result.segments.size());
  occ::log::info("screening charge     {:12.5f} e", result.screening_charge);
  occ::log::info("conductor stabilisation {:9.5f} Hartree",
                 result.energy_conductor - result.energy_gas);
  if (dispersion.known)
    occ::log::info("dispersion e/kB      {:12.4f} K ({})", dispersion.epsilon,
                   occ::solvent::sigma::dispersion_class_name(
                       dispersion.klass));
  else
    occ::log::warn("no COSMO-SAC-dsp parameter for this molecule; the "
                   "dispersion term will be skipped wherever it is used");

  std::string path = config.output_filename;
  if (path.empty())
    path = std::filesystem::path(config.geometry_filename).stem().string() +
           ".sigma";
  occ::solvent::sigma::write_sigma_profile(
      path, molecule.name(), profile, params, result.cavity_area,
      result.cavity_volume, dispersion);
  occ::log::info("wrote sigma profile to {}", path);

  // openCOSMO-RS descriptors: sigma on its own averaging radius plus the
  // correlation density sigma_orth. Cheap once the cavity exists.
  occ::solvent::sigma::RSParameters rs;
  auto rs_segments = result.segments;
  occ::solvent::sigma::average_sigma(rs_segments, rs.r_av, 1.0);
  occ::solvent::sigma::average_sigma_orth(rs_segments, rs.r_av, rs.r_corr,
                                          rs.sigma_orth_factor);
  const auto rs_solute = occ::solvent::sigma::RSComponent::from_segments(
      rs_segments, result.cavity_volume, result.cavity_area);

  if (!config.segments_filename.empty()) {
    occ::solvent::sigma::write_rs_segments(
        config.segments_filename, molecule.name(), rs_solute,
        rs_segments.atomic_number, rs, config.method, config.basis);
    occ::log::info("wrote openCOSMO-RS segments to {}",
                   config.segments_filename);
  }

  auto report_opencosmors = [&](const occ::solvent::sigma::RSComponent &solvent,
                                const std::string &label) {
    occ::solvent::sigma::RSOptions rs_options;
    rs_options.temperature = config.temperature;
    occ::solvent::sigma::RSSolventModel model(solvent, rs, rs_options);
    auto energy = occ::solvent::sigma::rs_solvation_free_energy(
        model, rs_solute, rs_segments,
        result.energy_conductor - result.energy_gas, config.num_rings,
        config.solvent_volume_liquid,
        occ::solvent::sigma::SolvationParameters::opencosmors_24a());

    const double k = occ::units::AU_TO_KJ_PER_MOL;
    occ::log::info("openCOSMO-RS 24a solvation free energy in '{}' (kJ/mol)",
                   label);
    occ::log::info("  E_diel           {:10.3f}", energy.dielectric * k);
    occ::log::info("  residual         {:10.3f}", energy.residual * k);
    occ::log::info("  combinatorial    {:10.3f}", energy.combinatorial * k);
    occ::log::info("  cavity           {:10.3f}", energy.cavity * k);
    occ::log::info("  ring             {:10.3f}", energy.ring * k);
    occ::log::info("  reference state  {:10.3f}", energy.reference_state * k);
    occ::log::info("  constant         {:10.3f}", energy.constant * k);
    occ::log::info("  total            {:10.3f}", energy.total() * k);
  };

  // A cached solvent ensemble avoids recomputing the solvent cavity, which
  // is what makes a solvent screen cheap.
  if (config.opencosmors && !config.solvent.empty()) {
    auto store = occ::solvent::sigma::RSProfileStore::standard();
    auto solvent = store.get(config.solvent);
    if (solvent.r_av > 0.0 &&
        std::abs(solvent.r_av - rs.r_av) > 1e-12)
      throw std::runtime_error(fmt::format(
          "solvent '{}' was averaged on r_av = {} but the parameters in use "
          "specify {}",
          config.solvent, solvent.r_av, rs.r_av));
    if (!solvent.basis.empty() && solvent.basis != config.basis)
      occ::log::warn("solvent '{}' segments were computed with {}/{} but this "
                     "solute uses {}/{}; the descriptors are not comparable",
                     config.solvent, solvent.method, solvent.basis,
                     config.method, config.basis);
    report_opencosmors(solvent.component, config.solvent);
    return;
  }

  if (config.solvent.empty() && config.solvent_geometry.empty())
    return;

  // Solvation energy of this molecule in the named solvent, when a profile
  // for it is available: E_diel (gas -> conductor) + the residual contraction.
  if (!config.solvent.empty()) {
    auto store = occ::solvent::sigma::ProfileStore::standard();
    if (store.contains(config.solvent)) {
      occ::solvent::sigma::PotentialOptions options;
      options.temperature = config.temperature;
      occ::solvent::sigma::SolventModel solvent(store.get(config.solvent),
                                                params, options);
      const double e_diel = (result.energy_conductor - result.energy_gas) *
                            occ::units::AU_TO_KJ_PER_MOL;
      const double residual =
          solvent.segment_energies(result.segments).sum() *
          occ::units::AU_TO_KJ_PER_MOL;
      const double hbond =
          solvent.segment_hbond_area(result.segments).sum();
      occ::log::info("solvation in '{}': E_diel {:.4f} + residual {:.4f} = "
                     "{:.4f} kJ/mol  (hydrogen-bonded area {:.3f} A^2)",
                     config.solvent, e_diel, residual, e_diel + residual,
                     hbond);
    }
  }

  // openCOSMO-RS: both cavities are built here, so no stored profile is
  // involved and the solvent carries the segment descriptors its kernel
  // needs rather than a binned profile.
  if (!config.solvent_geometry.empty()) {
    auto solvent_molecule =
        occ::io::molecule_from_xyz_file(config.solvent_geometry);
    occ::gto::AOBasis solvent_basis =
        occ::gto::AOBasis::load(solvent_molecule.atoms(), config.basis);
    solvent_basis.set_pure(!config.cartesian);
    occ::dft::DFT solvent_gas(config.method, solvent_basis);
    occ::qm::SCF<occ::dft::DFT> solvent_scf(solvent_gas);
    solvent_scf.compute_scf_energy();
    auto solvent_conductor =
        occ::driver::conductor_profile(solvent_scf.wavefunction(), settings);

    auto solvent_segments = solvent_conductor.segments;
    occ::solvent::sigma::average_sigma(solvent_segments, rs.r_av, 1.0);
    occ::solvent::sigma::average_sigma_orth(solvent_segments, rs.r_av,
                                            rs.r_corr, rs.sigma_orth_factor);
    report_opencosmors(occ::solvent::sigma::RSComponent::from_segments(
                           solvent_segments, solvent_conductor.cavity_volume,
                           solvent_conductor.cavity_area),
                       solvent_molecule.name());
    return;
  }

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
