#include <fmt/os.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/dft/dft.h>
#include <occ/driver/cosmors_driver.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_cosmors.h>
#include <occ/qm/scf.h>
#include <occ/solvent/cosmors.h>
#include <occ/solvent/cosmors_io.h>

namespace occ::main {

CLI::App *add_cosmors_subcommand(CLI::App &app) {
  CLI::App *cosmors = app.add_subcommand(
      "cosmo-rs", "compute an openCOSMO-RS solvation free energy");
  auto config = std::make_shared<CosmoRSConfig>();

  cosmors->add_option("geometry", config->geometry_filename,
                    "input geometry file (xyz)")
      ->required();
  cosmors->add_option("--method", config->method, "DFT functional");
  cosmors->add_option("--basis", config->basis, "basis set");
  cosmors->add_option("--solvent", config->solvent,
                    "solvent name, resolved against the shipped segment "
                    "ensembles");
  cosmors->add_option("--solvent-geometry", config->solvent_geometry,
                    "solvent geometry file, as an alternative to --solvent: "
                    "its conductor cavity is computed rather than loaded");
  cosmors->add_option("--write-segments", config->segments_filename,
                    "write this molecule's segment ensemble (.rsseg) so it "
                    "can be reused as a cached solvent");
  cosmors->add_option("--liquid-volume", config->liquid_volume,
                    "liquid-phase volume per solute molecule (Angstrom^3) for "
                    "the reference-state term; omitted when unset");
  cosmors->add_option("--rings", config->num_rings,
                    "rings in the solute for the ring correction (default: "
                    "counted from the bond graph)");
  cosmors->add_option("--probe", config->probe_radius,
                    "solvent probe radius used to build the cavity (Angstrom)");
  cosmors->add_option("--temperature", config->temperature, "temperature (K)");
  cosmors->add_option("--angular-points", config->angular_points,
                    "Lebedev order per atom for the cavity");
  cosmors->add_flag("--unconstrained-charge", config->unconstrained_charge,
                  "do not constrain the surface charge to -q");
  cosmors->add_flag("--cartesian", config->cartesian,
                  "use cartesian (6d) basis functions");

  cosmors->fallthrough();
  cosmors->callback([config]() { run_cosmors_subcommand(*config); });
  return cosmors;
}

void run_cosmors_subcommand(CosmoRSConfig const &config) {
  namespace cosmors = occ::solvent::cosmors;

  if (!config.solvent.empty() && !config.solvent_geometry.empty())
    throw std::runtime_error(
        "--solvent and --solvent-geometry both name a solvent; pick one");

  auto molecule = occ::io::molecule_from_xyz_file(config.geometry_filename);

  occ::gto::AOBasis basis =
      occ::gto::AOBasis::load(molecule.atoms(), config.basis);
  basis.set_pure(!config.cartesian);

  occ::dft::DFT gas(config.method, basis);
  occ::qm::SCF<occ::dft::DFT> gas_scf(gas);
  gas_scf.compute_scf_energy();

  occ::driver::ConductorSettings settings;
  settings.method = config.method;
  settings.basis = config.basis;
  settings.pure_spherical = !config.cartesian;
  settings.probe_radius_angs = config.probe_radius;
  settings.angular_points = config.angular_points;
  settings.constrain_charge = !config.unconstrained_charge;

  auto result =
      occ::driver::conductor_profile(gas_scf.wavefunction(), settings);
  const cosmors::Parameters rs = settings.parameters;

  occ::log::info("cavity area          {:12.5f} Angstrom^2", result.cavity_area);
  occ::log::info("cavity volume        {:12.5f} Angstrom^3",
                 result.cavity_volume);
  occ::log::info("segments             {:12d}", result.segments.size());
  occ::log::info("screening charge     {:12.5f} e", result.screening_charge);
  occ::log::info("conductor stabilisation {:9.5f} Hartree",
                 result.energy_conductor - result.energy_gas);

  const auto solute = cosmors::Component::from_segments(
      result.segments, result.cavity_volume, result.cavity_area);

  if (!config.segments_filename.empty()) {
    cosmors::write_segments(config.segments_filename, molecule.name(), solute,
                             result.segments.atomic_number, rs, config.method,
                             config.basis);
    occ::log::info("wrote segment ensemble to {}", config.segments_filename);
  }

  if (config.solvent.empty() && config.solvent_geometry.empty())
    return;

  const int num_rings = (config.num_rings >= 0) ? config.num_rings
                                                : cosmors::ring_count(molecule);

  auto report = [&](const cosmors::Component &solvent,
                    const std::string &label) {
    cosmors::ActivityOptions options;
    options.temperature = config.temperature;
    cosmors::SolventModel model(solvent, rs, options);
    auto energy = cosmors::solvation_free_energy(
        model, solute, result.segments,
        result.energy_conductor - result.energy_gas, num_rings,
        config.liquid_volume, cosmors::SolvationParameters::v24a());

    const double k = occ::units::AU_TO_KJ_PER_MOL;
    occ::log::info("openCOSMO-RS 24a solvation free energy in '{}' at {:.2f} K "
                   "({} ring(s), kJ/mol)",
                   label, config.temperature, num_rings);
    occ::log::info("  E_diel           {:10.3f}", energy.dielectric * k);
    occ::log::info("  residual         {:10.3f}", energy.residual * k);
    occ::log::info("  combinatorial    {:10.3f}", energy.combinatorial * k);
    occ::log::info("  cavity           {:10.3f}", energy.cavity * k);
    occ::log::info("  ring             {:10.3f}", energy.ring * k);
    occ::log::info("  reference state  {:10.3f}", energy.reference_state * k);
    occ::log::info("  constant         {:10.3f}", energy.constant * k);
    occ::log::info("  total            {:10.3f}", energy.total() * k);
  };

  // A cached solvent ensemble avoids recomputing the solvent cavity, which is
  // what makes a solvent screen cheap.
  if (!config.solvent.empty()) {
    auto store = cosmors::SegmentStore::standard();
    auto solvent = store.get(config.solvent);
    if (solvent.r_av > 0.0 && std::abs(solvent.r_av - rs.r_av) > 1e-12)
      throw std::runtime_error(fmt::format(
          "solvent '{}' was averaged on r_av = {} but the parameters in use "
          "specify {}",
          config.solvent, solvent.r_av, rs.r_av));
    if (!solvent.basis.empty() && solvent.basis != config.basis)
      occ::log::warn("solvent '{}' segments were computed with {}/{} but this "
                     "solute uses {}/{}; the descriptors are not comparable",
                     config.solvent, solvent.method, solvent.basis,
                     config.method, config.basis);
    report(solvent.component, config.solvent);
    return;
  }

  // Both cavities built here, so the solvent carries the segment descriptors
  // the kernel needs rather than a stored ensemble.
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
  report(cosmors::Component::from_segments(solvent_conductor.segments,
                                           solvent_conductor.cavity_volume,
                                           solvent_conductor.cavity_area),
         solvent_molecule.name());
}

} // namespace occ::main
