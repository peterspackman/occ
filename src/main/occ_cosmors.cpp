#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/driver/cosmors_driver.h>
#include <occ/dft/dft.h>
#include <occ/io/xyz.h>
#include <occ/main/occ_cosmors.h>
#include <occ/qm/scf.h>
#include <occ/solvent/cosmors_io.h>

namespace occ::main {

namespace {

void report(const occ::driver::CosmoRSSolvation &result,
            const std::string &solvent, double temperature) {
  const auto &e = result.energy;
  const double k = occ::units::AU_TO_KJ_PER_MOL;
  occ::log::info("cavity area          {:12.5f} Angstrom^2", result.cavity_area);
  occ::log::info("cavity volume        {:12.5f} Angstrom^3",
                 result.cavity_volume);
  occ::log::info("openCOSMO-RS 24a solvation free energy in '{}' at {:.2f} K "
                 "({} ring(s), kJ/mol)",
                 solvent, temperature, result.num_rings);
  occ::log::info("  dielectric       {:10.3f}   gas -> ideal conductor",
                 e.dielectric * k);
  occ::log::info("  residual         {:10.3f}   RT ln(gamma_res)",
                 e.residual * k);
  occ::log::info("  combinatorial    {:10.3f}   RT ln(gamma_comb)",
                 e.combinatorial * k);
  occ::log::info("  van der Waals    {:10.3f}   -sum_a tau_a A_a", e.vdw * k);
  occ::log::info("  ring             {:10.3f}   -omega_ring n_ring",
                 e.ring * k);
  occ::log::info("  reference state  {:10.3f}   -RT ln(v_gas/v_liquid)",
                 e.reference_state * k);
  occ::log::info("  eta              {:10.3f}   fitted intercept", e.eta * k);
  occ::log::info("  total            {:10.3f}", e.total() * k);
}

} // namespace

CLI::App *add_cosmors_subcommand(CLI::App &app) {
  CLI::App *cosmors = app.add_subcommand(
      "cosmo-rs", "compute an openCOSMO-RS solvation free energy");
  auto config = std::make_shared<CosmoRSConfig>();

  cosmors->add_option("geometry", config->geometry_filename,
                      "input geometry file (xyz)");
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
                      "liquid-phase volume per solute molecule (Angstrom^3) "
                      "for the reference-state term; omitted when unset");
  cosmors->add_option("--rings", config->num_rings,
                      "rings in the solute for the ring correction (default: "
                      "counted from the bond graph)");
  cosmors->add_option("--probe", config->probe_radius,
                      "solvent probe radius used to build the cavity "
                      "(Angstrom)");
  cosmors->add_option("--temperature", config->temperature, "temperature (K)");
  cosmors->add_option("--angular-points", config->angular_points,
                      "Lebedev order per atom for the cavity");
  cosmors->add_flag("--unconstrained-charge", config->unconstrained_charge,
                    "do not constrain the surface charge to -q");
  cosmors->add_flag("--cartesian", config->cartesian,
                    "use cartesian (6d) basis functions");
  cosmors->add_flag("--list-available-solvents", config->list_solvents,
                    "list solvents with a cached segment ensemble and exit");

  cosmors->fallthrough();
  cosmors->callback([config]() { run_cosmors_subcommand(*config); });
  return cosmors;
}

void run_cosmors_subcommand(CosmoRSConfig const &config) {
  namespace cosmors = occ::solvent::cosmors;

  if (config.list_solvents) {
    for (const auto &name : occ::driver::available_cosmors_solvents())
      occ::log::warn("{}", name);
    return;
  }
  if (config.geometry_filename.empty())
    throw std::runtime_error("a geometry is required");
  if (!config.solvent.empty() && !config.solvent_geometry.empty())
    throw std::runtime_error(
        "--solvent and --solvent-geometry both name a solvent; pick one");

  auto solute = occ::io::molecule_from_xyz_file(config.geometry_filename);

  occ::driver::CosmoRSSolvationSettings settings;
  settings.method = config.method;
  settings.basis = config.basis;
  settings.pure_spherical = !config.cartesian;
  settings.probe_radius_angs = config.probe_radius;
  settings.angular_points = config.angular_points;
  settings.constrain_charge = !config.unconstrained_charge;
  settings.temperature = config.temperature;
  settings.liquid_volume = config.liquid_volume;
  settings.num_rings = config.num_rings;

  // Writing an ensemble needs only the solute's own cavity, so it is worth
  // doing even when no solvent was named.
  if (config.solvent.empty() && config.solvent_geometry.empty() &&
      config.segments_filename.empty())
    throw std::runtime_error(
        "nothing to do: give --solvent, --solvent-geometry or "
        "--write-segments");

  if (!config.solvent.empty()) {
    auto result = occ::driver::cosmors_solvation_free_energy(
        solute, config.solvent, settings);
    report(result, config.solvent, config.temperature);
    if (!config.segments_filename.empty())
      occ::log::warn("--write-segments needs the solute's own ensemble; run "
                     "without a solvent to produce one");
    return;
  }

  if (!config.solvent_geometry.empty()) {
    auto solvent =
        occ::io::molecule_from_xyz_file(config.solvent_geometry);
    auto result =
        occ::driver::cosmors_solvation_free_energy(solute, solvent, settings);
    report(result, solvent.name(), config.temperature);
    return;
  }

  // Segment ensemble only.
  occ::driver::ConductorSettings conductor;
  conductor.method = config.method;
  conductor.basis = config.basis;
  conductor.pure_spherical = !config.cartesian;
  conductor.probe_radius_angs = config.probe_radius;
  conductor.angular_points = config.angular_points;
  conductor.constrain_charge = !config.unconstrained_charge;

  occ::gto::AOBasis basis =
      occ::gto::AOBasis::load(solute.atoms(), config.basis);
  basis.set_pure(!config.cartesian);
  occ::dft::DFT gas(config.method, basis);
  occ::qm::SCF<occ::dft::DFT> gas_scf(gas);
  gas_scf.compute_scf_energy();
  auto result =
      occ::driver::conductor_profile(gas_scf.wavefunction(), conductor);
  auto component = cosmors::Component::from_segments(
      result.segments, result.cavity_volume, result.cavity_area);
  cosmors::write_segments(config.segments_filename, solute.name(), component,
                          conductor.parameters, config.method, config.basis);
  occ::log::info("wrote segment ensemble to {}", config.segments_filename);
}

} // namespace occ::main
