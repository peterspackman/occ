#include <filesystem>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/driver/sigma_solvation.h>
#include <occ/solvent/sigma_io.h>

namespace fs = std::filesystem;

namespace occ::driver {

namespace {

qm::Wavefunction conductor_wavefunction(const std::string &cache_path,
                                        const qm::Wavefunction &gas,
                                        const SigmaSolvationSettings &settings) {
  if (fs::exists(cache_path)) {
    occ::log::info("Loading cached conductor wavefunction from {}", cache_path);
    return qm::Wavefunction::load(cache_path);
  }

  SigmaProfileSettings profile_settings;
  profile_settings.method = settings.method;
  profile_settings.basis = settings.basis;
  profile_settings.pure_spherical = settings.pure_spherical;
  profile_settings.angular_points = settings.angular_points;
  profile_settings.model = settings.model;

  auto result = conductor_profile(gas, profile_settings);
  auto wavefunction = result.wavefunction;
  occ::log::info("Writing conductor wavefunction to {}", cache_path);
  wavefunction.save(cache_path);
  return wavefunction;
}

/// gas -> ideal conductor, the electrostatic half of the solvation free
/// energy. Solvent independent by construction.
double result_energy_difference(const qm::Wavefunction &conductor,
                                const qm::Wavefunction &gas) {
  return conductor.energy.total - gas.energy.total;
}

} // namespace

SigmaSolvationResult
sigma_solvation(const std::string &basename,
                const std::vector<core::Molecule> &molecules,
                const std::vector<qm::Wavefunction> &gas_wavefunctions,
                const std::string &solvent,
                const SigmaSolvationSettings &settings) {
  const auto params = solvent::sigma::Parameters::for_model(settings.model);
  solvent::sigma::PotentialOptions options;
  options.temperature = settings.temperature;

  auto store = solvent::sigma::ProfileStore::standard();
  solvent::sigma::SolventModel model(store.get(solvent), params, options);
  occ::log::info("Sigma solvation: {} in '{}', potential converged in {} "
                 "iterations",
                 solvent::sigma::model_name(settings.model), solvent,
                 model.potential().iterations);

  SigmaSolvationResult result;
  const size_t n = gas_wavefunctions.size();
  result.surfaces.reserve(n);
  result.wavefunctions.reserve(n);
  result.reorganisation.reserve(n);
  result.hbond_area.reserve(n);

  for (size_t i = 0; i < n; i++) {
    const auto cache =
        fmt::format("{}_{}_conductor.owf.json", basename, i);
    auto wavefunction =
        conductor_wavefunction(cache, gas_wavefunctions[i], settings);
    Vec dielectric;
    auto segments = conductor_segments(wavefunction, params, 0.0,
                                       settings.angular_points, true,
                                       &dielectric);

    auto scrf_surface = model.solvation_surface(segments);

    // The two branches carry the two halves of the COSMO-RS decomposition,
    // which map onto the shape cg already consumes:
    //   coulomb <- E_diel, gas to ideal conductor. Large, solvent
    //              independent, and the bulk of the magnitude.
    //   cds     <- mu_res, conductor to real solvent. Carries all of the
    //              solvent dependence.
    // Keeping them apart means cg reports them separately, exactly as it does
    // for the SMD electrostatic and CDS terms.
    cg::SMDSolventSurfaces surfaces;
    surfaces.coulomb.positions = scrf_surface.positions;
    surfaces.coulomb.areas = scrf_surface.areas;
    surfaces.coulomb.energies = dielectric;

    surfaces.cds.positions = scrf_surface.positions;
    surfaces.cds.areas = scrf_surface.areas;
    surfaces.cds.energies = scrf_surface.energies;

    // The electronic relaxation cost has no per-element decomposition, so it
    // is spread by area — the same convention the SMD path uses.
    const double e_diel_total =
        result_energy_difference(wavefunction, gas_wavefunctions[i]);
    const double relaxation = e_diel_total - dielectric.sum();
    surfaces.electronic_energies =
        (relaxation / scrf_surface.areas.sum()) * scrf_surface.areas;
    surfaces.electronic_contribution = relaxation;
    surfaces.total_solvation_energy = e_diel_total + scrf_surface.total_energy();

    occ::log::info("  molecule {}: {} segments, E_diel {:.3f} + mu_res {:.3f} "
                   "= {:.3f} kJ/mol",
                   i, segments.size(),
                   e_diel_total * occ::units::AU_TO_KJ_PER_MOL,
                   scrf_surface.total_energy() * occ::units::AU_TO_KJ_PER_MOL,
                   surfaces.total_solvation_energy *
                       occ::units::AU_TO_KJ_PER_MOL);

    result.surfaces.push_back(std::move(surfaces));
    result.wavefunctions.push_back(std::move(wavefunction));
    result.reorganisation.push_back(model.segment_reorganisation(segments));
    result.hbond_area.push_back(model.segment_hbond_area(segments));
  }
  return result;
}

} // namespace occ::driver
