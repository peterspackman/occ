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
    auto segments = conductor_segments(wavefunction, params, 0.0,
                                       settings.angular_points);

    auto scrf_surface = model.solvation_surface(segments);
    cg::SMDSolventSurfaces surfaces;
    surfaces.coulomb.positions = scrf_surface.positions;
    surfaces.coulomb.areas = scrf_surface.areas;
    surfaces.coulomb.energies = scrf_surface.energies;
    // No CDS branch: the sigma potential already carries the non-electrostatic
    // part of the residual term.
    surfaces.cds.positions = Mat3N(3, 0);
    surfaces.cds.areas = Vec(0);
    surfaces.cds.energies = Vec(0);
    surfaces.electronic_energies = Vec::Zero(scrf_surface.areas.size());
    surfaces.total_solvation_energy = scrf_surface.total_energy();

    occ::log::info("  molecule {}: {} segments, residual {:.3f} kJ/mol", i,
                   segments.size(),
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
