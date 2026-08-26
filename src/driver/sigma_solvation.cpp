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
                const SolventSpec &solvent,
                const SigmaSolvationSettings &settings) {
  const auto params = solvent::sigma::Parameters::for_model(settings.model);
  solvent::sigma::PotentialOptions options;
  options.temperature = settings.temperature;

  auto store = solvent::sigma::ProfileStore::standard();
  std::vector<solvent::sigma::Component> components;
  components.reserve(solvent.components.size());
  for (const auto &name : solvent.components)
    components.push_back(store.get(name));
  auto mixture = solvent.is_mixture()
                     ? solvent::sigma::mix_components(components,
                                                      solvent.mole_fractions)
                     : components.front();

  solvent::sigma::SolventModel model(std::move(mixture), params, options);
  occ::log::info("Sigma solvation: {} in '{}' at {:.2f} K, potential "
                 "converged in {} iterations",
                 solvent::sigma::model_name(settings.model),
                 solvent.to_string(), options.temperature,
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

    // One cavity, carrying both halves of the COSMO-RS decomposition as
    // separate channels plus the per-contact descriptors:
    //   dielectric <- gas to ideal conductor. Large, solvent independent.
    //   residual   <- conductor to real solvent. All of the solvent
    //                 dependence lives here.
    const double e_diel_total =
        result_energy_difference(wavefunction, gas_wavefunctions[i]);
    const double relaxation = e_diel_total - dielectric.sum();

    cg::SolvationData surfaces;
    cg::CavitySurface cavity;
    cavity.name = "conductor";
    cavity.positions = scrf_surface.positions;
    cavity.areas = scrf_surface.areas;
    cavity.energies.push_back({"dielectric", dielectric});
    cavity.energies.push_back({"residual", scrf_surface.energies});
    // No per-element decomposition for the relaxation; spread it by area.
    cavity.energies.push_back(
        {"electronic", (relaxation / scrf_surface.areas.sum()) *
                           scrf_surface.areas});

    Vec reorganisation = model.segment_reorganisation(segments);
    Vec hbond_area = model.segment_hbond_area(segments);
    // Reported in kJ/mol to sit alongside the cg energies; hbond_area stays
    // in Angstrom^2.
    cavity.descriptors.push_back(
        {"reorganisation", reorganisation * occ::units::AU_TO_KJ_PER_MOL});
    cavity.descriptors.push_back({"hbond_area", hbond_area});
    surfaces.cavities.push_back(std::move(cavity));

    surfaces.electronic_contribution = relaxation;
    surfaces.total_solvation_energy = surfaces.total_energy();

    occ::log::info("  molecule {}: {} segments, E_diel {:.3f} + mu_res {:.3f} "
                   "= {:.3f} kJ/mol",
                   i, segments.size(),
                   e_diel_total * occ::units::AU_TO_KJ_PER_MOL,
                   scrf_surface.total_energy() * occ::units::AU_TO_KJ_PER_MOL,
                   surfaces.total_solvation_energy *
                       occ::units::AU_TO_KJ_PER_MOL);

    result.surfaces.push_back(std::move(surfaces));
    result.wavefunctions.push_back(std::move(wavefunction));
    result.reorganisation.push_back(std::move(reorganisation));
    result.hbond_area.push_back(std::move(hbond_area));
  }
  return result;
}

} // namespace occ::driver
