#include <filesystem>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/driver/sigma_solvation.h>
#include <occ/solvent/opencosmors.h>
#include <occ/solvent/opencosmors_io.h>
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

    // Measurement only for now: the openCOSMO-RS cavity term is not yet part
    // of the total, since its parameters were fitted against that model's
    // interaction kernel rather than the COSMO-SAC one used here.
    {
      const auto solvation_params =
          solvent::sigma::SolvationParameters::opencosmors_24a();
      const Vec cavity_energies =
          solvent::sigma::segment_cavity_energies(segments, solvation_params);
      const auto missing = solvent::sigma::unparameterised_elements(
          segments, solvation_params);
      occ::log::info("  openCOSMO-RS cavity term {:.3f} kJ/mol over {:.1f} A^2",
                     cavity_energies.sum() * occ::units::AU_TO_KJ_PER_MOL,
                     segments.total_area());
      for (const auto &[z, area] : solvent::sigma::area_per_element(segments))
        occ::log::info("    element {:3d}: {:8.2f} A^2", z, area);
      if (!missing.empty())
        occ::log::warn("    {} element(s) outside the openCOSMO-RS 24a set",
                       missing.size());
    }

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

CGSolvationResult
opencosmors_solvation(const std::string &basename,
                      const std::vector<core::Molecule> &molecules,
                      const std::vector<qm::Wavefunction> &gas_wavefunctions,
                      const SolventSpec &solvent,
                      const SigmaSolvationSettings &settings) {
  const solvent::sigma::RSParameters rs;
  solvent::sigma::RSOptions options;
  options.temperature = settings.temperature;

  auto store = solvent::sigma::RSProfileStore::standard();
  std::vector<solvent::sigma::RSComponent> components;
  components.reserve(solvent.components.size());
  for (const auto &name : solvent.components) {
    auto file = store.get(name);
    if (!file.basis.empty() && file.basis != settings.basis)
      occ::log::warn("solvent '{}' segments were computed with {}/{} but this "
                     "run uses {}/{}; the descriptors are not comparable",
                     name, file.method, file.basis, settings.method,
                     settings.basis);
    components.push_back(std::move(file.component));
  }
  auto mixture =
      solvent.is_mixture()
          ? solvent::sigma::mix_rs_components(components, solvent.mole_fractions)
          : components.front();

  solvent::sigma::RSSolventModel model(std::move(mixture), rs, options);
  const auto solvation_params =
      solvent::sigma::SolvationParameters::opencosmors_24a();
  occ::log::info("openCOSMO-RS 24a solvation: '{}' at {:.2f} K",
                 solvent.to_string(), options.temperature);
  // Only the surface-additive channels are produced here. The ring,
  // reference-state and eta terms of the openCOSMO-RS solvation free energy
  // are per-molecule; they cancel in the attachment-energy differences the
  // facet energies are built from, but they are absent from the per-molecule
  // total the solution thermodynamics use, which instead receives the
  // pipeline's own 1.89 kcal/mol concentration shift.
  occ::log::warn("openCOSMO-RS: facet energies are complete, but the reported "
                 "dG solution and log S omit this model's ring, reference "
                 "state and eta terms; use 'occ sigma --opencosmors' for an "
                 "absolute solvation free energy");

  // The averaging convention the segments arrive with belongs to COSMO-SAC;
  // the RS descriptors are rebuilt on their own radii below.
  const auto sac_params = solvent::sigma::Parameters::for_model(settings.model);

  CGSolvationResult result;
  const size_t n = gas_wavefunctions.size();
  result.surfaces.reserve(n);
  result.wavefunctions.reserve(n);

  for (size_t i = 0; i < n; i++) {
    const auto cache = fmt::format("{}_{}_conductor.owf.json", basename, i);
    auto wavefunction =
        conductor_wavefunction(cache, gas_wavefunctions[i], settings);
    Vec dielectric;
    auto segments =
        conductor_segments(wavefunction, sac_params, 0.0,
                           settings.angular_points, true, &dielectric);
    solvent::sigma::average_sigma(segments, rs.r_av, 1.0);
    solvent::sigma::average_sigma_orth(segments, rs.r_av, rs.r_corr,
                                       rs.sigma_orth_factor);

    // Volume and cavity area are left unset: they feed only the combinatorial
    // and reference-state terms, which are per-molecule and omitted here.
    auto solute = solvent::sigma::RSComponent::from_segments(segments, 0.0);

    const Vec residual = model.segment_energies(solute);
    const Vec cavity =
        solvent::sigma::segment_cavity_energies(segments, solvation_params);
    const auto missing = solvent::sigma::unparameterised_elements(
        segments, solvation_params);
    if (!missing.empty())
      occ::log::warn("molecule {}: {} element(s) have no openCOSMO-RS cavity "
                     "parameter and contribute nothing to that channel",
                     i, missing.size());

    const double e_diel =
        result_energy_difference(wavefunction, gas_wavefunctions[i]);
    const double relaxation = e_diel - dielectric.sum();

    cg::SolvationData surfaces;
    cg::CavitySurface surface;
    surface.name = "conductor";
    surface.positions = segments.positions;
    surface.areas = segments.areas;
    surface.energies.push_back({"dielectric", dielectric});
    surface.energies.push_back({"residual", residual});
    surface.energies.push_back({"cavity", cavity});
    // No per-element decomposition for the relaxation; spread it by area.
    surface.energies.push_back(
        {"electronic", (relaxation / segments.areas.sum()) * segments.areas});
    surfaces.cavities.push_back(std::move(surface));

    surfaces.electronic_contribution = relaxation;
    surfaces.total_solvation_energy = surfaces.total_energy();

    occ::log::info("  molecule {}: {} segments, E_diel {:.3f} + residual "
                   "{:.3f} + cavity {:.3f} = {:.3f} kJ/mol",
                   i, segments.size(),
                   e_diel * occ::units::AU_TO_KJ_PER_MOL,
                   residual.sum() * occ::units::AU_TO_KJ_PER_MOL,
                   cavity.sum() * occ::units::AU_TO_KJ_PER_MOL,
                   surfaces.total_solvation_energy *
                       occ::units::AU_TO_KJ_PER_MOL);

    result.surfaces.push_back(std::move(surfaces));
    result.wavefunctions.push_back(std::move(wavefunction));
  }
  return result;
}

} // namespace occ::driver
