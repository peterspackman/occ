#include <filesystem>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/driver/cosmors_solvation.h>
#include <occ/solvent/cosmors.h>
#include <occ/solvent/cosmors_io.h>

namespace fs = std::filesystem;

namespace occ::driver {

namespace {

qm::Wavefunction conductor_wavefunction(const std::string &cache_path,
                                        const qm::Wavefunction &gas,
                                        const CosmoRSSettings &settings) {
  if (fs::exists(cache_path)) {
    occ::log::info("Loading cached conductor wavefunction from {}", cache_path);
    return qm::Wavefunction::load(cache_path);
  }

  ConductorSettings profile_settings;
  profile_settings.method = settings.method;
  profile_settings.basis = settings.basis;
  profile_settings.pure_spherical = settings.pure_spherical;
  profile_settings.angular_points = settings.angular_points;

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

CGSolvationResult
cosmors_solvation(const std::string &basename,
                      const std::vector<core::Molecule> &molecules,
                      const std::vector<qm::Wavefunction> &gas_wavefunctions,
                      const SolventSpec &solvent,
                      const CosmoRSSettings &settings) {
  const solvent::cosmors::Parameters rs;
  solvent::cosmors::ActivityOptions options;
  options.temperature = settings.temperature;

  auto store = solvent::cosmors::SegmentStore::standard();
  std::vector<solvent::cosmors::Component> components;
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
          ? solvent::cosmors::mix_components(components, solvent.mole_fractions)
          : components.front();

  solvent::cosmors::SolventModel model(std::move(mixture), rs, options);
  const auto solvation_params =
      solvent::cosmors::SolvationParameters::v24a();
  occ::log::info("openCOSMO-RS 24a solvation: '{}' at {:.2f} K",
                 solvent.to_string(), options.temperature);
  // The surface-additive channels (dielectric, residual, cavity) carry the
  // per-contact decomposition the facet energies are built from. The
  // combinatorial, ring, reference-state and eta terms are per-molecule: they
  // cancel in attachment-energy differences, so they stay out of the
  // per-segment channels, but the solution thermodynamics need them and they
  // are added to the molecular total below.

  CGSolvationResult result;
  const size_t n = gas_wavefunctions.size();
  result.surfaces.reserve(n);
  result.wavefunctions.reserve(n);

  for (size_t i = 0; i < n; i++) {
    const auto cache = fmt::format("{}_{}_conductor.owf.json", basename, i);
    auto wavefunction =
        conductor_wavefunction(cache, gas_wavefunctions[i], settings);
    Vec dielectric;
    double cavity_volume = 0.0;
    auto segments =
        conductor_segments(wavefunction, rs, 0.0, settings.angular_points,
                           true, &dielectric, &cavity_volume);

    auto solute = solvent::cosmors::Component::from_segments(
        segments, cavity_volume, segments.total_area());

    const Vec residual = model.segment_energies(solute);
    const Vec cavity =
        solvent::cosmors::segment_cavity_energies(segments, solvation_params);
    const auto missing = solvent::cosmors::unparameterised_elements(
        segments, solvation_params);
    if (!missing.empty())
      occ::log::warn("molecule {}: {} element(s) have no openCOSMO-RS cavity "
                     "parameter and contribute nothing to that channel",
                     i, missing.size());

    const double e_diel =
        result_energy_difference(wavefunction, gas_wavefunctions[i]);
    const double relaxation = e_diel - dielectric.sum();

    // Per-molecule terms. The cycle rank of the bond graph counts rings for
    // a connected molecule; without bonds it is zero and the ring term drops
    // out, as it should for an acyclic solute.
    const auto &mol = molecules[i];
    const int num_rings =
        mol.bonds().empty()
            ? 0
            : std::max<int>(0, static_cast<int>(mol.bonds().size()) -
                                   static_cast<int>(mol.size()) + 1);
    const auto molecular = solvent::cosmors::solvation_free_energy(
        model, solute, segments, e_diel, num_rings,
        settings.volume_per_molecule, solvation_params);

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
    // The per-segment channels cover dielectric, residual and cavity; take
    // the rest of the model's total from the molecular assembly so the
    // reported solvation free energy is the whole of it.
    const double per_molecule = molecular.combinatorial + molecular.ring +
                                molecular.reference_state + molecular.constant;
    surfaces.total_solvation_energy = surfaces.total_energy() + per_molecule;

    const double k = occ::units::AU_TO_KJ_PER_MOL;
    occ::log::info("  molecule {}: {} segments, {} ring(s), cavity {:.1f} A^3",
                   i, segments.size(), num_rings, cavity_volume);
    occ::log::info("    E_diel {:8.3f}  residual {:8.3f}  cavity {:8.3f}",
                   e_diel * k, residual.sum() * k, cavity.sum() * k);
    occ::log::info("    comb   {:8.3f}  ring     {:8.3f}  ref.st {:8.3f}  "
                   "eta {:8.3f}",
                   molecular.combinatorial * k, molecular.ring * k,
                   molecular.reference_state * k, molecular.constant * k);
    occ::log::info("    total  {:8.3f} kJ/mol",
                   surfaces.total_solvation_energy * k);

    result.surfaces.push_back(std::move(surfaces));
    result.wavefunctions.push_back(std::move(wavefunction));
  }
  return result;
}

} // namespace occ::driver
