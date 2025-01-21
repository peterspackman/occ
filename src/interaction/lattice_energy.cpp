#include <occ/core/log.h>
#include <occ/core/progress.h>
#include <occ/core/timings.h>
#include <occ/interaction/coulomb.h>
#include <occ/interaction/lattice_energy.h>
#include <occ/interaction/pair_energy_store.h>
#include <occ/interaction/wolf.h>

namespace occ::interaction {

LatticeEnergyCalculator::LatticeEnergyCalculator(
    std::unique_ptr<EnergyModelBase> model, const crystal::Crystal &crystal,
    const std::string &basename, LatticeConvergenceSettings settings)
    : m_energy_model(std::move(model)), m_crystal(crystal),
      m_basename(basename), m_settings(settings) {

  if (m_settings.wolf_sum) {
    initialize_wolf_sum();
  }
}

void LatticeEnergyCalculator::initialize_wolf_sum() {
  WolfParameters params;
  params.cutoff = m_settings.max_radius;
  params.alpha = 0.2; // Default alpha value

  m_wolf_sum = std::make_unique<WolfSum>(params);
  m_wolf_sum->initialize(m_crystal, *m_energy_model);
}

bool LatticeEnergyCalculator::is_converged(double current,
                                           double previous) const {
  return std::abs(current - previous) <= m_settings.energy_tolerance;
}

void LatticeEnergyCalculator::report_convergence_statistics(
    const CEEnergyComponents &total) {

  occ::log::debug("Total polarization term:  {:.3f}",
                  total.polarization_kjmol() * 0.5);
  occ::log::debug("Total coulomb term:       {:.3f}", total.coulomb_kjmol() * 0.5);
  occ::log::debug("Total dispersion term:    {:.3f}", total.dispersion_kjmol() * 0.5);
  occ::log::debug("Total repulsion term:     {:.3f}", total.repulsion_kjmol() * 0.5);
  occ::log::debug("Total exchange term:      {:.3f}", total.exchange_kjmol() * 0.5);
}

CEEnergyComponents
compute_cycle_energy(const std::vector<CEEnergyComponents> &energies,
                     const crystal::CrystalDimers &all_dimers) {
  CEEnergyComponents total;
  const auto &mol_neighbors = all_dimers.molecule_neighbors;

  // Loop over all molecules and their neighbors
  for (const auto &n : mol_neighbors) {
    for (const auto &[dimer, unique_idx] : n) {
      const auto &e = energies[unique_idx];
      if (e.is_computed) {
        total += e;
      }
    }
  }
  return total;
}

LatticeEnergyResult LatticeEnergyCalculator::compute() {
  occ::timing::StopWatch sw;
  double lattice_energy = 0.0;
  double previous_lattice_energy = 0.0;
  double current_radius =
      std::max(m_settings.radius_increment, m_settings.min_radius);
  size_t cycle = 1;

  auto all_dimers = m_crystal.symmetry_unique_dimers(m_settings.max_radius);
  const auto &dimers = all_dimers.unique_dimers;
  std::vector<CEEnergyComponents> converged_energies(dimers.size());
  std::vector<double> charge_energies(dimers.size());

  occ::log::info("Found {} symmetry unique dimers within max radius {:.3f}",
                 dimers.size(), m_settings.max_radius);

  do {
    previous_lattice_energy = lattice_energy;

    // Create energy store for this cycle
    PairEnergyStore store{PairEnergyStore::Kind::XYZ,
                          fmt::format("{}_dimers", m_basename)};

    // Setup progress tracking
    size_t dimers_to_compute = 0;
    for (size_t i = 0; i < dimers.size(); i++) {
      if (dimers[i].nearest_distance() <= current_radius &&
          !converged_energies[i].is_computed) {
        dimers_to_compute++;
      }
    }
    occ::core::ProgressTracker progress(dimers_to_compute);

    // Compute energies for this radius
    size_t current_dimer = 0;
    size_t computed_dimers = 0;

    for (const auto &dimer : dimers) {
      sw.start();
      const auto &a = dimer.a();
      const auto &b = dimer.b();
      std::string dimer_name = dimer.name();

      if (dimer.nearest_distance() <= current_radius &&
          !converged_energies[current_dimer].is_computed) {

        if (!store.load(current_dimer, dimer,
                        converged_energies[current_dimer])) {
          // Compute new energy
          converged_energies[current_dimer] =
              m_energy_model->compute_energy(dimer);
          store.save(current_dimer, dimer, converged_energies[current_dimer]);

          if (m_settings.wolf_sum) {
            charge_energies[current_dimer] =
                coulomb_interaction_energy_asym_charges(
                    dimer, m_wolf_sum->asymmetric_charges());
          }
        }

        progress.update(computed_dimers, dimers_to_compute,
                        fmt::format("E[{}|{}]: {}", a.asymmetric_molecule_idx(),
                                    b.asymmetric_molecule_idx(), dimer_name));
        computed_dimers++;
      }
      current_dimer++;
      sw.stop();
    }

    CEEnergyComponents total =
        compute_cycle_energy(converged_energies, all_dimers);

    lattice_energy = 0.5 * total.total_kjmol();
    if (m_settings.wolf_sum) {
      double wolf_correction =
          m_wolf_sum->compute_correction(charge_energies, converged_energies);
      lattice_energy =
          m_energy_model->coulomb_scale_factor() * wolf_correction +
          0.5 * total.total_kjmol();
    }

    report_convergence_statistics(total);
    occ::log::info("Cycle {} lattice energy: {}", cycle, lattice_energy);

    cycle++;
    current_radius += m_settings.radius_increment;
  } while (!is_converged(lattice_energy, previous_lattice_energy));

  // Prepare final result
  for (size_t i = 0; i < converged_energies.size(); i++) {
    const auto &e = converged_energies[i];
    auto &dimer = all_dimers.unique_dimers[i];

    if (e.is_computed) {
      dimer.set_interaction_energy(e.coulomb_kjmol(), "Coulomb");
      dimer.set_interaction_energy(e.exchange_kjmol(), "Exchange");
      dimer.set_interaction_energy(e.repulsion_kjmol(), "Repulsion");
      dimer.set_interaction_energy(e.dispersion_kjmol(), "Dispersion");
      dimer.set_interaction_energy(e.polarization_kjmol(), "Polarization");
      dimer.set_interaction_energy(e.total_kjmol(), "Total");
    }
  }

  return {lattice_energy, all_dimers, converged_energies};
}

} // namespace occ::interaction
