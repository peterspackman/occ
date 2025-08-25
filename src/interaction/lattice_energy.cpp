#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/progress.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/coulomb.h>
#include <occ/interaction/lattice_energy.h>
#include <occ/interaction/pair_energy_store.h>
#include <occ/interaction/polarization_partitioning.h>
#include <occ/interaction/wolf.h>
#include <atomic>
#include <mutex>

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
  occ::log::debug("Total coulomb term:       {:.3f}",
                  total.coulomb_kjmol() * 0.5);
  occ::log::debug("Total dispersion term:    {:.3f}",
                  total.dispersion_kjmol() * 0.5);
  occ::log::debug("Total repulsion term:     {:.3f}",
                  total.repulsion_kjmol() * 0.5);
  occ::log::debug("Total exchange term:      {:.3f}",
                  total.exchange_kjmol() * 0.5);
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

    // Create a vector of dimers that need computation for parallel processing
    struct DimerTask {
      size_t index;
      const occ::core::Dimer* dimer;
    };
    
    std::vector<DimerTask> tasks_to_compute;
    for (size_t i = 0; i < dimers.size(); i++) {
      if (dimers[i].nearest_distance() <= current_radius &&
          !converged_energies[i].is_computed) {
        tasks_to_compute.push_back({i, &dimers[i]});
      }
    }
    
    // Parallel computation of energies with real-time progress updates using TBB
    if (!tasks_to_compute.empty()) {
      std::atomic<size_t> computed_dimers{0};
      std::mutex progress_mutex;
      
      occ::parallel::parallel_for(size_t(0), tasks_to_compute.size(), [&](size_t task_idx) {
        const auto& task = tasks_to_compute[task_idx];
        size_t dimer_idx = task.index;
        const auto& dimer = *task.dimer;
        
        if (!store.load(dimer_idx, dimer, converged_energies[dimer_idx])) {
          // Compute new energy - TBB handles nested parallelism automatically
          converged_energies[dimer_idx] = m_energy_model->compute_energy(dimer);
          // Ensure total energy is properly computed with model scaling factors
          m_energy_model->compute_total_energy(converged_energies[dimer_idx]);
          store.save(dimer_idx, dimer, converged_energies[dimer_idx]);
          
          if (m_settings.wolf_sum) {
            charge_energies[dimer_idx] =
                coulomb_interaction_energy_asym_charges(
                    dimer, m_wolf_sum->asymmetric_charges());
          }
        }
        
        // Update progress with mutex protection
        {
          std::lock_guard<std::mutex> lock(progress_mutex);
          size_t current_progress = computed_dimers.fetch_add(1) + 1;
          const auto &a = dimer.a();
          const auto &b = dimer.b();
          std::string dimer_name = dimer.name();
          progress.update(current_progress, dimers_to_compute,
                          fmt::format("E[{}|{}]: {}", a.asymmetric_molecule_idx(),
                                      b.asymmetric_molecule_idx(), dimer_name));
        }
      });
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

  // Apply crystal field polarization partitioning if enabled and recalculate lattice energy
  double final_lattice_energy = lattice_energy;
  if (m_settings.crystal_field_polarization) {
    double converged_radius = current_radius - m_settings.radius_increment;
    apply_crystal_field_polarization_partitioning(all_dimers, converged_energies, converged_radius);
    
    // Recalculate lattice energy with updated total energies
    CEEnergyComponents updated_total = compute_cycle_energy(converged_energies, all_dimers);
    final_lattice_energy = 0.5 * updated_total.total_kjmol();
    
    occ::log::info("Updated lattice energy after crystal field partitioning: {:.3f} kJ/mol", final_lattice_energy);
  }

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

  return {final_lattice_energy, all_dimers, converged_energies};
}

namespace {

/**
 * @brief Calculate total polarization energy from computed dimer energies
 */
double calculate_total_polarization_energy(const std::vector<CEEnergyComponents> &energies) {
  double total = 0.0;
  for (const auto& e : energies) {
    if (e.is_computed) {
      total += e.polarization_kjmol();
    }
  }
  return total;
}

/**
 * @brief Initialize total electric fields for each unit cell molecule
 */
std::vector<Mat3N> initialize_total_fields(const std::vector<core::Molecule> &molecules) {
  std::vector<Mat3N> fields;
  fields.reserve(molecules.size());
  
  for (const auto &mol : molecules) {
    fields.emplace_back(Mat3N::Zero(3, mol.size()));
  }
  return fields;
}

/**
 * @brief Apply gradient-based partitioning using stored fields and contributions
 */
void apply_gradient_partitioning(
    crystal::CrystalDimers &dimers,
    std::vector<CEEnergyComponents> &energies,
    const std::vector<core::Molecule> &unique_molecules,
    const std::vector<Mat3N> &total_fields,
    const std::vector<std::vector<double>> &pair_contributions,
    CEEnergyModel *ce_model) {

  size_t partitioned_count = 0;
  
  for (size_t i = 0; i < dimers.unique_dimers.size(); ++i) {
    if (!energies[i].is_computed) continue;
    
    const auto &dimer = dimers.unique_dimers[i];
    size_t mol_a_type = dimer.a().asymmetric_molecule_idx();
    size_t mol_b_type = dimer.b().asymmetric_molecule_idx();
    
    if (mol_a_type >= total_fields.size() || mol_b_type >= total_fields.size()) {
      continue;
    }
    
    try {
      // Get the asymmetric unit total fields
      const Mat3N& asym_field_a = total_fields[mol_a_type];
      const Mat3N& asym_field_b = total_fields[mol_b_type];
      
      // Transform the total fields based on the molecules' orientations
      // The fields need to be rotated to match the actual molecule orientations
      auto [rot_a, trans_a] = dimer.a().asymmetric_unit_transformation();
      auto [rot_b, trans_b] = dimer.b().asymmetric_unit_transformation();
      
      // Apply rotation to the fields (electric fields transform as vectors)
      Mat3N total_field_a = rot_a * asym_field_a;
      Mat3N total_field_b = rot_b * asym_field_b;
      
      // Get pair fields: A <- B and B <- A
      Mat3N pair_field_a = ce_model->compute_electric_field(dimer);  // Field at A from B
      core::Dimer reversed_dimer(dimer.b(), dimer.a());
      Mat3N pair_field_b = ce_model->compute_electric_field(reversed_dimer);  // Field at B from A
      
      // Apply gradient-based partitioning
      auto contribution = polarization_partitioning::partition_crystal_polarization_energy(
        dimer, total_field_a, total_field_b, pair_field_a, pair_field_b,
        ce_model->get_polarizabilities(dimer.a()), ce_model->get_polarizabilities(dimer.b())
      );
      
      // Replace the pairwise polarization energy with partitioned energy
      double old_pol_kjmol = energies[i].polarization_kjmol();
      energies[i].polarization = contribution.total_energy;
      double new_pol_kjmol = energies[i].polarization_kjmol();
      
      occ::log::debug("Partitioned dimer {} polarization: {:.6f} -> {:.6f} kJ/mol",
                      i, old_pol_kjmol, new_pol_kjmol);
      
      partitioned_count++;
    } catch (const std::exception &e) {
      occ::log::warn("Failed to partition polarization for dimer {}: {}", i, e.what());
    }
  }
  
  occ::log::info("Successfully partitioned polarization energy for {} dimers", partitioned_count);
}

} // anonymous namespace

void LatticeEnergyCalculator::apply_crystal_field_polarization_partitioning(
    crystal::CrystalDimers &dimers, 
    std::vector<CEEnergyComponents> &energies,
    double radius) {
  
  occ::log::info("Computing crystal field polarization with radius {:.3f}", radius);
  
  // Validate that we have a CE energy model
  auto *ce_model = dynamic_cast<CEEnergyModel*>(m_energy_model.get());
  if (!ce_model) {
    occ::log::warn("Crystal field polarization requires CE energy model, skipping");
    return;
  }

  // Report original polarization energy breakdown
  const double total_original_pol = calculate_total_polarization_energy(energies);
  occ::log::info("Total original polarization energy: {:.6f} kJ/mol", total_original_pol);

  occ::log::info("Using converged radius {:.3f} Å for crystal field calculation", radius);
  
  // Data structures to store field contributions for partitioning
  const auto &unique_molecules = m_crystal.symmetry_unique_molecules();
  std::vector<Mat3N> total_fields(unique_molecules.size());
  std::vector<double> crystal_pol_energies(unique_molecules.size());
  std::vector<std::vector<double>> pair_contributions(unique_molecules.size());
  
  // Initialize storage
  for (size_t mol_type = 0; mol_type < unique_molecules.size(); ++mol_type) {
    const auto &molecule = unique_molecules[mol_type];
    total_fields[mol_type] = Mat3N::Zero(3, molecule.size());
    pair_contributions[mol_type].resize(dimers.unique_dimers.size(), 0.0);
  }
  
  // For each symmetry unique molecule, do crystal field analysis and store contributions
  for (size_t mol_type = 0; mol_type < unique_molecules.size(); ++mol_type) {
    const auto &central_molecule = unique_molecules[mol_type];
    
    occ::log::info("\n=== Molecule type {} analysis ===", mol_type);
    
    // a) Sum up pairwise polarization energies for this molecule type
    double pairwise_pol_sum = 0.0;
    size_t pair_count = 0;
    
    // Sum over all neighbors (not just unique) to match crystal field calculation
    const auto &neighbors = dimers.molecule_neighbors[mol_type];
    for (const auto &[neighbor_dimer, unique_idx] : neighbors) {
      if (energies[unique_idx].is_computed) {
        // Each neighbor contributes half its pairwise polarization energy to this molecule
        pairwise_pol_sum += 0.5 * energies[unique_idx].polarization_kjmol();
        pair_count++;
      }
    }
    
    occ::log::info("Pairwise polarization sum (ALL {} neighbors): {:.6f} kJ/mol", pair_count, pairwise_pol_sum);
    
    // b) Accumulate crystal field from all neighbors
    occ::log::info("Accumulating fields for molecule type {}", mol_type);
    int neighbor_idx = 0;
    double sum_individual_pol = 0.0;
    for (const auto &[neighbor_dimer, unique_idx] : neighbors) {
      if (!energies[unique_idx].is_computed) continue;
      
      // The neighbor_dimer has the central molecule as a() and neighbor as b()
      // Check what molecule a() actually is
      const auto &mol_a = neighbor_dimer.a();
      occ::log::info("  Neighbor {} mol_a: asym_idx={}, unit_cell_idx={}", 
                     neighbor_idx, mol_a.asymmetric_molecule_idx(), mol_a.unit_cell_molecule_idx());
      
      Mat3N neighbor_field = ce_model->compute_electric_field(neighbor_dimer);
      
      // Debug: print field contribution from this neighbor
      auto field_norms = neighbor_field.colwise().norm();
      double max_field = field_norms.maxCoeff();
      double avg_field = field_norms.mean();
      
      // Compute polarization from this single neighbor's field alone
      double single_neighbor_pol_energy = ce_model->compute_crystal_field_polarization_energy(central_molecule, neighbor_field);
      double single_neighbor_pol_kjmol = single_neighbor_pol_energy * occ::units::AU_TO_KJ_PER_MOL;
      sum_individual_pol += single_neighbor_pol_kjmol;
      
      occ::log::info("  Neighbor {}: unique_idx={}, max_field={:.4f} au, avg_field={:.4f} au, single_neighbor_pol={:.4f} kJ/mol",
                      neighbor_idx, unique_idx, max_field, avg_field, single_neighbor_pol_kjmol);
      
      total_fields[mol_type] += neighbor_field;
      neighbor_idx++;
    }
    
    occ::log::info("Sum of single-neighbor polarizations (ALL {} neighbors): {:.6f} kJ/mol", neighbor_idx, sum_individual_pol);
    
    // Debug: report field magnitudes
    auto field_norms = total_fields[mol_type].colwise().norm();
    double max_field = field_norms.maxCoeff();
    double avg_field = field_norms.mean();
    
    size_t neighbor_count = 0;
    for (const auto &[neighbor_dimer, unique_idx] : neighbors) {
      if (energies[unique_idx].is_computed) neighbor_count++;
    }
    
    occ::log::info("Combined field stats: max={:.4f} au, avg={:.4f} au (from {} neighbors)", 
                   max_field, avg_field, neighbor_count);
    
    // c) Compute crystal field polarization energy
    double crystal_pol_energy = ce_model->compute_crystal_field_polarization_energy(central_molecule, total_fields[mol_type]);
    crystal_pol_energies[mol_type] = crystal_pol_energy;
    double crystal_pol_kjmol = crystal_pol_energy * occ::units::AU_TO_KJ_PER_MOL;
    
    occ::log::info("Combined crystal field polarization: {:.6f} kJ/mol", crystal_pol_kjmol);
    occ::log::info("Difference (combined - sum_single): {:.6f} kJ/mol", crystal_pol_kjmol - sum_individual_pol);
    occ::log::info("Difference (combined - pairwise_sum): {:.6f} kJ/mol", crystal_pol_kjmol - pairwise_pol_sum);
  }
  
  // Now apply gradient-based partitioning using stored contributions
  occ::log::info("\n=== Applying gradient-based partitioning ===");
  apply_gradient_partitioning(dimers, energies, unique_molecules, total_fields, pair_contributions, ce_model);
  
  // Recalculate total energies for all computed dimers using the energy model's scale factors
  for (auto& e : energies) {
    if (e.is_computed) {
      m_energy_model->compute_total_energy(e);
    }
  }
  
  // Report the total partitioned polarization energy
  // Sum over all neighbors (not just unique dimers) to get the correct total
  double total_partitioned_pol = 0.0;
  for (size_t mol_type = 0; mol_type < unique_molecules.size(); ++mol_type) {
    const auto &neighbors = dimers.molecule_neighbors[mol_type];
    for (const auto &[neighbor_dimer, unique_idx] : neighbors) {
      if (energies[unique_idx].is_computed) {
        // Each neighbor contributes half its partitioned energy to avoid double counting
        total_partitioned_pol += 0.5 * energies[unique_idx].polarization_kjmol();
      }
    }
  }
  
  // Also calculate what the final table will show (sum of all pairs with 0.5 factor)
  double table_sum = 0.0;
  for (size_t mol_type = 0; mol_type < unique_molecules.size(); ++mol_type) {
    const auto &neighbors = dimers.molecule_neighbors[mol_type];
    for (const auto &[neighbor_dimer, unique_idx] : neighbors) {
      if (energies[unique_idx].is_computed) {
        table_sum += energies[unique_idx].polarization_kjmol();
      }
    }
  }
  
  double total_crystal_field_pol = 0.0;
  for (double energy : crystal_pol_energies) {
    total_crystal_field_pol += energy * occ::units::AU_TO_KJ_PER_MOL;
  }
  
  occ::log::info("Total partitioned polarization energy (from neighbors): {:.6f} kJ/mol", total_partitioned_pol);
  occ::log::info("Total from final table (0.5 * sum): 0.5 * {:.6f} = {:.6f} kJ/mol", table_sum, 0.5 * table_sum);
  occ::log::info("Total crystal field polarization energy: {:.6f} kJ/mol", total_crystal_field_pol);
  occ::log::info("Conservation check: crystal field ({:.6f}) vs partitioned ({:.6f}) difference: {:.6f} kJ/mol", 
                 total_crystal_field_pol, total_partitioned_pol, total_partitioned_pol - total_crystal_field_pol);
  
  occ::log::info("Crystal field polarization computation completed for {} molecule types", unique_molecules.size());
}

} // namespace occ::interaction
