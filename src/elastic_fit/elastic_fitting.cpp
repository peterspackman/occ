#include <ankerl/unordered_dense.h>
#include <fmt/os.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/elastic_fit/elastic_fitting.h>

using occ::units::degrees;

namespace occ::elastic_fit {

// FittingSettings conversion moved to main module to avoid circular
// dependencies

ElasticFitter::ElasticFitter(const FittingSettings &settings)
    : m_settings(settings) {}

ElasticFitInput ElasticFitter::convert_elat_to_input(
    const occ::interaction::ElatResults &elat_data) {
  ElasticFitInput input;

  // Copy basic crystal info
  const auto &uc = elat_data.crystal.unit_cell();
  input.lattice_vectors = uc.direct();
  input.volume = elat_data.crystal.volume();
  input.title = elat_data.title;
  input.model = elat_data.model;

  // Copy molecule data
  const auto &uc_mols = elat_data.crystal.unit_cell_molecules();
  input.molecules.reserve(uc_mols.size());
  for (const auto &mol : uc_mols) {
    MoleculeInput mol_input;
    mol_input.id = mol.unit_cell_molecule_idx();
    mol_input.mass = mol.molar_mass() * 1000.0;  // Convert kg/mol to g/mol
    mol_input.center_of_mass = mol.center_of_mass();
    input.molecules.push_back(mol_input);
  }

  // Generate unit cell dimers and map energies
  const auto &symm_dimers = elat_data.lattice_energy_result.dimers;

  // Determine radius from symmetry-unique dimers
  double max_radius = 0.0;
  for (const auto &dimer : symm_dimers.unique_dimers) {
    double radius = dimer.nearest_distance();
    if (radius > max_radius) {
      max_radius = radius;
    }
  }

  occ::log::info("Converting elat data: using radius {:.3f} A", max_radius);

  // Generate unit cell dimers
  auto uc_dimers = elat_data.crystal.unit_cell_dimers(max_radius);

  // Map energies from symmetry-unique dimers to unit cell dimers
  for (size_t i = 0; i < uc_dimers.unique_dimers.size(); i++) {
    if (i < symm_dimers.unique_dimers.size()) {
      double energy = symm_dimers.unique_dimers[i].interaction_energy();
      uc_dimers.unique_dimers[i].set_interaction_energy(energy);
      uc_dimers.unique_dimers[i].set_interaction_energies(
          symm_dimers.unique_dimers[i].interaction_energies());
    }
  }

  // Convert dimers to PairInput
  for (const auto &mol_pairs : uc_dimers.molecule_neighbors) {
    for (const auto &[dimer, unique_idx] : mol_pairs) {
      if (unique_idx < 0 ||
          unique_idx >= static_cast<int>(uc_dimers.unique_dimers.size())) {
        continue;
      }

      const auto &unique_dimer = uc_dimers.unique_dimers[unique_idx];
      double total_energy = unique_dimer.interaction_energy();

      // Skip zero energy dimers
      if (total_energy == 0.0)
        continue;

      PairInput pair;
      pair.molecule_a = dimer.a().unit_cell_molecule_idx();
      pair.molecule_b = dimer.b().unit_cell_molecule_idx();
      pair.v_ab_com = dimer.v_ab_com();
      pair.energy = total_energy;
      input.pairs.push_back(pair);
    }
  }

  occ::log::info("Converted {} molecules and {} pairs from elat data",
                 input.molecules.size(), input.pairs.size());

  return input;
}

std::unique_ptr<PotentialBase> ElasticFitter::create_potential_from_pair(
    const PairInput &pair, const MoleculeInput &mol_a,
    const MoleculeInput &mol_b, double adjusted_energy) const {

  double distance = pair.v_ab_com.norm();
  const occ::Vec3 &rvec = pair.v_ab_com;

  switch (m_settings.potential_type) {
  case PotentialType::MORSE: {
    double D0 = -1.0 * adjusted_energy;
    double h = std::pow(10, 13);
    double conversion_factor = 1.6605388e-24 * std::pow(h, 2) * 6.0221418;
    // Mass stored in g/mol, convert to kg/mol for calculation
    double m = std::sqrt((mol_a.mass / 1000.0) * (mol_b.mass / 1000.0));
    double k = m * conversion_factor; // kj/mol/angstrom^2
    double alpha = std::sqrt(k / (2 * std::abs(D0)));

    auto potential = std::make_unique<MorseWrapper>(D0, distance, alpha, rvec);

    occ::log::debug("Added Morse potential: {} for pair at distance {:.4f}",
                    potential->to_string(), distance);

    return potential;
  }
  case PotentialType::LJ: {
    double eps = -1.0 * adjusted_energy;
    auto potential = std::make_unique<LJWrapper>(eps, distance, rvec);

    occ::log::debug("Added LJ potential: {:30} for pair at distance {:.4f}",
                    potential->to_string(), distance);

    return potential;
  }
  case PotentialType::LJ_A: {
    throw std::runtime_error("LJ_A should only be used for positive energies");
  }
  default:
    throw std::runtime_error("Unknown potential type");
  }
}

PES ElasticFitter::build_pes_from_input(const ElasticFitInput &input) {
  PES pes(input.volume, input.lattice_vectors);
  pes.set_scale(m_settings.scale_factor);
  pes.set_temperature(m_settings.temperature);

  // Calculate energy shift
  double energy_shift = 0.0;
  if (m_settings.max_to_zero) {
    if (m_settings.include_positive) {
      occ::log::warn("Can't include positive and set max to zero.");
    } else {
      // Find the maximum pair energy
      double max_energy = 0.0;
      for (const auto &pair : input.pairs) {
        if (pair.energy > max_energy) {
          max_energy = pair.energy;
        }
      }
      energy_shift = max_energy;
      if (energy_shift != 0.0) {
        pes.set_shift(energy_shift);
        occ::log::info("Shifting all pair energies down by {:.4f} kJ/mol",
                       energy_shift);
      }
    }
  }

  size_t discarded_count = 0;
  double discarded_total_energy = 0.0;

  // Build a map from molecule id to MoleculeInput for easy lookup
  ankerl::unordered_dense::map<int, const MoleculeInput *> mol_map;
  for (const auto &mol : input.molecules) {
    mol_map[mol.id] = &mol;
  }

  for (const auto &pair : input.pairs) {
    double adjusted_energy = pair.energy - energy_shift;

    // Handle positive energies
    if (adjusted_energy > 0.0) {
      if (!m_settings.include_positive) {
        occ::log::debug("Skipping pair with positive total energy {:.4f}",
                        adjusted_energy);
        discarded_count++;
        discarded_total_energy += adjusted_energy;
        continue;
      }

      // Create LJ_A potential for positive energies
      const double eps = -1.0 * adjusted_energy;
      double distance = pair.v_ab_com.norm();
      auto potential = std::make_unique<LJ_AWrapper>(eps, distance, pair.v_ab_com);

      occ::log::debug(
          "Added LJ_A potential: {:30} for pair at distance {:.4f}",
          potential->to_string(), distance);

      // Set mass data
      auto it_a = mol_map.find(pair.molecule_a);
      auto it_b = mol_map.find(pair.molecule_b);
      if (it_a != mol_map.end() && it_b != mol_map.end()) {
        potential->set_pair_mass(
            std::make_pair(it_a->second->mass, it_b->second->mass));
      }

      // Set pair indices
      potential->set_uc_pair_indices(
          std::make_pair(pair.molecule_a, pair.molecule_b));
      potential->set_pair_indices(
          std::make_pair(pair.molecule_a, pair.molecule_b));

      pes.add_potential(std::move(potential));
      continue;
    }

    // Create potential for negative energies
    auto it_a = mol_map.find(pair.molecule_a);
    auto it_b = mol_map.find(pair.molecule_b);
    if (it_a == mol_map.end() || it_b == mol_map.end()) {
      occ::log::warn("Missing molecule data for pair ({}, {})", pair.molecule_a,
                     pair.molecule_b);
      continue;
    }

    auto potential = create_potential_from_pair(pair, *it_a->second,
                                                 *it_b->second, adjusted_energy);
    if (potential) {
      // Set mass data
      potential->set_pair_mass(
          std::make_pair(it_a->second->mass, it_b->second->mass));

      // Set pair indices
      potential->set_uc_pair_indices(
          std::make_pair(pair.molecule_a, pair.molecule_b));
      potential->set_pair_indices(
          std::make_pair(pair.molecule_a, pair.molecule_b));

      pes.add_potential(std::move(potential));
    }
  }

  if (discarded_count > 0) {
    occ::log::warn("Discarded {} pairs with positive interaction energies "
                   "(total: {:.3f} kJ/mol)",
                   discarded_count, discarded_total_energy / 2.0);
  }

  return pes;
}

FittingResults
ElasticFitter::fit_elastic_tensor(const ElasticFitInput &input) {
  FittingResults results;

  // Build PES from minimal input
  PES pes = build_pes_from_input(input);

  // Calculate lattice energy and elastic tensor
  results.lattice_energy = pes.lattice_energy();
  results.elastic_tensor = pes.compute_elastic_tensor(
      input.volume, m_settings.solver_type, m_settings.svd_threshold,
      m_settings.save_debug_matrices);

  results.total_potentials_created = pes.number_of_potentials();

  // Handle energy shifting if needed
  if (m_settings.max_to_zero) {
    double og_elat =
        results.lattice_energy + pes.number_of_potentials() * pes.shift() / 2.0;
    results.energy_shift_applied = pes.shift();

    occ::log::info("Unaltered lattice energy {:.3f} kJ/(mole unit cells)",
                   og_elat);
    occ::log::info("Shifted lattice energy {:.3f} kJ/(mole unit cells)",
                   results.lattice_energy);

    // Scale the elastic tensor
    results.elastic_tensor *= og_elat / results.lattice_energy;
    occ::log::info("Applied energy scaling to elastic tensor");
  }

  // Run phonon analysis if requested
  if (m_settings.animate_phonons || m_settings.temperature > 0.0) {
    if (pes.has_crystal()) {
      pes.phonons(m_settings.shrinking_factors, m_settings.shift,
                  m_settings.animate_phonons);
    } else {
      occ::log::warn(
          "Phonon analysis requires full Crystal object, skipping...");
    }
  }

  return results;
}

FittingResults ElasticFitter::fit_elastic_tensor(
    const occ::interaction::ElatResults &elat_data) {
  // Convert to minimal input and use the new path
  ElasticFitInput input = convert_elat_to_input(elat_data);

  // Use the new fitting method
  FittingResults results = fit_elastic_tensor(input);

  // Generate GULP strings if needed (optional, for external use)
  if (!m_settings.gulp_file.empty()) {
    results.gulp_strings = generate_gulp_input(elat_data, results);
  }

  return results;
}

PES ElasticFitter::build_pes_from_elat_data(
    const occ::interaction::ElatResults &elat_data) {
  PES pes(elat_data.crystal);
  pes.set_scale(m_settings.scale_factor);
  pes.set_temperature(m_settings.temperature);

  // Calculate energy shift
  double energy_shift = calculate_energy_shift(elat_data);
  if (energy_shift != 0.0) {
    pes.set_shift(energy_shift);
    occ::log::info("Shifting all pair energies down by {:.4f} kJ/mol",
                   energy_shift);
  }

  size_t discarded_count = 0;
  double discarded_total_energy = 0.0;

  // For elastic fitting, generate unit cell dimers and map energies from
  // symmetry-unique dimers
  const auto &symm_dimers = elat_data.lattice_energy_result.dimers;

  // Determine radius from symmetry-unique dimers (max nearest_atom_distance)
  double max_radius = 0.0;
  for (const auto &dimer : symm_dimers.unique_dimers) {
    double radius = dimer.nearest_distance();
    if (radius > max_radius) {
      max_radius = radius;
    }
  }

  occ::log::info(
      "Using radius {:.3f} A to generate unit cell dimers for elastic fitting",
      max_radius);

  // Generate unit cell dimers using the determined radius
  auto uc_dimers = elat_data.crystal.unit_cell_dimers(max_radius);

  // CRITICAL: Map energies from symmetry-unique dimers to unit cell dimers
  // The unit cell dimers have unique_idx that should map to
  // symm_dimers.unique_dimers
  for (size_t i = 0; i < uc_dimers.unique_dimers.size(); i++) {
    if (i < symm_dimers.unique_dimers.size()) {
      // Copy energy from corresponding symmetry-unique dimer
      double energy = symm_dimers.unique_dimers[i].interaction_energy();
      uc_dimers.unique_dimers[i].set_interaction_energy(energy);
      uc_dimers.unique_dimers[i].set_interaction_energies(
          symm_dimers.unique_dimers[i].interaction_energies());
    }
  }

  // Debug: count total dimers
  size_t total_uc_dimers = 0;
  for (const auto &mol_pairs : uc_dimers.molecule_neighbors) {
    total_uc_dimers += mol_pairs.size();
  }
  occ::log::info("Generated {} unit cell dimer groups with {} total dimers for "
                 "elastic fitting",
                 uc_dimers.molecule_neighbors.size(), total_uc_dimers);
  occ::log::info("Mapped energies from {} symmetry-unique dimers",
                 symm_dimers.unique_dimers.size());

  for (const auto &mol_pairs : uc_dimers.molecule_neighbors) {
    for (const auto &[dimer, unique_idx] : mol_pairs) {
      // Get energy from the unit cell dimer (which now has energies mapped)
      if (unique_idx < 0 ||
          unique_idx >= static_cast<int>(uc_dimers.unique_dimers.size())) {
        continue;
      }

      const auto &unique_dimer = uc_dimers.unique_dimers[unique_idx];
      double total_energy = unique_dimer.interaction_energy();

      // Skip zero energy dimers
      if (total_energy == 0.0)
        continue;

      double adjusted_energy = total_energy - energy_shift;

      // Handle positive energies
      if (adjusted_energy > 0.0) {
        if (!m_settings.include_positive) {
          occ::log::debug("Skipping dimer with positive total energy {:.4f}",
                          adjusted_energy);
          discarded_count++;
          discarded_total_energy += adjusted_energy;
          continue;
        }

        // Create LJ_A potential for positive energies
        const double eps = -1.0 * adjusted_energy;
        double distance = dimer.center_of_mass_distance();
        occ::Vec3 rvec = dimer.v_ab_com();
        auto potential = std::make_unique<LJ_AWrapper>(eps, distance, rvec);

        occ::log::debug(
            "Added LJ_A potential: {:30} for dimer at distance {:.4f}",
            potential->to_string(), distance);

        // Set mass data from dimer
        double mass_a = dimer.a().molar_mass();
        double mass_b = dimer.b().molar_mass();
        potential->set_pair_mass(std::make_pair(mass_a, mass_b));

        // Set pair indices
        int uc_idx_a = dimer.a().unit_cell_molecule_idx();
        int uc_idx_b = dimer.b().unit_cell_molecule_idx();
        potential->set_uc_pair_indices(std::make_pair(uc_idx_a, uc_idx_b));
        potential->set_pair_indices(std::make_pair(uc_idx_a, uc_idx_b));

        pes.add_potential(std::move(potential));
        continue;
      }

      // Create potential for negative energies using dimer data
      auto potential = create_potential_from_dimer(dimer, adjusted_energy);
      if (potential) {
        // Set mass data from dimer
        double mass_a = dimer.a().molar_mass();
        double mass_b = dimer.b().molar_mass();
        potential->set_pair_mass(std::make_pair(mass_a, mass_b));

        // Set pair indices - these might be critical for elastic tensor
        // calculation
        int uc_idx_a = dimer.a().unit_cell_molecule_idx();
        int uc_idx_b = dimer.b().unit_cell_molecule_idx();
        potential->set_uc_pair_indices(std::make_pair(uc_idx_a, uc_idx_b));
        potential->set_pair_indices(std::make_pair(uc_idx_a, uc_idx_b));

        pes.add_potential(std::move(potential));
      }
    }
  }

  if (discarded_count > 0) {
    occ::log::warn("Discarded {} pairs with positive interaction energies "
                   "(total: {:.3f} kJ/mol)",
                   discarded_count, discarded_total_energy / 2.0);
  }

  return pes;
}

std::unique_ptr<PotentialBase>
ElasticFitter::create_potential_from_dimer(const occ::core::Dimer &dimer,
                                           double adjusted_energy) const {

  switch (m_settings.potential_type) {
  case PotentialType::MORSE: {
    double D0 = -1.0 * adjusted_energy;
    double h = std::pow(10, 13);
    double conversion_factor = 1.6605388e-24 * std::pow(h, 2) * 6.0221418;
    double mass_a = dimer.a().molar_mass();
    double mass_b = dimer.b().molar_mass();
    double m = std::sqrt(mass_a * mass_b);
    double k = m * conversion_factor; // kj/mol/angstrom^2
    double alpha = std::sqrt(k / (2 * std::abs(D0)));
    double distance = dimer.center_of_mass_distance();
    occ::Vec3 rvec = dimer.v_ab_com();

    auto potential = std::make_unique<MorseWrapper>(D0, distance, alpha, rvec);

    occ::log::debug("Added Morse potential: {} for dimer at distance {:.4f}",
                    potential->to_string(), distance);

    return potential;
  }
  case PotentialType::LJ: {
    double eps = -1.0 * adjusted_energy;
    double distance = dimer.center_of_mass_distance();
    occ::Vec3 rvec = dimer.v_ab_com();
    auto potential = std::make_unique<LJWrapper>(eps, distance, rvec);

    occ::log::debug("Added LJ potential: {:30} for dimer at distance {:.4f}",
                    potential->to_string(), distance);

    return potential;
  }
  case PotentialType::LJ_A: {
    throw std::runtime_error("LJ_A should only be used for positive energies");
  }
  default:
    throw std::runtime_error("Unknown potential type");
  }
}

double ElasticFitter::calculate_energy_shift(
    const occ::interaction::ElatResults &elat_data) const {
  if (!m_settings.max_to_zero) {
    return 0.0;
  }

  if (m_settings.include_positive) {
    occ::log::warn("Can't include positive and set max to zero.");
    return 0.0;
  }

  // Find the maximum pair energy from the symmetry-unique dimers (these have
  // the original energies)
  double max_energy = 0.0;
  const auto &dimers = elat_data.lattice_energy_result.dimers;
  for (const auto &dimer : dimers.unique_dimers) {
    double energy = dimer.interaction_energy();
    if (energy > max_energy) {
      max_energy = energy;
    }
  }

  return max_energy;
}

std::vector<std::string> ElasticFitter::generate_gulp_crystal_strings(
    const occ::interaction::ElatResults &elat_data) const {
  std::vector<std::string> gulp_strings;

  gulp_strings.push_back("conp prop phon noden hessian");
  gulp_strings.push_back("");
  gulp_strings.push_back("cell");

  const auto &uc = elat_data.crystal.unit_cell();
  const auto &lengths = uc.lengths();
  const auto &angles = uc.angles();
  std::string cry_str =
      fmt::format("{:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f} {:12.8f}",
                  lengths[0], lengths[1], lengths[2], degrees(angles[0]),
                  degrees(angles[1]), degrees(angles[2]));
  gulp_strings.push_back(cry_str);
  gulp_strings.push_back("");
  gulp_strings.push_back("cart");

  const auto &uc_mols = elat_data.crystal.unit_cell_molecules();
  for (const auto &mol : uc_mols) {
    const auto &com = mol.center_of_mass();
    const int mol_idx = mol.unit_cell_molecule_idx();
    std::string gulp_str = fmt::format("X{} core {:12.8f} {:12.8f} {:12.8f}",
                                       mol_idx + 1, com[0], com[1], com[2]);
    gulp_strings.push_back(gulp_str);
  }

  gulp_strings.push_back("");
  gulp_strings.push_back("element");
  for (const auto &mol : uc_mols) {
    double m = mol.molar_mass() * 1000;
    const int mol_idx = mol.unit_cell_molecule_idx();
    std::string gulp_str = fmt::format("mass X{} {:8.4f}", mol_idx + 1, m);
    gulp_strings.push_back(gulp_str);
  }

  gulp_strings.push_back("end");
  gulp_strings.push_back("");
  gulp_strings.push_back("space");
  gulp_strings.push_back("1");
  gulp_strings.push_back("");

  return gulp_strings;
}

std::vector<std::string>
ElasticFitter::generate_gulp_potential_strings() const {
  std::vector<std::string> strings;

  if (m_settings.potential_type == PotentialType::MORSE) {
    strings.push_back("morse inter kjmol");
  } else {
    strings.push_back("lennard epsilon kjmol");
  }

  return strings;
}

std::vector<std::string> ElasticFitter::generate_gulp_input(
    const occ::interaction::ElatResults &elat_data,
    const FittingResults &results) const {
  return results.gulp_strings;
}

void ElasticFitter::print_elastic_tensor(const occ::Mat6 &tensor,
                                         const std::string &title) {
  fmt::print("{}\n", title);

  for (int i = 0; i < 6; ++i) {
    // Print upper triangle only with proper spacing
    for (int k = 0; k < i; ++k) {
      fmt::print("{:9}", ""); // Empty space for lower triangle
    }
    for (int j = i; j < 6; ++j) {
      fmt::print("{:9.3f}", tensor(i, j));
    }
    fmt::print("\n");
  }
  fmt::print("\n"); // Empty line
}

void ElasticFitter::save_elastic_tensor(const occ::Mat6 &tensor,
                                        const std::string &filename) {
  auto dest = fmt::output_file(filename);
  dest.print("{}", format_matrix(tensor));
}

} // namespace occ::elastic_fit
