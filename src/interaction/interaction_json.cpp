#include <occ/interaction/interaction_json.h>
#include <occ/io/crystal_json.h>
#include <occ/crystal/dimer_labeller.h>
#include <ankerl/unordered_dense.h>
#include <fmt/format.h>
#include <fstream>

namespace occ::interaction {

void to_json(nlohmann::json &j, const CEEnergyComponents &c) {
  j["Coulomb"] = c.coulomb;
  j["Exchange"] = c.exchange;
  j["Repulsion"] = c.repulsion;
  j["Dispersion"] = c.dispersion;
  j["Polarization"] = c.polarization;
  j["Total"] = c.total;
}

void from_json(const nlohmann::json &j, CEEnergyComponents &c) {
  if (j.contains("Coulomb"))
    j.at("Coulomb").get_to(c.coulomb);
  if (j.contains("coulomb"))
    j.at("coulomb").get_to(c.coulomb);

  if (j.contains("Exchange"))
    j.at("Exchange").get_to(c.exchange);
  if (j.contains("exchange"))
    j.at("exchange").get_to(c.exchange);

  if (j.contains("Repulsion"))
    j.at("Repulsion").get_to(c.repulsion);
  if (j.contains("repulsion"))
    j.at("repulsion").get_to(c.repulsion);

  if (j.contains("Dispersion"))
    j.at("Dispersion").get_to(c.dispersion);
  if (j.contains("dispersion"))
    j.at("dispersion").get_to(c.dispersion);

  if (j.contains("Polarization"))
    j.at("Polarization").get_to(c.polarization);
  if (j.contains("polarization"))
    j.at("polarization").get_to(c.polarization);

  if (j.contains("Total"))
    j.at("Total").get_to(c.total);
  if (j.contains("total"))
    j.at("total").get_to(c.total);
}

void write_elat_json(const std::string& filename, const ElatResults& results) {
  using occ::crystal::SymmetryDimerLabeller;

  nlohmann::json j;
  j["result_type"] = "elat";
  j["title"] = results.title;
  j["crystal"] = results.crystal;
  j["model"] = results.model;
  j["has_permutation_symmetry"] = true;

  const auto& crystal = results.crystal;
  const auto& dimers = results.lattice_energy_result.dimers;
  const auto& uc_atoms = crystal.unit_cell_atoms();

  auto dimer_labeller = SymmetryDimerLabeller(crystal);
  dimer_labeller.connection = "-";
  dimer_labeller.format.fmt_string = "{}";

  j["pairs"] = {};
  for (const auto &mol_pairs : dimers.molecule_neighbors) {
    nlohmann::json m;
    for (const auto &[dimer, unique_idx] : mol_pairs) {
      const auto &unique_dimer = dimers.unique_dimers[unique_idx];
      if (unique_dimer.interaction_energy() == 0.0)
        continue;

      nlohmann::json d;
      nlohmann::json e;

      // Label generation
      auto label = dimer_labeller(dimer);
      d["Label"] = label;
      d["Unique Index"] = unique_idx;

      // Energy components
      const auto &energies = unique_dimer.interaction_energies();
      for (const auto &[k, v] : energies) {
        e[k] = v;
      }
      d["energies"] = e;

      // Nearest neighbor calculation based on distance threshold
      bool is_nearest = dimer.nearest_distance() <= 4.0;
      d["Nearest Neighbor"] = is_nearest;

      // Unit cell atom offsets
      nlohmann::json offsets_a = {};
      {
        const auto &a = dimer.a();
        const auto &a_uc_idx = a.unit_cell_idx();
        const auto &a_uc_shift = a.unit_cell_shift();
        for (int i = 0; i < a_uc_idx.rows(); i++) {
          offsets_a.push_back(std::array<int, 4>{a_uc_idx(i), a_uc_shift(0, i),
                                                 a_uc_shift(1, i),
                                                 a_uc_shift(2, i)});
        }
      }

      nlohmann::json offsets_b = {};
      {
        const auto &b = dimer.b();
        const auto &b_uc_idx = b.unit_cell_idx();
        const auto &b_uc_shift = b.unit_cell_shift();
        for (int i = 0; i < b_uc_idx.rows(); i++) {
          offsets_b.push_back(std::array<int, 4>{b_uc_idx(i), b_uc_shift(0, i),
                                                 b_uc_shift(1, i),
                                                 b_uc_shift(2, i)});
        }
      }
      d["uc_atom_offsets"] = {offsets_a, offsets_b};

      m.push_back(d);
    }
    j["pairs"].push_back(m);
  }

  std::ofstream dest(filename);
  dest << j.dump(2);
}

ElatResults read_elat_json(const std::string& filename) {
  // Load JSON file
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open JSON file: " + filename);
  }

  nlohmann::json j;
  file >> j;

  // Validate JSON format
  if (j["result_type"] != "elat") {
    throw std::runtime_error("Invalid JSON: not an elat result file");
  }

  if (!j.contains("pairs")) {
    throw std::runtime_error("Need 'pairs' in JSON output.");
  }

  if (!j.contains("crystal")) {
    throw std::runtime_error("Need 'crystal' in JSON output.");
  }

  // Extract basic info
  occ::crystal::Crystal crystal = j["crystal"];
  std::string title = j.value("title", "");
  std::string model = j.value("model", "");

  // Extract pairs data and reconstruct CrystalDimers with energies
  const auto& pairs_json = j["pairs"];

  // Build energy mapping from JSON
  ankerl::unordered_dense::map<int, CEEnergyComponents> energy_components;
  ankerl::unordered_dense::map<int, double> total_energies;

  for (size_t mol_idx = 0; mol_idx < pairs_json.size(); mol_idx++) {
    for (const auto& pair_data : pairs_json[mol_idx]) {
      int unique_idx = pair_data["Unique Index"];

      if (pair_data.contains("energies")) {
        const auto& energies_json = pair_data["energies"];

        CEEnergyComponents components;
        if (energies_json.contains("Coulomb")) components.coulomb = energies_json["Coulomb"];
        if (energies_json.contains("Exchange")) components.exchange = energies_json["Exchange"];
        if (energies_json.contains("Repulsion")) components.repulsion = energies_json["Repulsion"];
        if (energies_json.contains("Dispersion")) components.dispersion = energies_json["Dispersion"];
        if (energies_json.contains("Polarization")) components.polarization = energies_json["Polarization"];
        if (energies_json.contains("Total")) components.total = energies_json["Total"];

        energy_components[unique_idx] = components;
        total_energies[unique_idx] = components.total;
      }
    }
  }

  // Generate CrystalDimers from crystal using appropriate radius
  // We need to determine the radius from the JSON or use a reasonable default
  double radius = 15.0; // Default radius
  if (j.contains("radius")) {
    radius = j["radius"];
  }

  occ::crystal::CrystalDimers dimers = crystal.symmetry_unique_dimers(radius);

  // Map energies onto the computed dimers
  for (size_t i = 0; i < dimers.unique_dimers.size(); i++) {
    auto it = total_energies.find(i);
    if (it != total_energies.end()) {
      dimers.unique_dimers[i].set_interaction_energy(it->second);

      // Set detailed energy components if available
      auto comp_it = energy_components.find(i);
      if (comp_it != energy_components.end()) {
        ankerl::unordered_dense::map<std::string, double> energy_map;
        energy_map["Coulomb"] = comp_it->second.coulomb;
        energy_map["Exchange"] = comp_it->second.exchange;
        energy_map["Repulsion"] = comp_it->second.repulsion;
        energy_map["Dispersion"] = comp_it->second.dispersion;
        energy_map["Polarization"] = comp_it->second.polarization;
        energy_map["Total"] = comp_it->second.total;
        dimers.unique_dimers[i].set_interaction_energies(energy_map);
      }
    }
  }

  // Build LatticeEnergyResult
  LatticeEnergyResult lattice_result;
  lattice_result.dimers = std::move(dimers);
  lattice_result.energy_components.resize(lattice_result.dimers.unique_dimers.size());

  for (size_t i = 0; i < lattice_result.energy_components.size(); i++) {
    auto comp_it = energy_components.find(i);
    if (comp_it != energy_components.end()) {
      lattice_result.energy_components[i] = comp_it->second;
      lattice_result.energy_components[i].is_computed = true;
    }
  }

  // Calculate total lattice energy if available
  if (j.contains("lattice_energy")) {
    lattice_result.lattice_energy = j["lattice_energy"];
  }

  return ElatResults{
    std::move(crystal),
    std::move(lattice_result),
    title,
    model
  };
}

} // namespace occ::interaction
