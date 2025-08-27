#include <occ/core/dimer.h>
#include <occ/io/core_json.h>
#include <occ/io/eigen_json.h>

namespace occ::core {

void to_json(nlohmann::json &j, const Molecule &mol) {

  nlohmann::json elements;
  for (const auto &el : mol.elements()) {
    elements.push_back(el.symbol());
  }
  j["name"] = mol.name();
  j["elements"] = elements;
  Mat pos = mol.positions().transpose();
  j["positions"] = pos;

  if (mol.asymmetric_molecule_idx() > -1) {
    j["asym mol"] = mol.asymmetric_molecule_idx();
  }

  if (mol.unit_cell_molecule_idx() > -1) {
    j["uc mol"] = mol.unit_cell_molecule_idx();
  }

  const auto &asym_idx = mol.asymmetric_unit_idx();
  if (asym_idx.size() > 0) {
    j["asym atom"] = asym_idx.transpose();
  }

  const auto &uc_idx = mol.unit_cell_idx();
  if (asym_idx.size() > 0) {
    j["uc atom"] = uc_idx.transpose();
  }

  const auto &cell_shift = mol.cell_shift();
  j["cell shift"] = cell_shift;
}

void from_json(const nlohmann::json &j, occ::core::Molecule &mol) {
  // Parse elements and positions
  std::vector<occ::core::Element> elements;
  for (const auto &symbol : j["elements"]) {
    elements.push_back(occ::core::Element(symbol.get<std::string>()));
  }

  std::vector<std::array<double, 3>> positions = j["positions"];

  // Construct the molecule
  mol = occ::core::Molecule(elements, positions);

  // Set the name
  mol.set_name(j["name"].get<std::string>());

  // Set optional properties
  if (j.contains("asym mol")) {
    mol.set_asymmetric_molecule_idx(j["asym mol"].get<int>());
  }
  if (j.contains("uc mol")) {
    mol.set_unit_cell_molecule_idx(j["uc mol"].get<int>());
  }
  if (j.contains("asym atom")) {
    Eigen::VectorXi asym_idx = j["asym atom"].get<Eigen::VectorXi>();
    mol.set_asymmetric_unit_idx(asym_idx);
  }
  if (j.contains("uc atom")) {
    Eigen::VectorXi uc_idx = j["uc atom"].get<Eigen::VectorXi>();
    mol.set_unit_cell_idx(uc_idx);
  }
  if (j.contains("cell shift")) {
    auto cell_shift = j["cell shift"].get<IVec3>();
    mol.set_cell_shift(cell_shift);
  }
}

void to_json(nlohmann::json &j, const VibrationalModes &vib) {
  // Frequencies
  j["frequencies_cm"] = vib.frequencies_cm;
  j["frequencies_hartree"] = vib.frequencies_hartree;
  
  // Matrices
  j["normal_modes"] = vib.normal_modes;
  j["mass_weighted_hessian"] = vib.mass_weighted_hessian;
  
  // Optional original Hessian
  if (vib.hessian.size() > 0) {
    j["hessian"] = vib.hessian;
  }
  
  // Add summary statistics
  j["n_atoms"] = vib.n_atoms();
  j["n_modes"] = vib.n_modes();
  
  // Add sorted frequencies for convenience
  Vec all_freqs = vib.get_all_frequencies();
  j["sorted_frequencies_cm"] = all_freqs;
  
  // Add some metadata
  if (all_freqs.size() > 0) {
    j["lowest_freq_cm"] = all_freqs[0];
    j["highest_freq_cm"] = all_freqs[all_freqs.size() - 1];
  }
}

void from_json(const nlohmann::json &j, VibrationalModes &vib) {
  // Frequencies
  vib.frequencies_cm = j.at("frequencies_cm").get<Vec>();
  vib.frequencies_hartree = j.at("frequencies_hartree").get<Vec>();
  
  // Matrices
  vib.normal_modes = j.at("normal_modes").get<Mat>();
  vib.mass_weighted_hessian = j.at("mass_weighted_hessian").get<Mat>();
  
  // Optional original Hessian
  if (j.contains("hessian")) {
    vib.hessian = j["hessian"].get<Mat>();
  }
}

} // namespace occ::core

namespace nlohmann {
occ::core::Dimer adl_serializer<occ::core::Dimer>::from_json(const json &j) {
  occ::core::Dimer dimer(j.at("mol_a").get<occ::core::Molecule>(),
                         j.at("mol_b").get<occ::core::Molecule>());
  dimer.set_interaction_energy(j["interaction_energy"].get<double>());
  dimer.set_interaction_id(j["interaction_id"].get<int>());
  return dimer;
}

void adl_serializer<occ::core::Dimer>::to_json(json &j,
                                               const occ::core::Dimer &dimer) {
  j["mol_a"] = dimer.a();
  j["mol_b"] = dimer.b();
  j["interaction_energy"] = dimer.interaction_energy();
  j["interaction_id"] = dimer.interaction_id();
}

} // namespace nlohmann
