#include <occ/elastic_fit/elastic_fit_json.h>
#include <fstream>
#include <stdexcept>

namespace occ::elastic_fit {

void to_json(nlohmann::json &j, const MoleculeInput &m) {
  j = nlohmann::json{{"id", m.id},
                     {"mass", m.mass},
                     {"center_of_mass",
                      {m.center_of_mass(0), m.center_of_mass(1),
                       m.center_of_mass(2)}}};
}

void from_json(const nlohmann::json &j, MoleculeInput &m) {
  j.at("id").get_to(m.id);
  j.at("mass").get_to(m.mass);
  auto com = j.at("center_of_mass");
  m.center_of_mass = occ::Vec3(com[0].get<double>(), com[1].get<double>(),
                               com[2].get<double>());
}

void to_json(nlohmann::json &j, const PairInput &p) {
  j = nlohmann::json{
      {"molecule_a", p.molecule_a},
      {"molecule_b", p.molecule_b},
      {"v_ab_com", {p.v_ab_com(0), p.v_ab_com(1), p.v_ab_com(2)}},
      {"energy", p.energy}};
}

void from_json(const nlohmann::json &j, PairInput &p) {
  j.at("molecule_a").get_to(p.molecule_a);
  j.at("molecule_b").get_to(p.molecule_b);
  auto v = j.at("v_ab_com");
  p.v_ab_com =
      occ::Vec3(v[0].get<double>(), v[1].get<double>(), v[2].get<double>());
  j.at("energy").get_to(p.energy);
}

void to_json(nlohmann::json &j, const ElasticFitInput &input) {
  j["format_type"] = "elastic_fit_pairs";
  j["format_version"] = "1.0";

  // Lattice vectors as row vectors (easier to read)
  j["lattice_vectors"] = {
      {input.lattice_vectors(0, 0), input.lattice_vectors(0, 1),
       input.lattice_vectors(0, 2)},
      {input.lattice_vectors(1, 0), input.lattice_vectors(1, 1),
       input.lattice_vectors(1, 2)},
      {input.lattice_vectors(2, 0), input.lattice_vectors(2, 1),
       input.lattice_vectors(2, 2)}};

  j["volume"] = input.volume;

  // Metadata
  if (!input.title.empty()) {
    j["title"] = input.title;
  }
  if (!input.model.empty()) {
    j["model"] = input.model;
  }

  // Units for clarity
  j["units"] = {
      {"mass", "g/mol"}, {"distance", "angstrom"}, {"energy", "kJ/mol"}};

  // Molecules
  j["molecules"] = nlohmann::json::array();
  for (const auto &mol : input.molecules) {
    nlohmann::json mol_json;
    to_json(mol_json, mol);
    j["molecules"].push_back(mol_json);
  }

  // Pairs
  j["pairs"] = nlohmann::json::array();
  for (const auto &pair : input.pairs) {
    nlohmann::json pair_json;
    to_json(pair_json, pair);
    j["pairs"].push_back(pair_json);
  }
}

void from_json(const nlohmann::json &j, ElasticFitInput &input) {
  // Validate format type
  if (j.contains("format_type")) {
    std::string format_type = j.at("format_type");
    if (format_type != "elastic_fit_pairs") {
      throw std::runtime_error(
          "Invalid format_type: expected 'elastic_fit_pairs', got '" +
          format_type + "'");
    }
  }

  // Lattice vectors
  auto lv = j.at("lattice_vectors");
  for (int i = 0; i < 3; i++) {
    for (int k = 0; k < 3; k++) {
      input.lattice_vectors(i, k) = lv[i][k].get<double>();
    }
  }

  j.at("volume").get_to(input.volume);

  // Optional metadata
  if (j.contains("title")) {
    j.at("title").get_to(input.title);
  }
  if (j.contains("model")) {
    j.at("model").get_to(input.model);
  }

  // Molecules
  input.molecules.clear();
  for (const auto &mol_json : j.at("molecules")) {
    MoleculeInput mol;
    from_json(mol_json, mol);
    input.molecules.push_back(mol);
  }

  // Pairs
  input.pairs.clear();
  for (const auto &pair_json : j.at("pairs")) {
    PairInput pair;
    from_json(pair_json, pair);
    input.pairs.push_back(pair);
  }
}

void write_elastic_fit_json(const std::string &filename,
                            const ElasticFitInput &input) {
  nlohmann::json j;
  to_json(j, input);

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for writing: " + filename);
  }
  file << j.dump(2);
}

ElasticFitInput read_elastic_fit_json(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file for reading: " + filename);
  }

  nlohmann::json j;
  file >> j;

  ElasticFitInput input;
  from_json(j, input);
  return input;
}

} // namespace occ::elastic_fit
