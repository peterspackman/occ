#include <Eigen/Geometry>
#include <trajan/core/topology.h>

namespace trajan::core {

Topology::Topology(const BondGraph &bond_graph) : m_bond_graph(bond_graph) {
  const auto &adj_list = m_bond_graph.get_adjacency_list();

  for (const auto &[node_id, neighbours] : adj_list) {
    for (const auto &[neighbour_id, bond] : neighbours) {
      if (node_id < neighbour_id) {
        auto bond_pair = this->make_bond_pair(node_id, neighbour_id);
        Bond corrected_bond = bond;
        corrected_bond.idxs = bond_pair;
        m_bond_storage[bond_pair] = corrected_bond;
      }
    }
  }

  this->generate_all_from_bonds();
}

Topology::Topology(const std::vector<Atom> &atoms) : m_bond_graph(atoms) {
  for (size_t i = 0; i < atoms.size(); i++) {
    for (size_t j = i + 1; j < atoms.size(); j++) {
      auto bond_opt = atoms[i].is_bonded(atoms[j]);
      if (bond_opt.has_value()) {
        Bond bond = bond_opt.value();
        auto bond_pair = this->make_bond_pair(i, j);
        bond.idxs = bond_pair;

        m_bond_graph.add_edge(i, j, bond);
        m_bond_storage[bond_pair] = bond;
      }
    }
  }
  this->generate_all_from_bonds();
}

void Topology::add_bond(size_t atom1, size_t atom2, double bond_length) {
  auto bond_pair = this->make_bond_pair(atom1, atom2);

  if (m_bond_storage.find(bond_pair) != m_bond_storage.end()) {
    trajan::log::warn(
        fmt::format("Bond ({} {}) already in Topology.", atom1, atom2));
    return;
  }

  Bond bond(bond_length);
  bond.idxs = bond_pair;

  m_bond_graph.add_edge(atom1, atom2, bond);
  m_bond_storage[bond_pair] = bond;
}

void Topology::remove_bond(size_t atom1, size_t atom2) {
  auto bond_pair = this->make_bond_pair(atom1, atom2);

  if (m_bond_storage.find(bond_pair) == m_bond_storage.end()) {
    trajan::log::warn(
        fmt::format("Bond ({} {}) not in Topology.", atom1, atom2));
    return;
  }

  m_bond_graph.remove_edge(atom1, atom2);
  m_bond_storage.erase(bond_pair);
}

bool Topology::has_bond(size_t atom1, size_t atom2) const {
  auto bond_pair = this->make_bond_pair(atom1, atom2);
  return m_bond_storage.find(bond_pair) != m_bond_storage.end();
}

void Topology::clear_bonds() {
  m_bond_graph.clear_edges();
  m_bond_storage.clear();
  this->clear_angles();
  this->clear_dihedrals();
}

std::vector<Bond> Topology::get_bonds() const {
  std::vector<Bond> bonds;
  bonds.reserve(m_bond_storage.size());

  for (const auto &[bond_pair, bond] : m_bond_storage) {
    bonds.push_back(bond);
  }

  return bonds;
}

std::optional<Bond> Topology::get_bond(size_t atom1, size_t atom2) const {
  auto bond_pair = this->make_bond_pair(atom1, atom2);
  auto it = m_bond_storage.find(bond_pair);

  if (it != m_bond_storage.end()) {
    return it->second;
  }

  return std::nullopt;
}

bool Topology::update_bond(size_t atom1, size_t atom2,
                           const Bond &updated_bond) {
  auto bond_pair = this->make_bond_pair(atom1, atom2);
  auto it = m_bond_storage.find(bond_pair);

  if (it != m_bond_storage.end()) {
    Bond corrected_bond = updated_bond;
    corrected_bond.idxs = bond_pair;

    it->second = corrected_bond;

    m_bond_graph.add_edge(atom1, atom2, corrected_bond);

    return true;
  }

  return false;
}

void Topology::add_angle(size_t atom1, size_t centre, size_t atom3) {
  Angle angle(atom1, centre, atom3);
  if (m_angle_set.find(angle) != m_angle_set.end()) {
    trajan::log::warn(fmt::format("Angle ({} {} {}) already in Topology.",
                                  atom1, centre, atom3));
    return;
  }
  m_angles.push_back(angle);
  this->update_angle_structures();
}

void Topology::remove_angle(size_t atom1, size_t centre, size_t atom3) {
  Angle target(atom1, centre, atom3);
  if (m_angle_set.find(target) == m_angle_set.end()) {
    trajan::log::warn(
        fmt::format("Angle ({} {} {}) not in Topology.", atom1, centre, atom3));
    return;
  }
  m_angles.erase(std::remove(m_angles.begin(), m_angles.end(), target),
                 m_angles.end());
  this->update_angle_structures();
}

bool Topology::has_angle(size_t atom1, size_t centre, size_t atom3) const {
  Angle target(atom1, centre, atom3);
  return m_angle_set.find(target) != m_angle_set.end();
}

void Topology::clear_angles() {
  m_angles.clear();
  m_angle_set.clear();
}

void Topology::add_dihedral(size_t atom1, size_t atom2, size_t atom3,
                            size_t atom4, DihedralType type) {
  Dihedral dihedral(atom1, atom2, atom3, atom4, type);
  if (m_dihedral_set.find(dihedral) != m_dihedral_set.end()) {
    trajan::log::warn(
        fmt::format("Dihedral ({} {} {} {}, {}) already in Topology.", atom1,
                    atom2, atom3, atom4, dihedral_type_to_string(type)));
    return;
  }
  m_dihedrals.push_back(dihedral);
  this->update_dihedral_structures();
}

void Topology::remove_dihedral(size_t atom1, size_t atom2, size_t atom3,
                               size_t atom4) {
  Dihedral target(atom1, atom2, atom3, atom4);
  if (m_dihedral_set.find(target) == m_dihedral_set.end()) {
    trajan::log::warn(fmt::format("Dihedral ({} {} {} {}) not in Topology.",
                                  atom1, atom2, atom3, atom4));
    return;
  }
  m_dihedrals.erase(std::remove(m_dihedrals.begin(), m_dihedrals.end(), target),
                    m_dihedrals.end());
  this->update_dihedral_structures();
}

bool Topology::has_dihedral(size_t atom1, size_t atom2, size_t atom3,
                            size_t atom4) const {
  Dihedral target1(atom1, atom2, atom3, atom4);
  bool test1 = m_dihedral_set.find(target1) != m_dihedral_set.end();
  Dihedral target2(atom4, atom3, atom2, atom1);
  bool test2 = m_dihedral_set.find(target2) != m_dihedral_set.end();
  return test1 || test2;
}

void Topology::clear_dihedrals() {
  m_dihedrals.clear();
  m_dihedral_set.clear();
}

void Topology::generate_angles_from_bonds() {
  this->clear_angles();

  const auto &adj_list = m_bond_graph.get_adjacency_list();
  for (const auto &[atom_idx, neighbours] : adj_list) {
    if (neighbours.size() < 2) {
      continue;
    }
    auto angles = this->find_angles_around_atom(atom_idx);
    for (const auto &angle : angles) {
      m_angles.push_back(angle);
    }
  }

  this->update_angle_structures();
}

void Topology::generate_proper_dihedrals_from_bonds() {
  m_dihedrals.erase(std::remove_if(m_dihedrals.begin(), m_dihedrals.end(),
                                   [](const Dihedral &d) {
                                     return d.type == DihedralType::PROPER;
                                   }),
                    m_dihedrals.end());

  std::vector<Dihedral> new_dihedrals;

  for (const auto &[bond_pair, bond] : m_bond_storage) {
    size_t atom1 = bond_pair.first;
    size_t atom2 = bond_pair.second;

    auto dihedrals = this->find_proper_dihedrals_for_bond(atom1, atom2);
    new_dihedrals.insert(new_dihedrals.end(), dihedrals.begin(),
                         dihedrals.end());
  }

  std::sort(new_dihedrals.begin(), new_dihedrals.end());
  new_dihedrals.erase(std::unique(new_dihedrals.begin(), new_dihedrals.end()),
                      new_dihedrals.end());

  for (const auto &dihedral : new_dihedrals) {
    m_dihedrals.push_back(dihedral);
  }

  this->update_dihedral_structures();
}

void Topology::generate_improper_dihedrals_from_bonds() {
  std::vector<Dihedral> improper_dihedrals;
  const auto &adj_list = m_bond_graph.get_adjacency_list();

  for (const auto &[atom_idx, neighbours] : adj_list) {
    if (neighbours.size() != 3) {
      continue;
    }
    auto impropers = this->find_improper_dihedrals_around_atom(atom_idx);
    improper_dihedrals.insert(improper_dihedrals.end(), impropers.begin(),
                              impropers.end());
  }

  for (const auto &dihedral : improper_dihedrals) {
    m_dihedrals.push_back(dihedral);
  }

  this->update_dihedral_structures();
}

void Topology::generate_all_from_bonds() {
  this->generate_angles_from_bonds();
  this->generate_proper_dihedrals_from_bonds();
  this->generate_improper_dihedrals_from_bonds();
}

std::vector<size_t> Topology::get_bonded_atoms(size_t atom_idx) const {
  const auto &adj_list = m_bond_graph.get_adjacency_list();
  auto it = adj_list.find(atom_idx);
  if (it != adj_list.end()) {
    std::vector<size_t> bonded;
    bonded.reserve(it->second.size());
    for (const auto &[neighbour_id, bond] : it->second) {
      bonded.push_back(neighbour_id);
    }
    return bonded;
  }
  return {};
}

std::vector<Bond> Topology::get_bonds_involving_atom(size_t atom_idx) const {
  std::vector<Bond> bonds;

  for (const auto &[bond_pair, bond] : m_bond_storage) {
    if (bond_pair.first == atom_idx || bond_pair.second == atom_idx) {
      bonds.push_back(bond);
    }
  }

  return bonds;
}

std::vector<size_t> Topology::get_atoms_at_distance(size_t atom_idx,
                                                    size_t distance) const {
  if (distance == 0)
    return {atom_idx};
  if (distance == 1)
    return this->get_bonded_atoms(atom_idx);

  ankerl::unordered_dense::set<size_t> current_level = {atom_idx};
  ankerl::unordered_dense::set<size_t> visited = {atom_idx};

  for (size_t d = 1; d < distance; ++d) {
    ankerl::unordered_dense::set<size_t> next_level;
    for (size_t atom : current_level) {
      auto neighbours = this->get_bonded_atoms(atom);
      for (size_t neighbour : neighbours) {
        if (!visited.contains(neighbour)) {
          next_level.insert(neighbour);
          visited.insert(neighbour);
        }
      }
    }
    current_level = std::move(next_level);
  }

  ankerl::unordered_dense::set<size_t> final_level;
  for (size_t atom : current_level) {
    auto neighbours = this->get_bonded_atoms(atom);
    for (size_t neighbour : neighbours) {
      if (!visited.contains(neighbour)) {
        final_level.insert(neighbour);
      }
    }
  }

  return std::vector<size_t>(final_level.begin(), final_level.end());
}

std::vector<Molecule> Topology::extract_molecules() const {
  std::vector<Molecule> molecules{};

  auto components = m_bond_graph.find_connected_components();

  for (const auto &component : components) {
    molecules.emplace_back(component);
  }

  return molecules;
}

size_t Topology::num_proper_dihedrals() const {
  return std::count_if(
      m_dihedrals.begin(), m_dihedrals.end(),
      [](const Dihedral &d) { return d.type == DihedralType::PROPER; });
}

size_t Topology::num_improper_dihedrals() const {
  return std::count_if(
      m_dihedrals.begin(), m_dihedrals.end(),
      [](const Dihedral &d) { return d.type == DihedralType::IMPROPER; });
}

void Topology::update_angle_structures() {
  m_angle_set.clear();
  for (const auto &angle : m_angles) {
    m_angle_set.insert(angle);
  }
}

void Topology::update_dihedral_structures() {
  m_dihedral_set.clear();
  for (const auto &dihedral : m_dihedrals) {
    m_dihedral_set.insert(dihedral);
  }
}

std::vector<Angle> Topology::find_angles_around_atom(size_t centre_atom) const {
  std::vector<Angle> angles;
  auto neighbours = this->get_bonded_atoms(centre_atom);

  if (neighbours.size() < 2) {
    return angles;
  }

  for (size_t i = 0; i < neighbours.size(); i++) {
    for (size_t j = i + 1; j < neighbours.size(); j++) {
      Angle angle(neighbours[i], centre_atom, neighbours[j]);
      angles.push_back(angle);
    }
  }

  return angles;
}

std::vector<Dihedral>
Topology::find_proper_dihedrals_for_bond(size_t atom1, size_t atom2) const {
  std::vector<Dihedral> dihedrals;

  auto neighbours1 = this->get_bonded_atoms(atom1);
  auto neighbours2 = this->get_bonded_atoms(atom2);

  neighbours1.erase(std::remove(neighbours1.begin(), neighbours1.end(), atom2),
                    neighbours1.end());
  neighbours2.erase(std::remove(neighbours2.begin(), neighbours2.end(), atom1),
                    neighbours2.end());

  for (size_t n1 : neighbours1) {
    for (size_t n2 : neighbours2) {
      Dihedral dihedral(n1, atom1, atom2, n2, DihedralType::PROPER);
      dihedrals.push_back(dihedral);
    }
  }

  return dihedrals;
}

std::vector<Dihedral>
Topology::find_improper_dihedrals_around_atom(size_t centre_atom) const {
  std::vector<Dihedral> impropers;
  auto neighbours = get_bonded_atoms(centre_atom);

  if (neighbours.size() == 3) {
    Dihedral improper(centre_atom, neighbours[0], neighbours[1], neighbours[2],
                      DihedralType::IMPROPER);
    impropers.push_back(improper);
  }

  return impropers;
}

double Topology::calculate_angle(const Vec3 &pos1, const Vec3 &pos_center,
                                 const Vec3 &pos3) {
  Vec3 vec1 = pos1 - pos_center;
  Vec3 vec3 = pos3 - pos_center;

  double dot_product = vec1.dot(vec3);
  double magnitude_product = vec1.norm() * vec3.norm();

  if (magnitude_product == 0.0)
    return 0.0;

  double cos_angle = dot_product / magnitude_product;
  cos_angle = std::clamp(cos_angle, -1.0, 1.0);

  return std::acos(cos_angle);
}

double Topology::calculate_dihedral(const Vec3 &pos1, const Vec3 &pos2,
                                    const Vec3 &pos3, const Vec3 &pos4) {
  Vec3 vec12 = pos2 - pos1;
  Vec3 vec23 = pos3 - pos2;
  Vec3 vec34 = pos4 - pos3;

  Vec3 normal1 = vec12.cross(vec23);
  Vec3 normal2 = vec23.cross(vec34);

  double dot_product = normal1.dot(normal2);
  double magnitude_product = normal1.norm() * normal2.norm();

  if (magnitude_product == 0.0)
    return 0.0;

  double cos_angle = dot_product / magnitude_product;
  cos_angle = std::clamp(cos_angle, -1.0, 1.0);

  return std::acos(cos_angle);
}

bool Topology::validate_topology() const { return check_issues().empty(); }

std::vector<std::string> Topology::check_issues() const {
  std::vector<std::string> issues;

  for (const auto &angle : m_angles) {
    if (angle.atom1() == angle.center_atom() ||
        angle.atom3() == angle.center_atom() ||
        angle.atom1() == angle.atom3()) {
      issues.push_back(fmt::format("Invalid angle: {} {} {}", angle.atom1(),
                                   angle.center_atom(), angle.atom3()));
    }
  }

  for (const auto &dihedral : m_dihedrals) {
    ankerl::unordered_dense::set<size_t> unique_atoms(
        dihedral.atom_indices.begin(), dihedral.atom_indices.end());
    if (unique_atoms.size() != 4) {
      issues.push_back(
          fmt::format("Invalid dihedral: {} {} {} {}", dihedral.atom_indices[0],
                      dihedral.atom_indices[1], dihedral.atom_indices[2],
                      dihedral.atom_indices[3]));
    }
  }

  return issues;
}

void Topology::print_summary() const {
  trajan::log::info("Topology Summary:\n");
  trajan::log::info("  Bonds: {}\n", this->num_bonds());
  trajan::log::info("  Angles: {}\n", this->num_angles());
  trajan::log::info("  Proper Dihedrals: {}\n", this->num_proper_dihedrals());
  trajan::log::info("  Improper Dihedrals: {}\n",
                    this->num_improper_dihedrals());
}

std::string Topology::to_string() const {
  return fmt::format("Topology(bonds={}, angles={}, proper_dihedrals={}, "
                     "improper_dihedrals={})",
                     this->num_bonds(), this->num_angles(),
                     this->num_proper_dihedrals(),
                     this->num_improper_dihedrals());
}

} // namespace trajan::core
