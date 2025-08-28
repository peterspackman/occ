#include <trajan/core/molecule.h>

namespace trajan::core {

Molecule::Molecule(const std::vector<Atom> &atoms)
    : m_atomic_numbers(atoms.size()), m_positions(3, atoms.size()),
      m_partial_charges(Vec::Zero(atoms.size())) {
  m_elements.reserve(atoms.size());
  for (size_t i = 0; i < atoms.size(); i++) {
    const Atom &atom = atoms[i];
    m_elements.push_back(atom.element);
    m_atomic_numbers(i) = atom.element.atomic_number();
    m_positions(0, i) = atom.x;
    m_positions(1, i) = atom.y;
    m_positions(2, i) = atom.z;
  }
  m_name = chemical_formula(m_elements);
}

Molecule::Molecule(const graph::ConnectedComponent<Atom, Bond> &cc) {
  auto atoms = cc.get_nodes();
  auto bonds = cc.get_edges();
  m_elements.reserve(atoms.size());
  m_atoms.reserve(atoms.size());
  m_atomic_numbers.resize(atoms.size());
  m_positions.resize(3, atoms.size());
  m_partial_charges.resize(atoms.size(), 0.0);
  size_t i = 0;
  for (const auto &[idx, atom] : atoms) {
    m_atoms.push_back(atom);
    m_elements.push_back(atom.element);
    m_atomic_numbers(i) = atom.element.atomic_number();
    m_positions(0, i) = atom.x;
    m_positions(1, i) = atom.y;
    m_positions(2, i) = atom.z;
    i++;
  }
  m_name = chemical_formula(m_elements);
  m_bonds.reserve(bonds.size());
  for (auto &[p, bond] : bonds) {
    bond.idxs = p;
    m_bonds.push_back(bond);
  }
  this->type = "UNK";
};

Vec Molecule::atomic_masses() const {
  Vec masses(size());
  for (size_t i = 0; i < masses.size(); i++) {
    masses(i) = static_cast<double>(m_elements[i].mass());
  }
  return masses;
}

Vec3 Molecule::centroid() const { return m_positions.rowwise().mean(); }

Vec3 Molecule::centre_of_mass() const {
  trajan::RowVec masses = atomic_masses();
  masses.array() /= masses.sum();
  return (m_positions.array().rowwise() * masses.array()).rowwise().sum();
}

}; // namespace trajan::core
