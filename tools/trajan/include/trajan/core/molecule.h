#pragma once
#include <string>
#include <trajan/core/atom.h>
#include <trajan/core/element.h>
#include <trajan/core/graph.h>
#include <trajan/core/linear_algebra.h>
#include <vector>

namespace trajan::core {

class BondGraph : public graph::Graph<Atom, Bond> {
public:
  BondGraph() = default;
  BondGraph(const std::vector<Atom> &atoms) : Graph<Atom, Bond>(atoms) {};

private:
  NodeID get_node_id_from_node(const Atom &atom) const override {
    return atom.index;
  };
};

class Molecule {
public:
  double x, y, z;
  std::string type;
  size_t index;

  enum Origin {
    Cartesian,   /**< The Cartesian origin i.e. (0, 0, 0) in R3 */
    Centroid,    /**< The molecular centroid i.e. average position of atoms
                    (ignoring mass) */
    CentreOfMass /**< The centre of mass i.e. the weighted average positon */
  };

  inline explicit Molecule() {};
  Molecule(const std::vector<Atom> &atoms);
  Molecule(const graph::ConnectedComponent<Atom, Bond> &cc);
  Molecule(const Molecule &other, Vec3 shift)
      : x(other.x + shift.x()), y(other.y + shift.y()), z(other.z + shift.z()) {
  }

  inline Vec3 position() const { return {x, y, z}; }

  Vec atomic_masses() const;

  Vec3 centroid() const;

  Vec3 centre_of_mass() const;

  inline double square_distance(const Molecule &other) const {
    double dx = other.x - x, dy = other.y - y, dz = other.z - z;
    return dx * dx + dy * dy + dz * dz;
  }

  inline size_t size() const { return m_atomic_numbers.size(); }

  inline bool operator==(const Molecule &rhs) const {
    return this->index == rhs.index;
  }

private:
  int charge = 0;
  std::string m_name{""};
  std::vector<Atom> m_atoms;
  std::vector<Bond> m_bonds;
  std::vector<Element> m_elements;
  IVec m_atomic_numbers;
  Mat3N m_positions;
  Vec m_partial_charges;
};

}; // namespace trajan::core
