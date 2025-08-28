#pragma once
#include <trajan/core/atom.h>
#include <trajan/core/graph.h>
#include <trajan/core/linear_algebra.h>
#include <trajan/core/molecule.h>
#include <trajan/core/util.h>
#include <vector>

namespace trajan::core {

struct Angle {
  std::array<size_t, 3> atom_indices; // [atom1, center_atom, atom3]
  double equilibrium_angle;           // in radians
  double force_constant;

  Angle() : equilibrium_angle(0.0), force_constant(0.0) {}
  Angle(size_t i, size_t j, size_t k)
      : atom_indices({i, j, k}), equilibrium_angle(0.0), force_constant(0.0) {}
  size_t center_atom() const { return atom_indices[1]; }
  size_t atom1() const { return atom_indices[0]; }
  size_t atom3() const { return atom_indices[2]; }
  bool operator==(const Angle &other) const {
    return atom_indices == other.atom_indices;
  }
  bool operator<(const Angle &other) const {
    return atom_indices < other.atom_indices;
  }
  struct Hash {
    size_t operator()(const Angle &angle) const {
      size_t h1 = std::hash<size_t>{}(angle.atom_indices[0]);
      size_t h2 = std::hash<size_t>{}(angle.atom_indices[1]);
      size_t h3 = std::hash<size_t>{}(angle.atom_indices[2]);
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };
};

enum class DihedralType {
  PROPER,  // normal four-body dihedral
  IMPROPER // improper dihedral (one atom bonded to three others)
};

inline std::string dihedral_type_to_string(DihedralType type) {
  switch (type) {
  case DihedralType::PROPER:
    return "PROPER";
  case DihedralType::IMPROPER:
    return "IMPROPER";
  default:
    return "UNKNOWN";
  }
}

inline DihedralType dihedral_type_from_string(const std::string &str) {
  std::string edit_str = trajan::util::trim_copy(str);
  trajan::util::to_upper(edit_str);

  if (edit_str == "PROPER")
    return DihedralType::PROPER;
  if (edit_str == "IMPROPER")
    return DihedralType::IMPROPER;

  throw std::invalid_argument(fmt::format("Unknown DihedralType: {}", str));
}

struct Dihedral {
  std::array<size_t, 4> atom_indices; // [atom1, atom2, atom3, atom4]
  DihedralType type;
  double equilibrium_angle; // in radians
  double force_constant;
  int multiplicity; // for periodic dihedrals
  Dihedral()
      : type(DihedralType::PROPER), equilibrium_angle(0.0), force_constant(0.0),
        multiplicity(1) {}
  Dihedral(size_t i, size_t j, size_t k, size_t l,
           DihedralType t = DihedralType::PROPER)
      : atom_indices({i, j, k, l}), type(t), equilibrium_angle(0.0),
        force_constant(0.0), multiplicity(1) {}
  size_t atom2() const { return atom_indices[1]; }
  size_t atom3() const { return atom_indices[2]; }
  bool operator==(const Dihedral &other) const {
    return atom_indices == other.atom_indices;
  }
  bool operator<(const Dihedral &other) const {
    return atom_indices < other.atom_indices;
  }
  struct Hash {
    size_t operator()(const Dihedral &dihedral) const {
      size_t h1 = std::hash<size_t>{}(dihedral.atom_indices[0]);
      size_t h2 = std::hash<size_t>{}(dihedral.atom_indices[1]);
      size_t h3 = std::hash<size_t>{}(dihedral.atom_indices[2]);
      size_t h4 = std::hash<size_t>{}(dihedral.atom_indices[3]);
      return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
  };
};

struct BondPairHash {
  size_t operator()(const std::pair<size_t, size_t> &p) const {
    return std::hash<size_t>{}(p.first) ^ (std::hash<size_t>{}(p.second) << 1);
  }
};

class Topology {
public:
  Topology() = default;
  Topology(const BondGraph &bond_graph);
  Topology(const std::vector<Atom> &atoms);

  Topology &operator=(const Topology &) = default;

  // Topology &operator=(const Topology &other) {
  //   if (this == &other)
  //     return *this;
  //
  //   trajan::log::critical("Copying m_bond_graph");
  //   m_bond_graph = other.m_bond_graph;
  //
  //   trajan::log::critical("Copying m_angles");
  //   m_angles = other.m_angles;
  //
  //   trajan::log::critical("Copying m_dihedrals");
  //   m_dihedrals = other.m_dihedrals;
  //
  //   trajan::log::critical("Copying m_angle_set");
  //   m_angle_set = other.m_angle_set;
  //
  //   trajan::log::critical("Copying m_dihedral_set");
  //   m_dihedral_set = other.m_dihedral_set;
  //
  //   trajan::log::critical("Copying m_bond_storage");
  //   m_bond_storage = other.m_bond_storage;
  //
  //   return *this;
  // }

  inline const std::vector<Atom> &get_atoms() const {
    return m_bond_graph.nodes();
  }
  inline size_t num_atoms() const { return m_bond_graph.num_nodes(); }

  void add_bond(size_t atom1, size_t atom2, double bond_length = 0.0);
  void remove_bond(size_t atom1, size_t atom2);
  bool has_bond(size_t atom1, size_t atom2) const;
  void clear_bonds();
  std::vector<Bond> get_bonds() const;

  void add_angle(size_t atom1, size_t center, size_t atom3);
  void remove_angle(size_t atom1, size_t center, size_t atom3);
  bool has_angle(size_t atom1, size_t center, size_t atom3) const;
  void clear_angles();
  inline const std::vector<Angle> &get_angles() const { return m_angles; }

  void add_dihedral(size_t atom1, size_t atom2, size_t atom3, size_t atom4,
                    DihedralType type = DihedralType::PROPER);
  void remove_dihedral(size_t atom1, size_t atom2, size_t atom3, size_t atom4);
  bool has_dihedral(size_t atom1, size_t atom2, size_t atom3,
                    size_t atom4) const;
  void clear_dihedrals();
  inline const std::vector<Dihedral> &get_dihedrals() const {
    return m_dihedrals;
  }

  void generate_angles_from_bonds();
  void generate_proper_dihedrals_from_bonds();
  void generate_improper_dihedrals_from_bonds();
  void generate_all_from_bonds();

  // graph-based queries - delegates to BondGraph
  std::vector<size_t> get_bonded_atoms(size_t atom_idx) const;
  std::vector<size_t> get_atoms_at_distance(size_t atom_idx,
                                            size_t distance) const;
  std::optional<Bond> get_bond(size_t atom1, size_t atom2) const;
  bool update_bond(size_t atom1, size_t atom2, const Bond &updated_bond);
  std::vector<Bond> get_bonds_involving_atom(size_t atom_idx) const;

  std::vector<Molecule> extract_molecules() const;

  // inline size_t num_bonds() const { return get_bonds().size(); }
  inline size_t num_bonds() const { return m_bond_storage.size(); }
  inline size_t num_angles() const { return m_angles.size(); }
  inline size_t num_dihedrals() const { return m_dihedrals.size(); }
  size_t num_proper_dihedrals() const;
  size_t num_improper_dihedrals() const;

  bool validate_topology() const;
  std::vector<std::string> check_issues() const;

  void print_summary() const;
  std::string to_string() const;

  // access to underlying graph
  const BondGraph &get_bond_graph() const { return m_bond_graph; }
  BondGraph &get_bond_graph() { return m_bond_graph; }

private:
  BondGraph m_bond_graph;
  std::vector<Angle> m_angles;
  std::vector<Dihedral> m_dihedrals;

  // fast lookup structures for angles and dihedrals
  ankerl::unordered_dense::set<Angle, Angle::Hash> m_angle_set;
  ankerl::unordered_dense::set<Dihedral, Dihedral::Hash> m_dihedral_set;
  ankerl::unordered_dense::map<std::pair<size_t, size_t>, Bond, BondPairHash>
      m_bond_storage;

  inline std::pair<size_t, size_t> make_bond_pair(size_t atom1,
                                                  size_t atom2) const {
    return {std::min(atom1, atom2), std::max(atom1, atom2)};
  }

  void update_angle_structures();
  void update_dihedral_structures();

  std::vector<Angle> find_angles_around_atom(size_t center_atom) const;
  std::vector<Dihedral> find_proper_dihedrals_for_bond(size_t atom1,
                                                       size_t atom2) const;
  std::vector<Dihedral>
  find_improper_dihedrals_around_atom(size_t center_atom) const;

  static double calculate_angle(const Vec3 &pos1, const Vec3 &pos_center,
                                const Vec3 &pos3);
  static double calculate_dihedral(const Vec3 &pos1, const Vec3 &pos2,
                                   const Vec3 &pos3, const Vec3 &pos4);
};

} // namespace trajan::core
