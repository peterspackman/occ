#pragma once

#include <string>
#include <trajan/core/topology.h>
#include <vector>

namespace trajan::energy {

using core::Atom;
using core::Topology;

enum class UFFAtomType {
  H_,
  H_b,
  He4_4,
  Li,
  Be3_2,
  B_3,
  B_2,
  C_3,
  C_R,
  C_2,
  C_1,
  N_3,
  N_R,
  N_2,
  N_1,
  O_3,
  O_3_2,
  O_R,
  O_2,
  O_1,
  F_,
  Ne4_4,
  Na,
  Mg3_2,
  Al3,
  Si3,
  P_3_3,
  P_3_5,
  P_3_q,
  S_3_2,
  S_3_4,
  S_3_6,
  S_R,
  S_2,
  Cl,
  Ar4_4,
  K,
  Ca6_2,
  Sc3_3,
  Ti3_4,
  Ti6_4,
  V_3_5,
  Cr6_3,
  Mn6_2,
  Fe3_2,
  Fe6_2,
  Co6_3,
  Ni4_2,
  Cu3_1,
  Zn3_2,
  Ga3_3,
  Ge3,
  As3_3,
  Se3_2,
  Br,
  Kr4_4,
  Rb,
  Sr6_2,
  Y_3_3,
  Zr3_4,
  Nb3_5,
  Mo6_6,
  Mo3_6,
  Tc6_5,
  Ru6_2,
  Rh6_3,
  Pd4_2,
  Ag1_1,
  Cd3_2,
  In3_3,
  Sn3,
  Sb3_3,
  Te3_2,
  I,
  Xe4_4,
  Cs,
  Ba6_2,
  La3_3,
  Ce6_3,
  Pr6_3,
  Nd6_3,
  Pm6_3,
  Sm6_3,
  Eu6_3,
  Gd6_3,
  Tb6_3,
  Dy6_3,
  Ho6_3,
  Er6_3,
  Tm6_3,
  Yb6_3,
  Lu6_3,
  Hf3_4,
  Ta3_5,
  W_6_6,
  W_3_4,
  W_3_6,
  Re6_5,
  Re3_7,
  Os6_6,
  Ir6_3,
  Pt4_2,
  Au4_3,
  Hg1_2,
  Tl3_3,
  Pb3,
  Bi3_3,
  Po3_2,
  At,
  Rn4_4,
  Fr,
  Ra6_2,
  Ac6_3,
  Th6_4,
  Pa6_4,
  U_6_4,
  Np6_4,
  Pu6_4,
  Am6_4,
  Cm6_3,
  Bk6_3,
  Cf6_3,
  Es6_3,
  Fm6_3,
  Md6_3,
  No6_3,
  Lw6_3,
  UNKNOWN
};

inline std::string uff_type_to_string(UFFAtomType type) {
  static const std::unordered_map<UFFAtomType, std::string> type_names{
      {UFFAtomType::H_, "H_"},
      {UFFAtomType::H_b, "H_b"},
      {UFFAtomType::He4_4, "He4_4"},
      {UFFAtomType::Li, "Li"},
      {UFFAtomType::Be3_2, "Be3_2"},
      {UFFAtomType::B_3, "B_3"},
      {UFFAtomType::B_2, "B_2"},
      {UFFAtomType::C_3, "C_3"},
      {UFFAtomType::C_R, "C_R"},
      {UFFAtomType::C_2, "C_2"},
      {UFFAtomType::C_1, "C_1"},
      {UFFAtomType::N_3, "N_3"},
      {UFFAtomType::N_R, "N_R"},
      {UFFAtomType::N_2, "N_2"},
      {UFFAtomType::N_1, "N_1"},
      {UFFAtomType::O_3, "O_3"},
      {UFFAtomType::O_3_2, "O_3_2"},
      {UFFAtomType::O_R, "O_R"},
      {UFFAtomType::O_2, "O_2"},
      {UFFAtomType::O_1, "O_1"},
      {UFFAtomType::F_, "F_"},
      {UFFAtomType::Ne4_4, "Ne4_4"},
      {UFFAtomType::Na, "Na"},
      {UFFAtomType::Mg3_2, "Mg3_2"},
      {UFFAtomType::Al3, "Al3"},
      {UFFAtomType::Si3, "Si3"},
      {UFFAtomType::P_3_3, "P_3_3"},
      {UFFAtomType::P_3_5, "P_3_5"},
      {UFFAtomType::P_3_q, "P_3_q"},
      {UFFAtomType::S_3_2, "S_3_2"},
      {UFFAtomType::S_3_4, "S_3_4"},
      {UFFAtomType::S_3_6, "S_3_6"},
      {UFFAtomType::S_R, "S_R"},
      {UFFAtomType::S_2, "S_2"},
      {UFFAtomType::Cl, "Cl"},
      {UFFAtomType::Ar4_4, "Ar4_4"},
      {UFFAtomType::K, "K"},
      {UFFAtomType::Ca6_2, "Ca6_2"},
      {UFFAtomType::UNKNOWN, "UNKNOWN"}};
  auto it = type_names.find(type);
  return it != type_names.end() ? it->second : "UNKNOWN";
}

class UFFTypeAssigner {
public:
  explicit UFFTypeAssigner(const Topology &topology) : m_topology(topology) {}

  std::vector<UFFAtomType> assign_types() {
    m_topology.generate_cyclic_structures();
    const auto &atoms = m_topology.atoms();
    std::vector<UFFAtomType> types;
    types.reserve(atoms.size());

    for (size_t i = 0; i < atoms.size(); ++i) {
      types.push_back(assign_type(i));
    }
    return types;
  }

  UFFAtomType assign_type(size_t atom_idx) {
    const auto &atoms = m_topology.atoms();
    if (atom_idx >= atoms.size())
      return UFFAtomType::UNKNOWN;

    const Atom &atom = atoms[atom_idx];
    int z = atom.atomic_number();
    auto bonded = m_topology.get_bonded_atoms(atom_idx);
    int coord = bonded.size();
    int val_electrons = get_valence_electrons(z);

    // Calculate coordination based on VSEPR if applicable
    if (val_electrons > 0) {
      coord = calculate_coordination(atom_idx, z, val_electrons, coord);
    }

    // Route to specific classification
    switch (z) {
    case 1:
      return classify_hydrogen(atom_idx, bonded);
    case 2:
      return UFFAtomType::He4_4;
    case 3:
      return UFFAtomType::Li;
    case 4:
      return UFFAtomType::Be3_2;
    case 5:
      return (coord == 3) ? UFFAtomType::B_2 : UFFAtomType::B_3;
    case 6:
      return classify_carbon(atom_idx, bonded, coord);
    case 7:
      return classify_nitrogen(atom_idx, bonded, coord);
    case 8:
      return classify_oxygen(atom_idx, bonded, coord);
    case 9:
      return UFFAtomType::F_;
    case 10:
      return UFFAtomType::Ne4_4;
    case 11:
      return UFFAtomType::Na;
    case 12:
      return UFFAtomType::Mg3_2;
    case 13:
      return UFFAtomType::Al3;
    case 14:
      return UFFAtomType::Si3;
    case 15:
      return classify_phosphorus(atom_idx, coord);
    case 16:
      return classify_sulfur(atom_idx, bonded, coord);
    case 17:
      return UFFAtomType::Cl;
    case 18:
      return UFFAtomType::Ar4_4;
    case 19:
      return UFFAtomType::K;
    case 20:
      return UFFAtomType::Ca6_2;
    case 21:
      return UFFAtomType::Sc3_3;
    case 22:
      return (coord == 6) ? UFFAtomType::Ti6_4 : UFFAtomType::Ti3_4;
    case 23:
      return UFFAtomType::V_3_5;
    case 24:
      return UFFAtomType::Cr6_3;
    case 25:
      return UFFAtomType::Mn6_2;
    case 26:
      return (coord == 6) ? UFFAtomType::Fe6_2 : UFFAtomType::Fe3_2;
    case 27:
      return UFFAtomType::Co6_3;
    case 28:
      return UFFAtomType::Ni4_2;
    case 29:
      return UFFAtomType::Cu3_1;
    case 30:
      return UFFAtomType::Zn3_2;
    case 31:
      return UFFAtomType::Ga3_3;
    case 32:
      return UFFAtomType::Ge3;
    case 33:
      return UFFAtomType::As3_3;
    case 34:
      return UFFAtomType::Se3_2;
    case 35:
      return UFFAtomType::Br;
    case 36:
      return UFFAtomType::Kr4_4;
    case 37:
      return UFFAtomType::Rb;
    case 38:
      return UFFAtomType::Sr6_2;
    case 39:
      return UFFAtomType::Y_3_3;
    case 40:
      return UFFAtomType::Zr3_4;
    case 41:
      return UFFAtomType::Nb3_5;
    case 42:
      return (coord == 6) ? UFFAtomType::Mo6_6 : UFFAtomType::Mo3_6;
    case 43:
      return UFFAtomType::Tc6_5;
    case 44:
      return UFFAtomType::Ru6_2;
    case 45:
      return UFFAtomType::Rh6_3;
    case 46:
      return UFFAtomType::Pd4_2;
    case 47:
      return UFFAtomType::Ag1_1;
    case 48:
      return UFFAtomType::Cd3_2;
    case 49:
      return UFFAtomType::In3_3;
    case 50:
      return UFFAtomType::Sn3;
    case 51:
      return UFFAtomType::Sb3_3;
    case 52:
      return UFFAtomType::Te3_2;
    case 53:
      return UFFAtomType::I;
    case 54:
      return UFFAtomType::Xe4_4;
    case 55:
      return UFFAtomType::Cs;
    case 56:
      return UFFAtomType::Ba6_2;
    case 57 ... 71:
      return classify_lanthanide(z);
    case 72:
      return UFFAtomType::Hf3_4;
    case 73:
      return UFFAtomType::Ta3_5;
    case 74:
      return (coord == 6)   ? UFFAtomType::W_6_6
             : (coord == 4) ? UFFAtomType::W_3_4
                            : UFFAtomType::W_3_6;
    case 75:
      return (coord == 6) ? UFFAtomType::Re6_5 : UFFAtomType::Re3_7;
    case 76:
      return UFFAtomType::Os6_6;
    case 77:
      return UFFAtomType::Ir6_3;
    case 78:
      return UFFAtomType::Pt4_2;
    case 79:
      return UFFAtomType::Au4_3;
    case 80:
      return UFFAtomType::Hg1_2;
    case 81:
      return UFFAtomType::Tl3_3;
    case 82:
      return UFFAtomType::Pb3;
    case 83:
      return UFFAtomType::Bi3_3;
    case 84:
      return UFFAtomType::Po3_2;
    case 85:
      return UFFAtomType::At;
    case 86:
      return UFFAtomType::Rn4_4;
    case 87:
      return UFFAtomType::Fr;
    case 88:
      return UFFAtomType::Ra6_2;
    case 89:
      return UFFAtomType::Ac6_3;
    case 90:
      return UFFAtomType::Th6_4;
    case 91:
      return UFFAtomType::Pa6_4;
    case 92:
      return UFFAtomType::U_6_4;
    case 93:
      return UFFAtomType::Np6_4;
    case 94:
      return UFFAtomType::Pu6_4;
    case 95:
      return UFFAtomType::Am6_4;
    case 96:
      return UFFAtomType::Cm6_3;
    case 97:
      return UFFAtomType::Bk6_3;
    case 98:
      return UFFAtomType::Cf6_3;
    case 99:
      return UFFAtomType::Es6_3;
    case 100:
      return UFFAtomType::Fm6_3;
    case 101:
      return UFFAtomType::Md6_3;
    case 102:
      return UFFAtomType::No6_3;
    case 103:
      return UFFAtomType::Lw6_3;
    default:
      return UFFAtomType::UNKNOWN;
    }
  }

private:
  const Topology &m_topology;

  int get_valence_electrons(int z) const {
    switch (z) {
    case 15:
    case 33:
    case 51:
    case 83:
      return 5; // Group 15 (pnictogens)
    case 16:
    case 34:
    case 52:
    case 84:
      return 6; // Group 16 (chalcogens)
    case 35:
    case 53:
    case 85:
      return 7; // Group 17 (halogens)
    case 36:
    case 54:
    case 86:
      return 8; // Group 18 (noble gases)
    default:
      return 0;
    }
  }

  int calculate_coordination(size_t atom_idx, int z, int val_electrons,
                             int explicit_coord) const {
    const auto &atoms = m_topology.atoms();
    const Atom &atom = atoms[atom_idx];

    int formal_charge = 0; // Could extract from atom if available
    int valence = explicit_coord;
    int lone_pairs = static_cast<int>(
        std::ceil((val_electrons - formal_charge - valence) / 2.0));
    int total_coord = explicit_coord + lone_pairs;

    if (z == 16 && count_free_oxygens(atom_idx) == 3) {
      return 2; // SO3, planar
    }

    if (lone_pairs == 0 && explicit_coord == 3 && valence == 6) {
      return 2; // Hexavalent planar
    }

    if (lone_pairs == 0 && explicit_coord == 7) {
      return 7; // Pentagonal bipyramidal
    }

    if (explicit_coord > 4) {
      return explicit_coord;
    }

    int coord_diff = val_electrons - explicit_coord;
    if (std::abs(coord_diff) > 2) {
      return explicit_coord - 1;
    }

    return total_coord;
  }

  int count_free_oxygens(size_t atom_idx) const {
    const auto &atoms = m_topology.atoms();
    auto bonded = m_topology.get_bonded_atoms(atom_idx);
    int count = 0;
    for (size_t neighbor_idx : bonded) {
      if (atoms[neighbor_idx].atomic_number() == 8) {
        count++;
      }
    }
    return count;
  }

  bool is_aromatic(size_t atom_idx) const {

    int z = m_topology.atoms()[atom_idx].atomic_number();
    if (z != 6 & z != 7 & z != 8 & z != 16) {
      return false;
    }

    core::AtomGraph graph = m_topology.get_atom_graph();

    return graph.is_in_cycle(atom_idx, 8);
  }

  UFFAtomType classify_hydrogen(size_t atom_idx,
                                const std::vector<size_t> &bonded) const {
    if (bonded.size() == 2) {
      const auto &atoms = m_topology.atoms();
      if (atoms[bonded[0]].atomic_number() == 5 &&
          atoms[bonded[1]].atomic_number() == 5) {
        return UFFAtomType::H_b;
      }
    }
    return UFFAtomType::H_;
  }

  UFFAtomType classify_carbon(size_t atom_idx,
                              const std::vector<size_t> &bonded,
                              int coord) const {
    if (coord == 2)
      return UFFAtomType::C_1;
    if (coord == 3)
      return is_aromatic(atom_idx) ? UFFAtomType::C_R : UFFAtomType::C_2;
    if (coord == 4)
      return UFFAtomType::C_3;
    return UFFAtomType::UNKNOWN;
  }

  UFFAtomType classify_nitrogen(size_t atom_idx,
                                const std::vector<size_t> &bonded,
                                int coord) const {
    if (coord == 1)
      return UFFAtomType::N_1;
    if (coord == 2)
      return is_aromatic(atom_idx) ? UFFAtomType::N_R : UFFAtomType::N_2;
    if (coord == 3 || coord == 4) {
      return is_aromatic(atom_idx) ? UFFAtomType::N_R : UFFAtomType::N_3;
    }
    return UFFAtomType::UNKNOWN;
  }

  UFFAtomType classify_oxygen(size_t atom_idx,
                              const std::vector<size_t> &bonded,
                              int coord) const {
    const auto &atoms = m_topology.atoms();

    if (coord == 1)
      return UFFAtomType::O_2;

    if (coord == 2) {
      if (is_aromatic(atom_idx))
        return UFFAtomType::O_R;

      for (size_t neighbor_idx : bonded) {
        if (atoms[neighbor_idx].atomic_number() == 14) {
          return UFFAtomType::O_3_2;
        }
      }

      return UFFAtomType::O_3;
    }

    return UFFAtomType::UNKNOWN;
  }

  UFFAtomType classify_phosphorus(size_t atom_idx, int coord) const {
    if (coord == 5)
      return UFFAtomType::P_3_5;
    if (coord == 4)
      return UFFAtomType::P_3_q;
    return UFFAtomType::P_3_3;
  }

  UFFAtomType classify_sulfur(size_t atom_idx,
                              const std::vector<size_t> &bonded,
                              int coord) const {
    if (coord == 1)
      return UFFAtomType::S_2;
    if (coord == 2) {
      return is_aromatic(atom_idx) ? UFFAtomType::S_R : UFFAtomType::S_3_2;
    }
    if (coord == 4)
      return UFFAtomType::S_3_4;
    if (coord == 6)
      return UFFAtomType::S_3_6;
    return UFFAtomType::S_3_2;
  }

  UFFAtomType classify_lanthanide(int z) const {
    switch (z) {
    case 57:
      return UFFAtomType::La3_3;
    case 58:
      return UFFAtomType::Ce6_3;
    case 59:
      return UFFAtomType::Pr6_3;
    case 60:
      return UFFAtomType::Nd6_3;
    case 61:
      return UFFAtomType::Pm6_3;
    case 62:
      return UFFAtomType::Sm6_3;
    case 63:
      return UFFAtomType::Eu6_3;
    case 64:
      return UFFAtomType::Gd6_3;
    case 65:
      return UFFAtomType::Tb6_3;
    case 66:
      return UFFAtomType::Dy6_3;
    case 67:
      return UFFAtomType::Ho6_3;
    case 68:
      return UFFAtomType::Er6_3;
    case 69:
      return UFFAtomType::Tm6_3;
    case 70:
      return UFFAtomType::Yb6_3;
    case 71:
      return UFFAtomType::Lu6_3;
    default:
      return UFFAtomType::UNKNOWN;
    }
  }
};
} // namespace trajan::energy
