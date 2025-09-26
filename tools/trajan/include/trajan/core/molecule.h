#pragma once
#include <occ/core/molecule.h>
#include <string>
#include <trajan/core/atom.h>
// #include <trajan/core/element.h>
// #include <trajan/core/graph.h>
#include <occ/core/bondgraph.h>
// #include <trajan/core/linear_algebra.h>
#include <vector>

namespace trajan::core {

using Atom = trajan::core::EnhancedAtom;

class EnhancedMolecule : public occ::core::Molecule {
public:
  std::string type;
  int index;
  std::optional<int> uindex;
  std::vector<Atom> enhanced_atoms;

  inline bool operator==(const EnhancedMolecule &rhs) const {
    return this->index == rhs.index;
  };

  EnhancedMolecule(const std::vector<Atom> &atoms);
};

}; // namespace trajan::core
