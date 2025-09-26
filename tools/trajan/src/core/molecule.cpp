#include <trajan/core/molecule.h>

namespace trajan::core {

static std::vector<occ::core::Atom>
convert_atoms(const std::vector<Atom> &atoms) {
  std::vector<occ::core::Atom> occ_atoms;
  occ_atoms.reserve(atoms.size());

  for (const auto &atom : atoms) {
    const auto &pos = atom.position();
    occ::core::Atom occ_atom(atom.atomic_number(), pos[0], pos[1], pos[2]);
    occ_atoms.push_back(occ_atom);
  }

  return occ_atoms;
}

EnhancedMolecule::EnhancedMolecule(const std::vector<Atom> &atoms)
    : occ::core::Molecule(convert_atoms(atoms)), enhanced_atoms(atoms) {}

}; // namespace trajan::core
