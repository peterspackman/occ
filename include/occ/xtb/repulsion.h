#pragma once
#include <occ/core/atom.h>

namespace occ::xtb {

class Gfn2Parameters;

// GFN2 short-range repulsion energy:
//   E_rep = sum_{i<j} (Zeff_i Zeff_j / r_ij) * exp(-sqrt(α_i α_j) r_ij^kExp)
// where kExp = globals.kexp (=1.5) for heavy-heavy pairs and kexplight (=1.0)
// when either atom is H or He. Atom positions are in Bohr.
double repulsion_energy(const std::vector<core::Atom> &atoms,
                        const Gfn2Parameters &params);

} // namespace occ::xtb
