#pragma once
#include <occ/core/atom.h>
#include <occ/gto/shell.h>

namespace occ::xtb {

class Gfn2Parameters;

// Build the GFN2 valence basis (STO-NG expansion) for the given atoms.
// Returns a spherical AOBasis named "gfn2-xtb".
//
// Throws std::runtime_error if any atom's element is missing from `params`.
gto::AOBasis build_aobasis(const std::vector<core::Atom> &atoms,
                           const Gfn2Parameters &params);

} // namespace occ::xtb
