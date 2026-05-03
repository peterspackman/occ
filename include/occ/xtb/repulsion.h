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

// Periodic variant: E = ½ Σ_T Σ_{i,j} V(r_i - r_j - T) excluding the
// (T=0, i=j) self-pair. Each translation is from `translations`.
struct LatticeImage;
double repulsion_energy_periodic(
    const std::vector<core::Atom> &atoms, const Gfn2Parameters &params,
    const std::vector<LatticeImage> &translations);

// Analytical gradient of the repulsion energy. Returns (E, dE/dR) where
// dE/dR is a 3 × N matrix in Hartree/Bohr.
struct RepulsionEnergyGradient {
  double energy;
  Mat3N gradient;
};

RepulsionEnergyGradient
repulsion_energy_and_gradient(const std::vector<core::Atom> &atoms,
                              const Gfn2Parameters &params);

} // namespace occ::xtb
