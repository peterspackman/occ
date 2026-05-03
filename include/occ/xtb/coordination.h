#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

namespace occ::xtb {

// xtb's "GFN" coordination number flavor (used by GFN1/GFN2 H0 diagonal):
//   count_ij = expCount(10, r, r0) * expCount(20, r, r0 + 2)
//   expCount(k, r, r0) = 1 / (1 + exp(-k * (r0/r - 1)))
//   r0 = (R_cov_i + R_cov_j)
// Atom positions are taken in **Bohr** (matching occ::core::Atom).
// Pyykko-D3 covalent radii are used internally, pre-scaled by 4/3 ·  a₀/Å.
Vec gfn_coordination_numbers(const std::vector<core::Atom> &atoms);

// Periodic variant: sums over all lattice translations (excluding the
// (T=0, i=j) self-pair). `lattice_translations` provides the T-vectors in
// Bohr; the (0,0,0) entry must be present.
struct LatticeImage;
Vec gfn_coordination_numbers_periodic(
    const std::vector<core::Atom> &atoms,
    const std::vector<LatticeImage> &lattice_translations);

// Pauling-EN-weighted variant ("cov" flavor) — included for completeness;
// not used by H0 in GFN2 but useful for repulsion / multipole damping.
//   count_ij' = ε_AB · count_ij,  ε_AB = k4 · exp(-(|EN_A - EN_B| - k5)^2 / k6²)
// (Currently unused; left as a stub for Phase 2/3 extensions.)

} // namespace occ::xtb
