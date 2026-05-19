#pragma once
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::xtb {

// Auto-pick the Ewald screening parameter α from the cell volume:
//   α = √π / V^(1/3)   (1/Bohr)
// This roughly balances real-space erfc cost vs reciprocal-space cost on
// typical molecular-crystal cells. Used by both periodic γ and AES Ewald
// builders so they always agree on α and lattice/G truncation when called on
// the same geometry.
double auto_ewald_alpha(double volume_bohr3);

// Real-space cutoff for the erfc/erf-based Ewald sum at tolerance `tol`:
//   erfc(αR)/R ~ exp(-α²R²) / (√π · αR · R) ≤ tol  →  R ≥ √(-ln tol)/α
// Returns the cutoff in Bohr, with a safety margin of +1 Bohr.
double ewald_real_cutoff(double alpha, double tol);

// Reciprocal-space cutoff in 1/Bohr for the same tolerance:
//   exp(-G²/(4α²)) ≤ tol → G ≥ 2α·√(-ln tol).
double ewald_recip_cutoff(double alpha, double tol);

// Enumerate G vectors in the reciprocal lattice with 0 < |G| ≤ recip_cutoff.
// `reciprocal_bohr` columns are b₁, b₂, b₃ (each a Vec3 in 1/Bohr).
std::vector<Vec3> enumerate_g_vectors(const Mat3 &reciprocal_bohr,
                                       double recip_cutoff);

} // namespace occ::xtb
