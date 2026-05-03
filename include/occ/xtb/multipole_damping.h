#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

namespace occ::xtb {

class Gfn2Parameters;

// CN-dependent atomic radii used by the anisotropic multipole damping in
// xtb. Smoothly interpolates between the element's `multiRad` and
// `aesrmax` based on the deviation of CN from the "valence CN":
//   r(CN) = r0 + (rmax - r0) / (1 + exp(-aesexp * (CN - valenceCN - aesshift)))
Vec multipole_radii(const std::vector<core::Atom> &atoms, const Vec &cn,
                    const Gfn2Parameters &params);

// Pair-wise damped Coulomb tables for multipole-multipole interactions.
//   gab_n(i,j) = damp_n * R_ij^{-n}
//   damp_n = 1 / (1 + 6 * (rco / R_ij)^kdmp_n),  rco = (r_i + r_j) / 2
// Returns two symmetric n_atom × n_atom matrices for n = 3 and n = 5.
struct DampedCoulomb {
  Mat gab3;
  Mat gab5;
};

DampedCoulomb damped_multipole_coulomb(const std::vector<core::Atom> &atoms,
                                       const Vec &mp_radii,
                                       const Gfn2Parameters &params);

} // namespace occ::xtb
