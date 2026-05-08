#pragma once
#include <array>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/xtb/anisotropic.h>
#include <occ/xtb/periodic.h>
#include <vector>

namespace occ::xtb {

class Gfn2Parameters;
struct CammMoments;

// Pre-built atom-pair interaction tensors for periodic multipole ES, computed
// via Ewald split (real-space damped + erfc + reciprocal G-sum + self/back-
// ground corrections). Mirrors tblite's get_multipole_matrix_3d:
//
//   sd[α](i, j)        — charge → dipole field tensor
//   dd[α][β](i, j)     — dipole → dipole field tensor
//   sq[p](i, j)        — charge → quadrupole field tensor (p in OCC's qp order
//                         {xx, xy, yy, xz, yz, zz}, same as CammMoments::qp)
//
// `vec` in the kernel uses tblite's sign convention: `vec = R_i - R_j - T`,
// and the inner Ewald loop runs over T spanning the full set of lattice
// translations (real-space erfc cutoff). The reciprocal sum runs over G with
// |G| ≤ recip_cutoff.
//
// Geometry-cached: depends on atom positions, mp_radii, and the cell. NOT on
// charges or moments. Rebuild only when geometry changes.
struct MultipolePairTensors {
  std::array<Mat, 3> sd;
  std::array<std::array<Mat, 3>, 3> dd;
  std::array<Mat, 6> sq;
  double alpha;
  double real_cutoff;
  double recip_cutoff;
};

// Auto-pick alpha = sqrt(pi)/V^(1/3); pass alpha_user > 0 to override.
// `tol` controls erfc and reciprocal cutoffs (1e-10 is a safe default).
MultipolePairTensors
build_multipole_ewald_tensors(const PeriodicSystem &sys,
                              const Vec &mp_radii,
                              const Gfn2Parameters &params,
                              double tol = 1e-10,
                              double alpha_user = 0.0);

// Molecular variant: builds sd/dd/sq pair tensors with the same kernel as the
// Ewald build but with no lattice sum, no Ewald split, no reciprocal/self/
// background terms. Use for the molecular SCC path so it can share the same
// tensor-based potential / energy / H1 routines as the periodic path.
MultipolePairTensors
build_molecular_multipole_tensors(const std::vector<core::Atom> &atoms,
                                   const Vec &mp_radii,
                                   const Gfn2Parameters &params);

// Anisotropic ES energy (charge-dipole + charge-quadrupole + dipole-dipole)
// using the pre-built pair tensors. Adds the on-site polarization term
// (kernel·μ² + kernel·Σ Q²) per atom.
AnisotropicEnergy
anisotropic_energy_ewald(const std::vector<core::Atom> &atoms, const Vec &q,
                          const CammMoments &m,
                          const MultipolePairTensors &tensors,
                          const Gfn2Parameters &params);

// Per-atom potentials acting on charges (vs), atomic dipoles (vd), and
// atomic quadrupoles (vq) from the Ewald pair tensors. vq layout in xtb's
// qpint order {xx, yy, zz, xy, xz, yz}, matching the H1 update's vq layout.
// These are the strict tensor-contraction derivatives of the Ewald energy
// (no gauge corrections — the AO multipole integrals carry the per-atom
// origin shift, so the H1 routine consumes these potentials directly).
AnisotropicPotentials
anisotropic_potentials_ewald(const std::vector<core::Atom> &atoms,
                              const Vec &q, const CammMoments &m,
                              const MultipolePairTensors &tensors,
                              const Gfn2Parameters &params);

} // namespace occ::xtb
