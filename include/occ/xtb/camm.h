#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

namespace occ::xtb {

// Cumulative atomic multipole moments (CAMM) — atomic dipoles and traceless
// Cartesian quadrupoles obtained by Mulliken-like partitioning of the density
// matrix using dipole and quadrupole AO integrals at the global origin.
//
// Storage matches xtb's `aespot.f90` convention so the energy/H1 routines
// can use the data directly:
//   dipm  : Mat3N  (3 × n_atoms),   [x, y, z]_atom
//   qp    : Mat    (6 × n_atoms),   [xx, xy, yy, xz, yz, zz]_atom (traceless)
struct CammMoments {
  Mat3N dipm;
  Mat qp; // 6 × n_atoms
};

// Compute CAMM dipoles and (traceless Cartesian) quadrupoles for the atoms,
// given the AO density matrix P (in the same basis as the integrals), the
// overlap matrix S, the dipole AO matrices D (origin = (0,0,0)), and the
// quadrupole AO matrices Q (in {xx, xy, xz, yy, yz, zz} order, origin = 0).
CammMoments compute_camm_moments(const std::vector<core::Atom> &atoms,
                                 const std::vector<int> &bf_to_atom,
                                 const Mat &P, const Mat &S,
                                 const MatTriple &D,
                                 const std::array<Mat, 6> &Q);

} // namespace occ::xtb
