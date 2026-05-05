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

// Periodic CAMM following dftbplus's `getAtomicMultipolePopulation`. Takes
// pre-built per-atom-centered Ket/Bra Bloch-summed multipole AO matrices (see
// `PeriodicMultipoleAO` in `periodic_integrals.h`):
//   Ket(μ, ν) — atom-of-row-centered (origin at R_{atom_of(μ)}, cell 0)
//   Bra(μ, ν) — atom-of-col-image-centered (origin at R_{atom_of(ν)} + T)
// Off-diagonal pair (j, i) with j < i contributes pij·Ket(j,i) to atom_of(j)
// and pij·Bra(j,i) to atom_of(i). Diagonal pair contributes pii·Ket(i,i) to
// atom_of(i) once. Sign is the molecular CAMM convention (m.dipm = -atomic
// dipole, electron sign).
//
// Reduces exactly to molecular `compute_camm_moments` at the molecular limit
// (T = 0 only): D_ket = D_origin0 - R_row·S, D_bra = D_origin0 - R_col·S, so
// the partition reproduces the molecular `xa·ps - pdm` formula per atom.
CammMoments compute_camm_moments_periodic(
    const std::vector<core::Atom> &atoms,
    const std::vector<int> &bf_to_atom,
    const Mat &P,
    const MatTriple &D_ket, const MatTriple &D_bra,
    const std::array<Mat, 6> &Q_ket, const std::array<Mat, 6> &Q_bra);

} // namespace occ::xtb
