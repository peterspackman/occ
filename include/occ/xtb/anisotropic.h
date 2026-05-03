#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

namespace occ::xtb {

class Gfn2Parameters;
struct CammMoments;
struct DampedCoulomb;

struct AnisotropicEnergy {
  double aes;       // long-range multipole-multipole interaction (xtb "aes")
  double polariz;   // on-site polarization (xtb "anisotropic XC")
  double total() const { return aes + polariz; }
};

// Compute the anisotropic electrostatic energy from CAMM moments.
// `q` is the per-atom positive partial charge (xtb convention: q_A = z_A − pop).
// `damped` provides the gab3 / gab5 damped Coulomb tables.
AnisotropicEnergy
anisotropic_energy(const std::vector<core::Atom> &atoms, const Vec &q,
                   const CammMoments &m, const DampedCoulomb &damped,
                   const Gfn2Parameters &params);

// Per-atom potentials acting on charges (vs), atomic dipoles (vd), and
// atomic quadrupoles (vq). vq is laid out in `qpint` order (xx, yy, zz, xy,
// xz, yz) — this matches the storage of `quadrupole_ao_matrices(...)` after
// remapping inside `apply_anisotropic_h1`. Hartree units throughout.
struct AnisotropicPotentials {
  Vec vs;     // n_atoms
  Mat3N vd;   // 3 × n_atoms
  Mat vq;     // 6 × n_atoms (xx, yy, zz, xy, xz, yz)
};

AnisotropicPotentials
anisotropic_potentials(const std::vector<core::Atom> &atoms, const Vec &q,
                       const CammMoments &m, const DampedCoulomb &damped,
                       const Gfn2Parameters &params);

// Add the CAMM-induced shift to the AO Fock matrix H (in Hartree). Inputs
// are the overlap, the dipole AO matrices D, the quadrupole AO matrices Q
// (in {xx, xy, xz, yy, yz, zz} order, i.e. as returned by
// `quadrupole_ao_matrices`), the bf→atom mapping, and the per-atom
// potentials. Adds to H in place.
void apply_anisotropic_h1(Mat &H, const Mat &S, const MatTriple &D,
                          const std::array<Mat, 6> &Q,
                          const std::vector<int> &bf_to_atom,
                          const AnisotropicPotentials &pot);

} // namespace occ::xtb
