#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/xtb/periodic.h>
#include <occ/xtb/periodic_integrals.h>

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
// remapping inside `apply_anisotropic_h1_periodic`. Hartree units throughout.
struct AnisotropicPotentials {
  Vec vs;     // n_atoms
  Mat3N vd;   // 3 × n_atoms
  Mat vq;     // 6 × n_atoms (xx, yy, zz, xy, xz, yz)
};

AnisotropicPotentials
anisotropic_potentials(const std::vector<core::Atom> &atoms, const Vec &q,
                       const CammMoments &m, const DampedCoulomb &damped,
                       const Gfn2Parameters &params);

// Add the CAMM-induced shift to the AO Fock matrix H. Uses Ket (atom-of-row-centered)
// on the row side and Bra (atom-of-col-image-centered) on the column side,
// each paired with its respective atomic potential — symmetric averaging is
// no longer correct when the AOs sit at different absolute positions in the
// Bloch sum.
void apply_anisotropic_h1_periodic(
    Mat &H, const Mat &S,
    const MatTriple &D_ket, const MatTriple &D_bra,
    const std::array<Mat, 6> &Q_ket, const std::array<Mat, 6> &Q_bra,
    const std::vector<int> &bf_to_atom,
    const AnisotropicPotentials &pot);

// Complex variant of apply_anisotropic_h1_periodic for the k-point H(k). Same
// formula, just with Bloch-summed (complex) AO matrices at k. Hermiticity of
// H1(k) is preserved because D_bra(k) = D_ket(k)^H by construction.
void apply_anisotropic_h1_kpoint(
    CMat &H, const CMat &S,
    const CMatTriple &D_ket, const CMatTriple &D_bra,
    const std::array<CMat, 6> &Q_ket, const std::array<CMat, 6> &Q_bra,
    const std::vector<int> &bf_to_atom,
    const AnisotropicPotentials &pot);

} // namespace occ::xtb
