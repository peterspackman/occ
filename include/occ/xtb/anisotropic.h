#pragma once
#include <array>
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

// Explicit ∂E_AES/∂R at frozen (q, m, mp_radii) plus the per-atom
// ∂E_AES/∂CN_A chain through R_co = ½(mp_radii(i) + mp_radii(j)). Closed-form
// pair-loop gradient that mirrors `anisotropic_energy`'s pair sum term-by-
// term (charge-dipole · g3, charge-quadrupole · g5, dipole-dipole · g5) plus
// the kernel derivatives ∂g3/∂R, ∂g5/∂R. Skips the on-site polarization (no
// explicit R-dependence at frozen multipoles) and the SCC density response
// (frozen q, μ, Q).
//
// Pass an empty / zero-size `dmp_radii_dcn` to skip the CN chain (e.g., to
// reproduce a frozen-radii gradient). Otherwise its length must equal nat.
struct AnisotropicPairGradient {
  Mat3N grad_explicit;  // ∂E_AES/∂R_iα at frozen mp_radii (3 × N, Ha/Bohr).
  Vec   dE_dcn;         // ∂E_AES/∂CN_A per atom (length N, Ha).
};

AnisotropicPairGradient
anisotropic_pair_gradient_with_dcn(const std::vector<core::Atom> &atoms,
                                    const Vec &q, const CammMoments &m,
                                    const Vec &mp_radii,
                                    const Vec &dmp_radii_dcn,
                                    const Gfn2Parameters &params);

// Per-atom potentials acting on charges (vs), atomic dipoles (vd), and
// atomic quadrupoles (vq) — the variational conjugates of (q, μ_xtb,
// Q_xtb) at the converged density. vq is laid out in `qpint` order
// (xx, yy, zz, xy, xz, yz). The sign convention is
//   vd = +∂E_aniso/∂μ_xtb,  vq = +∂E_aniso/∂Q_xtb_qpint,  vs = +∂E_aniso/∂q
// (Hartree units throughout). The actual builder is
// `anisotropic_potentials_ewald` (multipole_ewald.h) — works for both
// molecular and periodic systems via `build_molecular_multipole_tensors`
// or `build_multipole_ewald_tensors` respectively.
struct AnisotropicPotentials {
  Vec vs;     // n_atoms
  Mat3N vd;   // 3 × n_atoms
  Mat vq;     // 6 × n_atoms (xx, yy, zz, xy, xz, yz)
};

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

// Pulay-like nuclear gradient piece from the SCC density's response to AO
// multipole integrals. At fixed P, the converged μ_A and Q_A both depend on
// R via the per-atom-centered AO matrices D_bra and Q_bra (see
// `compute_camm_moments_periodic`). This function returns
//
//   ∂(Σ_A vd_α(A) · μ_A_α  +  Σ_A Σ_l vq_l(A) · Q_A_l) / ∂R |_{P fixed}
//
// using the bra-side AO derivative integrals (`int1e_irp` and `int1e_irrp`,
// wrapped in `dipole_ao_grad` / `quadrupole_ao_grad`) and the centering chain
// through R_(atom of ν) · S, R_(atom of ν) · D, etc.  The result is added
// directly to the gradient by the caller.
//
// `D_origin0` and `Q_origin0` must be the BARE AO multipole matrices at common
// origin O = 0 (output of `dipole_ao_matrices` / `quadrupole_ao_matrices`),
// NOT the per-atom-centered Bra/Ket variants.  `Q_origin0` must be the raw
// Cartesian quadrupole (xx, xy, xz, yy, yz, zz) — the traceless transform is
// applied internally to match the CAMM partition.
//
// `ovlp_grad` is `engine.one_electron_operator_grad(Op::overlap)` (libcint
// convention: `ovlp_grad[γ](μ, ν) = ⟨∂_γ φ_μ | φ_ν⟩`).  `D_origin0` is the
// bare AO dipole matrix at common origin O = 0 (output of
// `dipole_ao_matrices`); the Q-bra centering chain doesn't actually need
// Q_origin0 (only ∂Q/∂R, supplied via `irrp`, plus the algebraic
// δ_αγ·D_β + δ_βγ·D_α IBP terms).
Mat3N anisotropic_density_pulay_gradient(
    const std::vector<core::Atom> &atoms,
    const std::vector<int> &bf_to_atom,
    const Mat &P, const Mat &S,
    const MatTriple &D_origin0,
    const std::array<MatTriple, 3> &irp,        // dipole_ao_grad output
    const std::array<MatTriple, 6> &irrp,       // quadrupole_ao_grad output
    const MatTriple &ovlp_grad,
    const AnisotropicPotentials &pot);

} // namespace occ::xtb
