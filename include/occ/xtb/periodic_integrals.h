#pragma once
#include <complex>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/xtb/periodic.h>
#include <vector>

namespace occ::xtb {

class Gfn2Parameters;

using CMat = Eigen::MatrixXcd;

// Per-translation real-space matrices.
//
// For a real-space integral matrix M (overlap, dipole, H0, ...) the periodic
// generalisation in tight-binding is built from blocks
//   M^T_{μν}  =  <φ_μ(r) | Ô | φ_ν(r - T)>
// where μ runs over basis functions of the central cell and ν runs over the
// same basis functions translated by T. The Bloch-summed matrix at k is
//   M(k) = Σ_T M^T exp(i k · T)
// (real-symmetric for k=0, complex-Hermitian otherwise).

// Compute the per-T overlap blocks. The (T=0) entry is the standard
// molecular overlap of the central basis.
std::vector<Mat>
periodic_overlap_blocks(const PeriodicSystem &sys,
                        const Gfn2Parameters &params,
                        const std::vector<LatticeImage> &translations);

// Compute the per-T H0 blocks. Requires the per-T overlap blocks (avoids
// recomputation) and the periodic CN vector (same length as central atoms).
std::vector<Mat>
periodic_h0_blocks(const PeriodicSystem &sys, const Gfn2Parameters &params,
                   const std::vector<LatticeImage> &translations,
                   const std::vector<Mat> &S_per_T, const Vec &cn);

// Bra/Ket atom-centered multipole AO matrices, dftbplus convention. For each
// AO pair (μ, ν) with atom_of(μ) = A_μ (cell 0) and atom_of(ν) = A_ν (looped
// over images of T):
//   Ket(μ, ν) = AO multipole integral with origin at R_{A_μ} (cell-0 row
//               atom) — so the integral is "atom-of-row-centered".
//   Bra(μ, ν) = AO multipole integral with origin at R_{A_ν} + T (image-T
//               column atom) — "atom-of-col-image-centered".
// Bloch-summed at Γ.
//
// Periodic CAMM partition (mirrors dftbplus's `getAtomicMultipolePopulation`):
//   mpat[A_μ] += Σ_{ν} P(μ, ν) · Ket(μ, ν)  for each AO row μ
//   mpat[A_ν] += Σ_{μ} P(μ, ν) · Bra(μ, ν)  for each AO column ν
struct PeriodicMultipoleAO {
  MatTriple D_ket;          // dipole AO, atom-of-row-centered (for CAMM partition)
  MatTriple D_bra;          // dipole AO, atom-of-col-image-centered
  std::array<Mat, 6> Q_ket; // quadrupole AO, atom-of-row-centered (traceless)
  std::array<Mat, 6> Q_bra; // quadrupole AO, atom-of-col-image-centered (traceless)
};

PeriodicMultipoleAO build_periodic_multipole_ao(
    const PeriodicSystem &sys, const Gfn2Parameters &params,
    const std::vector<LatticeImage> &translations);

// Per-translation real-space multipole AO blocks. translations[i] corresponds
// to all per_T[i] entries. Same Bra/Ket convention as `PeriodicMultipoleAO`,
// just kept un-summed so callers can Bloch-sum at arbitrary k. Q_ket / Q_bra
// already have the traceless-Cartesian transform applied per-T (linear, so
// algebraically the same as applying once after summing).
struct PeriodicMultipoleAOBlocks {
  std::vector<MatTriple> D_ket;          // dipole AO, atom-of-row-centered
  std::vector<MatTriple> D_bra;          // dipole AO, atom-of-col-image-centered
  std::vector<std::array<Mat, 6>> Q_ket; // traceless quadrupole, row-centered
  std::vector<std::array<Mat, 6>> Q_bra; // traceless quadrupole, col-centered
};

PeriodicMultipoleAOBlocks build_periodic_multipole_ao_blocks(
    const PeriodicSystem &sys, const Gfn2Parameters &params,
    const std::vector<LatticeImage> &translations);

// Complex 3-component AO matrix — Bloch sum of a per-T MatTriple at k.
struct CMatTriple {
  CMat x, y, z;
};

CMatTriple bloch_sum_triple(const std::vector<MatTriple> &per_T,
                             const std::vector<LatticeImage> &translations,
                             const Vec3 &k);

std::array<CMat, 6>
bloch_sum_array6(const std::vector<std::array<Mat, 6>> &per_T,
                 const std::vector<LatticeImage> &translations, const Vec3 &k);

// In-place transform from full Cartesian quadrupole AO matrices (xx, xy, xz,
// yy, yz, zz) to traceless-Cartesian, matching tblite's
// integral/multipole.f90 convention:
//
//   Q'_xx = 1.5·Q_xx - 0.5·(Q_xx + Q_yy + Q_zz)
//   Q'_xy = 1.5·Q_xy   ;   Q'_yy = 1.5·Q_yy - 0.5·(...)
//   Q'_xz = 1.5·Q_xz   ;   Q'_yz = 1.5·Q_yz
//   Q'_zz = 1.5·Q_zz - 0.5·(...)
//
// Applied per AO pair (μ, ν). For the H1 contribution `0.5·Q_AO·vq[A]` to
// match tblite's update, Q_AO must be the traceless form (otherwise the
// trace component of vq couples to the non-zero trace of Cartesian Q).
void apply_traceless_quadrupole_transform(std::array<Mat, 6> &Q);

// Apply per-atom origin centering (Bra/Ket convention, matches tblite) to
// raw common-origin-0 AO multipole matrices. `D_origin0` and `Q_origin0`
// are the outputs of `dipole_ao_matrices` / `quadrupole_ao_matrices` (the
// latter in raw {xx, xy, xz, yy, yz, zz} Cartesian form, NOT traceless yet).
// The returned struct's Q_ket / Q_bra are traceless-Cartesian, ready for
// `compute_camm_moments_periodic` / `apply_anisotropic_h1_periodic`.
PeriodicMultipoleAO
center_multipole_ao(const std::vector<core::Atom> &atoms,
                    const std::vector<int> &bf_to_atom,
                    const Mat &S,
                    const MatTriple &D_origin0,
                    const std::array<Mat, 6> &Q_origin0);

// Molecular variant: returns the same struct as the periodic build but with
// T = 0 only (no lattice). D_ket / D_bra carry the per-row / per-col atomic
// origin shifts; the molecular SCC can use these with the same
// compute_camm_moments_periodic + apply_anisotropic_h1_periodic pipeline as
// the periodic SCC. This keeps both paths on tblite's atom-centered
// convention end-to-end. The (atoms, params) overload builds its own
// IntegralEngine internally; callers that already have one (the SCC path,
// the analytical gradient) should prefer `center_multipole_ao` directly.
PeriodicMultipoleAO
build_molecular_multipole_ao(const std::vector<core::Atom> &atoms,
                              const Gfn2Parameters &params);

// Bloch-sum a set of per-T real matrices into a complex matrix at k (Bohr⁻¹):
//   M(k) = Σ_T M^T exp(i k · T)
CMat bloch_sum(const std::vector<Mat> &M_per_T,
               const std::vector<LatticeImage> &translations, const Vec3 &k);

// Bloch-sum at the Γ point — the sum is real-valued since exp(i·0) = 1, so
// returns a plain Mat for efficiency.
Mat bloch_sum_gamma(const std::vector<Mat> &M_per_T);

// Solve the complex Hermitian generalized eigenvalue problem
//   H · C = ε · S · C
// via canonical orthogonalization (X = U·s^(-1/2) where S = U·s·U^H), then
// standard Hermitian diagonalization of X^H·H·X. Returns ascending eigen-
// values and eigenvectors normalized so that C^H · S · C = I.
struct CGenSolveResult {
  Vec eigenvalues;
  CMat eigenvectors;
};
CGenSolveResult solve_generalized_hermitian(const CMat &H, const CMat &S,
                                             double s_eps = 1e-10);

} // namespace occ::xtb
