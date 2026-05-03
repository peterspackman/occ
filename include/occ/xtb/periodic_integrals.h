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
struct PeriodicAOMatrices {
  std::vector<Mat> S;   // overlap, indexed by translation
  std::vector<Mat> H0;  // extended Hückel core Hamiltonian
};

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

// Bloch-sum a set of per-T real matrices into a complex matrix at k (Bohr⁻¹):
//   M(k) = Σ_T M^T exp(i k · T)
CMat bloch_sum(const std::vector<Mat> &M_per_T,
               const std::vector<LatticeImage> &translations, const Vec3 &k);

// Bloch-sum at the Γ point — the sum is real-valued since exp(i·0) = 1, so
// returns a plain Mat for efficiency.
Mat bloch_sum_gamma(const std::vector<Mat> &M_per_T);

} // namespace occ::xtb
