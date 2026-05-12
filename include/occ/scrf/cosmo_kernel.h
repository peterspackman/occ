#pragma once
#include <Eigen/LU>
#include <occ/core/linear_algebra.h>

/// Self-consistent reaction-field (SCRF) primitives shared between the
/// HF/DFT-side `occ::solvent::COSMO` driver and the xTB-side CPCM-X / SMD
/// implicit-solvation models. The intent is that this module owns the COSMO /
/// CPCM math (cavity A matrix, atom↔cavity B operator, pre-solved response,
/// frozen-cavity analytical gradient) while higher-level wrappers in
/// `occ::solvent` and `occ::xtb` (and the forthcoming `occ::scrf::ReactionFieldEngine`)
/// glue this kernel to the rest of their pipeline (cavity build, parameter
/// lookup, energy accounting, gradient flow).
///
/// Everything here is parameterised on raw Eigen arrays so the library
/// depends only on `occ_core` — no dependency on `occ_solvent` or any
/// particular cavity representation.
namespace occ::scrf::detail {

/// Build the dense COSMO A matrix on a discretised cavity:
///   A(i, j) = 1/|r_i − r_j|, off-diagonal
///   A(i, i) = 1.07·√(4π/S_i) ≈ 3.793051240937804 / √S_i
/// `cavity_points` is 3 × ncav (Bohr), `cavity_areas` is ncav (Bohr²).
Mat build_cosmo_A(const Mat3N &cavity_points, const Vec &cavity_areas);

/// Build the atom↔cavity Coulomb operator B(i, a) = 1/|r_i − R_a|.
/// Useful when the source potential at each cavity point is being driven by
/// point-like atomic charges (e.g. xTB Mulliken charges):
///   φ_i = (B · q_atom)_i
/// `cavity_points` is 3 × ncav, `atom_positions` is 3 × natom (both Bohr).
Mat build_atom_cavity_coulomb(const Mat3N &cavity_points,
                              const Mat3N &atom_positions);

/// Pre-solved CPCM/COSMO response operator on a cavity, driven by atom-
/// centred point-charge sources. Caches:
///
///   B(i, a)  = 1/|r_i − R_a|                                   (ncav × natom)
///   A(i, j)  = 1/|r_i − r_j|, diag 1.07·√(4π/S_i)              (ncav × ncav)
///   G        = −f(ε) · A^{-1} · B   (so σ = G · q)             (ncav × natom)
///   J_solv   = B^T · G              (symmetric, neg-def)       (natom × natom)
///
/// Once held, the per-iteration cost of evaluating the solvation shift
/// collapses to two GEMVs:
///   σ      = G · q
///   V_solv = J_solv · q
///   E_solv = ½ q · V_solv  =  ½ σ · φ
struct CosmoResponse {
  Mat B;
  Mat G;
  Mat J_solv;
};

/// Build the pre-solved response. `epsilon` is the solvent relative
/// permittivity; `x` selects the f(ε) convention in `f(ε) = (ε−1)/(ε+x)`:
/// `x = 0` is the CPCM ideal-conductor convention, `x = 0.5` is Klamt COSMO.
/// Empty cavity (`cavity_points.cols() == 0`) returns zero-sized matrices.
CosmoResponse build_cosmo_response(const Mat3N &atom_positions_bohr,
                                   const Mat3N &cavity_points,
                                   const Vec &cavity_areas, double epsilon,
                                   double x);

/// Frozen-cavity analytical gradient of the polarisation energy with respect
/// to atomic positions:
///
///   ∂E_es/∂R_c =  − Σ_{i: a_i=c} σ_i · g_i
///                + q_c · h_c
///                + (1/f(ε)) · Σ_{i: a_i=c} σ_i · t_i
///                + diagonal-A term (smooth cavity only — see below)
///
///   g_i = Σ_a  q_a · (r_i − R_a) / |r_i − R_a|³     (field at cavity i from q)
///   t_i = −Σ_{j≠i} σ_j · (r_i − r_j) / |r_i − r_j|³ (field at cavity i from σ)
///   h_c = Σ_i σ_i · (r_i − R_c) / |r_i − R_c|³      (field at atom c from σ)
///
/// With a boolean cavity (smoothing_width = 0), per-element areas are
/// geometry-independent and ∂A_ii/∂R = 0. With a smooth cavity, each per-
/// element area depends smoothly on every other atom's position through the
/// erf-based smoothstep weight, giving
///
///   ∂A_ii/∂R_c = −½ A_ii · ∂ln(weight_i)/∂R_c
///   ∂ln(weight_i)/∂R_c = Σ_{k ≠ a_i} (s'/s)|d_ik · ∂d_ik/∂R_c
///
/// where `s(d) = ½(1 + erf((d − t_k)/w))`, `t_k = r_k`. Pass `atom_radii_bohr`
/// and `smoothing_width_bohr > 0` to enable this term; leave
/// `smoothing_width_bohr = 0` to skip it (boolean-cavity default).
///
/// Returns 3 × N_atoms in Hartree/Bohr.
[[nodiscard]] Mat3N
cosmo_gradient_frozen(const Mat3N &atom_positions_bohr,
                      const Mat3N &cavity_points, const Vec &cavity_areas,
                      const IVec &cavity_atom_index, const Vec &atom_charges,
                      const Vec &sigma, double f_epsilon,
                      const Vec &atom_radii_bohr = Vec(),
                      double smoothing_width_bohr = 0.0);

} // namespace occ::scrf::detail
