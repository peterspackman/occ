#pragma once
#include <Eigen/LU>
#include <occ/core/linear_algebra.h>
#include <occ/solvent/surface.h>

namespace occ::xtb::cosmo {

/// Dense COSMO/CPCM cavity response, pre-solved against an atom-centered
/// point-charge source.
///
/// Shared by `CpcmXSolvationModel` and `SmdSolvationModel` (and any future
/// implicit-solvent variant that uses the same atom→cavity Coulomb +
/// classical-COSMO-A response). Solver-cached state:
///
///   B(i, a)  = 1 / |r_i − R_a|                                (ncav × natom)
///   A(i, j)  = 1 / |r_i − r_j| with diag 1.07·√(4π/S_i)       (ncav × ncav)
///   G        = −f(ε) · A^{-1} · B   (so σ = G · q)            (ncav × natom)
///   J_solv   = B^T · G              (symmetric, neg-def)      (natom × natom)
///
/// With these in hand the per-iteration cost of evaluating the solvation
/// shift collapses to two GEMVs:
///
///   σ      = G · q
///   V_solv = J_solv · q
///   E_solv = ½ q · V_solv = ½ σ · φ
struct Response {
  /// `B(i, a) = 1 / |r_i − R_a|` (ncav × natom). Kept around so callers can
  /// reconstruct the per-element source potential `φ_i = (B · q)_i` for the
  /// per-element energy decomposition `½ σ_i · φ_i`.
  Mat B;
  /// `G = −f(ε) · A^{-1} · B` (ncav × natom). σ = G · q.
  Mat G;
  /// `J_solv = B^T · G` (natom × natom). Symmetric, negative-definite.
  /// V_solv = J_solv · q.
  Mat J_solv;
};

/// Build the COSMO response for the cavity defined by `surface` evaluated
/// against atom-centered sources at `atom_positions_bohr`.
///
///   epsilon : solvent relative permittivity
///   x       : convention parameter in f(ε) = (ε−1)/(ε+x). `x = 0` is the
///             CPCM ideal-conductor convention, `x = 0.5` is Klamt COSMO.
Response build(const Mat3N &atom_positions_bohr,
               const occ::solvent::surface::Surface &surface,
               double epsilon, double x);

/// Frozen-cavity analytical gradient of the polarisation energy with respect
/// to atomic positions:
///
///   ∂E_es/∂R_c = -Σ_{i: a_i=c} σ_i · g_i  +  q_c · h_c
///                + (1/f(ε)) · Σ_{i: a_i=c} σ_i · t_i
///                + diagonal-A term (smooth cavity only — see below)
///
///   g_i = Σ_a  q_a · (r_i - R_a) / |r_i - R_a|³     (field at cavity i from q)
///   t_i = -Σ_{j≠i} σ_j · (r_i - r_j) / |r_i - r_j|³ (field at cavity i from σ)
///   h_c = Σ_i σ_i · (r_i - R_c) / |r_i - R_c|³      (field at atom c from σ)
///
/// With a boolean cavity (smoothing_width = 0), per-element areas are
/// geometry-independent and ∂A_ii/∂R = 0. With a smooth cavity
/// (`solvent_surface(..., smoothing_width > 0)`), each per-element area
/// depends smoothly on every other atom's position through the smoothstep
/// weight, giving
///
///   ∂A_ii/∂R_c = -½ A_ii · ∂ln(weight_i)/∂R_c
///   ∂ln(weight_i)/∂R_c = Σ_{k ≠ a_i} (s'/s)|d_ik · ∂d_ik/∂R_c
///
/// where `s(d) = ½(1 + erf((d − t_k)/w))` with `t_k = r_k` (post-shift). Pass
/// `atom_radii_bohr` and `smoothing_width_bohr` to enable this term; leave
/// `smoothing_width_bohr = 0` to skip it (boolean-cavity default).
///
/// Returns 3 × N_atoms in Hartree/Bohr.
[[nodiscard]] Mat3N
gradient(const Mat3N &atom_positions_bohr,
         const occ::solvent::surface::Surface &surface,
         const Vec &atom_charges, const Vec &sigma, double f_epsilon,
         const Vec &atom_radii_bohr = Vec(),
         double smoothing_width_bohr = 0.0);

} // namespace occ::xtb::cosmo
