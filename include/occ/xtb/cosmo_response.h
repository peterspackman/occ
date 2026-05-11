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

} // namespace occ::xtb::cosmo
