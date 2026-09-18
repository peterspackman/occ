#pragma once
#include <Eigen/LU>
#include <array>
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

/// Geometric factors of the atom-centred multipole expansion evaluated at a
/// displacement `d = r_cavity − R_atom`:
///
///   φ = q·t0 + μ·t1 + Σ_p Θ_p·t2[p]
///
/// `t2` is ordered (xx, xy, yy, xz, yz, zz) — the layout of
/// `occ::xtb::CammMoments::qp` — and its off-diagonal entries carry the factor
/// of two that the full αβ contraction implies, so each `t2[p]` is conjugate
/// to the singly-stored component `Θ_p`. The `−tr(Θ)/(3d³)` piece is kept so
/// the kernel is the exact expansion for any Θ, not only a traceless one.
struct MultipoleKernel {
  double t0;
  Vec3 t1;
  std::array<double, 6> t2;
  /// Damping factors applied to `t1` and `t2` (1 when undamped) and their
  /// derivatives with respect to `rco`. A model whose cut-off radii depend on
  /// the geometry — GFN2's do, through the coordination number — needs these
  /// to close its gradient.
  double f3{1.0}, f5{1.0};
  double df3_drco{0.0}, df5_drco{0.0};
};

/// Short-range damping of the dipole and quadrupole kernels,
/// `f_n(d) = 1/(1 + 6·(rco/d)^kdmp_n)`, applied to `t1` with `n = 3` and to
/// `t2` with `n = 5`. This is the damping GFN2 already uses for its own
/// anisotropic electrostatics: CAMM moments are partitioned density, not point
/// multipoles, and the bare 1/d² and 1/d³ kernels overshoot badly when
/// evaluated as close in as a solvation cavity sits. The monopole term is
/// never damped — it has to stay exact for the cavity's Gauss-law behaviour.
///
/// `rco <= 0` disables damping, which is the default.
struct MultipoleDamping {
  double rco{0.0};
  double kdmp3{3.0};
  double kdmp5{4.0};

  bool active() const { return rco > 0.0; }
};

MultipoleKernel multipole_kernel(const Vec3 &d, MultipoleDamping damping = {});

/// ∂φ/∂d for one atom's moments, the gradient counterpart of
/// `multipole_kernel`. `theta` points at six contiguous quadrupole components
/// in the same layout, or is null for a charge-and-dipole source.
Vec3 multipole_kernel_gradient(const Vec3 &d, double q, const Vec3 &mu,
                               const double *theta,
                               MultipoleDamping damping = {});

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
///
/// `atom_dipoles` (3 × N) and `atom_quadrupoles` (6 × N, `multipole_kernel`
/// layout) extend the source from point charges to the full atom-centred
/// expansion; pass them empty for the charge-only field. They enter through
/// `g` and `h` only — the σ-on-σ and diagonal-A terms see the cavity, not the
/// source, and are unchanged.
[[nodiscard]] Mat3N
cosmo_gradient_frozen(const Mat3N &atom_positions_bohr,
                      const Mat3N &cavity_points, const Vec &cavity_areas,
                      const IVec &cavity_atom_index, const Vec &atom_charges,
                      const Vec &sigma, double f_epsilon,
                      const Vec &atom_radii_bohr = Vec(),
                      double smoothing_width_bohr = 0.0,
                      const Mat3N &atom_dipoles = Mat3N(),
                      const Mat &atom_quadrupoles = Mat(),
                      const Vec &damping_rco_bohr = Vec(),
                      double kdmp3 = 3.0, double kdmp5 = 4.0);

} // namespace occ::scrf::detail
