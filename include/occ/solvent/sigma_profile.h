#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/solvent/surface.h>

/// Screening-charge-density (σ) machinery underlying the COSMO-RS solvation
/// model.
///
/// Units follow the published parameter tables and are converted at the
/// boundary: σ in e/Å², segment areas in Å², distances in Å (positions are
/// stored in Bohr to match `surface::Surface`).
namespace occ::solvent::sigma {

/// Per-segment COSMO data.
struct Segments {
  Mat3N positions;    ///< Bohr
  Vec areas;          ///< Å²
  Vec sigma;          ///< e/Å², raw q_i / a_i
  Vec sigma_averaged; ///< e/Å², after `average_sigma`
  /// e/Å², after `average_sigma_orth`; empty for models that do not use it.
  Vec sigma_orth;
  IVec atom_index;
  IVec atomic_number; ///< of the parent atom

  Eigen::Index size() const { return areas.size(); }
  double total_area() const { return areas.sum(); }
  /// Σ a_i σ_i. For a conductor cavity this approaches −q_solute; the
  /// deviation measures the cavity discretisation error.
  double total_charge() const;
  double total_charge_averaged() const;
};

/// Build segments from a COSMO cavity and the apparent surface charges.
///
/// `charges` are segment charges in e, not densities — the COSMO A matrix
/// carries units of inverse length, so `σ = A⁻¹(−f φ)` is a charge.
Segments segments_from_cavity(const surface::Surface &cavity,
                              const Vec &charges, const IVec &atomic_numbers);

/// Segment averaging onto the effective contact scale:
///
///     σ̄_i = Σ_j σ_j w_ij / Σ_j w_ij
///     w_ij = (r_j² r_av²)/(r_j² + r_av²) · exp[ −f_decay d_ij²/(r_j² + r_av²) ]
///
/// with `r_j = √(a_j/π)` the radius of an equal-area disc. Fills
/// `sigma_averaged`. The Gaussian is truncated where it falls below ~1e-12.
///
/// openCOSMO-RS averages on `r_av = 0.5 Å` with `f_decay = 1`.
void average_sigma(Segments &segments, double r_av_angs, double f_decay = 1.0);

/// The same average returned rather than stored, so a second descriptor can
/// be built on a different radius without disturbing `sigma_averaged`.
Vec averaged_sigma(const Segments &segments, double r_av_angs,
                   double f_decay = 1.0);

/// Fill `sigma_orth`, the correlation screening charge density COSMO-RS
/// misfit uses:
///
///     σ⊥ = avg(r_corr) − factor · avg(r_av)
///
/// openCOSMO-RS takes `r_av = 0.5 Å`, `r_corr = 1.0 Å` and `factor = 0.816`.
/// Requires `sigma_averaged` to have been filled on the same `r_av`.
void average_sigma_orth(Segments &segments, double r_av_angs,
                        double r_corr_angs, double factor = 0.816);

} // namespace occ::solvent::sigma
