#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::solvent::surface {

struct Surface {
  Mat3N vertices;
  Vec areas;
  IVec atom_index;
};

/// Build an atom-centered Lebedev cavity surface.
///
/// `axis_aligned = true` (default) preserves the legacy behaviour of
/// pre-rotating the molecule into its principal-axes frame before
/// placing the Lebedev grid — fine for energy-only callers (SMD HF/DFT
/// pipeline). For analytical gradients, set `axis_aligned = false` so
/// each cavity point sits rigidly on its parent atom; otherwise the
/// global rotation chain `∂axes/∂R` breaks the frozen-cavity assumption
/// in `cosmo::gradient`.
///
/// `smoothing_width_bohr = 0` (default) uses the legacy boolean mask
/// (point is either fully included or fully dropped). With
/// `smoothing_width_bohr > 0`, each cavity point gets a continuous
/// weight in [0, 1]:
///
///     weight_j = Π_{k ≠ atom_j} smoothstep(|r_j − R_k|, r_k, w)
///     smoothstep(d, t, w) = ½ (1 + erf((d − t)/w))
///
/// Distances are measured between the post-shift cavity vertex and atom
/// centres so the gradient code can reconstruct the same weights from
/// `surface.vertices` and atom radii. Areas are multiplied by the weight
/// and points whose weight falls below 1e-10 are dropped. This makes the
/// cavity, and any energy derived from it, C∞ with respect to atomic
/// positions — required for clean analytical gradients.
Surface solvent_surface(const Vec &radii, const IVec &atomic_numbers,
                        const Mat3N &positions,
                        double solvent_radius_angs = 0.4,
                        bool axis_aligned = true,
                        double smoothing_width_bohr = 0.0);

IVec nearest_atom_index(const Mat3N &atom_positions,
                        const Mat3N &element_centers);
} // namespace occ::solvent::surface
