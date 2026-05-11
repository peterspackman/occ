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
Surface solvent_surface(const Vec &radii, const IVec &atomic_numbers,
                        const Mat3N &positions,
                        double solvent_radius_angs = 0.4,
                        bool axis_aligned = true);

IVec nearest_atom_index(const Mat3N &atom_positions,
                        const Mat3N &element_centers);
} // namespace occ::solvent::surface
