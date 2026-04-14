#pragma once
#include <occ/mults/ewald_sum.h>
#include <occ/mults/cutoff_spline.h>
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::mults {

struct NeighborPair;  // Forward declaration

/// Result of explicit Ewald strain derivative computation.
struct EwaldExplicitStrainTerms {
    Vec6 grad = Vec6::Zero();
    Mat6 hess = Mat6::Zero();
    Mat strain_site_mixed;
};

/// Compute Ewald strain derivatives (gradient + Hessian) using AD6 dual numbers.
///
/// Evaluates dE/dE_i and d²E/dE_i dE_j for the Ewald correction terms
/// (real-space erf, reciprocal-space, self-energy) under affine cell strain.
///
/// @param sites Ewald site positions, charges, and dipoles
/// @param unit_cell Current unit cell
/// @param neighbors Molecule pair list with cell shifts
/// @param mol_site_indices Mapping from molecule index to site indices
/// @param cutoff_radius COM cutoff for electrostatic pair gate
/// @param use_com_gate Whether to apply COM-based pair gating
/// @param elec_site_cutoff Per-site cutoff (0 = no cutoff)
/// @param params Ewald alpha and kmax parameters
/// @param taper Optional electrostatic radial taper
/// @param lattice_cache Precomputed G-vectors (or nullptr)
/// @param include_strain_state If true, compute mixed strain-site derivatives
EwaldExplicitStrainTerms compute_ewald_explicit_strain_terms(
    const std::vector<EwaldSite>& sites,
    const crystal::UnitCell& unit_cell,
    const std::vector<NeighborPair>& neighbors,
    const std::vector<std::vector<size_t>>& mol_site_indices,
    double cutoff_radius,
    bool use_com_gate,
    double elec_site_cutoff,
    const EwaldParams& params,
    const CutoffSpline* taper,
    const EwaldLatticeCache* lattice_cache,
    bool include_strain_state = false);

} // namespace occ::mults
