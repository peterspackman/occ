#pragma once
#include <occ/mults/cartesian_multipole.h>
#include <occ/mults/cartesian_molecule.h>

namespace occ::mults {

/// Shift a Cartesian multipole from its current origin to a new origin.
///
/// Given multipole moments M_{t'u'v'} at position r_s, computes moments
/// about a new origin r_0 using the binomial shift formula:
///
///   M'_{tuv} = sum_{t'<=t, u'<=u, v'<=v}  C(t,t') C(u,u') C(v,v')
///              * dx^(t-t') * dy^(u-u') * dz^(v-v') * M_{t'u'v'}
///
/// where d = r_s - r_0 (displacement from new origin to old site).
///
/// The output is accumulated (added to), not overwritten. Caller must
/// zero the output multipole before the first call if needed.
///
/// @param input       Source multipole at the old origin
/// @param input_rank  Maximum rank of non-zero components in input
/// @param displacement  Vector d = r_s - r_0 (old site minus new origin)
/// @param output      Destination multipole (accumulated into)
void shift_multipole_to_origin(const CartesianMultipole<4> &input,
                               int input_rank,
                               const Vec3 &displacement,
                               CartesianMultipole<4> &output);

/// Merge all sites of a molecule into a single effective site.
///
/// The new origin is the charge-weighted centroid if the total charge
/// is non-negligible, otherwise the geometric centroid.
///
/// @param mol  Input molecule with multiple sites
/// @return     Single-site molecule at the merged origin
CartesianMolecule merge_to_single_site(const CartesianMolecule &mol);

/// Merge all sites of a molecule into a single effective site at a
/// specified origin.
///
/// @param mol     Input molecule with multiple sites
/// @param origin  The position for the merged site
/// @return        Single-site molecule at origin
CartesianMolecule merge_to_single_site(const CartesianMolecule &mol,
                                       const Vec3 &origin);

} // namespace occ::mults
