#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::opt {

/// Lindh's model Hessian (R. Lindh, A. Bernhardsson, G. Karlstrom and
/// P.-A. Malmqvist, Chem. Phys. Lett. 241, 423 (1995)) in Cartesian
/// coordinates, Hartree/Bohr^2.
///
/// Every pair, bend and torsion of atoms contributes k q' q'^T, with q' the
/// coordinate's Cartesian derivative and k = k_r rho_ij, k_f rho_ij rho_jk or
/// k_t rho_ij rho_jk rho_kl, where rho_ij = exp(alpha (r_ref^2 - r_ij^2)) by
/// periodic-table row. Terms whose weight is below `threshold` are dropped.
///
/// positions in Angstrom (3 x N).
Mat lindh_model_hessian(const Mat3N &positions, const IVec &atomic_numbers,
                        double threshold = 1e-4);

/// Orthonormal basis (3N x 6, or 5 for a linear and 3 for a single atom) of
/// rigid translations and rotations about the centroid of x (any units).
Mat rigid_body_basis(const Mat3N &x);

} // namespace occ::opt
