#pragma once
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_energy_setup.h>
#include <occ/core/linear_algebra.h>

namespace occ::mults {

/// Build the 3x3 strain tensor for a single Voigt component.
///
/// Voigt convention:
///   0 -> E_1 = eps_11   (xx)
///   1 -> E_2 = eps_22   (yy)
///   2 -> E_3 = eps_33   (zz)
///   3 -> E_4 = 2*eps_23 (yz)
///   4 -> E_5 = 2*eps_13 (xz)
///   5 -> E_6 = 2*eps_12 (xy)
Mat3 voigt_strain_tensor(int voigt_index, double magnitude);

/// Result from strained crystal computation.
struct StrainedResult {
    double energy = 0.0; ///< Total energy (kJ/mol)
    Vec gradient;         ///< 6N energy gradient [-force, +torque]
};

/// Compute energy and gradient of a strained crystal.
///
/// Applies strain eps to the unit cell: direct' = (I + eps) * direct.
/// Fractional coordinates are unchanged, so Cartesian positions shift
/// according to the new lattice vectors.
StrainedResult compute_strained(
    const CrystalEnergySetup &setup,
    const Mat3 &strain,
    const std::vector<NeighborPair> *fixed_neighbors = nullptr,
    const std::vector<std::vector<bool>> *fixed_site_masks = nullptr);

/// Convenience: compute only energy at strained geometry.
inline double compute_strained_energy(
    const CrystalEnergySetup &setup,
    const Mat3 &strain,
    const std::vector<NeighborPair> *fixed_neighbors = nullptr,
    const std::vector<std::vector<bool>> *fixed_site_masks = nullptr) {
    return compute_strained(setup, strain, fixed_neighbors,
                            fixed_site_masks)
        .energy;
}

/// Compute 6 strain derivatives dU/dE_i by central finite differences.
///
/// Uses a fixed neighbor list from the unstrained crystal to avoid
/// discontinuities from the hard cutoff boundary.
///
/// @return Vec6 of dU/dE_i in eV per unit cell
Vec6 compute_strain_derivatives_fd(const CrystalEnergySetup &setup,
                                   double delta = 1e-4);

/// Compute the 6x6 clamped elastic stiffness tensor C_ij analytically.
///
/// @return 6x6 stiffness tensor in GPa
Mat6 compute_elastic_constants_fd(const CrystalEnergySetup &setup,
                                  double delta = 1e-3);

/// Compute relaxed-ion elastic constants via Schur complement.
///
/// C_relaxed = W_ee - W_ei * W_ii^{-1} * W_ie
///
/// @return 6x6 stiffness tensor in GPa (relaxed-ion)
Mat6 compute_relaxed_elastic_constants_fd(const CrystalEnergySetup &setup,
                                          double delta = 1e-3);

} // namespace occ::mults
