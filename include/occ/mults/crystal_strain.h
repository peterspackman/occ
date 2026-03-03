#pragma once
#include <occ/mults/crystal_energy.h>
#include <occ/mults/dmacrys_input.h>
#include <occ/core/linear_algebra.h>

namespace occ::mults {

/**
 * @brief Build the 3x3 strain tensor for a single Voigt component.
 *
 * Voigt convention:
 *   0 -> E_1 = eps_11   (xx)
 *   1 -> E_2 = eps_22   (yy)
 *   2 -> E_3 = eps_33   (zz)
 *   3 -> E_4 = 2*eps_23 (yz)
 *   4 -> E_5 = 2*eps_13 (xz)
 *   5 -> E_6 = 2*eps_12 (xy)
 *
 * @param voigt_index Index 0..5 (Voigt notation)
 * @param magnitude Strain magnitude (e.g. 1e-4)
 * @return 3x3 strain tensor eps (symmetric)
 */
Mat3 voigt_strain_tensor(int voigt_index, double magnitude);

/**
 * @brief Result from strained crystal computation.
 */
struct StrainedResult {
    double energy = 0.0; ///< Total energy (kJ/mol)
    Vec gradient;         ///< 6N energy gradient [-force, +torque]
};

/**
 * @brief Compute energy and gradient of a strained crystal.
 *
 * Applies strain eps to the unit cell: direct' = (I + eps) * direct.
 * Fractional coordinates are unchanged, so Cartesian positions shift
 * according to the new lattice vectors.
 */
StrainedResult compute_strained(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    const Mat3& strain,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    int max_interaction_order = -1,
    const std::vector<NeighborPair>* fixed_neighbors = nullptr,
    const std::vector<std::vector<bool>>* fixed_site_masks = nullptr);

/// Convenience: compute only energy at strained geometry.
inline double compute_strained_energy(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    const Mat3& strain,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    int max_interaction_order = -1,
    const std::vector<NeighborPair>* fixed_neighbors = nullptr,
    const std::vector<std::vector<bool>>* fixed_site_masks = nullptr) {
    return compute_strained(input, crystal, multipoles, buck_params,
                            strain, cutoff, use_ewald, alpha, kmax,
                            max_interaction_order,
                            fixed_neighbors, fixed_site_masks).energy;
}

/// Convenience: compute only gradient at strained geometry.
inline Vec compute_strained_gradient(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    const Mat3& strain,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    int max_interaction_order = -1,
    const std::vector<NeighborPair>* fixed_neighbors = nullptr,
    const std::vector<std::vector<bool>>* fixed_site_masks = nullptr) {
    return compute_strained(input, crystal, multipoles, buck_params,
                            strain, cutoff, use_ewald, alpha, kmax,
                            max_interaction_order,
                            fixed_neighbors, fixed_site_masks).gradient;
}

/**
 * @brief Compute 6 strain derivatives dU/dE_i by central finite differences.
 *
 * Uses a fixed neighbor list from the unstrained crystal to avoid
 * discontinuities from the hard cutoff boundary.
 *
 * @return Vec6 of dU/dE_i in eV per unit cell
 */
Vec6 compute_strain_derivatives_fd(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    double delta = 1e-4,
    int max_interaction_order = -1);

/**
 * @brief Compute the 6x6 clamped elastic stiffness tensor C_ij analytically.
 *
 * @return 6x6 stiffness tensor in GPa
 */
Mat6 compute_elastic_constants_fd(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    double delta = 1e-3,
    int max_interaction_order = -1);

/**
 * @brief Compute relaxed-ion elastic constants via Schur complement.
 *
 * C_relaxed = W_ee - W_ei * W_ii^{-1} * W_ie
 *
 * When the configured model lacks some second derivatives (currently Ewald
 * and/or electrostatic taper terms), this returns the corresponding
 * approximation and logs a warning.
 *
 * @return 6x6 stiffness tensor in GPa (relaxed-ion)
 */
Mat6 compute_relaxed_elastic_constants_fd(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    double delta = 1e-3,
    int max_interaction_order = -1);

} // namespace occ::mults
