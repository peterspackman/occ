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
 * @brief Compute energy of a strained crystal.
 *
 * Applies strain eps to the unit cell: direct' = (I + eps) * direct.
 * Fractional coordinates are unchanged, so Cartesian positions shift
 * according to the new lattice vectors. A new CrystalEnergy is built
 * from scratch with the deformed cell.
 *
 * @param input     DMACRYS input data (asymmetric unit, multipoles, potentials)
 * @param crystal   Reference (unstrained) crystal
 * @param multipoles Multipole sources for the reference crystal
 * @param buck_params Buckingham parameters
 * @param strain    3x3 strain tensor eps (deformation = I + eps)
 * @param cutoff    Neighbor cutoff (Angstrom)
 * @param use_ewald Whether to apply Ewald correction
 * @param alpha     Ewald alpha parameter (1/Angstrom)
 * @param kmax      Ewald kmax
 * @param max_interaction_order Max interaction order (-1 = no truncation, 4 = DMACRYS)
 * @param fixed_neighbors If non-null, use this neighbor list instead of rebuilding
 * @param fixed_site_masks If non-null, use these atom-pair masks for Buckingham
 * @return Total energy in kJ/mol (per unit cell)
 */
double compute_strained_energy(
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

/**
 * @brief Compute 6 strain derivatives dU/dE_i by central finite differences.
 *
 * Uses a fixed neighbor list from the unstrained crystal to avoid
 * discontinuities from the hard cutoff boundary.
 *
 * For each Voigt component i:
 *   dU/dE_i = [U(+delta) - U(-delta)] / (2*delta)
 *
 * @param input     DMACRYS input data
 * @param crystal   Reference crystal
 * @param multipoles Multipole sources
 * @param buck_params Buckingham parameters
 * @param cutoff    Neighbor cutoff (Angstrom)
 * @param use_ewald Whether to use Ewald
 * @param alpha     Ewald alpha
 * @param kmax      Ewald kmax
 * @param delta     FD step size (default 1e-4)
 * @param max_interaction_order Max interaction order (-1 = no truncation)
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
 * @brief Compute the 6x6 elastic stiffness tensor C_ij by FD.
 *
 * C_ij = (1/V) * d^2 U / (dE_i dE_j)
 *
 * Diagonal: C_ii = (1/V) * [U(+d) - 2*U(0) + U(-d)] / d^2
 * Off-diag: C_ij = (1/V) * [U(+di,+dj) - U(+di,-dj) - U(-di,+dj) + U(-di,-dj)] / (4*d^2)
 *
 * @param input     DMACRYS input data
 * @param crystal   Reference crystal
 * @param multipoles Multipole sources
 * @param buck_params Buckingham parameters
 * @param cutoff    Neighbor cutoff (Angstrom)
 * @param use_ewald Whether to use Ewald
 * @param alpha     Ewald alpha
 * @param kmax      Ewald kmax
 * @param delta     FD step size (default 1e-3)
 * @param max_interaction_order Max interaction order (-1 = no truncation)
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
 * @brief Compute the 6N energy gradient at a strained geometry.
 *
 * Returns the energy gradient (not forces) packed as:
 *   grad[6i..6i+2] = -force_i  (∂U/∂r for translations)
 *   grad[6i+3..6i+5] = +torque_i (∂U/∂θ for rotations)
 *
 * This matches the sign convention used in compute_with_hessian().
 */
Vec compute_strained_gradient(
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

/**
 * @brief Compute relaxed-ion elastic constants via Schur complement.
 *
 * C_relaxed = W_εε - W_εi · W_ii⁻¹ · W_iε
 *
 * Where W_εε is the strain-strain 2nd derivative (clamped elastic constants),
 * W_ii is the internal DOF Hessian, and W_εi is the strain-internal coupling.
 * Internal relaxation lowers stiffness compared to clamped-ion constants.
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
