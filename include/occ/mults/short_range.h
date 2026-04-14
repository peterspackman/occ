#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::mults {

/**
 * @brief Parameters for the Lennard-Jones 12-6 potential.
 *
 * The Lennard-Jones potential is commonly used to model van der Waals interactions:
 * V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
 *
 * @param epsilon Well depth (energy units). Typical values: 0.1-1.0 kcal/mol
 * @param sigma Zero-crossing distance (length units). Typical values: 2.5-4.0 Å
 *
 * Common parameterizations:
 * - Ar-Ar: ε = 0.238 kcal/mol, σ = 3.40 Å (OPLS)
 * - Ne-Ne: ε = 0.069 kcal/mol, σ = 2.78 Å (OPLS)
 * - Kr-Kr: ε = 0.317 kcal/mol, σ = 3.69 Å (OPLS)
 *
 * References:
 * - Jorgensen et al., J. Am. Chem. Soc. 118, 11225 (1996)
 * - Jones, Proc. R. Soc. Lond. A 106, 463 (1924)
 */
struct LennardJonesParams {
    double epsilon;  ///< Well depth (energy units)
    double sigma;    ///< Zero-crossing distance (length units)
};

/**
 * @brief Parameters for the Buckingham (exp-6) potential.
 *
 * The Buckingham potential provides a more realistic representation of repulsion:
 * V(r) = A*exp(-B*r) - C/r^6
 *
 * @param A Repulsion parameter (energy units). Typical values: 10^4-10^6 kcal/mol
 * @param B Exponential decay parameter (1/length units). Typical values: 3-4 Å^-1
 * @param C Dispersion parameter (energy * length^6 units). Typical values: 10^2-10^3 kcal·Å^6/mol
 *
 * Common parameterizations:
 * - Ar-Ar: A = 1.69×10^5 kcal/mol, B = 3.60 Å^-1, C = 99.5 kcal·Å^6/mol
 * - O-O (water): A = 2.55×10^5 kcal/mol, B = 3.55 Å^-1, C = 396.0 kcal·Å^6/mol
 *
 * Note: The Buckingham potential can become unphysical (negative, attractive) at very
 * small distances due to the exponential term. Use with caution at r < σ/2.
 *
 * References:
 * - Buckingham, Proc. R. Soc. Lond. A 168, 264 (1938)
 * - Stone, The Theory of Intermolecular Forces, 2nd ed. (2013)
 */
struct BuckinghamParams {
    double A;  ///< Repulsion parameter (energy units)
    double B;  ///< Exponential decay parameter (1/length units)
    double C;  ///< Dispersion parameter (energy * length^6 units)
};

/**
 * @brief Parameters for anisotropic Born-Mayer repulsion potential.
 *
 * The potential is:
 *   V(R) = exp(-α·R) × exp(α·ρ)
 *   ρ = ρ₀₀ + ρ₂₀·P₂(cosθ₁) + ρ₀₂·P₂(cosθ₂)
 *
 * where P₂(x) = (3x² - 1)/2, θ₁ is the angle between site 1's
 * anisotropy z-axis and the inter-site unit vector, θ₂ is the angle
 * between site 2's anisotropy z-axis and the inter-site unit vector.
 *
 * This is an additive term on top of the isotropic Buckingham potential.
 */
struct AnisotropicRepulsionParams {
    double alpha;   ///< Exponential decay parameter (Å⁻¹), typically 1/rho_iso
    double rho_00;  ///< Isotropic ρ (Å)
    double rho_20;  ///< P₂ anisotropy coefficient for site 1 (Å)
    double rho_02;  ///< P₂ anisotropy coefficient for site 2 (Å)
};

/**
 * @brief Class for computing short-range pairwise interactions.
 *
 * Provides implementations of Lennard-Jones and Buckingham potentials with
 * analytical first and second derivatives. These are commonly used in molecular
 * mechanics and molecular dynamics simulations.
 *
 * All functions are static and thread-safe. The class provides:
 * - Individual energy/derivative calculations
 * - Combined calculations (more efficient when all are needed)
 * - Transformation from scalar derivatives to vector forces and Hessians
 *
 * Usage example:
 * @code
 * Vec3 r_A(0, 0, 0);
 * Vec3 r_B(3.5, 0, 0);  // 3.5 Angstrom separation
 *
 * // Lennard-Jones for Ar-Ar
 * LennardJonesParams lj_params{.epsilon = 0.238, .sigma = 3.40};
 *
 * // Compute energy and derivatives
 * auto result = ShortRangeInteraction::lennard_jones_all(
 *     (r_B - r_A).norm(), lj_params);
 *
 * // Convert to forces and Hessian
 * auto forces = ShortRangeInteraction::compute_force_hessian(
 *     r_A, r_B, result.first_derivative, result.second_derivative);
 * @endcode
 */
class ShortRangeInteraction {
public:
    /**
     * @brief Result structure containing energy and derivatives.
     */
    struct EnergyAndDerivatives {
        double energy;              ///< Potential energy V(r)
        double first_derivative;    ///< First derivative dV/dr
        double second_derivative;   ///< Second derivative d²V/dr²
    };

    /**
     * @brief Result structure containing forces and Hessian for both sites.
     *
     * For a central potential V(r) between sites A and B:
     * - Forces satisfy Newton's 3rd law: F_A + F_B = 0
     * - Hessian is symmetric and satisfies: H_AA + H_AB + H_BA + H_BB = 0
     */
    struct ForceAndHessian {
        Vec3 force_A;    ///< Force on site A: F_A = -(dV/dr) * (r_vec/r)
        Vec3 force_B;    ///< Force on site B: F_B = -F_A
        Mat3 hessian_AA; ///< Hessian d²V/dx_A dx_A
        Mat3 hessian_AB; ///< Hessian d²V/dx_A dx_B
        Mat3 hessian_BB; ///< Hessian d²V/dx_B dx_B
    };

    /**
     * @brief Result from anisotropic repulsion computation.
     */
    struct AnisotropicResult {
        double energy;        ///< Potential energy
        Vec3 force_A;         ///< Force on site A (Cartesian, energy/length)
        Vec3 force_B;         ///< Force on site B (Cartesian, energy/length)
        Vec3 torque_axis_A;   ///< dE/dψ for rotation of axis A (lab frame)
        Vec3 torque_axis_B;   ///< dE/dψ for rotation of axis B (lab frame)
    };

    /**
     * @brief Compute anisotropic Born-Mayer repulsion energy, forces, and axis torques.
     *
     * @param pos_a Position of site A
     * @param pos_b Position of site B
     * @param axis_a Unit z-axis of site A in lab frame (zero = isotropic for this site)
     * @param axis_b Unit z-axis of site B in lab frame (zero = isotropic for this site)
     * @param params Anisotropic repulsion parameters
     * @return AnisotropicResult with energy, forces, and axis torques
     */
    static AnisotropicResult anisotropic_repulsion(
        const Vec3& pos_a, const Vec3& pos_b,
        const Vec3& axis_a, const Vec3& axis_b,
        const AnisotropicRepulsionParams& params);

    // ==================== Lennard-Jones Potential ====================

    /**
     * @brief Compute Lennard-Jones energy: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
     *
     * @param r Distance between sites (must be > 0)
     * @param params Lennard-Jones parameters (ε, σ)
     * @return Energy in same units as epsilon
     */
    static double lennard_jones_energy(double r, const LennardJonesParams& params);

    /**
     * @brief Compute first derivative: dV/dr = 24ε/r[(σ/r)^6 - 2(σ/r)^12]
     *
     * @param r Distance between sites (must be > 0)
     * @param params Lennard-Jones parameters (ε, σ)
     * @return First derivative dV/dr
     */
    static double lennard_jones_derivative(double r, const LennardJonesParams& params);

    /**
     * @brief Compute second derivative: d²V/dr² = 24ε/r²[26(σ/r)^12 - 7(σ/r)^6]
     *
     * @param r Distance between sites (must be > 0)
     * @param params Lennard-Jones parameters (ε, σ)
     * @return Second derivative d²V/dr²
     */
    static double lennard_jones_second_derivative(double r, const LennardJonesParams& params);

    /**
     * @brief Compute Lennard-Jones energy and both derivatives (more efficient).
     *
     * This is more efficient than calling the individual functions when all
     * quantities are needed, as it reuses intermediate calculations.
     *
     * @param r Distance between sites (must be > 0)
     * @param params Lennard-Jones parameters (ε, σ)
     * @return Structure containing energy, first derivative, and second derivative
     */
    static EnergyAndDerivatives lennard_jones_all(double r, const LennardJonesParams& params);

    // ==================== Buckingham Potential ====================

    /**
     * @brief Compute Buckingham energy: V(r) = A*exp(-B*r) - C/r^6
     *
     * @param r Distance between sites (must be > 0)
     * @param params Buckingham parameters (A, B, C)
     * @return Energy in same units as A
     */
    static double buckingham_energy(double r, const BuckinghamParams& params);

    /**
     * @brief Compute first derivative: dV/dr = -A*B*exp(-B*r) + 6C/r^7
     *
     * @param r Distance between sites (must be > 0)
     * @param params Buckingham parameters (A, B, C)
     * @return First derivative dV/dr
     */
    static double buckingham_derivative(double r, const BuckinghamParams& params);

    /**
     * @brief Compute second derivative: d²V/dr² = A*B²*exp(-B*r) - 42C/r^8
     *
     * @param r Distance between sites (must be > 0)
     * @param params Buckingham parameters (A, B, C)
     * @return Second derivative d²V/dr²
     */
    static double buckingham_second_derivative(double r, const BuckinghamParams& params);

    /**
     * @brief Compute Buckingham energy and both derivatives (more efficient).
     *
     * This is more efficient than calling the individual functions when all
     * quantities are needed, as it reuses intermediate calculations.
     *
     * @param r Distance between sites (must be > 0)
     * @param params Buckingham parameters (A, B, C)
     * @return Structure containing energy, first derivative, and second derivative
     */
    static EnergyAndDerivatives buckingham_all(double r, const BuckinghamParams& params);

    // ==================== Force and Hessian Transformations ====================

    /**
     * @brief Convert radial derivative to force vector.
     *
     * For a central potential V(r), the force on site A is:
     * F_A = -(dV/dr) * (r_vec/r) where r_vec = r_B - r_A
     *
     * @param dE_dr First derivative dV/dr (scalar)
     * @param r_vec Displacement vector r_B - r_A
     * @return Force vector on site A (force on B is -F_A)
     */
    static Vec3 derivative_to_force(double dE_dr, const Vec3& r_vec);

    /**
     * @brief Compute force vectors and Hessian matrices for both sites.
     *
     * For a central potential V(r) between sites A and B, computes:
     * - Forces: F_A = -(dV/dr) * (r_vec/r), F_B = -F_A
     * - Hessians: H_ij = d²V/dx_i dx_j
     *
     * The Hessian for a central potential has two terms:
     * 1. Curvature term: (d²V/dr²) * (r_i/r) * (r_j/r)
     * 2. Angular term: (dV/dr/r) * [δ_ij - (r_i/r)(r_j/r)]
     *
     * The resulting Hessian satisfies:
     * - H_AB = -H_AA (from symmetry)
     * - H_BB = H_AA (from symmetry)
     * - H_AA + H_AB + H_BA + H_BB = 0 (translational invariance)
     *
     * @param r_A Position of site A
     * @param r_B Position of site B
     * @param dE_dr First derivative dV/dr
     * @param d2E_dr2 Second derivative d²V/dr²
     * @return Structure containing forces and Hessians
     */
    static ForceAndHessian compute_force_hessian(
        const Vec3& r_A, const Vec3& r_B,
        double dE_dr, double d2E_dr2);

private:
    // Minimum distance cutoff to avoid singularities (in same units as input)
    static constexpr double MIN_DISTANCE = 1e-10;
};

} // namespace occ::mults
