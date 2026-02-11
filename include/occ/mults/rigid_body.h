#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/torque.h>
#include <Eigen/Geometry>
#include <string>

namespace occ::mults {

/**
 * @brief State of a rigid body for molecular dynamics simulations
 *
 * This structure contains all kinematic and dynamic state information for
 * a rigid molecule, including position, orientation, linear/angular velocities,
 * forces, torques, and multipole moments.
 *
 * ## Orientation Representations
 *
 * Multiple representations are supported:
 * - **Quaternion** (primary storage): Used for MD integration, numerically stable
 * - **Angle-axis** (optimization): Minimal parametrization, no singularities
 * - **Euler angles** (I/O): User-friendly input/output (ZYZ convention)
 *
 * ### When to Use Each:
 * - Quaternions: MD time integration, internal storage
 * - Angle-axis: Geometry optimization, gradient-based minimization
 * - Euler angles: Reading/writing input files, user interface
 *
 * ### Conversion Example:
 * ```cpp
 * RigidBodyState mol;
 * mol.set_euler_angles(0.5, 0.3, 0.2);  // User input
 * Vec3 p = mol.get_angle_axis();         // For optimization
 * // ... optimize ...
 * mol.set_angle_axis(p_optimized);       // Update state
 * Vec3 euler = mol.get_euler_angles();   // For output
 * ```
 */
struct RigidBodyState {
    // ========== Identification ==========
    std::string name;                 // Molecule name/label
    int id;                           // Unique identifier

    // ========== Translational State ==========
    Vec3 position;                    // Center of mass position (bohr)
    Vec3 velocity;                    // Linear velocity (bohr/fs)
    Vec3 force;                       // Total force on center of mass (hartree/bohr)
    double mass;                      // Total mass (amu)

    // ========== Rotational State (Quaternion representation) ==========
    Eigen::Quaterniond quaternion;    // Orientation quaternion (w, x, y, z)
    Vec3 angular_velocity_body;       // Angular velocity in body frame (rad/fs)
    Vec3 torque_body;                 // Torque in body frame (hartree)

    // ========== Rotational State (Euler angles - cached) ==========
    Vec3 euler_angles;                // ZYZ Euler angles [alpha, beta, gamma] (radians)
    Vec3 torque_euler;                // Torque w.r.t. Euler angles (hartree)
    bool euler_valid;                 // Whether euler_angles is up-to-date

    // ========== Inertia Tensor ==========
    Mat3 inertia_body;                // Moment of inertia tensor in body frame (amu*bohr^2)
    Mat3 inertia_inv_body;            // Inverse of inertia tensor (cached)

    // ========== Multipole Moments ==========
    occ::dma::Mult multipole_body;    // Multipoles in body frame (permanent, single-site legacy)
    occ::dma::Mult multipole_lab;     // Multipoles in lab frame (cached, single-site legacy)
    bool multipole_lab_valid;         // Whether multipole_lab needs update

    // ========== Multi-site Multipole Support ==========
    /// Body-frame site: multipole expansion at an offset from the center of mass.
    /// All sites share the same rotation (rigid body).
    struct BodySite {
        occ::dma::Mult multipole;
        Vec3 offset = Vec3::Zero();  // Position relative to COM (body frame)
    };
    std::vector<BodySite> sites_body; // Multi-site body-frame data (empty = use multipole_body)

    // ========== Constructors ==========

    /**
     * @brief Default constructor
     */
    RigidBodyState()
        : name(""), id(-1), position(Vec3::Zero()), velocity(Vec3::Zero()),
          force(Vec3::Zero()), mass(1.0), quaternion(Eigen::Quaterniond::Identity()),
          angular_velocity_body(Vec3::Zero()), torque_body(Vec3::Zero()),
          euler_angles(Vec3::Zero()), torque_euler(Vec3::Zero()),
          euler_valid(false), inertia_body(Mat3::Identity()),
          inertia_inv_body(Mat3::Identity()), multipole_body(0), multipole_lab(0),
          multipole_lab_valid(false) {}

    /**
     * @brief Construct with basic parameters
     */
    RigidBodyState(const std::string& mol_name, int mol_id, const Vec3& pos,
                   double mol_mass, int max_rank = 4)
        : name(mol_name), id(mol_id), position(pos), velocity(Vec3::Zero()),
          force(Vec3::Zero()), mass(mol_mass), quaternion(Eigen::Quaterniond::Identity()),
          angular_velocity_body(Vec3::Zero()), torque_body(Vec3::Zero()),
          euler_angles(Vec3::Zero()), torque_euler(Vec3::Zero()),
          euler_valid(true), inertia_body(Mat3::Identity()),
          inertia_inv_body(Mat3::Identity()), multipole_body(max_rank),
          multipole_lab(max_rank), multipole_lab_valid(false) {}

    // ========== Orientation Management ==========

    /**
     * @brief Get rotation matrix from current quaternion
     */
    Mat3 rotation_matrix() const;

    /**
     * @brief Set orientation from Euler angles (ZYZ convention)
     */
    void set_euler_angles(double alpha, double beta, double gamma);

    /**
     * @brief Get Euler angles from current quaternion (cached if possible)
     */
    Vec3 get_euler_angles();

    /**
     * @brief Set orientation from quaternion
     */
    void set_quaternion(const Eigen::Quaterniond& q);

    /**
     * @brief Set orientation from rotation matrix
     */
    void set_rotation_matrix(const Mat3& R);

    /**
     * @brief Get orientation as angle-axis vector
     *
     * Returns p = psi * n where:
     * - psi = rotation angle in [0, pi]
     * - n = unit rotation axis
     *
     * This is the parametrization Orient uses for optimization.
     *
     * @return Angle-axis vector (3 components)
     */
    Vec3 get_angle_axis() const;

    /**
     * @brief Set orientation from angle-axis vector
     *
     * @param p Angle-axis vector where |p| = rotation angle
     */
    void set_angle_axis(const Vec3& p);

    /**
     * @brief Compute Jacobian relating angle-axis to Euler angle derivatives
     *
     * For gradient transformation in optimization:
     *   grad_aa = J^T * grad_euler
     * where grad_euler = [dE/dalpha, dE/dbeta, dE/dgamma]
     *
     * Implements Orient's aaderiv formula from interact.f90
     *
     * @return 3x3 Jacobian matrix
     */
    Mat3 angle_axis_jacobian() const;

    /**
     * @brief Compute rotation matrix derivatives with respect to angle-axis components
     *
     * Computes ∂M/∂p_i for i=1,2,3 where M is the rotation matrix and p is the
     * angle-axis vector. This implements Orient's aaderiv routine.
     *
     * The derivatives are computed analytically using the Rodrigues formula:
     *   M = I + sin(ψ)·[n]× + (1-cos(ψ))·[n]×²
     *
     * where ψ = |p|, n = p/ψ, and [n]× is the skew-symmetric (cross-product) matrix.
     *
     * For small rotations (ψ < 1e-8), uses first-order approximation.
     * For general rotations, uses exact derivatives via chain rule.
     *
     * @return Array of 3 matrices, where result[i] = ∂M/∂p_i
     */
    std::array<Mat3, 3> rotation_matrix_derivatives() const;

    /**
     * @brief Compute second derivatives of rotation matrix w.r.t. angle-axis components
     *
     * Computes ∂²M/∂p_k∂p_l for k,l = 0,1,2 (9 matrices total).
     * The result is stored in row-major order: result[3*k + l] = ∂²M/∂p_k∂p_l
     *
     * For small rotations (ψ < 1e-8), returns zeros (second-order term).
     * For general rotations, computed analytically from Rodrigues formula.
     *
     * @return Array of 9 matrices, where result[3*k + l] = ∂²M/∂p_k∂p_l
     */
    std::array<Mat3, 9> rotation_matrix_second_derivatives() const;

    // ========== Multipole Management ==========

    /**
     * @brief Update lab-frame multipoles from body-frame multipoles
     *
     * This rotates the body-frame multipoles using the current orientation
     * and caches the result in multipole_lab.
     */
    void update_lab_multipoles();

    /**
     * @brief Get multipoles in lab frame (updating if necessary)
     */
    const occ::dma::Mult& get_lab_multipoles();

    /**
     * @brief Mark lab multipoles as invalid (call after orientation change)
     */
    void invalidate_lab_multipoles() { multipole_lab_valid = false; }

    // ========== Multi-site Helpers ==========

    /**
     * @brief Check if this molecule uses multi-site representation
     */
    bool is_multi_site() const { return !sites_body.empty(); }

    /**
     * @brief Get number of multipole sites
     */
    int num_sites() const {
        return sites_body.empty() ? 1 : static_cast<int>(sites_body.size());
    }

    /**
     * @brief Set single-site multipole (clears any multi-site data)
     */
    void set_single_site(const occ::dma::Mult& mult) {
        multipole_body = mult;
        sites_body.clear();
        multipole_lab_valid = false;
    }

    /**
     * @brief Set multi-site data (replaces single-site multipole_body)
     */
    void set_sites(const std::vector<BodySite>& sites) {
        sites_body = sites;
        multipole_lab_valid = false;
    }

    // ========== Inertia Tensor Management ==========

    /**
     * @brief Set inertia tensor (and compute inverse)
     */
    void set_inertia_tensor(const Mat3& I);

    /**
     * @brief Set isotropic (spherical) inertia
     */
    void set_spherical_inertia(double I_moment);

    /**
     * @brief Set diagonal inertia tensor
     */
    void set_diagonal_inertia(double Ixx, double Iyy, double Izz);

    /**
     * @brief Get inertia tensor in space frame (rotated from body frame)
     */
    Mat3 inertia_space() const;

    // ========== Energy Calculations ==========

    /**
     * @brief Compute translational kinetic energy
     */
    double translational_kinetic_energy() const;

    /**
     * @brief Compute rotational kinetic energy
     *
     * KE_rot = (1/2) * omega^T * I_body * omega
     */
    double rotational_kinetic_energy() const;

    /**
     * @brief Compute total kinetic energy
     */
    double kinetic_energy() const;

    // ========== Angular Momentum ==========

    /**
     * @brief Compute angular momentum in body frame
     */
    Vec3 angular_momentum_body() const;

    /**
     * @brief Compute angular momentum in space frame
     */
    Vec3 angular_momentum_space() const;
};

} // namespace occ::mults
