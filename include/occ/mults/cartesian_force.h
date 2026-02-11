#pragma once
#include <occ/mults/cartesian_molecule.h>
#include <occ/core/linear_algebra.h>
#include <Eigen/Geometry>
#include <vector>

namespace occ::mults {

/// Energy and force gradient for a single site pair.
struct CartesianForceResult {
    double energy = 0.0;
    Vec3 gradient = Vec3::Zero(); // dE/dR where R = posB - posA
                                  // Force on A = +gradient
                                  // Force on B = -gradient
};

/// Per-site forces for a molecule pair interaction.
struct MoleculeForceResult {
    double energy = 0.0;
    std::vector<Vec3> forces_A; // Force on each site of molecule A
    std::vector<Vec3> forces_B; // Force on each site of molecule B
};

/// Rigid body force and torque (lever-arm only, no multipole rotation).
struct RigidBodyForceResult {
    double energy = 0.0;
    Vec3 force = Vec3::Zero();       // Total translational force (lab frame)
    Vec3 torque_lab = Vec3::Zero();  // Torque in lab frame
    Vec3 torque_body = Vec3::Zero(); // Torque in body frame: M^T * torque_lab
};

/// Full rigid body result including multipole rotation contribution.
struct FullRigidBodyResult {
    double energy = 0.0;
    Vec3 force_A = Vec3::Zero();           // Translational force on A
    Vec3 force_B = Vec3::Zero();           // Translational force on B
    Vec3 torque_A_body = Vec3::Zero();     // Total torque on A (body frame)
    Vec3 torque_B_body = Vec3::Zero();     // Total torque on B (body frame)
    Vec3 grad_angle_axis_A = Vec3::Zero(); // dE/dp_A (angle-axis gradient)
    Vec3 grad_angle_axis_B = Vec3::Zero(); // dE/dp_B (angle-axis gradient)
};

/**
 * @brief Convert angle-axis gradient to quaternion gradient
 *
 * For optimization with quaternions, this computes the gradient in the
 * tangent space of the unit quaternion manifold. The result can be used
 * directly for gradient descent:
 *
 *   q_new = (q - step * grad_quat).normalized()
 *
 * Or equivalently using quaternion multiplication:
 *
 *   delta_q = Quaternion(1, -step*g[0]/2, -step*g[1]/2, -step*g[2]/2)
 *   q_new = (delta_q * q).normalized()
 *
 * @param q Current orientation quaternion (must be normalized)
 * @param grad_angle_axis The angle-axis gradient (∂E/∂θ for rotations about lab axes)
 * @return Quaternion gradient in tangent space (4 components: w, x, y, z)
 */
inline Eigen::Vector4d quaternion_gradient(
    const Eigen::Quaterniond& q,
    const Vec3& grad_angle_axis) {
    // The angle-axis gradient g represents ∂E/∂θ for infinitesimal rotations
    // about lab-frame axes. For a quaternion q, a small rotation δθ corresponds
    // to multiplying by δq ≈ (1, δθ/2).
    //
    // The quaternion gradient is the derivative of E with respect to q components,
    // projected onto the tangent space (perpendicular to q).
    //
    // For left-multiplication (lab-frame rotation): δq * q
    // The gradient is: ∂E/∂q = (1/2) * (0, g) * q  (quaternion product)

    Eigen::Quaterniond omega_quat(0.0,
                                   grad_angle_axis[0] / 2.0,
                                   grad_angle_axis[1] / 2.0,
                                   grad_angle_axis[2] / 2.0);
    Eigen::Quaterniond grad_quat = omega_quat * q;

    return grad_quat.coeffs();  // Returns (x, y, z, w) - Eigen's internal order
}

/**
 * @brief Apply quaternion gradient update (gradient descent step)
 *
 * Updates the quaternion in the direction of steepest descent.
 *
 * @param q Current orientation quaternion (will be modified in place)
 * @param grad_angle_axis The angle-axis gradient
 * @param step Step size (learning rate)
 */
inline void apply_quaternion_gradient_step(
    Eigen::Quaterniond& q,
    const Vec3& grad_angle_axis,
    double step) {
    // Small rotation in direction of negative gradient
    Eigen::Quaterniond delta(1.0,
                              -step * grad_angle_axis[0] / 2.0,
                              -step * grad_angle_axis[1] / 2.0,
                              -step * grad_angle_axis[2] / 2.0);
    q = delta * q;
    q.normalize();
}

/// Compute energy AND force gradient for a single pair of precomputed sites.
CartesianForceResult compute_site_pair_energy_force(
    const CartesianSite &siteA,
    const CartesianSite &siteB);

/// Compute per-site forces for two molecules.
MoleculeForceResult compute_molecule_forces(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB);

/// Aggregate per-site forces into rigid body force and lever-arm torque.
RigidBodyForceResult aggregate_rigid_body_forces(
    const std::vector<Vec3> &site_forces,
    const std::vector<Vec3> &site_positions,
    const Vec3 &center_of_mass,
    const Mat3 &rotation = Mat3::Identity());

/// Full force/torque including multipole rotation contribution.
/// Requires body-frame data in both molecules.
/// If site_cutoff > 0, skip site pairs with distance > site_cutoff (Angstrom).
FullRigidBodyResult compute_molecule_forces_torques(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB,
    double site_cutoff = 0.0);

} // namespace occ::mults
