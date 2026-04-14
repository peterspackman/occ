#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>

namespace occ::mults {

// Forward declarations
class MultipoleInteractions;

/**
 * @brief Result structure for torque calculations
 *
 * Contains forces and torques in various representations for rigid body dynamics
 * and optimization.
 *
 * Note: For optimization using angle-axis coordinates, convert torque_euler
 * to angle-axis gradients using RigidBodyState::angle_axis_jacobian():
 *
 *   grad_aa = J^T * torque_euler
 *
 * where J = state.angle_axis_jacobian()
 */
struct TorqueResult {
    Vec3 force;              // Force in Cartesian coordinates (lab frame)
    Vec3 torque_euler;       // Torque w.r.t. Euler angles (dE/dalpha, dE/dbeta, dE/dgamma)
    Vec3 torque_body;        // Torque in body-fixed frame
    Vec3 torque_space;       // Torque in space-fixed (lab) frame
    Vec3 grad_angle_axis;    // Gradient w.r.t. angle-axis parameters (dE/dpx, dE/dpy, dE/dpz)

    TorqueResult() : force(Vec3::Zero()), torque_euler(Vec3::Zero()),
                     torque_body(Vec3::Zero()), torque_space(Vec3::Zero()),
                     grad_angle_axis(Vec3::Zero()) {}
};

/**
 * @brief Jacobian relating Euler angle rates to angular velocity
 *
 * For ZYZ Euler angles (alpha, beta, gamma), the angular velocity in body frame is:
 *   omega_body = J(alpha, beta, gamma) * [dalpha/dt, dbeta/dt, dgamma/dt]^T
 *
 * The Jacobian J is:
 *   [ sin(beta)*sin(gamma),  cos(gamma),  0 ]
 *   [ sin(beta)*cos(gamma), -sin(gamma),  0 ]
 *   [ cos(beta),             0,           1 ]
 */
struct EulerJacobian {
    Mat3 J;      // Jacobian: omega_body = J * euler_rates
    Mat3 J_inv;  // Inverse Jacobian: euler_rates = J_inv * omega_body

    EulerJacobian(double alpha, double beta, double gamma);
};

/**
 * @brief Calculate forces and torques on rigid bodies from multipole interactions
 *
 * This class provides methods to compute forces and torques acting on molecules
 * due to multipole-multipole interactions. It supports both finite-difference
 * and analytical approaches (when available).
 */
class TorqueCalculation {
public:
    /**
     * @brief Compute forces and torques on a molecule from multipole interaction
     *
     * @param mult1 First multipole (body frame)
     * @param pos1 Position of first molecule
     * @param euler1 Euler angles of first molecule [alpha, beta, gamma]
     * @param mult2 Second multipole (body frame)
     * @param pos2 Position of second molecule
     * @param euler2 Euler angles of second molecule [alpha, beta, gamma]
     * @param which_molecule Which molecule to compute torque for (1 or 2)
     * @return TorqueResult containing force and torques in various representations
     */
    static TorqueResult compute_torque(
        const occ::dma::Mult& mult1, const Vec3& pos1, const Vec3& euler1,
        const occ::dma::Mult& mult2, const Vec3& pos2, const Vec3& euler2,
        int which_molecule = 1);

    /**
     * @brief Compute forces and torques using analytical derivatives
     *
     * Uses S-function derivatives and derivative transformation matrices
     * to compute exact forces and torques. This is much faster than
     * finite differences (no extra energy evaluations needed).
     *
     * @param mult1_body First multipole in body frame
     * @param pos1 Position of first molecule
     * @param euler1 Euler angles of first molecule [alpha, beta, gamma]
     * @param mult2_body Second multipole in body frame
     * @param pos2 Position of second molecule
     * @param euler2 Euler angles of second molecule [alpha, beta, gamma]
     * @param which_molecule Which molecule to compute torque for (1 or 2)
     * @return TorqueResult with analytical forces and torques
     */
    static TorqueResult compute_torque_analytical(
        const occ::dma::Mult& mult1_body, const Vec3& pos1, const Vec3& euler1,
        const occ::dma::Mult& mult2_body, const Vec3& pos2, const Vec3& euler2,
        int which_molecule = 1);

    /**
     * @brief Compute torque using finite differences (for validation)
     *
     * Computes torques by numerically differentiating the energy with respect
     * to Euler angles and positions. This is slower but more general than
     * analytical methods.
     *
     * @param mult1_body First multipole in body frame
     * @param pos1 Position of first molecule
     * @param euler1 Euler angles of first molecule [alpha, beta, gamma]
     * @param mult2_body Second multipole in body frame
     * @param pos2 Position of second molecule
     * @param euler2 Euler angles of second molecule [alpha, beta, gamma]
     * @param which_molecule Which molecule to compute torque for (1 or 2)
     * @param delta Step size for finite differences (default: 1e-6)
     * @return TorqueResult with finite-difference torques
     */
    static TorqueResult compute_torque_finite_diff(
        const occ::dma::Mult& mult1_body, const Vec3& pos1, const Vec3& euler1,
        const occ::dma::Mult& mult2_body, const Vec3& pos2, const Vec3& euler2,
        int which_molecule = 1,
        double delta = 1e-6);

    /**
     * @brief Convert Euler angle derivatives to body-frame torque
     *
     * Given torques with respect to Euler angles (dE/dalpha, dE/dbeta, dE/dgamma),
     * convert to torque in the body-fixed frame using the Euler Jacobian transpose.
     *
     * Relationship: tau_body = J^T * tau_euler
     *
     * @param torque_euler Torque w.r.t. Euler angles
     * @param euler Euler angles [alpha, beta, gamma]
     * @return Torque in body-fixed frame
     */
    static Vec3 euler_to_body_torque(const Vec3& torque_euler, const Vec3& euler);

    /**
     * @brief Convert Euler angle derivatives to space-frame torque
     *
     * @param torque_euler Torque w.r.t. Euler angles
     * @param euler Euler angles [alpha, beta, gamma]
     * @return Torque in space-fixed (lab) frame
     */
    static Vec3 euler_to_space_torque(const Vec3& torque_euler, const Vec3& euler);

private:
    // Helper: compute interaction energy for given configuration
    static double compute_interaction_energy(
        const occ::dma::Mult& mult1_body, const Vec3& pos1, const Vec3& euler1,
        const occ::dma::Mult& mult2_body, const Vec3& pos2, const Vec3& euler2);
};

} // namespace occ::mults
