#pragma once
#include <occ/mults/rigid_body.h>
#include <vector>

namespace occ::mults {

/**
 * @brief Rigid body molecular dynamics integrator
 *
 * This class implements the velocity Verlet algorithm for rigid body dynamics,
 * integrating both translational and rotational equations of motion.
 *
 * The quaternion-based formulation avoids gimbal lock and provides numerical
 * stability for arbitrary rotations.
 */
class RigidBodyDynamics {
public:
    /**
     * @brief Configuration for MD simulations
     */
    struct Config {
        double timestep;              // Integration timestep (fs)
        int num_steps;                // Total number of MD steps
        int output_frequency;         // Write trajectory every N steps
        bool use_quaternions;         // Use quaternion integration (recommended)
        bool conserve_momentum;       // Remove COM motion (not yet implemented)

        Config()
            : timestep(0.5), num_steps(1000), output_frequency(10),
              use_quaternions(true), conserve_momentum(false) {}
    };

    /**
     * @brief Perform one velocity Verlet integration step
     *
     * This integrates both translational and rotational degrees of freedom
     * using the velocity Verlet algorithm:
     *
     * 1. Update positions and orientations using current velocities
     * 2. Compute new forces and torques
     * 3. Update velocities using average of old and new forces/torques
     *
     * @param molecules Vector of rigid body states to integrate
     * @param dt Timestep in femtoseconds
     */
    static void velocity_verlet_step(std::vector<RigidBodyState>& molecules, double dt);

    /**
     * @brief Compute forces and torques on all molecules
     *
     * Calculates pairwise multipole interactions and updates force/torque
     * on each molecule.
     *
     * @param molecules Vector of rigid body states
     */
    static void compute_forces_torques(std::vector<RigidBodyState>& molecules);

    /**
     * @brief Update positions using current velocities (half-step)
     *
     * r(t+dt) = r(t) + v(t)*dt + (1/2)*a(t)*dt^2
     *
     * @param molecules Vector of rigid bodies
     * @param dt Timestep
     */
    static void update_positions(std::vector<RigidBodyState>& molecules, double dt);

    /**
     * @brief Update velocities using forces (half-step)
     *
     * v(t+dt) = v(t) + (1/2)*[a(t) + a(t+dt)]*dt
     *
     * @param molecules Vector of rigid bodies
     * @param dt Timestep
     * @param first_half If true, do first half-step; if false, do second half-step
     */
    static void update_velocities(std::vector<RigidBodyState>& molecules, double dt,
                                  bool first_half);

    /**
     * @brief Update orientations using quaternion integration
     *
     * Integrates quaternion using angular velocity in body frame:
     * dq/dt = (1/2) * q * omega_quat
     *
     * @param molecules Vector of rigid bodies
     * @param dt Timestep
     */
    static void update_orientations_quaternion(std::vector<RigidBodyState>& molecules,
                                               double dt);

    /**
     * @brief Update angular velocities using torques (half-step)
     *
     * Integrates Euler's equation in body frame:
     * dL/dt = tau - omega x L
     *
     * @param molecules Vector of rigid bodies
     * @param dt Timestep
     * @param first_half If true, do first half-step; if false, do second half-step
     */
    static void update_angular_velocities(std::vector<RigidBodyState>& molecules,
                                         double dt, bool first_half);

    /**
     * @brief Compute total kinetic energy of system
     *
     * @param molecules Vector of rigid bodies
     * @return Total kinetic energy (translational + rotational)
     */
    static double compute_kinetic_energy(const std::vector<RigidBodyState>& molecules);

    /**
     * @brief Compute total potential energy of system
     *
     * @param molecules Vector of rigid bodies
     * @return Total potential energy (sum of all pairwise interactions)
     */
    static double compute_potential_energy(const std::vector<RigidBodyState>& molecules);

    /**
     * @brief Compute total energy (kinetic + potential)
     *
     * @param molecules Vector of rigid bodies
     * @return Total energy
     */
    static double compute_total_energy(const std::vector<RigidBodyState>& molecules);

    /**
     * @brief Compute total linear momentum of system
     *
     * @param molecules Vector of rigid bodies
     * @return Total linear momentum vector
     */
    static Vec3 compute_linear_momentum(const std::vector<RigidBodyState>& molecules);

    /**
     * @brief Compute total angular momentum of system
     *
     * @param molecules Vector of rigid bodies
     * @return Total angular momentum vector (in space frame)
     */
    static Vec3 compute_angular_momentum(const std::vector<RigidBodyState>& molecules);

    /**
     * @brief Normalize all quaternions in the system
     *
     * Call this periodically to prevent numerical drift from unit quaternion constraint.
     *
     * @param molecules Vector of rigid bodies
     */
    static void normalize_quaternions(std::vector<RigidBodyState>& molecules);
};

} // namespace occ::mults
