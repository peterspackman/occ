#include <occ/mults/rigid_body_dynamics.h>
#include <occ/mults/torque.h>
#include <occ/mults/multipole_interactions.h>
#include <occ/mults/cartesian_force.h>
#include <iostream>

namespace occ::mults {

// ==================== Integration Steps ====================

void RigidBodyDynamics::update_positions(std::vector<RigidBodyState>& molecules, double dt) {
    for (auto& mol : molecules) {
        // r(t+dt) = r(t) + v(t)*dt + (1/2)*a(t)*dt^2
        Vec3 accel = mol.force / mol.mass;
        mol.position += mol.velocity * dt + 0.5 * accel * dt * dt;
    }
}

void RigidBodyDynamics::update_velocities(std::vector<RigidBodyState>& molecules,
                                         double dt, bool first_half) {
    for (auto& mol : molecules) {
        Vec3 accel = mol.force / mol.mass;
        if (first_half) {
            // First half-step: v(t+dt/2) = v(t) + (1/2)*a(t)*dt
            mol.velocity += 0.5 * accel * dt;
        } else {
            // Second half-step: v(t+dt) = v(t+dt/2) + (1/2)*a(t+dt)*dt
            mol.velocity += 0.5 * accel * dt;
        }
    }
}

void RigidBodyDynamics::update_orientations_quaternion(
    std::vector<RigidBodyState>& molecules, double dt) {

    for (auto& mol : molecules) {
        // Quaternion time derivative: dq/dt = (1/2) * q * omega_quat
        // where omega_quat = (0, omega_x, omega_y, omega_z)
        const Vec3& omega = mol.angular_velocity_body;

        // Create pure imaginary quaternion from angular velocity
        Eigen::Quaterniond omega_quat(0.0, omega(0), omega(1), omega(2));

        // Compute quaternion derivative
        Eigen::Quaterniond q_dot = mol.quaternion * omega_quat;
        q_dot.coeffs() *= 0.5;

        // Integrate: q(t+dt) = q(t) + dt * dq/dt
        mol.quaternion.coeffs() += dt * q_dot.coeffs();

        // Normalize to maintain unit quaternion
        mol.quaternion.normalize();

        // Mark Euler angles and lab multipoles as invalid
        mol.euler_valid = false;
        mol.invalidate_lab_multipoles();
    }
}

void RigidBodyDynamics::update_angular_velocities(
    std::vector<RigidBodyState>& molecules, double dt, bool first_half) {

    for (auto& mol : molecules) {
        // In body frame, Euler's equation is:
        // dL/dt = tau - omega x L
        // where L = I * omega
        //
        // For simplicity, we use: d(omega)/dt = I^{-1} * tau
        // (This neglects the omega x L term, which is small for small timesteps
        //  and would require iterative solution otherwise)

        Vec3 angular_accel = mol.inertia_inv_body * mol.torque_body;

        if (first_half) {
            // First half-step
            mol.angular_velocity_body += 0.5 * angular_accel * dt;
        } else {
            // Second half-step
            mol.angular_velocity_body += 0.5 * angular_accel * dt;
        }
    }
}

void RigidBodyDynamics::compute_forces_torques(std::vector<RigidBodyState>& molecules) {
    // Zero out forces and torques
    for (auto& mol : molecules) {
        mol.force.setZero();
        mol.torque_body.setZero();
        mol.torque_euler.setZero();
    }

    // Update lab-frame multipoles for all molecules
    for (auto& mol : molecules) {
        mol.update_lab_multipoles();
    }

    // Compute pairwise interactions
    for (size_t i = 0; i < molecules.size(); i++) {
        for (size_t j = i + 1; j < molecules.size(); j++) {
            auto& mol_i = molecules[i];
            auto& mol_j = molecules[j];

            // Get Euler angles for both molecules
            Vec3 euler_i = mol_i.get_euler_angles();
            Vec3 euler_j = mol_j.get_euler_angles();

            // Compute torque on molecule i
            TorqueResult torque_i = TorqueCalculation::compute_torque(
                mol_i.multipole_body, mol_i.position, euler_i,
                mol_j.multipole_body, mol_j.position, euler_j,
                1  // molecule 1
            );

            // Compute torque on molecule j
            TorqueResult torque_j = TorqueCalculation::compute_torque(
                mol_i.multipole_body, mol_i.position, euler_i,
                mol_j.multipole_body, mol_j.position, euler_j,
                2  // molecule 2
            );

            // Accumulate forces (Newton's third law: F_j = -F_i)
            mol_i.force += torque_i.force;
            mol_j.force -= torque_i.force;  // Equal and opposite

            // Accumulate torques
            mol_i.torque_body += torque_i.torque_body;
            mol_i.torque_euler += torque_i.torque_euler;

            mol_j.torque_body += torque_j.torque_body;
            mol_j.torque_euler += torque_j.torque_euler;
        }
    }
}

void RigidBodyDynamics::velocity_verlet_step(std::vector<RigidBodyState>& molecules,
                                            double dt) {
    // 1. Half-step velocity update
    update_velocities(molecules, dt, true);
    update_angular_velocities(molecules, dt, true);

    // 2. Full-step position update
    update_positions(molecules, dt);
    update_orientations_quaternion(molecules, dt);

    // 3. Compute new forces and torques
    compute_forces_torques(molecules);

    // 4. Half-step velocity update
    update_velocities(molecules, dt, false);
    update_angular_velocities(molecules, dt, false);

    // 5. Normalize quaternions to prevent drift
    normalize_quaternions(molecules);
}

// ==================== Energy and Momentum ====================

double RigidBodyDynamics::compute_kinetic_energy(
    const std::vector<RigidBodyState>& molecules) {

    double KE = 0.0;
    for (const auto& mol : molecules) {
        KE += mol.kinetic_energy();
    }
    return KE;
}

double RigidBodyDynamics::compute_potential_energy(
    const std::vector<RigidBodyState>& molecules) {

    double PE = 0.0;

    // Compute pairwise interaction energies
    MultipoleInteractions::Config config;
    config.max_rank = 4;  // Adjust as needed
    MultipoleInteractions interactions(config);

    for (size_t i = 0; i < molecules.size(); i++) {
        for (size_t j = i + 1; j < molecules.size(); j++) {
            // Get non-const references to update multipoles if needed
            RigidBodyState& mol_i = const_cast<RigidBodyState&>(molecules[i]);
            RigidBodyState& mol_j = const_cast<RigidBodyState&>(molecules[j]);

            // Ensure lab multipoles are up to date
            mol_i.update_lab_multipoles();
            mol_j.update_lab_multipoles();

            const occ::dma::Mult& mult_i = mol_i.multipole_lab;
            const occ::dma::Mult& mult_j = mol_j.multipole_lab;

            double E_ij = interactions.compute_interaction_energy(
                mult_i, mol_i.position,
                mult_j, mol_j.position
            );

            PE += E_ij;
        }
    }

    return PE;
}

double RigidBodyDynamics::compute_total_energy(
    const std::vector<RigidBodyState>& molecules) {
    return compute_kinetic_energy(molecules) + compute_potential_energy(molecules);
}

Vec3 RigidBodyDynamics::compute_linear_momentum(
    const std::vector<RigidBodyState>& molecules) {

    Vec3 p = Vec3::Zero();
    for (const auto& mol : molecules) {
        p += mol.mass * mol.velocity;
    }
    return p;
}

Vec3 RigidBodyDynamics::compute_angular_momentum(
    const std::vector<RigidBodyState>& molecules) {

    Vec3 L = Vec3::Zero();
    for (const auto& mol : molecules) {
        // Orbital angular momentum: r x p
        Vec3 L_orbital = mol.position.cross(mol.mass * mol.velocity);

        // Spin angular momentum (in space frame)
        Vec3 L_spin = mol.angular_momentum_space();

        L += L_orbital + L_spin;
    }
    return L;
}

void RigidBodyDynamics::normalize_quaternions(std::vector<RigidBodyState>& molecules) {
    for (auto& mol : molecules) {
        mol.quaternion.normalize();
    }
}

// ==================== Cartesian Engine Methods ====================

double RigidBodyDynamics::compute_forces_torques_cartesian(
    std::vector<RigidBodyState>& molecules) {

    // Zero out forces and torques
    for (auto& mol : molecules) {
        mol.force.setZero();
        mol.torque_body.setZero();
        mol.torque_euler.setZero();
    }

    // Build MultipleSources from each RigidBodyState
    std::vector<MultipoleSource> sources;
    sources.reserve(molecules.size());
    for (const auto& mol : molecules) {
        sources.push_back(multipole_source_from_rigid_body(mol));
    }

    // Get CartesianMolecules (triggers lazy build with body-frame data)
    std::vector<const CartesianMolecule*> cartesians;
    cartesians.reserve(molecules.size());
    for (const auto& src : sources) {
        cartesians.push_back(&src.cartesian());
    }

    // Compute pairwise interactions
    double total_energy = 0.0;
    for (size_t i = 0; i < molecules.size(); i++) {
        for (size_t j = i + 1; j < molecules.size(); j++) {
            // One call per pair computes both molecules' forces and torques
            FullRigidBodyResult result = compute_molecule_forces_torques(
                *cartesians[i], *cartesians[j]);

            total_energy += result.energy;

            // Accumulate translational forces (Newton's third law built in)
            molecules[i].force += result.force_A;
            molecules[j].force += result.force_B;

            // Use the pre-computed body-frame torques directly.
            // These are validated against finite differences in cartesian_force_test.cpp.
            // Note: The Cartesian engine uses a different orientation parametrization
            // (infinitesimal lab-frame rotations) than the S-function engine (Euler angles),
            // so torque_euler conversion is not straightforward. For quaternion-based
            // dynamics, torque_body is what's needed.
            molecules[i].torque_body += result.torque_A_body;
            molecules[j].torque_body += result.torque_B_body;
        }
    }

    return total_energy;
}

double RigidBodyDynamics::compute_potential_energy_cartesian(
    const std::vector<RigidBodyState>& molecules) {

    // Build MultipleSources from each RigidBodyState
    std::vector<MultipoleSource> sources;
    sources.reserve(molecules.size());
    for (const auto& mol : molecules) {
        sources.push_back(multipole_source_from_rigid_body(mol));
    }

    // Get CartesianMolecules
    std::vector<const CartesianMolecule*> cartesians;
    cartesians.reserve(molecules.size());
    for (const auto& src : sources) {
        cartesians.push_back(&src.cartesian());
    }

    // Compute pairwise interaction energies
    double total_energy = 0.0;
    for (size_t i = 0; i < molecules.size(); i++) {
        for (size_t j = i + 1; j < molecules.size(); j++) {
            total_energy += compute_molecule_interaction(
                *cartesians[i], *cartesians[j]);
        }
    }

    return total_energy;
}

} // namespace occ::mults
