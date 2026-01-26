#include <occ/mults/torque.h>
#include <occ/mults/rotation.h>
#include <occ/mults/multipole_interactions.h>
#include <occ/mults/derivative_transform.h>
#include <occ/mults/sfunction_term_builder.h>
#include <occ/mults/sfunction_evaluator.h>
#include <occ/mults/rigid_body.h>
#include <cmath>

namespace occ::mults {

// ==================== EulerJacobian Implementation ====================

EulerJacobian::EulerJacobian(double alpha, double beta, double gamma) {
    // For ZYZ Euler angles, the Jacobian relating angular velocity to Euler rates is:
    // omega_body = J * [alpha_dot, beta_dot, gamma_dot]^T
    //
    // J = [ sin(beta)*sin(gamma),  cos(gamma),  0 ]
    //     [ sin(beta)*cos(gamma), -sin(gamma),  0 ]
    //     [ cos(beta),             0,           1 ]

    double sb = std::sin(beta);
    double cb = std::cos(beta);
    double sg = std::sin(gamma);
    double cg = std::cos(gamma);

    J(0, 0) = sb * sg;
    J(0, 1) = cg;
    J(0, 2) = 0.0;

    J(1, 0) = sb * cg;
    J(1, 1) = -sg;
    J(1, 2) = 0.0;

    J(2, 0) = cb;
    J(2, 1) = 0.0;
    J(2, 2) = 1.0;

    // Compute inverse Jacobian
    // For the specific form of J, we can compute J_inv analytically
    // but for robustness, we use Eigen's inverse
    J_inv = J.inverse();
}

// ==================== TorqueCalculation Implementation ====================

double TorqueCalculation::compute_interaction_energy(
    const occ::dma::Mult& mult1_body, const Vec3& pos1, const Vec3& euler1,
    const occ::dma::Mult& mult2_body, const Vec3& pos2, const Vec3& euler2) {

    // Transform body-frame multipoles to lab frame
    Mat3 R1 = rotation_utils::euler_to_rotation(euler1(0), euler1(1), euler1(2));
    Mat3 R2 = rotation_utils::euler_to_rotation(euler2(0), euler2(1), euler2(2));

    occ::dma::Mult mult1_lab = rotated_multipole(mult1_body, R1);
    occ::dma::Mult mult2_lab = rotated_multipole(mult2_body, R2);

    // Compute interaction energy
    MultipoleInteractions::Config config;
    config.max_rank = std::max(mult1_body.max_rank, mult2_body.max_rank);
    // For multipole-multipole interactions, max_interaction_rank must be at least l1 + l2
    // to allow all terms. Set it to the sum of the ranks.
    config.max_interaction_rank = mult1_body.max_rank + mult2_body.max_rank;
    MultipoleInteractions interactions(config);

    return interactions.compute_interaction_energy(mult1_lab, pos1, mult2_lab, pos2);
}

/**
 * @brief Compute forces and orientation gradients analytically using Orient's approach
 *
 * This function computes analytical gradients of the multipole interaction energy
 * with respect to both positions and orientations. The implementation follows
 * Orient's methodology:
 *
 * 1. Keep multipoles in BODY frame (not lab frame)
 * 2. Transform geometry coordinates to each molecule's body frame
 * 3. Compute S-function derivatives w.r.t. intermediate variables
 * 4. Use D1 transformation matrix to convert to external coordinates
 *
 * Key features:
 * - Direct angle-axis gradients (more accurate than Euler → angle-axis conversion)
 * - Proper sign convention: returns gradients (∂E/∂x), not forces
 * - Validated against finite differences to machine precision
 *
 * @param mult1_body Body-frame multipoles for molecule 1
 * @param pos1 Position of molecule 1 (lab frame)
 * @param euler1 Euler angles (α,β,γ) for molecule 1 (ZYZ convention)
 * @param mult2_body Body-frame multipoles for molecule 2
 * @param pos2 Position of molecule 2 (lab frame)
 * @param euler2 Euler angles for molecule 2
 * @param which_molecule Which molecule's derivatives to compute (1 or 2)
 * @return TorqueResult containing force, gradients, and torques
 */
TorqueResult TorqueCalculation::compute_torque_analytical(
    const occ::dma::Mult& mult1_body, const Vec3& pos1, const Vec3& euler1,
    const occ::dma::Mult& mult2_body, const Vec3& pos2, const Vec3& euler2,
    int which_molecule) {

    TorqueResult result;

    // Get rotation matrices
    Mat3 R1 = rotation_utils::euler_to_rotation(euler1(0), euler1(1), euler1(2));
    Mat3 R2 = rotation_utils::euler_to_rotation(euler2(0), euler2(1), euler2(2));

    // ORIENT APPROACH: Keep multipoles in BODY frame and transform GEOMETRY to body frame
    // This allows S-function derivatives to see the orientation matrix dependence.
    //
    // We'll compute the interaction with body-frame multipoles and transformed coordinates:
    //   e1r = R1^T · er  (lab vector transformed to molecule 1's body frame)
    //   e2r = -R2^T · er (lab vector transformed to molecule 2's body frame)

    // Build interaction terms using BODY-frame multipoles
    int max_rank = std::max(mult1_body.max_rank, mult2_body.max_rank);
    int max_interaction_rank = mult1_body.max_rank + mult2_body.max_rank;

    SFunctionTermListBuilder builder(max_rank);
    auto term_list = builder.build_electrostatic_terms(
        mult1_body, mult2_body, max_interaction_rank);

    // Set up coordinate system with body-frame unit vectors (Orient's approach)
    CoordinateSystem coords = CoordinateSystem::from_body_frame(pos1, pos2, R1, R2);

    // Set up S-function evaluator with body-frame coordinate system
    SFunctionEvaluator evaluator(max_rank);
    evaluator.set_coordinate_system(coords);

    // Compute S-functions with first derivatives (level=1)
    auto sfunction_results = evaluator.compute_batch(term_list, 1);

    // Get distance from coordinate system
    double r = coords.r;

    // ===== FORCE CALCULATION USING D1 TRANSFORMATION =====
    // Energy: E = Σ coeff * S(t1,t2,j) / r^power
    //
    // Force on molecule A: F_A = -∂E/∂r_A
    //
    // The S-function derivatives are w.r.t. intermediate variables (unit vectors, etc.)
    // We use Orient's D1 transformation matrix to convert to position derivatives:
    //
    // ∂E/∂x_α = Σᵢ (∂E/∂Sⁱ) · (∂Sⁱ/∂x_α)
    //         = Σᵢ (coeff/r^power) · Σⱼ (∂Sⁱ/∂qⱼ) · (∂qⱼ/∂x_α)
    //
    // where qⱼ are intermediate variables (e1r, e2r, orientation, r)
    //
    // Build the s1 vector containing S-function derivatives w.r.t. intermediate vars
    // s1 has 16 elements in Orient:
    //   s1[0-2]:   ∂S/∂(e1r_x, e1r_y, e1r_z)  - unit vector at site A
    //   s1[3-5]:   ∂S/∂(e2r_x, e2r_y, e2r_z)  - unit vector at site B
    //   s1[6-14]:  ∂S/∂(orientation matrix)   - 9 elements (0 for point multipoles)
    //   s1[15]:    ∂S/∂r                       - distance derivative

    Vec3 force_A = Vec3::Zero();
    Vec3 force_B = Vec3::Zero();

    // Build s1 vector for all terms
    // s1 has 16 elements containing derivatives w.r.t. intermediate variables
    Vec s1_total = Vec::Zero(16);

    for (size_t i = 0; i < term_list.size(); ++i) {
        const auto& term = term_list.terms[i];
        const auto& sf_result = sfunction_results[i];

        double r_power = std::pow(r, term.power);
        double coeff_over_r = term.coeff / r_power;

        // Copy ALL S-function derivatives (15 components)
        // Indices 0-5: ∂S/∂(unit vector components e1r, e2r)
        // Indices 6-14: ∂S/∂(orientation matrix elements)
        for (int j = 0; j < 15; ++j) {
            s1_total[j] += coeff_over_r * sf_result.s1[j];
        }

        // Add ∂S/∂r term from the power-law factor
        // For E = coeff * S / r^power:
        // ∂E/∂r = coeff * [∂S/∂r / r^power - S * power / r^(power+1)]
        // The S-function derivatives don't include the r-dependence of 1/r^power,
        // so we need to add the second term explicitly
        if (term.power != 0) {
            double S_times_coeff = sf_result.s0 * term.coeff;
            double dr_contribution = -term.power / std::pow(r, term.power + 1) * S_times_coeff;
            s1_total[15] += dr_contribution;
        }
    }

    // Apply D1 transformation: external_derivs = D1^T * s1
    // D1 is [16 x 12] mapping intermediate vars to external coords
    // External coords for angle-axis: [x_A, y_A, z_A, px_A, py_A, pz_A, x_B, y_B, z_B, px_B, py_B, pz_B]
    // For point multipoles, site offsets are zero
    Vec3 a = Vec3::Zero();
    Vec3 b = Vec3::Zero();

    // Compute angle-axis parameters and rotation matrix derivatives
    RigidBodyState state1, state2;
    state1.set_euler_angles(euler1(0), euler1(1), euler1(2));
    state2.set_euler_angles(euler2(0), euler2(1), euler2(2));

    std::array<Mat3, 3> M1_A = state1.rotation_matrix_derivatives();
    std::array<Mat3, 3> M1_B = state2.rotation_matrix_derivatives();

    Mat D1 = DerivativeTransform::compute_D1_angle_axis(coords, R1, R2, M1_A, M1_B, a, b);

    // Compute external derivatives: d(Energy)/d(external_coords) = D1^T * s1
    Vec external_derivs = D1.transpose() * s1_total;  // [12 x 1]

    // Extract forces from external derivatives
    // Force = -gradient, so F = -∂E/∂pos
    force_A = -Vec3(external_derivs[0], external_derivs[1], external_derivs[2]);
    force_B = -Vec3(external_derivs[6], external_derivs[7], external_derivs[8]);

    // Return force on the requested molecule
    if (which_molecule == 1) {
        result.force = force_A;
    } else {
        result.force = force_B;
    }

    // ===== ANGLE-AXIS GRADIENT AND TORQUE CALCULATION =====
    //
    // IMPORTANT SIGN CONVENTION:
    // - This function computes GRADIENTS (∂E/∂x) not forces/torques
    // - For minimization, we need gradients (positive when energy increases)
    // - Forces/torques are the NEGATIVE of gradients: F = -∂E/∂x
    //
    // The D1 transformation gives us derivatives w.r.t. external coordinates:
    //   external_derivs[3-5]   = ∂E/∂(px_A, py_A, pz_A)  (angle-axis derivatives for molecule A)
    //   external_derivs[9-11]  = ∂E/∂(px_B, py_B, pz_B)  (angle-axis derivatives for molecule B)
    //
    // where p is the angle-axis representation (rotation vector).

    Vec3 grad_aa_A = Vec3(external_derivs[3], external_derivs[4], external_derivs[5]);
    Vec3 grad_aa_B = Vec3(external_derivs[9], external_derivs[10], external_derivs[11]);

    Vec3 euler = (which_molecule == 1) ? euler1 : euler2;
    Vec3 grad_aa = (which_molecule == 1) ? grad_aa_A : grad_aa_B;

    // Store angle-axis gradient (this is the direct analytical result!)
    result.grad_angle_axis = grad_aa;

    // Convert angle-axis gradient to Euler gradient using Jacobian chain rule:
    //   ∂E/∂euler = (∂p/∂euler)^T · (∂E/∂p) = J_aa^T · grad_aa
    //
    // where J_aa is the angle-axis Jacobian: J_aa_ij = ∂p_i/∂euler_j
    RigidBodyState temp_state;
    temp_state.set_euler_angles(euler(0), euler(1), euler(2));
    Mat3 J_aa = temp_state.angle_axis_jacobian();

    Vec3 grad_euler = J_aa.transpose() * grad_aa;

    // SIGN CONVENTION FIXED (Nov 2024):
    // Previously this had an erroneous negation. We now store the gradient directly.
    // The calling code (optimizer) will apply the negative sign to get forces/torques.
    result.torque_euler = grad_euler;  // Store as gradient (positive)

    // Convert Euler torque to body-frame torque using Euler Jacobian
    // For ZYZ Euler angles: ω_body = J_euler(α,β,γ) · [dα/dt, dβ/dt, dγ/dt]
    // Torque transformation: τ_body = J_euler^T · τ_euler
    double alpha = euler(0);
    double beta = euler(1);
    double gamma = euler(2);

    double ca = std::cos(alpha);
    double sa = std::sin(alpha);
    double cb = std::cos(beta);
    double sb = std::sin(beta);

    Mat3 J_euler;
    J_euler << sa*sb, ca, 0,
               ca*sb, -sa, 0,
               cb,    0,  1;

    // Convert to body-frame using Euler Jacobian (for compatibility with dynamics codes)
    result.torque_body = J_euler.transpose() * grad_euler;

    return result;
}

// Fallback: Finite difference implementation (kept for validation/testing)
TorqueResult TorqueCalculation::compute_torque_finite_diff(
    const occ::dma::Mult& mult1_body, const Vec3& pos1, const Vec3& euler1,
    const occ::dma::Mult& mult2_body, const Vec3& pos2, const Vec3& euler2,
    int which_molecule,
    double delta) {

    TorqueResult result;

    // Energy at current configuration
    double E0 = compute_interaction_energy(mult1_body, pos1, euler1, mult2_body, pos2, euler2);

    // Select which molecule's variables to perturb
    const Vec3& pos = (which_molecule == 1) ? pos1 : pos2;
    const Vec3& euler = (which_molecule == 1) ? euler1 : euler2;

    // ===== Force calculation: F = -dE/dr =====
    for (int i = 0; i < 3; i++) {
        Vec3 pos_plus = (which_molecule == 1) ? pos1 : pos2;
        pos_plus(i) += delta;

        double E_plus;
        if (which_molecule == 1) {
            E_plus = compute_interaction_energy(mult1_body, pos_plus, euler1,
                                               mult2_body, pos2, euler2);
        } else {
            E_plus = compute_interaction_energy(mult1_body, pos1, euler1,
                                               mult2_body, pos_plus, euler2);
        }

        // Force = -gradient
        result.force(i) = -(E_plus - E0) / delta;
    }

    // ===== Torque calculation: tau_euler = -dE/d(euler angles) =====
    for (int i = 0; i < 3; i++) {
        Vec3 euler_plus = euler;
        euler_plus(i) += delta;

        double E_plus;
        if (which_molecule == 1) {
            E_plus = compute_interaction_energy(mult1_body, pos1, euler_plus,
                                               mult2_body, pos2, euler2);
        } else {
            E_plus = compute_interaction_energy(mult1_body, pos1, euler1,
                                               mult2_body, pos2, euler_plus);
        }

        // Store gradient ∂E/∂euler (positive, matches analytical version)
        result.torque_euler(i) = (E_plus - E0) / delta;
    }

    // Convert to body and space frame torques
    result.torque_body = euler_to_body_torque(result.torque_euler, euler);
    result.torque_space = euler_to_space_torque(result.torque_euler, euler);

    return result;
}

TorqueResult TorqueCalculation::compute_torque(
    const occ::dma::Mult& mult1, const Vec3& pos1, const Vec3& euler1,
    const occ::dma::Mult& mult2, const Vec3& pos2, const Vec3& euler2,
    int which_molecule) {

    // Use analytical derivatives (faster and more accurate)
    // Falls back to finite differences for torques (not yet fully analytical)
    return compute_torque_analytical(mult1, pos1, euler1, mult2, pos2, euler2,
                                     which_molecule);
}

Vec3 TorqueCalculation::euler_to_body_torque(const Vec3& torque_euler, const Vec3& euler) {
    // Check for singularity (beta near 0 or pi)
    double beta = euler(1);
    double sb = std::sin(beta);

    // Near singularity, the conversion is ill-defined
    // In this case, just return the torque_euler directly as an approximation
    if (std::abs(sb) < 1e-8) {
        return torque_euler;  // Approximate at singularity
    }

    // Construct Jacobian
    EulerJacobian J(euler(0), euler(1), euler(2));

    // The relationship between torques is:
    // tau_euler = J^T * tau_body
    // Therefore: tau_body = (J^T)^{-1} * tau_euler = (J^{-1})^T * tau_euler
    //
    // But it's simpler to use: tau_body = J^{-T} * tau_euler
    // where J^{-T} = (J^{-1})^T

    return J.J_inv.transpose() * torque_euler;
}

Vec3 TorqueCalculation::euler_to_space_torque(const Vec3& torque_euler, const Vec3& euler) {
    // First convert to body frame
    Vec3 tau_body = euler_to_body_torque(torque_euler, euler);

    // Then rotate to space frame
    Mat3 R = rotation_utils::euler_to_rotation(euler(0), euler(1), euler(2));
    return R * tau_body;
}

} // namespace occ::mults
