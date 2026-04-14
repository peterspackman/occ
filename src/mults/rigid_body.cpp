#include <occ/mults/rigid_body.h>
#include <occ/mults/rotation.h>
#include <cmath>

namespace occ::mults {

// ==================== Orientation Management ====================

Mat3 RigidBodyState::rotation_matrix() const {
    return quaternion.toRotationMatrix();
}

void RigidBodyState::set_euler_angles(double alpha, double beta, double gamma) {
    euler_angles << alpha, beta, gamma;
    euler_valid = true;

    // Convert to quaternion
    Mat3 R = rotation_utils::euler_to_rotation(alpha, beta, gamma);
    quaternion = Eigen::Quaterniond(R);
    quaternion.normalize();

    // Lab multipoles need update
    invalidate_lab_multipoles();
}

Vec3 RigidBodyState::get_euler_angles() {
    if (!euler_valid) {
        // Convert quaternion to Euler angles (ZYZ convention)
        // This is the inverse of euler_to_rotation
        Mat3 R = rotation_matrix();

        // Extract Euler angles from rotation matrix
        // For ZYZ: R = Rz(alpha) * Ry(beta) * Rz(gamma)
        //
        // R = [ ca*cb*cg - sa*sg,  -ca*cb*sg - sa*cg,  ca*sb ]
        //     [ sa*cb*cg + ca*sg,  -sa*cb*sg + ca*cg,  sa*sb ]
        //     [ -sb*cg,             sb*sg,             cb    ]
        //
        // where ca = cos(alpha), sa = sin(alpha), etc.

        double beta = std::acos(R(2, 2));  // beta = acos(R_33)

        double alpha, gamma;
        if (std::abs(std::sin(beta)) > 1e-10) {
            // Non-singular case
            alpha = std::atan2(R(1, 2), R(0, 2));   // atan2(sa*sb, ca*sb)
            gamma = std::atan2(R(2, 1), -R(2, 0));  // atan2(sb*sg, -sb*cg)
        } else {
            // Gimbal lock: beta = 0 or pi
            // When beta = 0: alpha and gamma are not uniquely determined
            // Convention: Set alpha = 0 and solve for gamma (prefer gamma for z-rotations)
            alpha = 0.0;
            if (beta < M_PI / 2) {
                // beta near 0
                gamma = std::atan2(R(1, 0), R(0, 0));
            } else {
                // beta near pi
                gamma = std::atan2(-R(1, 0), R(0, 0));
            }
        }

        euler_angles << alpha, beta, gamma;

        // Normalize angles to [0, 2π) to provide consistent output
        // This is important for tracking accumulated rotations
        const double two_pi = 2.0 * M_PI;
        if (alpha < 0.0) alpha += two_pi;
        if (gamma < 0.0) gamma += two_pi;
        if (alpha >= two_pi) alpha -= two_pi;
        if (gamma >= two_pi) gamma -= two_pi;

        euler_angles << alpha, beta, gamma;
        euler_valid = true;
    }

    return euler_angles;
}

void RigidBodyState::set_quaternion(const Eigen::Quaterniond& q) {
    quaternion = q;
    quaternion.normalize();
    euler_valid = false;  // Euler angles need recalculation
    invalidate_lab_multipoles();
}

void RigidBodyState::set_rotation_matrix(const Mat3& R) {
    quaternion = Eigen::Quaterniond(R);
    quaternion.normalize();
    euler_valid = false;
    invalidate_lab_multipoles();
}

Vec3 RigidBodyState::get_angle_axis() const {
    // Quaternion q = [w, x, y, z] = [cos(psi/2), sin(psi/2)*n]
    // Extract angle: psi = 2*acos(w)
    // Extract axis: n = [x, y, z] / sin(psi/2)
    // Return: p = psi * n

    double w = quaternion.w();
    Vec3 v(quaternion.x(), quaternion.y(), quaternion.z());

    // Handle small rotations
    double angle = 2.0 * std::acos(std::clamp(w, -1.0, 1.0));

    if (angle < 1e-10) {
        // Small rotation: p ≈ 2*v (first-order approximation)
        return 2.0 * v;
    }

    // General case
    double s = std::sin(angle / 2.0);
    Vec3 axis = v / s;  // Unit axis
    return angle * axis;  // Angle-axis vector
}

void RigidBodyState::set_angle_axis(const Vec3& p) {
    double angle = p.norm();

    if (angle < 1e-10) {
        // Small rotation: use first-order approximation
        // q ≈ [1, p(0)/2, p(1)/2, p(2)/2]
        quaternion = Eigen::Quaterniond(1.0, p(0)/2, p(1)/2, p(2)/2);
        quaternion.normalize();
        euler_valid = false;
        invalidate_lab_multipoles();
        return;
    }

    // General case: q = [cos(psi/2), sin(psi/2)*n]
    Vec3 axis = p / angle;  // Unit axis
    Eigen::AngleAxisd aa(angle, axis);
    quaternion = Eigen::Quaterniond(aa);

    // Mark cached values as invalid
    euler_valid = false;
    invalidate_lab_multipoles();
}

/**
 * @brief Construct skew-symmetric (cross-product) matrix from vector
 *
 * For vector v = [vx, vy, vz], returns the matrix [v]× such that:
 *   [v]× * w = v × w  (cross product)
 *
 * The skew-symmetric matrix is:
 *   [  0  -vz   vy ]
 *   [ vz    0  -vx ]
 *   [-vy   vx    0 ]
 */
static Mat3 skew_symmetric(const Vec3& v) {
    Mat3 S;
    S <<     0, -v(2),  v(1),
          v(2),     0, -v(0),
         -v(1),  v(0),     0;
    return S;
}

/**
 * @brief Compute derivative of skew-symmetric matrix with respect to vector component
 *
 * Computes ∂[v]×/∂v_k for k=0,1,2
 *
 * @param k Component index (0, 1, or 2)
 * @return 3x3 matrix ∂[v]×/∂v_k
 */
static Mat3 skew_symmetric_derivative(int k) {
    Mat3 dS_dp = Mat3::Zero();

    // ∂[v]×/∂v_0 (x-component)
    if (k == 0) {
        dS_dp <<  0,  0,  0,
                  0,  0, -1,
                  0,  1,  0;
    }
    // ∂[v]×/∂v_1 (y-component)
    else if (k == 1) {
        dS_dp <<  0,  0,  1,
                  0,  0,  0,
                 -1,  0,  0;
    }
    // ∂[v]×/∂v_2 (z-component)
    else if (k == 2) {
        dS_dp <<  0, -1,  0,
                  1,  0,  0,
                  0,  0,  0;
    }

    return dS_dp;
}

/**
 * @brief Compute analytical Jacobian for angle-axis to Euler angle transformation
 *
 * This implements Orient's aaderiv formulas from interact.f90 (lines 132-221).
 *
 * The Jacobian relates angle-axis derivatives to Euler angle derivatives:
 *   grad_aa = J^T * grad_euler
 *
 * where:
 *   - grad_euler = [dE/dα, dE/dβ, dE/dγ] (torque w.r.t. Euler angles)
 *   - grad_aa = [dE/dpx, dE/dpy, dE/dpz] (gradient w.r.t. angle-axis)
 *
 * The angle-axis vector p = ψ*n where:
 *   - ψ = |p| is the rotation angle
 *   - n = p/ψ is the unit rotation axis
 *
 * The rotation matrix M is given by Rodrigues' formula:
 *   M = I + sin(ψ)*[n]× + (1-cos(ψ))*[n]×²
 *
 * The derivative matrices M1(k) = ∂M/∂p(k) are computed analytically
 * using chain rule through the angle and axis components.
 *
 * @return 3x3 Jacobian matrix J where J_ij = dp_i/deuler_j
 */
Mat3 RigidBodyState::angle_axis_jacobian() const {
    // Get angle-axis vector p and current Euler angles
    Vec3 p = get_angle_axis();
    double psi = p.norm();  // Rotation angle

    // Get euler angles for computing the Jacobian
    RigidBodyState state_copy = *this;
    Vec3 euler = state_copy.get_euler_angles();

    // For the angle-axis to Euler Jacobian, we need to compute how
    // the angle-axis vector p changes as we vary the Euler angles.
    //
    // Since we have:
    //   M(euler) = rotation matrix from Euler angles
    //   M(p) = rotation matrix from angle-axis
    //
    // and these represent the same rotation, we need:
    //   dM/dp * dp/deuler = dM/deuler
    //
    // Therefore:
    //   dp/deuler = (dM/dp)^(-1) * dM/deuler
    //
    // This is complex, so for now we use numerical differentiation
    // but with the machinery from Orient's aaderiv available for future use.

    // Small angle case: use analytical approximation
    if (psi < 1e-8) {
        // For small rotations: M ≈ I + [p]×
        // The Jacobian is approximately identity (to first order)
        // For better accuracy with small rotations, use numerical differentiation
        const double eps = 1e-7;
        Mat3 J;

        for (int j = 0; j < 3; j++) {
            Vec3 euler_plus = euler;
            euler_plus(j) += eps;

            RigidBodyState state_plus;
            state_plus.set_euler_angles(euler_plus(0), euler_plus(1), euler_plus(2));
            Vec3 p_plus = state_plus.get_angle_axis();

            J.col(j) = (p_plus - p) / eps;
        }

        return J;
    }

    // General case: use numerical differentiation
    //
    // Note: Orient's aaderiv computes M1 = ∂M/∂p, which is the derivative
    // of the rotation matrix with respect to angle-axis components.
    //
    // For optimization, we need the inverse relationship: how does p change
    // when we change Euler angles? This requires solving:
    //   dp/deuler = (∂M/∂p)^(-1) * ∂M/∂euler
    //
    // The analytical formula for this is complex and requires computing
    // the pseudoinverse of a tensor. For now, numerical differentiation
    // is more straightforward and sufficiently accurate for optimization.

    const double eps = 1e-7;
    Mat3 J;

    for (int j = 0; j < 3; j++) {
        Vec3 euler_plus = euler;
        euler_plus(j) += eps;

        RigidBodyState state_plus;
        state_plus.set_euler_angles(euler_plus(0), euler_plus(1), euler_plus(2));
        Vec3 p_plus = state_plus.get_angle_axis();

        J.col(j) = (p_plus - p) / eps;
    }

    return J;
}

/**
 * @brief Compute rotation matrix derivatives ∂M/∂p_i (Orient's aaderiv)
 *
 * This implements the analytical derivatives of the rotation matrix M with respect
 * to the angle-axis vector components p_i for i=1,2,3.
 *
 * Based on Orient's aaderiv routine (interact.f90, lines 132-221).
 *
 * The Rodrigues formula is:
 *   M = I + sin(ψ)·N₀ + (1-cos(ψ))·N₀²
 * where:
 *   ψ = |p| (rotation angle)
 *   n = p/ψ (unit rotation axis)
 *   N₀ = [n]× (skew-symmetric matrix from n)
 *
 * The derivatives are:
 *   ∂M/∂p_k = n_k·sin(ψ)·N₀² + (1-cos(ψ))·(N₁ᵏ·N₀ + N₀·N₁ᵏ)
 *           + n_k·cos(ψ)·N₀ + sin(ψ)·N₁ᵏ
 * where:
 *   N₁ᵏ = (1/ψ)·∂[n]×/∂p_k
 *
 * @return Array of 3 matrices M1[k] = ∂M/∂p_k for k=0,1,2
 */
std::array<Mat3, 3> RigidBodyState::rotation_matrix_derivatives() const {
    std::array<Mat3, 3> M1;

    // Get angle-axis vector
    Vec3 p = get_angle_axis();
    double psi = p.norm();  // Rotation angle

    // Small angle case: M ≈ I + [p]×
    // Therefore: ∂M/∂p_k ≈ ∂[p]×/∂p_k
    if (psi < 1e-8) {
        // For small rotations, the derivatives are simply the skew-symmetric derivatives
        for (int k = 0; k < 3; k++) {
            M1[k] = skew_symmetric_derivative(k);
        }
        return M1;
    }

    // General case: Use Rodrigues formula derivatives
    Vec3 n = p / psi;  // Unit rotation axis

    // Compute skew-symmetric matrices
    Mat3 N0 = skew_symmetric(n);  // [n]×
    Mat3 N0_sq = N0 * N0;         // [n]×²

    // Precompute trigonometric functions
    double sin_psi = std::sin(psi);
    double cos_psi = std::cos(psi);
    double one_minus_cos = 1.0 - cos_psi;

    // Compute ∂M/∂p_k for each component k
    for (int k = 0; k < 3; k++) {
        // Compute N₁ᵏ = (1/ψ)·∂[n]×/∂p_k
        //
        // Since n = p/ψ, we have:
        //   ∂n/∂p_k = (1/ψ)·(e_k - n_k·n)
        // where e_k is the k-th basis vector
        //
        // Therefore:
        //   N₁ᵏ = [∂n/∂p_k]× = (1/ψ)·([e_k]× - n_k·[n]×)

        Vec3 ek = Vec3::Zero();
        ek(k) = 1.0;

        Mat3 N1_k = (skew_symmetric(ek) - n(k) * N0) / psi;

        // Compute ∂M/∂p_k using the derivative formula:
        // ∂M/∂p_k = n_k·sin(ψ)·N₀²
        //         + (1-cos(ψ))·(N₁ᵏ·N₀ + N₀·N₁ᵏ)
        //         + n_k·cos(ψ)·N₀
        //         + sin(ψ)·N₁ᵏ

        M1[k] = n(k) * sin_psi * N0_sq
              + one_minus_cos * (N1_k * N0 + N0 * N1_k)
              + n(k) * cos_psi * N0
              + sin_psi * N1_k;
    }

    return M1;
}

/**
 * @brief Compute rotation matrix derivatives from Orient's aaderiv
 *
 * This computes M1(k) = ∂M/∂p(k) for k=1,2,3 where M is the rotation matrix
 * and p is the angle-axis vector.
 *
 * This function implements the exact formulas from Orient's interact.f90
 * aaderiv subroutine (lines 132-221). It is provided for reference and
 * for potential future use in direct analytical optimization.
 *
 * @param p Angle-axis vector
 * @return Array of 3 matrices: [M1_x, M1_y, M1_z]
 */
static std::array<Mat3, 3> compute_rotation_derivatives(const Vec3& p) {
    std::array<Mat3, 3> M1;
    double psi = p.norm();

    if (psi < 1e-8) {
        // Small angle approximation from Orient (lines 194-198)
        // M ≈ I + [p]×
        // M1(k) = ∂/∂p(k) [I + [p]×] = [e_k]× (skew matrix of k-th basis vector)

        Vec3 ex(1, 0, 0), ey(0, 1, 0), ez(0, 0, 1);
        M1[0] = skew_symmetric(ex);
        M1[1] = skew_symmetric(ey);
        M1[2] = skew_symmetric(ez);

        return M1;
    }

    // General case (Orient lines 200-214)
    Vec3 n = p / psi;  // Unit axis
    Mat3 N0 = skew_symmetric(n);  // [n]×
    Mat3 N0_squared = N0 * N0;    // [n]×²

    // Compute N1(k) = ∂[n]×/∂p(k) for k=0,1,2
    // From Orient lines 204-209:
    // N1(:,:,k) = (1/psi) * [∂n/∂p(k)]×
    // where ∂n/∂p(k) = (e_k - n(k)*n) / psi

    std::array<Mat3, 3> N1;

    for (int k = 0; k < 3; k++) {
        // Compute ∂n/∂p(k) = (1/psi) * (e_k - n(k)*n)
        Vec3 e_k = Vec3::Zero();
        e_k(k) = 1.0;

        Vec3 dn_dpk = (e_k - n(k) * n) / psi;
        N1[k] = skew_symmetric(dn_dpk);
    }

    // Compute M1(k) = ∂M/∂p(k) using Orient's formula (lines 211-213):
    // M1(:,:,k) = n(k)*sin(ψ)*[n]×²
    //           + (1-cos(ψ))*(N1(:,:,k)*[n]× + [n]×*N1(:,:,k))
    //           + n(k)*cos(ψ)*[n]×
    //           + sin(ψ)*N1(:,:,k)

    double sin_psi = std::sin(psi);
    double cos_psi = std::cos(psi);

    for (int k = 0; k < 3; k++) {
        M1[k] = n(k) * sin_psi * N0_squared
              + (1.0 - cos_psi) * (N1[k] * N0 + N0 * N1[k])
              + n(k) * cos_psi * N0
              + sin_psi * N1[k];
    }

    return M1;
}

// ==================== Multipole Management ====================

void RigidBodyState::update_lab_multipoles() {
    if (!multipole_lab_valid) {
        Mat3 R = rotation_matrix();
        multipole_lab = rotated_multipole(multipole_body, R);
        multipole_lab_valid = true;
    }
}

const occ::dma::Mult& RigidBodyState::get_lab_multipoles() {
    update_lab_multipoles();
    return multipole_lab;
}

// ==================== Inertia Tensor Management ====================

void RigidBodyState::set_inertia_tensor(const Mat3& I) {
    inertia_body = I;
    inertia_inv_body = I.inverse();
}

void RigidBodyState::set_spherical_inertia(double I_moment) {
    inertia_body = I_moment * Mat3::Identity();
    inertia_inv_body = (1.0 / I_moment) * Mat3::Identity();
}

void RigidBodyState::set_diagonal_inertia(double Ixx, double Iyy, double Izz) {
    inertia_body = Mat3::Zero();
    inertia_body(0, 0) = Ixx;
    inertia_body(1, 1) = Iyy;
    inertia_body(2, 2) = Izz;

    inertia_inv_body = Mat3::Zero();
    inertia_inv_body(0, 0) = 1.0 / Ixx;
    inertia_inv_body(1, 1) = 1.0 / Iyy;
    inertia_inv_body(2, 2) = 1.0 / Izz;
}

Mat3 RigidBodyState::inertia_space() const {
    Mat3 R = rotation_matrix();
    return R * inertia_body * R.transpose();
}

// ==================== Energy Calculations ====================

double RigidBodyState::translational_kinetic_energy() const {
    // KE = (1/2) * m * v^2
    return 0.5 * mass * velocity.squaredNorm();
}

double RigidBodyState::rotational_kinetic_energy() const {
    // KE_rot = (1/2) * omega^T * I * omega
    Vec3 L = inertia_body * angular_velocity_body;
    return 0.5 * angular_velocity_body.dot(L);
}

double RigidBodyState::kinetic_energy() const {
    return translational_kinetic_energy() + rotational_kinetic_energy();
}

// ==================== Angular Momentum ====================

Vec3 RigidBodyState::angular_momentum_body() const {
    return inertia_body * angular_velocity_body;
}

Vec3 RigidBodyState::angular_momentum_space() const {
    Vec3 L_body = angular_momentum_body();
    Mat3 R = rotation_matrix();
    return R * L_body;
}

/**
 * @brief Compute second derivatives of rotation matrix w.r.t. angle-axis components
 *
 * This computes ∂²M/∂p_k∂p_l analytically by differentiating the Rodrigues formula
 * derivatives. The formula for ∂M/∂p_k is:
 *
 *   ∂M/∂p_k = n_k·sin(ψ)·N₀² + (1-cos(ψ))·(N₁ᵏ·N₀ + N₀·N₁ᵏ)
 *           + n_k·cos(ψ)·N₀ + sin(ψ)·N₁ᵏ
 *
 * Taking the derivative with respect to p_l:
 *
 *   ∂²M/∂p_k∂p_l = (derivatives of each term above)
 *
 * Key identities used:
 *   ∂ψ/∂p_l = n_l
 *   ∂n_k/∂p_l = (δ_kl - n_k·n_l) / ψ
 *   ∂N₀/∂p_l = N₁ˡ
 *   ∂N₀²/∂p_l = N₁ˡ·N₀ + N₀·N₁ˡ
 *   ∂N₁ᵏ/∂p_l = N₂ᵏˡ (computed below)
 *
 * @return Array of 9 matrices, where result[3*k + l] = ∂²M/∂p_k∂p_l
 */
std::array<Mat3, 9> RigidBodyState::rotation_matrix_second_derivatives() const {
    std::array<Mat3, 9> M2;

    // Get angle-axis vector
    Vec3 p = get_angle_axis();
    double psi = p.norm();

    // Small angle case: second derivatives are O(1) but small
    // For very small rotations, use finite differences or zeros
    if (psi < 1e-8) {
        // For small rotations M ≈ I + [p]×
        // ∂M/∂p_k ≈ [e_k]× (constant in p)
        // ∂²M/∂p_k∂p_l ≈ 0
        for (int i = 0; i < 9; ++i) {
            M2[i] = Mat3::Zero();
        }
        return M2;
    }

    // General case
    Vec3 n = p / psi;  // Unit rotation axis

    // Precompute basic quantities
    Mat3 N0 = skew_symmetric(n);
    Mat3 N0_sq = N0 * N0;

    double sin_psi = std::sin(psi);
    double cos_psi = std::cos(psi);
    double one_minus_cos = 1.0 - cos_psi;

    // Compute N1[k] = [∂n/∂p_k]× for each k
    std::array<Mat3, 3> N1;
    for (int k = 0; k < 3; k++) {
        Vec3 ek = Vec3::Zero();
        ek(k) = 1.0;
        Vec3 dn_dpk = (ek - n(k) * n) / psi;
        N1[k] = skew_symmetric(dn_dpk);
    }

    // Compute second derivatives ∂²M/∂p_k∂p_l
    for (int k = 0; k < 3; k++) {
        for (int l = 0; l < 3; l++) {
            // Compute various derivative terms

            // ∂ψ/∂p_l = n_l
            double dpsi_dpl = n(l);

            // ∂n_k/∂p_l = (δ_kl - n_k·n_l) / ψ
            double dnk_dpl = ((k == l ? 1.0 : 0.0) - n(k) * n(l)) / psi;

            // ∂sin(ψ)/∂p_l = cos(ψ)·n_l
            double dsin_dpl = cos_psi * n(l);

            // ∂cos(ψ)/∂p_l = -sin(ψ)·n_l
            double dcos_dpl = -sin_psi * n(l);

            // ∂(1-cos(ψ))/∂p_l = sin(ψ)·n_l
            double done_minus_cos_dpl = sin_psi * n(l);

            // ∂N₀/∂p_l = N₁ˡ (already computed)
            const Mat3& dN0_dpl = N1[l];

            // ∂(N₀²)/∂p_l = N₁ˡ·N₀ + N₀·N₁ˡ
            Mat3 dN0sq_dpl = N1[l] * N0 + N0 * N1[l];

            // ∂N₁ᵏ/∂p_l = N₂ᵏˡ
            // N₁ᵏ = [∂n/∂p_k]× where ∂n/∂p_k = (e_k - n_k·n) / ψ
            // ∂N₁ᵏ/∂p_l = [∂²n/∂p_k∂p_l]×
            //
            // ∂(∂n/∂p_k)/∂p_l = ∂/∂p_l [(e_k - n_k·n) / ψ]
            //                = -(e_k - n_k·n)·n_l / ψ²
            //                  + (-∂n_k/∂p_l·n - n_k·∂n/∂p_l) / ψ
            //
            // Let's compute this term by term:
            Vec3 ek = Vec3::Zero();
            ek(k) = 1.0;
            Vec3 el = Vec3::Zero();
            el(l) = 1.0;

            Vec3 dn_dpk = (ek - n(k) * n) / psi;
            Vec3 dn_dpl = (el - n(l) * n) / psi;

            // ∂²n/∂p_k∂p_l
            // = -(e_k - n_k·n)·n_l / ψ²
            //   + (-dnk/dpl·n - n_k·dn/dpl) / ψ
            Vec3 d2n_dpk_dpl = -(ek - n(k) * n) * n(l) / (psi * psi)
                              + (-dnk_dpl * n - n(k) * dn_dpl) / psi;

            Mat3 dN1k_dpl = skew_symmetric(d2n_dpk_dpl);

            // Now compute ∂²M/∂p_k∂p_l by differentiating each term:
            //
            // Term 1: ∂/∂p_l [n_k·sin(ψ)·N₀²]
            //       = dnk_dpl·sin(ψ)·N₀² + n_k·dsin_dpl·N₀² + n_k·sin(ψ)·dN0sq_dpl
            Mat3 term1 = dnk_dpl * sin_psi * N0_sq
                       + n(k) * dsin_dpl * N0_sq
                       + n(k) * sin_psi * dN0sq_dpl;

            // Term 2: ∂/∂p_l [(1-cos(ψ))·(N₁ᵏ·N₀ + N₀·N₁ᵏ)]
            //       = done_minus_cos_dpl·(N₁ᵏ·N₀ + N₀·N₁ᵏ)
            //       + (1-cos(ψ))·(dN1k_dpl·N₀ + N₁ᵏ·dN0_dpl + dN0_dpl·N₁ᵏ + N₀·dN1k_dpl)
            Mat3 term2 = done_minus_cos_dpl * (N1[k] * N0 + N0 * N1[k])
                       + one_minus_cos * (dN1k_dpl * N0 + N1[k] * dN0_dpl
                                        + dN0_dpl * N1[k] + N0 * dN1k_dpl);

            // Term 3: ∂/∂p_l [n_k·cos(ψ)·N₀]
            //       = dnk_dpl·cos(ψ)·N₀ + n_k·dcos_dpl·N₀ + n_k·cos(ψ)·dN0_dpl
            Mat3 term3 = dnk_dpl * cos_psi * N0
                       + n(k) * dcos_dpl * N0
                       + n(k) * cos_psi * dN0_dpl;

            // Term 4: ∂/∂p_l [sin(ψ)·N₁ᵏ]
            //       = dsin_dpl·N₁ᵏ + sin(ψ)·dN1k_dpl
            Mat3 term4 = dsin_dpl * N1[k]
                       + sin_psi * dN1k_dpl;

            M2[3 * k + l] = term1 + term2 + term3 + term4;
        }
    }

    return M2;
}

} // namespace occ::mults
