#include <occ/mults/cartesian_hessian.h>
#include <occ/mults/interaction_tensor.h>
#include <occ/mults/cartesian_rotation.h>
#include <occ/mults/cartesian_kernels.h>
#include <occ/ints/rints.h>
#include <cmath>

namespace occ::mults {

using occ::ints::hermite_index;

// ============================================================================
// Helper functions
// ============================================================================

Mat PairHessianResult::pack_full_hessian() const {
    Mat H = Mat::Zero(12, 12);

    // Order: [pos_A(3), rot_A(3), pos_B(3), rot_B(3)]

    // Position-Position blocks
    H.block<3, 3>(0, 0) = H_posA_posA;
    H.block<3, 3>(0, 6) = H_posA_posB;
    H.block<3, 3>(6, 0) = H_posA_posB.transpose();
    H.block<3, 3>(6, 6) = H_posB_posB;

    // Position-Rotation cross blocks
    H.block<3, 3>(0, 3) = H_posA_rotA;
    H.block<3, 3>(3, 0) = H_posA_rotA.transpose();
    H.block<3, 3>(0, 9) = H_posA_rotB;
    H.block<3, 3>(9, 0) = H_posA_rotB.transpose();
    H.block<3, 3>(6, 3) = H_posB_rotA;
    H.block<3, 3>(3, 6) = H_posB_rotA.transpose();
    H.block<3, 3>(6, 9) = H_posB_rotB;
    H.block<3, 3>(9, 6) = H_posB_rotB.transpose();

    // Rotation-Rotation blocks
    H.block<3, 3>(3, 3) = H_rotA_rotA;
    H.block<3, 3>(3, 9) = H_rotA_rotB;
    H.block<3, 3>(9, 3) = H_rotA_rotB.transpose();
    H.block<3, 3>(9, 9) = H_rotB_rotB;

    return H;
}

Mat6 PairHessianResult::pack_position_hessian() const {
    Mat6 H;
    H.block<3, 3>(0, 0) = H_posA_posA;
    H.block<3, 3>(0, 3) = H_posA_posB;
    H.block<3, 3>(3, 0) = H_posA_posB.transpose();
    H.block<3, 3>(3, 3) = H_posB_posB;
    return H;
}

// ============================================================================
// Charge-charge Hessian (rank 0 - rank 0)
// ============================================================================

PairHessianResult compute_charge_charge_hessian(
    const Vec3 &posA, double qA,
    const Vec3 &posB, double qB) {

    PairHessianResult result;

    Vec3 R = posB - posA;
    double Rx = R[0], Ry = R[1], Rz = R[2];

    // Compute T-tensor up to order 2 for Hessian
    InteractionTensor<2> T;
    compute_interaction_tensor<2>(Rx, Ry, Rz, T);

    double qAqB = qA * qB;

    // Energy: E = qA * qB * T(0,0,0)
    result.energy = qAqB * T(0, 0, 0);

    // Force on A (gradient w.r.t. posA is negative of gradient w.r.t. R):
    // ∂E/∂posA_k = -∂E/∂R_k = -qA * qB * T(1,0,0) etc.
    // Force = -gradient, so F_A = qA * qB * T^(1)
    result.force_A = Vec3(qAqB * T(1, 0, 0),
                          qAqB * T(0, 1, 0),
                          qAqB * T(0, 0, 1));
    result.force_B = -result.force_A;  // Newton's 3rd law

    // Position-Position Hessian:
    // ∂²E/∂posA_k ∂posA_l = ∂²E/∂R_k ∂R_l = qA * qB * T^(2)_kl
    // Note: ∂/∂posA = -∂/∂R, so ∂²/∂posA² = ∂²/∂R²
    result.H_posA_posA(0, 0) = qAqB * T(2, 0, 0);
    result.H_posA_posA(0, 1) = qAqB * T(1, 1, 0);
    result.H_posA_posA(0, 2) = qAqB * T(1, 0, 1);
    result.H_posA_posA(1, 0) = qAqB * T(1, 1, 0);
    result.H_posA_posA(1, 1) = qAqB * T(0, 2, 0);
    result.H_posA_posA(1, 2) = qAqB * T(0, 1, 1);
    result.H_posA_posA(2, 0) = qAqB * T(1, 0, 1);
    result.H_posA_posA(2, 1) = qAqB * T(0, 1, 1);
    result.H_posA_posA(2, 2) = qAqB * T(0, 0, 2);

    // H_posA_posB = ∂²E/∂posA_k ∂posB_l = -∂²E/∂R_k ∂R_l = -H_posA_posA
    result.H_posA_posB = -result.H_posA_posA;

    // H_posB_posB = ∂²E/∂posB_k ∂posB_l = ∂²E/∂R_k ∂R_l = H_posA_posA
    result.H_posB_posB = result.H_posA_posA;

    // No rotation Hessian for point charges (rank 0)
    // All rotation blocks remain zero

    return result;
}

// ============================================================================
// Charge-dipole Hessian (rank 0 - rank 1)
// ============================================================================

PairHessianResult compute_charge_dipole_hessian(
    const Vec3 &posA, double qA,
    const Vec3 &posB, const Vec3 &dipole_B,
    const Vec3 &body_dipole_B,
    const Mat3 &M,
    const std::array<Mat3, 3> &dM,
    const std::array<Mat3, 9> *d2M) {

    PairHessianResult result;

    Vec3 R = posB - posA;
    double Rx = R[0], Ry = R[1], Rz = R[2];

    // Compute T-tensor up to order 3 for Hessian (need T^(1), T^(2), T^(3))
    InteractionTensor<3> T;
    compute_interaction_tensor<3>(Rx, Ry, Rz, T);

    // Energy: E = qA * T^(1)_j * μ_B^j
    // where T^(1)_j = [T(1,0,0), T(0,1,0), T(0,0,1)]
    double T1x = T(1, 0, 0);
    double T1y = T(0, 1, 0);
    double T1z = T(0, 0, 1);

    result.energy = qA * (T1x * dipole_B[0] + T1y * dipole_B[1] + T1z * dipole_B[2]);

    // Force on A (gradient w.r.t. posA):
    // ∂E/∂posA_k = -∂E/∂R_k = -qA * T^(2)_jk * μ_B^j
    // Force = -gradient
    for (int k = 0; k < 3; ++k) {
        int idx_xk = hermite_index(1 + (k == 0 ? 1 : 0),
                                   0 + (k == 1 ? 1 : 0),
                                   0 + (k == 2 ? 1 : 0));
        int idx_yk = hermite_index(0 + (k == 0 ? 1 : 0),
                                   1 + (k == 1 ? 1 : 0),
                                   0 + (k == 2 ? 1 : 0));
        int idx_zk = hermite_index(0 + (k == 0 ? 1 : 0),
                                   0 + (k == 1 ? 1 : 0),
                                   1 + (k == 2 ? 1 : 0));

        double T2_xk = T.data[idx_xk];  // T(1+dk0, dk1, dk2)
        double T2_yk = T.data[idx_yk];  // T(dk0, 1+dk1, dk2)
        double T2_zk = T.data[idx_zk];  // T(dk0, dk1, 1+dk2)

        // ∂E/∂R_k = qA * (T2_xk * μx + T2_yk * μy + T2_zk * μz)
        double grad_k = qA * (T2_xk * dipole_B[0] + T2_yk * dipole_B[1] + T2_zk * dipole_B[2]);

        result.force_A[k] = grad_k;  // F = +dE/dR (since dE/dposA = -dE/dR)
    }
    result.force_B = -result.force_A;

    // Position-Position Hessian:
    // ∂²E/∂R_k ∂R_l = qA * T^(3)_jkl * μ_B^j
    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            // T^(3)_jkl: j=x,y,z, k=direction, l=direction
            // For j=x: T(1+dk+dl, 0, 0) etc.
            int t = 1 + (k == 0 ? 1 : 0) + (l == 0 ? 1 : 0);
            int u = 0 + (k == 1 ? 1 : 0) + (l == 1 ? 1 : 0);
            int v = 0 + (k == 2 ? 1 : 0) + (l == 2 ? 1 : 0);

            double T3_xkl = T(t, u, v);

            t = 0 + (k == 0 ? 1 : 0) + (l == 0 ? 1 : 0);
            u = 1 + (k == 1 ? 1 : 0) + (l == 1 ? 1 : 0);
            v = 0 + (k == 2 ? 1 : 0) + (l == 2 ? 1 : 0);

            double T3_ykl = T(t, u, v);

            t = 0 + (k == 0 ? 1 : 0) + (l == 0 ? 1 : 0);
            u = 0 + (k == 1 ? 1 : 0) + (l == 1 ? 1 : 0);
            v = 1 + (k == 2 ? 1 : 0) + (l == 2 ? 1 : 0);

            double T3_zkl = T(t, u, v);

            result.H_posA_posA(k, l) = qA * (T3_xkl * dipole_B[0] +
                                             T3_ykl * dipole_B[1] +
                                             T3_zkl * dipole_B[2]);
        }
    }
    result.H_posA_posB = -result.H_posA_posA;
    result.H_posB_posB = result.H_posA_posA;

    // Position-Rotation Hessian for B's rotation:
    // ∂²E/∂R_k ∂θ_B^m = qA * T^(2)_jk * ∂μ_B^j/∂θ_B^m
    //
    // μ_B = M * μ_body (dipole rotation)
    // ∂μ_B/∂θ_m = dM_m * μ_body

    for (int m = 0; m < 3; ++m) {
        Vec3 dmu_dm = dM[m] * body_dipole_B;

        for (int k = 0; k < 3; ++k) {
            // T^(2)_jk for j=x,y,z
            double T2_xk = T(1 + (k == 0 ? 1 : 0), (k == 1 ? 1 : 0), (k == 2 ? 1 : 0));
            double T2_yk = T((k == 0 ? 1 : 0), 1 + (k == 1 ? 1 : 0), (k == 2 ? 1 : 0));
            double T2_zk = T((k == 0 ? 1 : 0), (k == 1 ? 1 : 0), 1 + (k == 2 ? 1 : 0));

            // ∂²E/∂R_k ∂θ_m = qA * (T2_xk * dmu_x + T2_yk * dmu_y + T2_zk * dmu_z)
            result.H_posA_rotB(k, m) = -qA * (T2_xk * dmu_dm[0] +
                                              T2_yk * dmu_dm[1] +
                                              T2_zk * dmu_dm[2]);
            result.H_posB_rotB(k, m) = -result.H_posA_rotB(k, m);
        }
    }

    // Gradient w.r.t. B's rotation (angle-axis):
    // ∂E/∂θ_B^m = qA * T^(1)_j * ∂μ_B^j/∂θ_B^m
    for (int m = 0; m < 3; ++m) {
        Vec3 dmu_dm = dM[m] * body_dipole_B;
        result.grad_angle_axis_B[m] = qA * (T1x * dmu_dm[0] + T1y * dmu_dm[1] + T1z * dmu_dm[2]);
    }

    // Rotation-Rotation Hessian for B's rotation:
    // ∂²E/∂θ_B^m ∂θ_B^n = qA * T^(1)_j * ∂²μ_B^j/∂θ_B^m ∂θ_B^n
    //
    // ∂²μ_B/∂θ_m ∂θ_n = d2M[3*m + n] * μ_body
    if (d2M != nullptr) {
        for (int m = 0; m < 3; ++m) {
            for (int n = 0; n < 3; ++n) {
                Vec3 d2mu_dmn = (*d2M)[3 * m + n] * body_dipole_B;
                result.H_rotB_rotB(m, n) = qA * (T1x * d2mu_dmn[0] +
                                                 T1y * d2mu_dmn[1] +
                                                 T1z * d2mu_dmn[2]);
            }
        }
    }

    // No rotation Hessian for A (charge at A doesn't rotate)
    // H_posA_rotA, H_rotA_rotA, H_rotA_rotB all remain zero

    return result;
}

// ============================================================================
// Molecule-level truncated Hessian (charge-charge + charge-dipole only)
// ============================================================================

PairHessianResult compute_molecule_hessian_truncated(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB) {

    PairHessianResult result;

    // Get rotation data for both molecules
    const bool hasBodyA = molA.body_data.has_value();
    const bool hasBodyB = molB.body_data.has_value();

    std::array<Mat3, 3> dM_A, dM_B;
    std::array<Mat3, 9> d2M_A, d2M_B;
    Mat3 M_A = Mat3::Identity(), M_B = Mat3::Identity();

    if (hasBodyA) {
        M_A = molA.body_data->rotation;
        // Need to compute rotation matrix derivatives from body_data
        // For now, we don't have them stored, so we'll skip rotation-A terms
    }

    if (hasBodyB) {
        M_B = molB.body_data->rotation;
        // Similar for B
    }

    // Loop over site pairs
    for (size_t i = 0; i < molA.sites.size(); ++i) {
        const auto &sA = molA.sites[i];
        if (sA.rank < 0) continue;

        for (size_t j = 0; j < molB.sites.size(); ++j) {
            const auto &sB = molB.sites[j];
            if (sB.rank < 0) continue;

            Vec3 R = sB.position - sA.position;

            // Charge-charge (rank 0 - rank 0)
            if (sA.rank == 0 && sB.rank == 0) {
                double qA = sA.cart.data[0];  // charge
                double qB = sB.cart.data[0];
                auto pair_result = compute_charge_charge_hessian(sA.position, qA, sB.position, qB);

                result.energy += pair_result.energy;
                result.force_A += pair_result.force_A;
                result.force_B += pair_result.force_B;
                result.H_posA_posA += pair_result.H_posA_posA;
                result.H_posA_posB += pair_result.H_posA_posB;
                result.H_posB_posB += pair_result.H_posB_posB;
            }
            // Charge-dipole (rank 0 - rank 1): A is charge, B has dipole
            else if (sA.rank == 0 && sB.rank >= 1) {
                double qA = sA.cart.data[0];
                Vec3 dipole_B(sB.cart.data[hermite_index(1, 0, 0)],
                              sB.cart.data[hermite_index(0, 1, 0)],
                              sB.cart.data[hermite_index(0, 0, 1)]);

                // Get body-frame dipole for rotation derivatives
                Vec3 body_dipole_B = Vec3::Zero();
                if (hasBodyB && j < molB.body_data->body_multipoles.size()) {
                    const auto &body_cart = molB.body_data->body_multipoles[j];
                    body_dipole_B = Vec3(body_cart.data[hermite_index(1, 0, 0)],
                                         body_cart.data[hermite_index(0, 1, 0)],
                                         body_cart.data[hermite_index(0, 0, 1)]);
                }

                // For now, use simple position-only Hessian (skip rotation terms)
                auto pair_result = compute_charge_dipole_hessian(
                    sA.position, qA, sB.position, dipole_B,
                    body_dipole_B, M_B, dM_B, nullptr);

                result.energy += pair_result.energy;
                result.force_A += pair_result.force_A;
                result.force_B += pair_result.force_B;
                result.H_posA_posA += pair_result.H_posA_posA;
                result.H_posA_posB += pair_result.H_posA_posB;
                result.H_posB_posB += pair_result.H_posB_posB;
                // Skip rotation terms for now
            }
            // Dipole-charge (rank 1 - rank 0): A has dipole, B is charge
            else if (sA.rank >= 1 && sB.rank == 0) {
                double qB = sB.cart.data[0];
                Vec3 dipole_A(sA.cart.data[hermite_index(1, 0, 0)],
                              sA.cart.data[hermite_index(0, 1, 0)],
                              sA.cart.data[hermite_index(0, 0, 1)]);

                // Get body-frame dipole for rotation derivatives
                Vec3 body_dipole_A = Vec3::Zero();
                if (hasBodyA && i < molA.body_data->body_multipoles.size()) {
                    const auto &body_cart = molA.body_data->body_multipoles[i];
                    body_dipole_A = Vec3(body_cart.data[hermite_index(1, 0, 0)],
                                         body_cart.data[hermite_index(0, 1, 0)],
                                         body_cart.data[hermite_index(0, 0, 1)]);
                }

                // Swap A and B roles, negate R
                auto pair_result = compute_charge_dipole_hessian(
                    sB.position, qB, sA.position, dipole_A,
                    body_dipole_A, M_A, dM_A, nullptr);

                // Results need to be swapped back
                result.energy += pair_result.energy;
                result.force_A += pair_result.force_B;  // Swapped
                result.force_B += pair_result.force_A;
                result.H_posA_posA += pair_result.H_posB_posB;  // Swapped
                result.H_posA_posB += pair_result.H_posA_posB.transpose();
                result.H_posB_posB += pair_result.H_posA_posA;
                // Skip rotation terms for now
            }
            // Higher rank interactions: skip Hessian contribution
            // (they still contribute to energy/gradient via other code paths)
        }
    }

    return result;
}

} // namespace occ::mults
