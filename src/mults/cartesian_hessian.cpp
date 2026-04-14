#include <occ/mults/cartesian_hessian.h>
#include <occ/mults/interaction_tensor.h>
#include <occ/mults/cartesian_rotation.h>
#include <occ/mults/cartesian_kernels.h>
#include <occ/core/units.h>
#include <occ/ints/rints.h>
#include <cmath>

namespace occ::mults {

using occ::ints::hermite_index;

namespace {

constexpr double kEnergyConv =
    occ::units::AU_TO_KJ_PER_MOL;
constexpr double kForceConv =
    occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM;
constexpr double kHessianConv =
    occ::units::AU_TO_KJ_PER_MOL /
    (occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM);

std::array<Mat3, 3> so3_generators() {
    std::array<Mat3, 3> G;
    G[0] << 0, 0, 0,  0, 0, -1,  0, 1, 0;
    G[1] << 0, 0, 1,  0, 0, 0,  -1, 0, 0;
    G[2] << 0, -1, 0,  1, 0, 0,  0, 0, 0;
    return G;
}

std::array<Mat3, 3> rotation_matrix_first_derivatives(const Mat3& M) {
    std::array<Mat3, 3> dM;
    const auto G = so3_generators();
    for (int k = 0; k < 3; ++k) {
        dM[k] = G[k] * M;
    }
    return dM;
}

std::array<Mat3, 9> rotation_matrix_second_derivatives(const Mat3& M) {
    std::array<Mat3, 9> d2M;
    const auto G = so3_generators();
    for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
            d2M[3 * k + l] = 0.5 * (G[k] * G[l] + G[l] * G[k]) * M;
        }
    }
    return d2M;
}

} // namespace

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
    (void)M;

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
        result.grad_angle_axis_B[m] = qA * (T1x * dmu_dm[0] +
                                            T1y * dmu_dm[1] +
                                            T1z * dmu_dm[2]);
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
// T-tensor dispatch for Hessian (order = max_interaction_order + 2)
// ============================================================================

/// Compute interaction tensor and contract with multipole weights/derivatives.
/// Returns accumulated E, g, H and field arrays for the rotation Hessian.
struct SitePairContractionResult {
    static constexpr int kMaxComp = occ::ints::nhermsum(4);
    double E = 0.0;
    Vec3 g = Vec3::Zero();
    Mat3 H = Mat3::Zero();
    std::array<double, kMaxComp> fieldA0{};
    std::array<double, kMaxComp> fieldB0{};
    std::array<std::array<double, kMaxComp>, 3> fieldA_R{};
    std::array<std::array<double, kMaxComp>, 3> fieldB_R{};
    std::array<std::array<double, kMaxComp>, 3> fieldA_from_dB{};
    std::array<std::array<double, kMaxComp>, 3> fieldB_from_dA{};
};

template<int TOrder>
static SitePairContractionResult compute_site_pair_contraction(
    const double* wA, const std::array<std::array<double, occ::ints::nhermsum(4)>, 3>& dwA,
    const double* wB, const std::array<std::array<double, occ::ints::nhermsum(4)>, 3>& dwB,
    int nA, int nB,
    const Vec3& R_bohr,
    bool use_order_cutoff, int max_interaction_order) {

    SitePairContractionResult r;

    InteractionTensor<TOrder> T;
    compute_interaction_tensor<TOrder>(R_bohr[0], R_bohr[1], R_bohr[2], T);

    for (int ia = 0; ia < nA; ++ia) {
        const auto [ta, ua, va] = kernel_detail::tuv4.entries[ia];
        const int la = ta + ua + va;
        const double wAi = wA[ia];
        if (wAi == 0.0 &&
            dwA[0][ia] == 0.0 && dwA[1][ia] == 0.0 &&
            dwA[2][ia] == 0.0) {
            continue;
        }

        for (int jb = 0; jb < nB; ++jb) {
            const auto [tb, ub, vb] = kernel_detail::tuv4.entries[jb];
            const int lb = tb + ub + vb;
            if (use_order_cutoff &&
                (la + lb) > max_interaction_order) {
                continue;
            }
            const int t = ta + tb;
            const int u = ua + ub;
            const int v = va + vb;

            const double T0 = T(t, u, v);
            const double Tx = T(t + 1, u, v);
            const double Ty = T(t, u + 1, v);
            const double Tz = T(t, u, v + 1);
            const double Txx = T(t + 2, u, v);
            const double Txy = T(t + 1, u + 1, v);
            const double Txz = T(t + 1, u, v + 1);
            const double Tyy = T(t, u + 2, v);
            const double Tyz = T(t, u + 1, v + 1);
            const double Tzz = T(t, u, v + 2);

            const double wBj = wB[jb];
            const double wp = wAi * wBj;

            r.E += wp * T0;
            r.g[0] += wp * Tx;
            r.g[1] += wp * Ty;
            r.g[2] += wp * Tz;
            r.H(0, 0) += wp * Txx;
            r.H(0, 1) += wp * Txy;
            r.H(0, 2) += wp * Txz;
            r.H(1, 0) += wp * Txy;
            r.H(1, 1) += wp * Tyy;
            r.H(1, 2) += wp * Tyz;
            r.H(2, 0) += wp * Txz;
            r.H(2, 1) += wp * Tyz;
            r.H(2, 2) += wp * Tzz;

            r.fieldA0[ia] += wBj * T0;
            r.fieldA_R[0][ia] += wBj * Tx;
            r.fieldA_R[1][ia] += wBj * Ty;
            r.fieldA_R[2][ia] += wBj * Tz;
            for (int n = 0; n < 3; ++n) {
                r.fieldA_from_dB[n][ia] += dwB[n][jb] * T0;
            }

            r.fieldB0[jb] += wAi * T0;
            r.fieldB_R[0][jb] += wAi * Tx;
            r.fieldB_R[1][jb] += wAi * Ty;
            r.fieldB_R[2][jb] += wAi * Tz;
            for (int m = 0; m < 3; ++m) {
                r.fieldB_from_dA[m][jb] += dwA[m][ia] * T0;
            }
        }
    }
    return r;
}

/// Runtime dispatch for site pair contraction based on max interaction order.
static SitePairContractionResult dispatch_site_pair_contraction(
    const double* wA, const std::array<std::array<double, occ::ints::nhermsum(4)>, 3>& dwA,
    const double* wB, const std::array<std::array<double, occ::ints::nhermsum(4)>, 3>& dwB,
    int nA, int nB,
    const Vec3& R_bohr,
    bool use_order_cutoff, int max_interaction_order) {

    if (!use_order_cutoff) {
        return compute_site_pair_contraction<10>(
            wA, dwA, wB, dwB, nA, nB, R_bohr, false, -1);
    }
    // T-tensor order = max_interaction_order + 2 (for Hessian)
    switch (max_interaction_order) {
    case 0: return compute_site_pair_contraction<2>(
        wA, dwA, wB, dwB, nA, nB, R_bohr, true, 0);
    case 1: return compute_site_pair_contraction<3>(
        wA, dwA, wB, dwB, nA, nB, R_bohr, true, 1);
    case 2: return compute_site_pair_contraction<4>(
        wA, dwA, wB, dwB, nA, nB, R_bohr, true, 2);
    case 3: return compute_site_pair_contraction<5>(
        wA, dwA, wB, dwB, nA, nB, R_bohr, true, 3);
    case 4: return compute_site_pair_contraction<6>(
        wA, dwA, wB, dwB, nA, nB, R_bohr, true, 4);
    case 5: return compute_site_pair_contraction<7>(
        wA, dwA, wB, dwB, nA, nB, R_bohr, true, 5);
    case 6: return compute_site_pair_contraction<8>(
        wA, dwA, wB, dwB, nA, nB, R_bohr, true, 6);
    case 7: return compute_site_pair_contraction<9>(
        wA, dwA, wB, dwB, nA, nB, R_bohr, true, 7);
    default: return compute_site_pair_contraction<10>(
        wA, dwA, wB, dwB, nA, nB, R_bohr, true, max_interaction_order);
    }
}

// ============================================================================
// Molecule-level rigid-body Hessian (all Cartesian multipole ranks)
// ============================================================================

// ============================================================================
// Public: build per-molecule Hessian data (rotation derivatives)
// ============================================================================

MoleculeHessianData build_molecule_hessian_data(
    const CartesianMolecule& mol, bool signed_side) {

    const double to_bohr = 1.0 / occ::units::BOHR_TO_ANGSTROM;
    MoleculeHessianData result;
    result.sites.resize(mol.sites.size());

    const bool has_body = mol.body_data.has_value();
    Mat3 M = Mat3::Identity();
    std::array<Mat3, 3> dM;
    std::array<Mat3, 9> d2M;
    for (int k = 0; k < 3; ++k) dM[k].setZero();
    for (int k = 0; k < 9; ++k) d2M[k].setZero();

    if (has_body) {
        M = mol.body_data->rotation;
        dM = rotation_matrix_first_derivatives(M);
        d2M = rotation_matrix_second_derivatives(M);
    }

    for (size_t i = 0; i < mol.sites.size(); ++i) {
        auto& out = result.sites[i];
        for (int m = 0; m < 3; ++m) {
            out.dlever[m] = Vec3::Zero();
            for (int n = 0; n < 3; ++n) {
                out.d2lever[m][n] = Vec3::Zero();
            }
        }

        const auto& site = mol.sites[i];
        out.rank = site.rank;
        if (site.rank < 0) continue;

        const int ncomp = occ::ints::nhermsum(site.rank);
        for (int c = 0; c < ncomp; ++c) {
            const double w = signed_side
                ? kernel_detail::weights4.sign_inv_fact[c]
                : kernel_detail::weights4.inv_fact[c];
            out.w[c] = w * site.cart.data[c];
        }

        if (!has_body || i >= mol.body_data->body_offsets.size()) {
            continue;
        }

        const Vec3& body_offset = mol.body_data->body_offsets[i];
        out.lever = (M * body_offset) * to_bohr;
        for (int m = 0; m < 3; ++m) {
            out.dlever[m] = (dM[m] * body_offset) * to_bohr;
            for (int n = 0; n < 3; ++n) {
                out.d2lever[m][n] = (d2M[3 * m + n] * body_offset) * to_bohr;
            }
        }

        if (site.rank <= 0 || i >= mol.body_data->body_multipoles.size()) {
            continue;
        }

        const auto& body_cart = mol.body_data->body_multipoles[i];
        for (int m = 0; m < 3; ++m) {
            CartesianMultipole<4> d_cart;
            rotate_cartesian_multipole_derivative<4>(
                body_cart, M, dM[m], d_cart);
            for (int c = 0; c < ncomp; ++c) {
                const double w = signed_side
                    ? kernel_detail::weights4.sign_inv_fact[c]
                    : kernel_detail::weights4.inv_fact[c];
                out.dw[m][c] = w * d_cart.data[c];
            }

            for (int n = 0; n < 3; ++n) {
                CartesianMultipole<4> d2_cart;
                rotate_cartesian_multipole_second_derivative<4>(
                    body_cart, M, dM[m], dM[n], d2M[3 * m + n], d2_cart);
                for (int c = 0; c < ncomp; ++c) {
                    const double w = signed_side
                        ? kernel_detail::weights4.sign_inv_fact[c]
                        : kernel_detail::weights4.inv_fact[c];
                    out.d2w[m][n][c] = w * d2_cart.data[c];
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Internal implementation using precomputed site derivatives
// ============================================================================

static PairHessianResult compute_molecule_hessian_impl(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB,
    const std::vector<SiteHessianDerivatives>& site_data_A,
    const std::vector<SiteHessianDerivatives>& site_data_B,
    const Vec3& offset_B,
    double site_cutoff,
    int max_interaction_order,
    const CutoffSpline* taper,
    bool taper_hessian) {

    PairHessianResult result;
    const double to_bohr = 1.0 / occ::units::BOHR_TO_ANGSTROM;
    constexpr int kMaxComp = occ::ints::nhermsum(4);

    using SiteDerivatives = SiteHessianDerivatives;

    auto dot_n = [](const std::array<double, kMaxComp>& a,
                    const std::array<double, kMaxComp>& b,
                    int n) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    };

    auto scale_to_kjmol = [](PairHessianResult &r) {
        r.energy *= kEnergyConv;
        r.force_A *= kForceConv;
        r.force_B *= kForceConv;
        r.grad_angle_axis_A *= kEnergyConv;
        r.grad_angle_axis_B *= kEnergyConv;

        r.H_posA_posA *= kHessianConv;
        r.H_posA_posB *= kHessianConv;
        r.H_posB_posB *= kHessianConv;

        r.H_posA_rotA *= kForceConv;
        r.H_posA_rotB *= kForceConv;
        r.H_posB_rotA *= kForceConv;
        r.H_posB_rotB *= kForceConv;

        r.H_rotA_rotA *= kEnergyConv;
        r.H_rotA_rotB *= kEnergyConv;
        r.H_rotB_rotB *= kEnergyConv;
    };

    const bool use_site_cutoff = (site_cutoff > 0.0);
    const bool use_order_cutoff = (max_interaction_order >= 0);
    const bool use_taper = (taper != nullptr && taper->is_valid());

    auto apply_radial_taper = [&](
        PairHessianResult& r, const Vec3& R_ang,
        const std::array<Vec3, 3>& dR_dthetaA_ang,
        const std::array<Vec3, 3>& dR_dthetaB_ang,
        const std::array<std::array<Vec3, 3>, 3>& d2R_AA_ang,
        const std::array<std::array<Vec3, 3>, 3>& d2R_BB_ang) {
        if (!use_taper) return;

        const double dist = R_ang.norm();
        CutoffSplineValue sw = evaluate_cutoff_spline(dist, *taper);
        if (sw.value <= 0.0) {
            r = {};
            return;
        }

        const double E0 = r.energy;
        Vec g0(12);
        g0.segment<3>(0) = -r.force_A;
        g0.segment<3>(3) = r.grad_angle_axis_A;
        g0.segment<3>(6) = -r.force_B;
        g0.segment<3>(9) = r.grad_angle_axis_B;

        // q = [xA(3), thetaA(3), xB(3), thetaB(3)]
        // Generic chain rule:
        //   g = f g0 + E0 * f_q
        //   H = f H0 + f_q g0^T + g0 f_q^T + E0 * f_qq
        Vec rq = Vec::Zero(12);

        if (dist > 1e-12) {
            const Vec3 u = R_ang / dist;
            std::array<Vec3, 12> dR;
            for (int a = 0; a < 12; ++a) dR[a] = Vec3::Zero();
            dR[0] = -Vec3::UnitX();
            dR[1] = -Vec3::UnitY();
            dR[2] = -Vec3::UnitZ();
            dR[6] = Vec3::UnitX();
            dR[7] = Vec3::UnitY();
            dR[8] = Vec3::UnitZ();
            for (int m = 0; m < 3; ++m) {
                dR[3 + m] = dR_dthetaA_ang[m];
                dR[9 + m] = dR_dthetaB_ang[m];
            }

            for (int p = 0; p < 12; ++p) {
                rq[p] = u.dot(dR[p]);
            }

            {
                Mat H0 = r.pack_full_hessian();
                const double f = sw.value;

                Mat H;
                if (taper_hessian) {
                    // Full chain rule: H = f*H0 + fq*g0^T + g0*fq^T + E0*fqq
                    // Physically correct second derivative of f(R)*E(q).
                    Mat rqq = Mat::Zero(12, 12);
                    for (int p = 0; p < 12; ++p) {
                        for (int q = 0; q < 12; ++q) {
                            Vec3 d2R = Vec3::Zero();
                            if (p >= 3 && p < 6 && q >= 3 && q < 6) {
                                d2R = d2R_AA_ang[p - 3][q - 3];
                            } else if (p >= 9 && p < 12 && q >= 9 && q < 12) {
                                d2R = d2R_BB_ang[p - 9][q - 9];
                            }
                            rqq(p, q) = u.dot(d2R) +
                                        (dR[p].dot(dR[q]) - rq[p] * rq[q]) / dist;
                        }
                    }

                    Vec fq_hess = sw.first_derivative * rq;
                    Mat fqq_hess = sw.second_derivative * (rq * rq.transpose()) +
                                   sw.first_derivative * rqq;

                    H = f * H0 + fq_hess * g0.transpose() +
                        g0 * fq_hess.transpose() + E0 * fqq_hess;
                } else {
                    // Same full chain rule — taper_hessian flag is unused.
                    // DMACRYS uses an approximate treatment (untapered SEC),
                    // but our full chain rule is mathematically correct and
                    // validated against finite differences.
                    Mat rqq = Mat::Zero(12, 12);
                    for (int p = 0; p < 12; ++p) {
                        for (int q = 0; q < 12; ++q) {
                            Vec3 d2R = Vec3::Zero();
                            if (p >= 3 && p < 6 && q >= 3 && q < 6) {
                                d2R = d2R_AA_ang[p - 3][q - 3];
                            } else if (p >= 9 && p < 12 && q >= 9 && q < 12) {
                                d2R = d2R_BB_ang[p - 9][q - 9];
                            }
                            rqq(p, q) = u.dot(d2R) +
                                        (dR[p].dot(dR[q]) - rq[p] * rq[q]) / dist;
                        }
                    }

                    Vec fq_hess = sw.first_derivative * rq;
                    Mat fqq_hess = sw.second_derivative * (rq * rq.transpose()) +
                                   sw.first_derivative * rqq;

                    H = f * H0 + fq_hess * g0.transpose() +
                        g0 * fq_hess.transpose() + E0 * fqq_hess;
                }

                r.H_posA_posA = H.block<3, 3>(0, 0);
                r.H_posA_posB = H.block<3, 3>(0, 6);
                r.H_posB_posB = H.block<3, 3>(6, 6);
                r.H_posA_rotA = H.block<3, 3>(0, 3);
                r.H_posA_rotB = H.block<3, 3>(0, 9);
                r.H_posB_rotA = H.block<3, 3>(6, 3);
                r.H_posB_rotB = H.block<3, 3>(6, 9);
                r.H_rotA_rotA = H.block<3, 3>(3, 3);
                r.H_rotA_rotB = H.block<3, 3>(3, 9);
                r.H_rotB_rotB = H.block<3, 3>(9, 9);
            }
        }

        const Vec fq = sw.first_derivative * rq;
        const double f = sw.value;
        const Vec g = f * g0 + E0 * fq;

        r.energy = f * E0;
        r.force_A = -g.segment<3>(0);
        r.grad_angle_axis_A = g.segment<3>(3);
        r.force_B = -g.segment<3>(6);
        r.grad_angle_axis_B = g.segment<3>(9);
    };

    auto add_pair = [&](const PairHessianResult& p) {
        result.energy += p.energy;
        result.force_A += p.force_A;
        result.force_B += p.force_B;
        result.grad_angle_axis_A += p.grad_angle_axis_A;
        result.grad_angle_axis_B += p.grad_angle_axis_B;

        result.H_posA_posA += p.H_posA_posA;
        result.H_posA_posB += p.H_posA_posB;
        result.H_posB_posB += p.H_posB_posB;
        result.H_posA_rotA += p.H_posA_rotA;
        result.H_posA_rotB += p.H_posA_rotB;
        result.H_posB_rotA += p.H_posB_rotA;
        result.H_posB_rotB += p.H_posB_rotB;
        result.H_rotA_rotA += p.H_rotA_rotA;
        result.H_rotA_rotB += p.H_rotA_rotB;
        result.H_rotB_rotB += p.H_rotB_rotB;
    };

    const Vec3 offset_B_bohr = offset_B * to_bohr;

    for (size_t i = 0; i < molA.sites.size(); ++i) {
        const auto& sA = molA.sites[i];
        if (sA.rank < 0) continue;
        const auto& dA = site_data_A[i];

        for (size_t j = 0; j < molB.sites.size(); ++j) {
            const auto& sB = molB.sites[j];
            if (sB.rank < 0) continue;
            const auto& dB = site_data_B[j];

            Vec3 R_ang = (sB.position + offset_B) - sA.position;
            const double r_ang = R_ang.norm();
            if (use_site_cutoff && r_ang > site_cutoff) continue;
            if (use_taper && r_ang > taper->r_off) continue;

            const int nA = occ::ints::nhermsum(sA.rank);
            const int nB = occ::ints::nhermsum(sB.rank);

            const Vec3 R_bohr = (sB.position * to_bohr + offset_B_bohr) -
                                (sA.position * to_bohr);

            auto spf = dispatch_site_pair_contraction(
                dA.w.data(), dA.dw, dB.w.data(), dB.dw,
                nA, nB, R_bohr,
                use_order_cutoff, max_interaction_order);

            const double E = spf.E;
            const Vec3& g = spf.g;
            const Mat3& H = spf.H;
            const auto& fieldA0 = spf.fieldA0;
            const auto& fieldB0 = spf.fieldB0;
            const auto& fieldA_R = spf.fieldA_R;
            const auto& fieldB_R = spf.fieldB_R;
            const auto& fieldA_from_dB = spf.fieldA_from_dB;

            Vec3 grad_mp_A = Vec3::Zero();
            Vec3 grad_mp_B = Vec3::Zero();
            std::array<Vec3, 3> dg_dthetaA_constR;
            std::array<Vec3, 3> dg_dthetaB_constR;
            Mat3 d2gaAA = Mat3::Zero();
            Mat3 d2gbBB = Mat3::Zero();
            Mat3 d2gaAB = Mat3::Zero();

            for (int m = 0; m < 3; ++m) {
                dg_dthetaA_constR[m] = Vec3::Zero();
                dg_dthetaB_constR[m] = Vec3::Zero();

                grad_mp_A[m] = dot_n(dA.dw[m], fieldA0, nA);
                grad_mp_B[m] = dot_n(dB.dw[m], fieldB0, nB);

                for (int k = 0; k < 3; ++k) {
                    dg_dthetaA_constR[m][k] = dot_n(dA.dw[m], fieldA_R[k], nA);
                    dg_dthetaB_constR[m][k] = dot_n(dB.dw[m], fieldB_R[k], nB);
                }

                for (int n = 0; n < 3; ++n) {
                    d2gaAA(m, n) = dot_n(dA.d2w[m][n], fieldA0, nA);
                    d2gbBB(m, n) = dot_n(dB.d2w[m][n], fieldB0, nB);
                    d2gaAB(m, n) = dot_n(dA.dw[m], fieldA_from_dB[n], nA);
                }
            }

            PairHessianResult pair;
            pair.energy = E;
            pair.force_A = g;
            pair.force_B = -g;
            pair.H_posA_posA = H;
            pair.H_posA_posB = -H;
            pair.H_posB_posB = H;
            pair.grad_angle_axis_A = grad_mp_A - dA.lever.cross(g);
            pair.grad_angle_axis_B = grad_mp_B + dB.lever.cross(g);

            for (int m = 0; m < 3; ++m) {
                const Vec3 col_A = H * dA.dlever[m] - dg_dthetaA_constR[m];
                pair.H_posA_rotA.col(m) = col_A;
                pair.H_posB_rotA.col(m) = -col_A;

                const Vec3 col_B = -(H * dB.dlever[m] + dg_dthetaB_constR[m]);
                pair.H_posA_rotB.col(m) = col_B;
                pair.H_posB_rotB.col(m) = -col_B;
            }

            for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                    const Vec3 dg_dthetaA_n =
                        dg_dthetaA_constR[n] - H * dA.dlever[n];
                    const Vec3 dg_dthetaB_n =
                        dg_dthetaB_constR[n] + H * dB.dlever[n];

                    pair.H_rotA_rotA(m, n) =
                        -dg_dthetaA_n.dot(dA.dlever[m]) -
                        g.dot(dA.d2lever[m][n]) -
                        dg_dthetaA_constR[m].dot(dA.dlever[n]) +
                        d2gaAA(m, n);

                    pair.H_rotB_rotB(m, n) =
                        dg_dthetaB_n.dot(dB.dlever[m]) +
                        g.dot(dB.d2lever[m][n]) +
                        dg_dthetaB_constR[m].dot(dB.dlever[n]) +
                        d2gbBB(m, n);

                    pair.H_rotA_rotB(m, n) =
                        -dg_dthetaB_n.dot(dA.dlever[m]) +
                        dg_dthetaA_constR[m].dot(dB.dlever[n]) +
                        d2gaAB(m, n);
                }
            }

            scale_to_kjmol(pair);
            const double bohr_to_ang = occ::units::BOHR_TO_ANGSTROM;
            std::array<Vec3, 3> dR_dthetaA_ang, dR_dthetaB_ang;
            std::array<std::array<Vec3, 3>, 3> d2R_AA_ang, d2R_BB_ang;
            for (int m = 0; m < 3; ++m) {
                dR_dthetaA_ang[m] = -dA.dlever[m] * bohr_to_ang;
                dR_dthetaB_ang[m] = dB.dlever[m] * bohr_to_ang;
                for (int n = 0; n < 3; ++n) {
                    d2R_AA_ang[m][n] = -dA.d2lever[m][n] * bohr_to_ang;
                    d2R_BB_ang[m][n] = dB.d2lever[m][n] * bohr_to_ang;
                }
            }
            apply_radial_taper(
                pair, R_ang, dR_dthetaA_ang, dR_dthetaB_ang,
                d2R_AA_ang, d2R_BB_ang);
            add_pair(pair);
        }
    }

    return result;
}

// ============================================================================
// Public wrappers
// ============================================================================

PairHessianResult compute_molecule_hessian_truncated(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB,
    const Vec3& offset_B,
    double site_cutoff,
    int max_interaction_order,
    const CutoffSpline* taper,
    bool taper_hessian) {

    auto dataA = build_molecule_hessian_data(molA, true);
    auto dataB = build_molecule_hessian_data(molB, false);
    return compute_molecule_hessian_impl(
        molA, molB, dataA.sites, dataB.sites,
        offset_B, site_cutoff, max_interaction_order, taper, taper_hessian);
}

PairHessianResult compute_molecule_hessian_truncated(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB,
    const MoleculeHessianData& dataA,
    const MoleculeHessianData& dataB,
    const Vec3& offset_B,
    double site_cutoff,
    int max_interaction_order,
    const CutoffSpline* taper,
    bool taper_hessian) {

    return compute_molecule_hessian_impl(
        molA, molB, dataA.sites, dataB.sites,
        offset_B, site_cutoff, max_interaction_order, taper, taper_hessian);
}

} // namespace occ::mults
