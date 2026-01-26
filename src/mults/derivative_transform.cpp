#include <occ/mults/derivative_transform.h>
#include <stdexcept>
#include <fmt/core.h>
#include <fmt/format.h>

namespace occ::mults {

Vec3 DerivativeTransform::cross(const Vec3& a, const Vec3& b) {
    Vec3 result;
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
    return result;
}

Mat DerivativeTransform::compute_D1_angle_axis(
    const CoordinateSystem& coords,
    const Mat3& M_A,
    const Mat3& M_B,
    const std::array<Mat3, 3>& M1_A,
    const std::array<Mat3, 3>& M1_B,
    const Vec3& a,
    const Vec3& b) {

    // Initialize D1 matrix [16 x 12]
    Mat D1 = Mat::Zero(NUM_INTERMEDIATE_VARS, NUM_EXTERNAL_COORDS);

    const double r = coords.r;
    if (r < 1e-15) {
        throw std::runtime_error("Distance too small for derivative calculation");
    }

    // Extract coordinate components (following Orient's pairinfo_aa setup)
    const Vec3& rab = coords.rab;  // rb - ra
    const Vec3& er = coords.er;    // unit vector

    // CRITICAL: For angle-axis parametrization, we use BODY-FRAME-PROJECTED geometry:
    //   e1r = M_A^T · er  (unit vector in A's body frame)
    //   e2r = -M_B^T · er (unit vector in B's body frame, with sign flip)
    //
    // These unit vectors DO depend on molecular rotations through M_A and M_B,
    // so their derivatives w.r.t. angle-axis parameters are NON-ZERO.
    //
    // For point multipoles: srotlink = I (no site rotation relative to COM)
    //                       slink = 0 (sites at COM)
    // So: se_A = M_A, se_B = M_B (body frame orientations)
    Mat3 se_A = M_A;  // Site orientation = molecular orientation
    Mat3 se_B = M_B;  // Site orientation = molecular orientation

    // Unit vectors in body frames (from CoordinateSystem, precomputed)
    // coords should have been created with from_body_frame(ra, rb, M_A, M_B)
    Vec3 e1r = coords.raxyz();    // e1r_body from coords
    Vec3 e2r = coords.rbxyz();    // e2r_body from coords

    // Helper vectors (Orient line 358-359)
    Vec3 ra_vec = rab + a;  // From A's COM to site B
    Vec3 rb_vec = rab - b;  // From site A to B's COM

    // ========== TRANSLATIONAL DERIVATIVES (columns 0-2, 6-8) ==========
    // Following Orient lines 383-394

    // Distance r (row 15) - Orient line 384-385
    D1(15, 0) = -rab[0] / r;
    D1(15, 1) = -rab[1] / r;
    D1(15, 2) = -rab[2] / r;
    D1(15, 6) = rab[0] / r;
    D1(15, 7) = rab[1] / r;
    D1(15, 8) = rab[2] / r;

    // R·wa (rows 0-2) - Orient line 387-388
    // ∂(rab · M_A)/∂x_A = -M_A^T, ∂(rab · M_A)/∂x_B = +M_A^T
    D1.block<3,3>(0, 0) = -se_A.transpose();
    D1.block<3,3>(0, 6) = se_A.transpose();

    // -R·wb (rows 3-5) - Orient line 390-391
    // ∂(-rab · M_B)/∂x_A = +M_B^T, ∂(-rab · M_B)/∂x_B = -M_B^T
    D1.block<3,3>(3, 0) = se_B.transpose();
    D1.block<3,3>(3, 6) = -se_B.transpose();

    // wa·wb (rows 6-14) - Orient line 393-394: Independent of position
    D1.block<9,3>(6, 0).setZero();
    D1.block<9,3>(6, 6).setZero();

    // ========== ROTATIONAL DERIVATIVES (columns 3-5, 9-11) ==========
    // These use M1 derivatives following Orient's pairinfo_aa lines 398-424

    for (int i = 0; i < 3; i++) {
        const Mat3& M1A = M1_A[i];
        const Mat3& M1B = M1_B[i];

        // Distance r (row 15) - Orient line 411
        // ∂r/∂pA_i: Rotating A changes position of site at offset a
        D1(15, 3+i) = -ra_vec.dot(M1A * a) / r;

        // ∂r/∂pB_i: Rotating B changes position of site at offset b
        D1(15, 9+i) = rb_vec.dot(M1B * b) / r;

        // rab·wa (rows 0-2) - Orient line 414
        // ∂(rab · M_A)/∂p_A_i = rab · M1_A_i
        // For point multipoles: ra_vec = rab, srotlink = I
        D1.col(3+i).segment<3>(0) = M1A.transpose() * ra_vec;

        // ∂(rab · M_A)/∂p_B_i = 0 (M_A doesn't depend on B's rotation)
        D1.col(9+i).segment<3>(0) = Vec3::Zero();

        // -rab·wb (rows 3-5) - Orient line 418
        // ∂(-rab · M_B)/∂p_A_i = 0 (M_B doesn't depend on A's rotation)
        D1.col(3+i).segment<3>(3) = Vec3::Zero();

        // ∂(-rab · M_B)/∂p_B_i = -rab · M1_B_i
        // For point multipoles: rb_vec = rab, srotlink = I
        D1.col(9+i).segment<3>(3) = -M1B.transpose() * rb_vec;

        // wa·wb (rows 6-14) - Orient line 420-423
        // Orientation dot products: xx(i,j) = M_A^T · M_B (relative multipole orientation)
        // For point multipoles with srotlink = I:
        //   ∂(M_A^T · M_B)/∂pA_i = M1_A(:,:,i)^T · M_B
        //   ∂(M_A^T · M_B)/∂pB_i = M_A^T · M1_B(:,:,i)
        Mat3 dxx_dpA = M1A.transpose() * M_B;
        Mat3 dxx_dpB = M_A.transpose() * M1B;

        // Flatten 3x3 matrix to 9-vector in ROW-MAJOR order to match S-function indexing
        // S-function derivatives use row-major: s1[6]=cxx, s1[7]=cxy, s1[8]=cxz, s1[9]=cyx, etc.
        // Order: cxx, cxy, cxz, cyx, cyy, cyz, czx, czy, czz
        Eigen::Matrix<double, 9, 1> dxx_dpA_vec;
        Eigen::Matrix<double, 9, 1> dxx_dpB_vec;
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                dxx_dpA_vec[row * 3 + col] = dxx_dpA(row, col);
                dxx_dpB_vec[row * 3 + col] = dxx_dpB(row, col);
            }
        }

        D1.col(3+i).segment<9>(6) = dxx_dpA_vec;
        D1.col(9+i).segment<9>(6) = dxx_dpB_vec;
    }

    // ========== CONVERT TO UNIT VECTOR DERIVATIVES ==========
    // Transform ∂(rab·wa)/∂p → ∂(er·wa)/∂p
    // Following Orient lines 434-437

    for (int ip = 0; ip < NUM_EXTERNAL_COORDS; ++ip) {
        for (int i = 0; i < 3; ++i) {
            // Unit vector for site A (e1r)
            D1(i, ip) = (D1(i, ip) - e1r[i] * D1(15, ip)) / r;

            // Unit vector for site B (e2r)
            D1(3 + i, ip) = (D1(3 + i, ip) - e2r[i] * D1(15, ip)) / r;
        }
    }

    // DEBUG (disabled)
    // static bool printed = false;
    // if (!printed) {
    //     fmt::print("\n=== D1 angle-axis Debug ===\n");
    //     fmt::print("rab = [{:.6f}, {:.6f}, {:.6f}], r = {:.6f}\n", rab[0], rab[1], rab[2], r);
    //     fmt::print("e1r = [{:.6f}, {:.6f}, {:.6f}]\n", e1r[0], e1r[1], e1r[2]);
    //     fmt::print("e2r = [{:.6f}, {:.6f}, {:.6f}]\n", e2r[0], e2r[1], e2r[2]);
    //     fmt::print("D1 rows 0-2 (e1r), cols 3-5 (px_A, py_A, pz_A):\n");
    //     for (int i = 0; i < 3; i++) {
    //         fmt::print("  [{:.6e}, {:.6e}, {:.6e}]\n", D1(i, 3), D1(i, 4), D1(i, 5));
    //     }
    //     fmt::print("D1 rows 6-14 (orientation), cols 3-5 (px_A, py_A, pz_A):\n");
    //     for (int i = 6; i <= 14; i++) {
    //         fmt::print("  [{:.6e}, {:.6e}, {:.6e}]\n", D1(i, 3), D1(i, 4), D1(i, 5));
    //     }
    //     fmt::print("D1 row 15 (r), cols 3-5 (px_A, py_A, pz_A):\n");
    //     fmt::print("  [{:.6e}, {:.6e}, {:.6e}]\n", D1(15, 3), D1(15, 4), D1(15, 5));
    //
    //     fmt::print("\nM1_A derivatives:\n");
    //     for (int k = 0; k < 3; k++) {
    //         fmt::print("M1_A[{}]:\n", k);
    //         for (int i = 0; i < 3; i++) {
    //             fmt::print("  [{:.6e}, {:.6e}, {:.6e}]\n", M1_A[k](i,0), M1_A[k](i,1), M1_A[k](i,2));
    //         }
    //     }
    //     printed = true;
    // }

    return D1;
}

Mat DerivativeTransform::compute_D1(const CoordinateSystem& coords,
                                     const Vec3& a, const Vec3& b) {
    // Initialize D1 matrix [16 x 12]
    Mat D1 = Mat::Zero(NUM_INTERMEDIATE_VARS, NUM_EXTERNAL_COORDS);

    const double r = coords.r;
    if (r < 1e-15) {
        throw std::runtime_error("Distance too small for derivative calculation");
    }

    // Extract coordinate components
    const Vec3& rab = coords.rab;  // rb - ra
    const Vec3& er = coords.er;    // unit vector

    // Orient's e1r and e2r (unit vectors in local frames)
    // For point multipoles with identity orientation matrix:
    Vec3 e1r = coords.raxyz();  // +er
    Vec3 e2r = coords.rbxyz();  // -er

    // Compute helper vectors (following Orient's mlinfo.f90)
    Vec3 ra = rab + a;  // R + a (constant w.r.t. rotations of A)
    Vec3 rb = rab - b;  // R - b (constant w.r.t. rotations of B)

    // Cross products needed for first derivatives
    Vec3 rxa = cross(rab, a);
    Vec3 rxb = cross(rab, b);

    // For point multipoles, we need cross products with local axes
    // Since orientation matrix is identity, local axes = global axes
    // These would be used if we had molecular rotations
    // For now, simplified for point multipoles

    // --------------------------------------------------------------------------
    // First derivatives of |R| (q(16) in Orient, index 15 in C++)
    // --------------------------------------------------------------------------
    // ∂R/∂x_A = -rab/R
    D1(15, 0) = -rab[0] / r;
    D1(15, 1) = -rab[1] / r;
    D1(15, 2) = -rab[2] / r;

    // ∂R/∂tau_A = (rab × a) / R (for molecular rotations)
    D1(15, 3) = rxa[0] / r;
    D1(15, 4) = rxa[1] / r;
    D1(15, 5) = rxa[2] / r;

    // ∂R/∂x_B = +rab/R
    D1(15, 6) = rab[0] / r;
    D1(15, 7) = rab[1] / r;
    D1(15, 8) = rab[2] / r;

    // ∂R/∂tau_B = -(rab × b) / R
    D1(15, 9) = -rxb[0] / r;
    D1(15, 10) = -rxb[1] / r;
    D1(15, 11) = -rxb[2] / r;

    // --------------------------------------------------------------------------
    // First derivatives of w1.R (q(1:3), indices 0-2)
    // Note: These are derivatives of the VECTOR R, not unit vector
    // Will be converted to unit vector derivatives later
    // --------------------------------------------------------------------------

    // For point multipoles with identity orientation (se = I):
    // w1(i,j) = delta(i,j), so w1.R = R in the molecular frame = rab in global frame

    // ∂(w1.R)/∂x_A = -w1 = -I (negative identity for each component)
    for (int j = 0; j < 3; ++j) {
        D1(j, 0) = (j == 0) ? -1.0 : 0.0;  // -delta(j, 0)
        D1(j, 1) = (j == 1) ? -1.0 : 0.0;  // -delta(j, 1)
        D1(j, 2) = (j == 2) ? -1.0 : 0.0;  // -delta(j, 2)
    }

    // ∂(w1.R)/∂tau_A = -(R+a) × w1
    // For identity orientation and each component:
    // This requires the cross product of (rab + a) with each local axis
    // Local axis i = e_i (standard basis vector)
    for (int j = 0; j < 3; ++j) {
        Vec3 local_axis = Vec3::Zero();
        local_axis[j] = 1.0;
        Vec3 raxw1 = cross(ra, local_axis);
        D1(j, 3) = -raxw1[0];
        D1(j, 4) = -raxw1[1];
        D1(j, 5) = -raxw1[2];
    }

    // ∂(w1.R)/∂x_B = +w1 = +I
    for (int j = 0; j < 3; ++j) {
        D1(j, 6) = (j == 0) ? 1.0 : 0.0;
        D1(j, 7) = (j == 1) ? 1.0 : 0.0;
        D1(j, 8) = (j == 2) ? 1.0 : 0.0;
    }

    // ∂(w1.R)/∂tau_B = -w1 × b
    for (int j = 0; j < 3; ++j) {
        Vec3 local_axis = Vec3::Zero();
        local_axis[j] = 1.0;
        Vec3 w1xb = cross(local_axis, b);
        D1(j, 9) = -w1xb[0];
        D1(j, 10) = -w1xb[1];
        D1(j, 11) = -w1xb[2];
    }

    // --------------------------------------------------------------------------
    // First derivatives of w2.R (q(4:6), indices 3-5)
    // --------------------------------------------------------------------------

    // ∂(w2.R)/∂x_A = +w2 = +I (for identity orientation)
    for (int j = 0; j < 3; ++j) {
        D1(3 + j, 0) = (j == 0) ? 1.0 : 0.0;
        D1(3 + j, 1) = (j == 1) ? 1.0 : 0.0;
        D1(3 + j, 2) = (j == 2) ? 1.0 : 0.0;
    }

    // ∂(w2.R)/∂tau_A = -w2 × a
    for (int j = 0; j < 3; ++j) {
        Vec3 local_axis = Vec3::Zero();
        local_axis[j] = 1.0;
        Vec3 w2xa = cross(local_axis, a);
        D1(3 + j, 3) = -w2xa[0];
        D1(3 + j, 4) = -w2xa[1];
        D1(3 + j, 5) = -w2xa[2];
    }

    // ∂(w2.R)/∂x_B = -w2 = -I
    for (int j = 0; j < 3; ++j) {
        D1(3 + j, 6) = (j == 0) ? -1.0 : 0.0;
        D1(3 + j, 7) = (j == 1) ? -1.0 : 0.0;
        D1(3 + j, 8) = (j == 2) ? -1.0 : 0.0;
    }

    // ∂(w2.R)/∂tau_B = (R-b) × w2
    for (int j = 0; j < 3; ++j) {
        Vec3 local_axis = Vec3::Zero();
        local_axis[j] = 1.0;
        Vec3 rbxw2 = cross(rb, local_axis);
        D1(3 + j, 9) = rbxw2[0];
        D1(3 + j, 10) = rbxw2[1];
        D1(3 + j, 11) = rbxw2[2];
    }

    // --------------------------------------------------------------------------
    // First derivatives of w1.w2 (q(7:15), indices 6-14)
    // Orientation matrix elements: xx(i,j) = w1(i) . w2(j)
    // --------------------------------------------------------------------------

    // For identity orientation matrices: xx(i,j) = delta(i,j)
    // ∂(w1.w2)/∂x_A = 0 (translation doesn't affect orientation)
    // ∂(w1.w2)/∂x_B = 0

    for (int j1 = 0; j1 < 3; ++j1) {
        for (int j2 = 0; j2 < 3; ++j2) {
            int idx = 6 + j1 * 3 + j2;  // Flatten 3x3 matrix in row-major order

            // Translation derivatives are zero
            D1(idx, 0) = 0.0;
            D1(idx, 1) = 0.0;
            D1(idx, 2) = 0.0;
            D1(idx, 6) = 0.0;
            D1(idx, 7) = 0.0;
            D1(idx, 8) = 0.0;

            // Rotation derivatives: w1 × w2
            // For identity: ∂(w1(j1).w2(j2))/∂tau_A = (w1(j1) × w1(j2))
            Vec3 w1_j1 = Vec3::Zero(); w1_j1[j1] = 1.0;
            Vec3 w2_j2 = Vec3::Zero(); w2_j2[j2] = 1.0;
            Vec3 w1xw2 = cross(w1_j1, w2_j2);

            D1(idx, 3) = w1xw2[0];
            D1(idx, 4) = w1xw2[1];
            D1(idx, 5) = w1xw2[2];

            D1(idx, 9) = -w1xw2[0];
            D1(idx, 10) = -w1xw2[1];
            D1(idx, 11) = -w1xw2[2];
        }
    }

    // --------------------------------------------------------------------------
    // Convert derivatives of w1.R and w2.R to unit vector derivatives
    // Following Orient's formula:
    // ∂(e1r)/∂x = [∂(w1.R)/∂x - e1r · ∂R/∂x] / R
    // --------------------------------------------------------------------------

    for (int ip = 0; ip < NUM_EXTERNAL_COORDS; ++ip) {
        for (int i = 0; i < 3; ++i) {
            // Unit vector for site A (e1r)
            D1(i, ip) = (D1(i, ip) - e1r[i] * D1(15, ip)) / r;

            // Unit vector for site B (e2r)
            D1(3 + i, ip) = (D1(3 + i, ip) - e2r[i] * D1(15, ip)) / r;
        }
    }

    return D1;
}

std::array<Mat, DerivativeTransform::NUM_INTERMEDIATE_VARS>
DerivativeTransform::compute_D2(const CoordinateSystem& coords,
                                 const Vec3& a, const Vec3& b) {
    // Initialize D2 tensor as array of 16 matrices, each [12 x 12]
    std::array<Mat, NUM_INTERMEDIATE_VARS> D2;
    for (int iq = 0; iq < NUM_INTERMEDIATE_VARS; ++iq) {
        D2[iq] = Mat::Zero(NUM_EXTERNAL_COORDS, NUM_EXTERNAL_COORDS);
    }

    const double r = coords.r;
    if (r < 1e-15) {
        throw std::runtime_error("Distance too small for derivative calculation");
    }

    // Extract coordinate components
    const Vec3& rab = coords.rab;
    const Vec3& er = coords.er;
    Vec3 e1r = coords.raxyz();
    Vec3 e2r = coords.rbxyz();

    // Compute helper vectors
    Vec3 ra = rab + a;
    Vec3 rb = rab - b;

    // Scalar products needed for second derivatives
    double ab = a.dot(b);
    double raa = ra.dot(a);
    double rbb = rb.dot(b);

    // Additional scalar products with local axes (for identity orientation)
    Vec aw1 = Vec::Zero(3);
    Vec aw2 = Vec::Zero(3);
    Vec bw1 = Vec::Zero(3);
    Vec bw2 = Vec::Zero(3);
    Vec w1ra = Vec::Zero(3);
    Vec w2rb = Vec::Zero(3);

    for (int i = 0; i < 3; ++i) {
        // For identity orientation: w(i) = e_i
        aw1[i] = a[i];
        aw2[i] = a[i];
        bw1[i] = b[i];
        bw2[i] = b[i];
        w1ra[i] = ra[i];
        w2rb[i] = rb[i];
    }

    // Permutation indices for cross product derivatives
    std::array<int, 3> i1 = {1, 2, 0};
    std::array<int, 3> i2 = {2, 0, 1};

    // --------------------------------------------------------------------------
    // Second derivatives of R.R (will divide by 2R to get derivatives of R)
    // Following Orient's mlinfo.f90 lines 248-289
    // --------------------------------------------------------------------------

    Mat& D2_R = D2[15];  // q(16) = R, index 15

    // Diagonal blocks for pure translations
    for (int i = 0; i < 3; ++i) {
        D2_R(i, i) = 1.0;           // ∂²(R²)/∂x_A_i²
        D2_R(6+i, i) = -1.0;        // ∂²(R²)/∂x_A_i ∂x_B_i
        D2_R(i, 6+i) = -1.0;        // ∂²(R²)/∂x_B_i ∂x_A_i
        D2_R(6+i, 6+i) = 1.0;       // ∂²(R²)/∂x_B_i²
    }

    // Mixed torque-torque derivatives
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            D2_R(3+i, 3+j) = -ra[i] * a[j];
            D2_R(3+i, 9+j) = b[i] * a[j];
            D2_R(9+i, 3+j) = a[i] * b[j];
            D2_R(9+i, 9+j) = rb[i] * b[j];
        }
        D2_R(3+i, 3+i) += raa;
        D2_R(3+i, 9+i) -= ab;
        D2_R(9+i, 3+i) -= ab;
        D2_R(9+i, 9+i) -= rbb;
    }

    // Cross derivatives (position-torque coupling)
    for (int i = 0; i < 3; ++i) {
        int j = i1[i];
        int k = i2[i];

        D2_R(j, 3+k) = a[i];
        D2_R(k, 3+j) = -a[i];
        D2_R(j, 9+k) = -b[i];
        D2_R(k, 9+j) = b[i];

        D2_R(3+j, k) = -a[i];
        D2_R(3+k, j) = a[i];
        D2_R(3+j, 6+k) = a[i];
        D2_R(3+k, 6+j) = -a[i];

        D2_R(6+j, 3+k) = -a[i];
        D2_R(6+k, 3+j) = a[i];
        D2_R(6+j, 9+k) = b[i];
        D2_R(6+k, 9+j) = -b[i];

        D2_R(9+j, k) = b[i];
        D2_R(9+k, j) = -b[i];
        D2_R(9+j, 6+k) = -b[i];
        D2_R(9+k, 6+j) = b[i];
    }

    // Get first derivatives for conversion
    Mat D1 = compute_D1(coords, a, b);

    // Convert from ∂²(R²) to ∂²R using: ∂²R/∂x∂y = [∂²(R²) - ∂R/∂x · ∂R/∂y] / R
    for (int i = 0; i < NUM_EXTERNAL_COORDS; ++i) {
        for (int j = 0; j < NUM_EXTERNAL_COORDS; ++j) {
            D2_R(i, j) = (D2_R(i, j) - D1(15, i) * D1(15, j)) / r;
        }
    }

    // --------------------------------------------------------------------------
    // Second derivatives of w1.R and w2.R
    // Following Orient's mlinfo.f90 lines 291-349
    // --------------------------------------------------------------------------

    for (int k = 0; k < 3; ++k) {
        Mat& D2_w1k = D2[k];      // q(k+1) in Orient, index k in C++
        Mat& D2_w2k = D2[3 + k];  // q(k+4) in Orient, index 3+k in C++

        for (int i = 0; i < 3; ++i) {
            int j1 = i1[i];
            int j2 = i2[i];

            // w1(k).R derivatives
            for (int j = 0; j < 3; ++j) {
                D2_w1k(3+i, 3+j) = ra[i] * ((k == j) ? 1.0 : 0.0);
                D2_w1k(3+i, 9+j) = -b[i] * ((k == j) ? 1.0 : 0.0);
                D2_w1k(9+i, 3+j) = -((k == i) ? 1.0 : 0.0) * b[j];
                D2_w1k(9+i, 9+j) = ((k == i) ? 1.0 : 0.0) * b[j];
            }
            D2_w1k(3+i, 3+i) += -w1ra[k];
            D2_w1k(3+i, 9+i) += bw1[k];
            D2_w1k(9+i, 3+i) += bw1[k];
            D2_w1k(9+i, 9+i) += -bw1[k];

            double delta_ik = (k == i) ? 1.0 : 0.0;
            D2_w1k(j1, 3+j2) = -delta_ik;
            D2_w1k(j2, 3+j1) = delta_ik;
            D2_w1k(3+j1, j2) = delta_ik;
            D2_w1k(3+j2, j1) = -delta_ik;
            D2_w1k(3+j1, 6+j2) = -delta_ik;
            D2_w1k(3+j2, 6+j1) = delta_ik;
            D2_w1k(6+j1, 3+j2) = delta_ik;
            D2_w1k(6+j2, 3+j1) = -delta_ik;

            // w2(k).R derivatives
            for (int j = 0; j < 3; ++j) {
                D2_w2k(3+i, 3+j) = ((k == i) ? 1.0 : 0.0) * a[j];
                D2_w2k(3+i, 9+j) = -((k == i) ? 1.0 : 0.0) * a[j];
                D2_w2k(9+i, 3+j) = -a[i] * ((k == j) ? 1.0 : 0.0);
                D2_w2k(9+i, 9+j) = -rb[i] * ((k == j) ? 1.0 : 0.0);
            }
            D2_w2k(3+i, 3+i) += -aw2[k];
            D2_w2k(3+i, 9+i) += aw2[k];
            D2_w2k(9+i, 3+i) += aw2[k];
            D2_w2k(9+i, 9+i) += w2rb[k];

            D2_w2k(j1, 9+j2) = delta_ik;
            D2_w2k(j2, 9+j1) = -delta_ik;
            D2_w2k(6+j1, 9+j2) = -delta_ik;
            D2_w2k(6+j2, 9+j1) = delta_ik;
            D2_w2k(9+j1, j2) = -delta_ik;
            D2_w2k(9+j2, j1) = delta_ik;
            D2_w2k(9+j1, 6+j2) = delta_ik;
            D2_w2k(9+j2, 6+j1) = -delta_ik;
        }
    }

    // --------------------------------------------------------------------------
    // Second derivatives of w1.w2 (orientation matrix)
    // Following Orient's mlinfo.f90 lines 335-349
    // --------------------------------------------------------------------------

    for (int k1 = 0; k1 < 3; ++k1) {
        for (int k2 = 0; k2 < 3; ++k2) {
            int idx = 6 + k1 * 3 + k2;  // Flatten in row-major order
            Mat& D2_xx = D2[idx];

            double xx_k1k2 = (k1 == k2) ? 1.0 : 0.0;  // Identity orientation

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    double delta_jk1 = (j == k1) ? 1.0 : 0.0;
                    double delta_ik2 = (i == k2) ? 1.0 : 0.0;
                    double delta_ik1 = (i == k1) ? 1.0 : 0.0;
                    double delta_jk2 = (j == k2) ? 1.0 : 0.0;

                    D2_xx(3+i, 3+j) = delta_jk1 * delta_ik2;
                    D2_xx(3+i, 9+j) = -delta_jk1 * delta_ik2;
                    D2_xx(9+i, 3+j) = -delta_ik1 * delta_jk2;
                    D2_xx(9+i, 9+j) = delta_ik1 * delta_jk2;
                }
                D2_xx(3+i, 3+i) += -xx_k1k2;
                D2_xx(3+i, 9+i) += xx_k1k2;
                D2_xx(9+i, 3+i) += xx_k1k2;
                D2_xx(9+i, 9+i) += -xx_k1k2;
            }
        }
    }

    // --------------------------------------------------------------------------
    // Convert w1.R and w2.R derivatives to unit vector derivatives
    // Following Orient's formula (lines 351-520 in mlinfo.f90)
    // --------------------------------------------------------------------------

    for (int j = 0; j < NUM_EXTERNAL_COORDS; ++j) {
        for (int i = 0; i < NUM_EXTERNAL_COORDS; ++i) {
            for (int k = 0; k < 3; ++k) {
                // e1r components
                D2[k](i, j) = (D2[k](i, j) - e1r[k] * D2[15](i, j)
                              - D1(15, i) * D1(k, j) - D1(15, j) * D1(k, i)) / r;

                // e2r components
                D2[3+k](i, j) = (D2[3+k](i, j) - e2r[k] * D2[15](i, j)
                                - D1(15, i) * D1(3+k, j) - D1(15, j) * D1(3+k, i)) / r;
            }
        }
    }

    // Ensure symmetry in all D2 matrices (second derivatives are symmetric)
    for (int iq = 0; iq < NUM_INTERMEDIATE_VARS; ++iq) {
        for (int i = 0; i < NUM_EXTERNAL_COORDS; ++i) {
            for (int j = i + 1; j < NUM_EXTERNAL_COORDS; ++j) {
                // Average the two entries to enforce symmetry
                double avg = 0.5 * (D2[iq](i, j) + D2[iq](j, i));
                D2[iq](i, j) = avg;
                D2[iq](j, i) = avg;
            }
        }
    }

    return D2;
}

Mat DerivativeTransform::compute_D1S(const Mat& S1, const Mat& D1) {
    // D1S[ip][ix] = Σⱼ S1[iq][ix] · D1[iq][ip]
    // S1 is [15 x nmax], D1 is [16 x 12], result is [12 x nmax]

    const int nmax = S1.cols();
    Mat D1S = Mat::Zero(NUM_EXTERNAL_COORDS, nmax);

    // Matrix multiplication: D1S = D1^T(0:14, :) * S1
    // Only use first 15 rows of D1 (q(1:15), indices 0-14)
    for (int ix = 0; ix < nmax; ++ix) {
        for (int ip = 0; ip < NUM_EXTERNAL_COORDS; ++ip) {
            double sum = 0.0;
            for (int iq = 0; iq < NUM_FIRST_DERIV_VARS; ++iq) {
                sum += S1(iq, ix) * D1(iq, ip);
            }
            D1S(ip, ix) = sum;
        }
    }

    return D1S;
}

std::vector<Mat> DerivativeTransform::compute_D2S(
    const Mat& S1, const Mat& S2, const Mat& D1,
    const std::array<Mat, NUM_INTERMEDIATE_VARS>& D2) {

    const int nmax = S1.cols();

    // D2S is a vector of matrices, one [12 x 12] matrix per S-function
    std::vector<Mat> D2S(nmax, Mat::Zero(NUM_EXTERNAL_COORDS, NUM_EXTERNAL_COORDS));

    // Following Orient's optimized computation (sfns.F90 lines 172-231)
    for (int ix = 0; ix < nmax; ++ix) {
        // Initialize to zero
        Mat& D2S_ix = D2S[ix];

        // First term: Σⱼₖ S2[packed(j,k)] · D1[iq][ip] · D1[jq][jp]
        int kq = 0;
        for (int iq = 0; iq < NUM_FIRST_DERIV_VARS; ++iq) {
            double f = 1.0;
            for (int jq = 0; jq <= iq; ++jq) {
                if (jq == iq) f = 0.5;

                if (std::abs(S2(kq, ix)) > 1e-15) {
                    for (int jp = 0; jp < NUM_EXTERNAL_COORDS; ++jp) {
                        for (int ip = 0; ip < NUM_EXTERNAL_COORDS; ++ip) {
                            D2S_ix(ip, jp) += f * S2(kq, ix) *
                                (D1(iq, ip) * D1(jq, jp) + D1(iq, jp) * D1(jq, ip));
                        }
                    }
                }
                ++kq;
            }
        }

        // Second term: Σⱼ S1[iq] · D2[iq][ip][jp]
        for (int iq = 0; iq < NUM_FIRST_DERIV_VARS; ++iq) {
            if (std::abs(S1(iq, ix)) > 1e-15) {
                for (int jp = 0; jp < NUM_EXTERNAL_COORDS; ++jp) {
                    for (int ip = 0; ip < NUM_EXTERNAL_COORDS; ++ip) {
                        D2S_ix(ip, jp) += S1(iq, ix) * D2[iq](ip, jp);
                    }
                }
            }
        }
    }

    return D2S;
}

std::pair<int, int> DerivativeTransform::unpack_symmetric_index(int kq) {
    // Convert packed index to (i, j) where i >= j
    // kq = 0 → (0, 0)
    // kq = 1 → (1, 0)
    // kq = 2 → (1, 1)
    // kq = 3 → (2, 0)
    // etc.

    int i = 0;
    int sum = 0;
    while (sum + i + 1 <= kq) {
        sum += i + 1;
        ++i;
    }
    int j = kq - sum;

    return {i, j};
}

int DerivativeTransform::pack_symmetric_index(int i, int j) {
    if (j > i) std::swap(i, j);  // Ensure i >= j
    return i * (i + 1) / 2 + j;
}

} // namespace occ::mults
