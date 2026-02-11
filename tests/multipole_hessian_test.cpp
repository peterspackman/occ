#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/mults/cartesian_force.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cartesian_interaction.h>
#include <occ/mults/cartesian_hessian.h>
#include <occ/mults/rigid_body.h>
#include <occ/dma/mult.h>
#include <fmt/core.h>

using namespace occ::mults;
using namespace occ;
using Catch::Approx;
using dma::Mult;

/// Build a rotation matrix from axis-angle (Rodrigues).
static Mat3 axis_angle_rotation(const Vec3 &axis, double angle) {
    Vec3 n = axis.normalized();
    double c = std::cos(angle), s = std::sin(angle);
    Mat3 K;
    K << 0, -n[2], n[1],
         n[2], 0, -n[0],
         -n[1], n[0], 0;
    return Mat3::Identity() + s * K + (1.0 - c) * K * K;
}

/// Build a CartesianMolecule from body-frame data with rotation.
static CartesianMolecule build_molecule(
    const std::vector<std::pair<Mult, Vec3>>& body_sites,
    const Mat3& rotation,
    const Vec3& center) {
    return CartesianMolecule::from_body_frame_with_rotation(body_sites, rotation, center);
}

/// Compute energy for a given configuration.
static double compute_energy(
    const std::vector<std::pair<Mult, Vec3>>& body_A,
    const std::vector<std::pair<Mult, Vec3>>& body_B,
    const Mat3& rot_A, const Vec3& center_A,
    const Mat3& rot_B, const Vec3& center_B) {
    auto molA = build_molecule(body_A, rot_A, center_A);
    auto molB = build_molecule(body_B, rot_B, center_B);
    return compute_molecule_interaction(molA, molB);
}

/// Compute gradient w.r.t. center_A position.
static Vec3 compute_grad_center_A(
    const std::vector<std::pair<Mult, Vec3>>& body_A,
    const std::vector<std::pair<Mult, Vec3>>& body_B,
    const Mat3& rot_A, const Vec3& center_A,
    const Mat3& rot_B, const Vec3& center_B) {
    auto molA = build_molecule(body_A, rot_A, center_A);
    auto molB = build_molecule(body_B, rot_B, center_B);
    auto result = compute_molecule_forces_torques(molA, molB);
    return -result.force_A;  // gradient = -force
}

/// Compute gradient w.r.t. angle-axis of molecule A.
static Vec3 compute_grad_angle_A(
    const std::vector<std::pair<Mult, Vec3>>& body_A,
    const std::vector<std::pair<Mult, Vec3>>& body_B,
    const Mat3& rot_A, const Vec3& center_A,
    const Mat3& rot_B, const Vec3& center_B) {
    auto molA = build_molecule(body_A, rot_A, center_A);
    auto molB = build_molecule(body_B, rot_B, center_B);
    auto result = compute_molecule_forces_torques(molA, molB);
    return result.grad_angle_axis_A;
}

// ============================================================================
// Test: Numerical Hessian from gradient finite differences
// ============================================================================

TEST_CASE("Translation-translation Hessian via gradient FD", "[hessian][translation]") {
    // Two dipolar molecules
    Mult m1(2), m2(2);
    m1.Q00() = -0.5;
    m1.Q10() = 1.0;
    m2.Q00() = 0.5;
    m2.Q10() = -0.8;

    Vec3 body_offset(0, 0, 0);
    std::vector<std::pair<Mult, Vec3>> body_A = {{m1, body_offset}};
    std::vector<std::pair<Mult, Vec3>> body_B = {{m2, body_offset}};

    Mat3 rot_A = axis_angle_rotation(Vec3(0, 1, 0), 0.3);
    Mat3 rot_B = Mat3::Identity();
    Vec3 center_A(0, 0, 0);
    Vec3 center_B(5, 2, -1);

    // Compute gradient at reference point
    Vec3 grad_ref = compute_grad_center_A(body_A, body_B, rot_A, center_A, rot_B, center_B);

    // Numerical Hessian d²E/dxA_i dxA_j via gradient finite difference
    const double h = 1e-5;
    Mat3 numerical_hessian;

    for (int j = 0; j < 3; ++j) {
        Vec3 center_plus = center_A;
        Vec3 center_minus = center_A;
        center_plus[j] += h;
        center_minus[j] -= h;

        Vec3 grad_plus = compute_grad_center_A(body_A, body_B, rot_A, center_plus, rot_B, center_B);
        Vec3 grad_minus = compute_grad_center_A(body_A, body_B, rot_A, center_minus, rot_B, center_B);

        for (int i = 0; i < 3; ++i) {
            numerical_hessian(i, j) = (grad_plus[i] - grad_minus[i]) / (2.0 * h);
        }
    }

    SECTION("Hessian is symmetric") {
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                INFO(fmt::format("H({},{})={:.10e}  H({},{})={:.10e}",
                                i, j, numerical_hessian(i, j),
                                j, i, numerical_hessian(j, i)));
                REQUIRE(numerical_hessian(i, j) == Approx(numerical_hessian(j, i)).epsilon(1e-6));
            }
        }
    }

    SECTION("Hessian has reasonable magnitude") {
        // For electrostatic interactions at ~5 angstrom, Hessian should be non-trivial
        double hess_norm = numerical_hessian.norm();
        INFO(fmt::format("Hessian Frobenius norm: {:.6e}", hess_norm));
        REQUIRE(hess_norm > 1e-6);  // Should have curvature
        REQUIRE(hess_norm < 1e6);   // But not crazy large
    }
}

TEST_CASE("Rotation-rotation Hessian via gradient FD", "[hessian][rotation]") {
    // NOTE: grad_angle_axis is the derivative with respect to infinitesimal rotation
    // about LAB FRAME (space frame) axes. For FD, we should use left-multiplication:
    //   R' = exp(h*e_k) * R
    // This corresponds to a small rotation about lab axis k.
    //
    // However, the Hessian d²E/dθi dθj (where θi is rotation about lab axis i)
    // is NOT symmetric in general because rotations don't commute.
    // The symmetry H_ij = H_ji only holds when the energy is a function of
    // commuting variables.
    //
    // For L-BFGS optimization, we don't need the exact Hessian - the L-BFGS
    // approximation handles non-commutativity naturally. What matters is:
    // 1. The gradient is correct (validated by energy FD)
    // 2. The gradient is smooth (allows L-BFGS to work)

    Mult m1(2), m2(2);
    m1.Q00() = -0.5;
    m1.Q10() = 1.0;
    m1.Q11c() = 0.5;
    m2.Q00() = 0.5;
    m2.Q10() = -0.8;
    m2.Q11s() = 0.3;

    Vec3 body_offset(0, 0, 0);
    std::vector<std::pair<Mult, Vec3>> body_A = {{m1, body_offset}};
    std::vector<std::pair<Mult, Vec3>> body_B = {{m2, body_offset}};

    Mat3 rot_A = axis_angle_rotation(Vec3(1, 1, 0).normalized(), 0.4);
    Mat3 rot_B = Mat3::Identity();
    Vec3 center_A(0, 0, 0);
    Vec3 center_B(5, 2, -1);

    // Numerical Hessian d²E/dθA_i dθA_j via gradient finite difference
    // Using lab-frame rotations (left multiplication)
    const double h = 1e-5;
    Mat3 numerical_hessian;

    for (int j = 0; j < 3; ++j) {
        Vec3 axis = Vec3::Zero();
        axis[j] = 1.0;

        Mat3 dR_plus = axis_angle_rotation(axis, h);
        Mat3 dR_minus = axis_angle_rotation(axis, -h);
        Mat3 rot_plus = dR_plus * rot_A;    // Left multiplication = lab frame rotation
        Mat3 rot_minus = dR_minus * rot_A;

        Vec3 grad_plus = compute_grad_angle_A(body_A, body_B, rot_plus, center_A, rot_B, center_B);
        Vec3 grad_minus = compute_grad_angle_A(body_A, body_B, rot_minus, center_A, rot_B, center_B);

        for (int i = 0; i < 3; ++i) {
            numerical_hessian(i, j) = (grad_plus[i] - grad_minus[i]) / (2.0 * h);
        }
    }

    SECTION("Hessian has reasonable magnitude") {
        double hess_norm = numerical_hessian.norm();
        INFO(fmt::format("Rotation Hessian Frobenius norm: {:.6e}", hess_norm));
        REQUIRE(hess_norm > 1e-8);  // Should have curvature
        REQUIRE(hess_norm < 1e4);   // But not crazy large
    }

    SECTION("Gradient is smooth under rotation perturbation") {
        // Check that gradient changes smoothly as we rotate
        Vec3 axis(0, 1, 0);
        double h_test = 0.01;  // 0.01 radian ~ 0.6 degrees
        Mat3 dR = axis_angle_rotation(axis, h_test);
        Mat3 rot_perturbed = dR * rot_A;

        Vec3 grad_orig = compute_grad_angle_A(body_A, body_B, rot_A, center_A, rot_B, center_B);
        Vec3 grad_perturbed = compute_grad_angle_A(body_A, body_B, rot_perturbed, center_A, rot_B, center_B);

        Vec3 diff = grad_perturbed - grad_orig;
        INFO(fmt::format("Gradient change for 0.01 rad rotation: |dg| = {:.6e}", diff.norm()));
        // Gradient should change smoothly (not jump discontinuously)
        REQUIRE(diff.norm() < 0.1);  // Less than 0.1 kJ/mol/rad change per 0.01 rad
    }
}

TEST_CASE("Mixed translation-rotation Hessian", "[hessian][mixed]") {
    // NOTE: The mixed Hessian d²E/dxi dθj = d²E/dθj dxi should be symmetric
    // because translation and rotation commute as operations.

    Mult m1(2), m2(2);
    m1.Q00() = -0.5;
    m1.Q10() = 1.0;
    m2.Q00() = 0.5;
    m2.Q10() = -0.8;

    Vec3 body_offset(0, 0, 0);
    std::vector<std::pair<Mult, Vec3>> body_A = {{m1, body_offset}};
    std::vector<std::pair<Mult, Vec3>> body_B = {{m2, body_offset}};

    Mat3 rot_A = axis_angle_rotation(Vec3(0, 1, 0), 0.3);
    Mat3 rot_B = Mat3::Identity();
    Vec3 center_A(0, 0, 0);
    Vec3 center_B(5, 2, -1);

    // d²E/dxA_i dθA_j via FD (perturb rotation in lab frame)
    const double h = 1e-5;
    Mat3 hess_x_theta;  // d(grad_x) / dθ

    for (int j = 0; j < 3; ++j) {
        Vec3 axis = Vec3::Zero();
        axis[j] = 1.0;
        Mat3 dR_plus = axis_angle_rotation(axis, h);
        Mat3 dR_minus = axis_angle_rotation(axis, -h);
        Mat3 rot_plus = dR_plus * rot_A;    // Left multiplication = lab frame
        Mat3 rot_minus = dR_minus * rot_A;

        Vec3 grad_x_plus = compute_grad_center_A(body_A, body_B, rot_plus, center_A, rot_B, center_B);
        Vec3 grad_x_minus = compute_grad_center_A(body_A, body_B, rot_minus, center_A, rot_B, center_B);

        for (int i = 0; i < 3; ++i) {
            hess_x_theta(i, j) = (grad_x_plus[i] - grad_x_minus[i]) / (2.0 * h);
        }
    }

    // d²E/dθA_i dxA_j via FD
    Mat3 hess_theta_x;  // d(grad_θ) / dx

    for (int j = 0; j < 3; ++j) {
        Vec3 center_plus = center_A;
        Vec3 center_minus = center_A;
        center_plus[j] += h;
        center_minus[j] -= h;

        Vec3 grad_theta_plus = compute_grad_angle_A(body_A, body_B, rot_A, center_plus, rot_B, center_B);
        Vec3 grad_theta_minus = compute_grad_angle_A(body_A, body_B, rot_A, center_minus, rot_B, center_B);

        for (int i = 0; i < 3; ++i) {
            hess_theta_x(i, j) = (grad_theta_plus[i] - grad_theta_minus[i]) / (2.0 * h);
        }
    }

    SECTION("Cross Hessians are transposes of each other") {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                INFO(fmt::format("Hx_theta({},{})={:.10e}  Htheta_x({},{})={:.10e}",
                                i, j, hess_x_theta(i, j), j, i, hess_theta_x(j, i)));
                double diff = std::abs(hess_x_theta(i, j) - hess_theta_x(j, i));
                double avg = 0.5 * (std::abs(hess_x_theta(i, j)) + std::abs(hess_theta_x(j, i)));
                if (avg > 1e-10) {
                    REQUIRE(diff / avg < 0.1);  // Within 10%
                }
            }
        }
    }
}

TEST_CASE("Hessian eigenvalues are real", "[hessian][eigenvalues]") {
    // For a stable equilibrium, eigenvalues should be positive
    // For a saddle point or unstable, can have negative eigenvalues
    // But they should always be REAL for a properly computed Hessian

    Mult m1(2), m2(2);
    m1.Q00() = -0.5;
    m1.Q10() = 0.5;
    m2.Q00() = 0.5;

    Vec3 body_offset(0, 0, 0);
    std::vector<std::pair<Mult, Vec3>> body_A = {{m1, body_offset}};
    std::vector<std::pair<Mult, Vec3>> body_B = {{m2, body_offset}};

    Mat3 rot_A = Mat3::Identity();
    Mat3 rot_B = Mat3::Identity();
    Vec3 center_A(0, 0, 0);
    Vec3 center_B(4, 0, 0);

    // Build 6x6 Hessian for molecule A (3 translation + 3 rotation)
    const double h = 1e-5;
    Mat6 full_hessian = Mat6::Zero();

    auto compute_full_grad = [&](const Vec3& center, const Mat3& rot) -> Vec6 {
        auto molA = build_molecule(body_A, rot, center);
        auto molB = build_molecule(body_B, rot_B, center_B);
        auto result = compute_molecule_forces_torques(molA, molB);
        Vec6 grad;
        grad.head<3>() = -result.force_A;
        grad.tail<3>() = result.grad_angle_axis_A;
        return grad;
    };

    Vec6 grad_ref = compute_full_grad(center_A, rot_A);

    // Translation columns
    for (int j = 0; j < 3; ++j) {
        Vec3 c_plus = center_A, c_minus = center_A;
        c_plus[j] += h;
        c_minus[j] -= h;
        Vec6 g_plus = compute_full_grad(c_plus, rot_A);
        Vec6 g_minus = compute_full_grad(c_minus, rot_A);
        full_hessian.col(j) = (g_plus - g_minus) / (2.0 * h);
    }

    // Rotation columns - use lab-frame rotations (left multiplication)
    for (int j = 0; j < 3; ++j) {
        Vec3 axis = Vec3::Zero();
        axis[j] = 1.0;
        Mat3 dR_plus = axis_angle_rotation(axis, h);
        Mat3 dR_minus = axis_angle_rotation(axis, -h);
        Vec6 g_plus = compute_full_grad(center_A, dR_plus * rot_A);
        Vec6 g_minus = compute_full_grad(center_A, dR_minus * rot_A);
        full_hessian.col(3 + j) = (g_plus - g_minus) / (2.0 * h);
    }

    // Symmetrize (finite difference introduces small asymmetry)
    Mat6 hess_sym = 0.5 * (full_hessian + full_hessian.transpose());

    // Check eigenvalues
    Eigen::SelfAdjointEigenSolver<Mat6> es(hess_sym);
    Vec6 eigenvalues = es.eigenvalues();

    INFO(fmt::format("Eigenvalues: {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}",
                     eigenvalues[0], eigenvalues[1], eigenvalues[2],
                     eigenvalues[3], eigenvalues[4], eigenvalues[5]));

    SECTION("All eigenvalues are real (solver succeeded)") {
        REQUIRE(es.info() == Eigen::Success);
    }

    SECTION("Eigenvalues are finite") {
        for (int i = 0; i < 6; ++i) {
            REQUIRE(std::isfinite(eigenvalues[i]));
        }
    }
}

// ============================================================================
// Test: Gradient consistency check (validates 1st derivs work for Hessian)
// ============================================================================

TEST_CASE("Gradient is smooth enough for Hessian", "[hessian][smoothness]") {
    // If gradients are smooth, the Hessian (2nd FD of gradient) should be well-behaved
    Mult m1(4), m2(4);
    m1.Q00() = -0.669;
    m1.Q10() = 0.234;
    m1.Q20() = -0.123;
    m1.Q30() = 0.05;
    m2.Q00() = 0.5;
    m2.Q11c() = 0.1;
    m2.Q22c() = 0.05;

    Vec3 body_offset(0, 0, 0);
    std::vector<std::pair<Mult, Vec3>> body_A = {{m1, body_offset}};
    std::vector<std::pair<Mult, Vec3>> body_B = {{m2, body_offset}};

    Mat3 rot_A = axis_angle_rotation(Vec3(1, 0, 0), 0.2);
    Mat3 rot_B = axis_angle_rotation(Vec3(0, 1, 0), -0.3);
    Vec3 center_A(0, 0, 0);
    Vec3 center_B(5, 2, -1);

    // Check gradient at multiple h values for smoothness
    std::vector<double> h_values = {1e-3, 1e-4, 1e-5, 1e-6, 1e-7};
    std::vector<Mat3> hessians;

    for (double h : h_values) {
        Mat3 hess;
        for (int j = 0; j < 3; ++j) {
            Vec3 center_plus = center_A;
            Vec3 center_minus = center_A;
            center_plus[j] += h;
            center_minus[j] -= h;

            Vec3 grad_plus = compute_grad_center_A(body_A, body_B, rot_A, center_plus, rot_B, center_B);
            Vec3 grad_minus = compute_grad_center_A(body_A, body_B, rot_A, center_minus, rot_B, center_B);

            for (int i = 0; i < 3; ++i) {
                hess(i, j) = (grad_plus[i] - grad_minus[i]) / (2.0 * h);
            }
        }
        hessians.push_back(hess);
    }

    SECTION("Hessian converges as h decreases") {
        // The Hessian should converge as h gets smaller (up to numerical precision)
        // Compare h=1e-4 to h=1e-5 and h=1e-5 to h=1e-6
        double diff_1 = (hessians[1] - hessians[2]).norm();
        double diff_2 = (hessians[2] - hessians[3]).norm();

        INFO(fmt::format("Hessian diff (1e-4 vs 1e-5): {:.6e}", diff_1));
        INFO(fmt::format("Hessian diff (1e-5 vs 1e-6): {:.6e}", diff_2));

        // Both differences should be small
        REQUIRE(diff_1 < 1e-3);
        REQUIRE(diff_2 < 1e-4);
    }
}

// ============================================================================
// Test: Analytical Charge-Charge Hessian
// ============================================================================

TEST_CASE("Analytical charge-charge Hessian vs numerical", "[hessian][analytical][charge]") {
    // Two point charges
    double qA = -0.5;  // Negative charge
    double qB = 0.8;   // Positive charge
    Vec3 posA(0, 0, 0);
    Vec3 posB(4, 2, -1);  // ~4.6 Angstrom separation

    // Compute analytical Hessian
    auto result = compute_charge_charge_hessian(posA, qA, posB, qB);

    // Numerical Hessian via energy finite differences
    const double h = 1e-5;

    auto energy_func = [&](const Vec3& pA, const Vec3& pB) {
        Vec3 R = pB - pA;
        double r = R.norm();
        return qA * qB / r;  // Coulomb law (in whatever units)
    };

    // Reference energy
    double E0 = energy_func(posA, posB);
    INFO(fmt::format("Analytical energy: {:.10e}, Expected: {:.10e}",
                     result.energy, E0));
    REQUIRE(result.energy == Approx(E0).epsilon(1e-10));

    // Numerical gradient at A
    Vec3 grad_A_num;
    for (int i = 0; i < 3; ++i) {
        Vec3 pA_plus = posA, pA_minus = posA;
        pA_plus[i] += h;
        pA_minus[i] -= h;
        grad_A_num[i] = (energy_func(pA_plus, posB) - energy_func(pA_minus, posB)) / (2.0 * h);
    }

    // Analytical gradient (force_A = -grad_posA for electrostatics)
    Vec3 grad_A_ana = -result.force_A;

    INFO(fmt::format("Gradient A - Analytical: ({:.8e}, {:.8e}, {:.8e})",
                     grad_A_ana[0], grad_A_ana[1], grad_A_ana[2]));
    INFO(fmt::format("Gradient A - Numerical:  ({:.8e}, {:.8e}, {:.8e})",
                     grad_A_num[0], grad_A_num[1], grad_A_num[2]));

    SECTION("Analytical gradient matches numerical") {
        for (int i = 0; i < 3; ++i) {
            REQUIRE(grad_A_ana[i] == Approx(grad_A_num[i]).epsilon(1e-6));
        }
    }

    // Numerical Hessian ∂²E/∂posA_i ∂posA_j
    Mat3 hess_AA_num;
    for (int j = 0; j < 3; ++j) {
        Vec3 pA_plus = posA, pA_minus = posA;
        pA_plus[j] += h;
        pA_minus[j] -= h;

        // Gradient at perturbed points
        Vec3 grad_plus, grad_minus;
        for (int i = 0; i < 3; ++i) {
            Vec3 pp_plus = pA_plus, pp_minus = pA_plus;
            Vec3 pm_plus = pA_minus, pm_minus = pA_minus;
            pp_plus[i] += h;
            pp_minus[i] -= h;
            pm_plus[i] += h;
            pm_minus[i] -= h;

            grad_plus[i] = (energy_func(pp_plus, posB) - energy_func(pp_minus, posB)) / (2.0 * h);
            grad_minus[i] = (energy_func(pm_plus, posB) - energy_func(pm_minus, posB)) / (2.0 * h);
        }
        hess_AA_num.col(j) = (grad_plus - grad_minus) / (2.0 * h);
    }

    INFO(fmt::format("Analytical H_posA_posA:\n{:.8e} {:.8e} {:.8e}\n{:.8e} {:.8e} {:.8e}\n{:.8e} {:.8e} {:.8e}",
                     result.H_posA_posA(0,0), result.H_posA_posA(0,1), result.H_posA_posA(0,2),
                     result.H_posA_posA(1,0), result.H_posA_posA(1,1), result.H_posA_posA(1,2),
                     result.H_posA_posA(2,0), result.H_posA_posA(2,1), result.H_posA_posA(2,2)));
    INFO(fmt::format("Numerical H_posA_posA:\n{:.8e} {:.8e} {:.8e}\n{:.8e} {:.8e} {:.8e}\n{:.8e} {:.8e} {:.8e}",
                     hess_AA_num(0,0), hess_AA_num(0,1), hess_AA_num(0,2),
                     hess_AA_num(1,0), hess_AA_num(1,1), hess_AA_num(1,2),
                     hess_AA_num(2,0), hess_AA_num(2,1), hess_AA_num(2,2)));

    SECTION("Analytical Hessian matches numerical") {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(result.H_posA_posA(i,j) == Approx(hess_AA_num(i,j)).epsilon(1e-4));
            }
        }
    }

    SECTION("Position Hessians are symmetric") {
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                REQUIRE(result.H_posA_posA(i,j) == Approx(result.H_posA_posA(j,i)).epsilon(1e-12));
            }
        }
    }

    SECTION("Cross Hessians have correct relationship") {
        // H_posA_posB = -H_posA_posA for two-body potential
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(result.H_posA_posB(i,j) == Approx(-result.H_posA_posA(i,j)).epsilon(1e-12));
            }
        }
        // H_posB_posB = H_posA_posA
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                REQUIRE(result.H_posB_posB(i,j) == Approx(result.H_posA_posA(i,j)).epsilon(1e-12));
            }
        }
    }
}

// ============================================================================
// Test: Rotation Matrix Second Derivatives
// ============================================================================

TEST_CASE("Rotation matrix second derivative", "[hessian][rotation_derivative]") {
    // Test that ∂²M/∂p_k∂p_l is computed correctly via numerical FD

    Vec3 angle_axis(0.3, -0.2, 0.5);  // Non-trivial rotation

    // Build a RigidBodyState to get the derivatives
    // For this test, we'll use numerical differentiation as reference

    auto rotation_from_aa = [](const Vec3& p) -> Mat3 {
        double angle = p.norm();
        if (angle < 1e-12) {
            return Mat3::Identity();
        }
        Vec3 axis = p / angle;
        return Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    };

    auto first_deriv = [&](const Vec3& p, int k) -> Mat3 {
        const double h = 1e-6;
        Vec3 p_plus = p, p_minus = p;
        p_plus[k] += h;
        p_minus[k] -= h;
        return (rotation_from_aa(p_plus) - rotation_from_aa(p_minus)) / (2.0 * h);
    };

    auto second_deriv = [&](const Vec3& p, int k, int l) -> Mat3 {
        const double h = 1e-5;
        Vec3 p_plus = p, p_minus = p;
        p_plus[l] += h;
        p_minus[l] -= h;
        return (first_deriv(p_plus, k) - first_deriv(p_minus, k)) / (2.0 * h);
    };

    // Use RigidBodyState to get analytical second derivative
    occ::mults::RigidBodyState state;
    state.set_angle_axis(angle_axis);
    auto M2_analytical = state.rotation_matrix_second_derivatives();

    SECTION("Second derivatives match numerical") {
        for (int k = 0; k < 3; ++k) {
            for (int l = 0; l < 3; ++l) {
                Mat3 M2_num = second_deriv(angle_axis, k, l);
                Mat3 M2_ana = M2_analytical[3 * k + l];

                double diff = (M2_num - M2_ana).norm();
                double mag = M2_num.norm();

                INFO(fmt::format("d²M/dp_{} dp_{}: num_norm={:.6e}, ana_norm={:.6e}, diff={:.6e}",
                                k, l, mag, M2_ana.norm(), diff));

                if (mag > 1e-8) {
                    REQUIRE(diff / mag < 0.01);  // Within 1% relative error
                } else {
                    REQUIRE(diff < 1e-6);  // Absolute error for small values
                }
            }
        }
    }
}

// ============================================================================
// Test: Charge-Dipole Hessian Position Block
// ============================================================================

TEST_CASE("Charge-dipole position Hessian", "[hessian][analytical][dipole]") {
    // Charge at A, dipole at B
    double qA = 0.5;
    Vec3 posA(0, 0, 0);
    Vec3 posB(4, 1, -1);

    // Lab-frame dipole (fixed orientation for this test)
    Vec3 dipole_B(0.3, -0.2, 0.1);
    Vec3 body_dipole_B = dipole_B;  // No rotation
    Mat3 M = Mat3::Identity();
    std::array<Mat3, 3> dM = {Mat3::Zero(), Mat3::Zero(), Mat3::Zero()};

    // Compute analytical
    auto result = compute_charge_dipole_hessian(posA, qA, posB, dipole_B,
                                                 body_dipole_B, M, dM, nullptr);

    // Numerical energy function
    auto energy_func = [&](const Vec3& pA, const Vec3& pB) {
        Vec3 R = pB - pA;
        double r = R.norm();
        // E = q_A * T^(1)_j * μ_B^j = -q_A * R_j / r³ * μ_B^j
        double T1x = -R[0] / (r * r * r);
        double T1y = -R[1] / (r * r * r);
        double T1z = -R[2] / (r * r * r);
        return qA * (T1x * dipole_B[0] + T1y * dipole_B[1] + T1z * dipole_B[2]);
    };

    double E0 = energy_func(posA, posB);
    INFO(fmt::format("Analytical energy: {:.10e}, Numerical: {:.10e}",
                     result.energy, E0));
    REQUIRE(result.energy == Approx(E0).epsilon(1e-8));

    // Numerical Hessian
    const double h = 1e-5;
    Mat3 hess_AA_num;

    for (int j = 0; j < 3; ++j) {
        for (int i = 0; i < 3; ++i) {
            Vec3 pp = posA, pm = posA, mp = posA, mm = posA;
            pp[i] += h; pp[j] += h;
            pm[i] += h; pm[j] -= h;
            mp[i] -= h; mp[j] += h;
            mm[i] -= h; mm[j] -= h;

            hess_AA_num(i, j) = (energy_func(pp, posB) - energy_func(pm, posB)
                               - energy_func(mp, posB) + energy_func(mm, posB)) / (4.0 * h * h);
        }
    }

    SECTION("Position Hessian matches numerical") {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                INFO(fmt::format("H({},{}) analytical={:.8e}, numerical={:.8e}",
                                i, j, result.H_posA_posA(i,j), hess_AA_num(i,j)));
                REQUIRE(result.H_posA_posA(i,j) == Approx(hess_AA_num(i,j)).epsilon(1e-4));
            }
        }
    }
}
