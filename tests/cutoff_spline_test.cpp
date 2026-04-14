#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/cartesian_force.h>
#include <occ/mults/cartesian_hessian.h>
#include <occ/mults/cutoff_spline.h>
#include <algorithm>
#include <cmath>

using Catch::Approx;
using namespace occ;
using namespace occ::mults;

static CartesianMolecule one_charge_molecule(double q, const Vec3& center) {
    dma::Mult m(0);
    m.Q00() = q;
    std::vector<std::pair<dma::Mult, Vec3>> body_sites = {{m, Vec3::Zero()}};
    return CartesianMolecule::from_body_frame_with_rotation(
        body_sites, Mat3::Identity(), center);
}

static Mat3 axis_angle_rotation(const Vec3& axis, double angle) {
    Vec3 n = axis.normalized();
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    Mat3 K;
    K << 0.0, -n[2], n[1],
         n[2], 0.0, -n[0],
        -n[1], n[0], 0.0;
    return Mat3::Identity() + s * K + (1.0 - c) * K * K;
}

static CartesianMolecule build_molecule(
    const std::vector<std::pair<dma::Mult, Vec3>>& body_sites,
    const Mat3& rotation, const Vec3& center) {
    return CartesianMolecule::from_body_frame_with_rotation(
        body_sites, rotation, center);
}

static Vec full_pair_gradient(
    const std::vector<std::pair<dma::Mult, Vec3>>& body_A,
    const std::vector<std::pair<dma::Mult, Vec3>>& body_B,
    const Mat3& rot_A, const Vec3& center_A,
    const Mat3& rot_B, const Vec3& center_B,
    const CutoffSpline& taper,
    double site_cutoff = 0.0,
    int max_order = -1) {

    auto molA = build_molecule(body_A, rot_A, center_A);
    auto molB = build_molecule(body_B, rot_B, center_B);
    auto fr = compute_molecule_forces_torques(
        molA, molB, site_cutoff, max_order, Vec3::Zero(), &taper);

    Vec g(12);
    g.segment<3>(0) = -fr.force_A;
    g.segment<3>(3) = fr.grad_angle_axis_A;
    g.segment<3>(6) = -fr.force_B;
    g.segment<3>(9) = fr.grad_angle_axis_B;
    return g;
}

TEST_CASE("Rank4+order-cutoff+taper Hessian remains analytic",
          "[mults][cutoff][hessian][rank4]") {
    dma::Mult a0(4), a1(4), b0(4), b1(4);
    a0.Q00() = -0.47; a0.Q10() = 0.22; a0.Q11c() = -0.09; a0.Q20() = 0.06; a0.Q22s() = 0.03; a0.Q30() = -0.02; a0.Q33c() = 0.015; a0.Q40() = -0.008;
    a1.Q00() = 0.31;  a1.Q10() = -0.18; a1.Q11s() = 0.07; a1.Q21c() = -0.04; a1.Q22c() = 0.025; a1.Q31s() = 0.01; a1.Q32c() = -0.008; a1.Q44s() = 0.004;
    b0.Q00() = 0.42;  b0.Q10() = -0.20; b0.Q11s() = 0.11; b0.Q20() = -0.05; b0.Q21s() = 0.02; b0.Q30() = 0.018; b0.Q33s() = -0.014; b0.Q41c() = 0.006;
    b1.Q00() = -0.26; b1.Q10() = 0.15;  b1.Q11c() = -0.06; b1.Q22s() = 0.03; b1.Q31c() = -0.012; b1.Q32s() = 0.009; b1.Q42c() = -0.005; b1.Q44c() = 0.003;

    std::vector<std::pair<dma::Mult, Vec3>> body_A = {
        {a0, Vec3(0.30, -0.24, 0.12)},
        {a1, Vec3(-0.27, 0.17, -0.15)}
    };
    std::vector<std::pair<dma::Mult, Vec3>> body_B = {
        {b0, Vec3(-0.29, 0.19, 0.10)},
        {b1, Vec3(0.24, -0.22, -0.18)}
    };

    const Mat3 rot_A = axis_angle_rotation(Vec3(0.5, 0.4, -0.3).normalized(), 0.33);
    const Mat3 rot_B = axis_angle_rotation(Vec3(-0.6, 0.2, 0.7).normalized(), -0.27);
    const Vec3 center_A(-0.35, 0.18, -0.12);
    const Vec3 center_B(2.95, -0.36, 0.63);

    CutoffSpline taper;
    taper.enabled = true;
    taper.r_on = 2.0;
    taper.r_off = 4.0;
    taper.order = 3;
    const double site_cutoff = 4.0;
    const int max_order = 4;

    auto molA = build_molecule(body_A, rot_A, center_A);
    auto molB = build_molecule(body_B, rot_B, center_B);
    auto ana = compute_molecule_hessian_truncated(
        molA, molB, Vec3::Zero(), site_cutoff, max_order, &taper);
    Mat H_ana = ana.pack_full_hessian();

    const double h = 2e-5;
    Mat H_num = Mat::Zero(12, 12);
    auto gradient_at = [&](const Vec& x) {
        Vec3 cA = center_A + x.segment<3>(0);
        Vec3 cB = center_B + x.segment<3>(6);
        Mat3 rA = rot_A;
        Mat3 rB = rot_B;
        for (int k = 0; k < 3; ++k) {
            Vec3 axis = Vec3::Zero();
            axis[k] = 1.0;
            rA = axis_angle_rotation(axis, x[3 + k]) * rA;
            rB = axis_angle_rotation(axis, x[9 + k]) * rB;
        }
        return full_pair_gradient(body_A, body_B, rA, cA, rB, cB, taper,
                                  site_cutoff, max_order);
    };

    for (int j = 0; j < 12; ++j) {
        Vec xp = Vec::Zero(12), xm = Vec::Zero(12);
        xp[j] = h;
        xm[j] = -h;
        H_num.col(j) = (gradient_at(xp) - gradient_at(xm)) / (2.0 * h);
    }
    Mat H_num_sym = 0.5 * (H_num + H_num.transpose());
    const double denom = std::max(1e-10, H_num_sym.norm());
    const double rel = (H_ana - H_num_sym).norm() / denom;
    INFO("rank4 tapered/order-cutoff relative Hessian error = " << rel);
    CHECK(rel < 5e-2);
}

TEST_CASE("DMACRYS cubic spline cutoff basics", "[mults][cutoff][spli]") {
    auto below = evaluate_cutoff_spline(1.5, 2.0, 4.0, 3);
    CHECK(below.value == Approx(1.0));
    CHECK(below.first_derivative == Approx(0.0));
    CHECK(below.second_derivative == Approx(0.0));

    auto at_on = evaluate_cutoff_spline(2.0, 2.0, 4.0, 3);
    CHECK(at_on.value == Approx(1.0));
    CHECK(at_on.first_derivative == Approx(0.0).margin(1e-12));

    auto mid = evaluate_cutoff_spline(3.0, 2.0, 4.0, 3);
    CHECK(mid.value == Approx(0.5).margin(1e-12));
    CHECK(mid.first_derivative == Approx(-0.75).margin(1e-12));

    auto at_off = evaluate_cutoff_spline(4.0, 2.0, 4.0, 3);
    CHECK(at_off.value == Approx(0.0).margin(1e-12));
    CHECK(at_off.first_derivative == Approx(0.0).margin(1e-12));
}

TEST_CASE("Spline taper applies product-rule force term", "[mults][cutoff][force]") {
    auto molA = one_charge_molecule(+1.0, Vec3(0.0, 0.0, 0.0));
    auto molB = one_charge_molecule(+1.0, Vec3(3.0, 0.0, 0.0));

    auto ref = compute_molecule_forces_torques(molA, molB);

    CutoffSpline taper;
    taper.enabled = true;
    taper.r_on = 2.0;
    taper.r_off = 4.0;
    taper.order = 3;

    auto tap = compute_molecule_forces_torques(molA, molB, 0.0, -1, Vec3::Zero(), &taper);

    auto sw = evaluate_cutoff_spline(3.0, taper);
    REQUIRE(sw.value > 0.0);

    CHECK(tap.energy == Approx(sw.value * ref.energy).epsilon(1e-12));

    Vec3 expected_force = sw.value * ref.force_A;
    expected_force += (ref.energy * sw.first_derivative) * Vec3::UnitX();

    CHECK(tap.force_A[0] == Approx(expected_force[0]).epsilon(1e-11));
    CHECK(tap.force_A[1] == Approx(expected_force[1]).margin(1e-12));
    CHECK(tap.force_A[2] == Approx(expected_force[2]).margin(1e-12));
}

TEST_CASE("Truncated electrostatic Hessian honors B offset", "[mults][hessian][offset]") {
    auto molA = one_charge_molecule(+1.0, Vec3(0.0, 0.0, 0.0));
    auto molB = one_charge_molecule(-1.0, Vec3(0.7, -0.2, 0.4));
    Vec3 offset(2.5, -0.3, 0.8);

    auto h_off = compute_molecule_hessian_truncated(molA, molB, offset);

    auto molB_shifted = one_charge_molecule(-1.0, Vec3(0.7, -0.2, 0.4) + offset);
    auto h_abs = compute_molecule_hessian_truncated(molA, molB_shifted);

    CHECK(h_off.energy == Approx(h_abs.energy).epsilon(1e-12));
    CHECK((h_off.force_A - h_abs.force_A).norm() == Approx(0.0).margin(1e-12));
    CHECK((h_off.H_posA_posA - h_abs.H_posA_posA).norm() == Approx(0.0).margin(1e-12));
    CHECK((h_off.H_posA_posB - h_abs.H_posA_posB).norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("Spline taper keeps full rigid-body Hessian analytic", "[mults][cutoff][hessian]") {
    dma::Mult a0(3), a1(2), b0(3), b1(2);
    a0.Q00() = -0.60; a0.Q10() = 0.35; a0.Q11c() = -0.12; a0.Q20() = 0.08; a0.Q21s() = 0.03;
    a1.Q00() = 0.25;  a1.Q10() = -0.20; a1.Q11s() = 0.09; a1.Q22c() = -0.04;
    b0.Q00() = 0.55;  b0.Q10() = -0.28; b0.Q11s() = 0.15; b0.Q20() = -0.06; b0.Q21c() = 0.02;
    b1.Q00() = -0.22; b1.Q10() = 0.18;  b1.Q11c() = -0.07; b1.Q22s() = 0.05;

    std::vector<std::pair<dma::Mult, Vec3>> body_A = {
        {a0, Vec3(0.35, -0.22, 0.14)},
        {a1, Vec3(-0.28, 0.19, -0.11)}
    };
    std::vector<std::pair<dma::Mult, Vec3>> body_B = {
        {b0, Vec3(-0.31, 0.17, 0.09)},
        {b1, Vec3(0.26, -0.21, -0.16)}
    };

    const Mat3 rot_A = axis_angle_rotation(Vec3(0.4, 0.7, -0.2).normalized(), 0.37);
    const Mat3 rot_B = axis_angle_rotation(Vec3(-0.5, 0.1, 0.8).normalized(), -0.29);
    const Vec3 center_A(-0.4, 0.2, -0.1);
    const Vec3 center_B(2.9, -0.4, 0.6); // in taper region (2 < r < 4)

    CutoffSpline taper;
    taper.enabled = true;
    taper.r_on = 2.0;
    taper.r_off = 4.0;
    taper.order = 3;

    auto molA = build_molecule(body_A, rot_A, center_A);
    auto molB = build_molecule(body_B, rot_B, center_B);
    auto ana = compute_molecule_hessian_truncated(
        molA, molB, Vec3::Zero(), 0.0, -1, &taper);
    Mat H_ana = ana.pack_full_hessian();
    Mat H_ana_no_taper = compute_molecule_hessian_truncated(
        molA, molB, Vec3::Zero(), 0.0, -1, nullptr).pack_full_hessian();

    const double h = 2e-5;
    Mat H_num = Mat::Zero(12, 12);
    auto gradient_at = [&](const Vec& x) {
        Vec3 cA = center_A + x.segment<3>(0);
        Vec3 cB = center_B + x.segment<3>(6);
        Mat3 rA = rot_A;
        Mat3 rB = rot_B;
        for (int k = 0; k < 3; ++k) {
            Vec3 axis = Vec3::Zero();
            axis[k] = 1.0;
            rA = axis_angle_rotation(axis, x[3 + k]) * rA;
            rB = axis_angle_rotation(axis, x[9 + k]) * rB;
        }
        return full_pair_gradient(body_A, body_B, rA, cA, rB, cB, taper);
    };

    for (int j = 0; j < 12; ++j) {
        Vec xp = Vec::Zero(12), xm = Vec::Zero(12);
        xp[j] = h;
        xm[j] = -h;
        H_num.col(j) = (gradient_at(xp) - gradient_at(xm)) / (2.0 * h);
    }

    Mat H_num_sym = 0.5 * (H_num + H_num.transpose());
    const double denom = std::max(1e-10, H_num_sym.norm());
    const double rel = (H_ana - H_num_sym).norm() / denom;
    const double rel_no_taper = (H_ana_no_taper - H_num_sym).norm() / denom;
    INFO("relative Hessian error = " << rel);
    INFO("relative Hessian error (analytic no-taper vs tapered FD) = " << rel_no_taper);
    INFO("pos-pos block rel = "
         << (H_ana.block<3,3>(0,0) - H_num_sym.block<3,3>(0,0)).norm() /
                std::max(1e-10, H_num_sym.block<3,3>(0,0).norm()));
    INFO("pos-rot(A) block rel = "
         << (H_ana.block<3,3>(0,3) - H_num_sym.block<3,3>(0,3)).norm() /
                std::max(1e-10, H_num_sym.block<3,3>(0,3).norm()));
    INFO("rot-rot(A) block rel = "
         << (H_ana.block<3,3>(3,3) - H_num_sym.block<3,3>(3,3)).norm() /
                std::max(1e-10, H_num_sym.block<3,3>(3,3).norm()));
    INFO("rot-rot(AB) block rel = "
         << (H_ana.block<3,3>(3,9) - H_num_sym.block<3,3>(3,9)).norm() /
                std::max(1e-10, H_num_sym.block<3,3>(3,9).norm()));
    CHECK(rel < 5e-2);
}
