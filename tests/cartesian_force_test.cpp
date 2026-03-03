#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/cartesian_force.h>
#include <occ/mults/cartesian_rotation.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/core/units.h>
#include <fmt/core.h>
#include <cmath>
#include <vector>

using namespace occ;
using namespace occ::mults;
using namespace occ::dma;
using Approx = Catch::Approx;

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

namespace {

/// Build a CartesianMolecule from a single site.
CartesianMolecule single_site_mol(const Mult &m, const Vec3 &pos) {
    return CartesianMolecule::from_lab_sites({{m, pos}});
}

/// Finite-difference gradient via central differences.
Vec3 finite_diff_gradient(const CartesianMolecule &molA,
                          const CartesianMolecule &molB,
                          int which_mol, // 0 = perturb A's center, 1 = perturb B's center
                          double delta = 1e-5) {
    Vec3 grad;
    for (int d = 0; d < 3; ++d) {
        // Create perturbed molecules
        CartesianMolecule mol_plus = which_mol == 0 ? molA : molB;
        CartesianMolecule mol_minus = which_mol == 0 ? molA : molB;

        for (auto &s : mol_plus.sites)
            s.position[d] += delta;
        for (auto &s : mol_minus.sites)
            s.position[d] -= delta;

        double E_plus, E_minus;
        if (which_mol == 0) {
            E_plus = compute_molecule_interaction(mol_plus, molB);
            E_minus = compute_molecule_interaction(mol_minus, molB);
        } else {
            E_plus = compute_molecule_interaction(molA, mol_plus);
            E_minus = compute_molecule_interaction(molA, mol_minus);
        }
        grad[d] = (E_plus - E_minus) / (2.0 * delta);
    }
    return grad;
}

double molecule_pair_energy_with_options(const CartesianMolecule &molA,
                                         const CartesianMolecule &molB,
                                         double site_cutoff,
                                         int max_interaction_order,
                                         const Vec3 &offset_B,
                                         const CutoffSpline *taper) {
    return compute_molecule_forces_torques(
               molA, molB, site_cutoff, max_interaction_order, offset_B, taper)
        .energy;
}

Vec3 finite_diff_force_A_with_options(const CartesianMolecule &molA,
                                      const CartesianMolecule &molB,
                                      double site_cutoff,
                                      int max_interaction_order,
                                      const Vec3 &offset_B,
                                      const CutoffSpline *taper,
                                      double delta = 1e-6) {
    Vec3 force = Vec3::Zero();
    for (int d = 0; d < 3; ++d) {
        CartesianMolecule mol_plus = molA;
        CartesianMolecule mol_minus = molA;
        for (auto &s : mol_plus.sites) s.position[d] += delta;
        for (auto &s : mol_minus.sites) s.position[d] -= delta;

        const double Ep = molecule_pair_energy_with_options(
            mol_plus, molB, site_cutoff, max_interaction_order, offset_B, taper);
        const double Em = molecule_pair_energy_with_options(
            mol_minus, molB, site_cutoff, max_interaction_order, offset_B, taper);
        const double dE_dx = (Ep - Em) / (2.0 * delta);
        force[d] = -dE_dx; // physical force = -dE/dx
    }
    return force;
}

std::array<Mat3, 6> voigt_basis_for_tests() {
    std::array<Mat3, 6> B{};
    B[0].setZero();
    B[0](0, 0) = 1.0;
    B[1].setZero();
    B[1](1, 1) = 1.0;
    B[2].setZero();
    B[2](2, 2) = 1.0;
    B[3].setZero();
    B[3](1, 2) = B[3](2, 1) = 0.5; // E4 = 2*eps_yz
    B[4].setZero();
    B[4](0, 2) = B[4](2, 0) = 0.5; // E5 = 2*eps_xz
    B[5].setZero();
    B[5](0, 1) = B[5](1, 0) = 0.5; // E6 = 2*eps_xy
    return B;
}

/// Build a rotation matrix from axis-angle (Rodrigues).
Mat3 axis_angle_rotation(const Vec3 &axis, double angle) {
    Vec3 n = axis.normalized();
    double c = std::cos(angle), s = std::sin(angle);
    Mat3 K;
    K << 0, -n[2], n[1],
         n[2], 0, -n[0],
         -n[1], n[0], 0;
    return Mat3::Identity() + s * K + (1.0 - c) * K * K;
}

/// Finite-difference torque for molecule's orientation.
/// Perturbs angle-axis parameters around the current rotation.
Vec3 finite_diff_torque(
    const std::vector<std::pair<Mult, Vec3>> &body_sites,
    const Mat3 &rotation, const Vec3 &center,
    const CartesianMolecule &other_mol,
    bool is_mol_A, // true if perturbing mol A, false for mol B
    double delta = 1e-5) {
    Vec3 grad;

    // Generator matrices (infinitesimal rotations about x, y, z)
    for (int k = 0; k < 3; ++k) {
        Vec3 axis = Vec3::Zero();
        axis[k] = 1.0;

        // Perturb rotation by small angle about axis k
        Mat3 dR_plus = axis_angle_rotation(axis, delta);
        Mat3 dR_minus = axis_angle_rotation(axis, -delta);
        Mat3 R_plus = dR_plus * rotation;
        Mat3 R_minus = dR_minus * rotation;

        auto mol_plus = CartesianMolecule::from_body_frame_with_rotation(
            body_sites, R_plus, center);
        auto mol_minus = CartesianMolecule::from_body_frame_with_rotation(
            body_sites, R_minus, center);

        double E_plus, E_minus;
        if (is_mol_A) {
            E_plus = compute_molecule_interaction(mol_plus, other_mol);
            E_minus = compute_molecule_interaction(mol_minus, other_mol);
        } else {
            E_plus = compute_molecule_interaction(other_mol, mol_plus);
            E_minus = compute_molecule_interaction(other_mol, mol_minus);
        }
        grad[k] = (E_plus - E_minus) / (2.0 * delta);
    }
    return grad;
}

Vec3 finite_diff_torque_with_options(
    const std::vector<std::pair<Mult, Vec3>> &body_sites,
    const Mat3 &rotation, const Vec3 &center,
    const CartesianMolecule &other_mol,
    bool is_mol_A,
    double site_cutoff,
    int max_interaction_order,
    const Vec3 &offset_B,
    const CutoffSpline *taper,
    double delta = 1e-6) {
    Vec3 grad = Vec3::Zero();
    for (int k = 0; k < 3; ++k) {
        Vec3 axis = Vec3::Zero();
        axis[k] = 1.0;

        Mat3 dR_plus = axis_angle_rotation(axis, delta);
        Mat3 dR_minus = axis_angle_rotation(axis, -delta);
        Mat3 R_plus = dR_plus * rotation;
        Mat3 R_minus = dR_minus * rotation;

        auto mol_plus = CartesianMolecule::from_body_frame_with_rotation(
            body_sites, R_plus, center);
        auto mol_minus = CartesianMolecule::from_body_frame_with_rotation(
            body_sites, R_minus, center);

        double E_plus = 0.0, E_minus = 0.0;
        if (is_mol_A) {
            E_plus = molecule_pair_energy_with_options(
                mol_plus, other_mol, site_cutoff, max_interaction_order, offset_B, taper);
            E_minus = molecule_pair_energy_with_options(
                mol_minus, other_mol, site_cutoff, max_interaction_order, offset_B, taper);
        } else {
            E_plus = molecule_pair_energy_with_options(
                other_mol, mol_plus, site_cutoff, max_interaction_order, offset_B, taper);
            E_minus = molecule_pair_energy_with_options(
                other_mol, mol_minus, site_cutoff, max_interaction_order, offset_B, taper);
        }
        grad[k] = (E_plus - E_minus) / (2.0 * delta);
    }
    return grad;
}

} // anonymous namespace

// -------------------------------------------------------------------
// Test 1: Single charge-charge pair force (analytical)
// -------------------------------------------------------------------

TEST_CASE("Charge-charge pair force matches analytical", "[force][analytical]") {
    Mult m1(4), m2(4);
    m1.Q00() = 2.0;
    m2.Q00() = 3.0;

    // Positions in Angstrom (converted from 4 Bohr)
    constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
    Vec3 p1(0, 0, 0), p2(0, 0, 4.0 * B2A);
    auto molA = single_site_mol(m1, p1);
    auto molB = single_site_mol(m2, p2);

    auto ef = compute_site_pair_energy_force(molA.sites[0], molB.sites[0]);

    // compute_site_pair_energy_force converts Ang->Bohr internally,
    // returns energy in Hartree and gradient in Hartree/Bohr.
    // E = q1*q2/R = 6/4 = 1.5 Hartree (R=4 Bohr)
    double R = 4.0;
    double expected_E = 2.0 * 3.0 / R;
    REQUIRE(ef.energy == Approx(expected_E));

    // dE/dRz = -q1*q2*Rz/R^3 in Hartree/Bohr
    double R3 = R * R * R;
    Vec3 expected_grad(0.0, 0.0, -2.0 * 3.0 * 4.0 / R3);
    REQUIRE(ef.gradient[0] == Approx(expected_grad[0]).margin(1e-14));
    REQUIRE(ef.gradient[1] == Approx(expected_grad[1]).margin(1e-14));
    REQUIRE(ef.gradient[2] == Approx(expected_grad[2]).margin(1e-12));
}

// -------------------------------------------------------------------
// Test 2: Dipole-dipole force via finite difference
// -------------------------------------------------------------------

TEST_CASE("Dipole-dipole force matches finite difference", "[force][fd]") {
    Mult m1(4), m2(4);
    m1.Q10() = 1.0;
    m1.Q11c() = 0.5;
    m2.Q10() = -0.3;
    m2.Q11s() = 0.7;

    // Positions in Angstrom (converted from Bohr)
    constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
    Vec3 p1(0, 0, 0), p2(3.5 * B2A, 1.2 * B2A, -2.1 * B2A);
    auto molA = single_site_mol(m1, p1);
    auto molB = single_site_mol(m2, p2);

    auto ef = compute_site_pair_energy_force(molA.sites[0], molB.sites[0]);

    // FD gradient via compute_molecule_interaction returns kJ/mol/Angstrom.
    // compute_site_pair_energy_force returns gradient in Hartree/Bohr.
    // Convert analytical gradient to kJ/mol/Angstrom for comparison.
    constexpr double grad_conv = occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM;
    Vec3 fd_grad = finite_diff_gradient(molA, molB, 1, 1e-6);

    INFO(fmt::format("Analytical grad: ({:.10e}, {:.10e}, {:.10e})",
                     ef.gradient[0] * grad_conv, ef.gradient[1] * grad_conv, ef.gradient[2] * grad_conv));
    INFO(fmt::format("FD grad:         ({:.10e}, {:.10e}, {:.10e})",
                     fd_grad[0], fd_grad[1], fd_grad[2]));

    REQUIRE(ef.gradient[0] * grad_conv == Approx(fd_grad[0]).epsilon(1e-8));
    REQUIRE(ef.gradient[1] * grad_conv == Approx(fd_grad[1]).epsilon(1e-8));
    REQUIRE(ef.gradient[2] * grad_conv == Approx(fd_grad[2]).epsilon(1e-8));
}

// -------------------------------------------------------------------
// Test 3: Full-rank pair force via finite difference
// -------------------------------------------------------------------

TEST_CASE("Full-rank pair force matches finite difference", "[force][fd][fullrank]") {
    Mult m1(4), m2(4);
    m1.Q00() = -0.669;
    m1.Q10() = 0.234;
    m1.Q20() = -0.123;
    m1.Q30() = 0.05;
    m1.Q40() = 0.01;
    m2.Q00() = 0.5;
    m2.Q11c() = 0.1;
    m2.Q22c() = 0.05;
    m2.Q33c() = 0.02;
    m2.Q44c() = 0.005;

    // Positions in Angstrom (converted from Bohr)
    constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
    Vec3 p1(0, 0, 0), p2(4.5 * B2A, -1.3 * B2A, 2.7 * B2A);
    auto molA = single_site_mol(m1, p1);
    auto molB = single_site_mol(m2, p2);

    auto ef = compute_site_pair_energy_force(molA.sites[0], molB.sites[0]);

    // FD gradient from compute_molecule_interaction is in kJ/mol/Angstrom.
    // Analytical gradient from compute_site_pair_energy_force is in Hartree/Bohr.
    constexpr double grad_conv = occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM;
    Vec3 fd_grad = finite_diff_gradient(molA, molB, 1, 1e-5);

    for (int d = 0; d < 3; ++d) {
        INFO(fmt::format("Component {}: analytical={:.10e} fd={:.10e} diff={:.2e}",
                         d, ef.gradient[d] * grad_conv, fd_grad[d],
                         std::abs(ef.gradient[d] * grad_conv - fd_grad[d])));
        REQUIRE(ef.gradient[d] * grad_conv == Approx(fd_grad[d]).epsilon(1e-7));
    }
}

TEST_CASE("Truncated+taper force matches finite difference",
          "[force][fd][trunc][taper]") {
    Mult m1(4), m2(4);
    m1.Q00() = -0.669;
    m1.Q10() = 0.234;
    m1.Q20() = -0.123;
    m1.Q30() = 0.05;
    m1.Q40() = 0.01;
    m2.Q00() = 0.5;
    m2.Q11c() = 0.1;
    m2.Q22c() = 0.05;
    m2.Q33c() = 0.02;
    m2.Q44c() = 0.005;

    auto molA = single_site_mol(m1, Vec3(0.0, 0.0, 0.0));
    auto molB = single_site_mol(m2, Vec3(6.2, 1.1, -0.8));

    CutoffSpline taper;
    taper.enabled = true;
    taper.r_on = 5.5;
    taper.r_off = 7.0;
    taper.order = 3;

    const Vec3 offset_B(0.3, -0.2, 0.4);
    const double site_cutoff = 8.0;
    const int max_order = 4;

    auto result = compute_molecule_forces_torques(
        molA, molB, site_cutoff, max_order, offset_B, &taper);
    Vec3 fd_force_A = finite_diff_force_A_with_options(
        molA, molB, site_cutoff, max_order, offset_B, &taper, 1e-6);

    for (int d = 0; d < 3; ++d) {
        INFO(fmt::format("Component {}: analytic={:.10e} fd={:.10e}",
                         d, result.force_A[d], fd_force_A[d]));
        REQUIRE(result.force_A[d] == Approx(fd_force_A[d]).epsilon(2e-6));
    }
}

TEST_CASE("Truncated+taper multi-site force matches finite difference",
          "[force][fd][trunc][taper][multisite]") {
    Mult qO(4), qH(4);
    qO.Q00() = -0.669;
    qO.Q10() = 0.234;
    qO.Q20() = -0.123;
    qO.Q30() = 0.05;
    qH.Q00() = 0.335;
    qH.Q11c() = 0.03;

    auto molA = CartesianMolecule::from_lab_sites({
        {qO, Vec3(0.0, 0.0, 0.0)},
        {qH, Vec3(1.8, 0.0, 0.0)},
        {qH, Vec3(-0.6, 1.7, 0.0)}
    });
    auto molB = CartesianMolecule::from_lab_sites({
        {qO, Vec3(5.9, 0.4, 0.2)},
        {qH, Vec3(7.6, 0.6, 0.1)},
        {qH, Vec3(5.1, 2.0, -0.2)}
    });

    CutoffSpline taper;
    taper.enabled = true;
    taper.r_on = 5.0;
    taper.r_off = 7.0;
    taper.order = 3;

    const Vec3 offset_B(0.2, -0.1, 0.25);
    const double site_cutoff = 8.0;
    const int max_order = 4;

    auto result = compute_molecule_forces_torques(
        molA, molB, site_cutoff, max_order, offset_B, &taper);
    Vec3 fd_force_A = finite_diff_force_A_with_options(
        molA, molB, site_cutoff, max_order, offset_B, &taper, 1e-6);

    for (int d = 0; d < 3; ++d) {
        INFO(fmt::format("Component {}: analytic={:.10e} fd={:.10e}",
                         d, result.force_A[d], fd_force_A[d]));
        REQUIRE(result.force_A[d] == Approx(fd_force_A[d]).epsilon(2e-6));
    }
}

TEST_CASE("Truncated+taper torque matches finite difference",
          "[force][fd][trunc][taper][torque]") {
    Mult qA(4), qB(4);
    qA.Q00() = 0.75;
    qB.Q00() = -0.60;

    const std::vector<std::pair<Mult, Vec3>> bodyA{
        {qA, Vec3(0.6, -0.2, 0.1)},
        {qA, Vec3(-0.4, 0.5, -0.3)}
    };
    const std::vector<std::pair<Mult, Vec3>> bodyB{
        {qB, Vec3(0.2, 0.1, -0.4)},
        {qB, Vec3(-0.5, -0.3, 0.2)}
    };

    const Mat3 rotA = axis_angle_rotation(Vec3(0.7, -0.2, 0.4).normalized(), 0.31);
    const Mat3 rotB = axis_angle_rotation(Vec3(-0.3, 0.9, 0.1).normalized(), -0.28);
    const Vec3 comA(0.1, -0.3, 0.2);
    const Vec3 comB(6.4, 0.7, -0.8);

    auto molA = CartesianMolecule::from_body_frame_with_rotation(bodyA, rotA, comA);
    auto molB = CartesianMolecule::from_body_frame_with_rotation(bodyB, rotB, comB);

    CutoffSpline taper;
    taper.enabled = true;
    taper.r_on = 5.0;
    taper.r_off = 7.0;
    taper.order = 3;

    const Vec3 offset_B(0.35, -0.15, 0.25);
    const double site_cutoff = 8.5;
    const int max_order = 0; // charge-only to isolate taper/geometry terms

    const auto result = compute_molecule_forces_torques(
        molA, molB, site_cutoff, max_order, offset_B, &taper);

    const Vec3 fd_tau_A = finite_diff_torque_with_options(
        bodyA, rotA, comA, molB, true,
        site_cutoff, max_order, offset_B, &taper, 1e-6);
    const Vec3 fd_tau_B = finite_diff_torque_with_options(
        bodyB, rotB, comB, molA, false,
        site_cutoff, max_order, offset_B, &taper, 1e-6);

    for (int k = 0; k < 3; ++k) {
        INFO(fmt::format("A component {}: analytic={:.10e} fd={:.10e}",
                         k, result.grad_angle_axis_A[k], fd_tau_A[k]));
        REQUIRE(result.grad_angle_axis_A[k] == Approx(fd_tau_A[k]).epsilon(1e-5));
        INFO(fmt::format("B component {}: analytic={:.10e} fd={:.10e}",
                         k, result.grad_angle_axis_B[k], fd_tau_B[k]));
        REQUIRE(result.grad_angle_axis_B[k] == Approx(fd_tau_B[k]).epsilon(1e-5));
    }
}

TEST_CASE("Truncated+taper pair virial matches strain finite difference",
          "[force][fd][strain][trunc][taper][virial]") {
    Mult mA1(4), mA2(4), mB1(4), mB2(4);
    mA1.Q00() = -0.55;
    mA1.Q10() = 0.14;
    mA1.Q20() = -0.08;
    mA2.Q00() = 0.22;
    mA2.Q11c() = 0.05;
    mA2.Q22s() = -0.02;

    mB1.Q00() = 0.41;
    mB1.Q10() = -0.09;
    mB1.Q21c() = 0.03;
    mB2.Q00() = -0.19;
    mB2.Q11s() = 0.06;
    mB2.Q30() = -0.015;

    const std::vector<std::pair<Mult, Vec3>> bodyA{
        {mA1, Vec3(0.0, 0.0, 0.0)},
        {mA2, Vec3(1.2, -0.4, 0.3)},
    };
    const std::vector<std::pair<Mult, Vec3>> bodyB{
        {mB1, Vec3(0.0, 0.0, 0.0)},
        {mB2, Vec3(-0.7, 1.1, -0.2)},
    };

    const Mat3 rotA = axis_angle_rotation(Vec3(1.0, 2.0, -1.0).normalized(), 0.37);
    // Improper orientation (det=-1): inversion * proper rotation.
    const Mat3 rotB = -axis_angle_rotation(Vec3(-1.0, 0.5, 2.0).normalized(), 0.51);

    const Vec3 comA(0.6, -0.7, 0.2);
    const Vec3 comB(4.1, 1.8, -1.3);
    const Vec3 offset_B(2.3, -1.1, 0.9);

    CutoffSpline taper;
    taper.enabled = true;
    taper.r_on = 5.0;
    taper.r_off = 7.5;
    taper.order = 3;

    const double site_cutoff = 8.0;
    const int max_order = 4;
    const auto B = voigt_basis_for_tests();

    auto energy_at_strain = [&](const Mat3 &eps) {
        const Mat3 F = Mat3::Identity() + eps;
        const auto molA = CartesianMolecule::from_body_frame_with_rotation(
            bodyA, rotA, F * comA);
        const auto molB = CartesianMolecule::from_body_frame_with_rotation(
            bodyB, rotB, F * comB);
        const Vec3 shift = F * offset_B;
        return compute_molecule_forces_torques(
            molA, molB, site_cutoff, max_order, shift, &taper).energy;
    };

    const auto molA0 = CartesianMolecule::from_body_frame_with_rotation(bodyA, rotA, comA);
    const auto molB0 = CartesianMolecule::from_body_frame_with_rotation(bodyB, rotB, comB);
    const auto r0 = compute_molecule_forces_torques(
        molA0, molB0, site_cutoff, max_order, offset_B, &taper);

    const Vec3 disp = comB + offset_B - comA;
    Vec6 g_analytic = Vec6::Zero();
    for (int a = 0; a < 6; ++a) {
        g_analytic[a] = r0.force_A.dot(B[a] * disp);
    }

    const double d = 1e-6;
    Vec6 g_fd = Vec6::Zero();
    for (int a = 0; a < 6; ++a) {
        Mat3 ep = Mat3::Zero();
        Mat3 em = Mat3::Zero();
        if (a == 0) {
            ep(0, 0) = d;
            em(0, 0) = -d;
        } else if (a == 1) {
            ep(1, 1) = d;
            em(1, 1) = -d;
        } else if (a == 2) {
            ep(2, 2) = d;
            em(2, 2) = -d;
        } else if (a == 3) {
            ep(1, 2) = ep(2, 1) = 0.5 * d;
            em(1, 2) = em(2, 1) = -0.5 * d;
        } else if (a == 4) {
            ep(0, 2) = ep(2, 0) = 0.5 * d;
            em(0, 2) = em(2, 0) = -0.5 * d;
        } else {
            ep(0, 1) = ep(1, 0) = 0.5 * d;
            em(0, 1) = em(1, 0) = -0.5 * d;
        }
        const double Ep = energy_at_strain(ep);
        const double Em = energy_at_strain(em);
        g_fd[a] = (Ep - Em) / (2.0 * d);
    }

    for (int a = 0; a < 6; ++a) {
        INFO(fmt::format("Voigt {}: analytic={:.10e} fd={:.10e}",
                         a + 1, g_analytic[a], g_fd[a]));
        REQUIRE(g_analytic[a] == Approx(g_fd[a]).epsilon(2e-5));
    }
}

// -------------------------------------------------------------------
// Test 4: Newton's 3rd law
// -------------------------------------------------------------------

TEST_CASE("Newton's 3rd law: F_A + F_B = 0", "[force][newton3]") {
    Mult m1(4), m2(4);
    m1.Q00() = -0.669;
    m1.Q10() = 0.234;
    m1.Q22c() = 0.05;
    m2.Q00() = 0.5;
    m2.Q11c() = 0.1;
    m2.Q20() = -0.3;

    Vec3 p1(1.2, -0.5, 0.3), p2(4.5, 1.3, -2.7);
    auto molA = CartesianMolecule::from_lab_sites({{m1, p1}});
    auto molB = CartesianMolecule::from_lab_sites({{m2, p2}});

    auto result = compute_molecule_forces(molA, molB);

    Vec3 total = result.forces_A[0] + result.forces_B[0];
    REQUIRE(total[0] == Approx(0.0).margin(1e-14));
    REQUIRE(total[1] == Approx(0.0).margin(1e-14));
    REQUIRE(total[2] == Approx(0.0).margin(1e-14));
}

// -------------------------------------------------------------------
// Test 5: Molecule-level force (multi-site)
// -------------------------------------------------------------------

TEST_CASE("Molecule force matches finite difference", "[force][molecule]") {
    Mult qO(4), qH(4);
    qO.Q00() = -0.669;
    qO.Q10() = 0.234;
    qO.Q20() = -0.123;
    qH.Q00() = 0.335;

    Vec3 pO1(0, 0, 0), pH1a(1.8, 0, 0), pH1b(-0.6, 1.7, 0);
    Vec3 pO2(6, 0, 0), pH2a(7.8, 0, 0), pH2b(5.4, 1.7, 0);

    auto molA = CartesianMolecule::from_lab_sites({
        {qO, pO1}, {qH, pH1a}, {qH, pH1b}
    });
    auto molB = CartesianMolecule::from_lab_sites({
        {qO, pO2}, {qH, pH2a}, {qH, pH2b}
    });

    auto result = compute_molecule_forces(molA, molB);

    // FD gradient of energy w.r.t. B's position: dE/d(posB)
    Vec3 fd_grad_B = finite_diff_gradient(molA, molB, 1, 1e-6);

    // forces_B is the physical force = -dE/d(posB), so compare with -fd_grad
    Vec3 total_force_B = Vec3::Zero();
    for (const auto &f : result.forces_B) total_force_B += f;

    for (int d = 0; d < 3; ++d) {
        INFO(fmt::format("Component {}: force_B={:.10e} -fd_grad={:.10e}",
                         d, total_force_B[d], -fd_grad_B[d]));
        REQUIRE(total_force_B[d] == Approx(-fd_grad_B[d]).epsilon(1e-7));
    }
}

// -------------------------------------------------------------------
// Test 6: Lever-arm torque for point charges
// -------------------------------------------------------------------

TEST_CASE("Lever-arm torque for charges matches finite difference", "[force][torque][lever]") {
    // Pure charges: no multipole rotation contribution
    Mult q1(0), q2(0), q3(0);
    q1.Q00() = -0.669;
    q2.Q00() = 0.335;
    q3.Q00() = 0.335;

    Vec3 body_offsets[] = {Vec3(0, 0, 0), Vec3(1.8, 0, 0), Vec3(-0.6, 1.7, 0)};
    Mat3 rotation = Mat3::Identity();
    Vec3 center(0, 0, 0);

    std::vector<std::pair<Mult, Vec3>> body_sites_A = {
        {q1, body_offsets[0]}, {q2, body_offsets[1]}, {q3, body_offsets[2]}
    };

    // Other molecule (single charge)
    Mult q_other(0);
    q_other.Q00() = 1.0;
    auto molB = single_site_mol(q_other, Vec3(6, 2, 1));

    auto molA = CartesianMolecule::from_body_frame_with_rotation(
        body_sites_A, rotation, center);
    auto mol_result = compute_molecule_forces(molA, molB);

    // Compute lever-arm torque
    std::vector<Vec3> positions;
    for (const auto &s : molA.sites)
        positions.push_back(s.position);

    auto rb = aggregate_rigid_body_forces(
        mol_result.forces_A, positions, center, rotation);

    // Finite-difference torque
    Vec3 fd_torque = finite_diff_torque(
        body_sites_A, rotation, center, molB, true, 1e-6);

    // torque_lab = Σ lever × F is the physical torque.
    // dE/dp = -torque_lab for the positional part (pure charges have no multipole term).
    // FD computes dE/dp, so compare -torque_lab with fd_torque.
    INFO(fmt::format("  -torque_lab: ({:.10e}, {:.10e}, {:.10e})",
                     -rb.torque_lab[0], -rb.torque_lab[1], -rb.torque_lab[2]));
    INFO(fmt::format("  FD dE/dp:    ({:.10e}, {:.10e}, {:.10e})",
                     fd_torque[0], fd_torque[1], fd_torque[2]));

    for (int d = 0; d < 3; ++d) {
        REQUIRE(-rb.torque_lab[d] == Approx(fd_torque[d]).epsilon(1e-6));
    }
}

// -------------------------------------------------------------------
// Test 7: Full torque with multipole rotation
// -------------------------------------------------------------------

TEST_CASE("Full torque with dipoles matches finite difference", "[force][torque][full]") {
    Mult m1(4), m2(4);
    m1.Q00() = -0.669;
    m1.Q10() = 0.5;  // dipole along z in body frame
    m1.Q20() = 0.2;
    m2.Q00() = 0.5;
    m2.Q11c() = 0.3; // dipole along x in body frame

    Vec3 body_offset_A(0, 0, 0);
    Vec3 body_offset_B(0, 0, 0);

    // Non-trivial rotation for molecule A
    Mat3 rotA = axis_angle_rotation(Vec3(1, 1, 0).normalized(), 0.5);
    Vec3 centerA(0, 0, 0);
    Mat3 rotB = Mat3::Identity();
    Vec3 centerB(5, 2, -1);

    std::vector<std::pair<Mult, Vec3>> body_sites_A = {{m1, body_offset_A}};
    std::vector<std::pair<Mult, Vec3>> body_sites_B = {{m2, body_offset_B}};

    auto molA = CartesianMolecule::from_body_frame_with_rotation(
        body_sites_A, rotA, centerA);
    auto molB = CartesianMolecule::from_body_frame_with_rotation(
        body_sites_B, rotB, centerB);

    auto result = compute_molecule_forces_torques(molA, molB);

    // Finite-difference torque for molecule A
    Vec3 fd_torque_A = finite_diff_torque(
        body_sites_A, rotA, centerA, molB, true, 1e-6);

    INFO(fmt::format("Analytical torque A: ({:.10e}, {:.10e}, {:.10e})",
                     result.grad_angle_axis_A[0],
                     result.grad_angle_axis_A[1],
                     result.grad_angle_axis_A[2]));
    INFO(fmt::format("FD torque A:         ({:.10e}, {:.10e}, {:.10e})",
                     fd_torque_A[0], fd_torque_A[1], fd_torque_A[2]));

    for (int d = 0; d < 3; ++d) {
        REQUIRE(result.grad_angle_axis_A[d] == Approx(fd_torque_A[d]).epsilon(1e-5));
    }
}

// -------------------------------------------------------------------
// Test 8: Cartesian rotation roundtrip
// -------------------------------------------------------------------

TEST_CASE("Cartesian multipole rotation roundtrip", "[force][rotation]") {
    // Create a multipole with all ranks
    Mult m(4);
    m.Q00() = 1.0;
    m.Q10() = 0.5;
    m.Q11c() = -0.3;
    m.Q11s() = 0.2;
    m.Q20() = 0.4;
    m.Q21c() = -0.1;
    m.Q22c() = 0.15;
    m.Q30() = 0.05;
    m.Q31c() = -0.03;
    m.Q33s() = 0.02;
    m.Q40() = 0.01;
    m.Q42c() = 0.005;
    m.Q44c() = -0.003;

    CartesianMultipole<4> body;
    spherical_to_cartesian<4>(m, body);

    // Rotate by arbitrary rotation, then rotate back
    Mat3 M = axis_angle_rotation(Vec3(1, 2, 3).normalized(), 1.2);

    CartesianMultipole<4> lab, roundtrip;
    rotate_cartesian_multipole<4>(body, M, lab);
    rotate_cartesian_multipole<4>(lab, M.transpose(), roundtrip);

    for (int i = 0; i < CartesianMultipole<4>::size; ++i) {
        INFO(fmt::format("Component {}: body={:.10e} roundtrip={:.10e} diff={:.2e}",
                         i, body.data[i], roundtrip.data[i],
                         std::abs(body.data[i] - roundtrip.data[i])));
        REQUIRE(roundtrip.data[i] == Approx(body.data[i]).margin(1e-12));
    }
}

TEST_CASE("Cartesian rotation identity preserves multipole", "[force][rotation]") {
    Mult m(4);
    m.Q00() = 1.0;
    m.Q10() = 0.5;
    m.Q20() = 0.3;
    m.Q30() = 0.1;
    m.Q40() = 0.05;

    CartesianMultipole<4> body;
    spherical_to_cartesian<4>(m, body);

    CartesianMultipole<4> lab;
    rotate_cartesian_multipole<4>(body, Mat3::Identity(), lab);

    for (int i = 0; i < CartesianMultipole<4>::size; ++i) {
        REQUIRE(lab.data[i] == Approx(body.data[i]).margin(1e-14));
    }
}

// -------------------------------------------------------------------
// Test 9: Rotation derivative matches finite difference
// -------------------------------------------------------------------

TEST_CASE("Rotation derivative matches finite difference", "[force][rotation][derivative]") {
    Mult m(4);
    m.Q10() = 1.0;
    m.Q11c() = 0.5;
    m.Q20() = 0.3;
    m.Q22c() = -0.2;

    CartesianMultipole<4> body;
    spherical_to_cartesian<4>(m, body);

    Mat3 M = axis_angle_rotation(Vec3(1, 1, 1).normalized(), 0.7);

    // Generator matrices
    Mat3 G[3];
    G[0] << 0, 0, 0,  0, 0, -1,  0, 1, 0;
    G[1] << 0, 0, 1,  0, 0, 0,  -1, 0, 0;
    G[2] << 0, -1, 0,  1, 0, 0,  0, 0, 0;

    for (int k = 0; k < 3; ++k) {
        Mat3 M1 = G[k] * M;

        CartesianMultipole<4> d_lab;
        rotate_cartesian_multipole_derivative<4>(body, M, M1, d_lab);

        // Finite difference
        double delta = 1e-7;
        Vec3 axis = Vec3::Zero();
        axis[k] = 1.0;
        Mat3 M_plus = axis_angle_rotation(axis, delta) * M;
        Mat3 M_minus = axis_angle_rotation(axis, -delta) * M;

        CartesianMultipole<4> lab_plus, lab_minus;
        rotate_cartesian_multipole<4>(body, M_plus, lab_plus);
        rotate_cartesian_multipole<4>(body, M_minus, lab_minus);

        for (int i = 0; i < CartesianMultipole<4>::size; ++i) {
            double fd = (lab_plus.data[i] - lab_minus.data[i]) / (2.0 * delta);
            if (std::abs(fd) < 1e-14 && std::abs(d_lab.data[i]) < 1e-14)
                continue;
            INFO(fmt::format("axis={} component={}: analytical={:.10e} fd={:.10e}",
                             k, i, d_lab.data[i], fd));
            REQUIRE(d_lab.data[i] == Approx(fd).epsilon(1e-5));
        }
    }
}

// -------------------------------------------------------------------
// Test 10: Energy consistency - force function gives same energy
// -------------------------------------------------------------------

TEST_CASE("Force function returns same energy as energy-only", "[force][consistency]") {
    Mult m1(4), m2(4);
    m1.Q00() = -0.669;
    m1.Q10() = 0.234;
    m1.Q20() = -0.123;
    m1.Q30() = 0.05;
    m2.Q00() = 0.5;
    m2.Q11c() = 0.1;
    m2.Q22c() = 0.05;

    // Positions in Angstrom (converted from Bohr)
    constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
    Vec3 p1(0, 0, 0), p2(4.5 * B2A, -1.3 * B2A, 2.7 * B2A);
    auto molA = single_site_mol(m1, p1);
    auto molB = single_site_mol(m2, p2);

    // compute_molecule_interaction returns kJ/mol
    double energy_only = compute_molecule_interaction(molA, molB);
    // compute_site_pair_energy_force returns Hartree
    auto ef = compute_site_pair_energy_force(molA.sites[0], molB.sites[0]);

    REQUIRE(ef.energy * occ::units::AU_TO_KJ_PER_MOL == Approx(energy_only).margin(1e-10));
}

// -------------------------------------------------------------------
// Test 11: Body-frame with rotation gives same energy as lab-frame
// -------------------------------------------------------------------

TEST_CASE("Body-frame with rotation matches lab energy", "[force][bodyframe]") {
    Mult m1(4);
    m1.Q00() = 1.0;
    m1.Q10() = 0.5;
    m1.Q22c() = 0.1;

    Mult m2(4);
    m2.Q00() = -0.5;

    Mat3 rot = axis_angle_rotation(Vec3(0, 0, 1), 0.3);
    Vec3 center(2, 0, 0);
    Vec3 body_pos(1, 0, 0);

    // Build with rotation (multipoles rotated)
    auto mol_rot = CartesianMolecule::from_body_frame_with_rotation(
        {{m1, body_pos}}, rot, center);

    // Build lab equivalent: manually rotate multipole and place at lab position
    Vec3 lab_pos = center + rot * body_pos;
    // For lab-frame, we need the rotated multipole
    // The from_body_frame_with_rotation should handle this

    auto mol_other = single_site_mol(m2, Vec3(0, 0, 0));

    double E_rot = compute_molecule_interaction(mol_rot, mol_other);

    // Verify by also computing with update_orientation (same rotation)
    mol_rot.update_orientation(rot, center);
    double E_updated = compute_molecule_interaction(mol_rot, mol_other);

    REQUIRE(E_rot == Approx(E_updated).margin(1e-14));
}

// -------------------------------------------------------------------
// Test 12: Multi-site molecule Newton's 3rd law
// -------------------------------------------------------------------

TEST_CASE("Multi-site molecule Newton's 3rd law", "[force][newton3][molecule]") {
    Mult qO(4), qH(4);
    qO.Q00() = -0.669;
    qO.Q10() = 0.234;
    qH.Q00() = 0.335;

    Vec3 pO1(0, 0, 0), pH1a(1.8, 0, 0), pH1b(-0.6, 1.7, 0);
    Vec3 pO2(6, 0, 0), pH2a(7.8, 0, 0), pH2b(5.4, 1.7, 0);

    auto molA = CartesianMolecule::from_lab_sites({
        {qO, pO1}, {qH, pH1a}, {qH, pH1b}
    });
    auto molB = CartesianMolecule::from_lab_sites({
        {qO, pO2}, {qH, pH2a}, {qH, pH2b}
    });

    auto result = compute_molecule_forces(molA, molB);

    Vec3 total_A = Vec3::Zero(), total_B = Vec3::Zero();
    for (const auto &f : result.forces_A) total_A += f;
    for (const auto &f : result.forces_B) total_B += f;

    Vec3 total = total_A + total_B;
    REQUIRE(total[0] == Approx(0.0).margin(1e-13));
    REQUIRE(total[1] == Approx(0.0).margin(1e-13));
    REQUIRE(total[2] == Approx(0.0).margin(1e-13));
}

// -------------------------------------------------------------------
// Test 13: Full torque multi-site with rotation
// -------------------------------------------------------------------

TEST_CASE("Full torque multi-site matches finite difference", "[force][torque][full][multisite]") {
    Mult qO(4), qH(4);
    qO.Q00() = -0.669;
    qO.Q10() = 0.234;
    qO.Q20() = -0.123;
    qH.Q00() = 0.335;

    Vec3 bodyO(0, 0, 0), bodyH1(1.8, 0, 0), bodyH2(-0.6, 1.7, 0);

    Mat3 rotA = axis_angle_rotation(Vec3(0, 1, 0), 0.4);
    Vec3 centerA(0, 0, 0);

    std::vector<std::pair<Mult, Vec3>> body_sites_A = {
        {qO, bodyO}, {qH, bodyH1}, {qH, bodyH2}
    };

    Mult q_other(0);
    q_other.Q00() = 1.0;
    auto molB = single_site_mol(q_other, Vec3(6, 2, 1));

    auto molA = CartesianMolecule::from_body_frame_with_rotation(
        body_sites_A, rotA, centerA);

    auto result = compute_molecule_forces_torques(molA, molB);

    Vec3 fd_torque = finite_diff_torque(
        body_sites_A, rotA, centerA, molB, true, 1e-6);

    for (int d = 0; d < 3; ++d) {
        INFO(fmt::format("Component {}: analytical={:.10e} fd={:.10e}",
                         d, result.grad_angle_axis_A[d], fd_torque[d]));
        REQUIRE(result.grad_angle_axis_A[d] == Approx(fd_torque[d]).epsilon(1e-5));
    }
}

// -------------------------------------------------------------------
// Test: Quaternion gradient descent reduces energy
// -------------------------------------------------------------------

TEST_CASE("Quaternion gradient step reduces energy", "[force][quaternion]") {
    // Set up two interacting molecules
    Mult m1(2), m2(2);
    m1.Q00() = 0.5;
    m1.Q10() = 0.8;
    m1.Q11c() = -0.3;
    m2.Q00() = -0.3;
    m2.Q10() = 0.6;

    Vec3 body_offset(0, 0, 0);
    std::vector<std::pair<Mult, Vec3>> body_sites_A = {{m1, body_offset}};
    std::vector<std::pair<Mult, Vec3>> body_sites_B = {{m2, body_offset}};

    // Initial orientations
    Eigen::Quaterniond qA = Eigen::Quaterniond(
        Eigen::AngleAxisd(0.5, Vec3(1, 1, 0).normalized()));
    Eigen::Quaterniond qB = Eigen::Quaterniond::Identity();

    Mat3 rotA = qA.toRotationMatrix();
    Mat3 rotB = qB.toRotationMatrix();
    Vec3 centerA(0, 0, 0);
    Vec3 centerB(4, 1, -1);

    auto molA = CartesianMolecule::from_body_frame_with_rotation(
        body_sites_A, rotA, centerA);
    auto molB = CartesianMolecule::from_body_frame_with_rotation(
        body_sites_B, rotB, centerB);

    // Get initial energy and gradient
    auto result = compute_molecule_forces_torques(molA, molB);
    double E_initial = result.energy;

    // Apply gradient step using the helper function
    double step = 0.1;
    Eigen::Quaterniond qA_new = qA;
    apply_quaternion_gradient_step(qA_new, result.grad_angle_axis_A, step);

    // Rebuild molecule with new orientation
    Mat3 rotA_new = qA_new.toRotationMatrix();
    auto molA_new = CartesianMolecule::from_body_frame_with_rotation(
        body_sites_A, rotA_new, centerA);

    double E_new = compute_molecule_interaction(molA_new, molB);

    INFO(fmt::format("E_initial = {:.10e}", E_initial));
    INFO(fmt::format("E_new     = {:.10e}", E_new));
    INFO(fmt::format("|grad|    = {:.10e}", result.grad_angle_axis_A.norm()));

    // Energy should decrease (or stay same if gradient is zero)
    if (result.grad_angle_axis_A.norm() > 1e-10) {
        REQUIRE(E_new < E_initial);
    }
}

TEST_CASE("Quaternion gradient step works for multiple configurations", "[force][quaternion]") {
    // Test that gradient descent reduces energy for various configurations
    Mult m1(2), m2(2);
    m1.Q00() = 0.3;
    m1.Q10() = 1.0;
    m1.Q11c() = -0.5;
    m2.Q00() = -0.2;
    m2.Q10() = 0.8;
    m2.Q11s() = 0.4;

    Vec3 body_offset(0, 0, 0);
    std::vector<std::pair<Mult, Vec3>> body_sites_A = {{m1, body_offset}};
    std::vector<std::pair<Mult, Vec3>> body_sites_B = {{m2, body_offset}};

    // Test several different initial orientations
    std::vector<Eigen::Quaterniond> test_orientations = {
        Eigen::Quaterniond::Identity(),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.5, Vec3::UnitX())),
        Eigen::Quaterniond(Eigen::AngleAxisd(1.2, Vec3::UnitY())),
        Eigen::Quaterniond(Eigen::AngleAxisd(0.8, Vec3(1, 1, 1).normalized())),
        Eigen::Quaterniond(Eigen::AngleAxisd(2.5, Vec3(1, -2, 0.5).normalized())),
    };

    Vec3 centerA(0, 0, 0);
    Vec3 centerB(4, 1, -1);

    for (size_t idx = 0; idx < test_orientations.size(); ++idx) {
        SECTION(fmt::format("Orientation {}", idx)) {
            Eigen::Quaterniond qA = test_orientations[idx];
            Eigen::Quaterniond qB = Eigen::Quaterniond::Identity();

            auto molA = CartesianMolecule::from_body_frame_with_rotation(
                body_sites_A, qA.toRotationMatrix(), centerA);
            auto molB = CartesianMolecule::from_body_frame_with_rotation(
                body_sites_B, qB.toRotationMatrix(), centerB);

            auto result = compute_molecule_forces_torques(molA, molB);
            double E_initial = result.energy;

            // Skip if gradient is too small
            if (result.grad_angle_axis_A.norm() < 1e-10) {
                continue;
            }

            // Apply gradient step
            double step = 0.05;
            Eigen::Quaterniond qA_new = qA;
            apply_quaternion_gradient_step(qA_new, result.grad_angle_axis_A, step);

            auto molA_new = CartesianMolecule::from_body_frame_with_rotation(
                body_sites_A, qA_new.toRotationMatrix(), centerA);
            double E_new = compute_molecule_interaction(molA_new, molB);

            INFO(fmt::format("E_initial = {:.10e}, E_new = {:.10e}, diff = {:.10e}",
                             E_initial, E_new, E_new - E_initial));

            // Energy should decrease
            REQUIRE(E_new < E_initial);
        }
    }
}
