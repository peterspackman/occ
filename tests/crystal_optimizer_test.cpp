#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/mults/lbfgs.h>
#include <occ/mults/trust_region.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_optimizer.h>
#include <occ/dma/mult.h>
#include <cmath>

using namespace occ;
using namespace occ::mults;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ============================================================================
// L-BFGS Tests
// ============================================================================

TEST_CASE("LBFGS: Rosenbrock function minimization", "[crystal_optimizer][lbfgs]") {
    // Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // Minimum at (1, 1)
    auto rosenbrock = [](const Vec& x, Vec& g) -> double {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        g[0] = -2.0 * a - 400.0 * x[0] * b;
        g[1] = 200.0 * b;
        return a * a + 100.0 * b * b;
    };

    LBFGSSettings settings;
    settings.gradient_tol = 1e-6;
    LBFGS optimizer(settings);

    Vec x0(2);
    x0 << -1.0, 1.0;

    auto result = optimizer.minimize(rosenbrock, x0, 500);

    REQUIRE(result.converged);
    CHECK_THAT(result.x[0], WithinAbs(1.0, 1e-4));
    CHECK_THAT(result.x[1], WithinAbs(1.0, 1e-4));
    CHECK_THAT(result.final_energy, WithinAbs(0.0, 1e-8));
}

TEST_CASE("LBFGS: Quadratic function", "[crystal_optimizer][lbfgs]") {
    // f(x) = (x-2)^T A (x-2) where A is diagonal
    Vec target(3);
    target << 2.0, -1.0, 0.5;
    Vec diag(3);
    diag << 1.0, 2.0, 3.0;

    auto quadratic = [&](const Vec& x, Vec& g) -> double {
        Vec diff = x - target;
        g = 2.0 * diag.asDiagonal() * diff;
        return diff.dot(diag.asDiagonal() * diff);
    };

    LBFGSSettings settings;
    settings.gradient_tol = 1e-8;
    LBFGS optimizer(settings);

    Vec x0 = Vec::Zero(3);
    auto result = optimizer.minimize(quadratic, x0, 100);

    REQUIRE(result.converged);
    CHECK_THAT(result.x[0], WithinAbs(target[0], 1e-6));
    CHECK_THAT(result.x[1], WithinAbs(target[1], 1e-6));
    CHECK_THAT(result.x[2], WithinAbs(target[2], 1e-6));
}

TEST_CASE("LBFGS: 6D optimization (translation + rotation)", "[crystal_optimizer][lbfgs]") {
    // Simulate rigid body DOF: 3 translation + 3 angle-axis
    // Target: position = (1, 2, 3), rotation = small angle about z
    Vec target(6);
    target << 1.0, 2.0, 3.0, 0.0, 0.0, 0.1;  // pos + angle-axis

    auto objective = [&](const Vec& x, Vec& g) -> double {
        Vec diff = x - target;
        g = 2.0 * diff;
        return diff.squaredNorm();
    };

    LBFGS optimizer;
    Vec x0 = Vec::Zero(6);
    auto result = optimizer.minimize(objective, x0, 100);

    REQUIRE(result.converged);
    for (int i = 0; i < 6; ++i) {
        CHECK_THAT(result.x[i], WithinAbs(target[i], 1e-5));
    }
}

// ============================================================================
// Trust Region Newton Tests
// ============================================================================

TEST_CASE("TrustRegion: Quadratic function", "[crystal_optimizer][trust_region]") {
    // f(x) = 0.5 * x^T A x - b^T x
    // Minimum at x = A^{-1} b
    Mat A(3, 3);
    A << 4.0, 1.0, 0.0,
         1.0, 3.0, 0.5,
         0.0, 0.5, 2.0;
    Vec b(3);
    b << 1.0, 2.0, 3.0;
    Vec optimal = A.ldlt().solve(b);

    auto objective = [&](const Vec& x) -> std::pair<double, Vec> {
        double f = 0.5 * x.dot(A * x) - b.dot(x);
        Vec g = A * x - b;
        return {f, g};
    };

    auto hessian = [&](const Vec& x) -> Mat {
        return A;  // Constant Hessian for quadratic
    };

    TrustRegionSettings settings;
    settings.gradient_tol = 1e-10;
    TrustRegion optimizer(settings);

    Vec x0 = Vec::Zero(3);
    auto result = optimizer.minimize(objective, hessian, x0);

    REQUIRE(result.converged);
    CHECK_THAT(result.x[0], WithinAbs(optimal[0], 1e-8));
    CHECK_THAT(result.x[1], WithinAbs(optimal[1], 1e-8));
    CHECK_THAT(result.x[2], WithinAbs(optimal[2], 1e-8));
    CHECK(result.iterations <= 3);  // Should converge in few iterations for quadratic
}

TEST_CASE("TrustRegion: Rosenbrock function", "[crystal_optimizer][trust_region]") {
    // Rosenbrock: harder test for trust region
    auto objective = [](const Vec& x) -> std::pair<double, Vec> {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        Vec g(2);
        g[0] = -2.0 * a - 400.0 * x[0] * b;
        g[1] = 200.0 * b;
        return {a * a + 100.0 * b * b, g};
    };

    // Numerical Hessian (Rosenbrock has complex analytical Hessian)
    auto hessian = [&objective](const Vec& x) -> Mat {
        const double h = 1e-5;
        int n = x.size();
        Mat H(n, n);
        auto [f0, g0] = objective(x);
        for (int j = 0; j < n; ++j) {
            Vec x_plus = x;
            Vec x_minus = x;
            x_plus[j] += h;
            x_minus[j] -= h;
            auto [fp, gp] = objective(x_plus);
            auto [fm, gm] = objective(x_minus);
            H.col(j) = (gp - gm) / (2.0 * h);
        }
        return 0.5 * (H + H.transpose());  // Symmetrize
    };

    TrustRegionSettings settings;
    settings.gradient_tol = 1e-6;
    settings.max_iterations = 100;
    TrustRegion optimizer(settings);

    Vec x0(2);
    x0 << -1.0, 1.0;
    auto result = optimizer.minimize(objective, hessian, x0);

    REQUIRE(result.converged);
    CHECK_THAT(result.x[0], WithinAbs(1.0, 1e-4));
    CHECK_THAT(result.x[1], WithinAbs(1.0, 1e-4));
}

TEST_CASE("TrustRegion: Indefinite Hessian handling", "[crystal_optimizer][trust_region]") {
    // f(x) = x1^2 - x2^2 + 0.1*x1 (saddle near origin, but with gradient)
    // Start slightly off the saddle - should escape along negative curvature
    auto objective = [](const Vec& x) -> std::pair<double, Vec> {
        Vec g(2);
        g[0] = 2.0 * x[0] + 0.1;
        g[1] = -2.0 * x[1];
        return {x[0] * x[0] - x[1] * x[1] + 0.1 * x[0], g};
    };

    auto hessian = [](const Vec& x) -> Mat {
        Mat H(2, 2);
        H << 2.0, 0.0,
             0.0, -2.0;  // Indefinite!
        return H;
    };

    TrustRegionSettings settings;
    settings.initial_radius = 1.0;
    settings.max_iterations = 10;
    TrustRegion optimizer(settings);

    Vec x0(2);
    x0 << 0.0, 0.01;  // Slight perturbation
    auto result = optimizer.minimize(objective, hessian, x0);

    // Trust region should detect negative curvature and move to boundary
    // along the x2 direction (where curvature is negative)
    INFO("Final x = (" << result.x[0] << ", " << result.x[1] << ")");
    // The optimizer should have moved - either along gradient or neg curvature
    CHECK(result.x.norm() > 0.05);
}

// ============================================================================
// MoleculeState Tests
// ============================================================================

TEST_CASE("MoleculeState: Identity rotation", "[crystal_optimizer][molecule_state]") {
    MoleculeState state;
    state.position = Vec3(1.0, 2.0, 3.0);
    state.angle_axis = Vec3::Zero();

    Mat3 R = state.rotation_matrix();
    CHECK(R.isApprox(Mat3::Identity(), 1e-12));
}

TEST_CASE("MoleculeState: 90 degree rotation about z", "[crystal_optimizer][molecule_state]") {
    MoleculeState state;
    state.position = Vec3::Zero();
    state.angle_axis = Vec3(0.0, 0.0, M_PI / 2.0);

    Mat3 R = state.rotation_matrix();

    // Rotating (1, 0, 0) by 90 deg about z should give (0, 1, 0)
    Vec3 v(1.0, 0.0, 0.0);
    Vec3 rotated = R * v;

    CHECK_THAT(rotated[0], WithinAbs(0.0, 1e-10));
    CHECK_THAT(rotated[1], WithinAbs(1.0, 1e-10));
    CHECK_THAT(rotated[2], WithinAbs(0.0, 1e-10));
}

TEST_CASE("MoleculeState: Round-trip from_rotation", "[crystal_optimizer][molecule_state]") {
    // Create state with arbitrary rotation
    MoleculeState original;
    original.position = Vec3(1.5, -2.3, 0.7);
    original.angle_axis = Vec3(0.3, -0.2, 0.5);  // ~0.6 rad rotation

    Mat3 R = original.rotation_matrix();

    // Reconstruct from rotation matrix
    MoleculeState reconstructed = MoleculeState::from_rotation(original.position, R);

    // Check rotation matrices match (angle-axis representation is not unique)
    Mat3 R_reconstructed = reconstructed.rotation_matrix();
    CHECK(R.isApprox(R_reconstructed, 1e-10));
}

// ============================================================================
// CrystalEnergyResult Tests
// ============================================================================

TEST_CASE("CrystalEnergyResult: pack_gradient", "[crystal_optimizer][energy_result]") {
    CrystalEnergyResult result;
    result.forces.resize(2);
    result.torques.resize(2);

    result.forces[0] = Vec3(1.0, 2.0, 3.0);
    result.forces[1] = Vec3(4.0, 5.0, 6.0);
    result.torques[0] = Vec3(0.1, 0.2, 0.3);
    result.torques[1] = Vec3(0.4, 0.5, 0.6);

    Vec grad = result.pack_gradient();

    REQUIRE(grad.size() == 12);
    CHECK_THAT(grad[0], WithinAbs(1.0, 1e-12));
    CHECK_THAT(grad[1], WithinAbs(2.0, 1e-12));
    CHECK_THAT(grad[2], WithinAbs(3.0, 1e-12));
    CHECK_THAT(grad[3], WithinAbs(0.1, 1e-12));
    CHECK_THAT(grad[4], WithinAbs(0.2, 1e-12));
    CHECK_THAT(grad[5], WithinAbs(0.3, 1e-12));
    CHECK_THAT(grad[6], WithinAbs(4.0, 1e-12));
    CHECK_THAT(grad[7], WithinAbs(5.0, 1e-12));
    CHECK_THAT(grad[8], WithinAbs(6.0, 1e-12));
}

// ============================================================================
// Williams DE Parameters Tests
// ============================================================================

TEST_CASE("Williams DE parameters: Basic coverage", "[crystal_optimizer][force_field]") {
    auto params = CrystalEnergy::williams_de_params();

    // Check some key pairs exist
    REQUIRE(params.count({1, 1}) == 1);  // H-H
    REQUIRE(params.count({6, 6}) == 1);  // C-C
    REQUIRE(params.count({7, 7}) == 1);  // N-N
    REQUIRE(params.count({8, 8}) == 1);  // O-O

    // Check cross-terms are symmetric
    REQUIRE(params.count({1, 6}) == 1);
    REQUIRE(params.count({6, 1}) == 1);
    auto hc_16 = params[{1, 6}];
    auto hc_61 = params[{6, 1}];
    CHECK_THAT(hc_16.A, WithinAbs(hc_61.A, 1e-10));

    // Check reasonable values for H-H
    auto hh = params[{1, 1}];
    CHECK(hh.A > 0);
    CHECK(hh.B > 0);
    CHECK(hh.C > 0);
}

TEST_CASE("Williams DE: Energy at typical distances", "[crystal_optimizer][force_field]") {
    auto params = CrystalEnergy::williams_de_params();

    // H-H at 2.4 Angstrom (van der Waals contact)
    auto hh = params[{1, 1}];
    double r = 2.4;
    auto result = ShortRangeInteraction::buckingham_all(r, hh);

    // Should be slightly repulsive at this distance
    // Energy should be finite and reasonable
    CHECK(std::isfinite(result.energy));
    CHECK(std::isfinite(result.first_derivative));

    // At larger distance, should be attractive (dispersion dominates)
    double r_large = 4.0;
    auto result_large = ShortRangeInteraction::buckingham_all(r_large, hh);
    CHECK(result_large.energy < 0);  // Attractive
}

// ============================================================================
// Integration Tests (require Crystal infrastructure)
// ============================================================================

// These tests would require a real Crystal object, which we skip for now
// and test in integration with actual CIF files

TEST_CASE("LBFGS: Convergence callback", "[crystal_optimizer][lbfgs]") {
    auto quadratic = [](const Vec& x, Vec& g) -> double {
        g = 2.0 * x;
        return x.squaredNorm();
    };

    LBFGS optimizer;
    Vec x0(2);
    x0 << 5.0, 3.0;

    int callback_count = 0;
    auto callback = [&callback_count](int iter, const Vec& x, double f, const Vec& g) {
        callback_count++;
        return true;  // Continue
    };

    auto result = optimizer.minimize(quadratic, x0, callback, 100);

    REQUIRE(result.converged);
    CHECK(callback_count > 0);
    // callback_count = iterations + 1 because callback is called for iteration 0 too
    CHECK(callback_count == result.iterations + 1);
}

TEST_CASE("LBFGS: Early stop via callback", "[crystal_optimizer][lbfgs]") {
    // Rosenbrock from far starting point to ensure many iterations needed
    auto rosenbrock = [](const Vec& x, Vec& g) -> double {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        g[0] = -2.0 * a - 400.0 * x[0] * b;
        g[1] = 200.0 * b;
        return a * a + 100.0 * b * b;
    };

    LBFGSSettings settings;
    settings.gradient_tol = 1e-15;  // Extremely tight
    settings.energy_tol = 1e-20;
    settings.x_tol = 1e-20;
    LBFGS optimizer(settings);

    Vec x0(2);
    x0 << -5.0, 5.0;  // Far from minimum

    int stop_at = 3;
    auto callback = [&stop_at](int iter, const Vec& x, double f, const Vec& g) {
        return iter < stop_at;
    };

    auto result = optimizer.minimize(rosenbrock, x0, callback, 500);

    // Main check: callback was honored and iteration stopped
    CHECK(result.iterations <= stop_at);

    // If not converged naturally, should be callback stop
    if (!result.converged) {
        CHECK(result.termination_reason == "Stopped by callback");
    }
}
