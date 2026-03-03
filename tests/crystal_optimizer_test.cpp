#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/mults/lbfgs.h>
#include <occ/mults/mstmin.h>
#include <occ/mults/trust_region.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_optimizer.h>
#include <occ/mults/dmacrys_input.h>
#include <occ/dma/mult.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <cmath>

using namespace occ;
using namespace occ::mults;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static const std::string AXOSOW_JSON = CMAKE_SOURCE_DIR "/tests/data/dmacrys/AXOSOW.json";

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
// MSTMIN Tests
// ============================================================================

TEST_CASE("MSTMIN: Rosenbrock function minimization", "[crystal_optimizer][mstmin]") {
    auto rosenbrock = [](const Vec& x, Vec& g) -> double {
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        g[0] = -2.0 * a - 400.0 * x[0] * b;
        g[1] = 200.0 * b;
        return a * a + 100.0 * b * b;
    };

    MSTMINSettings settings;
    settings.gradient_tol = 1e-6;
    settings.step_tol = 1e-6;
    settings.max_displacement = 0.25;
    settings.energy_tol = 1e-12;
    MSTMIN optimizer(settings);

    Vec x0(2);
    x0 << -1.0, 1.0;

    auto result = optimizer.minimize(rosenbrock, x0, 500);

    REQUIRE(result.converged);
    CHECK_THAT(result.x[0], WithinAbs(1.0, 1e-4));
    CHECK_THAT(result.x[1], WithinAbs(1.0, 1e-4));
    CHECK_THAT(result.final_energy, WithinAbs(0.0, 1e-7));
}

TEST_CASE("MSTMIN: Quadratic function", "[crystal_optimizer][mstmin]") {
    Vec target(3);
    target << 2.0, -1.0, 0.5;
    Vec diag(3);
    diag << 1.0, 2.0, 3.0;

    auto quadratic = [&](const Vec& x, Vec& g) -> double {
        Vec diff = x - target;
        g = 2.0 * diag.asDiagonal() * diff;
        return diff.dot(diag.asDiagonal() * diff);
    };

    MSTMINSettings settings;
    settings.gradient_tol = 1e-9;
    settings.step_tol = 1e-9;
    settings.max_displacement = 0.5;
    settings.energy_tol = 1e-14;
    MSTMIN optimizer(settings);

    Vec x0 = Vec::Zero(3);
    auto result = optimizer.minimize(quadratic, x0, 100);

    REQUIRE(result.converged);
    CHECK_THAT(result.x[0], WithinAbs(target[0], 1e-6));
    CHECK_THAT(result.x[1], WithinAbs(target[1], 1e-6));
    CHECK_THAT(result.x[2], WithinAbs(target[2], 1e-6));
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

// ============================================================================
// Symmetry Mapping Tests
// ============================================================================

TEST_CASE("SymmetryMapping: AXOSOW Z'=1 Pbca", "[crystal_optimizer][symmetry]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);

    auto mapping = build_symmetry_mapping(crystal);

    // Pbca: Z=8, Z'=1
    CHECK(mapping.num_independent == 1);
    CHECK(mapping.num_uc_molecules == 8);

    // All 8 UC molecules should map to independent molecule 0
    for (int j = 0; j < mapping.num_uc_molecules; ++j) {
        CHECK(mapping.uc_molecules[j].independent_idx == 0);
    }

    // Independent molecule 0 should have 8 UC images
    REQUIRE(mapping.independent[0].uc_indices.size() == 8);

    // Z'=1 at general position: full 6 DOF (3 trans + 3 rot)
    CHECK(mapping.independent[0].trans_dof == 3);
    CHECK(mapping.independent[0].rot_dof == 3);
    CHECK(mapping.total_dof() == 6);
}

TEST_CASE("SymmetryMapping: generate and round-trip UC states", "[crystal_optimizer][symmetry]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    auto mapping = build_symmetry_mapping(crystal);

    // Create CrystalEnergy to get initial states
    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, false);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);

    auto uc_states = calc.initial_states();

    // Extract independent state from the first UC image
    int ref_uc = mapping.independent[0].uc_indices[0];
    const auto& uc_info = mapping.uc_molecules[ref_uc];

    MoleculeState indep_state;
    indep_state.position =
        uc_info.R_cart.transpose() * (uc_states[ref_uc].position - uc_info.t_cart);
    Mat3 R_uc = uc_states[ref_uc].rotation_matrix();
    Mat3 R_indep = uc_info.R_cart.transpose() * R_uc;
    indep_state = MoleculeState::from_rotation(indep_state.position, R_indep);

    // Generate UC states from independent state
    std::vector<MoleculeState> indep_states = {indep_state};
    auto generated = generate_uc_states(indep_states, mapping);

    REQUIRE(generated.size() == 8);

    // Check that generated positions are close to original UC positions
    for (int j = 0; j < 8; ++j) {
        double pos_diff = (generated[j].position - uc_states[j].position).norm();
        INFO("UC mol " << j << ": pos diff = " << pos_diff);
        CHECK(pos_diff < 0.01);  // Should be very close (within rounding)

        Mat3 R_ref = uc_states[j].rotation_matrix();
        Mat3 R_gen = generated[j].rotation_matrix();
        CHECK_THAT((R_ref - R_gen).norm(), WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("MoleculeState preserves improper orientations", "[crystal_optimizer][symmetry]") {
    Mat3 Q = Mat3::Identity();
    Q(0, 0) = -1.0; // Reflection in yz plane (det = -1)

    MoleculeState state = MoleculeState::from_rotation(Vec3::Zero(), Q);
    CHECK(state.parity == -1);

    Mat3 recon = state.rotation_matrix();
    CHECK_THAT((Q - recon).norm(), WithinAbs(0.0, 1e-10));
    CHECK_THAT(recon.determinant(), WithinAbs(-1.0, 1e-10));
}

TEST_CASE("CrystalOptimizer: symmetry DOF count", "[crystal_optimizer][symmetry]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    // Symmetry mode keeps full independent DOF; fix_first_* applies only in legacy mode.
    CrystalOptimizerSettings settings;
    settings.use_symmetry = true;
    settings.fix_first_translation = true;
    settings.fix_first_rotation = false;
    settings.force_field = ForceFieldType::Custom;
    settings.use_ewald = false;

    CrystalOptimizer optimizer(crystal, multipoles, settings);

    // Z'=1 at general position: full 6 DOF
    CHECK(optimizer.num_parameters() == 6);
    CHECK_FALSE(optimizer.settings().fix_first_translation);
    CHECK_FALSE(optimizer.settings().fix_first_rotation);
}

TEST_CASE("CrystalOptimizer: symmetry vs legacy energy match", "[crystal_optimizer][symmetry]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    // Setup with symmetry
    CrystalOptimizerSettings sym_settings;
    sym_settings.use_symmetry = true;
    sym_settings.fix_first_translation = true;
    sym_settings.fix_first_rotation = false;
    sym_settings.force_field = ForceFieldType::Custom;
    sym_settings.use_cartesian_engine = true;
    sym_settings.use_ewald = false;

    CrystalOptimizer sym_opt(crystal, multipoles, sym_settings);
    setup_crystal_energy_from_dmacrys(sym_opt.energy_calculator(), input, crystal, multipoles);
    for (const auto& [key, p] : buck_params)
        sym_opt.energy_calculator().set_buckingham_params(key.first, key.second, p);
    sym_opt.reinitialize_states();

    // Setup without symmetry
    CrystalOptimizerSettings legacy_settings = sym_settings;
    legacy_settings.use_symmetry = false;

    CrystalOptimizer legacy_opt(crystal, multipoles, legacy_settings);
    setup_crystal_energy_from_dmacrys(legacy_opt.energy_calculator(), input, crystal, multipoles);
    for (const auto& [key, p] : buck_params)
        legacy_opt.energy_calculator().set_buckingham_params(key.first, key.second, p);
    legacy_opt.reinitialize_states();

    // Compute energies
    auto sym_result = sym_opt.compute_energy_gradient();
    auto legacy_result = legacy_opt.compute_energy_gradient();

    INFO("Symmetric energy: " << sym_result.total_energy);
    INFO("Legacy energy: " << legacy_result.total_energy);

    // Energies should match since both use the same UC states
    CHECK_THAT(sym_result.total_energy,
               WithinAbs(legacy_result.total_energy, 0.01));
}

TEST_CASE("CrystalOptimizer: symmetry gradient check", "[crystal_optimizer][symmetry]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    CrystalOptimizerSettings settings;
    settings.use_symmetry = true;
    settings.fix_first_translation = true;
    settings.fix_first_rotation = false;
    settings.force_field = ForceFieldType::Custom;
    settings.use_cartesian_engine = true;
    settings.use_ewald = false;
    settings.max_interaction_order = 4;

    CrystalOptimizer optimizer(crystal, multipoles, settings);
    setup_crystal_energy_from_dmacrys(optimizer.energy_calculator(), input, crystal, multipoles);
    for (const auto& [key, p] : buck_params)
        optimizer.energy_calculator().set_buckingham_params(key.first, key.second, p);
    optimizer.reinitialize_states();

    // Get current parameters
    Vec params = optimizer.get_parameters();
    REQUIRE(params.size() == 6);  // 3 translation + 3 rotation DOF

    // Compute reference energy
    optimizer.set_parameters(params);
    double energy = optimizer.energy_calculator().compute(optimizer.states()).total_energy;

    // 4-point finite-difference gradient (O(h^4) accuracy)
    const double h = 1e-4;  // Larger step for better signal-to-noise
    Vec fd_grad(params.size());

    for (int i = 0; i < params.size(); ++i) {
        Vec pp = params, pm = params, p2p = params, p2m = params;
        pp[i] += h;
        pm[i] -= h;
        p2p[i] += 2.0 * h;
        p2m[i] -= 2.0 * h;

        optimizer.set_parameters(pp);
        double ep = optimizer.energy_calculator().compute(optimizer.states()).total_energy;
        optimizer.set_parameters(pm);
        double em = optimizer.energy_calculator().compute(optimizer.states()).total_energy;
        optimizer.set_parameters(p2p);
        double e2p = optimizer.energy_calculator().compute(optimizer.states()).total_energy;
        optimizer.set_parameters(p2m);
        double e2m = optimizer.energy_calculator().compute(optimizer.states()).total_energy;

        fd_grad[i] = (-e2p + 8.0 * ep - 8.0 * em + e2m) / (12.0 * h);
        INFO("DOF " << i << ": fd_grad=" << fd_grad[i]);
    }

    // Verify: moving along negative gradient direction decreases energy
    double step = 1e-5;
    Vec params_down = params - step * fd_grad;
    optimizer.set_parameters(params_down);
    double e_down = optimizer.energy_calculator().compute(optimizer.states()).total_energy;

    INFO("Energy: " << energy << " -> " << e_down << " (delta=" << (e_down - energy) << ")");
    CHECK(e_down < energy);  // Energy should decrease along negative gradient

    // Cross-check: FD at two step sizes should agree (relative tolerance)
    const double h2 = h / 2.0;
    for (int i = 0; i < params.size(); ++i) {
        Vec pp2 = params, pm2 = params, p2p2 = params, p2m2 = params;
        pp2[i] += h2;
        pm2[i] -= h2;
        p2p2[i] += 2.0 * h2;
        p2m2[i] -= 2.0 * h2;

        optimizer.set_parameters(pp2);
        double ep2 = optimizer.energy_calculator().compute(optimizer.states()).total_energy;
        optimizer.set_parameters(pm2);
        double em2 = optimizer.energy_calculator().compute(optimizer.states()).total_energy;
        optimizer.set_parameters(p2p2);
        double e2p2 = optimizer.energy_calculator().compute(optimizer.states()).total_energy;
        optimizer.set_parameters(p2m2);
        double e2m2 = optimizer.energy_calculator().compute(optimizer.states()).total_energy;

        double fd2 = (-e2p2 + 8.0 * ep2 - 8.0 * em2 + e2m2) / (12.0 * h2);

        double rel_diff = std::abs(fd_grad[i] - fd2) / (std::abs(fd_grad[i]) + 1e-10);
        double abs_diff = std::abs(fd_grad[i] - fd2);
        INFO("DOF " << i << ": fd(h)=" << fd_grad[i] << " fd(h/2)=" << fd2
             << " rel_diff=" << rel_diff << " abs_diff=" << abs_diff);
        // 4-pt FD at h and h/2 should agree reasonably.
        // Small gradients in high-curvature regions may show larger relative error.
        CHECK((rel_diff < 0.01 || abs_diff < 2.0));
    }

    // Reset
    optimizer.set_parameters(params);
}

TEST_CASE("SymmetryMapping: accumulate_gradients consistency", "[crystal_optimizer][symmetry]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto mapping = build_symmetry_mapping(crystal);

    // Create mock UC-level forces/torques
    std::vector<Vec3> uc_forces(8), uc_torques(8);
    for (int j = 0; j < 8; ++j) {
        uc_forces[j] = Vec3(1.0, 2.0, 3.0) * (j + 1);  // Distinct per molecule
        uc_torques[j] = Vec3(0.1, 0.2, 0.3) * (j + 1);
    }

    std::vector<Vec3> indep_forces, indep_torques;
    accumulate_gradients(uc_forces, uc_torques, mapping, indep_forces, indep_torques);

    // Should have 1 independent molecule
    REQUIRE(indep_forces.size() == 1);
    REQUIRE(indep_torques.size() == 1);

    // The accumulated force should be non-zero (sum of back-rotated forces)
    CHECK(indep_forces[0].norm() > 0.0);
    CHECK(indep_torques[0].norm() > 0.0);
}

TEST_CASE("CrystalOptimizer: cell mode guardrails", "[crystal_optimizer][cell]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    CrystalOptimizerSettings settings;
    settings.use_symmetry = true;
    settings.optimize_cell = true;
    settings.method = OptimizationMethod::TrustRegion;
    settings.fix_first_translation = true;
    settings.fix_first_rotation = false;

    CrystalOptimizer optimizer(crystal, multipoles, settings);

    CHECK(optimizer.settings().use_symmetry);
    CHECK(optimizer.settings().method == OptimizationMethod::MSTMIN);
    CHECK(optimizer.num_parameters() == 9);  // 6 molecular + 3 active cell DOF (orthorhombic)
}

TEST_CASE("CrystalOptimizer: TrustRegion exact-Hessian gating",
          "[crystal_optimizer][hessian]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    SECTION("Keep TrustRegion when exact Ewald Hessian is available") {
        CrystalOptimizerSettings settings;
        settings.method = OptimizationMethod::TrustRegion;
        settings.use_symmetry = false;
        settings.use_cartesian_engine = true;  // includes multipole electrostatics
        settings.use_ewald = true;
        settings.require_exact_hessian = true;
        settings.force_field = ForceFieldType::Custom;

        CrystalOptimizer optimizer(crystal, multipoles, settings);
        CHECK(optimizer.settings().method == OptimizationMethod::TrustRegion);

        auto h = optimizer.energy_calculator().compute_with_hessian(optimizer.states());
        CHECK(h.exact_for_model);
        CHECK(h.includes_ewald_terms);
    }

    SECTION("Keep exact-Hessian status with electrostatic taper enabled") {
        CrystalOptimizerSettings settings;
        settings.method = OptimizationMethod::TrustRegion;
        settings.use_symmetry = false;
        settings.use_cartesian_engine = true;
        settings.use_ewald = true;
        settings.require_exact_hessian = true;
        settings.force_field = ForceFieldType::Custom;

        CrystalOptimizer optimizer(crystal, multipoles, settings);
        optimizer.energy_calculator().set_electrostatic_taper(10.0, 12.0, 3);
        CHECK(optimizer.settings().method == OptimizationMethod::TrustRegion);
        CHECK(optimizer.energy_calculator().can_compute_exact_hessian());

        auto h = optimizer.energy_calculator().compute_with_hessian(optimizer.states());
        CHECK(h.exact_for_model);
    }

    SECTION("Keep TrustRegion when exact Hessian is available") {
        CrystalOptimizerSettings settings;
        settings.method = OptimizationMethod::TrustRegion;
        settings.use_symmetry = false;
        settings.use_cartesian_engine = false; // short-range only model
        settings.use_ewald = false;
        settings.require_exact_hessian = true;
        settings.force_field = ForceFieldType::Custom;

        CrystalOptimizer optimizer(crystal, multipoles, settings);
        CHECK(optimizer.settings().method == OptimizationMethod::TrustRegion);

        auto h = optimizer.energy_calculator().compute_with_hessian(optimizer.states());
        CHECK(h.exact_for_model);
    }

    SECTION("Keep TrustRegion for non-Ewald Cartesian multipoles") {
        CrystalOptimizerSettings settings;
        settings.method = OptimizationMethod::TrustRegion;
        settings.use_symmetry = false;
        settings.use_cartesian_engine = true;  // full multipole electrostatics
        settings.use_ewald = false;            // required for exact Hessian
        settings.require_exact_hessian = true;
        settings.force_field = ForceFieldType::Custom;

        CrystalOptimizer optimizer(crystal, multipoles, settings);
        CHECK(optimizer.settings().method == OptimizationMethod::TrustRegion);

        auto h = optimizer.energy_calculator().compute_with_hessian(optimizer.states());
        CHECK(h.exact_for_model);
        CHECK_FALSE(h.includes_ewald_terms);
    }
}

TEST_CASE("CrystalOptimizer: symmetry+cell strain constraints",
          "[crystal_optimizer][cell][symmetry]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    SECTION("Orthorhombic default: shear strains are constrained") {
        CrystalOptimizerSettings settings;
        settings.method = OptimizationMethod::LBFGS;
        settings.use_symmetry = true;
        settings.optimize_cell = true;
        settings.force_field = ForceFieldType::Custom;
        settings.use_ewald = false;
        settings.constrain_cell_strain_by_lattice = true;

        CrystalOptimizer optimizer(crystal, multipoles, settings);
        Vec p0 = optimizer.get_parameters();
        REQUIRE(p0.size() == optimizer.num_parameters());
        REQUIRE(p0.size() == 9); // 6 molecular + [E1,E2,E3]

        Vec p1 = p0;
        const int cell_offset = static_cast<int>(p1.size()) - 3;
        p1[cell_offset + 0] = 6e-4;   // E1 (active)
        p1[cell_offset + 1] = -4e-4;  // E2 (active)
        p1[cell_offset + 2] = 2e-4;   // E3 (active)

        optimizer.set_parameters(p1);
        Vec p2 = optimizer.get_parameters();

        CHECK_THAT(p2[cell_offset + 0], WithinAbs(p1[cell_offset + 0], 1e-12));
        CHECK_THAT(p2[cell_offset + 1], WithinAbs(p1[cell_offset + 1], 1e-12));
        CHECK_THAT(p2[cell_offset + 2], WithinAbs(p1[cell_offset + 2], 1e-12));
    }

    SECTION("Monoclinic default: only one shear component is active") {
        auto uc = crystal.unit_cell();
        occ::crystal::UnitCell mono_uc(
            uc.a(), uc.b(), uc.c(),
            occ::units::PI * 0.5, occ::units::radians(110.0), occ::units::PI * 0.5);
        occ::crystal::Crystal mono_crystal(
            crystal.asymmetric_unit(), crystal.space_group(), mono_uc);

        CrystalOptimizerSettings settings;
        settings.method = OptimizationMethod::LBFGS;
        settings.use_symmetry = true;
        settings.optimize_cell = true;
        settings.force_field = ForceFieldType::Custom;
        settings.use_ewald = false;
        settings.constrain_cell_strain_by_lattice = true;

        CrystalOptimizer optimizer(mono_crystal, multipoles, settings);
        Vec p = optimizer.get_parameters();
        REQUIRE(p.size() == 10); // 6 molecular + [E1,E2,E3,E5]
        const int cell_offset = static_cast<int>(p.size()) - 4;
        p[cell_offset + 0] = 2e-4;    // E1
        p[cell_offset + 1] = -3e-4;   // E2
        p[cell_offset + 2] = 4e-4;    // E3
        p[cell_offset + 3] = -2e-3;   // E5 (beta shear, active)

        optimizer.set_parameters(p);
        Vec p2 = optimizer.get_parameters();
        CHECK_THAT(p2[cell_offset + 0], WithinAbs(p[cell_offset + 0], 1e-12));
        CHECK_THAT(p2[cell_offset + 1], WithinAbs(p[cell_offset + 1], 1e-12));
        CHECK_THAT(p2[cell_offset + 2], WithinAbs(p[cell_offset + 2], 1e-12));
        CHECK_THAT(p2[cell_offset + 3], WithinAbs(p[cell_offset + 3], 1e-12));
    }

    SECTION("Constraints can be disabled explicitly") {
        CrystalOptimizerSettings settings;
        settings.method = OptimizationMethod::LBFGS;
        settings.use_symmetry = true;
        settings.optimize_cell = true;
        settings.force_field = ForceFieldType::Custom;
        settings.use_ewald = false;
        settings.constrain_cell_strain_by_lattice = false;

        CrystalOptimizer optimizer(crystal, multipoles, settings);
        Vec p0 = optimizer.get_parameters();
        REQUIRE(p0.size() == 12); // 6 molecular + full 6 cell DOF
        Vec p1 = p0;
        const int cell_offset = static_cast<int>(p1.size()) - 6;
        p1[cell_offset + 3] = 3e-4; // E4
        p1[cell_offset + 4] = -2e-4; // E5

        optimizer.set_parameters(p1);
        Vec p2 = optimizer.get_parameters();
        CHECK_THAT(p2[cell_offset + 3], WithinAbs(p1[cell_offset + 3], 1e-12));
        CHECK_THAT(p2[cell_offset + 4], WithinAbs(p1[cell_offset + 4], 1e-12));
    }
}

TEST_CASE("CrystalOptimizer: cell strain preserves fractional COM",
          "[crystal_optimizer][cell]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    CrystalOptimizerSettings settings;
    settings.use_symmetry = false;
    settings.optimize_cell = true;
    settings.force_field = ForceFieldType::Custom;
    settings.use_ewald = false;

    CrystalOptimizer optimizer(crystal, multipoles, settings);
    setup_crystal_energy_from_dmacrys(optimizer.energy_calculator(), input, crystal, multipoles);
    for (const auto& [key, p] : buck_params) {
        optimizer.energy_calculator().set_buckingham_params(key.first, key.second, p);
    }
    optimizer.reinitialize_states();

    Vec params = optimizer.get_parameters();
    int cell_offset = params.size() - 6;

    const auto& crystal0 = optimizer.energy_calculator().crystal();
    Vec3 frac0 = crystal0.to_fractional(optimizer.states()[1].position);
    Mat3 direct0 = crystal0.unit_cell().direct();

    params[cell_offset + 0] = 8e-4;
    params[cell_offset + 1] = -6e-4;
    params[cell_offset + 2] = 5e-4;
    params[cell_offset + 5] = 2e-4;
    optimizer.set_parameters(params);

    const auto& crystal1 = optimizer.energy_calculator().crystal();
    Vec3 frac1 = crystal1.to_fractional(optimizer.states()[1].position);
    Mat3 direct1 = crystal1.unit_cell().direct();

    CHECK_FALSE(direct0.isApprox(direct1, 1e-12));
    Vec3 frac_diff = frac1 - frac0;
    frac_diff -= frac_diff.array().round().matrix();
    INFO("frac0 = " << frac0.transpose());
    INFO("frac1 = " << frac1.transpose());
    INFO("frac_diff = " << frac_diff.transpose());
    CHECK(frac_diff.norm() < 2e-4);
}
