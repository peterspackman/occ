#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/mults/short_range.h>
#include <cmath>

using namespace occ;
using namespace occ::mults;
using Approx = Catch::Approx;

// Numerical derivative helper function
template<typename Func>
double numerical_derivative(Func f, double x, double h = 1e-6) {
    return (f(x + h) - f(x - h)) / (2.0 * h);
}

// Numerical second derivative helper function
template<typename Func>
double numerical_second_derivative(Func f, double x, double h = 1e-5) {
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
}

// ==================== Lennard-Jones Tests ====================

TEST_CASE("Lennard-Jones potential - basic properties", "[short_range][lennard_jones]") {
    LennardJonesParams params{.epsilon = 0.238, .sigma = 3.40};

    SECTION("Energy at equilibrium distance") {
        // Minimum is at r = 2^(1/6) * σ ≈ 1.122 * σ
        const double r_eq = std::pow(2.0, 1.0/6.0) * params.sigma;
        const double energy = ShortRangeInteraction::lennard_jones_energy(r_eq, params);

        // At equilibrium, energy = -ε
        REQUIRE(energy == Approx(-params.epsilon).margin(1e-12));
    }

    SECTION("Energy at sigma") {
        // At r = σ, energy should be zero
        const double energy = ShortRangeInteraction::lennard_jones_energy(params.sigma, params);
        REQUIRE(energy == Approx(0.0).margin(1e-12));
    }

    SECTION("Energy is repulsive at short distances") {
        const double r_short = 0.5 * params.sigma;
        const double energy = ShortRangeInteraction::lennard_jones_energy(r_short, params);
        REQUIRE(energy > 0.0);
    }

    SECTION("Energy is attractive at long distances") {
        const double r_long = 2.0 * params.sigma;
        const double energy = ShortRangeInteraction::lennard_jones_energy(r_long, params);
        REQUIRE(energy < 0.0);
    }

    SECTION("Energy approaches zero at large r") {
        const double r_large = 10.0 * params.sigma;
        const double energy = ShortRangeInteraction::lennard_jones_energy(r_large, params);
        REQUIRE(std::abs(energy) < 1e-6);
    }
}

TEST_CASE("Lennard-Jones potential - derivatives", "[short_range][lennard_jones]") {
    LennardJonesParams params{.epsilon = 0.238, .sigma = 3.40};

    SECTION("First derivative matches numerical derivative") {
        std::vector<double> test_distances = {2.5, 3.0, 3.5, 4.0, 5.0};

        for (double r : test_distances) {
            auto energy_func = [&](double x) {
                return ShortRangeInteraction::lennard_jones_energy(x, params);
            };

            double analytical = ShortRangeInteraction::lennard_jones_derivative(r, params);
            double numerical = numerical_derivative(energy_func, r);

            INFO("Testing at r = " << r);
            REQUIRE(analytical == Approx(numerical).epsilon(1e-6));
        }
    }

    SECTION("Second derivative matches numerical derivative") {
        std::vector<double> test_distances = {2.5, 3.0, 3.5, 4.0, 5.0};

        for (double r : test_distances) {
            auto energy_func = [&](double x) {
                return ShortRangeInteraction::lennard_jones_energy(x, params);
            };

            double analytical = ShortRangeInteraction::lennard_jones_second_derivative(r, params);
            double numerical = numerical_second_derivative(energy_func, r);

            INFO("Testing at r = " << r);
            REQUIRE(analytical == Approx(numerical).epsilon(1e-4));
        }
    }

    SECTION("Derivative is zero at equilibrium") {
        const double r_eq = std::pow(2.0, 1.0/6.0) * params.sigma;
        const double deriv = ShortRangeInteraction::lennard_jones_derivative(r_eq, params);
        REQUIRE(deriv == Approx(0.0).margin(1e-10));
    }

    SECTION("Derivative is negative at short distances (repulsive regime)") {
        // At r < r_eq, dV/dr < 0 (energy decreases toward minimum)
        const double r_short = 0.8 * params.sigma;
        const double deriv = ShortRangeInteraction::lennard_jones_derivative(r_short, params);
        REQUIRE(deriv < 0.0);
    }

    SECTION("Derivative is positive at long distances (attractive regime)") {
        // At r > r_eq, dV/dr > 0 (energy increases away from minimum)
        const double r_long = 2.0 * params.sigma;
        const double deriv = ShortRangeInteraction::lennard_jones_derivative(r_long, params);
        REQUIRE(deriv > 0.0);
    }
}

TEST_CASE("Lennard-Jones potential - combined calculation", "[short_range][lennard_jones]") {
    LennardJonesParams params{.epsilon = 0.238, .sigma = 3.40};

    SECTION("lennard_jones_all returns consistent results") {
        std::vector<double> test_distances = {2.5, 3.0, 3.5, 4.0, 5.0};

        for (double r : test_distances) {
            auto result = ShortRangeInteraction::lennard_jones_all(r, params);

            double energy = ShortRangeInteraction::lennard_jones_energy(r, params);
            double deriv = ShortRangeInteraction::lennard_jones_derivative(r, params);
            double deriv2 = ShortRangeInteraction::lennard_jones_second_derivative(r, params);

            INFO("Testing at r = " << r);
            REQUIRE(result.energy == Approx(energy).margin(1e-14));
            REQUIRE(result.first_derivative == Approx(deriv).margin(1e-14));
            REQUIRE(result.second_derivative == Approx(deriv2).margin(1e-14));
        }
    }
}

// ==================== Buckingham Potential Tests ====================

TEST_CASE("Buckingham potential - basic properties", "[short_range][buckingham]") {
    BuckinghamParams params{.A = 1.69e5, .B = 3.60, .C = 99.5};

    SECTION("Energy is repulsive at very short distances") {
        const double r_short = 2.0;
        const double energy = ShortRangeInteraction::buckingham_energy(r_short, params);
        REQUIRE(energy > 0.0);
    }

    SECTION("Energy has minimum (attractive well)") {
        // Find approximate minimum by checking a range
        double r_min = 0.0;
        double e_min = 1e10;

        for (double r = 2.5; r < 5.0; r += 0.1) {
            double e = ShortRangeInteraction::buckingham_energy(r, params);
            if (e < e_min) {
                e_min = e;
                r_min = r;
            }
        }

        REQUIRE(e_min < 0.0);  // Should have attractive well
        REQUIRE(r_min > 0.0);  // Should have positive minimum position
    }

    SECTION("Energy approaches zero at large r") {
        const double r_large = 20.0;
        const double energy = ShortRangeInteraction::buckingham_energy(r_large, params);
        REQUIRE(std::abs(energy) < 1e-3);
    }
}

TEST_CASE("Buckingham potential - derivatives", "[short_range][buckingham]") {
    BuckinghamParams params{.A = 1.69e5, .B = 3.60, .C = 99.5};

    SECTION("First derivative matches numerical derivative") {
        std::vector<double> test_distances = {2.5, 3.0, 3.5, 4.0, 5.0};

        for (double r : test_distances) {
            auto energy_func = [&](double x) {
                return ShortRangeInteraction::buckingham_energy(x, params);
            };

            double analytical = ShortRangeInteraction::buckingham_derivative(r, params);
            double numerical = numerical_derivative(energy_func, r);

            INFO("Testing at r = " << r);
            REQUIRE(analytical == Approx(numerical).epsilon(1e-6));
        }
    }

    SECTION("Second derivative matches numerical derivative") {
        std::vector<double> test_distances = {2.5, 3.0, 3.5, 4.0, 5.0};

        for (double r : test_distances) {
            auto energy_func = [&](double x) {
                return ShortRangeInteraction::buckingham_energy(x, params);
            };

            double analytical = ShortRangeInteraction::buckingham_second_derivative(r, params);
            double numerical = numerical_second_derivative(energy_func, r);

            INFO("Testing at r = " << r);
            REQUIRE(analytical == Approx(numerical).epsilon(1e-4));
        }
    }
}

TEST_CASE("Buckingham potential - combined calculation", "[short_range][buckingham]") {
    BuckinghamParams params{.A = 1.69e5, .B = 3.60, .C = 99.5};

    SECTION("buckingham_all returns consistent results") {
        std::vector<double> test_distances = {2.5, 3.0, 3.5, 4.0, 5.0};

        for (double r : test_distances) {
            auto result = ShortRangeInteraction::buckingham_all(r, params);

            double energy = ShortRangeInteraction::buckingham_energy(r, params);
            double deriv = ShortRangeInteraction::buckingham_derivative(r, params);
            double deriv2 = ShortRangeInteraction::buckingham_second_derivative(r, params);

            INFO("Testing at r = " << r);
            REQUIRE(result.energy == Approx(energy).margin(1e-10));
            REQUIRE(result.first_derivative == Approx(deriv).margin(1e-10));
            REQUIRE(result.second_derivative == Approx(deriv2).margin(1e-10));
        }
    }
}

// ==================== Force and Hessian Tests ====================

TEST_CASE("Force calculation - basic properties", "[short_range][force]") {
    SECTION("Force is zero when derivative is zero") {
        Vec3 r_vec(1.0, 0.0, 0.0);
        double dE_dr = 0.0;

        Vec3 force = ShortRangeInteraction::derivative_to_force(dE_dr, r_vec);
        REQUIRE(force.norm() == Approx(0.0).margin(1e-14));
    }

    SECTION("Force points along displacement vector") {
        Vec3 r_vec(3.0, 4.0, 0.0);  // r = 5.0
        double dE_dr = 2.0;  // Arbitrary positive value

        Vec3 force = ShortRangeInteraction::derivative_to_force(dE_dr, r_vec);

        // Force should be parallel to r_vec
        Vec3 r_normalized = r_vec.normalized();
        Vec3 f_normalized = force.normalized();

        // Check if parallel (dot product = ±1)
        double dot = r_normalized.dot(f_normalized);
        REQUIRE(std::abs(std::abs(dot) - 1.0) < 1e-12);
    }

    SECTION("Force magnitude scales with derivative") {
        Vec3 r_vec(1.0, 0.0, 0.0);

        double dE_dr1 = 1.0;
        double dE_dr2 = 2.0;

        Vec3 force1 = ShortRangeInteraction::derivative_to_force(dE_dr1, r_vec);
        Vec3 force2 = ShortRangeInteraction::derivative_to_force(dE_dr2, r_vec);

        REQUIRE(force2.norm() == Approx(2.0 * force1.norm()).margin(1e-14));
    }
}

TEST_CASE("Force and Hessian - Newton's 3rd law", "[short_range][force][hessian]") {
    Vec3 r_A(0.0, 0.0, 0.0);
    Vec3 r_B(3.5, 0.0, 0.0);

    double dE_dr = 1.5;
    double d2E_dr2 = -0.8;

    auto result = ShortRangeInteraction::compute_force_hessian(r_A, r_B, dE_dr, d2E_dr2);

    SECTION("Forces sum to zero (Newton's 3rd law)") {
        Vec3 total_force = result.force_A + result.force_B;
        REQUIRE(total_force.norm() == Approx(0.0).margin(1e-14));
    }

    SECTION("Forces are equal and opposite") {
        REQUIRE(result.force_A.norm() == Approx(result.force_B.norm()).margin(1e-14));
        REQUIRE(result.force_A.dot(result.force_B) == Approx(-result.force_A.squaredNorm()).margin(1e-14));
    }
}

TEST_CASE("Force and Hessian - symmetry properties", "[short_range][force][hessian]") {
    Vec3 r_A(1.0, 2.0, 3.0);
    Vec3 r_B(4.0, 5.0, 6.0);

    double dE_dr = 1.2;
    double d2E_dr2 = 0.5;

    auto result = ShortRangeInteraction::compute_force_hessian(r_A, r_B, dE_dr, d2E_dr2);

    SECTION("Hessian AA is symmetric") {
        REQUIRE(result.hessian_AA.isApprox(result.hessian_AA.transpose(), 1e-14));
    }

    SECTION("Hessian BB is symmetric") {
        REQUIRE(result.hessian_BB.isApprox(result.hessian_BB.transpose(), 1e-14));
    }

    SECTION("Hessian AB = -Hessian AA") {
        REQUIRE(result.hessian_AB.isApprox(-result.hessian_AA, 1e-14));
    }

    SECTION("Hessian BB = Hessian AA") {
        REQUIRE(result.hessian_BB.isApprox(result.hessian_AA, 1e-14));
    }

    SECTION("Hessians satisfy translational invariance") {
        // Sum of all Hessian blocks should be zero
        Mat3 total = result.hessian_AA + result.hessian_AB +
                     result.hessian_AB.transpose() + result.hessian_BB;
        REQUIRE(total.norm() == Approx(0.0).margin(1e-14));
    }
}

TEST_CASE("Force and Hessian - numerical validation", "[short_range][force][hessian]") {
    LennardJonesParams params{.epsilon = 0.238, .sigma = 3.40};

    Vec3 r_A(0.0, 0.0, 0.0);
    Vec3 r_B(3.5, 0.0, 0.0);

    SECTION("Forces match analytical derivatives") {
        auto result = ShortRangeInteraction::lennard_jones_all(
            (r_B - r_A).norm(), params);

        auto forces = ShortRangeInteraction::compute_force_hessian(
            r_A, r_B, result.first_derivative, result.second_derivative);

        // Analytical force should be -dE/dr along x direction
        double expected_fx = -result.first_derivative;
        REQUIRE(forces.force_A(0) == Approx(expected_fx).margin(1e-14));
        REQUIRE(forces.force_A(1) == Approx(0.0).margin(1e-14));
        REQUIRE(forces.force_A(2) == Approx(0.0).margin(1e-14));
    }

    SECTION("Hessian matches numerical derivatives") {
        // Compute analytical Hessian
        auto result = ShortRangeInteraction::lennard_jones_all(
            (r_B - r_A).norm(), params);
        auto hessian_result = ShortRangeInteraction::compute_force_hessian(
            r_A, r_B, result.first_derivative, result.second_derivative);

        // Numerical Hessian via finite differences
        const double h = 1e-6;

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                // Perturbation vectors
                Vec3 delta_i = Vec3::Zero();
                delta_i(i) = h;

                // Forward difference
                Vec3 r_A_plus = r_A + delta_i;
                double r_plus = (r_B - r_A_plus).norm();
                auto result_plus = ShortRangeInteraction::lennard_jones_all(r_plus, params);
                auto forces_plus = ShortRangeInteraction::derivative_to_force(
                    result_plus.first_derivative, r_B - r_A_plus);

                // Backward difference
                Vec3 r_A_minus = r_A - delta_i;
                double r_minus = (r_B - r_A_minus).norm();
                auto result_minus = ShortRangeInteraction::lennard_jones_all(r_minus, params);
                auto forces_minus = ShortRangeInteraction::derivative_to_force(
                    result_minus.first_derivative, r_B - r_A_minus);

                // Numerical derivative of force component j
                double numerical_hessian_ij = (forces_plus(j) - forces_minus(j)) / (2.0 * h);
                double analytical_hessian_ij = hessian_result.hessian_AA(i, j);

                INFO("Hessian element (" << i << "," << j << ")");
                REQUIRE(analytical_hessian_ij == Approx(numerical_hessian_ij).epsilon(1e-5));
            }
        }
    }
}

TEST_CASE("Integration test - Lennard-Jones with forces", "[short_range][integration]") {
    // Argon-Argon interaction
    LennardJonesParams ar_ar{.epsilon = 0.238, .sigma = 3.40};

    SECTION("Complete workflow at attractive distance") {
        // Use distance clearly beyond equilibrium (r_eq ≈ 3.82Å)
        Vec3 r_A(0.0, 0.0, 0.0);
        Vec3 r_B(4.5, 0.0, 0.0);

        double r = (r_B - r_A).norm();
        auto energy_result = ShortRangeInteraction::lennard_jones_all(r, ar_ar);
        auto force_result = ShortRangeInteraction::compute_force_hessian(
            r_A, r_B, energy_result.first_derivative, energy_result.second_derivative);

        // Check that energy is reasonable
        REQUIRE(energy_result.energy < 0.0);  // Should be attractive at this distance
        REQUIRE(energy_result.energy > -ar_ar.epsilon);  // Should be above minimum

        // At r > r_eq, dV/dr > 0 (energy increases as r increases - pulling apart)
        // Force = -dV/dr * unit_vec pulls particles together
        REQUIRE(energy_result.first_derivative > 0.0);  // dV/dr positive
        REQUIRE(force_result.force_A(0) < 0.0);  // Force on A points toward B (negative x direction)
        REQUIRE(force_result.force_B(0) > 0.0);  // Force on B points toward A (positive x direction)

        // In the attractive region (r > r_eq), d²V/dr² can be negative
        // (the potential is concave near the minimum)
    }

    SECTION("Complete workflow at repulsive distance") {
        // Use distance clearly before equilibrium (r < r_eq ≈ 3.82Å)
        Vec3 r_A(0.0, 0.0, 0.0);
        Vec3 r_B(3.0, 0.0, 0.0);

        double r = (r_B - r_A).norm();
        auto energy_result = ShortRangeInteraction::lennard_jones_all(r, ar_ar);
        auto force_result = ShortRangeInteraction::compute_force_hessian(
            r_A, r_B, energy_result.first_derivative, energy_result.second_derivative);

        // At r < r_eq, dV/dr < 0 (energy decreases as r increases - want to get closer)
        // Force = -dV/dr * unit_vec pushes particles apart
        REQUIRE(energy_result.first_derivative < 0.0);  // dV/dr negative
        REQUIRE(force_result.force_A(0) > 0.0);  // Force on A points away from B (positive x direction)
        REQUIRE(force_result.force_B(0) < 0.0);  // Force on B points away from A (negative x direction)
    }
}

TEST_CASE("Integration test - Buckingham with forces", "[short_range][integration]") {
    // Argon-Argon Buckingham parameters
    BuckinghamParams ar_ar{.A = 1.69e5, .B = 3.60, .C = 99.5};

    Vec3 r_A(0.0, 0.0, 0.0);
    Vec3 r_B(3.5, 0.0, 0.0);

    SECTION("Complete workflow") {
        double r = (r_B - r_A).norm();
        auto energy_result = ShortRangeInteraction::buckingham_all(r, ar_ar);
        auto force_result = ShortRangeInteraction::compute_force_hessian(
            r_A, r_B, energy_result.first_derivative, energy_result.second_derivative);

        // Check that forces sum to zero
        Vec3 total_force = force_result.force_A + force_result.force_B;
        REQUIRE(total_force.norm() == Approx(0.0).margin(1e-12));

        // Check Hessian symmetry
        REQUIRE(force_result.hessian_AA.isApprox(force_result.hessian_AA.transpose(), 1e-14));
    }
}

TEST_CASE("Edge cases - very small distances", "[short_range][edge_cases]") {
    LennardJonesParams params{.epsilon = 0.238, .sigma = 3.40};

    SECTION("Distance below MIN_DISTANCE throws exception") {
        REQUIRE_THROWS_AS(
            ShortRangeInteraction::lennard_jones_energy(1e-12, params),
            std::domain_error);
    }

    SECTION("Very small but valid distance") {
        double r = 1e-9;  // Just above MIN_DISTANCE
        double energy = ShortRangeInteraction::lennard_jones_energy(r, params);
        REQUIRE(std::isfinite(energy));
        REQUIRE(energy > 0.0);  // Should be strongly repulsive
    }
}

TEST_CASE("Edge cases - different coordinate systems", "[short_range][edge_cases]") {
    LennardJonesParams params{.epsilon = 0.238, .sigma = 3.40};

    SECTION("Force direction changes with site positions") {
        Vec3 r_A(1.0, 2.0, 3.0);
        Vec3 r_B1(4.0, 2.0, 3.0);  // Displacement along x
        Vec3 r_B2(1.0, 5.0, 3.0);  // Displacement along y
        Vec3 r_B3(1.0, 2.0, 6.0);  // Displacement along z

        auto result = ShortRangeInteraction::lennard_jones_all(3.0, params);

        auto forces1 = ShortRangeInteraction::compute_force_hessian(
            r_A, r_B1, result.first_derivative, result.second_derivative);
        auto forces2 = ShortRangeInteraction::compute_force_hessian(
            r_A, r_B2, result.first_derivative, result.second_derivative);
        auto forces3 = ShortRangeInteraction::compute_force_hessian(
            r_A, r_B3, result.first_derivative, result.second_derivative);

        // Force magnitudes should be equal (same r)
        REQUIRE(forces1.force_A.norm() == Approx(forces2.force_A.norm()).margin(1e-14));
        REQUIRE(forces2.force_A.norm() == Approx(forces3.force_A.norm()).margin(1e-14));

        // Force directions should differ
        REQUIRE(forces1.force_A.dot(forces2.force_A) == Approx(0.0).margin(1e-14));
        REQUIRE(forces2.force_A.dot(forces3.force_A) == Approx(0.0).margin(1e-14));
    }
}
