#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/mults/derivative_transform.h>
#include <occ/mults/sfunctions.h>
#include <iostream>
#include <fmt/core.h>

using namespace occ::mults;
using namespace occ;
using Catch::Approx;

TEST_CASE("DerivativeTransform - Basic matrix dimensions", "[mults][derivatives]") {
    Vec3 ra(-2.0, -2.0, -2.0);
    Vec3 rb(2.0, 2.0, 2.0);

    auto coords = CoordinateSystem::from_points(ra, rb);

    SECTION("D1 matrix dimensions") {
        Mat D1 = DerivativeTransform::compute_D1(coords);

        REQUIRE(D1.rows() == DerivativeTransform::NUM_INTERMEDIATE_VARS);
        REQUIRE(D1.cols() == DerivativeTransform::NUM_EXTERNAL_COORDS);
    }

    SECTION("D2 tensor dimensions") {
        auto D2 = DerivativeTransform::compute_D2(coords);

        REQUIRE(D2.size() == DerivativeTransform::NUM_INTERMEDIATE_VARS);
        for (const auto& mat : D2) {
            REQUIRE(mat.rows() == DerivativeTransform::NUM_EXTERNAL_COORDS);
            REQUIRE(mat.cols() == DerivativeTransform::NUM_EXTERNAL_COORDS);
        }
    }
}

TEST_CASE("DerivativeTransform - Symmetric index packing", "[mults][derivatives]") {
    SECTION("Unpack symmetric indices") {
        // Test a few known cases
        auto [i0, j0] = DerivativeTransform::unpack_symmetric_index(0);
        REQUIRE(i0 == 0);
        REQUIRE(j0 == 0);

        auto [i1, j1] = DerivativeTransform::unpack_symmetric_index(1);
        REQUIRE(i1 == 1);
        REQUIRE(j1 == 0);

        auto [i2, j2] = DerivativeTransform::unpack_symmetric_index(2);
        REQUIRE(i2 == 1);
        REQUIRE(j2 == 1);

        auto [i3, j3] = DerivativeTransform::unpack_symmetric_index(3);
        REQUIRE(i3 == 2);
        REQUIRE(j3 == 0);
    }

    SECTION("Pack-unpack roundtrip") {
        for (int i = 0; i < 15; ++i) {
            for (int j = 0; j <= i; ++j) {
                int kq = DerivativeTransform::pack_symmetric_index(i, j);
                auto [ii, jj] = DerivativeTransform::unpack_symmetric_index(kq);
                REQUIRE(ii == i);
                REQUIRE(jj == j);
            }
        }
    }
}

TEST_CASE("DerivativeTransform - D1 point multipoles", "[mults][derivatives]") {
    // Test case: point multipoles at specific positions
    Vec3 ra(-2.0, -2.0, -2.0);
    Vec3 rb(2.0, 2.0, 2.0);

    auto coords = CoordinateSystem::from_points(ra, rb);
    Mat D1 = DerivativeTransform::compute_D1(coords);

    const double r = coords.r;

    SECTION("Distance derivatives") {
        // ∂R/∂x_A should be -rab/r
        REQUIRE(D1(15, 0) == Approx(-coords.rab[0] / r));
        REQUIRE(D1(15, 1) == Approx(-coords.rab[1] / r));
        REQUIRE(D1(15, 2) == Approx(-coords.rab[2] / r));

        // ∂R/∂x_B should be +rab/r
        REQUIRE(D1(15, 6) == Approx(coords.rab[0] / r));
        REQUIRE(D1(15, 7) == Approx(coords.rab[1] / r));
        REQUIRE(D1(15, 8) == Approx(coords.rab[2] / r));
    }

    SECTION("Unit vector derivatives for point multipoles") {
        // For point multipoles (a=0, b=0), translation derivatives simplify
        // ∂(e1r_x)/∂x_A should be approximately -1/r (for unit vector in x)
        // More precisely: ∂(er_i)/∂x_j = [delta_ij - er_i * er_j] / r

        double er_x = coords.er[0];
        double er_y = coords.er[1];
        double er_z = coords.er[2];

        // e1r = +er for our convention
        // ∂(e1r_x)/∂x_A_x = ∂(er_x)/∂x_A_x = -(1 - er_x²) / r
        double expected_dex_dxa = -(1.0 - er_x * er_x) / r;
        REQUIRE(D1(0, 0) == Approx(expected_dex_dxa).margin(1e-10));
    }

    SECTION("Torque derivatives for point multipoles") {
        // For point multipoles (a=0, b=0), torque derivatives should be zero
        // because there's no lever arm

        // Check some torque derivative components
        REQUIRE(D1(15, 3) == Approx(0.0).margin(1e-10));  // ∂R/∂tau_A_x
        REQUIRE(D1(15, 4) == Approx(0.0).margin(1e-10));  // ∂R/∂tau_A_y
        REQUIRE(D1(15, 5) == Approx(0.0).margin(1e-10));  // ∂R/∂tau_A_z
    }
}

TEST_CASE("DerivativeTransform - D1 with molecular structure", "[mults][derivatives]") {
    // Test with non-zero site vectors (molecular structure)
    Vec3 ra(0.0, 0.0, 0.0);
    Vec3 rb(5.0, 0.0, 0.0);
    Vec3 a(0.0, 1.0, 0.0);  // Site A offset from COM (perpendicular to rab)
    Vec3 b(0.0, 0.0, 1.0);  // Site B offset from COM (perpendicular to rab)

    auto coords = CoordinateSystem::from_points(ra, rb);
    Mat D1 = DerivativeTransform::compute_D1(coords, a, b);

    SECTION("Torque derivatives are non-zero with structure") {
        // With non-zero a and b perpendicular to rab, torque derivatives should be non-zero
        // ∂R/∂tau_A involves rab × a, which should be non-zero when a ⊥ rab

        double torque_deriv_magnitude = std::sqrt(
            D1(15, 3) * D1(15, 3) +
            D1(15, 4) * D1(15, 4) +
            D1(15, 5) * D1(15, 5)
        );

        REQUIRE(torque_deriv_magnitude > 1e-10);
    }
}

TEST_CASE("DerivativeTransform - D1S transformation", "[mults][derivatives]") {
    Vec3 ra(-2.0, -2.0, -2.0);
    Vec3 rb(2.0, 2.0, 2.0);

    auto coords = CoordinateSystem::from_points(ra, rb);
    Mat D1 = DerivativeTransform::compute_D1(coords);

    SECTION("D1S matrix dimensions") {
        // Create a dummy S1 matrix [15 x nmax]
        int nmax = 10;
        Mat S1 = Mat::Random(DerivativeTransform::NUM_FIRST_DERIV_VARS, nmax);

        Mat D1S = DerivativeTransform::compute_D1S(S1, D1);

        REQUIRE(D1S.rows() == DerivativeTransform::NUM_EXTERNAL_COORDS);
        REQUIRE(D1S.cols() == nmax);
    }

    SECTION("D1S transformation correctness") {
        // Use a simple S1 matrix with known values
        int nmax = 3;
        Mat S1 = Mat::Zero(DerivativeTransform::NUM_FIRST_DERIV_VARS, nmax);

        // S-function 0: derivative only w.r.t. q(0) = e1r_x
        S1(0, 0) = 1.0;

        // S-function 1: derivative only w.r.t. q(15) = r (but this is excluded in D1S)

        // S-function 2: derivative w.r.t. q(1) = e1r_y
        S1(1, 2) = 2.0;

        Mat D1S = DerivativeTransform::compute_D1S(S1, D1);

        // D1S(ip, 0) should be D1(0, ip) since S1(0,0) = 1 and others are 0
        for (int ip = 0; ip < DerivativeTransform::NUM_EXTERNAL_COORDS; ++ip) {
            REQUIRE(D1S(ip, 0) == Approx(D1(0, ip)));
        }

        // D1S(ip, 2) should be 2 * D1(1, ip)
        for (int ip = 0; ip < DerivativeTransform::NUM_EXTERNAL_COORDS; ++ip) {
            REQUIRE(D1S(ip, 2) == Approx(2.0 * D1(1, ip)));
        }
    }
}

TEST_CASE("DerivativeTransform - D2 symmetry", "[mults][derivatives]") {
    Vec3 ra(-2.0, -2.0, -2.0);
    Vec3 rb(2.0, 2.0, 2.0);

    auto coords = CoordinateSystem::from_points(ra, rb);
    auto D2 = DerivativeTransform::compute_D2(coords);

    SECTION("D2 matrices are symmetric") {
        // Second derivatives should be symmetric: ∂²f/∂x∂y = ∂²f/∂y∂x
        for (int iq = 0; iq < DerivativeTransform::NUM_INTERMEDIATE_VARS; ++iq) {
            for (int i = 0; i < DerivativeTransform::NUM_EXTERNAL_COORDS; ++i) {
                for (int j = 0; j < DerivativeTransform::NUM_EXTERNAL_COORDS; ++j) {
                    REQUIRE(D2[iq](i, j) == Approx(D2[iq](j, i)).margin(1e-12));
                }
            }
        }
    }
}

TEST_CASE("DerivativeTransform - Numerical derivative validation", "[mults][derivatives]") {
    // Validate D1 against numerical derivatives
    Vec3 ra(1.0, 2.0, 3.0);
    Vec3 rb(4.0, 5.0, 6.0);

    auto coords = CoordinateSystem::from_points(ra, rb);
    Mat D1 = DerivativeTransform::compute_D1(coords);

    const double h = 1e-6;  // Finite difference step

    SECTION("Distance derivative numerical check") {
        // ∂R/∂x_A_x
        Vec3 ra_plus = ra;
        ra_plus[0] += h;
        auto coords_plus = CoordinateSystem::from_points(ra_plus, rb);

        double numerical_deriv = (coords_plus.r - coords.r) / h;
        double analytical_deriv = D1(15, 0);

        REQUIRE(analytical_deriv == Approx(numerical_deriv).epsilon(1e-5));
    }

    SECTION("Unit vector derivative numerical check") {
        // ∂(e1r_y)/∂y_B
        Vec3 rb_plus = rb;
        rb_plus[1] += h;
        auto coords_plus = CoordinateSystem::from_points(ra, rb_plus);

        double numerical_deriv = (coords_plus.ray() - coords.ray()) / h;
        double analytical_deriv = D1(1, 7);  // q(2) = e1r_y, coord 7 = y_B

        REQUIRE(analytical_deriv == Approx(numerical_deriv).epsilon(1e-5));
    }
}

TEST_CASE("DerivativeTransform - Integration with SFunctions", "[mults][derivatives][integration]") {
    // Test the full pipeline: SFunctions → S1/S2 → D1S/D2S transformation

    Vec3 ra(-2.0, -2.0, -2.0);
    Vec3 rb(2.0, 2.0, 2.0);

    SFunctions sfuncs(4);  // Max rank 4
    sfuncs.set_coordinates(ra, rb);

    // Compute a simple S-function with derivatives
    auto result = sfuncs.compute_s_function(0, 0, 0, 2);  // Charge-charge with 2nd derivs

    SECTION("S-function has correct derivative dimensions") {
        REQUIRE(result.s1.size() == SFunctions::NUM_FIRST_DERIVS);
        REQUIRE(result.s2.size() == SFunctions::NUM_SECOND_DERIVS);
    }

    SECTION("Can transform S-function derivatives") {
        auto coords = sfuncs.coordinate_system();
        Mat D1 = DerivativeTransform::compute_D1(coords);

        // Create S1 matrix from single S-function
        Mat S1 = Mat::Zero(DerivativeTransform::NUM_FIRST_DERIV_VARS, 1);
        for (int i = 0; i < DerivativeTransform::NUM_FIRST_DERIV_VARS; ++i) {
            S1(i, 0) = result.s1[i];
        }

        Mat D1S = DerivativeTransform::compute_D1S(S1, D1);

        // D1S should be [12 x 1] - derivatives w.r.t. external coords
        REQUIRE(D1S.rows() == 12);
        REQUIRE(D1S.cols() == 1);

        // For charge-charge (constant S-function), derivatives should be zero
        // or very small
        for (int i = 0; i < 12; ++i) {
            REQUIRE(D1S(i, 0) == Approx(0.0).margin(1e-10));
        }
    }
}

TEST_CASE("DerivativeTransform - Real multipole interaction", "[mults][derivatives][integration]") {
    // Test with a real multipole interaction that has non-zero derivatives

    Vec3 ra(0.0, 0.0, 0.0);
    Vec3 rb(3.0, 0.0, 0.0);

    SFunctions sfuncs(2);
    sfuncs.set_coordinates(ra, rb);

    // Charge-dipole interaction (should have non-zero derivatives)
    auto result = sfuncs.compute_s_function(0, 1, 1, 1);  // level=1 for first derivs

    SECTION("Charge-dipole has non-zero S1 derivatives") {
        double s1_norm = result.s1.norm();
        REQUIRE(s1_norm > 1e-10);  // Should have non-zero derivatives
    }

    SECTION("Transform to Cartesian derivatives") {
        auto coords = sfuncs.coordinate_system();
        Mat D1 = DerivativeTransform::compute_D1(coords);

        Mat S1 = Mat::Zero(DerivativeTransform::NUM_FIRST_DERIV_VARS, 1);
        for (int i = 0; i < DerivativeTransform::NUM_FIRST_DERIV_VARS; ++i) {
            S1(i, 0) = result.s1[i];
        }

        Mat D1S = DerivativeTransform::compute_D1S(S1, D1);

        // Should have non-zero Cartesian derivatives
        double d1s_norm = D1S.norm();
        REQUIRE(d1s_norm > 1e-10);

        // Print for inspection (optional, can comment out)
        fmt::print("\nCharge-dipole S-function:\n");
        fmt::print("  S0 = {:.6e}\n", result.s0);
        fmt::print("  ||D1S|| = {:.6e}\n", d1s_norm);
        fmt::print("  Cartesian derivatives (first 6):\n");
        for (int i = 0; i < 6; ++i) {
            fmt::print("    dS/dx[{}] = {:.6e}\n", i, D1S(i, 0));
        }
    }
}
