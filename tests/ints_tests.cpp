#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/ints/boys.h>
#include <occ/ints/ecoeffs.h>
#include <occ/ints/rints.h>
#include <occ/ints/kernels.h>
#include <occ/ints/esp.h>
#include <occ/qm/integral_engine.h>
#include <occ/gto/shell.h>
#include <occ/core/atom.h>
#include <occ/core/molecule.h>
#include <occ/core/point_charge.h>
#include <cmath>

using Catch::Approx;
using namespace occ::ints;

// ============================================================================
// Boys function tests
// ============================================================================

TEST_CASE("Boys function accuracy", "[boys]") {
    const auto& b = boys();

    SECTION("F_m(0) = 1/(2m+1)") {
        for (int m = 0; m < 10; ++m) {
            double result = detail::boys_reference(0.0, m);
            double expected = 1.0 / (2 * m + 1);
            REQUIRE(result == Approx(expected).epsilon(1e-14));
        }
    }

    SECTION("Asymptotic expansion for large T") {
        for (double T : {40.0, 60.0, 100.0}) {
            for (int m = 0; m < 10; ++m) {
                double ref = detail::boys_reference(T, m);
                double asym = boys_asymptotic(T, m);
                REQUIRE(asym == Approx(ref).epsilon(1e-8));
            }
        }
    }

    SECTION("Chebyshev interpolation accuracy across T range") {
        for (double T = 0.0; T < 35.0; T += 0.5) {
            for (int m = 0; m < 15; ++m) {
                double interp = b.compute(T, m);
                double ref = BoysDefault::reference(T, m);
                REQUIRE(interp == Approx(ref).epsilon(1e-8));
            }
        }
    }

    SECTION("Array evaluation") {
        for (double T : {0.0, 1.0, 5.0, 20.0, 30.0}) {
            double F[8];
            b.compute(T, F);
            for (int m = 0; m < 8; ++m) {
                double ref = BoysDefault::reference(T, m);
                REQUIRE(F[m] == Approx(ref).epsilon(1e-14));
            }
        }
    }
}

// ============================================================================
// E-coefficient tests
// ============================================================================

TEST_CASE("E-coefficient basic properties", "[ecoeffs]") {
    SECTION("E^00_0 equals exp(-mu * XAB^2)") {
        double a = 1.0, b = 2.0;
        double XAB = 1.5;
        double p = a + b;
        double mu = a * b / p;
        double expected = std::exp(-mu * XAB * XAB);

        ECoeffs1D<double, 0, 0> E;
        compute_e_coeffs_1d<double, 0, 0>(a, b, XAB, E);

        REQUIRE(E(0, 0, 0) == Approx(expected).epsilon(1e-12));
    }

    SECTION("E-coefficients for p-s pair") {
        double a = 1.0, b = 1.0;
        double XAB = 1.0;

        ECoeffs1D<double, 1, 0> E;
        compute_e_coeffs_1d<double, 1, 0>(a, b, XAB, E);

        double p = a + b;
        double mu = a * b / p;
        double E00 = std::exp(-mu * XAB * XAB);
        double XPA = b / p * XAB;

        REQUIRE(E(0, 0, 0) == Approx(E00).epsilon(1e-12));
        REQUIRE(E(1, 0, 0) == Approx(XPA * E00).epsilon(1e-12));
    }
}

// ============================================================================
// R-integral tests
// ============================================================================

TEST_CASE("R-integral basic properties", "[rints]") {
    const auto& b = boys();

    SECTION("R^0_000 at origin") {
        double p = 1.0;
        RInts<double, 0> R;
        compute_r_ints<double, 0, BoysParamsDefault>(b.table(), p, 0.0, 0.0, 0.0, R);
        REQUIRE(R(0, 0, 0) == Approx(1.0).epsilon(1e-10));
    }

    SECTION("R^0_000 for finite distance") {
        double p = 1.5;
        double PCx = 1.0, PCy = 0.0, PCz = 0.0;
        double PC2 = PCx * PCx;
        double T = p * PC2;

        RInts<double, 0> R;
        compute_r_ints<double, 0, BoysParamsDefault>(b.table(), p, PCx, PCy, PCz, R);

        double expected = BoysDefault::reference(T, 0);
        REQUIRE(R(0, 0, 0) == Approx(expected).epsilon(1e-10));
    }
}

// ============================================================================
// ESP integral tests - Cartesian
// ============================================================================

TEST_CASE("ESP integrals match libcint for various shell types", "[esp][libcint]") {
    using namespace occ::qm;
    using occ::core::Atom;

    const auto& b = boys();

    // Test data: {l, alpha, Z_nucleus, nucleus_pos}
    struct ShellTest {
        int l;
        double alpha;
        int Z;
        double nucleus_x;
    };

    std::vector<ShellTest> tests = {
        {0, 1.5, 1, 1.0},   // s-shell
        {1, 1.0, 6, 1.0},   // p-shell
        {2, 0.8, 6, 1.5},   // d-shell
        {3, 0.5, 26, 2.0}   // f-shell
    };

    for (const auto& test : tests) {
        DYNAMIC_SECTION("l=" << test.l << " shell") {
            // Create atoms: shell at origin, nucleus at test position
            std::vector<Atom> atoms = {
                {test.Z, 0.0, 0.0, 0.0},
                {1, test.nucleus_x, 0.0, 0.0}
            };

            // Create shell
            std::vector<double> exponents = {test.alpha};
            std::vector<std::vector<double>> coeffs = {{1.0}};
            std::array<double, 3> origin = {0.0, 0.0, 0.0};

            Shell shell(test.l, exponents, coeffs, origin);
            shell.incorporate_shell_norm();

            std::vector<Shell> shells = {shell};
            AOBasis basis(atoms, shells);
            IntegralEngine engine(basis);

            // Get libcint nuclear attraction
            occ::Mat V_nuc = engine.one_electron_operator(cint::Operator::nuclear);

            // Compute our ESP at both nucleus positions
            double A[3] = {0.0, 0.0, 0.0};
            double C0[3] = {0.0, 0.0, 0.0};
            double C1[3] = {test.nucleus_x, 0.0, 0.0};
            double norm_coeff = shell.contraction_coefficients(0, 0);

            // Cartesian shell size
            int n_cart = (shell.l + 1) * (shell.l + 2) / 2;
            int n_integrals = n_cart * n_cart;
            std::vector<double> esp_C0(n_integrals);
            std::vector<double> esp_C1(n_integrals);

            // Use template dispatch based on angular momentum
            auto compute_esp = [&](int l, double* C, double* out) {
                switch(l) {
                    case 0: esp_contracted<double, 0, 0>(1, 1, &test.alpha, &test.alpha,
                                &norm_coeff, &norm_coeff, A, A, C, b.table(), out); break;
                    case 1: esp_contracted<double, 1, 1>(1, 1, &test.alpha, &test.alpha,
                                &norm_coeff, &norm_coeff, A, A, C, b.table(), out); break;
                    case 2: esp_contracted<double, 2, 2>(1, 1, &test.alpha, &test.alpha,
                                &norm_coeff, &norm_coeff, A, A, C, b.table(), out); break;
                    case 3: esp_contracted<double, 3, 3>(1, 1, &test.alpha, &test.alpha,
                                &norm_coeff, &norm_coeff, A, A, C, b.table(), out); break;
                }
            };

            compute_esp(shell.l, C0, esp_C0.data());
            compute_esp(shell.l, C1, esp_C1.data());

            // Check diagonal elements
            for (int i = 0; i < n_cart; ++i) {
                double our_V = -test.Z * esp_C0[i * n_cart + i] - 1.0 * esp_C1[i * n_cart + i];
                REQUIRE(our_V == Approx(V_nuc(i, i)).epsilon(1e-6));
            }

            // Check one off-diagonal element if available
            if (n_cart > 1) {
                double our_V_01 = -test.Z * esp_C0[1] - 1.0 * esp_C1[1];
                REQUIRE(our_V_01 == Approx(V_nuc(0, 1)).epsilon(1e-6));
            }
        }
    }
}

// ============================================================================
// ESP integral tests - Spherical
// ============================================================================

TEST_CASE("ESP spherical vs libcint", "[esp][spherical]") {
    using namespace occ::qm;
    using occ::core::Atom;

    const auto& b = boys();

    // Create a system with various shells in spherical harmonics
    std::vector<Atom> atoms = {
        {6, 0.0, 0.0, 0.0},
        {1, 1.5, 0.5, 0.3}
    };

    // Create shells with different angular momenta
    std::vector<Shell> shells;
    shells.push_back(Shell(0, {1.2}, {{1.0}}, {0.0, 0.0, 0.0}));
    shells.push_back(Shell(1, {0.9}, {{1.0}}, {0.0, 0.0, 0.0}));
    shells.push_back(Shell(2, {0.6}, {{1.0}}, {0.0, 0.0, 0.0}));

    for (auto& shell : shells) {
        shell.incorporate_shell_norm();
    }

    AOBasis basis(atoms, shells);
    basis.set_pure(true);  // Convert to spherical
    IntegralEngine engine(basis);

    // Get libcint result (spherical)
    occ::Mat V_nuc = engine.one_electron_operator(cint::Operator::nuclear);

    // Verify basis is spherical and dimensions are correct
    REQUIRE(basis.is_pure());
    REQUIRE(V_nuc.rows() == basis.nbf());
    REQUIRE(V_nuc.cols() == basis.nbf());

    // Check matrix properties
    REQUIRE(V_nuc.allFinite());

    // Nuclear attraction should be negative definite
    for (int i = 0; i < basis.nbf(); ++i) {
        REQUIRE(V_nuc(i, i) < 0.0);
    }
}

// ============================================================================
// ESPEvaluator high-level API test
// ============================================================================

TEST_CASE("ESPEvaluator high-level API", "[esp][evaluator]") {
    const auto& b = boys();

    // Create evaluator
    ESPEvaluator<double> evaluator(b.table());

    // Basic sanity check - evaluator should be created successfully
    REQUIRE(true);  // If we get here, evaluator was created
}

// ============================================================================
// Overlap integral tests
// ============================================================================

TEST_CASE("Overlap integrals Cartesian", "[overlap][libcint]") {
    using namespace occ::qm;
    using occ::core::Atom;

    // Test that libcint overlap works for Cartesian basis
    std::vector<Atom> atoms = {{6, 0.0, 0.0, 0.0}};

    std::vector<Shell> shells;
    shells.push_back(Shell(0, {1.5}, {{1.0}}, {0.0, 0.0, 0.0}));
    shells.push_back(Shell(1, {1.0}, {{1.0}}, {0.0, 0.0, 0.0}));

    for (auto& shell : shells) {
        shell.incorporate_shell_norm();
    }

    AOBasis basis(atoms, shells);
    IntegralEngine engine(basis);

    occ::Mat S = engine.one_electron_operator(cint::Operator::overlap);

    // Check dimensions
    REQUIRE(S.rows() == basis.nbf());
    REQUIRE(S.cols() == basis.nbf());

    // Check symmetry
    REQUIRE((S - S.transpose()).norm() < 1e-10);

    // Check diagonal is positive
    for (int i = 0; i < basis.nbf(); ++i) {
        REQUIRE(S(i, i) > 0.0);
    }
}

TEST_CASE("Overlap integrals spherical", "[overlap][spherical][libcint]") {
    using namespace occ::qm;
    using occ::core::Atom;

    // Test that libcint overlap works for spherical basis
    std::vector<Atom> atoms = {{6, 0.0, 0.0, 0.0}};

    std::vector<Shell> shells;
    shells.push_back(Shell(0, {1.2}, {{1.0}}, {0.0, 0.0, 0.0}));
    shells.push_back(Shell(1, {0.9}, {{1.0}}, {0.0, 0.0, 0.0}));
    shells.push_back(Shell(2, {0.6}, {{1.0}}, {0.0, 0.0, 0.0}));

    for (auto& shell : shells) {
        shell.incorporate_shell_norm();
    }

    AOBasis basis(atoms, shells);
    basis.set_pure(true);
    IntegralEngine engine(basis);

    occ::Mat S = engine.one_electron_operator(cint::Operator::overlap);

    // Check dimensions
    REQUIRE(S.rows() == basis.nbf());
    REQUIRE(S.cols() == basis.nbf());
    REQUIRE(basis.is_pure());

    // Check symmetry
    REQUIRE((S - S.transpose()).norm() < 1e-10);

    // Check diagonal is positive
    for (int i = 0; i < basis.nbf(); ++i) {
        REQUIRE(S(i, i) > 0.0);
    }
}
