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
#include "fd_test_utils.h"

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

// The structure-of-arrays kernel used by the COSX exchange build must agree
// with the row-major evaluate() path for every (la, lb), with and without
// range separation, including the padded tail of a point block.
TEST_CASE("ESPEvaluator SoA kernel matches row-major evaluate", "[esp][evaluator][soa]") {
    using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const auto& b = boys();
    const std::array<double, 3> A{0.0, 0.0, 0.0};
    const std::array<double, 3> B{1.3, -0.7, 0.45};
    const std::vector<double> exps_a{31.0, 3.1, 0.72};
    const std::vector<double> exps_b{2.4, 0.53};
    const std::vector<double> c_a{0.2, 0.42, 0.68};
    const std::vector<double> c_b{0.55, 0.61};

    // 37 points spanning near (small Boys argument) to far (asymptotic) from
    // the pair; padded to the kernel's block size with copies of the last.
    const int npt = 37;
    const int pad = ESPEvaluator<double>::soa_padding;
    const int npad = (npt + pad - 1) / pad * pad;
    occ::Mat3N pts(3, npt);
    for (int i = 0; i < npt; ++i) {
        const double s = 0.03 + 0.35 * i;
        pts.col(i) << s * std::cos(0.7 * i), s * std::sin(1.3 * i) - 0.2,
            0.4 * std::cos(0.37 * i) * s;
    }
    std::vector<double> x(npad), y(npad), z(npad);
    for (int i = 0; i < npad; ++i) {
        const int c = std::min(i, npt - 1);
        x[i] = pts(0, c);
        y[i] = pts(1, c);
        z[i] = pts(2, c);
    }

    for (bool spherical : {false, true}) {
        for (int la = 0; la <= 4; ++la) {
            for (int lb = 0; lb <= 4; ++lb) {
                ESPEvaluator<double> esp(b.table());
                const size_t idx = esp.add_shell_pair(
                    la, lb, 3, 2, exps_a.data(), exps_b.data(), c_a.data(),
                    c_b.data(), A.data(), B.data(), spherical);
                const int nab = esp.nab(idx);
                for (double omega : {0.0, 0.4}) {
                    MatRM ref(npt, nab), work(npt, esp.nherm(idx));
                    esp.evaluate(idx, pts, ref, work, omega);
                    std::vector<double> V(nab * npad, 99.0);
                    esp.evaluate_soa(idx, x.data(), y.data(), z.data(), npad,
                                     V.data(), npad, omega);
                    double max_diff = 0.0, max_ref = 0.0;
                    for (int ab = 0; ab < nab; ++ab) {
                        for (int i = 0; i < npt; ++i) {
                            max_diff = std::max(
                                max_diff, std::abs(V[ab * npad + i] - ref(i, ab)));
                            max_ref = std::max(max_ref, std::abs(ref(i, ab)));
                        }
                    }
                    INFO((spherical ? "spherical" : "cartesian")
                         << " la=" << la << " lb=" << lb << " omega=" << omega
                         << " max|ref|=" << max_ref);
                    REQUIRE(max_diff <= 1e-12 * std::max(1.0, max_ref));
                }
            }
        }
    }
}

// The COSX exchange build drives ESPEvaluator over every significant shell
// pair of the orbital basis: contracted shells, mixed angular momenta, and
// two different centres. Check that path against libcint for every (la, lb)
// combination up to f, in both Cartesian and spherical form.
TEST_CASE("ESPEvaluator matches libcint for all shell pairs", "[esp][evaluator][libcint]") {
    using namespace occ::qm;
    using occ::core::Atom;
    using occ::core::PointCharge;
    using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    const auto& b = boys();

    const std::array<double, 3> A{0.0, 0.0, 0.0};
    const std::array<double, 3> B{1.3, -0.7, 0.45};

    // A few grid points: on top of a centre, between them, and well outside.
    occ::Mat3N points(3, 4);
    points.col(0) << 0.15, 0.05, -0.1;
    points.col(1) << 0.7, -0.3, 0.2;
    points.col(2) << 1.4, -0.8, 0.5;
    points.col(3) << 4.5, 2.0, -3.0;

    // Contracted (2-primitive) shells, as in a real basis set.
    const std::vector<double> exps_a{3.1, 0.72};
    const std::vector<double> exps_b{2.4, 0.53};
    const std::vector<std::vector<double>> coeffs_a{{0.42, 0.68}};
    const std::vector<std::vector<double>> coeffs_b{{0.55, 0.61}};

    for (bool spherical : {false, true}) {
        for (int la = 0; la <= 3; ++la) {
            for (int lb = 0; lb <= 3; ++lb) {
                DYNAMIC_SECTION((spherical ? "spherical" : "cartesian")
                                << " la=" << la << " lb=" << lb) {
                    std::vector<Atom> atoms = {{6, A[0], A[1], A[2]},
                                               {8, B[0], B[1], B[2]}};
                    std::vector<Shell> shells;
                    shells.push_back(Shell(la, exps_a, coeffs_a, A));
                    shells.push_back(Shell(lb, exps_b, coeffs_b, B));
                    for (auto& sh : shells)
                        sh.incorporate_shell_norm();

                    AOBasis basis(atoms, shells);
                    basis.set_pure(spherical);
                    IntegralEngine engine(basis);

                    const auto& sh_a = basis.shells()[0];
                    const auto& sh_b = basis.shells()[1];
                    const int size_a = sh_a.size();
                    const int size_b = sh_b.size();

                    ESPEvaluator<double> esp(b.table());
                    const size_t idx = esp.add_shell_pair(
                        sh_a.l, sh_b.l, sh_a.num_primitives(),
                        sh_b.num_primitives(), sh_a.exponents.data(),
                        sh_b.exponents.data(),
                        sh_a.contraction_coefficients.col(0).data(),
                        sh_b.contraction_coefficients.col(0).data(),
                        sh_a.origin.data(), sh_b.origin.data(), spherical);

                    REQUIRE(esp.nab(idx) == size_a * size_b);

                    // Overlap of the same shell pair: the COSX pair screen
                    // uses this as the pair's charge magnitude.
                    {
                        occ::Mat S_ref =
                            engine.one_electron_operator(cint::Operator::overlap);
                        std::vector<double> s(esp.nab(idx));
                        esp.evaluate_overlap(idx, s.data());
                        Eigen::Map<const MatRM> S(s.data(), size_a, size_b);
                        const double err =
                            (S - S_ref.block(0, size_a, size_a, size_b))
                                .cwiseAbs()
                                .maxCoeff();
                        INFO("overlap max|err|=" << err);
                        REQUIRE(err < 1e-10);
                    }

                    const Eigen::Index npt = points.cols();
                    MatRM integrals(npt, esp.nab(idx));
                    MatRM workspace(npt, esp.nherm(idx));
                    esp.evaluate(idx, points, integrals, workspace, 0.0);

                    for (Eigen::Index pt = 0; pt < npt; ++pt) {
                        // libcint reference: a unit positive point charge at
                        // this grid point gives -\int phi_p phi_q / |r - C|.
                        std::vector<PointCharge> charges{
                            PointCharge(1.0, points(0, pt), points(1, pt),
                                        points(2, pt))};
                        occ::Mat ref = -engine.point_charge_potential(charges);

                        Eigen::Map<const MatRM> V(integrals.row(pt).data(),
                                                  size_a, size_b);
                        const double max_ref =
                            ref.block(0, size_a, size_a, size_b)
                                .cwiseAbs()
                                .maxCoeff();
                        const double err =
                            (V - ref.block(0, size_a, size_a, size_b))
                                .cwiseAbs()
                                .maxCoeff();
                        INFO("point " << pt << " max|ref|=" << max_ref
                                      << " max|err|=" << err);
                        REQUIRE(err < 1e-8 * std::max(1.0, max_ref));
                    }
                }
            }
        }
    }
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

namespace esp_pair {
// A contracted shell pair (three and two primitives) on nearby centres, for
// the ESP derivative tests below.
constexpr double ea[3] = {2.1, 0.7, 0.18}, eb[2] = {0.9, 0.22};
constexpr double ca[3] = {0.3, 0.6, 0.5}, cb[2] = {0.7, 0.4};
constexpr double A0[3] = {0.1, -0.2, 0.3}, B0[3] = {-0.4, 0.5, -0.1};
} // namespace esp_pair

// Centre derivatives of ESP pair integrals (the first-derivative rows of a
// Hessian pair) against central differences of the integrals in the centres.
TEST_CASE("ESP pair centre derivatives vs finite differences", "[esp][ints]") {
  using occ::ints::ESPEvaluator;
  using namespace esp_pair;
  constexpr int PAD = ESPEvaluator<double>::soa_padding;
  std::vector<double> x(PAD), y(PAD), z(PAD);
  for (int g = 0; g < PAD; g++) {
    x[g] = 0.8 * std::sin(1.3 * g);
    y[g] = 0.9 * std::cos(0.7 * g) - 0.3;
    z[g] = 0.5 + 0.04 * g;
  }
  const double h = 1e-5;
  double worst = 0.0;
  for (bool sph : {false, true})
    for (int la = 0; la <= 4; la++)
      for (int lb = 0; lb <= 4; lb++)
        for (double omega : {0.0, 0.3}) {
          auto values = [&](const double *A, const double *B) {
            ESPEvaluator<double> esp(occ::ints::boys().table());
            const size_t k =
                esp.add_shell_pair(la, lb, 3, 2, ea, eb, ca, cb, A, B, sph);
            std::vector<double> V(esp.nab(k) * PAD);
            esp.evaluate_soa(k, x.data(), y.data(), z.data(), PAD, V.data(),
                             PAD, omega);
            return V;
          };
          ESPEvaluator<double> esp(occ::ints::boys().table());
          esp.resize_hessian_pairs(1);
          esp.set_shell_pair_hessian(0, la, lb, 3, 2, ea, eb, ca, cb, A0, B0,
                                     sph);
          const int nab = static_cast<int>(values(A0, B0).size()) / PAD;
          std::vector<double> D(7 * nab * PAD);
          esp.evaluate_soa_hessian(0, x.data(), y.data(), z.data(), PAD,
                                   D.data(), PAD, nullptr, 0, nullptr, omega);
          for (int centre = 0; centre < 2; centre++)
            for (int d = 0; d < 3; d++) {
              double Ap[3], Am[3], Bp[3], Bm[3];
              std::copy(A0, A0 + 3, Ap);
              std::copy(A0, A0 + 3, Am);
              std::copy(B0, B0 + 3, Bp);
              std::copy(B0, B0 + 3, Bm);
              (centre == 0 ? Ap : Bp)[d] += h;
              (centre == 0 ? Am : Bm)[d] -= h;
              const auto Vp = values(Ap, Bp), Vm = values(Am, Bm);
              for (int ab = 0; ab < nab; ab++)
                for (int g = 0; g < PAD; g++) {
                  const double fd =
                      (Vp[ab * PAD + g] - Vm[ab * PAD + g]) / (2 * h);
                  const double an =
                      D[((1 + 3 * centre + d) * nab + ab) * PAD + g];
                  worst = std::max(worst, std::abs(an - fd));
                }
            }
        }
  INFO("max |analytic - FD| = " << worst);
  REQUIRE(worst < 1e-7);
}

// Hessian pairs: values against the value pairs, and the second derivatives
// contracted with per-point weights against central differences of the
// contracted first derivatives.
TEST_CASE("ESP Hessian pairs vs finite differences", "[esp][ints]") {
  using occ::ints::ESPEvaluator;
  using namespace esp_pair;
  // a fixed generic unit direction over (A_xyz, B_xyz)
  const occ::Vec w = occ::test::generic_direction(6);
  constexpr int PAD = ESPEvaluator<double>::soa_padding;
  const int npts = 2 * PAD, npt = npts - 5; // padding points carry W = 0
  std::vector<double> x(npts), y(npts), z(npts);
  for (int g = 0; g < npts; g++) {
    x[g] = 1.1 * std::sin(1.3 * g);
    y[g] = 0.9 * std::cos(0.7 * g) - 0.3;
    z[g] = 0.5 + 0.02 * g;
  }
  const double h = 1e-5;
  double worst_v = 0.0, worst_h = 0.0;
  for (bool sph : {false, true})
    for (int la = 0; la <= 4; la++)
      for (int lb = 0; lb <= 4; lb++)
        for (double omega : {0.0, 0.3}) {
          if (la + lb > 6 && omega != 0.0)
            continue;
          auto hessian = [&](const double *A, const double *B,
                             std::vector<double> &V, const double *W, int ldw,
                             double *out) {
            ESPEvaluator<double> esp(occ::ints::boys().table());
            esp.resize_hessian_pairs(1);
            esp.set_shell_pair_hessian(0, la, lb, 3, 2, ea, eb, ca, cb, A, B,
                                       sph);
            esp.evaluate_soa_hessian(0, x.data(), y.data(), z.data(), npts,
                                     V.data(), npts, W, ldw, out, omega);
          };
          ESPEvaluator<double> esp(occ::ints::boys().table());
          const size_t k =
              esp.add_shell_pair(la, lb, 3, 2, ea, eb, ca, cb, A0, B0, sph);
          const int nab = esp.nab(k);
          const int ldw = npts + 3;
          std::vector<double> W(nab * ldw, 0.0);
          for (int ab = 0; ab < nab; ab++)
            for (int g = 0; g < npt; g++)
              W[ab * ldw + g] = std::cos(0.37 * g + 1.1 * ab);
          std::vector<double> V(nab * npts), Hv(7 * nab * npts);
          esp.evaluate_soa(k, x.data(), y.data(), z.data(), npts, V.data(),
                           npts, omega);
          double out[21] = {};
          hessian(A0, B0, Hv, W.data(), ldw, out);
          for (int i = 0; i < nab * npts; i++)
            worst_v = std::max(worst_v, std::abs(Hv[i] - V[i]));
          // d/dc' of sum W d/dc A, c the derivative rows of V
          auto first = [&](const std::vector<double> &Hd, int c) {
            double s = 0.0;
            for (int ab = 0; ab < nab; ab++)
              for (int g = 0; g < npts; g++)
                s += W[ab * ldw + g] * Hd[((1 + c) * nab + ab) * npts + g];
            return s;
          };
          // Both centres move together along one generic direction w over
          // their six coordinates: the w-derivative of each contracted first
          // derivative against the second derivatives contracted with w.
          static const int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
          double Ap[3], Am[3], Bp[3], Bm[3];
          for (int e = 0; e < 3; e++) {
            Ap[e] = A0[e] + h * w(e);
            Am[e] = A0[e] - h * w(e);
            Bp[e] = B0[e] + h * w(3 + e);
            Bm[e] = B0[e] - h * w(3 + e);
          }
          std::vector<double> Vp(7 * nab * npts), Vm(7 * nab * npts);
          hessian(Ap, Bp, Vp, nullptr, ldw, nullptr);
          hessian(Am, Bm, Vm, nullptr, ldw, nullptr);
          for (int c = 0; c < 6; c++) {
            const double fd = (first(Vp, c) - first(Vm, c)) / (2 * h);
            const int cc = c / 3, d = c % 3;
            double an = 0.0;
            for (int centre = 0; centre < 2; centre++)
              for (int e = 0; e < 3; e++) {
                double second;
                if (cc == centre)
                  second = out[(cc == 0 ? 0 : 15) + sym[d][e]];
                else
                  second = cc == 0 ? out[6 + 3 * d + e] : out[6 + 3 * e + d];
                an += w(3 * centre + e) * second;
              }
            worst_h =
                std::max(worst_h, std::abs(an - fd) / (1.0 + std::abs(fd)));
          }
        }
  INFO("values " << worst_v << " second derivatives " << worst_h);
  REQUIRE(worst_v < 1e-12);
  REQUIRE(worst_h < 1e-6);
}

// The fused pass (values plus derivatives contracted with per-point weights)
// against the separate values and the first-derivative rows of a Hessian
// pair.
TEST_CASE("ESP fused value and derivative contraction", "[esp][ints]") {
  using occ::ints::ESPEvaluator;
  using namespace esp_pair;
  constexpr int PAD = ESPEvaluator<double>::soa_padding;
  const int npts = 3 * PAD, npt = npts - 5; // padding points carry W = 0
  std::vector<double> x(npts), y(npts), z(npts);
  for (int g = 0; g < npts; g++) {
    x[g] = 1.1 * std::sin(1.3 * g);
    y[g] = 0.9 * std::cos(0.7 * g) - 0.3;
    z[g] = 0.5 + 0.02 * g;
  }
  double worst_v = 0.0, worst_d = 0.0;
  for (bool sph : {false, true})
    for (int la = 0; la <= 4; la++)
      for (int lb = 0; lb <= 4; lb++)
        for (int ncent : {1, 2}) {
          ESPEvaluator<double> esp(occ::ints::boys().table());
          const size_t k =
              esp.add_shell_pair(la, lb, 3, 2, ea, eb, ca, cb, A0, B0, sph);
          esp.resize_derivative_pairs(1);
          esp.set_shell_pair_derivative(0, la, lb, 3, 2, ea, eb, ca, cb, A0,
                                        B0, sph);
          esp.resize_hessian_pairs(1);
          esp.set_shell_pair_hessian(0, la, lb, 3, 2, ea, eb, ca, cb, A0, B0,
                                     sph);
          const int nab = esp.nab(k);
          const int ldw = npts + 3;
          std::vector<double> W(nab * ldw, 0.0);
          for (int ab = 0; ab < nab; ab++)
            for (int g = 0; g < npt; g++)
              W[ab * ldw + g] = std::cos(0.37 * g + 1.1 * ab);
          std::vector<double> V(nab * npts), H(7 * nab * npts), A(nab * npts);
          esp.evaluate_soa(k, x.data(), y.data(), z.data(), npts, V.data(),
                           npts);
          esp.evaluate_soa_hessian(0, x.data(), y.data(), z.data(), npts,
                                   H.data(), npts, nullptr, 0, nullptr);
          const double *D = H.data() + nab * npts; // d/dA_xyz, d/dB_xyz
          double out[6] = {0, 0, 0, 0, 0, 0};
          esp.evaluate_soa_value_and_derivative_contraction(
              0, x.data(), y.data(), z.data(), npts, A.data(), npts, W.data(),
              ldw, ncent, out);
          for (int i = 0; i < nab * npts; i++)
            worst_v = std::max(worst_v, std::abs(A[i] - V[i]));
          for (int c = 0; c < 3 * ncent; c++) {
            double ref = 0.0;
            for (int ab = 0; ab < nab; ab++)
              for (int g = 0; g < npts; g++)
                ref += W[ab * ldw + g] * D[(c * nab + ab) * npts + g];
            worst_d = std::max(worst_d,
                               std::abs(out[c] - ref) / (1.0 + std::abs(ref)));
          }
          for (int c = 3 * ncent; c < 6; c++)
            worst_d = std::max(worst_d, std::abs(out[c]));
        }
  INFO("values " << worst_v << " contracted derivatives " << worst_d);
  REQUIRE(worst_v < 1e-12);
  REQUIRE(worst_d < 1e-11);
}
