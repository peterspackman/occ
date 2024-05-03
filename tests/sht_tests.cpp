#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <occ/sht/legendre.h>
#include <occ/sht/sht.h>
#include <occ/sht/spherical_harmonics.h>
#include <occ/sht/clebsch.h>
#include <occ/sht/wigner3j.h>

double func(double theta, double phi) {
    return (1.0 + 0.01 * std::cos(theta) +
            0.1 * (3.0 * std::cos(theta) * std::cos(theta) - 1.0) +
            (std::cos(phi) + 0.3 * std::sin(phi)) * std::sin(theta) +
            (std::cos(2.0 * phi) + 0.1 * std::sin(2.0 * phi)) *
                std::sin(theta) * std::sin(theta) *
                (7.0 * std::cos(theta) * std::cos(theta) - 1.0) * 3.0 / 8.0);
}

TEST_CASE("SHT Analysis complex function", "[sht]") {

    occ::sht::SHT sht(4);
    occ::CMat values = sht.values_on_grid_complex(func);
    occ::CVec expected(sht.nlm());
    expected << std::sqrt(4 * M_PI), // 0 0
        std::complex<double>(std::sqrt(2 * M_PI / 3),
                             0.3 * std::sqrt(2 * M_PI / 3)), // 1 -1
        0.01 * std::sqrt(4 * M_PI / 3.0),                    // 1 0
        std::complex<double>(-std::sqrt(2 * M_PI / 3),
                             0.3 * std::sqrt(2 * M_PI / 3)), // 1 1
        0.0,                                                 // 2 -2
        0.0,                                                 // 2 -1
        0.1 * std::sqrt(16 * M_PI / 5.0),                    // 2  0
        0.0,                                                 // 2  1
        0.0,                                                 // 2  2
        0.0,                                                 // 3 -3
        0.0,                                                 // 3 -2
        0.0,                                                 // 3 -1
        0.0,                                                 // 3  0
        0.0,                                                 // 3  1
        0.0,                                                 // 3  2
        0.0,                                                 // 3  3
        0.0,                                                 // 4 -4
        0.0,                                                 // 4 -3
        std::complex<double>(0.5 * std::sqrt(2 * M_PI / 5.0),
                             0.05 * std::sqrt(2.0 * M_PI / 5.0)), // 4 -2
        0.0,                                                      // 4 -1
        0.0,                                                      // 4  0
        0.0,                                                      // 4  1
        std::complex<double>(0.5 * std::sqrt(2 * M_PI / 5.0),
                             -0.05 * std::sqrt(2.0 * M_PI / 5.0)), // 4  2
        0.0,                                                       // 4  3
        0.0;                                                       // 4  4
                                                                   //
    occ::CVec coeffs = sht.analysis_cplx(values);
    REQUIRE(occ::util::all_close(coeffs, expected, 1e-7, 1e-12));

    occ::CMat values_synth = sht.synthesis_cplx(coeffs);
    REQUIRE(occ::util::all_close(values, values_synth, 1e-7, 1e-12));
}

TEST_CASE("SHT Analysis real function", "[sht]") {

    occ::sht::SHT sht(4);
    occ::Mat values = sht.values_on_grid_real(func);
    occ::CVec expected(sht.nplm());

    expected << std::sqrt(4 * M_PI),      // 0  0
        0.01 * std::sqrt(4 * M_PI / 3.0), // 1  0
        0.1 * std::sqrt(16 * M_PI / 5.0), // 2  0
        0.0,                              // 3  0
        0.0,                              // 4  0
        std::complex<double>(-std::sqrt(2 * M_PI / 3),
                             0.3 * std::sqrt(2 * M_PI / 3)), // 1 1
        0.0,                                                 // 2  1
        0.0,                                                 // 3  1
        0.0,                                                 // 4  1
        0.0,                                                 // 2  2
        0.0,                                                 // 3  2
        std::complex<double>(0.5 * std::sqrt(2 * M_PI / 5.0),
                             -0.05 * std::sqrt(2.0 * M_PI / 5.0)), // 4  2
        0.0,                                                       // 3  3
        0.0,                                                       // 4  3
        0.0;                                                       // 4  4

    occ::CVec coeffs = sht.analysis_real(values);
    REQUIRE(occ::util::all_close(coeffs, expected, 1e-7, 1e-12));

    occ::Mat values_synth = sht.synthesis_real(coeffs);
    REQUIRE(occ::util::all_close(values, values_synth, 1e-7, 1e-12));
}

TEST_CASE("SHT Associated Legendre polynomial evaluate in batch", "[sht]") {

    occ::sht::AssocLegendreP plm(4);
    occ::Vec expected((4 + 1) * (4 + 2) / 2);
    occ::Vec result((4 + 1) * (4 + 2) / 2);

    expected << 0.28209479177387814, 0.24430125595145993, -0.07884789131313,
        -0.326529291016351, -0.24462907724141, 0.2992067103010745,
        0.33452327177864455, 0.06997056236064664, -0.25606603842001846,
        0.2897056515173922, 0.3832445536624809, 0.18816934037548755,
        0.27099482274755193, 0.4064922341213279, 0.24892463950030275;

    plm.evaluate_batch(0.5, result);

    REQUIRE(occ::util::all_close(expected, result));
}

TEST_CASE("Spherical harmonics") {
    occ::sht::SphericalHarmonics sph(4, true);
    using cplx = std::complex<double>;
    using Catch::Matchers::WithinAbs;

    auto cmp = [](cplx a, cplx b) {
	REQUIRE_THAT(a.real(), WithinAbs(b.real(), 1e-12));
	REQUIRE_THAT(a.imag(), WithinAbs(b.imag(), 1e-12));
    };
    SECTION("Angular") {
	occ::CVec eval = sph.evaluate(M_PI/4, M_PI/4);
	cmp(eval(0), cplx(0.28209479177387814, 0.0));
	cmp(eval(1), cplx(0.17274707473566775, -0.17274707473566772));
	cmp(eval(2), cplx(0.34549414947133544,0));
	cmp(eval(3), cplx(-0.17274707473566775, -0.17274707473566772));
	cmp(eval(4), cplx(1.1826236627522426e-17, -0.19313710101159473));
	cmp(eval(5), cplx(0.27313710764801974, -0.2731371076480197));
	cmp(eval(6), cplx(0.15769578262626002, +0));
	cmp(eval(7), cplx(-0.2731371076480198, -0.27313710764801974));
	cmp(eval(8), cplx(1.1826236627522428e-17, +0.19313710101159476));
	cmp(eval(9), cplx(-0.104305955908196, -0.10430595590819601));
	cmp(eval(10), cplx(2.212486281755292e-17, -0.3613264303300692));
	cmp(eval(11), cplx(0.24238513808561293, -0.24238513808561288));
	cmp(eval(12), cplx(-0.13193775767639848, +0));
	cmp(eval(13), cplx(-0.24238513808561296, -0.24238513808561293));
	cmp(eval(14), cplx(2.2124862817552912e-17, +0.3613264303300691));
	cmp(eval(15), cplx(0.104305955908196, -0.10430595590819601));
	cmp(eval(16), cplx(-0.11063317311124561, -1.3548656133020197e-17));
	cmp(eval(17), cplx(-0.22126634622249125, -0.2212663462224913));
	cmp(eval(18), cplx(2.5604553376501068e-17, -0.4181540897233056));
	cmp(eval(19), cplx(0.08363081794466115, -0.08363081794466114));
	cmp(eval(20), cplx(-0.34380302747441394, +0));
	cmp(eval(21), cplx(-0.08363081794466115, -0.08363081794466114));
	cmp(eval(22), cplx(2.5604553376501068e-17, +0.4181540897233056));
	cmp(eval(23), cplx(0.22126634622249125, -0.2212663462224913));
	cmp(eval(24), cplx(-0.11063317311124561, +1.3548656133020197e-17));
    }

    SECTION("Cartesian") {
	occ::Vec3 pos(2.0, 0.0, 1.0);
	double theta = 0.0, phi = 0.0;
	occ::CVec eval = sph.evaluate(pos);
	occ::CVec eval_ang = sph.evaluate(theta, phi);
	cmp(eval(0),  eval_ang(0));
	cmp(eval(1),  eval_ang(1));
	cmp(eval(2),  eval_ang(2));
	cmp(eval(3),  eval_ang(3));
	cmp(eval(4),  eval_ang(4));
	cmp(eval(5),  eval_ang(5));
	cmp(eval(6),  eval_ang(6));
	cmp(eval(7),  eval_ang(7));
	cmp(eval(8),  eval_ang(8));
	cmp(eval(9),  eval_ang(9));
	cmp(eval(10), eval_ang(10));
	cmp(eval(11), eval_ang(11));
	cmp(eval(12), eval_ang(12));
	cmp(eval(13), eval_ang(13));
	cmp(eval(14), eval_ang(14));
	cmp(eval(15), eval_ang(15));
	cmp(eval(16), eval_ang(16));
	cmp(eval(17), eval_ang(17));
	cmp(eval(18), eval_ang(18));
	cmp(eval(19), eval_ang(19));
	cmp(eval(20), eval_ang(20));
	cmp(eval(21), eval_ang(21));
	cmp(eval(22), eval_ang(22));
	cmp(eval(23), eval_ang(23));
	cmp(eval(24), eval_ang(24));
    }
}

TEST_CASE("Clebsch") {
    // j1 = m1 = j2 = m2 = 1/2, j = 1, m = 1
    REQUIRE(occ::sht::clebsch(1, 1, 1, 1, 2, 2) == Catch::Approx(1.0));
    // j1 = j2 = m2 = 1/2, m1 = -1/2, j = 0, m = 0
    REQUIRE(occ::sht::clebsch(1, -1, 1, 1, 0, 0) == Catch::Approx(-std::sqrt(0.5)));
    REQUIRE(occ::sht::clebsch(1, -1, 1, 1, 0, 0) == Catch::Approx(-std::sqrt(0.5)));
    REQUIRE(occ::sht::clebsch(1, -1, 1, 1, 2, 0) == Catch::Approx(std::sqrt(0.5)));
    REQUIRE(occ::sht::clebsch(3, 3, 2, 0, 5, 3) == Catch::Approx(std::sqrt(0.4)));
}

TEST_CASE("wigner_3j", "[wigner_symbols]") {
    using occ::sht::wigner3j;
    using occ::sht::wigner3j_single;
    using Catch::Approx;

    SECTION("Selection rules") {
        REQUIRE(wigner3j_single(2, 2, 1, 1, 1, -2) == Approx(0.0));
        REQUIRE(wigner3j_single(2, 2, 1, 1, 2, -2) == Approx(0.0));
        REQUIRE(wigner3j_single(2, 2, 1, 2, 1, -2) == Approx(0.0));
    }

    SECTION("Analytical formula") {
        double l1 = 2, l2 = 2, l3 = 2, m1 = 0, m2 = 0, m3 = 0;
        double expected = - std::sqrt(2.0 / 35.0);
        REQUIRE(wigner3j_single(l1, l2, l3, m1, m2, m3) == Approx(expected));
    }

    SECTION("Symmetry properties") {
        double l1 = 3, l2 = 2, l3 = 1, m1 = 1, m2 = 0, m3 = -1;
        double expected = sqrt(2.0 / 35);

        REQUIRE(wigner3j_single(l1, l2, l3, m1, m2, m3) == Approx(expected));
        REQUIRE(wigner3j_single(l1, l3, l2, m1, m3, m2) == Approx(expected));
        REQUIRE(wigner3j_single(l2, l1, l3, m2, m1, m3) == Approx(expected));
        REQUIRE(wigner3j_single(l2, l3, l1, m2, m3, m1) == Approx(expected));
        REQUIRE(wigner3j_single(l3, l1, l2, m3, m1, m2) == Approx(expected));
        REQUIRE(wigner3j_single(l3, l2, l1, m3, m2, m1) == Approx(expected));

        REQUIRE(wigner3j_single(l1, l2, l3, -m1, -m2, -m3) == Approx(expected));
        REQUIRE(wigner3j_single(l1, l3, l2, -m1, -m3, -m2) == Approx(expected));
        REQUIRE(wigner3j_single(l2, l1, l3, -m2, -m1, -m3) == Approx(expected));
        REQUIRE(wigner3j_single(l2, l3, l1, -m2, -m3, -m1) == Approx(expected));
        REQUIRE(wigner3j_single(l3, l1, l2, -m3, -m1, -m2) == Approx(expected));
        REQUIRE(wigner3j_single(l3, l2, l1, -m3, -m2, -m1) == Approx(expected));

        REQUIRE(wigner3j_single(l1, l2, l3, m1, -m2, -m3) == Approx(0.0));
        REQUIRE(wigner3j_single(l1, l3, l2, m1, -m3, -m2) == Approx(0.0));
        REQUIRE(wigner3j_single(l2, l1, l3, m2, -m1, -m3) == Approx(expected));
        REQUIRE(wigner3j_single(l2, l3, l1, m2, -m3, -m1) == Approx(expected));
        REQUIRE(wigner3j_single(l3, l1, l2, m3, -m1, -m2) == Approx(0.0));
        REQUIRE(wigner3j_single(l3, l2, l1, m3, -m2, -m1) == Approx(0.0));
    }
}
