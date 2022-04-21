#include "catch.hpp"
#include <occ/3rdparty/eigen-fmt/fmt.h>
#include <occ/core/util.h>
#include <occ/sht/sht.h>

double func(double theta, double phi) {
    return (1.0 + 0.01 * std::cos(theta) +
            0.1 * (3.0 * std::cos(theta) * std::cos(theta) - 1.0) +
            (std::cos(phi) + 0.3 * std::sin(phi)) * std::sin(theta) +
            (std::cos(2.0 * phi) + 0.1 * std::sin(2.0 * phi)) *
                std::sin(theta) * std::sin(theta) *
                (7.0 * std::cos(theta) * std::cos(theta) - 1.0) * 3.0 / 8.0);
}

TEST_CASE("Analysis complex", "[sht]") {

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

TEST_CASE("Analysis real", "[sht]") {

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
