#include <occ/sht/sht.h>
#include "catch.hpp"
#include <fmt/ostream.h>
#include <occ/core/util.h>

double func(double theta, double phi) {
    return 1.0 + 0.01*std::cos(theta)  + 0.1*(3.*std::cos(theta)*std::cos(theta) - 1.0)	// Y00, Y10, Y20
    + (std::cos(phi) + 0.3*std::sin(phi)) * std::sin(theta)	// Y11
    + (std::cos(2.*phi) + 0.1*std::sin(2.*phi)) * std::sin(theta)*std::sin(theta) * (7.0* std::cos(theta)*std::cos(theta) - 1.0) * 3./8.; 	// Y42
}



TEST_CASE("Analysis complex", "[sht]") {
    size_t l_max = 4;
    using occ::sht::SHT;
    occ::sht::SHT sht(l_max);
    occ::sht::AssocLegendreP Plm(l_max);
    auto coeffs = sht.analysis_complex(func);

    for(size_t i = 0; i < coeffs.size(); i++) {
	const auto coeff = coeffs[i];
	if(i == SHT::idx_c(0, 0)) {
	    REQUIRE(coeff.real() == Approx(std::sqrt(4 * M_PI)));
	}
	else if (i == SHT::idx_c(1, 0)) {
	    REQUIRE(coeff.real() == Approx(0.01 * std::sqrt(4 * M_PI / 3.0)));
	}
	else if (i == SHT::idx_c(2, 0)) {
	    REQUIRE(coeff.real() == Approx(0.1 * std::sqrt(16 * M_PI / 5.0)));
	}
	else if (i == SHT::idx_c(1, 1)) {
	    REQUIRE(coeff.real() == Approx(-std::sqrt(2 * M_PI / 3.0)));
	    REQUIRE(coeff.imag() == Approx(0.3 * std::sqrt(2.0 * M_PI /3.0)));;
	}
	else if (i == SHT::idx_c(1, -1)) {
	    REQUIRE(coeff.real() == Approx(-std::sqrt(2 * M_PI / 3.0)));
	    REQUIRE(coeff.imag() == Approx(-0.3 * std::sqrt(2.0 * M_PI /3.0)));
	}
	else if (i == SHT::idx_c(4, 2)) {
	    REQUIRE(coeff.real() == Approx(0.5 * std::sqrt(2 * M_PI / 5.0)));
	    REQUIRE(coeff.imag() == Approx(0.05 * std::sqrt(2.0 * M_PI /5.0)));
	}
	else if (i == SHT::idx_c(4, -2)) {
	    REQUIRE(coeff.real() == Approx(0.5 * std::sqrt(2 * M_PI / 5.0)));
	    REQUIRE(coeff.imag() == Approx(-0.05 * std::sqrt(2.0 * M_PI /5.0)));
	}
	else {
	    if(std::abs(coeff.real()) > 1e-12 || std::abs(coeff.imag()) > 1e-12) {
		int l = std::floor(std::sqrt(i));
		int m = i - l * l - l;
		fmt::print("coeffs({}) l = {}, m = {} is nonzero: ({}, {})\n", i, l, m, coeff.real(), coeff.imag());
		fmt::print("P({}, {}, 0.5) = {}\n", l, m, Plm(l, m, 0.5));
		REQUIRE(coeff.real() == 0.0);
		REQUIRE(coeff.imag() == 0.0);
	    }
	}
    }

    occ::CMat values = sht.synthesis_complex(coeffs);
    occ::CMat ref_values = sht.values_on_grid_complex(func);

    for(size_t i = 0; i < values.rows(); i++) {
	for(size_t j = 0; j < values.cols(); j++) {
	    REQUIRE(values(i, j).real() == Approx(ref_values(i, j).real()));
	}
    }
}

TEST_CASE("Analysis real", "[sht]") {
    size_t l_max = 25;
    using occ::sht::SHT;
    occ::sht::SHT sht(l_max);
    occ::sht::AssocLegendreP Plm(l_max);
    auto t0 = std::chrono::high_resolution_clock::now();
    auto coeffs = sht.analysis_real(func);
    auto t1 = std::chrono::high_resolution_clock::now();

    fmt::print("Analysis took {} s\n",std::chrono::duration<double>(t1 - t0).count());
    occ::timing::print_timings();

    int i = 0;
    for(int m = 0; m <= l_max; m++) {
	for(int l = m; l <= l_max; l++) {
	    if(m == 0 && l == 0) {
		REQUIRE(coeffs(i).real() == Approx(std::sqrt(4 * M_PI)));
	    }
	    else if (m == 0 && l == 1) {
		REQUIRE(coeffs(i).real() == Approx(0.01 * std::sqrt(4 * M_PI / 3.0)));
	    }
	    else if (m == 0 && l == 2) {
		REQUIRE(coeffs(i).real() == Approx(0.1 * std::sqrt(16 * M_PI / 5.0)));
	    }
	    else if (m == 1 && l == 1) {
		REQUIRE(coeffs(i).real() == Approx(-std::sqrt(2 * M_PI / 3.0)));
		REQUIRE(coeffs(i).imag() == Approx(0.3 * std::sqrt(2.0 * M_PI /3.0)));;
	    }
	    else if (m == 2 && l == 4) {
		REQUIRE(coeffs(i).real() == Approx(0.5 * std::sqrt(2 * M_PI / 5.0)));
		REQUIRE(coeffs(i).imag() == Approx(0.05 * std::sqrt(2.0 * M_PI /5.0)));
	    }
	    else {
		if(std::abs(coeffs(i).real()) > 1e-12 || std::abs(coeffs(i).imag()) > 1e-12) {
		    CHECK(coeffs(i).real() == 0.0);
		    CHECK(coeffs(i).imag() == 0.0);
		}
	    }
	    i++;
	}
    }
    
    occ::Mat values = sht.synthesis_real(coeffs);
    occ::Mat ref_values = sht.values_on_grid_real(func);

    for(size_t i = 0; i < values.rows(); i++) {
	for(size_t j = 0; j < values.cols(); j++) {
	    REQUIRE(values(i, j) == Approx(ref_values(i, j)));
	}
    }
}
