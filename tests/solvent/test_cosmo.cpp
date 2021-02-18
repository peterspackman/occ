#include <tonto/solvent/cosmo.h>
#include <fmt/ostream.h>
#include <tonto/core/timings.h>
#include "catch.hpp"

using tonto::solvent::COSMO;
using tonto::Vec;
using tonto::Mat;
using tonto::Mat3N;

TEST_CASE("COSMO", "[solvent]") {

    auto pts = Mat3N(3, 4);
    pts << 1.0, 0.0, 0.0, 0.0,
           0.0, 1.0, 0.0, 1.0,
           0.0, 0.0, 1.0, 1.0;
    auto areas = Vec(4);
    areas << 0.01, 0.02, 0.03, 0.04;

    auto charges = Vec(4);
    charges << -0.01, 0.01, -0.02, 0.01;
    const COSMO c(78.4);
    COSMO::Result result = c(pts, areas, charges);
    fmt::print("Final energy: {}\n", result.energy);
    fmt::print("Initial:\n{}\n", result.initial);

    REQUIRE(result.energy == Approx(-1.48692397577355e-05));

}
