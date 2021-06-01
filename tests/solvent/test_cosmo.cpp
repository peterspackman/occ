#include <occ/solvent/cosmo.h>
#include <fmt/ostream.h>
#include <occ/core/timings.h>
#include "catch.hpp"

using occ::solvent::COSMO;
using occ::Vec;
using occ::Mat;
using occ::Mat3N;

TEST_CASE("COSMO", "[solvent]") {

    auto pts = Mat3N(3, 12);
    pts << -0.525731, 0.525731, -0.525731,  0.525731,       0.0,      0.0,       0.0,       0.0,  0.850651, 0.850651, -0.850651, -0.850651,
            0.850651, 0.850651, -0.850651, -0.850651, -0.525731, 0.525731, -0.525731,  0.525731,       0.0,      0.0,       0.0,       0.0,
            0.0,           0.0,       0.0,       0.0,  0.850651, 0.850651, -0.850651, -0.850651, -0.525731, 0.525731, -0.525731,  0.527531;

    auto areas = Vec(12);
    areas.setConstant(0.79787845);

    auto charges = Vec(12);
    charges.topRows(6).setConstant(-0.05);
    charges.bottomRows(6).setConstant(0.05);
    const COSMO c(78.4);
    COSMO::Result result = c(pts, areas, charges);
    fmt::print("Final energy: {}\n", result.energy);
    fmt::print("InitialFinal charges:\n{}\n", result.initial);
    fmt::print("Converged charges:\n{}\n", result.converged);

    REQUIRE(result.energy == Approx(-0.0042078133));

}
