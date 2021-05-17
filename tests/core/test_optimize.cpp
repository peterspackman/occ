#include <occ/core/linear_algebra.h>
#include "catch.hpp"
#include <fmt/ostream.h>
#include <occ/core/timings.h>
#include <occ/core/optimize.h>

double sx(double x)
{
    return std::sin(x);
}

double ax(double x)
{
    return std::abs(x - 4);
}


TEST_CASE("Brent") {
    std::function<double(double)> x2 = [](double x) { return (x - 0.5) * (x - 0.5); };
    occ::opt::Brent brent(x2);
    double xmin = brent.xmin();
    fmt::print("Found minimum of (x - 0.5)^2 in {} evaluations: ({}, {})\n", brent.num_calls(), xmin, brent.f_xmin());
    REQUIRE(xmin == Approx(0.5));
    occ::opt::Brent brentsin(sx);
    xmin = brentsin.xmin();
    fmt::print("Found a minimum of sin(x) in {} evaluations: ({}, {})\n", brentsin.num_calls(), xmin, brentsin.f_xmin());
    REQUIRE(std::abs(xmin) == Approx(M_PI / 2));

    occ::opt::Brent ba(ax);
    xmin = ba.xmin();
    fmt::print("Found minimum of abs(x - 4) in {} evaluations ({}, {})\n", ba.num_calls(), xmin, ba.f_xmin());
}
