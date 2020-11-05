#include "linear_algebra.h"
#include "catch.hpp"
#include <fmt/ostream.h>
#include "timings.h"
#include "optimize.h"

double x3(double x)
{
    return std::sin(x);
}

TEST_CASE("Brent") {
    std::function<double(double)> x2 = [](double x) { return (x - 0.5) * (x - 0.5); };
    tonto::opt::Brent brent(x2);
    double xmin = brent.xmin();
    fmt::print("Found minimum in {} evaluations: {}\n", brent.num_calls(), xmin);
    REQUIRE(xmin == Approx(0.5));
    tonto::opt::Brent brentsin(x3);
    xmin = brentsin.xmin();
    fmt::print("Found minimum in {} evaluations: {}\n", brentsin.num_calls(), xmin);
    REQUIRE(std::abs(xmin) == Approx(M_PI / 2));
}
