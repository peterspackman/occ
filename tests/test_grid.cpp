#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "lebedev.h"
#include <fmt/ostream.h>

TEST_CASE("lebedev", "[grid]")
{
    auto grid = tonto::grid::lebedev(110);
    fmt::print("grid:\n{}\n", grid);
    REQUIRE(grid.rows() == 110);
    REQUIRE(grid.cols() == 4);

    BENCHMARK("Lebedev 86") {
        return tonto::grid::lebedev(86);
    };

    BENCHMARK("Lebedev 590") {
        return tonto::grid::lebedev(590);
    };

    BENCHMARK("Lebedev 5810") {
        return tonto::grid::lebedev(5810);
    };
}
