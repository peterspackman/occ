#include "tonto/geometry/morton.h"
#include "catch.hpp"
#include <iostream>

using tonto::geometry::MIndex;

TEST_CASE("Morton Index", "[geometry]")
{
    MIndex m{1};
    MIndex c1 = m.child(7);
    MIndex c2 = c1.child(3);

    REQUIRE(m.level() == 0);
    REQUIRE(c1.size() == Approx(0.25));
    REQUIRE(c2.size() == Approx(0.125));
    auto center = c2.center();
    REQUIRE(center.x == Approx(0.875));
    REQUIRE(center.y == Approx(0.875));
    REQUIRE(center.z == Approx(0.625));

    REQUIRE(c2.primal(1, 3).code == 0x0);
    REQUIRE(c2.dual(1, 3).code == 0xb6db6db6db6db6db);
}
