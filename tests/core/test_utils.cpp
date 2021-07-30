#include <occ/core/util.h>
#include "catch.hpp"

TEST_CASE("all_close", "[util]")
{
    using occ::util::all_close;

    occ::Mat x = occ::Mat::Identity(3, 3);
    occ::Mat x2 = occ::Mat::Identity(3, 3) * 2;
    REQUIRE(all_close(x, x));
    REQUIRE(!all_close(x, x2));
    REQUIRE(all_close(x, x2 * 0.5));
}


TEST_CASE("is_close", "[util]")
{
    using occ::util::is_close;
    double x1 = 1e-6;
    double x2 = 2e-6;
    REQUIRE(!is_close(x1, x2));
    REQUIRE(is_close(x1, x2 / 2));
    REQUIRE(is_close(1e-17, 1e-18));
}

TEST_CASE("is_even, is_odd", "[util]")
{
    using occ::util::is_even, occ::util::is_odd;
    REQUIRE(is_even(2));
    REQUIRE(is_odd(1));
    REQUIRE(is_even(0));
    REQUIRE(is_odd(1312321413));
    REQUIRE(!is_odd(1312421412));
    REQUIRE(!is_even(1312421411));
}

TEST_CASE("smallest_common_factor", "[util]")
{
    using occ::util::smallest_common_factor;
    REQUIRE(smallest_common_factor(1, 3) == 1);
    REQUIRE(smallest_common_factor(-1, 3) == 1);
    REQUIRE(smallest_common_factor(-2, 4) == 2);
    REQUIRE(smallest_common_factor(-4, 12) == 4);
    REQUIRE(smallest_common_factor(12, 12) == 12);
    REQUIRE(smallest_common_factor(0, 12) == 12);
}


TEST_CASE("human_readable_size", "[util]")
{
    using occ::util::human_readable_size;

    REQUIRE(human_readable_size(1024, "B") == "1.00KiB");
    REQUIRE(human_readable_size(1024*1024, "B") == "1.00MiB");
    REQUIRE(human_readable_size(1024*1024*1024, "B") == "1.00GiB");
    REQUIRE(human_readable_size(0, "B") == "0.00B");
}

