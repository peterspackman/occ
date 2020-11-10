#include "gto.h"
#include "catch.hpp"
#include "util.h"
#include "basisset.h"

using tonto::util::all_close;

TEST_CASE("Basic rotations", "[mo_rotation]")
{
    tonto::Mat3 rot = tonto::Mat3::Identity(3, 3);
    auto drot = tonto::gto::cartesian_gaussian_rotation_matrix<2>(rot);
    REQUIRE(all_close(drot, tonto::MatRM::Identity(6, 6)));

    auto frot = tonto::gto::cartesian_gaussian_rotation_matrix<3>(rot);
    REQUIRE(all_close(frot, tonto::MatRM::Identity(10, 10)));
}

