#include <occ/crystal/symmetryoperation.h>
#include <occ/core/util.h>
#include "catch.hpp"

using occ::crystal::SymmetryOperation;
using occ::util::all_close;

TEST_CASE("SymmetryOperation constructor", "[symmetry_operation]")
{
    REQUIRE(SymmetryOperation("x,y,z").is_identity());
    REQUIRE(SymmetryOperation(16484).is_identity());
}

TEST_CASE("SymmetryOperation seitz", "[symmetry_operation]")
{
    auto id = SymmetryOperation(16484);
    REQUIRE(
        all_close(
            id.seitz(),
            Eigen::Matrix4d::Identity()
        )
    );
    REQUIRE(all_close(id.rotation(), id.inverted().rotation()));
}
