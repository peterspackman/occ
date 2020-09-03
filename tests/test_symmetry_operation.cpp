#include "symmetryoperation.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("SymmetryOperation constructor", "[symmetry_operation]")
{
    using craso::crystal::SymmetryOperation;
    REQUIRE(SymmetryOperation("x,y,z").is_identity());
    REQUIRE(SymmetryOperation(16484).is_identity());
}
