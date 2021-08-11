#include "catch.hpp"
#include <occ/gto/gto.h>
#include <fmt/ostream.h>

using occ::Mat;

TEST_CASE("spherical <-> cartesian") {
    for(int i = 0; i < 5; i++)
    {
        Mat x = Mat::Identity(2 * i + 1, 2 * i + 1);
        Mat s2c = occ::gto::spherical_to_cartesian_transformation_matrix(i);
        Mat c2s = occ::gto::cartesian_to_spherical_transformation_matrix(i);
        fmt::print("L = {}\n{}\n", i, s2c * c2s);

    }
}
