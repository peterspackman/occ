#include "catch.hpp"
#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <vector>
#include <iostream>


TEST_CASE("Water DFT grid", "[dft]")
{

    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    libint2::BasisSet basis("3-21G", atoms);

    SECTION("Grid generation") {
        craso::dft::DFTGrid grid(basis, atoms);
        grid.set_min_angular_points(12);
        grid.set_max_angular_points(20);
        auto pts = grid.grid_points(0);
        assert(pts.cols() == 1564);
    }

}
