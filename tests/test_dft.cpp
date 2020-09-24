#include "catch.hpp"
#include "dft.h"
#include "molecule.h"
#include <vector>
#include <iostream>


TEST_CASE("Water DFT grid", "[dft]")
{
    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    craso::chem::Molecule mol(atoms);

    SECTION("Grid generation") {
        craso::dft::DFTGrid grid(mol);
        grid.set_min_angular_points(12);
        grid.set_max_angular_points(20);
        REQUIRE(grid.atomic_numbers() == mol.atomic_numbers());
        std::cout << "Grid" << std::endl;
        std::cout << grid.grid_points(0).transpose() << std::endl;

    }

}
