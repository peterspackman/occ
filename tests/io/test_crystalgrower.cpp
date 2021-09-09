#include "catch.hpp"
#include <occ/io/crystalgrower.h>
#include <occ/crystal/crystal.h>
#include <iostream>


auto acetic_crystal()
{
    const std::vector<std::string> labels = {"C1", "C2", "H1", "H2", "H3", "H4", "O1", "O2"};
    occ::IVec nums(labels.size());
    occ::Mat positions(labels.size(), 3);
    for(size_t i = 0; i < labels.size(); i++)
    {
        nums(i) = occ::chem::Element(labels[i]).atomic_number();
    }
    positions << 
        0.16510, 0.28580,  0.17090,
        0.08940, 0.37620,  0.34810,
        0.18200, 0.05100, -0.11600,
        0.12800, 0.51000,  0.49100,
        0.03300, 0.54000,  0.27900,
        0.05300, 0.16800,  0.42100,
        0.12870, 0.10750,  0.00000,
        0.25290, 0.37030,  0.17690;
    
    occ::crystal::AsymmetricUnit asym(positions.transpose(), nums, labels);
    occ::crystal::SpaceGroup sg(33);
    occ::crystal::UnitCell cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);

    return occ::crystal::Crystal(asym, sg, cell);
}


TEST_CASE("Write acetic structure file", "[write]")
{
    auto acetic = acetic_crystal();
    occ::io::crystalgrower::StructureWriter writer(std::cout);
    writer.write(acetic);
    REQUIRE(true);
}
