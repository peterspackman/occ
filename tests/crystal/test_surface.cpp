#include <occ/crystal/surface.h>
#include <occ/crystal/crystal.h>
#include <occ/core/timings.h>
#include "catch.hpp"

using occ::crystal::Crystal;
using occ::crystal::AsymmetricUnit;
using occ::crystal::UnitCell;
using occ::crystal::SpaceGroup;
using occ::crystal::SymmetryOperation;
using occ::crystal::Surface;
using occ::crystal::MillerIndex;
using occ::util::all_close;

auto acetic_acid_crystal()
{
    const std::vector<std::string> labels = {"C1", "C2", "H1", "H2", "H3", "H4", "O1", "O2"};
    occ::IVec nums(labels.size());
    occ::Mat positions(labels.size(), 3);
    for(size_t i = 0; i < labels.size(); i++)
    {
        nums(i) = occ::core::Element(labels[i]).atomic_number();
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
    AsymmetricUnit asym = AsymmetricUnit(positions.transpose(), nums, labels);
    SpaceGroup sg(33);
    UnitCell cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);
    return Crystal(asym, sg, cell);
}

TEST_CASE("Surface constructor", "[crystal, surface]")
{
    Crystal a = acetic_acid_crystal();
    MillerIndex m{0, 1, 0};
    Surface surf(m, a);
    surf.print();
}


TEST_CASE("Surface generation", "[crystal, surface]")
{
    Crystal a = acetic_acid_crystal();
    occ::timing::StopWatch sw;
    sw.start();
    auto surfaces = occ::crystal::generate_surfaces(a, 0.1);
    sw.stop();
    size_t n = 0;
    fmt::print("Top 10 surfaces\n");
    for(const auto& surf: surfaces)
    {
        surf.print();
        n++;
        if(n > 10) break;
    }
    fmt::print("Generation took {} s\n", sw.read());
}
