#include <occ/crystal/crystal.h>
#include <occ/core/util.h>
#include <fmt/ostream.h>
#include "catch.hpp"

using occ::crystal::Crystal;
using occ::crystal::AsymmetricUnit;
using occ::crystal::UnitCell;
using occ::crystal::SpaceGroup;
using occ::crystal::SymmetryOperation;
using occ::util::all_close;
using occ::util::deg2rad;


auto ice_ii_asym()
{
    const std::vector<std::string> labels = {
        "O1", "H1", "H2", "O2", "H3", "H4", "O3", "H5", "H6", "O4", "H7", "H8",
        "O5", "H9", "H10", "O6", "H11", "H12", "O7", "H13", "H14", "O8", "H15",
        "H16", "O9", "H17", "H18", "O10", "H19", "H20", "O11", "H21", "H22", "O12",
        "H23", "H24",
    };

    occ::IVec nums(labels.size());
    occ::Mat positions(labels.size(), 3);
    for(size_t i = 0; i < labels.size(); i++)
    {
        nums(i) = occ::chem::Element(labels[i]).atomic_number();
    }

    positions << 
        0.273328954083, 0.026479033257, 0.855073668062, 0.152000330304,
        0.043488909374, 0.793595454907, 0.420775085827, 0.191165194485,
        0.996362203192, 0.144924657237, 0.726669877048, 0.973520141937,
        0.206402797363, 0.847998481439, 0.956510183901, 0.003636687868,
        0.579223433079, 0.808833958746, 0.026477491142, 0.855072949204,
        0.273328854276, 0.043487719387, 0.793594459529, 0.152000553858,
        0.191163388489, 0.996362120061, 0.420774953988, 0.726670757782,
        0.973520932681, 0.144926297633, 0.847999275418, 0.956510882297,
        0.206404294889, 0.579224602173, 0.808834869258, 0.003637530197,
        0.855073412561, 0.273329597478, 0.026478702027, 0.793594909621,
        0.152000771295, 0.043488316376, 0.996362312075, 0.420775512814,
        0.191164826329, 0.973520717390, 0.144925628579, 0.726671054509,
        0.956510600982, 0.206403626100, 0.847999547813, 0.808834607385,
        0.003637609551, 0.579224562315, 0.477029330652, 0.749805220756,
        0.331717174202, 0.402360172390, 0.720795433576, 0.401054786853,
        0.368036378343, 0.742284933413, 0.207434128329, 0.668282055550,
        0.522969467265, 0.250193622013, 0.598945169999, 0.597639203188,
        0.279204514235, 0.792565160978, 0.631962548905, 0.257714022497,
        0.749805496250, 0.331717033025, 0.477029827575, 0.720795009402,
        0.401054437437, 0.402360618546, 0.742284706875, 0.207433751728,
        0.368036342085, 0.522969071341, 0.250193392512, 0.668282780114,
        0.597638176364, 0.279203622225, 0.598945231951, 0.631962932785,
        0.257715003205, 0.792566578018, 0.331715381178, 0.477028907327,
        0.749804544234, 0.401053887354, 0.402360576463, 0.720795552111,
        0.207432480540, 0.368035542438, 0.742284142147, 0.250193225247,
        0.668282913065, 0.522970147212, 0.279203658434, 0.598945325854,
        0.597639149965, 0.257715011998, 0.792566781760, 0.631964289620;

    return AsymmetricUnit(positions.transpose(), nums, labels);
}

auto acetic_asym()
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
    return AsymmetricUnit(positions.transpose(), nums, labels);
}

void print_asymmetric_unit(const AsymmetricUnit &asym)
{
    for(size_t i = 0; i < asym.size(); i++)
    {
        fmt::print("{:<6s} {:>3d}   {:.6f} {:.6f} {:.6f}\n", asym.labels[i], asym.atomic_numbers(i),
                   asym.positions(0, i), asym.positions(1, i), asym.positions(2, i));
    }
}

TEST_CASE("AsymmetricUnit constructor", "[crystal]")
{

    AsymmetricUnit asym = ice_ii_asym();
    print_asymmetric_unit(asym);
    REQUIRE(asym.labels.size() == 36);
    std::vector<std::string> old_labels = asym.labels;
    asym.generate_default_labels();
    print_asymmetric_unit(asym);
    REQUIRE(old_labels == asym.labels);
}

TEST_CASE("UnitCell constructor", "[crystal]")
{
    UnitCell ice = occ::crystal::rhombohedral_cell(7.78, deg2rad(113.1));
}

TEST_CASE("ice_ii molecules", "[crystal]")
{
    AsymmetricUnit asym = ice_ii_asym();
    SpaceGroup sg(1);
    UnitCell cell = occ::crystal::rhombohedral_cell(7.78, deg2rad(113.1));
    Crystal ice_ii(asym, sg, cell);
    REQUIRE(ice_ii.symmetry_operations().size() == 1);
    fmt::print("Unit cell molecules:\n");
    for(const auto& mol: ice_ii.unit_cell_molecules())
    {
        fmt::print("{}\n", mol.name());
    }
    REQUIRE(ice_ii.symmetry_unique_molecules().size() == 12);
    fmt::print("Asymmetric unit molecules:\n");
    for(const auto& mol: ice_ii.symmetry_unique_molecules())
    {
        fmt::print("{}\n", mol.name());
    }
}


SymmetryOperation dimer_symop(const occ::chem::Dimer &dimer, const Crystal &crystal)
{
    const auto& a = dimer.a();
    const auto& b = dimer.b();

    int sa_int = a.asymmetric_unit_symop()(0);
    int sb_int = b.asymmetric_unit_symop()(0);

    SymmetryOperation symop_a(sa_int);
    SymmetryOperation symop_b(sb_int);

    auto symop_ab = symop_b * symop_a.inverted();
    occ::Vec3 c_a = symop_ab(crystal.to_fractional(a.positions())).rowwise().mean();
    occ::Vec3 v_ab = crystal.to_fractional(b.centroid()) - c_a;

    symop_ab = symop_ab.translated(v_ab);
    return symop_ab;
}

TEST_CASE("acetic molecules", "[crystal]")
{
    AsymmetricUnit asym = acetic_asym();
    SpaceGroup sg(33);
    UnitCell cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);

    Crystal acetic(asym, sg, cell);
    REQUIRE(acetic.unit_cell_molecules().size() == 4);
    fmt::print("Unit cell molecules:\n");
    for(const auto& mol: acetic.unit_cell_molecules())
    {
        fmt::print("{}\n", mol.name());
    }
    REQUIRE(acetic.symmetry_unique_molecules().size() == 1);
    fmt::print("Asymmetric unit molecules:\n");
    for(const auto& mol: acetic.symmetry_unique_molecules())
    {
        fmt::print("{}\n", mol.name());
    }

    auto crystal_dimers = acetic.symmetry_unique_dimers(3.8);
    const auto &dimers = crystal_dimers.unique_dimers;
    REQUIRE(dimers.size() == 7);
    fmt::print("Dimers\n");
    for(const auto& dimer: dimers)
    {
        auto s_ab = dimer_symop(dimer, acetic);
        fmt::print("R = {:.3f}, symop = {}\n", dimer.nearest_distance(), s_ab.to_string());
    }

    const auto &mol_neighbors = crystal_dimers.molecule_neighbors;
    for(size_t i = 0; i < mol_neighbors.size(); i++)
    {
        const auto& n = mol_neighbors[i];
        fmt::print("Neighbors for molecule {}\n", i);
        size_t j = 0;
        for(const auto& dimer: n)
        {
            auto s_ab = dimer_symop(dimer, acetic);
            fmt::print("R = {:.3f}, symop = {}, unique_idx = {}\n",
                       dimer.nearest_distance(), s_ab.to_string(),
                       crystal_dimers.unique_dimer_idx[i][j]);
            j++;
        }
    }

}
