#include "catch.hpp"
#include <fmt/ostream.h>
#include <occ/core/util.h>
#include <occ/qm/basisset.h>
#include <occ/qm/occshell.h>

TEST_CASE("H2O/6-31G") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    occ::qm::BasisSet basis("6-31G", atoms);
    auto sp_shells = occ::qm::pople_sp_shells(basis);
    //    for(const auto& x: sp_shells) {
    //        fmt::print("{}\n", x);
    //    }
}

TEST_CASE("spherical_to_cartesian") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    occ::qm::BasisSet basis("6-31G", atoms);
    basis.set_pure(true);
    //    for(const auto& x: basis) {
    //        fmt::print("{}\n", x);
    //    }
}

TEST_CASE("AOBasis load") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    occ::qm::AOBasis basis = occ::qm::AOBasis::load(atoms, "6-31G");
    for (const auto &sh : basis.shells()) {
        std::cout << sh << '\n';
    }
}
