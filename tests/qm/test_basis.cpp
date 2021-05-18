#include "catch.hpp"
#include <occ/qm/basisset.h>
#include <fmt/ostream.h>
#include <occ/core/util.h>

TEST_CASE("H2O/6-31G") {
    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    occ::qm::BasisSet basis("6-31G", atoms);
    auto sp_shells = occ::qm::pople_sp_shells(basis);
    for(const auto& x: sp_shells) {
        fmt::print("{}\n", x);
    }
}
