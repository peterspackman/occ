#include "catch.hpp"
#include "hf.h"
#include "basisset.h"
#include <fmt/ostream.h>

using tonto::qm::BasisSet;
using tonto::hf::HartreeFock;

TEST_CASE("GTO vals H2/STO-3G") {
    std::vector<libint2::Atom> atoms {
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.398397}
    };
    BasisSet basis("sto-3g", atoms);
    auto grid_pts = tonto::Mat3N::Random(3, 4);
    HartreeFock hf(atoms, basis);
    auto field_values = hf.nuclear_electric_field_contribution(grid_pts);
    fmt::print("Grid points\n{}\n", grid_pts);
    fmt::print("Nuclear E field values:\n{}\n", field_values);
}
