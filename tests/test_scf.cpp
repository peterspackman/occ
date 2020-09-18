#include "scf.h"
#include "hf.h"
#include "catch.hpp"
#include <iostream>
#include <vector>

using craso::scf::SCF;
using craso::hf::HartreeFock;

TEST_CASE("Water SCF", "[scf]")
{
    if (!libint2::initialized()) libint2::initialize();
    libint2::Shell::do_enforce_unit_normalization(false);
    SECTION("Initialize") {
        std::vector<libint2::Atom> atoms{
            {8,  -0.7021961, -0.0560603,  0.0099423},
            {1,  -1.0221932,  0.8467758, -0.0114887},
            {1,   0.2575211,  0.0421215,  0.0052190}
        };

        libint2::BasisSet obs("STO-3G", atoms);
        HartreeFock hf(atoms, obs);
        craso::scf::SCF<HartreeFock> scf(hf);
        double e = scf.compute_scf_energy();
        REQUIRE(e == Approx(-73.241652074251).epsilon(1e-8));
    }

}
