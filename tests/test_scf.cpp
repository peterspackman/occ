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
            {8, -1.32695761, -0.10593856, 0.01878821},
            {1, -1.93166418, 1.60017351, -0.02171049},
            {1, 0.48664409, 0.07959806, 0.00986248}
        };

        libint2::BasisSet obs("STO-3G", atoms);
        HartreeFock hf(atoms, obs);
        craso::scf::SCF<HartreeFock> scf(hf);
        double e = scf.compute_scf_energy();
        REQUIRE(e == Approx(-74.963706080054).epsilon(1e-8));
    }

}
