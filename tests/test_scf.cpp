#include "scf.h"
#include "hf.h"
#include "catch.hpp"
#include <iostream>
#include <vector>

using tonto::hf::HartreeFock;

TEST_CASE("Water SCF", "[scf]")
{
    libint2::Shell::do_enforce_unit_normalization(false);
    if (!libint2::initialized()) libint2::initialize();
    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };

    SECTION("STO-3G") {
        libint2::BasisSet obs("STO-3G", atoms);
        HartreeFock hf(atoms, obs);
        tonto::scf::RestrictedSCF<HartreeFock> scf(hf);
        double e = scf.compute_scf_energy();
        REQUIRE(e == Approx(-74.963706080054).epsilon(1e-8));
    }

    SECTION("3-21G") {
        libint2::BasisSet obs("3-21G", atoms);
        HartreeFock hf(atoms, obs);
        tonto::scf::RestrictedSCF<HartreeFock> scf(hf);
        double e = scf.compute_scf_energy();
        REQUIRE(e == Approx(-75.585325673488).epsilon(1e-8));
    }


}
