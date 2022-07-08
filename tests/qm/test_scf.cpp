#include "catch.hpp"
#include <iostream>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <vector>

using occ::hf::HartreeFock;
using occ::qm::SpinorbitalKind;

TEST_CASE("Water SCF", "[scf]") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};

    SECTION("STO-3G") {
        auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");
        HartreeFock hf(obs);
        occ::scf::SCF<HartreeFock, SpinorbitalKind::Restricted> scf(hf);
        scf.energy_convergence_threshold = 1e-8;
        double e = scf.compute_scf_energy();
        REQUIRE(e == Approx(-74.963706080054).epsilon(1e-8));
    }

    SECTION("3-21G") {
        auto obs = occ::qm::AOBasis::load(atoms, "3-21G");
        HartreeFock hf(obs);
        occ::scf::SCF<HartreeFock, SpinorbitalKind::Restricted> scf(hf);
        scf.energy_convergence_threshold = 1e-8;
        double e = scf.compute_scf_energy();
        REQUIRE(e == Approx(-75.585325673488).epsilon(1e-8));
    }
}
