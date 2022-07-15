#include "catch.hpp"
#include <fmt/ostream.h>
#include <occ/crystal/crystal.h>
#include <occ/interaction/wolf.h>

/* Dimer tests */
using occ::crystal::AsymmetricUnit;
using occ::crystal::Crystal;
using occ::crystal::SpaceGroup;
using occ::crystal::UnitCell;
using occ::interaction::wolf_coulomb_energy;

auto nacl_crystal() {
    const std::vector<std::string> labels = {"Na1", "Cl1"};
    occ::IVec nums(labels.size());
    occ::Mat positions(3, labels.size());
    for (size_t i = 0; i < labels.size(); i++) {
        nums(i) = occ::core::Element(labels[i]).atomic_number();
    }
    positions << 0.00000, 0.50000, 0.00000, 0.50000, 0.00000, 0.50000;
    AsymmetricUnit asym = AsymmetricUnit(positions, nums, labels);
    SpaceGroup sg(1);
    UnitCell cell = occ::crystal::rhombohedral_cell(3.9598, M_PI / 3);
    return Crystal(asym, sg, cell);
}

TEST_CASE("NaCl wolf sum", "[interaction,wolf]") {
    auto nacl = nacl_crystal();

    occ::Vec charges(2);
    charges << 1.0, -1.0;

    double radius = 16.0;
    double eta = 0.2;

    auto surrounds = nacl.asymmetric_unit_atom_surroundings(radius);
    occ::interaction::WolfParams params{radius, eta};
    double wolf_energy = 0.0;
    occ::Mat3N cart_pos_asym =
        nacl.to_cartesian(nacl.asymmetric_unit().positions);
    for (int i = 0; i < surrounds.size(); i++) {
        double qi = charges(i);
        occ::Vec3 pi = cart_pos_asym.col(i);
        occ::Vec qj(surrounds[i].size());
        for (int j = 0; j < qj.rows(); j++) {
            qj(j) = charges(surrounds[i].asym_idx(j));
        }
        wolf_energy += wolf_coulomb_energy(qi, pi, qj, surrounds[i].cart_pos);
    }
    fmt::print("Wolf energy (NaCl): {}\n", wolf_energy);
    REQUIRE(wolf_energy == Approx(-0.3302735899347252));
}
