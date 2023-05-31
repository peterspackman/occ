#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/ostream.h>
#include <occ/crystal/crystal.h>
#include <occ/interaction/pair_potential.h>
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
    REQUIRE(wolf_energy == Catch::Approx(-0.3302735899347252));
}

TEST_CASE("Dreiding type HB repulsion", "[interaction]") {
    occ::IVec els_a(3);
    els_a << 8, 1, 1;
    occ::Mat3N pos_a(3, 3);
    pos_a << -0.70219605, -0.05606026, 0.00994226, -1.02219322, 0.84677578,
        -0.01148871, 0.25752106, 0.0421215, 0.005219;
    pos_a.transposeInPlace();
    occ::IVec els_b(3);
    els_b << 8, 1, 1;

    occ::Mat3N pos_b(3, 3);
    pos_b << 2.21510240e+00, 2.67620530e-02, 6.33988000e-04, 2.59172401e+00,
        -4.11618013e-01, 7.66758370e-01, 2.58736671e+00, -4.49450922e-01,
        -7.44768514e-01;
    pos_b.transposeInPlace();

    occ::core::Dimer dimer(occ::core::Molecule(els_a, pos_a),
                           occ::core::Molecule(els_b, pos_b));

    double t = occ::interaction::dreiding_type_hb_correction(6.6, 2.2, dimer);
    REQUIRE(t == Catch::Approx(-1.95340359422375));
}
