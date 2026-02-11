#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/interaction_tensor.h>
#include <occ/mults/cartesian_multipole.h>
#include <occ/mults/cartesian_interaction.h>
#include <occ/mults/multipole_interactions.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/core/units.h>
#include <fmt/core.h>
#include <cmath>
#include <string>
#include <vector>

#include "orient_reference_data.h"

using namespace occ;
using namespace occ::mults;
using namespace occ::dma;
using Approx = Catch::Approx;
using occ::ints::hermite_index;

// -------------------------------------------------------------------
// Stage 1: Interaction tensor tests
// -------------------------------------------------------------------

TEST_CASE("Interaction tensor analytical values",
          "[cartesian][tensor]") {

    SECTION("T_000 = 1/R for R along z") {
        double R = 3.0;
        InteractionTensor<4> T;
        compute_interaction_tensor<4>(0.0, 0.0, R, T);

        REQUIRE(T(0, 0, 0) == Approx(1.0 / R));
    }

    SECTION("T_001 = -Rz/R^3 for R along z") {
        double R = 3.0;
        InteractionTensor<4> T;
        compute_interaction_tensor<4>(0.0, 0.0, R, T);

        double expected = -R / (R * R * R); // -Rz/R^3
        REQUIRE(T(0, 0, 1) == Approx(expected));
    }

    SECTION("T_002 = (3Rz^2 - R^2) / R^5 for R along z") {
        double Rz = 3.0;
        double R = Rz;
        InteractionTensor<4> T;
        compute_interaction_tensor<4>(0.0, 0.0, Rz, T);

        double R5 = R * R * R * R * R;
        double expected = (3.0 * Rz * Rz - R * R) / R5;
        REQUIRE(T(0, 0, 2) == Approx(expected));
    }

    SECTION("T_100 = -Rx/R^3 for general R") {
        double Rx = 1.0, Ry = 2.0, Rz = 3.0;
        double R2 = Rx * Rx + Ry * Ry + Rz * Rz;
        double R = std::sqrt(R2);
        double R3 = R * R * R;

        InteractionTensor<4> T;
        compute_interaction_tensor<4>(Rx, Ry, Rz, T);

        REQUIRE(T(1, 0, 0) == Approx(-Rx / R3));
        REQUIRE(T(0, 1, 0) == Approx(-Ry / R3));
        REQUIRE(T(0, 0, 1) == Approx(-Rz / R3));
    }

    SECTION("Laplacian of T vanishes: T_200 + T_020 + T_002 = 0") {
        double Rx = 1.5, Ry = -2.3, Rz = 0.7;
        InteractionTensor<4> T;
        compute_interaction_tensor<4>(Rx, Ry, Rz, T);

        double laplacian = T(2, 0, 0) + T(0, 2, 0) + T(0, 0, 2);
        REQUIRE(laplacian == Approx(0.0).margin(1e-12));
    }

    SECTION("Higher-order Laplacian: T_{t+2,u,v} + T_{t,u+2,v} + T_{t,u,v+2} = 0") {
        double Rx = 1.5, Ry = -2.3, Rz = 0.7;
        InteractionTensor<6> T;
        compute_interaction_tensor<6>(Rx, Ry, Rz, T);

        for (int l = 2; l <= 4; ++l) {
            for (int t = 0; t <= l - 2; ++t) {
                for (int u = 0; u <= l - 2 - t; ++u) {
                    int v = l - 2 - t - u;
                    double lap = T(t + 2, u, v) + T(t, u + 2, v) + T(t, u, v + 2);
                    INFO(fmt::format("Laplacian at ({},{},{}): {:.2e}", t, u, v, lap));
                    REQUIRE(lap == Approx(0.0).margin(1e-10));
                }
            }
        }
    }
}

// -------------------------------------------------------------------
// Stage 2: Spherical-to-Cartesian conversion tests
// -------------------------------------------------------------------

TEST_CASE("Spherical to Cartesian conversion",
          "[cartesian][conversion]") {

    SECTION("L=0: monopole") {
        Mult m(0);
        m.Q00() = 2.5;
        CartesianMultipole<4> cart;
        spherical_to_cartesian<4>(m, cart);
        REQUIRE(cart(0, 0, 0) == Approx(2.5));
    }

    SECTION("L=1: dipole") {
        Mult m(1);
        m.Q10() = 1.0;
        m.Q11c() = 2.0;
        m.Q11s() = 3.0;
        CartesianMultipole<4> cart;
        spherical_to_cartesian<4>(m, cart);

        REQUIRE(cart(0, 0, 1) == Approx(1.0)); // z = Q10
        REQUIRE(cart(1, 0, 0) == Approx(2.0)); // x = Q11c
        REQUIRE(cart(0, 1, 0) == Approx(3.0)); // y = Q11s
    }

    SECTION("L=2: tracelessness") {
        Mult m(2);
        m.Q20() = 1.0;
        m.Q22c() = 0.5;
        CartesianMultipole<4> cart;
        spherical_to_cartesian<4>(m, cart);

        double trace = cart(2, 0, 0) + cart(0, 2, 0) + cart(0, 0, 2);
        REQUIRE(trace == Approx(0.0).margin(1e-14));
    }

    SECTION("L=2: specific values from Q20=1") {
        Mult m(4);
        m.Q20() = 1.0;
        CartesianMultipole<4> cart;
        spherical_to_cartesian<4>(m, cart);

        REQUIRE(cart(0, 0, 2) == Approx(2.0 / 3.0));
        REQUIRE(cart(2, 0, 0) == Approx(-1.0 / 3.0));
        REQUIRE(cart(0, 2, 0) == Approx(-1.0 / 3.0));
    }

    SECTION("L=3: tracelessness") {
        Mult m(4);
        m.Q30() = 1.0;
        m.Q31c() = 0.5;
        m.Q32c() = 0.3;
        m.Q33c() = 0.2;
        CartesianMultipole<4> cart;
        spherical_to_cartesian<4>(m, cart);

        // Trace conditions for rank 3
        double trace1 = cart(3, 0, 0) + cart(1, 2, 0) + cart(1, 0, 2);
        double trace2 = cart(2, 1, 0) + cart(0, 3, 0) + cart(0, 1, 2);
        double trace3 = cart(2, 0, 1) + cart(0, 2, 1) + cart(0, 0, 3);

        REQUIRE(trace1 == Approx(0.0).margin(1e-14));
        REQUIRE(trace2 == Approx(0.0).margin(1e-14));
        REQUIRE(trace3 == Approx(0.0).margin(1e-14));
    }

    SECTION("L=4: tracelessness") {
        Mult m(4);
        m.Q40() = 1.0;
        m.Q42c() = 0.5;
        m.Q44c() = 0.3;
        CartesianMultipole<4> cart;
        spherical_to_cartesian<4>(m, cart);

        // Trace conditions for rank 4: sum over any pair of indices
        double tr1 = cart(4, 0, 0) + cart(2, 2, 0) + cart(2, 0, 2);
        double tr2 = cart(2, 2, 0) + cart(0, 4, 0) + cart(0, 2, 2);
        double tr3 = cart(2, 0, 2) + cart(0, 2, 2) + cart(0, 0, 4);
        double tr4 = cart(3, 1, 0) + cart(1, 3, 0) + cart(1, 1, 2);
        double tr5 = cart(3, 0, 1) + cart(1, 2, 1) + cart(1, 0, 3);
        double tr6 = cart(2, 1, 1) + cart(0, 3, 1) + cart(0, 1, 3);

        REQUIRE(tr1 == Approx(0.0).margin(1e-13));
        REQUIRE(tr2 == Approx(0.0).margin(1e-13));
        REQUIRE(tr3 == Approx(0.0).margin(1e-13));
        REQUIRE(tr4 == Approx(0.0).margin(1e-13));
        REQUIRE(tr5 == Approx(0.0).margin(1e-13));
        REQUIRE(tr6 == Approx(0.0).margin(1e-13));
    }
}

// -------------------------------------------------------------------
// Stage 3: Energy cross-validation against analytical values
// -------------------------------------------------------------------

TEST_CASE("Cartesian interaction energy analytical",
          "[cartesian][energy]") {

    CartesianInteractions engine;

    SECTION("Charge-charge at R=3 along z: E = 1/3") {
        Mult m1(4), m2(4);
        m1.Q00() = 1.0;
        m2.Q00() = 1.0;
        Vec3 p1(0, 0, 0), p2(0, 0, 3);

        double E = engine.compute_interaction_energy(m1, p1, m2, p2);
        REQUIRE(E == Approx(1.0 / 3.0));
    }

    SECTION("Charge-dipole Q00*Q10 at R=3 along z: E = -1/9") {
        Mult m1(4), m2(4);
        m1.Q00() = 1.0;
        m2.Q10() = 1.0;
        Vec3 p1(0, 0, 0), p2(0, 0, 3);

        double E = engine.compute_interaction_energy(m1, p1, m2, p2);
        REQUIRE(E == Approx(-1.0 / 9.0));
    }

    SECTION("Dipole-dipole Q10*Q10 at R=3 along z: E = -2/27") {
        Mult m1(4), m2(4);
        m1.Q10() = 1.0;
        m2.Q10() = 1.0;
        Vec3 p1(0, 0, 0), p2(0, 0, 3);

        double E = engine.compute_interaction_energy(m1, p1, m2, p2);
        REQUIRE(E == Approx(-2.0 / 27.0));
    }

    SECTION("Charge-quadrupole Q00*Q20 at R=3 along z: E = 1/27") {
        Mult m1(4), m2(4);
        m1.Q00() = 1.0;
        m2.Q20() = 1.0;
        Vec3 p1(0, 0, 0), p2(0, 0, 3);

        double E = engine.compute_interaction_energy(m1, p1, m2, p2);
        REQUIRE(E == Approx(1.0 / 27.0));
    }

    SECTION("Charge-charge at R=5: E = 1/5 = 0.2") {
        Mult m1(4), m2(4);
        m1.Q00() = 1.0;
        m2.Q00() = 1.0;
        Vec3 p1(0, 0, 0), p2(5, 0, 0);

        double E = engine.compute_interaction_energy(m1, p1, m2, p2);
        REQUIRE(E == Approx(0.2));
    }
}

// -------------------------------------------------------------------
// Stage 4: Cross-validation against S-function engine
// -------------------------------------------------------------------

namespace {

// Return the multipole rank for component index i in the 25-component
// ordering: Q00, Q10..Q11s, Q20..Q22s, Q30..Q33s, Q40..Q44s
int component_rank(int i) {
    if (i < 1) return 0;
    if (i < 4) return 1;
    if (i < 9) return 2;
    if (i < 16) return 3;
    return 4;
}

} // anonymous namespace

TEST_CASE("Cartesian vs S-function engine cross-validation",
          "[cartesian][cross_validation]") {

    CartesianInteractions cart_engine;
    MultipoleInteractions::Config sf_config;
    sf_config.max_rank = 4;
    MultipoleInteractions sf_engine(sf_config);

    Vec3 pos1(0.0, 0.0, 0.0);

    // All 25 multipole component names
    const std::vector<std::string> components = {
        "Q00",
        "Q10", "Q11c", "Q11s",
        "Q20", "Q21c", "Q21s", "Q22c", "Q22s",
        "Q30", "Q31c", "Q31s", "Q32c", "Q32s", "Q33c", "Q33s",
        "Q40", "Q41c", "Q41s", "Q42c", "Q42s", "Q43c", "Q43s", "Q44c", "Q44s"
    };

    // 14 test directions at distance 5
    const std::vector<Vec3> directions = {
        Vec3( 5, 0, 0), Vec3(-5, 0, 0),
        Vec3( 0, 5, 0), Vec3( 0,-5, 0),
        Vec3( 0, 0, 5), Vec3( 0, 0,-5),
        Vec3( 2.886751345948129, 2.886751345948129, 2.886751345948129),
        Vec3( 2.886751345948129, 2.886751345948129,-2.886751345948129),
        Vec3( 2.886751345948129,-2.886751345948129, 2.886751345948129),
        Vec3( 2.886751345948129,-2.886751345948129,-2.886751345948129),
        Vec3(-2.886751345948129, 2.886751345948129, 2.886751345948129),
        Vec3(-2.886751345948129, 2.886751345948129,-2.886751345948129),
        Vec3(-2.886751345948129,-2.886751345948129, 2.886751345948129),
        Vec3(-2.886751345948129,-2.886751345948129,-2.886751345948129),
    };

    // The S-function engine only supports rank sums up to 4 (i.e. the
    // interaction rank for which S-functions are implemented).  The
    // Cartesian engine handles all rank sums up to 2*MaxL = 8.
    // Cross-validate only the pairs where both engines are expected to
    // produce results (rank_A + rank_B <= 4).
    constexpr int sf_max_rank_sum = 4;

    int total = 0, passed = 0, failed = 0, skipped_zero = 0, skipped_ranksum = 0;

    for (int i = 0; i < static_cast<int>(components.size()); ++i) {
        int rank_i = component_rank(i);
        for (int j = i; j < static_cast<int>(components.size()); ++j) {
            int rank_j = component_rank(j);
            if (rank_i + rank_j > sf_max_rank_sum) {
                ++skipped_ranksum;
                continue;
            }

            for (int d = 0; d < static_cast<int>(directions.size()); ++d) {
                Mult m1(4), m2(4);
                m1.get_component(components[i]) = 1.0;
                m2.get_component(components[j]) = 1.0;

                Vec3 pos2 = directions[d];

                double sf_energy = sf_engine.compute_interaction_energy(
                    m1, pos1, m2, pos2);
                double cart_energy = cart_engine.compute_interaction_energy(
                    m1, pos1, m2, pos2);

                ++total;
                if (std::abs(sf_energy) < 1e-15 && std::abs(cart_energy) < 1e-15) {
                    ++skipped_zero;
                    continue;
                }

                double diff = std::abs(cart_energy - sf_energy);
                bool match = diff < 1e-10
                    || diff < 1e-6 * std::max(std::abs(sf_energy), std::abs(cart_energy));

                if (match) {
                    ++passed;
                } else {
                    ++failed;
                    UNSCOPED_INFO(fmt::format(
                        "MISMATCH {} x {} dir{}: SF={:.10e} Cart={:.10e} diff={:.2e}",
                        components[i], components[j], d,
                        sf_energy, cart_energy, diff));
                }

                CHECK(match);
            }
        }
    }

    UNSCOPED_INFO(fmt::format(
        "Cross-validation: {} tested, {} passed, {} zero, {} skipped (rank sum > {}), {} failed",
        total, passed, skipped_zero, skipped_ranksum, sf_max_rank_sum, failed));
    REQUIRE(failed == 0);
}

// -------------------------------------------------------------------
// Stage 4b: Cartesian engine against Orient reference data
// -------------------------------------------------------------------

TEST_CASE("Cartesian engine vs Orient reference",
          "[cartesian][orient_reference]") {

    CartesianInteractions engine;
    Vec3 pos1(0.0, 0.0, 0.0);

    int passed = 0, failed = 0;

    for (int i = 0; i < NUM_ORIENT_REFERENCE_ENTRIES; i++) {
        const auto &entry = ORIENT_REFERENCE_DATA[i];
        Vec3 pos2(entry.pos2_x, entry.pos2_y, entry.pos2_z);

        Mult m1(4), m2(4);
        m1.get_component(entry.site1) = 1.0;
        m2.get_component(entry.site2) = 1.0;

        double energy = engine.compute_interaction_energy(m1, pos1, m2, pos2);

        std::string label = fmt::format("{} x {} dir{}",
                                         entry.site1, entry.site2,
                                         entry.direction_idx);

        bool match = std::abs(energy - entry.energy_hartree) < 1e-6;
        if (match) {
            passed++;
        } else {
            failed++;
            UNSCOPED_INFO(fmt::format(
                "MISMATCH {}: Cart={:.10e} Orient={:.10e} diff={:.2e}",
                label, energy, entry.energy_hartree,
                std::abs(energy - entry.energy_hartree)));
        }

        REQUIRE(energy == Approx(entry.energy_hartree).margin(1e-6));
    }

    UNSCOPED_INFO(fmt::format(
        "Orient reference: {} tested, {} passed, {} failed",
        NUM_ORIENT_REFERENCE_ENTRIES, passed, failed));
}

// -------------------------------------------------------------------
// Stage 5: Molecule batch API tests
// -------------------------------------------------------------------

TEST_CASE("Molecule batch API matches per-pair engine",
          "[cartesian][molecule]") {

    CartesianInteractions engine;

    SECTION("Two single-site molecules (charge-charge)") {
        Mult m1(4), m2(4);
        m1.Q00() = 1.0;
        m2.Q00() = 1.0;
        constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
        Vec3 p1(0, 0, 0), p2(0, 0, 3);

        auto molA = CartesianMolecule::from_lab_sites({{m1, p1 * B2A}});
        auto molB = CartesianMolecule::from_lab_sites({{m2, p2 * B2A}});

        double batch_E = compute_molecule_interaction(molA, molB);
        double pair_E = engine.compute_interaction_energy(m1, p1, m2, p2);

        REQUIRE(batch_E == Approx(pair_E * occ::units::AU_TO_KJ_PER_MOL).margin(1e-10));
    }

    SECTION("Single-site with full multipoles") {
        Mult m1(4), m2(4);
        m1.Q00() = -0.669;
        m1.Q10() = 0.234;
        m1.Q20() = -0.123;
        m2.Q00() = 0.5;
        m2.Q11c() = 0.1;
        m2.Q22c() = 0.05;
        constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
        Vec3 p1(0, 0, 0), p2(3, 2, 1);

        auto molA = CartesianMolecule::from_lab_sites({{m1, p1 * B2A}});
        auto molB = CartesianMolecule::from_lab_sites({{m2, p2 * B2A}});

        double batch_E = compute_molecule_interaction(molA, molB);
        double pair_E = engine.compute_interaction_energy(m1, p1, m2, p2);

        REQUIRE(batch_E == Approx(pair_E * occ::units::AU_TO_KJ_PER_MOL).margin(1e-10));
    }

    SECTION("Multi-site water-like molecules") {
        // 3 sites per molecule
        Mult qO(4), qH(4);
        qO.Q00() = -0.669;
        qO.Q10() = 0.234;
        qH.Q00() = 0.335;

        constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
        Vec3 pO1(0, 0, 0), pH1a(1.8, 0, 0), pH1b(-0.6, 1.7, 0);
        Vec3 pO2(5, 0, 0), pH2a(6.8, 0, 0), pH2b(4.4, 1.7, 0);

        auto molA = CartesianMolecule::from_lab_sites({
            {qO, pO1 * B2A}, {qH, pH1a * B2A}, {qH, pH1b * B2A}
        });
        auto molB = CartesianMolecule::from_lab_sites({
            {qO, pO2 * B2A}, {qH, pH2a * B2A}, {qH, pH2b * B2A}
        });

        double batch_E = compute_molecule_interaction(molA, molB);

        // Compute per-pair reference (engine works in Bohr/Hartree)
        double ref_E = 0.0;
        std::vector<std::pair<Mult, Vec3>> sitesA = {{qO, pO1}, {qH, pH1a}, {qH, pH1b}};
        std::vector<std::pair<Mult, Vec3>> sitesB = {{qO, pO2}, {qH, pH2a}, {qH, pH2b}};
        for (const auto &[mA, pA] : sitesA) {
            for (const auto &[mB, pB] : sitesB) {
                ref_E += engine.compute_interaction_energy(mA, pA, mB, pB);
            }
        }

        REQUIRE(batch_E == Approx(ref_E * occ::units::AU_TO_KJ_PER_MOL).margin(1e-10));
    }

    SECTION("Body-frame construction matches lab-frame") {
        Mult m1(4), m2(4);
        m1.Q00() = 1.0;
        m1.Q10() = 0.5;
        m2.Q00() = -0.5;

        // Positions in Angstrom
        Vec3 body_pos(1.0, 0, 0);
        Mat3 rot = Mat3::Identity();
        Vec3 center(3, 0, 0);

        auto mol_body = CartesianMolecule::from_body_frame(
            {{m1, body_pos}}, rot, center);
        auto mol_lab = CartesianMolecule::from_lab_sites(
            {{m1, center + body_pos}});

        auto mol_other = CartesianMolecule::from_lab_sites(
            {{m2, Vec3(0, 0, 0)}});

        double body_E = compute_molecule_interaction(mol_body, mol_other);
        double lab_E = compute_molecule_interaction(mol_lab, mol_other);

        REQUIRE(body_E == Approx(lab_E).margin(1e-14));
    }

    SECTION("SIMD batch matches scalar molecule interaction") {
        // 3-site molecules with mixed ranks
        Mult qO(4), qH(4);
        qO.Q00() = -0.669;
        qO.Q10() = 0.234;
        qO.Q20() = -0.123;
        qH.Q00() = 0.335;

        Vec3 pO1(0, 0, 0), pH1a(1.8, 0, 0), pH1b(-0.6, 1.7, 0);
        Vec3 pO2(5, 0, 0), pH2a(6.8, 0, 0), pH2b(4.4, 1.7, 0);

        auto molA = CartesianMolecule::from_lab_sites({
            {qO, pO1}, {qH, pH1a}, {qH, pH1b}
        });
        auto molB = CartesianMolecule::from_lab_sites({
            {qO, pO2}, {qH, pH2a}, {qH, pH2b}
        });

        double scalar_E = compute_molecule_interaction(molA, molB);
        double simd_E = compute_molecule_interaction_simd(molA, molB);

        REQUIRE(simd_E == Approx(scalar_E).margin(1e-14));
    }

    SECTION("site_pair_energy matches molecule interaction for single pair") {
        Mult m1(4), m2(4);
        m1.Q00() = 1.0;
        m1.Q10() = 0.3;
        m2.Q00() = 0.5;
        constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
        Vec3 p1(0, 0, 0), p2(4, 0, 0);

        // Molecule API expects Angstrom positions, returns kJ/mol
        auto molA = CartesianMolecule::from_lab_sites({{m1, p1 * B2A}});
        auto molB = CartesianMolecule::from_lab_sites({{m2, p2 * B2A}});

        double mol_E = compute_molecule_interaction(molA, molB);

        // Site pair energy with Bohr positions returns Hartree
        auto molA_bohr = CartesianMolecule::from_lab_sites({{m1, p1}});
        auto molB_bohr = CartesianMolecule::from_lab_sites({{m2, p2}});
        double pair_E = compute_site_pair_energy(molA_bohr.sites[0], molB_bohr.sites[0]);

        REQUIRE(mol_E == Approx(pair_E * occ::units::AU_TO_KJ_PER_MOL).margin(1e-10));
    }
}
