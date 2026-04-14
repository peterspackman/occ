#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/mults/multipole_coarsening.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cartesian_multipole.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/dma/mult.h>
#include <occ/ints/rints.h>
#include <fmt/core.h>
#include <cmath>
#include <vector>

using namespace occ;
using namespace occ::mults;
using namespace occ::dma;
using Approx = Catch::Approx;
using occ::ints::hermite_index;

// -------------------------------------------------------------------
// Test 1: Charge shift produces exact higher moments
// -------------------------------------------------------------------

TEST_CASE("Charge shift produces exact dipole and quadrupole",
          "[coarsening]") {

    SECTION("Charge q at displacement d produces dipole q*d") {
        CartesianMultipole<4> input;
        double q = 2.5;
        input(0, 0, 0) = q;

        Vec3 d(1.0, -0.5, 0.3);
        CartesianMultipole<4> output;
        shift_multipole_to_origin(input, 0, d, output);

        // Charge preserved
        REQUIRE(output(0, 0, 0) == Approx(q));

        // Dipole = q * d
        REQUIRE(output(1, 0, 0) == Approx(q * d[0]));
        REQUIRE(output(0, 1, 0) == Approx(q * d[1]));
        REQUIRE(output(0, 0, 1) == Approx(q * d[2]));

        // Quadrupole = q * d_i * d_j
        REQUIRE(output(2, 0, 0) == Approx(q * d[0] * d[0]));
        REQUIRE(output(0, 2, 0) == Approx(q * d[1] * d[1]));
        REQUIRE(output(0, 0, 2) == Approx(q * d[2] * d[2]));
        REQUIRE(output(1, 1, 0) == Approx(q * d[0] * d[1]));
        REQUIRE(output(1, 0, 1) == Approx(q * d[0] * d[2]));
        REQUIRE(output(0, 1, 1) == Approx(q * d[1] * d[2]));
    }

    SECTION("Charge at displacement produces octupole q*dx*dy*dz") {
        CartesianMultipole<4> input;
        double q = 1.0;
        input(0, 0, 0) = q;

        Vec3 d(0.5, 0.7, -0.3);
        CartesianMultipole<4> output;
        shift_multipole_to_origin(input, 0, d, output);

        // Octupole component = q * dx * dy * dz
        REQUIRE(output(1, 1, 1) == Approx(q * d[0] * d[1] * d[2]));

        // Hexadecapole component = q * dx^2 * dy * dz
        REQUIRE(output(2, 1, 1) == Approx(q * d[0] * d[0] * d[1] * d[2]));
    }
}

// -------------------------------------------------------------------
// Test 2: Dipole shift produces correct higher moments
// -------------------------------------------------------------------

TEST_CASE("Dipole shift produces correct higher moments",
          "[coarsening]") {

    CartesianMultipole<4> input;
    double mu_z = 1.5;
    input(0, 0, 1) = mu_z; // z-dipole

    Vec3 d(0.0, 0.0, 2.0); // displacement along z
    CartesianMultipole<4> output;
    shift_multipole_to_origin(input, 1, d, output);

    // Charge = 0 (dipole doesn't produce monopole)
    REQUIRE(output(0, 0, 0) == Approx(0.0).margin(1e-15));

    // Dipole preserved: mu_z (from original) only
    REQUIRE(output(0, 0, 1) == Approx(mu_z));
    REQUIRE(output(1, 0, 0) == Approx(0.0).margin(1e-15));

    // Quadrupole: shifted dipole produces M_002 = mu_z * dz
    // From binomial: M'_{002} = C(0,0)*C(0,0)*C(2,1)*dz^1*mu_z = 2*dz*mu_z
    REQUIRE(output(0, 0, 2) == Approx(2.0 * d[2] * mu_z));

    // Cross terms: M'_{101} = C(1,0)*dx^1*C(0,0)*C(1,1)*mu_z = 0 (dx=0)
    REQUIRE(output(1, 0, 1) == Approx(0.0).margin(1e-15));
}

// -------------------------------------------------------------------
// Test 3: Two opposite charges merge to dipole
// -------------------------------------------------------------------

TEST_CASE("Two opposite charges merge to dipole",
          "[coarsening]") {

    // +1 at (0,0,-1 Ang) and -1 at (0,0,+1 Ang), merge at origin.
    // merge_to_single_site converts displacements to Bohr internally.
    constexpr double A2B = 1.0 / occ::units::BOHR_TO_ANGSTROM;
    CartesianMolecule mol;

    CartesianSite site_pos;
    site_pos.cart(0, 0, 0) = 1.0;
    site_pos.position = Vec3(0, 0, -1.0); // Angstrom
    site_pos.rank = 0;

    CartesianSite site_neg;
    site_neg.cart(0, 0, 0) = -1.0;
    site_neg.position = Vec3(0, 0, 1.0); // Angstrom
    site_neg.rank = 0;

    mol.sites.push_back(site_pos);
    mol.sites.push_back(site_neg);

    auto merged = merge_to_single_site(mol, Vec3(0, 0, 0));

    REQUIRE(merged.sites.size() == 1);
    const auto &m = merged.sites[0].cart;

    // Total charge = +1 + (-1) = 0
    REQUIRE(m(0, 0, 0) == Approx(0.0).margin(1e-15));

    // Dipole: sum of q_i * d_i (in Bohr)
    // +1 * (0,0,-A2B) + (-1) * (0,0,+A2B) = (0,0,-2*A2B)
    REQUIRE(m(1, 0, 0) == Approx(0.0).margin(1e-15));
    REQUIRE(m(0, 1, 0) == Approx(0.0).margin(1e-15));
    REQUIRE(m(0, 0, 1) == Approx(-2.0 * A2B));

    // Quadrupole: sum of q_i * d_iz^2 (in Bohr^2)
    // +1 * (-A2B)^2 + (-1) * (A2B)^2 = 0
    REQUIRE(m(0, 0, 2) == Approx(0.0).margin(1e-15));

    // Merged site at origin
    REQUIRE(merged.sites[0].position.norm() == Approx(0.0).margin(1e-15));
}

// -------------------------------------------------------------------
// Test 4: Total charge preservation for multi-site molecule
// -------------------------------------------------------------------

TEST_CASE("Total charge preserved after merge",
          "[coarsening]") {

    // Water-like: O with -0.669, two H with +0.335 each
    Mult qO(2), qH(0);
    qO.Q00() = -0.669;
    qO.Q10() = 0.234;
    qO.Q20() = -0.123;
    qH.Q00() = 0.335;

    constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
    Vec3 pO(0, 0, 0), pH1(1.8 * B2A, 0, 0), pH2(-0.6 * B2A, 1.7 * B2A, 0);

    auto mol = CartesianMolecule::from_lab_sites({
        {qO, pO}, {qH, pH1}, {qH, pH2}
    });

    auto merged = merge_to_single_site(mol);

    // Total charge: -0.669 + 0.335 + 0.335 = 0.001
    double total_q = 0.0;
    for (const auto &site : mol.sites) {
        total_q += site.cart(0, 0, 0);
    }
    REQUIRE(merged.sites[0].cart(0, 0, 0) == Approx(total_q));
}

// -------------------------------------------------------------------
// Test 5: Single-site merge is identity
// -------------------------------------------------------------------

TEST_CASE("Single-site merge is identity",
          "[coarsening]") {

    Mult m(4);
    m.Q00() = 1.0;
    m.Q10() = 0.5;
    m.Q11c() = -0.3;
    m.Q20() = 0.2;
    m.Q33c() = 0.1;

    Vec3 pos(3.0, -1.0, 2.0);
    auto mol = CartesianMolecule::from_lab_sites({{m, pos}});

    // Merge at the site's own position => identity
    auto merged = merge_to_single_site(mol, pos);

    REQUIRE(merged.sites.size() == 1);
    REQUIRE(merged.sites[0].position[0] == Approx(pos[0]));
    REQUIRE(merged.sites[0].position[1] == Approx(pos[1]));
    REQUIRE(merged.sites[0].position[2] == Approx(pos[2]));

    // All multipole components should match
    constexpr int size = CartesianMultipole<4>::size;
    for (int i = 0; i < size; ++i) {
        INFO(fmt::format("Component {}: original={:.10e} merged={:.10e}",
                         i, mol.sites[0].cart.data[i],
                         merged.sites[0].cart.data[i]));
        REQUIRE(merged.sites[0].cart.data[i] ==
                Approx(mol.sites[0].cart.data[i]).margin(1e-14));
    }
}

// -------------------------------------------------------------------
// Test 6: Merged vs exact energy convergence with distance
// -------------------------------------------------------------------

TEST_CASE("Merged energy converges to exact at large distance",
          "[coarsening]") {

    // Build two 3-site "molecules" with mixed multipoles
    Mult qO(2), qH(0);
    qO.Q00() = -0.669;
    qO.Q10() = 0.234;
    qO.Q20() = -0.123;
    qH.Q00() = 0.335;

    // Molecule A centered near origin (positions in Angstrom)
    constexpr double B2A = occ::units::BOHR_TO_ANGSTROM;
    auto molA = CartesianMolecule::from_lab_sites({
        {qO, Vec3(0, 0, 0)},
        {qH, Vec3(1.8 * B2A, 0, 0)},
        {qH, Vec3(-0.6 * B2A, 1.7 * B2A, 0)}
    });

    // Merge A
    auto mergedA = merge_to_single_site(molA);

    // Test at increasing distances (in Bohr, converted to Angstrom).
    // Start at R=10 (molecule size ~2 Bohr, so d/R ~ 0.2).
    std::vector<double> distances_bohr = {10, 20, 40, 80, 160};
    std::vector<double> rel_errors;

    for (double R_bohr : distances_bohr) {
        double R_ang = R_bohr * B2A;

        // Molecule B is a copy of A, shifted along x
        auto molB = CartesianMolecule::from_lab_sites({
            {qO, Vec3(R_ang, 0, 0)},
            {qH, Vec3(R_ang + 1.8 * B2A, 0, 0)},
            {qH, Vec3(R_ang - 0.6 * B2A, 1.7 * B2A, 0)}
        });

        auto mergedB = merge_to_single_site(molB);

        double exact_E = compute_molecule_interaction(molA, molB);
        double merged_E = compute_molecule_interaction(mergedA, mergedB);

        double rel_error = (std::abs(exact_E) > 1e-15)
            ? std::abs(merged_E - exact_E) / std::abs(exact_E)
            : std::abs(merged_E - exact_E);

        INFO(fmt::format("R={:.0f} Bohr: exact={:.8e} merged={:.8e} rel_err={:.4e}",
                         R_bohr, exact_E, merged_E, rel_error));

        rel_errors.push_back(rel_error);
    }

    // Overall convergence: error at largest R should be much smaller than
    // at smallest R
    INFO(fmt::format("Error at R=10: {:.4e}, R=160: {:.4e}",
                     rel_errors.front(), rel_errors.back()));
    REQUIRE(rel_errors.back() < rel_errors.front());

    // At R=160 Bohr, error should be small (< 1%)
    INFO(fmt::format("Final relative error at R=160: {:.4e}", rel_errors.back()));
    REQUIRE(rel_errors.back() < 0.01);
}
