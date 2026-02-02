#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/multipole_source.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cartesian_force.h>
#include <occ/mults/rigid_body.h>
#include <fmt/core.h>
#include <cmath>

using namespace occ;
using namespace occ::mults;
using namespace occ::dma;
using Approx = Catch::Approx;

// -------------------------------------------------------------------
// Test 1: Construction + cartesian() access
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource construction and cartesian access",
          "[multipole_source]") {

    SECTION("Body-frame construction with identity rotation") {
        Mult m(4);
        m.Q00() = 1.0;
        m.Q10() = 0.5;

        MultipoleSource::BodySite site;
        site.multipole = m;
        site.offset = Vec3(1.0, 0.0, 0.0);

        MultipoleSource source({site});
        // Default orientation: identity rotation, origin center
        REQUIRE(source.num_sites() == 1);
        REQUIRE(source.rotation().isApprox(Mat3::Identity()));
        REQUIRE(source.center().isApprox(Vec3::Zero()));

        const auto &cart = source.cartesian();
        REQUIRE(cart.sites.size() == 1);
        REQUIRE(cart.sites[0].position.isApprox(Vec3(1.0, 0.0, 0.0)));
        REQUIRE(cart.sites[0].rank >= 0);
    }

    SECTION("Single-site lab-frame convenience constructor") {
        Mult m(4);
        m.Q00() = 2.0;

        MultipoleSource source(m, Vec3(3.0, 0.0, 0.0));
        REQUIRE(source.num_sites() == 1);

        const auto &cart = source.cartesian();
        REQUIRE(cart.sites.size() == 1);
        REQUIRE(cart.sites[0].position.isApprox(Vec3(3.0, 0.0, 0.0)));
        REQUIRE(cart.sites[0].cart.data[0] == Approx(2.0));
    }

    SECTION("from_lab_sites construction") {
        Mult m1(4), m2(4);
        m1.Q00() = 1.0;
        m2.Q00() = -0.5;

        auto source = MultipoleSource::from_lab_sites({
            {m1, Vec3(0.0, 0.0, 0.0)},
            {m2, Vec3(2.0, 0.0, 0.0)}
        });
        REQUIRE(source.num_sites() == 2);

        const auto &cart = source.cartesian();
        REQUIRE(cart.sites.size() == 2);
        REQUIRE(cart.sites[0].position.isApprox(Vec3(0.0, 0.0, 0.0)));
        REQUIRE(cart.sites[1].position.isApprox(Vec3(2.0, 0.0, 0.0)));
    }
}

// -------------------------------------------------------------------
// Test 2: Potential matches manual probe
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource potential matches manual probe",
          "[multipole_source]") {

    Mult m(4);
    m.Q00() = -0.669;
    m.Q10() = 0.234;
    m.Q20() = -0.123;

    MultipoleSource source(m, Vec3(0.0, 0.0, 0.0));
    Vec3 point(3.0, 2.0, 1.0);

    double phi = source.compute_potential(point);

    // Manual probe computation
    CartesianSite probe;
    probe.cart.data[0] = 1.0;
    probe.position = point;
    probe.rank = 0;

    double manual = compute_site_pair_energy(source.cartesian().sites[0], probe);

    REQUIRE(phi == Approx(manual).margin(1e-14));
}

// -------------------------------------------------------------------
// Test 3: Batch matches single-point
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource batch potential matches single-point",
          "[multipole_source]") {

    Mult m(4);
    m.Q00() = 1.0;
    m.Q10() = 0.3;
    m.Q22c() = 0.1;

    MultipoleSource source(m, Vec3(0.0, 0.0, 0.0));

    Mat3N grid(3, 4);
    grid.col(0) = Vec3(3.0, 0.0, 0.0);
    grid.col(1) = Vec3(0.0, 4.0, 0.0);
    grid.col(2) = Vec3(0.0, 0.0, 5.0);
    grid.col(3) = Vec3(2.0, 2.0, 2.0);

    Mat3NConstRef grid_ref = grid;
    Vec batch = source.compute_potential(grid_ref);

    for (int i = 0; i < 4; ++i) {
        double single = source.compute_potential(Vec3(grid.col(i)));
        REQUIRE(batch(i) == Approx(single).margin(1e-14));
    }
}

// -------------------------------------------------------------------
// Test 4: Field matches finite-difference of potential
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource field matches finite-difference of potential",
          "[multipole_source]") {

    Mult m(4);
    m.Q00() = 1.0;
    m.Q10() = 0.5;
    m.Q20() = -0.2;
    m.Q11c() = 0.3;

    MultipoleSource source(m, Vec3(0.0, 0.0, 0.0));
    Vec3 point(3.0, 2.0, 1.0);

    Vec3 field = source.compute_field(point);

    const double h = 1e-6;
    Vec3 fd_field;
    for (int d = 0; d < 3; ++d) {
        Vec3 pp = point, pm = point;
        pp[d] += h;
        pm[d] -= h;
        double phi_p = source.compute_potential(pp);
        double phi_m = source.compute_potential(pm);
        fd_field[d] = -(phi_p - phi_m) / (2.0 * h);
    }

    for (int d = 0; d < 3; ++d) {
        INFO(fmt::format("d={}: field={:.10e} fd={:.10e}", d, field[d], fd_field[d]));
        REQUIRE(field[d] == Approx(fd_field[d]).margin(1e-8));
    }
}

// -------------------------------------------------------------------
// Test 5: Field matches negative gradient from force kernel
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource field matches negative gradient",
          "[multipole_source]") {

    Mult m(4);
    m.Q00() = -0.669;
    m.Q10() = 0.234;
    m.Q20() = -0.123;

    MultipoleSource source(m, Vec3(0.0, 0.0, 0.0));
    Vec3 point(3.0, 2.0, 1.0);

    Vec3 field = source.compute_field(point);

    // Manual: E = -gradient of site-probe energy
    CartesianSite probe;
    probe.cart.data[0] = 1.0;
    probe.position = point;
    probe.rank = 0;

    auto ef = compute_site_pair_energy_force(
        source.cartesian().sites[0], probe);
    Vec3 expected = -ef.gradient;

    for (int d = 0; d < 3; ++d) {
        REQUIRE(field[d] == Approx(expected[d]).margin(1e-14));
    }
}

// -------------------------------------------------------------------
// Test 6: Multi-site water molecule
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource multi-site water potential is sum of sites",
          "[multipole_source]") {

    Mult qO(4), qH(4);
    qO.Q00() = -0.669;
    qO.Q10() = 0.234;
    qH.Q00() = 0.335;

    Vec3 offsetO(0.0, 0.0, 0.0);
    Vec3 offsetH1(1.8, 0.0, 0.0);
    Vec3 offsetH2(-0.6, 1.7, 0.0);

    MultipoleSource::BodySite sO, sH1, sH2;
    sO.multipole = qO;
    sO.offset = offsetO;
    sH1.multipole = qH;
    sH1.offset = offsetH1;
    sH2.multipole = qH;
    sH2.offset = offsetH2;

    MultipoleSource source({sO, sH1, sH2});
    REQUIRE(source.num_sites() == 3);

    Vec3 point(5.0, 3.0, 2.0);
    double phi = source.compute_potential(point);

    // Sum individual single-site sources
    MultipoleSource srcO(qO, offsetO);
    MultipoleSource srcH1(qH, offsetH1);
    MultipoleSource srcH2(qH, offsetH2);
    double sum = srcO.compute_potential(point)
               + srcH1.compute_potential(point)
               + srcH2.compute_potential(point);

    REQUIRE(phi == Approx(sum).margin(1e-14));
}

// -------------------------------------------------------------------
// Test 7: Orientation update invalidation
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource orientation update invalidates Cartesian",
          "[multipole_source]") {

    Mult m(4);
    m.Q00() = 1.0;
    m.Q10() = 0.5;

    MultipoleSource::BodySite site;
    site.multipole = m;
    site.offset = Vec3(1.0, 0.0, 0.0);

    MultipoleSource source({site});
    Vec3 point(5.0, 0.0, 0.0);

    double phi1 = source.compute_potential(point);

    // Rotate 90 degrees about z: site moves from (1,0,0) to (0,1,0)
    Mat3 rot90z;
    rot90z << 0, -1, 0,
              1,  0, 0,
              0,  0, 1;
    source.set_orientation(rot90z, Vec3::Zero());

    double phi2 = source.compute_potential(point);

    // Energy should differ because multipole is rotated and site moved
    REQUIRE(phi1 != Approx(phi2).margin(1e-10));

    // Verify the site position changed
    const auto &cart = source.cartesian();
    REQUIRE(cart.sites[0].position.isApprox(Vec3(0.0, 1.0, 0.0), 1e-14));
}

// -------------------------------------------------------------------
// Test 8: RigidBodyState bridge
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource RigidBodyState bridge",
          "[multipole_source]") {

    RigidBodyState rb("test", 0, Vec3(2.0, 0.0, 0.0), 1.0, 4);
    rb.multipole_body.Q00() = 1.0;
    rb.multipole_body.Q10() = 0.5;
    // Identity quaternion (default) → identity rotation

    auto source = multipole_source_from_rigid_body(rb);
    REQUIRE(source.num_sites() == 1);

    Vec3 point(5.0, 0.0, 0.0);
    double phi = source.compute_potential(point);

    // Compare with direct Cartesian computation
    auto mol = CartesianMolecule::from_lab_sites(
        {{rb.multipole_body, rb.position}});
    CartesianSite probe;
    probe.cart.data[0] = 1.0;
    probe.position = point;
    probe.rank = 0;
    double ref = compute_site_pair_energy(mol.sites[0], probe);

    REQUIRE(phi == Approx(ref).margin(1e-14));

    // Test sync_orientation after rotating the rigid body
    rb.set_euler_angles(0.5, 0.3, 0.2);
    sync_orientation(source, rb);
    double phi_rotated = source.compute_potential(point);

    // Build reference with rotated multipole
    auto mol_rot = CartesianMolecule::from_body_frame_with_rotation(
        {{rb.multipole_body, Vec3::Zero()}}, rb.rotation_matrix(), rb.position);
    double ref_rotated = compute_site_pair_energy(mol_rot.sites[0], probe);

    REQUIRE(phi_rotated == Approx(ref_rotated).margin(1e-14));
}

// -------------------------------------------------------------------
// Test 9: Pairwise via cartesian()
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource pairwise via cartesian()",
          "[multipole_source]") {

    Mult m1(4), m2(4);
    m1.Q00() = -0.669;
    m1.Q10() = 0.234;
    m1.Q20() = -0.123;
    m2.Q00() = 0.5;
    m2.Q11c() = 0.1;

    MultipoleSource srcA(m1, Vec3(0.0, 0.0, 0.0));
    MultipoleSource srcB(m2, Vec3(5.0, 2.0, 1.0));

    double mol_E = compute_molecule_interaction(
        srcA.cartesian(), srcB.cartesian());

    // Compare with direct CartesianInteractions
    auto molA = CartesianMolecule::from_lab_sites({{m1, Vec3(0.0, 0.0, 0.0)}});
    auto molB = CartesianMolecule::from_lab_sites({{m2, Vec3(5.0, 2.0, 1.0)}});
    double ref_E = compute_molecule_interaction(molA, molB);

    REQUIRE(mol_E == Approx(ref_E).margin(1e-14));
}

// -------------------------------------------------------------------
// Test 10: from_lab_sites matches body-frame with identity rotation
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource from_lab_sites matches body-frame identity",
          "[multipole_source]") {

    Mult m1(4), m2(4);
    m1.Q00() = 1.0;
    m1.Q10() = 0.3;
    m2.Q00() = -0.5;
    m2.Q11c() = 0.2;

    Vec3 p1(0.0, 0.0, 0.0), p2(2.0, 1.0, 0.0);

    auto lab_source = MultipoleSource::from_lab_sites({{m1, p1}, {m2, p2}});

    MultipoleSource::BodySite bs1, bs2;
    bs1.multipole = m1;
    bs1.offset = p1;
    bs2.multipole = m2;
    bs2.offset = p2;
    MultipoleSource body_source({bs1, bs2});
    // Identity rotation + zero center → same as lab sites

    Vec3 point(5.0, 3.0, 2.0);
    double phi_lab = lab_source.compute_potential(point);
    double phi_body = body_source.compute_potential(point);

    REQUIRE(phi_lab == Approx(phi_body).margin(1e-14));
}

// -------------------------------------------------------------------
// Benchmark: SIMD batch vs manual probe loop
// -------------------------------------------------------------------

TEST_CASE("MultipoleSource batch benchmark",
          "[multipole_source][benchmark]") {

    // Hexadecapole source: exercises the most expensive rank (4)
    Mult m(4);
    m.Q00() = -0.669;
    m.Q10() = 0.234;
    m.Q11c() = 0.15;
    m.Q11s() = -0.08;
    m.Q20() = -0.123;
    m.Q22c() = 0.05;
    m.Q30() = 0.01;
    m.Q40() = 0.005;

    MultipoleSource source(m, Vec3(0.0, 0.0, 0.0));
    const auto &cart = source.cartesian();

    // Build 10x10x10 = 1000 grid points
    constexpr int Ngrid = 10;
    constexpr int N = Ngrid * Ngrid * Ngrid;
    Mat3N grid(3, N);
    int idx = 0;
    for (int ix = 0; ix < Ngrid; ++ix) {
        for (int iy = 0; iy < Ngrid; ++iy) {
            for (int iz = 0; iz < Ngrid; ++iz) {
                grid(0, idx) = 3.0 + ix * 0.5;
                grid(1, idx) = -2.0 + iy * 0.5;
                grid(2, idx) = -2.0 + iz * 0.5;
                ++idx;
            }
        }
    }
    Mat3NConstRef grid_ref = grid;

    SECTION("Potential: 1000 grid points") {
        // Verify both give same answer
        Vec batch_result = source.compute_potential(grid_ref);

        Vec manual_result = Vec::Zero(N);
        CartesianSite probe;
        probe.cart.data[0] = 1.0;
        probe.rank = 0;
        for (int p = 0; p < N; ++p) {
            probe.position = grid.col(p);
            manual_result(p) = compute_site_pair_energy(cart.sites[0], probe);
        }

        for (int p = 0; p < N; ++p) {
            REQUIRE(batch_result(p) == Approx(manual_result(p)).margin(1e-13));
        }

        BENCHMARK("Manual probe loop (1000 pts)") {
            Vec result = Vec::Zero(N);
            CartesianSite pr;
            pr.cart.data[0] = 1.0;
            pr.rank = 0;
            for (int p = 0; p < N; ++p) {
                pr.position = grid.col(p);
                result(p) = compute_site_pair_energy(cart.sites[0], pr);
            }
            return result;
        };

        BENCHMARK("MultipoleSource batch (1000 pts)") {
            return source.compute_potential(grid_ref);
        };
    }

    SECTION("Field: 1000 grid points") {
        Mat3N batch_result = source.compute_field(grid_ref);

        Mat3N manual_result = Mat3N::Zero(3, N);
        CartesianSite probe;
        probe.cart.data[0] = 1.0;
        probe.rank = 0;
        for (int p = 0; p < N; ++p) {
            probe.position = grid.col(p);
            auto ef = compute_site_pair_energy_force(cart.sites[0], probe);
            manual_result.col(p) = -ef.gradient;
        }

        for (int p = 0; p < N; ++p) {
            for (int d = 0; d < 3; ++d) {
                REQUIRE(batch_result(d, p) ==
                        Approx(manual_result(d, p)).margin(1e-13));
            }
        }

        BENCHMARK("Manual probe+force loop (1000 pts)") {
            Mat3N result = Mat3N::Zero(3, N);
            CartesianSite pr;
            pr.cart.data[0] = 1.0;
            pr.rank = 0;
            for (int p = 0; p < N; ++p) {
                pr.position = grid.col(p);
                auto ef = compute_site_pair_energy_force(cart.sites[0], pr);
                result.col(p) = -ef.gradient;
            }
            return result;
        };

        BENCHMARK("MultipoleSource field batch (1000 pts)") {
            return source.compute_field(grid_ref);
        };
    }
}
