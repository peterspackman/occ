#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <occ/mults/cartesian_interaction.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cartesian_force.h>
#include <occ/mults/multipole_interactions.h>
#include <occ/mults/torque.h>
#include <occ/mults/rotation.h>
#include <fmt/core.h>
#include <cmath>
#include <string>
#include <vector>

using namespace occ;
using namespace occ::mults;
using namespace occ::dma;

namespace {

// All 25 multipole component names (rank 0..4)
const std::vector<std::string> ALL_COMPONENTS = {
    "Q00",
    "Q10", "Q11c", "Q11s",
    "Q20", "Q21c", "Q21s", "Q22c", "Q22s",
    "Q30", "Q31c", "Q31s", "Q32c", "Q32s", "Q33c", "Q33s",
    "Q40", "Q41c", "Q41s", "Q42c", "Q42s", "Q43c", "Q43s", "Q44c", "Q44s"
};

// 14 test directions at distance 5
Vec3 directions[] = {
    Vec3( 5, 0, 0), Vec3(-5, 0, 0),
    Vec3( 0, 5, 0), Vec3( 0,-5, 0),
    Vec3( 0, 0, 5), Vec3( 0, 0,-5),
    Vec3( 2.887, 2.887, 2.887), Vec3( 2.887, 2.887,-2.887),
    Vec3( 2.887,-2.887, 2.887), Vec3( 2.887,-2.887,-2.887),
    Vec3(-2.887, 2.887, 2.887), Vec3(-2.887, 2.887,-2.887),
    Vec3(-2.887,-2.887, 2.887), Vec3(-2.887,-2.887,-2.887),
};

} // anonymous namespace

TEST_CASE("Interaction engine benchmarks", "[mults][benchmark]") {

    CartesianInteractions cart_engine;
    MultipoleInteractions::Config sf_config;
    sf_config.max_rank = 4;
    MultipoleInteractions sf_engine(sf_config);

    Vec3 pos1(0, 0, 0);
    Vec3 pos2(5, 0, 0);

    // Single-pair benchmarks
    SECTION("Charge-charge") {
        Mult m1(4), m2(4);
        m1.Q00() = 1.0;
        m2.Q00() = 1.0;

        BENCHMARK("S-function: charge-charge") {
            return sf_engine.compute_interaction_energy(m1, pos1, m2, pos2);
        };

        BENCHMARK("Cartesian:  charge-charge") {
            return cart_engine.compute_interaction_energy(m1, pos1, m2, pos2);
        };
    }

    SECTION("Dipole-dipole") {
        Mult m1(4), m2(4);
        m1.Q10() = 1.0;
        m2.Q10() = 1.0;

        BENCHMARK("S-function: dipole-dipole") {
            return sf_engine.compute_interaction_energy(m1, pos1, m2, pos2);
        };

        BENCHMARK("Cartesian:  dipole-dipole") {
            return cart_engine.compute_interaction_energy(m1, pos1, m2, pos2);
        };
    }

    SECTION("Quadrupole-quadrupole") {
        Mult m1(4), m2(4);
        m1.Q20() = 1.0;
        m2.Q20() = 1.0;

        BENCHMARK("S-function: quad-quad") {
            return sf_engine.compute_interaction_energy(m1, pos1, m2, pos2);
        };

        BENCHMARK("Cartesian:  quad-quad") {
            return cart_engine.compute_interaction_energy(m1, pos1, m2, pos2);
        };
    }

    SECTION("Hexadecapole-hexadecapole") {
        Mult m1(4), m2(4);
        m1.Q40() = 1.0;
        m2.Q40() = 1.0;

        BENCHMARK("S-function: hex-hex") {
            return sf_engine.compute_interaction_energy(m1, pos1, m2, pos2);
        };

        BENCHMARK("Cartesian:  hex-hex") {
            return cart_engine.compute_interaction_energy(m1, pos1, m2, pos2);
        };
    }

    SECTION("Dipole-dipole 14 directions") {
        Mult m1(4), m2(4);
        m1.Q10() = 1.0;
        m2.Q10() = 1.0;

        BENCHMARK("S-function: dipole-dipole 14 dirs") {
            double sum = 0.0;
            for (auto &d : directions) {
                sum += sf_engine.compute_interaction_energy(m1, pos1, m2, d);
            }
            return sum;
        };

        BENCHMARK("Cartesian:  dipole-dipole 14 dirs") {
            double sum = 0.0;
            for (auto &d : directions) {
                sum += cart_engine.compute_interaction_energy(m1, pos1, m2, d);
            }
            return sum;
        };
    }

    SECTION("All 325 unique pairs, one direction") {
        BENCHMARK("S-function: all 325 pairs") {
            double sum = 0.0;
            for (int i = 0; i < 25; ++i) {
                for (int j = i; j < 25; ++j) {
                    Mult m1(4), m2(4);
                    m1.get_component(ALL_COMPONENTS[i]) = 1.0;
                    m2.get_component(ALL_COMPONENTS[j]) = 1.0;
                    sum += sf_engine.compute_interaction_energy(m1, pos1, m2, pos2);
                }
            }
            return sum;
        };

        BENCHMARK("Cartesian:  all 325 pairs") {
            double sum = 0.0;
            for (int i = 0; i < 25; ++i) {
                for (int j = i; j < 25; ++j) {
                    Mult m1(4), m2(4);
                    m1.get_component(ALL_COMPONENTS[i]) = 1.0;
                    m2.get_component(ALL_COMPONENTS[j]) = 1.0;
                    sum += cart_engine.compute_interaction_energy(m1, pos1, m2, pos2);
                }
            }
            return sum;
        };
    }

    SECTION("Molecule batch: water dimer (3x3 sites)") {
        // Build two 3-site "water" molecules with realistic-ish multipoles
        Mult qO(4), qH(4);
        qO.Q00() = -0.669;
        qO.Q10() = 0.234;
        qO.Q20() = -0.123;
        qH.Q00() = 0.335;

        Vec3 pO1(0, 0, 0), pH1a(1.8, 0, 0), pH1b(-0.6, 1.7, 0);
        Vec3 pO2(5, 0, 0), pH2a(6.8, 0, 0), pH2b(4.4, 1.7, 0);

        // Pre-build CartesianMolecule objects (outside benchmark loop)
        auto molA = CartesianMolecule::from_lab_sites({
            {qO, pO1}, {qH, pH1a}, {qH, pH1b}
        });
        auto molB = CartesianMolecule::from_lab_sites({
            {qO, pO2}, {qH, pH2a}, {qH, pH2b}
        });

        // Build site data for per-pair reference
        std::vector<std::pair<Mult, Vec3>> sitesA = {{qO, pO1}, {qH, pH1a}, {qH, pH1b}};
        std::vector<std::pair<Mult, Vec3>> sitesB = {{qO, pO2}, {qH, pH2a}, {qH, pH2b}};

        BENCHMARK("Cartesian per-pair: water dimer 3x3") {
            double sum = 0.0;
            for (const auto &[mA, pA] : sitesA) {
                for (const auto &[mB, pB] : sitesB) {
                    sum += cart_engine.compute_interaction_energy(mA, pA, mB, pB);
                }
            }
            return sum;
        };

        BENCHMARK("Cartesian batch:    water dimer 3x3") {
            return compute_molecule_interaction(molA, molB);
        };

        BENCHMARK("Cartesian SIMD:     water dimer 3x3") {
            return compute_molecule_interaction_simd(molA, molB);
        };

        BENCHMARK("S-function:         water dimer 3x3") {
            double sum = 0.0;
            for (const auto &[mA, pA] : sitesA) {
                for (const auto &[mB, pB] : sitesB) {
                    sum += sf_engine.compute_interaction_energy(mA, pA, mB, pB);
                }
            }
            return sum;
        };
    }
}

// ===================================================================
// Cross-engine validation: S-function vs Cartesian with rotated molecules
// ===================================================================

TEST_CASE("Cross-engine: rotated single-site energy",
          "[mults][crossval]") {
    using Approx = Catch::Approx;

    // NOTE: S-function engine silently drops terms with rank_sum > 5.
    // Use multipoles with max rank_sum ≤ 5 for cross-engine validation.
    // The Cartesian engine handles all rank sums up to 2*MaxL = 8.

    MultipoleInteractions::Config sf_config;
    sf_config.max_rank = 4;
    MultipoleInteractions sf_engine(sf_config);

    // Multipoles with rich components but rank ≤ 2 (max rank_sum = 4)
    Mult m1(4), m2(4);
    m1.Q00() = -0.669;
    m1.Q10() = 0.234;
    m1.Q11c() = -0.15;
    m1.Q20() = -0.123;
    m1.Q22c() = 0.08;

    m2.Q00() = 0.5;
    m2.Q10() = -0.3;
    m2.Q11s() = 0.2;
    m2.Q20() = 0.15;
    m2.Q21c() = -0.06;
    m2.Q22c() = 0.04;

    Vec3 pos1(0, 0, 0);
    Vec3 pos2(5.0, 2.0, -1.5);

    // Lab-frame energy (no rotation) — validates energy kernel agreement
    {
        double E_sf_lab = sf_engine.compute_interaction_energy(m1, pos1, m2, pos2);
        auto molA_lab = CartesianMolecule::from_lab_sites({{m1, pos1}});
        auto molB_lab = CartesianMolecule::from_lab_sites({{m2, pos2}});
        double E_cart_lab = compute_molecule_interaction(molA_lab, molB_lab);
        fmt::print("  Lab-frame energy (no rotation):\n");
        fmt::print("    S-func: {:.12e}  Cart: {:.12e}  Diff: {:.2e}\n",
                   E_sf_lab, E_cart_lab, std::abs(E_sf_lab - E_cart_lab));
        REQUIRE(E_cart_lab == Approx(E_sf_lab).epsilon(1e-10));
    }

    // Rotated energy — validates Cartesian rotation matches Wigner D rotation
    Vec3 euler1(0.7, 0.4, -0.3);
    Vec3 euler2(-0.5, 0.8, 1.2);
    Mat3 R1 = rotation_utils::euler_to_rotation(euler1[0], euler1[1], euler1[2]);
    Mat3 R2 = rotation_utils::euler_to_rotation(euler2[0], euler2[1], euler2[2]);

    Mult m1_lab = rotated_multipole(m1, R1);
    Mult m2_lab = rotated_multipole(m2, R2);
    double E_sf = sf_engine.compute_interaction_energy(m1_lab, pos1, m2_lab, pos2);

    auto molA = CartesianMolecule::from_body_frame_with_rotation({{m1, Vec3::Zero()}}, R1, pos1);
    auto molB = CartesianMolecule::from_body_frame_with_rotation({{m2, Vec3::Zero()}}, R2, pos2);
    double E_cart = compute_molecule_interaction(molA, molB);

    // Cartesian energy with Wigner-D-rotated lab multipoles (bypass Cartesian rotation)
    auto molA_sflab = CartesianMolecule::from_lab_sites({{m1_lab, pos1}});
    auto molB_sflab = CartesianMolecule::from_lab_sites({{m2_lab, pos2}});
    double E_cart_sflab = compute_molecule_interaction(molA_sflab, molB_sflab);

    fmt::print("  Rotated energy:\n");
    fmt::print("    S-func (Wigner D):       {:.12e}\n", E_sf);
    fmt::print("    Cart (Cartesian rot):    {:.12e}\n", E_cart);
    fmt::print("    Cart (Wigner D lab):     {:.12e}\n", E_cart_sflab);
    fmt::print("    Diff SF vs Cart(cart rot): {:.2e}\n", std::abs(E_sf - E_cart));
    fmt::print("    Diff SF vs Cart(WignerD):  {:.2e}\n", std::abs(E_sf - E_cart_sflab));

    // Energy kernel with Wigner D lab multipoles must match
    REQUIRE(E_cart_sflab == Approx(E_sf).epsilon(1e-10));
    // Cartesian rotation must give same result as Wigner D
    REQUIRE(E_cart == Approx(E_sf).epsilon(1e-10));
}

TEST_CASE("Cross-engine: rotated single-site force + torque",
          "[mults][crossval]") {
    using Approx = Catch::Approx;

    // Rank ≤ 2 multipoles → rank_sum ≤ 4 (within S-function capability)
    Mult m1(4), m2(4);
    m1.Q00() = -0.669;
    m1.Q10() = 0.234;
    m1.Q11c() = -0.15;
    m1.Q20() = -0.123;
    m1.Q22c() = 0.08;

    m2.Q00() = 0.5;
    m2.Q10() = -0.3;
    m2.Q11s() = 0.2;
    m2.Q20() = 0.15;

    Vec3 euler1(0.7, 0.4, -0.3);
    Vec3 euler2(-0.5, 0.8, 1.2);
    Vec3 pos1(0, 0, 0);
    Vec3 pos2(5.0, 2.0, -1.5);

    Mat3 R1 = rotation_utils::euler_to_rotation(euler1[0], euler1[1], euler1[2]);
    Mat3 R2 = rotation_utils::euler_to_rotation(euler2[0], euler2[1], euler2[2]);

    // Cartesian: analytical force + torque
    auto molA = CartesianMolecule::from_body_frame_with_rotation(
        {{m1, Vec3::Zero()}}, R1, pos1);
    auto molB = CartesianMolecule::from_body_frame_with_rotation(
        {{m2, Vec3::Zero()}}, R2, pos2);
    auto cart_result = compute_molecule_forces_torques(molA, molB);

    // --- Validate Cartesian force via FD ---
    double delta = 1e-7;
    Vec3 cart_fd_force_A;
    {
        for (int d = 0; d < 3; ++d) {
            auto molA_p = molA;
            auto molA_m = molA;
            molA_p.sites[0].position[d] += delta;
            molA_m.sites[0].position[d] -= delta;
            double Ep = compute_molecule_interaction(molA_p, molB);
            double Em = compute_molecule_interaction(molA_m, molB);
            cart_fd_force_A[d] = -(Ep - Em) / (2 * delta);
        }
    }

    fmt::print("\n  === Cartesian force A: analytical vs FD ===\n");
    for (int d = 0; d < 3; ++d) {
        fmt::print("    d={}: analytical={:.10e}  FD={:.10e}  diff={:.2e}\n",
                   d, cart_result.force_A[d], cart_fd_force_A[d],
                   cart_result.force_A[d] - cart_fd_force_A[d]);
        REQUIRE(cart_result.force_A[d] == Approx(cart_fd_force_A[d]).epsilon(1e-6));
    }

    // --- Validate Cartesian angle-axis gradient A via FD ---
    Vec3 cart_fd_grad_aa_A;
    {
        for (int k = 0; k < 3; ++k) {
            // Perturb angle-axis parameter p_k
            Vec3 dp = Vec3::Zero();
            dp[k] = delta;

            // Forward: apply infinitesimal rotation about axis k
            // R_new = exp(dp) * R ≈ (I + [dp]_x) * R
            Mat3 skew;
            skew << 0, -dp[2], dp[1],  dp[2], 0, -dp[0],  -dp[1], dp[0], 0;
            Mat3 R1_p = (Mat3::Identity() + skew) * R1;
            auto molA_p = CartesianMolecule::from_body_frame_with_rotation(
                {{m1, Vec3::Zero()}}, R1_p, pos1);

            Mat3 R1_m = (Mat3::Identity() - skew) * R1;
            auto molA_m = CartesianMolecule::from_body_frame_with_rotation(
                {{m1, Vec3::Zero()}}, R1_m, pos1);

            double Ep = compute_molecule_interaction(molA_p, molB);
            double Em = compute_molecule_interaction(molA_m, molB);
            cart_fd_grad_aa_A[k] = (Ep - Em) / (2 * delta);
        }
    }

    fmt::print("  === Cartesian grad_angle_axis A: analytical vs FD ===\n");
    for (int k = 0; k < 3; ++k) {
        fmt::print("    k={}: analytical={:.10e}  FD={:.10e}  diff={:.2e}\n",
                   k, cart_result.grad_angle_axis_A[k], cart_fd_grad_aa_A[k],
                   cart_result.grad_angle_axis_A[k] - cart_fd_grad_aa_A[k]);
        REQUIRE(cart_result.grad_angle_axis_A[k] ==
                Approx(cart_fd_grad_aa_A[k]).epsilon(1e-5));
    }

    // --- Validate Cartesian angle-axis gradient B via FD ---
    Vec3 cart_fd_grad_aa_B;
    {
        for (int k = 0; k < 3; ++k) {
            Vec3 dp = Vec3::Zero();
            dp[k] = delta;
            Mat3 skew;
            skew << 0, -dp[2], dp[1],  dp[2], 0, -dp[0],  -dp[1], dp[0], 0;

            Mat3 R2_p = (Mat3::Identity() + skew) * R2;
            auto molB_p = CartesianMolecule::from_body_frame_with_rotation(
                {{m2, Vec3::Zero()}}, R2_p, pos2);
            Mat3 R2_m = (Mat3::Identity() - skew) * R2;
            auto molB_m = CartesianMolecule::from_body_frame_with_rotation(
                {{m2, Vec3::Zero()}}, R2_m, pos2);

            double Ep = compute_molecule_interaction(molA, molB_p);
            double Em = compute_molecule_interaction(molA, molB_m);
            cart_fd_grad_aa_B[k] = (Ep - Em) / (2 * delta);
        }
    }

    fmt::print("  === Cartesian grad_angle_axis B: analytical vs FD ===\n");
    for (int k = 0; k < 3; ++k) {
        fmt::print("    k={}: analytical={:.10e}  FD={:.10e}  diff={:.2e}\n",
                   k, cart_result.grad_angle_axis_B[k], cart_fd_grad_aa_B[k],
                   cart_result.grad_angle_axis_B[k] - cart_fd_grad_aa_B[k]);
        REQUIRE(cart_result.grad_angle_axis_B[k] ==
                Approx(cart_fd_grad_aa_B[k]).epsilon(1e-5));
    }

    // --- Cross-engine comparison (informational) ---
    // NOTE: S-function analytical force/torque has known ~5e-6 accuracy issue
    // compared to its own FD. The Cartesian engine matches its FD to ~1e-10.
    auto sf_result1 = TorqueCalculation::compute_torque_analytical(
        m1, pos1, euler1, m2, pos2, euler2, 1);
    auto sf_result2 = TorqueCalculation::compute_torque_analytical(
        m1, pos1, euler1, m2, pos2, euler2, 2);

    fmt::print("\n  === Cross-engine comparison (informational) ===\n");
    fmt::print("  Force A: Cart=({:.6e},{:.6e},{:.6e}) SF=({:.6e},{:.6e},{:.6e})\n",
               cart_result.force_A[0], cart_result.force_A[1], cart_result.force_A[2],
               sf_result1.force[0], sf_result1.force[1], sf_result1.force[2]);
    fmt::print("  Force B: Cart=({:.6e},{:.6e},{:.6e}) SF=({:.6e},{:.6e},{:.6e})\n",
               cart_result.force_B[0], cart_result.force_B[1], cart_result.force_B[2],
               sf_result2.force[0], sf_result2.force[1], sf_result2.force[2]);
    fmt::print("  Grad AA A: Cart=({:.6e},{:.6e},{:.6e}) SF=({:.6e},{:.6e},{:.6e})\n",
               cart_result.grad_angle_axis_A[0], cart_result.grad_angle_axis_A[1],
               cart_result.grad_angle_axis_A[2],
               sf_result1.grad_angle_axis[0], sf_result1.grad_angle_axis[1],
               sf_result1.grad_angle_axis[2]);
    fmt::print("  Grad AA B: Cart=({:.6e},{:.6e},{:.6e}) SF=({:.6e},{:.6e},{:.6e})\n",
               cart_result.grad_angle_axis_B[0], cart_result.grad_angle_axis_B[1],
               cart_result.grad_angle_axis_B[2],
               sf_result2.grad_angle_axis[0], sf_result2.grad_angle_axis[1],
               sf_result2.grad_angle_axis[2]);
}

TEST_CASE("Cross-engine: rotated water dimer energy",
          "[mults][crossval]") {
    using Approx = Catch::Approx;

    MultipoleInteractions::Config sf_config;
    sf_config.max_rank = 4;
    MultipoleInteractions sf_engine(sf_config);

    // Water-like multipoles
    Mult qO(4), qH(4);
    qO.Q00() = -0.669;
    qO.Q10() = 0.234;
    qO.Q20() = -0.123;
    qO.Q22c() = 0.08;
    qH.Q00() = 0.335;
    qH.Q10() = 0.05;

    // Body-frame site offsets
    Vec3 bodyO(0, 0, 0), bodyH1(1.8, 0, 0), bodyH2(-0.6, 1.7, 0);

    // Non-trivial orientations
    Vec3 euler_A(0.3, 0.7, -0.5);
    Vec3 euler_B(-0.8, 0.2, 1.0);
    Mat3 R_A = rotation_utils::euler_to_rotation(euler_A[0], euler_A[1], euler_A[2]);
    Mat3 R_B = rotation_utils::euler_to_rotation(euler_B[0], euler_B[1], euler_B[2]);

    Vec3 center_A(0, 0, 0);
    Vec3 center_B(6, 2, -1);

    // --- S-function energy: rotate multipoles to lab, sum over site pairs ---
    Mult qO_labA = rotated_multipole(qO, R_A);
    Mult qH_labA = rotated_multipole(qH, R_A);
    Mult qO_labB = rotated_multipole(qO, R_B);
    Mult qH_labB = rotated_multipole(qH, R_B);

    // Lab-frame positions
    Vec3 pO_A = center_A + R_A * bodyO;
    Vec3 pH1_A = center_A + R_A * bodyH1;
    Vec3 pH2_A = center_A + R_A * bodyH2;
    Vec3 pO_B = center_B + R_B * bodyO;
    Vec3 pH1_B = center_B + R_B * bodyH1;
    Vec3 pH2_B = center_B + R_B * bodyH2;

    std::vector<std::pair<Mult, Vec3>> labA = {
        {qO_labA, pO_A}, {qH_labA, pH1_A}, {qH_labA, pH2_A}
    };
    std::vector<std::pair<Mult, Vec3>> labB = {
        {qO_labB, pO_B}, {qH_labB, pH1_B}, {qH_labB, pH2_B}
    };

    double E_sf = 0.0;
    for (const auto &[mA, pA] : labA) {
        for (const auto &[mB, pB] : labB) {
            E_sf += sf_engine.compute_interaction_energy(mA, pA, mB, pB);
        }
    }

    // --- Cartesian energy ---
    std::vector<std::pair<Mult, Vec3>> body_sites_A = {
        {qO, bodyO}, {qH, bodyH1}, {qH, bodyH2}
    };
    std::vector<std::pair<Mult, Vec3>> body_sites_B = {
        {qO, bodyO}, {qH, bodyH1}, {qH, bodyH2}
    };

    auto molA = CartesianMolecule::from_body_frame_with_rotation(body_sites_A, R_A, center_A);
    auto molB = CartesianMolecule::from_body_frame_with_rotation(body_sites_B, R_B, center_B);
    double E_cart = compute_molecule_interaction(molA, molB);

    // --- Cartesian force + torque ---
    auto cart_ft = compute_molecule_forces_torques(molA, molB);

    fmt::print("\n  Rotated water dimer energy:\n");
    fmt::print("    S-function: {:.12e}\n", E_sf);
    fmt::print("    Cartesian:  {:.12e}\n", E_cart);
    fmt::print("    Cart F+T:   {:.12e}\n", cart_ft.energy);
    fmt::print("    Diff: {:.2e}\n", std::abs(E_sf - E_cart));

    REQUIRE(E_cart == Approx(E_sf).epsilon(1e-10));
    REQUIRE(cart_ft.energy == Approx(E_sf).epsilon(1e-10));
}

// ===================================================================
// Performance: S-function vs Cartesian for rotated molecule pairs
// ===================================================================

TEST_CASE("Rotated molecule benchmarks", "[mults][benchmark][rotated]") {

    CartesianInteractions cart_engine;
    MultipoleInteractions::Config sf_config;
    sf_config.max_rank = 4;
    MultipoleInteractions sf_engine(sf_config);

    // Rich multipoles
    Mult m1(4), m2(4);
    m1.Q00() = -0.669; m1.Q10() = 0.234; m1.Q11c() = -0.15;
    m1.Q20() = -0.123; m1.Q22c() = 0.08; m1.Q30() = 0.05; m1.Q40() = 0.01;
    m2.Q00() = 0.5; m2.Q10() = -0.3; m2.Q11s() = 0.2;
    m2.Q20() = 0.15; m2.Q21c() = -0.06;

    Vec3 euler1(0.7, 0.4, -0.3);
    Vec3 euler2(-0.5, 0.8, 1.2);
    Mat3 R1 = rotation_utils::euler_to_rotation(euler1[0], euler1[1], euler1[2]);
    Mat3 R2 = rotation_utils::euler_to_rotation(euler2[0], euler2[1], euler2[2]);
    Vec3 pos1(0, 0, 0), pos2(5.0, 2.0, -1.5);

    // Pre-build molecules for Cartesian batch API
    auto molA = CartesianMolecule::from_body_frame_with_rotation(
        {{m1, Vec3::Zero()}}, R1, pos1);
    auto molB = CartesianMolecule::from_body_frame_with_rotation(
        {{m2, Vec3::Zero()}}, R2, pos2);

    // Pre-rotate for S-function lab-frame energy
    Mult m1_lab = rotated_multipole(m1, R1);
    Mult m2_lab = rotated_multipole(m2, R2);

    SECTION("Single-site energy") {
        BENCHMARK("S-function: energy (lab-frame)") {
            return sf_engine.compute_interaction_energy(m1_lab, pos1, m2_lab, pos2);
        };

        BENCHMARK("Cartesian:  energy (body-frame)") {
            return compute_molecule_interaction(molA, molB);
        };
    }

    SECTION("Single-site force + torque") {
        BENCHMARK("S-function: analytical force+torque (mol 1)") {
            return TorqueCalculation::compute_torque_analytical(
                m1, pos1, euler1, m2, pos2, euler2, 1);
        };

        BENCHMARK("Cartesian:  force+torque (both mols)") {
            return compute_molecule_forces_torques(molA, molB);
        };
    }

    SECTION("Water dimer (3x3 sites) rotated") {
        Mult qO(4), qH(4);
        qO.Q00() = -0.669; qO.Q10() = 0.234; qO.Q20() = -0.123; qO.Q22c() = 0.08;
        qH.Q00() = 0.335; qH.Q10() = 0.05;

        Vec3 bodyO(0, 0, 0), bodyH1(1.8, 0, 0), bodyH2(-0.6, 1.7, 0);

        // Cartesian: pre-built rotated molecules
        auto wA = CartesianMolecule::from_body_frame_with_rotation(
            {{qO, bodyO}, {qH, bodyH1}, {qH, bodyH2}}, R1, pos1);
        auto wB = CartesianMolecule::from_body_frame_with_rotation(
            {{qO, bodyO}, {qH, bodyH1}, {qH, bodyH2}}, R2, pos2);

        // S-function: pre-rotated lab multipoles and positions
        Mult qO_labA = rotated_multipole(qO, R1);
        Mult qH_labA = rotated_multipole(qH, R1);
        Mult qO_labB = rotated_multipole(qO, R2);
        Mult qH_labB = rotated_multipole(qH, R2);

        std::vector<std::pair<Mult, Vec3>> labA = {
            {qO_labA, pos1 + R1 * bodyO},
            {qH_labA, pos1 + R1 * bodyH1},
            {qH_labA, pos1 + R1 * bodyH2}
        };
        std::vector<std::pair<Mult, Vec3>> labB = {
            {qO_labB, pos2 + R2 * bodyO},
            {qH_labB, pos2 + R2 * bodyH1},
            {qH_labB, pos2 + R2 * bodyH2}
        };

        BENCHMARK("S-function:     water dimer 3x3 rotated") {
            double sum = 0.0;
            for (const auto &[mA, pA] : labA)
                for (const auto &[mB, pB] : labB)
                    sum += sf_engine.compute_interaction_energy(mA, pA, mB, pB);
            return sum;
        };

        BENCHMARK("Cartesian:      water dimer 3x3 energy") {
            return compute_molecule_interaction(wA, wB);
        };

        BENCHMARK("Cartesian SIMD: water dimer 3x3 energy") {
            return compute_molecule_interaction_simd(wA, wB);
        };

        BENCHMARK("Cartesian:      water dimer 3x3 force+torque") {
            return compute_molecule_forces_torques(wA, wB);
        };
    }
}
