#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <occ/mults/coordinate_system.h>
#include <cmath>

using namespace occ::mults;
using occ::Vec3;
using Catch::Approx;

TEST_CASE("CoordinateSystem - basic functionality", "[mults][coordinates]") {
    
    SECTION("Simple axis-aligned case") {
        Vec3 ra(0.0, 0.0, 0.0);
        Vec3 rb(1.0, 0.0, 0.0);  // Unit vector along x-axis
        auto cs = CoordinateSystem::from_points(ra, rb);
        
        // Basic calculations
        REQUIRE(cs.r == Approx(1.0));
        REQUIRE(cs.dx() == Approx(1.0));
        REQUIRE(cs.dy() == Approx(0.0));
        REQUIRE(cs.dz() == Approx(0.0));
        
        // Unit vector should be (1, 0, 0)
        REQUIRE(cs.er[0] == Approx(1.0));
        REQUIRE(cs.er[1] == Approx(0.0));
        REQUIRE(cs.er[2] == Approx(0.0));
        
        // OCC convention: rax = +er, rbx = -er
        REQUIRE(cs.rax() == Approx(1.0));
        REQUIRE(cs.ray() == Approx(0.0));
        REQUIRE(cs.raz() == Approx(0.0));
        REQUIRE(cs.rbx() == Approx(-1.0));
        REQUIRE(cs.rby() == Approx(0.0));
        REQUIRE(cs.rbz() == Approx(0.0));
        
        // Validation
        REQUIRE(cs.is_valid());
    }
    
    SECTION("Diagonal case - matches Orient test coordinates") {
        // This matches our Orient validation case
        Vec3 ra(-2.0, -2.0, -2.0);
        Vec3 rb(2.0, 2.0, 2.0);
        auto cs = CoordinateSystem::from_points(ra, rb);
        
        // Distance should be sqrt((4)² + (4)² + (4)²) = sqrt(48) = 4*sqrt(3)
        double expected_r = 4.0 * std::sqrt(3.0);
        REQUIRE(cs.r == Approx(expected_r));
        
        // Unit vector components should all be 1/sqrt(3) ≈ 0.57735...
        double expected_unit = 1.0 / std::sqrt(3.0);
        REQUIRE(cs.er[0] == Approx(expected_unit));
        REQUIRE(cs.er[1] == Approx(expected_unit));
        REQUIRE(cs.er[2] == Approx(expected_unit));
        
        // OCC coordinates: rax = +er, rbx = -er
        REQUIRE(cs.rax() == Approx(expected_unit));
        REQUIRE(cs.ray() == Approx(expected_unit));
        REQUIRE(cs.raz() == Approx(expected_unit));
        REQUIRE(cs.rbx() == Approx(-expected_unit));
        REQUIRE(cs.rby() == Approx(-expected_unit));
        REQUIRE(cs.rbz() == Approx(-expected_unit));
        
        REQUIRE(cs.is_valid());
    }
    
    SECTION("2D case in xy-plane") {
        Vec3 ra(-1.0, -1.0, 0.0);
        Vec3 rb(1.0, 1.0, 0.0);
        auto cs = CoordinateSystem::from_points(ra, rb);
        
        // Distance should be sqrt(4 + 4) = 2*sqrt(2)
        double expected_r = 2.0 * std::sqrt(2.0);
        REQUIRE(cs.r == Approx(expected_r));
        
        // Unit vector should be (1/√2, 1/√2, 0)
        double expected_unit = 1.0 / std::sqrt(2.0);
        REQUIRE(cs.er[0] == Approx(expected_unit));
        REQUIRE(cs.er[1] == Approx(expected_unit));
        REQUIRE(cs.er[2] == Approx(0.0));
        
        // OCC coordinates: rax = +er, rbx = -er
        REQUIRE(cs.rax() == Approx(expected_unit));
        REQUIRE(cs.ray() == Approx(expected_unit));
        REQUIRE(cs.raz() == Approx(0.0));
        REQUIRE(cs.rbx() == Approx(-expected_unit));
        REQUIRE(cs.rby() == Approx(-expected_unit));
        REQUIRE(cs.rbz() == Approx(0.0));
        
        REQUIRE(cs.is_valid());
    }
}

TEST_CASE("CoordinateSystem - Orient validation", "[mults][coordinates]") {
    
    SECTION("Match Orient debug output - case 1") {
        // Orient debug: rax=-0.577, rbx=+0.577 (Orient convention: rax=-er, rbx=+er)
        // OCC convention: rax=+er, rbx=-er (opposite signs, same math via kernel swap)
        Vec3 ra(0.0, 0.0, 0.0);  // molecule position
        Vec3 rb(2.0, 2.0, 2.0);  // grid point
        auto cs = CoordinateSystem::from_points(ra, rb);

        bool matches = cs.matches_orient_debug(
             0.57735027,  0.57735027,  0.57735027,  // rax, ray, raz (OCC: +er)
            -0.57735027, -0.57735027, -0.57735027,  // rbx, rby, rbz (OCC: -er)
             3.46410162                             // r
        );
        REQUIRE(matches);
    }

    SECTION("Match Orient debug output - case 2") {
        // Orient debug: rax=-0.707, rbx=+0.707 (Orient convention: rax=-er, rbx=+er)
        // OCC convention: rax=+er, rbx=-er (opposite signs, same math via kernel swap)
        Vec3 ra(-1.0, -1.0, 0.0);
        Vec3 rb(1.0, 1.0, 0.0);
        auto cs = CoordinateSystem::from_points(ra, rb);

        bool matches = cs.matches_orient_debug(
             0.70710678,  0.70710678,  0.00000000,  // rax, ray, raz (OCC: +er)
            -0.70710678, -0.70710678,  0.00000000,  // rbx, rby, rbz (OCC: -er)
             2.82842712                             // r
        );
        REQUIRE(matches);
    }
}

TEST_CASE("CoordinateSystem - edge cases", "[mults][coordinates]") {
    
    SECTION("Same points (degenerate case)") {
        Vec3 ra(1.0, 2.0, 3.0);
        Vec3 rb(1.0, 2.0, 3.0);
        auto cs = CoordinateSystem::from_points(ra, rb);
        
        REQUIRE(cs.r == Approx(0.0));
        REQUIRE(cs.dx() == Approx(0.0));
        REQUIRE(cs.dy() == Approx(0.0)); 
        REQUIRE(cs.dz() == Approx(0.0));
        
        // Unit vector should be zero for degenerate case
        REQUIRE(cs.er[0] == Approx(0.0));
        REQUIRE(cs.er[1] == Approx(0.0));
        REQUIRE(cs.er[2] == Approx(0.0));
        
        // Orient coordinates should also be zero
        REQUIRE(cs.rax() == Approx(0.0));
        REQUIRE(cs.ray() == Approx(0.0));
        REQUIRE(cs.raz() == Approx(0.0));
        REQUIRE(cs.rbx() == Approx(0.0));
        REQUIRE(cs.rby() == Approx(0.0));
        REQUIRE(cs.rbz() == Approx(0.0));
    }
    
    SECTION("Very small distance") {
        Vec3 ra(0.0, 0.0, 0.0);
        Vec3 rb(1e-15, 0.0, 0.0);
        auto cs = CoordinateSystem::from_points(ra, rb);
        
        REQUIRE(cs.r == Approx(1e-15));
        // For very small distances, unit vector calculation may be unstable
        // but should still be valid mathematically
        if (cs.r > 1e-12) {
            REQUIRE(cs.is_valid());
        }
    }
}

TEST_CASE("CoordinateSystem - mathematical properties", "[mults][coordinates]") {
    
    SECTION("Unit vector normalization") {
        Vec3 ra(1.5, -2.3, 4.7);
        Vec3 rb(-3.1, 5.2, -1.8);
        auto cs = CoordinateSystem::from_points(ra, rb);
        
        // Unit vector should have length 1
        REQUIRE(cs.er.norm() == Approx(1.0));
        
        // raxyz and rbxyz should be opposite unit vectors
        REQUIRE((cs.raxyz() + cs.rbxyz()).norm() == Approx(0.0).margin(1e-12));
        
        // Both should have unit length
        REQUIRE(cs.raxyz().norm() == Approx(1.0));
        REQUIRE(cs.rbxyz().norm() == Approx(1.0));
        
        REQUIRE(cs.is_valid());
    }
    
    SECTION("Coordinate consistency") {
        Vec3 ra(0.1, 0.2, 0.3);
        Vec3 rb(0.4, 0.5, 0.6);
        auto cs = CoordinateSystem::from_points(ra, rb);
        
        // Raw coordinates should match input
        REQUIRE(cs.raw_rax() == Approx(0.1));
        REQUIRE(cs.raw_ray() == Approx(0.2));
        REQUIRE(cs.raw_raz() == Approx(0.3));
        REQUIRE(cs.raw_rbx() == Approx(0.4));
        REQUIRE(cs.raw_rby() == Approx(0.5));
        REQUIRE(cs.raw_rbz() == Approx(0.6));
        
        // Distance components
        REQUIRE(cs.dx() == Approx(0.3));
        REQUIRE(cs.dy() == Approx(0.3));
        REQUIRE(cs.dz() == Approx(0.3));
        
        // Overall consistency
        REQUIRE(cs.is_valid());
    }
}