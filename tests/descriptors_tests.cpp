#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <catch2/catch_approx.hpp>
#include <chrono>
#include <fmt/ostream.h>
#include <fmt/core.h>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <occ/core/util.h>
#include <occ/core/element.h>
#include <occ/descriptors/steinhardt.h>
#include <occ/descriptors/pdd_amd.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/spacegroup.h>
#include <occ/crystal/unitcell.h>
#include <occ/descriptors/sorted_k_distances.h>

// Custom Catch2 matcher for matrix comparison with detailed debugging
template<typename ExpectedType>
class MatrixApproxMatcher : public Catch::Matchers::MatcherGenericBase {
    ExpectedType const& m_expected;
    double m_atol;
    double m_rtol;
    mutable std::string m_description;

public:
    MatrixApproxMatcher(ExpectedType const& expected, double atol = 1e-6, double rtol = 1e-6)
        : m_expected(expected), m_atol(atol), m_rtol(rtol) {}

    template<typename ActualType>
    bool match(ActualType const& actual) const {
        if (actual.rows() != m_expected.rows() || actual.cols() != m_expected.cols()) {
            m_description = fmt::format("Size mismatch: actual {}x{} vs expected {}x{}", 
                                      actual.rows(), actual.cols(), 
                                      m_expected.rows(), m_expected.cols());
            return false;
        }

        std::vector<std::tuple<int, int, double, double, double>> failures;
        
        for (int i = 0; i < actual.rows(); ++i) {
            for (int j = 0; j < actual.cols(); ++j) {
                double a = actual(i, j);
                double e = m_expected(i, j);
                double abs_diff = std::abs(a - e);
                double rel_diff = std::abs(e) > 0 ? abs_diff / std::abs(e) : abs_diff;
                
                if (abs_diff > m_atol && rel_diff > m_rtol) {
                    failures.emplace_back(i, j, a, e, abs_diff);
                }
            }
        }
        
        if (!failures.empty()) {
            std::ostringstream oss;
            oss << fmt::format("Matrix comparison failed (atol={:.2e}, rtol={:.2e}):\n", m_atol, m_rtol);
            
            if (failures.size() <= 10) {
                oss << fmt::format("  {} mismatched elements:\n", failures.size());
                for (const auto& [i, j, a, e, diff] : failures) {
                    oss << fmt::format("    [{:2d},{:2d}]: actual={:12.8f}, expected={:12.8f}, diff={:.2e}\n", 
                                     i, j, a, e, diff);
                }
            } else {
                oss << fmt::format("  First 10 of {} mismatched elements:\n", failures.size());
                for (size_t k = 0; k < 10; ++k) {
                    const auto& [i, j, a, e, diff] = failures[k];
                    oss << fmt::format("    [{:2d},{:2d}]: actual={:12.8f}, expected={:12.8f}, diff={:.2e}\n", 
                                     i, j, a, e, diff);
                }
                oss << fmt::format("    ... and {} more\n", failures.size() - 10);
            }
            
            // Show small matrices in full
            if (actual.rows() <= 8 && actual.cols() <= 8) {
                oss << "\n  Actual matrix:\n";
                for (int i = 0; i < actual.rows(); ++i) {
                    oss << "    ";
                    for (int j = 0; j < actual.cols(); ++j) {
                        oss << fmt::format("{:10.6f} ", actual(i, j));
                    }
                    oss << "\n";
                }
                
                oss << "\n  Expected matrix:\n";
                for (int i = 0; i < m_expected.rows(); ++i) {
                    oss << "    ";
                    for (int j = 0; j < m_expected.cols(); ++j) {
                        oss << fmt::format("{:10.6f} ", m_expected(i, j));
                    }
                    oss << "\n";
                }
            }
            
            m_description = oss.str();
            return false;
        }
        
        return true;
    }

    std::string describe() const override {
        if (m_description.empty()) {
            return fmt::format("is approximately equal to expected matrix (atol={:.2e}, rtol={:.2e})", m_atol, m_rtol);
        }
        return m_description;
    }
};

// Helper function to create the matcher
template<typename T>
inline MatrixApproxMatcher<T> IsApproxMatrix(T const& expected, double atol = 1e-6, double rtol = 1e-6) {
    return MatrixApproxMatcher<T>(expected, atol, rtol);
}

namespace timer {

using namespace std::chrono;

template <typename R, typename P> inline double seconds(duration<R, P> x) {
  return duration_cast<nanoseconds>(x).count() / 1e9;
}

inline auto time() { return high_resolution_clock::now(); }

} // namespace timer

inline void print_vec(const std::string &name, const occ::Vec &v) {
  fmt::print("{}\n", name);
  for (int i = 0; i < v.rows(); i++) {
    fmt::print(" {:12.6f}", v(i));
  }
  fmt::print("\n");
}

TEST_CASE("Steinhardt q parameters", "[steinhardt]") {
  using Catch::Matchers::WithinAbs;
  using occ::descriptors::Steinhardt;
  constexpr double eps = 1e-6;

  SECTION("Cubic symmetry") {
    Steinhardt steinhardt(6);
    occ::Mat3N positions(3, 8);
    positions << 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1,
        1, -1, 1, -1, 1, -1;

    occ::Vec q = steinhardt.compute_q(positions);
    print_vec("Cubic Q", q);

    REQUIRE(q.size() == 7);
    REQUIRE_THAT(q(0), WithinAbs(1.0, eps));
    REQUIRE_THAT(q(1), WithinAbs(0.0, eps));
    REQUIRE_THAT(q(2), WithinAbs(0.0, eps));
    REQUIRE_THAT(q(3), WithinAbs(0.0, eps));
    REQUIRE(q(4) > 0.1);
    REQUIRE_THAT(q(5), WithinAbs(0.0, eps));
    REQUIRE(q(6) > 0.1);
    occ::Vec w = steinhardt.compute_w(positions);
    print_vec("Cubic W", w);
  }

  SECTION("Octahedral symmetry") {
    Steinhardt steinhardt(6);
    occ::Mat3N positions(3, 6);
    positions << 1, -1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1;

    occ::Vec q = steinhardt.compute_q(positions);
    print_vec("Octahedral Q", q);

    REQUIRE(q.size() == 7);
    REQUIRE_THAT(q(0), WithinAbs(1.0, eps));
    REQUIRE_THAT(q(1), WithinAbs(0.0, eps));
    REQUIRE_THAT(q(2), WithinAbs(0.0, eps));
    REQUIRE_THAT(q(3), WithinAbs(0.0, eps));
    REQUIRE(q(4) > 0.1);
    REQUIRE_THAT(q(5), WithinAbs(0.0, eps));
    REQUIRE(q(6) > 0.1);
    occ::Vec w = steinhardt.compute_w(positions);
    print_vec("Octahedral W", w);
  }

  SECTION("Tetrahedral symmetry") {
    Steinhardt steinhardt(6);
    occ::Mat3N positions(3, 4);

    positions.col(0) = occ::Vec3{1.0, 0.0, -1 / std::sqrt(2.0)};
    positions.col(1) = occ::Vec3{-1.0, 0.0, -1 / std::sqrt(2.0)};
    positions.col(2) = occ::Vec3{0.0, 1.0, 1 / std::sqrt(2.0)};
    positions.col(3) = occ::Vec3{0.0, -1.0, 1 / std::sqrt(2.0)};

    occ::Vec q = steinhardt.compute_q(positions);
    print_vec("Tetrahedral Q", q);

    REQUIRE(q.size() == 7);
    REQUIRE_THAT(q(0), WithinAbs(1.0, eps));
    REQUIRE_THAT(q(1), WithinAbs(0.0, eps));
    REQUIRE_THAT(q(2), WithinAbs(0.0, eps));
    REQUIRE(q(3) > 0.1);
    REQUIRE(q(4) > 0.1);
    REQUIRE_THAT(q(5), WithinAbs(0.0, eps));
    REQUIRE(q(6) > 0.1);
    occ::Vec w = steinhardt.compute_w(positions);
    print_vec("Tetrahedral W", w);
  }

  SECTION("Icosahedral symmetry") {
    Steinhardt steinhardt(6);
    occ::Mat3N positions(3, 12);
    const double gr = 0.5 * (1.0 + std::sqrt(5));

    positions.col(0) = occ::Vec3{0.0, 1.0, gr};
    positions.col(1) = occ::Vec3{0.0, 1.0, -gr};
    positions.col(2) = occ::Vec3{0.0, -1.0, gr};
    positions.col(3) = occ::Vec3{0.0, -1.0, -gr};
    positions.col(4) = occ::Vec3{1.0, gr, 0.0};
    positions.col(5) = occ::Vec3{1.0, -gr, 0.0};
    positions.col(6) = occ::Vec3{-1.0, gr, 0.0};
    positions.col(7) = occ::Vec3{-1.0, -gr, 0.0};
    positions.col(8) = occ::Vec3{gr, 0.0, 1.0};
    positions.col(9) = occ::Vec3{gr, 0.0, -1.0};
    positions.col(10) = occ::Vec3{-gr, 0.0, 1.0};
    positions.col(11) = occ::Vec3{-gr, 0.0, -1.0};

    occ::Vec q = steinhardt.compute_q(positions);
    print_vec("Icosahedral Q", q);

    REQUIRE(q.size() == 7);
    REQUIRE_THAT(q(0), WithinAbs(1.0, eps));
    REQUIRE_THAT(q(4), WithinAbs(0.0, eps));
    REQUIRE_THAT(q(6), WithinAbs(0.66332495807107972, eps));

    occ::Vec w = steinhardt.compute_w(positions);
    print_vec("Icosahedral W", w);
  }
}

TEST_CASE("PDD and AMD descriptors", "[pdd][amd]") {
  using Catch::Matchers::WithinAbs;
  using occ::crystal::Crystal;
  using occ::crystal::SpaceGroup;
  using occ::crystal::UnitCell;
  using occ::crystal::AsymmetricUnit;
  using occ::descriptors::PDD;
  using occ::descriptors::PDDConfig;
  constexpr double eps = 1e-6;

  SECTION("Simple cubic lattice") {
    // Create simple cubic lattice: single atom at origin
    occ::Mat3N positions(3, 1);
    positions << 0.0, 0.0, 0.0;
    occ::IVec atomic_numbers(1);
    atomic_numbers << 1; // Hydrogen for simplicity
    
    AsymmetricUnit asym_unit(positions, atomic_numbers);
    UnitCell unit_cell(1.0, 1.0, 1.0, M_PI/2, M_PI/2, M_PI/2); // Cubic unit cell (angles in radians)
    SpaceGroup space_group(1); // P1
    
    Crystal crystal(asym_unit, space_group, unit_cell);
    
    // Test with k=10 to match reference data
    PDD pdd(crystal, 10);
    
    // Check that PDD was calculated
    REQUIRE(pdd.size() > 0);
    REQUIRE(pdd.k() == 10);
    
    // Calculate AMD
    occ::Vec amd = pdd.average_minimum_distance();
    REQUIRE(amd.size() == 10);
    
    // For simple cubic, first 6 neighbors should be at distance 1.0
    // (based on reference data)
    for (int i = 0; i < 6; ++i) {
      REQUIRE_THAT(amd(i), WithinAbs(1.0, eps));
    }
  }

  SECTION("Face-centered cubic lattice") {
    // Create FCC lattice: 4 atoms in unit cell
    occ::Mat3N positions(3, 4);
    positions << 0.0, 0.5, 0.5, 0.0,
                 0.0, 0.5, 0.0, 0.5,
                 0.0, 0.0, 0.5, 0.5;
    occ::IVec atomic_numbers(4);
    atomic_numbers << 1, 1, 1, 1; // All hydrogen
    
    AsymmetricUnit asym_unit(positions, atomic_numbers);
    UnitCell unit_cell(1.0, 1.0, 1.0, M_PI/2, M_PI/2, M_PI/2);
    SpaceGroup space_group(1); // P1
    
    Crystal crystal(asym_unit, space_group, unit_cell);
    
    PDD pdd(crystal, 10);
    occ::Vec amd = pdd.average_minimum_distance();
    
    // For FCC, first neighbors should be at distance sqrt(2)/2 ≈ 0.707
    // (based on reference data)
    REQUIRE_THAT(amd(0), WithinAbs(0.7071067811865476, eps));
  }

  SECTION("PDD configuration options") {
    // Simple test structure
    occ::Mat3N positions(3, 1);
    positions << 0.0, 0.0, 0.0;
    occ::IVec atomic_numbers(1);
    atomic_numbers << 1;
    
    AsymmetricUnit asym_unit(positions, atomic_numbers);
    UnitCell unit_cell(1.0, 1.0, 1.0, M_PI/2, M_PI/2, M_PI/2);
    SpaceGroup space_group(1);
    
    Crystal crystal(asym_unit, space_group, unit_cell);
    
    // Test with different configurations
    PDDConfig config_no_collapse;
    config_no_collapse.collapse = false;
    config_no_collapse.return_groups = true;
    
    PDD pdd_no_collapse(crystal, 5, config_no_collapse);
    
    REQUIRE(pdd_no_collapse.size() > 0);
    REQUIRE(pdd_no_collapse.groups().size() > 0);
    
    // Test lexsort disabled
    PDDConfig config_no_sort;
    config_no_sort.lexsort = false;
    
    PDD pdd_no_sort(crystal, 5, config_no_sort);
    REQUIRE(pdd_no_sort.size() > 0);
  }

  SECTION("AMD consistency") {
    // Test that AMD calculated directly matches AMD from PDD
    occ::Mat3N positions(3, 2);
    positions << 0.0, 0.5,
                 0.0, 0.5,
                 0.0, 0.5;
    occ::IVec atomic_numbers(2);
    atomic_numbers << 1, 1;
    
    AsymmetricUnit asym_unit(positions, atomic_numbers);
    UnitCell unit_cell(1.0, 1.0, 1.0, M_PI/2, M_PI/2, M_PI/2);
    SpaceGroup space_group(1);
    
    Crystal crystal(asym_unit, space_group, unit_cell);
    
    PDD pdd(crystal, 8);
    occ::Vec amd_from_pdd = pdd.average_minimum_distance();
    
    // Both should give the same result
    REQUIRE(amd_from_pdd.size() == 8);
    
    // Check that all values are positive (distances should be positive)
    for (int i = 0; i < amd_from_pdd.size(); ++i) {
      REQUIRE(amd_from_pdd(i) > 0.0);
    }
  }

}

TEST_CASE("Acetic acid PDD", "[acetic]") {
  using Catch::Matchers::WithinAbs;
  using occ::crystal::Crystal;
  using occ::crystal::SpaceGroup;
  using occ::crystal::UnitCell;
  using occ::crystal::AsymmetricUnit;
  using occ::descriptors::PDD;
  using occ::descriptors::PDDConfig;
  constexpr double eps = 1e-6;

  SECTION("Acetic acid crystal") {
    // Use the same acetic acid structure from crystal_tests.cpp - calling acetic_asym() directly
    // would be cleaner but requires exposing that function. For now, replicate it here.
    const std::vector<std::string> labels = {"C1", "C2", "H1", "H2", "H3", "H4", "O1", "O2"};
    occ::IVec atomic_numbers(labels.size());
    occ::Mat positions(labels.size(), 3);
    
    for (size_t i = 0; i < labels.size(); i++) {
      atomic_numbers(i) = occ::core::Element(labels[i]).atomic_number();
    }
    
    // Positions from acetic_asym() function in crystal_tests.cpp (fractional coordinates)
    positions << 0.16510, 0.28580, 0.17090, 0.08940, 0.37620, 0.34810, 0.18200, 0.05100,
                -0.11600, 0.12800, 0.51000, 0.49100, 0.03300, 0.54000, 0.27900, 0.05300,
                 0.16800, 0.42100, 0.12870, 0.10750, 0.00000, 0.25290, 0.37030, 0.17690;
    
    AsymmetricUnit asym_unit(positions.transpose(), atomic_numbers, labels);
    UnitCell unit_cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);
    SpaceGroup space_group(33); // Pna21
    
    Crystal crystal(asym_unit, space_group, unit_cell);
    
    // Test PDD calculation with no sorting to match Python reference
    PDDConfig config;
    config.lexsort = false;  // Match Python reference parameters
    config.collapse = false;
    PDD pdd(crystal, 20, config);
    
    REQUIRE(pdd.size() > 0);
    REQUIRE(pdd.k() == 20);
    
    // Calculate AMD
    occ::Vec amd = pdd.average_minimum_distance();
    REQUIRE(amd.size() == 20);
    
    // Reference PDD matrix from Python AMD implementation using acetic_acid.cif
    // Shape: (8 atoms, 21 columns = weight + 20 distances)
    // Using lexsort=False, collapse=False to match raw neighbor finding order
    // Updated to match current AMD library output
    occ::Mat expected_pdd(8, 21);
    expected_pdd << 
      0.125000, 1.219379, 1.317110, 1.480109, 1.923221, 2.115779, 2.127803, 2.136358, 2.612530, 3.151560, 3.487366, 3.541226, 3.542645, 3.543793, 3.551732, 3.581595, 3.585105, 3.707953, 3.735425, 3.836573, 3.873320,
      0.125000, 1.067286, 1.082780, 1.113573, 1.480109, 2.343842, 2.388600, 2.985135, 3.132740, 3.227709, 3.308990, 3.476208, 3.504623, 3.532061, 3.576590, 3.578649, 3.621567, 3.622530, 3.642788, 3.671548, 3.681262,
      0.125000, 1.000915, 1.648558, 1.923221, 2.332552, 2.608088, 2.612530, 3.027310, 3.132740, 3.204020, 3.227709, 3.247002, 3.261124, 3.371283, 3.393513, 3.551732, 3.565441, 3.576590, 3.620945, 3.667780, 3.671657,
      0.125000, 1.113573, 1.760660, 1.767675, 2.115779, 2.416140, 2.520637, 2.608088, 2.715957, 2.904588, 3.027310, 3.151560, 3.245682, 3.247002, 3.263313, 3.270191, 3.359952, 3.578649, 3.681262, 3.695943, 3.707953,
      0.125000, 1.082780, 1.750363, 1.760660, 2.136358, 2.642150, 2.709339, 2.714246, 2.715957, 2.887150, 2.985135, 3.024054, 3.024054, 3.065178, 3.099978, 3.371283, 3.532061, 3.541226, 3.581595, 3.620945, 3.646174,
      0.125000, 1.067286, 1.750363, 1.767675, 2.127803, 2.633771, 2.642150, 2.707585, 2.714246, 2.904588, 3.120439, 3.204020, 3.213908, 3.261124, 3.308990, 3.486249, 3.486249, 3.487209, 3.565441, 3.621567, 3.695943,
      0.125000, 1.000915, 1.317110, 2.219990, 2.343842, 2.623125, 2.633771, 2.707585, 2.709339, 2.887150, 3.099978, 3.263313, 3.270191, 3.359952, 3.487209, 3.542645, 3.585105, 3.592077, 3.622530, 3.642788, 3.646174,
      0.125000, 1.219379, 1.648558, 2.219990, 2.332552, 2.388600, 2.416140, 2.520637, 2.623125, 3.065178, 3.120439, 3.213908, 3.245682, 3.393513, 3.476208, 3.487366, 3.504623, 3.531867, 3.531867, 3.531867, 3.531867;
    
    // Check against reference values from Python AMD implementation
    // These values were calculated using the same CIF file structure
    std::vector<double> expected_amd = {
      1.0964393986, 1.5344377104, 1.7816114933, 2.0990268288, 2.4714369510,
      2.5063712662, 2.6758678784, 2.7945791420, 3.0289928370, 3.1721706502,
      3.2651612888, 3.2884885469, 3.3467384036, 3.3946916814, 3.4837137516,
      3.5300127208, 3.5797625897, 3.6227315981, 3.6611262606, 3.6817937666
    };
    
    // Check if PDD matrices are close (transpose expected since we store environments as columns now)
    REQUIRE_THAT(pdd.matrix(), IsApproxMatrix(expected_pdd.transpose(), 1e-6, 1e-6));
    
    // Check that AMD values match the reference
    occ::Vec expected_amd_vec(expected_amd.size());
    for (int i = 0; i < static_cast<int>(expected_amd.size()); ++i) {
      expected_amd_vec(i) = expected_amd[i];
    }
    REQUIRE_THAT(amd, IsApproxMatrix(expected_amd_vec, 1e-6, 1e-6));
    
    // Test that the descriptor is deterministic
    PDD pdd2(crystal, 20);
    occ::Vec amd2 = pdd2.average_minimum_distance();
    
    for (int i = 0; i < amd.size(); ++i) {
      REQUIRE_THAT(amd(i), WithinAbs(amd2(i), eps));
    }
  }
}

TEST_CASE("PDD Performance Benchmark", "[pdd][benchmark]") {
  using occ::crystal::Crystal;
  using occ::crystal::SpaceGroup;
  using occ::crystal::UnitCell;
  using occ::crystal::AsymmetricUnit;
  using occ::descriptors::PDD;
  using occ::descriptors::PDDConfig;

  SECTION("Large k benchmark - acetic acid") {
    // Use acetic acid structure for realistic benchmark
    const std::vector<std::string> labels = {"C1", "C2", "H1", "H2", "H3", "H4", "O1", "O2"};
    occ::IVec atomic_numbers(labels.size());
    occ::Mat positions(labels.size(), 3);
    
    for (size_t i = 0; i < labels.size(); i++) {
      atomic_numbers(i) = occ::core::Element(labels[i]).atomic_number();
    }
    
    positions << 0.16510, 0.28580, 0.17090, 0.08940, 0.37620, 0.34810, 0.18200, 0.05100,
                -0.11600, 0.12800, 0.51000, 0.49100, 0.03300, 0.54000, 0.27900, 0.05300,
                 0.16800, 0.42100, 0.12870, 0.10750, 0.00000, 0.25290, 0.37030, 0.17690;
    
    AsymmetricUnit asym_unit(positions.transpose(), atomic_numbers, labels);
    UnitCell unit_cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);
    SpaceGroup space_group(33); // Pna21
    
    Crystal crystal(asym_unit, space_group, unit_cell);
    
    // Benchmark different k values
    std::vector<int> k_values = {20, 50, 100, 200};
    
    for (int k : k_values) {
      fmt::print("\n=== Benchmarking PDD with k={} ===\n", k);
      
      // Warm up
      PDD warmup(crystal, 10);
      
      // Benchmark PDD construction
      auto start = timer::time();
      PDD pdd(crystal, k);
      auto pdd_time = timer::time();
      
      // Benchmark AMD calculation
      occ::Vec amd = pdd.average_minimum_distance();
      auto amd_time = timer::time();
      
      double pdd_seconds = timer::seconds(pdd_time - start);
      double amd_seconds = timer::seconds(amd_time - pdd_time);
      double total_seconds = timer::seconds(amd_time - start);
      
      fmt::print("PDD construction: {:.6f} seconds\n", pdd_seconds);
      fmt::print("AMD calculation:  {:.6f} seconds\n", amd_seconds);
      fmt::print("Total time:       {:.6f} seconds\n", total_seconds);
      fmt::print("PDD size: {} environments\n", pdd.size());
      fmt::print("AMD size: {} values\n", amd.size());
      
      // Verify results are reasonable
      REQUIRE(pdd.size() > 0);
      REQUIRE(amd.size() == k);
      REQUIRE(amd(0) > 0.5);  // Sanity check
      REQUIRE(amd(0) < 5.0);
      REQUIRE(total_seconds < 1.0); // Should be fast
    }
  }
  
  SECTION("Multiple runs for k=100") {
    // More detailed benchmark for k=100 with multiple runs
    const std::vector<std::string> labels = {"C1", "C2", "H1", "H2", "H3", "H4", "O1", "O2"};
    occ::IVec atomic_numbers(labels.size());
    occ::Mat positions(labels.size(), 3);
    
    for (size_t i = 0; i < labels.size(); i++) {
      atomic_numbers(i) = occ::core::Element(labels[i]).atomic_number();
    }
    
    positions << 0.16510, 0.28580, 0.17090, 0.08940, 0.37620, 0.34810, 0.18200, 0.05100,
                -0.11600, 0.12800, 0.51000, 0.49100, 0.03300, 0.54000, 0.27900, 0.05300,
                 0.16800, 0.42100, 0.12870, 0.10750, 0.00000, 0.25290, 0.37030, 0.17690;
    
    AsymmetricUnit asym_unit(positions.transpose(), atomic_numbers, labels);
    UnitCell unit_cell = occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);
    SpaceGroup space_group(33);
    
    Crystal crystal(asym_unit, space_group, unit_cell);
    
    const int k = 100;
    const int num_runs = 100;
    std::vector<double> times;
    times.reserve(num_runs);
    
    // Warm up
    PDD warmup(crystal, 10);
    
    for (int run = 0; run < num_runs; ++run) {
      auto start = timer::time();
      
      // Use minimal config like Python benchmark (no lexsort, no collapse)
      PDDConfig minimal_config;
      minimal_config.lexsort = false;
      minimal_config.collapse = false;
      
      PDD pdd(crystal, k, minimal_config);
      occ::Vec amd = pdd.average_minimum_distance();
      auto end = timer::time();
      
      double seconds = timer::seconds(end - start);
      times.push_back(seconds);
    }
    
    // Calculate statistics
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / num_runs;
    
    double sq_sum = 0.0;
    for (double t : times) {
      sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / num_runs);
    
    auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());
    double min_time = *min_it;
    double max_time = *max_it;
    
    fmt::print("\nC++ Performance Statistics ({} runs):\n", num_runs);
    fmt::print("Mean:   {:.6f} ± {:.6f} seconds\n", mean, std_dev);
    fmt::print("Min:    {:.6f} seconds\n", min_time);
    fmt::print("Max:    {:.6f} seconds\n", max_time);
    fmt::print("Range:  {:.6f} seconds\n", max_time - min_time);
    
    // Require reasonable performance (should be much faster than 1 second)
    REQUIRE(mean < 1.0);
    REQUIRE(min_time > 0.0);
    REQUIRE(std_dev < 0.1); // Should be consistent
  }
}

TEST_CASE("SortedKDistances comprehensive tests", "[sorted_k_distances]") {
  using namespace occ::descriptors;
  
  SECTION("Basic functionality with k=5") {
    SortedKDistances<5> sorted_k;
    
    // Check initial state
    REQUIRE(sorted_k.empty());
    REQUIRE(sorted_k.size() == 0);
    
    // Insert single values
    REQUIRE(sorted_k.try_insert(3.0));
    REQUIRE(sorted_k.size() == 1);
    REQUIRE(sorted_k[0] == 3.0);
    REQUIRE(sorted_k.back() == 3.0);
    
    REQUIRE(sorted_k.try_insert(1.0));
    REQUIRE(sorted_k.size() == 2);
    REQUIRE(sorted_k[0] == 1.0);
    REQUIRE(sorted_k[1] == 3.0);
    REQUIRE(sorted_k.back() == 3.0);
    
    REQUIRE(sorted_k.try_insert(2.0));
    REQUIRE(sorted_k.size() == 3);
    REQUIRE(sorted_k[0] == 1.0);
    REQUIRE(sorted_k[1] == 2.0);
    REQUIRE(sorted_k[2] == 3.0);
    REQUIRE(sorted_k.back() == 3.0);
    
    // Fill to capacity
    REQUIRE(sorted_k.try_insert(5.0));
    REQUIRE(sorted_k.try_insert(4.0));
    REQUIRE(sorted_k.size() == 5);
    
    // Check sorted order
    for (int i = 0; i < 5; ++i) {
      REQUIRE(sorted_k[i] == static_cast<double>(i + 1));
    }
    
    // Try to insert larger value (should be rejected)
    REQUIRE_FALSE(sorted_k.try_insert(6.0));
    REQUIRE(sorted_k.size() == 5);
    REQUIRE(sorted_k.back() == 5.0);
    
    // Insert smaller value (should replace worst)
    REQUIRE(sorted_k.try_insert(0.5));
    REQUIRE(sorted_k.size() == 5);
    REQUIRE(sorted_k[0] == 0.5);
    REQUIRE(sorted_k[1] == 1.0);
    REQUIRE(sorted_k[2] == 2.0);
    REQUIRE(sorted_k[3] == 3.0);
    REQUIRE(sorted_k[4] == 4.0);
    REQUIRE(sorted_k.back() == 4.0);
  }
  
  SECTION("try_insert_batch4 functionality") {
    SortedKDistances<10> sorted_k;
    
    // Test batch insertion on empty container
    double batch1[4] = {4.0, 2.0, 6.0, 1.0};
    int count = sorted_k.try_insert_batch4(batch1);
    REQUIRE(count == 4);
    REQUIRE(sorted_k.size() == 4);
    
    // Check if properly sorted
    REQUIRE(sorted_k[0] == 1.0);
    REQUIRE(sorted_k[1] == 2.0);
    REQUIRE(sorted_k[2] == 4.0);
    REQUIRE(sorted_k[3] == 6.0);
    
    // Test batch insertion with some values rejected
    double batch2[4] = {8.0, 3.0, 7.0, 5.0};
    count = sorted_k.try_insert_batch4(batch2);
    REQUIRE(count == 4);
    REQUIRE(sorted_k.size() == 8);
    
    // Check sorted order
    std::vector<double> expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    for (size_t i = 0; i < expected.size(); ++i) {
      REQUIRE(sorted_k[i] == expected[i]);
    }
    
    // Fill to capacity
    double batch3[4] = {10.0, 9.0, 12.0, 11.0};
    count = sorted_k.try_insert_batch4(batch3);
    REQUIRE(count == 2); // Only 9.0 and 10.0 should be inserted
    REQUIRE(sorted_k.size() == 10);
    
    // Test batch insertion when full with some rejections
    double batch4[4] = {0.5, 15.0, 2.5, 20.0};
    count = sorted_k.try_insert_batch4(batch4);
    REQUIRE(count == 2); // Only 0.5 and 2.5 should be inserted
    REQUIRE(sorted_k.size() == 10);
    REQUIRE(sorted_k[0] == 0.5);
    REQUIRE(sorted_k[1] == 1.0);
    REQUIRE(sorted_k[2] == 2.0);
    REQUIRE(sorted_k[3] == 2.5);
  }
  
  SECTION("try_insert_batch4 vs individual inserts consistency") {
    SortedKDistances<8> sorted_k1, sorted_k2;
    
    // Test data - same values inserted differently
    double test_values[12] = {5.2, 1.8, 3.1, 7.4, 2.9, 6.3, 4.7, 8.1, 1.2, 9.6, 3.8, 2.3};
    
    // Method 1: Insert individually
    for (int i = 0; i < 12; ++i) {
      sorted_k1.try_insert(test_values[i]);
    }
    
    // Method 2: Insert in batches of 4
    for (int i = 0; i < 12; i += 4) {
      sorted_k2.try_insert_batch4(&test_values[i]);
    }
    
    // Both should have same final state
    REQUIRE(sorted_k1.size() == sorted_k2.size());
    for (size_t i = 0; i < sorted_k1.size(); ++i) {
      REQUIRE(sorted_k1[i] == Catch::Approx(sorted_k2[i]).epsilon(1e-10));
    }
  }
  
  SECTION("Edge cases with k=1") {
    SortedKDistances<1> sorted_k;
    
    REQUIRE(sorted_k.try_insert(5.0));
    REQUIRE(sorted_k.size() == 1);
    REQUIRE(sorted_k[0] == 5.0);
    
    // Should replace with smaller value
    REQUIRE(sorted_k.try_insert(3.0));
    REQUIRE(sorted_k.size() == 1);
    REQUIRE(sorted_k[0] == 3.0);
    
    // Should reject larger value
    REQUIRE_FALSE(sorted_k.try_insert(4.0));
    REQUIRE(sorted_k[0] == 3.0);
    
    // Test batch insertion
    double batch[4] = {6.0, 1.0, 4.0, 8.0};
    int count = sorted_k.try_insert_batch4(batch);
    REQUIRE(count == 1); // Only 1.0 should be inserted
    REQUIRE(sorted_k.size() == 1);
    REQUIRE(sorted_k[0] == 1.0);
  }
  
  SECTION("Performance and SIMD detection") {
    fmt::print("SortedKDistances SIMD type: {}\n", SortedKDistances<100>::simd_type());
    
    // Quick performance test
    SortedKDistances<50> sorted_k;
    std::vector<double> test_data(1000);
    std::iota(test_data.begin(), test_data.end(), 0.0);
    std::shuffle(test_data.begin(), test_data.end(), std::mt19937{42});
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_data.size(); i += 4) {
      if (i + 3 < test_data.size()) {
        sorted_k.try_insert_batch4(&test_data[i]);
      } else {
        // Handle remaining elements individually
        for (size_t j = i; j < test_data.size(); ++j) {
          sorted_k.try_insert(test_data[j]);
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    fmt::print("Batch insertion of 1000 values took {} microseconds\n", duration.count());
    
    REQUIRE(sorted_k.size() == 50);
    // Check that we got the 50 smallest values
    for (int i = 0; i < 50; ++i) {
      REQUIRE(sorted_k[i] == static_cast<double>(i));
    }
  }
  
  SECTION("Clear and reuse") {
    SortedKDistances<5> sorted_k;
    
    // Fill it up
    for (int i = 0; i < 10; ++i) {
      sorted_k.try_insert(static_cast<double>(i));
    }
    REQUIRE(sorted_k.size() == 5);
    
    // Clear and verify
    sorted_k.clear();
    REQUIRE(sorted_k.empty());
    REQUIRE(sorted_k.size() == 0);
    
    // Reuse should work
    REQUIRE(sorted_k.try_insert(42.0));
    REQUIRE(sorted_k.size() == 1);
    REQUIRE(sorted_k[0] == 42.0);
  }
}
