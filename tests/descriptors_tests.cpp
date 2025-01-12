#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <chrono>
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/util.h>
#include <occ/descriptors/steinhardt.h>

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
