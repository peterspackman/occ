#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/descriptors/steinhardt.h>
#include <occ/core/util.h>
#include <fmt/ostream.h>

namespace timer {

using namespace std::chrono;

template <typename R, typename P>
inline double seconds(duration<R, P> x) {
  return duration_cast<nanoseconds>(x).count() / 1e9;
}

inline auto time() {
  return high_resolution_clock::now();
}

} // namespace timer



TEST_CASE("Steinhardt q parameters", "[steinhardt]") {
    using Catch::Matchers::WithinAbs;
    using occ::descriptors::Steinhardt;
    constexpr double eps = 1e-6;

    SECTION("Cubic symmetry") {
        Steinhardt steinhardt(6);
        occ::Mat3N positions(3, 8);
        positions <<  1,  1,  1,  1, -1, -1, -1, -1,
                      1,  1, -1, -1,  1,  1, -1, -1, 
                      1, -1,  1, -1,  1, -1,  1, -1;

        occ::Vec q = steinhardt.compute_q(positions);
	fmt::print("Cubic Q\n{}\n", q);

        REQUIRE(q.size() == 7);
        REQUIRE_THAT(q(0), WithinAbs(1.0, eps));
        REQUIRE_THAT(q(1), WithinAbs(0.0, eps));
        REQUIRE_THAT(q(2), WithinAbs(0.0, eps));
        REQUIRE_THAT(q(3), WithinAbs(0.0, eps));
        REQUIRE(q(4) > 0.1);
        REQUIRE_THAT(q(5), WithinAbs(0.0, eps));
        REQUIRE(q(6) > 0.1);
	occ::Vec w = steinhardt.compute_w(positions);
	fmt::print("Cubic W\n{}\n", w);
    }

    SECTION("Octahedral symmetry") {
        Steinhardt steinhardt(6);
        occ::Mat3N positions(3, 6);
        positions <<  1, -1,  0,  0,  0,  0,
                      0,  0,  1, -1,  0,  0,
                      0,  0,  0,  0,  1, -1;

        occ::Vec q = steinhardt.compute_q(positions);
	fmt::print("Octahedral Q\n{}\n", q);

        REQUIRE(q.size() == 7);
        REQUIRE_THAT(q(0), WithinAbs(1.0, eps));
        REQUIRE_THAT(q(1), WithinAbs(0.0, eps));
        REQUIRE_THAT(q(2), WithinAbs(0.0, eps));
        REQUIRE_THAT(q(3), WithinAbs(0.0, eps));
        REQUIRE(q(4) > 0.1);
        REQUIRE_THAT(q(5), WithinAbs(0.0, eps));
        REQUIRE(q(6) > 0.1);
	occ::Vec w = steinhardt.compute_w(positions);
	fmt::print("Octahedral W\n{}\n", w);
    }

    SECTION("Tetrahedral symmetry") {
        Steinhardt steinhardt(6);
        occ::Mat3N positions(3, 4);

        positions.col(0) = occ::Vec3{ 1.0,  0.0, -1 / std::sqrt(2.0)};
        positions.col(1) = occ::Vec3{-1.0,  0.0, -1 / std::sqrt(2.0)};
        positions.col(2) = occ::Vec3{ 0.0,  1.0,  1 / std::sqrt(2.0)};
        positions.col(3) = occ::Vec3{ 0.0, -1.0,  1 / std::sqrt(2.0)};

        occ::Vec q = steinhardt.compute_q(positions);
	fmt::print("Tetrahedral Q\n{}\n", q);

        REQUIRE(q.size() == 7);
        REQUIRE_THAT(q(0), WithinAbs(1.0, eps));
        REQUIRE_THAT(q(1), WithinAbs(0.0, eps));
        REQUIRE_THAT(q(2), WithinAbs(0.0, eps));
        REQUIRE(q(3) > 0.1);
        REQUIRE(q(4) > 0.1);
        REQUIRE_THAT(q(5), WithinAbs(0.0, eps));
        REQUIRE(q(6) > 0.1);
	occ::Vec w = steinhardt.compute_w(positions);
	fmt::print("Tetrahedral W\n{}\n", w);
    }

    SECTION("Icosahedral symmetry") {
        Steinhardt steinhardt(6);
        occ::Mat3N positions(3, 12);

	positions.col(0) =  occ::Vec3{ 0.52573111,  0.85065081,  0.        };
	positions.col(1) =  occ::Vec3{ 0.52573111,  0.85065081,  0.        };
	positions.col(2) =  occ::Vec3{-0.52573111, -0.85065081,  0.        };
	positions.col(3) =  occ::Vec3{-0.52573111, -0.85065081,  0.        };
	positions.col(4) =  occ::Vec3{ 0.        , -0.52573111,  0.85065081};
	positions.col(5) =  occ::Vec3{ 0.        ,  0.52573111,  0.85065081};
	positions.col(6) =  occ::Vec3{ 0.        , -0.52573111, -0.85065081};
	positions.col(7) =  occ::Vec3{ 0.        ,  0.52573111, -0.85065081};
	positions.col(8) =  occ::Vec3{ 0.85065081,  0.        , -0.52573111};
	positions.col(9) =  occ::Vec3{ 0.85065081,  0.        ,  0.52573111};
	positions.col(10) = occ::Vec3{-0.85065081,  0.        , -0.52573111};
	positions.col(11) = occ::Vec3{-0.85065081,  0.        ,  0.52573111};

	fmt::print("Mean pos\n{}\n", positions.rowwise().mean());

        occ::Vec q = steinhardt.compute_q(positions);
	fmt::print("Icosahedral Q\n{}\n", q);

        REQUIRE(q.size() == 7);
        REQUIRE_THAT(q(0), WithinAbs(1.0, eps));
        REQUIRE_THAT(q(1), WithinAbs(0.0, eps));
        REQUIRE(q(2) > 0.1);
        REQUIRE_THAT(q(3), WithinAbs(0.0, eps));
	REQUIRE(q(4) > 0.1);
        REQUIRE_THAT(q(5), WithinAbs(0.0, eps));
        REQUIRE(q(6) > 0.1);

	occ::Vec w = steinhardt.compute_w(positions);
	fmt::print("Icosahedral W\n{}\n", w);
    }
}
