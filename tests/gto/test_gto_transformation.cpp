#include "catch.hpp"
#include <occ/gto/gto.h>
#include <fmt/ostream.h>
#include <occ/core/util.h>

using occ::Mat;

TEST_CASE("spherical <-> cartesian gaussian transforms") {
    using occ::util::all_close;
    SECTION("c c^-1 = I") {
	for(int i = 0; i < 8; i++)
	{
	    Mat x = Mat::Identity(2 * i + 1, 2 * i + 1);
	    Mat c = occ::gto::cartesian_to_spherical_transformation_matrix(i);
	    Mat cinv = occ::gto::spherical_to_cartesian_transformation_matrix(i);
	    REQUIRE(all_close(x, c * cinv, 1e-14, 1e-14));
	    fmt::print("c c^-1 (l = {})\n{}\n", i, c * cinv);
	}
    }

    SECTION("Transform with rotation") {
	constexpr int l = 2;
	Mat rotation = Mat::Zero(3, 3);
	// i.e. rotation of 90 deg about x
	rotation << 1,  0,  0,
		    0,  0, -1,
		    0,  1,  0;
	Mat rotation_reverse = Mat::Zero(3, 3);
	// i.e. rotation of - 90 deg about x
	rotation_reverse << 1,  0,  0,
			    0,  0,  1,
			    0, -1,  0;

	// cartesians
	// xx
	// xy
	// xz
	// yy
	// yz
	// zz
        Mat grot = occ::gto::cartesian_gaussian_rotation_matrix<l>(rotation);
        Mat grot2 = occ::gto::cartesian_gaussian_rotation_matrix<l>(rotation_reverse);

	// sqrt(3)*x*y -> -sqrt(3) xz idx = 0
	// sqrt(3)*y*z -> -sqrt(3) yz idx = 1
	//
	// -r**2/2 + 3*z**2/2 = z**2 - (x**2 + y**2)/2 -> y**2 - (x**2 + z**2)/2 idx = 2
	// r ** 2 = x**2 + y**2 + z**2
	// -x**2 = -r**2 + y**2 + z**2
	// = (y**2 + y**2 - r**2  + y**2 + z**2 + z**2) / 2
	// = (3*y**2 + z**2 - r**2)/2
	//
	// sqrt(3)*x*z -> sqrt(3) xy idx = 3
	// sqrt(3)*(x**2 - y**2)/2 -> sqrt(3) * (x**2 - z**2)/2 idx = 4

	Mat x = Mat::Random(5, 13);
	x(0, 0) = 2.5;
	x(1, 1) = 0.5;
	x(2, 1) = 0.67;
	x(3, 2) = 1;
	x(4, 2) = 2;
	Mat c = occ::gto::cartesian_to_spherical_transformation_matrix(l);
	Mat cinv = occ::gto::spherical_to_cartesian_transformation_matrix(l);
	Mat srot = c * grot * cinv;

	fmt::print("c {} {}\n{}\n", c.rows(), c.cols(), c);
	fmt::print("c^-1 {} {}\n{}\n", cinv.rows(), cinv.cols(), cinv);
	fmt::print("coeffs sph ({} {})\n{}\n", x.rows(), x.cols(), x);
	fmt::print("srot ({} {})\n{}\n", srot.rows(), srot.cols(), srot);

	Mat xcart = cinv * x;
	Mat xcart_rot = grot * xcart;
	Mat xsph_rot = c * xcart_rot;
	fmt::print("cartesian ({} {})\n{}\n", xcart.rows(), xcart.cols(), xcart);
	fmt::print("cartesian rotated\n{}\n", xcart_rot);
	fmt::print("spherical rotated\n{}\n", xsph_rot);
	fmt::print("spherical rotated2\n{}\n", srot * x);

	Mat xcart2 = cinv * xsph_rot;
	Mat xcart_rot2 = grot2 * xcart2;
	Mat xsph_rot2 = c * xcart_rot2;
	fmt::print("cartesian\n{}\n", xcart2);
	fmt::print("cartesian rotated\n{}\n", xcart_rot2);
	fmt::print("spherical rotated\n{}\n", xsph_rot2);
	REQUIRE(all_close(x, xsph_rot2, 1e-12, 1e-12));

    }
}
