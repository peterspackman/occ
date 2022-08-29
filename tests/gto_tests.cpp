#include <catch2/catch_test_macros.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <vector>

using occ::Mat;

// GTO

TEST_CASE("GTO values, derivatives & 2nd derivatives H2/STO-3G") {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                       {1, 0.0, 0.0, 1.398397}};
    occ::qm::AOBasis aobasis = occ::qm::AOBasis::load(atoms, "sto-3g");
    auto grid_pts = Mat::Zero(3, 4);
    auto gto_values = occ::gto::evaluate_basis(aobasis, grid_pts, 2);
    fmt::print("Gto values\nphi:\n{}\n", gto_values.phi);
    fmt::print("phi_x\n{}\n", gto_values.phi_x);
    fmt::print("phi_y\n{}\n", gto_values.phi_y);
    fmt::print("phi_z\n{}\n", gto_values.phi_z);
    fmt::print("phi_xx\n{}\n", gto_values.phi_xx);
    fmt::print("phi_xy\n{}\n", gto_values.phi_xy);
    fmt::print("phi_xz\n{}\n", gto_values.phi_xz);
    fmt::print("phi_yy\n{}\n", gto_values.phi_yy);
    fmt::print("phi_yz\n{}\n", gto_values.phi_yz);
    fmt::print("phi_zz\n{}\n", gto_values.phi_zz);

    Mat D(2, 2);
    D.setConstant(0.60245569);
    auto rho = occ::density::evaluate_density_on_grid<2>(aobasis, D, grid_pts);
    fmt::print("Rho\n{}\n", rho);
}

TEST_CASE("GTO values, derivatives & density H2/3-21G") {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                       {1, 0.0, 0.0, 1.398397}};
    occ::qm::AOBasis basis = occ::qm::AOBasis::load(atoms, "3-21G");
    auto grid_pts = Mat::Identity(3, 4);
    auto gto_values = occ::gto::evaluate_basis(basis, grid_pts, 1);
    fmt::print("Gto values\nphi:\n{}\n", gto_values.phi);
    fmt::print("phi_x\n{}\n", gto_values.phi_x);
    fmt::print("phi_y\n{}\n", gto_values.phi_y);
    fmt::print("phi_z\n{}\n", gto_values.phi_z);

    Mat D(4, 4);
    D << 0.175416203439, 0.181496024303, 0.175416203439, 0.181496024303,
        0.181496024303, 0.187786568128, 0.181496024303, 0.187786568128,
        0.175416203439, 0.181496024303, 0.175416203439, 0.181496024303,
        0.181496024303, 0.187786568128, 0.181496024303, 0.187786568128;

    auto rho = occ::density::evaluate_density_on_grid<1>(basis, D, grid_pts);
    fmt::print("Rho\n{}\n", rho);
}

TEST_CASE("GTO values, derivatives & density H2/STO-3G Unrestricted") {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                       {1, 0.0, 0.0, 1.398397}};
    occ::qm::AOBasis basis = occ::qm::AOBasis::load(atoms, "sto-3g");
    auto grid_pts = Mat::Identity(3, 4);
    auto gto_values = occ::gto::evaluate_basis(basis, grid_pts, 1);
    fmt::print("Gto values\nphi:\n{}\n", gto_values.phi);
    fmt::print("phi_x\n{}\n", gto_values.phi_x);
    fmt::print("phi_y\n{}\n", gto_values.phi_y);
    fmt::print("phi_z\n{}\n", gto_values.phi_z);

    Mat D(4, 2);
    D.block(0, 0, 2, 2).setConstant(0.30122784);
    D.block(2, 0, 2, 2).setConstant(0.30122784);
    auto rho = occ::density::evaluate_density_on_grid<
        1, occ::qm::SpinorbitalKind::Unrestricted>(basis, D, grid_pts);
    fmt::print("Rho alpha\n{}\n", occ::qm::block::a(rho));
    fmt::print("Rho beta\n{}\n", occ::qm::block::b(rho));
}

TEST_CASE("Spherical Gaussian basis <-> Cartesian Gaussian basis transforms") {
    using occ::util::all_close;
    SECTION("c c^-1 = I") {
        for (int i = 0; i < 8; i++) {
            Mat x = Mat::Identity(2 * i + 1, 2 * i + 1);
            Mat c = occ::gto::cartesian_to_spherical_transformation_matrix(i);
            Mat cinv =
                occ::gto::spherical_to_cartesian_transformation_matrix(i);
            REQUIRE(all_close(x, c * cinv, 1e-14, 1e-14));
            fmt::print("c c^-1 (l = {})\n{}\n", i, c * cinv);
        }
    }

    SECTION("Transform with rotation") {
        constexpr int l = 2;
        Mat rotation = Mat::Zero(3, 3);
        // i.e. rotation of 90 deg about x
        rotation << 1, 0, 0, 0, 0, -1, 0, 1, 0;
        Mat rotation_reverse = Mat::Zero(3, 3);
        // i.e. rotation of - 90 deg about x
        rotation_reverse << 1, 0, 0, 0, 0, 1, 0, -1, 0;

        // cartesians
        // xx
        // xy
        // xz
        // yy
        // yz
        // zz
        Mat grot = occ::gto::cartesian_gaussian_rotation_matrix<l>(rotation);
        Mat grot2 =
            occ::gto::cartesian_gaussian_rotation_matrix<l>(rotation_reverse);

        // sqrt(3)*x*y -> -sqrt(3) xz idx = 0
        // sqrt(3)*y*z -> -sqrt(3) yz idx = 1
        //
        // -r**2/2 + 3*z**2/2 = z**2 - (x**2 + y**2)/2 -> y**2 - (x**2 + z**2)/2
        // idx = 2 r ** 2 = x**2 + y**2 + z**2 -x**2 = -r**2 + y**2 + z**2 =
        // (y**2 + y**2 - r**2  + y**2 + z**2 + z**2) / 2 = (3*y**2 + z**2 -
        // r**2)/2
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
        fmt::print("cartesian ({} {})\n{}\n", xcart.rows(), xcart.cols(),
                   xcart);
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
