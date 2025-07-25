#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/conditioning_orthogonalizer.h>
#include <occ/core/util.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/gto/rotation.h>
#include <occ/qm/hf.h>
#include <occ/qm/mo.h>
#include <occ/qm/integral_engine.h>
#include <vector>
#include "test_utils.h"

using Catch::Approx;
using occ::format_matrix;
using occ::Mat;

// GTO

TEST_CASE("GTO values derivatives & 2nd derivatives H2/STO-3G") {
  std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                     {1, 0.0, 0.0, 1.398397}};
  occ::qm::AOBasis aobasis = occ::qm::AOBasis::load(atoms, "sto-3g");
  Mat grid_pts = Mat::Identity(3, 4);
  auto gto_values = occ::gto::evaluate_basis(aobasis, grid_pts, 2);
  fmt::print("Gto values\nphi:\n{}\n", format_matrix(gto_values.phi));
  fmt::print("phi_x\n{}\n", format_matrix(gto_values.phi_x));
  fmt::print("phi_y\n{}\n", format_matrix(gto_values.phi_y));
  fmt::print("phi_z\n{}\n", format_matrix(gto_values.phi_z));
  fmt::print("phi_xx\n{}\n", format_matrix(gto_values.phi_xx));
  fmt::print("phi_xy\n{}\n", format_matrix(gto_values.phi_xy));
  fmt::print("phi_xz\n{}\n", format_matrix(gto_values.phi_xz));
  fmt::print("phi_yy\n{}\n", format_matrix(gto_values.phi_yy));
  fmt::print("phi_yz\n{}\n", format_matrix(gto_values.phi_yz));
  fmt::print("phi_zz\n{}\n", format_matrix(gto_values.phi_zz));

  Mat D(2, 2);
  D.setConstant(0.60245569);
  auto rho = occ::density::evaluate_density_on_grid<2>(aobasis, D, grid_pts);
  fmt::print("Rho\n{}\n", format_matrix(rho));
}

TEST_CASE("GTO values derivatives & density H2/3-21G") {
  std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                     {1, 0.0, 0.0, 1.398397}};
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(atoms, "3-21G");
  auto grid_pts = Mat::Identity(3, 4);
  auto gto_values = occ::gto::evaluate_basis(basis, grid_pts, 1);
  fmt::print("Gto values\nphi:\n{}\n", format_matrix(gto_values.phi));
  fmt::print("phi_x\n{}\n", format_matrix(gto_values.phi_x));
  fmt::print("phi_y\n{}\n", format_matrix(gto_values.phi_y));
  fmt::print("phi_z\n{}\n", format_matrix(gto_values.phi_z));

  Mat D(4, 4);
  D << 0.175416203439, 0.181496024303, 0.175416203439, 0.181496024303,
      0.181496024303, 0.187786568128, 0.181496024303, 0.187786568128,
      0.175416203439, 0.181496024303, 0.175416203439, 0.181496024303,
      0.181496024303, 0.187786568128, 0.181496024303, 0.187786568128;

  auto rho = occ::density::evaluate_density_on_grid<1>(basis, D, grid_pts);
  fmt::print("Rho\n{}\n", format_matrix(rho));
}

TEST_CASE("GTO values derivatives & density H2/STO-3G Unrestricted") {
  std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                     {1, 0.0, 0.0, 1.398397}};
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(atoms, "sto-3g");
  auto grid_pts = Mat::Identity(3, 4);
  auto gto_values = occ::gto::evaluate_basis(basis, grid_pts, 1);
  fmt::print("Gto values\nphi:\n{}\n", format_matrix(gto_values.phi));
  fmt::print("phi_x\n{}\n", format_matrix(gto_values.phi_x));
  fmt::print("phi_y\n{}\n", format_matrix(gto_values.phi_y));
  fmt::print("phi_z\n{}\n", format_matrix(gto_values.phi_z));

  Mat D(4, 2);
  D.block(0, 0, 2, 2).setConstant(0.30122784);
  D.block(2, 0, 2, 2).setConstant(0.30122784);
  auto rho = occ::density::evaluate_density_on_grid<
      1, occ::qm::SpinorbitalKind::Unrestricted>(basis, D, grid_pts);
  fmt::print("Rho alpha\n{}\n", format_matrix(occ::qm::block::a(rho)));
  fmt::print("Rho beta\n{}\n", format_matrix(occ::qm::block::b(rho)));
}

TEST_CASE("Spherical GTO rotations") {
  // These tests have been modified from
  // https://github.com/google/spherical-harmonics
  // which is licensed under the Apache-2.0 license
  // (compatible with GPLv3)
  //
  // The band-level rotation matrices for a rotation about the z-axis are
  // relatively simple so we can compute them closed form and make sure the
  // recursive general approach works properly.
  // This closed form comes from [1].
  using occ::Mat;
  using occ::Mat3;
  using occ::util::all_close;

  SECTION("Closed form z-axis rotation") {
    double alpha = M_PI / 4.0;
    Eigen::Quaterniond rz(Eigen::AngleAxisd(alpha, Eigen::Vector3d::UnitZ()));

    Mat3 rotation_matrix = rz.normalized().toRotationMatrix();

    auto rotations =
        occ::gto::spherical_gaussian_rotation_matrices(3, rotation_matrix);

    fmt::print("Rotation matrix\n{}\n", format_matrix(rotation_matrix));
    // order 0
    Mat r0(1, 1);
    r0.setConstant(1.0);

    REQUIRE(all_close(r0, rotations[0], 1e-10, 1e-10));

    // order 1
    Mat r1(3, 3);
    r1 << std::cos(alpha), 0, std::sin(alpha), 0, 1, 0, -std::sin(alpha), 0,
        std::cos(alpha);
    fmt::print("Found  (l == 1)\n{}\n", format_matrix(rotations[1]));
    fmt::print("Expect (l == 1)\n{}\n", format_matrix(r1));
    REQUIRE(all_close(r1, rotations[1], 1e-10, 1e-10));

    // order 2
    Mat r2(5, 5);
    r2 << cos(2 * alpha), 0, 0, 0, sin(2 * alpha), 0, cos(alpha), 0, sin(alpha),
        0, 0, 0, 1, 0, 0, 0, -sin(alpha), 0, cos(alpha), 0, -sin(2 * alpha), 0,
        0, 0, cos(2 * alpha);
    fmt::print("Found  (l == 2)\n{}\n", format_matrix(rotations[2]));
    fmt::print("Expect (l == 2)\n{}\n", format_matrix(r2));
    REQUIRE(all_close(r2, rotations[2], 1e-10, 1e-10));

    // order 3
    Mat r3(7, 7);
    r3 << cos(3 * alpha), 0, 0, 0, 0, 0, sin(3 * alpha), 0, cos(2 * alpha), 0,
        0, 0, sin(2 * alpha), 0, 0, 0, cos(alpha), 0, sin(alpha), 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, -sin(alpha), 0, cos(alpha), 0, 0, 0, -sin(2 * alpha),
        0, 0, 0, cos(2 * alpha), 0, -sin(3 * alpha), 0, 0, 0, 0, 0,
        cos(3 * alpha);
    fmt::print("Found  (l == 3)\n{}\n", format_matrix(rotations[3]));
    fmt::print("Expect (l == 3)\n{}\n", format_matrix(r2));
    REQUIRE(all_close(r3, rotations[3], 1e-10, 1e-10));
  }

  SECTION("Closed form bands") {
    // Use an arbitrary rotation
    Eigen::Quaterniond r(Eigen::AngleAxisd(
        0.423, Eigen::Vector3d(0.234, -0.642, 0.829).normalized()));
    Mat3 rot = r.toRotationMatrix();

    // Create rotation for band 1 and 2
    auto rotations = occ::gto::spherical_gaussian_rotation_matrices(3, rot);

    // For l = 1, the transformation matrix for the coefficients is
    // relatively easy to derive. If R is the rotation matrix, the elements
    // of the transform can be described as: Mij = integral_over_sphere Yi(R
    // * s)Yj(s) ds. For l = 1, we have:
    //   Y0(s) = -0.5sqrt(3/pi)s.y
    //   Y1(s) = 0.5sqrt(3/pi)s.z
    //   Y2(s) = -0.5sqrt(3/pi)s.x
    // Note that these Yi include the Condon-Shortely phase. The expectent
    // matrix M is equal to:
    //   [ R11  -R12   R10
    //    -R21   R22  -R20
    //     R01  -R02   R00 ]
    // In [1]'s Appendix summarizing [4], this is given without the negative
    // signs and is a simple permutation, but that is because [4] does not
    // include the Condon-Shortely phase in their definition of the SH basis
    // functions.
    Eigen::Matrix3d band_1 = rotations[1];

    REQUIRE(rot(1, 1) == band_1(0, 0));
    REQUIRE(rot(1, 2) == band_1(0, 1));
    REQUIRE(rot(1, 0) == band_1(0, 2));
    REQUIRE(rot(2, 1) == band_1(1, 0));
    REQUIRE(rot(2, 2) == band_1(1, 1));
    REQUIRE(rot(2, 0) == band_1(1, 2));
    REQUIRE(rot(0, 1) == band_1(2, 0));
    REQUIRE(rot(0, 2) == band_1(2, 1));
    REQUIRE(rot(0, 0) == band_1(2, 2));

    // The l = 2 band transformation is significantly more complex in terms
    // of R, and a CAS program was used to arrive at these equations (plus a
    // fair amount of simplification by hand afterwards).
    Mat band_2 = rotations[2];
    REQUIRE(rot(0, 0) * rot(1, 1) + rot(0, 1) * rot(1, 0) ==
            Approx(band_2(0, 0)).epsilon(1e-10));
    REQUIRE(rot(0, 1) * rot(1, 2) + rot(0, 2) * rot(1, 1) ==
            Approx(band_2(0, 1)).epsilon(1e-10));
    REQUIRE(-std::sqrt(3) / 3 *
                (rot(0, 0) * rot(1, 0) + rot(0, 1) * rot(1, 1) -
                 2 * rot(0, 2) * rot(1, 2)) ==
            Approx(band_2(0, 2)).epsilon(1e-10));
    REQUIRE(rot(0, 0) * rot(1, 2) + rot(0, 2) * rot(1, 0) ==
            Approx(band_2(0, 3)).epsilon(1e-10));
    REQUIRE(rot(0, 0) * rot(1, 0) - rot(0, 1) * rot(1, 1) ==
            Approx(band_2(0, 4)).epsilon(1e-10));

    REQUIRE(rot(1, 0) * rot(2, 1) + rot(1, 1) * rot(2, 0) ==
            Approx(band_2(1, 0)).epsilon(1e-10));
    REQUIRE(rot(1, 1) * rot(2, 2) + rot(1, 2) * rot(2, 1) ==
            Approx(band_2(1, 1)).epsilon(1e-10));
    REQUIRE(-std::sqrt(3) / 3 *
                (rot(1, 0) * rot(2, 0) + rot(1, 1) * rot(2, 1) -
                 2 * rot(1, 2) * rot(2, 2)) ==
            Approx(band_2(1, 2)).epsilon(1e-10));
    REQUIRE(rot(1, 0) * rot(2, 2) + rot(1, 2) * rot(2, 0) ==
            Approx(band_2(1, 3)).epsilon(1e-10));
    REQUIRE(rot(1, 0) * rot(2, 0) - rot(1, 1) * rot(2, 1) ==
            Approx(band_2(1, 4)).epsilon(1e-10));

    REQUIRE(-std::sqrt(3) / 3 *
                (rot(0, 0) * rot(0, 1) + rot(1, 0) * rot(1, 1) -
                 2 * rot(2, 0) * rot(2, 1)) ==
            Approx(band_2(2, 0)).epsilon(1e-10));
    REQUIRE(-std::sqrt(3) / 3 *
                (rot(0, 1) * rot(0, 2) + rot(1, 1) * rot(1, 2) -
                 2 * rot(2, 1) * rot(2, 2)) ==
            Approx(band_2(2, 1)).epsilon(1e-10));
    REQUIRE(-0.5 * (1 - 3 * rot(2, 2) * rot(2, 2)) ==
            Approx(band_2(2, 2)).epsilon(1e-10));
    REQUIRE(-std::sqrt(3) / 3 *
                (rot(0, 0) * rot(0, 2) + rot(1, 0) * rot(1, 2) -
                 2 * rot(2, 0) * rot(2, 2)) ==
            Approx(band_2(2, 3)).epsilon(1e-10));
    REQUIRE(std::sqrt(3) / 6 *
                (-rot(0, 0) * rot(0, 0) + rot(0, 1) * rot(0, 1) -
                 rot(1, 0) * rot(1, 0) + rot(1, 1) * rot(1, 1) +
                 2 * rot(2, 0) * rot(2, 0) - 2 * rot(2, 1) * rot(2, 1)) ==
            Approx(band_2(2, 4)).epsilon(1e-10));

    REQUIRE(rot(0, 0) * rot(2, 1) + rot(0, 1) * rot(2, 0) ==
            Approx(band_2(3, 0)).epsilon(1e-10));
    REQUIRE(rot(0, 1) * rot(2, 2) + rot(0, 2) * rot(2, 1) ==
            Approx(band_2(3, 1)).epsilon(1e-10));
    REQUIRE(-std::sqrt(3) / 3 *
                (rot(0, 0) * rot(2, 0) + rot(0, 1) * rot(2, 1) -
                 2 * rot(0, 2) * rot(2, 2)) ==
            Approx(band_2(3, 2)).epsilon(1e-10));
    REQUIRE(rot(0, 0) * rot(2, 2) + rot(0, 2) * rot(2, 0) ==
            Approx(band_2(3, 3)).epsilon(1e-10));
    REQUIRE(rot(0, 0) * rot(2, 0) - rot(0, 1) * rot(2, 1) ==
            Approx(band_2(3, 4)).epsilon(1e-10));

    REQUIRE(rot(0, 0) * rot(0, 1) - rot(1, 0) * rot(1, 1) ==
            Approx(band_2(4, 0)).epsilon(1e-10));
    REQUIRE(rot(0, 1) * rot(0, 2) - rot(1, 1) * rot(1, 2) ==
            Approx(band_2(4, 1)).epsilon(1e-10));
    REQUIRE(std::sqrt(3) / 6 *
                (-rot(0, 0) * rot(0, 0) - rot(0, 1) * rot(0, 1) +
                 rot(1, 0) * rot(1, 0) + rot(1, 1) * rot(1, 1) +
                 2 * rot(0, 2) * rot(0, 2) - 2 * rot(1, 2) * rot(1, 2)) ==
            Approx(band_2(4, 2)).epsilon(1e-10));
    REQUIRE(rot(0, 0) * rot(0, 2) - rot(1, 0) * rot(1, 2) ==
            Approx(band_2(4, 3)).epsilon(1e-10));
    REQUIRE(0.5 * (rot(0, 0) * rot(0, 0) - rot(0, 1) * rot(0, 1) -
                   rot(1, 0) * rot(1, 0) + rot(1, 1) * rot(1, 1)) ==
            Approx(band_2(4, 4)).epsilon(1e-10));
  }
}

TEST_CASE("Density matrix transformation preserves population", "[gto][density][transformation]") {
  // Create a simple molecule with d orbitals to test spherical->cartesian transformation
  std::vector<occ::core::Atom> atoms{{6, 0.0, 0.0, 0.0}};  // Carbon atom
  
  // Use a basis set with d functions (spherical)
  occ::qm::AOBasis basis_sph = occ::qm::AOBasis::load(atoms, "cc-pvtz");
  basis_sph.set_pure(true);  // Ensure spherical
  
  // Create a simple density matrix (just identity for testing)
  int nbf_sph = basis_sph.nbf();
  Mat D_sph = Mat::Identity(nbf_sph, nbf_sph) * 0.1;  // Small occupation
  
  // Calculate overlap matrix for spherical basis
  occ::qm::IntegralEngine eng_sph(basis_sph);
  Mat S_sph = eng_sph.one_electron_operator(occ::qm::IntegralEngine::Op::overlap);
  
  // Calculate original Mulliken populations: trace(D * S)
  double pop_original = (D_sph * S_sph).trace();
  
  // Transform density matrix to cartesian
  Mat D_cart = occ::gto::transform_density_matrix_spherical_to_cartesian(basis_sph, D_sph);
  
  // Create cartesian basis and its overlap matrix
  occ::qm::AOBasis basis_cart = basis_sph;
  basis_cart.set_pure(false);
  occ::qm::IntegralEngine eng_cart(basis_cart);
  Mat S_cart = eng_cart.one_electron_operator(occ::qm::IntegralEngine::Op::overlap);
  
  // Calculate transformed Mulliken populations: trace(D_cart * S_cart)
  double pop_transformed = (D_cart * S_cart).trace();
  
  // Population should be preserved
  REQUIRE(pop_transformed == Approx(pop_original).epsilon(1e-12));
  
  // Also test that the density matrix dimensions changed appropriately
  REQUIRE(D_cart.rows() == basis_cart.nbf());
  REQUIRE(D_cart.cols() == basis_cart.nbf());
  REQUIRE(basis_cart.nbf() > basis_sph.nbf());  // Cartesian should have more functions
}
