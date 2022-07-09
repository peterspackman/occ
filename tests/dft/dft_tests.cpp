#include "catch.hpp"
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/atom.h>
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/dft/grid.h>
#include <occ/dft/lebedev.h>
#include <occ/dft/seminumerical_exchange.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/qm/hf.h>
#include <occ/qm/shell.h>
#include <occ/qm/scf.h>
#include <vector>

// DFT

TEST_CASE("Density Functional", "[lda]") {
    occ::dft::DensityFunctional lda("xc_lda_x");
    occ::dft::DensityFunctional lda_u("xc_lda_x", true);
    occ::dft::DensityFunctional::Params params(
        5, occ::dft::DensityFunctional::Family::LDA,
        occ::qm::SpinorbitalKind::Restricted);
    REQUIRE(params.rho.size() == 5);
    params.rho = occ::Vec::LinSpaced(5, 0, 1);
    fmt::print("Rho:\n{}\n", params.rho);
    auto res = lda.evaluate(params);
    fmt::print("exc:\n{}\nvrho\n{}\n", res.exc, res.vrho);

    occ::dft::DensityFunctional::Params params_u(
        5, occ::dft::DensityFunctional::Family::LDA,
        occ::qm::SpinorbitalKind::Unrestricted);
    REQUIRE(params_u.rho.size() == 10);
    for (size_t i = 0; i < params.rho.rows(); i++) {
        params_u.rho(2 * i) = params.rho(i);
        params_u.rho(2 * i + 1) = params.rho(i);
    }
    fmt::print("Rho interleaved:\n{}\n", params_u.rho);
    auto res1 = lda_u.evaluate(params_u);
    fmt::print("exc:\n{}\nvrho\n{}\n", res1.exc, res1.vrho);

    params_u.rho = params.rho.replicate(2, 1);
    fmt::print("Rho block:\n{}\n", params_u.rho);
    auto res2 = lda_u.evaluate(params_u);
    fmt::print("exc:\n{}\nvrho\n{}\n", res2.exc, res2.vrho);
}

TEST_CASE("gga", "[gga]") {
    using occ::util::all_close;
    namespace block = occ::qm::block;
    occ::dft::DensityFunctional gga("xc_gga_x_pbe");
    occ::dft::DensityFunctional gga_u("xc_gga_x_pbe", true);
    occ::Mat rho(8, 4);
    rho << 3.009700173159170558e-02, -6.373586084157208120e-02,
        -0.000000000000000000e+00, 1.865655330995227498e-02,
        3.009700173159170558e-02, -0.000000000000000000e+00,
        -6.373586084157208120e-02, 1.865655330995227498e-02,
        1.508591677559790178e-01, -0.000000000000000000e+00,
        -0.000000000000000000e+00, 1.303514966003518905e-01,
        1.775122853194434636e-01, -0.000000000000000000e+00,
        -0.000000000000000000e+00, 7.842601108050306635e-02,
        3.009700173159170558e-02, -6.373586084157208120e-02,
        -0.000000000000000000e+00, 1.865655330995227498e-02,
        3.009700173159170558e-02, -0.000000000000000000e+00,
        -6.373586084157208120e-02, 1.865655330995227498e-02,
        1.508591677559790178e-01, -0.000000000000000000e+00,
        -0.000000000000000000e+00, 1.303514966003518905e-01,
        1.775122853194434636e-01, -0.000000000000000000e+00,
        -0.000000000000000000e+00, 7.842601108050306635e-02;
    occ::dft::DensityFunctional::Params params(
        4, occ::dft::DensityFunctional::Family::GGA,
        occ::qm::SpinorbitalKind::Restricted);
    REQUIRE(params.rho.size() == 4);
    params.rho.col(0) = block::a(rho).col(0);
    auto rho_a = block::a(rho), rho_b = block::b(rho);
    fmt::print("Rho_a:\n{}\n", rho_a);
    params.sigma.col(0) = rho_a.col(1).array() * rho_a.col(1).array() +
                          rho_a.col(2).array() * rho_a.col(2).array() +
                          rho_a.col(3).array() * rho_a.col(3).array();
    fmt::print("GGA-----\nRho:\n{}\n\nsigma\n{}\n", params.rho, params.sigma);
    auto res = gga.evaluate(params);
    fmt::print("exc:\n{}\nvrho\n{}\nvsigma\n{}\n", res.exc, res.vrho,
               res.vsigma);

    occ::Vec expected_exc(4);
    expected_exc << -0.27851489, -0.27851489, -0.39899553, -0.41654061;
    REQUIRE(all_close(expected_exc, res.exc, 1e-6));

    occ::dft::DensityFunctional::Params params_u(
        4, occ::dft::DensityFunctional::Family::GGA,
        occ::qm::SpinorbitalKind::Unrestricted);
    REQUIRE(params_u.rho.size() == 8);
    params_u.rho.col(0) = rho_a.col(0);
    params_u.rho.col(1) = rho_b.col(0);
    params_u.sigma.col(0) = rho_a.col(1).array() * rho_a.col(1).array() +
                            rho_a.col(2).array() * rho_a.col(2).array() +
                            rho_a.col(3).array() * rho_a.col(3).array();
    params_u.sigma.col(1) = rho_a.col(1).array() * rho_b.col(1).array() +
                            rho_a.col(2).array() * rho_b.col(2).array() +
                            rho_a.col(3).array() * rho_b.col(3).array();
    params_u.sigma.col(2) = rho_b.col(1).array() * rho_b.col(1).array() +
                            rho_b.col(2).array() * rho_b.col(2).array() +
                            rho_b.col(3).array() * rho_b.col(3).array();
    fmt::print("rho_xyz\n{}\n{}\n", rho_a.block(0, 1, 4, 3),
               rho_b.block(0, 1, 4, 3));
    fmt::print("\n\nRho interleaved:\n{}\nsigma\n{}\n", params_u.rho,
               params_u.sigma);
    auto res1 = gga_u.evaluate(params_u);
    fmt::print("exc:\n{}\nvrho\n{}\nvsigma\n{}\n", res1.exc, res1.vrho,
               res1.vsigma);
    // assert(all_close(expected_exc, res1.exc, 1e-6));
}

// Grid
using occ::util::all_close;

TEST_CASE("lebedev", "[grid]") {
    auto grid = occ::grid::lebedev(110);
    fmt::print("grid:\n{}\n", grid);
    REQUIRE(grid.rows() == 110);
    REQUIRE(grid.cols() == 4);

    /*
    BENCHMARK("Lebedev 86") {
        return occ::grid::lebedev(86);
    };0

    BENCHMARK("Lebedev 590") {
        return occ::grid::lebedev(590);
    };

    BENCHMARK("Lebedev 5810") {
        return occ::grid::lebedev(5810);
    };
    */
}

TEST_CASE("Becke radial grid", "[radial]") {
    auto radial = occ::dft::generate_becke_radial_grid(3, 0.6614041435977716);
    occ::Vec3 expected_pts{9.21217133, 0.66140414, 0.04748668};
    occ::Vec3 expected_weights{77.17570606, 1.3852416, 0.39782349};
    fmt::print("Becke radial grid:\n{} == {}\n{} == {}\n",
               radial.points.transpose(), expected_pts.transpose(),
               radial.weights.transpose(), expected_weights.transpose());

    REQUIRE(all_close(radial.points, expected_pts, 1e-5));
    REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
    /*
    BENCHMARK("Becke radial 10") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 50") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 80") {
        return occ::dft::generate_becke_radial_grid(10);
    };
    */
}

TEST_CASE("Gauss-Chebyshev radial grid", "[radial]") {
    auto radial = occ::dft::generate_gauss_chebyshev_radial_grid(3);
    occ::Vec3 expected_pts{8.66025404e-01, 6.123234e-17, -8.66025404e-01};
    occ::Vec3 expected_weights{1.04719755, 1.04719755, 1.04719755};
    fmt::print("Gauss-Chebyshev radial grid:\n{} == {}\n{} == {}\n",
               radial.points.transpose(), expected_pts.transpose(),
               radial.weights.transpose(), expected_weights.transpose());

    REQUIRE(all_close(radial.points, expected_pts, 1e-5));
    REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
}

TEST_CASE("Mura-Knowles radial grid", "[radial]") {
    auto radial = occ::dft::generate_mura_knowles_radial_grid(3, 1);
    occ::Vec3 expected_pts{0.02412997, 0.69436324, 4.49497829};
    occ::Vec3 expected_weights{0.14511628, 1.48571429, 8.57142857};
    fmt::print("Mura-Knowles radial grid:\n{} == {}\n{} == {}\n",
               radial.points.transpose(), expected_pts.transpose(),
               radial.weights.transpose(), expected_weights.transpose());

    REQUIRE(all_close(radial.points, expected_pts, 1e-5));
    REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
    /*
    BENCHMARK("Becke radial 10") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 50") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 80") {
        return occ::dft::generate_becke_radial_grid(10);
    };
    */
}

TEST_CASE("Treutler-Alrichs radial grid", "[radial]") {
    auto radial = occ::dft::generate_treutler_alrichs_radial_grid(3);
    occ::Vec3 expected_pts{0.10934791, 1, 3.82014324};
    occ::Vec3 expected_weights{0.34905607, 1.60432893, 4.51614622};
    fmt::print("Treutler-Alrichs radial grid:\n{} == {}\n{} == {}\n",
               radial.points.transpose(), expected_pts.transpose(),
               radial.weights.transpose(), expected_weights.transpose());

    REQUIRE(all_close(radial.points, expected_pts, 1e-5));
    REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
    /*
    BENCHMARK("Becke radial 10") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 50") {
        return occ::dft::generate_becke_radial_grid(10);
    };

    BENCHMARK("Becke radial 80") {
        return occ::dft::generate_becke_radial_grid(10);
    };
    */
}

TEST_CASE("atom grid", "[grid]") {
    auto atom = occ::dft::generate_atom_grid(1, 302, 50);
    // fmt::print("Atom grid:\n{}\n", atom.points.transpose());
    fmt::print("Sum weights\n{}\n", atom.weights.array().sum());
    fmt::print("Shape: {} {}\n", atom.points.rows(), atom.points.cols());
    fmt::print("Max weight\n{}\n", atom.weights.maxCoeff());

    /*
    BENCHMARK("Atom Carbon") {
        return occ::dft::generate_atom_grid(6);
    };

    BENCHMARK("Atom Carbon 590") {
        return occ::dft::generate_atom_grid(6, 590);
    };
*/
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                       {1, 0.0, 0.0, 1.39839733}};
    auto basis = occ::qm::AOBasis::load(atoms, "sto-3g");
    occ::dft::MolecularGrid mgrid(basis);

    auto grid = mgrid.generate_partitioned_atom_grid(0);
    // fmt::print("{}\n{}\n", grid.points, grid.atomic_number);
}

// Seminumerical Exchange

TEST_CASE("Water DFT", "[scf]") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    auto basis = occ::qm::AOBasis::load(atoms, "def2-tzvp");
    basis.set_pure(false);
    auto hf = occ::hf::HartreeFock(basis);
    occ::scf::SCF<occ::hf::HartreeFock, occ::qm::SpinorbitalKind::Restricted>
        scf(hf);
    double e = scf.compute_scf_energy();

    occ::dft::AtomGridSettings settings;
    settings.max_angular_points = 302;
    settings.radial_precision = 1e-12;
    fmt::print("Construct\n");
    occ::dft::cosx::SemiNumericalExchange sgx(basis, settings);
    fmt::print("Construct done\n");

    occ::timing::StopWatch<2> sw;
    sw.start(0);
    fmt::print("Compute K SGX\n");
    occ::Mat result =
        sgx.compute_K(occ::qm::SpinorbitalKind::Restricted, scf.mo);
    sw.stop(0);
    fmt::print("Compute K SGX done\n");
    occ::Mat Jexact, Kexact;
    sw.start(1);
    std::tie(Jexact, Kexact) = hf.compute_JK(
        occ::qm::SpinorbitalKind::Restricted, scf.mo, 1e-12, occ::Mat());
    sw.stop(1);
    int i, j;
    fmt::print("K - Kexact: {:12.8f}\n",
               (result - Kexact).array().cwiseAbs().maxCoeff(&i, &j));
    /*
    const auto &bf2shell = sgx.engine().aobasis().bf_to_shell();
    int s1 = bf2shell[i];
    const auto &sh1 = sgx.engine().aobasis()[s1];
    int s2 = bf2shell[i];
    const auto &sh2 = sgx.engine().aobasis()[s2];

    fmt::print("Shells:\n");
    std::cout << s1 << sh1 << '\n' << s2 << sh2 << '\n';
    int bf1 = sgx.engine().aobasis().first_bf()[s1];
    int bf2 = sgx.engine().aobasis().first_bf()[s2];

    fmt::print("K SGX\n{}\n", result.block(bf1, bf2, sh1.size(), sh2.size()));
    fmt::print("K exact\n{}\n", Kexact.block(bf1, bf2, sh1.size(), sh2.size()));
    */
    fmt::print("Speedup = ({} vs. {}) {:.3f} times\n", sw.read(0), sw.read(1),
               sw.read(1) / sw.read(0));

    occ::timing::print_timings();
}
