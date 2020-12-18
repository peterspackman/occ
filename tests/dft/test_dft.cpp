#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include <tonto/dft/dft.h>
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <libint2.hpp>
#include <vector>
#include <iostream>
#include <tonto/qm/hf.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <tonto/qm/density.h>
#include <tonto/qm/gto.h>
#include <tonto/core/util.h>

TEST_CASE("Density Functional", "[lda]") {
    tonto::dft::DensityFunctional lda("xc_lda_x");
    tonto::dft::DensityFunctional lda_u("xc_lda_x", true);
    tonto::dft::DensityFunctional::Params params(5, tonto::dft::DensityFunctional::Family::LDA, tonto::qm::SpinorbitalKind::Restricted);
    REQUIRE(params.rho.size() == 5);
    params.rho = tonto::Vec::LinSpaced(5, 0, 1);
    fmt::print("Rho:\n{}\n", params.rho);
    auto res = lda.evaluate(params);
    fmt::print("exc:\n{}\nvrho\n{}\n", res.exc, res.vrho);

    tonto::dft::DensityFunctional::Params params_u(5, tonto::dft::DensityFunctional::Family::LDA, tonto::qm::SpinorbitalKind::Unrestricted);
    REQUIRE(params_u.rho.size() == 10);
    for(size_t i = 0; i < params.rho.rows(); i++) {
        params_u.rho(2*i) = params.rho(i);
        params_u.rho(2*i + 1) = params.rho(i);
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
    using tonto::util::all_close;
    tonto::dft::DensityFunctional gga("xc_gga_x_pbe");
    tonto::dft::DensityFunctional gga_u("xc_gga_x_pbe", true);
    tonto::Mat rho(8, 4);
    rho << 3.009700173159170558e-02, -6.373586084157208120e-02, -0.000000000000000000e+00, 1.865655330995227498e-02,
           3.009700173159170558e-02, -0.000000000000000000e+00, -6.373586084157208120e-02, 1.865655330995227498e-02,
           1.508591677559790178e-01, -0.000000000000000000e+00, -0.000000000000000000e+00, 1.303514966003518905e-01,
           1.775122853194434636e-01, -0.000000000000000000e+00, -0.000000000000000000e+00, 7.842601108050306635e-02,
           3.009700173159170558e-02, -6.373586084157208120e-02, -0.000000000000000000e+00, 1.865655330995227498e-02,
           3.009700173159170558e-02, -0.000000000000000000e+00, -6.373586084157208120e-02, 1.865655330995227498e-02,
           1.508591677559790178e-01, -0.000000000000000000e+00, -0.000000000000000000e+00, 1.303514966003518905e-01,
           1.775122853194434636e-01, -0.000000000000000000e+00, -0.000000000000000000e+00, 7.842601108050306635e-02;
    tonto::dft::DensityFunctional::Params params(4, tonto::dft::DensityFunctional::Family::GGA, tonto::qm::SpinorbitalKind::Restricted);
    REQUIRE(params.rho.size() == 4);
    params.rho.col(0) = rho.alpha().col(0);
    auto rho_a = rho.alpha(), rho_b = rho.beta();
    fmt::print("Rho_a:\n{}\n", rho_a);
    params.sigma.col(0) = rho_a.col(1).array() * rho_a.col(1).array() + rho_a.col(2).array() * rho_a.col(2).array() + rho_a.col(3).array() * rho_a.col(3).array();
    fmt::print("GGA-----\nRho:\n{}\n\nsigma\n{}\n", params.rho, params.sigma);
    auto res = gga.evaluate(params);
    fmt::print("exc:\n{}\nvrho\n{}\nvsigma\n{}\n", res.exc, res.vrho, res.vsigma);

    tonto::Vec expected_exc(4);
    expected_exc << -0.27851489, -0.27851489, -0.39899553, -0.41654061;
    REQUIRE(all_close(expected_exc, res.exc, 1e-6));

    tonto::dft::DensityFunctional::Params params_u(4, tonto::dft::DensityFunctional::Family::GGA, tonto::qm::SpinorbitalKind::Unrestricted);
    REQUIRE(params_u.rho.size() == 8);
    params_u.rho.col(0) = rho_a.col(0);
    params_u.rho.col(1) = rho_b.col(0);
    params_u.sigma.col(0) = rho_a.col(1).array() * rho_a.col(1).array() + rho_a.col(2).array() * rho_a.col(2).array() + rho_a.col(3).array() * rho_a.col(3).array();
    params_u.sigma.col(1) = rho_a.col(1).array() * rho_b.col(1).array() + rho_a.col(2).array() * rho_b.col(2).array() + rho_a.col(3).array() * rho_b.col(3).array();
    params_u.sigma.col(2) = rho_b.col(1).array() * rho_b.col(1).array() + rho_b.col(2).array() * rho_b.col(2).array() + rho_b.col(3).array() * rho_b.col(3).array();
    fmt::print("rho_xyz\n{}\n{}\n", rho_a.block(0, 1, 4, 3), rho_b.block(0, 1, 4, 3));
    fmt::print("\n\nRho interleaved:\n{}\nsigma\n{}\n", params_u.rho, params_u.sigma);
    auto res1 = gga_u.evaluate(params_u);
    fmt::print("exc:\n{}\nvrho\n{}\nvsigma\n{}\n", res1.exc, res1.vrho, res1.vsigma);
   // assert(all_close(expected_exc, res1.exc, 1e-6));
}

