#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <libint2.hpp>
#include <vector>
#include <iostream>
#include "hf.h"
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "density.h"
#include "gto.h"
#include "util.h"

TEST_CASE("Water DFT grid", "[dft]")
{

    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    libint2::BasisSet basis("3-21G", atoms);

    SECTION("Grid generation") {
        tonto::dft::DFTGrid grid(basis, atoms);
        grid.set_min_angular_points(12);
        grid.set_max_angular_points(20);
        auto [pts, weights] = grid.grid_points(0);
        assert(pts.cols() == 1564);
        auto [hpts_a, weights_a] = grid.grid_points(1);
        auto [hpts_b, weights_b] = grid.grid_points(1);
        assert(hpts_a.cols() == hpts_b.cols());
    }

    SECTION("Density Functional") {
        tonto::dft::DensityFunctional lda("xc_lda_x");
        tonto::dft::DensityFunctional lda_u("xc_lda_x", true);
        tonto::dft::DensityFunctional::Params params(5, tonto::dft::DensityFunctional::Family::LDA, tonto::qm::SpinorbitalKind::Restricted);
        assert(params.rho.size() == 5);
        params.rho = tonto::Vec::LinSpaced(5, 0, 1);
        fmt::print("Rho:\n{}\n", params.rho);
        auto res = lda.evaluate(params);
        fmt::print("exc:\n{}\nvrho\n{}\n", res.exc, res.vrho);

        tonto::dft::DensityFunctional::Params params_u(5, tonto::dft::DensityFunctional::Family::LDA, tonto::qm::SpinorbitalKind::Unrestricted);
        assert(params_u.rho.size() == 10);
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

}
