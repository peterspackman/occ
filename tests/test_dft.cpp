#include "catch.hpp"
#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
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
        auto pts = grid.grid_points(0);
        assert(pts.cols() == 1564);
        auto hpts_a = grid.grid_points(1);
        auto hpts_b = grid.grid_points(1);
        assert(hpts_a.cols() == hpts_b.cols());
    }

    std::vector<libint2::Atom> atomsH2{
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.7}
    };
    libint2::BasisSet basisH2("sto-3g", atomsH2);

    tonto::MatRM DH2 = tonto::MatRM::Constant(2, 2, 0.8394261);

    SECTION("Grid rho") {
        tonto::Mat4N pts(4, 3);
        pts << 1, 0, 0, 
               0, 1, 0,
               0, 0, 1, 
               1, 1, 1;
        tonto::Vec expected_rho(3);
        expected_rho << 0.07057915015926258, 0.07057915015926258, 0.2528812862521075;
        fmt::print("Pts:\n{}\n", pts);
        auto rho = tonto::density::evaluate(basisH2, atomsH2, DH2, pts);
        fmt::print("rho:\n{}\n", rho);
        REQUIRE(tonto::util::all_close(expected_rho, rho));
        for(const auto& lmn : tonto::gto::cartesian_ordering(2)) {
            fmt::print("{}: {} {} {}\n", 2, lmn.l, lmn.m, lmn.n);
        }
        fmt::print("{}\n", tonto::gto::Momenta{0, 1, 0}.to_string());
        fmt::print("{}\n", tonto::gto::Momenta{2, 1, 0}.to_string());
    }
}
