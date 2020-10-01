#include "catch.hpp"
#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <vector>
#include <iostream>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "hf.h"

double eval_gto(const libint2::Shell &s1, const double x[3])
{
    double result = 0.0;
    fmt::print("Point: {} {} {}\n", x[0], x[1], x[2]);
    double x1 = x[0] - s1.O[0];
    double x2 = x[1] - s1.O[1];
    double x3 = x[2] - s1.O[2];
    double r2 = x1 * x1 + x2 * x2 + x3 * x3;
    for(size_t i = 0; i < s1.alpha.size(); ++i)
    {
        double alpha = s1.alpha[i];
        double e = exp(-alpha * r2);
        for(const auto& contr: s1.contr)
        {
            const double coeff = contr.coeff[i];
            if(contr.l > 0) fmt::print("l > 0 not done\n");
            result += coeff * e;
        }
    }
    return result;
}

tonto::Vec evaluate_density(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::Mat4N &grid_pts)
{
    tonto::Vec rho = tonto::Vec::Zero(grid_pts.cols());
    const size_t natoms = atoms.size();
    const size_t nshells = basis.size();
    tonto::Mat rho_s = tonto::Mat::Zero(nshells, grid_pts.cols());
    int evaluated = -1;
    for(int i = 0; i < nshells; i++)
    {
        const auto& s1 = basis[i];
        if(i > evaluated) {
            fmt::print("evaluating rho_{}\n", i);
            for(size_t pt = 0; pt < grid_pts.cols(); pt++)
            {
                rho_s(i, pt) += eval_gto(s1, grid_pts.col(pt).data());
            }
            evaluated = i;
        }
        fmt::print("rho_{}:\n{}\n", i, rho_s.row(i));
        for(int j = i; j < nshells; j++)
        {
            if (i == j) {
                rho.array() += D(i, i) * rho_s.row(i).array() * rho_s.row(i).array();
            }
            else {
                if(j > evaluated) {
                    fmt::print("evaluating rho_{}\n", j);
                    const auto& s2 = basis[j];
                    for(size_t pt = 0; pt < grid_pts.cols(); pt++)
                    {
                        rho_s(j, pt) += eval_gto(s2, grid_pts.col(pt).data());
                    }
                    evaluated = j;
                }
                fmt::print("rho_{}:\n{}\n", j, rho_s.row(j));
                rho.array() += 2 * D(i, j) * rho_s.row(i).array() * rho_s.row(j).array();
            }
        }
    }
    return rho;
}

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
        fmt::print("Pts:\n{}\n", pts);
        auto rho = evaluate_density(basisH2, atomsH2, DH2, pts);
        fmt::print("rho:\n{}\n", rho);
    }
}
