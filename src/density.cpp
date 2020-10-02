#include "density.h"
#include <libint2/shell.h>
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <libint2/cartesian.h>
#include <fmt/core.h>
#include <fmt/ostream.h>


namespace tonto::density {

tonto::Vec eval_gto(const libint2::Shell &s1, const double x[3])
{
    tonto::Vec result = tonto::Vec::Zero(s1.cartesian_size());
    fmt::print("Point: {} {} {}\n", x[0], x[1], x[2]);
    double x1 = x[0] - s1.O[0];
    double x2 = x[1] - s1.O[1];
    double x3 = x[2] - s1.O[2];
    double r2 = x1 * x1 + x2 * x2 + x3 * x3;
    double expfac = 0.0;
    for(size_t i = 0; i < s1.nprim(); ++i)
    {
        double alpha = s1.alpha[i];
        double e = exp(-alpha * r2);
        for(const auto& contr: s1.contr)
        {
            const double coeff = contr.coeff[i];
            int l = 0, m = 0, n = 0;
            int offset = 0;
            FOR_CART(l, m, n, contr.l);
                double tmp = coeff * e;
                double xfac = 1.0, yfac = 1.0, zfac = 1.0;
                for(int ii = 0; ii < l; ii++) xfac *= x1;
                for(int ii = 0; ii < l; ii++) yfac *= x2;
                for(int ii = 0; ii < l; ii++) zfac *= x3;
                result(offset) += xfac * yfac * zfac * tmp;
                offset++;
            END_FOR_CART
        }
    }
    return result;
}

tonto::Vec evaluate(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::Mat4N &grid_pts)
{
    tonto::Vec rho = tonto::Vec::Zero(grid_pts.cols());
    const size_t natoms = atoms.size();
    const size_t nshells = basis.size();
    tonto::Mat rho_s = tonto::Mat::Zero(D.rows(), grid_pts.cols());
    auto shell2bf = basis.shell2bf();
    for(int i = 0; i < nshells; i++)
    {
        const auto& s1 = basis[i];
        size_t bf = shell2bf[i];
        size_t n_bf = s1.size();
        for(size_t pt = 0; pt < grid_pts.cols(); pt++)
        {
            rho_s.block(bf, pt, n_bf, 1) += eval_gto(s1, grid_pts.col(pt).data());
        }
    }
    fmt::print("rho_s\n{}\n", rho_s);
    for(int bf1 = 0; bf1 < rho_s.rows(); bf1++) {
        for(int bf2 = bf1; bf2 < rho_s.rows(); bf2++) {
            if(bf1 == bf2) rho.array() += D(bf1, bf2) * rho_s.row(bf1).array() * rho_s.row(bf2).array();
            else rho.array() += 2 * D(bf1, bf2) * rho_s.row(bf1).array() * rho_s.row(bf2).array();
        }
    }
    return rho;
}
}
