#include "density.h"
#include <libint2/shell.h>
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <libint2/cartesian.h>

namespace tonto::density {

tonto::Vec eval_gto(const libint2::Shell &s1, const double x[3])
{
    tonto::Vec result = tonto::Vec::Zero(s1.size());
    double dx = x[0] - s1.O[0];
    double dy = x[1] - s1.O[1];
    double dz = x[2] - s1.O[2];
    double r2 = dx * dx + dy * dy + dz * dz;
    for(size_t i = 0; i < s1.nprim(); ++i)
    {
        double alpha = s1.alpha[i];
        double expfac = exp(-alpha * r2);
        for(const auto& contr: s1.contr)
        {
            const double coeff = contr.coeff[i];
            int l = 0, m = 0, n = 0;
            int offset = 0;
            FOR_CART(l, m, n, contr.l);
                double tmp = coeff * expfac;
                for(int px = 0; px < l; px++) tmp *= dx;
                for(int py = 0; py < m; py++) tmp *= dy;
                for(int pz = 0; pz < n; pz++) tmp *= dz;
                result(offset) += tmp;
                offset++;
            END_FOR_CART
        }
    }
    return result;
}

tonto::Mat evaluate_gtos(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::Mat4N &grid_pts)
{
    const auto nshells = basis.size();
    tonto::Mat gto_vals = tonto::Mat::Zero(basis.nbf(), grid_pts.cols());
    auto shell2bf = basis.shell2bf();
    for(auto i = 0; i < nshells; i++)
    {
        const auto& s1 = basis[i];
        size_t bf = shell2bf[i];
        size_t n_bf = s1.size();
        for(auto pt = 0; pt < grid_pts.cols(); pt++)
        {
            gto_vals.block(bf, pt, n_bf, 1) += eval_gto(s1, grid_pts.col(pt).data());
        }
    }
    return gto_vals;
}

tonto::Vec evaluate(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::Mat4N &grid_pts)
{
    tonto::Vec rho = tonto::Vec::Zero(grid_pts.cols());
    auto gto_vals = evaluate_gtos(basis, atoms, grid_pts);
    for(int bf1 = 0; bf1 < gto_vals.rows(); bf1++) {
        for(int bf2 = bf1; bf2 < gto_vals.rows(); bf2++) {
            if(bf1 == bf2) rho.array() += D(bf1, bf2) * gto_vals.row(bf1).array() * gto_vals.row(bf2).array();
            else rho.array() += 2 * D(bf1, bf2) * gto_vals.row(bf1).array() * gto_vals.row(bf2).array();
        }
    }
    return rho;
}
}
