#include "density.h"
#include <libint2/shell.h>
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <libint2/cartesian.h>
#include <fmt/core.h>

namespace tonto::density {

void eval_shell(const libint2::Shell &shell, const Eigen::Ref<const tonto::Mat>& dists, Eigen::Ref<tonto::Mat>& result)
{
    double dx = dists(0, 0); double dy = dists(0, 1); double dz = dists(0, 2); double r2 = dists(0, 3);
    bool do_gradients = result.cols() > 1;
    for(size_t i = 0; i < shell.nprim(); ++i)
    {
        double alpha = shell.alpha[i];
        double expfac = exp(-alpha * r2);
        for(const auto& contr: shell.contr)
        {
            const double coeff{contr.coeff[i]};
            int l, m, n;
            int offset{0};
            FOR_CART(l, m, n, contr.l)
                double tmp{coeff * expfac};
                for(int px = 0; px < l; px++) tmp *= dx;
                for(int py = 0; py < m; py++) tmp *= dy;
                for(int pz = 0; pz < n; pz++) tmp *= dz;
                result(offset, 0) += tmp;
                offset++;
            END_FOR_CART
            /*
             * df/dx = (l / x - 2 \alpha x) * f(x)
             * df2/dx2 = ((l - 1)/x - 2 \alpha x) * f(x)
             */
        }
    }
}

tonto::Mat evaluate_gtos(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatN4 &grid_pts)
{
    const auto natoms = atoms.size();
    const auto npts = grid_pts.rows();
    tonto::Mat gto_vals = tonto::Mat::Zero(basis.nbf(), grid_pts.rows());
    auto shell2bf = basis.shell2bf();
    auto atom2shell = basis.atom2shell(atoms);

    for(size_t i = 0; i < natoms; i++)
    {
        const auto& atom = atoms[i];
        tonto::MatN4 dists(npts, 4);
        for(auto pt = 0; pt < npts; pt++) {
            double dx = grid_pts(pt, 0) - atom.x;
            double dy = grid_pts(pt, 1) - atom.y;
            double dz = grid_pts(pt, 2) - atom.z;
            double r2 = dx * dx + dy * dy + dz * dz;
            dists(pt, 0) = dx; dists(pt, 1) = dy; dists(pt, 2) = dz; dists(pt, 3) = r2;
        }
        for(const auto& shell_idx: atom2shell[i]) {
            const auto& shell = basis[shell_idx];
            size_t bf = shell2bf[shell_idx];
            size_t n_bf = shell.size();
            for(auto pt = 0; pt < grid_pts.rows(); pt++)
            {
                Eigen::Ref<tonto::Mat> block(gto_vals.block(bf, pt, n_bf, 1));
                eval_shell(shell, dists.row(pt), block);
            }
        }
    }
    return gto_vals;
}

tonto::Vec evaluate(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::MatN4 &grid_pts)
{
    tonto::Vec rho = tonto::Vec::Zero(grid_pts.rows());
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
