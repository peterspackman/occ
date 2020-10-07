#include "density.h"
#include <libint2/shell.h>
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <libint2/cartesian.h>

namespace tonto::density {

void eval_shell_S(const libint2::Shell &shell, const Eigen::Ref<const tonto::Mat>& dists, Eigen::Ref<tonto::Mat>& result, int derivative)
{
    double dx = dists(0, 0); double dy = dists(0, 1); double dz = dists(0, 2); double r2 = dists(0, 3);
    for(size_t i = 0; i < shell.nprim(); ++i)
    {
        double alpha = shell.alpha[i];
        double expfac = exp(-alpha * r2);
        for(const auto& contr: shell.contr)
        {
            const double coeff{contr.coeff[i]};
            double tmp = coeff * expfac;
            double a2;
            double ddx, pdx, ddy, pdy, ddz, pdz;
            result(0, 0) += tmp;
            if(derivative >=1) {
                a2 = 2 * alpha;
                pdx = (a2 * dx);
                pdy = (a2 * dy);
                pdz = (a2 * dz);
                ddx = tmp * pdx;
                ddy = tmp * pdy;
                ddz = tmp * pdz;
                result(0, 1) += ddx;
                result(0, 2) += ddy;
                result(0, 3) += ddz;
            }
            if(derivative >= 2) {
                result(0, 4) += ddx * pdx - a2 * tmp;
                result(0, 5) += ddx * pdy;
                result(0, 6) += ddx * pdz;
                result(0, 7) += ddy * pdy - a2 * tmp;
                result(0, 8) += ddy * pdz;
                result(0, 9) += ddz * pdz - a2 * tmp;
            }
        }
    }
}


void eval_shell(const libint2::Shell &shell, const Eigen::Ref<const tonto::Mat>& dists, Eigen::Ref<tonto::Mat>& result, int derivative)
{
    double dx = dists(0, 0); double dy = dists(0, 1); double dz = dists(0, 2); double r2 = dists(0, 3);
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
                double ddx, ddy, ddz, pdx, pdy, pdz;
                if(derivative >=1) {
                    pdx = (l / dx - 2 * alpha * dx);
                    pdy = (m / dy - 2 * alpha * dy);
                    pdz = (n / dz - 2 * alpha * dz);
                    ddx = tmp * pdx;
                    ddy = tmp * pdy;
                    ddz = tmp * pdz;
                    result(offset, 1) += ddx;
                    result(offset, 2) += ddy;
                    result(offset, 3) += ddz;
                }
                if(derivative >= 2) {
                    double pdx2 = (-l / (dx * dx) - 2 * alpha);
                    double pdy2 = (-m / (dy * dy) - 2 * alpha);
                    double pdz2 = (-n / (dz * dz) - 2 * alpha);
                    result(offset, 4) += ddx * pdx + tmp * pdx2;
                    result(offset, 5) += ddx * pdy;
                    result(offset, 6) += ddx * pdz;
                    result(offset, 7) += ddy * pdy + tmp * pdy2;
                    result(offset, 8) += ddy * pdz;
                    result(offset, 9) += ddz * pdz + tmp * pdz2;
                }
                offset++;
            END_FOR_CART
        }
    }
}

tonto::Mat evaluate_gtos(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatN4 &grid_pts, int derivative)
{
    const auto natoms = atoms.size();
    const auto npts = grid_pts.rows();
    int n_components = num_components(derivative);
    tonto::Mat gto_vals = tonto::Mat::Zero(basis.nbf(), grid_pts.rows() * n_components);
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
                Eigen::Ref<tonto::Mat> block(gto_vals.block(bf, pt * n_components, n_bf, n_components));
                eval_shell(shell, dists.row(pt), block, derivative);
            }
        }
    }
    return gto_vals;
}

tonto::Vec evaluate(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::MatN4 &grid_pts,
    int derivative)
{
    int n_components = num_components(derivative);
    tonto::Vec rho = tonto::Vec::Zero(grid_pts.rows()* n_components);
    auto gto_vals = evaluate_gtos(basis, atoms, grid_pts, derivative);
    for(int bf1 = 0; bf1 < gto_vals.rows(); bf1++) {
        for(int bf2 = bf1; bf2 < gto_vals.rows(); bf2++) {
            if(bf1 == bf2) rho.array() += D(bf1, bf2) * gto_vals.row(bf1).array() * gto_vals.row(bf2).array();
            else rho.array() += 2 * D(bf1, bf2) * gto_vals.row(bf1).array() * gto_vals.row(bf2).array();
        }
    }
    return rho;
}
}
