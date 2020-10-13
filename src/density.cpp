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
    bool dx0 = (dx == 0.0), dy0 = (dy == 0), dz0 = (dz == 0);
    constexpr int LMAX = 5;
    std::array<double, LMAX> bx, by, bz;
    bx[0] = 1; by[0] = 1; bz[0] = 1;
    bx[1] = dx; by[1] = dy; bz[1] = dz;
    for(size_t i = 2; i < LMAX; i++) {
        bx[i] = bx[i-1] * dx;
        by[i] = by[i-1] * dy;
        bz[i] = bz[i-1] * dz;
    }
    for(size_t i = 0; i < shell.nprim(); ++i)
    {
        double alpha = shell.alpha[i];
        double expfac = exp(-alpha * r2);
        for(const auto& contr: shell.contr)
        {
            const double coeff{contr.coeff[i]};
            int l, m, n;
            int offset{0};
            double cexp{coeff * expfac};
            double ax2 = -2 * alpha * dx, ay2 = -2 * alpha * dy, az2 = -2 * alpha * dz;
            double ddx, ddy, ddz;
            FOR_CART(l, m, n, contr.l)
                double xn = bx[l], yn = by[m], zn = bz[n];
                double tmp = cexp * xn * yn * zn;
                result(offset, 0) += tmp;
                if(derivative >= 1) {
                    ddx = cexp * (l * ((l - 1 > 0) ? bx[l - 1] : 1.0) * yn * zn + ax2* xn * yn * zn);
                    ddy = cexp * (m * ((m - 1 > 0) ? by[m - 1] : 1.0) * xn * zn + ay2* xn * yn * zn);
                    ddz = cexp * (n * ((n - 1 > 0) ? bz[n - 1] : 1.0) * xn * yn + az2 * xn * yn * zn);
                    result(offset, 1) += ddx;
                    result(offset, 2) += ddy;
                    result(offset, 3) += ddz;
                }
                offset++;
            END_FOR_CART
        }
    }
}

void eval_shell_points(size_t bf, size_t n_bf, const libint2::Shell &shell, const tonto::MatN4& dists, tonto::Mat& result, int derivative)
{
    size_t n_pt = dists.rows();
    size_t n_prim = shell.nprim();
    size_t n_components = num_components(derivative);
    constexpr size_t LMAX{5};
    for(const auto& contraction: shell.contr) {
        switch (contraction.l) {
        case 0: {
            for(size_t n = 0; n < n_pt; n++) {
                double x = dists(n, 0);
                double y = dists(n, 1);
                double z = dists(n, 2);
                double r2 = dists(n, 3);
                size_t offset = n * n_components;
                double cc_exp_r2 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    cc_exp_r2 += cexp;
                    if (derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;

                result(bf, offset) += cc_exp_r2;
                if ( derivative > 0)
                {
                    result(bf, offset + 1) += g1 * x;
                    result(bf, offset + 2) += g1 * y;
                    result(bf, offset + 3) += g1 * z;
                }
            }
            break;
        }
        case 1: {
            for (size_t n = 0; n < n_pt; n++) {
                double x = dists(n, 0);
                double y = dists(n, 1);
                double z = dists(n, 2);
                double r2 = dists(n, 3);
                size_t offset = n * n_components;
                double g0 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    g0 += cexp;
                    if (derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;
                double g1x = g1 * x;
                double g1y = g1 * y;
                double g1z = g1 * z;
                result(bf, offset) += x * g0;
                result(bf + 1, offset) += y * g0;
                result(bf + 2, offset) += z * g0;
                if(derivative > 0) {
                    result(bf, offset + 1) += g0 + x*g1x;
                    result(bf, offset + 2) += x * g1y;
                    result(bf, offset + 3) += x * g1z;
                    result(bf + 1, offset + 1) += y*g1x;
                    result(bf + 1, offset + 2) += g0 + y * g1y;
                    result(bf + 1, offset + 3) += y * g1z;
                    result(bf + 2, offset + 1) += z * g1x;
                    result(bf + 2, offset + 2) += z * g1y;
                    result(bf + 2, offset + 3) += g0 + z * g1z;
                }
            }
            break;
        }
        default: {
            std::array<double, LMAX> bx, by, bz, gxb, gyb, gzb;
            bx[0] = 1.0; by[0] = 1.0; bz[0] = 1.0;
            for (size_t n = 0; n < n_pt; n++) {
                double x = dists(n, 0);
                double y = dists(n, 1);
                double z = dists(n, 2);
                double r2 = dists(n, 3);
                size_t offset = n * n_components;
                double g0 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    g0 += cexp;
                    if (derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;
                double g1x = g1 * x;
                double g1y = g1 * y;
                double g1z = g1 * z;
                double g1xx = g1x * x;
                double g1yy = g1y * y;
                double g1zz = g1z * z;
                double bxb = x, byb = y, bzb = z;
                bx[1] = x; by[1] = y; bz[1] = z;
                gxb[0] = g1x; gyb[0] = g1y; gzb[0] = g1z;
                gxb[1] = g0 + g1xx; gyb[1] = g0 + g1yy; gzb[1] = g0 + g1zz;

                for(size_t b = 2; b <= contraction.l; b++) {
                    gxb[b] = (b * g0 + g1xx) * bxb;
                    gyb[b] = (b * g0 + g1yy) * byb;
                    gzb[b] = (b * g0 + g1zz) * bzb;
                    bxb *= x; byb *= y; bzb *= z;
                    bx[b] = bxb; by[b] = byb; bz[b] = bzb;
                }
                int L, M, N;
                size_t lmn = 0;
                FOR_CART(L, M, N, contraction.l)
                    bxb = bx[L]; byb = by[M]; bzb = bz[N];
                    double by_bz = byb * bzb;
                    result(bf + lmn, offset) = bxb * by_bz * g0;
                    if(derivative > 0) {
                        result(bf + lmn, offset + 1) = gxb[L] * by_bz;
                        result(bf + lmn, offset + 2) = bxb * gyb[M] * bzb;
                        result(bf + lmn, offset + 3) = bxb * byb * gzb[N];
                    }
                    lmn++;
                END_FOR_CART
            }
        }
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
            eval_shell_points(bf, n_bf, shell, dists, gto_vals, derivative);
        }
    }
    return gto_vals;
}


tonto::Mat evaluate(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::MatN4 &grid_pts,
    int derivative)
{
    int n_components = num_components(derivative);
    tonto::Mat rho = tonto::Mat::Zero(grid_pts.rows(), n_components);
    auto gto_vals = evaluate_gtos(basis, atoms, grid_pts, derivative);
    for(int bf1 = 0; bf1 < gto_vals.rows(); bf1++) {
        const auto& g1 = gto_vals.row(bf1);
        for(int bf2 = 0; bf2 < gto_vals.rows(); bf2++) {
            const auto& g2 = gto_vals.row(bf2);
            const auto Dab = D(bf1, bf2);
            for(size_t i = 0; i < grid_pts.rows(); i++) {
                size_t offset = n_components * i;
                rho(i, 0) += Dab * g1(offset) * g2(offset);
                if(n_components >= 4) {
                    rho(i, 1) += Dab * (g1(offset) * g2(offset + 1) + g2(offset) * g1(offset + 1));
                    rho(i, 2) += Dab * (g1(offset) * g2(offset + 2) + g2(offset) * g1(offset + 2));
                    rho(i, 3) += Dab * (g1(offset) * g2(offset + 3) + g2(offset) * g1(offset + 3));
                }
            }
        }
    }
    return rho;
}
}
