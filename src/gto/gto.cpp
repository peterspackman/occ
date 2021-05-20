#include <occ/gto/gto.h>

namespace occ::gto {

namespace impl {

void add_shell_contribution(
    size_t bf, const libint2::Shell &shell, const Eigen::Ref<const occ::Mat>& dists,
    GTOValues<0>& result,
    const Eigen::Ref<const occ::MaskArray>& mask)
{
    size_t n_pt = dists.cols();
    size_t n_prim = shell.nprim();
    constexpr size_t LMAX{8};
    int l = shell.contr[0].l;
    Eigen::Map<const occ::Vec> coeffs(shell.contr[0].coeff.data(), n_prim, 1),
        alpha(shell.alpha.data(), n_prim, 1);
    occ::Vec nalpha = - alpha;;
    occ::Vec tmp(n_prim);
    switch (l) {
        case 0: {
            occ::timing::start(occ::timing::category::gto_s);
            for(size_t pt = 0; pt < n_pt; pt++) {
                if(!mask(pt)) continue;
                double r2 = dists(3, pt);
                double g0{0.0};
                tmp = Eigen::exp(nalpha.array() * r2);

                for(int i = 0; i < n_prim; i++)
                {
                    double cexp = coeffs(i) * tmp(i);
                    g0 += cexp;
                }
                result.phi(pt, bf) += g0;
            }
            occ::timing::stop(occ::timing::category::gto_s);
            break;
        }
        case 1: {
            occ::timing::start(occ::timing::category::gto_p);
            for (size_t pt = 0; pt < n_pt; pt++) {
                if(!mask(pt)) continue;
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double g0{0.0};
                tmp = Eigen::exp(nalpha.array() * r2);

                for(int i = 0; i < n_prim; i++)
                {
                    double cexp = coeffs(i) * tmp(i);
                    g0 += cexp;
                }


                result.phi(pt, bf) += x * g0;
                result.phi(pt, bf + 1) += y * g0;
                result.phi(pt, bf + 2) += z * g0;
            }
            occ::timing::stop(occ::timing::category::gto_p);
            break;
        }
        default: {
            occ::timing::start(occ::timing::category::gto_gen);
            std::array<double, LMAX> bx, by, bz;
            bx[0] = 1.0; by[0] = 1.0; bz[0] = 1.0;
            for (size_t pt = 0; pt < n_pt; pt++) {
                if(!mask(pt)) continue;
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double g0{0.0};
                tmp = Eigen::exp(nalpha.array() * r2);

                for(int i = 0; i < n_prim; i++)
                {
                    double cexp = coeffs(i) * tmp(i);
                    g0 += cexp;
                }

                double bxb = x, byb = y, bzb = z;
                bx[1] = x; by[1] = y; bz[1] = z;

                for(size_t b = 2; b <= l; b++) {
                    bxb *= x; byb *= y; bzb *= z;
                    bx[b] = bxb; by[b] = byb; bz[b] = bzb;
                }
                size_t offset = 0;
                auto func = [&](int L, int M, int N, int LL) {
                    bxb = bx[L]; byb = by[M]; bzb = bz[N];
                    double by_bz = byb * bzb;
                    result.phi(pt, bf + offset) += bxb * by_bz * g0;
                    offset++;
                };
                occ::gto::iterate_over_shell<true>(func, l);
            }
            occ::timing::stop(occ::timing::category::gto_gen);
        }
    }
}

void add_shell_contribution(
    size_t bf, const libint2::Shell &shell, const Eigen::Ref<const occ::Mat>& dists,
    GTOValues<1>& result,
    const Eigen::Ref<const occ::MaskArray>& mask)
{
    size_t n_pt = dists.cols();
    size_t n_prim = shell.nprim();
    constexpr size_t LMAX{8};
    int l = shell.contr[0].l;
    Eigen::Map<const occ::Vec> coeffs(shell.contr[0].coeff.data(), n_prim, 1),
        alpha(shell.alpha.data(), n_prim, 1);
    occ::Vec nalpha = - alpha;;
    occ::Vec tmp(n_prim);
    switch (l) {
        case 0: {
            occ::timing::start(occ::timing::category::gto_s);
            for(size_t pt = 0; pt < n_pt; pt++) {
                if(!mask(pt)) continue;
                double r2 = dists(3, pt);
                double g1{0.0};
                double g0{0.0};
                tmp = Eigen::exp(nalpha.array() * r2);

                for(int i = 0; i < n_prim; i++)
                {
                    double cexp = coeffs(i) * tmp(i);
                    g0 += cexp;
                    g1 += alpha(i) * cexp;
                }
                result.phi(pt, bf) += g0;

                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                g1 *= -2;
                result.phi_x(pt, bf) += g1 * x;
                result.phi_y(pt, bf) += g1 * y;
                result.phi_z(pt, bf) += g1 * z;
            }
            occ::timing::stop(occ::timing::category::gto_s);
            break;
        }
        case 1: {
            occ::timing::start(occ::timing::category::gto_p);
            for (size_t pt = 0; pt < n_pt; pt++) {
                if(!mask(pt)) continue;
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double g0{0.0}, g1{0.0};
                tmp = Eigen::exp(nalpha.array() * r2);

                for(int i = 0; i < n_prim; i++)
                {
                    double cexp = coeffs(i) * tmp(i);
                    g0 += cexp;
                    g1 += alpha(i) * cexp;
                }


                result.phi(pt, bf) += x * g0;
                result.phi(pt, bf + 1) += y * g0;
                result.phi(pt, bf + 2) += z * g0;

                g1 = -2 * g1;
                double g1x = g1 * x;
                double g1y = g1 * y;
                double g1z = g1 * z;

                result.phi_x(pt, bf) += g0 + x*g1x;
                result.phi_y(pt, bf) += x * g1y;
                result.phi_z(pt, bf) += x * g1z;
                result.phi_x(pt, bf + 1) += y*g1x;
                result.phi_y(pt, bf + 1) += g0 + y * g1y;
                result.phi_z(pt, bf + 1) += y * g1z;
                result.phi_x(pt, bf + 2) += z * g1x;
                result.phi_y(pt, bf + 2) += z * g1y;
                result.phi_z(pt, bf + 2) += g0 + z * g1z;
            }
            occ::timing::stop(occ::timing::category::gto_p);
            break;
        }
        default: {
            occ::timing::start(occ::timing::category::gto_gen);
            std::array<double, LMAX> bx, by, bz, gxb, gyb, gzb;
            bx[0] = 1.0; by[0] = 1.0; bz[0] = 1.0;
            for (size_t pt = 0; pt < n_pt; pt++) {
                if(!mask(pt)) continue;
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double g0{0.0}, g1{0.0};
                tmp = Eigen::exp(nalpha.array() * r2);

                for(int i = 0; i < n_prim; i++)
                {
                    double cexp = coeffs(i) * tmp(i);
                    g0 += cexp;
                    g1 += alpha(i) * cexp;
                }

                double g1x, g1y, g1z, g1xx, g1yy, g1zz;
                double bxb = x, byb = y, bzb = z;
                bx[1] = x; by[1] = y; bz[1] = z;

                g1 = -2 * g1;
                g1x = g1 * x;
                g1y = g1 * y;
                g1z = g1 * z;
                g1xx = g1x * x;
                g1yy = g1y * y;
                g1zz = g1z * z;
                gxb[0] = g1x; gyb[0] = g1y; gzb[0] = g1z;
                gxb[1] = g0 + g1xx; gyb[1] = g0 + g1yy; gzb[1] = g0 + g1zz;

                for(size_t b = 2; b <= l; b++) {
                    gxb[b] = (b * g0 + g1xx) * bxb;
                    gyb[b] = (b * g0 + g1yy) * byb;
                    gzb[b] = (b * g0 + g1zz) * bzb;
                    bxb *= x; byb *= y; bzb *= z;
                    bx[b] = bxb; by[b] = byb; bz[b] = bzb;
                }
                size_t offset = 0;
                auto func = [&](int L, int M, int N, int LL) {
                    bxb = bx[L]; byb = by[M]; bzb = bz[N];
                    double by_bz = byb * bzb;
                    result.phi(pt, bf + offset) += bxb * by_bz * g0;
                    result.phi_x(pt, bf + offset) += gxb[L] * by_bz;
                    result.phi_y(pt, bf + offset) += bxb * gyb[M] * bzb;
                    result.phi_z(pt, bf + offset) += bxb * byb * gzb[N];
                    offset++;
                };
                occ::gto::iterate_over_shell<true>(func, l);
            }
            occ::timing::stop(occ::timing::category::gto_gen);
        }
    }
}

}

}
