#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/timings.h>
#include <occ/core/logger.h>
#include <occ/core/util.h>
#include <occ/qm/basisset.h>
#include <string>
#include <vector>
#include <array>
#include <occ/gto/shell_order.h>
#include <fmt/core.h>

namespace occ::gto {

using occ::qm::BasisSet;
const inline char shell_labels[] = "SPDFGHIKMNOQRTUVWXYZ";

inline std::string component_label(int i, int j, int k, int l) {
    std::string label{shell_labels[l]};
    for(int c = 0; c < i; c++) label += "x";
    for(int c = 0; c < j; c++) label += "y";
    for(int c = 0; c < k; c++) label += "z";
    return label;
}

inline std::vector<std::string> shell_component_labels(int l)
{
    if(l == 0) return {std::string{shell_labels[0]}};
    int i, j, k;

    std::vector<std::string> labels;
    auto f = [&labels](int i, int j, int k, int l) {
        labels.push_back(component_label(i, j, k, l));
    };
    occ::gto::iterate_over_shell<true>(f, l);
    return labels;
}

template<size_t max_derivative>
struct GTOValues {
    GTOValues(size_t nbf, size_t npts) : phi(npts, nbf) {}
    inline void set_zero() {
        phi.setZero();
    }
    occ::Mat phi;
};

template<>
struct GTOValues<1>
{
    GTOValues(size_t nbf, size_t npts) : phi(npts, nbf), phi_x(npts, nbf), phi_y(npts, nbf), phi_z(npts, nbf) {}
    inline void set_zero() {
        phi.setZero();
        phi_x.setZero();
        phi_y.setZero();
        phi_z.setZero();
    }
    occ::Mat phi;
    occ::Mat phi_x;
    occ::Mat phi_y;
    occ::Mat phi_z;
};

inline double cartesian_normalization_factor(int l, int m, int n)
{
    int angular_momenta = l + m + n;
    using occ::util::double_factorial;
    return sqrt(double_factorial(angular_momenta) / (double_factorial(l) * double_factorial(m) * double_factorial(n)));
}

struct Momenta {
    int l{0};
    int m{0};
    int n{0};

    std::string to_string() const {
        int am = l + m + n;
        static char lsymb[] = "SPDFGHIKMNOQRTUVWXYZ";
        if (am == 0) return std::string(1, lsymb[0]);

        std::string suffix = "";
        for(int i = 0; i < l; i++) suffix += "x";
        for(int i = 0; i < m; i++) suffix += "y";
        for(int i = 0; i < n; i++) suffix += "z";

        return std::string(1, lsymb[am]) + suffix;
    }
};

inline std::vector<Momenta> cartesian_ordering(int l) {
    if(l == 0) return {{0, 0, 0}};
    int i = 0, j = 0, k = 0;
    std::vector<Momenta> powers;
    auto f = [&powers](int i, int j, int k, int l) {
        powers.push_back({i, j, k});
    };
    occ::gto::iterate_over_shell<true>(f, l);
    return powers;
}

namespace impl {

template<size_t max_derivative>
void add_shell_contribution(size_t bf, const libint2::Shell &shell, const Eigen::Ref<const occ::Mat>& dists,
                    GTOValues<max_derivative>& result,
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
                    if constexpr (max_derivative > 0) g1 += alpha(i) * cexp;
                }
                result.phi(pt, bf) += g0;

                if constexpr ( max_derivative > 0)
                {
                    double x = dists(0, pt);
                    double y = dists(1, pt);
                    double z = dists(2, pt);
                    g1 *= -2;
                    result.phi_x(pt, bf) += g1 * x;
                    result.phi_y(pt, bf) += g1 * y;
                    result.phi_z(pt, bf) += g1 * z;
                }
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
                    if constexpr (max_derivative > 0) g1 += alpha(i) * cexp;
                }


                result.phi(pt, bf) += x * g0;
                result.phi(pt, bf + 1) += y * g0;
                result.phi(pt, bf + 2) += z * g0;

                if constexpr (max_derivative > 0) {
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
            }
            occ::timing::stop(occ::timing::category::gto_p);
            break;
        }
        default: {
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
                    if constexpr (max_derivative > 0) g1 += alpha(i) * cexp;
                }

                double g1x, g1y, g1z, g1xx, g1yy, g1zz;
                double bxb = x, byb = y, bzb = z;
                bx[1] = x; by[1] = y; bz[1] = z;

                if constexpr ( max_derivative > 0)
                {
                    g1 = -2 * g1;
                    g1x = g1 * x;
                    g1y = g1 * y;
                    g1z = g1 * z;
                    g1xx = g1x * x;
                    g1yy = g1y * y;
                    g1zz = g1z * z;
                    gxb[0] = g1x; gyb[0] = g1y; gzb[0] = g1z;
                    gxb[1] = g0 + g1xx; gyb[1] = g0 + g1yy; gzb[1] = g0 + g1zz;
                }

                for(size_t b = 2; b <= l; b++) {
                    if constexpr ( max_derivative > 0)
                    {
                        gxb[b] = (b * g0 + g1xx) * bxb;
                        gyb[b] = (b * g0 + g1yy) * byb;
                        gzb[b] = (b * g0 + g1zz) * bzb;
                    }
                    bxb *= x; byb *= y; bzb *= z;
                    bx[b] = bxb; by[b] = byb; bz[b] = bzb;
                }
                size_t offset = 0;
                auto func = [&](int L, int M, int N, int LL) {
                    bxb = bx[L]; byb = by[M]; bzb = bz[N];
                    double by_bz = byb * bzb;
                    result.phi(pt, bf + offset) += bxb * by_bz * g0;
                    if constexpr (max_derivative > 0) {
                        result.phi_x(pt, bf + offset) += gxb[L] * by_bz;
                        result.phi_y(pt, bf + offset) += bxb * gyb[M] * bzb;
                        result.phi_z(pt, bf + offset) += bxb * byb * gzb[N];
                    }
                    offset++;
                };
                occ::gto::iterate_over_shell<true>(func, l);
            }
        }
    }
}
}

template<size_t max_derivative>
GTOValues<max_derivative> evaluate_basis_on_grid(const BasisSet &basis,
                                         const std::vector<libint2::Atom> &atoms,
                                         const occ::Mat &grid_pts)
{
    occ::timing::start(occ::timing::category::gto);
    size_t nbf = basis.nbf();
    size_t npts = grid_pts.cols();
    size_t natoms = atoms.size();
    GTOValues<max_derivative> gto_values(nbf, npts);
    gto_values.set_zero();
    auto shell2bf = basis.shell2bf();
    auto atom2shell = basis.atom2shell(atoms);
    constexpr auto EXPCUTOFF{50};

    for(size_t i = 0; i < natoms; i++)
    {
        occ::timing::start(occ::timing::category::gto_dist);
        const auto& atom = atoms[i];
        occ::Mat dists(4, npts);
        occ::MaskArray mask(npts);
        occ::Vec3 xyz(atom.x, atom.y, atom.z);

        dists.block(0, 0, 3, npts) = grid_pts.block(0, 0, 3, npts).colwise() - xyz;
        dists.row(3) = dists.block(0, 0, 3, npts).colwise().squaredNorm();
        occ::timing::stop(occ::timing::category::gto_dist);
        for(const auto& shell_idx: atom2shell[i]) {
            occ::timing::start(occ::timing::category::gto_mask);
            const auto& shell = basis[shell_idx];
            size_t bf = shell2bf[shell_idx];
            mask.setConstant(false);
            for(size_t pt = 0; pt < npts; pt++) {
                for(size_t prim = 0; prim < shell.nprim(); prim++) {
                    if((shell.alpha[prim] * dists(3, pt) - shell.max_ln_coeff[prim]) < EXPCUTOFF) {
                        mask(pt) = true;
                        break;
                    }
                }
            }
            occ::timing::stop(occ::timing::category::gto_mask);
            occ::timing::start(occ::timing::category::gto_shell);
            impl::add_shell_contribution<max_derivative>(bf, shell, dists, gto_values, mask);
            occ::timing::stop(occ::timing::category::gto_shell);
        }
    }
    occ::timing::stop(occ::timing::category::gto);
    return gto_values;
}

template<int angular_momentum>
std::vector<std::array<int, angular_momentum>> cartesian_gaussian_power_index_arrays()
{
    std::vector<std::array<int, angular_momentum>> result;
    int l, m, n;
    auto f = [&result](int l, int m, int n, int LL) {
        std::array<int, angular_momentum> powers;
        int idx = 0;
        for(int i = 0; i < l; i++) {
            powers[idx] = 0;
            idx++;
        }
        for(int i = 0; i < m; i++) {
            powers[idx] = 1;
            idx++;
        }
        for(int i = 0; i < n; i++) {
            powers[idx] = 2;
            idx++;
        }
        result.push_back(powers);
    };
    occ::gto::iterate_over_shell<true>(f, angular_momentum);
    return result;
}


/*
 * Result should be R: an MxM rotation matrix for P: a MxN set of coordinates
 * giving results P' = R P
 */
template<int l>
occ::MatRM cartesian_gaussian_rotation_matrix(const occ::Mat3 rotation)
{
    constexpr int num_moments = (l + 1) * (l + 2) / 2;
    occ::MatRM result = occ::MatRM::Zero(num_moments, num_moments);
    auto cg_powers = cartesian_gaussian_power_index_arrays<l>();
    for(int i = 0; i < num_moments; i++)
    {
        const auto ix = cg_powers[i];
        for(int j = 0; j < num_moments; j++)
        {
            std::array<int, l> jx = cg_powers[j];
            do {
                double tmp{1};
                for(int k = 0; k < ix.size(); k++) {
                    int u = ix[k], v = jx[k];
                    tmp *= rotation(v, u);
                }
                result(j, i) += tmp;
            } while(std::next_permutation(jx.begin(), jx.end()));
        }
    }
    return result;
}
}
