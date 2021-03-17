#pragma once
#include <tonto/core/linear_algebra.h>
#include <tonto/core/timings.h>
#include <tonto/core/logger.h>
#include <tonto/core/util.h>
#include <tonto/qm/basisset.h>
#include <string>
#include <vector>
#include <array>
#include <libint2/cgshell_ordering.h>
#include <libint2/shgshell_ordering.h>
#include <fmt/core.h>

namespace tonto::gto {

using tonto::qm::BasisSet;
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
    FOR_CART(i, j, k, l)
        labels.push_back(component_label(i, j, k, l));
    END_FOR_CART
    return labels;
}

template<size_t max_derivative>
struct GTOValues {
    GTOValues(size_t nbf, size_t npts) : phi(npts, nbf) {}
    inline void set_zero() {
        phi.setZero();
    }
    tonto::Mat phi;
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
    tonto::Mat phi;
    tonto::Mat phi_x;
    tonto::Mat phi_y;
    tonto::Mat phi_z;
};

inline double cartesian_normalization_factor(int l, int m, int n)
{
    int angular_momenta = l + m + n;
    using tonto::util::double_factorial;
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
    FOR_CART(i,j,k,l)
        powers.push_back({i, j, k});
    END_FOR_CART
    return powers;
}

namespace impl {

template<size_t max_derivative>
void add_shell_contribution(size_t bf, const libint2::Shell &shell, const Eigen::Ref<const tonto::Mat>& dists,
                    GTOValues<max_derivative>& result,
                    const Eigen::Ref<const tonto::MaskArray>& mask)
{
    size_t n_pt = dists.cols();
    size_t n_prim = shell.nprim();
    constexpr size_t LMAX{8};
    for(const auto& contraction: shell.contr) {
        switch (contraction.l) {
        case 0: {
            for(size_t pt = 0; pt < n_pt; pt++) {
                if(!mask(pt)) continue;
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double cc_exp_r2 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    cc_exp_r2 += cexp;
                    if constexpr (max_derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;

                result.phi(pt, bf) += cc_exp_r2;
                if constexpr ( max_derivative > 0)
                {
                    result.phi_x(pt, bf) += g1 * x;
                    result.phi_y(pt, bf) += g1 * y;
                    result.phi_z(pt, bf) += g1 * z;
                }
            }
            break;
        }
        case 1: {
            for (size_t pt = 0; pt < n_pt; pt++) {
                if(!mask(pt)) continue;
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double g0 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    g0 += cexp;
                    if constexpr (max_derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;
                double g1x = g1 * x;
                double g1y = g1 * y;
                double g1z = g1 * z;
                result.phi(pt, bf) += x * g0;
                result.phi(pt, bf + 1) += y * g0;
                result.phi(pt, bf + 2) += z * g0;
                if constexpr (max_derivative > 0) {
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
                double g0 = 0.0;
                double g1{0.0};
                #pragma omp simd
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    g0 += cexp;
                    if constexpr (max_derivative > 0) {
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
                size_t offset = 0;
                FOR_CART(L, M, N, contraction.l)
                    bxb = bx[L]; byb = by[M]; bzb = bz[N];
                    double by_bz = byb * bzb;
                    result.phi(pt, bf + offset) += bxb * by_bz * g0;
                    if constexpr (max_derivative > 0) {
                        result.phi_x(pt, bf + offset) += gxb[L] * by_bz;
                        result.phi_y(pt, bf + offset) += bxb * gyb[M] * bzb;
                        result.phi_z(pt, bf + offset) += bxb * byb * gzb[N];
                    }
                    offset++;
                END_FOR_CART
            }
        }
        }
    }
}

template<size_t max_derivative>
void add_shell_contribution_block(size_t bf, const libint2::Shell &shell, const Eigen::Ref<const tonto::Mat>& dists,
                    GTOValues<max_derivative>& result, size_t offset, size_t N)
{
    size_t n_pt = dists.cols();
    size_t n_prim = shell.nprim();
    constexpr size_t LMAX{5};
    for(const auto& contraction: shell.contr) {
        switch (contraction.l) {
        case 0: {
            for(size_t pt = offset; pt < offset + N; pt++) {
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double cc_exp_r2 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    cc_exp_r2 += cexp;
                    if constexpr (max_derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;

                result.phi(pt, bf) += cc_exp_r2;
                if constexpr ( max_derivative > 0)
                {
                    result.phi_x(pt, bf) += g1 * x;
                    result.phi_y(pt, bf) += g1 * y;
                    result.phi_z(pt, bf) += g1 * z;
                }
            }
            break;
        }
        case 1: {
            for (size_t pt = offset; pt <  offset + N; pt++) {
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double g0 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    g0 += cexp;
                    if constexpr (max_derivative > 0) {
                        g1 += shell.alpha[i] * cexp;
                    }
                }
                g1 *= -2;
                double g1x = g1 * x;
                double g1y = g1 * y;
                double g1z = g1 * z;
                result.phi(pt, bf) += x * g0;
                result.phi(pt, bf + 1) += y * g0;
                result.phi(pt, bf + 2) += z * g0;
                if constexpr (max_derivative > 0) {
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
            break;
        }
        default: {
            std::array<double, LMAX> bx, by, bz, gxb, gyb, gzb;
            bx[0] = 1.0; by[0] = 1.0; bz[0] = 1.0;
            for (size_t pt = offset; pt < offset + N; pt++) {
                double x = dists(0, pt);
                double y = dists(1, pt);
                double z = dists(2, pt);
                double r2 = dists(3, pt);
                double g0 = 0.0;
                double g1{0.0};
                for(int i = 0; i < n_prim; i++) {
                    double cexp = contraction.coeff[i] * exp(- shell.alpha[i] * r2);
                    g0 += cexp;
                    if constexpr (max_derivative > 0) {
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
                size_t offset = 0;
                FOR_CART(L, M, N, contraction.l)
                    bxb = bx[L]; byb = by[M]; bzb = bz[N];
                    double by_bz = byb * bzb;
                    result.phi(pt, bf + offset) += bxb * by_bz * g0;
                    if constexpr (max_derivative > 0) {
                        result.phi_x(pt, bf + offset) += gxb[L] * by_bz;
                        result.phi_y(pt, bf + offset) += bxb * gyb[M] * bzb;
                        result.phi_z(pt, bf + offset) += bxb * byb * gzb[N];
                    }
                    offset++;
                END_FOR_CART
            }
        }
        }
    }
}

}

template<size_t max_derivative>
GTOValues<max_derivative> evaluate_basis_on_grid(const BasisSet &basis,
                                         const std::vector<libint2::Atom> &atoms,
                                         const tonto::Mat &grid_pts)
{
    tonto::timing::start(tonto::timing::category::gto);
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
        const auto& atom = atoms[i];
        tonto::Mat dists(4, npts);
        tonto::MaskArray mask(npts);
        tonto::Vec3 xyz(atom.x, atom.y, atom.z);

        dists.block(0, 0, 3, npts) = grid_pts.block(0, 0, 3, npts).colwise() - xyz;
        dists.row(3) = dists.block(0, 0, 3, npts).colwise().squaredNorm();
        for(const auto& shell_idx: atom2shell[i]) {
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
            impl::add_shell_contribution<max_derivative>(bf, shell, dists, gto_values, mask);
        }
    }
    tonto::timing::stop(tonto::timing::category::gto);
    return gto_values;
}

template<int angular_momentum>
std::vector<std::array<int, angular_momentum>> cartesian_gaussian_power_index_arrays()
{
    std::vector<std::array<int, angular_momentum>> result;
    int l, m, n;
    FOR_CART(l, m, n, angular_momentum)
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
    END_FOR_CART
    return result;
}


/*
 * Result should be R: an MxM rotation matrix for P: a MxN set of coordinates
 * giving results P' = R P
 */
template<int l>
tonto::MatRM cartesian_gaussian_rotation_matrix(const tonto::Mat3 rotation)
{
    constexpr int num_moments = (l + 1) * (l + 2) / 2;
    tonto::MatRM result = tonto::MatRM::Zero(num_moments, num_moments);
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
