#pragma once
#include <array>
#include <fmt/core.h>
#include <gau2grid/gau2grid.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/gto/shell_order.h>
#include <occ/qm/basisset.h>
#include <occ/qm/occshell.h>
#include <string>
#include <vector>

namespace occ::gto {

using occ::qm::BasisSet;
constexpr inline char shell_labels[] = "SPDFGHIKMNOQRTUVWXYZ";

inline std::string component_label(int i, int j, int k, int l) {
    std::string label{shell_labels[l]};
    for (int c = 0; c < i; c++)
        label += "x";
    for (int c = 0; c < j; c++)
        label += "y";
    for (int c = 0; c < k; c++)
        label += "z";
    return label;
}

template <bool Cartesian = true>
inline std::vector<std::string> shell_component_labels(int l) {
    if (l == 0)
        return {std::string{shell_labels[0]}};
    int i, j, k;

    std::vector<std::string> labels;
    auto f = [&labels](int i, int j, int k, int l) {
        labels.push_back(component_label(i, j, k, l));
    };
    occ::gto::iterate_over_shell<Cartesian>(f, l);
    return labels;
}

struct GTOValues {

    inline void reserve(size_t nbf, size_t npts, int derivative_order) {
        phi = Mat(npts, nbf);
        if (derivative_order > 0) {
            phi_x = Mat(npts, nbf);
            phi_y = Mat(npts, nbf);
            phi_z = Mat(npts, nbf);
        }
        if (derivative_order > 1) {
            phi_xx = Mat(npts, nbf);
            phi_xy = Mat(npts, nbf);
            phi_xz = Mat(npts, nbf);
            phi_yy = Mat(npts, nbf);
            phi_yz = Mat(npts, nbf);
            phi_zz = Mat(npts, nbf);
        }
    }

    inline void set_zero() {
        phi.setZero();
        phi_x.setZero();
        phi_y.setZero();
        phi_z.setZero();
        phi_xx.setZero();
        phi_xy.setZero();
        phi_xz.setZero();
        phi_yy.setZero();
        phi_yz.setZero();
        phi_zz.setZero();
    }
    Mat phi;
    Mat phi_x;
    Mat phi_y;
    Mat phi_z;
    Mat phi_xx;
    Mat phi_xy;
    Mat phi_xz;
    Mat phi_yy;
    Mat phi_yz;
    Mat phi_zz;
};

constexpr unsigned int num_subshells(bool cartesian, unsigned int l) {
    if (l == 0)
        return 1;
    if (l == 1)
        return 3;
    if (cartesian)
        return (l + 2) * (l + 1) / 2;
    return 2 * l + 1;
}

inline double cartesian_normalization_factor(int l, int m, int n) {
    int angular_momenta = l + m + n;
    using occ::util::double_factorial;
    return sqrt(
        double_factorial(angular_momenta) /
        (double_factorial(l) * double_factorial(m) * double_factorial(n)));
}

struct Momenta {
    int l{0};
    int m{0};
    int n{0};

    std::string to_string() const {
        int am = l + m + n;
        if (am == 0)
            return std::string(1, shell_labels[0]);

        std::string suffix = "";
        for (int i = 0; i < l; i++)
            suffix += "x";
        for (int i = 0; i < m; i++)
            suffix += "y";
        for (int i = 0; i < n; i++)
            suffix += "z";

        return std::string(1, shell_labels[am]) + suffix;
    }
};

struct MomentaSpherical {
    int l{0};
    int m{0};
};

inline std::vector<Momenta> cartesian_subshell_ordering(int l) {
    if (l == 0)
        return {{0, 0, 0}};
    int i = 0, j = 0, k = 0;
    std::vector<Momenta> powers;
    auto f = [&powers](int i, int j, int k, int l) {
        powers.push_back({i, j, k});
    };
    occ::gto::iterate_over_shell<true>(f, l);
    return powers;
}

inline std::vector<MomentaSpherical> spherical_subshell_ordering(int l) {
    if (l == 0)
        return {MomentaSpherical{}};
    int m = 0;
    std::vector<MomentaSpherical> moments;
    auto f = [&moments](int l, int m) { moments.push_back({l, m}); };
    occ::gto::iterate_over_shell<false>(f, l);
    return moments;
}

template <int max_derivative = 0>
void evaluate_basis(const qm::AOBasis &basis, const occ::Mat &grid_pts,
                    GTOValues &gto_values) {
    occ::timing::start(occ::timing::category::gto);
    size_t nbf = basis.nbf();
    size_t npts = grid_pts.cols();
    size_t natoms = basis.atoms().size();
    gto_values.reserve(nbf, npts, max_derivative);
    gto_values.set_zero();
    auto shell2bf = basis.first_bf();
    auto atom2shell = basis.atom_to_shell();
    for (size_t i = 0; i < natoms; i++) {
        for (const auto &shell_idx : atom2shell[i]) {
            occ::timing::start(occ::timing::category::gto_shell);
            size_t bf = shell2bf[shell_idx];
            double *output = gto_values.phi.col(bf).data();
            const double *xyz = grid_pts.data();
            long int xyz_stride = 3;
            const auto &sh = basis[shell_idx];
            const double *coeffs = sh.contraction_coefficients.data();
            const double *alpha = sh.exponents.data();
            const double *center = sh.origin.data();
            int L = sh.l;
            int order = (sh.kind == qm::OccShell::Kind::Spherical)
                            ? GG_SPHERICAL_CCA
                            : GG_CARTESIAN_CCA;
            if constexpr (max_derivative == 0) {
                gg_collocation(L, npts, xyz, xyz_stride, sh.num_primitives(),
                               coeffs, alpha, center, order, output);
            } else if constexpr (max_derivative == 1) {
                double *x_out = gto_values.phi_x.col(bf).data();
                double *y_out = gto_values.phi_y.col(bf).data();
                double *z_out = gto_values.phi_z.col(bf).data();
                gg_collocation_deriv1(
                    L, npts, xyz, xyz_stride, sh.num_primitives(), coeffs,
                    alpha, center, order, output, x_out, y_out, z_out);
            } else if constexpr (max_derivative == 2) {
                double *x_out = gto_values.phi_x.col(bf).data();
                double *y_out = gto_values.phi_y.col(bf).data();
                double *z_out = gto_values.phi_z.col(bf).data();
                double *xx_out = gto_values.phi_xx.col(bf).data();
                double *xy_out = gto_values.phi_xy.col(bf).data();
                double *xz_out = gto_values.phi_xz.col(bf).data();
                double *yy_out = gto_values.phi_yy.col(bf).data();
                double *yz_out = gto_values.phi_yz.col(bf).data();
                double *zz_out = gto_values.phi_zz.col(bf).data();
                gg_collocation_deriv2(
                    L, npts, xyz, xyz_stride, sh.num_primitives(), coeffs,
                    alpha, center, order, output, x_out, y_out, z_out, xx_out,
                    xy_out, xz_out, yy_out, yz_out, zz_out);
            }
            occ::timing::stop(occ::timing::category::gto_shell);
        }
    }
    occ::timing::stop(occ::timing::category::gto);
}

void evaluate_basis(const qm::AOBasis &basis, const occ::Mat &grid_pts,
                    GTOValues &gto_values, int max_derivative);

inline GTOValues evaluate_basis(const qm::AOBasis &basis,
                                const occ::Mat &grid_pts, int max_derivative) {
    GTOValues gto_values;
    evaluate_basis(basis, grid_pts, gto_values, max_derivative);
    return gto_values;
}

template <int max_derivative = 0>
void evaluate_basis(const BasisSet &basis,
                    const std::vector<occ::core::Atom> &atoms,
                    const occ::Mat &grid_pts, GTOValues &gto_values) {
    occ::timing::start(occ::timing::category::gto);
    size_t nbf = basis.nbf();
    size_t npts = grid_pts.cols();
    size_t natoms = atoms.size();
    gto_values.reserve(nbf, npts, max_derivative);
    gto_values.set_zero();
    auto shell2bf = basis.shell2bf();
    auto atom2shell = basis.atom2shell(atoms);
    for (size_t i = 0; i < natoms; i++) {
        for (const auto &shell_idx : atom2shell[i]) {
            occ::timing::start(occ::timing::category::gto_shell);
            size_t bf = shell2bf[shell_idx];
            double *output = gto_values.phi.col(bf).data();
            const double *xyz = grid_pts.data();
            long int xyz_stride = 3;
            const auto &sh = basis[shell_idx];
            const double *coeffs = sh.contr[0].coeff.data();
            const double *alpha = sh.alpha.data();
            const double *center = sh.O.data();
            int L = sh.contr[0].l;
            int order =
                (sh.contr[0].pure) ? GG_SPHERICAL_CCA : GG_CARTESIAN_CCA;
            if constexpr (max_derivative == 0) {
                gg_collocation(L, npts, xyz, xyz_stride, sh.nprim(), coeffs,
                               alpha, center, order, output);
            } else if constexpr (max_derivative == 1) {
                double *x_out = gto_values.phi_x.col(bf).data();
                double *y_out = gto_values.phi_y.col(bf).data();
                double *z_out = gto_values.phi_z.col(bf).data();
                gg_collocation_deriv1(L, npts, xyz, xyz_stride, sh.nprim(),
                                      coeffs, alpha, center, order, output,
                                      x_out, y_out, z_out);
            } else if constexpr (max_derivative == 2) {
                double *x_out = gto_values.phi_x.col(bf).data();
                double *y_out = gto_values.phi_y.col(bf).data();
                double *z_out = gto_values.phi_z.col(bf).data();
                double *xx_out = gto_values.phi_xx.col(bf).data();
                double *xy_out = gto_values.phi_xy.col(bf).data();
                double *xz_out = gto_values.phi_xz.col(bf).data();
                double *yy_out = gto_values.phi_yy.col(bf).data();
                double *yz_out = gto_values.phi_yz.col(bf).data();
                double *zz_out = gto_values.phi_zz.col(bf).data();
                gg_collocation_deriv2(L, npts, xyz, xyz_stride, sh.nprim(),
                                      coeffs, alpha, center, order, output,
                                      x_out, y_out, z_out, xx_out, xy_out,
                                      xz_out, yy_out, yz_out, zz_out);
            }
            occ::timing::stop(occ::timing::category::gto_shell);
        }
    }
    occ::timing::stop(occ::timing::category::gto);
}

void evaluate_basis(const BasisSet &basis,
                    const std::vector<occ::core::Atom> &atoms,
                    const occ::Mat &grid_pts, GTOValues &gto_values,
                    int max_derivative);

inline GTOValues evaluate_basis(const BasisSet &basis,
                                const std::vector<occ::core::Atom> &atoms,
                                const occ::Mat &grid_pts, int max_derivative) {
    GTOValues gto_values;
    evaluate_basis(basis, atoms, grid_pts, gto_values, max_derivative);
    return gto_values;
}

template <int angular_momentum>
std::vector<std::array<int, angular_momentum>>
cartesian_gaussian_power_index_arrays() {
    std::vector<std::array<int, angular_momentum>> result;
    auto f = [&result](int l, int m, int n, int LL) {
        std::array<int, angular_momentum> powers;
        int idx = 0;
        for (int i = 0; i < l; i++) {
            powers[idx] = 0;
            idx++;
        }
        for (int i = 0; i < m; i++) {
            powers[idx] = 1;
            idx++;
        }
        for (int i = 0; i < n; i++) {
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
template <int l>
Mat cartesian_gaussian_rotation_matrix(const occ::Mat3 rotation) {
    constexpr int num_moments = (l + 1) * (l + 2) / 2;
    Mat result = Mat::Zero(num_moments, num_moments);
    auto cg_powers = cartesian_gaussian_power_index_arrays<l>();
    int p1_idx = 0;
    for (const auto &p1 : cg_powers) {
        int p2_idx = 0;
        // copy as we're permuting p2
        for (auto p2 : cg_powers) {
            do {
                double tmp{1.0};
                for (int k = 0; k < l; k++) {
                    tmp *= rotation(p2[k], p1[k]);
                }
                result(p2_idx, p1_idx) += tmp;
            } while (std::next_permutation(p2.begin(), p2.end()));
            p2_idx++;
        }
        p1_idx++;
    }
    return result;
}

Mat cartesian_to_spherical_transformation_matrix(int l);
Mat spherical_to_cartesian_transformation_matrix(int l);

} // namespace occ::gto
