#pragma once
#include <array>
#include <fmt/core.h>
#include <gau2grid/gau2grid.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/gto/shell_order.h>
#include <occ/qm/shell.h>
#include <string>
#include <vector>

namespace occ::gto {

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
    double result =
        sqrt(double_factorial(angular_momenta) /
             (double_factorial(l) * double_factorial(m) * double_factorial(n)));
    if (angular_momenta > 1) {
        result /= 2 * std::sqrt(M_PI / (2 * angular_momenta + 1));
    }
    return result;
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
    std::vector<MomentaSpherical> moments;
    auto f = [&moments](int l, int m) { moments.push_back({l, m}); };
    occ::gto::iterate_over_shell<false>(f, l);
    return moments;
}

void evaluate_basis(const qm::AOBasis &basis, const occ::Mat &grid_pts,
                    GTOValues &gto_values, int max_derivative);

inline GTOValues evaluate_basis(const qm::AOBasis &basis,
                                const occ::Mat &grid_pts, int max_derivative) {
    GTOValues gto_values;
    evaluate_basis(basis, grid_pts, gto_values, max_derivative);
    return gto_values;
}

Vec evaluate_decay_cutoff(const qm::AOBasis &basis);

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

Mat cartesian_to_spherical_transformation_matrix(int l);
Mat spherical_to_cartesian_transformation_matrix(int l);

} // namespace occ::gto
