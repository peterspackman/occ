#pragma once
#include <occ/mults/cartesian_multipole.h>
#include <occ/core/linear_algebra.h>
#include <algorithm>
#include <cmath>

namespace occ::mults {

namespace rotation_detail {

inline double ipow(double x, int n) {
    switch (n) {
        case 0: return 1.0;
        case 1: return x;
        case 2: return x * x;
        case 3: return x * x * x;
        case 4: return x * x * x * x;
        default: return std::pow(x, n);
    }
}

inline constexpr double fact(int n) {
    constexpr double table[] = {1, 1, 2, 6, 24, 120, 720, 5040};
    return table[n];
}

/// Compute the rotation kernel element R_l(tuv, abc; M).
///
/// For a symmetric rank-l Cartesian tensor, the rotation from body to lab
/// involves summing over all 3x3 transportation matrices p with
/// row sums (t,u,v) and column sums (a,b,c).
///
/// K = sum_p [t!/(pxx!pxy!pxz!) * u!/(pyx!pyy!pyz!) * v!/(pzx!pzy!pzz!)]
///     * M(0,0)^pxx * M(0,1)^pxy * ... * M(2,2)^pzz
inline double rotation_kernel(int t, int u, int v,
                               int a, int b, int c,
                               const Mat3 &M) {
    double coeff = 0.0;

    for (int pxx = 0; pxx <= std::min(t, a); ++pxx) {
        for (int pxy = 0; pxy <= std::min(t - pxx, b); ++pxy) {
            int pxz = t - pxx - pxy;
            if (pxz > c) continue;

            for (int pyx = 0; pyx <= std::min(u, a - pxx); ++pyx) {
                int pzx = a - pxx - pyx;
                if (pzx > v) continue;

                for (int pyy = 0; pyy <= std::min(u - pyx, b - pxy); ++pyy) {
                    int pyz = u - pyx - pyy;
                    int pzy = b - pxy - pyy;
                    int pzz = v - pzx - pzy;

                    if (pyz < 0 || pzy < 0 || pzz < 0) continue;

                    double mc = (fact(t) / (fact(pxx) * fact(pxy) * fact(pxz)))
                              * (fact(u) / (fact(pyx) * fact(pyy) * fact(pyz)))
                              * (fact(v) / (fact(pzx) * fact(pzy) * fact(pzz)));

                    double mp = ipow(M(0, 0), pxx) * ipow(M(0, 1), pxy)
                              * ipow(M(0, 2), pxz) * ipow(M(1, 0), pyx)
                              * ipow(M(1, 1), pyy) * ipow(M(1, 2), pyz)
                              * ipow(M(2, 0), pzx) * ipow(M(2, 1), pzy)
                              * ipow(M(2, 2), pzz);

                    coeff += mc * mp;
                }
            }
        }
    }
    return coeff;
}

/// Compute the derivative of the rotation kernel w.r.t. angle-axis parameter.
///
/// Given M1 = dM/dp_k, uses the product rule:
/// dK/dp_k = sum_p mc * sum_{i,j: p(i,j)>0} [p(i,j) * M(i,j)^(p(i,j)-1) * M1(i,j)
///           * product of remaining M factors]
inline double rotation_kernel_derivative(int t, int u, int v,
                                          int a, int b, int c,
                                          const Mat3 &M, const Mat3 &M1) {
    double dcoeff = 0.0;

    for (int pxx = 0; pxx <= std::min(t, a); ++pxx) {
        for (int pxy = 0; pxy <= std::min(t - pxx, b); ++pxy) {
            int pxz = t - pxx - pxy;
            if (pxz > c) continue;

            for (int pyx = 0; pyx <= std::min(u, a - pxx); ++pyx) {
                int pzx = a - pxx - pyx;
                if (pzx > v) continue;

                for (int pyy = 0; pyy <= std::min(u - pyx, b - pxy); ++pyy) {
                    int pyz = u - pyx - pyy;
                    int pzy = b - pxy - pyy;
                    int pzz = v - pzx - pzy;

                    if (pyz < 0 || pzy < 0 || pzz < 0) continue;

                    double mc = (fact(t) / (fact(pxx) * fact(pxy) * fact(pxz)))
                              * (fact(u) / (fact(pyx) * fact(pyy) * fact(pyz)))
                              * (fact(v) / (fact(pzx) * fact(pzy) * fact(pzz)));

                    int p[3][3] = {
                        {pxx, pxy, pxz},
                        {pyx, pyy, pyz},
                        {pzx, pzy, pzz}
                    };

                    // For each (ri, rj), replace one M factor with M1
                    double deriv_term = 0.0;
                    for (int ri = 0; ri < 3; ++ri) {
                        for (int rj = 0; rj < 3; ++rj) {
                            if (p[ri][rj] == 0) continue;
                            double prod = 1.0;
                            for (int i = 0; i < 3; ++i) {
                                for (int j = 0; j < 3; ++j) {
                                    if (i == ri && j == rj) {
                                        prod *= p[ri][rj]
                                              * ipow(M(i, j), p[i][j] - 1)
                                              * M1(i, j);
                                    } else {
                                        prod *= ipow(M(i, j), p[i][j]);
                                    }
                                }
                            }
                            deriv_term += prod;
                        }
                    }

                    dcoeff += mc * deriv_term;
                }
            }
        }
    }
    return dcoeff;
}

} // namespace rotation_detail

/// Rotate a CartesianMultipole from body frame to lab frame.
///
/// For each rank l, applies:
///   lab(t,u,v) = sum_{a+b+c=l} K(tuv, abc; M) * body(a,b,c)
template <int MaxL>
void rotate_cartesian_multipole(
    const CartesianMultipole<MaxL> &body,
    const Mat3 &M,
    CartesianMultipole<MaxL> &lab) {
    using occ::ints::hermite_index;

    // Rank 0: scalar
    lab.data[0] = body.data[0];

    for (int l = 1; l <= MaxL; ++l) {
        for (int t = l; t >= 0; --t) {
            for (int u = l - t; u >= 0; --u) {
                int v = l - t - u;
                double val = 0.0;
                for (int a = l; a >= 0; --a) {
                    for (int b = l - a; b >= 0; --b) {
                        int c = l - a - b;
                        double bv = body.data[hermite_index(a, b, c)];
                        if (bv == 0.0) continue;
                        val += rotation_detail::rotation_kernel(t, u, v, a, b, c, M) * bv;
                    }
                }
                lab.data[hermite_index(t, u, v)] = val;
            }
        }
    }
}

/// Compute derivative of lab-frame multipole w.r.t. angle-axis parameter.
///
/// Given M1 = dM/dp_k, computes d(lab)/dp_k for the rotated multipole.
template <int MaxL>
void rotate_cartesian_multipole_derivative(
    const CartesianMultipole<MaxL> &body,
    const Mat3 &M,
    const Mat3 &M1,
    CartesianMultipole<MaxL> &d_lab) {
    using occ::ints::hermite_index;

    // Rank 0: derivative is zero (scalar is rotation-invariant)
    d_lab.data[0] = 0.0;

    for (int l = 1; l <= MaxL; ++l) {
        for (int t = l; t >= 0; --t) {
            for (int u = l - t; u >= 0; --u) {
                int v = l - t - u;
                double val = 0.0;
                for (int a = l; a >= 0; --a) {
                    for (int b = l - a; b >= 0; --b) {
                        int c = l - a - b;
                        double bv = body.data[hermite_index(a, b, c)];
                        if (bv == 0.0) continue;
                        val += rotation_detail::rotation_kernel_derivative(
                                   t, u, v, a, b, c, M, M1) * bv;
                    }
                }
                d_lab.data[hermite_index(t, u, v)] = val;
            }
        }
    }
}

} // namespace occ::mults
