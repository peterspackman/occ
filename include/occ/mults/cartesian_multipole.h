#pragma once
#include <occ/ints/rints.h>
#include <occ/dma/mult.h>
#include <cmath>
#include <cstring>

namespace occ::mults {

using occ::ints::nherm;
using occ::ints::nhermsum;
using occ::ints::hermite_index;

/// Traceless Cartesian multipole moments Theta_{tuv}.
///
/// Storage layout follows RInts: hermite_index(t,u,v) with all ranks
/// from 0 to MaxL stored contiguously. Total size = nhermsum(MaxL).
///
/// @tparam MaxL Maximum multipole rank
template <int MaxL>
struct CartesianMultipole {
    static constexpr int size = nhermsum(MaxL);
    alignas(64) double data[size];

    CartesianMultipole() { std::memset(data, 0, sizeof(data)); }

    double &operator()(int t, int u, int v) {
        return data[hermite_index(t, u, v)];
    }

    double operator()(int t, int u, int v) const {
        return data[hermite_index(t, u, v)];
    }

    /// Return the highest rank with any non-zero component, or -1 if empty.
    int effective_rank(double tol = 1e-15) const {
        for (int l = MaxL; l >= 0; --l) {
            for (int t = 0; t <= l; ++t) {
                for (int u = 0; u <= l - t; ++u) {
                    if (std::abs(data[hermite_index(t, u, l - t - u)]) > tol)
                        return l;
                }
            }
        }
        return -1;
    }
};

namespace detail {

inline constexpr double sqrt3 = 1.7320508075688772935;
inline constexpr double sqrt5 = 2.2360679774997896964;
inline constexpr double sqrt6 = 2.4494897427831780982;
inline constexpr double sqrt7 = 2.6457513110645905905;
inline constexpr double sqrt10 = 3.1622776601683795;
inline constexpr double sqrt15 = 3.872983346207417;
inline constexpr double sqrt35 = 5.916079783099616;
inline constexpr double sqrt70 = 8.366600265340756;

/// Factorial lookup for small values
inline constexpr double factorial(int n) {
    constexpr double table[] = {1, 1, 2, 6, 24, 120, 720, 5040};
    return table[n];
}

/// Convert spherical multipole to traceless Cartesian for rank 0
template <int MaxL>
void convert_l0(const occ::dma::Mult &sph, CartesianMultipole<MaxL> &cart) {
    cart(0, 0, 0) = sph.Q00();
}

/// Convert spherical multipole to traceless Cartesian for rank 1
template <int MaxL>
void convert_l1(const occ::dma::Mult &sph, CartesianMultipole<MaxL> &cart) {
    if constexpr (MaxL < 1) return;
    cart(1, 0, 0) = sph.Q11c();  // x
    cart(0, 1, 0) = sph.Q11s();  // y
    cart(0, 0, 1) = sph.Q10();   // z
}

/// Convert spherical multipole to traceless Cartesian for rank 2
///
/// Theta_002 = 2*Q20/3
/// Theta_200 = -Q20/3 + Q22c/sqrt(3)
/// Theta_020 = -Q20/3 - Q22c/sqrt(3)
/// Theta_101 = Q21c/sqrt(3)
/// Theta_011 = Q21s/sqrt(3)
/// Theta_110 = Q22s/sqrt(3)
template <int MaxL>
void convert_l2(const occ::dma::Mult &sph, CartesianMultipole<MaxL> &cart) {
    if constexpr (MaxL < 2) return;
    double Q20 = sph.Q20();
    double Q21c = sph.Q21c();
    double Q21s = sph.Q21s();
    double Q22c = sph.Q22c();
    double Q22s = sph.Q22s();

    double inv_sqrt3 = 1.0 / sqrt3;

    cart(0, 0, 2) = 2.0 * Q20 / 3.0;
    cart(2, 0, 0) = -Q20 / 3.0 + Q22c * inv_sqrt3;
    cart(0, 2, 0) = -Q20 / 3.0 - Q22c * inv_sqrt3;
    cart(1, 0, 1) = Q21c * inv_sqrt3;
    cart(0, 1, 1) = Q21s * inv_sqrt3;
    cart(1, 1, 0) = Q22s * inv_sqrt3;
}

/// Convert spherical multipole to traceless Cartesian for rank 3
///
/// Derived from Stone's solid harmonics (Table B.1) with tracelessness.
template <int MaxL>
void convert_l3(const occ::dma::Mult &sph, CartesianMultipole<MaxL> &cart) {
    if constexpr (MaxL < 3) return;
    double Q30 = sph.Q30();
    double Q31c = sph.Q31c();
    double Q31s = sph.Q31s();
    double Q32c = sph.Q32c();
    double Q32s = sph.Q32s();
    double Q33c = sph.Q33c();
    double Q33s = sph.Q33s();

    double inv_sqrt6 = 1.0 / sqrt6;
    double inv_sqrt10 = 1.0 / sqrt10;
    double inv_sqrt15 = 1.0 / sqrt15;
    double c31c = Q31c * inv_sqrt6;  // Q31c / sqrt(6)
    double c31s = Q31s * inv_sqrt6;  // Q31s / sqrt(6)

    cart(0, 0, 3) = 2.0 * Q30 / 5.0;
    cart(2, 0, 1) = -Q30 / 5.0 + Q32c * inv_sqrt15;
    cart(0, 2, 1) = -Q30 / 5.0 - Q32c * inv_sqrt15;
    cart(1, 0, 2) = 4.0 * c31c / 5.0;
    cart(0, 1, 2) = 4.0 * c31s / 5.0;
    cart(1, 1, 1) = Q32s * inv_sqrt15;
    cart(3, 0, 0) = -3.0 * c31c / 5.0 + Q33c * inv_sqrt10;
    cart(1, 2, 0) = -c31c / 5.0 - Q33c * inv_sqrt10;
    cart(2, 1, 0) = -c31s / 5.0 + Q33s * inv_sqrt10;
    cart(0, 3, 0) = -3.0 * c31s / 5.0 - Q33s * inv_sqrt10;
}

/// Convert spherical multipole to traceless Cartesian for rank 4
///
/// Derived from Stone's solid harmonics with tracelessness constraints.
template <int MaxL>
void convert_l4(const occ::dma::Mult &sph, CartesianMultipole<MaxL> &cart) {
    if constexpr (MaxL < 4) return;
    double Q40 = sph.Q40();
    double Q41c = sph.Q41c();
    double Q41s = sph.Q41s();
    double Q42c = sph.Q42c();
    double Q42s = sph.Q42s();
    double Q43c = sph.Q43c();
    double Q43s = sph.Q43s();
    double Q44c = sph.Q44c();
    double Q44s = sph.Q44s();

    double inv_sqrt5 = 1.0 / sqrt5;
    double inv_sqrt10 = 1.0 / sqrt10;
    double inv_sqrt35 = 1.0 / sqrt35;
    double inv_sqrt70 = 1.0 / sqrt70;
    double inv_7sqrt5 = 1.0 / (7.0 * sqrt5);
    double inv_7sqrt10 = 1.0 / (7.0 * sqrt10);

    // Diagonal components (involving Q40, Q42c, Q44c)
    cart(0, 0, 4) = 8.0 * Q40 / 35.0;
    cart(4, 0, 0) = 3.0 * Q40 / 35.0 + Q44c * inv_sqrt35
                    - 2.0 * Q42c * inv_7sqrt5;
    cart(0, 4, 0) = 3.0 * Q40 / 35.0 + Q44c * inv_sqrt35
                    + 2.0 * Q42c * inv_7sqrt5;
    cart(2, 2, 0) = Q40 / 35.0 - Q44c * inv_sqrt35;
    cart(2, 0, 2) = -4.0 * Q40 / 35.0 + 2.0 * Q42c * inv_7sqrt5;
    cart(0, 2, 2) = -4.0 * Q40 / 35.0 - 2.0 * Q42c * inv_7sqrt5;

    // xy-plane components (involving Q42s, Q44s)
    cart(3, 1, 0) = Q44s * inv_sqrt35 - Q42s * inv_7sqrt5;
    cart(1, 3, 0) = -Q42s * inv_7sqrt5 - Q44s * inv_sqrt35;
    cart(1, 1, 2) = 2.0 * Q42s * inv_7sqrt5;

    // xz-plane components (involving Q41c, Q43c)
    cart(3, 0, 1) = -3.0 * Q41c * inv_7sqrt10 + Q43c * inv_sqrt70;
    cart(1, 2, 1) = -Q41c * inv_7sqrt10 - Q43c * inv_sqrt70;
    cart(1, 0, 3) = 4.0 * Q41c * inv_7sqrt10;

    // yz-plane components (involving Q41s, Q43s)
    cart(2, 1, 1) = Q43s * inv_sqrt70 - Q41s * inv_7sqrt10;
    cart(0, 3, 1) = -3.0 * Q41s * inv_7sqrt10 - Q43s * inv_sqrt70;
    cart(0, 1, 3) = 4.0 * Q41s * inv_7sqrt10;
}

} // namespace detail

/// Convert spherical multipole (Mult) to traceless Cartesian multipole.
///
/// The output Theta_{tuv} satisfies tracelessness:
///   Theta_{t+2,u,v} + Theta_{t,u+2,v} + Theta_{t,u,v+2} = 0
/// for all valid (t,u,v) with t+u+v = l-2.
///
/// @tparam MaxL  Maximum rank to convert (0..4 supported)
/// @param sph    Input spherical multipole (Stone convention)
/// @param cart   Output traceless Cartesian multipole
template <int MaxL>
void spherical_to_cartesian(const occ::dma::Mult &sph,
                            CartesianMultipole<MaxL> &cart) {
    static_assert(MaxL <= 4, "spherical_to_cartesian supports MaxL <= 4");
    detail::convert_l0(sph, cart);
    if constexpr (MaxL >= 1) detail::convert_l1(sph, cart);
    if constexpr (MaxL >= 2) detail::convert_l2(sph, cart);
    if constexpr (MaxL >= 3) detail::convert_l3(sph, cart);
    if constexpr (MaxL >= 4) detail::convert_l4(sph, cart);
}

} // namespace occ::mults
