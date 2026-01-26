#pragma once
#include <occ/ints/kernels.h>
#include <stdexcept>
#include <string>

namespace occ::ints {

// ============================================================================
// Templated 3-center ERI kernels (μν|P)
// Similar to ESP but with Gaussian auxiliary function instead of point charge
// ============================================================================

/// 3-center ERI primitive integral (μν|P)
/// Uses templated RInts for efficiency (no dynamic allocation)
template <typename T, int LA, int LB, int LC, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED
void eri3c_primitive(T a, T b, T gamma,
                     const T* A, const T* B, const T* C,
                     const T* boys_table,
                     T* integrals) {
    constexpr int L_ab = LA + LB;
    constexpr int L = L_ab + LC;
    constexpr int na = ncart(LA);
    constexpr int nb = ncart(LB);
    constexpr int nc = ncart(LC);

    // Combined exponent and center for bra
    const T p = a + b;
    const T Px = (a * A[0] + b * B[0]) / p;
    const T Py = (a * A[1] + b * B[1]) / p;
    const T Pz = (a * A[2] + b * B[2]) / p;

    // Combined parameter for R-integrals
    const T pq = p + gamma;
    const T alpha = p * gamma / pq;

    // Vector from P to C
    const T PCx = Px - C[0];
    const T PCy = Py - C[1];
    const T PCz = Pz - C[2];

    // Compute E-coefficients for bra pair
    ECoeffs3D<T, LA, LB> E_ab;
    compute_e_coeffs_3d<T, LA, LB>(a, b, A, B, E_ab);

    // Compute single-center E-coefficients for aux (1D recurrence)
    // E^{l,0}_t for single center: E^0_0 = 1, E^{l+1}_t = (1/2γ) E^l_{t-1} + (t+1) E^l_{t+1}
    T E_c_1d[(LC + 1) * (LC + 1)];
    const T half_gamma_inv = T(0.5) / gamma;
    for (int i = 0; i <= LC; ++i) {
        for (int t = 0; t <= LC; ++t) {
            E_c_1d[i * (LC + 1) + t] = T(0);
        }
    }
    E_c_1d[0] = T(1);
    for (int i = 0; i < LC; ++i) {
        for (int t = 0; t <= i + 1; ++t) {
            T val = T(0);
            if (t > 0) val += half_gamma_inv * E_c_1d[i * (LC + 1) + t - 1];
            if (t + 1 <= i) val += T(t + 1) * E_c_1d[i * (LC + 1) + t + 1];
            E_c_1d[(i + 1) * (LC + 1) + t] = val;
        }
    }

    // Compute R-integrals (templated, fixed-size)
    RInts<T, L> R;
    compute_r_ints<T, L, BoysParams>(boys_table, alpha, PCx, PCy, PCz, R);

    // Prefactor: 2π^{5/2} / (p × γ × √(p+γ)) × sph_a × sph_b × sph_c
    constexpr T fac_a = spherical_harmonic_factor<T>(LA);
    constexpr T fac_b = spherical_harmonic_factor<T>(LB);
    constexpr T fac_c = spherical_harmonic_factor<T>(LC);
    const T pi_2p5 = BoysConstants<T>::pi * BoysConstants<T>::pi * std::sqrt(BoysConstants<T>::pi);
    const T prefactor = T(2) * pi_2p5 / (p * gamma * std::sqrt(pq)) * fac_a * fac_b * fac_c;

    // Contract: integrals[ab, c] = Σ_{tuv,xyz} E^ab_{tuv} × E^c_{xyz} × R_{t+x,u+y,v+z}
    // Optimized: factor x,y,z components separately
    int idx = 0;
    for (int ia = LA; ia >= 0; --ia) {
        for (int ja = LA - ia; ja >= 0; --ja) {
            int ka = LA - ia - ja;
            for (int ib = LB; ib >= 0; --ib) {
                for (int jb = LB - ib; jb >= 0; --jb) {
                    int kb = LB - ib - jb;

                    // Loop over aux Cartesian components
                    for (int ic = LC; ic >= 0; --ic) {
                        for (int jc = LC - ic; jc >= 0; --jc) {
                            int kc = LC - ic - jc;

                            // Ket sign: (-1)^(ic+jc+kc)
                            T ket_sign = ((ic + jc + kc) & 1) ? T(-1) : T(1);

                            // Contract over all Hermite indices
                            T sum = T(0);
                            for (int t = 0; t <= ia + ib; ++t) {
                                T Et = E_ab.x(ia, ib, t);
                                for (int u = 0; u <= ja + jb; ++u) {
                                    T Eu = E_ab.y(ja, jb, u);
                                    for (int v = 0; v <= ka + kb; ++v) {
                                        T Ev = E_ab.z(ka, kb, v);
                                        T E_bra = Et * Eu * Ev;

                                        for (int tx = 0; tx <= ic; ++tx) {
                                            T Ecx = E_c_1d[ic * (LC + 1) + tx];
                                            for (int ty = 0; ty <= jc; ++ty) {
                                                T Ecy = E_c_1d[jc * (LC + 1) + ty];
                                                for (int tz = 0; tz <= kc; ++tz) {
                                                    T Ecz = E_c_1d[kc * (LC + 1) + tz];
                                                    sum += E_bra * Ecx * Ecy * Ecz *
                                                           R(t + tx, u + ty, v + tz);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            integrals[idx++] = prefactor * ket_sign * sum;
                        }
                    }
                }
            }
        }
    }
}

/// 3-center ERI contracted integral (μν|P) for shell triple
template <typename T, int LA, int LB, int LC, typename BoysParams = BoysParamsDefault>
void eri3c_contracted(int na_prim, int nb_prim, int nc_prim,
                      const T* exponents_a, const T* exponents_b, const T* exponents_c,
                      const T* coeffs_a, const T* coeffs_b, const T* coeffs_c,
                      const T* A, const T* B, const T* C,
                      const T* boys_table,
                      T* integrals) {
    constexpr int na = ncart(LA);
    constexpr int nb = ncart(LB);
    constexpr int nc = ncart(LC);
    constexpr int nab = na * nb;
    constexpr int nabc = nab * nc;

    // Initialize output
    for (int i = 0; i < nabc; ++i) {
        integrals[i] = T(0);
    }

    T prim_ints[nabc];

    // Sum over primitive triples
    for (int ia = 0; ia < na_prim; ++ia) {
        T a = exponents_a[ia];
        T ca = coeffs_a[ia];

        for (int ib = 0; ib < nb_prim; ++ib) {
            T b = exponents_b[ib];
            T cb = coeffs_b[ib];

            for (int ic = 0; ic < nc_prim; ++ic) {
                T gamma = exponents_c[ic];
                T cc = coeffs_c[ic];

                eri3c_primitive<T, LA, LB, LC, BoysParams>(
                    a, b, gamma, A, B, C, boys_table, prim_ints);

                T cabc = ca * cb * cc;
                for (int i = 0; i < nabc; ++i) {
                    integrals[i] += cabc * prim_ints[i];
                }
            }
        }
    }
}

/// Dispatch to templated 3c ERI kernel based on runtime angular momenta
template <typename T, typename BoysParams = BoysParamsDefault>
void eri3c_dispatch(int la, int lb, int lc,
                    int na_prim, int nb_prim, int nc_prim,
                    const T* exponents_a, const T* exponents_b, const T* exponents_c,
                    const T* coeffs_a, const T* coeffs_b, const T* coeffs_c,
                    const T* A, const T* B, const T* C,
                    const T* boys_table,
                    T* integrals) {
    // Dispatch macro
    #define DISPATCH_3C(LA, LB, LC) \
        if (la == LA && lb == LB && lc == LC) { \
            eri3c_contracted<T, LA, LB, LC, BoysParams>( \
                na_prim, nb_prim, nc_prim, \
                exponents_a, exponents_b, exponents_c, \
                coeffs_a, coeffs_b, coeffs_c, \
                A, B, C, boys_table, integrals); \
            return; \
        }

    // s functions (LC = 0)
    DISPATCH_3C(0, 0, 0)
    DISPATCH_3C(0, 1, 0) DISPATCH_3C(1, 0, 0) DISPATCH_3C(1, 1, 0)
    DISPATCH_3C(0, 2, 0) DISPATCH_3C(2, 0, 0) DISPATCH_3C(1, 2, 0) DISPATCH_3C(2, 1, 0) DISPATCH_3C(2, 2, 0)
    DISPATCH_3C(0, 3, 0) DISPATCH_3C(3, 0, 0) DISPATCH_3C(1, 3, 0) DISPATCH_3C(3, 1, 0) DISPATCH_3C(2, 3, 0) DISPATCH_3C(3, 2, 0) DISPATCH_3C(3, 3, 0)

    // p functions (LC = 1)
    DISPATCH_3C(0, 0, 1)
    DISPATCH_3C(0, 1, 1) DISPATCH_3C(1, 0, 1) DISPATCH_3C(1, 1, 1)
    DISPATCH_3C(0, 2, 1) DISPATCH_3C(2, 0, 1) DISPATCH_3C(1, 2, 1) DISPATCH_3C(2, 1, 1) DISPATCH_3C(2, 2, 1)
    DISPATCH_3C(0, 3, 1) DISPATCH_3C(3, 0, 1) DISPATCH_3C(1, 3, 1) DISPATCH_3C(3, 1, 1) DISPATCH_3C(2, 3, 1) DISPATCH_3C(3, 2, 1) DISPATCH_3C(3, 3, 1)

    // d functions (LC = 2)
    DISPATCH_3C(0, 0, 2)
    DISPATCH_3C(0, 1, 2) DISPATCH_3C(1, 0, 2) DISPATCH_3C(1, 1, 2)
    DISPATCH_3C(0, 2, 2) DISPATCH_3C(2, 0, 2) DISPATCH_3C(1, 2, 2) DISPATCH_3C(2, 1, 2) DISPATCH_3C(2, 2, 2)
    DISPATCH_3C(0, 3, 2) DISPATCH_3C(3, 0, 2) DISPATCH_3C(1, 3, 2) DISPATCH_3C(3, 1, 2) DISPATCH_3C(2, 3, 2) DISPATCH_3C(3, 2, 2) DISPATCH_3C(3, 3, 2)

    // f functions (LC = 3)
    DISPATCH_3C(0, 0, 3)
    DISPATCH_3C(0, 1, 3) DISPATCH_3C(1, 0, 3) DISPATCH_3C(1, 1, 3)
    DISPATCH_3C(0, 2, 3) DISPATCH_3C(2, 0, 3) DISPATCH_3C(1, 2, 3) DISPATCH_3C(2, 1, 3) DISPATCH_3C(2, 2, 3)
    DISPATCH_3C(0, 3, 3) DISPATCH_3C(3, 0, 3) DISPATCH_3C(1, 3, 3) DISPATCH_3C(3, 1, 3) DISPATCH_3C(2, 3, 3) DISPATCH_3C(3, 2, 3) DISPATCH_3C(3, 3, 3)

    // g functions (LC = 4)
    DISPATCH_3C(0, 0, 4)
    DISPATCH_3C(0, 1, 4) DISPATCH_3C(1, 0, 4) DISPATCH_3C(1, 1, 4)
    DISPATCH_3C(0, 2, 4) DISPATCH_3C(2, 0, 4) DISPATCH_3C(1, 2, 4) DISPATCH_3C(2, 1, 4) DISPATCH_3C(2, 2, 4)
    DISPATCH_3C(0, 3, 4) DISPATCH_3C(3, 0, 4) DISPATCH_3C(1, 3, 4) DISPATCH_3C(3, 1, 4) DISPATCH_3C(2, 3, 4) DISPATCH_3C(3, 2, 4) DISPATCH_3C(3, 3, 4)

    // h functions (LC = 5) - common in aux bases
    DISPATCH_3C(0, 0, 5)
    DISPATCH_3C(0, 1, 5) DISPATCH_3C(1, 0, 5) DISPATCH_3C(1, 1, 5)
    DISPATCH_3C(0, 2, 5) DISPATCH_3C(2, 0, 5) DISPATCH_3C(1, 2, 5) DISPATCH_3C(2, 1, 5) DISPATCH_3C(2, 2, 5)

    // i functions (LC = 6) - some aux bases
    DISPATCH_3C(0, 0, 6)
    DISPATCH_3C(0, 1, 6) DISPATCH_3C(1, 0, 6) DISPATCH_3C(1, 1, 6)
    DISPATCH_3C(0, 2, 6) DISPATCH_3C(2, 0, 6)

    #undef DISPATCH_3C

    // Fallback to dynamic version (should rarely happen)
    throw std::runtime_error("eri3c_dispatch: unsupported angular momentum combination: " +
                            std::to_string(la) + ", " + std::to_string(lb) + ", " + std::to_string(lc));
}

} // namespace occ::ints
