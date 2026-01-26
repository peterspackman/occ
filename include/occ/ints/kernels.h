#pragma once
#include <occ/ints/boys.h>
#include <occ/ints/ecoeffs.h>
#include <occ/ints/rints.h>
#include <Eigen/Dense>
#include <array>
#include <vector>

namespace occ::ints {

// Spherical harmonic normalization factors
// s: 1/(2*sqrt(pi)), p: sqrt(3/(4*pi)), d+: 1
template <typename T>
OCC_GPU_ENABLED OCC_GPU_INLINE
constexpr T spherical_harmonic_factor(int l) {
    return (l == 0) ? T(0.282094791773878143) :
           (l == 1) ? T(0.488602511902919921) : T(1);
}

// ============================================================================
// ESP Primitive and Contracted Kernels
// ============================================================================

/// ESP integral for a single primitive pair at a single point
///
/// @param a        Exponent of first primitive
/// @param b        Exponent of second primitive
/// @param A        Center of first primitive [3]
/// @param B        Center of second primitive [3]
/// @param C        Point at which to evaluate potential [3]
/// @param boys_table  Boys function interpolation table
/// @param integrals   Output: (la+1)(la+2)/2 × (lb+1)(lb+2)/2 integrals
///
/// The output layout is [Cartesian_a][Cartesian_b] in standard order
template <typename T, int LA, int LB, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED
void esp_primitive(T a, T b,
                   const T* A, const T* B, const T* C,
                   const T* boys_table,
                   T* integrals) {
    constexpr int L = LA + LB;
    constexpr int na = ncart(LA);
    constexpr int nb = ncart(LB);

    // Combined exponent and center
    const T p = a + b;
    const T Px = (a * A[0] + b * B[0]) / p;
    const T Py = (a * A[1] + b * B[1]) / p;
    const T Pz = (a * A[2] + b * B[2]) / p;

    // Vector from P to C
    const T PCx = Px - C[0];
    const T PCy = Py - C[1];
    const T PCz = Pz - C[2];

    // Compute E-coefficients
    ECoeffs3D<T, LA, LB> E;
    compute_e_coeffs_3d<T, LA, LB>(a, b, A, B, E);

    // Compute R-integrals
    RInts<T, L> R;
    compute_r_ints<T, L, BoysParams>(boys_table, p, PCx, PCy, PCz, R);

    // Prefactor: 2π/p × spherical_harmonic_factor(LA) × spherical_harmonic_factor(LB)
    constexpr T fac_a = spherical_harmonic_factor<T>(LA);
    constexpr T fac_b = spherical_harmonic_factor<T>(LB);
    const T prefactor = T(2) * BoysConstants<T>::pi / p * fac_a * fac_b;

    // Contract E-coefficients with R-integrals over Cartesian components
    int idx_ab = 0;
    for (int ia = LA; ia >= 0; --ia) {
        for (int ja = LA - ia; ja >= 0; --ja) {
            int ka = LA - ia - ja;
            for (int ib = LB; ib >= 0; --ib) {
                for (int jb = LB - ib; jb >= 0; --jb) {
                    int kb = LB - ib - jb;

                    T sum = T(0);
                    for (int t = 0; t <= ia + ib; ++t) {
                        T Et = E.x(ia, ib, t);
                        for (int u = 0; u <= ja + jb; ++u) {
                            T Eu = E.y(ja, jb, u);
                            for (int v = 0; v <= ka + kb; ++v) {
                                T Ev = E.z(ka, kb, v);
                                sum += Et * Eu * Ev * R(t, u, v);
                            }
                        }
                    }
                    integrals[idx_ab++] = prefactor * sum;
                }
            }
        }
    }
}

/// ESP integral for contracted shell pair at a single point
template <typename T, int LA, int LB, typename BoysParams = BoysParamsDefault>
OCC_GPU_ENABLED
void esp_contracted(int na_prim, int nb_prim,
                    const T* exponents_a, const T* exponents_b,
                    const T* coeffs_a, const T* coeffs_b,
                    const T* A, const T* B, const T* C,
                    const T* boys_table,
                    T* integrals) {
    constexpr int na = ncart(LA);
    constexpr int nb = ncart(LB);

    // Initialize output to zero
    for (int i = 0; i < na * nb; ++i) {
        integrals[i] = T(0);
    }

    // Temporary for primitive integrals
    T prim_ints[na * nb];

    // Sum over primitive pairs
    for (int ia = 0; ia < na_prim; ++ia) {
        T a = exponents_a[ia];
        T ca = coeffs_a[ia];

        for (int ib = 0; ib < nb_prim; ++ib) {
            T b = exponents_b[ib];
            T cb = coeffs_b[ib];

            // Compute primitive integrals
            esp_primitive<T, LA, LB, BoysParams>(a, b, A, B, C, boys_table, prim_ints);

            // Accumulate with contraction coefficients
            T cab = ca * cb;
            for (int i = 0; i < na * nb; ++i) {
                integrals[i] += cab * prim_ints[i];
            }
        }
    }
}

/// ESP integrals for a shell pair at multiple points (vectorized over points)
/// SHARK-style: E-coefficients computed once per primitive pair, reused for all points
template <typename T, int LA, int LB, typename BoysParams = BoysParamsDefault>
void esp_contracted_batch(int na_prim, int nb_prim,
                          const T* exponents_a, const T* exponents_b,
                          const T* coeffs_a, const T* coeffs_b,
                          const T* A, const T* B,
                          int npts, const T* C,
                          const T* boys_table,
                          T* integrals) {
    constexpr int L = LA + LB;
    constexpr int na = ncart(LA);
    constexpr int nb = ncart(LB);
    constexpr int nab = na * nb;

    // Spherical harmonic factors
    constexpr T fac_a = spherical_harmonic_factor<T>(LA);
    constexpr T fac_b = spherical_harmonic_factor<T>(LB);
    constexpr T sph_fac = fac_a * fac_b;

    // Initialize output to zero
    for (int i = 0; i < npts * nab; ++i) {
        integrals[i] = T(0);
    }

    // SHARK-style: precompute E-coefficients per primitive pair
    for (int prim_a = 0; prim_a < na_prim; ++prim_a) {
        const T a = exponents_a[prim_a];
        const T ca = coeffs_a[prim_a];

        for (int prim_b = 0; prim_b < nb_prim; ++prim_b) {
            const T b_exp = exponents_b[prim_b];
            const T cb = coeffs_b[prim_b];
            const T cab = ca * cb;

            const T p = a + b_exp;
            const T Px = (a * A[0] + b_exp * B[0]) / p;
            const T Py = (a * A[1] + b_exp * B[1]) / p;
            const T Pz = (a * A[2] + b_exp * B[2]) / p;

            const T prefactor = T(2) * BoysConstants<T>::pi / p * sph_fac * cab;

            // E-coefficients (independent of C)
            ECoeffs3D<T, LA, LB> E;
            compute_e_coeffs_3d<T, LA, LB>(a, b_exp, A, B, E);

            // Process all grid points with same E-coefficients
            for (int pt = 0; pt < npts; ++pt) {
                const T* Cpt = C + 3 * pt;
                T* out = integrals + pt * nab;

                const T PCx = Px - Cpt[0];
                const T PCy = Py - Cpt[1];
                const T PCz = Pz - Cpt[2];

                RInts<T, L> R;
                compute_r_ints<T, L, BoysParams>(boys_table, p, PCx, PCy, PCz, R);

                int idx_ab = 0;
                for (int ia = LA; ia >= 0; --ia) {
                    for (int ja = LA - ia; ja >= 0; --ja) {
                        int ka = LA - ia - ja;
                        for (int ib = LB; ib >= 0; --ib) {
                            for (int jb = LB - ib; jb >= 0; --jb) {
                                int kb = LB - ib - jb;

                                T sum = T(0);
                                for (int t = 0; t <= ia + ib; ++t) {
                                    T Et = E.x(ia, ib, t);
                                    for (int u = 0; u <= ja + jb; ++u) {
                                        T Eu = E.y(ja, jb, u);
                                        for (int v = 0; v <= ka + kb; ++v) {
                                            T Ev = E.z(ka, kb, v);
                                            sum += Et * Eu * Ev * R(t, u, v);
                                        }
                                    }
                                }
                                out[idx_ab++] += prefactor * sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Precomputed Shell Pair Data Structures
// ============================================================================

/// Data for a single primitive pair (precomputed at initialization)
template <typename T>
struct PrimitivePairData {
    T p;              // Combined exponent (a + b)
    T Px, Py, Pz;     // Product center
    T prefactor;      // 2π/p × sph_fac × ca × cb

    // E-matrix flattened: [nab, nherm] stored row-major
    std::vector<T> E_matrix;

    int nab;          // Number of basis function pairs
    int nherm;        // Number of Hermite indices
};

/// Data for a shell pair (all primitive pairs)
template <typename T>
struct ShellPairData {
    int la, lb;                              // Angular momenta
    std::array<T, 3> A, B;                   // Shell centers
    std::vector<PrimitivePairData<T>> primitives;

    int nab() const { return ncart(la) * ncart(lb); }
    int nherm() const { return nhermsum(la + lb); }
};

/// Build E-matrix for a primitive pair
template <typename T, int LA, int LB>
void build_e_matrix(const ECoeffs3D<T, LA, LB>& E, T* E_matrix) {
    constexpr int L = LA + LB;
    constexpr int na = ncart(LA);
    constexpr int nb = ncart(LB);
    constexpr int nherm = nhermsum(L);

    for (int i = 0; i < na * nb * nherm; ++i) {
        E_matrix[i] = T(0);
    }

    int row = 0;
    for (int ia = LA; ia >= 0; --ia) {
        for (int ja = LA - ia; ja >= 0; --ja) {
            int ka = LA - ia - ja;
            for (int ib = LB; ib >= 0; --ib) {
                for (int jb = LB - ib; jb >= 0; --jb) {
                    int kb = LB - ib - jb;

                    for (int t = 0; t <= ia + ib; ++t) {
                        T Et = E.x(ia, ib, t);
                        for (int u = 0; u <= ja + jb; ++u) {
                            T Eu = E.y(ja, jb, u);
                            for (int v = 0; v <= ka + kb; ++v) {
                                T Ev = E.z(ka, kb, v);
                                int col = hermite_index(t, u, v);
                                E_matrix[row * nherm + col] = Et * Eu * Ev;
                            }
                        }
                    }
                    ++row;
                }
            }
        }
    }
}

/// Precompute ShellPairData for a shell pair
template <typename T, int LA, int LB>
ShellPairData<T> precompute_shell_pair(
    int na_prim, int nb_prim,
    const T* exponents_a, const T* exponents_b,
    const T* coeffs_a, const T* coeffs_b,
    const T* A, const T* B)
{
    constexpr int L = LA + LB;
    constexpr int na = ncart(LA);
    constexpr int nb = ncart(LB);
    constexpr int nab = na * nb;
    constexpr int nherm = nhermsum(L);

    constexpr T fac_a = spherical_harmonic_factor<T>(LA);
    constexpr T fac_b = spherical_harmonic_factor<T>(LB);
    constexpr T sph_fac = fac_a * fac_b;

    ShellPairData<T> data;
    data.la = LA;
    data.lb = LB;
    data.A = {A[0], A[1], A[2]};
    data.B = {B[0], B[1], B[2]};

    data.primitives.reserve(na_prim * nb_prim);

    for (int ia = 0; ia < na_prim; ++ia) {
        const T a = exponents_a[ia];
        const T ca = coeffs_a[ia];

        for (int ib = 0; ib < nb_prim; ++ib) {
            const T b = exponents_b[ib];
            const T cb = coeffs_b[ib];

            PrimitivePairData<T> prim;
            prim.p = a + b;
            prim.Px = (a * A[0] + b * B[0]) / prim.p;
            prim.Py = (a * A[1] + b * B[1]) / prim.p;
            prim.Pz = (a * A[2] + b * B[2]) / prim.p;
            prim.prefactor = T(2) * BoysConstants<T>::pi / prim.p * sph_fac * ca * cb;
            prim.nab = nab;
            prim.nherm = nherm;

            prim.E_matrix.resize(nab * nherm);
            ECoeffs3D<T, LA, LB> E;
            compute_e_coeffs_3d<T, LA, LB>(a, b, A, B, E);
            build_e_matrix<T, LA, LB>(E, prim.E_matrix.data());

            data.primitives.push_back(std::move(prim));
        }
    }

    return data;
}

/// Calculate workspace size for esp_evaluate_with_precomputed
template <int LA, int LB>
constexpr size_t esp_workspace_size(int npts) {
    constexpr int L = LA + LB;
    constexpr int nherm = nhermsum(L);
    return npts * nherm;  // R_matrix: [npts, nherm]
}

/// ESP evaluation using precomputed ShellPairData
/// This is the hot path - E-matrices are already computed!
///
/// @param shell_pair  Precomputed shell pair data
/// @param npts        Number of grid points
/// @param C           Grid point coordinates [3 * npts]
/// @param boys_table  Boys function interpolation table
/// @param integrals   Output buffer [npts * nab]
/// @param workspace   Pre-allocated workspace (size >= esp_workspace_size<LA,LB>(npts))
template <typename T, int LA, int LB, typename BoysParams = BoysParamsDefault>
void esp_evaluate_with_precomputed(
    const ShellPairData<T>& shell_pair,
    int npts, const T* C,
    const T* boys_table,
    T* integrals,
    T* workspace)
{
    constexpr int L = LA + LB;
    constexpr int na = ncart(LA);
    constexpr int nb = ncart(LB);
    constexpr int nab = na * nb;
    constexpr int nherm_val = nhermsum(L);

    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Output: [npts, nab]
    Eigen::Map<MatRM> result(integrals, npts, nab);
    result.setZero();

    // R-integrals buffer: [npts, nherm] - use provided workspace
    Eigen::Map<MatRM> R_matrix(workspace, npts, nherm_val);

    // Process each precomputed primitive pair
    for (const auto& prim : shell_pair.primitives) {
        // Compute R-integrals for all grid points
        for (int pt = 0; pt < npts; ++pt) {
            const T* Cpt = C + 3 * pt;
            const T PCx = prim.Px - Cpt[0];
            const T PCy = prim.Py - Cpt[1];
            const T PCz = prim.Pz - Cpt[2];

            RInts<T, L> R;
            compute_r_ints<T, L, BoysParams>(boys_table, prim.p, PCx, PCy, PCz, R);

            for (int h = 0; h < nherm_val; ++h) {
                R_matrix(pt, h) = R.data[h];
            }
        }

        // Matrix multiply: result += prefactor * R_matrix @ E^T
        Eigen::Map<const MatRM> E_matrix(prim.E_matrix.data(), nab, nherm_val);
        result.noalias() += prim.prefactor * R_matrix * E_matrix.transpose();
    }
}

// ============================================================================
// Auxiliary Shell Data Structures (for Split-RI-J)
// ============================================================================

/// Data for a single auxiliary primitive (single-center)
template <typename T>
struct AuxPrimitiveData {
    T gamma;          // Exponent
    T coeff;          // Contraction coefficient
    std::vector<T> E_matrix;  // [nc, nherm] E-matrix for single center
    int nc;           // Number of Cartesian functions
    int nherm;        // Number of Hermite indices
};

/// Data for an auxiliary shell
template <typename T>
struct AuxShellData {
    int lc;                       // Angular momentum
    std::array<T, 3> C;           // Center
    std::vector<AuxPrimitiveData<T>> primitives;

    int nc() const { return ncart(lc); }
    int nherm() const { return nhermsum(lc); }
};

/// Build 1D E-coefficients for single-center auxiliary with exponent gamma
/// Uses recurrence: E^{l+1}_t = (1/2γ) E^l_{t-1} + (t+1) E^l_{t+1}
/// @param lc         Angular momentum
/// @param gamma      Exponent
/// @param E_1d       Output: [(lc+1) × (lc+1)] 1D E-coefficients
template <typename T>
void build_aux_e_1d(int lc, T gamma, T* E_1d) {
    const T half_gamma_inv = T(0.5) / gamma;
    const int stride = lc + 1;

    // Initialize to zero
    for (int i = 0; i <= lc; ++i) {
        for (int t = 0; t <= lc; ++t) {
            E_1d[i * stride + t] = T(0);
        }
    }

    // E^{0}_0 = 1
    E_1d[0] = T(1);

    // Build higher angular momenta using recurrence
    for (int i = 0; i < lc; ++i) {
        for (int t = 0; t <= i + 1; ++t) {
            T val = T(0);
            if (t > 0) val += half_gamma_inv * E_1d[i * stride + t - 1];
            if (t + 1 <= i) val += T(t + 1) * E_1d[i * stride + t + 1];
            E_1d[(i + 1) * stride + t] = val;
        }
    }
}

/// Build E-matrix for single-center auxiliary shell with ket signs
/// @param E_1d       1D E-coefficients from build_aux_e_1d
/// @param lc         Angular momentum
/// @param E_matrix   Output: [nc × nherm] E-matrix with ket signs baked in
template <typename T>
void build_aux_e_matrix_with_gamma(const T* E_1d, int lc, T* E_matrix) {
    const int nc = ncart(lc);
    const int nherm = nhermsum(lc);
    const int stride = lc + 1;

    // Initialize to zero
    for (int i = 0; i < nc * nherm; ++i) {
        E_matrix[i] = T(0);
    }

    // Build E_c matrix with ket signs (-1)^(tx+ty+tz)
    int row = 0;
    for (int ic = lc; ic >= 0; --ic) {
        for (int jc = lc - ic; jc >= 0; --jc) {
            int kc = lc - ic - jc;

            for (int tx = 0; tx <= ic; ++tx) {
                T Ecx = E_1d[ic * stride + tx];
                for (int ty = 0; ty <= jc; ++ty) {
                    T Ecy = E_1d[jc * stride + ty];
                    for (int tz = 0; tz <= kc; ++tz) {
                        T Ecz = E_1d[kc * stride + tz];
                        T ket_sign = ((tx + ty + tz) & 1) ? T(-1) : T(1);
                        int h = hermite_index(tx, ty, tz);
                        E_matrix[row * nherm + h] = ket_sign * Ecx * Ecy * Ecz;
                    }
                }
            }
            ++row;
        }
    }
}

/// Precompute auxiliary shell data with gamma-dependent E-coefficients
/// @param nprim      Number of primitives
/// @param exponents  Primitive exponents [nprim]
/// @param coeffs     Contraction coefficients [nprim]
/// @param C          Shell center [3]
template <typename T, int L>
AuxShellData<T> precompute_aux_shell(
    int nprim, const T* exponents, const T* coeffs, const T* C)
{
    constexpr int nc = ncart(L);
    constexpr int nherm = nhermsum(L);

    AuxShellData<T> data;
    data.lc = L;
    data.C = {C[0], C[1], C[2]};
    data.primitives.reserve(nprim);

    // Temporary buffer for 1D E-coefficients
    T E_1d[(L + 1) * (L + 1)];

    for (int ip = 0; ip < nprim; ++ip) {
        AuxPrimitiveData<T> prim;
        prim.gamma = exponents[ip];
        prim.coeff = coeffs[ip];
        prim.nc = nc;
        prim.nherm = nherm;
        prim.E_matrix.resize(nc * nherm);

        // Build E-matrix with gamma-dependent coefficients
        build_aux_e_1d<T>(L, prim.gamma, E_1d);
        build_aux_e_matrix_with_gamma<T>(E_1d, L, prim.E_matrix.data());

        data.primitives.push_back(std::move(prim));
    }

    return data;
}

// ============================================================================
// Dispatch Functions for Runtime Angular Momentum
// ============================================================================

/// Dispatch shell pair precomputation based on runtime angular momenta
/// Supports angular momenta up to L=4 (g functions)
template <typename T>
ShellPairData<T> precompute_shell_pair_dispatch(
    int la, int lb,
    int na_prim, int nb_prim,
    const T* exponents_a, const T* exponents_b,
    const T* coeffs_a, const T* coeffs_b,
    const T* A, const T* B)
{
    // Dispatch based on (la, lb) combination
    #define DISPATCH_PAIR(LA, LB) \
        if (la == LA && lb == LB) \
            return precompute_shell_pair<T, LA, LB>(na_prim, nb_prim, \
                exponents_a, exponents_b, coeffs_a, coeffs_b, A, B)

    // s-s through g-g (all 25 combinations)
    DISPATCH_PAIR(0, 0);
    DISPATCH_PAIR(0, 1); DISPATCH_PAIR(1, 0); DISPATCH_PAIR(1, 1);
    DISPATCH_PAIR(0, 2); DISPATCH_PAIR(1, 2); DISPATCH_PAIR(2, 0); DISPATCH_PAIR(2, 1); DISPATCH_PAIR(2, 2);
    DISPATCH_PAIR(0, 3); DISPATCH_PAIR(1, 3); DISPATCH_PAIR(2, 3); DISPATCH_PAIR(3, 0); DISPATCH_PAIR(3, 1); DISPATCH_PAIR(3, 2); DISPATCH_PAIR(3, 3);
    DISPATCH_PAIR(0, 4); DISPATCH_PAIR(1, 4); DISPATCH_PAIR(2, 4); DISPATCH_PAIR(3, 4); DISPATCH_PAIR(4, 0); DISPATCH_PAIR(4, 1); DISPATCH_PAIR(4, 2); DISPATCH_PAIR(4, 3); DISPATCH_PAIR(4, 4);

    #undef DISPATCH_PAIR

    // Fallback: throw for unsupported angular momenta
    throw std::runtime_error("Unsupported angular momentum pair: la=" +
        std::to_string(la) + ", lb=" + std::to_string(lb));
}

/// Dispatch auxiliary shell precomputation based on runtime angular momentum
template <typename T>
AuxShellData<T> precompute_aux_shell_dispatch(
    int lc,
    int nprim, const T* exponents, const T* coeffs, const T* C)
{
    switch (lc) {
        case 0: return precompute_aux_shell<T, 0>(nprim, exponents, coeffs, C);
        case 1: return precompute_aux_shell<T, 1>(nprim, exponents, coeffs, C);
        case 2: return precompute_aux_shell<T, 2>(nprim, exponents, coeffs, C);
        case 3: return precompute_aux_shell<T, 3>(nprim, exponents, coeffs, C);
        case 4: return precompute_aux_shell<T, 4>(nprim, exponents, coeffs, C);
        case 5: return precompute_aux_shell<T, 5>(nprim, exponents, coeffs, C);
        case 6: return precompute_aux_shell<T, 6>(nprim, exponents, coeffs, C);
        default:
            throw std::runtime_error("Unsupported auxiliary angular momentum: " +
                std::to_string(lc));
    }
}

// ============================================================================
// 3-Center and 2-Center ERI Kernels (for Split-RI-J)
// ============================================================================

/// 3-center ERI (μν|P) using precomputed data
///
/// Computes: (μν|P) = Σ_prims (2π^{5/2})/(p*q*√(p+q)) × ca×cb×cc
///                    × Σ_{tuv,xyz} E^{ab}_{tuv} × E^c_{xyz} × R_{t+x,u+y,v+z}(α, PC_scaled)
/// where p = α_a + α_b, q = γ_c, α = pq/(p+q)
///
/// @param ao_pair    Precomputed AO shell pair data
/// @param aux_shell  Precomputed auxiliary shell data
/// @param boys_table Boys function interpolation table
/// @param integrals  Output: [nab, nc] integrals
template <typename T, int LA, int LB, int LC, typename BoysParams = BoysParamsDefault>
void eri3c_evaluate_with_precomputed(
    const ShellPairData<T>& ao_pair,
    const AuxShellData<T>& aux_shell,
    const T* boys_table,
    T* integrals)
{
    constexpr int L_total = LA + LB + LC;
    constexpr int nab = ncart(LA) * ncart(LB);
    constexpr int nc = ncart(LC);
    constexpr int nherm_ab = nhermsum(LA + LB);
    constexpr int nherm_c = nhermsum(LC);

    // Initialize output to zero
    for (int i = 0; i < nab * nc; ++i) {
        integrals[i] = T(0);
    }

    // Loop over AO primitive pairs
    for (const auto& ao_prim : ao_pair.primitives) {
        const T p = ao_prim.p;
        const T Px = ao_prim.Px;
        const T Py = ao_prim.Py;
        const T Pz = ao_prim.Pz;

        // Loop over auxiliary primitives
        for (const auto& aux_prim : aux_shell.primitives) {
            const T q = aux_prim.gamma;
            const T cc = aux_prim.coeff;

            // Combined quantities
            const T pq = p + q;
            const T alpha = p * q / pq;

            // Distance from product center P to auxiliary center C
            const T PCx = Px - aux_shell.C[0];
            const T PCy = Py - aux_shell.C[1];
            const T PCz = Pz - aux_shell.C[2];

            // Scaled distance for R-integrals
            const T scale = alpha / p;
            const T Rx = scale * PCx;
            const T Ry = scale * PCy;
            const T Rz = scale * PCz;

            // Prefactor: (2π^{5/2}) / (p * q * √(p+q)) × prefactor_ab × cc
            // where prefactor_ab already contains 2π/p × ca × cb
            // So total prefactor = prefactor_ab × cc × π^{3/2} / (q * √(p+q))
            const T prefactor = ao_prim.prefactor * cc *
                std::pow(BoysConstants<T>::pi / pq, T(1.5));

            // Compute R-integrals
            RInts<T, L_total> R;
            compute_r_ints<T, L_total, BoysParams>(boys_table, alpha, Rx, Ry, Rz, R);

            // Contract E-matrices with R-integrals
            // (μν|P) = prefactor × Σ_{tuv} E^{ab}_{tuv} × Σ_{xyz} E^c_{xyz} × R_{t+x,u+y,v+z}
            const T* E_ab = ao_prim.E_matrix.data();
            const T* E_c = aux_prim.E_matrix.data();

            for (int ab = 0; ab < nab; ++ab) {
                for (int c = 0; c < nc; ++c) {
                    T sum = T(0);

                    // Sum over Hermite indices for AO pair
                    for (int t = 0; t <= LA + LB; ++t) {
                        for (int u = 0; u <= LA + LB - t; ++u) {
                            for (int v = 0; v <= LA + LB - t - u; ++v) {
                                int h_ab = hermite_index(t, u, v);
                                T E_ab_h = E_ab[ab * nherm_ab + h_ab];
                                if (E_ab_h == T(0)) continue;

                                // Sum over Hermite indices for aux
                                for (int tx = 0; tx <= LC; ++tx) {
                                    for (int ux = 0; ux <= LC - tx; ++ux) {
                                        for (int vx = 0; vx <= LC - tx - ux; ++vx) {
                                            int h_c = hermite_index(tx, ux, vx);
                                            T E_c_h = E_c[c * nherm_c + h_c];
                                            if (E_c_h == T(0)) continue;

                                            sum += E_ab_h * E_c_h *
                                                   R(t + tx, u + ux, v + vx);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    integrals[ab * nc + c] += prefactor * sum;
                }
            }
        }
    }
}

/// 2-center Coulomb (P|Q) using precomputed data
///
/// Computes: (P|Q) = Σ_prims (2π)/(p+q) × cp × cq × Σ_{tuv} E^{PQ}_{tuv} × R_{tuv}(p+q, P-Q)
///
/// @param shell_P    Precomputed auxiliary shell P data
/// @param shell_Q    Precomputed auxiliary shell Q data
/// @param boys_table Boys function interpolation table
/// @param integrals  Output: [nc_P, nc_Q] integrals
template <typename T, int LP, int LQ, typename BoysParams = BoysParamsDefault>
void eri2c_evaluate_with_precomputed(
    const AuxShellData<T>& shell_P,
    const AuxShellData<T>& shell_Q,
    const T* boys_table,
    T* integrals)
{
    constexpr int L = LP + LQ;
    constexpr int nc_P = ncart(LP);
    constexpr int nc_Q = ncart(LQ);

    // Initialize output to zero
    for (int i = 0; i < nc_P * nc_Q; ++i) {
        integrals[i] = T(0);
    }

    // Centers
    const T Px = shell_P.C[0], Py = shell_P.C[1], Pz = shell_P.C[2];
    const T Qx = shell_Q.C[0], Qy = shell_Q.C[1], Qz = shell_Q.C[2];

    // Loop over primitive pairs
    for (const auto& prim_P : shell_P.primitives) {
        const T ap = prim_P.gamma;
        const T cp = prim_P.coeff;

        for (const auto& prim_Q : shell_Q.primitives) {
            const T aq = prim_Q.gamma;
            const T cq = prim_Q.coeff;

            const T p = ap + aq;
            const T prefactor = T(2) * BoysConstants<T>::pi / p * cp * cq;

            // Weighted center
            const T Wx = (ap * Px + aq * Qx) / p;
            const T Wy = (ap * Py + aq * Qy) / p;
            const T Wz = (ap * Pz + aq * Qz) / p;

            // Distance from weighted center to Q
            const T WQx = Wx - Qx;
            const T WQy = Wy - Qy;
            const T WQz = Wz - Qz;

            // Compute R-integrals
            RInts<T, L> R;
            compute_r_ints<T, L, BoysParams>(boys_table, p, WQx, WQy, WQz, R);

            // Compute E-coefficients for the pair
            ECoeffs3D<T, LP, LQ> E;
            T A[3] = {Px, Py, Pz};
            T B[3] = {Qx, Qy, Qz};
            compute_e_coeffs_3d<T, LP, LQ>(ap, aq, A, B, E);

            // Contract
            int idx = 0;
            for (int ip = LP; ip >= 0; --ip) {
                for (int jp = LP - ip; jp >= 0; --jp) {
                    int kp = LP - ip - jp;

                    for (int iq = LQ; iq >= 0; --iq) {
                        for (int jq = LQ - iq; jq >= 0; --jq) {
                            int kq = LQ - iq - jq;

                            T sum = T(0);
                            for (int t = 0; t <= ip + iq; ++t) {
                                T Et = E.x(ip, iq, t);
                                for (int u = 0; u <= jp + jq; ++u) {
                                    T Eu = E.y(jp, jq, u);
                                    for (int v = 0; v <= kp + kq; ++v) {
                                        T Ev = E.z(kp, kq, v);
                                        sum += Et * Eu * Ev * R(t, u, v);
                                    }
                                }
                            }
                            integrals[idx++] += prefactor * sum;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Dynamic Angular Momentum Versions
// ============================================================================

/// ESP integral with runtime angular momentum
template <typename T, typename BoysParams = BoysParamsDefault>
void esp_primitive_dynamic(int la, int lb,
                           T a, T b,
                           const T* A, const T* B, const T* C,
                           const T* boys_table,
                           T* integrals) {
    const int L = la + lb;
    const int na = ncart(la);
    const int nb = ncart(lb);

    const T p = a + b;
    const T Px = (a * A[0] + b * B[0]) / p;
    const T Py = (a * A[1] + b * B[1]) / p;
    const T Pz = (a * A[2] + b * B[2]) / p;

    const T PCx = Px - C[0];
    const T PCy = Py - C[1];
    const T PCz = Pz - C[2];

    ECoeffs1DDynamic<T> Ex, Ey, Ez;
    compute_e_coeffs_1d_dynamic(la, lb, a, b, B[0] - A[0], Ex);
    compute_e_coeffs_1d_dynamic(la, lb, a, b, B[1] - A[1], Ey);
    compute_e_coeffs_1d_dynamic(la, lb, a, b, B[2] - A[2], Ez);

    RIntsDynamic<T> R;
    compute_r_ints_dynamic<T, BoysParams>(boys_table, L, p, PCx, PCy, PCz, R);

    const T fac_a = spherical_harmonic_factor<T>(la);
    const T fac_b = spherical_harmonic_factor<T>(lb);
    const T prefactor = T(2) * BoysConstants<T>::pi / p * fac_a * fac_b;

    int idx_ab = 0;
    for (int i_a = la; i_a >= 0; --i_a) {
        for (int j_a = la - i_a; j_a >= 0; --j_a) {
            int k_a = la - i_a - j_a;

            for (int i_b = lb; i_b >= 0; --i_b) {
                for (int j_b = lb - i_b; j_b >= 0; --j_b) {
                    int k_b = lb - i_b - j_b;

                    T sum = T(0);

                    for (int t = 0; t <= i_a + i_b; ++t) {
                        T Et = Ex(i_a, i_b, t);
                        for (int u = 0; u <= j_a + j_b; ++u) {
                            T Eu = Ey(j_a, j_b, u);
                            for (int v = 0; v <= k_a + k_b; ++v) {
                                T Ev = Ez(k_a, k_b, v);
                                sum += Et * Eu * Ev * R(t, u, v);
                            }
                        }
                    }

                    integrals[idx_ab++] = prefactor * sum;
                }
            }
        }
    }
}

/// Contracted ESP with runtime angular momentum
template <typename T, typename BoysParams = BoysParamsDefault>
void esp_contracted_dynamic(int la, int lb,
                            int na_prim, int nb_prim,
                            const T* exponents_a, const T* exponents_b,
                            const T* coeffs_a, const T* coeffs_b,
                            const T* A, const T* B, const T* C,
                            const T* boys_table,
                            T* integrals) {
    const int na = ncart(la);
    const int nb = ncart(lb);
    const int nab = na * nb;

    for (int i = 0; i < nab; ++i) {
        integrals[i] = T(0);
    }

    T prim_ints[RINTS_MAX_SIZE];

    for (int ia = 0; ia < na_prim; ++ia) {
        T ea = exponents_a[ia];
        T ca = coeffs_a[ia];

        for (int ib = 0; ib < nb_prim; ++ib) {
            T eb = exponents_b[ib];
            T cb = coeffs_b[ib];

            esp_primitive_dynamic<T, BoysParams>(
                la, lb, ea, eb, A, B, C, boys_table, prim_ints
            );

            T cab = ca * cb;
            for (int i = 0; i < nab; ++i) {
                integrals[i] += cab * prim_ints[i];
            }
        }
    }
}

// ============================================================================
// Dynamic Versions of 3-Center and 2-Center ERIs
// ============================================================================

/// Dynamic 3-center ERI (μν|P) for runtime angular momentum
template <typename T, typename BoysParams = BoysParamsDefault>
void eri3c_dynamic(
    int la, int lb, int lc,
    int na_prim, int nb_prim, int nc_prim,
    const T* exponents_a, const T* exponents_b, const T* exponents_c,
    const T* coeffs_a, const T* coeffs_b, const T* coeffs_c,
    const T* A, const T* B, const T* C,
    const T* boys_table,
    T* integrals)
{
    const int L = la + lb + lc;
    const int na = ncart(la);
    const int nb = ncart(lb);
    const int nc = ncart(lc);
    const int nab = na * nb;

    // Initialize output
    for (int i = 0; i < nab * nc; ++i) {
        integrals[i] = T(0);
    }

    // Loop over primitives
    for (int ia = 0; ia < na_prim; ++ia) {
        const T a = exponents_a[ia];
        const T ca = coeffs_a[ia];

        for (int ib = 0; ib < nb_prim; ++ib) {
            const T b = exponents_b[ib];
            const T cb = coeffs_b[ib];

            // AO pair quantities
            const T p = a + b;
            const T Px = (a * A[0] + b * B[0]) / p;
            const T Py = (a * A[1] + b * B[1]) / p;
            const T Pz = (a * A[2] + b * B[2]) / p;

            // E-coefficients for AO pair (includes K_ab = exp(-μ_ab × |A-B|²) internally)
            ECoeffs1DDynamic<T> Ex_ab, Ey_ab, Ez_ab;
            compute_e_coeffs_1d_dynamic(la, lb, a, b, B[0] - A[0], Ex_ab);
            compute_e_coeffs_1d_dynamic(la, lb, a, b, B[1] - A[1], Ey_ab);
            compute_e_coeffs_1d_dynamic(la, lb, a, b, B[2] - A[2], Ez_ab);

            for (int ic = 0; ic < nc_prim; ++ic) {
                const T gamma = exponents_c[ic];
                const T cc = coeffs_c[ic];

                // Combined quantities
                const T pq = p + gamma;
                const T alpha = p * gamma / pq;

                // Distance from P to C (used directly for R-integrals)
                const T PCx = Px - C[0];
                const T PCy = Py - C[1];
                const T PCz = Pz - C[2];

                // Spherical harmonic factors (OCC convention: s=1/(2√π), p=√(3/(4π)), L≥2=1.0)
                auto sph_factor = [](int l) -> T {
                    if (l == 0) return T(0.28209479177387814);  // 1/(2√π)
                    if (l == 1) return T(0.48860251190291992);  // √(3/(4π))
                    return T(1.0);
                };
                const T sph_a = sph_factor(la);
                const T sph_b = sph_factor(lb);
                const T sph_c = sph_factor(lc);

                // Prefactor for 3c ERI: 2π^{5/2} / (p × γ × √(p+γ)) × ca × cb × cc × sph_factors
                // From working GPU Split-RI-J in occ-gints/src/metal/split_ri_j_metal.mm
                // K_ab is already in E-coefficients (from ecoeffs.h line 51: E00_0 = exp(-mu*XAB²))
                const T pi_2p5 = BoysConstants<T>::pi * BoysConstants<T>::pi * std::sqrt(BoysConstants<T>::pi);
                const T prefactor = T(2) * pi_2p5 /
                    (p * gamma * std::sqrt(pq)) * ca * cb * cc * sph_a * sph_b * sph_c;

                // Compute single-center E-coefficients for auxiliary function
                // E^l_t for x^l exp(-γx²): E^0_0=1, E^{l+1}_t = (1/(2γ))×E^l_{t-1} + (t+1)×E^l_{t+1}
                const T half_gamma_inv = T(0.5) / gamma;
                T E_c[(LMAX + 1) * (LMAX + 1)];
                for (int i = 0; i <= lc; ++i) {
                    for (int t = 0; t <= lc; ++t) {
                        E_c[i * (lc + 1) + t] = T(0);
                    }
                }
                E_c[0] = T(1);  // E^0_0 = 1
                for (int i = 0; i < lc; ++i) {
                    for (int t = 0; t <= i + 1; ++t) {
                        T val = T(0);
                        if (t > 0) val += half_gamma_inv * E_c[i * (lc + 1) + t - 1];
                        if (t + 1 <= i) val += T(t + 1) * E_c[i * (lc + 1) + t + 1];
                        E_c[(i + 1) * (lc + 1) + t] = val;
                    }
                }

                // R-integrals using reduced exponent α and distance P-C
                RIntsDynamic<T> R;
                compute_r_ints_dynamic<T, BoysParams>(
                    boys_table, L, alpha, PCx, PCy, PCz, R);

                // Contract: (μν|ρ) = prefactor × Σ_{tuv} E^{ab}_{tuv} × Σ_{xyz} E^{c}_{xyz} × R_{t+x,u+y,v+z}
                int idx_ab = 0;
                for (int i_a = la; i_a >= 0; --i_a) {
                    for (int j_a = la - i_a; j_a >= 0; --j_a) {
                        int k_a = la - i_a - j_a;
                        for (int i_b = lb; i_b >= 0; --i_b) {
                            for (int j_b = lb - i_b; j_b >= 0; --j_b) {
                                int k_b = lb - i_b - j_b;

                                int idx_c = 0;
                                for (int i_c = lc; i_c >= 0; --i_c) {
                                    for (int j_c = lc - i_c; j_c >= 0; --j_c) {
                                        int k_c = lc - i_c - j_c;

                                        T sum = T(0);
                                        // Sum over AO Hermite indices
                                        for (int t = 0; t <= i_a + i_b; ++t) {
                                            T Et = Ex_ab(i_a, i_b, t);
                                            for (int u = 0; u <= j_a + j_b; ++u) {
                                                T Eu = Ey_ab(j_a, j_b, u);
                                                for (int v = 0; v <= k_a + k_b; ++v) {
                                                    T Ev = Ez_ab(k_a, k_b, v);
                                                    // Sum over auxiliary Hermite indices
                                                    for (int tx = 0; tx <= i_c; ++tx) {
                                                        T Ecx = E_c[i_c * (lc + 1) + tx];
                                                        for (int ty = 0; ty <= j_c; ++ty) {
                                                            T Ecy = E_c[j_c * (lc + 1) + ty];
                                                            for (int tz = 0; tz <= k_c; ++tz) {
                                                                T Ecz = E_c[k_c * (lc + 1) + tz];
                                                                // Ket sign: (-1)^(tx+ty+tz) per occ convention
                                                                T ket_sign = ((tx + ty + tz) & 1) ? T(-1) : T(1);
                                                                sum += ket_sign * Et * Eu * Ev * Ecx * Ecy * Ecz *
                                                                       R(t + tx, u + ty, v + tz);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        integrals[idx_ab * nc + idx_c] += prefactor * sum;
                                        ++idx_c;
                                    }
                                }
                                ++idx_ab;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Dynamic 2-center Coulomb (P|Q) for runtime angular momentum
template <typename T, typename BoysParams = BoysParamsDefault>
void eri2c_dynamic(
    int lp, int lq,
    int np_prim, int nq_prim,
    const T* exponents_p, const T* exponents_q,
    const T* coeffs_p, const T* coeffs_q,
    const T* P, const T* Q,
    const T* boys_table,
    T* integrals)
{
    const int L = lp + lq;
    const int nc_p = ncart(lp);
    const int nc_q = ncart(lq);

    // Initialize output
    for (int i = 0; i < nc_p * nc_q; ++i) {
        integrals[i] = T(0);
    }

    // Loop over primitives
    for (int ip = 0; ip < np_prim; ++ip) {
        const T ap = exponents_p[ip];
        const T cp = coeffs_p[ip];

        for (int iq = 0; iq < nq_prim; ++iq) {
            const T aq = exponents_q[iq];
            const T cq = coeffs_q[iq];

            const T p = ap + aq;
            const T prefactor = T(2) * BoysConstants<T>::pi / p * cp * cq;

            // Weighted center
            const T Wx = (ap * P[0] + aq * Q[0]) / p;
            const T Wy = (ap * P[1] + aq * Q[1]) / p;
            const T Wz = (ap * P[2] + aq * Q[2]) / p;

            // Distance from weighted center to Q
            const T WQx = Wx - Q[0];
            const T WQy = Wy - Q[1];
            const T WQz = Wz - Q[2];

            // E-coefficients
            ECoeffs1DDynamic<T> Ex, Ey, Ez;
            compute_e_coeffs_1d_dynamic(lp, lq, ap, aq, Q[0] - P[0], Ex);
            compute_e_coeffs_1d_dynamic(lp, lq, ap, aq, Q[1] - P[1], Ey);
            compute_e_coeffs_1d_dynamic(lp, lq, ap, aq, Q[2] - P[2], Ez);

            // R-integrals
            RIntsDynamic<T> R;
            compute_r_ints_dynamic<T, BoysParams>(boys_table, L, p, WQx, WQy, WQz, R);

            // Contract
            int idx = 0;
            for (int i_p = lp; i_p >= 0; --i_p) {
                for (int j_p = lp - i_p; j_p >= 0; --j_p) {
                    int k_p = lp - i_p - j_p;

                    for (int i_q = lq; i_q >= 0; --i_q) {
                        for (int j_q = lq - i_q; j_q >= 0; --j_q) {
                            int k_q = lq - i_q - j_q;

                            T sum = T(0);
                            for (int t = 0; t <= i_p + i_q; ++t) {
                                T Et = Ex(i_p, i_q, t);
                                for (int u = 0; u <= j_p + j_q; ++u) {
                                    T Eu = Ey(j_p, j_q, u);
                                    for (int v = 0; v <= k_p + k_q; ++v) {
                                        T Ev = Ez(k_p, k_q, v);
                                        sum += Et * Eu * Ev * R(t, u, v);
                                    }
                                }
                            }
                            integrals[idx++] += prefactor * sum;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Matmul-based 3-Center ERI (optimized contraction)
// ============================================================================

/// Build E_ab matrix [nab × nherm_ab] from 3D E-coefficients
/// Row order: standard Cartesian order (descending in first index)
template <typename T>
void build_E_ab_matrix_dynamic(
    const ECoeffs1DDynamic<T>& Ex, const ECoeffs1DDynamic<T>& Ey, const ECoeffs1DDynamic<T>& Ez,
    int la, int lb, T* E_ab)
{
    const int L_ab = la + lb;
    const int nherm_ab = nhermsum(L_ab);

    int row = 0;
    for (int i_a = la; i_a >= 0; --i_a) {
        for (int j_a = la - i_a; j_a >= 0; --j_a) {
            int k_a = la - i_a - j_a;
            for (int i_b = lb; i_b >= 0; --i_b) {
                for (int j_b = lb - i_b; j_b >= 0; --j_b) {
                    int k_b = lb - i_b - j_b;

                    // Zero this row
                    for (int h = 0; h < nherm_ab; ++h) {
                        E_ab[row * nherm_ab + h] = T(0);
                    }

                    // Fill non-zero entries
                    for (int t = 0; t <= i_a + i_b; ++t) {
                        T Et = Ex(i_a, i_b, t);
                        for (int u = 0; u <= j_a + j_b; ++u) {
                            T Eu = Ey(j_a, j_b, u);
                            for (int v = 0; v <= k_a + k_b; ++v) {
                                T Ev = Ez(k_a, k_b, v);
                                int h = hermite_index(t, u, v);
                                E_ab[row * nherm_ab + h] = Et * Eu * Ev;
                            }
                        }
                    }
                    ++row;
                }
            }
        }
    }
}

/// Build E_c matrix [nc × nherm_c] from single-center E-coefficients with ket signs
/// The sign (-1)^(tx+ty+tz) is baked into the matrix
template <typename T>
void build_E_c_matrix_dynamic(const T* E_c_1d, int lc, T* E_c_matrix)
{
    const int nherm_c = nhermsum(lc);

    int row = 0;
    for (int i_c = lc; i_c >= 0; --i_c) {
        for (int j_c = lc - i_c; j_c >= 0; --j_c) {
            int k_c = lc - i_c - j_c;

            // Zero this row
            for (int h = 0; h < nherm_c; ++h) {
                E_c_matrix[row * nherm_c + h] = T(0);
            }

            // Fill non-zero entries with ket signs
            for (int tx = 0; tx <= i_c; ++tx) {
                T Ecx = E_c_1d[i_c * (lc + 1) + tx];
                for (int ty = 0; ty <= j_c; ++ty) {
                    T Ecy = E_c_1d[j_c * (lc + 1) + ty];
                    for (int tz = 0; tz <= k_c; ++tz) {
                        T Ecz = E_c_1d[k_c * (lc + 1) + tz];
                        T ket_sign = ((tx + ty + tz) & 1) ? T(-1) : T(1);
                        int h = hermite_index(tx, ty, tz);
                        E_c_matrix[row * nherm_c + h] = ket_sign * Ecx * Ecy * Ecz;
                    }
                }
            }
            ++row;
        }
    }
}

/// Build R_tensor [nherm_ab × nherm_c] from R-integrals
/// R_tensor[h_ab, h_c] = R[combined_hermite_index(h_ab, h_c)]
template <typename T>
void build_R_tensor_dynamic(const RIntsDynamic<T>& R, int L_ab, int lc, T* R_tensor)
{
    const int nherm_ab = nhermsum(L_ab);
    const int nherm_c = nhermsum(lc);

    // For each Hermite index pair, compute the combined R-integral
    for (int t = 0; t <= L_ab; ++t) {
        for (int u = 0; u <= L_ab - t; ++u) {
            for (int v = 0; v <= L_ab - t - u; ++v) {
                int h_ab = hermite_index(t, u, v);

                for (int tx = 0; tx <= lc; ++tx) {
                    for (int ty = 0; ty <= lc - tx; ++ty) {
                        for (int tz = 0; tz <= lc - tx - ty; ++tz) {
                            int h_c = hermite_index(tx, ty, tz);
                            R_tensor[h_ab * nherm_c + h_c] = R(t + tx, u + ty, v + tz);
                        }
                    }
                }
            }
        }
    }
}

/// Dynamic 3-center ERI using matrix multiplication for contraction
/// This version pre-allocates workspace and uses Eigen matmul
template <typename T, typename BoysParams = BoysParamsDefault>
void eri3c_dynamic_matmul(
    int la, int lb, int lc,
    int na_prim, int nb_prim, int nc_prim,
    const T* exponents_a, const T* exponents_b, const T* exponents_c,
    const T* coeffs_a, const T* coeffs_b, const T* coeffs_c,
    const T* A, const T* B, const T* C,
    const T* boys_table,
    T* integrals)
{
    const int L_ab = la + lb;
    const int L = L_ab + lc;
    const int na = ncart(la);
    const int nb = ncart(lb);
    const int nc = ncart(lc);
    const int nab = na * nb;
    const int nherm_ab = nhermsum(L_ab);
    const int nherm_c = nhermsum(lc);

    // Pre-allocate workspace buffers (avoid allocations in inner loops)
    std::vector<T> E_ab_buf(nab * nherm_ab);
    std::vector<T> E_c_buf(nc * nherm_c);
    std::vector<T> R_tensor_buf(nherm_ab * nherm_c);
    std::vector<T> temp_buf(nab * nherm_c);

    // Single-center E coefficients buffer (reused across aux primitives)
    T E_c_1d[(LMAX + 1) * (LMAX + 1)];

    // Eigen maps (no allocation, just view into buffers)
    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<MatRM> E_ab(E_ab_buf.data(), nab, nherm_ab);
    Eigen::Map<MatRM> E_c(E_c_buf.data(), nc, nherm_c);
    Eigen::Map<MatRM> R_tensor(R_tensor_buf.data(), nherm_ab, nherm_c);
    Eigen::Map<MatRM> temp(temp_buf.data(), nab, nherm_c);
    Eigen::Map<MatRM> result(integrals, nab, nc);

    result.setZero();

    // Spherical harmonic factors
    auto sph_factor = [](int l) -> T {
        if (l == 0) return T(0.28209479177387814);
        if (l == 1) return T(0.48860251190291992);
        return T(1.0);
    };
    const T sph_a = sph_factor(la);
    const T sph_b = sph_factor(lb);
    const T sph_c = sph_factor(lc);

    // Loop over AO primitive pairs
    for (int ia = 0; ia < na_prim; ++ia) {
        const T a = exponents_a[ia];
        const T ca = coeffs_a[ia];

        for (int ib = 0; ib < nb_prim; ++ib) {
            const T b = exponents_b[ib];
            const T cb = coeffs_b[ib];

            const T p = a + b;
            const T Px = (a * A[0] + b * B[0]) / p;
            const T Py = (a * A[1] + b * B[1]) / p;
            const T Pz = (a * A[2] + b * B[2]) / p;

            // E-coefficients for AO pair
            ECoeffs1DDynamic<T> Ex_ab, Ey_ab, Ez_ab;
            compute_e_coeffs_1d_dynamic(la, lb, a, b, B[0] - A[0], Ex_ab);
            compute_e_coeffs_1d_dynamic(la, lb, a, b, B[1] - A[1], Ey_ab);
            compute_e_coeffs_1d_dynamic(la, lb, a, b, B[2] - A[2], Ez_ab);

            // Build E_ab matrix once per AO primitive pair
            build_E_ab_matrix_dynamic(Ex_ab, Ey_ab, Ez_ab, la, lb, E_ab_buf.data());

            // Loop over aux primitives
            for (int ic = 0; ic < nc_prim; ++ic) {
                const T gamma = exponents_c[ic];
                const T cc = coeffs_c[ic];

                const T pq = p + gamma;
                const T alpha = p * gamma / pq;

                // Distance from P to C
                const T PCx = Px - C[0];
                const T PCy = Py - C[1];
                const T PCz = Pz - C[2];

                // Prefactor
                const T pi_2p5 = BoysConstants<T>::pi * BoysConstants<T>::pi * std::sqrt(BoysConstants<T>::pi);
                const T prefactor = T(2) * pi_2p5 /
                    (p * gamma * std::sqrt(pq)) * ca * cb * cc * sph_a * sph_b * sph_c;

                // Build single-center E coefficients for aux
                const T half_gamma_inv = T(0.5) / gamma;
                for (int i = 0; i <= lc; ++i) {
                    for (int t = 0; t <= lc; ++t) {
                        E_c_1d[i * (lc + 1) + t] = T(0);
                    }
                }
                E_c_1d[0] = T(1);
                for (int i = 0; i < lc; ++i) {
                    for (int t = 0; t <= i + 1; ++t) {
                        T val = T(0);
                        if (t > 0) val += half_gamma_inv * E_c_1d[i * (lc + 1) + t - 1];
                        if (t + 1 <= i) val += T(t + 1) * E_c_1d[i * (lc + 1) + t + 1];
                        E_c_1d[(i + 1) * (lc + 1) + t] = val;
                    }
                }

                // Build E_c matrix with ket signs
                build_E_c_matrix_dynamic(E_c_1d, lc, E_c_buf.data());

                // Compute R-integrals
                RIntsDynamic<T> R;
                compute_r_ints_dynamic<T, BoysParams>(boys_table, L, alpha, PCx, PCy, PCz, R);

                // Build R_tensor
                build_R_tensor_dynamic(R, L_ab, lc, R_tensor_buf.data());

                // Matmul contraction: result += prefactor * E_ab @ R_tensor @ E_c^T
                temp.noalias() = E_ab * R_tensor;
                result.noalias() += prefactor * temp * E_c.transpose();
            }
        }
    }
}

/// Dynamic 3-center ERI using matrix multiplication with external workspace
/// This version eliminates internal allocations by using caller-provided buffers.
///
/// @param la, lb, lc       Angular momenta
/// @param na_prim, nb_prim, nc_prim  Number of primitives
/// @param exponents_a/b/c  Primitive exponents
/// @param coeffs_a/b/c     Contraction coefficients
/// @param A, B, C          Shell centers
/// @param boys_table       Boys function interpolation table
/// @param integrals        Output buffer [nab × nc]
/// @param E_ab_workspace   Workspace for E_ab matrix [nab × nherm_ab]
/// @param E_c_workspace    Workspace for E_c matrix [nc × nherm_c]
/// @param R_workspace      Workspace for R tensor [nherm_ab × nherm_c]
/// @param temp_workspace   Workspace for temp matrix [nab × nherm_c]
template <typename T, typename BoysParams = BoysParamsDefault>
void eri3c_dynamic_matmul_workspace(
    int la, int lb, int lc,
    int na_prim, int nb_prim, int nc_prim,
    const T* exponents_a, const T* exponents_b, const T* exponents_c,
    const T* coeffs_a, const T* coeffs_b, const T* coeffs_c,
    const T* A, const T* B, const T* C,
    const T* boys_table,
    T* integrals,
    T* E_ab_workspace,
    T* E_c_workspace,
    T* R_workspace,
    T* temp_workspace)
{
    const int L_ab = la + lb;
    const int L = L_ab + lc;
    const int na = ncart(la);
    const int nb = ncart(lb);
    const int nc = ncart(lc);
    const int nab = na * nb;
    const int nherm_ab = nhermsum(L_ab);
    const int nherm_c = nhermsum(lc);

    // Single-center E coefficients buffer (small, stack-allocated)
    T E_c_1d[(LMAX + 1) * (LMAX + 1)];

    // Eigen maps over workspace (no allocation)
    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<MatRM> E_ab(E_ab_workspace, nab, nherm_ab);
    Eigen::Map<MatRM> E_c(E_c_workspace, nc, nherm_c);
    Eigen::Map<MatRM> R_tensor(R_workspace, nherm_ab, nherm_c);
    Eigen::Map<MatRM> temp(temp_workspace, nab, nherm_c);
    Eigen::Map<MatRM> result(integrals, nab, nc);

    result.setZero();

    // Spherical harmonic factors
    auto sph_factor = [](int l) -> T {
        if (l == 0) return T(0.28209479177387814);
        if (l == 1) return T(0.48860251190291992);
        return T(1.0);
    };
    const T sph_a = sph_factor(la);
    const T sph_b = sph_factor(lb);
    const T sph_c = sph_factor(lc);

    // Loop over AO primitive pairs
    for (int ia = 0; ia < na_prim; ++ia) {
        const T a = exponents_a[ia];
        const T ca = coeffs_a[ia];

        for (int ib = 0; ib < nb_prim; ++ib) {
            const T b = exponents_b[ib];
            const T cb = coeffs_b[ib];

            const T p = a + b;
            const T Px = (a * A[0] + b * B[0]) / p;
            const T Py = (a * A[1] + b * B[1]) / p;
            const T Pz = (a * A[2] + b * B[2]) / p;

            // E-coefficients for AO pair
            ECoeffs1DDynamic<T> Ex_ab, Ey_ab, Ez_ab;
            compute_e_coeffs_1d_dynamic(la, lb, a, b, B[0] - A[0], Ex_ab);
            compute_e_coeffs_1d_dynamic(la, lb, a, b, B[1] - A[1], Ey_ab);
            compute_e_coeffs_1d_dynamic(la, lb, a, b, B[2] - A[2], Ez_ab);

            // Build E_ab matrix once per AO primitive pair
            build_E_ab_matrix_dynamic(Ex_ab, Ey_ab, Ez_ab, la, lb, E_ab_workspace);

            // Loop over aux primitives
            for (int ic = 0; ic < nc_prim; ++ic) {
                const T gamma = exponents_c[ic];
                const T cc = coeffs_c[ic];

                const T pq = p + gamma;
                const T alpha = p * gamma / pq;

                // Distance from P to C
                const T PCx = Px - C[0];
                const T PCy = Py - C[1];
                const T PCz = Pz - C[2];

                // Prefactor
                const T pi_2p5 = BoysConstants<T>::pi * BoysConstants<T>::pi * std::sqrt(BoysConstants<T>::pi);
                const T prefactor = T(2) * pi_2p5 /
                    (p * gamma * std::sqrt(pq)) * ca * cb * cc * sph_a * sph_b * sph_c;

                // Build single-center E coefficients for aux
                const T half_gamma_inv = T(0.5) / gamma;
                for (int i = 0; i <= lc; ++i) {
                    for (int t = 0; t <= lc; ++t) {
                        E_c_1d[i * (lc + 1) + t] = T(0);
                    }
                }
                E_c_1d[0] = T(1);
                for (int i = 0; i < lc; ++i) {
                    for (int t = 0; t <= i + 1; ++t) {
                        T val = T(0);
                        if (t > 0) val += half_gamma_inv * E_c_1d[i * (lc + 1) + t - 1];
                        if (t + 1 <= i) val += T(t + 1) * E_c_1d[i * (lc + 1) + t + 1];
                        E_c_1d[(i + 1) * (lc + 1) + t] = val;
                    }
                }

                // Build E_c matrix with ket signs
                build_E_c_matrix_dynamic(E_c_1d, lc, E_c_workspace);

                // Compute R-integrals
                RIntsDynamic<T> R;
                compute_r_ints_dynamic<T, BoysParams>(boys_table, L, alpha, PCx, PCy, PCz, R);

                // Build R_tensor
                build_R_tensor_dynamic(R, L_ab, lc, R_workspace);

                // Matmul contraction: result += prefactor * E_ab @ R_tensor @ E_c^T
                temp.noalias() = E_ab * R_tensor;
                result.noalias() += prefactor * temp * E_c.transpose();
            }
        }
    }
}

/// Required workspace sizes for eri3c_dynamic_matmul_workspace
/// Returns sizes as tuple: (E_ab_size, E_c_size, R_tensor_size, temp_size)
inline std::tuple<size_t, size_t, size_t, size_t>
eri3c_workspace_sizes(int la, int lb, int lc) {
    const int nab = ncart(la) * ncart(lb);
    const int nc = ncart(lc);
    const int nherm_ab = nhermsum(la + lb);
    const int nherm_c = nhermsum(lc);

    return {
        static_cast<size_t>(nab * nherm_ab),     // E_ab
        static_cast<size_t>(nc * nherm_c),        // E_c
        static_cast<size_t>(nherm_ab * nherm_c),  // R_tensor
        static_cast<size_t>(nab * nherm_c)        // temp
    };
}

/// Maximum workspace sizes given max angular momenta
/// Useful for pre-allocating workspace for all shell combinations
inline std::tuple<size_t, size_t, size_t, size_t>
eri3c_workspace_sizes_max(int max_l_ao, int max_l_aux) {
    const int max_nab = ncart(max_l_ao) * ncart(max_l_ao);
    const int max_nc = ncart(max_l_aux);
    const int max_nherm_ab = nhermsum(2 * max_l_ao);
    const int max_nherm_c = nhermsum(max_l_aux);

    return {
        static_cast<size_t>(max_nab * max_nherm_ab),     // E_ab
        static_cast<size_t>(max_nc * max_nherm_c),        // E_c
        static_cast<size_t>(max_nherm_ab * max_nherm_c),  // R_tensor
        static_cast<size_t>(max_nab * max_nherm_c)        // temp
    };
}

} // namespace occ::ints
