#pragma once
#include <occ/ints/kernels.h>
#include <occ/ints/rints.h>
#include <Eigen/Dense>
#include <vector>

namespace occ::ints {

// ============================================================================
// TRUE Split-RI-J Hermite Transformation Kernels
// ============================================================================
//
// These kernels implement the transformations from Neese 2003 (Scheme 4):
//
// Forward pass (compute g from density D):
//   Eq. 17: X_q = sum_{mu,nu} D_{mu,nu} * E^{mu nu}_q * (-1)^q
//   Eq. 18: Y_p = sum_q R_pq * X_q
//   Eq. 19: g_r = sum_p E^r_p * Y_p
//
// Backward pass (compute J from d):
//   Eq. 20: T_p = sum_r E^r_p * d_r
//   Eq. 21: U_q = sum_p R_pq * T_p  (same R_pq!)
//   Eq. 22: J_{mu,nu} = sum_q (-1)^q * E^{mu nu}_q * U_q
//
// Key insight: R_pq is computed once and reused for both passes.
// ============================================================================

/// Compute Hermite sign factor (-1)^(t+u+v) for a given Hermite index
template <typename T>
OCC_GPU_ENABLED OCC_GPU_INLINE
constexpr T hermite_sign(int t, int u, int v) {
    return ((t + u + v) & 1) ? T(-1) : T(1);
}

/// Apply Hermite signs to a vector
/// out[h] = sign(h) * in[h] where sign(h) = (-1)^(t+u+v)
template <typename T>
void apply_hermite_signs(const T* in, int L_max, T* out) {
    for (int t = 0; t <= L_max; ++t) {
        for (int u = 0; u <= L_max - t; ++u) {
            for (int v = 0; v <= L_max - t - u; ++v) {
                int h = hermite_index(t, u, v);
                T sign = hermite_sign<T>(t, u, v);
                out[h] = sign * in[h];
            }
        }
    }
}

// ============================================================================
// Forward Pass Kernels
// ============================================================================

/// Eq. 17: Transform density block to Hermite basis
/// X_q = sum_{ab} D[ab] * E[ab, q] * (-1)^q
///
/// This is the first step of Split-RI-J: contract density with E-matrix.
/// The sign (-1)^q is applied here.
///
/// @param D_block    Density matrix block [na * nb] (flattened)
/// @param E_ab       E-matrix [nab, nherm_ab] row-major (from primitives, accumulated)
/// @param na, nb     Shell sizes
/// @param nherm_ab   Number of Hermite functions
/// @param X_out      Output Hermite density [nherm_ab]
/// @param apply_sign If true, apply (-1)^q signs (default true)
template <typename T>
void density_to_hermite(
    const T* D_block,       // [na * nb]
    const T* E_ab,          // [nab * nherm_ab]
    int na, int nb,
    int nherm_ab,
    T* X_out,               // [nherm_ab]
    bool apply_sign = true)
{
    using VecMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    using ConstVecMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatRMMap = Eigen::Map<const MatRM>;

    const int nab = na * nb;
    ConstMatRMMap E(E_ab, nab, nherm_ab);
    ConstVecMap D(D_block, nab);
    VecMap X(X_out, nherm_ab);

    // X = E^T @ D (contract Cartesian pairs to Hermite)
    X.noalias() = E.transpose() * D;

    // Apply (-1)^q signs if requested
    if (apply_sign) {
        int L_ab = 0;
        while (nhermsum(L_ab) < nherm_ab) ++L_ab;
        apply_hermite_signs(X_out, L_ab, X_out);
    }
}

/// Eq. 18: Contract Hermite ERI tensor with Hermite density (forward)
/// Y_p = sum_q R_pq * X_q
///
/// @param R_pq       Hermite ERI tensor [nherm_ab * nherm_c] row-major
/// @param X_q        Hermite density [nherm_ab]
/// @param nherm_ab   Rows of R_pq (AO pair Hermite count)
/// @param nherm_c    Cols of R_pq (aux Hermite count)
/// @param Y_p        Output [nherm_c]
template <typename T>
void contract_hermite_eri_forward(
    const T* R_pq,          // [nherm_ab * nherm_c]
    const T* X_q,           // [nherm_ab]
    int nherm_ab,
    int nherm_c,
    T* Y_p)                 // [nherm_c]
{
    using VecMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    using ConstVecMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatRMMap = Eigen::Map<const MatRM>;

    ConstMatRMMap R(R_pq, nherm_ab, nherm_c);
    ConstVecMap X(X_q, nherm_ab);
    VecMap Y(Y_p, nherm_c);

    // Y = R^T @ X (contract over AO Hermite indices p)
    Y.noalias() = R.transpose() * X;
}

/// Eq. 19: Transform from Hermite to aux orbital basis
/// g_r = sum_p E^r_p * Y_p
///
/// @param E_c        Aux E-matrix [nc * nherm_c] row-major
/// @param Y_p        Hermite intermediate [nherm_c]
/// @param nc         Number of aux Cartesian functions
/// @param nherm_c    Number of aux Hermite functions
/// @param g_out      Output [nc] (accumulated)
template <typename T>
void hermite_to_aux(
    const T* E_c,           // [nc * nherm_c]
    const T* Y_p,           // [nherm_c]
    int nc,
    int nherm_c,
    T* g_out)               // [nc]
{
    using VecMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    using ConstVecMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatRMMap = Eigen::Map<const MatRM>;

    ConstMatRMMap E(E_c, nc, nherm_c);
    ConstVecMap Y(Y_p, nherm_c);
    VecMap g(g_out, nc);

    // g = E @ Y
    g.noalias() = E * Y;
}

// ============================================================================
// Backward Pass Kernels
// ============================================================================

/// Eq. 20: Transform from aux orbital to Hermite basis
/// T_p = sum_r E^r_p * d_r
///
/// @param E_c        Aux E-matrix [nc * nherm_c] row-major
/// @param d_r        Fitted coefficients [nc]
/// @param nc         Number of aux Cartesian functions
/// @param nherm_c    Number of aux Hermite functions
/// @param T_out      Output [nherm_c]
template <typename Scalar>
void aux_to_hermite(
    const Scalar* E_c,           // [nc * nherm_c]
    const Scalar* d_r,           // [nc]
    int nc,
    int nherm_c,
    Scalar* T_out)               // [nherm_c]
{
    using VecMap = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;
    using ConstVecMap = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;
    using MatRM = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatRMMap = Eigen::Map<const MatRM>;

    ConstMatRMMap E(E_c, nc, nherm_c);
    ConstVecMap d(d_r, nc);
    VecMap T_vec(T_out, nherm_c);

    // T = E^T @ d
    T_vec.noalias() = E.transpose() * d;
}

/// Eq. 21: Contract Hermite ERI tensor (backward)
/// U_q = sum_p R_pq * T_p
///
/// NOTE: Uses the SAME R_pq as forward pass - this is the key insight!
///
/// @param R_pq       Hermite ERI tensor [nherm_ab * nherm_c] row-major
/// @param T_p        Hermite aux coefficients [nherm_c]
/// @param nherm_ab   Rows of R_pq
/// @param nherm_c    Cols of R_pq
/// @param U_q        Output [nherm_ab]
template <typename Scalar>
void contract_hermite_eri_backward(
    const Scalar* R_pq,          // [nherm_ab * nherm_c]
    const Scalar* T_p,           // [nherm_c]
    int nherm_ab,
    int nherm_c,
    Scalar* U_q)                 // [nherm_ab]
{
    using VecMap = Eigen::Map<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;
    using ConstVecMap = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>;
    using MatRM = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatRMMap = Eigen::Map<const MatRM>;

    ConstMatRMMap R(R_pq, nherm_ab, nherm_c);
    ConstVecMap T_vec(T_p, nherm_c);
    VecMap U(U_q, nherm_ab);

    // U = R @ T (note: NOT transposed, unlike forward pass)
    U.noalias() = R * T_vec;
}

/// Eq. 22: Transform from Hermite to orbital basis (build J block)
/// J_{mu,nu} = sum_q (-1)^q * E^{mu nu}_q * U_q
///
/// @param E_ab       E-matrix [nab * nherm_ab] row-major
/// @param U_q        Hermite coefficients [nherm_ab] (signs already applied)
/// @param na, nb     Shell sizes
/// @param nherm_ab   Number of Hermite functions
/// @param J_block    Output [na * nb] (accumulated)
/// @param apply_sign If true, apply (-1)^q to U before contraction
template <typename T>
void hermite_to_orbital(
    const T* E_ab,          // [nab * nherm_ab]
    const T* U_q,           // [nherm_ab]
    int na, int nb,
    int nherm_ab,
    T* J_block,             // [na * nb]
    bool apply_sign = true)
{
    using VecMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    using ConstVecMap = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    using MatRM = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatRMMap = Eigen::Map<const MatRM>;

    const int nab = na * nb;
    ConstMatRMMap E(E_ab, nab, nherm_ab);
    VecMap J(J_block, nab);

    if (apply_sign) {
        // Apply (-1)^q signs to U
        int L_ab = 0;
        while (nhermsum(L_ab) < nherm_ab) ++L_ab;

        std::vector<T> U_signed(nherm_ab);
        apply_hermite_signs(U_q, L_ab, U_signed.data());

        ConstVecMap U(U_signed.data(), nherm_ab);
        J.noalias() += E * U;
    } else {
        ConstVecMap U(U_q, nherm_ab);
        J.noalias() += E * U;
    }
}

// ============================================================================
// R_pq Tensor Computation
// ============================================================================

/// Compute the accumulated R_pq tensor for a shell pair + aux shell
///
/// R_pq connects Hermite indices from AO pair (index q, size nherm_ab)
/// to Hermite indices from aux shell (index p, size nherm_c).
///
/// The tensor is: R_pq[q, p] = sum_prims prefactor * R(t_q + t_p, u_q + u_p, v_q + v_p)
///
/// where (t_q, u_q, v_q) are Hermite indices for the AO pair and
/// (t_p, u_p, v_p) are Hermite indices for the aux shell.
///
/// @param ao_pair    Precomputed AO shell pair data
/// @param aux_shell  Precomputed aux shell data
/// @param boys_table Boys function interpolation table
/// @param R_pq_out   Output tensor [nherm_ab * nherm_c]
template <typename T, typename BoysParams = BoysParamsDefault>
void compute_R_pq_tensor(
    const ShellPairData<T>& ao_pair,
    const AuxShellData<T>& aux_shell,
    const T* boys_table,
    T* R_pq_out)
{
    const int la = ao_pair.la;
    const int lb = ao_pair.lb;
    const int lc = aux_shell.lc;
    const int L_ab = la + lb;
    const int L_total = L_ab + lc;
    const int nherm_ab = nhermsum(L_ab);
    const int nherm_c = nhermsum(lc);

    // Zero output
    for (int i = 0; i < nherm_ab * nherm_c; ++i) {
        R_pq_out[i] = T(0);
    }

    // Prefactor constants
    const T pi_1p5 = std::pow(BoysConstants<T>::pi, T(1.5));

    // Loop over AO primitive pairs
    for (const auto& ao_prim : ao_pair.primitives) {
        const T p = ao_prim.p;
        const T Px = ao_prim.Px;
        const T Py = ao_prim.Py;
        const T Pz = ao_prim.Pz;
        const T prefactor_ab = ao_prim.prefactor;

        // Loop over aux primitives
        for (const auto& aux_prim : aux_shell.primitives) {
            const T gamma = aux_prim.gamma;
            const T cc = aux_prim.coeff;

            const T pq = p + gamma;
            const T alpha = p * gamma / pq;

            // Distance from P to C
            const T PCx = Px - aux_shell.C[0];
            const T PCy = Py - aux_shell.C[1];
            const T PCz = Pz - aux_shell.C[2];

            // Prefactor combines ao_prim.prefactor with aux factors
            // ao_prim.prefactor = 2*pi/p * sph_a * sph_b * ca * cb (includes K_ab)
            // Full prefactor = prefactor_ab * cc * sph_c * pi^{3/2} / (gamma * sqrt(pq))
            const T sph_c = spherical_harmonic_factor<T>(lc);
            const T prefactor = prefactor_ab * cc * sph_c * pi_1p5 / (gamma * std::sqrt(pq));

            // Compute R-integrals
            RIntsDynamic<T> R;
            compute_r_ints_dynamic<T, BoysParams>(boys_table, L_total, alpha, PCx, PCy, PCz, R);

            // Accumulate into R_pq tensor
            // R_pq[h_ab, h_c] += prefactor * R[combined_index]
            for (int t_ab = 0; t_ab <= L_ab; ++t_ab) {
                for (int u_ab = 0; u_ab <= L_ab - t_ab; ++u_ab) {
                    for (int v_ab = 0; v_ab <= L_ab - t_ab - u_ab; ++v_ab) {
                        int h_ab = hermite_index(t_ab, u_ab, v_ab);

                        for (int t_c = 0; t_c <= lc; ++t_c) {
                            for (int u_c = 0; u_c <= lc - t_c; ++u_c) {
                                for (int v_c = 0; v_c <= lc - t_c - u_c; ++v_c) {
                                    int h_c = hermite_index(t_c, u_c, v_c);
                                    R_pq_out[h_ab * nherm_c + h_c] +=
                                        prefactor * R(t_ab + t_c, u_ab + u_c, v_ab + v_c);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Compute E_ab matrix accumulated over all primitives in a shell pair
/// This is needed for the forward and backward transformations.
///
/// NOTE: For TRUE Split-RI-J, we do NOT include prefactors here.
/// The E-matrix is just a pure basis transformation weighted by contraction coefficients.
/// All integral prefactors (2π/p, spherical factors, etc.) go into R_pq.
///
/// @param shell_pair Precomputed shell pair data
/// @param E_ab_out   Output [nab * nherm_ab] row-major
template <typename T>
void compute_accumulated_E_ab(
    const ShellPairData<T>& shell_pair,
    T* E_ab_out)
{
    const int nab = shell_pair.nab();
    const int nherm_ab = shell_pair.nherm();

    // Zero output
    for (int i = 0; i < nab * nherm_ab; ++i) {
        E_ab_out[i] = T(0);
    }

    // Accumulate over all primitives - just E-matrices, no integral prefactors
    // The prim.E_matrix is the pure E-coefficient transformation
    for (const auto& prim : shell_pair.primitives) {
        for (int i = 0; i < nab * nherm_ab; ++i) {
            E_ab_out[i] += prim.E_matrix[i];
        }
    }
}

/// Compute E_c matrix accumulated over all primitives in an aux shell
///
/// @param aux_shell  Precomputed aux shell data
/// @param E_c_out    Output [nc * nherm_c] row-major
template <typename T>
void compute_accumulated_E_c(
    const AuxShellData<T>& aux_shell,
    T* E_c_out)
{
    const int nc = aux_shell.nc();
    const int nherm_c = aux_shell.nherm();

    // Zero output
    for (int i = 0; i < nc * nherm_c; ++i) {
        E_c_out[i] = T(0);
    }

    // Accumulate over all primitives
    for (const auto& prim : aux_shell.primitives) {
        for (int i = 0; i < nc * nherm_c; ++i) {
            E_c_out[i] += prim.coeff * prim.E_matrix[i];
        }
    }
}

// ============================================================================
// Workspace Helpers
// ============================================================================

/// Compute maximum workspace sizes for Split-RI-J
///
/// @param max_l_ao   Maximum AO angular momentum
/// @param max_l_aux  Maximum aux angular momentum
/// @return Tuple of (max_nherm_ab, max_nherm_c, max_nab, max_nc)
inline std::tuple<size_t, size_t, size_t, size_t>
split_rij_workspace_sizes(int max_l_ao, int max_l_aux) {
    const int max_nherm_ab = nhermsum(2 * max_l_ao);
    const int max_nherm_c = nhermsum(max_l_aux);
    const int max_nab = ncart(max_l_ao) * ncart(max_l_ao);
    const int max_nc = ncart(max_l_aux);

    return {max_nherm_ab, max_nherm_c, max_nab, max_nc};
}

} // namespace occ::ints
