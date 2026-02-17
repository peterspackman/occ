#pragma once
#include <occ/mults/interaction_tensor.h>
#include <occ/mults/cartesian_multipole.h>

namespace occ::mults {

namespace kernel_detail {

/// (t, u, v) tuple stored for each hermite index.
struct TUV {
    int t, u, v;
};

/// Build the index-to-(t,u,v) table for all ranks up to MaxL at compile time.
/// Table is ordered by hermite_index: rank 0, rank 1, ..., rank MaxL.
template <int MaxL>
struct TUVTable {
    static constexpr int N = nhermsum(MaxL);
    TUV entries[N];

    constexpr TUVTable() : entries{} {
        int idx = 0;
        for (int l = 0; l <= MaxL; ++l) {
            for (int t = l; t >= 0; --t) {
                for (int uv = l - t; uv >= 0; --uv) {
                    int u = l - t - uv;
                    int v = uv;
                    // Place at the correct hermite_index position
                    int hi = hermite_index(t, u, v);
                    entries[hi] = {t, u, v};
                    ++idx;
                }
            }
        }
    }
};

/// Global constexpr table for MaxL=4 (35 entries) and MaxL=8 (165 entries).
inline constexpr TUVTable<4> tuv4{};
inline constexpr TUVTable<8> tuv8{};

/// Factorial lookup (sufficient for max index 8).
inline constexpr double fact(int n) {
    constexpr double table[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320};
    return table[n];
}

/// Precomputed weight table: (-1)^l / (t! u! v!) for site A,
/// and 1 / (t! u! v!) for site B.
template <int MaxL>
struct WeightTable {
    static constexpr int N = nhermsum(MaxL);
    double sign_inv_fact[N]; // (-1)^l / (t!u!v!)
    double inv_fact[N];      // 1 / (t!u!v!)

    constexpr WeightTable() : sign_inv_fact{}, inv_fact{} {
        constexpr TUVTable<MaxL> tuv{};
        for (int i = 0; i < N; ++i) {
            auto [t, u, v] = tuv.entries[i];
            int l = t + u + v;
            double ifact = 1.0 / (fact(t) * fact(u) * fact(v));
            double sign = (l % 2 == 0) ? 1.0 : -1.0;
            sign_inv_fact[i] = sign * ifact;
            inv_fact[i] = ifact;
        }
    }
};

inline constexpr WeightTable<4> weights4{};

/// Precomputed hermite_index addition tables for contraction hot paths.
///
/// For each pair of hermite indices (i, j), stores the combined index
/// hermite_index(ta+tb, ua+ub, va+vb) and the three shifted-index variants
/// for force gradients. Replaces ~10 integer ops per hermite_index call
/// with a single array lookup.
///
/// Table size: N×N × 4 × 4 bytes. For MaxL=4: 35×35×16 = ~19KB (fits in L1).
template <int MaxL>
struct HermiteAddTable {
    static constexpr int N = nhermsum(MaxL);
    int energy[N][N];    // hermite_index(ta+tb, ua+ub, va+vb)
    int force_x[N][N];   // hermite_index(ta+tb+1, ua+ub, va+vb)
    int force_y[N][N];   // hermite_index(ta+tb, ua+ub+1, va+vb)
    int force_z[N][N];   // hermite_index(ta+tb, ua+ub, va+vb+1)

    constexpr HermiteAddTable() : energy{}, force_x{}, force_y{}, force_z{} {
        constexpr TUVTable<MaxL> tuv{};
        for (int i = 0; i < N; ++i) {
            int ta = tuv.entries[i].t;
            int ua = tuv.entries[i].u;
            int va = tuv.entries[i].v;
            for (int j = 0; j < N; ++j) {
                int tb = tuv.entries[j].t;
                int ub = tuv.entries[j].u;
                int vb = tuv.entries[j].v;
                energy[i][j]  = hermite_index(ta+tb, ua+ub, va+vb);
                force_x[i][j] = hermite_index(ta+tb+1, ua+ub, va+vb);
                force_y[i][j] = hermite_index(ta+tb, ua+ub+1, va+vb);
                force_z[i][j] = hermite_index(ta+tb, ua+ub, va+vb+1);
            }
        }
    }
};

inline constexpr HermiteAddTable<4> add_table4{};

/// Lightweight energy + gradient result (no Eigen dependency).
struct EnergyGradient {
    double energy;
    double grad[3]; // dE/dR_x, dE/dR_y, dE/dR_z
};

} // namespace kernel_detail

/// Preweighted contraction: compute energy for multipoles with known ranks.
///
/// Precomputes wA[i] = (-1)^l / (t!u!v!) * A[i] and
///             wB[j] = 1/(t!u!v!) * B[j]
/// then contracts: E = sum_ij wA[i] * T(ta+tb, ua+ub, va+vb) * wB[j]
///
/// Only iterates over indices up to rankA and rankB respectively.
///
/// @tparam Order  The interaction tensor order (= rankA + rankB)
template <int Order>
double contract_ranked(const CartesianMultipole<4> &A, int rankA,
                       const InteractionTensor<Order> &T,
                       const CartesianMultipole<4> &B, int rankB) {
    using namespace kernel_detail;

    const int nA = nhermsum(rankA);
    const int nB = nhermsum(rankB);

    // Precompute weighted multipole values
    alignas(64) double wA[nhermsum(4)];
    alignas(64) double wB[nhermsum(4)];

    for (int i = 0; i < nA; ++i)
        wA[i] = weights4.sign_inv_fact[i] * A.data[i];
    for (int j = 0; j < nB; ++j)
        wB[j] = weights4.inv_fact[j] * B.data[j];

    double energy = 0.0;

    for (int i = 0; i < nA; ++i) {
        if (wA[i] == 0.0) continue;
        double eA = 0.0;
        for (int j = 0; j < nB; ++j) {
            eA += T.data[add_table4.energy[i][j]] * wB[j];
        }
        energy += wA[i] * eA;
    }

    return energy;
}

/// Preweighted contraction with force: compute energy AND gradient dE/dR.
///
/// Uses the T-tensor derivative property:
///   dT(t,u,v)/dR_x = T(t+1,u,v)
///   dT(t,u,v)/dR_y = T(t,u+1,v)
///   dT(t,u,v)/dR_z = T(t,u,v+1)
///
/// Requires InteractionTensor<Order+1> to accommodate the shifted indices.
///
/// @tparam Order  The interaction tensor order (= rankA + rankB)
template <int Order>
kernel_detail::EnergyGradient contract_ranked_with_force(
    const CartesianMultipole<4> &A, int rankA,
    const InteractionTensor<Order + 1> &T,
    const CartesianMultipole<4> &B, int rankB) {
    using namespace kernel_detail;

    const int nA = nhermsum(rankA);
    const int nB = nhermsum(rankB);

    alignas(64) double wA[nhermsum(4)];
    alignas(64) double wB[nhermsum(4)];

    for (int i = 0; i < nA; ++i)
        wA[i] = weights4.sign_inv_fact[i] * A.data[i];
    for (int j = 0; j < nB; ++j)
        wB[j] = weights4.inv_fact[j] * B.data[j];

    double energy = 0.0;
    double gx = 0.0, gy = 0.0, gz = 0.0;

    for (int i = 0; i < nA; ++i) {
        if (wA[i] == 0.0) continue;
        double eA = 0.0, gxA = 0.0, gyA = 0.0, gzA = 0.0;
        for (int j = 0; j < nB; ++j) {
            double wBj = wB[j];
            eA  += T.data[add_table4.energy[i][j]] * wBj;
            gxA += T.data[add_table4.force_x[i][j]] * wBj;
            gyA += T.data[add_table4.force_y[i][j]] * wBj;
            gzA += T.data[add_table4.force_z[i][j]] * wBj;
        }
        energy += wA[i] * eA;
        gx += wA[i] * gxA;
        gy += wA[i] * gyA;
        gz += wA[i] * gzA;
    }

    return {energy, {gx, gy, gz}};
}

/// Compute per-site interaction field at a local site from another site's multipoles.
///
/// field[tuv] = sum_{j} T(ta+tb, ua+ub, va+vb) * w_other[j]
///
/// When other_signed=false: w_other = inv_fact * other (for B-side "other", no sign)
/// When other_signed=true:  w_other = sign_inv_fact * other (for A-side "other", (-1)^l sign)
///
/// Use other_signed=false when computing field at A from B's multipoles.
/// Use other_signed=true when computing field at B from A's multipoles.
template <int Order, bool OtherSigned = false>
void compute_interaction_field(
    int rankLocal,
    const InteractionTensor<Order> &T,
    const CartesianMultipole<4> &other, int rankOther,
    CartesianMultipole<4> &field) {
    using namespace kernel_detail;

    const int nLocal = nhermsum(rankLocal);
    const int nOther = nhermsum(rankOther);

    alignas(64) double wOther[nhermsum(4)];
    for (int j = 0; j < nOther; ++j) {
        if constexpr (OtherSigned)
            wOther[j] = weights4.sign_inv_fact[j] * other.data[j];
        else
            wOther[j] = weights4.inv_fact[j] * other.data[j];
    }

    for (int i = 0; i < nLocal; ++i) {
        double f = 0.0;
        for (int j = 0; j < nOther; ++j) {
            f += T.data[add_table4.energy[i][j]] * wOther[j];
        }
        field.data[i] = f;
    }
}

/// Compute interaction field from a pre-computed T-tensor of higher order.
///
/// Same as compute_interaction_field, but accepts InteractionTensor<HigherOrder>
/// and only accesses elements up to rankLocal + rankOther. This avoids
/// recomputing the T-tensor when we already have one at sufficient order.
template <int HigherOrder, bool OtherSigned = false>
void compute_interaction_field_from_tensor(
    int rankLocal,
    const InteractionTensor<HigherOrder> &T,
    const CartesianMultipole<4> &other, int rankOther,
    CartesianMultipole<4> &field) {
    using namespace kernel_detail;

    const int nLocal = nhermsum(rankLocal);
    const int nOther = nhermsum(rankOther);

    alignas(64) double wOther[nhermsum(4)];
    for (int j = 0; j < nOther; ++j) {
        if constexpr (OtherSigned)
            wOther[j] = weights4.sign_inv_fact[j] * other.data[j];
        else
            wOther[j] = weights4.inv_fact[j] * other.data[j];
    }

    for (int i = 0; i < nLocal; ++i) {
        double f = 0.0;
        for (int j = 0; j < nOther; ++j) {
            f += T.data[add_table4.energy[i][j]] * wOther[j];
        }
        field.data[i] = f;
    }
}

/// Combined energy+force+fields kernel.
///
/// Computes InteractionTensor<Order+1> once and extracts:
/// 1. Energy and force gradient (from contract_ranked_with_force)
/// 2. Interaction field at A from B (OtherSigned=false)
/// 3. Interaction field at B from A (OtherSigned=true)
///
/// This replaces 3 separate T-tensor computations with 1.
struct EnergyForceFields {
    kernel_detail::EnergyGradient eg;
    CartesianMultipole<4> fieldA;
    CartesianMultipole<4> fieldB;
};

template <int Order>
EnergyForceFields compute_pair_ef_and_fields(
    const CartesianMultipole<4> &cartA, int rankA,
    const CartesianMultipole<4> &cartB, int rankB,
    double Rx, double Ry, double Rz) {
    EnergyForceFields result;

    // Compute T-tensor at Order+1 (needed for force gradient)
    InteractionTensor<Order + 1> T;
    compute_interaction_tensor<Order + 1>(Rx, Ry, Rz, T);

    // Energy and force from the Order+1 tensor
    result.eg = contract_ranked_with_force<Order>(cartA, rankA, T, cartB, rankB);

    // Field at A from B: uses T elements up to Order (subset of Order+1)
    compute_interaction_field_from_tensor<Order + 1, false>(
        rankA, T, cartB, rankB, result.fieldA);

    // Field at B from A: uses T elements up to Order (subset of Order+1)
    compute_interaction_field_from_tensor<Order + 1, true>(
        rankB, T, cartA, rankA, result.fieldB);

    return result;
}

} // namespace occ::mults
