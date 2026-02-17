#include <occ/mults/cartesian_force.h>
#include <occ/mults/interaction_tensor.h>
#include <occ/core/units.h>
#include <cmath>

namespace occ::mults {

namespace {

/// Dispatch to order-specific tensor computation for a site pair (with force).
template <int Order>
kernel_detail::EnergyGradient compute_pair_ef_at_order(
    const CartesianMultipole<4> &cartA, int rankA,
    const CartesianMultipole<4> &cartB, int rankB,
    double Rx, double Ry, double Rz) {
    InteractionTensor<Order + 1> T;
    compute_interaction_tensor<Order + 1>(Rx, Ry, Rz, T);
    return contract_ranked_with_force<Order>(cartA, rankA, T, cartB, rankB);
}

kernel_detail::EnergyGradient dispatch_pair_ef(
    const CartesianMultipole<4> &cartA, int rankA,
    const CartesianMultipole<4> &cartB, int rankB,
    double Rx, double Ry, double Rz) {
    int order = rankA + rankB;
    switch (order) {
        case 0: return compute_pair_ef_at_order<0>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 1: return compute_pair_ef_at_order<1>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 2: return compute_pair_ef_at_order<2>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 3: return compute_pair_ef_at_order<3>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 4: return compute_pair_ef_at_order<4>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 5: return compute_pair_ef_at_order<5>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 6: return compute_pair_ef_at_order<6>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 7: return compute_pair_ef_at_order<7>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 8: return compute_pair_ef_at_order<8>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        default: return {0.0, {0.0, 0.0, 0.0}};
    }
}

/// Dispatch combined energy+force+fields computation (single T-tensor).
EnergyForceFields dispatch_pair_ef_and_fields(
    const CartesianMultipole<4> &cartA, int rankA,
    const CartesianMultipole<4> &cartB, int rankB,
    double Rx, double Ry, double Rz) {
    int order = rankA + rankB;
    switch (order) {
        case 0: return compute_pair_ef_and_fields<0>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 1: return compute_pair_ef_and_fields<1>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 2: return compute_pair_ef_and_fields<2>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 3: return compute_pair_ef_and_fields<3>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 4: return compute_pair_ef_and_fields<4>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 5: return compute_pair_ef_and_fields<5>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 6: return compute_pair_ef_and_fields<6>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 7: return compute_pair_ef_and_fields<7>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 8: return compute_pair_ef_and_fields<8>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        default: return {};
    }
}

} // anonymous namespace

// -------------------------------------------------------------------
// Stage 1: Per-site force computation
// -------------------------------------------------------------------

CartesianForceResult compute_site_pair_energy_force(
    const CartesianSite &siteA,
    const CartesianSite &siteB) {
    if (siteA.rank < 0 || siteB.rank < 0)
        return {};
    // Positions are stored in Angstrom; interaction tensors assume Bohr.
    Vec3 R_ang = siteB.position - siteA.position;
    Vec3 R = R_ang / occ::units::BOHR_TO_ANGSTROM;
    auto eg = dispatch_pair_ef(siteA.cart, siteA.rank,
                               siteB.cart, siteB.rank,
                               R[0], R[1], R[2]);
    return {eg.energy, Vec3(eg.grad[0], eg.grad[1], eg.grad[2])};
}

// -------------------------------------------------------------------
// Stage 2: Molecule-level force aggregation
// -------------------------------------------------------------------

MoleculeForceResult compute_molecule_forces(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB) {
    MoleculeForceResult result;
    result.forces_A.assign(molA.sites.size(), Vec3::Zero());
    result.forces_B.assign(molB.sites.size(), Vec3::Zero());

    for (size_t i = 0; i < molA.sites.size(); ++i) {
        const auto &sA = molA.sites[i];
        if (sA.rank < 0) continue;
        for (size_t j = 0; j < molB.sites.size(); ++j) {
            const auto &sB = molB.sites[j];
            if (sB.rank < 0) continue;
            Vec3 R_ang = sB.position - sA.position;
            Vec3 R = R_ang / occ::units::BOHR_TO_ANGSTROM;
            auto eg = dispatch_pair_ef(sA.cart, sA.rank,
                                       sB.cart, sB.rank,
                                       R[0], R[1], R[2]);
            // Convert from Hartree/Bohr to kJ/mol/Angstrom
            Vec3 grad_bohr(eg.grad[0], eg.grad[1], eg.grad[2]);
            Vec3 grad = grad_bohr * (occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM);
            result.energy += eg.energy * occ::units::AU_TO_KJ_PER_MOL;
            result.forces_A[i] += grad;   // F_A = +dE/dR
            result.forces_B[j] -= grad;   // F_B = -dE/dR (Newton's 3rd law)
        }
    }
    return result;
}

RigidBodyForceResult aggregate_rigid_body_forces(
    const std::vector<Vec3> &site_forces,
    const std::vector<Vec3> &site_positions,
    const Vec3 &center_of_mass,
    const Mat3 &rotation) {
    RigidBodyForceResult result;

    for (size_t i = 0; i < site_forces.size(); ++i) {
        result.force += site_forces[i];
        Vec3 lever = site_positions[i] - center_of_mass;
        result.torque_lab += lever.cross(site_forces[i]);
    }
    result.torque_body = rotation.transpose() * result.torque_lab;
    return result;
}

// -------------------------------------------------------------------
// Stage 4: Full force/torque with multipole rotation contribution
// -------------------------------------------------------------------

/// Fused energy+force+fields kernel with pre-weighted multipoles and lookup tables.
///
/// Computes energy, force gradient, field at A from B, and field at B from A
/// in a single pass over (i,j) pairs. Only includes terms where lA + lB <= MaxOrder.
///
/// Key optimizations vs the previous 3-loop version:
/// 1. Precomputed hermite_index lookup tables (add_table4) replace ~10 int ops per call
/// 2. fieldA is a free byproduct of the energy inner sum (no separate loop)
/// 3. fieldB is accumulated with 1 extra FMA per inner iteration (no separate loop)
/// 4. Pre-weighted multipole arrays avoid redundant per-pair weighting
template <int MaxOrder>
static EnergyForceFields compute_pair_fused(
    const double *wA, int rankA,
    const double *wB, int rankB,
    double Rx, double Ry, double Rz) {
    using namespace kernel_detail;

    EnergyForceFields result;

    InteractionTensor<MaxOrder + 1> T;
    compute_interaction_tensor<MaxOrder + 1>(Rx, Ry, Rz, T);

    const int nA_eff = nhermsum(std::min(rankA, MaxOrder));
    const int nB_max = nhermsum(std::min(rankB, MaxOrder));

    double energy = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
    alignas(64) double fB[nhermsum(4)] = {};

    for (int i = 0; i < nA_eff; ++i) {
        auto [ta, ua, va] = tuv4.entries[i];
        int lA = ta + ua + va;
        int nB_eff = nhermsum(std::min(rankB, MaxOrder - lA));
        double wAi = wA[i];
        double eA = 0.0, gxA = 0.0, gyA = 0.0, gzA = 0.0;
        for (int j = 0; j < nB_eff; ++j) {
            double T_e = T.data[add_table4.energy[i][j]];
            double wBj = wB[j];
            eA  += T_e * wBj;
            gxA += T.data[add_table4.force_x[i][j]] * wBj;
            gyA += T.data[add_table4.force_y[i][j]] * wBj;
            gzA += T.data[add_table4.force_z[i][j]] * wBj;
            fB[j] += T_e * wAi;
        }
        result.fieldA.data[i] = eA;
        energy += wAi * eA;
        gx += wAi * gxA;
        gy += wAi * gyA;
        gz += wAi * gzA;
    }
    result.eg = {energy, {gx, gy, gz}};
    for (int j = 0; j < nB_max; ++j)
        result.fieldB.data[j] = fB[j];

    return result;
}

/// Dispatch fused kernel by max_order (runtime → compile-time).
static EnergyForceFields dispatch_pair_fused(
    const double *wA, int rankA,
    const double *wB, int rankB,
    double Rx, double Ry, double Rz,
    int max_order) {
    switch (max_order) {
        case 0: return compute_pair_fused<0>(wA, rankA, wB, rankB, Rx, Ry, Rz);
        case 1: return compute_pair_fused<1>(wA, rankA, wB, rankB, Rx, Ry, Rz);
        case 2: return compute_pair_fused<2>(wA, rankA, wB, rankB, Rx, Ry, Rz);
        case 3: return compute_pair_fused<3>(wA, rankA, wB, rankB, Rx, Ry, Rz);
        case 4: return compute_pair_fused<4>(wA, rankA, wB, rankB, Rx, Ry, Rz);
        case 5: return compute_pair_fused<5>(wA, rankA, wB, rankB, Rx, Ry, Rz);
        case 6: return compute_pair_fused<6>(wA, rankA, wB, rankB, Rx, Ry, Rz);
        case 7: return compute_pair_fused<7>(wA, rankA, wB, rankB, Rx, Ry, Rz);
        case 8: return compute_pair_fused<8>(wA, rankA, wB, rankB, Rx, Ry, Rz);
        default: return {};
    }
}

/// Combined energy+force+fields contraction with interaction-order filter.
/// Non-preweighted version for the non-hot-path (full-order) case.
template <int MaxOrder>
static EnergyForceFields compute_pair_ef_and_fields_filtered(
    const CartesianMultipole<4> &cartA, int rankA,
    const CartesianMultipole<4> &cartB, int rankB,
    double Rx, double Ry, double Rz) {
    using namespace kernel_detail;

    const int nA = nhermsum(rankA);
    const int nB = nhermsum(rankB);

    alignas(64) double wA[nhermsum(4)];
    alignas(64) double wB[nhermsum(4)];
    for (int i = 0; i < nA; ++i)
        wA[i] = weights4.sign_inv_fact[i] * cartA.data[i];
    for (int j = 0; j < nB; ++j)
        wB[j] = weights4.inv_fact[j] * cartB.data[j];

    return compute_pair_fused<MaxOrder>(wA, rankA, wB, rankB, Rx, Ry, Rz);
}

/// Dispatch filtered ef+fields by max_order (runtime → compile-time).
EnergyForceFields dispatch_pair_ef_and_fields_filtered(
    const CartesianMultipole<4> &cartA, int rankA,
    const CartesianMultipole<4> &cartB, int rankB,
    double Rx, double Ry, double Rz,
    int max_order) {
    switch (max_order) {
        case 0: return compute_pair_ef_and_fields_filtered<0>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 1: return compute_pair_ef_and_fields_filtered<1>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 2: return compute_pair_ef_and_fields_filtered<2>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 3: return compute_pair_ef_and_fields_filtered<3>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 4: return compute_pair_ef_and_fields_filtered<4>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 5: return compute_pair_ef_and_fields_filtered<5>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 6: return compute_pair_ef_and_fields_filtered<6>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 7: return compute_pair_ef_and_fields_filtered<7>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 8: return compute_pair_ef_and_fields_filtered<8>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        default: return {};
    }
}

/// Pre-weighted multipole data for a single site.
struct PreweightedSite {
    alignas(64) double w[nhermsum(4)];
};

FullRigidBodyResult compute_molecule_forces_torques(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB,
    double site_cutoff,
    int max_interaction_order,
    const Vec3 &offset_B) {
    using namespace kernel_detail;
    FullRigidBodyResult result;

    const size_t nA = molA.sites.size();
    const size_t nB = molB.sites.size();

    std::vector<Vec3> forces_A(nA, Vec3::Zero());
    std::vector<Vec3> forces_B(nB, Vec3::Zero());

    // Per-site interaction fields for torque computation
    std::vector<CartesianMultipole<4>> fields_A(nA);
    std::vector<CartesianMultipole<4>> fields_B(nB);

    const bool use_site_cutoff = (site_cutoff > 0.0);
    const bool use_order_cutoff = (max_interaction_order >= 0);

    // Precompute weighted multipoles once per molecule pair (Stage 3).
    // molA sites use sign_inv_fact weighting (A-side).
    // molB sites use inv_fact weighting (B-side).
    static constexpr size_t MAX_SITES_STACK = 16;
    PreweightedSite pw_stack_A[MAX_SITES_STACK], pw_stack_B[MAX_SITES_STACK];
    std::unique_ptr<PreweightedSite[]> pw_heap_A, pw_heap_B;
    PreweightedSite *pwA = nA <= MAX_SITES_STACK
        ? pw_stack_A
        : (pw_heap_A.reset(new PreweightedSite[nA]), pw_heap_A.get());
    PreweightedSite *pwB = nB <= MAX_SITES_STACK
        ? pw_stack_B
        : (pw_heap_B.reset(new PreweightedSite[nB]), pw_heap_B.get());

    for (size_t i = 0; i < nA; ++i) {
        const auto &s = molA.sites[i];
        if (s.rank < 0) continue;
        int nc = nhermsum(s.rank);
        for (int c = 0; c < nc; ++c)
            pwA[i].w[c] = weights4.sign_inv_fact[c] * s.cart.data[c];
    }
    for (size_t j = 0; j < nB; ++j) {
        const auto &s = molB.sites[j];
        if (s.rank < 0) continue;
        int nc = nhermsum(s.rank);
        for (int c = 0; c < nc; ++c)
            pwB[j].w[c] = weights4.inv_fact[c] * s.cart.data[c];
    }

    // Site-pair loop: compute energy, per-site forces, and interaction fields
    for (size_t i = 0; i < nA; ++i) {
        const auto &sA = molA.sites[i];
        if (sA.rank < 0) continue;
        for (size_t j = 0; j < nB; ++j) {
            const auto &sB = molB.sites[j];
            if (sB.rank < 0) continue;
            Vec3 R_ang = (sB.position + offset_B) - sA.position;
            if (use_site_cutoff && R_ang.norm() > site_cutoff) continue;
            Vec3 R = R_ang / occ::units::BOHR_TO_ANGSTROM;
            double Rx = R[0], Ry = R[1], Rz = R[2];

            EnergyForceFields eff;
            if (use_order_cutoff) {
                // Fused kernel with pre-weighted multipoles and lookup tables
                eff = dispatch_pair_fused(
                    pwA[i].w, sA.rank, pwB[j].w, sB.rank,
                    Rx, Ry, Rz, max_interaction_order);
            } else {
                // Full order: T-tensor at rankA+rankB+1
                eff = dispatch_pair_ef_and_fields(sA.cart, sA.rank,
                                                  sB.cart, sB.rank,
                                                  Rx, Ry, Rz);
            }
            Vec3 grad_bohr(eff.eg.grad[0], eff.eg.grad[1], eff.eg.grad[2]);
            Vec3 grad = grad_bohr * (occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM);
            result.energy += eff.eg.energy * occ::units::AU_TO_KJ_PER_MOL;
            forces_A[i] += grad;
            forces_B[j] -= grad;

            // Accumulate interaction fields for torque computation
            for (int k = 0; k < CartesianMultipole<4>::size; ++k) {
                fields_A[i].data[k] += eff.fieldA.data[k];
                fields_B[j].data[k] += eff.fieldB.data[k];
            }
        }
    }

    // Translational forces
    for (size_t i = 0; i < nA; ++i)
        result.force_A += forces_A[i];
    for (size_t j = 0; j < nB; ++j)
        result.force_B += forces_B[j];

    // Lever-arm torque and multipole rotation torque (requires body-frame data).
    // Rotation derivatives d_multipoles[k][i] are precomputed in BodyFrameData
    // when orientation is set, so torque evaluation just contracts them with fields.
    if (molA.body_data) {
        const auto &bd = *molA.body_data;
        Vec3 torque_lab_A = Vec3::Zero();
        for (size_t i = 0; i < nA; ++i) {
            Vec3 lever = molA.sites[i].position - bd.center;
            torque_lab_A += lever.cross(forces_A[i]);
        }

        // Multipole rotation torque: contract precomputed derivatives with fields
        Vec3 torque_multipole_A = Vec3::Zero();
        for (int k = 0; k < 3; ++k) {
            double tau_k = 0.0;
            for (size_t i = 0; i < nA; ++i) {
                if (molA.sites[i].rank <= 0) continue;
                const auto &d_cart = bd.d_multipoles[k][i];
                int nComp = nhermsum(molA.sites[i].rank);
                for (int c = 0; c < nComp; ++c) {
                    tau_k += weights4.sign_inv_fact[c] * d_cart.data[c]
                             * fields_A[i].data[c];
                }
            }
            torque_multipole_A[k] = tau_k * occ::units::AU_TO_KJ_PER_MOL;
        }

        // dE/dp = -(lever torque) + (multipole torque)
        result.grad_angle_axis_A = -torque_lab_A + torque_multipole_A;
        result.torque_A_body = bd.rotation.transpose() * result.grad_angle_axis_A;
    }

    if (molB.body_data) {
        const auto &bd = *molB.body_data;
        Vec3 torque_lab_B = Vec3::Zero();
        for (size_t j = 0; j < nB; ++j) {
            Vec3 lever = molB.sites[j].position - bd.center;
            torque_lab_B += lever.cross(forces_B[j]);
        }

        Vec3 torque_multipole_B = Vec3::Zero();
        for (int k = 0; k < 3; ++k) {
            double tau_k = 0.0;
            for (size_t j = 0; j < nB; ++j) {
                if (molB.sites[j].rank <= 0) continue;
                const auto &d_cart = bd.d_multipoles[k][j];
                // B-side uses inv_fact (no (-1)^l sign) for contraction
                int nComp = nhermsum(molB.sites[j].rank);
                for (int c = 0; c < nComp; ++c) {
                    tau_k += weights4.inv_fact[c] * d_cart.data[c]
                             * fields_B[j].data[c];
                }
            }
            torque_multipole_B[k] = tau_k * occ::units::AU_TO_KJ_PER_MOL;
        }

        result.grad_angle_axis_B = -torque_lab_B + torque_multipole_B;
        result.torque_B_body = bd.rotation.transpose() * result.grad_angle_axis_B;
    }

    return result;
}

} // namespace occ::mults
