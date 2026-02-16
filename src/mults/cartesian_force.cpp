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

/// Dispatch for interaction field computation, templatized on signedness.
/// Signed=false: field at A from B (B uses inv_fact, no sign).
/// Signed=true:  field at B from A (A uses sign_inv_fact, with (-1)^l sign).
template <int Order, bool Signed>
void compute_pair_field_impl(
    int rankLocal,
    const CartesianMultipole<4> &other, int rankOther,
    double Rx, double Ry, double Rz,
    CartesianMultipole<4> &field) {
    InteractionTensor<Order> T;
    compute_interaction_tensor<Order>(Rx, Ry, Rz, T);
    compute_interaction_field<Order, Signed>(rankLocal, T, other, rankOther, field);
}

template <bool Signed>
void dispatch_pair_field_impl(
    int rankLocal,
    const CartesianMultipole<4> &other, int rankOther,
    double Rx, double Ry, double Rz,
    CartesianMultipole<4> &field) {
    int order = rankLocal + rankOther;
    switch (order) {
        case 0: compute_pair_field_impl<0, Signed>(rankLocal, other, rankOther, Rx, Ry, Rz, field); break;
        case 1: compute_pair_field_impl<1, Signed>(rankLocal, other, rankOther, Rx, Ry, Rz, field); break;
        case 2: compute_pair_field_impl<2, Signed>(rankLocal, other, rankOther, Rx, Ry, Rz, field); break;
        case 3: compute_pair_field_impl<3, Signed>(rankLocal, other, rankOther, Rx, Ry, Rz, field); break;
        case 4: compute_pair_field_impl<4, Signed>(rankLocal, other, rankOther, Rx, Ry, Rz, field); break;
        case 5: compute_pair_field_impl<5, Signed>(rankLocal, other, rankOther, Rx, Ry, Rz, field); break;
        case 6: compute_pair_field_impl<6, Signed>(rankLocal, other, rankOther, Rx, Ry, Rz, field); break;
        case 7: compute_pair_field_impl<7, Signed>(rankLocal, other, rankOther, Rx, Ry, Rz, field); break;
        case 8: compute_pair_field_impl<8, Signed>(rankLocal, other, rankOther, Rx, Ry, Rz, field); break;
        default: break;
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

/// Energy-only contraction with interaction-order filter.
/// Computes E = sum_{lA+lB <= max_order} A[i] * T[i+j] * B[j]
/// where lA = rank of component i, lB = rank of component j.
static double contract_energy_order_filtered(
    const CartesianMultipole<4> &A, int rankA,
    const CartesianMultipole<4> &B, int rankB,
    double Rx, double Ry, double Rz,
    int max_order) {
    using namespace kernel_detail;

    // Compute T-tensor at the truncated order (not full rankA+rankB)
    int eff_order = std::min(rankA + rankB, max_order);

    // We need a T-tensor of order max_order+1 (for gradient) but here
    // we only need energy, so order max_order suffices.
    // Use order 8 T-tensor (the largest we support) and just don't
    // access entries beyond max_order.
    InteractionTensor<9> T;
    compute_interaction_tensor<9>(Rx, Ry, Rz, T);

    const int nA = nhermsum(rankA);
    const int nB = nhermsum(rankB);

    double energy = 0.0;
    for (int i = 0; i < nA; ++i) {
        auto [ta, ua, va] = tuv4.entries[i];
        int lA = ta + ua + va;
        double wA = weights4.sign_inv_fact[i] * A.data[i];
        if (wA == 0.0) continue;
        for (int j = 0; j < nB; ++j) {
            auto [tb, ub, vb] = tuv4.entries[j];
            int lB = tb + ub + vb;
            if (lA + lB > max_order) continue;
            double wB = weights4.inv_fact[j] * B.data[j];
            energy += wA * T.data[hermite_index(ta+tb, ua+ub, va+vb)] * wB;
        }
    }
    return energy;
}

FullRigidBodyResult compute_molecule_forces_torques(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB,
    double site_cutoff,
    int max_interaction_order) {
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

    // Single pass: compute energy, per-site forces, and interaction fields
    for (size_t i = 0; i < nA; ++i) {
        const auto &sA = molA.sites[i];
        if (sA.rank < 0) continue;
        for (size_t j = 0; j < nB; ++j) {
            const auto &sB = molB.sites[j];
            if (sB.rank < 0) continue;
            Vec3 R_ang = sB.position - sA.position;
            if (use_site_cutoff && R_ang.norm() > site_cutoff) continue;
            Vec3 R = R_ang / occ::units::BOHR_TO_ANGSTROM;
            double Rx = R[0], Ry = R[1], Rz = R[2];

            // Energy and force
            // When order-truncated, use filtered energy (no force/torque)
            if (use_order_cutoff && (sA.rank + sB.rank) > max_interaction_order) {
                // This pair has contributions at order > max, but may also
                // have contributions at order <= max. Use filtered contraction.
                double e = contract_energy_order_filtered(
                    sA.cart, sA.rank, sB.cart, sB.rank,
                    Rx, Ry, Rz, max_interaction_order);
                result.energy += e * occ::units::AU_TO_KJ_PER_MOL;
                // Skip force/torque for truncated pairs
                // (strain derivatives only need energy)
                continue;
            }

            auto eg = dispatch_pair_ef(sA.cart, sA.rank,
                                       sB.cart, sB.rank,
                                       Rx, Ry, Rz);
            Vec3 grad_bohr(eg.grad[0], eg.grad[1], eg.grad[2]);
            Vec3 grad = grad_bohr * (occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM);
            result.energy += eg.energy * occ::units::AU_TO_KJ_PER_MOL;
            forces_A[i] += grad;
            forces_B[j] -= grad;

            // Interaction field at site A from site B (for A's torque)
            CartesianMultipole<4> fieldA_ij;
            dispatch_pair_field_impl<false>(sA.rank, sB.cart, sB.rank,
                                            Rx, Ry, Rz, fieldA_ij);
            for (int k = 0; k < CartesianMultipole<4>::size; ++k)
                fields_A[i].data[k] += fieldA_ij.data[k];

            // Interaction field at site B from site A (for B's torque)
            // field_B[j] = Σ_i sign_inv_fact[i] * A[i] * T(R; ti+tj, ...)
            // Uses +R (same direction) and signed weights for A
            CartesianMultipole<4> fieldB_ji;
            dispatch_pair_field_impl<true>(sB.rank, sA.cart, sA.rank,
                                           Rx, Ry, Rz, fieldB_ji);
            for (int k = 0; k < CartesianMultipole<4>::size; ++k)
                fields_B[j].data[k] += fieldB_ji.data[k];
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
                using namespace kernel_detail;
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
                using namespace kernel_detail;
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
