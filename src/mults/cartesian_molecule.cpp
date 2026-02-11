#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cartesian_rotation.h>
#include <occ/mults/interaction_tensor.h>
#include <occ/mults/interaction_tensor_simd.h>
#include <occ/core/units.h>
#include <cmath>
#include <array>
#include <memory>

namespace occ::mults {

namespace {

/// SO(3) generator matrices for infinitesimal rotations about x, y, z axes.
/// dM/dp_k = G[k] * M at angle-axis parameter p = 0.
inline std::array<Mat3, 3> so3_generators() {
    std::array<Mat3, 3> G;
    G[0] << 0, 0, 0,  0, 0, -1,  0, 1, 0;
    G[1] << 0, 0, 1,  0, 0, 0,  -1, 0, 0;
    G[2] << 0, -1, 0,  1, 0, 0,  0, 0, 0;
    return G;
}

/// Precompute rotation derivatives for all sites.
/// d_multipoles[k][i] = d(lab_multipole_i)/dp_k
void compute_rotation_derivatives(
    BodyFrameData &bd,
    const std::vector<CartesianSite> &sites) {
    const auto G = so3_generators();
    const size_t n = sites.size();

    for (int k = 0; k < 3; ++k) {
        Mat3 dM_k = G[k] * bd.rotation;
        bd.d_multipoles[k].resize(n);
        for (size_t i = 0; i < n; ++i) {
            if (sites[i].rank <= 0) {
                bd.d_multipoles[k][i] = {};
                continue;
            }
            rotate_cartesian_multipole_derivative<4>(
                bd.body_multipoles[i], bd.rotation, dM_k,
                bd.d_multipoles[k][i]);
        }
    }
}

} // anonymous namespace

CartesianMolecule CartesianMolecule::from_lab_sites(
    const std::vector<std::pair<occ::dma::Mult, Vec3>> &site_data) {
    CartesianMolecule mol;
    mol.sites.reserve(site_data.size());
    for (const auto &[mult, pos] : site_data) {
        CartesianSite site;
        spherical_to_cartesian<4>(mult, site.cart);
        site.position = pos;
        site.rank = site.cart.effective_rank();
        mol.sites.push_back(std::move(site));
    }
    return mol;
}

CartesianMolecule CartesianMolecule::from_body_frame(
    const std::vector<std::pair<occ::dma::Mult, Vec3>> &body_sites,
    const Mat3 &rotation, const Vec3 &center) {
    CartesianMolecule mol;
    mol.sites.reserve(body_sites.size());
    for (const auto &[mult, body_pos] : body_sites) {
        CartesianSite site;
        spherical_to_cartesian<4>(mult, site.cart);
        site.position = center + rotation * body_pos;
        site.rank = site.cart.effective_rank();
        mol.sites.push_back(std::move(site));
    }
    return mol;
}

CartesianMolecule CartesianMolecule::from_body_frame_with_rotation(
    const std::vector<std::pair<occ::dma::Mult, Vec3>> &body_sites,
    const Mat3 &rotation, const Vec3 &center) {
    CartesianMolecule mol;
    mol.sites.reserve(body_sites.size());

    BodyFrameData bd;
    bd.center = center;
    bd.rotation = rotation;
    bd.body_multipoles.reserve(body_sites.size());
    bd.body_offsets.reserve(body_sites.size());

    for (const auto &[mult, body_pos] : body_sites) {
        // Convert spherical to Cartesian in body frame
        CartesianMultipole<4> body_cart;
        spherical_to_cartesian<4>(mult, body_cart);
        bd.body_multipoles.push_back(body_cart);
        bd.body_offsets.push_back(body_pos);

        // Rotate body-frame Cartesian multipole to lab frame
        CartesianSite site;
        rotate_cartesian_multipole<4>(body_cart, rotation, site.cart);
        site.position = center + rotation * body_pos;
        site.rank = site.cart.effective_rank();
        mol.sites.push_back(std::move(site));
    }

    compute_rotation_derivatives(bd, mol.sites);
    mol.body_data = std::move(bd);
    return mol;
}

void CartesianMolecule::update_positions(
    const Mat3 &rotation, const Vec3 &center,
    const std::vector<Vec3> &body_offsets) {
    for (size_t i = 0; i < sites.size() && i < body_offsets.size(); ++i) {
        sites[i].position = center + rotation * body_offsets[i];
    }
}

void CartesianMolecule::update_orientation(
    const Mat3 &new_rotation, const Vec3 &new_center) {
    if (!body_data) return;
    auto &bd = *body_data;
    bd.rotation = new_rotation;
    bd.center = new_center;
    for (size_t i = 0; i < sites.size(); ++i) {
        sites[i].position = new_center + new_rotation * bd.body_offsets[i];
        rotate_cartesian_multipole<4>(bd.body_multipoles[i], new_rotation,
                                      sites[i].cart);
        sites[i].rank = sites[i].cart.effective_rank();
    }
    compute_rotation_derivatives(bd, sites);
}

namespace {

/// Dispatch to order-specific tensor computation for a site pair.
template <int Order>
double compute_pair_at_order(const CartesianMultipole<4> &cartA, int rankA,
                             const CartesianMultipole<4> &cartB, int rankB,
                             double Rx, double Ry, double Rz) {
    InteractionTensor<Order> T;
    compute_interaction_tensor<Order>(Rx, Ry, Rz, T);
    return contract_ranked<Order>(cartA, rankA, T, cartB, rankB);
}

double dispatch_pair(const CartesianMultipole<4> &cartA, int rankA,
                     const CartesianMultipole<4> &cartB, int rankB,
                     double Rx, double Ry, double Rz) {
    int order = rankA + rankB;
    switch (order) {
        case 0: return compute_pair_at_order<0>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 1: return compute_pair_at_order<1>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 2: return compute_pair_at_order<2>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 3: return compute_pair_at_order<3>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 4: return compute_pair_at_order<4>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 5: return compute_pair_at_order<5>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 6: return compute_pair_at_order<6>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 7: return compute_pair_at_order<7>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        case 8: return compute_pair_at_order<8>(cartA, rankA, cartB, rankB, Rx, Ry, Rz);
        default: return 0.0;
    }
}

// ---- SIMD batch helpers ----

/// Pair descriptor for batching.
struct PairRef {
    const CartesianSite *siteA;
    const CartesianSite *siteB;
    double Rx, Ry, Rz;
};

/// Process a SIMD batch of pairs at a specific order.
template <int Order>
double process_batch(const PairRef *pairs, int count) {
    constexpr int B = simd_batch_size;
    double energy = 0.0;

    int i = 0;
    // Process full SIMD batches
    for (; i + B <= count; i += B) {
        alignas(64) double Rx[B], Ry[B], Rz[B];
        for (int b = 0; b < B; ++b) {
            Rx[b] = pairs[i + b].Rx;
            Ry[b] = pairs[i + b].Ry;
            Rz[b] = pairs[i + b].Rz;
        }

        InteractionTensorBatch<Order, B> T;
        compute_interaction_tensor_batch<Order>(Rx, Ry, Rz, T);

        for (int b = 0; b < B; ++b) {
            energy += contract_ranked_from_batch<Order, B>(
                pairs[i + b].siteA->cart, pairs[i + b].siteA->rank,
                T, b,
                pairs[i + b].siteB->cart, pairs[i + b].siteB->rank);
        }
    }

    // Scalar remainder
    for (; i < count; ++i) {
        energy += compute_pair_at_order<Order>(
            pairs[i].siteA->cart, pairs[i].siteA->rank,
            pairs[i].siteB->cart, pairs[i].siteB->rank,
            pairs[i].Rx, pairs[i].Ry, pairs[i].Rz);
    }

    return energy;
}

/// Process all pairs in a bucket for a given runtime order.
double dispatch_batch(int order, const PairRef *pairs, int count) {
    switch (order) {
        case 0: return process_batch<0>(pairs, count);
        case 1: return process_batch<1>(pairs, count);
        case 2: return process_batch<2>(pairs, count);
        case 3: return process_batch<3>(pairs, count);
        case 4: return process_batch<4>(pairs, count);
        case 5: return process_batch<5>(pairs, count);
        case 6: return process_batch<6>(pairs, count);
        case 7: return process_batch<7>(pairs, count);
        case 8: return process_batch<8>(pairs, count);
        default: return 0.0;
    }
}

} // anonymous namespace

double compute_site_pair_energy(
    const CartesianSite &siteA,
    const CartesianSite &siteB) {
    if (siteA.rank < 0 || siteB.rank < 0) return 0.0;
    Vec3 R = siteB.position - siteA.position;
    return dispatch_pair(siteA.cart, siteA.rank,
                         siteB.cart, siteB.rank,
                         R[0], R[1], R[2]);
}

double compute_molecule_interaction(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB) {
    double energy = 0.0;
    for (const auto &sA : molA.sites) {
        if (sA.rank < 0) continue;
        for (const auto &sB : molB.sites) {
            if (sB.rank < 0) continue;
            // Positions are in Angstrom; interaction tensors expect Bohr
            Vec3 R = (sB.position - sA.position) / occ::units::BOHR_TO_ANGSTROM;
            energy += dispatch_pair(sA.cart, sA.rank,
                                    sB.cart, sB.rank,
                                    R[0], R[1], R[2]);
        }
    }
    return energy * occ::units::AU_TO_KJ_PER_MOL;
}

double compute_molecule_interaction_simd(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB) {

    const int nA = static_cast<int>(molA.sites.size());
    const int nB = static_cast<int>(molB.sites.size());
    const int max_pairs = nA * nB;

    if (max_pairs == 0) return 0.0;

    // Stack-allocated pair buffer + order-indexed counts/offsets.
    // For small molecules (e.g. 10x10 = 100 pairs), this fits on the stack.
    // For larger cases, fall back to heap.
    constexpr int stack_limit = 256;
    PairRef stack_buf[stack_limit];
    std::unique_ptr<PairRef[]> heap_buf;
    PairRef *pairs = stack_buf;
    if (max_pairs > stack_limit) {
        heap_buf = std::make_unique<PairRef[]>(max_pairs);
        pairs = heap_buf.get();
    }

    // Count pairs per order (0..8)
    std::array<int, 9> counts{};
    int total = 0;

    // First pass: count per order
    for (int i = 0; i < nA; ++i) {
        if (molA.sites[i].rank < 0) continue;
        for (int j = 0; j < nB; ++j) {
            if (molB.sites[j].rank < 0) continue;
            int order = molA.sites[i].rank + molB.sites[j].rank;
            ++counts[order];
            ++total;
        }
    }

    // Compute offsets (prefix sum)
    std::array<int, 9> offsets{};
    for (int o = 1; o <= 8; ++o)
        offsets[o] = offsets[o - 1] + counts[o - 1];

    // Second pass: fill pairs sorted by order
    std::array<int, 9> pos = offsets; // current write position per order
    for (int i = 0; i < nA; ++i) {
        const auto &sA = molA.sites[i];
        if (sA.rank < 0) continue;
        for (int j = 0; j < nB; ++j) {
            const auto &sB = molB.sites[j];
            if (sB.rank < 0) continue;
            int order = sA.rank + sB.rank;
            // Positions are in Angstrom; interaction tensors expect Bohr
            Vec3 R = (sB.position - sA.position) / occ::units::BOHR_TO_ANGSTROM;
            pairs[pos[order]++] = {&sA, &sB, R[0], R[1], R[2]};
        }
    }

    // Process each order group with SIMD batching
    double energy = 0.0;
    for (int order = 0; order <= 8; ++order) {
        if (counts[order] == 0) continue;
        energy += dispatch_batch(order, &pairs[offsets[order]], counts[order]);
    }
    return energy * occ::units::AU_TO_KJ_PER_MOL;
}

} // namespace occ::mults
