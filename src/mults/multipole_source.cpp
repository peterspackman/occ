#include <occ/mults/multipole_source.h>
#include <occ/mults/interaction_tensor.h>
#include <occ/mults/interaction_tensor_simd.h>
#include <occ/mults/rigid_body.h>

namespace occ::mults {

// ======================================================================
// Specialized probe kernels: rank-0 probe eliminates inner B-loop
// ======================================================================

namespace {

using namespace kernel_detail;

// --- Scalar: single-point potential for one source site ---

template <int Order>
double probe_potential_scalar(const CartesianMultipole<4> &A, int rankA,
                              double Rx, double Ry, double Rz) {
    InteractionTensor<Order> T;
    compute_interaction_tensor<Order>(Rx, Ry, Rz, T);
    const int nA = nhermsum(rankA);
    double energy = 0.0;
    for (int i = 0; i < nA; ++i) {
        auto [ta, ua, va] = tuv4.entries[i];
        energy += weights4.sign_inv_fact[i] * A.data[i] * T(ta, ua, va);
    }
    return energy;
}

double dispatch_probe_potential_scalar(const CartesianMultipole<4> &A, int rankA,
                                       double Rx, double Ry, double Rz) {
    switch (rankA) {
        case 0: return probe_potential_scalar<0>(A, rankA, Rx, Ry, Rz);
        case 1: return probe_potential_scalar<1>(A, rankA, Rx, Ry, Rz);
        case 2: return probe_potential_scalar<2>(A, rankA, Rx, Ry, Rz);
        case 3: return probe_potential_scalar<3>(A, rankA, Rx, Ry, Rz);
        case 4: return probe_potential_scalar<4>(A, rankA, Rx, Ry, Rz);
        default: return 0.0;
    }
}

// --- Scalar: single-point field (potential + gradient) for one source site ---

template <int Order>
EnergyGradient probe_field_scalar(const CartesianMultipole<4> &A, int rankA,
                                   double Rx, double Ry, double Rz) {
    InteractionTensor<Order + 1> T;
    compute_interaction_tensor<Order + 1>(Rx, Ry, Rz, T);
    const int nA = nhermsum(rankA);
    double energy = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
    for (int i = 0; i < nA; ++i) {
        auto [ta, ua, va] = tuv4.entries[i];
        double w = weights4.sign_inv_fact[i] * A.data[i];
        energy += w * T(ta, ua, va);
        gx += w * T(ta + 1, ua, va);
        gy += w * T(ta, ua + 1, va);
        gz += w * T(ta, ua, va + 1);
    }
    return {energy, {gx, gy, gz}};
}

EnergyGradient dispatch_probe_field_scalar(const CartesianMultipole<4> &A, int rankA,
                                            double Rx, double Ry, double Rz) {
    switch (rankA) {
        case 0: return probe_field_scalar<0>(A, rankA, Rx, Ry, Rz);
        case 1: return probe_field_scalar<1>(A, rankA, Rx, Ry, Rz);
        case 2: return probe_field_scalar<2>(A, rankA, Rx, Ry, Rz);
        case 3: return probe_field_scalar<3>(A, rankA, Rx, Ry, Rz);
        case 4: return probe_field_scalar<4>(A, rankA, Rx, Ry, Rz);
        default: return {0.0, {0.0, 0.0, 0.0}};
    }
}

// --- SIMD-batched: N-point potential for one source site ---

template <int Order>
void probe_potential_batch(const CartesianSite &site,
                           Mat3NConstRef points,
                           double *result, int N) {
    constexpr int B = simd_batch_size;
    const int nA = nhermsum(site.rank);

    // Precompute weighted multipole once per site
    alignas(64) double wA[nhermsum(4)];
    for (int i = 0; i < nA; ++i)
        wA[i] = weights4.sign_inv_fact[i] * site.cart.data[i];

    const double sx = site.position.x();
    const double sy = site.position.y();
    const double sz = site.position.z();

    int p = 0;

    // SIMD batches
    for (; p + B <= N; p += B) {
        alignas(64) double Rx[B], Ry[B], Rz[B];
        for (int b = 0; b < B; ++b) {
            Rx[b] = points(0, p + b) - sx;
            Ry[b] = points(1, p + b) - sy;
            Rz[b] = points(2, p + b) - sz;
        }

        InteractionTensorBatch<Order, B> T;
        compute_interaction_tensor_batch<Order>(Rx, Ry, Rz, T);

        for (int b = 0; b < B; ++b) {
            double e = 0.0;
            for (int i = 0; i < nA; ++i) {
                auto [ta, ua, va] = tuv4.entries[i];
                e += wA[i] * T.data[hermite_index(ta, ua, va) * B + b];
            }
            result[p + b] += e;
        }
    }

    // Scalar remainder
    for (; p < N; ++p) {
        double Rx = points(0, p) - sx;
        double Ry = points(1, p) - sy;
        double Rz = points(2, p) - sz;
        InteractionTensor<Order> T;
        compute_interaction_tensor<Order>(Rx, Ry, Rz, T);
        double e = 0.0;
        for (int i = 0; i < nA; ++i) {
            auto [ta, ua, va] = tuv4.entries[i];
            e += wA[i] * T(ta, ua, va);
        }
        result[p] += e;
    }
}

void dispatch_probe_potential_batch(const CartesianSite &site,
                                    Mat3NConstRef points,
                                    double *result, int N) {
    switch (site.rank) {
        case 0: probe_potential_batch<0>(site, points, result, N); break;
        case 1: probe_potential_batch<1>(site, points, result, N); break;
        case 2: probe_potential_batch<2>(site, points, result, N); break;
        case 3: probe_potential_batch<3>(site, points, result, N); break;
        case 4: probe_potential_batch<4>(site, points, result, N); break;
        default: break;
    }
}

// --- SIMD-batched: N-point field for one source site ---

template <int Order>
void probe_field_batch(const CartesianSite &site,
                       Mat3NConstRef points,
                       Mat3N &result, int N) {
    constexpr int B = simd_batch_size;
    const int nA = nhermsum(site.rank);

    alignas(64) double wA[nhermsum(4)];
    for (int i = 0; i < nA; ++i)
        wA[i] = weights4.sign_inv_fact[i] * site.cart.data[i];

    const double sx = site.position.x();
    const double sy = site.position.y();
    const double sz = site.position.z();

    int p = 0;

    // SIMD batches (need Order+1 tensor for gradient)
    for (; p + B <= N; p += B) {
        alignas(64) double Rx[B], Ry[B], Rz[B];
        for (int b = 0; b < B; ++b) {
            Rx[b] = points(0, p + b) - sx;
            Ry[b] = points(1, p + b) - sy;
            Rz[b] = points(2, p + b) - sz;
        }

        InteractionTensorBatch<Order + 1, B> T;
        compute_interaction_tensor_batch<Order + 1>(Rx, Ry, Rz, T);

        for (int b = 0; b < B; ++b) {
            double gx = 0.0, gy = 0.0, gz = 0.0;
            for (int i = 0; i < nA; ++i) {
                auto [ta, ua, va] = tuv4.entries[i];
                double w = wA[i];
                gx += w * T.data[hermite_index(ta + 1, ua, va) * B + b];
                gy += w * T.data[hermite_index(ta, ua + 1, va) * B + b];
                gz += w * T.data[hermite_index(ta, ua, va + 1) * B + b];
            }
            // field = -gradient of potential
            result(0, p + b) -= gx;
            result(1, p + b) -= gy;
            result(2, p + b) -= gz;
        }
    }

    // Scalar remainder
    for (; p < N; ++p) {
        double Rx = points(0, p) - sx;
        double Ry = points(1, p) - sy;
        double Rz = points(2, p) - sz;
        InteractionTensor<Order + 1> T;
        compute_interaction_tensor<Order + 1>(Rx, Ry, Rz, T);
        double gx = 0.0, gy = 0.0, gz = 0.0;
        for (int i = 0; i < nA; ++i) {
            auto [ta, ua, va] = tuv4.entries[i];
            double w = wA[i];
            gx += w * T(ta + 1, ua, va);
            gy += w * T(ta, ua + 1, va);
            gz += w * T(ta, ua, va + 1);
        }
        result(0, p) -= gx;
        result(1, p) -= gy;
        result(2, p) -= gz;
    }
}

void dispatch_probe_field_batch(const CartesianSite &site,
                                Mat3NConstRef points,
                                Mat3N &result, int N) {
    switch (site.rank) {
        case 0: probe_field_batch<0>(site, points, result, N); break;
        case 1: probe_field_batch<1>(site, points, result, N); break;
        case 2: probe_field_batch<2>(site, points, result, N); break;
        case 3: probe_field_batch<3>(site, points, result, N); break;
        case 4: probe_field_batch<4>(site, points, result, N); break;
        default: break;
    }
}

} // anonymous namespace

// ======================================================================
// Construction
// ======================================================================

MultipoleSource::MultipoleSource(std::vector<BodySite> body_sites)
    : m_body_sites(std::move(body_sites)), m_has_body_data(true) {}

MultipoleSource::MultipoleSource(const occ::dma::Mult &multipole,
                                 const Vec3 &position)
    : m_has_body_data(false) {
    CartesianSite site;
    spherical_to_cartesian<4>(multipole, site.cart);
    site.position = position;
    site.rank = site.cart.effective_rank();
    m_cartesian.sites = {std::move(site)};
    m_center = position;
    m_cartesian_valid = true;
}

MultipoleSource MultipoleSource::from_lab_sites(
    const std::vector<std::pair<occ::dma::Mult, Vec3>> &site_data) {
    MultipoleSource source;
    source.m_has_body_data = false;
    source.m_cartesian = CartesianMolecule::from_lab_sites(site_data);
    source.m_cartesian_valid = true;
    if (!site_data.empty()) {
        Vec3 c = Vec3::Zero();
        for (const auto &[m, p] : site_data)
            c += p;
        source.m_center = c / static_cast<double>(site_data.size());
    }
    return source;
}

// ======================================================================
// Orientation
// ======================================================================

void MultipoleSource::set_orientation(const Mat3 &rotation, const Vec3 &center) {
    m_rotation = rotation;
    m_center = center;
    m_cartesian_valid = false;
}

const Mat3 &MultipoleSource::rotation() const { return m_rotation; }
const Vec3 &MultipoleSource::center() const { return m_center; }

int MultipoleSource::num_sites() const {
    if (m_has_body_data)
        return static_cast<int>(m_body_sites.size());
    ensure_cartesian();
    return static_cast<int>(m_cartesian.sites.size());
}

const std::vector<MultipoleSource::BodySite> &MultipoleSource::body_sites() const {
    return m_body_sites;
}

// ======================================================================
// Cartesian access (lazy)
// ======================================================================

const CartesianMolecule &MultipoleSource::cartesian() const {
    ensure_cartesian();
    return m_cartesian;
}

void MultipoleSource::ensure_cartesian() const {
    if (m_cartesian_valid)
        return;
    if (!m_has_body_data) {
        m_cartesian_valid = true;
        return;
    }

    std::vector<std::pair<occ::dma::Mult, Vec3>> pairs;
    pairs.reserve(m_body_sites.size());
    for (const auto &bs : m_body_sites)
        pairs.emplace_back(bs.multipole, bs.offset);

    if (!m_cartesian.body_data) {
        m_cartesian = CartesianMolecule::from_body_frame_with_rotation(
            pairs, m_rotation, m_center);
    } else {
        m_cartesian.update_orientation(m_rotation, m_center);
    }
    m_cartesian_valid = true;
}

// ======================================================================
// Single-point evaluation (specialized scalar probe kernels)
// ======================================================================

double MultipoleSource::compute_potential(const Vec3 &point) const {
    ensure_cartesian();
    double result = 0.0;
    for (const auto &site : m_cartesian.sites) {
        if (site.rank < 0)
            continue;
        Vec3 R = point - site.position;
        result += dispatch_probe_potential_scalar(
            site.cart, site.rank, R[0], R[1], R[2]);
    }
    return result;
}

Vec3 MultipoleSource::compute_field(const Vec3 &point) const {
    ensure_cartesian();
    Vec3 field = Vec3::Zero();
    for (const auto &site : m_cartesian.sites) {
        if (site.rank < 0)
            continue;
        Vec3 R = point - site.position;
        auto eg = dispatch_probe_field_scalar(
            site.cart, site.rank, R[0], R[1], R[2]);
        field[0] -= eg.grad[0];
        field[1] -= eg.grad[1];
        field[2] -= eg.grad[2];
    }
    return field;
}

// ======================================================================
// Batch evaluation (SIMD-batched probe kernels)
// ======================================================================

Vec MultipoleSource::compute_potential(Mat3NConstRef points) const {
    ensure_cartesian();
    const int N = static_cast<int>(points.cols());
    Vec result = Vec::Zero(N);

    for (const auto &site : m_cartesian.sites) {
        if (site.rank < 0)
            continue;
        dispatch_probe_potential_batch(site, points, result.data(), N);
    }
    return result;
}

Mat3N MultipoleSource::compute_field(Mat3NConstRef points) const {
    ensure_cartesian();
    const int N = static_cast<int>(points.cols());
    Mat3N result = Mat3N::Zero(3, N);

    for (const auto &site : m_cartesian.sites) {
        if (site.rank < 0)
            continue;
        dispatch_probe_field_batch(site, points, result, N);
    }
    return result;
}

// ======================================================================
// RigidBodyState bridge
// ======================================================================

MultipoleSource multipole_source_from_rigid_body(const RigidBodyState &rb) {
    std::vector<MultipoleSource::BodySite> sites;
    if (rb.is_multi_site()) {
        // Multi-site: convert RigidBodyState::BodySite to MultipoleSource::BodySite
        sites.reserve(rb.sites_body.size());
        for (const auto &s : rb.sites_body) {
            MultipoleSource::BodySite ms;
            ms.multipole = s.multipole;
            ms.offset = s.offset;
            sites.push_back(std::move(ms));
        }
    } else {
        // Single-site: use legacy multipole_body at origin
        MultipoleSource::BodySite site;
        site.multipole = rb.multipole_body;
        site.offset = Vec3::Zero();
        sites.push_back(std::move(site));
    }
    MultipoleSource source(std::move(sites));
    source.set_orientation(rb.rotation_matrix(), rb.position);
    return source;
}

void sync_orientation(MultipoleSource &source, const RigidBodyState &rb) {
    source.set_orientation(rb.rotation_matrix(), rb.position);
}

} // namespace occ::mults
