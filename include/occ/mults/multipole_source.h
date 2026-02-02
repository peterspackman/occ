#pragma once
#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cartesian_force.h>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <vector>

namespace occ::mults {

struct RigidBodyState; // forward declaration for bridge

class MultipoleSource {
public:
    struct BodySite {
        occ::dma::Mult multipole;
        Vec3 offset = Vec3::Zero(); // position relative to center (body frame)
    };

    // --- Construction ---

    MultipoleSource() = default;

    /// From body-frame sites. Orientation defaults to identity at origin.
    explicit MultipoleSource(std::vector<BodySite> body_sites);

    /// Single-site convenience (lab-frame, no rotation support).
    MultipoleSource(const occ::dma::Mult &multipole, const Vec3 &position);

    /// From pre-built lab-frame data (no body-frame retained, no rotation support).
    static MultipoleSource from_lab_sites(
        const std::vector<std::pair<occ::dma::Mult, Vec3>> &site_data);

    // --- Orientation ---

    void set_orientation(const Mat3 &rotation, const Vec3 &center);
    const Mat3 &rotation() const;
    const Vec3 &center() const;
    int num_sites() const;
    const std::vector<BodySite> &body_sites() const;

    // --- Cartesian access (lazy) ---

    const CartesianMolecule &cartesian() const;

    // --- Batch evaluation ---

    Vec compute_potential(Mat3NConstRef points) const;
    double compute_potential(const Vec3 &point) const;

    Mat3N compute_field(Mat3NConstRef points) const;
    Vec3 compute_field(const Vec3 &point) const;

private:
    std::vector<BodySite> m_body_sites;
    Mat3 m_rotation = Mat3::Identity();
    Vec3 m_center = Vec3::Zero();
    bool m_has_body_data = false;

    mutable CartesianMolecule m_cartesian;
    mutable bool m_cartesian_valid = false;

    void ensure_cartesian() const;
};

// --- RigidBodyState bridge (free functions) ---

MultipoleSource multipole_source_from_rigid_body(const RigidBodyState &rb);
void sync_orientation(MultipoleSource &source, const RigidBodyState &rb);

} // namespace occ::mults
