#pragma once
#include <occ/mults/cartesian_multipole.h>
#include <occ/mults/cartesian_kernels.h>
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <vector>
#include <array>
#include <utility>
#include <optional>

namespace occ::mults {

/// A single site with precomputed Cartesian multipole and cached rank.
struct CartesianSite {
    CartesianMultipole<4> cart;
    Vec3 position;
    int rank = -1; // effective rank, -1 if empty
};

/// Body-frame data for rigid body rotation support.
///
/// Precomputes and caches rotation derivatives so that torque evaluation
/// only needs to contract cached derivatives with interaction fields,
/// without re-enumerating the rotation kernel at force-evaluation time.
struct BodyFrameData {
    std::vector<CartesianMultipole<4>> body_multipoles; // body-frame Cartesian multipoles
    std::vector<Vec3> body_offsets;   // site positions relative to center (body frame)
    Vec3 center;                       // center of mass (lab frame)
    Mat3 rotation;                     // current rotation matrix (body → lab)

    /// Precomputed rotation derivatives: d_multipoles[k][i] is
    /// d(lab_multipole_i)/dp_k where p_k is the k-th angle-axis component
    /// (k=0,1,2 for infinitesimal rotations about x,y,z).
    /// Computed once per orientation change, reused across all force evaluations.
    std::array<std::vector<CartesianMultipole<4>>, 3> d_multipoles;
};

/// Collection of precomputed Cartesian sites for a molecule.
///
/// Caches the spherical-to-Cartesian conversion and effective_rank
/// so they are computed once per molecule rather than once per pair.
struct CartesianMolecule {
    std::vector<CartesianSite> sites;

    /// Body-frame data for rotation support (optional).
    std::optional<BodyFrameData> body_data;

    /// Build from lab-frame site data (multipole + position pairs).
    static CartesianMolecule from_lab_sites(
        const std::vector<std::pair<occ::dma::Mult, Vec3>> &site_data);

    /// Build from body-frame data with rotation and center-of-mass offset.
    /// Does NOT retain body-frame data (positions only).
    ///
    /// @param body_sites  Multipoles and body-frame positions
    /// @param rotation    3x3 rotation matrix (body → lab)
    /// @param center      Lab-frame center of mass
    static CartesianMolecule from_body_frame(
        const std::vector<std::pair<occ::dma::Mult, Vec3>> &body_sites,
        const Mat3 &rotation, const Vec3 &center);

    /// Build from body-frame data, retaining body-frame multipoles for
    /// rotation support. Both positions AND multipoles are rotated to lab.
    ///
    /// @param body_sites  Multipoles (body frame) and body-frame positions
    /// @param rotation    3x3 rotation matrix (body → lab)
    /// @param center      Lab-frame center of mass
    static CartesianMolecule from_body_frame_with_rotation(
        const std::vector<std::pair<occ::dma::Mult, Vec3>> &body_sites,
        const Mat3 &rotation, const Vec3 &center);

    /// Update lab-frame positions from body-frame offsets + rotation.
    ///
    /// This avoids re-converting spherical → Cartesian multipoles;
    /// only positions change.  The effective_rank is rotation-invariant
    /// and remains cached.
    void update_positions(const Mat3 &rotation, const Vec3 &center,
                          const std::vector<Vec3> &body_offsets);

    /// Update both positions AND multipoles for a new orientation.
    /// Requires body_data to be set (via from_body_frame_with_rotation).
    void update_orientation(const Mat3 &new_rotation, const Vec3 &new_center);
};

/// Compute total electrostatic interaction energy between two molecules.
///
/// Iterates over all site pairs (i in molA, j in molB), dispatching
/// to the appropriate tensor order for each pair.  Multipole conversions
/// and rank detection are already cached in the CartesianMolecule objects.
double compute_molecule_interaction(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB);

/// Compute interaction energy for a single pair of precomputed sites.
double compute_site_pair_energy(
    const CartesianSite &siteA,
    const CartesianSite &siteB);

/// Compute total interaction energy using SIMD-batched T-tensor computation.
///
/// Groups pairs by interaction order and processes groups in SIMD batches
/// of simd_batch_size pairs simultaneously.  Falls back to scalar for
/// remainder pairs in each group.
double compute_molecule_interaction_simd(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB);

} // namespace occ::mults
