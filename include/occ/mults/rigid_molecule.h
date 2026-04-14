#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>
#include <vector>

namespace occ::mults {

/// A rigid molecule with multipole sites and atom geometry.
///
/// This is the single source of truth for a molecule's body-frame data
/// and its placement in the crystal. Sites (multipole expansion points)
/// and atoms (for short-range interactions) are separate lists because
/// DMA can place multipoles on non-atom points (bond midpoints, lone pairs).
struct RigidMolecule {
    /// A multipole expansion site in the body frame.
    struct Site {
        Vec3 position;              ///< body-frame offset from COM (Angstrom)
        occ::dma::Mult multipole;
        int atom_index = -1;        ///< index into atoms[], -1 = non-atom site
        int short_range_type = 0;   ///< force-field type code (0 = untyped)
        Vec3 aniso_axis = Vec3::Zero(); ///< anisotropic repulsion z-axis (body frame)
    };

    /// An atom in the molecule (for geometry and short-range interactions).
    struct Atom {
        int atomic_number = 0;
        Vec3 position;              ///< body-frame offset from COM (Angstrom)
    };

    std::vector<Site> sites;        ///< multipole sites (may include non-atom sites)
    std::vector<Atom> atoms;        ///< real atoms (for Buckingham, neighbor detection)

    // --- Placement in crystal ---

    Vec3 com = Vec3::Zero();        ///< Cartesian COM position (Angstrom)
    Vec3 angle_axis = Vec3::Zero(); ///< proper rotation part (radians, angle-axis)
    int parity = 1;                 ///< +1 proper, -1 improper

    // --- Derived ---

    /// Full O(3) rotation matrix (det = parity).
    Mat3 rotation_matrix() const;

    /// Proper SO(3) rotation matrix (det = +1).
    Mat3 proper_rotation_matrix() const;

    /// Create from a full O(3) rotation matrix (extracts parity + angle-axis).
    static void set_from_rotation(RigidMolecule &mol, const Vec3 &pos,
                                  const Mat3 &R);
};

} // namespace occ::mults
