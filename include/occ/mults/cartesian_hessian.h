#pragma once
#include <occ/mults/cartesian_multipole.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/core/linear_algebra.h>

namespace occ::mults {

/// Result of Hessian computation for a pair of rigid bodies.
///
/// The Hessian is organized in blocks:
/// - Position-Position: 3x3 for each (molecule_i, molecule_j)
/// - Position-Rotation: 3x3 for each (position_i, rotation_j)
/// - Rotation-Position: 3x3 for each (rotation_i, position_j)
/// - Rotation-Rotation: 3x3 for each (rotation_i, rotation_j)
struct PairHessianResult {
    double energy = 0.0;

    // Forces (negative gradient w.r.t. position)
    Vec3 force_A = Vec3::Zero();
    Vec3 force_B = Vec3::Zero();

    // Gradient w.r.t. angle-axis (for torque)
    Vec3 grad_angle_axis_A = Vec3::Zero();
    Vec3 grad_angle_axis_B = Vec3::Zero();

    // Hessian blocks (w.r.t. position and angle-axis parameters)
    // ∂²E/∂pos_A_k ∂pos_A_l
    Mat3 H_posA_posA = Mat3::Zero();
    // ∂²E/∂pos_A_k ∂pos_B_l
    Mat3 H_posA_posB = Mat3::Zero();
    // ∂²E/∂pos_B_k ∂pos_B_l
    Mat3 H_posB_posB = Mat3::Zero();

    // ∂²E/∂pos_A_k ∂θ_A_l (position-rotation cross terms)
    Mat3 H_posA_rotA = Mat3::Zero();
    // ∂²E/∂pos_A_k ∂θ_B_l
    Mat3 H_posA_rotB = Mat3::Zero();
    // ∂²E/∂pos_B_k ∂θ_A_l
    Mat3 H_posB_rotA = Mat3::Zero();
    // ∂²E/∂pos_B_k ∂θ_B_l
    Mat3 H_posB_rotB = Mat3::Zero();

    // ∂²E/∂θ_A_k ∂θ_A_l (rotation-rotation)
    Mat3 H_rotA_rotA = Mat3::Zero();
    // ∂²E/∂θ_A_k ∂θ_B_l
    Mat3 H_rotA_rotB = Mat3::Zero();
    // ∂²E/∂θ_B_k ∂θ_B_l
    Mat3 H_rotB_rotB = Mat3::Zero();

    /// Pack the Hessian into a 12x12 matrix (6 DOF per molecule)
    /// Order: [pos_A, rot_A, pos_B, rot_B]
    Mat pack_full_hessian() const;

    /// Pack only the position-position Hessian (6x6)
    /// Order: [pos_A, pos_B]
    Mat6 pack_position_hessian() const;
};

/// Compute analytical Hessian for charge-charge interactions only.
///
/// This is the simplest case where both multipoles are rank 0 (charges).
/// The Hessian is entirely positional (no rotation dependence for point charges).
///
/// Energy: E = q_A * q_B / R
/// Gradient: ∂E/∂R_k = q_A * q_B * T^(1)_k
/// Hessian: ∂²E/∂R_k∂R_l = q_A * q_B * T^(2)_kl
///
/// @param posA Position of charge A
/// @param qA Charge at A
/// @param posB Position of charge B
/// @param qB Charge at B
/// @return PairHessianResult with energy, forces, and position Hessian
PairHessianResult compute_charge_charge_hessian(
    const Vec3 &posA, double qA,
    const Vec3 &posB, double qB);

/// Compute analytical Hessian for charge-dipole interactions.
///
/// Site A has charge q_A, site B has dipole μ_B (in lab frame).
///
/// Energy: E = q_A * T^(1)_j * μ_B^j
///
/// Position Hessian: ∂²E/∂R_k∂R_l = q_A * T^(3)_jkl * μ_B^j
///
/// Position-Rotation Hessian (for B's rotation):
///   ∂²E/∂R_k∂θ_B^m = q_A * T^(2)_jk * ∂μ_B^j/∂θ_B^m
///
/// Rotation-Rotation Hessian (for B's rotation):
///   ∂²E/∂θ_B^m∂θ_B^n = q_A * T^(1)_j * ∂²μ_B^j/∂θ_B^m∂θ_B^n
///
/// @param posA Position of charge A
/// @param qA Charge at A
/// @param posB Position of dipole B (center)
/// @param dipole_B Lab-frame dipole at B (already rotated)
/// @param body_dipole_B Body-frame dipole at B (for rotation derivatives)
/// @param M Rotation matrix (body to lab) for B
/// @param dM Array of 3 rotation matrix derivatives ∂M/∂θ_k
/// @param d2M Array of 9 second derivatives ∂²M/∂θ_k∂θ_l (optional, can be nullptr)
/// @return PairHessianResult with all Hessian blocks
PairHessianResult compute_charge_dipole_hessian(
    const Vec3 &posA, double qA,
    const Vec3 &posB, const Vec3 &dipole_B,
    const Vec3 &body_dipole_B,
    const Mat3 &M,
    const std::array<Mat3, 3> &dM,
    const std::array<Mat3, 9> *d2M = nullptr);

/// Compute analytical Hessian for molecule pair (charge-charge + charge-dipole only).
///
/// This is a truncated Hessian that includes:
/// - All charge-charge interactions (rank 0 - rank 0)
/// - All charge-dipole interactions (rank 0 - rank 1 and rank 1 - rank 0)
///
/// Higher multipole contributions are excluded from the Hessian but
/// still contribute to energy and gradient.
///
/// @param molA Molecule A with body frame data
/// @param molB Molecule B with body frame data
/// @return PairHessianResult with combined Hessian
PairHessianResult compute_molecule_hessian_truncated(
    const CartesianMolecule &molA,
    const CartesianMolecule &molB);

} // namespace occ::mults
