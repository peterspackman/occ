#pragma once
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::mults {

/**
 * @brief Projection matrix for removing rigid-body modes from optimization
 *
 * For a system of N rigid bodies with 6N degrees of freedom (3 position + 3 orientation each),
 * there are 6 spurious modes corresponding to global translation (3) and global rotation (3).
 * These cause the Hessian to have 6 zero eigenvalues, leading to divergence in optimization.
 *
 * Orient's approach (BUILDO routine):
 * - Construct a projection matrix P that removes these 6 modes
 * - Optimize in the (6N-6) dimensional space
 * - Gradient projection: grad_proj = P^T * grad_full
 * - Coordinate reconstruction: x_full = x_base + P * x_proj
 *
 * For 2 molecules (N=2):
 * - 12 DOF → 6 DOF (relative position + relative orientation)
 * - Simplest approach: Fix molecule 1 at origin with identity rotation
 * - Optimize only molecule 2's position and orientation relative to molecule 1
 */
class RigidBodyProjection {
public:
    /**
     * @brief Construct projection for N rigid bodies
     * @param n_molecules Number of rigid bodies
     */
    explicit RigidBodyProjection(size_t n_molecules);

    /**
     * @brief Get number of original degrees of freedom (6N)
     */
    size_t full_dof() const { return m_full_dof; }

    /**
     * @brief Get number of projected degrees of freedom (6N-6)
     */
    size_t projected_dof() const { return m_projected_dof; }

    /**
     * @brief Project full gradient to reduced space
     *
     * For optimization, we need: grad_proj = P^T * grad_full
     * This removes components corresponding to global translation/rotation.
     *
     * @param grad_full Gradient in full 6N space
     * @return Gradient in (6N-6) projected space
     */
    Vec project_gradient(const Vec& grad_full) const;

    /**
     * @brief Reconstruct full coordinates from projected coordinates
     *
     * Given projected coordinates x_proj in (6N-6) space, reconstruct
     * full coordinates: x_full = x_base + P * x_proj
     *
     * @param x_proj Coordinates in (6N-6) projected space
     * @param x_base Base configuration (typically initial geometry)
     * @return Coordinates in full 6N space
     */
    Vec reconstruct_full(const Vec& x_proj, const Vec& x_base) const;

    /**
     * @brief Project full coordinates to reduced space
     *
     * Given full coordinates, project to (6N-6) space relative to base:
     * x_proj = P^T * (x_full - x_base)
     *
     * @param x_full Coordinates in full 6N space
     * @param x_base Base configuration
     * @return Coordinates in (6N-6) projected space
     */
    Vec project_coordinates(const Vec& x_full, const Vec& x_base) const;

    /**
     * @brief Get projection matrix (for debugging/analysis)
     * @return (6N-6) × 6N projection matrix
     */
    const Mat& projection_matrix() const { return m_P; }

private:
    size_t m_n_molecules;     ///< Number of rigid bodies
    size_t m_full_dof;        ///< Full degrees of freedom (6N)
    size_t m_projected_dof;   ///< Projected degrees of freedom (6N-6)
    Mat m_P;                  ///< Projection matrix: (6N-6) × 6N

    /**
     * @brief Construct projection matrix
     *
     * For N=2 (simplest case):
     * - Fix molecule 1 at origin with identity rotation (6 constraints)
     * - Optimize molecule 2's 6 DOF
     * - P is a 6×12 matrix that extracts molecule 2's coordinates
     *
     * For N>2 (general case):
     * - Fix molecule 1 at origin with identity rotation (6 constraints)
     * - Optimize relative positions/orientations of molecules 2..N
     * - Use Gram-Schmidt to construct orthonormal basis in (6N-6) space
     */
    void build_projection_matrix();
};

/**
 * @brief Simple 2-molecule projection (most common case)
 *
 * For 2 molecules, the simplest projection:
 * - Molecule 1 fixed at origin with identity rotation
 * - Molecule 2 free to move (6 DOF)
 * - 12 DOF → 6 DOF
 *
 * This is equivalent to optimizing in relative coordinates.
 */
class TwoMoleculeProjection {
public:
    TwoMoleculeProjection() = default;

    /**
     * @brief Project gradient from 12-DOF to 6-DOF
     *
     * grad_proj = [grad_mol2_pos; grad_mol2_orient]
     *
     * @param grad_full 12-element gradient [mol1_pos, mol1_orient, mol2_pos, mol2_orient]
     * @return 6-element gradient [mol2_pos, mol2_orient]
     */
    Vec project_gradient(const Vec& grad_full) const;

    /**
     * @brief Reconstruct full 12-DOF from 6-DOF
     *
     * x_full = [x_base[0:6]; x_base[0:6] + x_proj]
     *
     * @param x_proj 6-element vector [mol2_pos, mol2_orient]
     * @param x_base 12-element base configuration
     * @return 12-element full coordinates
     */
    Vec reconstruct_full(const Vec& x_proj, const Vec& x_base) const;

    /**
     * @brief Project coordinates from 12-DOF to 6-DOF
     *
     * x_proj = x_full[6:12] - x_base[6:12]
     *
     * @param x_full 12-element full coordinates
     * @param x_base 12-element base configuration
     * @return 6-element projected coordinates
     */
    Vec project_coordinates(const Vec& x_full, const Vec& x_base) const;
};

} // namespace occ::mults
