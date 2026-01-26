#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>

namespace occ::mults {

/**
 * @brief Rotate multipole moments using Wigner D-matrices
 * 
 * This module provides functionality to rotate multipole moments stored in
 * occ::dma::Mult objects under 3D rotations. The implementation is based on
 * the proven algorithms from Orient's rotations.f90, specifically the wigner()
 * subroutine.
 * 
 * The rotation uses real spherical harmonic conventions matching Stone's
 * "Theory of Intermolecular Forces" with Racah normalization.
 */

/**
 * @brief Rotation matrix type
 */
using RotationMatrix = Mat3;

/**
 * @brief Wigner D-matrix for multipole rotation
 * 
 * Computes the transformation matrix D required to rotate real spherical 
 * harmonic multipole components (Q00, Q10, Q11c, Q11s, ..., QLLc, QLLs)
 * under a 3D rotation described by rotation matrix R.
 * 
 * @param R 3x3 rotation matrix (direction cosines)
 * @param lmax Maximum multipole rank to include
 * @return Square matrix D of size (lmax+1)^2 for transforming multipole components
 */
Mat wigner_d_matrix(const RotationMatrix& R, int lmax);

/**
 * @brief Rotate a multipole moment object
 * 
 * Applies a 3D rotation to all multipole components in a Mult object
 * up to its maximum rank. The rotation preserves the multipole expansion
 * while transforming to the new coordinate system.
 * 
 * @param mult Input multipole object (modified in place)
 * @param R 3x3 rotation matrix
 * @return Reference to the rotated multipole object
 */
occ::dma::Mult& rotate_multipole(occ::dma::Mult& mult, const RotationMatrix& R);

/**
 * @brief Create a rotated copy of a multipole object
 * 
 * @param mult Input multipole object (unchanged)
 * @param R 3x3 rotation matrix  
 * @return New multipole object with rotated components
 */
occ::dma::Mult rotated_multipole(const occ::dma::Mult& mult, const RotationMatrix& R);

/**
 * @brief Utility functions for rotation matrix creation
 */
namespace rotation_utils {

/**
 * @brief Create rotation matrix from Euler angles (ZYZ convention)
 * @param alpha First rotation around Z axis (radians)
 * @param beta Rotation around Y axis (radians) 
 * @param gamma Final rotation around Z axis (radians)
 * @return 3x3 rotation matrix
 */
RotationMatrix euler_to_rotation(double alpha, double beta, double gamma);

/**
 * @brief Create rotation matrix from axis-angle representation
 * @param axis Rotation axis (will be normalized)
 * @param angle Rotation angle in radians
 * @return 3x3 rotation matrix
 */
RotationMatrix axis_angle_to_rotation(const Vec3& axis, double angle);

/**
 * @brief Create rotation matrix from quaternion
 * @param q Quaternion as [w, x, y, z] (will be normalized)
 * @return 3x3 rotation matrix
 */
RotationMatrix quaternion_to_rotation(const Vec4& q);

/**
 * @brief Validate that a matrix is a proper rotation matrix
 * @param R Matrix to check
 * @return true if R is a valid rotation matrix
 */
bool is_rotation_matrix(const RotationMatrix& R, double tolerance = 1e-10);

} // namespace rotation_utils

} // namespace occ::mults