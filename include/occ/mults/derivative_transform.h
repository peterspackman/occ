#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/mults/coordinate_system.h>
#include <array>

namespace occ::mults {

/**
 * Derivative transformation matrices for multipole interactions
 *
 * This class implements Orient's derivative transformation system that converts
 * S-function derivatives into molecular Cartesian coordinate derivatives.
 *
 * Mathematical background:
 * =======================
 *
 * The S-functions depend on 16 intermediate variables (Orient's "cosine" array):
 *   q(1:3)   = e1r (unit inter-site vector in site A's frame)
 *   q(4:6)   = e2r (unit inter-site vector in site B's frame, with sign flip)
 *   q(7:15)  = xx (orientation matrix elements, 3x3 flattened)
 *   q(16)    = r (inter-site distance)
 *
 * These intermediate variables depend on 12 external coordinates:
 *   For molecule A:
 *     coords 1-3:  Position (xa, ya, za)
 *     coords 4-6:  Torque/rotation (tau_x, tau_y, tau_z)
 *   For molecule B:
 *     coords 7-9:  Position (xb, yb, zb)
 *     coords 10-12: Torque/rotation (tau_x, tau_y, tau_z)
 *
 * Transformation chain for first derivatives:
 * ===========================================
 *
 * ∂E/∂x_α = Σᵢ (∂E/∂Sⁱ) · (∂Sⁱ/∂x_α)
 *
 * where the chain rule gives:
 *
 * ∂Sⁱ/∂x_α = Σⱼ (∂Sⁱ/∂qⱼ) · (∂qⱼ/∂x_α)
 *
 * Orient computes:
 * - S1(iq,ix): First derivatives of S-function ix w.r.t. intermediate variable iq
 *              Dimension: [15 intermediate vars] x [nmax S-functions]
 *              (Only 15 because q(16)=r derivatives are handled separately)
 *
 * - D1(iq,ip): First derivatives of intermediate variable iq w.r.t. external coord ip
 *              Dimension: [16 intermediate vars] x [12 external coords]
 *
 * - D1S(ip,ix): Result of chain rule: ∂Sⁱˣ/∂x_ip = Σⱼ S1(iq,ix) · D1(iq,ip)
 *               Dimension: [12 external coords] x [nmax S-functions]
 *
 * Transformation chain for second derivatives:
 * ============================================
 *
 * ∂²E/∂x_α∂x_β = Σᵢⱼ (∂²E/∂Sⁱ∂Sʲ) · (∂Sⁱ/∂x_α) · (∂Sʲ/∂x_β)
 *              + Σᵢ (∂E/∂Sⁱ) · (∂²Sⁱ/∂x_α∂x_β)
 *
 * where:
 *
 * ∂²Sⁱ/∂x_α∂x_β = Σⱼₖ (∂²Sⁱ/∂qⱼ∂qₖ) · (∂qⱼ/∂x_α) · (∂qₖ/∂x_β)
 *               + Σⱼ (∂Sⁱ/∂qⱼ) · (∂²qⱼ/∂x_α∂x_β)
 *
 * Orient computes:
 * - S2(kq,ix): Second derivatives of S-function ix w.r.t. intermediate variables
 *              Stored as packed symmetric matrix (kq is packed index)
 *              Dimension: [120 packed pairs] x [nmax S-functions]
 *
 * - D2(iq,ip,jp): Second derivatives of intermediate variable iq
 *                 w.r.t. external coords ip and jp
 *                 Dimension: [16] x [12] x [12]
 *
 * - D2S(ip,jp,ix): Result: ∂²Sⁱˣ/∂x_ip∂x_jp
 *                  Computed as:
 *                  D2S = Σⱼₖ S2(packed(j,k),ix) · D1(iq,ip) · D1(jq,jp)
 *                      + Σⱼ S1(iq,ix) · D2(iq,ip,jp)
 *                  Dimension: [12] x [12] x [nmax S-functions]
 *
 * Coordinate system notes:
 * =======================
 *
 * The intermediate variables (q) are computed from the raw molecular coordinates:
 *
 * - Site positions: ra (site A), rb (site B) in global Cartesian frame
 * - Site vectors from molecular COM: a = ra - cm(A), b = rb - cm(B)
 * - Inter-site vector: rab = rb - ra
 * - Distance: r = |rab|
 * - Unit vector: er = rab / r
 *
 * The derivatives account for:
 * 1. Translation: Moving the molecular center of mass
 * 2. Rotation: Rotating the molecular frame (changes site positions and orientations)
 *
 * For point multipoles (no molecular structure):
 * - Site vector a = 0, b = 0
 * - Only translational derivatives are non-zero
 * - D1(1:6, 4:6) = 0, D1(1:6, 10:12) = 0 (no torque coupling)
 */
class DerivativeTransform {
public:
    // Dimensions matching Orient's implementation
    static constexpr int NUM_INTERMEDIATE_VARS = 16;  // cosine array size
    static constexpr int NUM_EXTERNAL_COORDS = 12;    // 6 per molecule (3 pos + 3 rot)
    static constexpr int NUM_FIRST_DERIV_VARS = 15;   // S1 only uses first 15 (no r deriv)
    static constexpr int NUM_SECOND_DERIV_PACKED = 120; // Packed symmetric matrix

    /**
     * Compute D1 matrix with angle-axis derivatives
     *
     * Following Orient's pairinfo_aa (interact.f90, lines 225-448).
     *
     * This version computes derivatives w.r.t. angle-axis parameters using
     * the precomputed M1 matrices (∂M/∂p_k) from rotation_matrix_derivatives().
     *
     * Input coordinate convention:
     *   coords[0:2]   - Position of molecule A (xa, ya, za)
     *   coords[3:5]   - Angle-axis of molecule A (px, py, pz)
     *   coords[6:8]   - Position of molecule B (xb, yb, zb)
     *   coords[9:11]  - Angle-axis of molecule B (px, py, pz)
     *
     * @param coords Coordinate system with site positions
     * @param M_A Rotation matrix for molecule A (current orientation)
     * @param M_B Rotation matrix for molecule B (current orientation)
     * @param M1_A Rotation matrix derivatives for molecule A [∂M_A/∂p_A]
     * @param M1_B Rotation matrix derivatives for molecule B [∂M_B/∂p_B]
     * @param a Site vector for site A relative to molecular COM (zero for point multipoles)
     * @param b Site vector for site B relative to molecular COM (zero for point multipoles)
     * @return D1 matrix [16 x 12]
     */
    static Mat compute_D1_angle_axis(
        const CoordinateSystem& coords,
        const Mat3& M_A,
        const Mat3& M_B,
        const std::array<Mat3, 3>& M1_A,
        const std::array<Mat3, 3>& M1_B,
        const Vec3& a = Vec3::Zero(),
        const Vec3& b = Vec3::Zero());

    /**
     * Compute D1 matrix: First derivatives of intermediate variables w.r.t. external coords
     *
     * Following Orient's mlinfo.f90 exactly.
     *
     * Input coordinate convention:
     *   coords[0:2]   - Position of molecule A (xa, ya, za)
     *   coords[3:5]   - Torque on molecule A (tau_x, tau_y, tau_z)
     *   coords[6:8]   - Position of molecule B (xb, yb, zb)
     *   coords[9:11]  - Torque on molecule B (tau_x, tau_y, tau_z)
     *
     * The torque derivatives use cross products:
     * - ∂(e1r)/∂tau_A involves (rab + a) × local_axis
     * - ∂(e2r)/∂tau_B involves (rab - b) × local_axis
     *
     * For point multipoles (a=0, b=0), torque derivatives vanish.
     *
     * @param coords Coordinate system with site positions
     * @param a Site vector for site A relative to molecular COM (zero for point multipoles)
     * @param b Site vector for site B relative to molecular COM (zero for point multipoles)
     * @return D1 matrix [16 x 12]
     */
    static Mat compute_D1(const CoordinateSystem& coords,
                          const Vec3& a = Vec3::Zero(),
                          const Vec3& b = Vec3::Zero());

    /**
     * Compute D2 matrix: Second derivatives of intermediate variables
     *
     * Following Orient's mlinfo.f90 exactly.
     *
     * This is a 3D array D2[iq][ip][jp] where:
     * - iq: intermediate variable index (0-15)
     * - ip, jp: external coordinate indices (0-11)
     *
     * The matrix is symmetric in (ip,jp) for each iq.
     *
     * @param coords Coordinate system
     * @param a Site vector for site A relative to molecular COM
     * @param b Site vector for site B relative to molecular COM
     * @return D2 tensor [16 x 12 x 12]
     */
    static std::array<Mat, NUM_INTERMEDIATE_VARS> compute_D2(
        const CoordinateSystem& coords,
        const Vec3& a = Vec3::Zero(),
        const Vec3& b = Vec3::Zero());

    /**
     * Transform S-function first derivatives to external coordinates
     *
     * Computes: D1S[ip][ix] = Σⱼ S1[iq][ix] · D1[iq][ip]
     *
     * This is a matrix-matrix multiplication where:
     * - S1 is [15 x nmax] (first derivatives of S-functions w.r.t. intermediate vars)
     * - D1 is [16 x 12] (first derivatives of intermediate vars w.r.t. external coords)
     * - D1S is [12 x nmax] (first derivatives of S-functions w.r.t. external coords)
     *
     * Note: Only the first 15 rows of D1 are used (q(16)=r handled separately in S1)
     *
     * @param S1 First derivatives from S-function computation [15 x nmax]
     * @param D1 Coordinate transformation matrix [16 x 12]
     * @return D1S transformed derivatives [12 x nmax]
     */
    static Mat compute_D1S(const Mat& S1, const Mat& D1);

    /**
     * Transform S-function second derivatives to external coordinates
     *
     * Computes: D2S[ip][jp][ix] = Σⱼₖ S2[packed(j,k)][ix] · D1[iq][ip] · D1[jq][jp]
     *                            + Σⱼ S1[iq][ix] · D2[iq][ip][jp]
     *
     * This combines:
     * 1. The product of S2 (second derivs w.r.t. intermediate vars) with D1²
     * 2. The product of S1 (first derivs) with D2 (second derivs of coordinates)
     *
     * @param S1 First derivatives [15 x nmax]
     * @param S2 Second derivatives (packed symmetric) [120 x nmax]
     * @param D1 First coordinate transformation [16 x 12]
     * @param D2 Second coordinate transformation [16 x 12 x 12]
     * @return D2S transformed second derivatives [12 x 12 x nmax]
     */
    static std::vector<Mat> compute_D2S(const Mat& S1, const Mat& S2,
                                        const Mat& D1,
                                        const std::array<Mat, NUM_INTERMEDIATE_VARS>& D2);

    /**
     * Utility: Convert packed symmetric matrix index to (i,j) pair
     *
     * Orient uses packed storage for symmetric matrices:
     * kq = 0, 1, 2, ..., 119 maps to (i,j) pairs:
     * (1,1), (2,1), (2,2), (3,1), (3,2), (3,3), ...
     *
     * This is lower-triangular storage with 1-based indexing in Fortran,
     * converted to 0-based for C++.
     *
     * @param kq Packed index (0-119)
     * @return (i,j) pair with 0-based indexing
     */
    static std::pair<int, int> unpack_symmetric_index(int kq);

    /**
     * Utility: Convert (i,j) pair to packed symmetric index
     *
     * Inverse of unpack_symmetric_index.
     *
     * @param i First index (0-14)
     * @param j Second index (0-14, j <= i)
     * @return Packed index kq
     */
    static int pack_symmetric_index(int i, int j);

private:
    /**
     * Helper: Compute cross product for a 3-vector
     */
    static Vec3 cross(const Vec3& a, const Vec3& b);
};

} // namespace occ::mults
