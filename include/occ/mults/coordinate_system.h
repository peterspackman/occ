#pragma once
#include <occ/core/linear_algebra.h>
#include <cmath>

namespace occ::mults {

/**
 * Lightweight POD struct for multipole coordinate system transformations
 * 
 * Implements Orient's exact coordinate convention:
 * - Site A coordinates: ra (raw input coordinates)
 * - Site B coordinates: rb (raw input coordinates)  
 * - Inter-site vector: rab = rb - ra
 * - Distance: r = |rab|
 * - Unit vector: er = rab / r
 * - Orient variables: rax,ray,raz = +er (unit vector from A to B, in body frame)
 *                    rbx,rby,rbz = -er (unit vector from B to A, in body frame)
 */
struct CoordinateSystem {
    // Raw input coordinates
    Vec3 ra, rb;
    
    // Computed quantities
    Vec3 rab;    // Inter-site vector: rb - ra
    double r;    // Distance: |rab|
    Vec3 er;     // Unit vector: rab / r (lab frame)

    // Body-frame unit vectors (for Orient's approach with rotated multipoles)
    // When use_body_frame = true, these override the lab-frame accessors
    bool use_body_frame = false;
    Vec3 e1r_body;  // Unit vector in molecule A's body frame: M_A^T * er
    Vec3 e2r_body;  // Unit vector in molecule B's body frame with sign: -M_B^T * er

    // Orientation matrix (3x3)
    // c(i,j) = dot product of axis i at site A with axis j at site B
    // Following Orient's convention: xx = matmul(transpose(M_A), M_B)
    // For point multipoles with no molecular rotation: identity matrix
    double cxx, cxy, cxz;
    double cyx, cyy, cyz;
    double czx, czy, czz;
    
    // Orient-compatible accessors
    // In lab-frame mode: use er and -er
    // In body-frame mode: use transformed vectors e1r_body and e2r_body
    Vec3 raxyz() const { return use_body_frame ? e1r_body : er; }
    Vec3 rbxyz() const { return use_body_frame ? e2r_body : -er; }

    // Individual component accessors matching Orient exactly
    double rax() const { return use_body_frame ? e1r_body[0] : er[0]; }
    double ray() const { return use_body_frame ? e1r_body[1] : er[1]; }
    double raz() const { return use_body_frame ? e1r_body[2] : er[2]; }
    double rbx() const { return use_body_frame ? e2r_body[0] : -er[0]; }
    double rby() const { return use_body_frame ? e2r_body[1] : -er[1]; }
    double rbz() const { return use_body_frame ? e2r_body[2] : -er[2]; }
    
    // Utility accessors
    double dx() const { return rab[0]; }   // rb.x - ra.x
    double dy() const { return rab[1]; }   // rb.y - ra.y  
    double dz() const { return rab[2]; }   // rb.z - ra.z
    
    // Raw coordinate accessors (for reference)
    double raw_rax() const { return ra[0]; }
    double raw_ray() const { return ra[1]; }
    double raw_raz() const { return ra[2]; }
    double raw_rbx() const { return rb[0]; }
    double raw_rby() const { return rb[1]; }
    double raw_rbz() const { return rb[2]; }
    
    /**
     * Initialize coordinate system from two points
     * Following Orient's mlinfo.f90 exactly:
     * - rab = rb - ra (site-to-site vector)
     * - r = |rab| (distance)
     * - er = rab / r (normalized inter-site vector)
     * - Orientation matrix initialized to identity (for point multipoles)
     */
    static CoordinateSystem from_points(const Vec3& ra_in, const Vec3& rb_in) {
        CoordinateSystem cs;
        cs.ra = ra_in;
        cs.rb = rb_in;
        cs.rab = rb_in - ra_in;
        cs.r = cs.rab.norm();

        if (cs.r > 1e-12) {
            cs.er = cs.rab / cs.r;
        } else {
            cs.er = Vec3::Zero(); // Handle degenerate case
        }

        // For point multipoles with no molecular rotation, orientation matrix is identity
        cs.cxx = 1.0; cs.cxy = 0.0; cs.cxz = 0.0;
        cs.cyx = 0.0; cs.cyy = 1.0; cs.cyz = 0.0;
        cs.czx = 0.0; cs.czy = 0.0; cs.czz = 1.0;

        return cs;
    }

    /**
     * Initialize coordinate system with body-frame unit vectors (Orient's approach)
     *
     * This implements Orient's body-frame approach where:
     * - Lab frame: er_lab = (rb - ra) / |rb - ra|
     * - Body frame for site A: e1r = M_A^T * er_lab
     * - Body frame for site B: e2r = -M_B^T * er_lab
     * - Orientation matrix: xx = M_A^T * M_B
     *
     * @param ra_in Position of site A (lab frame)
     * @param rb_in Position of site B (lab frame)
     * @param M_A Rotation matrix for molecule A (body to lab)
     * @param M_B Rotation matrix for molecule B (body to lab)
     */
    static CoordinateSystem from_body_frame(const Vec3& ra_in, const Vec3& rb_in,
                                           const Mat3& M_A, const Mat3& M_B) {
        CoordinateSystem cs;
        cs.ra = ra_in;
        cs.rb = rb_in;
        cs.rab = rb_in - ra_in;
        cs.r = cs.rab.norm();

        // Compute lab-frame unit vector
        if (cs.r > 1e-12) {
            cs.er = cs.rab / cs.r;
        } else {
            cs.er = Vec3::Zero();
        }

        // Transform to body frames (Orient's e1r and e2r)
        cs.use_body_frame = true;
        cs.e1r_body = M_A.transpose() * cs.er;   // e1r = M_A^T * er
        cs.e2r_body = -(M_B.transpose() * cs.er); // e2r = -M_B^T * er

        // Compute orientation matrix: xx = M_A^T * M_B
        Mat3 xx = M_A.transpose() * M_B;
        cs.cxx = xx(0,0); cs.cxy = xx(0,1); cs.cxz = xx(0,2);
        cs.cyx = xx(1,0); cs.cyy = xx(1,1); cs.cyz = xx(1,2);
        cs.czx = xx(2,0); cs.czy = xx(2,1); cs.czz = xx(2,2);

        return cs;
    }
    
    /**
     * Validate coordinate system consistency
     * Returns true if all computed quantities are consistent
     */
    bool is_valid(double tolerance = 1e-12) const {
        // Check distance calculation
        if (std::abs(r - rab.norm()) > tolerance) return false;
        
        // Check unit vector normalization  
        if (std::abs(er.norm() - 1.0) > tolerance && r > tolerance) return false;
        
        // Check unit vector calculation
        if (r > tolerance) {
            Vec3 expected_er = rab / r;
            if ((er - expected_er).norm() > tolerance) return false;
        }
        
        // Check Orient coordinate relationships
        if ((raxyz() - er).norm() > tolerance) return false;  // raxyz should equal +er
        if ((rbxyz() + er).norm() > tolerance) return false;  // rbxyz should equal -er
        
        return true;
    }
    
    /**
     * Check if coordinate system matches Orient's debug output
     * Useful for validating against known Orient calculations
     */
    bool matches_orient_debug(double orient_rax, double orient_ray, double orient_raz,
                             double orient_rbx, double orient_rby, double orient_rbz,
                             double orient_r, double tolerance = 1e-6) const {
        return std::abs(rax() - orient_rax) < tolerance &&
               std::abs(ray() - orient_ray) < tolerance &&
               std::abs(raz() - orient_raz) < tolerance &&
               std::abs(rbx() - orient_rbx) < tolerance &&
               std::abs(rby() - orient_rby) < tolerance &&
               std::abs(rbz() - orient_rbz) < tolerance &&
               std::abs(r - orient_r) < tolerance;
    }
};

} // namespace occ::mults