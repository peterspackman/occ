#include <occ/mults/rotation.h>
#include <occ/sht/wigner3j.h>
#include <occ/sht/clebsch.h>
#include <cmath>
#include <stdexcept>
#include <complex>

namespace occ::mults {

namespace rotation_utils {

bool is_rotation_matrix(const RotationMatrix& R, double tolerance) {
    // Check that R is orthogonal (R^T * R = I)
    Mat3 RtR = R.transpose() * R;
    Mat3 identity = Mat3::Identity();
    if (!RtR.isApprox(identity, tolerance)) return false;
    
    // Check that determinant is +1 (proper rotation)
    double det = R.determinant();
    return std::abs(det - 1.0) < tolerance;
}

RotationMatrix euler_to_rotation(double alpha, double beta, double gamma) {
    // ZYZ convention: R = Rz(gamma) * Ry(beta) * Rz(alpha)
    double ca = std::cos(alpha);
    double sa = std::sin(alpha);
    double cb = std::cos(beta);
    double sb = std::sin(beta);
    double cg = std::cos(gamma);
    double sg = std::sin(gamma);
    
    RotationMatrix R;
    R(0, 0) = ca * cb * cg - sa * sg;
    R(0, 1) = -ca * cb * sg - sa * cg;
    R(0, 2) = ca * sb;
    R(1, 0) = sa * cb * cg + ca * sg;
    R(1, 1) = -sa * cb * sg + ca * cg;
    R(1, 2) = sa * sb;
    R(2, 0) = -sb * cg;
    R(2, 1) = sb * sg;
    R(2, 2) = cb;
    
    return R;
}

RotationMatrix axis_angle_to_rotation(const Vec3& axis, double angle) {
    Vec3 n = axis.normalized();
    double c = std::cos(angle);
    double s = std::sin(angle);
    double one_minus_c = 1.0 - c;
    
    // Rodrigues' rotation formula
    RotationMatrix R;
    R(0, 0) = c + n(0) * n(0) * one_minus_c;
    R(0, 1) = n(0) * n(1) * one_minus_c - n(2) * s;
    R(0, 2) = n(0) * n(2) * one_minus_c + n(1) * s;
    R(1, 0) = n(1) * n(0) * one_minus_c + n(2) * s;
    R(1, 1) = c + n(1) * n(1) * one_minus_c;
    R(1, 2) = n(1) * n(2) * one_minus_c - n(0) * s;
    R(2, 0) = n(2) * n(0) * one_minus_c - n(1) * s;
    R(2, 1) = n(2) * n(1) * one_minus_c + n(0) * s;
    R(2, 2) = c + n(2) * n(2) * one_minus_c;
    
    return R;
}

RotationMatrix quaternion_to_rotation(const Vec4& q_in) {
    Vec4 q = q_in.normalized();
    double w = q(0), x = q(1), y = q(2), z = q(3);
    
    RotationMatrix R;
    R(0, 0) = 1 - 2 * (y*y + z*z);
    R(0, 1) = 2 * (x*y - w*z);
    R(0, 2) = 2 * (x*z + w*y);
    R(1, 0) = 2 * (x*y + w*z);
    R(1, 1) = 1 - 2 * (x*x + z*z);
    R(1, 2) = 2 * (y*z - w*x);
    R(2, 0) = 2 * (x*z - w*y);
    R(2, 1) = 2 * (y*z + w*x);
    R(2, 2) = 1 - 2 * (x*x + y*y);
    
    return R;
}

} // namespace rotation_utils

// Build rotation matrix directly using Orient's explicit formulas
Mat build_rank_block_orient(int l, const RotationMatrix& R) {
  int size = 2 * l + 1;
  Mat D = Mat::Zero(size, size);
  
  // Extract rotation matrix elements
  double xx = R(0, 0), xy = R(0, 1), xz = R(0, 2);
  double yx = R(1, 0), yy = R(1, 1), yz = R(1, 2);
  double zx = R(2, 0), zy = R(2, 1), zz = R(2, 2);
  
  // Useful constants from Orient
  const double rt2 = 1.4142135623730950488;
  const double rt3 = 1.7320508075688772935;
  const double rt5 = 2.2360679774997896964;
  const double rt7 = 2.6457513110645905905;
  const double rt6 = rt2 * rt3;
  const double rt10 = rt2 * rt5;
  const double rt14 = rt2 * rt7;
  const double rt15 = rt3 * rt5;
  const double rt21 = rt3 * rt7;
  const double rt30 = rt2 * rt3 * rt5;
  const double rt35 = rt5 * rt7;
  const double rt42 = rt2 * rt3 * rt7;
  const double rt70 = rt2 * rt5 * rt7;
  const double rt105 = rt3 * rt5 * rt7;
  
  if (l == 0) {
    D(0, 0) = 1.0;
  } else if (l == 1) {
    // Rank 1: C10, C11c, C11s maps to zz, xx, yy block with permutation
    D(0, 0) = zz; D(0, 1) = zx; D(0, 2) = zy;
    D(1, 0) = xz; D(1, 1) = xx; D(1, 2) = xy;
    D(2, 0) = yz; D(2, 1) = yx; D(2, 2) = yy;
  } else if (l == 2) {
    // Orient's explicit rank 2 formulas (indices 5-9 in Orient = 0-4 here)
    D(0, 0) = (3.0 * zz * zz - 1.0) / 2.0;
    D(0, 1) = rt3 * zx * zz;
    D(0, 2) = rt3 * zy * zz;
    D(0, 3) = (rt3 * (-2.0 * zy * zy - zz * zz + 1.0)) / 2.0;
    D(0, 4) = rt3 * zx * zy;
    
    D(1, 0) = rt3 * xz * zz;
    D(1, 1) = 2.0 * xx * zz - yy;
    D(1, 2) = yx + 2.0 * xy * zz;
    D(1, 3) = -2.0 * xy * zy - xz * zz;
    D(1, 4) = xx * zy + zx * xy;
    
    D(2, 0) = rt3 * yz * zz;
    D(2, 1) = 2.0 * yx * zz + xy;
    D(2, 2) = -xx + 2.0 * yy * zz;
    D(2, 3) = -2.0 * yy * zy - yz * zz;
    D(2, 4) = yx * zy + zx * yy;
    
    D(3, 0) = rt3 * (-2.0 * yz * yz - zz * zz + 1.0) / 2.0;
    D(3, 1) = -2.0 * yx * yz - zx * zz;
    D(3, 2) = -2.0 * yy * yz - zy * zz;
    D(3, 3) = (4.0 * yy * yy + 2.0 * zy * zy + 2.0 * yz * yz + zz * zz - 3.0) / 2.0;
    D(3, 4) = -2.0 * yx * yy - zx * zy;
    
    D(4, 0) = rt3 * xz * yz;
    D(4, 1) = xx * yz + yx * xz;
    D(4, 2) = xy * yz + yy * xz;
    D(4, 3) = -2.0 * xy * yy - xz * yz;
    D(4, 4) = xx * yy + yx * xy;
  } else if (l == 3) {
    // Orient's explicit rank 3 formulas (indices 10-16 in Orient = 0-6 here)
    D(0, 0) = (-8.0 * xx * yy + 8.0 * yx * xy + 5.0 * zz * zz * zz + 5.0 * zz) / 2.0;
    D(0, 1) = (rt6 * zx * (5.0 * zz * zz - 1.0)) / 4.0;
    D(0, 2) = (rt6 * zy * (5.0 * zz * zz - 1.0)) / 4.0;
    D(0, 3) = (rt15 * zz * (-2.0 * zy * zy - zz * zz + 1.0)) / 2.0;
    D(0, 4) = rt15 * zx * zy * zz;
    D(0, 5) = (rt10 * zx * (-4.0 * zy * zy - zz * zz + 1.0)) / 4.0;
    D(0, 6) = (rt10 * zy * (-4.0 * zy * zy - 3.0 * zz * zz + 3.0)) / 4.0;
    
    D(1, 0) = (rt3 * xz * (5.0 * zz * zz - 1.0)) / (2.0 * rt2);
    D(1, 1) = (-10.0 * xx * yy * yy + 15.0 * xx * zz * zz - xx + 10.0 * yx * xy * yy) / 4.0;
    D(1, 2) = (10.0 * xy * yz * yz + 15.0 * xy * zz * zz - 11.0 * xy - 10.0 * yy * xz * yz) / 4.0;
    D(1, 3) = (rt10 * (4.0 * xy * yy * yz - 4.0 * yy * yy * xz - 6.0 * zy * zy * xz - 3.0 * xz * zz * zz + 5.0 * xz)) / 4.0;
    D(1, 4) = rt10 * (-xx * yy * yz - yx * xy * yz + 2.0 * yx * yy * xz + 3.0 * zx * zy * xz) / 2.0;
    D(1, 5) = (rt15 * (-2.0 * xx * yy * yy - 4.0 * xx * zy * zy - xx * zz * zz + 3.0 * xx + 2.0 * yx * xy * yy)) / 4.0;
    D(1, 6) = (rt15 * (-4.0 * xy * zy * zy - 2.0 * xy * yz * yz - 3.0 * xy * zz * zz + 3.0 * xy + 2.0 * yy * xz * yz)) / 4.0;
    
    D(2, 0) = (rt3 * yz * (5.0 * zz * zz - 1.0)) / (2.0 * rt2);
    D(2, 1) = (10.0 * yx * zy * zy + 15.0 * yx * zz * zz - 11.0 * yx - 10.0 * zx * yy * zy) / 4.0;
    D(2, 2) = (5.0 * yy * zz * zz - yy + 10.0 * zy * yz * zz) / 4.0;
    D(2, 3) = (rt10 * (-4.0 * yy * zy * zz - 2.0 * zy * zy * yz - 3.0 * yz * zz * zz + yz)) / 4.0;
    D(2, 4) = (rt10 * (yx * zy * zz + zx * yy * zz + zx * zy * yz)) / 2.0;
    D(2, 5) = (rt15 * (-2.0 * yx * zy * zy - yx * zz * zz + yx - 2.0 * zx * yy * zy)) / 4.0;
    D(2, 6) = (rt15 * (-4.0 * yy * zy * zy - yy * zz * zz + yy - 2.0 * zy * yz * zz)) / 4.0;
    
    D(3, 0) = rt15 * zz * (-2.0 * yz * yz - zz * zz + 1.0) / 2.0;
    D(3, 1) = (rt10 * (4.0 * yx * yy * zy - 4.0 * zx * yy * yy - 6.0 * zx * yz * yz - 3.0 * zx * zz * zz + 5.0 * zx)) / 4.0;
    D(3, 2) = (rt10 * (-4.0 * yy * yz * zz - 2.0 * zy * yz * yz - 3.0 * zy * zz * zz + zy)) / 4.0;
    D(3, 3) = (-4.0 * xx * yy - 4.0 * yx * xy + 12.0 * yy * yy * zz + 6.0 * zy * zy * zz + 6.0 * yz * yz * zz + 3.0 * zz * zz * zz - 9.0 * zz) / 2.0;
    D(3, 4) = -6.0 * yx * yy * zz - 3.0 * zx * zy * zz - 4.0 * xy * yy - 2.0 * xz * yz;
    D(3, 5) = (rt6 * (4.0 * yx * yy * zy + 4.0 * zx * yy * yy + 4.0 * zx * zy * zy + 2.0 * zx * yz * yz + zx * zz * zz - 3.0 * zx)) / 4.0;
    D(3, 6) = (rt6 * (8.0 * yy * yy * zy + 4.0 * yy * yz * zz + 4.0 * zy * zy * zy + 2.0 * zy * yz * yz + 3.0 * zy * zz * zz - 5.0 * zy)) / 4.0;
    
    D(4, 0) = rt15 * xz * yz * zz;
    D(4, 1) = (rt10 * (-xx * yy * zy - yx * xy * zy + 2.0 * zx * xy * yy + 3.0 * zx * xz * yz)) / 2.0;
    D(4, 2) = (rt10 * (xy * yz * zz + yy * xz * zz + zy * xz * yz)) / 2.0;
    D(4, 3) = -4.0 * yx * yy - 2.0 * zx * zy - 6.0 * xy * yy * zz - 3.0 * xz * yz * zz;
    D(4, 4) = 3.0 * xx * yy * zz + 3.0 * yx * xy * zz - 4.0 * yy * yy - 2.0 * zy * zy - 2.0 * yz * yz - zz * zz + 3.0;
    D(4, 5) = (rt6 * (-xx * yy * zy - yx * xy * zy - 2.0 * zx * xy * yy - zx * xz * yz)) / 2.0;
    D(4, 6) = (rt6 * (-4.0 * xy * yy * zy - xy * yz * zz - yy * xz * zz - zy * xz * yz)) / 2.0;
    
    D(5, 0) = (rt5 * xz * (-4.0 * yz * yz - zz * zz + 1.0)) / (2.0 * rt2);
    D(5, 1) = (rt15 * (-2.0 * xx * yy * yy - 4.0 * xx * yz * yz - xx * zz * zz + 3.0 * xx + 2.0 * yx * xy * yy)) / 4.0;
    D(5, 2) = (rt15 * (-2.0 * xy * yz * yz - xy * zz * zz + xy - 2.0 * yy * xz * yz)) / 4.0;
    D(5, 3) = (rt6 * (4.0 * xy * yy * yz + 4.0 * yy * yy * xz + 2.0 * zy * zy * xz + 4.0 * xz * yz * yz + xz * zz * zz - 3.0 * xz)) / 4.0;
    D(5, 4) = (rt6 * (-xx * yy * yz - yx * xy * yz - 2.0 * yx * yy * xz - zx * zy * xz)) / 2.0;
    D(5, 5) = (10.0 * xx * yy * yy + 4.0 * xx * zy * zy + 4.0 * xx * yz * yz + xx * zz * zz - 7.0 * xx + 6.0 * yx * xy * yy) / 4.0;
    D(5, 6) = (16.0 * xy * yy * yy + 4.0 * xy * zy * zy + 6.0 * xy * yz * yz + 3.0 * xy * zz * zz - 7.0 * xy + 6.0 * yy * xz * yz) / 4.0;
    
    D(6, 0) = (rt5 * yz * (-4.0 * yz * yz - 3.0 * zz * zz + 3.0)) / (2.0 * rt2);
    D(6, 1) = (rt15 * (-2.0 * yx * zy * zy - 4.0 * yx * yz * yz - 3.0 * yx * zz * zz + 3.0 * yx + 2.0 * zx * yy * zy)) / 4.0;
    D(6, 2) = (rt15 * (-4.0 * yy * yz * yz - yy * zz * zz + yy - 2.0 * zy * yz * zz)) / 4.0;
    D(6, 3) = (rt6 * (8.0 * yy * yy * yz + 4.0 * yy * zy * zz + 2.0 * zy * zy * yz + 4.0 * yz * yz * yz + 3.0 * yz * zz * zz - 5.0 * yz)) / 4.0;
    D(6, 4) = (rt6 * (-4.0 * yx * yy * yz - yx * zy * zz - zx * yy * zz - zx * zy * yz)) / 2.0;
    D(6, 5) = (16.0 * yx * yy * yy + 6.0 * yx * zy * zy + 4.0 * yx * yz * yz + 3.0 * yx * zz * zz - 7.0 * yx + 6.0 * zx * yy * zy) / 4.0;
    D(6, 6) = (16.0 * yy * yy * yy + 12.0 * yy * zy * zy + 12.0 * yy * yz * yz + 3.0 * yy * zz * zz - 15.0 * yy + 6.0 * zy * yz * zz) / 4.0;
  } else if (l == 4) {
    // Orient's complete explicit rank 4 formulas (indices 17-25 in Orient = 0-8 here)
    // Row 0 (dd(17,*))
    D(0, 0) = (-68.0*yy*yy*zz*zz+68.0*yy*yy+136.0*yy*zy*yz*zz-68.0*zy*zy*yz*yz+68.0*zy*zy+68.0*yz*yz+35.0*zz*zz*zz*zz+38.0*zz*zz-65.0)/8.0;
    D(0, 1) = (rt10*(10.0*yx*yy*zy*zz-10.0*yx*zy*zy*yz+10.0*yx*yz-10.0*zx*yy*yy*zz+10.0*zx*yy*zy*yz+7.0*zx*zz*zz*zz+7.0*zx*zz))/4.0;
    D(0, 2) = (rt10*zy*zz*(7.0*zz*zz-3.0))/4.0;
    D(0, 3) = (rt5*(-14.0*zy*zy*zz*zz+2.0*zy*zy-7.0*zz*zz*zz*zz+8.0*zz*zz-1.0))/4.0;
    D(0, 4) = (rt5*zx*zy*(7.0*zz*zz-1.0))/2.0;
    D(0, 5) = (rt70*(2.0*yx*yy*zy*zz-2.0*yx*zy*zy*yz+2.0*yx*yz-2.0*zx*yy*yy*zz+2.0*zx*yy*zy*yz-4.0*zx*zy*zy*zz-zx*zz*zz*zz+3.0*zx*zz))/4.0;
    D(0, 6) = (rt70*zy*zz*(-4.0*zy*zy-3.0*zz*zz+3.0))/4.0;
    D(0, 7) = (rt35*(4.0*yy*yy*zz*zz-4.0*yy*yy-8.0*yy*zy*yz*zz+8.0*zy*zy*zy*zy+4.0*zy*zy*yz*yz+8.0*zy*zy*zz*zz-12.0*zy*zy-4.0*yz*yz+zz*zz*zz*zz-6.0*zz*zz+5.0))/8.0;
    D(0, 8) = (rt35*zx*zy*(-2.0*zy*zy-zz*zz+1.0))/2.0;
    
    // Row 1 (dd(18,*))
    D(1, 0) = (rt5*(10.0*xy*yy*yz*zz-10.0*xy*zy*yz*yz+10.0*xy*zy-10.0*yy*yy*xz*zz+10.0*yy*zy*xz*yz+7.0*xz*zz*zz*zz+7.0*xz*zz))/(2.0*rt2);
    D(1, 1) = (-24.0*xx*yy*yy*zz+28.0*xx*zz*zz*zz+28.0*xx*zz+24.0*yx*xy*yy*zz-31.0*yy*zz*zz+3.0*yy+34.0*zy*yz*zz)/4.0;
    D(1, 2) = (-34.0*yx*zy*zy-3.0*yx*zz*zz+31.0*yx+34.0*zx*yy*zy+24.0*xy*yz*yz*zz+28.0*xy*zz*zz*zz+4.0*xy*zz-24.0*yy*xz*yz*zz)/4.0;
    D(1, 3) = (rt2*(4.0*xy*yy*yz*zz+4.0*xy*zy*yz*yz+xy*zy*zz*zz-3.0*xy*zy-4.0*yy*yy*xz*zz-4.0*yy*zy*xz*yz-15.0*zy*zy*xz*zz-7.0*xz*zz*zz*zz+8.0*xz*zz))/2.0;
    D(1, 4) = (rt2*(-22.0*xx*yy*yy*zy-30.0*xx*zy*zy*zy-xx*zy*zz*zz+29.0*xx*zy+22.0*yx*xy*yy*zy+30.0*zx*xy*zy*zy+22.0*zx*xy*yz*yz+29.0*zx*xy*zz*zz-23.0*zx*xy-22.0*zx*yy*xz*yz))/4.0;
    D(1, 5) = (rt7*(-8.0*xx*yy*yy*zz-16.0*xx*zy*zy*zz-4.0*xx*zz*zz*zz+12.0*xx*zz+8.0*yx*xy*yy*zz+4.0*yy*zy*zy+yy*zz*zz-yy+2.0*zy*yz*zz))/4.0;
    D(1, 6) = (rt7*(-2.0*yx*zy*zy-yx*zz*zz+yx-2.0*zx*yy*zy-16.0*xy*zy*zy*zz-8.0*xy*yz*yz*zz-12.0*xy*zz*zz*zz+12.0*xy*zz+8.0*yy*xz*yz*zz))/4.0;
    D(1, 7) = (rt14*(-2.0*xy*yy*yz*zz+8.0*xy*zy*zy*zy+2.0*xy*zy*yz*yz+4.0*xy*zy*zz*zz-6.0*xy*zy+2.0*yy*yy*xz*zz-2.0*yy*zy*xz*yz+4.0*zy*zy*xz*zz+xz*zz*zz*zz-3.0*xz*zz))/4.0;
    D(1, 8) = (rt14*(-2.0*xx*yy*yy*zy-4.0*xx*zy*zy*zy-xx*zy*zz*zz+3.0*xx*zy+2.0*yx*xy*yy*zy-4.0*zx*xy*zy*zy-2.0*zx*xy*yz*yz-3.0*zx*xy*zz*zz+3.0*zx*xy+2.0*zx*yy*xz*yz))/4.0;
    
    // Row 2 (dd(19,*))
    D(2, 0) = (rt5*yz*zz*(7.0*zz*zz-3.0))/(2.0*rt2);
    D(2, 1) = (24.0*yx*zy*zy*zz+28.0*yx*zz*zz*zz+4.0*yx*zz-24.0*zx*yy*zy*zz-34.0*xy*yz*yz-3.0*xy*zz*zz+31.0*xy+34.0*yy*xz*yz)/4.0;
    D(2, 2) = (-34.0*xx*yy*yy+3.0*xx*zz*zz+3.0*xx+34.0*yx*xy*yy+4.0*yy*zz*zz*zz+28.0*yy*zz+24.0*zy*yz*zz*zz)/4.0;
    D(2, 3) = (rt2*(-7.0*yy*zy*zz*zz+yy*zy-7.0*zy*zy*yz*zz-7.0*yz*zz*zz*zz+4.0*yz*zz))/2.0;
    D(2, 4) = (rt2*(-8.0*yx*zy*zy*zy-yx*zy*zz*zz+7.0*yx*zy+8.0*zx*yy*zy*zy+7.0*zx*yy*zz*zz-zx*yy+22.0*zx*zy*yz*zz))/4.0;
    D(2, 5) = (rt7*(-8.0*yx*zy*zy*zz-4.0*yx*zz*zz*zz+4.0*yx*zz-8.0*zx*yy*zy*zz-4.0*xy*zy*zy-2.0*xy*yz*yz-3.0*xy*zz*zz+3.0*xy+2.0*yy*xz*yz))/4.0;
    D(2, 6) = (rt7*(2.0*xx*yy*yy+4.0*xx*zy*zy+xx*zz*zz-3.0*xx-2.0*yx*xy*yy-16.0*yy*zy*zy*zz-4.0*yy*zz*zz*zz+4.0*yy*zz-8.0*zy*yz*zz*zz))/4.0;
    D(2, 7) = (rt14*(8.0*yy*zy*zy*zy+4.0*yy*zy*zz*zz-4.0*yy*zy+4.0*zy*zy*yz*zz+yz*zz*zz*zz-yz*zz))/4.0;
    D(2, 8) = (rt14*(-2.0*yx*zy*zy*zy-yx*zy*zz*zz+yx*zy-6.0*zx*yy*zy*zy-zx*yy*zz*zz+zx*yy-2.0*zx*zy*yz*zz))/4.0;
    
    // Row 3 (dd(20,*))
    D(3, 0) = rt5*(-14.0*yz*yz*zz*zz+2.0*yz*yz-7.0*zz*zz*zz*zz+8.0*zz*zz-1.0)/4.0;
    D(3, 1) = (rt2*(4.0*yx*yy*zy*zz+4.0*yx*zy*zy*yz+yx*yz*zz*zz-3.0*yx*yz-4.0*zx*yy*yy*zz-4.0*zx*yy*zy*yz-15.0*zx*yz*yz*zz-7.0*zx*zz*zz*zz+8.0*zx*zz))/2.0;
    D(3, 2) = (rt2*(-7.0*yy*yz*zz*zz+yy*yz-7.0*zy*yz*yz*zz-7.0*zy*zz*zz*zz+4.0*zy*zz))/2.0;
    D(3, 3) = (14.0*yy*yy*zz*zz-10.0*yy*yy+14.0*zy*zy*yz*yz+14.0*zy*zy*zz*zz-12.0*zy*zy+14.0*yz*yz*zz*zz-12.0*yz*yz+7.0*zz*zz*zz*zz-20.0*zz*zz+11.0)/2.0;
    D(3, 4) = -7.0*yx*yy*zz*zz+5.0*yx*yy-7.0*zx*zy*yz*yz-7.0*zx*zy*zz*zz+6.0*zx*zy;
    D(3, 5) = (rt14*(4.0*yx*zy*zy*yz+yx*yz*zz*zz-3.0*yx*yz+4.0*zx*yy*yy*zz+4.0*zx*zy*zy*zz+zx*yz*yz*zz+zx*zz*zz*zz-4.0*zx*zz))/2.0;
    D(3, 6) = (rt14*(4.0*yy*yy*zy*zz+4.0*yy*zy*zy*yz+3.0*yy*yz*zz*zz-yy*yz+4.0*zy*zy*zy*zz+3.0*zy*yz*yz*zz+3.0*zy*zz*zz*zz-4.0*zy*zz))/2.0;
    D(3, 7) = (rt7*(-16.0*yy*yy*zy*zy-8.0*yy*yy*zz*zz+8.0*yy*yy-8.0*zy*zy*zy*zy-8.0*zy*zy*yz*yz-8.0*zy*zy*zz*zz+16.0*zy*zy-2.0*yz*yz*zz*zz+6.0*yz*yz-zz*zz*zz*zz+8.0*zz*zz-7.0))/4.0;
    D(3, 8) = rt7*(2.0*yx*yy*zy*zy+yx*yy*zz*zz-yx*yy+2.0*zx*yy*yy*zy+2.0*zx*zy*zy*zy+zx*zy*yz*yz+zx*zy*zz*zz-2.0*zx*zy);
    
    // Row 4 (dd(21,*))
    D(4, 0) = rt5*xz*yz*(7.0*zz*zz-1.0)/2.0;
    D(4, 1) = (rt2*(-22.0*xx*yy*yy*yz-30.0*xx*yz*yz*yz-xx*yz*zz*zz+29.0*xx*yz+22.0*yx*xy*yy*yz+22.0*yx*zy*zy*xz+30.0*yx*xz*yz*yz+29.0*yx*xz*zz*zz-23.0*yx*xz-22.0*zx*yy*zy*xz))/4.0;
    D(4, 2) = (rt2*(14.0*xy*yz*yz*yz+21.0*xy*yz*zz*zz-15.0*xy*yz-14.0*yy*xz*yz*yz+7.0*yy*xz*zz*zz-yy*xz))/4.0;
    D(4, 3) = -7.0*xy*yy*zz*zz+5.0*xy*yy-7.0*zy*zy*xz*yz-7.0*xz*yz*zz*zz+6.0*xz*yz;
    D(4, 4) = (-28.0*xx*yy*yy*yy-14.0*xx*yy*zy*zy-14.0*xx*yy*yz*yz+7.0*xx*yy*zz*zz+23.0*xx*yy+28.0*yx*xy*yy*yy+14.0*yx*xy*zy*zy+14.0*yx*xy*yz*yz+21.0*yx*xy*zz*zz-19.0*yx*xy)/2.0;
    D(4, 5) = (rt14*(2.0*xx*yy*yy*yz-4.0*xx*zy*zy*yz+2.0*xx*yz*yz*yz-xx*yz*zz*zz+xx*yz+6.0*yx*xy*yy*yz-8.0*yx*yy*yy*xz-6.0*yx*zy*zy*xz-2.0*yx*xz*yz*yz-3.0*yx*xz*zz*zz+5.0*yx*xz-6.0*zx*yy*zy*xz))/4.0;
    D(4, 6) = (rt14*(8.0*xy*yy*yy*yz-4.0*xy*zy*zy*yz-6.0*xy*yz*yz*yz-9.0*xy*yz*zz*zz+7.0*xy*yz-8.0*yy*yy*yy*xz-12.0*yy*zy*zy*xz+6.0*yy*xz*yz*yz-3.0*yy*xz*zz*zz+9.0*yy*xz))/4.0;
    D(4, 7) = (rt7*(8.0*xy*yy*zy*zy+4.0*xy*yy*zz*zz-4.0*xy*yy+4.0*zy*zy*xz*yz+xz*yz*zz*zz-3.0*xz*yz))/2.0;
    D(4, 8) = (rt7*(-4.0*xx*yy*zy*zy+2.0*xx*yy*yz*yz-xx*yy*zz*zz+xx*yy-4.0*yx*xy*zy*zy-2.0*yx*xy*yz*yz-3.0*yx*xy*zz*zz+3.0*yx*xy))/2.0;
    
    // Row 5 (dd(22,*))
    D(5, 0) = (rt35*(2.0*xy*yy*yz*zz-2.0*xy*zy*yz*yz+2.0*xy*zy-2.0*yy*yy*xz*zz+2.0*yy*zy*xz*yz-4.0*xz*yz*yz*zz-xz*zz*zz*zz+3.0*xz*zz))/(2.0*rt2);
    D(5, 1) = (rt7*(-8.0*xx*yy*yy*zz-16.0*xx*yz*yz*zz-4.0*xx*zz*zz*zz+12.0*xx*zz+8.0*yx*xy*yy*zz+4.0*yy*yz*yz+yy*zz*zz-yy+2.0*zy*yz*zz))/4.0;
    D(5, 2) = (rt7*(-2.0*yx*zy*zy-4.0*yx*yz*yz-3.0*yx*zz*zz+3.0*yx+2.0*zx*yy*zy-8.0*xy*yz*yz*zz-4.0*xy*zz*zz*zz+4.0*xy*zz-8.0*yy*xz*yz*zz))/4.0;
    D(5, 3) = (rt14*(4.0*xy*zy*yz*yz+xy*zy*zz*zz-3.0*xy*zy+4.0*yy*yy*xz*zz+zy*zy*xz*zz+4.0*xz*yz*yz*zz+xz*zz*zz*zz-4.0*xz*zz))/2.0;
    D(5, 4) = (rt14*(2.0*xx*yy*yy*zy+2.0*xx*zy*zy*zy-4.0*xx*zy*yz*yz-xx*zy*zz*zz+xx*zy+6.0*yx*xy*yy*zy-8.0*zx*xy*yy*yy-2.0*zx*xy*zy*zy-6.0*zx*xy*yz*yz-3.0*zx*xy*zz*zz+5.0*zx*xy-6.0*zx*yy*xz*yz))/4.0;
    D(5, 5) = (40.0*xx*yy*yy*zz+16.0*xx*zy*zy*zz+16.0*xx*yz*yz*zz+4.0*xx*zz*zz*zz-28.0*xx*zz+24.0*yx*xy*yy*zz-48.0*yy*yy*yy-36.0*yy*zy*zy-36.0*yy*yz*yz-9.0*yy*zz*zz+45.0*yy-18.0*zy*yz*zz)/4.0;
    D(5, 6) = (48.0*yx*yy*yy+18.0*yx*zy*zy+12.0*yx*yz*yz+9.0*yx*zz*zz-21.0*yx+18.0*zx*yy*zy+64.0*xy*yy*yy*zz+16.0*xy*zy*zy*zz+24.0*xy*yz*yz*zz+12.0*xy*zz*zz*zz-28.0*xy*zz+24.0*yy*xz*yz*zz)/4.0;
    D(5, 7) = (rt2*(-32.0*xy*yy*yy*zy-6.0*xy*yy*yz*zz-8.0*xy*zy*zy*zy-10.0*xy*zy*yz*yz-4.0*xy*zy*zz*zz+14.0*xy*zy-10.0*yy*yy*xz*zz-6.0*yy*zy*xz*yz-4.0*zy*zy*xz*zz-4.0*xz*yz*yz*zz-xz*zz*zz*zz+7.0*xz*zz))/4.0;
    D(5, 8) = (rt2*(10.0*xx*yy*yy*zy+4.0*xx*zy*zy*zy+4.0*xx*zy*yz*yz+xx*zy*zz*zz-7.0*xx*zy+6.0*yx*xy*yy*zy+16.0*zx*xy*yy*yy+4.0*zx*xy*zy*zy+6.0*zx*xy*yz*yz+3.0*zx*xy*zz*zz-7.0*zx*xy+6.0*zx*yy*xz*yz))/4.0;
    
    // Row 6 (dd(23,*))
    D(6, 0) = (rt35*yz*zz*(-4.0*yz*yz-3.0*zz*zz+3.0))/(2.0*rt2);
    D(6, 1) = (rt7*(-8.0*yx*zy*zy*zz-16.0*yx*yz*yz*zz-12.0*yx*zz*zz*zz+12.0*yx*zz+8.0*zx*yy*zy*zz-2.0*xy*yz*yz-xy*zz*zz+xy-2.0*yy*xz*yz))/4.0;
    D(6, 2) = (rt7*(2.0*xx*yy*yy+4.0*xx*yz*yz+xx*zz*zz-3.0*xx-2.0*yx*xy*yy-16.0*yy*yz*yz*zz-4.0*yy*zz*zz*zz+4.0*yy*zz-8.0*zy*yz*zz*zz))/4.0;
    D(6, 3) = (rt14*(4.0*yy*yy*yz*zz+4.0*yy*zy*yz*yz+3.0*yy*zy*zz*zz-yy*zy+3.0*zy*zy*yz*zz+4.0*yz*yz*yz*zz+3.0*yz*zz*zz*zz-4.0*yz*zz))/2.0;
    D(6, 4) = (rt14*(8.0*yx*yy*yy*zy-4.0*yx*zy*yz*yz-3.0*yx*zy*zz*zz+yx*zy-8.0*zx*yy*yy*yy-12.0*zx*yy*yz*yz-3.0*zx*yy*zz*zz+9.0*zx*yy-6.0*zx*zy*yz*zz))/4.0;
    D(6, 5) = (64.0*yx*yy*yy*zz+24.0*yx*zy*zy*zz+16.0*yx*yz*yz*zz+12.0*yx*zz*zz*zz-28.0*yx*zz+24.0*zx*yy*zy*zz+48.0*xy*yy*yy+12.0*xy*zy*zy+18.0*xy*yz*yz+9.0*xy*zz*zz-21.0*xy+18.0*yy*xz*yz)/4.0;
    D(6, 6) = (-30.0*xx*yy*yy-12.0*xx*zy*zy-12.0*xx*yz*yz-3.0*xx*zz*zz+21.0*xx-18.0*yx*xy*yy+64.0*yy*yy*yy*zz+48.0*yy*zy*zy*zz+48.0*yy*yz*yz*zz+12.0*yy*zz*zz*zz-60.0*yy*zz+24.0*zy*yz*zz*zz)/4.0;
    D(6, 7) = (rt2*(-32.0*yy*yy*yy*zy-16.0*yy*yy*yz*zz-24.0*yy*zy*zy*zy-16.0*yy*zy*yz*yz-12.0*yy*zy*zz*zz+28.0*yy*zy-12.0*zy*zy*yz*zz-4.0*yz*yz*yz*zz-3.0*yz*zz*zz*zz+7.0*yz*zz))/4.0;
    D(6, 8) = (rt2*(16.0*yx*yy*yy*zy+6.0*yx*zy*zy*zy+4.0*yx*zy*yz*yz+3.0*yx*zy*zz*zz-7.0*yx*zy+16.0*zx*yy*yy*yy+18.0*zx*yy*zy*zy+12.0*zx*yy*yz*yz+3.0*zx*yy*zz*zz-15.0*zx*yy+6.0*zx*zy*yz*zz))/4.0;
    
    // Row 7 (dd(24,*))
    D(7, 0) = rt35*(4.0*yy*yy*zz*zz-4.0*yy*yy-8.0*yy*zy*yz*zz+4.0*zy*zy*yz*yz-4.0*zy*zy+8.0*yz*yz*yz*yz+8.0*yz*yz*zz*zz-12.0*yz*yz+zz*zz*zz*zz-6.0*zz*zz+5.0)/8.0;
    D(7, 1) = (rt14*(-2.0*yx*yy*zy*zz+2.0*yx*zy*zy*yz+8.0*yx*yz*yz*yz+4.0*yx*yz*zz*zz-6.0*yx*yz+2.0*zx*yy*yy*zz-2.0*zx*yy*zy*yz+4.0*zx*yz*yz*zz+zx*zz*zz*zz-3.0*zx*zz))/4.0;
    D(7, 2) = (rt14*(8.0*yy*yz*yz*yz+4.0*yy*yz*zz*zz-4.0*yy*yz+4.0*zy*yz*yz*zz+zy*zz*zz*zz-zy*zz))/4.0;
    D(7, 3) = (rt7*(-16.0*yy*yy*yz*yz-8.0*yy*yy*zz*zz+8.0*yy*yy-8.0*zy*zy*yz*yz-2.0*zy*zy*zz*zz+6.0*zy*zy-8.0*yz*yz*yz*yz-8.0*yz*yz*zz*zz+16.0*yz*yz-zz*zz*zz*zz+8.0*zz*zz-7.0))/4.0;
    D(7, 4) = (rt7*(8.0*yx*yy*yz*yz+4.0*yx*yy*zz*zz-4.0*yx*yy+4.0*zx*zy*yz*yz+zx*zy*zz*zz-3.0*zx*zy))/2.0;
    D(7, 5) = (rt2*(-32.0*yx*yy*yy*yz-6.0*yx*yy*zy*zz-10.0*yx*zy*zy*yz-8.0*yx*yz*yz*yz-4.0*yx*yz*zz*zz+14.0*yx*yz-10.0*zx*yy*yy*zz-6.0*zx*yy*zy*yz-4.0*zx*zy*zy*zz-4.0*zx*yz*yz*zz-zx*zz*zz*zz+7.0*zx*zz))/4.0;
    D(7, 6) = (rt2*(-32.0*yy*yy*yy*yz-16.0*yy*yy*zy*zz-16.0*yy*zy*zy*yz-24.0*yy*yz*yz*yz-12.0*yy*yz*zz*zz+28.0*yy*yz-4.0*zy*zy*zy*zz-12.0*zy*yz*yz*zz-3.0*zy*zz*zz*zz+7.0*zy*zz))/4.0;
    D(7, 7) = (64.0*yy*yy*yy*yy+64.0*yy*yy*zy*zy+64.0*yy*yy*yz*yz+20.0*yy*yy*zz*zz-84.0*yy*yy+24.0*yy*zy*yz*zz+8.0*zy*zy*zy*zy+20.0*zy*zy*yz*yz+8.0*zy*zy*zz*zz-28.0*zy*zy+8.0*yz*yz*yz*yz+8.0*yz*yz*zz*zz-28.0*yz*yz+zz*zz*zz*zz-14.0*zz*zz+21.0)/8.0;
    D(7, 8) = (-16.0*yx*yy*yy*yy-8.0*yx*yy*zy*zy-8.0*yx*yy*yz*yz-4.0*yx*yy*zz*zz+12.0*yx*yy-8.0*zx*yy*yy*zy-2.0*zx*zy*zy*zy-4.0*zx*zy*yz*yz-zx*zy*zz*zz+5.0*zx*zy)/2.0;
    
    // Row 8 (dd(25,*))
    D(8, 0) = rt35*xz*yz*(-2.0*yz*yz-zz*zz+1.0)/2.0;
    D(8, 1) = (rt14*(-2.0*xx*yy*yy*yz-4.0*xx*yz*yz*yz-xx*yz*zz*zz+3.0*xx*yz+2.0*yx*xy*yy*yz-2.0*yx*zy*zy*xz-4.0*yx*xz*yz*yz-3.0*yx*xz*zz*zz+3.0*yx*xz+2.0*zx*yy*zy*xz))/4.0;
    D(8, 2) = (rt14*(-4.0*xy*yz*yz*yz-3.0*xy*yz*zz*zz+3.0*xy*yz-4.0*yy*xz*yz*yz-yy*xz*zz*zz+yy*xz))/4.0;
    D(8, 3) = rt7*(2.0*xy*yy*yz*yz+xy*yy*zz*zz-xy*yy+2.0*yy*yy*xz*yz+zy*zy*xz*yz+2.0*xz*yz*yz*yz+xz*yz*zz*zz-2.0*xz*yz);
    D(8, 4) = (rt7*(2.0*xx*yy*zy*zy-4.0*xx*yy*yz*yz-xx*yy*zz*zz+xx*yy-2.0*yx*xy*zy*zy-4.0*yx*xy*yz*yz-3.0*yx*xy*zz*zz+3.0*yx*xy))/2.0;
    D(8, 5) = (rt2*(10.0*xx*yy*yy*yz+4.0*xx*zy*zy*yz+4.0*xx*yz*yz*yz+xx*yz*zz*zz-7.0*xx*yz+6.0*yx*xy*yy*yz+16.0*yx*yy*yy*xz+6.0*yx*zy*zy*xz+4.0*yx*xz*yz*yz+3.0*yx*xz*zz*zz-7.0*yx*xz+6.0*zx*yy*zy*xz))/4.0;
    D(8, 6) = (rt2*(16.0*xy*yy*yy*yz+4.0*xy*zy*zy*yz+12.0*xy*yz*yz*yz+9.0*xy*yz*zz*zz-13.0*xy*yz+16.0*yy*yy*yy*xz+12.0*yy*zy*zy*xz+12.0*yy*xz*yz*yz+3.0*yy*xz*zz*zz-15.0*yy*xz))/4.0;
    D(8, 7) = (-16.0*xy*yy*yy*yy-8.0*xy*yy*zy*zy-8.0*xy*yy*yz*yz-4.0*xy*yy*zz*zz+12.0*xy*yy-8.0*yy*yy*xz*yz-4.0*zy*zy*xz*yz-2.0*xz*yz*yz*yz-xz*yz*zz*zz+5.0*xz*yz)/2.0;
    D(8, 8) = (8.0*xx*yy*yy*yy+4.0*xx*yy*zy*zy+4.0*xx*yy*yz*yz+xx*yy*zz*zz-7.0*xx*yy+8.0*yx*xy*yy*yy+4.0*yx*xy*zy*zy+4.0*yx*xy*yz*yz+3.0*yx*xy*zz*zz-5.0*yx*xy)/2.0;
  }
  
  return D;
}

Mat wigner_d_matrix(const RotationMatrix& R, int lmax) {
  if (lmax < 0) {
    throw std::invalid_argument("lmax must be non-negative");
  }

  int total_size = (lmax + 1) * (lmax + 1);
  Mat D = Mat::Zero(total_size, total_size);

  int offset = 0;
  for (int l = 0; l <= lmax; ++l) {
    int block_size = 2 * l + 1;
    Mat block = build_rank_block_orient(l, R);
    D.block(offset, offset, block_size, block_size) = block;
    offset += block_size;
  }

  return D;
}

occ::dma::Mult& rotate_multipole(occ::dma::Mult& mult, const RotationMatrix& R) {
    if (mult.max_rank < 0) return mult;
    
    Mat D = wigner_d_matrix(R, mult.max_rank);
    Vec q_rotated = D * mult.q;
    mult.q = q_rotated;
    
    return mult;
}

occ::dma::Mult rotated_multipole(const occ::dma::Mult& mult, const RotationMatrix& R) {
    occ::dma::Mult result = mult;  // Copy
    return rotate_multipole(result, R);
}

} // namespace occ::mults