#include <occ/core/kabsch.h>

namespace occ::core::linalg {

occ::Mat3 kabsch_rotation_matrix(const occ::Mat3N &a, const occ::Mat3N &b,
                                 bool ensure_proper_rotation) {
  /*
  Calculate the optimal rotation matrix `R` to rotate
  `A` onto `B`, minimising root-mean-square deviation so that
  this may be then calculated.

  See: https://en.wikipedia.org/wiki/Kabsch_algorithm

  Reference:
  ```
  Kabsch, W. Acta Cryst. A, 32, 922-923, (1976)
  DOI: http://dx.doi.org/10.1107/S0567739476001873
  ```
  Args:
      A : (3,N) matrix where N is the number of vectors and D
          is the dimension of each vector
      B : (3,N) matrix where N is the number of
          vectors and D is the dimension of each vector
  Returns:
      (D,D) rotation matrix where D is the dimension of each vector
  */

  // Calculate the covariance matrix
  occ::Mat3 cov = a * b.transpose();

  // Use singular value decomposition to calculate
  // the optimal rotation matrix
  Eigen::JacobiSVD<occ::Mat> svd(cov,
                                 Eigen::ComputeThinU | Eigen::ComputeThinV);
  // auto v, s, w = np.linalg.svd(cov)
  occ::Mat3N u = svd.matrixU();
  occ::MatN3 v = svd.matrixV();

  // check the determinant to ensure a right-handed
  // coordinate system
  occ::Mat d = occ::Mat::Identity(3, 3);
  if (ensure_proper_rotation)
    d(2, 2) = (u.determinant() * v.determinant() < 0.0) ? -1 : 1;
  return v * d * u.transpose();
}

} // namespace occ::core::linalg
