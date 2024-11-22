#pragma once
#include <Eigen/Dense>

namespace quickhull {

template <typename T> class Plane {
public:
  Eigen::Matrix<T, 3, 1> normal;

  // Signed distance (if normal is of length 1) to the plane from origin
  T m_D;

  // Normal length squared
  T m_sqrNLength;

  bool isPointOnPositiveSide(const Eigen::Matrix<T, 3, 1> &Q) const {
    T d = normal.dot(Q) + m_D;
    if (d >= 0)
      return true;
    return false;
  }

  Plane() = default;

  // Construct a plane using normal N and any point P on the plane
  Plane(const Eigen::Matrix<T, 3, 1> &N, const Eigen::Matrix<T, 3, 1> &P)
      : normal(N), m_D(-N.dot(P)), m_sqrNLength(normal.squaredNorm()) {}
};

} // namespace quickhull
