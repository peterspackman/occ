#pragma once
#include <Eigen/Dense>
#include <occ/geometry/plane.h>
#include <occ/geometry/ray.h>

namespace quickhull {

namespace mathutils {

inline float getSquaredDistanceBetweenPointAndRay(const Eigen::Vector3f &p,
                                                  const Ray<float> &r) {
  const Eigen::Vector3f s = p - r.m_S;
  float t = s.dot(r.m_V);
  return s.squaredNorm() - t * t * r.m_VInvLengthSquared;
}

// Note that the unit of distance returned is relative to plane's normal's
// length (divide by N.getNormalized() if needed to get the "real" distance).
inline float getSignedDistanceToPlane(Eigen::Vector3f v,
                                      const Plane<float> &p) {
  return p.normal.dot(v) + p.m_D;
}

inline Eigen::Vector3f triangle_normal(const Eigen::Vector3f &a,
                                       const Eigen::Vector3f &b,
                                       const Eigen::Vector3f &c) {
  // We want to get (a-c).crossProduct(b-c) without constructing temp vectors
  return (a - c).cross(b - c);
}

inline double getSquaredDistanceBetweenPointAndRay(const Eigen::Vector3d &p,
                                                   const Ray<double> &r) {
  const Eigen::Vector3d s = p - r.m_S;
  double t = s.dot(r.m_V);
  return s.squaredNorm() - t * t * r.m_VInvLengthSquared;
}

// Note that the unit of distance returned is relative to plane's normal's
// length (divide by N.getNormalized() if needed to get the "real" distance).
inline double getSignedDistanceToPlane(Eigen::Vector3d v,
                                       const Plane<double> &p) {
  return p.normal.dot(v) + p.m_D;
}

inline Eigen::Vector3d triangle_normal(const Eigen::Vector3d &a,
                                       const Eigen::Vector3d &b,
                                       const Eigen::Vector3d &c) {
  // We want to get (a-c).crossProduct(b-c) without constructing temp vectors
  return (a - c).cross(b - c);
}

} // namespace mathutils

} // namespace quickhull
