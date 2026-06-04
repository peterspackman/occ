#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/geometry/marching_cubes.h>

namespace occ::isosurface::impl {

// Local finite-difference gradient of the field in the functor's own
// (MC-local) coordinates. Using FD (rather than the functor's gradient(),
// which may be normalised or in a different basis) keeps everything correct
// and consistent across functor types and coordinate spaces.
template <typename Func>
inline FVec3 local_gradient(const Func &func, const FVec3 &p) {
  using occ::geometry::mc::impl::eval_point;
  constexpr float h = 1.0e-3f;
  FVec3 g;
  for (int i = 0; i < 3; i++) {
    FVec3 pp = p, pm = p;
    pp[i] += h;
    pm[i] -= h;
    g[i] = (eval_point(func, pp) - eval_point(func, pm)) / (2.0f * h);
  }
  return g;
}

// Newton-project a point onto {f = iso} along the (FD) gradient.
template <typename Func>
inline FVec3 project_to_isosurface(const Func &func, FVec3 p, float iso,
                                   int steps) {
  using occ::geometry::mc::impl::eval_point;
  for (int s = 0; s < steps; s++) {
    const FVec3 g = local_gradient(func, p);
    const float gn2 = g.squaredNorm();
    if (gn2 < 1.0e-20f)
      break;
    const float fv = eval_point(func, p) - iso;
    p -= (fv / gn2) * g;
  }
  return p;
}

} // namespace occ::isosurface::impl
