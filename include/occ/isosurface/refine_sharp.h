#pragma once
#include <ankerl/unordered_dense.h>
#include <array>
#include <cmath>
#include <occ/core/linear_algebra.h>
#include <occ/geometry/marching_cubes.h>
#include <occ/isosurface/mesh_utils.h>
#include <occ/isosurface/projection.h>
#include <utility>
#include <vector>

namespace occ::isosurface {

struct SharpRefineParams {
  int passes{0};                   // 0 = disabled
  float angle_threshold_deg{30.0f}; // dihedral angle to flag an edge as sharp
  int projection_steps{3};         // Newton steps onto the level set
};

// Adaptive mesh-space refinement of "sharp-ish" features (convex creases /
// corners) that marching cubes flat-cuts. Edges whose two incident faces meet
// at more than angle_threshold_deg are split at their midpoint, the midpoint is
// projected onto {f = iso}, and the 1-3 incident triangles are re-triangulated
// (red-green). This is crack-free by construction: a split edge is shared, so
// both incident faces see the same midpoint. Operates in MC-local coordinates
// on the flat (vertices, normals, curvature, faces) buffers in place.
template <typename Func>
void refine_sharp_edges(const Func &func, float isovalue,
                        const SharpRefineParams &params, std::vector<float> &V,
                        std::vector<float> &N, std::vector<float> &C,
                        std::vector<uint32_t> &F) {
  if (params.passes <= 0)
    return;

  // Weld coincident vertices first: without it a shared sharp edge looks like
  // two index-distinct boundary edges and adjacency / the red-green split would
  // be wrong (missed edges / real cracks). Idempotent if already welded.
  weld_vertices(V, N, C, F);

  const float cos_thresh =
      std::cos(double(params.angle_threshold_deg) * M_PI / 180.0);

  // Normals are computed in MC-local space; map them to cartesian for
  // non-orthogonal functors (e.g. void surfaces on a skewed lattice).
  FMat3 Jit = FMat3::Identity();
  if constexpr (occ::geometry::mc::impl::has_basis_transform<Func>::value)
    Jit = func.basis_transform();

  using Edge = std::pair<uint32_t, uint32_t>;
  struct EdgeHash {
    size_t operator()(const Edge &e) const {
      return ankerl::unordered_dense::detail::wyhash::hash(&e, sizeof(e));
    }
  };
  auto ekey = [](uint32_t a, uint32_t b) {
    return a < b ? Edge{a, b} : Edge{b, a};
  };
  auto vget = [&](uint32_t i) {
    return FVec3(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
  };

  for (int pass = 0; pass < params.passes; pass++) {
    const size_t nf = F.size() / 3;

    // Geometric (per-face) normals for the dihedral test.
    std::vector<FVec3> fn(nf);
    for (size_t f = 0; f < nf; f++) {
      FVec3 a = vget(F[3 * f]), b = vget(F[3 * f + 1]), c = vget(F[3 * f + 2]);
      fn[f] = (b - a).cross(c - a).normalized();
    }

    // Edge -> up to two incident faces.
    ankerl::unordered_dense::map<Edge, std::array<int, 2>, EdgeHash> einc;
    for (size_t f = 0; f < nf; f++)
      for (int e = 0; e < 3; e++) {
        auto k = ekey(F[3 * f + e], F[3 * f + (e + 1) % 3]);
        auto it = einc.find(k);
        if (it == einc.end())
          einc.emplace(k, std::array<int, 2>{int(f), -1});
        else
          it->second[1] = int(f);
      }

    // Flag sharp edges and create projected midpoint vertices.
    ankerl::unordered_dense::map<Edge, uint32_t, EdgeHash> mid;
    for (const auto &[k, faces] : einc) {
      if (faces[1] < 0)
        continue; // boundary edge: no dihedral
      if (fn[faces[0]].dot(fn[faces[1]]) >= cos_thresh)
        continue; // not sharp enough

      FVec3 a = vget(k.first), b = vget(k.second);
      FVec3 m = impl::project_to_isosurface(func, FVec3((a + b) * 0.5f),
                                            isovalue, params.projection_steps);
      const uint32_t idx = V.size() / 3;
      V.insert(V.end(), {m[0], m[1], m[2]});

      // Normal from the projected gradient, sign-aligned to the endpoints.
      FVec3 g = impl::local_gradient(func, m);
      FVec3 nn = (Jit * g).normalized();
      FVec3 na(N[3 * k.first], N[3 * k.first + 1], N[3 * k.first + 2]);
      FVec3 nb(N[3 * k.second], N[3 * k.second + 1], N[3 * k.second + 2]);
      if (nn.dot(na + nb) < 0.0f)
        nn = -nn;
      N.insert(N.end(), {nn[0], nn[1], nn[2]});

      // Curvature is interpolated (a true off-grid Hessian is unavailable).
      C.insert(C.end(), {0.5f * (C[2 * k.first] + C[2 * k.second]),
                         0.5f * (C[2 * k.first + 1] + C[2 * k.second + 1])});

      mid.emplace(k, idx);
    }

    if (mid.empty())
      break; // converged: nothing sharp left

    // Re-triangulate every face by which of its edges were split.
    std::vector<uint32_t> newF;
    newF.reserve(F.size() + 3 * mid.size());
    auto getmid = [&](uint32_t a, uint32_t b) -> int {
      auto it = mid.find(ekey(a, b));
      return it == mid.end() ? -1 : int(it->second);
    };
    auto tri = [&](uint32_t a, uint32_t b, uint32_t c) {
      newF.insert(newF.end(), {a, b, c});
    };
    for (size_t f = 0; f < nf; f++) {
      uint32_t v0 = F[3 * f], v1 = F[3 * f + 1], v2 = F[3 * f + 2];
      int m0 = getmid(v0, v1), m1 = getmid(v1, v2), m2 = getmid(v2, v0);
      const int b = (m0 >= 0 ? 1 : 0) | (m1 >= 0 ? 2 : 0) | (m2 >= 0 ? 4 : 0);
      switch (b) {
      case 0:
        tri(v0, v1, v2);
        break;
      case 1:
        tri(v0, m0, v2);
        tri(m0, v1, v2);
        break;
      case 2:
        tri(v0, v1, m1);
        tri(v0, m1, v2);
        break;
      case 4:
        tri(v0, v1, m2);
        tri(v1, v2, m2);
        break;
      case 3:
        tri(m0, v1, m1);
        tri(v0, m0, m1);
        tri(v0, m1, v2);
        break;
      case 6:
        tri(m1, v2, m2);
        tri(v0, v1, m1);
        tri(v0, m1, m2);
        break;
      case 5:
        tri(v0, m0, m2);
        tri(m0, v1, v2);
        tri(m0, v2, m2);
        break;
      case 7:
        tri(v0, m0, m2);
        tri(m0, v1, m1);
        tri(m2, m1, v2);
        tri(m0, m1, m2);
        break;
      }
    }
    F.swap(newF);
  }
}

} // namespace occ::isosurface
