#pragma once
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cmath>
#include <cstdint>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/geometry/half_edge.h>
#include <occ/isosurface/mesh_utils.h>
#include <occ/isosurface/projection.h>
#include <vector>

namespace occ::isosurface {

struct QualityParams {
  int iterations{0};              // 0 = off
  float feature_angle_deg{30.0f}; // edges/vertices sharper than this are fixed
  float relaxation_factor{0.0f};  // 0 = no relaxation; >0 enables it
  float collapse_ratio{0.0f};     // collapse edges < ratio * median (0 = off)
  int projection_steps{2};        // Newton steps to re-project moved vertices
};

namespace impl {

inline FVec3 vert_at(const std::vector<float> &V, int i) {
  return FVec3(V[3 * i], V[3 * i + 1], V[3 * i + 2]);
}

inline float tri_min_angle(const FVec3 &a, const FVec3 &b, const FVec3 &c) {
  auto angle = [](const FVec3 &p, const FVec3 &q, const FVec3 &r) {
    const FVec3 u = q - p, v = r - p;
    const float nu = u.norm(), nv = v.norm();
    if (nu < 1e-20f || nv < 1e-20f)
      return 0.0f;
    return std::acos(std::clamp(u.dot(v) / (nu * nv), -1.0f, 1.0f));
  };
  return std::min({angle(a, b, c), angle(b, c, a), angle(c, a, b)});
}

inline FVec3 face_normal(const FVec3 &a, const FVec3 &b, const FVec3 &c) {
  return (b - a).cross(c - a).normalized();
}

inline uint64_t edge_key(int a, int b) {
  const uint32_t lo = std::min(a, b), hi = std::max(a, b);
  return (static_cast<uint64_t>(lo) << 32) | hi;
}

// One sweep of feature-preserving, minimum-angle-increasing edge flips. A flip
// is rejected if it would create a < 1 deg triangle or invert a triangle
// (folding a non-convex quad). Topology only; vertices are untouched.
inline int flip_sweep(occ::geometry::HalfEdgeMesh &mesh,
                      const std::vector<float> &V, float feat_cos) {
  const float floor_rad = 1.0f * float(M_PI) / 180.0f;
  int flips = 0;
  for (int e = 0; e < mesh.n_halfedges(); e++) {
    if (mesh.is_boundary(e) || e > mesh.twin(e))
      continue;
    const int t = mesh.twin(e);
    const FVec3 A = vert_at(V, mesh.from(e)), B = vert_at(V, mesh.to(e));
    const FVec3 C = vert_at(V, mesh.to(mesh.next(e)));
    const FVec3 D = vert_at(V, mesh.to(mesh.next(t)));
    if (face_normal(A, B, C).dot(face_normal(B, A, D)) < feat_cos)
      continue; // feature
    const float before = std::min(tri_min_angle(A, B, C), tri_min_angle(B, A, D));
    const float after = std::min(tri_min_angle(A, C, D), tri_min_angle(B, D, C));
    if (after < floor_rad || after <= before + 1e-4f)
      continue;

    // Reject flips that would invert a triangle (non-convex quad). flip()
    // produces triangles wound (d,c,a) and (c,d,b); their normals must agree
    // with the summed normal of the old faces, else the surface folds over.
    const FVec3 navg = (B - A).cross(C - A) + (A - B).cross(D - B);
    if ((C - D).cross(A - D).dot(navg) <= 0.0f ||
        (D - C).cross(B - C).dot(navg) <= 0.0f)
      continue;

    if (mesh.flip(e))
      flips++;
  }
  return flips;
}

// One Jacobi sweep of tangential Laplacian relaxation, re-projected onto
// {f = iso}. Boundary and feature vertices are pinned. Returns moved count.
template <typename Func>
int relax_sweep(const Func &func, float iso,
                const occ::geometry::HalfEdgeMesh &mesh, std::vector<float> &V,
                float feat_cos, float lambda, int proj_steps) {
  const int nv = mesh.n_vertices();
  std::vector<char> pinned(nv, 0);
  for (int e = 0; e < mesh.n_halfedges(); e++) {
    if (mesh.is_boundary(e)) {
      pinned[mesh.from(e)] = pinned[mesh.to(e)] = 1;
      continue;
    }
    if (e > mesh.twin(e))
      continue;
    const int t = mesh.twin(e);
    const FVec3 A = vert_at(V, mesh.from(e)), B = vert_at(V, mesh.to(e));
    const FVec3 C = vert_at(V, mesh.to(mesh.next(e)));
    const FVec3 D = vert_at(V, mesh.to(mesh.next(t)));
    if (face_normal(A, B, C).dot(face_normal(B, A, D)) < feat_cos)
      pinned[mesh.from(e)] = pinned[mesh.to(e)] = 1;
  }

  std::vector<float> Vnew = V;
  std::vector<int> ring;
  int moved = 0;
  for (int v = 0; v < nv; v++) {
    if (pinned[v] || !mesh.vertex_one_ring(v, ring) || ring.size() < 3)
      continue;
    const FVec3 p = vert_at(V, v);
    FVec3 centroid = FVec3::Zero();
    for (int w : ring)
      centroid += vert_at(V, w);
    centroid /= float(ring.size());

    const FVec3 g = local_gradient(func, p);
    const float gn = g.norm();
    if (gn < 1.0e-12f)
      continue;
    const FVec3 n = g / gn;
    FVec3 delta = centroid - p;
    delta -= delta.dot(n) * n;
    FVec3 pnew = project_to_isosurface(func, p + lambda * delta, iso, proj_steps);

    // Reject the move if it would fold any incident triangle. The 1-ring is in
    // circulation order, so consecutive neighbours bound the incident faces.
    bool inverts = false;
    for (size_t i = 0; i < ring.size(); i++) {
      const FVec3 r0 = vert_at(V, ring[i]);
      const FVec3 r1 = vert_at(V, ring[(i + 1) % ring.size()]);
      if ((r0 - p).cross(r1 - p).dot((r0 - pnew).cross(r1 - pnew)) <= 0.0f) {
        inverts = true;
        break;
      }
    }
    if (inverts)
      continue;

    Vnew[3 * v] = pnew[0];
    Vnew[3 * v + 1] = pnew[1];
    Vnew[3 * v + 2] = pnew[2];
    moved++;
  }
  V.swap(Vnew);
  return moved;
}

// Collapse interior edges shorter than ratio * median edge length, removing
// the needle slivers that flips/relaxation cannot. Greedy (shortest first) with
// a lock set so collapses stay independent; the manifold link condition and a
// triangle-inversion test guard each collapse; the merged vertex is projected
// onto {f = iso}. Boundary edges/vertices are left untouched. In place; returns
// the number of collapses.
template <typename Func>
int collapse_short_edges(const Func &func, float iso, std::vector<float> &V,
                         std::vector<float> &N, std::vector<float> &C,
                         std::vector<uint32_t> &F, float ratio) {
  const int nv = int(V.size() / 3), nf = int(F.size() / 3);
  auto vat = [&](int i) { return FVec3(V[3 * i], V[3 * i + 1], V[3 * i + 2]); };

  std::vector<ankerl::unordered_dense::set<int>> nbr(nv);
  std::vector<std::vector<int>> vfaces(nv);
  ankerl::unordered_dense::map<uint64_t, int> efaces;
  for (int f = 0; f < nf; f++) {
    const int v[3] = {int(F[3 * f]), int(F[3 * f + 1]), int(F[3 * f + 2])};
    for (int i = 0; i < 3; i++) {
      vfaces[v[i]].push_back(f);
      nbr[v[i]].insert(v[(i + 1) % 3]);
      nbr[v[i]].insert(v[(i + 2) % 3]);
      efaces[edge_key(v[i], v[(i + 1) % 3])]++;
    }
  }
  std::vector<char> boundary(nv, 0);
  std::vector<float> lens;
  lens.reserve(efaces.size());
  for (auto &[k, count] : efaces) {
    const int a = int(k >> 32), b = int(k & 0xffffffff);
    if (count == 1)
      boundary[a] = boundary[b] = 1;
    lens.push_back((vat(a) - vat(b)).norm());
  }
  if (lens.empty())
    return 0;
  std::sort(lens.begin(), lens.end());
  const float min_len = ratio * lens[lens.size() / 2];

  struct Cand {
    float len;
    int u, v;
  };
  std::vector<Cand> cands;
  for (auto &[k, count] : efaces) {
    if (count != 2)
      continue;
    const int a = int(k >> 32), b = int(k & 0xffffffff);
    if (boundary[a] || boundary[b])
      continue;
    const float L = (vat(a) - vat(b)).norm();
    if (L < min_len)
      cands.push_back({L, a, b});
  }
  std::sort(cands.begin(), cands.end(),
            [](const Cand &x, const Cand &y) { return x.len < y.len; });

  std::vector<char> dead(nv, 0), locked(nv, 0);
  std::vector<uint32_t> vmap(nv);
  for (int i = 0; i < nv; i++)
    vmap[i] = i;

  int collapsed = 0;
  for (const auto &cd : cands) {
    const int u = cd.u, v = cd.v;
    if (locked[u] || locked[v])
      continue;

    // Opposite apexes of the two faces on edge (u, v).
    int apex[2], na = 0;
    bool bad = false;
    for (int f : vfaces[u]) {
      const int a = int(F[3 * f]), b = int(F[3 * f + 1]), c = int(F[3 * f + 2]);
      if ((a == v) || (b == v) || (c == v)) {
        const int w = (a != u && a != v) ? a : ((b != u && b != v) ? b : c);
        if (na < 2)
          apex[na] = w;
        na++;
      }
    }
    if (na != 2)
      continue;

    // Link condition: u and v share exactly the two apexes.
    int shared = 0;
    for (int w : nbr[u]) {
      if (w == u || w == v)
        continue;
      if (nbr[v].contains(w)) {
        shared++;
        if (w != apex[0] && w != apex[1])
          bad = true;
      }
    }
    if (bad || shared != 2)
      continue;

    const FVec3 newpos = project_to_isosurface(
        func, FVec3((vat(u) + vat(v)) * 0.5f), iso, 3);

    // Reject if any surviving incident triangle would invert or degenerate.
    auto inverts = [&](int vert) {
      for (int f : vfaces[vert]) {
        const int a = int(F[3 * f]), b = int(F[3 * f + 1]),
                  c = int(F[3 * f + 2]);
        const bool hu = (a == u || b == u || c == u);
        const bool hv = (a == v || b == v || c == v);
        if (hu && hv)
          continue; // collapsed face
        auto P = [&](int x) { return (x == u || x == v) ? newpos : vat(x); };
        const FVec3 no = (vat(b) - vat(a)).cross(vat(c) - vat(a));
        const FVec3 nn = (P(b) - P(a)).cross(P(c) - P(a));
        if (nn.dot(no) <= 0.0f || nn.norm() < 1e-12f)
          return true;
      }
      return false;
    };
    if (inverts(u) || inverts(v))
      continue;

    V[3 * u] = newpos[0];
    V[3 * u + 1] = newpos[1];
    V[3 * u + 2] = newpos[2];
    vmap[v] = u;
    dead[v] = 1;
    locked[u] = locked[v] = 1;
    for (int w : nbr[u])
      locked[w] = 1;
    for (int w : nbr[v])
      locked[w] = 1;
    collapsed++;
  }
  if (collapsed == 0)
    return 0;

  std::vector<uint32_t> Fn;
  Fn.reserve(F.size());
  for (int f = 0; f < nf; f++) {
    const uint32_t a = vmap[F[3 * f]], b = vmap[F[3 * f + 1]],
                   c = vmap[F[3 * f + 2]];
    if (a != b && b != c && a != c)
      Fn.insert(Fn.end(), {a, b, c});
  }
  std::vector<int> remap(nv, -1);
  std::vector<float> Vn, Nn, Cn;
  int cnt = 0;
  for (int i = 0; i < nv; i++) {
    if (dead[i])
      continue;
    remap[i] = cnt++;
    Vn.insert(Vn.end(), {V[3 * i], V[3 * i + 1], V[3 * i + 2]});
    Nn.insert(Nn.end(), {N[3 * i], N[3 * i + 1], N[3 * i + 2]});
    Cn.insert(Cn.end(), {C[2 * i], C[2 * i + 1]});
  }
  for (auto &idx : Fn)
    idx = remap[idx];
  V.swap(Vn);
  N.swap(Nn);
  C.swap(Cn);
  F.swap(Fn);
  return collapsed;
}

} // namespace impl

struct AngleStats {
  float min_deg{0};
  float p05_deg{0};
  float frac_below_30{0};
  int nfaces{0};
};

inline AngleStats triangle_angle_stats(const std::vector<float> &V,
                                       const std::vector<uint32_t> &F) {
  std::vector<float> mins;
  mins.reserve(F.size() / 3);
  for (size_t f = 0; f < F.size() / 3; f++)
    mins.push_back(impl::tri_min_angle(impl::vert_at(V, F[3 * f]),
                                       impl::vert_at(V, F[3 * f + 1]),
                                       impl::vert_at(V, F[3 * f + 2])) *
                   180.0f / float(M_PI));
  AngleStats s;
  s.nfaces = int(mins.size());
  if (mins.empty())
    return s;
  std::sort(mins.begin(), mins.end());
  s.min_deg = mins.front();
  s.p05_deg = mins[size_t(0.05 * mins.size())];
  int below = 0;
  for (float m : mins)
    if (m < 30.0f)
      below++;
  s.frac_below_30 = float(below) / mins.size();
  return s;
}

// Improve triangle quality: feature-preserving valence/min-angle edge flips,
// optional short-edge collapse (removes needle slivers), and optional
// tangential relaxation re-projected onto {f = iso}. Welds first.
template <typename Func>
void improve_mesh_quality(const Func &func, float isovalue,
                          const QualityParams &params, std::vector<float> &V,
                          std::vector<float> &N, std::vector<float> &C,
                          std::vector<uint32_t> &F) {
  if (params.iterations <= 0)
    return;
  weld_vertices(V, N, C, F);
  const float feat_cos =
      std::cos(double(params.feature_angle_deg) * M_PI / 180.0);

  int total_flips = 0, total_moved = 0, total_collapsed = 0;
  for (int it = 0; it < params.iterations; it++) {
    int collapsed = 0;
    if (params.collapse_ratio > 0.0f)
      collapsed = impl::collapse_short_edges(func, isovalue, V, N, C, F,
                                             params.collapse_ratio);

    occ::geometry::HalfEdgeMesh mesh(int(V.size() / 3), F);
    const int flips = impl::flip_sweep(mesh, V, feat_cos);
    int moved = 0;
    if (params.relaxation_factor > 0.0f)
      moved = impl::relax_sweep(func, isovalue, mesh, V, feat_cos,
                                params.relaxation_factor, params.projection_steps);
    mesh.export_faces(F);

    total_flips += flips;
    total_moved += moved;
    total_collapsed += collapsed;
    if (collapsed == 0 && flips == 0 && moved == 0)
      break;
  }

  occ::log::debug("Quality pass: {} flips, {} collapses, {} moves -> {} faces",
                  total_flips, total_collapsed, total_moved, F.size() / 3);
}

} // namespace occ::isosurface
