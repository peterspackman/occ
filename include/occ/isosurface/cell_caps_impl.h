#pragma once
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <occ/core/log.h>
#include <occ/core/parallel.h>

namespace occ::isosurface {

namespace cell_caps_impl {

// Number of cell faces a fractional point lies on (within tol).
inline int faces_at(const FVec3 &frac, float tol, std::array<int, 3> &out) {
  int n = 0;
  for (int a = 0; a < 3; a++) {
    if (frac(a) <= tol) {
      out[n++] = a * 2 + 0;
    } else if (frac(a) >= 1.0f - tol) {
      out[n++] = a * 2 + 1;
    }
  }
  return n;
}

// Linear interpolation factor for the iso crossing on a cell edge between
// values va (at u=0) and vb (at u=1). Both must be on opposite sides of iso.
inline float iso_offset(float va, float vb, float iso) {
  float d = vb - va;
  if (d == 0.0f)
    return 0.5f;
  return (iso - va) / d;
}

// 2D marching-squares cap polygon for one face cell, walking the cell
// boundary CCW (c00 → c10 → c11 → c01) and emitting void corners and iso
// crossings on edges between corners of differing classification. For the
// saddle cases (5 = c00+c11, 10 = c10+c01) this would produce a self-
// intersecting polygon; we handle them explicitly as two disjoint polygons.
//
// Output is a list of polygons (each a list of vertex labels). Vertex labels
// 0..3 are corners c00, c10, c11, c01; labels 4..7 are edge midpoints on
// edges b (c00-c10), r (c10-c11), t (c11-c01), l (c01-c00).
inline void
build_void_polys(int case_code,
                 std::vector<std::array<uint8_t, 6>> &out_polys_with_size) {
  out_polys_with_size.clear();
  if (case_code == 0)
    return;

  auto add = [&](std::initializer_list<uint8_t> verts) {
    std::array<uint8_t, 6> p{};
    p[0] = static_cast<uint8_t>(verts.size());
    int i = 1;
    for (auto v : verts)
      p[i++] = v;
    out_polys_with_size.push_back(p);
  };

  switch (case_code) {
  case 0b0001: // c00 only
    add({0, 4, 7});
    return;
  case 0b0010: // c10 only
    add({1, 5, 4});
    return;
  case 0b0100: // c11 only
    add({2, 6, 5});
    return;
  case 0b1000: // c01 only
    add({3, 7, 6});
    return;
  case 0b0011: // c00 + c10
    add({0, 1, 5, 7});
    return;
  case 0b0110: // c10 + c11
    add({1, 2, 6, 4});
    return;
  case 0b1100: // c11 + c01
    add({2, 3, 7, 5});
    return;
  case 0b1001: // c01 + c00
    add({3, 0, 4, 6});
    return;
  case 0b0101: // c00 + c11 (saddle): two disjoint corner polys
    add({0, 4, 7});
    add({2, 6, 5});
    return;
  case 0b1010: // c10 + c01 (saddle): two disjoint corner polys
    add({1, 5, 4});
    add({3, 7, 6});
    return;
  case 0b0111: // c00 + c10 + c11
    add({0, 1, 2, 6, 7});
    return;
  case 0b1011: // c00 + c10 + c01
    add({0, 1, 5, 6, 3});
    return;
  case 0b1101: // c00 + c11 + c01
    add({0, 4, 5, 2, 3});
    return;
  case 0b1110: // c10 + c11 + c01
    add({1, 2, 3, 7, 4});
    return;
  case 0b1111: // all void
    add({0, 1, 2, 3});
    return;
  default:
    return;
  }
}

// Pack a (kind, i, j) tuple into an int64 hash key for the per-face vertex
// dedup map.
inline int64_t pack_vkey(int kind, int i, int j) {
  return (int64_t(kind) << 40) | (int64_t(i) << 20) | int64_t(j);
}

} // namespace cell_caps_impl

template <typename F>
CellCapClassification
add_cell_caps(const F &functor, float iso,
              std::vector<float> &vertices_frac,
              std::vector<float> &normals, std::vector<float> &curvatures,
              std::vector<uint32_t> &indices, float on_face_tol) {
  using namespace cell_caps_impl;

  const size_t num_faces_natural = indices.size() / 3;

  Eigen::Vector3i samples = functor.cubes_per_side();
  // basis_transform returns J^-T = (M^-1)^T whose columns are the reciprocal
  // lattice vectors. Column `axis` is perpendicular to the (a, b)-style face
  // for that axis, so it's the cartesian normal direction (up to sign and
  // normalization).
  FMat3 J_inv_T = functor.basis_transform();

  size_t cap_tris_added = 0;
  size_t cap_verts_added = 0;

  for (int axis = 0; axis < 3; axis++) {
    int u_axis = (axis + 1) % 3;
    int v_axis = (axis + 2) % 3;
    int Nu = samples(u_axis);
    int Nv = samples(v_axis);

    if (Nu < 2 || Nv < 2)
      continue;

    for (int side = 0; side < 2; side++) {
      float face_frac = float(side);

      // Outward normal (cartesian) for this face. CCW polygon winding in the
      // (u_axis, v_axis) plane gives a +axis frac normal; for side=1 that's
      // outward, for side=0 we need the opposite winding so we flip below.
      FVec3 n_cart = J_inv_T.col(axis).normalized();
      if (side == 0)
        n_cart = -n_cart;

      // 1) Sample density on the face's Nu x Nv grid.
      FMat3N face_pts(3, Nu * Nv);
      for (int j = 0; j < Nv; j++) {
        for (int i = 0; i < Nu; i++) {
          int idx = j * Nu + i;
          face_pts(axis, idx) = face_frac;
          face_pts(u_axis, idx) = float(i) / float(Nu - 1);
          face_pts(v_axis, idx) = float(j) / float(Nv - 1);
        }
      }
      FVec face_vals(Nu * Nv);
      functor.batch(face_pts, face_vals);

      // 2) Per-face vertex dedup map: (kind, i, j) -> global vertex index.
      // kind 0 = corner at (i, j)
      // kind 1 = h-edge midpoint between (i, j) and (i+1, j)
      // kind 2 = v-edge midpoint between (i, j) and (i, j+1)
      ankerl::unordered_dense::map<int64_t, uint32_t> vmap;

      auto add_vertex = [&](float fu, float fv) -> uint32_t {
        FVec3 frac;
        frac(axis) = face_frac;
        frac(u_axis) = fu;
        frac(v_axis) = fv;
        uint32_t id = static_cast<uint32_t>(vertices_frac.size() / 3);
        vertices_frac.push_back(frac(0));
        vertices_frac.push_back(frac(1));
        vertices_frac.push_back(frac(2));
        normals.push_back(n_cart(0));
        normals.push_back(n_cart(1));
        normals.push_back(n_cart(2));
        curvatures.push_back(0.0f);
        curvatures.push_back(0.0f);
        cap_verts_added++;
        return id;
      };

      auto get_corner = [&](int i, int j) -> uint32_t {
        int64_t key = pack_vkey(0, i, j);
        auto it = vmap.find(key);
        if (it != vmap.end())
          return it->second;
        float fu = float(i) / float(Nu - 1);
        float fv = float(j) / float(Nv - 1);
        uint32_t id = add_vertex(fu, fv);
        vmap.emplace(key, id);
        return id;
      };

      auto get_h_edge = [&](int i, int j) -> uint32_t {
        // edge between corner (i, j) and (i+1, j)
        int64_t key = pack_vkey(1, i, j);
        auto it = vmap.find(key);
        if (it != vmap.end())
          return it->second;
        float va = face_vals(j * Nu + i);
        float vb = face_vals(j * Nu + (i + 1));
        float t = iso_offset(va, vb, iso);
        float fu = (float(i) + t) / float(Nu - 1);
        float fv = float(j) / float(Nv - 1);
        uint32_t id = add_vertex(fu, fv);
        vmap.emplace(key, id);
        return id;
      };

      auto get_v_edge = [&](int i, int j) -> uint32_t {
        // edge between corner (i, j) and (i, j+1)
        int64_t key = pack_vkey(2, i, j);
        auto it = vmap.find(key);
        if (it != vmap.end())
          return it->second;
        float va = face_vals(j * Nu + i);
        float vb = face_vals((j + 1) * Nu + i);
        float t = iso_offset(va, vb, iso);
        float fu = float(i) / float(Nu - 1);
        float fv = (float(j) + t) / float(Nv - 1);
        uint32_t id = add_vertex(fu, fv);
        vmap.emplace(key, id);
        return id;
      };

      // 3) For each (Nu-1) x (Nv-1) face cell, classify corners and emit
      //    cap triangles for the void portion.
      std::vector<std::array<uint8_t, 6>> polys;
      for (int j = 0; j < Nv - 1; j++) {
        for (int i = 0; i < Nu - 1; i++) {
          float v00 = face_vals(j * Nu + i);
          float v10 = face_vals(j * Nu + (i + 1));
          float v11 = face_vals((j + 1) * Nu + (i + 1));
          float v01 = face_vals((j + 1) * Nu + i);
          int code = 0;
          if (v00 < iso)
            code |= 0b0001;
          if (v10 < iso)
            code |= 0b0010;
          if (v11 < iso)
            code |= 0b0100;
          if (v01 < iso)
            code |= 0b1000;
          if (code == 0)
            continue;

          build_void_polys(code, polys);
          if (polys.empty())
            continue;

          // Map polygon labels (0..7) to actual vertex IDs in our mesh.
          auto label_id = [&](uint8_t lab) -> uint32_t {
            switch (lab) {
            case 0:
              return get_corner(i, j);
            case 1:
              return get_corner(i + 1, j);
            case 2:
              return get_corner(i + 1, j + 1);
            case 3:
              return get_corner(i, j + 1);
            case 4:
              return get_h_edge(i, j);
            case 5:
              return get_v_edge(i + 1, j);
            case 6:
              return get_h_edge(i, j + 1);
            case 7:
              return get_v_edge(i, j);
            }
            return 0;
          };

          for (const auto &p : polys) {
            uint8_t n = p[0];
            // Fan from p[1].
            uint32_t v0 = label_id(p[1]);
            for (int k = 2; k < n; k++) {
              uint32_t va = label_id(p[k]);
              uint32_t vb = label_id(p[k + 1]);
              if (side == 0) {
                // Side-0 face's outward normal is -axis; reverse winding.
                indices.push_back(v0);
                indices.push_back(vb);
                indices.push_back(va);
              } else {
                indices.push_back(v0);
                indices.push_back(va);
                indices.push_back(vb);
              }
              cap_tris_added++;
            }
          }
        }
      }
    }
  }

  occ::log::info("Void caps: +{} vertices, +{} cap triangles", cap_verts_added,
                 cap_tris_added);

  // Build classifications.
  CellCapClassification result;
  size_t total_verts = vertices_frac.size() / 3;
  size_t total_faces = indices.size() / 3;
  result.vertex_class = IVec::Zero(total_verts);
  result.face_class = IVec::Zero(total_faces);

  for (size_t v = 0; v < total_verts; v++) {
    FVec3 frac(vertices_frac[3 * v], vertices_frac[3 * v + 1],
               vertices_frac[3 * v + 2]);
    std::array<int, 3> faces_idx{};
    int n = faces_at(frac, on_face_tol, faces_idx);
    if (n == 0)
      result.vertex_class(v) = 0;
    else if (n == 1)
      result.vertex_class(v) = 2;
    else
      result.vertex_class(v) = 1;
  }
  for (size_t t = num_faces_natural; t < total_faces; t++)
    result.face_class(t) = 1;

  return result;
}

} // namespace occ::isosurface
