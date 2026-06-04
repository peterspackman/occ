#pragma once
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace occ::geometry {

// Compact, editable half-edge triangle mesh storing connectivity only. Vertex
// positions and per-vertex attributes are kept by the caller and indexed by
// vertex id; the operations here (currently edge flip) preserve vertex ids, so
// those external arrays stay valid. Boundary half-edges have twin() == NONE.
//
// Convention: each face owns three consecutive half-edges (3f, 3f+1, 3f+2);
// half-edge h goes from from(h) to to(h), next(h) walks its face CCW, and
// prev(h) == next(next(h)) for triangles.
class HalfEdgeMesh {
public:
  static constexpr int NONE = -1;

  HalfEdgeMesh(int nverts, const std::vector<uint32_t> &faces) {
    const int nf = static_cast<int>(faces.size() / 3);
    m_to.resize(3 * nf);
    m_twin.assign(3 * nf, NONE);
    m_next.resize(3 * nf);
    m_face.resize(3 * nf);
    m_face_half.resize(nf);
    m_vert_half.assign(nverts, NONE);

    std::unordered_map<int64_t, int> emap;
    emap.reserve(3 * nf * 2);
    auto ekey = [nverts](int a, int b) {
      return static_cast<int64_t>(a) * static_cast<int64_t>(nverts) + b;
    };

    for (int f = 0; f < nf; f++) {
      const int v[3] = {int(faces[3 * f]), int(faces[3 * f + 1]),
                        int(faces[3 * f + 2])};
      for (int i = 0; i < 3; i++) {
        const int h = 3 * f + i, from = v[i], to = v[(i + 1) % 3];
        m_to[h] = to;
        m_face[h] = f;
        m_next[h] = 3 * f + (i + 1) % 3;
        if (m_vert_half[from] == NONE)
          m_vert_half[from] = h;
        emap[ekey(from, to)] = h;
      }
      m_face_half[f] = 3 * f;
    }

    for (int h = 0; h < 3 * nf; h++) {
      const int to = m_to[h], from = m_to[prev(h)];
      auto it = emap.find(ekey(to, from));
      if (it != emap.end())
        m_twin[h] = it->second;
      m_edges.insert(ukey(from, to));
    }
  }

  int n_halfedges() const { return static_cast<int>(m_to.size()); }
  int n_faces() const { return static_cast<int>(m_face_half.size()); }
  int n_vertices() const { return static_cast<int>(m_vert_half.size()); }
  int to(int h) const { return m_to[h]; }
  int twin(int h) const { return m_twin[h]; }
  int next(int h) const { return m_next[h]; }
  int prev(int h) const { return m_next[m_next[h]]; }
  int from(int h) const { return m_to[prev(h)]; }
  int face(int h) const { return m_face[h]; }
  int face_halfedge(int f) const { return m_face_half[f]; }
  bool is_boundary(int h) const { return m_twin[h] == NONE; }
  bool edge_exists(int u, int v) const { return m_edges.contains(ukey(u, v)); }
  int vertex_halfedge(int v) const { return m_vert_half[v]; }

  // Collect the 1-ring neighbours of vertex v into nbrs. Returns false (and a
  // partial ring) if v is on the boundary, so callers can pin such vertices.
  bool vertex_one_ring(int v, std::vector<int> &nbrs) const {
    nbrs.clear();
    const int h0 = m_vert_half[v];
    if (h0 == NONE)
      return false;
    int h = h0;
    do {
      nbrs.push_back(m_to[h]);
      if (m_twin[h] == NONE)
        return false; // hit a boundary
      h = m_next[m_twin[h]];
    } while (h != h0);
    return true;
  }

  // Flip the shared diagonal of the two triangles adjacent to interior edge e.
  // No-op (returns false) on a boundary edge or when the would-be diagonal
  // already exists (which would make the mesh non-manifold).
  bool flip(int e) {
    if (m_twin[e] == NONE)
      return false;
    const int t = m_twin[e];
    const int eN = m_next[e], eNN = m_next[eN];
    const int tN = m_next[t], tNN = m_next[tN];
    const int a = m_to[eNN]; // from(e)
    const int b = m_to[e];
    const int c = m_to[eN]; // apex of e's face
    const int d = m_to[tN]; // apex of t's face
    if (c == d || m_edges.contains(ukey(c, d)))
      return false;
    const int f0 = m_face[e], f1 = m_face[t];

    m_to[e] = c;
    m_to[t] = d;
    m_next[e] = eNN;
    m_next[eNN] = tN;
    m_next[tN] = e;
    m_next[t] = tNN;
    m_next[tNN] = eN;
    m_next[eN] = t;
    m_face[e] = m_face[eNN] = m_face[tN] = f0;
    m_face[t] = m_face[tNN] = m_face[eN] = f1;
    m_face_half[f0] = e;
    m_face_half[f1] = t;
    m_vert_half[a] = tN;
    m_vert_half[b] = eN;
    m_vert_half[c] = t;
    m_vert_half[d] = e;
    m_edges.erase(ukey(a, b));
    m_edges.insert(ukey(c, d));
    return true;
  }

  void export_faces(std::vector<uint32_t> &out) const {
    out.clear();
    out.reserve(3 * n_faces());
    for (int f = 0; f < n_faces(); f++) {
      const int h0 = m_face_half[f], h1 = m_next[h0], h2 = m_next[h1];
      out.push_back(m_to[h0]);
      out.push_back(m_to[h1]);
      out.push_back(m_to[h2]);
    }
  }

  // Topological sanity check (used by tests): twins are mutual, faces are
  // 3-cycles, and all of a face's half-edges agree on the face id.
  bool check_valid() const {
    for (int h = 0; h < n_halfedges(); h++) {
      if (m_twin[h] != NONE && m_twin[m_twin[h]] != h)
        return false;
      if (m_next[m_next[m_next[h]]] != h)
        return false;
      if (m_face[h] != m_face[m_next[h]])
        return false;
    }
    return true;
  }

private:
  static uint64_t ukey(int a, int b) {
    const uint32_t lo = std::min(a, b), hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | hi;
  }

  std::vector<int> m_to, m_twin, m_next, m_face, m_face_half, m_vert_half;
  ankerl::unordered_dense::set<uint64_t> m_edges;
};

} // namespace occ::geometry
