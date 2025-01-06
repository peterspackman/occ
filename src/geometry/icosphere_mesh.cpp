#include <fmt/core.h>
#include <iostream>
#include <occ/core/log.h>
#include <occ/geometry/icosphere_mesh.h>

namespace occ::geometry {

std::pair<size_t, size_t> IcosphereMesh::compute_sizes(size_t level) {
  size_t v = 12;
  size_t f = 20;
  for (size_t i = 1; i <= level; i++) {
    int next_f = f * 4;
    int next_e = f * 3 / 2 * 2;
    v += next_e / 2;
    f = next_f;
  }
  return {v, f};
}

int IcosphereMesh::add_vertex(const Vec3 &p) {
  m_vertices.col(m_vertex_index) = p.normalized();
  return m_vertex_index++;
}

int IcosphereMesh::add_midpoint(size_t p1, size_t p2) {
  int64_t l = std::min(p1, p2);
  int64_t u = std::max(p1, p2);
  // assume that we won't have more than 2^32 points...
  int64_t key = (l << 32) + u;

  auto it = midpoint_cache.find(key);
  if (it != midpoint_cache.end()) {
    return it->second;
  }

  Vec3 point1 = m_vertices.col(p1);
  Vec3 point2 = m_vertices.col(p2);
  Vec3 middle = (point1 + point2).normalized();

  int i = add_vertex(middle);
  midpoint_cache[key] = i;
  return i;
}

Mat3N IcosphereMesh::initial_vertices() {
  const double t = (1.0 + std::sqrt(5.0)) / 2.0;
  Mat3N v(3, 12);
  // clang-format off
  v << -1,  1, -1,  1,  0,  0,  0,  0,  t,  t, -t, -t,
        t,  t, -t, -t, -1,  1, -1,  1,  0,  0,  0,  0,
        0,  0,  0,  0,  t,  t, -t, -t, -1,  1, -1,  1;
  // clang-format on
  v.colwise().normalize();
  return v;
}

IMat3N IcosphereMesh::initial_faces() {
  IMat3N f(3, 20);
  // clang-format off
  f <<  0, 0, 0,  0,  0, 1,  5, 11, 10, 7, 3, 3, 3, 3, 3, 4,  2,  6, 8, 9,
       11, 5, 1,  7, 10, 5, 11, 10,  7, 1, 9, 4, 2, 6, 8, 9,  4,  2, 6, 8,
        5, 1, 7, 10, 11, 9,  4,  2,  6, 8, 4, 2, 6, 8, 9, 5, 11, 10, 7, 1;
  // clang-format on
  return f;
}

IcosphereMesh::IcosphereMesh(size_t subdivisions) {
  auto [nverts, nfaces] = IcosphereMesh::compute_sizes(subdivisions);
  m_vertices.resize(3, nverts);
  midpoint_cache.clear();
  m_vertex_index = 0;

  m_vertices.leftCols(12) = IcosphereMesh::initial_vertices();
  m_vertex_index = 12;
  m_faces = IcosphereMesh::initial_faces();

  // Refine triangles
  for (size_t i = 0; i < subdivisions; i++) {
    IMat3N new_faces(3, m_faces.cols() * 4);
    size_t new_face_index = 0;
    for (size_t f = 0; f < m_faces.cols(); f++) {
      IVec3 tri = m_faces.col(f);
      int a = tri[0], b = tri[1], c = tri[2];
      int ab = add_midpoint(a, b);
      int bc = add_midpoint(b, c);
      int ca = add_midpoint(c, a);
      new_faces.col(new_face_index++) = IVec3{a, ab, ca};
      new_faces.col(new_face_index++) << IVec3{b, bc, ab};
      new_faces.col(new_face_index++) << IVec3{c, ca, bc};
      new_faces.col(new_face_index++) << IVec3{ab, bc, ca};
    }
    m_faces = new_faces;
  }
  occ::log::debug("IcosphereMesh final vertex count {}", m_vertices.cols());
  occ::log::debug("IcosphereMesh final face count   {}", m_faces.cols());
}

} // namespace occ::geometry
