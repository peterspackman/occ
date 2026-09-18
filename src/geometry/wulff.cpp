#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <map>
#include <numeric>
#include <occ/core/log.h>
#include <occ/geometry/wulff.h>
#include <set>
#include <utility>

namespace occ::geometry {

Mat3N project_to_plane(const Mat3N &points, const Vec3 &plane_normal) {
  Mat3N projected_points =
      points.array() -
      (plane_normal * (plane_normal.transpose() * points)).array();

  Vec3 a_vector = projected_points.col(1) - projected_points.col(0);
  Vec3 b_vector = plane_normal.cross(a_vector);

  Vec u = projected_points.transpose() * a_vector;
  Vec v = projected_points.transpose() * b_vector;

  Mat3N result = Mat3N::Zero(3, points.cols());
  result.row(0) = u.transpose();
  result.row(1) = v.transpose();

  return result;
}

namespace {

// Column order of coplanar points by angle about their centroid, measured in
// an in-plane basis built from the normal.
std::vector<size_t> angular_order(const Mat3N &points, const Vec3 &normal) {
  const Vec3 u = normal.unitOrthogonal();
  const Vec3 v = normal.cross(u);
  const Vec3 centroid = points.rowwise().mean();
  std::vector<double> angles(points.cols());
  for (int i = 0; i < points.cols(); ++i) {
    const Vec3 d = points.col(i) - centroid;
    angles[i] = std::atan2(v.dot(d), u.dot(d));
  }
  std::vector<size_t> order(points.cols());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&angles](size_t a, size_t b) { return angles[a] < angles[b]; });
  return order;
}

// A convex polyhedron: each face is a cyclically ordered loop of vertex
// indices, labelled with the facet whose plane it lies in (-1 for the faces of
// the initial bounding box).
struct Polyhedron {
  struct Face {
    int facet{-1};
    std::vector<int> loop;
  };
  std::vector<Vec3> vertices;
  std::vector<Face> faces;
};

Polyhedron bounding_box(double half_width) {
  Polyhedron box;
  for (int i = 0; i < 8; ++i)
    box.vertices.push_back(half_width * Vec3((i & 1) ? 1.0 : -1.0,
                                             (i & 2) ? 1.0 : -1.0,
                                             (i & 4) ? 1.0 : -1.0));
  box.faces = {{-1, {0, 2, 6, 4}}, {-1, {1, 5, 7, 3}},  // x = -w, +w
               {-1, {0, 4, 5, 1}}, {-1, {2, 3, 7, 6}},  // y = -w, +w
               {-1, {0, 1, 3, 2}}, {-1, {4, 6, 7, 5}}}; // z = -w, +w
  return box;
}

// Drop vertices no face refers to.
void compact(Polyhedron &poly) {
  std::vector<int> remap(poly.vertices.size(), -1);
  std::vector<Vec3> kept;
  for (auto &face : poly.faces) {
    for (int &k : face.loop) {
      if (remap[k] < 0) {
        remap[k] = kept.size();
        kept.push_back(poly.vertices[k]);
      }
      k = remap[k];
    }
  }
  poly.vertices = std::move(kept);
}

// Keep the part of `poly` with n.x <= energy; the new face is labelled
// `facet`. Each vertex is classified once and each crossed edge is split once,
// so the result is a valid polyhedron however close the plane comes to
// existing vertices or faces.
void clip(Polyhedron &poly, const Vec3 &n, double energy, int facet) {
  double scale = 0.0;
  for (const auto &v : poly.vertices)
    scale = std::max(scale, v.norm());
  const double tolerance = 1e-12 * scale; // rounding, nothing more

  std::vector<double> distance(poly.vertices.size());
  bool cuts = false;
  for (size_t k = 0; k < poly.vertices.size(); ++k) {
    distance[k] = n.dot(poly.vertices[k]) - energy;
    cuts = cuts || distance[k] > tolerance;
  }
  if (!cuts)
    return;

  std::map<std::pair<int, int>, int> edge_points;
  auto edge_point = [&](int i, int j) {
    const std::pair<int, int> key{std::min(i, j), std::max(i, j)};
    if (auto it = edge_points.find(key); it != edge_points.end())
      return it->second;
    const double t = distance[i] / (distance[i] - distance[j]);
    poly.vertices.push_back(poly.vertices[i] +
                            t * (poly.vertices[j] - poly.vertices[i]));
    distance.push_back(0.0);
    const int index = static_cast<int>(poly.vertices.size()) - 1;
    edge_points.emplace(key, index);
    return index;
  };

  std::vector<int> cap;
  auto add_to_cap = [&cap](int k) {
    if (std::find(cap.begin(), cap.end(), k) == cap.end())
      cap.push_back(k);
  };

  std::vector<Polyhedron::Face> faces;
  for (const auto &face : poly.faces) {
    std::vector<int> loop;
    const size_t m = face.loop.size();
    for (size_t a = 0; a < m; ++a) {
      const int i = face.loop[a];
      const int j = face.loop[(a + 1) % m];
      const double di = distance[i], dj = distance[j];
      if (di <= tolerance) {
        loop.push_back(i);
        if (di >= -tolerance)
          add_to_cap(i);
      }
      if ((di < -tolerance && dj > tolerance) ||
          (di > tolerance && dj < -tolerance)) {
        const int p = edge_point(i, j);
        loop.push_back(p);
        add_to_cap(p);
      }
    }
    if (loop.size() >= 3)
      faces.push_back({face.facet, std::move(loop)});
  }

  if (cap.size() >= 3) {
    Mat3N points(3, cap.size());
    for (size_t k = 0; k < cap.size(); ++k)
      points.col(k) = poly.vertices[cap[k]];
    std::vector<int> loop(cap.size());
    const auto order = angular_order(points, n);
    for (size_t k = 0; k < order.size(); ++k)
      loop[k] = cap[order[k]];
    faces.push_back({facet, std::move(loop)});
  }
  poly.faces = std::move(faces);
  compact(poly);
}

// Merge vertices within `tolerance` of each other. Two planes that nearly
// coincide cut a sliver face between them; merging collapses it, and a face
// left with fewer than three distinct vertices is dropped.
void merge_close_vertices(Polyhedron &poly, double tolerance) {
  std::vector<int> remap(poly.vertices.size());
  std::vector<Vec3> kept;
  for (size_t k = 0; k < poly.vertices.size(); ++k) {
    int found = -1;
    for (size_t m = 0; m < kept.size(); ++m) {
      if ((kept[m] - poly.vertices[k]).norm() <= tolerance) {
        found = static_cast<int>(m);
        break;
      }
    }
    if (found < 0) {
      found = static_cast<int>(kept.size());
      kept.push_back(poly.vertices[k]);
    }
    remap[k] = found;
  }
  poly.vertices = std::move(kept);

  std::vector<Polyhedron::Face> faces;
  for (const auto &face : poly.faces) {
    std::vector<int> loop;
    for (int k : face.loop) {
      if (loop.empty() || loop.back() != remap[k])
        loop.push_back(remap[k]);
    }
    while (loop.size() > 1 && loop.front() == loop.back())
      loop.pop_back();
    if (loop.size() >= 3)
      faces.push_back({face.facet, std::move(loop)});
  }
  poly.faces = std::move(faces);
  compact(poly);
}

} // namespace

void Facet::reorder(const Mat3N &points) {
  if (point_index.size() < 3)
    return;
  const auto order = angular_order(points, normal);
  std::vector<int> sorted_point_index(point_index.size());
  for (size_t i = 0; i < order.size(); ++i)
    sorted_point_index[i] = point_index[order[i]];
  point_index = std::move(sorted_point_index);
}

void Facet::reorder_and_triangulate(const Mat3N &all_points) {
  if (point_index.empty())
    return;

  // assumes we have at least 3 points
  const size_t N = point_index.size();
  Mat3N points = all_points(Eigen::all, point_index);

  reorder(points);

  this->triangles = IMat3N(3, N - 2);

  this->triangles.row(0).array() = point_index[0];
  this->triangles.row(1) =
      Eigen::Map<const IVec>(point_index.data() + 1, N - 2);
  this->triangles.row(2) =
      Eigen::Map<const IVec>(point_index.data() + 2, N - 2);
}

WulffConstruction::WulffConstruction(
    const Mat3N &facet_normals, const Vec &facet_energies,
    const std::vector<std::string> &facet_labels) {

  const size_t N = facet_energies.rows();
  for (size_t i = 0; i < N; i++) {
    const double energy = facet_energies(i);
    // dual = p / (|p|^2), since we haven't scaled p just divide by energy
    Vec3 dual = facet_normals.col(i).array() / energy;
    m_facets.push_back(Facet{energy, facet_normals.col(i),
                             (facet_labels.size() > i)
                                 ? facet_labels[i]
                                 : fmt::format("facet_{}", i),
                             dual});
  }
  build_polyhedron();
}

// The Wulff shape is the intersection of the half-spaces n.x <= gamma, built
// by clipping a bounding box with each facet plane in turn.
void WulffConstruction::build_polyhedron() {
  double max_energy = 0.0;
  for (const auto &facet : m_facets)
    max_energy = std::max(max_energy, facet.energy);

  Polyhedron poly;
  if (max_energy > 0.0) {
    poly = bounding_box(1e6 * max_energy);
    int non_positive = 0;
    for (size_t f = 0; f < m_facets.size(); f++) {
      if (m_facets[f].energy <= 0.0) {
        non_positive++;
        continue;
      }
      clip(poly, m_facets[f].normal, m_facets[f].energy, static_cast<int>(f));
    }
    if (non_positive > 0)
      occ::log::warn("Wulff construction: ignored {} facets with non-positive "
                     "energy",
                     non_positive);
    merge_close_vertices(poly, 1e-6 * max_energy);
  }

  const bool bounded =
      !poly.faces.empty() &&
      std::none_of(poly.faces.begin(), poly.faces.end(),
                   [](const Polyhedron::Face &face) { return face.facet < 0; });
  if (!bounded) {
    occ::log::warn("Wulff construction: the facets do not enclose a bounded "
                   "shape; returning an empty one");
    poly = Polyhedron{};
  }

  m_wulff_vertices = Mat3N(3, poly.vertices.size());
  for (size_t k = 0; k < poly.vertices.size(); k++)
    m_wulff_vertices.col(k) = poly.vertices[k];
  for (auto &facet : m_facets)
    facet.point_index.clear();
  for (const auto &face : poly.faces)
    m_facets[face.facet].point_index = face.loop;
  occ::log::debug("Wulff construction: {} vertices, {} of {} facets active",
                  poly.vertices.size(), poly.faces.size(), m_facets.size());

  size_t N = 0;
  for (auto &facet : m_facets) {
    facet.reorder_and_triangulate(m_wulff_vertices);
    if (facet.point_index.size() > 0) {
      N += facet.triangles.cols();
    }
  }

  m_wulff_triangles = IMat3N(3, N);
  m_wulff_triangle_indices = IVec(N);
  N = 0;
  for (int f = 0; f < m_facets.size(); f++) {
    const auto &facet = m_facets[f];
    if (facet.point_index.size() <= 0)
      continue;
    int size = facet.triangles.cols();
    m_wulff_triangles.block(0, N, 3, size) = facet.triangles;
    m_wulff_triangle_indices.block(N, 0, size, 1).array() = f;
    N += size;
  }
}

const Mat3N &WulffConstruction::vertices() const { return m_wulff_vertices; }

const IMat3N &WulffConstruction::triangles() const { return m_wulff_triangles; }

double WulffConstruction::facet_area(size_t i) const {
  const auto &facet = m_facets[i];
  double area = 0.0;
  for (int t = 0; t < facet.triangles.cols(); ++t) {
    const Vec3 v1 = m_wulff_vertices.col(facet.triangles(0, t));
    const Vec3 v2 = m_wulff_vertices.col(facet.triangles(1, t));
    const Vec3 v3 = m_wulff_vertices.col(facet.triangles(2, t));
    area += 0.5 * (v2 - v1).cross(v3 - v1).norm();
  }
  return area;
}

Vec WulffConstruction::facet_areas() const {
  Vec areas(m_facets.size());
  for (size_t i = 0; i < m_facets.size(); ++i)
    areas(i) = facet_area(i);
  return areas;
}

double WulffConstruction::total_area() const {
  double area = 0.0;
  for (size_t i = 0; i < m_facets.size(); ++i)
    area += facet_area(i);
  return area;
}

std::vector<WulffEdge> WulffConstruction::edges() const {
  std::vector<WulffEdge> result;
  // two active facets share an edge if they have exactly two vertices in common
  for (size_t a = 0; a < m_facets.size(); ++a) {
    if (m_facets[a].point_index.empty())
      continue;
    std::set<int> va(m_facets[a].point_index.begin(),
                     m_facets[a].point_index.end());
    for (size_t b = a + 1; b < m_facets.size(); ++b) {
      if (m_facets[b].point_index.empty())
        continue;
      std::vector<int> shared;
      for (int idx : m_facets[b].point_index) {
        if (va.count(idx))
          shared.push_back(idx);
      }
      if (shared.size() == 2) {
        double len =
            (m_wulff_vertices.col(shared[0]) - m_wulff_vertices.col(shared[1]))
                .norm();
        result.push_back(WulffEdge{a, b, shared[0], shared[1], len});
      }
    }
  }
  return result;
}

} // namespace occ::geometry
