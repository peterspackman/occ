#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <numeric>
#include <set>
#include <occ/core/format_matrix.h>
#include <occ/core/log.h>
#include <occ/geometry/quickhull.h>
#include <occ/geometry/wulff.h>

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

void Facet::reorder(const Mat3N &points) {
  if (point_index.empty())
    return;

  Mat3N points_2d = project_to_plane(points, normal);

  Vec3 centroid = points_2d.col(0);
  Mat3N directions = points_2d.colwise() - centroid;
  for (int i = 1; i < directions.cols(); ++i) {
    directions.col(i).normalize();
  }
  // Create a vector of indices for sorting
  std::vector<size_t> indices(point_index.size() - 1);
  std::iota(indices.begin(), indices.end(), 1);

  // Sort the indices based on the angles in the directions array
  std::sort(indices.begin(), indices.end(),
            [&directions](size_t i1, size_t i2) {
              double angle1 = std::atan2(directions(1, i1), directions(0, i1));
              double angle2 = std::atan2(directions(1, i2), directions(0, i2));
              return angle1 < angle2;
            });

  // Create a new sorted point_index array
  std::vector<int> sorted_point_index(point_index.size());
  sorted_point_index[0] = point_index[0]; // Keep the first index unchanged
  for (size_t i = 0; i < indices.size(); ++i) {
    sorted_point_index[i + 1] = point_index[indices[i]];
  }

  // Replace the original point_index with the sorted version
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

  Mat3N dual_points(3, N);
  for (int i = 0; i < N; i++) {
    double energy = facet_energies(i);
    // dual = p / (|p|^2), since we haven't scaled p just divide by energy
    Vec3 dual = facet_normals.col(i).array() / energy;
    m_facets.push_back(Facet{energy, facet_normals.col(i),
                             (facet_labels.size() > i)
                                 ? facet_labels[i]
                                 : fmt::format("facet_{}", i),
                             dual});
    dual_points.col(i) = dual;
  }
  quickhull::QuickHull<double> hull_builder;

  auto hull =
      hull_builder.getConvexHull(dual_points.data(), dual_points.cols(), true);

  occ::log::debug("Convex hull has {} faces, {} vertices",
                  hull.triangles().cols(), hull.vertices().cols());
  occ::log::debug("Hull triangles:\n{}",
                  format_matrix(hull.triangles(), "{:6d}"));
  occ::log::debug("Hull vertices:\n{}", format_matrix(hull.vertices()));
  IMat3N triangles = hull.triangles().cast<int>();
  extract_wulff_from_dual_hull_simplices(triangles);
}

void WulffConstruction::extract_wulff_from_dual_hull_simplices(
    const IMat3N &simplices) {

  m_wulff_vertices = Mat3N(3, simplices.cols());
  for (int i = 0; i < m_wulff_vertices.cols(); i++) {
    auto &facet_a = m_facets[simplices(0, i)];
    auto &facet_b = m_facets[simplices(1, i)];
    auto &facet_c = m_facets[simplices(2, i)];
    m_wulff_vertices.col(i) =
        (facet_b.dual - facet_a.dual).cross(facet_c.dual - facet_a.dual);

    double fac = m_wulff_vertices.col(i).dot(facet_a.normal);
    if (std::abs(fac) < 1e-6) {
      occ::log::warn("zero or near zero scaling factor in wulff construction - "
                     "check if system is 2D or is missing facets!");
    }

    double scale_factor = facet_a.energy / fac;
    m_wulff_vertices.col(i).array() *= scale_factor;

    // push_back facet_indices
    facet_a.point_index.push_back(i);
    facet_b.point_index.push_back(i);
    facet_c.point_index.push_back(i);
  }

  merge_coincident_vertices();

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

// The dual hull is triangulated, so a Wulff corner where more than three
// facets meet is produced once per dual simplex: coincident vertices with
// distinct indices. Merge them so facet polygons, edges and corners see one
// vertex per geometric corner.
void WulffConstruction::merge_coincident_vertices() {
  const int nv = m_wulff_vertices.cols();
  if (nv == 0)
    return;
  const double tol2 = std::pow(
      1e-8 * std::max(1.0, m_wulff_vertices.colwise().norm().maxCoeff()), 2);

  std::vector<int> remap(nv);
  std::vector<int> kept;
  for (int i = 0; i < nv; ++i) {
    int found = -1;
    for (size_t k = 0; k < kept.size(); ++k) {
      if ((m_wulff_vertices.col(i) - m_wulff_vertices.col(kept[k]))
              .squaredNorm() < tol2) {
        found = k;
        break;
      }
    }
    if (found < 0) {
      found = kept.size();
      kept.push_back(i);
    }
    remap[i] = found;
  }
  if (static_cast<int>(kept.size()) == nv)
    return;

  Mat3N merged(3, kept.size());
  for (size_t k = 0; k < kept.size(); ++k)
    merged.col(k) = m_wulff_vertices.col(kept[k]);
  m_wulff_vertices = merged;

  for (auto &facet : m_facets) {
    std::vector<int> updated;
    for (int idx : facet.point_index) {
      int v = remap[idx];
      if (std::find(updated.begin(), updated.end(), v) == updated.end())
        updated.push_back(v);
    }
    if (updated.size() < 3)
      updated.clear(); // degenerate facet -> inactive
    facet.point_index = std::move(updated);
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
