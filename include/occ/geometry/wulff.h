#pragma once
#include <occ/core/linear_algebra.h>
#include <string>
#include <vector>

namespace occ::geometry {

struct Facet {
  double energy{0.0};
  Vec3 normal;
  std::string label;
  Vec3 dual;
  std::vector<int> point_index{};
  IMat3N triangles;

  void reorder(const Mat3N &points);
  void reorder_and_triangulate(const Mat3N &points);
};

Mat3N project_to_plane(const Mat3N &, const Vec3 &);

/// \brief An edge of the Wulff polyhedron: two facets meeting along a segment.
struct WulffEdge {
  size_t facet_a{0};
  size_t facet_b{0};
  int vertex_a{-1};
  int vertex_b{-1};
  double length{0.0};
};

class WulffConstruction {
public:
  WulffConstruction(const Mat3N &facet_normals, const Vec &facet_energies,
                    const std::vector<std::string> &facet_labels = {});

  const Mat3N &vertices() const;
  const IMat3N &triangles() const;
  const auto &facets() const { return m_facets; }

  /// Area of facet \p i (sum of its triangle areas); 0 for inactive facets.
  double facet_area(size_t i) const;
  /// Area of every facet (inactive facets contribute 0).
  Vec facet_areas() const;
  /// Total surface area of the polyhedron.
  double total_area() const;
  /// Edges (active-facet pairs sharing two vertices) with their lengths.
  std::vector<WulffEdge> edges() const;

private:
  void extract_wulff_from_dual_hull_simplices(const IMat3N &simplices);
  void merge_coincident_vertices();

  std::vector<Facet> m_facets;

  Mat3N m_wulff_vertices;
  IMat3N m_wulff_triangles;

  IVec m_wulff_triangle_indices;
};

} // namespace occ::geometry
