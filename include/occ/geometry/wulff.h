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

class WulffConstruction {
public:
  WulffConstruction(const Mat3N &facet_normals, const Vec &facet_energies,
                    const std::vector<std::string> &facet_labels = {});

  const Mat3N &vertices() const;
  const IMat3N &triangles() const;
  const auto &facets() const { return m_facets; }

private:
  void extract_wulff_from_dual_hull_simplices(const IMat3N &simplices);

  std::vector<Facet> m_facets;

  Mat3N m_wulff_vertices;
  IMat3N m_wulff_triangles;

  IVec m_wulff_triangle_indices;
};

} // namespace occ::geometry
