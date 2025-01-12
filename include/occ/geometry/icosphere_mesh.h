#include <Eigen/Dense>
#include <ankerl/unordered_dense.h>
#include <occ/core/linear_algebra.h>

namespace occ::geometry {
class IcosphereMesh {

public:
  IcosphereMesh(size_t subdivisions = 0);
  inline const Mat3N &vertices() const { return m_vertices; }
  inline const IMat3N &faces() const { return m_faces; }
  static std::pair<size_t, size_t> compute_sizes(size_t);

  static Mat3N initial_vertices();
  static IMat3N initial_faces();

private:
  Eigen::Matrix3Xd m_vertices;
  Eigen::Matrix3Xi m_faces;
  ankerl::unordered_dense::map<int64_t, int> midpoint_cache;

  int add_vertex(const occ::Vec3 &p);
  int add_midpoint(size_t p1, size_t p2);

  size_t m_vertex_index{0};
};

} // namespace occ::geometry
