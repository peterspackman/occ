#pragma once
#include <ankerl/unordered_dense.h>
#include <cstdint>
#include <occ/core/linear_algebra.h>

namespace occ::io {

struct IsosurfaceMesh {
  IsosurfaceMesh() {}
  IsosurfaceMesh(size_t num_vertices, size_t num_faces)
      : vertices(3 * num_vertices), faces(3 * num_faces),
        normals(3 * num_vertices) {}
  std::vector<float> vertices;
  std::vector<uint32_t> faces;
  std::vector<float> normals;
  std::vector<float> mean_curvature;
  std::vector<float> gaussian_curvature;
};

inline IsosurfaceMesh mesh_from_vertices_faces(Eigen::Ref<const Mat3N> vertices, Eigen::Ref<const IMat3N> faces) {
    IsosurfaceMesh result;

    // Copy vertices
    result.vertices.resize(3 * vertices.cols());
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> vertices_map(result.vertices.data(), 3, vertices.cols());
    vertices_map = vertices.cast<float>();

    // Copy faces
    result.faces.resize(3 * faces.cols());
    Eigen::Map<Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>> faces_map(result.faces.data(), 3, faces.cols());
    faces_map = faces.cast<uint32_t>();

    return result;
}

struct VertexProperties {
  using FloatVertexProperties =
      ankerl::unordered_dense::map<std::string, std::vector<float>>;
  using IntVertexProperties =
      ankerl::unordered_dense::map<std::string, std::vector<int>>;

  template <typename T>
  void add_property(const std::string &name, const std::vector<T> &values) {
    if constexpr (std::is_same<T, float>::value) {
      fprops[name] = values;
    } else {
      iprops[name] = values;
    }
  }

  void add_property(const std::string &name, const FVec &values) {
    fprops[name] =
        std::vector<float>(values.data(), values.data() + values.size());
  }

  void add_property(const std::string &name, const IVec &values) {
    iprops[name] =
        std::vector<int>(values.data(), values.data() + values.size());
  }

  FloatVertexProperties fprops;
  IntVertexProperties iprops;
};

} // namespace occ::io
