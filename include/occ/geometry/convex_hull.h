#pragma once
#include <fstream>
#include <memory>
#include <occ/geometry/mesh.h>
#include <unordered_map>
#include <vector>

namespace quickhull {

template <typename T> class ConvexHull {
  Eigen::Matrix<T, 3, Eigen::Dynamic> m_vertices;
  std::vector<size_t> m_indices;

public:
  ConvexHull() {}

  ConvexHull(const Eigen::Matrix<T, 3, Eigen::Dynamic> &vertices,
             const std::vector<size_t> indices)
      : m_vertices(vertices), m_indices(indices) {}

  // Construct vertex and index buffers from half edge mesh and pointcloud
  ConvexHull(const MeshBuilder<T> &mesh,
             const Eigen::Matrix<T, 3, Eigen::Dynamic> &pointCloud, bool CCW) {

    std::vector<bool> faceProcessed(mesh.m_faces.size(), false);
    std::vector<size_t> faceStack;
    for (size_t i = 0; i < mesh.m_faces.size(); i++) {
      if (!mesh.m_faces[i].isDisabled()) {
        faceStack.push_back(i);
        break;
      }
    }
    if (faceStack.size() == 0) {
      return;
    }

    const size_t iCCW = CCW ? 1 : 0;
    const size_t finalMeshFaceCount =
        mesh.m_faces.size() - mesh.m_disabledFaces.size();
    m_indices.reserve(finalMeshFaceCount * 3);

    while (faceStack.size()) {
      auto it = faceStack.end() - 1;
      size_t top = *it;
      assert(!mesh.m_faces[top].isDisabled());
      faceStack.erase(it);
      if (faceProcessed[top]) {
        continue;
      } else {
        faceProcessed[top] = true;
        auto halfEdges = mesh.getHalfEdgeIndicesOfFace(mesh.m_faces[top]);
        size_t adjacent[] = {
            mesh.m_halfEdges[mesh.m_halfEdges[halfEdges[0]].m_opp].m_face,
            mesh.m_halfEdges[mesh.m_halfEdges[halfEdges[1]].m_opp].m_face,
            mesh.m_halfEdges[mesh.m_halfEdges[halfEdges[2]].m_opp].m_face};
        for (auto a : adjacent) {
          if (!faceProcessed[a] && !mesh.m_faces[a].isDisabled()) {
            faceStack.push_back(a);
          }
        }
        auto vertices = mesh.getVertexIndicesOfFace(mesh.m_faces[top]);
        m_indices.push_back(vertices[0]);
        m_indices.push_back(vertices[1 + iCCW]);
        m_indices.push_back(vertices[2 - iCCW]);
      }
    }
    m_vertices = pointCloud;
  }

  ConvexHull reduced() const {
    std::unordered_map<size_t, size_t> vertex_mapping;
    std::vector<size_t> new_indices;
    new_indices.reserve(m_indices.size());
    for (size_t idx : m_indices) {
      auto loc = vertex_mapping.find(idx);
      if (loc != vertex_mapping.end()) {
        new_indices.push_back(loc->second);
      } else {
        size_t new_idx = vertex_mapping.size();
        new_indices.push_back(new_idx);
        vertex_mapping.insert({idx, new_idx});
      }
    }
    std::vector<size_t> vertex_list(vertex_mapping.size());
    for (const auto &[idx, new_idx] : vertex_mapping) {
      vertex_list[new_idx] = idx;
    }
    return ConvexHull(m_vertices(Eigen::all, vertex_list), new_indices);
  }

  inline Eigen::Map<const Eigen::Matrix<size_t, 3, Eigen::Dynamic>>
  triangles() const {
    return Eigen::Map<const Eigen::Matrix<size_t, 3, Eigen::Dynamic>>(
        m_indices.data(), 3, m_indices.size() / 3);
  }
  inline const auto &indices() const { return m_indices; }
  inline auto &indices() { return m_indices; }
  inline const auto &vertices() const { return m_vertices; }
  inline auto &vertices() { return m_vertices; }

  // Export the mesh to a Waveform OBJ file
  void writeWaveformOBJ(const std::string &filename,
                        const std::string &objectName = "quickhull") const {
    std::ofstream objFile;
    objFile.open(filename);
    objFile << "o " << objectName << "\n";
    for (int i = 0; i < m_vertices.cols(); i++) {
      objFile << "v " << m_vertices(0, i) << " " << m_vertices(1, i) << " "
              << m_vertices(2, i) << "\n";
    }
    size_t triangleCount = m_indices.size() / 3;
    for (size_t i = 0; i < triangleCount; i++) {
      objFile << "f " << m_indices[i * 3] + 1 << " " << m_indices[i * 3 + 1] + 1
              << " " << m_indices[i * 3 + 2] + 1 << "\n";
    }
    objFile.close();
  }
};

} // namespace quickhull
