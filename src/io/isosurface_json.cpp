#include <fmt/os.h>
#include <fstream>
#include <occ/io/eigen_json.h>
#include <occ/io/isosurface_json.h>

namespace occ::io {

nlohmann::json isosurface_to_json(const isosurface::Isosurface &surf) {
  nlohmann::json j;

  // Metadata
  j["kind"] = surf.kind;
  j["description"] = surf.description;
  j["isovalue"] = surf.isovalue;
  j["separation"] = surf.separation;
  j["volume"] = surf.volume();
  j["surfaceArea"] = surf.surface_area();

  // Mesh data dimensions
  j["numVertices"] = surf.vertices.cols();
  j["numFaces"] = surf.faces.cols();

  // Vertices as flat array for efficiency
  nlohmann::json vertices = nlohmann::json::array();
  for (int i = 0; i < surf.vertices.cols(); ++i) {
    vertices.push_back(surf.vertices(0, i));
    vertices.push_back(surf.vertices(1, i));
    vertices.push_back(surf.vertices(2, i));
  }
  j["vertices"] = vertices;

  // Faces as flat array
  nlohmann::json faces = nlohmann::json::array();
  for (int i = 0; i < surf.faces.cols(); ++i) {
    faces.push_back(surf.faces(0, i));
    faces.push_back(surf.faces(1, i));
    faces.push_back(surf.faces(2, i));
  }
  j["faces"] = faces;

  // Normals if present
  if (surf.normals.cols() > 0) {
    nlohmann::json normals = nlohmann::json::array();
    for (int i = 0; i < surf.normals.cols(); ++i) {
      normals.push_back(surf.normals(0, i));
      normals.push_back(surf.normals(1, i));
      normals.push_back(surf.normals(2, i));
    }
    j["normals"] = normals;
  }

  // Curvature data if present
  if (surf.gaussian_curvature.size() > 0) {
    nlohmann::json gaussian_curvature = nlohmann::json::array();
    for (int i = 0; i < surf.gaussian_curvature.size(); ++i) {
      gaussian_curvature.push_back(surf.gaussian_curvature(i));
    }
    j["gaussianCurvature"] = gaussian_curvature;
  }

  if (surf.mean_curvature.size() > 0) {
    nlohmann::json mean_curvature = nlohmann::json::array();
    for (int i = 0; i < surf.mean_curvature.size(); ++i) {
      mean_curvature.push_back(surf.mean_curvature(i));
    }
    j["meanCurvature"] = mean_curvature;
  }

  // Properties
  if (surf.properties.count() > 0) {
    nlohmann::json props;
    for (const auto &kv : surf.properties.properties) {
      const auto &name = kv.first;
      const auto &prop = kv.second;
      std::visit(
          [&props, &name](const auto &values) {
            using ValueType = std::decay_t<decltype(values)>;
            nlohmann::json prop_array = nlohmann::json::array();
            for (int i = 0; i < values.size(); ++i) {
              prop_array.push_back(values(i));
            }
            props[name] = prop_array;
          },
          prop);
    }
    j["properties"] = props;
  }

  return j;
}

void write_isosurface_json(const std::string &filename,
                           const isosurface::Isosurface &surf) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  file << isosurface_to_json(surf).dump(2);
  file.close();
}

std::string isosurface_to_json_string(const isosurface::Isosurface &surf) {
  return isosurface_to_json(surf).dump();
}

} // namespace occ::io