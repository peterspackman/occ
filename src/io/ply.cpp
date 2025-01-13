#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/timings.h>
#include <occ/io/ply.h>
#include <occ/io/tinyply.h>

namespace occ::io {

// Add some utility functions
namespace detail {
inline void validate_mesh_data(const IsosurfaceMesh &mesh,
                               const VertexProperties &properties) {
  if (mesh.vertices.empty() || mesh.faces.empty()) {
    throw std::invalid_argument("Empty mesh data");
  }
  if (mesh.vertices.size() % 3 != 0 || mesh.faces.size() % 3 != 0) {
    throw std::invalid_argument("Invalid mesh data dimensions");
  }
  if (!mesh.normals.empty() && mesh.normals.size() != mesh.vertices.size()) {
    throw std::invalid_argument("Normals size doesn't match vertices");
  }

  const size_t vertex_count = mesh.vertices.size() / 3;
  for (const auto &[name, prop] : properties.fprops) {
    if (prop.size() != vertex_count) {
      throw std::invalid_argument(fmt::format(
          "Float property '{}' size ({}) doesn't match vertex count ({})", name,
          prop.size(), vertex_count));
    }
  }
  for (const auto &[name, prop] : properties.iprops) {
    if (prop.size() != vertex_count) {
      throw std::invalid_argument(fmt::format(
          "Int property '{}' size ({}) doesn't match vertex count ({})", name,
          prop.size(), vertex_count));
    }
  }
}
} // namespace detail

void write_ply_file(const std::string &filename,
                    const Eigen::Matrix3Xf &vertices,
                    const Eigen::Matrix3Xi &faces) {
  if (vertices.cols() == 0 || faces.cols() == 0) {
    throw std::invalid_argument("Empty mesh data");
  }
  auto file = fmt::output_file(filename);
  file.print("ply\n");
  file.print("format ascii 1.0\n");
  file.print("comment exported from OCC\n");
  file.print("element vertex {}\n", vertices.size() / 3);
  file.print("property float x\n");
  file.print("property float y\n");
  file.print("property float z\n");
  file.print("element face {}\n", faces.size() / 3);
  file.print("property list uchar int vertex_index\n");
  file.print("end_header\n");
  for (size_t idx = 0; idx < vertices.cols(); idx++) {
    file.print("{} {} {}\n", vertices(0, idx), vertices(1, idx),
               vertices(2, idx));
  }
  for (size_t idx = 0; idx < faces.cols(); idx++) {
    file.print("3 {} {} {}\n", faces(0, idx), faces(1, idx), faces(2, idx));
  }
}

void write_ply_mesh(const std::string &filename, const IsosurfaceMesh &mesh,
                    const VertexProperties &properties, bool binary) {
  detail::validate_mesh_data(mesh, properties);
  occ::timing::start(occ::timing::category::io);

  tinyply::PlyFile ply_file;

  const size_t vertex_count = mesh.vertices.size() / 3;
  const size_t face_count = mesh.faces.size() / 3;

  // Add vertices
  ply_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, vertex_count,
      reinterpret_cast<const uint8_t *>(mesh.vertices.data()),
      tinyply::Type::INVALID, 0);

  // Add normals if present
  if (!mesh.normals.empty()) {
    ply_file.add_properties_to_element(
        "vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, vertex_count,
        reinterpret_cast<const uint8_t *>(mesh.normals.data()),
        tinyply::Type::INVALID, 0);
  }

  // Add faces
  ply_file.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::UINT32, face_count,
      reinterpret_cast<const uint8_t *>(mesh.faces.data()),
      tinyply::Type::UINT32, 3);

  // Add float properties
  for (const auto &[name, values] : properties.fprops) {
    ply_file.add_properties_to_element(
        "vertex", {name}, tinyply::Type::FLOAT32, values.size(),
        reinterpret_cast<const uint8_t *>(values.data()),
        tinyply::Type::INVALID, 0);
  }

  // Add integer properties
  for (const auto &[name, values] : properties.iprops) {
    ply_file.add_properties_to_element(
        "vertex", {name}, tinyply::Type::INT32, values.size(),
        reinterpret_cast<const uint8_t *>(values.data()),
        tinyply::Type::INVALID, 0);
  }

  ply_file.get_comments().push_back("Generated by OCC");

  // This is an attempt to workaround a quirk on windows
  // where it would crash sometimes (silently i.e. no throw or segfault)
  // when writing out this file
  // The buffer size implications have not been thoroughly tested

  static constexpr size_t BUFFER_SIZE = 2 * 1024 * 1024; // 2MB buffer
  std::vector<char> buffer(BUFFER_SIZE);
  std::filebuf out_fb;
  out_fb.pubsetbuf(buffer.data(), buffer.size());

  if (!out_fb.open(filename,
                   binary ? std::ios::out | std::ios::binary : std::ios::out)) {
    throw std::runtime_error(
        fmt::format("Failed to open file for writing: {}", filename));
  }

  std::ostream out(&out_fb);

  ply_file.write(out, binary);
  out.flush();

  if (out.fail()) {
    throw std::runtime_error(
        fmt::format("Failed while writing to file: {}", filename));
  }

  if (!out_fb.close()) {
    throw std::runtime_error(fmt::format("Failed to close file: {}", filename));
  }
  occ::timing::stop(occ::timing::category::io);
}

} // namespace occ::io
