#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/timings.h>
#include <occ/io/ply.h>
#include <occ/io/tinyply.h>

using occ::isosurface::Isosurface;

namespace occ::io {

// Add some utility functions
namespace detail {
inline void validate_mesh_data(const Isosurface &isosurface) {
  if (isosurface.vertices.size() == 0) {
    throw std::invalid_argument("Isosurface has no vertex data");
  }

  if (isosurface.faces.size() == 0) {
    throw std::invalid_argument("Isosurface has no face data");
  }

  if (isosurface.normals.size() != isosurface.vertices.size()) {
    throw std::invalid_argument("Normals size doesn't match vertices");
  }

  const size_t vertex_count = isosurface.vertices.cols();
  for (const auto &[name, prop] : isosurface.properties.properties) {
    const auto name_copy = name;
    std::visit(
        [name_copy, vertex_count](const auto &values) {
          if (values.size() != vertex_count) {
            throw std::invalid_argument(fmt::format(
                "Property '{}' size ({}) doesn't match vertex count ({})",
                name_copy, values.size(), vertex_count));
          }
        },
        prop);
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

void write_ply_mesh(const std::string &filename, const Isosurface &isosurface,
                    bool binary) {
  detail::validate_mesh_data(isosurface);
  occ::timing::start(occ::timing::category::io);

  tinyply::PlyFile ply_file;
  const size_t vertex_count = isosurface.vertices.cols();
  const size_t face_count = isosurface.faces.cols();

  // Add vertices
  ply_file.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, vertex_count,
      reinterpret_cast<const uint8_t *>(isosurface.vertices.data()),
      tinyply::Type::INVALID, 0);

  // Add normals if present
  if (isosurface.normals.size() > 0) {
    ply_file.add_properties_to_element(
        "vertex", {"nx", "ny", "nz"}, tinyply::Type::FLOAT32, vertex_count,
        reinterpret_cast<const uint8_t *>(isosurface.normals.data()),
        tinyply::Type::INVALID, 0);
  }

  // Add faces
  ply_file.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::UINT32, face_count,
      reinterpret_cast<const uint8_t *>(isosurface.faces.data()),
      tinyply::Type::UINT32, 3);

  // Add curvature properties if present
  if (isosurface.gaussian_curvature.size() > 0) {
    ply_file.add_properties_to_element(
        "vertex", {"gaussian_curvature"}, tinyply::Type::FLOAT32, vertex_count,
        reinterpret_cast<const uint8_t *>(isosurface.gaussian_curvature.data()),
        tinyply::Type::INVALID, 0);
  }

  if (isosurface.mean_curvature.size() > 0) {
    ply_file.add_properties_to_element(
        "vertex", {"mean_curvature"}, tinyply::Type::FLOAT32, vertex_count,
        reinterpret_cast<const uint8_t *>(isosurface.mean_curvature.data()),
        tinyply::Type::INVALID, 0);
  }

  occ::log::debug("Need to write {} properties", isosurface.properties.count());

  // Add variant properties
  for (const auto &[name, prop] : isosurface.properties.properties) {
    const auto name_copy = name;
    std::visit(
        [name_copy, &ply_file](const auto &values) {
          using ValueType = std::decay_t<decltype(values)>;
          if constexpr (std::is_same_v<ValueType, FVec>) {
            ply_file.add_properties_to_element(
                "vertex", {name_copy}, tinyply::Type::FLOAT32, values.size(),
                reinterpret_cast<const uint8_t *>(values.data()),
                tinyply::Type::INVALID, 0);
            occ::log::debug("Writing float property: {}", name_copy);
          } else if constexpr (std::is_same_v<ValueType, IVec>) {
            ply_file.add_properties_to_element(
                "vertex", {name_copy}, tinyply::Type::INT32, values.size(),
                reinterpret_cast<const uint8_t *>(values.data()),
                tinyply::Type::INVALID, 0);
            occ::log::debug("Writing integer property: {}", name_copy);
          } else {
            occ::log::warn("Skipping writing surface property: {}", name_copy);
          }
        },
        prop);
  }

  ply_file.get_comments().push_back("Generated by OCC");

  // File writing with buffer
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
