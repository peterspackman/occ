#include <fmt/os.h>
#include <occ/io/obj.h>

namespace occ::io {

void write_obj_file(const std::string &filename,
                    const isosurface::Isosurface &mesh) {
  auto file = fmt::output_file(filename);
  file.print("# vertices\n");
  for (size_t idx = 0; idx < mesh.vertices.cols(); idx++) {
    file.print("v {} {} {}\n", mesh.vertices(0, idx), mesh.vertices(1, idx),
               mesh.vertices(2, idx));
  }
  if (mesh.normals.size() > 0) {
    file.print("# vertex normals\n");
    for (size_t idx = 0; idx < mesh.normals.cols(); idx++) {
      file.print("vn {} {} {}\n", mesh.normals(0, idx), mesh.normals(1, idx),
                 mesh.normals(2, idx));
    }
  }
  file.print("# faces\n");
  for (size_t idx = 0; idx < mesh.faces.cols(); idx++) {
    int f1 = mesh.faces(0, idx) + 1;
    int f2 = mesh.faces(1, idx) + 1;
    int f3 = mesh.faces(2, idx) + 1;
    file.print("f {}/{} {}/{} {}/{}\n", f1, f1, f2, f2, f3, f3);
  }
}

} // namespace occ::io
