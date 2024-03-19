#include <fmt/os.h>
#include <occ/io/obj.h>

namespace occ::io {

void write_obj_file(const std::string &filename, const IsosurfaceMesh &mesh,
                    const VertexProperties &properties) {
    auto file = fmt::output_file(filename);
    file.print("# vertices\n");
    for (size_t idx = 0; idx < mesh.vertices.size(); idx += 3) {
        file.print("v {} {} {}\n", mesh.vertices[idx], mesh.vertices[idx + 1],
                   mesh.vertices[idx + 2]);
    }
    if (mesh.normals.size() > 0) {
        file.print("# vertex normals\n");
        for (size_t idx = 0; idx < mesh.normals.size(); idx += 3) {
            file.print("vn {} {} {}\n", mesh.normals[idx],
                       mesh.normals[idx + 1], mesh.normals[idx + 2]);
        }
    }
    file.print("# faces\n");
    for (size_t idx = 0; idx < mesh.faces.size(); idx += 3) {
        int f1 = mesh.faces[idx] + 1;
        int f2 = mesh.faces[idx + 1] + 1;
        int f3 = mesh.faces[idx + 2] + 1;
        file.print("f {}/{} {}/{} {}/{}\n", f1, f1, f2, f2, f3, f3);
    }
}

} // namespace occ::io
