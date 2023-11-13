#include <occ/io/obj.h>
#include <fmt/os.h>

namespace occ::io {

void write_obj_file(const std::string &filename, const IsosurfaceMesh &mesh,
                    const VertexProperties &properties) {
    auto file = fmt::output_file(filename);
    file.print("# vertices\n");
    for (size_t idx = 0; idx < mesh.vertices.cols(); idx++) {
        file.print("v {} {} {}\n", mesh.vertices(0, idx), mesh.vertices(1, idx),
                   mesh.vertices(2, idx));
    }
    file.print("# vertex normals\n");
    for (size_t idx = 0; idx < mesh.vertices.cols(); idx++) {
        file.print("vn {} {} {}\n", mesh.normals(0, idx), mesh.normals(1, idx),
                   mesh.normals(2, idx));
    }
    file.print("# faces\n");
    for (size_t idx = 0; idx < mesh.faces.cols(); idx++) {
        int f1 = mesh.faces(0, idx) + 1;
        int f2 = mesh.faces(1, idx) + 1;
        int f3 = mesh.faces(2, idx) + 1;
        file.print("f {}/{} {}/{} {}/{}\n", f1, f1, f2, f2, f3, f3);
    }
    file.print("# dnorm di\n");
    for (size_t idx = 0; idx < properties.dnorm.rows(); idx++) {
        file.print("vt {} {} {}\n", properties.dnorm(idx), properties.de(idx),
                   properties.de(idx));
    }
}

}
