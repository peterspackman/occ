#include <occ/io/ply.h>
#include <fmt/os.h>
#include <occ/io/tinyply.h>

namespace occ::io {

void write_ply_file(const std::string &filename,
                    const Eigen::Matrix3Xf &vertices,
                    const Eigen::Matrix3Xi &faces) {
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

}
