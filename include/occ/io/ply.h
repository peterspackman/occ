#pragma once
#include <occ/core/linear_algebra.h>
#include <string>

namespace occ::io {

void write_ply_file(const std::string &filename,
                    const Eigen::Matrix3Xf &vertices,
                    const Eigen::Matrix3Xi &faces);

}
