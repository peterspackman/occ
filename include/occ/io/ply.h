#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/isosurface/isosurface.h>
#include <string>

namespace occ::io {

void write_ply_file(const std::string &filename,
                    const Eigen::Matrix3Xf &vertices,
                    const Eigen::Matrix3Xi &faces);

void write_ply_mesh(const std::string &filename,
                    const isosurface::Isosurface &isosurface,
                    bool binary = true);
} // namespace occ::io
