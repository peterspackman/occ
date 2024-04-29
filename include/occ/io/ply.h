#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/io/isosurface_mesh.h>
#include <string>

namespace occ::io {

void write_ply_file(const std::string &filename,
                    const Eigen::Matrix3Xf &vertices,
                    const Eigen::Matrix3Xi &faces);

void write_ply_mesh(const std::string &filename, const IsosurfaceMesh &mesh,
		    const VertexProperties &properties, bool binary = true);
}
