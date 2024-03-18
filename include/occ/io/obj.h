#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/io/isosurface_mesh.h>

namespace occ::io {



void write_obj_file(const std::string &filename, const IsosurfaceMesh &mesh,
                    const VertexProperties &properties);

}
