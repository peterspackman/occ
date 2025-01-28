#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/isosurface/isosurface.h>

namespace occ::io {

void write_obj_file(const std::string &filename,
                    const isosurface::Isosurface &isosurface);

}
