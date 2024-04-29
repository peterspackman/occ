#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::isosurface {

struct SurfaceCurvature {
    FVec mean, gaussian, k1, k2, curvedness, shape_index;
};

SurfaceCurvature calculate_curvature(const std::vector<float> &mean, const std::vector<float> &gaussian);

}
