#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core {
// same as indexing='ij' order in numpy, could easily generalize this
std::pair<Mat, Mat> meshgrid(const Vec &, const Vec &);
} // namespace occ::core
