#pragma once
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::gto {

std::vector<Mat> spherical_gaussian_rotation_matrices(int lmax,
                                                      const Mat3 &rotation);

}
