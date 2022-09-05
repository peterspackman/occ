#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core::linalg {

occ::Mat3 kabsch_rotation_matrix(const occ::Mat3N &, const occ::Mat3N &,
                                 bool ensure_proper_rotation = true);

}
