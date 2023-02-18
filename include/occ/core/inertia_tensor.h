#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core {

Mat3 inertia_tensor(Eigen::Ref<const Vec> masses,
                    Eigen::Ref<const Mat3N> positions);

}
