#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::dma {

void solid_harmonics(Eigen::Ref<const Vec3> pos, int j, Eigen::Ref<Vec> r);

}
