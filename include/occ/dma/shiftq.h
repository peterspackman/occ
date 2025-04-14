#pragma once
#include <occ/dma/mult.h>

namespace occ::dma {

void shiftq(const Mult &q1, int l1, int m1, Mult &q2, int m2,
            Eigen::Ref<const Vec3> pos);
}
