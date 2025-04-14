#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dma/mult.h>

namespace occ::dma {
void addqlm(int l, int lmax, double f, Eigen::Ref<const Vec> gx,
            Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz, Mult &out);

void addql0(int l, double f, Eigen::Ref<const Vec> gx,
            Eigen::Ref<const Vec> gy, Eigen::Ref<const Vec> gz,
            Mult& out);

} // namespace occ::dma
