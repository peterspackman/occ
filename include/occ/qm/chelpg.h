#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::qm {
class Wavefunction;

Vec chelpg_charges(const Wavefunction &wfn, Eigen::Ref<Mat3N> grid_points);

} // namespace occ::qm
