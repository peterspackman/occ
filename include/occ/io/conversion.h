#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/basisset.h>

namespace occ::io::conversion {

namespace orb {

Mat from_gaussian_order_cartesian(const occ::qm::BasisSet &, const Mat &);
Mat to_gaussian_order_cartesian(const occ::qm::BasisSet &, const Mat &);

} // namespace orb

} // namespace occ::io::conversion
