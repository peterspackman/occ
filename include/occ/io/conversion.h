#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/shell.h>

namespace occ::io::conversion {

namespace orb {

Mat from_gaussian_order_cartesian(const occ::qm::AOBasis &, const Mat &);
Mat to_gaussian_order_cartesian(const occ::qm::AOBasis &, const Mat &);
Mat from_gaussian_order_spherical(const occ::qm::AOBasis &, const Mat &);
Mat to_gaussian_order_spherical(const occ::qm::AOBasis &, const Mat &);

} // namespace orb

} // namespace occ::io::conversion
