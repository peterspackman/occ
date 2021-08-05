#pragma once
#include <occ/qm/basisset.h>
#include <occ/core/linear_algebra.h>

namespace occ::io::conversion {

namespace orb {

Mat from_gaussian_order_cartesian(const occ::qm::BasisSet&, const Mat&);
Mat to_gaussian_order_cartesian(const occ::qm::BasisSet&, const Mat&);

}

}
