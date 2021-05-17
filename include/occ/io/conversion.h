#pragma once
#include <occ/qm/basisset.h>
#include <occ/core/linear_algebra.h>

namespace occ::io::conversion {

namespace orb {

occ::MatRM from_gaussian(const occ::qm::BasisSet&, const occ::MatRM&);
occ::MatRM to_gaussian(const occ::qm::BasisSet&, const occ::MatRM&);

}

}
