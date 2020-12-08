#pragma once
#include <tonto/qm/basisset.h>
#include <tonto/core/linear_algebra.h>

namespace tonto::io::conversion {

namespace orb {

tonto::MatRM from_gaussian(const tonto::qm::BasisSet&, const tonto::MatRM&);
tonto::MatRM to_gaussian(const tonto::qm::BasisSet&, const tonto::MatRM&);

}

}
