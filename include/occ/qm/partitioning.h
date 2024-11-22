#pragma once
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

Vec mulliken_partition(const AOBasis &basis, const MolecularOrbitals &mo,
                       Eigen::Ref<const Mat> op);

} // namespace occ::qm
