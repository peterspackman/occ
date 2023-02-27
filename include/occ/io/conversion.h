#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/mo.h>
#include <occ/qm/shell.h>

namespace occ::io::conversion {

namespace orb {

Mat from_gaussian_order_cartesian(const qm::AOBasis &, const Mat &);
Mat to_gaussian_order_cartesian(const qm::AOBasis &, const Mat &);
Mat from_gaussian_order_spherical(const qm::AOBasis &, const Mat &);
Mat to_gaussian_order_spherical(const qm::AOBasis &, const Mat &);

qm::MolecularOrbitals to_gaussian_order(const qm::AOBasis &,
                                        const qm::MolecularOrbitals &);

qm::MolecularOrbitals from_gaussian_order(const qm::AOBasis &,
                                          const qm::MolecularOrbitals &);

} // namespace orb

} // namespace occ::io::conversion
