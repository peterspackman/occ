#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/mo.h>
#include <occ/gto/shell.h>

namespace occ::io::conversion {

namespace orb {

Mat from_gaussian_order_cartesian(const gto::AOBasis &, const Mat &);
Mat to_gaussian_order_cartesian(const gto::AOBasis &, const Mat &);
Mat from_gaussian_order_spherical(const gto::AOBasis &, const Mat &);
Mat to_gaussian_order_spherical(const gto::AOBasis &, const Mat &);

qm::MolecularOrbitals to_gaussian_order(const gto::AOBasis &,
                                        const qm::MolecularOrbitals &);

qm::MolecularOrbitals from_gaussian_order(const gto::AOBasis &,
                                          const qm::MolecularOrbitals &);

} // namespace orb

} // namespace occ::io::conversion
