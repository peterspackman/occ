#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core {

struct SymmetricEigenResult {
  Vec eigenvalues;  ///< ascending
  Mat eigenvectors; ///< orthonormal, column i belongs to eigenvalue i
};

/// Eigenvalues and eigenvectors of the real symmetric matrix A (its lower
/// triangle is read).
///
/// With a LAPACK (OCC_HAVE_LAPACK) this is the divide-and-conquer dsyevd,
/// several times faster than Eigen's QR-iteration solver even on one core;
/// otherwise Eigen's SelfAdjointEigenSolver. Call it outside TBB parallel
/// regions: a threaded LAPACK runs its own threads, sized to occ's.
SymmetricEigenResult eigh(Eigen::Ref<const Mat> A);

} // namespace occ::core
