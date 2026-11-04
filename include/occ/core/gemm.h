#pragma once
#include <algorithm>
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>

// Eigen's matrix product is single-threaded in this build. `pgemm` is the
// codebase-native way to use all threads for a large one: TBB over disjoint
// output blocks, a serial Eigen GEMM per block (as in integrals.cpp
// mo_transform_packed).

namespace occ::core {

/// C(m x n) = A(m x k) * B(k x n). With accumulate, C += A*B. With transA, A is
/// passed as (k x m) and A^T*B is formed.
///
/// Whether to parallelise here depends on the GEMM backend:
///  - On Apple with Accelerate, the GEMM runs on the AMX matrix co-processor,
///    which is shared per CPU cluster: one dgemm already saturates it (measured:
///    8 concurrent dgemms = 1.2x one) and a single AMX call (~346 GFLOP/s) beats
///    Eigen NEON on all 8 cores (~237). So issue exactly ONE call -- TBB-blocking
///    would only spawn contending AMX calls.
///  - Everywhere else (Linux default has no system BLAS; we do not assume a
///    Linux BLAS is threaded) the underlying GEMM is single-threaded (Eigen's own
///    NEON/AVX, ~52 GFLOP/s), so we parallelise over output blocks of the larger
///    dimension ourselves (measured ~4.6x on 8 cores).
inline void pgemm(Eigen::Ref<occ::Mat> C, Eigen::Ref<const occ::Mat> A,
                  Eigen::Ref<const occ::Mat> B, bool accumulate = false,
                  bool transA = false) {
  const Eigen::Index m = C.rows(), n = C.cols();
  if (m == 0 || n == 0)
    return;
#if defined(__APPLE__) && defined(EIGEN_USE_BLAS)
  if (transA)
    accumulate ? (C.noalias() += A.transpose() * B)
               : (C.noalias() = A.transpose() * B);
  else
    accumulate ? (C.noalias() += A * B) : (C.noalias() = A * B);
#else
  auto colblock = [&](Eigen::Index j0, Eigen::Index jb) {
    auto Cb = C.middleCols(j0, jb);
    const auto Bb = B.middleCols(j0, jb);
    if (transA)
      accumulate ? (Cb.noalias() += A.transpose() * Bb)
                 : (Cb.noalias() = A.transpose() * Bb);
    else
      accumulate ? (Cb.noalias() += A * Bb) : (Cb.noalias() = A * Bb);
  };
  auto rowblock = [&](Eigen::Index i0, Eigen::Index ib) {
    auto Cb = C.middleRows(i0, ib);
    if (transA) // A is (k x m): row i of C uses column i of A
      accumulate ? (Cb.noalias() += A.middleCols(i0, ib).transpose() * B)
                 : (Cb.noalias() = A.middleCols(i0, ib).transpose() * B);
    else
      accumulate ? (Cb.noalias() += A.middleRows(i0, ib) * B)
                 : (Cb.noalias() = A.middleRows(i0, ib) * B);
  };
  const Eigen::Index nt =
      std::max<Eigen::Index>(1, occ::parallel::get_num_threads());
  const Eigen::Index dim = std::max(m, n); // block the larger output dimension
  const Eigen::Index bs = std::max<Eigen::Index>(64, (dim + nt - 1) / nt);
  const size_t nblk = static_cast<size_t>((dim + bs - 1) / bs);
  if (nblk <= 1) {
    colblock(0, n);
    return;
  }
  const bool rows = (m >= n);
  occ::parallel::parallel_for(size_t(0), nblk, [&](size_t blk) {
    const Eigen::Index b0 = static_cast<Eigen::Index>(blk) * bs;
    const Eigen::Index bb = std::min(bs, dim - b0);
    rows ? rowblock(b0, bb) : colblock(b0, bb);
  });
#endif
}

} // namespace occ::core
