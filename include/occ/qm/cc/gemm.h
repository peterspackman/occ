#pragma once
#include <algorithm>
#include <array>
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <unsupported/Eigen/CXX11/Tensor>

// Small GEMM helpers for the coupled-cluster hot paths. Eigen's tensor
// contract() and matrix product are both single-threaded in this build, so the
// dense O(o^3 v^3) / O(o^4 v^2) terms in update_amps run on one core. `pgemm`
// reproduces the codebase-native pattern (TBB over disjoint output columns, a
// serial Eigen GEMM per block -- as in integrals.cpp mo_transform_packed) so the
// big contractions, once reshaped to matrices, use all threads.

namespace occ::qm::cc {

/// C(m x n) = A(m x k) * B(k x n), parallelised over column blocks of C (each a
/// serial Eigen GEMM). With accumulate, C += A*B. With transA, A is passed as
/// (k x m) and A^T*B is formed. Output columns are disjoint across threads, so
/// the concurrent writes are race-free.
inline void pgemm(Eigen::Ref<occ::Mat> C, Eigen::Ref<const occ::Mat> A,
                  Eigen::Ref<const occ::Mat> B, bool accumulate = false,
                  bool transA = false) {
  const Eigen::Index n = B.cols();
  if (n == 0)
    return;
#ifdef EIGEN_USE_BLAS
  // A threaded, optimised BLAS (e.g. Apple Accelerate) already parallelises the
  // GEMM internally -- issue a single call and do NOT stack TBB on top (that
  // would oversubscribe). The reshape in pcon2 is what routes the contraction
  // through dgemm.
  if (transA) {
    if (accumulate)
      C.noalias() += A.transpose() * B;
    else
      C.noalias() = A.transpose() * B;
  } else {
    if (accumulate)
      C.noalias() += A * B;
    else
      C.noalias() = A * B;
  }
#else
  // No BLAS: parallelise the serial Eigen GEMM ourselves, one wide column block
  // per thread (A streams once per thread; fat cache-efficient GEMMs).
  const Eigen::Index nt =
      std::max<Eigen::Index>(1, occ::parallel::get_num_threads());
  const Eigen::Index bs = std::max<Eigen::Index>(32, (n + nt - 1) / nt);
  const size_t nblk = static_cast<size_t>((n + bs - 1) / bs);
  occ::parallel::parallel_for(size_t(0), nblk, [&](size_t blk) {
    const Eigen::Index j0 = static_cast<Eigen::Index>(blk) * bs;
    const Eigen::Index jb = std::min(bs, n - j0);
    auto Cb = C.middleCols(j0, jb);
    const auto Bb = B.middleCols(j0, jb);
    if (transA) {
      if (accumulate)
        Cb.noalias() += A.transpose() * Bb;
      else
        Cb.noalias() = A.transpose() * Bb;
    } else {
      if (accumulate)
        Cb.noalias() += A * Bb;
      else
        Cb.noalias() = A * Bb;
    }
  });
#endif
}

/// View a contiguous (column-major) 4-tensor's data as an (r x c) matrix.
/// r*c must equal the tensor size; the caller is responsible for having laid the
/// tensor out (via .shuffle()) so that this flat reinterpretation is meaningful.
inline Eigen::Map<const occ::Mat> mat_view(const Eigen::Tensor<double, 4> &t,
                                           Eigen::Index r, Eigen::Index c) {
  return Eigen::Map<const occ::Mat>(t.data(), r, c);
}

/// Parallel two-index contraction of two 4-tensors -- a drop-in, multithreaded
/// replacement for `A.contract(B, {(a0,b0),(a1,b1)})`. Pairs A axis a0 with B
/// axis b0 and a1 with b1; returns the (A-free..., B-free...) tensor in Eigen's
/// contract output order (so an existing trailing `.shuffle()` still applies).
/// Reshapes each operand once (free indices grouped as the GEMM's outer dim,
/// contracted indices as the shared dim) and dispatches to pgemm.
inline Eigen::Tensor<double, 4> pcon2(const Eigen::Tensor<double, 4> &A, int a0,
                                      int a1, const Eigen::Tensor<double, 4> &B,
                                      int b0, int b1) {
  using T4 = Eigen::Tensor<double, 4>;
  std::array<int, 2> fa{}, fb{};
  for (int x = 0, n = 0; x < 4; ++x)
    if (x != a0 && x != a1)
      fa[n++] = x;
  for (int x = 0, n = 0; x < 4; ++x)
    if (x != b0 && x != b1)
      fb[n++] = x;
  const Eigen::Index fa0 = A.dimension(fa[0]), fa1 = A.dimension(fa[1]);
  const Eigen::Index k0 = A.dimension(a0), k1 = A.dimension(a1);
  const Eigen::Index fb0 = B.dimension(fb[0]), fb1 = B.dimension(fb[1]);
  // A -> (fa0, fa1, a0, a1); B -> (b0, b1, fb0, fb1)
  const T4 As = A.shuffle(Eigen::array<int, 4>{fa[0], fa[1], a0, a1});
  const T4 Bs = B.shuffle(Eigen::array<int, 4>{b0, b1, fb[0], fb[1]});
  occ::Mat C(fa0 * fa1, fb0 * fb1);
  pgemm(C, mat_view(As, fa0 * fa1, k0 * k1), mat_view(Bs, k0 * k1, fb0 * fb1));
  T4 out(fa0, fa1, fb0, fb1);
  std::copy(C.data(), C.data() + C.size(), out.data());
  return out;
}

} // namespace occ::qm::cc
