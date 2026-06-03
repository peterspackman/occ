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

/// View a contiguous (column-major) 4-tensor's data as an (r x c) matrix.
/// r*c must equal the tensor size; the caller is responsible for having laid the
/// tensor out (via .shuffle()) so that this flat reinterpretation is meaningful.
inline Eigen::Map<const occ::Mat> mat_view(const Eigen::Tensor<double, 4> &t,
                                           Eigen::Index r, Eigen::Index c) {
  return Eigen::Map<const occ::Mat>(t.data(), r, c);
}

/// Parallel tensor permute (transpose): writes contiguous `out` = A.shuffle(perm)
/// -- out has dims Odim[k] = Adims[perm[k]] and out(j...) = A(source). The
/// strided gather from A is split across TBB over contiguous output blocks. This
/// replaces Eigen's serial .shuffle(): in a transpose-GEMM (TTGT) contraction the
/// permutation is the bottleneck (Matthews, TBLIS, arXiv:1607.00291), so it must
/// run on all cores even though the GEMM is handed to a black-box BLAS.
template <int N>
inline void ppermute(double *out, const double *A,
                     const Eigen::array<Eigen::Index, N> &Adims,
                     const Eigen::array<int, N> &perm) {
  Eigen::array<Eigen::Index, N> Odim{}, srcStr{}, Astride{};
  Eigen::Index s = 1;
  for (int k = 0; k < N; ++k) {
    Astride[k] = s;
    s *= Adims[k];
  }
  Eigen::Index sz = 1;
  for (int k = 0; k < N; ++k) {
    Odim[k] = Adims[perm[k]];
    srcStr[k] = Astride[perm[k]];
    sz *= Odim[k];
  }
  // gather a contiguous output range [n0,n1) via an odometer over the output
  // multi-index, advancing the (strided) source offset incrementally.
  auto run = [&](Eigen::Index n0, Eigen::Index n1) {
    Eigen::array<Eigen::Index, N> idx{};
    Eigen::Index rem = n0, src = 0;
    for (int k = 0; k < N; ++k) {
      idx[k] = rem % Odim[k];
      rem /= Odim[k];
      src += idx[k] * srcStr[k];
    }
    for (Eigen::Index n = n0; n < n1; ++n) {
      out[n] = A[src];
      for (int k = 0; k < N; ++k) {
        if (++idx[k] < Odim[k]) {
          src += srcStr[k];
          break;
        }
        idx[k] = 0;
        src -= srcStr[k] * (Odim[k] - 1);
      }
    }
  };
  const Eigen::Index nt =
      std::max<Eigen::Index>(1, occ::parallel::get_num_threads());
  const Eigen::Index bs = std::max<Eigen::Index>(8192, (sz + nt - 1) / nt);
  const size_t nblk = static_cast<size_t>((sz + bs - 1) / bs);
  if (nblk <= 1) {
    run(0, sz);
    return;
  }
  occ::parallel::parallel_for(size_t(0), nblk, [&](size_t blk) {
    const Eigen::Index n0 = static_cast<Eigen::Index>(blk) * bs;
    run(n0, std::min(sz, n0 + bs));
  });
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
  // A -> (fa0, fa1, a0, a1); B -> (b0, b1, fb0, fb1), via parallel transpose
  std::vector<double> Abuf(static_cast<size_t>(A.size()));
  std::vector<double> Bbuf(static_cast<size_t>(B.size()));
  const Eigen::array<Eigen::Index, 4> Adims{A.dimension(0), A.dimension(1),
                                            A.dimension(2), A.dimension(3)};
  const Eigen::array<Eigen::Index, 4> Bdims{B.dimension(0), B.dimension(1),
                                            B.dimension(2), B.dimension(3)};
  ppermute<4>(Abuf.data(), A.data(), Adims,
              Eigen::array<int, 4>{fa[0], fa[1], a0, a1});
  ppermute<4>(Bbuf.data(), B.data(), Bdims,
              Eigen::array<int, 4>{b0, b1, fb[0], fb[1]});
  occ::Mat C(fa0 * fa1, fb0 * fb1);
  pgemm(C, Eigen::Map<const occ::Mat>(Abuf.data(), fa0 * fa1, k0 * k1),
        Eigen::Map<const occ::Mat>(Bbuf.data(), k0 * k1, fb0 * fb1));
  T4 out(fa0, fa1, fb0, fb1);
  std::copy(C.data(), C.data() + C.size(), out.data());
  return out;
}

/// General N-index parallel tensor contraction -- a drop-in, multithreaded
/// replacement for `A.contract(B, pairs)` that routes through pgemm (BLAS), so
/// it parallelises where Eigen's tensor contract() (serial gebp_kernel) does
/// not. Each pair (a,b) contracts A axis a with B axis b. Result is returned in
/// Eigen's contract output order (A free axes ascending, then B free axes
/// ascending), so any existing trailing `.shuffle()` still applies.
template <int RA, int RB, int NC>
Eigen::Tensor<double, RA + RB - 2 * NC>
pcon(const Eigen::Tensor<double, RA> &A, const Eigen::Tensor<double, RB> &B,
     const std::array<Eigen::IndexPair<int>, NC> &pairs) {
  constexpr int FA = RA - NC; // # free A axes
  constexpr int FB = RB - NC; // # free B axes
  constexpr int RO = FA + FB; // output rank
  std::array<bool, RA> aC{};
  std::array<bool, RB> bC{};
  Eigen::array<int, RA> shA{};
  Eigen::array<int, RB> shB{};
  Eigen::Index con = 1;
  // contracted axes: A's go last (pair order), B's go first (pair order)
  for (int i = 0; i < NC; ++i) {
    aC[pairs[i].first] = true;
    bC[pairs[i].second] = true;
    shA[FA + i] = pairs[i].first;
    shB[i] = pairs[i].second;
    con *= A.dimension(pairs[i].first);
  }
  Eigen::array<Eigen::Index, RO> outDims{};
  Eigen::Index freeA = 1, freeB = 1;
  for (int x = 0, n = 0; x < RA; ++x)
    if (!aC[x]) {
      shA[n] = x;
      outDims[n] = A.dimension(x);
      freeA *= A.dimension(x);
      ++n;
    }
  for (int x = 0, n = 0; x < RB; ++x)
    if (!bC[x]) {
      shB[NC + n] = x;
      outDims[FA + n] = B.dimension(x);
      freeB *= B.dimension(x);
      ++n;
    }
  // parallel transpose each operand into [freeA..., con...] / [con..., freeB...]
  std::vector<double> Abuf(static_cast<size_t>(A.size()));
  std::vector<double> Bbuf(static_cast<size_t>(B.size()));
  Eigen::array<Eigen::Index, RA> Adims;
  for (int k = 0; k < RA; ++k)
    Adims[k] = A.dimension(k);
  Eigen::array<Eigen::Index, RB> Bdims;
  for (int k = 0; k < RB; ++k)
    Bdims[k] = B.dimension(k);
  ppermute<RA>(Abuf.data(), A.data(), Adims, shA);
  ppermute<RB>(Bbuf.data(), B.data(), Bdims, shB);
  occ::Mat C(freeA, freeB);
  pgemm(C, Eigen::Map<const occ::Mat>(Abuf.data(), freeA, con),
        Eigen::Map<const occ::Mat>(Bbuf.data(), con, freeB));
  Eigen::Tensor<double, RO> out(outDims);
  std::copy(C.data(), C.data() + C.size(), out.data());
  return out;
}

} // namespace occ::qm::cc
