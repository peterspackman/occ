#pragma once
// Small pieces shared by the derivative-integral code that calls libcint's
// integral functions directly (the ones IntegralEnvironment has no helper
// for): the call itself, an owning optimizer handle, the list of significant
// shell pairs, and the scatter of 3x3 atom blocks into a 3N x 3N Hessian.
#include <occ/core/linear_algebra.h>
#include <occ/qm/cint_interface.h>
#include <utility>
#include <vector>

namespace occ::qm::detail {

using Intor = libcint::CINTIntegralFunction;

/// Owns a libcint optimizer (null: none).
struct OptHolder {
  libcint::CINTOpt *ptr{nullptr};
  OptHolder() = default;
  OptHolder(const OptHolder &) = delete;
  OptHolder &operator=(const OptHolder &) = delete;
  OptHolder(OptHolder &&o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  OptHolder &operator=(OptHolder &&o) noexcept {
    std::swap(ptr, o.ptr);
    return *this;
  }
  ~OptHolder() {
    if (ptr)
      libcint::CINTdel_optimizer(&ptr);
  }
};

/// f(out, shls) on env; false when libcint finds the integrals zero.
inline bool call(Intor *f, cint::IntegralEnvironment &env, double *out,
                 int *shls, libcint::CINTOpt *opt = nullptr) {
  return f(out, nullptr, shls, env.atom_data_ptr(), env.num_atoms(),
           env.basis_data_ptr(), env.num_basis(), env.env_data_ptr(), opt,
           nullptr) != 0;
}

/// Unordered significant shell pairs (p >= q) from a shell-pair list.
inline std::vector<std::pair<int, int>>
unique_pairs(const std::vector<std::vector<size_t>> &shellpairs) {
  std::vector<std::pair<int, int>> pairs;
  for (size_t p = 0; p < shellpairs.size(); p++)
    for (const auto q : shellpairs[p])
      pairs.emplace_back(static_cast<int>(p), static_cast<int>(q));
  return pairs;
}

/// H(3A.., 3B..) += blk.
inline void add_block(Mat &H, int A, int B, const Mat3 &blk) {
  H.block<3, 3>(3 * A, 3 * B) += blk;
}

/// Second derivatives of an integral over a function on atom M, one on atom
/// N and an operator centre on atom C (a nucleus, a point charge or an
/// auxiliary function), from its MM, NN and MN blocks: the integral depends
/// only on the differences of the three positions, so every block involving
/// C follows by translational invariance. C < 0 is a centre fixed in space.
inline void add_operator_centre_blocks(Mat &H, int M, int N, int C,
                                       const Mat3 &MM, const Mat3 &NN,
                                       const Mat3 &MN) {
  const Mat3 NM = MN.transpose();
  add_block(H, M, M, MM);
  add_block(H, N, N, NN);
  add_block(H, M, N, MN);
  add_block(H, N, M, NM);
  if (C < 0)
    return;
  const Mat3 MC = -(MM + MN), NC = -(NM + NN);
  add_block(H, M, C, MC);
  add_block(H, C, M, MC.transpose());
  add_block(H, N, C, NC);
  add_block(H, C, N, NC.transpose());
  add_block(H, C, C, -(MC + NC));
}

} // namespace occ::qm::detail
