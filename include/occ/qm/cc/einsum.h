#pragma once
#include <array>
#include <occ/core/linear_algebra.h>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>

// numpy-style einsum and transpose on (column-major) Eigen tensors, so
// equations ported from numpy / PySCF read term for term like the original:
//
//   T2 w1 = einsum<2>("ja,jb->ba", fov, t1);
//   T4 tau = t2 + einsum<4>("ia,jb->ijab", t1, t1);
//   woooo += transpose(woooo, {1, 0, 3, 2}); // safe: transpose evaluates
//
// einsum lays both operands out as [free, summed] x [summed, free] matrices
// (skipping the permute when they already are) and contracts them with one
// pgemm, so it threads like the hand-written GEMM contractions in update_amps.
// transpose always returns a fresh tensor, never a lazy shuffle, so
// `X += transpose(X, ...)` cannot alias.

namespace occ::qm::cc {

namespace detail {
/// out = in with out.dim(k) = in.dim(perm[k]); column-major, rank <= 6
void permute(const double *in, const Eigen::Index *dims, int rank,
             const int *perm, double *out);

/// the result's dimensions for spec on operands of the given shapes
void einsum_shape(std::string_view spec, const Eigen::Index *da, int ra,
                  const Eigen::Index *db, int rb, Eigen::Index *dc, int rc);

/// the contraction itself, into out (sized from einsum_shape)
void einsum_contract(std::string_view spec, const double *a,
                     const Eigen::Index *da, int ra, const double *b,
                     const Eigen::Index *db, int rb, double *out, int rc);

template <typename E> constexpr int rank_of() {
  return Eigen::internal::traits<E>::NumDimensions;
}

// a plain tensor passes through by reference; an expression is evaluated
template <typename E> decltype(auto) evaluated(const E &e) {
  if constexpr (std::is_same_v<E, Eigen::Tensor<double, rank_of<E>()>>)
    return (e);
  else
    return Eigen::Tensor<double, rank_of<E>()>(e);
}

template <int R>
std::array<Eigen::Index, R> dims_of(const Eigen::Tensor<double, R> &t) {
  std::array<Eigen::Index, R> d;
  for (int k = 0; k < R; k++)
    d[k] = t.dimension(k);
  return d;
}
} // namespace detail

/// numpy a.transpose(perm): out.dimension(k) = a.dimension(perm[k]), evaluated
template <int R>
Eigen::Tensor<double, R>
transpose(const Eigen::Tensor<double, R> &a,
          const std::type_identity_t<std::array<int, R>> &perm) {
  const auto d = detail::dims_of(a);
  Eigen::array<Eigen::Index, R> od;
  for (int k = 0; k < R; k++)
    od[k] = d[perm[k]];
  Eigen::Tensor<double, R> out(od);
  detail::permute(a.data(), d.data(), R, perm.data(), out.data());
  return out;
}

/// numpy einsum of two operands with an RC-index result: a label in both
/// inputs and not in the output is summed; a label may not repeat within one
/// operand, nor appear in both inputs and the output (no batch or trace terms).
/// The operands may be tensors or tensor expressions.
template <int RC, typename A, typename B>
Eigen::Tensor<double, RC> einsum(std::string_view spec, const A &a,
                                 const B &b) {
  constexpr int RA = detail::rank_of<A>(), RB = detail::rank_of<B>();
  const auto &ta = detail::evaluated(a);
  const auto &tb = detail::evaluated(b);
  const auto da = detail::dims_of(ta);
  const auto db = detail::dims_of(tb);
  std::array<Eigen::Index, RC> dc;
  detail::einsum_shape(spec, da.data(), RA, db.data(), RB, dc.data(), RC);
  Eigen::array<Eigen::Index, RC> od;
  for (int k = 0; k < RC; k++)
    od[k] = dc[k];
  Eigen::Tensor<double, RC> out(od);
  detail::einsum_contract(spec, ta.data(), da.data(), RA, tb.data(), db.data(),
                          RB, out.data(), RC);
  return out;
}

/// sum_x a(x) b(x) over two tensors of the same shape
template <int R>
double dot(const Eigen::Tensor<double, R> &a,
           const Eigen::Tensor<double, R> &b) {
  if (detail::dims_of(a) != detail::dims_of(b))
    throw std::runtime_error("dot: tensor shapes differ");
  return Eigen::Map<const occ::Vec>(a.data(), a.size())
      .dot(Eigen::Map<const occ::Vec>(b.data(), b.size()));
}

/// Frobenius norm of a tensor
template <int R> double norm(const Eigen::Tensor<double, R> &a) {
  return Eigen::Map<const occ::Vec>(a.data(), a.size()).norm();
}

/// largest |a(x)|
template <int R> double max_abs(const Eigen::Tensor<double, R> &a) {
  return a.size() ? Eigen::Map<const occ::Vec>(a.data(), a.size())
                        .cwiseAbs()
                        .maxCoeff()
                  : 0.0;
}

/// a matrix as a 2-index tensor (both column-major: a plain copy)
inline Eigen::Tensor<double, 2> as_tensor(const occ::Mat &m) {
  return Eigen::TensorMap<const Eigen::Tensor<double, 2>>(m.data(), m.rows(),
                                                          m.cols());
}

/// a 2-index tensor as a matrix
inline occ::Mat as_matrix(const Eigen::Tensor<double, 2> &t) {
  return Eigen::Map<const occ::Mat>(t.data(), t.dimension(0), t.dimension(1));
}

} // namespace occ::qm::cc
