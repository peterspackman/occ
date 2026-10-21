#pragma once
// Internal to occ_cc: a dense row-major tensor (rank <= 6) with numpy's einsum
// and transpose, so the PySCF-ported equations of the correlated densities
// read term for term like their Python originals (chemist notation, t2 as
// ijab, t3 as ijkabc). Contractions go through pgemm.
#include <Eigen/Dense>
#include <cmath>
#include <occ/core/linear_algebra.h>
#include <stdexcept>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace occ::qm::cc::detail {

using ivec = std::vector<int>;
using vec = std::vector<double>;
using RowMat =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MapR = Eigen::Map<RowMat>;
using CMapR = Eigen::Map<const RowMat>;

// C = alpha op(A) op(B) + beta C on row-major arrays with leading dimensions
// (cblas_dgemm's row-major form), serial: for use inside parallel loops
void gemm_rm(bool ta, bool tb, int m, int n, int k, double alpha,
             const double *A, int lda, const double *B, int ldb, double beta,
             double *C, int ldc);

struct Tn {
  ivec d;
  vec v;
  Tn() = default;
  explicit Tn(const ivec &dims) : d(dims), v(count(dims), 0.0) {}
  static size_t count(const ivec &dims) {
    size_t n = 1;
    for (int k = 0; k < static_cast<int>(dims.size()); k++)
      n *= static_cast<size_t>(dims[k]);
    return n;
  }
  void same_shape(const Tn &o) const {
    if (o.d != d)
      throw std::runtime_error("einsum: tensor shapes differ");
  }
  Tn &operator+=(const Tn &o) {
    same_shape(o);
    for (size_t i = 0; i < v.size(); i++)
      v[i] += o.v[i];
    return *this;
  }
  Tn &operator-=(const Tn &o) {
    same_shape(o);
    for (size_t i = 0; i < v.size(); i++)
      v[i] -= o.v[i];
    return *this;
  }
  Tn &operator*=(const double s) {
    for (double &x : v)
      x *= s;
    return *this;
  }
  double &at(const int i, const int j) {
    return v[static_cast<size_t>(i) * d[1] + j];
  }
  double at(const int i, const int j) const {
    return v[static_cast<size_t>(i) * d[1] + j];
  }
  double norm() const {
    double s = 0.0;
    for (const double x : v)
      s += x * x;
    return std::sqrt(s);
  }
};
Tn operator+(Tn a, const Tn &b);
Tn operator-(Tn a, const Tn &b);
Tn operator*(Tn a, double s);
Tn divided(Tn a, const Tn &d);
double dot(const Tn &a, const Tn &b);

/// numpy a.transpose(perm): out.d[k] = a.d[perm[k]]
Tn T(const Tn &a, const ivec &perm);

/// numpy einsum of two operands: a label in both inputs and not in the output
/// is summed; a label may not repeat within one operand, nor be shared by both
/// inputs and the output (no batch or trace terms)
Tn einsum(const std::string &spec, const Tn &a, const Tn &b);

Tn from_tensor(const Eigen::Tensor<double, 2> &t);
Tn from_tensor(const Eigen::Tensor<double, 4> &t);
Eigen::Tensor<double, 4> to_tensor(const Tn &a);
Tn block(const occ::Mat &m, int r0, int c0, int nr, int nc);
Tn as_tn(const occ::Mat &R);

} // namespace occ::qm::cc::detail
