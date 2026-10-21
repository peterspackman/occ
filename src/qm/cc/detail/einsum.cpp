#include "einsum.h"
#include <algorithm>
#include <array>
#include <occ/core/parallel.h>
#include <occ/qm/cc/gemm.h>

namespace occ::qm::cc::detail {

void gemm_rm(const bool ta, const bool tb, const int m, const int n,
             const int k, const double alpha, const double *A, const int lda,
             const double *B, const int ldb, const double beta, double *C,
             const int ldc) {
  using Stride = Eigen::OuterStride<>;
  Eigen::Map<RowMat, 0, Stride> Cm(C, m, n, Stride(ldc));
  const Eigen::Map<const RowMat, 0, Stride> Am(A, ta ? k : m, ta ? m : k,
                                               Stride(lda)),
      Bm(B, tb ? n : k, tb ? k : n, Stride(ldb));
  if (beta == 0.0)
    Cm.setZero();
  else if (beta != 1.0)
    Cm *= beta;
  if (!ta && !tb)
    Cm.noalias() += alpha * Am * Bm;
  else if (ta && !tb)
    Cm.noalias() += alpha * Am.transpose() * Bm;
  else if (!ta && tb)
    Cm.noalias() += alpha * Am * Bm.transpose();
  else
    Cm.noalias() += alpha * Am.transpose() * Bm.transpose();
}

Tn operator+(Tn a, const Tn &b) {
  a += b;
  return a;
}
Tn operator-(Tn a, const Tn &b) {
  a -= b;
  return a;
}
Tn operator*(Tn a, const double s) {
  a *= s;
  return a;
}
Tn divided(Tn a, const Tn &d) {
  a.same_shape(d);
  for (size_t i = 0; i < a.v.size(); i++)
    a.v[i] /= d.v[i];
  return a;
}
double dot(const Tn &a, const Tn &b) {
  a.same_shape(b);
  double s = 0.0;
  for (size_t i = 0; i < a.v.size(); i++)
    s += a.v[i] * b.v[i];
  return s;
}

// numpy a.transpose(perm): out.d[k] = a.d[perm[k]]. Padded at the front, so the
// innermost loop runs over the last real output axis, a plain copy when that
// axis stays last
Tn T(const Tn &a, const ivec &perm) {
  constexpr int R = 6;
  const int r = static_cast<int>(a.d.size()), pad = R - r;
  std::array<int, R> ad{};
  std::array<size_t, R> stride{};
  for (int k = 0; k < R; k++)
    ad[k] = k < pad ? 1 : a.d[k - pad];
  size_t s = 1;
  for (int k = R - 1; k >= 0; k--) {
    stride[k] = s;
    s *= static_cast<size_t>(ad[k]);
  }
  ivec od(r);
  for (int k = 0; k < r; k++)
    od[k] = a.d[perm[k]];
  Tn out(od);
  std::array<int, R> n{};
  std::array<size_t, R> st{};
  for (int k = 0; k < R; k++) {
    const int src = k < pad ? k : pad + perm[k - pad];
    n[k] = ad[src];
    st[k] = stride[src];
  }
  const size_t outer = static_cast<size_t>(n[0]) * n[1] * n[2] * n[3] * n[4];
  auto rows = [&](const size_t q0, const size_t q1) {
    for (size_t q = q0; q < q1; q++) {
      size_t rem = q, base = 0;
      for (int k = 4; k >= 0; k--) {
        base += (rem % n[k]) * st[k];
        rem /= n[k];
      }
      double *dst = out.v.data() + q * n[5];
      const double *src = a.v.data() + base;
      if (st[5] == 1)
        std::copy_n(src, n[5], dst);
      else
        for (int x5 = 0; x5 < n[5]; x5++)
          dst[x5] = src[x5 * st[5]];
    }
  };
  if (out.v.size() <= (size_t(1) << 18))
    rows(0, outer);
  else {
    const size_t nb = std::min<size_t>(
                     outer,
                     4 * static_cast<size_t>(occ::parallel::get_num_threads())),
                 per = (outer + nb - 1) / nb;
    occ::parallel::parallel_for(size_t(0), nb, [&](const size_t b) {
      rows(b * per, std::min(outer, (b + 1) * per));
    });
  }
  return out;
}

static bool identity(const ivec &perm) {
  for (size_t k = 0; k < perm.size(); k++)
    if (perm[k] != static_cast<int>(k))
      return false;
  return true;
}

// numpy einsum of two operands: a label in both inputs and not in the output is
// summed; a label may not repeat within one operand, nor be shared by both
// inputs and the output (no batch or trace terms here)
Tn einsum(const std::string &spec, const Tn &a, const Tn &b) {
  const size_t comma = spec.find(','), arrow = spec.find("->");
  const std::string la = spec.substr(0, comma),
                    lb = spec.substr(comma + 1, arrow - comma - 1),
                    lo = spec.substr(arrow + 2);
  if (la.size() != a.d.size() || lb.size() != b.d.size())
    throw std::runtime_error("einsum: '" + spec +
                             "' does not match the operand ranks");
  std::string con, fa, fb;
  for (const char c : la)
    (lb.find(c) != std::string::npos && lo.find(c) == std::string::npos ? con
                                                                        : fa) +=
        c;
  for (const char c : lb)
    if (con.find(c) == std::string::npos)
      fb += c;
  for (const char c : fa)
    if (fb.find(c) != std::string::npos)
      throw std::runtime_error("einsum: '" + spec + "' has a batch label");
  ivec pa, pb, cd;
  size_t M = 1, K = 1, N = 1;
  for (const char c : fa) {
    pa.push_back(static_cast<int>(la.find(c)));
    cd.push_back(a.d[la.find(c)]);
    M *= a.d[la.find(c)];
  }
  for (const char c : con) {
    if (a.d[la.find(c)] != b.d[lb.find(c)])
      throw std::runtime_error("einsum: '" + spec +
                               "' sums over unequal dimensions");
    pa.push_back(static_cast<int>(la.find(c)));
    pb.push_back(static_cast<int>(lb.find(c)));
    K *= a.d[la.find(c)];
  }
  for (const char c : fb) {
    pb.push_back(static_cast<int>(lb.find(c)));
    cd.push_back(b.d[lb.find(c)]);
    N *= b.d[lb.find(c)];
  }
  Tn at, bt;
  if (!identity(pa))
    at = T(a, pa);
  if (!identity(pb))
    bt = T(b, pb);
  const double *ap = at.v.empty() ? a.v.data() : at.v.data(),
               *bp = bt.v.empty() ? b.v.data() : bt.v.data();
  Tn c(cd);
  // row-major C(M x N) = A(M x K) B(K x N) is column-major C^T = B^T A^T
  pgemm(Eigen::Map<occ::Mat>(c.v.data(), N, M),
        Eigen::Map<const occ::Mat>(bp, N, K),
        Eigen::Map<const occ::Mat>(ap, K, M));
  const std::string lc = fa + fb;
  ivec po;
  for (const char x : lo)
    po.push_back(static_cast<int>(lc.find(x)));
  return identity(po) ? c : T(c, po);
}

Tn from_tensor(const Eigen::Tensor<double, 2> &t) {
  Tn out({static_cast<int>(t.dimension(0)), static_cast<int>(t.dimension(1))});
  for (int i = 0; i < out.d[0]; i++)
    for (int j = 0; j < out.d[1]; j++)
      out.at(i, j) = t(i, j);
  return out;
}

Tn from_tensor(const Eigen::Tensor<double, 4> &t) {
  Tn out({static_cast<int>(t.dimension(0)), static_cast<int>(t.dimension(1)),
          static_cast<int>(t.dimension(2)), static_cast<int>(t.dimension(3))});
  size_t w = 0;
  for (int i = 0; i < out.d[0]; i++)
    for (int j = 0; j < out.d[1]; j++)
      for (int k = 0; k < out.d[2]; k++)
        for (int l = 0; l < out.d[3]; l++)
          out.v[w++] = t(i, j, k, l);
  return out;
}

Eigen::Tensor<double, 4> to_tensor(const Tn &a) {
  Eigen::Tensor<double, 4> t(a.d[0], a.d[1], a.d[2], a.d[3]);
  size_t w = 0;
  for (int i = 0; i < a.d[0]; i++)
    for (int j = 0; j < a.d[1]; j++)
      for (int k = 0; k < a.d[2]; k++)
        for (int l = 0; l < a.d[3]; l++)
          t(i, j, k, l) = a.v[w++];
  return t;
}

Tn block(const occ::Mat &m, const int r0, const int c0, const int nr,
         const int nc) {
  Tn out({nr, nc});
  for (int i = 0; i < nr; i++)
    for (int j = 0; j < nc; j++)
      out.at(i, j) = m(r0 + i, c0 + j);
  return out;
}

Tn as_tn(const occ::Mat &R) {
  Tn r({static_cast<int>(R.rows()), static_cast<int>(R.cols())});
  for (int k = 0; k < R.rows(); k++)
    for (int i = 0; i < R.cols(); i++)
      r.at(k, i) = R(k, i);
  return r;
}

} // namespace occ::qm::cc::detail
