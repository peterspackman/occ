#pragma once
// Shared pieces of the density-fitting derivative code (RI-J and RI-K
// gradients and second derivatives): libcint call helpers and loops over
// AO shell pairs and auxiliary shells with plain or first-derivative
// three-centre integrals.
#include "libcint_derivatives.h"
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/integral_engine.h>
#include <utility>
#include <vector>

namespace occ::qm::detail::dfd {

using ShellKind = gto::Shell::Kind;
using Op = cint::Operator;

using detail::add_block;
using detail::add_operator_centre_blocks;
using detail::call;
using detail::Intor;
using detail::OptHolder;

// Unordered significant AO shell pairs (p >= q as the engine lists them).
inline std::vector<std::pair<int, int>>
unordered_pairs(const IntegralEngine &ao) {
  return detail::unique_pairs(ao.shellpairs());
}

// The ip1 integrals for both orders of every unordered AO pair and every aux
// shell R, in parallel: f(a, b, R, na, nb, nr, buf) with
// buf[x][r][j][i] = (d_x a_i b_j | R_r), electron coordinate of a. Every
// caller puts the centre derivative on a's atom and its opposite on R's
// (translational invariance), so triples with a on R's atom are skipped.
// Which index the parallel tasks own. AuxShell (the default) keeps each
// aux shell's fitted quantities in one task; ShellPair lets a callback write
// the (a, b) and (b, a) blocks of a shared matrix without races.
enum class TripleTasks { AuxShell, ShellPair };

template <ShellKind kind, TripleTasks tasks = TripleTasks::AuxShell,
          typename F>
void for_each_ip1_triple(IntegralEngine &ao, F &&f) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const int nsh_ao = static_cast<int>(aobasis.size());
  const int nsh_aux = static_cast<int>(auxbasis.size());
  const auto pairs = unordered_pairs(ao);
  cint::Optimizer opt(ao.env(), Op::coulomb, 3, 1);
  const size_t bufsize = 3 * aobasis.max_shell_size() *
                         aobasis.max_shell_size() * auxbasis.max_shell_size();
  occ::parallel::thread_local_storage<std::vector<double>> tl_buf(
      [bufsize]() { return std::vector<double>(bufsize); });
  const auto &ao_atom = aobasis.shell_to_atom();
  const auto &aux_atom = auxbasis.shell_to_atom();
  auto triple = [&](int p, int q, int R) {
    auto &buf = tl_buf.local();
    auto one = [&](int a, int b) {
      if (ao_atom[a] == aux_atom[R])
        return;
      const auto dims = ao.env().three_center_helper_grad<Op::coulomb, kind>(
          {a, b, R + nsh_ao}, opt.optimizer_ptr(), buf.data(), nullptr);
      if (dims[0] >= 0)
        f(a, b, R, dims[0], dims[1], dims[2], buf.data());
    };
    one(p, q);
    if (p != q)
      one(q, p);
  };
  if constexpr (tasks == TripleTasks::AuxShell) {
    occ::parallel::parallel_for(0, nsh_aux, [&](int R) {
      for (const auto &[p, q] : pairs)
        triple(p, q, R);
    });
  } else {
    occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t k) {
      for (int R = 0; R < nsh_aux; R++)
        triple(pairs[k].first, pairs[k].second, R);
    });
  }
}

// Parallel over unordered AO pairs: f(p, q, na, nb, I) with I the pair's
// integrals over every auxiliary function, (na * nb) x naux column-major
// (I(i + j * na, P) = (p_i q_j | P)), so contractions over the auxiliary
// index, and over many densities or coefficient vectors at once, are GEMMs.
template <ShellKind kind, typename F>
void for_each_pair_block(IntegralEngine &ao, F &&f) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const int nsh_ao = static_cast<int>(aobasis.size());
  const auto &first_aux = auxbasis.first_bf();
  const Eigen::Index naux = auxbasis.nbf();
  const auto pairs = unordered_pairs(ao);
  cint::Optimizer opt(ao.env(), Op::coulomb, 3);
  const size_t bufsize = aobasis.max_shell_size() * aobasis.max_shell_size() *
                         auxbasis.max_shell_size();
  struct TL {
    std::vector<double> buf;
    Mat I;
  };
  occ::parallel::thread_local_storage<TL> tls(
      [bufsize]() { return TL{std::vector<double>(bufsize), Mat()}; });
  occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t k) {
    const auto [p, q] = pairs[k];
    auto &t = tls.local();
    const int na = aobasis[p].size(), nb = aobasis[q].size();
    t.I.setZero(na * nb, naux);
    bool any = false;
    for (int R = 0; R < static_cast<int>(auxbasis.size()); R++) {
      const auto dims = ao.env().three_center_helper<Op::coulomb, kind>(
          {p, q, R + nsh_ao}, opt.optimizer_ptr(), t.buf.data(), nullptr);
      if (dims[0] < 0)
        continue;
      any = true;
      t.I.middleCols(first_aux[R], dims[2]) =
          Eigen::Map<const Mat>(t.buf.data(), na * nb, dims[2]);
    }
    if (any)
      f(p, q, na, nb, t.I);
  });
}

template <ShellKind kind, typename F>
void for_each_metric_ip1_impl(IntegralEngine &aux, F &&f) {
  const auto &auxbasis = aux.aobasis();
  const auto &atom = auxbasis.shell_to_atom();
  const int nsh = static_cast<int>(auxbasis.size());
  cint::Optimizer opt(aux.env(), Op::coulomb, 2, 1);
  const size_t bufsize =
      3 * auxbasis.max_shell_size() * auxbasis.max_shell_size();
  occ::parallel::thread_local_storage<std::vector<double>> tl_buf(
      [bufsize]() { return std::vector<double>(bufsize); });
  occ::parallel::parallel_for(0, nsh, [&](int P) {
    auto &buf = tl_buf.local();
    for (int Q = 0; Q < nsh; Q++) {
      if (atom[P] == atom[Q])
        continue;
      const auto dims = aux.env().two_center_helper_grad<Op::coulomb, kind>(
          {P, Q}, opt.optimizer_ptr(), buf.data(), nullptr);
      f(P, Q, dims[0], dims[1], buf.data());
    }
  });
}

// Parallel over auxiliary shells P, with the two-centre ip1 integrals against
// every Q on another atom (on the same atom both centres move together):
// f(P, Q, np, nq, buf) with buf[x][j][i] = (d_x P_i | Q_j), electron
// coordinate of P. Tasks own P.
template <typename F> void for_each_metric_ip1(IntegralEngine &aux, F &&f) {
  if (aux.is_spherical())
    for_each_metric_ip1_impl<ShellKind::Spherical>(aux, std::forward<F>(f));
  else
    for_each_metric_ip1_impl<ShellKind::Cartesian>(aux, std::forward<F>(f));
}

// sum_PQ c(P, Q, I_x) (dV/dX) over all atoms (3 x natom), for a contraction
// c of the block I_x = (d_x P | Q) (np x nq) that is symmetric in P and Q:
// each ordered pair gives the centre derivative of P, counted twice.
template <typename C>
Mat3N metric_gradient(IntegralEngine &aux, int natom, C &&contract) {
  const auto &atom = aux.aobasis().shell_to_atom();
  occ::parallel::thread_local_storage<Mat3N> tl(Mat3N::Zero(3, natom));
  for_each_metric_ip1(aux, [&](int P, int Q, int np, int nq,
                               const double *buf) {
    Vec3 s;
    for (int x = 0; x < 3; x++)
      s(x) = contract(P, Q,
                      Eigen::Map<const Mat>(buf + x * np * nq, np, nq));
    tl.local().col(atom[P]) -= 2.0 * s; // centre derivative is -ip1
  });
  Mat3N grad = Mat3N::Zero(3, natom);
  for (const auto &t : tl)
    grad += t;
  return grad;
}

// Contractions of a metric derivative block I (np x nq, aux shells P and Q)
// with d d^T for a vector d, or with a symmetric matrix M (naux x naux).
inline auto vector_metric_contraction(const IntegralEngine &aux,
                                      const Vec &d) {
  return [&first = aux.aobasis().first_bf(),
          &d](int P, int Q, const Eigen::Map<const Mat> &I) {
    return d.segment(first[P], I.rows())
        .dot(I * d.segment(first[Q], I.cols()));
  };
}

inline auto matrix_metric_contraction(const IntegralEngine &aux,
                                      const Mat &M) {
  return [&first = aux.aobasis().first_bf(),
          &M](int P, int Q, const Eigen::Map<const Mat> &I) {
    return M.block(first[P], first[Q], I.rows(), I.cols())
        .cwiseProduct(I)
        .sum();
  };
}

// d^T (dV/dX) d over all atoms (3 x natom).
inline Mat3N metric_term(IntegralEngine &aux, const Vec &d, int natom) {
  return metric_gradient(aux, natom, vector_metric_contraction(aux, d));
}

// sum_PQ M_PQ (dV/dX)_PQ over all atoms for a symmetric M (naux x naux).
inline Mat3N metric_term_matrix(IntegralEngine &aux, const Mat &M,
                                int natom) {
  return metric_gradient(aux, natom, matrix_metric_contraction(aux, M));
}

// Second derivatives (3N x 3N) of the metric term sum_PQ c(P, Q, V_PQ), for a
// contraction c of the block of shells P, Q as in metric_gradient, times
// scale. A two-centre integral depends only on R_P - R_Q, so with
// S = c(ipip1 on P) the blocks are PP = QQ = S, PQ = QP^T = -S; pairs on one
// atom do not contribute.
template <typename C>
Mat metric_hessian(IntegralEngine &aux, double scale, C &&contract) {
  const auto &basis = aux.aobasis();
  const auto &atom = basis.shell_to_atom();
  const int nsh = static_cast<int>(basis.size());
  const int n3 = 3 * static_cast<int>(basis.atoms().size());
  Intor *ipip1 = aux.is_spherical() ? libcint::int2c2e_ipip1_sph
                                    : libcint::int2c2e_ipip1_cart;
  auto &env = aux.env();
  struct TL {
    std::vector<double> buf;
    Mat H;
    OptHolder o;
  };
  occ::parallel::thread_local_storage<TL> tls([&]() {
    TL t{std::vector<double>(9 * basis.max_shell_size() *
                             basis.max_shell_size()),
         Mat::Zero(n3, n3)};
    libcint::int2c2e_ipip1_optimizer(&t.o.ptr, env.atom_data_ptr(),
                                     env.num_atoms(), env.basis_data_ptr(),
                                     env.num_basis(), env.env_data_ptr());
    return t;
  });
  occ::parallel::parallel_for(0, nsh, [&](int P) {
    auto &t = tls.local();
    const int np = basis[P].size();
    for (int Q = 0; Q < nsh; Q++) {
      if (atom[P] == atom[Q])
        continue;
      const int nq = basis[Q].size();
      int shls[2] = {P, Q};
      if (!call(ipip1, env, t.buf.data(), shls, t.o.ptr))
        continue;
      Mat3 S;
      for (int c = 0; c < 9; c++)
        S(c / 3, c % 3) = contract(
            P, Q, Eigen::Map<const Mat>(t.buf.data() + c * np * nq, np, nq));
      S *= scale;
      add_block(t.H, atom[P], atom[P], S);
      add_block(t.H, atom[Q], atom[Q], S);
      add_block(t.H, atom[P], atom[Q], -S);
      add_block(t.H, atom[Q], atom[P], -S.transpose());
    }
  });
  Mat H = Mat::Zero(n3, n3);
  for (const auto &t : tls)
    H += t.H;
  return H;
}

// Second derivatives (3N x 3N) of sum_{mn P} (mn|P) W_mn^P for weights W
// symmetric in m and n, times scale. Per unordered AO shell pair (p, q) and
// aux shell R not all on one atom, weights(p, q, R, nr, W) fills W[r]
// (np x nq) for the functions r of R; the pp and qq blocks come from ipip1 in
// both orders and the pq block from ipvip1, each counted twice for p != q
// (once per order of the pair), and the aux centre's by translational
// invariance.
template <ShellKind kind, typename Weights>
Mat three_center_hessian(IntegralEngine &ao, double scale,
                         Weights &&weights) {
  const auto &aobasis = ao.aobasis();
  const auto &auxbasis = ao.auxbasis();
  const auto &ao_atom = aobasis.shell_to_atom();
  const auto &aux_atom = auxbasis.shell_to_atom();
  const int nsh_ao = static_cast<int>(aobasis.size());
  const int nsh_aux = static_cast<int>(auxbasis.size());
  const int n3 = 3 * static_cast<int>(aobasis.atoms().size());
  constexpr bool sph = kind == ShellKind::Spherical;
  Intor *ipip1 =
      sph ? libcint::int3c2e_ipip1_sph : libcint::int3c2e_ipip1_cart;
  Intor *ipvip1 =
      sph ? libcint::int3c2e_ipvip1_sph : libcint::int3c2e_ipvip1_cart;
  const auto pairs = unordered_pairs(ao);
  const size_t bufsize = 9 * aobasis.max_shell_size() *
                         aobasis.max_shell_size() * auxbasis.max_shell_size();
  auto &env = ao.env();
  struct TL {
    std::vector<double> buf;
    std::vector<Mat> W;
    Mat H;
    OptHolder o_ipip1, o_ipvip1;
  };
  occ::parallel::thread_local_storage<TL> tls([&]() {
    TL t{std::vector<double>(bufsize), {}, Mat::Zero(n3, n3)};
    libcint::int3c2e_ipip1_optimizer(&t.o_ipip1.ptr, env.atom_data_ptr(),
                                     env.num_atoms(), env.basis_data_ptr(),
                                     env.num_basis(), env.env_data_ptr());
    libcint::int3c2e_ipvip1_optimizer(&t.o_ipvip1.ptr, env.atom_data_ptr(),
                                      env.num_atoms(), env.basis_data_ptr(),
                                      env.num_basis(), env.env_data_ptr());
    return t;
  });
  occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t k) {
    const auto [p, q] = pairs[k];
    auto &t = tls.local();
    const int A = ao_atom[p], B = ao_atom[q];
    const double fac = scale * ((p == q) ? 1.0 : 2.0);
    const int na = aobasis[p].size(), nb = aobasis[q].size();
    for (int R = 0; R < nsh_aux; R++) {
      const int C = aux_atom[R];
      if (A == B && B == C)
        continue;
      const int nr = auxbasis[R].size();
      if (t.W.size() < static_cast<size_t>(nr))
        t.W.resize(nr);
      weights(p, q, R, nr, t.W);
      // buf[c][r][j][i] = d2/dc (a_i b_j | R_r) contracted with W[r] in the
      // pair's (p, q) order, transposed when the buffer is for (q, p).
      auto contract = [&](bool swap) {
        Mat3 out;
        const int n1 = swap ? nb : na, n2 = swap ? na : nb;
        const size_t n = static_cast<size_t>(n1) * n2 * nr;
        for (int c = 0; c < 9; c++) {
          double s = 0.0;
          for (int r = 0; r < nr; r++) {
            Eigen::Map<const Mat> I(t.buf.data() + c * n + r * n1 * n2, n1,
                                    n2);
            s += swap ? t.W[r].transpose().cwiseProduct(I).sum()
                      : t.W[r].cwiseProduct(I).sum();
          }
          out(c / 3, c % 3) = s;
        }
        return out;
      };
      int s_pq[3] = {p, q, R + nsh_ao}, s_qp[3] = {q, p, R + nsh_ao};
      Mat3 AA = Mat3::Zero(), BB = Mat3::Zero(), AB = Mat3::Zero();
      if (call(ipip1, env, t.buf.data(), s_pq, t.o_ipip1.ptr))
        AA = fac * contract(false);
      if (call(ipip1, env, t.buf.data(), s_qp, t.o_ipip1.ptr))
        BB = fac * contract(true);
      if (call(ipvip1, env, t.buf.data(), s_pq, t.o_ipvip1.ptr))
        AB = fac * contract(false);
      add_operator_centre_blocks(t.H, A, B, C, AA, BB, AB);
    }
  });
  Mat H = Mat::Zero(n3, n3);
  for (const auto &t : tls)
    H += t.H;
  return H;
}

} // namespace occ::qm::detail::dfd
