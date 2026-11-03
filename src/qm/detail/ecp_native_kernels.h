#pragma once
// ECP operator, gradient, Hessian and derivative-matrix kernels on top of the
// native integrals in occ/ints/ecp.h.
//
// Every kernel loops over ECP centres (each atom carrying an ECP) and, per
// centre, over the significant shell pairs in parallel. The integrals depend
// only on the basis centres relative to C, so only derivatives with respect
// to basis centres away from C are evaluated; everything involving C comes
// from translational invariance (a basis centre on C contributes through C).
#include "kernel_traits.h"
#include <occ/core/parallel.h>
#include <occ/gto/gto.h>
#include <occ/ints/ecoeffs.h>
#include <occ/ints/ecp.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm::detail {

namespace ecp_native {

using ShellPairList = std::vector<std::vector<size_t>>;
using ShellKind = Shell::Kind;

inline std::vector<std::pair<int, int>>
flatten_pairs(const AOBasis &basis, const ShellPairList &shellpairs) {
  std::vector<std::pair<int, int>> pairs;
  if (shellpairs.empty()) {
    for (int p = 0; p < static_cast<int>(basis.size()); p++)
      for (int q = 0; q <= p; q++)
        pairs.push_back({p, q});
  } else {
    for (int p = 0; p < static_cast<int>(shellpairs.size()); p++)
      for (auto q : shellpairs[p])
        pairs.push_back({p, static_cast<int>(q)});
  }
  return pairs;
}

inline std::vector<Mat> cart2sph_matrices(const AOBasis &basis) {
  int lmax = 0;
  for (const auto &sh : basis.shells())
    lmax = std::max(lmax, static_cast<int>(sh.l));
  std::vector<Mat> c2s;
  for (int l = 0; l <= lmax; l++)
    c2s.push_back(occ::gto::cartesian_to_spherical_transformation_matrix(l));
  return c2s;
}

// Per-thread evaluator workspace, created on first use.
struct TLWorkspace {
  std::shared_ptr<occ::ints::EcpEvaluator::Workspace> ws;
  occ::ints::EcpEvaluator::Workspace &get(const occ::ints::EcpEvaluator &ev) {
    if (!ws)
      ws = ev.make_workspace();
    return *ws;
  }
};

} // namespace ecp_native

/// V_ecp in the AO basis.
template <ShellKind kind>
Mat ecp_native_operator_kernel(const AOBasis &basis,
                               const occ::ints::EcpEvaluator &ev,
                               const ShellPairList &shellpairs) {
  using namespace ecp_native;
  const auto pairs = flatten_pairs(basis, shellpairs);
  const auto c2s = kind == ShellKind::Spherical ? cart2sph_matrices(basis)
                                                : std::vector<Mat>{};
  const auto &first = basis.first_bf();
  const auto nbf = basis.nbf();
  occ::parallel::thread_local_storage<Mat> tl_v(Mat::Zero(nbf, nbf));
  occ::parallel::thread_local_storage<TLWorkspace> tl_ws;
  for (size_t c = 0; c < ev.centres().size(); c++) {
    auto cd = ev.prepare(static_cast<int>(c), 0);
    occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t i) {
      auto &ws = tl_ws.local().get(ev);
      const auto [p, q] = pairs[i];
      if (!ev.significant(*cd, p, q))
        return;
      MatRM blk(occ::ints::ncart(basis[p].l), occ::ints::ncart(basis[q].l));
      ev.compute(*cd, p, q, ws, blk);
      Mat &V = tl_v.local();
      if constexpr (kind == ShellKind::Spherical) {
        const Mat s = c2s[basis[p].l] * blk * c2s[basis[q].l].transpose();
        V.block(first[p], first[q], s.rows(), s.cols()) += s;
      } else {
        V.block(first[p], first[q], blk.rows(), blk.cols()) += blk;
      }
    });
  }
  Mat V = Mat::Zero(nbf, nbf);
  for (const auto &t : tl_v)
    V += t;
  // fill the upper triangle from the lower (pairs are q <= p)
  for (size_t p = 0; p < basis.size(); p++)
    for (size_t q = 0; q < p; q++) {
      const int r0 = first[p], c0 = first[q];
      const int nr = basis[p].size(), nc = basis[q].size();
      V.block(c0, r0, nc, nr) = V.block(r0, c0, nr, nc).transpose();
    }
  return V;
}

namespace ecp_native {

// D block in the Cartesian basis of the pair, so contractions can be done
// against raw Cartesian derivative blocks.
template <ShellKind kind>
inline MatRM cartesian_density_block(const Mat &D, const AOBasis &basis,
                                     const std::vector<Mat> &c2s, int p,
                                     int q) {
  const auto &first = basis.first_bf();
  const auto blk =
      D.block(first[p], first[q], basis[p].size(), basis[q].size());
  if constexpr (kind == ShellKind::Spherical)
    return c2s[basis[p].l].transpose() * blk * c2s[basis[q].l];
  else
    return blk;
}

} // namespace ecp_native

/// Tr(D dV_ecp/dR) per atom.
template <ShellKind kind>
Mat3N ecp_native_gradient_kernel(const AOBasis &basis,
                                 const occ::ints::EcpEvaluator &ev,
                                 const ShellPairList &shellpairs,
                                 const Mat &D) {
  using namespace ecp_native;
  const auto pairs = flatten_pairs(basis, shellpairs);
  const auto c2s = kind == ShellKind::Spherical ? cart2sph_matrices(basis)
                                                : std::vector<Mat>{};
  const auto &s2a = basis.shell_to_atom();
  const size_t natoms = basis.atoms().size();
  occ::parallel::thread_local_storage<Mat3N> tl_g(Mat3N::Zero(3, natoms));
  occ::parallel::thread_local_storage<TLWorkspace> tl_ws;
  for (size_t c = 0; c < ev.centres().size(); c++) {
    const int catom = ev.centres()[c].atom;
    auto cd = ev.prepare(static_cast<int>(c), 1);
    occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t i) {
      auto &ws = tl_ws.local().get(ev);
      const auto [p, q] = pairs[i];
      const int A = s2a[p], B = s2a[q];
      const bool want_a = A != catom, want_b = B != catom;
      if ((!want_a && !want_b) || !ev.significant(*cd, p, q))
        return;
      std::array<MatRM, 6> d;
      ev.compute_first_derivatives(*cd, p, q, want_a, want_b, ws, d);
      const MatRM Dc = cartesian_density_block<kind>(D, basis, c2s, p, q);
      const double weight = (p == q) ? 1.0 : 2.0;
      Mat3N &g = tl_g.local();
      for (int x = 0; x < 3; x++) {
        double ga = 0.0, gb = 0.0;
        if (want_a)
          ga = weight * Dc.cwiseProduct(d[x]).sum();
        if (want_b)
          gb = weight * Dc.cwiseProduct(d[3 + x]).sum();
        g(x, A) += ga;
        g(x, B) += gb;
        g(x, catom) -= ga + gb;
      }
    });
  }
  Mat3N g = Mat3N::Zero(3, natoms);
  for (const auto &t : tl_g)
    g += t;
  return g;
}

/// dV_ecp/dR for every nuclear coordinate, added into H1[3 * atom + xyz].
template <ShellKind kind>
void ecp_native_derivative_matrices_kernel(const AOBasis &basis,
                                           const occ::ints::EcpEvaluator &ev,
                                           const ShellPairList &shellpairs,
                                           std::vector<Mat> &H1) {
  using namespace ecp_native;
  const auto pairs = flatten_pairs(basis, shellpairs);
  const auto c2s = kind == ShellKind::Spherical ? cart2sph_matrices(basis)
                                                : std::vector<Mat>{};
  const auto &s2a = basis.shell_to_atom();
  const auto &first = basis.first_bf();
  const auto nbf = basis.nbf();
  const size_t n3 = H1.size();
  occ::parallel::thread_local_storage<std::vector<Mat>> tl(
      [&]() { return std::vector<Mat>(n3, Mat::Zero(nbf, nbf)); });
  occ::parallel::thread_local_storage<TLWorkspace> tl_ws;
  for (size_t c = 0; c < ev.centres().size(); c++) {
    const int catom = ev.centres()[c].atom;
    auto cd = ev.prepare(static_cast<int>(c), 1);
    occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t i) {
      auto &ws = tl_ws.local().get(ev);
      const auto [p, q] = pairs[i];
      const int A = s2a[p], B = s2a[q];
      const bool want_a = A != catom, want_b = B != catom;
      if ((!want_a && !want_b) || !ev.significant(*cd, p, q))
        return;
      std::array<MatRM, 6> d;
      ev.compute_first_derivatives(*cd, p, q, want_a, want_b, ws, d);
      auto &out = tl.local();
      const int r0 = first[p], c0 = first[q];
      auto add = [&](int atom, int x, const MatRM &raw) {
        Mat blk;
        if constexpr (kind == ShellKind::Spherical)
          blk = c2s[basis[p].l] * raw * c2s[basis[q].l].transpose();
        else
          blk = raw;
        Mat &M = out[3 * atom + x];
        M.block(r0, c0, blk.rows(), blk.cols()) += blk;
        if (p != q)
          M.block(c0, r0, blk.cols(), blk.rows()) += blk.transpose();
      };
      for (int x = 0; x < 3; x++) {
        MatRM dc = MatRM::Zero(d[want_a ? x : 3 + x].rows(),
                               d[want_a ? x : 3 + x].cols());
        if (want_a) {
          add(A, x, d[x]);
          dc -= d[x];
        }
        if (want_b) {
          add(B, x, d[3 + x]);
          dc -= d[3 + x];
        }
        add(catom, x, dc);
      }
    });
  }
  for (const auto &t : tl)
    for (size_t k = 0; k < n3; k++)
      H1[k] += t[k];
}

/// Skeleton ECP Hessian sum P d2 V_ecp / dR dR (3N x 3N).
template <ShellKind kind>
Mat ecp_native_hessian_kernel(const AOBasis &basis,
                              const occ::ints::EcpEvaluator &ev,
                              const ShellPairList &shellpairs, const Mat &P) {
  using namespace ecp_native;
  const auto pairs = flatten_pairs(basis, shellpairs);
  const auto c2s = kind == ShellKind::Spherical ? cart2sph_matrices(basis)
                                                : std::vector<Mat>{};
  const auto &s2a = basis.shell_to_atom();
  const int natom = static_cast<int>(basis.atoms().size());
  occ::parallel::thread_local_storage<Mat> tl(
      [&]() { return Mat(Mat::Zero(3 * natom, 3 * natom)); });
  occ::parallel::thread_local_storage<TLWorkspace> tl_ws;
  for (size_t c = 0; c < ev.centres().size(); c++) {
    const int C = ev.centres()[c].atom;
    auto cd = ev.prepare(static_cast<int>(c), 2);
    occ::parallel::parallel_for(size_t(0), pairs.size(), [&](size_t i) {
      auto &ws = tl_ws.local().get(ev);
      const auto [p, q] = pairs[i];
      const int A = s2a[p], B = s2a[q];
      const bool a_on_c = A == C, b_on_c = B == C;
      if ((a_on_c && b_on_c) || !ev.significant(*cd, p, q))
        return;
      std::array<MatRM, 6> aa, bb;
      std::array<MatRM, 9> ab;
      ev.compute_second_derivatives(*cd, p, q, !a_on_c, !a_on_c && !b_on_c,
                                    !b_on_c, ws, aa, ab, bb);
      const MatRM Pc = cartesian_density_block<kind>(P, basis, c2s, p, q);
      const double weight = (p == q) ? 1.0 : 2.0;
      auto contract = [&](const MatRM &blk) {
        return weight * Pc.cwiseProduct(blk).sum();
      };
      Mat &H = tl.local();
      auto add = [&](int a, int b, const Mat3 &M) {
        H.block<3, 3>(3 * a, 3 * b) += M;
      };
      if (!a_on_c && !b_on_c) {
        Mat3 AA, BB, AB;
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++) {
            AA(i, j) = contract(aa[occ::ints::ecp_sym_index(i, j)]);
            BB(i, j) = contract(bb[occ::ints::ecp_sym_index(i, j)]);
            AB(i, j) = contract(ab[3 * i + j]);
          }
        const Mat3 BA = AB.transpose();
        const Mat3 AC = -(AA + AB), BC = -(BA + BB);
        add(A, A, AA);
        add(B, B, BB);
        add(A, B, AB);
        add(B, A, BA);
        add(A, C, AC);
        add(C, A, AC.transpose());
        add(B, C, BC);
        add(C, B, BC.transpose());
        add(C, C, AA + AB + BA + BB);
      } else {
        // One basis centre on C: the integral depends only on the other
        // centre X relative to C.
        const int X = a_on_c ? B : A;
        const auto &xx = a_on_c ? bb : aa;
        Mat3 XX;
        for (int i = 0; i < 3; i++)
          for (int j = 0; j < 3; j++)
            XX(i, j) = contract(xx[occ::ints::ecp_sym_index(i, j)]);
        add(X, X, XX);
        add(X, C, -XX);
        add(C, X, -XX);
        add(C, C, XX);
      }
    });
  }
  Mat H = Mat::Zero(3 * natom, 3 * natom);
  for (const auto &t : tl)
    H += t;
  return H;
}

} // namespace occ::qm::detail
