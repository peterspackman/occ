#pragma once
#include "libcint_derivatives.h"
#include "quartet_derivatives.h"
#include "kernel_traits.h"
#include <occ/core/parallel.h>
#include <occ/qm/integral_engine.h>
#include <memory>
#include <occ/qm/shellblock_norm.h>

// Two-electron (Coulomb + exact exchange) nuclear gradient, contracted inside
// the integral loop, with no derivative Fock matrices formed. It
//
//  - walks unique quartets only (p >= q, r >= s, pq >= rs), generated inside
//    the parallel loop, so memory is O(natom) per thread;
//  - folds the eight-fold permutational degeneracy into the two-particle
//    density element Gamma, so each quartet is contracted once;
//  - evaluates (nabla mu nu | lambda sigma) only for the centres it needs: the
//    four shells fall into groups by atom, the largest group is obtained by
//    translational invariance, and a quartet whose shells all sit on one atom
//    contributes nothing and is skipped before any integral is evaluated.
//
// Energy convention and the shared quartet pieces: quartet_derivatives.h.

namespace occ::qm::detail {

template <ShellKind kind>
inline size_t max_cache_size_2e_ip1(cint::IntegralEnvironment &env,
                                    size_t nsh) {
  size_t result = 0;
  for (int i = 0; i < static_cast<int>(nsh); i++) {
    std::array<int, 4> shls{i, i, i, i};
    int n = 0;
    if constexpr (kind == ShellKind::Spherical) {
      n = libcint::int2e_ip1_sph(nullptr, nullptr, shls.data(),
                                 env.atom_data_ptr(), env.num_atoms(),
                                 env.basis_data_ptr(), env.num_basis(),
                                 env.env_data_ptr(), nullptr, nullptr);
    } else {
      n = libcint::int2e_ip1_cart(nullptr, nullptr, shls.data(),
                                  env.atom_data_ptr(), env.num_atoms(),
                                  env.basis_data_ptr(), env.num_basis(),
                                  env.env_data_ptr(), nullptr, nullptr);
    }
    result = std::max(result, static_cast<size_t>(n));
  }
  return result;
}

template <SpinorbitalKind sk, ShellKind kind>
Mat3N jk_gradient_contracted(cint::IntegralEnvironment &env,
                             const AOBasis &basis,
                             const ShellPairList &shellpairs,
                             const MolecularOrbitals &mo, double jf,
                             double kf, double precision, const Mat &Schwarz) {
  static_assert(sk != SpinorbitalKind::General,
                "general spinorbitals use the matrix path");
  constexpr bool U = sk == SpinorbitalKind::Unrestricted;
  const size_t nsh = basis.size();
  const size_t natom = basis.atoms().size();
  const auto &first_bf = basis.first_bf();
  const auto &shell_atom = basis.shell_to_atom();

  const auto tpd = two_particle_density<kind>(basis, mo);
  const auto pairs = unique_pairs(shellpairs);
  const size_t npair = pairs.size();

  const size_t cache_size = max_cache_size_2e_ip1<kind>(env, nsh);
  const size_t buf_size = env.buffer_size_2e(1);

  struct ThreadData {
    std::unique_ptr<occ::qm::cint::Optimizer> opt;
    std::vector<double> buffer, cache, gamma;
    Mat3N grad;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    return ThreadData{
        std::make_unique<occ::qm::cint::Optimizer>(env, Op::coulomb, 4, 1),
                      std::vector<double>(buf_size),
                      std::vector<double>(cache_size),
                      std::vector<double>(buf_size / 3),
                      Mat3N::Zero(3, natom)};
  });

  auto ip1 = [&](ThreadData &t, std::array<int, 4> shls) {
    return env.four_center_helper_grad<Op::coulomb, kind>(
        shls, t.opt->optimizer_ptr(), t.buffer.data(), t.cache.data());
  };

  // Process row ij of the pair-pair triangle: pairs[ij] x pairs[0..ij].
  auto process_row = [&](size_t ij) {
    auto &t = tls.local();
    const auto [p, q] = pairs[ij];
    const int np = basis[p].size(), nq = basis[q].size();
    const int bp = first_bf[p], bq = first_bf[q];
    const double pq_deg = (p == q) ? 1.0 : 2.0;
    for (size_t kl = 0; kl <= ij; kl++) {
      const auto [r, s] = pairs[kl];
      const auto qc = quartet_centres(shell_atom[p], shell_atom[q],
                                      shell_atom[r], shell_atom[s]);
      if (qc.trivial)
        continue;
      if (screened_out(tpd, Schwarz, p, q, r, s, jf, kf, precision))
        continue;

      const int nr = basis[r].size(), ns = basis[s].size();
      const int br = first_bf[r], bs = first_bf[s];
      const double deg = pq_deg * ((r == s) ? 1.0 : 2.0) *
                         ((p == r && q == s) ? 1.0 : 2.0);

      const int nn[4] = {np, nq, nr, ns};
      double *G = t.gamma.data();
      fill_gamma<U>(G, tpd, bp, bq, br, bs, nn, deg, jf, kf);

      Vec3 total = Vec3::Zero();
      for (int a = 0; a < qc.nneeded; a++) {
        const int centre = qc.needed[a];
        int perm[4];
        centre_first(centre, perm);
        const int sh[4] = {p, q, r, s};
        auto dims = ip1(t, {sh[perm[0]], sh[perm[1]], sh[perm[2]], sh[perm[3]]});
        if (dims[0] < 0)
          continue;
        double g[3];
        contract4<3>(G, t.buffer.data(), nn, perm, g);
        // d/dA phi(r - A) = -nabla phi: the integral is the negative.
        const Vec3 d(-g[0], -g[1], -g[2]);
        t.grad.col(qc.atoms[centre]) += d;
        total += d;
      }
      t.grad.col(qc.omit) -= total;
    }
  };

  // Rows near the end of the triangle carry the most quartets; hand those out
  // first so the stragglers at the end are cheap.
  occ::parallel::parallel_for(size_t(0), npair, [&](size_t k) {
    process_row(npair - 1 - k);
  });

  Mat3N result = Mat3N::Zero(3, natom);
  for (const auto &t : tls)
    result += t.grad;
  return result;
}

} // namespace occ::qm::detail
