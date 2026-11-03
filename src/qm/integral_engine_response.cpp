// Two-electron response matrices for many densities at once:
//
//   G[D1] = jf * 2 J[D1] - kf * K[D1]      (restricted, D1 per spin)
//
// the linear response of occ's restricted Fock build, which is what a CPHF
// solver applies on every iteration -- for a Hessian, to 3N densities at a
// time. The generic batched J/K kernel digests each integral into each
// density's own matrices one element at a time, which for tens of densities
// is pure memory traffic. Here densities are stored with the density index
// fastest, so every shell quartet's J and K updates are small dense matrix
// products over all densities at once.

#include "detail/kernel_traits.h"
#include "detail/libcint_derivatives.h"
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/shellblock_norm.h>

namespace occ::qm {

using ShellKind = Shell::Kind;
using Op = cint::Operator;

namespace {

using RowMat =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Density-major storage in shell-pair blocks: the block of shells (r, s) is
// rows [offset(r, s), offset(r, s) + nr * ns), element (i, j) of the block at
// row offset + i + nr * j, all densities along the row. Every gather and
// scatter of a quartet is then a contiguous block.
struct BlockLayout {
  int nsh{0};
  std::vector<Eigen::Index> offset;
  Eigen::Index rows{0};
  BlockLayout(const AOBasis &basis) : nsh(static_cast<int>(basis.size())) {
    offset.resize(static_cast<size_t>(nsh) * nsh);
    for (int r = 0; r < nsh; r++)
      for (int s = 0; s < nsh; s++) {
        offset[r * nsh + s] = rows;
        rows += static_cast<Eigen::Index>(basis[r].size()) * basis[s].size();
      }
  }
  inline Eigen::Index at(int r, int s) const { return offset[r * nsh + s]; }
};

template <ShellKind kind>
std::vector<Mat> response_kernel(cint::IntegralEnvironment &env,
                                 const AOBasis &basis,
                                 const std::vector<std::vector<size_t>> &sp,
                                 const std::vector<Mat> &D, int nj, double jf,
                                 double kf, double precision,
                                 const Mat &Schwarz) {
  const int nbf = static_cast<int>(basis.nbf());
  const int nd = static_cast<int>(D.size());
  const auto &first_bf = basis.first_bf();
  const bool do_k = kf != 0.0;

  const auto pairs = detail::unique_pairs(sp);
  const size_t npair = pairs.size();

  // Total-density convention (libint-style digestion): Dt = 2 D1 gives
  // J[Dt] - 1/2 K[Dt] = 2 J[D1] - K[D1] after the final symmetrisation.
  const BlockLayout L(basis);
  const int nsh = L.nsh;
  RowMat Dall(L.rows, nd);
  Mat Dnorm;
  for (int k = 0; k < nd; k++) {
    for (int r = 0; r < nsh; r++)
      for (int s2 = 0; s2 < nsh; s2++) {
        const int nr = basis[r].size(), ns = basis[s2].size();
        const Eigen::Index off = L.at(r, s2);
        for (int j = 0; j < ns; j++)
          for (int i = 0; i < nr; i++)
            Dall(off + i + nr * j, k) =
                2.0 * D[k](first_bf[r] + i, first_bf[s2] + j);
      }
    Mat n = shellblock_norm<SpinorbitalKind::Restricted, kind>(basis, D[k]);
    Dnorm = (k == 0) ? n : Mat(Dnorm.cwiseMax(n));
  }
  const bool screen = Schwarz.size() > 0;

  const size_t nmax4 = env.buffer_size_2e(0);
  struct ThreadData {
    std::unique_ptr<cint::Optimizer> opt;
    std::vector<double> buf, kbuf1, kbuf2;
    RowMat G;
  };
  occ::parallel::thread_local_storage<ThreadData> tls([&]() {
    return ThreadData{std::make_unique<cint::Optimizer>(env, Op::coulomb, 4),
                      std::vector<double>(nmax4), std::vector<double>(nmax4),
                      std::vector<double>(nmax4),
                      RowMat::Zero(L.rows, nd)};
  });

  auto process_row = [&](size_t ij) {
    auto &t = tls.local();
    const auto [p, q] = pairs[ij];
    const int np = basis[p].size(), nq = basis[q].size();
    for (size_t kl = 0; kl <= ij; kl++) {
      const auto [r, s] = pairs[kl];
      if (screen) {
        const double dmax =
            std::max({Dnorm(p, q), Dnorm(r, s), Dnorm(p, r), Dnorm(q, s),
                      Dnorm(p, s), Dnorm(q, r)});
        if (dmax * Schwarz(p, q) * Schwarz(r, s) < precision)
          continue;
      }
      std::array<int, 4> shls{p, q, r, s};
      auto dims = env.four_center_helper<Op::coulomb, kind>(
          shls, t.opt->optimizer_ptr(), t.buf.data(), nullptr);
      if (dims[0] < 0)
        continue;
      const int nr = basis[r].size(), ns = basis[s].size();
      const double deg = ((p == q) ? 1.0 : 2.0) * ((r == s) ? 1.0 : 2.0) *
                         ((ij == kl) ? 1.0 : 2.0);

      // Coulomb: I as (pq) x (rs), column-major straight from libcint.
      Eigen::Map<const Mat> IJ(t.buf.data(), np * nq, nr * ns);
      auto blk = [&](RowMat &X, int a, int b) {
        return X.middleRows(L.at(a, b),
                            static_cast<Eigen::Index>(basis[a].size()) *
                                basis[b].size());
      };
      auto cblk = [&](int a, int b) {
        return Dall.middleRows(L.at(a, b),
                               static_cast<Eigen::Index>(basis[a].size()) *
                                   basis[b].size());
      };
      if (jf != 0.0 && nj > 0) {
        const double cj = jf * deg;
        blk(t.G, p, q).leftCols(nj).noalias() +=
            cj * (IJ * cblk(r, s).leftCols(nj));
        blk(t.G, r, s).leftCols(nj).noalias() +=
            cj * (IJ.transpose() * cblk(p, q).leftCols(nj));
      }
      if (do_k) {
        // K1: (p r) x (q s);  K2: (p s) x (q r)
        double *k1 = t.kbuf1.data(), *k2 = t.kbuf2.data();
        for (int d = 0, n = 0; d < ns; d++)
          for (int c = 0; c < nr; c++)
            for (int b = 0; b < nq; b++)
              for (int a = 0; a < np; a++, n++) {
                const double v = t.buf[n];
                k1[(a + np * c) + (np * nr) * (b + nq * d)] = v;
                k2[(a + np * d) + (np * ns) * (b + nq * c)] = v;
              }
        const double ck = -0.25 * kf * deg;
        Eigen::Map<const Mat> K1(k1, np * nr, nq * ns);
        Eigen::Map<const Mat> K2(k2, np * ns, nq * nr);
        blk(t.G, p, r).noalias() += ck * (K1 * cblk(q, s));
        blk(t.G, q, s).noalias() += ck * (K1.transpose() * cblk(p, r));
        blk(t.G, p, s).noalias() += ck * (K2 * cblk(q, r));
        blk(t.G, q, r).noalias() += ck * (K2.transpose() * cblk(p, s));
      }
    }
  };

  occ::parallel::parallel_for(size_t(0), npair, [&](size_t k) {
    process_row(npair - 1 - k);
  });

  RowMat Gall = RowMat::Zero(L.rows, nd);
  for (const auto &t : tls)
    Gall += t.G;
  std::vector<Mat> result(nd, Mat(nbf, nbf));
  for (int k = 0; k < nd; k++) {
    Mat &G = result[k];
    for (int r = 0; r < nsh; r++)
      for (int s2 = 0; s2 < nsh; s2++) {
        const int nr = basis[r].size(), ns = basis[s2].size();
        const Eigen::Index off = L.at(r, s2);
        for (int j = 0; j < ns; j++)
          for (int i = 0; i < nr; i++)
            G(first_bf[r] + i, first_bf[s2] + j) = Gall(off + i + nr * j, k);
      }
    G = 0.25 * (G + G.transpose()).eval();
  }
  return result;
}

} // namespace

std::vector<Mat>
IntegralEngine::coulomb_exchange_response(const std::vector<Mat> &D, double jf,
                                          double kf, const Mat &Schwarz,
                                          size_t exchange_only) const {
  if (D.empty())
    return {};
  occ::timing::start(occ::timing::category::fock);
  // Bound the per-thread accumulators: nthreads x nbf^2 x chunk doubles.
  const size_t nbf = m_aobasis.nbf();
  const size_t nthreads = std::max(1, occ::parallel::get_num_threads());
  const size_t budget = size_t(1) << 30; // bytes
  const size_t chunk = std::max<size_t>(
      1, std::min(D.size(), budget / (nthreads * nbf * nbf * sizeof(double))));
  const size_t ncoulomb = D.size() - std::min(D.size(), exchange_only);
  std::vector<Mat> result;
  result.reserve(D.size());
  for (size_t start = 0; start < D.size(); start += chunk) {
    std::vector<Mat> part(D.begin() + start,
                          D.begin() + std::min(D.size(), start + chunk));
    const int nj = static_cast<int>(
        std::min(part.size(), ncoulomb - std::min(ncoulomb, start)));
    auto G = is_spherical()
                 ? response_kernel<ShellKind::Spherical>(
                       m_env, m_aobasis, m_shellpairs, part, nj, jf, kf,
                       m_precision, Schwarz)
                 : response_kernel<ShellKind::Cartesian>(
                       m_env, m_aobasis, m_shellpairs, part, nj, jf, kf,
                       m_precision, Schwarz);
    for (auto &g : G)
      result.push_back(std::move(g));
  }
  occ::timing::stop(occ::timing::category::fock);
  return result;
}

} // namespace occ::qm
