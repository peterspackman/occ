#include "detail/df_kernels.h"
#include <algorithm>
#include <fmt/core.h>
#include <new>
#include <numeric>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/qm/df_packed_store.h>
#include <stdexcept>

namespace occ::qm {

using ShellKind = gto::Shell::Kind;
using IntegralResult = IntegralEngine::IntegralResult<3>;

size_t PackedDFStore::bytes_for(const gto::AOBasis &basis,
                                const ShellPairList &shellpairs, size_t naux) {
  size_t rows = 0;
  for (size_t p = 0; p < shellpairs.size(); p++)
    for (size_t q : shellpairs[p])
      rows += basis[p].size() * basis[q].size();
  return rows * naux * sizeof(double);
}

void PackedDFStore::clear() {
  m_ints.resize(0, 0);
  m_blocks.clear();
  m_unstored.clear();
  m_built = false;
  m_complete = false;
}

void PackedDFStore::build(IntegralEngine &engine, const CoulombMetric &metric,
                          size_t max_bytes) {
  clear();
  m_built = true;
  const auto &ao = engine.aobasis();
  const auto &aux = engine.auxbasis();
  const auto &sp = engine.shellpairs();
  const auto &fbf = ao.first_bf();
  const size_t nsh = ao.size();
  m_nbf = ao.nbf();
  m_naux = aux.nbf();

  // Pairs ranked by a rough estimate of the cost of recomputing them per
  // stored byte: contraction depth and angular momentum make a pair dear, its
  // size makes it big. A heuristic, not a measured cost model.
  struct Candidate {
    int p, q;
    double priority;
  };
  std::vector<Candidate> candidates;
  for (size_t p = 0; p < nsh; p++)
    for (size_t q : sp[p]) {
      const double n = double(ao[p].size()) * ao[q].size();
      const double cost = double(ao[p].num_primitives()) *
                          ao[q].num_primitives() * (ao[p].l + ao[q].l + 1);
      candidates.push_back({int(p), int(q), cost / n});
    }
  const size_t row_bytes = m_naux * sizeof(double);
  std::vector<char> keep(candidates.size(), 1);
  if (bytes_for(ao, sp, m_naux) > max_bytes) {
    std::vector<size_t> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return candidates[a].priority > candidates[b].priority;
    });
    size_t used = 0;
    for (size_t k : order) {
      const auto &c = candidates[k];
      const size_t b = ao[c.p].size() * ao[c.q].size() * row_bytes;
      keep[k] = used + b <= max_bytes;
      if (keep[k])
        used += b;
    }
  }

  m_unstored.assign(nsh, {});
  ShellPairList stored(nsh);
  std::vector<int> block_of(nsh * nsh, -1);
  size_t rows = 0;
  for (size_t k = 0; k < candidates.size(); k++) {
    const auto &c = candidates[k];
    if (!keep[k]) {
      m_unstored[c.p].push_back(c.q);
      continue;
    }
    const int np = ao[c.p].size(), nq = ao[c.q].size();
    block_of[size_t(c.p) * nsh + c.q] = m_blocks.size();
    m_blocks.push_back({c.p, c.q, fbf[c.p], fbf[c.q], np, nq, rows});
    stored[c.p].push_back(c.q);
    rows += size_t(np) * nq;
  }
  m_complete = m_blocks.size() == candidates.size();
  if (m_blocks.empty())
    return;

  occ::log::info("Storing packed 3-centre integrals: {} of {} shell pairs, "
                 "{:.0f} MiB (dense would be {:.0f} MiB)",
                 m_blocks.size(), candidates.size(),
                 rows * row_bytes / 1048576.0,
                 double(m_nbf) * m_nbf * row_bytes / 1048576.0);
  // Uninitialised, then zeroed one column per task: the first touch of a
  // multi-GiB array is what costs, and it parallelises.
  try {
    m_ints.resize(rows, m_naux);
  } catch (const std::bad_alloc &) {
    throw std::runtime_error(fmt::format(
        "could not allocate {:.0f} MiB for the stored 3-centre integrals "
        "({} basis functions, {} auxiliary). Use the direct density fitting "
        "policy, or a smaller store memory limit.",
        rows * row_bytes / 1048576.0, m_nbf, m_naux));
  }
  occ::parallel::parallel_for(size_t(0), m_naux,
                              [&](size_t P) { m_ints.col(P).setZero(); });

  auto store = [&](const IntegralResult &args) {
    const Block &b =
        m_blocks[block_of[size_t(args.shell[0]) * nsh + args.shell[1]]];
    const size_t n = size_t(args.dims[0]) * args.dims[1];
    for (int P = 0; P < args.dims[2]; P++)
      std::copy(args.buffer + P * n, args.buffer + (P + 1) * n,
                m_ints.col(args.bf[2] + P).data() + b.row);
  };
  cint::Optimizer opt(engine.env(), cint::Operator::coulomb, 3);
  if (engine.is_spherical())
    detail::compute_three_center_integrals_tbb<ShellKind::Spherical>(
        store, engine.env(), ao, aux, stored, opt);
  else
    detail::compute_three_center_integrals_tbb<ShellKind::Cartesian>(
        store, engine.env(), ao, aux, stored, opt);
  if (m_complete)
    fold(metric);
}

void PackedDFStore::fold(const CoulombMetric &metric) {
  occ::log::debug("Folding the Coulomb metric into the packed integrals");
  const size_t rows = m_ints.rows(), chunk = 512;
  const size_t nchunks = (rows + chunk - 1) / chunk;
  // The eigen fallback of the metric drops vectors: fewer columns survive
  Eigen::Index kept = 0;
  occ::parallel::parallel_for(size_t(0), nchunks, [&](size_t c) {
    const size_t r0 = c * chunk, nr = std::min(chunk, rows - r0);
    Mat t = metric.half_inverse_apply(m_ints.middleRows(r0, nr).transpose());
    m_ints.middleRows(r0, nr).leftCols(t.rows()) = t.transpose();
    if (c == 0)
      kept = t.rows();
  });
  m_naux = kept;
}

void PackedDFStore::unpack(size_t P, Mat &dense) const {
  const double *col = m_ints.col(P).data();
  for (const auto &b : m_blocks) {
    Eigen::Map<const Mat> M(col + b.row, b.np, b.nq);
    dense.block(b.fp, b.fq, b.np, b.nq) = M;
    if (b.p != b.q)
      dense.block(b.fq, b.fp, b.nq, b.np) = M.transpose();
  }
}

Mat PackedDFStore::contract(const std::vector<Mat> &densities) const {
  const size_t nc = densities.size(), rows = m_ints.rows();
  Mat d(rows, nc);
  for (size_t c = 0; c < nc; c++) {
    const Mat &D = densities[c];
    for (const auto &b : m_blocks) {
      const double w = b.p == b.q ? 1.0 : 2.0;
      for (int j = 0; j < b.nq; j++)
        for (int i = 0; i < b.np; i++)
          d(b.row + i + size_t(b.np) * j, c) = w * D(b.fp + i, b.fq + j);
    }
  }
  Mat g(m_naux, nc);
  occ::parallel::parallel_for(size_t(0), m_naux, [&](size_t P) {
    g.row(P) = m_ints.col(P).transpose() * d;
  });
  return g;
}

std::vector<Mat> PackedDFStore::expand(const Mat &x) const {
  const size_t nc = x.cols(), rows = m_ints.rows(), chunk = 4096;
  Mat jp(rows, nc);
  occ::parallel::parallel_for(
      size_t(0), (rows + chunk - 1) / chunk, [&](size_t c) {
        const size_t r0 = c * chunk, nr = std::min(chunk, rows - r0);
        jp.middleRows(r0, nr).noalias() =
            m_ints.middleRows(r0, nr).leftCols(m_naux) * x.topRows(m_naux);
      });
  std::vector<Mat> J(nc, Mat::Zero(m_nbf, m_nbf));
  for (size_t c = 0; c < nc; c++)
    for (const auto &b : m_blocks) {
      Eigen::Map<const Mat> M(jp.col(c).data() + b.row, b.np, b.nq);
      J[c].block(b.fp, b.fq, b.np, b.nq) = M;
      if (b.p != b.q)
        J[c].block(b.fq, b.fp, b.nq, b.np) = M.transpose();
    }
  return J;
}

std::vector<Mat>
PackedDFStore::exchange(const std::vector<Mat> &C,
                        const std::vector<std::pair<int, int>> &pairs) const {
  const size_t nbf = m_nbf, ns = C.size(), np = pairs.size();
  occ::parallel::thread_local_storage<Mat> dense(
      [nbf]() { return Mat::Zero(nbf, nbf); });
  occ::parallel::thread_local_storage<std::vector<Mat>> W(
      [&]() { return std::vector<Mat>(ns); });
  occ::parallel::thread_local_storage<std::vector<Mat>> K_local(
      [&]() { return std::vector<Mat>(np, Mat::Zero(nbf, nbf)); });
  occ::parallel::parallel_for(size_t(0), m_naux, [&](size_t P) {
    Mat &B = dense.local();
    unpack(P, B);
    auto &w = W.local();
    for (size_t s = 0; s < ns; s++)
      w[s].noalias() = B * C[s];
    auto &K = K_local.local();
    for (size_t k = 0; k < np; k++) {
      const auto [s, t] = pairs[k];
      if (s == t)
        K[k].selfadjointView<Eigen::Lower>().rankUpdate(w[s]);
      else
        K[k].noalias() += w[s] * w[t].transpose();
    }
  });
  std::vector<Mat> K(np, Mat::Zero(nbf, nbf));
  for (const auto &local : K_local)
    for (size_t k = 0; k < np; k++)
      K[k] += local[k];
  for (size_t k = 0; k < np; k++)
    if (pairs[k].first == pairs[k].second)
      K[k] = K[k].selfadjointView<Eigen::Lower>();
  return K;
}

Mat PackedDFStore::half_transform(Eigen::Ref<const Mat> C_left,
                                  Eigen::Ref<const Mat> C_right) const {
  const Eigen::Index nL = C_left.cols(), nR = C_right.cols();
  const size_t nbf = m_nbf;
  occ::parallel::thread_local_storage<Mat> dense(
      [nbf]() { return Mat::Zero(nbf, nbf); });
  Mat B(nL * nR, m_naux);
  occ::parallel::parallel_for(size_t(0), m_naux, [&](size_t P) {
    Mat &BP = dense.local();
    unpack(P, BP);
    Eigen::Map<Mat>(B.col(P).data(), nR, nL).noalias() =
        C_right.transpose() * (BP * C_left);
  });
  return B;
}

} // namespace occ::qm
