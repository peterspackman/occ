#include <occ/core/parallel.h>
#include <occ/ints/boys.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

void IntegralEngine::ensure_esp_initialized() const {
  if (m_esp_initialized)
    return;

  m_esp_evaluator = std::make_unique<occ::ints::ESPEvaluator<double>>(
      occ::ints::boys().table());

  const auto &shells = m_aobasis.shells();
  bool spherical = is_spherical();

  // Add all significant shell pairs (matching shellpair list order)
  for (size_t p = 0; p < m_shellpairs.size(); p++) {
    for (size_t q : m_shellpairs[p]) {
      const auto &sh_p = shells[p];
      const auto &sh_q = shells[q];
      m_esp_evaluator->add_shell_pair(
          sh_p.l, sh_q.l, sh_p.num_primitives(), sh_q.num_primitives(),
          sh_p.exponents.data(), sh_q.exponents.data(),
          sh_p.contraction_coefficients.col(0).data(),
          sh_q.contraction_coefficients.col(0).data(), sh_p.origin.data(),
          sh_q.origin.data(), spherical);
    }
  }
  m_esp_initialized = true;
}

Vec IntegralEngine::electric_potential_mmd(const MolecularOrbitals &mo,
                                           const Mat3N &points) const {
  ensure_esp_initialized();

  const size_t npts = points.cols();
  const auto &shells = m_aobasis.shells();
  const auto &first_bf = m_aobasis.first_bf();
  const size_t nbf = m_aobasis.nbf();

  // Spin-summed density, and per shell pair a screen on the summed
  // magnitudes of every spin block, so cancellation between blocks cannot
  // hide a pair.
  Mat Dtot = Mat::Zero(nbf, nbf);
  std::vector<Mat> block_store;
  switch (mo.kind) {
  case SpinorbitalKind::Restricted:
    block_store = {mo.D};
    break;
  case SpinorbitalKind::Unrestricted:
    block_store = {qm::block::a(mo.D), qm::block::b(mo.D)};
    break;
  case SpinorbitalKind::General:
    block_store = {qm::block::aa(mo.D), qm::block::ab(mo.D),
                   qm::block::ba(mo.D), qm::block::bb(mo.D)};
    break;
  }
  for (const auto &Db : block_store)
    Dtot += Db;

  struct Pair {
    int bp, np, bq, nq;
    size_t esp_idx;
    double scale;
  };
  std::vector<Pair> pairs;
  size_t esp_idx = 0;
  for (size_t p = 0; p < m_shellpairs.size(); p++) {
    for (size_t q : m_shellpairs[p]) {
      const int bp = first_bf[p], bq = first_bf[q];
      const int np = shells[p].size(), nq = shells[q].size();
      double dsum = 0.0;
      for (const auto &Db : block_store) {
        dsum += Db.block(bp, bq, np, nq).cwiseAbs().sum();
        if (p != q)
          dsum += Db.block(bq, bp, nq, np).cwiseAbs().sum();
      }
      // Off-diagonal pairs stand for both (p, q) and (q, p).
      if (dsum >= 1e-12)
        pairs.push_back({bp, np, bq, nq, esp_idx, p == q ? 1.0 : 2.0});
      esp_idx++;
    }
  }

  // Point-fastest ESP kernel: X[ab * npad + g] over a padded batch.
  constexpr int PAD = occ::ints::ESPEvaluator<double>::soa_padding;
  constexpr size_t BATCH_SIZE = 256;
  const size_t num_batches = (npts + BATCH_SIZE - 1) / BATCH_SIZE;
  Vec potential = Vec::Zero(npts);

  struct Scratch {
    std::vector<double> x, y, z, X;
  };
  occ::parallel::thread_local_storage<Scratch> scratch_local;

  occ::parallel::parallel_for(size_t(0), num_batches, [&](size_t batch_idx) {
    const size_t start = batch_idx * BATCH_SIZE;
    const size_t count = std::min(BATCH_SIZE, npts - start);
    const int npad = static_cast<int>((count + PAD - 1) / PAD * PAD);
    auto &ws = scratch_local.local();
    ws.x.resize(npad);
    ws.y.resize(npad);
    ws.z.resize(npad);
    for (int g = 0; g < npad; g++) {
      const size_t c = start + std::min<size_t>(g, count - 1);
      ws.x[g] = points(0, c);
      ws.y[g] = points(1, c);
      ws.z[g] = points(2, c);
    }
    std::vector<double> pot(count, 0.0);
    for (const auto &pr : pairs) {
      const int nab = pr.np * pr.nq;
      ws.X.resize(static_cast<size_t>(nab) * npad);
      m_esp_evaluator->evaluate_soa(pr.esp_idx, ws.x.data(), ws.y.data(),
                                    ws.z.data(), npad, ws.X.data(), npad);
      // V(pt) += sum_ab D(a, b) (ab|1/r|pt)
      for (int a = 0; a < pr.np; a++)
        for (int b = 0; b < pr.nq; b++) {
          const double d = pr.scale * Dtot(pr.bp + a, pr.bq + b);
          if (d == 0.0)
            continue;
          const double *Xab = ws.X.data() + (a * pr.nq + b) * npad;
          for (size_t g = 0; g < count; g++)
            pot[g] += d * Xab[g];
        }
    }
    // Batches own disjoint point ranges.
    for (size_t g = 0; g < count; g++)
      potential(start + g) = pot[g];
  });

  // Factor of 2 to match libcint convention (see three_center_kernels.h)
  // Negative sign because our ESP integrals have opposite sign convention
  return -2.0 * potential;
}

} // namespace occ::qm
