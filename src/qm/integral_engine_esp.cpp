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

  Vec potential = Vec::Zero(npts);
  constexpr size_t BATCH_SIZE = 256;

  // Process in batches for cache efficiency
  occ::parallel::thread_local_storage<Vec> pot_local(
      [npts]() { return Vec::Zero(npts); });

  size_t num_batches = (npts + BATCH_SIZE - 1) / BATCH_SIZE;

  // Thread-local workspace buffers
  using MatRM =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  occ::parallel::thread_local_storage<MatRM> workspace_local;
  occ::parallel::thread_local_storage<MatRM> integrals_local;

  occ::parallel::parallel_for(size_t(0), num_batches, [&](size_t batch_idx) {
    size_t start = batch_idx * BATCH_SIZE;
    size_t count = std::min(BATCH_SIZE, npts - start);

    auto pts_block = points.middleCols(start, count);
    auto &pot = pot_local.local();
    auto &workspace = workspace_local.local();
    auto &integrals = integrals_local.local();

    // For each shell pair, evaluate ESP integrals and contract with density
    size_t esp_idx = 0;
    for (size_t p = 0; p < m_shellpairs.size(); p++) {
      for (size_t q : m_shellpairs[p]) {
        int bf_p = first_bf[p];
        int bf_q = first_bf[q];
        int size_p = shells[p].size();
        int size_q = shells[q].size();

        // Get density block and check if negligible
        // Handle different spinorbital kinds
        double D_pq_sum = 0.0;
        switch (mo.kind) {
        case SpinorbitalKind::Restricted:
          D_pq_sum = mo.D.block(bf_p, bf_q, size_p, size_q).cwiseAbs().sum();
          if (p != q) {
            D_pq_sum +=
                mo.D.block(bf_q, bf_p, size_q, size_p).cwiseAbs().sum();
          }
          break;
        case SpinorbitalKind::Unrestricted: {
          auto Da = qm::block::a(mo.D);
          auto Db = qm::block::b(mo.D);
          D_pq_sum = Da.block(bf_p, bf_q, size_p, size_q).cwiseAbs().sum() +
                     Db.block(bf_p, bf_q, size_p, size_q).cwiseAbs().sum();
          if (p != q) {
            D_pq_sum += Da.block(bf_q, bf_p, size_q, size_p).cwiseAbs().sum() +
                        Db.block(bf_q, bf_p, size_q, size_p).cwiseAbs().sum();
          }
          break;
        }
        case SpinorbitalKind::General: {
          auto Daa = qm::block::aa(mo.D);
          auto Dab = qm::block::ab(mo.D);
          auto Dba = qm::block::ba(mo.D);
          auto Dbb = qm::block::bb(mo.D);
          D_pq_sum = Daa.block(bf_p, bf_q, size_p, size_q).cwiseAbs().sum() +
                     Dab.block(bf_p, bf_q, size_p, size_q).cwiseAbs().sum() +
                     Dba.block(bf_p, bf_q, size_p, size_q).cwiseAbs().sum() +
                     Dbb.block(bf_p, bf_q, size_p, size_q).cwiseAbs().sum();
          if (p != q) {
            D_pq_sum += Daa.block(bf_q, bf_p, size_q, size_p).cwiseAbs().sum() +
                        Dab.block(bf_q, bf_p, size_q, size_p).cwiseAbs().sum() +
                        Dba.block(bf_q, bf_p, size_q, size_p).cwiseAbs().sum() +
                        Dbb.block(bf_q, bf_p, size_q, size_p).cwiseAbs().sum();
          }
          break;
        }
        }

        // Skip if density negligible
        if (D_pq_sum < 1e-12) {
          esp_idx++;
          continue;
        }

        // Resize buffers if needed
        int nab = m_esp_evaluator->nab(esp_idx);
        int nherm = m_esp_evaluator->nherm(esp_idx);
        if (workspace.rows() != static_cast<Eigen::Index>(count) ||
            workspace.cols() != nherm) {
          workspace.resize(count, nherm);
        }
        if (integrals.rows() != static_cast<Eigen::Index>(count) ||
            integrals.cols() != nab) {
          integrals.resize(count, nab);
        }

        m_esp_evaluator->evaluate(esp_idx, pts_block, integrals, workspace);

        // Contract: V(pt) += sum_pq D(p,q) * (pq|1/r|pt)
        // Factor of 2 for off-diagonal shell pairs (p != q)
        double scale = (p == q) ? 1.0 : 2.0;

        for (size_t pt = 0; pt < count; pt++) {
          Eigen::Map<const MatRM> V(integrals.row(pt).data(), size_p, size_q);

          double contrib = 0.0;
          switch (mo.kind) {
          case SpinorbitalKind::Restricted:
            contrib =
                (mo.D.block(bf_p, bf_q, size_p, size_q).array() * V.array())
                    .sum();
            break;
          case SpinorbitalKind::Unrestricted: {
            auto Da = qm::block::a(mo.D);
            auto Db = qm::block::b(mo.D);
            contrib = ((Da.block(bf_p, bf_q, size_p, size_q).array() +
                        Db.block(bf_p, bf_q, size_p, size_q).array()) *
                       V.array())
                          .sum();
            break;
          }
          case SpinorbitalKind::General: {
            auto Daa = qm::block::aa(mo.D);
            auto Dab = qm::block::ab(mo.D);
            auto Dba = qm::block::ba(mo.D);
            auto Dbb = qm::block::bb(mo.D);
            contrib = ((Daa.block(bf_p, bf_q, size_p, size_q).array() +
                        Dab.block(bf_p, bf_q, size_p, size_q).array() +
                        Dba.block(bf_p, bf_q, size_p, size_q).array() +
                        Dbb.block(bf_p, bf_q, size_p, size_q).array()) *
                       V.array())
                          .sum();
            break;
          }
          }
          pot(start + pt) += scale * contrib;
        }
        esp_idx++;
      }
    }
  });

  // Reduce results from all threads
  for (const auto &local : pot_local) {
    potential += local;
  }

  // Factor of 2 to match libcint convention (see three_center_kernels.h)
  // Negative sign because our ESP integrals have opposite sign convention
  return -2.0 * potential;
}

} // namespace occ::qm
