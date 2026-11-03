#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/cphf.h>
#include <algorithm>
#include <stdexcept>

namespace occ::qm::cphf {

std::vector<SpinChannel> spin_channels(const MolecularOrbitals &mo) {
  const Eigen::Index nbf = mo.C.rows();
  auto channel = [](const Mat &C, const Vec &e, Eigen::Index nocc) {
    const Eigen::Index nvirt = C.cols() - nocc;
    return SpinChannel{C.leftCols(nocc), C.rightCols(nvirt), e.head(nocc),
                       e.segment(nocc, nvirt)};
  };
  switch (mo.kind) {
  case SpinorbitalKind::Restricted:
    return {channel(mo.C, mo.energies, mo.n_alpha)};
  case SpinorbitalKind::Unrestricted: {
    const Eigen::Index n = nbf / 2;
    return {channel(mo.C.topRows(n), mo.energies.head(n), mo.n_alpha),
            channel(mo.C.bottomRows(n), mo.energies.tail(n), mo.n_beta)};
  }
  default:
    throw std::runtime_error("cphf: general spinorbitals not supported");
  }
}

Mat occupied_orbitals(const MolecularOrbitals &mo) {
  const auto ch = spin_channels(mo);
  Eigen::Index n = 0;
  for (const auto &c : ch)
    n += c.Co.cols();
  Mat Co(ch[0].Co.rows(), n);
  n = 0;
  for (const auto &c : ch) {
    Co.middleCols(n, c.Co.cols()) = c.Co;
    n += c.Co.cols();
  }
  return Co;
}

Mat contract_occupied(const Mat &F, const MolecularOrbitals &mo) {
  const auto ch = spin_channels(mo);
  const Eigen::Index nbf = F.cols();
  const bool per_spin = F.rows() == 2 * nbf;
  Eigen::Index n = 0;
  for (const auto &c : ch)
    n += c.Co.cols();
  Mat out(nbf, n);
  n = 0;
  for (size_t s = 0; s < ch.size(); s++) {
    const auto Fs = F.middleRows(per_spin ? s * nbf : 0, nbf);
    out.middleCols(n, ch[s].Co.cols()).noalias() = Fs * ch[s].Co;
    n += ch[s].Co.cols();
  }
  return out;
}

Solver::Solver(const MolecularOrbitals &mo, ResponseFunction G,
               Settings settings)
    : m_channels(spin_channels(mo)), m_G(std::move(G)),
      m_settings(settings) {
  Eigen::Index size = 0, column = 0;
  for (const auto &c : m_channels) {
    if (c.Co.cols() <= 0 || c.Cv.cols() <= 0)
      throw std::runtime_error("cphf: need occupied and virtual orbitals");
    m_offset.push_back(size);
    m_column.push_back(column);
    size += c.Cv.cols() * c.Co.cols();
    column += c.Co.cols();
  }
  m_delta = vo_zero();
  for (size_t s = 0; s < m_channels.size(); s++) {
    const auto &c = m_channels[s];
    block(m_delta, s) = c.ev.replicate(1, c.Co.cols()) -
                        c.eo.transpose().replicate(c.Cv.cols(), 1);
  }
}

Mat Solver::vo_zero() const {
  if (m_channels.size() == 1)
    return Mat::Zero(m_channels[0].Cv.cols(), m_channels[0].Co.cols());
  const auto &b = m_channels.back();
  return Mat::Zero(m_offset.back() + b.Cv.cols() * b.Co.cols(), 1);
}

Eigen::Map<Mat> Solver::block(Mat &U, int s) const {
  const auto &c = m_channels[s];
  return Eigen::Map<Mat>(U.data() + m_offset[s], c.Cv.cols(), c.Co.cols());
}

Eigen::Map<const Mat> Solver::block(const Mat &U, int s) const {
  const auto &c = m_channels[s];
  return Eigen::Map<const Mat>(U.data() + m_offset[s], c.Cv.cols(),
                               c.Co.cols());
}

Mat Solver::vo_block(const Mat &X) const {
  const Eigen::Index nbf = X.cols();
  const bool per_spin = X.rows() == 2 * nbf;
  Mat out = vo_zero();
  for (size_t s = 0; s < m_channels.size(); s++) {
    const auto &c = m_channels[s];
    block(out, s) =
        c.Cv.transpose() * X.middleRows(per_spin ? s * nbf : 0, nbf) * c.Co;
  }
  return out;
}

std::vector<Mat> Solver::apply(const std::vector<Mat> &U) const {
  std::vector<Mat> G;
  return apply(U, G);
}

std::vector<Mat> Solver::apply(const std::vector<Mat> &U,
                               std::vector<Mat> &G) const {
  const Eigen::Index nbf = m_channels[0].Co.rows();
  const Eigen::Index ncol = m_column.back() + m_channels.back().Co.cols();
  std::vector<Mat> Cx(U.size(), Mat(nbf, ncol));
  for (size_t k = 0; k < U.size(); k++)
    for (size_t s = 0; s < m_channels.size(); s++)
      Cx[k].middleCols(m_column[s], m_channels[s].Co.cols()).noalias() =
          m_channels[s].Cv * block(U[k], s);
  G = m_G(Cx);
  std::vector<Mat> result;
  result.reserve(U.size());
  for (size_t k = 0; k < U.size(); k++) {
    Mat r = m_delta.cwiseProduct(U[k]);
    for (size_t s = 0; s < m_channels.size(); s++) {
      const auto &c = m_channels[s];
      block(r, s).noalias() +=
          c.Cv.transpose() * G[k].middleCols(m_column[s], c.Co.cols());
    }
    result.push_back(std::move(r));
  }
  return result;
}

namespace {
double dot(const Mat &a, const Mat &b) {
  return a.cwiseProduct(b).sum();
}
} // namespace

// Preconditioned conjugate gradient, one independent recurrence per RHS, the
// expensive operator application batched over all RHS still running. The
// orbital Hessian of a stable SCF is symmetric positive definite, and
// e_a - e_i is an excellent diagonal preconditioner, so this converges in a
// handful of iterations.
Result Solver::solve(const std::vector<Mat> &B,
                     const std::vector<Mat> *U0) const {
  occ::timing::start(occ::timing::category::fock);
  const size_t nrhs = B.size();
  Result result;
  result.U.resize(nrhs);
  result.residuals.assign(nrhs, 0.0);

  const Mat precond = m_delta.cwiseInverse();
  std::vector<Mat> r(nrhs), z(nrhs), p(nrhs);
  std::vector<double> rz(nrhs, 0.0);
  std::vector<bool> active(nrhs, true);

  // Start from the given guess, or the uncoupled solution.
  for (size_t k = 0; k < nrhs; k++)
    result.U[k] = U0 ? (*U0)[k] : Mat(B[k].cwiseProduct(precond));
  {
    auto AU = apply(result.U, result.G);
    result.response_builds += static_cast<int>(nrhs);
    for (size_t k = 0; k < nrhs; k++) {
      r[k] = B[k] - AU[k];
      z[k] = r[k].cwiseProduct(precond);
      p[k] = z[k];
      rz[k] = dot(r[k], z[k]);
      result.residuals[k] = r[k].cwiseAbs().maxCoeff();
      active[k] = result.residuals[k] >= m_settings.tolerance;
    }
  }

  for (int iter = 1; iter <= m_settings.max_iterations; iter++) {
    std::vector<size_t> idx;
    for (size_t k = 0; k < nrhs; k++)
      if (active[k])
        idx.push_back(k);
    if (idx.empty()) {
      result.converged = true;
      break;
    }
    result.iterations = iter;

    std::vector<Mat> batch;
    batch.reserve(idx.size());
    for (auto k : idx)
      batch.push_back(p[k]);
    std::vector<Mat> Gp;
    auto Ap = apply(batch, Gp);
    result.response_builds += static_cast<int>(idx.size());

    double max_res = 0.0;
    for (size_t n = 0; n < idx.size(); n++) {
      const size_t k = idx[n];
      const double alpha = rz[k] / dot(p[k], Ap[n]);
      result.U[k] += alpha * p[k];
      result.G[k] += alpha * Gp[n];
      r[k] -= alpha * Ap[n];
      result.residuals[k] = r[k].cwiseAbs().maxCoeff();
      max_res = std::max(max_res, result.residuals[k]);
      if (result.residuals[k] < m_settings.tolerance) {
        active[k] = false;
        continue;
      }
      z[k] = r[k].cwiseProduct(precond);
      const double rz_new = dot(r[k], z[k]);
      p[k] = z[k] + (rz_new / rz[k]) * p[k];
      rz[k] = rz_new;
    }
    occ::log::info("  CPHF iteration {:3d}: {:4d} perturbations unconverged, "
                   "max residual {:.2e}",
                   iter, std::count(active.begin(), active.end(), true),
                   max_res);
  }
  if (!result.converged) {
    result.converged = std::none_of(active.begin(), active.end(),
                                    [](bool a) { return a; });
  }
  if (!result.converged)
    occ::log::warn("CPHF did not converge in {} iterations",
                   m_settings.max_iterations);
  result.R = std::move(r);
  occ::timing::stop(occ::timing::category::fock);
  return result;
}

} // namespace occ::qm::cphf
