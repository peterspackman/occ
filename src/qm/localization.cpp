#include <Eigen/Eigenvalues>
#include <cmath>
#include <occ/core/log.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/localization.h>
#include <occ/qm/wavefunction.h>
#include <stdexcept>

namespace occ::qm {

namespace {

// Both populations are q_A(s, t) = 1/2 sum_{mu in A} (X_s Y_t + X_t Y_s) for a
// pair of AO-coefficient matrices that rotate with the orbitals: X = C, Y = S C
// (Mulliken) or X = Y = S^1/2 C (Lowdin, Y left empty and X used for both)
struct PopulationFactors {
  Mat X, Y;
  const Mat &y() const { return Y.size() ? Y : X; }
};

PopulationFactors population_factors(const Mat &S, const Mat &C,
                                     PopulationKind kind) {
  if (kind == PopulationKind::Mulliken)
    return {C, S * C};
  Eigen::SelfAdjointEigenSolver<Mat> es(S);
  if (es.info() != Eigen::Success)
    throw std::runtime_error("pipek_mezey: overlap matrix diagonalization "
                             "failed");
  return {es.operatorSqrt() * C, Mat()};
}

Mat atomic_populations(const PopulationFactors &f,
                       const std::vector<int> &bf_atom, int natoms) {
  Mat q = Mat::Zero(natoms, f.X.cols());
  for (Eigen::Index k = 0; k < f.X.cols(); k++)
    for (Eigen::Index mu = 0; mu < f.X.rows(); mu++)
      q(bf_atom[mu], k) += f.X(mu, k) * f.y()(mu, k);
  return q;
}

} // namespace

LocalizedOrbitals pipek_mezey(const AOBasis &basis, const Mat &S, const Mat &C,
                              const PipekMezeySettings &settings) {
  const auto &bf_atom = basis.bf_to_atom();
  const int natoms = static_cast<int>(basis.atoms().size());
  const int nbf = static_cast<int>(C.rows()), n = static_cast<int>(C.cols());
  if (nbf != static_cast<int>(basis.nbf()) || S.rows() != nbf ||
      S.cols() != nbf)
    throw std::runtime_error(
        "pipek_mezey: orbital coefficients and overlap must match the basis");

  PopulationFactors f = population_factors(S, C, settings.population);
  LocalizedOrbitals result;
  result.U = Mat::Identity(n, n);
  std::vector<double> qss(natoms), qtt(natoms), qst(natoms);

  // rotate columns (a, b) of M by the angle with cosine c and sine s
  auto rotate = [](Mat &M, int a, int b, double c, double s) {
    const Vec ma = M.col(a), mb = M.col(b);
    M.col(a) = c * ma + s * mb;
    M.col(b) = c * mb - s * ma;
  };

  double max_rot = 0.0, gain = 0.0;
  for (int sweep = 0; sweep < settings.max_sweeps; sweep++) {
    max_rot = 0.0;
    gain = 0.0;
    for (int a = 0; a < n; a++) {
      for (int b = a + 1; b < n; b++) {
        const Mat &Y = f.y();
        std::fill(qss.begin(), qss.end(), 0.0);
        std::fill(qtt.begin(), qtt.end(), 0.0);
        std::fill(qst.begin(), qst.end(), 0.0);
        for (int mu = 0; mu < nbf; mu++) {
          const int A = bf_atom[mu];
          qss[A] += f.X(mu, a) * Y(mu, a);
          qtt[A] += f.X(mu, b) * Y(mu, b);
          qst[A] += 0.5 * (f.X(mu, a) * Y(mu, b) + f.X(mu, b) * Y(mu, a));
        }
        double x = 0.0, y = 0.0;
        for (int A = 0; A < natoms; A++) {
          const double d = qss[A] - qtt[A];
          x += qst[A] * qst[A] - 0.25 * d * d;
          y += qst[A] * d;
        }
        // the functional changes by x (1 - cos 4g) + y sin 4g; cos 4g = -x / r,
        // sin 4g = y / r maximizes it, a gain of x + r. Pairs that gain
        // nothing (a flat direction, e.g. among equivalent pi bonds) are left
        // alone: their angle would be rounding noise
        const double r = std::sqrt(x * x + y * y);
        if (x + r < 1e-14)
          continue;
        const double g = 0.25 * std::atan2(y, -x);
        max_rot = std::max(max_rot, std::abs(g));
        gain += x + r;
        const double c = std::cos(g), s = std::sin(g);
        rotate(f.X, a, b, c, s);
        if (f.Y.size())
          rotate(f.Y, a, b, c, s);
        rotate(result.U, a, b, c, s);
      }
    }
    result.sweeps = sweep + 1;
    // on the functional, not the angles: Jacobi sweeps converge linearly, and
    // strongly coupled pairs can keep rotating by ~1e-7 rad per sweep long
    // after the functional has settled
    if (gain < settings.tolerance) {
      result.converged = true;
      break;
    }
  }
  if (!result.converged)
    occ::log::warn("Pipek-Mezey: functional still rising by {:.3e} (largest "
                   "rotation {:.3e}) after {} sweeps",
                   gain, max_rot, result.sweeps);

  result.C = C * result.U;
  result.populations = atomic_populations(f, bf_atom, natoms);
  result.functional = result.populations.squaredNorm();
  occ::log::debug(
      "Pipek-Mezey: {} orbitals, functional {:.10f} after {} sweeps", n,
      result.functional, result.sweeps);
  return result;
}

LocalizedOrbitals pipek_mezey(const Wavefunction &wfn, int n_frozen,
                              const PipekMezeySettings &settings) {
  if (wfn.mo.kind != SpinorbitalKind::Restricted ||
      wfn.mo.n_alpha != wfn.mo.n_beta)
    throw std::runtime_error(
        "pipek_mezey: needs a closed-shell restricted wavefunction");
  const int nocc = static_cast<int>(wfn.mo.n_alpha);
  if (n_frozen < 0 || n_frozen >= nocc)
    throw std::runtime_error("pipek_mezey: n_frozen must be in [0, n_occ)");
  const IntegralEngine engine(wfn.basis);
  const Mat S = engine.one_electron_operator(cint::Operator::overlap);
  return pipek_mezey(wfn.basis, S,
                     wfn.mo.C.middleCols(n_frozen, nocc - n_frozen), settings);
}

} // namespace occ::qm
