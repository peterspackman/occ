#include <Eigen/LU>
#include <cmath>
#include <fmt/core.h>
#include <limits>
#include <occ/core/constants.h>
#include <occ/core/log.h>
#include <occ/solvent/sigma_potential.h>
#include <stdexcept>

namespace occ::solvent::sigma {

namespace {

/// Row-wise ⟨f⟩ under the pairing distribution.
Vec pairing_mean(const Mat &pairing, const Mat &f) {
  return (pairing.array() * f.array()).rowwise().sum();
}

struct Moments {
  Vec mean;
  Vec variance;
  Vec variance_misfit;
  Vec variance_hbond;
  Vec covariance;
  Vec hbond_probability;
  Vec pairing_entropy;
};

/// Every moment is a contraction over the same pairing matrix, which is why
/// the misfit and H-bond terms are kept apart in the kernel.
Moments contract_moments(const Mat &pairing, const Kernel &kernel,
                         const Vec &profile_flat, double beta) {
  const Mat &mf = kernel.misfit;
  const Mat &hb = kernel.hbond;
  const Mat total = mf + hb;

  Moments m;
  m.mean = pairing_mean(pairing, total);
  const Vec mean_mf = pairing_mean(pairing, mf);
  const Vec mean_hb = pairing_mean(pairing, hb);

  m.variance =
      pairing_mean(pairing, total.cwiseProduct(total)) - m.mean.cwiseAbs2();
  m.variance_misfit =
      pairing_mean(pairing, mf.cwiseProduct(mf)) - mean_mf.cwiseAbs2();
  m.variance_hbond =
      pairing_mean(pairing, hb.cwiseProduct(hb)) - mean_hb.cwiseAbs2();
  m.covariance = pairing_mean(pairing, mf.cwiseProduct(hb)) -
                 mean_mf.cwiseProduct(mean_hb);

  // The H-bond term is non-positive wherever it is active, so its sign
  // identifies the hydrogen-bonded part of the partner ensemble.
  Mat active = (hb.array() < 0.0).cast<double>();
  m.hbond_probability = pairing_mean(pairing, active);

  const Eigen::Index dim = pairing.rows();
  m.pairing_entropy = Vec::Zero(dim);
  for (Eigen::Index i = 0; i < dim; i++) {
    double kl = 0.0;
    for (Eigen::Index j = 0; j < dim; j++) {
      const double p = pairing(i, j);
      if (p > 0.0 && profile_flat(j) > 0.0)
        kl += p * std::log(p / profile_flat(j));
    }
    m.pairing_entropy(i) = kl / beta;
  }
  return m;
}

/// Pairing matrix and log-partition function for the current μ.
void pairing_and_log_partition(const Vec &profile_flat, const Mat &energy,
                               const Vec &mu, double beta, Mat &pairing,
                               Vec &log_z) {
  const Eigen::Index dim = profile_flat.size();
  const double neg_inf = -std::numeric_limits<double>::infinity();

  Vec log_p(dim);
  for (Eigen::Index j = 0; j < dim; j++)
    log_p(j) = (profile_flat(j) > 0.0) ? std::log(profile_flat(j)) : neg_inf;

  pairing.resize(dim, dim);
  log_z.resize(dim);

  for (Eigen::Index i = 0; i < dim; i++) {
    // Empty bins contribute exp(-inf) = 0 without any branching, provided
    // the shift is finite.
    Vec w = log_p + beta * (mu - energy.row(i).transpose());
    const double shift = w.maxCoeff();
    if (!std::isfinite(shift))
      throw std::runtime_error(
          "sigma potential: solvent profile has no occupied bins");
    Vec e = (w.array() - shift).exp();
    const double sum = e.sum();
    log_z(i) = shift + std::log(sum);
    pairing.row(i) = e.transpose() / sum;
  }
}

} // namespace

Vec flatten(const Mat &values) {
  Vec flat(values.size());
  const Eigen::Index n = values.rows();
  for (Eigen::Index c = 0; c < values.cols(); c++)
    flat.segment(c * n, n) = values.col(c);
  return flat;
}

Mat unflatten(const Vec &flat, const Grid &grid, int num_classes) {
  Mat values(grid.n, num_classes);
  for (int c = 0; c < num_classes; c++)
    values.col(c) = flat.segment(static_cast<Eigen::Index>(c) * grid.n, grid.n);
  return values;
}

Mat pairing_matrix(const Vec &profile_flat, const Kernel &kernel,
                   const Vec &mu, double temperature) {
  const double beta = 1.0 / (kernel.gas_constant * temperature);
  Mat pairing;
  Vec log_z;
  pairing_and_log_partition(profile_flat, kernel.total(), mu, beta, pairing,
                            log_z);
  return pairing;
}

Potential solve_sigma_potential(const Profile &solvent, const Kernel &kernel,
                                const PotentialOptions &options) {
  if (solvent.grid != kernel.grid)
    throw std::runtime_error("solve_sigma_potential: profile and kernel grids "
                             "do not match");
  if (solvent.num_classes() != kernel.num_classes)
    throw std::runtime_error(fmt::format(
        "solve_sigma_potential: profile has {} classes, kernel has {}",
        solvent.num_classes(), kernel.num_classes));

  const Eigen::Index dim = kernel.dim();
  const Vec profile_flat = flatten(solvent.normalized());
  const Mat energy = kernel.total();
  const double beta = 1.0 / (kernel.gas_constant * options.temperature);

  Vec mu = Vec::Zero(dim);
  Mat pairing;
  Vec log_z;
  Moments moments;
  Vec previous_variance = Vec::Zero(dim);

  Potential result;
  result.grid = kernel.grid;
  result.num_classes = kernel.num_classes;
  result.temperature = options.temperature;

  for (int iteration = 1; iteration <= options.max_iterations; iteration++) {
    pairing_and_log_partition(profile_flat, energy, mu, beta, pairing, log_z);
    const Vec mu_updated = -log_z / beta;
    moments = contract_moments(pairing, kernel, profile_flat, beta);

    const double residual_mu = (mu_updated - mu).cwiseAbs().maxCoeff();
    const double residual_variance =
        (iteration == 1)
            ? std::numeric_limits<double>::infinity()
            : (moments.variance - previous_variance).cwiseAbs().maxCoeff();
    previous_variance = moments.variance;

    result.iterations = iteration;
    result.residual_mu = residual_mu;
    result.residual_variance = residual_variance;

    if (residual_mu < options.tolerance_mu &&
        residual_variance < options.tolerance_variance) {
      mu = mu_updated;
      result.converged = true;
      break;
    }

    if (options.use_newton) {
      Mat jacobian = Mat::Identity(dim, dim) + pairing;
      mu += jacobian.partialPivLu().solve(mu_updated - mu);
    } else {
      mu = (1.0 - options.mixing) * mu + options.mixing * mu_updated;
    }
  }

  if (!result.converged) {
    // Refresh the moments so the reported profiles match the returned mu.
    pairing_and_log_partition(profile_flat, energy, mu, beta, pairing, log_z);
    moments = contract_moments(pairing, kernel, profile_flat, beta);
    occ::log::warn("sigma potential did not converge in {} iterations "
                   "(residual mu = {:.3e}, variance = {:.3e})",
                   result.iterations, result.residual_mu,
                   result.residual_variance);
  }

  const Grid &grid = kernel.grid;
  const int nc = kernel.num_classes;
  result.mu = unflatten(mu, grid, nc);
  result.mean_energy = unflatten(moments.mean, grid, nc);
  result.variance = unflatten(moments.variance, grid, nc);
  result.variance_misfit = unflatten(moments.variance_misfit, grid, nc);
  result.variance_hbond = unflatten(moments.variance_hbond, grid, nc);
  result.covariance = unflatten(moments.covariance, grid, nc);
  result.hbond_probability = unflatten(moments.hbond_probability, grid, nc);
  result.pairing_entropy = unflatten(moments.pairing_entropy, grid, nc);
  return result;
}

double residual_ln_gamma(const Profile &solute,
                         const Potential &solvent_potential,
                         const Potential &solute_potential, double a_eff,
                         double gas_constant) {
  const double rt = gas_constant * solvent_potential.temperature;
  const Mat difference = solvent_potential.mu - solute_potential.mu;
  return contract(solute, difference) / (a_eff * rt);
}

} // namespace occ::solvent::sigma
