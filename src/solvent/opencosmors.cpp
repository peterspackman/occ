#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <limits>
#include <occ/core/constants.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/solvent/opencosmors.h>
#include <stdexcept>

namespace occ::solvent::sigma {

SolvationParameters SolvationParameters::opencosmors_24a() {
  // openCOSMO-RS 24a, as distributed with openCOSMO-RS_py
  // (TUHH-TVT/openCOSMO-RS_py, parameterization.py). kcal/mol/Å².
  SolvationParameters params;
  params.tau = {
      {1, 2.933803e-02},  // H
      {6, 2.287904e-02},  // C
      {7, 7.007681e-04},  // N
      {8, 3.545052e-03},  // O
      {9, 5.608829e-03},  // F
      {14, 4.215503e-03}, // Si
      {15, 3.607977e-03}, // P
      {16, 3.498700e-02}, // S
      {17, 3.414282e-02}, // Cl
      {35, 4.085111e-02}, // Br
  };
  params.eta = -4.448499;       // kcal/mol
  params.omega_ring = 2.6302510e-01; // kcal/mol
  return params;
}

RSParameters RSParameters::opencosmors_24a() { return RSParameters{}; }

RSComponent RSComponent::from_segments(const Segments &segments, double volume,
                                       double cavity_area) {
  if (segments.sigma_orth.size() != segments.size())
    throw std::runtime_error(
        "RSComponent::from_segments: sigma_orth has not been computed");
  RSComponent out;
  out.sigma = segments.sigma_averaged;
  out.sigma_orth = segments.sigma_orth;
  out.area = segments.areas;
  out.volume = volume;
  out.cavity_area = cavity_area;
  return out;
}

RSComponent mix_rs_components(const std::vector<RSComponent> &components,
                              const Vec &mole_fractions) {
  if (components.empty())
    throw std::runtime_error("mix_rs_components: no components");
  if (static_cast<Eigen::Index>(components.size()) != mole_fractions.size())
    throw std::runtime_error(
        "mix_rs_components: one mole fraction per component is required");
  const double sum = mole_fractions.sum();
  if (sum <= 0.0)
    throw std::runtime_error(
        "mix_rs_components: mole fractions are not positive");

  Eigen::Index total = 0;
  for (const auto &c : components)
    total += c.size();

  RSComponent out;
  out.sigma = Vec(total);
  out.sigma_orth = Vec(total);
  out.area = Vec(total);

  Eigen::Index at = 0;
  for (size_t m = 0; m < components.size(); m++) {
    const auto &c = components[m];
    const double x = mole_fractions(m) / sum;
    out.sigma.segment(at, c.size()) = c.sigma;
    out.sigma_orth.segment(at, c.size()) = c.sigma_orth;
    out.area.segment(at, c.size()) = x * c.area;
    out.volume += x * c.volume;
    out.cavity_area += x * c.total_area();
    at += c.size();
  }
  return out;
}

Mat rs_interaction_energies(const RSComponent &a, const RSComponent &b,
                            const RSParameters &params, double temperature) {
  const Eigen::Index na = a.size(), nb = b.size();
  Mat out(na, nb);

  const double misfit_prefactor = 0.5 * params.mf_alpha * params.a_eff;

  // Donor and acceptor excesses. A segment donates only below -sigma_thresh
  // and accepts only above it, so the clamps subsume the sign switches the
  // reference applies separately.
  auto donor = [&](double sigma) {
    return std::min(0.0, sigma + params.hb_sigma_thresh);
  };
  auto acceptor = [&](double sigma) {
    return std::max(0.0, sigma - params.hb_sigma_thresh);
  };

  const double scale =
      1.0 - params.hb_c_T + params.hb_c_T * (298.15 / temperature);
  const double hb_prefactor =
      (scale > 0.0) ? params.hb_c * scale * params.a_eff : 0.0;

  for (Eigen::Index i = 0; i < na; i++) {
    const double sigma_i = a.sigma(i), orth_i = a.sigma_orth(i);
    const double donor_i = donor(sigma_i), acceptor_i = acceptor(sigma_i);
    for (Eigen::Index j = 0; j < nb; j++) {
      const double sigma_sum = sigma_i + b.sigma(j);
      const double orth_sum = orth_i + b.sigma_orth(j);
      const double misfit = misfit_prefactor * sigma_sum *
                            (sigma_sum + params.mf_f_corr * orth_sum);
      // Symmetrised: either segment may be the donor.
      const double hbond = acceptor_i * donor(b.sigma(j)) +
                           acceptor(b.sigma(j)) * donor_i;
      out(i, j) = misfit + hb_prefactor * hbond;
    }
  }
  return out;
}

namespace {

/// `exp(-A/RT)`, rejecting the input if any exponent is large enough to
/// overflow. Over the parameterised range these run to about ±15, so the
/// fixed point can be iterated in plain arithmetic — one matrix-vector
/// product per step instead of a log-sum-exp over every pair.
Mat boltzmann_factors(const Mat &interaction, double temperature) {
  const double rt =
      occ::constants::molar_gas_constant<double> * temperature;
  Mat exponent = -interaction / rt;
  const double largest = exponent.maxCoeff();
  if (largest > 500.0)
    throw std::runtime_error(fmt::format(
        "openCOSMO-RS interaction energy {:.3g} kJ/mol at {:.1f} K overflows "
        "the Boltzmann factor; the segment descriptors are out of range",
        -largest * rt / 1000.0, temperature));
  return exponent.array().exp().matrix();
}

/// ln Γ for the pooled ensemble.
Vec solve_segment_activities(const Mat &interaction, const Vec &fraction,
                             const RSOptions &options) {
  const Eigen::Index n = fraction.size();
  const Mat tau = boltzmann_factors(interaction, options.temperature);

  Vec gamma = Vec::Ones(n);
  for (int iteration = 0; iteration < options.max_iterations; iteration++) {
    const Vec updated =
        (tau * fraction.cwiseProduct(gamma)).cwiseInverse();
    const double change =
        (updated - gamma).cwiseQuotient(gamma).cwiseAbs().maxCoeff();
    gamma += options.mixing * (updated - gamma);
    if (change < options.tolerance)
      return gamma.array().log();
  }

  if (options.throw_on_failure)
    throw std::runtime_error(fmt::format(
        "openCOSMO-RS segment activities did not converge in {} iterations",
        options.max_iterations));
  occ::log::warn("openCOSMO-RS segment activities did not converge");
  return gamma.array().log();
}

/// Pool every component's segments into one ensemble, with the segment mole
/// fractions the mixture implies.
struct PooledEnsemble {
  RSComponent all;
  Vec fraction;
  std::vector<Eigen::Index> offset;
};

PooledEnsemble pool(const std::vector<RSComponent> &components,
                    const Vec &mole_fractions) {
  Eigen::Index total = 0;
  for (const auto &c : components)
    total += c.size();

  PooledEnsemble out;
  out.all.sigma = Vec(total);
  out.all.sigma_orth = Vec(total);
  out.all.area = Vec(total);
  out.fraction = Vec(total);
  out.offset.reserve(components.size());

  Eigen::Index at = 0;
  for (size_t m = 0; m < components.size(); m++) {
    const auto &c = components[m];
    out.offset.push_back(at);
    out.all.sigma.segment(at, c.size()) = c.sigma;
    out.all.sigma_orth.segment(at, c.size()) = c.sigma_orth;
    out.all.area.segment(at, c.size()) = c.area;
    out.fraction.segment(at, c.size()) = mole_fractions(m) * c.area;
    at += c.size();
  }
  const double norm = out.fraction.sum();
  if (norm > 0.0)
    out.fraction /= norm;
  return out;
}

} // namespace

Vec rs_residual_ln_gamma(const std::vector<RSComponent> &components,
                         const Vec &mole_fractions,
                         const RSParameters &params,
                         const RSOptions &options) {
  const auto ensemble = pool(components, mole_fractions);
  const Mat interaction = rs_interaction_energies(
      ensemble.all, ensemble.all, params, options.temperature);
  const Vec ln_gamma =
      solve_segment_activities(interaction, ensemble.fraction, options);

  Vec out = Vec::Zero(components.size());
  for (size_t m = 0; m < components.size(); m++) {
    const auto &c = components[m];
    const Eigen::Index at = ensemble.offset[m];
    for (Eigen::Index i = 0; i < c.size(); i++)
      out(m) += (c.area(i) / params.a_eff) * ln_gamma(at + i);
  }
  return out;
}

Vec rs_combinatorial_ln_gamma(const std::vector<RSComponent> &components,
                              const Vec &mole_fractions,
                              const RSParameters &params) {
  const Eigen::Index n = components.size();
  Vec volume(n), area(n);
  for (Eigen::Index m = 0; m < n; m++) {
    volume(m) = components[m].volume;
    area(m) = components[m].total_area();
  }

  const double mean_volume = mole_fractions.dot(volume);
  const double mean_area = mole_fractions.dot(area);

  Vec out(n);
  for (Eigen::Index m = 0; m < n; m++) {
    const double phi = volume(m) / mean_volume;
    const double theta = area(m) / mean_area;
    const double ratio = phi / theta;
    out(m) = std::log(phi) + 1.0 - phi -
             params.comb_z * 0.5 * (area(m) / params.comb_a_std) *
                 (std::log(ratio) + 1.0 - ratio);
  }
  return out;
}

RSSolventModel::RSSolventModel(RSComponent solvent, RSParameters params,
                               RSOptions options)
    : m_solvent(std::move(solvent)), m_params(std::move(params)),
      m_options(std::move(options)) {
  const auto ensemble = pool({m_solvent}, Vec::Ones(1));
  m_fraction = ensemble.fraction;
  const Mat interaction = rs_interaction_energies(
      ensemble.all, ensemble.all, m_params, m_options.temperature);
  m_ln_gamma =
      solve_segment_activities(interaction, m_fraction, m_options);
}

Vec RSSolventModel::segment_energies(const RSComponent &solute) const {
  // Test particle against the converged solvent: at infinite dilution the
  // solute does not perturb the solvent's own activities.
  const Mat interaction = rs_interaction_energies(solute, m_solvent, m_params,
                                                  m_options.temperature);
  const double rt = occ::constants::molar_gas_constant<double> *
                    m_options.temperature;
  const Mat tau = boltzmann_factors(interaction, m_options.temperature);
  const Vec weight = m_fraction.cwiseProduct(m_ln_gamma.array().exp().matrix());
  const Vec ln_gamma = -(tau * weight).array().log();

  return (solute.area.array() / m_params.a_eff * rt * ln_gamma.array() /
          (occ::units::AU_TO_KJ_PER_MOL * 1000.0))
      .matrix();
}

double RSSolventModel::residual_energy(const RSComponent &solute) const {
  return segment_energies(solute).sum();
}

double RSSolventModel::combinatorial_energy(const RSComponent &solute) const {
  // Infinite dilution: the mixture averages are the pure solvent's.
  const double phi = solute.volume / m_solvent.volume;
  const double theta = solute.total_area() / m_solvent.total_area();
  const double ratio = phi / theta;
  const double ln_gamma =
      std::log(phi) + 1.0 - phi -
      m_params.comb_z * 0.5 * (solute.total_area() / m_params.comb_a_std) *
          (std::log(ratio) + 1.0 - ratio);
  const double rt = occ::constants::molar_gas_constant<double> *
                    m_options.temperature;
  return rt * ln_gamma / (occ::units::AU_TO_KJ_PER_MOL * 1000.0);
}

RSSolvationEnergy rs_solvation_free_energy(
    const RSSolventModel &solvent, const RSComponent &solute,
    const Segments &segments, double dielectric, int num_rings,
    double volume_liquid, const SolvationParameters &solvation_params) {
  RSSolvationEnergy out;
  out.dielectric = dielectric;
  out.residual = solvent.residual_energy(solute);
  out.combinatorial = solvent.combinatorial_energy(solute);
  out.cavity = segment_cavity_energies(segments, solvation_params).sum();

  const double to_hartree = 1.0 / occ::units::AU_TO_KCAL_PER_MOL;
  out.ring = -solvation_params.omega_ring * num_rings * to_hartree;
  out.constant = -solvation_params.eta * to_hartree;

  if (volume_liquid > 0.0) {
    constexpr double bar_in_pascal = 1.0e5;
    constexpr double cubic_metre_in_cubic_angstrom = 1.0e30;
    const double temperature = solvent.options().temperature;
    const double volume_gas = occ::constants::boltzmann<double> * temperature /
                              bar_in_pascal * cubic_metre_in_cubic_angstrom;
    const double rt =
        occ::constants::molar_gas_constant<double> * temperature / 1000.0;
    out.reference_state = -(rt * std::log(volume_gas / volume_liquid)) /
                          occ::units::AU_TO_KJ_PER_MOL;
  }
  return out;
}

Vec segment_cavity_energies(const Segments &segments,
                            const SolvationParameters &params) {
  const Eigen::Index n = segments.size();
  Vec out = Vec::Zero(n);
  const double to_hartree = 1.0 / occ::units::AU_TO_KCAL_PER_MOL;
  for (Eigen::Index i = 0; i < n; i++) {
    const auto entry = params.tau.find(segments.atomic_number(i));
    if (entry == params.tau.end())
      continue;
    out(i) = -entry->second * segments.areas(i) * to_hartree;
  }
  return out;
}

int ring_count(const core::Molecule &molecule) {
  if (molecule.bonds().empty())
    return 0;
  return std::max<int>(0, static_cast<int>(molecule.bonds().size()) -
                              static_cast<int>(molecule.size()) + 1);
}

std::vector<int> unparameterised_elements(const Segments &segments,
                                          const SolvationParameters &params) {
  std::vector<int> missing;
  for (Eigen::Index i = 0; i < segments.size(); i++) {
    const int z = segments.atomic_number(i);
    if (params.tau.contains(z))
      continue;
    if (std::find(missing.begin(), missing.end(), z) == missing.end())
      missing.push_back(z);
  }
  std::sort(missing.begin(), missing.end());
  return missing;
}

ankerl::unordered_dense::map<int, double>
area_per_element(const Segments &segments) {
  ankerl::unordered_dense::map<int, double> out;
  for (Eigen::Index i = 0; i < segments.size(); i++)
    out[segments.atomic_number(i)] += segments.areas(i);
  return out;
}

double molecular_solvation_terms(int num_rings, double volume_liquid,
                                 double temperature,
                                 const SolvationParameters &params) {
  const double to_hartree = 1.0 / occ::units::AU_TO_KCAL_PER_MOL;
  double total = -params.omega_ring * num_rings * to_hartree;
  total -= params.eta * to_hartree;

  if (volume_liquid > 0.0) {
    // Ideal gas volume per molecule at `temperature` and 1 bar, in Å³, so
    // the ratio matches `volume_liquid`.
    constexpr double bar_in_pascal = 1.0e5;
    constexpr double cubic_metre_in_cubic_angstrom = 1.0e30;
    const double volume_gas = occ::constants::boltzmann<double> * temperature /
                              bar_in_pascal * cubic_metre_in_cubic_angstrom;
    const double rt =
        occ::constants::molar_gas_constant<double> * temperature / 1000.0;
    total -= (rt * std::log(volume_gas / volume_liquid)) /
             occ::units::AU_TO_KJ_PER_MOL;
  }
  return total;
}

} // namespace occ::solvent::sigma
