#include <cmath>
#include <fmt/core.h>
#include <occ/solvent/sigma_activity.h>
#include <stdexcept>

namespace occ::solvent::sigma {

namespace {

void check_sizes(const std::vector<Component> &components,
                 const Vec &mole_fractions) {
  if (components.empty())
    throw std::runtime_error("activity: no components");
  if (mole_fractions.size() != static_cast<Eigen::Index>(components.size()))
    throw std::runtime_error(
        fmt::format("activity: {} mole fractions for {} components",
                    mole_fractions.size(), components.size()));
}

} // namespace

Vec combinatorial_ln_gamma(const std::vector<Component> &components,
                           const Vec &mole_fractions,
                           const Parameters &params) {
  check_sizes(components, mole_fractions);
  const Eigen::Index n = mole_fractions.size();
  const double half_z = 0.5 * params.z_coordination;

  Vec r(n), q(n), l(n);
  for (Eigen::Index i = 0; i < n; i++) {
    r(i) = components[i].volume / params.r0;
    q(i) = components[i].area() / params.q0;
    l(i) = half_z * (r(i) - q(i)) - (r(i) - 1.0);
  }

  const double sum_xr = mole_fractions.dot(r);
  const double sum_xq = mole_fractions.dot(q);
  const double sum_xl = mole_fractions.dot(l);

  Vec out(n);
  for (Eigen::Index i = 0; i < n; i++) {
    // phi_i/x_i and theta_i/phi_i are both free of x_i, so infinite dilution
    // is well defined.
    const double phi_over_x = r(i) / sum_xr;
    const double theta_over_phi = (q(i) * sum_xr) / (r(i) * sum_xq);
    out(i) = std::log(phi_over_x) + half_z * q(i) * std::log(theta_over_phi) +
             l(i) - phi_over_x * sum_xl;
  }
  return out;
}

Vec residual_ln_gamma(const std::vector<Component> &components,
                      const Vec &mole_fractions, const Parameters &params,
                      const PotentialOptions &options) {
  check_sizes(components, mole_fractions);
  const Eigen::Index n = mole_fractions.size();
  const Grid &grid = components.front().profile.grid;

  const Kernel kernel = build_kernel(grid, params, options.temperature);

  std::vector<Profile> profiles;
  profiles.reserve(components.size());
  for (const auto &component : components)
    profiles.push_back(component.profile);

  const Profile mixture = mix_profiles(profiles, mole_fractions);
  const Potential mixture_potential =
      solve_sigma_potential(mixture, kernel, options);

  const double rt = params.gas_constant * options.temperature;
  Vec out(n);
  for (Eigen::Index i = 0; i < n; i++) {
    const Potential pure =
        solve_sigma_potential(components[i].profile, kernel, options);
    out(i) = contract(components[i].profile, mixture_potential.mu - pure.mu) /
             (params.a_eff * rt);
  }
  return out;
}

Vec activity_coefficients(const std::vector<Component> &components,
                          const Vec &mole_fractions, const Parameters &params,
                          const PotentialOptions &options) {
  Vec ln_gamma = combinatorial_ln_gamma(components, mole_fractions, params) +
                 residual_ln_gamma(components, mole_fractions, params, options);
  if (components.size() == 2)
    ln_gamma += dispersion_ln_gamma(components[0].dispersion,
                                    components[1].dispersion, mole_fractions);
  return ln_gamma;
}

} // namespace occ::solvent::sigma
