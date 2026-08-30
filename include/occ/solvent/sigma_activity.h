#pragma once
#include <occ/solvent/sigma_dispersion.h>
#include <occ/solvent/sigma_potential.h>
#include <vector>

namespace occ::solvent::sigma {

/// One component of a liquid mixture: its σ-profile plus the cavity volume
/// the combinatorial term needs. The cavity area is the profile's total.
struct Component {
  Profile profile; ///< area per bin, Å²
  double volume{0.0}; ///< Å³
  /// Left unknown when the profile carries no dispersion metadata, in which
  /// case the dispersion term contributes nothing.
  Dispersion dispersion{};

  double area() const { return profile.total_area(); }
};

/// Staverman–Guggenheim combinatorial contribution to `ln γ` for each
/// component, with `r_i = V_i/r0` and `q_i = A_i/q0`:
///
///     ln γ_i^comb = ln(φ_i/x_i) + (z/2) q_i ln(θ_i/φ_i)
///                   + l_i − (φ_i/x_i) Σ_j x_j l_j
///     l_i = (z/2)(r_i − q_i) − (r_i − 1)
///
/// Evaluated through `φ_i/x_i = r_i/Σ_j x_j r_j` so it stays finite at
/// infinite dilution.
Vec combinatorial_ln_gamma(const std::vector<Component> &components,
                           const Vec &mole_fractions,
                           const Parameters &params);

/// Residual contribution to `ln γ` for each component:
///
///     ln γ_i^res = Σ_σ A_i p_i(σ) [μ_S(σ) − μ_i(σ)] / (a_eff · RT)
///
/// Solves the σ-potential once for the mixture and once for each pure
/// component.
Vec residual_ln_gamma(const std::vector<Component> &components,
                      const Vec &mole_fractions, const Parameters &params,
                      const PotentialOptions &options = {});

/// `combinatorial_ln_gamma + residual_ln_gamma`, plus the COSMO-SAC-dsp
/// dispersion term for a binary whose components both carry a dispersion
/// parameter.
///
/// The dispersion term is defined for two components only, so a mixture of
/// three or more gets the first two contributions alone.
Vec activity_coefficients(const std::vector<Component> &components,
                          const Vec &mole_fractions, const Parameters &params,
                          const PotentialOptions &options = {});

} // namespace occ::solvent::sigma
