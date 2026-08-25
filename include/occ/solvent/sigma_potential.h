#pragma once
#include <occ/core/constants.h>
#include <occ/solvent/sigma_kernel.h>

namespace occ::solvent::sigma {

/// Gas constant in the units this module works in, kcal mol⁻¹ K⁻¹. Derived
/// from `occ::constants` so `RT` here matches the rest of occ.
inline constexpr double gas_constant_kcal =
    occ::constants::boltzmann<double> * occ::constants::avogadro<double> /
    4184.0;

struct PotentialOptions {
  double temperature{298.15}; ///< K

  /// Damping for the Picard update, `μ ← (1−ω)μ + ω g(μ)`. The fixed-point
  /// map has eigenvalue −1 along the constant direction, so the undamped
  /// iteration oscillates; ω = 0.5 cancels that mode exactly.
  double mixing{0.5};

  /// Convergence is gated on the potential *and* the variance profile: the
  /// second moment settles more slowly, so a tolerance tuned on μ alone can
  /// return an under-converged variance.
  double tolerance_mu{1e-10};
  double tolerance_variance{1e-8};

  int max_iterations{500};

  /// Use the exact Newton step. The Jacobian of the fixed-point map is `−P`,
  /// so the update solves `(I + P) δ = g(μ) − μ` with the pairing matrix that
  /// has already been formed.
  bool use_newton{true};
};

/// The σ-potential together with the moment profiles of the pairing energy.
///
/// Every array is `grid.n × num_classes` and lives on the same grid, so any
/// of them can be handed to `contract` / `contract_segments` unchanged.
/// Energies are kcal/mol per contact; variances are (kcal/mol)².
struct Potential {
  Grid grid;
  int num_classes{1};
  double temperature{298.15};

  Mat mu;          ///< σ-potential, `RT ln Γ(σ)`
  Mat mean_energy; ///< ⟨E⟩(σ) under the pairing distribution
  Mat variance;    ///< Var(E)(σ)
  Mat variance_misfit;
  Mat variance_hbond;
  Mat covariance; ///< Cov(E_misfit, E_hbond); `variance` = sum + 2·cov
  Mat hbond_probability; ///< ∫_HB P(σ′|σ) dσ′
  Mat pairing_entropy;   ///< `KL(P‖p)/β`, the entropic part of μ

  int iterations{0};
  bool converged{false};
  double residual_mu{0.0};
  double residual_variance{0.0};

  /// σ-potential per unit area (kcal mol⁻¹ Å⁻²), which is what contracts
  /// against segment areas.
  Mat mu_per_area(double a_eff) const { return mu / a_eff; }
};

/// Conditional pairing distribution
/// `P(σ′|σ) ∝ p(σ′) exp{β[μ(σ′) − E(σ,σ′)]}`, row-normalised, in the
/// composite `class * grid.n + bin` index. Rows are the conditioning σ.
///
/// Accumulated in log space with a max subtraction; the H-bond term makes
/// the exponent large enough that the direct form overflows.
Mat pairing_matrix(const Vec &profile_flat, const Kernel &kernel,
                   const Vec &mu, double temperature);

/// Flatten a `grid.n × num_classes` profile to the composite index, and back.
Vec flatten(const Mat &values);
Mat unflatten(const Vec &flat, const Grid &grid, int num_classes);

/// Solve for the σ-potential of `solvent` and evaluate the moment profiles.
///
/// `solvent` is normalised internally, so only its shape matters. The
/// potential is evaluated over the whole grid, including bins where the
/// solvent has no area — a solute segment can still land there.
Potential solve_sigma_potential(const Profile &solvent, const Kernel &kernel,
                                const PotentialOptions &options = {});

/// Residual contribution to `ln γ` for a solute with profile `solute` in a
/// solvent whose potential is `solvent_potential`:
///
///     ln γ^res = Σ_σ A_X p̂_X(σ) [μ_S(σ) − μ_X(σ)] / (a_eff · RT)
///
/// which is `contract(solute, μ_S − μ_X)` up to that prefactor.
double residual_ln_gamma(const Profile &solute,
                         const Potential &solvent_potential,
                         const Potential &solute_potential, double a_eff);

} // namespace occ::solvent::sigma
