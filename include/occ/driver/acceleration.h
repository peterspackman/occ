#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <occ/core/log.h>
#include <occ/gto/shell.h>
#include <occ/io/occ_input.h>
#include <occ/qm/fitting_basis.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/seminumerical_exchange.h>
#include <string>
#include <type_traits>

// Forward declaration: only used in a discarded `if constexpr` branch unless the
// including translation unit instantiates apply_acceleration<DFT> (in which case
// it includes the full <occ/dft/dft.h>).
namespace occ::dft {
class DFT;
}

namespace occ::driver {

// Resolved SCF acceleration choices for a single calculation.
struct AccelerationPlan {
  std::string df_basis; ///< auxiliary basis for density fitting ("" => none)
  bool use_cosx{false}; ///< seminumerical (COSX) exchange for the K term
};

/**
 * @brief Decide SCF acceleration (density fitting / COSX) for a calculation.
 *
 * @param policy             requested RIPolicy (Auto by default)
 * @param orbital_basis_name primary orbital basis name (for aux-basis lookup)
 * @param nbf                number of basis functions (for the COSX crossover)
 * @param exact_exchange     fraction of exact (HF) exchange: 1.0 for HF, the
 *                           hybrid mixing fraction for DFT, 0.0 for a pure GGA
 * @param user_df_basis      explicit --df-basis/--aux value ("" if unset)
 * @param user_cosx          explicit --cosx flag
 * @param df_exchange_bytes  size of the DF-K intermediate, 0 if unknown
 *
 * Explicit user settings always win; Auto only fills in choices left unset.
 * The Auto rule (ORCA-style): density-fit the Coulomb term for every SCF
 * method, and for exact exchange use DF-K below the basis-function crossover,
 * seminumerical COSX above it.
 */
/**
 * @brief Memory budget for the DF exchange intermediate, in bytes.
 *
 * DF-K half-transforms the three-centre integrals into B[i][mu][P], which is
 * nocc x nbf x ndf doubles. That intermediate cannot be blocked over the
 * auxiliary index -- K = sum_i B_i V^-1 B_i^T, and V^-1 couples every
 * auxiliary function to every other -- so its size is a hard requirement of
 * the method rather than something an implementation can trade away.
 *
 * COSX needs no such intermediate, which makes it the right choice once the
 * DF-K workspace stops fitting, independent of how many basis functions
 * there are.
 */
inline std::size_t df_exchange_memory_budget() {
#ifdef __EMSCRIPTEN__
  // The whole wasm heap is capped at 1 GB and the wavefunction, grids and
  // Fock matrices share it.
  constexpr std::size_t default_mib = 192;
#else
  constexpr std::size_t default_mib = 2048;
#endif
  // The budget is a property of the machine rather than of the calculation,
  // so it is set by the environment rather than the input file.
  if (const char *env = std::getenv("OCC_DF_EXCHANGE_BUDGET_MB")) {
    char *end = nullptr;
    const unsigned long long mib = std::strtoull(env, &end, 10);
    if (end != env && mib > 0)
      return static_cast<std::size_t>(mib) * 1024 * 1024;
    occ::log::warn("ignoring OCC_DF_EXCHANGE_BUDGET_MB='{}': expected a "
                   "positive number of MiB",
                   env);
  }
  return default_mib * 1024 * 1024;
}

/// Bytes DF-K would need for its half-transformed intermediate.
inline std::size_t df_exchange_intermediate_bytes(std::size_t nocc,
                                                  std::size_t nbf,
                                                  std::size_t ndf) {
  return nocc * nbf * ndf * sizeof(double);
}

inline AccelerationPlan
plan_acceleration(io::RIPolicy policy, const std::string &orbital_basis_name,
                  std::size_t nbf, double exact_exchange,
                  const std::string &user_df_basis, bool user_cosx,
                  std::size_t df_exchange_bytes = 0) {
  using occ::qm::FittingKind;
  using occ::qm::resolve_fitting_basis;

  AccelerationPlan plan;
  plan.df_basis = user_df_basis;
  plan.use_cosx = user_cosx;

  const bool has_exact_exchange = exact_exchange != 0.0;

  switch (policy) {
  case io::RIPolicy::None:
    // Conventional integrals unless the user explicitly asked for DF/COSX.
    break;
  case io::RIPolicy::JK:
    if (plan.df_basis.empty())
      plan.df_basis =
          resolve_fitting_basis(orbital_basis_name, FittingKind::JK);
    plan.use_cosx = false;
    break;
  case io::RIPolicy::COSX:
    if (plan.df_basis.empty())
      plan.df_basis =
          resolve_fitting_basis(orbital_basis_name, FittingKind::JK);
    plan.use_cosx = has_exact_exchange;
    break;
  case io::RIPolicy::Auto:
    // Only act if the user hasn't already chosen DF or COSX explicitly.
    if (plan.df_basis.empty() && !plan.use_cosx) {
      plan.df_basis =
          resolve_fitting_basis(orbital_basis_name, FittingKind::JK);
      if (has_exact_exchange &&
          nbf > static_cast<std::size_t>(occ::qm::cosx_nbf_crossover()))
        plan.use_cosx = true;
      // A basis-size threshold alone misses the case that actually fails: the
      // DF-K intermediate scales as nocc x nbf x ndf, so a modest orbital
      // basis with a large fitting basis can exceed the budget well below the
      // crossover. `df_exchange_bytes` is zero when the caller cannot size it.
      if (has_exact_exchange && !plan.use_cosx && df_exchange_bytes > 0 &&
          df_exchange_bytes > df_exchange_memory_budget()) {
        occ::log::info(
            "DF exchange would need {} MiB for its half-transformed "
            "integrals, over the {} MiB budget; using COSX instead",
            df_exchange_bytes / (1024 * 1024),
            df_exchange_memory_budget() / (1024 * 1024));
        plan.use_cosx = true;
      }
    }
    break;
  }
  return plan;
}

/**
 * @brief Apply the active acceleration policy (DF / COSX) to an SCF procedure.
 *
 * Works for both HartreeFock and DFT (both expose the DF and COSX setters).
 * COSX is only enabled for exact exchange, never for range-separated hybrids
 * (COSX cannot handle range separation).
 *
 * @param allow_cosx pass false for gradient-producing calculations (geometry
 *        optimisation, frequencies): COSX has no analytic gradient, so it is
 *        downgraded to DF exchange there.
 */
template <typename Proc>
void apply_acceleration(Proc &proc, std::size_t nbf, const io::OccInput &config,
                        bool allow_cosx = true) {
  double exchange_factor = 1.0; // HF: full exact exchange
  bool range_separated = false;
  if constexpr (std::is_same<Proc, occ::dft::DFT>::value) {
    exchange_factor = proc.exact_exchange_factor();
    range_separated = proc.range_separated_parameters().omega != 0.0;
  }
  const bool has_exact_exchange = exchange_factor != 0.0 || range_separated;
  // COSX now supports range separation (long-range erf exchange), so a
  // range-separated hybrid is eligible for the COSX crossover. Treat it as
  // having exact exchange even when the short-range fraction is zero (e.g.
  // LC-omegaPBE), since the long-range exact exchange still needs a K build.
  const double cosx_exchange = range_separated ? 1.0 : exchange_factor;

  // Size the DF-K intermediate so the policy can weigh memory as well as
  // basis size. The fitting basis has to be resolved and loaded to count its
  // functions; that happens again in set_density_fitting_basis, but only once
  // per SCF setup rather than per iteration.
  std::size_t df_exchange_bytes = 0;
  if (has_exact_exchange && config.method.ri_policy == io::RIPolicy::Auto &&
      config.basis.df_name.empty() && !config.method.use_cosx) {
    const auto aux_name = occ::qm::resolve_fitting_basis(
        config.basis.name, occ::qm::FittingKind::JK);
    if (!aux_name.empty()) {
      try {
        const auto &ao = proc.aobasis();
        auto aux = occ::gto::AOBasis::load(ao.atoms(), aux_name);
        const int electrons =
            ao.effective_nuclear_charge() - static_cast<int>(config.electronic.charge);
        const std::size_t nocc =
            static_cast<std::size_t>(std::max(electrons, 0) + 1) / 2;
        df_exchange_bytes =
            df_exchange_intermediate_bytes(nocc, nbf, aux.nbf());
      } catch (const std::exception &e) {
        occ::log::debug("could not size the DF exchange intermediate: {}",
                        e.what());
      }
    }
  }

  AccelerationPlan accel = plan_acceleration(
      config.method.ri_policy, config.basis.name, nbf, cosx_exchange,
      config.basis.df_name, config.method.use_cosx, df_exchange_bytes);

  // COSX has no analytic gradient; gradient-producing drivers downgrade it to
  // DF exchange (the gradient itself stays on exact integrals regardless).
  if (!allow_cosx && accel.use_cosx) {
    occ::log::warn("COSX has no gradient; using DF exchange for this "
                   "gradient calculation");
    accel.use_cosx = false;
    if (accel.df_basis.empty())
      accel.df_basis = occ::qm::resolve_fitting_basis(
          config.basis.name, occ::qm::FittingKind::JK);
  }

  // Describe the resolved two-electron treatment in plain terms: how the
  // Coulomb (J) and exact-exchange (K) parts are evaluated. (Aligned labels:
  // "Coulomb  (J):" and "Exchange (K):" are both 13 characters.)
  if (accel.df_basis.empty() && !accel.use_cosx) {
    occ::log::info("Two-electron integrals: exact (4-center)");
  } else {
    if (config.method.use_split_ri_j)
      occ::log::info("Coulomb  (J): Split-RI-J density fitting [{}]",
                     accel.df_basis);
    else if (!accel.df_basis.empty())
      occ::log::info("Coulomb  (J): density fitting [{}]", accel.df_basis);
    else
      occ::log::info("Coulomb  (J): exact (4-center)");

    if (has_exact_exchange) {
      if (accel.use_cosx)
        occ::log::info("Exchange (K): seminumerical exchange (COSX, {})",
                       occ::numint::cosx_grid_level_to_string(
                           config.method.cosx_grid_level));
      else if (!accel.df_basis.empty())
        occ::log::info("Exchange (K): density fitting [{}]", accel.df_basis);
      else
        occ::log::info("Exchange (K): exact (4-center)");
    }
  }

  if (!accel.df_basis.empty()) {
    proc.set_density_fitting_basis(accel.df_basis,
                                   config.basis.df_auto_threshold);
    // Direct is the default -- storing the three-centre integrals needs
    // nbf^2 x ndf doubles, which is what exhausted the browser heap. The flag
    // is kept because it still forces the issue for a caller that has changed
    // the policy itself; on the default path it is a no-op.
    if (config.method.use_direct_df_kernels)
      proc.set_density_fitting_policy(occ::qm::IntegralEngineDF::Policy::Direct);
    if (config.method.use_split_ri_j)
      proc.set_coulomb_method(occ::qm::CoulombMethod::SplitRIJ);
  }

  if (accel.use_cosx) {
    proc.set_cosx_exchange(config.method.cosx_grid_level);
    occ::qm::cosx::Settings cosx_settings;
    cosx_settings.screen_threshold = config.method.cosx.screen_threshold;
    cosx_settings.margin = config.method.cosx.margin;
    cosx_settings.f_threshold = config.method.cosx.f_threshold;
    proc.set_cosx_settings(cosx_settings);
  }
}

} // namespace occ::driver
