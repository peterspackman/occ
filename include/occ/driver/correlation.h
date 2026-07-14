#pragma once
#include <occ/qm/wavefunction.h>
#include <string>

namespace occ::driver {

/**
 * @brief Options for a post-HF correlation calculation on a converged SCF
 * wavefunction.
 *
 * All fields are plain strings/scalars so language bindings (Python, JS, Lua)
 * can expose them directly. Backend prefixes in `method` ("ri-mp2",
 * "thc-ccsd(t)") are honoured exactly as on the CLI.
 */
struct CorrelationOptions {
  std::string method{"mp2"}; ///< mp2 | ccsd | ccsd(t) (with optional ri-/df-/thc- prefix)
  std::string backend{"auto"};      ///< auto | conventional/exact | ri/df | thc
  std::string aux_basis{};          ///< empty = auto-resolve (correlation fitting)
  std::string spin_scaling{"none"}; ///< none | scs | sos (MP2 only)
  int n_frozen{-1};       ///< -1 = auto (chemical core), 0 = all-electron
  double max_memory_gb{1.0}; ///< integral-build memory budget
  int max_cycle{100};        ///< CC amplitude iterations
  double tol{1e-9};          ///< CC convergence on the correlation energy
  // THC backend knobs
  double thc_c_isdf{6.0};                  ///< THC rank = c * n_select
  std::string thc_isdf_method{"cholesky"}; ///< cholesky | qr
  int thc_grid_angular{110};               ///< candidate-grid max angular pts
  double thc_grid_radial_precision{1e-7};  ///< candidate-grid radial precision
  int laplace_points{14};                  ///< THC-MP2 Laplace quadrature points
};

/**
 * @brief Result of a post-HF correlation calculation. Components that do not
 * apply to the method run are left at zero.
 */
struct CorrelationResult {
  std::string method;    ///< canonical label, e.g. "RI-MP2", "CCSD(T)"
  double scf_energy{0.0};
  double correlation_energy{0.0}; ///< correlation used in `total_energy`
                                  ///< (incl. spin scaling and triples)
  double total_energy{0.0};
  double same_spin{0.0};          ///< MP2 components
  double opposite_spin{0.0};
  double scaled_correlation{0.0}; ///< SCS/SOS-MP2 correlation
  double ccsd_correlation{0.0};   ///< CCSD correlation (excl. triples)
  double triples_correction{0.0}; ///< perturbative (T)
  int iterations{0};              ///< CC iterations (0 for MP2)
  bool converged{true};
  int n_frozen{0}; ///< frozen occupied orbitals actually used
};

/**
 * @brief Run an MP2 / CCSD / CCSD(T) calculation on a converged SCF
 * wavefunction, with the same backend dispatch, auxiliary-basis resolution and
 * frozen-core handling as the CLI. Handles restricted and unrestricted
 * references.
 */
CorrelationResult run_correlation(const qm::Wavefunction &wfn,
                                  const CorrelationOptions &opts = {});

} // namespace occ::driver
