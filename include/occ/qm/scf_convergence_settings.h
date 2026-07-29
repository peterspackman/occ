#pragma once

namespace occ::qm {

/// Strategy for DIIS extrapolation during SCF
enum class DiisStrategy {
  CDIIS,       ///< Standard commutator DIIS only
  EDIIS_CDIIS, ///< EDIIS early, switch to CDIIS when error is small
  ADIIS_CDIIS  ///< ADIIS early, switch to CDIIS when error is small (recommended)
};

struct SCFConvergenceSettings {
  double energy_threshold{1e-6};
  double commutator_threshold{1e-5};
  // Start accumulating F += G(ΔD) once the SCF is in the well-behaved regime.
  // This shares the ADIIS→CDIIS switch point below: earlier than that the
  // density still swings enough that ΔD is no smaller than D, so incremental
  // builds screen no better while accumulating drift. Must stay looser than
  // `commutator_threshold`, or the SCF converges before it can ever engage.
  double incremental_fock_threshold{1e-2};

  // DIIS strategy settings
  DiisStrategy diis_strategy{DiisStrategy::ADIIS_CDIIS};
  double diis_switch_threshold{0.01};  // Switch from ADIIS/EDIIS to CDIIS when error below this

  // Level shifting: applied to virtual orbitals to stabilize convergence
  double level_shift{0.3};  // Hartree, applied when diis_error > level_shift_threshold
  double level_shift_threshold{0.1};  // Only apply level shift when error is large

  inline bool energy_converged(double energy_difference) const {
    return energy_difference < energy_threshold;
  }

  inline bool commutator_converged(double commutator_difference) const {
    return commutator_difference < commutator_threshold;
  }

  inline bool energy_and_commutator_converged(double ediff,
                                              double cdiff) const {
    return energy_converged(ediff) && commutator_converged(cdiff);
  }

  inline bool start_incremental_fock(double diis_error) const {
    return diis_error < incremental_fock_threshold;
  }

  inline double effective_level_shift(double diis_error) const {
    // Apply level shift only when error is large
    if (diis_error > level_shift_threshold) {
      return level_shift;
    }
    return 0.0;
  }
};

} // namespace occ::qm
