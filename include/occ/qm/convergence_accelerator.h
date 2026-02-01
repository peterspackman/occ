#pragma once
#include <occ/qm/adiis.h>
#include <occ/qm/cdiis.h>
#include <occ/qm/ediis.h>
#include <occ/qm/scf_convergence_settings.h>

namespace occ::qm {

/// Encapsulates SCF convergence acceleration strategies (CDIIS, EDIIS, ADIIS)
class ConvergenceAccelerator {
public:
  ConvergenceAccelerator(DiisStrategy strategy = DiisStrategy::ADIIS_CDIIS,
                         double switch_threshold = 0.1);

  /// Update with current SCF state, returns extrapolated Fock matrix
  Mat update(SpinorbitalKind kind, const Mat &S, const Mat &D, const Mat &F,
             double energy);

  /// Current commutator error estimate (from CDIIS)
  double error() const { return m_error; }

  /// Maximum error from CDIIS
  double max_error() const { return m_cdiis.max_error(); }

  /// Reset all internal DIIS histories
  void reset();

  /// Change strategy
  void set_strategy(DiisStrategy strategy) { m_strategy = strategy; }

  /// Change switch threshold
  void set_switch_threshold(double threshold) { m_switch_threshold = threshold; }

private:
  DiisStrategy m_strategy;
  double m_switch_threshold;
  double m_error{1.0};
  bool m_using_cdiis{false};  // Track if we've switched to CDIIS

  CDIIS m_cdiis;
  EDIIS m_ediis;
  ADIIS m_adiis;
};

} // namespace occ::qm
