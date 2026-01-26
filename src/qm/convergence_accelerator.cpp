#include <occ/core/log.h>
#include <occ/qm/convergence_accelerator.h>

namespace occ::qm {

ConvergenceAccelerator::ConvergenceAccelerator(DiisStrategy strategy,
                                               double switch_threshold)
    : m_strategy{strategy}, m_switch_threshold{switch_threshold} {}

void ConvergenceAccelerator::reset() {
  m_cdiis = CDIIS{};
  m_ediis.reset();
  m_adiis.reset();
  m_error = 1.0;
  m_using_cdiis = false;
}

Mat ConvergenceAccelerator::update(SpinorbitalKind kind, const Mat &S,
                                   const Mat &D, const Mat &F, double energy) {
  // Always use CDIIS to get error estimate
  Mat F_cdiis = m_cdiis.update(S, D, F);
  m_error = m_cdiis.max_error();

  // For CDIIS-only strategy, we're done
  if (m_strategy == DiisStrategy::CDIIS) {
    return F_cdiis;
  }

  // For hybrid strategies, use ADIIS/EDIIS when error is large
  if (m_error > m_switch_threshold) {
    if (m_strategy == DiisStrategy::ADIIS_CDIIS) {
      return m_adiis.update(kind, D, F);
    } else { // EDIIS_CDIIS
      return m_ediis.update(kind, D, F, energy);
    }
  }

  // Error is small enough, switch to CDIIS
  if (!m_using_cdiis) {
    m_using_cdiis = true;
    const char *from = (m_strategy == DiisStrategy::ADIIS_CDIIS) ? "ADIIS" : "EDIIS";
    occ::log::debug("Switching from {} to CDIIS (error {:.2e} < threshold {:.2e})",
                    from, m_error, m_switch_threshold);
  }
  return F_cdiis;
}

} // namespace occ::qm
