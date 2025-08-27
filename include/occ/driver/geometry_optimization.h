#pragma once
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>
#include <occ/core/vibration.h>

namespace occ::driver {

/**
 * @brief Perform geometry optimization
 * 
 * @param config Input configuration
 * @return Optimized wavefunction
 */
qm::Wavefunction geometry_optimization(const io::OccInput &config);

/**
 * @brief Perform geometry optimization with optional vibrational analysis
 * 
 * @param config Input configuration
 * @param run_frequencies If true, compute vibrational frequencies after optimization
 * @return Pair of optimized wavefunction and vibrational modes (empty if not computed)
 */
std::pair<qm::Wavefunction, core::VibrationalModes> 
geometry_optimization_with_frequencies(const io::OccInput &config, bool run_frequencies = true);

} // namespace occ::driver
