#pragma once
#include <occ/core/vibration.h>
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>

namespace occ::driver {

/**
 * @brief Configuration options for vibrational frequency analysis
 */
struct VibrationalAnalysisConfig {
    bool compute_frequencies = true;          ///< Compute vibrational frequencies
    bool project_tr_rot = false;              ///< Project out translation/rotation modes
    double step_size = 0.005;                 ///< Step size for finite differences (Bohr)
    bool use_acoustic_sum_rule = true;        ///< Use acoustic sum rule optimization
    bool save_hessian = false;                ///< Save Hessian matrix to file
    std::string hessian_filename = "hessian.json"; ///< Filename for saved Hessian
    bool save_results = false;                ///< Save vibrational analysis results
    std::string results_filename = "frequencies.json"; ///< Filename for saved results
};

/**
 * @brief Perform vibrational frequency analysis on an optimized geometry
 * 
 * This function computes the molecular Hessian using finite differences
 * and performs normal mode analysis to obtain vibrational frequencies.
 * It can be called after geometry optimization to characterize the 
 * stationary point.
 * 
 * @param config Input configuration from OCC input file
 * @param wfn Converged wavefunction from optimization or single point
 * @param vib_config Configuration options for vibrational analysis
 * @return VibrationalModes Complete vibrational analysis results
 */
core::VibrationalModes vibrational_analysis(const io::OccInput &config,
                                           const qm::Wavefunction &wfn,
                                           const VibrationalAnalysisConfig &vib_config = {});

/**
 * @brief Convenience function for standard frequency analysis
 * 
 * Uses default settings optimized for most common use cases:
 * - Finite differences with acoustic sum rule
 * - Step size: 0.005 Bohr
 * - No projection of translational/rotational modes
 * 
 * @param config Input configuration from OCC input file
 * @param wfn Converged wavefunction from optimization or single point
 * @return VibrationalModes Complete vibrational analysis results
 */
core::VibrationalModes vibrational_analysis(const io::OccInput &config,
                                           const qm::Wavefunction &wfn);

} // namespace occ::driver