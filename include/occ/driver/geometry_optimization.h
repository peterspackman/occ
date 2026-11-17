#pragma once
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>
#include <occ/core/vibration.h>

#include <optional>

namespace occ::xtb {
class XtbCalculator;
}

namespace occ::driver {

/// Charge, spin, electronic temperature and solvent from the input.
void configure_xtb_calculator(xtb::XtbCalculator &calc,
                              const io::OccInput &config);

/**
 * @brief Perform geometry optimization
 * 
 * @param config Input configuration
 * @return Optimized wavefunction
 */
/// For a saddle-point search, `initial_mode` (Cartesian, 3N) chooses the
/// mode followed at the start in place of the input's ts_mode.
qm::Wavefunction geometry_optimization(const io::OccInput &config,
                                       const Vec &initial_mode = {});

/**
 * @brief Energy and analytic gradient at the input geometry, evaluated as a
 * converged optimisation step would be (COSX on its gradient grids).
 *
 * @return Wavefunction and gradient in Hartree/Bohr
 */
std::pair<qm::Wavefunction, Mat3N>
energy_and_gradient(const io::OccInput &config);

/// Energy and analytic gradient (Hartree/Bohr) at a geometry, as in an
/// optimisation step: the SCF warm-started from `previous` when given, a
/// GFN2-xTB calculator kept in `gfn2` between calls when given.
std::pair<qm::Wavefunction, Mat3N>
energy_and_gradient_at(const io::OccInput &config, const core::Molecule &m,
                       const qm::Wavefunction *previous = nullptr,
                       std::optional<xtb::XtbCalculator> *gfn2 = nullptr);

/**
 * @brief Perform geometry optimization with optional vibrational analysis
 * 
 * @param config Input configuration
 * @param run_frequencies If true, compute vibrational frequencies after optimization
 * @return Pair of optimized wavefunction and vibrational modes (empty if not computed)
 */
std::pair<qm::Wavefunction, core::VibrationalModes>
geometry_optimization_with_frequencies(const io::OccInput &config,
                                       bool run_frequencies = true,
                                       const Vec &initial_mode = {});

} // namespace occ::driver
