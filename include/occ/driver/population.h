#pragma once
#include <occ/qm/wavefunction.h>
#include <string>
#include <vector>

namespace occ::driver {

/// Names accepted by partial_charges(), in a stable order.
const std::vector<std::string> &partial_charge_schemes();

/**
 * @brief Partial atomic charges (e) of a wavefunction by scheme.
 *
 * mulliken and lowdin partition the density matrix; chelpg fits the
 * wavefunction's ESP (for GFN2-xTB its point-multipole ESP); hirshfeld
 * partitions an all-electron density against free-atom densities; eeq uses
 * the geometry and total charge only; xtb returns the stored GFN2-xTB charges.
 * Throws for an unknown scheme, or one that does not apply to `wfn` (xtb
 * without xTB data, hirshfeld on a valence-only GFN2-xTB density).
 */
Vec partial_charges(const qm::Wavefunction &wfn, const std::string &scheme);

} // namespace occ::driver
