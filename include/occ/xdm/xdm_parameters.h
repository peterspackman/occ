#pragma once
#include <occ/xdm/xdm.h>
#include <string>
#include <optional>

namespace occ::xdm {

/**
 * @brief Get functional-specific XDM damping parameters
 *
 * Returns XDM damping parameters (a1, a2) for the given functional name.
 * Parameters are based on literature values.
 *
 * @param functional Name of the functional (case-insensitive)
 * @return XDM::Parameters if found, std::nullopt otherwise
 *
 * References:
 * - Becke & Johnson, J. Chem. Phys. 127, 154108 (2007)
 * - Otero-de-la-Roza & Johnson, J. Chem. Phys. 138, 204109 (2013)
 */
std::optional<XDM::Parameters> get_xdm_parameters(const std::string &functional);

} // namespace occ::xdm
