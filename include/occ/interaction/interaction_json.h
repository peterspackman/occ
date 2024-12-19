#pragma once
#include <nlohmann/json.hpp>
#include <occ/interaction/pairinteraction.h>

namespace occ::interaction {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CEEnergyComponents, coulomb, exchange,
                                   repulsion, polarization, dispersion, total)

} // namespace occ::interaction
