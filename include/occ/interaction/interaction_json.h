#pragma once
#include <nlohmann/json.hpp>
#include <occ/interaction/pairinteraction.h>

namespace occ::interaction {

void to_json(nlohmann::json &j, const CEEnergyComponents &);
void from_json(const nlohmann::json &j, CEEnergyComponents &);

} // namespace occ::interaction
