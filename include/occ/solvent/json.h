#pragma once
#include <nlohmann/json.hpp>
#include <occ/solvent/smd_parameters.h>

namespace occ::solvent {

void to_json(nlohmann::json &j, const SMDSolventParameters &);

void from_json(const nlohmann::json &j, SMDSolventParameters &);

} // namespace occ::solvent
