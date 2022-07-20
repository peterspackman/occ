#pragma once
#include <nlohmann/json.hpp>
#include <occ/core/dimer.h>

namespace occ::core {

void to_json(nlohmann::json &j, const Dimer &);
void to_json(nlohmann::json &j, const Molecule &);

} // namespace occ::core
