#pragma once
#include <nlohmann/json.hpp>

namespace occ::core {

class Dimer;
class Molecule;
void to_json(nlohmann::json &j, const Dimer &);
void to_json(nlohmann::json &j, const Molecule &l);

} // namespace occ::core
