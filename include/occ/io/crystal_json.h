#pragma once
#include <nlohmann/json.hpp>
#include <occ/crystal/crystal.h>

namespace occ::crystal {

void to_json(nlohmann::json &j, const Crystal &);
void to_json(nlohmann::json &j, const AsymmetricUnit &);
void to_json(nlohmann::json &j, const SpaceGroup &);
void to_json(nlohmann::json &j, const SymmetryOperation &);
void to_json(nlohmann::json &j, const UnitCell &);
void to_json(nlohmann::json &j, const CrystalAtomRegion &);
void to_json(nlohmann::json &j, const CrystalDimers &);

} // namespace occ::crystal
