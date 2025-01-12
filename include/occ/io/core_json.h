#pragma once
#include <nlohmann/json.hpp>
#include <occ/core/dimer.h>

namespace occ::core {

void to_json(nlohmann::json &j, const Molecule &);
void from_json(const nlohmann::json &j, Molecule &);

} // namespace occ::core

namespace nlohmann {
template <> struct adl_serializer<occ::core::Dimer> {
  static occ::core::Dimer from_json(const json &);
  static void to_json(json &j, const occ::core::Dimer &);
};
} // namespace nlohmann
