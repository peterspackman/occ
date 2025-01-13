#pragma once
#include <nlohmann/json.hpp>
#include <occ/crystal/crystal.h>

namespace occ::crystal {

void to_json(nlohmann::json &j, const CrystalDimers &);
void from_json(const nlohmann::json &j, CrystalDimers &);

void to_json(nlohmann::json &j, const AsymmetricUnit &);
void from_json(const nlohmann::json &j, AsymmetricUnit &);

void to_json(nlohmann::json &j, const UnitCell &);
void from_json(const nlohmann::json &j, UnitCell &);

void to_json(nlohmann::json &j, const CrystalAtomRegion &);
void from_json(const nlohmann::json &j, CrystalAtomRegion &);

} // namespace occ::crystal

namespace nlohmann {
template <> struct adl_serializer<occ::crystal::SymmetryOperation> {
  static occ::crystal::SymmetryOperation from_json(const json &);
  static void to_json(json &j, const occ::crystal::SymmetryOperation &);
};

template <> struct adl_serializer<occ::crystal::SpaceGroup> {
  static occ::crystal::SpaceGroup from_json(const json &);
  static void to_json(json &j, const occ::crystal::SpaceGroup &);
};

template <> struct adl_serializer<occ::crystal::Crystal> {
  static occ::crystal::Crystal from_json(const json &);
  static void to_json(json &j, const occ::crystal::Crystal &);
};

} // namespace nlohmann
