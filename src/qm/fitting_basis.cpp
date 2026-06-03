#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/qm/fitting_basis.h>
#include <unordered_map>

namespace fs = std::filesystem;

namespace occ::qm {

namespace {

struct FittingDefaults {
  std::string jk_default{"def2-universal-jkfit"};
  std::string corr_default{"def2-tzvp-rifit"};
  int cosx_nbf_crossover{600};
  // orbital basis name -> {jk, corr}; either may be empty (use default).
  std::unordered_map<std::string, std::pair<std::string, std::string>> map;
};

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

// Normalise an orbital basis name to the form used as a key in the JSON map:
// lowercase, without a trailing ".json".
std::string canonical_key(const std::string &name) {
  std::string key = to_lower(name);
  if (key.ends_with(".json"))
    key.erase(key.size() - 5);
  return key;
}

FittingDefaults load_defaults() {
  FittingDefaults defaults;

  const char *dir = occ::get_data_directory();
  std::string base = dir ? std::string(dir) : std::string(".");
  fs::path path = fs::path(base) / "basis" / "fitting_defaults.json";

  if (!fs::exists(path)) {
    occ::log::debug("fitting_defaults.json not found at {}; using built-in "
                    "auxiliary basis defaults",
                    path.string());
    return defaults;
  }

  try {
    std::ifstream file(path.string());
    nlohmann::json j;
    file >> j;

    if (j.contains("scf_jk_default"))
      defaults.jk_default = j.at("scf_jk_default").get<std::string>();
    if (j.contains("corr_default"))
      defaults.corr_default = j.at("corr_default").get<std::string>();
    if (j.contains("policy") && j.at("policy").contains("cosx_nbf_crossover"))
      defaults.cosx_nbf_crossover =
          j.at("policy").at("cosx_nbf_crossover").get<int>();

    if (j.contains("map")) {
      for (const auto &[name, entry] : j.at("map").items()) {
        std::string jk = entry.value("jk", std::string{});
        std::string corr = entry.value("corr", std::string{});
        defaults.map[canonical_key(name)] = {jk, corr};
      }
    }
    occ::log::debug("Loaded {} auxiliary basis defaults from {}",
                    defaults.map.size(), path.string());
  } catch (const std::exception &e) {
    occ::log::warn("Failed to parse {}: {}; using built-in auxiliary basis "
                   "defaults",
                   path.string(), e.what());
    return FittingDefaults{};
  }

  return defaults;
}

const FittingDefaults &defaults() {
  static const FittingDefaults instance = load_defaults();
  return instance;
}

} // namespace

std::string resolve_fitting_basis(const std::string &orbital_basis_name,
                                  FittingKind kind) {
  const FittingDefaults &d = defaults();
  const std::string &fallback =
      (kind == FittingKind::JK) ? d.jk_default : d.corr_default;

  auto it = d.map.find(canonical_key(orbital_basis_name));
  if (it != d.map.end()) {
    const std::string &mapped =
        (kind == FittingKind::JK) ? it->second.first : it->second.second;
    if (!mapped.empty())
      return mapped;
  }
  return fallback;
}

int cosx_nbf_crossover() { return defaults().cosx_nbf_crossover; }

} // namespace occ::qm
