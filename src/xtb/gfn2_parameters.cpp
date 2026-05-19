#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/xtb/gfn2_parameters.h>
#include <stdexcept>

namespace occ::xtb {

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

ShellParam shell_from_json(const json &j) {
  ShellParam s;
  s.n = j.at("n").get<int>();
  s.l = j.at("l").get<int>();
  s.n_prim = j.at("n_prim").get<int>();
  s.is_valence = j.at("is_valence").get<bool>();
  s.self_energy_ev = j.at("self_energy_ev").get<double>();
  s.slater_exponent = j.at("slater_exponent").get<double>();
  s.kcn_au = j.at("kcn_au").get<double>();
  s.shell_poly = j.at("shell_poly").get<double>();
  s.ref_occ = j.at("ref_occ").get<double>();
  s.shell_hardness_au = j.at("shell_hardness_au").get<double>();
  return s;
}

ElementParam element_from_json(const json &j) {
  ElementParam e;
  e.z = j.at("z").get<int>();
  e.ao = j.at("ao").get<std::string>();
  e.pauling_en = j.at("pauling_en").get<double>();
  e.atomic_hardness = j.at("atomic_hardness").get<double>();
  e.third_order_atom_au = j.at("third_order_atom_au").get<double>();
  e.rep_alpha = j.at("rep_alpha").get<double>();
  e.rep_zeff = j.at("rep_zeff").get<double>();
  e.dip_kernel = j.at("dip_kernel").get<double>();
  e.quad_kernel = j.at("quad_kernel").get<double>();
  for (const auto &js : j.at("shells")) {
    e.shells.push_back(shell_from_json(js));
  }
  return e;
}

double get_or(const json &j, const char *key, double fallback) {
  auto it = j.find(key);
  return (it != j.end() && !it->is_null()) ? it->get<double>() : fallback;
}

GlobalParam globals_from_json(const json &j) {
  GlobalParam g;
  g.kshell[0] = get_or(j, "ks", 0.0);
  g.kshell[1] = get_or(j, "kp", 0.0);
  g.kshell[2] = get_or(j, "kd", 0.0);
  g.kshell[3] = get_or(j, "kf", 0.0);
  g.ksp = get_or(j, "ksp", 0.0);
  g.ksd = get_or(j, "ksd", 0.0);
  g.kpd = get_or(j, "kpd", 0.0);
  g.kdiff = get_or(j, "kdiff", 0.0);
  g.enscale = get_or(j, "enscale", 0.0);
  g.enscale4 = get_or(j, "enscale4", 0.0);
  g.ipeashift_au = get_or(j, "ipeashift_au", 0.0);
  g.aesshift = get_or(j, "aesshift", 0.0);
  g.aesexp = get_or(j, "aesexp", 0.0);
  g.aesrmax = get_or(j, "aesrmax", 0.0);
  g.aesdmp3 = get_or(j, "aesdmp3", 0.0);
  g.aesdmp5 = get_or(j, "aesdmp5", 0.0);
  g.alphaj = get_or(j, "alphaj", 0.0);
  g.a1 = get_or(j, "a1", 0.0);
  g.a2 = get_or(j, "a2", 0.0);
  g.s6 = get_or(j, "s6", 1.0);
  g.s8 = get_or(j, "s8", 0.0);
  g.s9 = get_or(j, "s9", 0.0);
  g.kexp = get_or(j, "kexp", 0.0);
  g.kexplight = get_or(j, "kexplight", 0.0);
  auto it = j.find("gam3shell");
  if (it != j.end() && it->is_array()) {
    for (size_t l = 0; l < 4 && l < it->size(); ++l) {
      const auto &row = it->at(l);
      if (row.is_array() && row.size() >= 2) {
        g.gam3shell[l][0] = row.at(0).get<double>();
        g.gam3shell[l][1] = row.at(1).get<double>();
      }
    }
  }
  return g;
}

std::string default_param_path() {
  const char *env = occ::get_data_directory();
  std::string base = env ? std::string(env) : std::string(".");
  return base + "/xtb/gfn2.json";
}

Gfn2Parameters parse(const json &j) {
  Gfn2Parameters p;
  p.set_method(j.value("method", std::string{"GFN2-xTB"}));
  p.set_doi(j.value("doi", std::string{}));
  p.set_globals(globals_from_json(j.at("globals")));
  for (const auto &je : j.at("elements")) {
    p.add_element(element_from_json(je));
  }
  return p;
}

} // namespace

Gfn2Parameters Gfn2Parameters::load(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open GFN2 parameter file: " + path);
  }
  json j;
  try {
    in >> j;
  } catch (const json::parse_error &e) {
    throw std::runtime_error("GFN2 parameter file is not valid JSON (" + path +
                             "): " + e.what());
  }
  return parse(j);
}

Gfn2Parameters Gfn2Parameters::load_default() {
  std::string path = default_param_path();
  if (!fs::exists(path)) {
    occ::log::warn("GFN2 parameter file not found at '{}', "
                   "checking current directory",
                   path);
    if (fs::exists("xtb/gfn2.json")) {
      path = "xtb/gfn2.json";
    } else if (fs::exists("gfn2.json")) {
      path = "gfn2.json";
    } else {
      throw std::runtime_error("Cannot locate GFN2 parameter file (looked at "
                               "share/xtb/gfn2.json, xtb/gfn2.json, "
                               "gfn2.json)");
    }
  }
  return load(path);
}

const ElementParam *Gfn2Parameters::element(int z) const {
  if (z < 1 || z > static_cast<int>(m_elements.size())) {
    return nullptr;
  }
  // elements are stored in Z-order; verify and look up.
  const auto &e = m_elements[z - 1];
  if (e.z == z)
    return &e;
  // Fallback: linear search if order isn't dense (shouldn't happen).
  for (const auto &cand : m_elements) {
    if (cand.z == z)
      return &cand;
  }
  return nullptr;
}

} // namespace occ::xtb
