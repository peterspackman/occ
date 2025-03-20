#include <ankerl/unordered_dense.h>
#include <filesystem>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/dft/dft_method.h>

namespace fs = std::filesystem;

namespace occ::dft {

using dfid = DensityFunctional::Identifier;

inline const ankerl::unordered_dense::map<std::string, MethodDefinition>
    builtin_definitions = {
        {"b3lyp", {{{dfid::hyb_gga_xc_b3lyp, 1.0}}}},
        {"b3pw91", {{{dfid::hyb_gga_xc_b3pw91, 1.0}}}},
        {"pbeh", {{{dfid::hyb_gga_xc_pbeh}}}},
        {"b97", {{{dfid::hyb_gga_xc_b97}}}},
        {"b3lyp5", {{{dfid::hyb_gga_xc_b3lyp5}}}},
        {"pbe1pbe", {{{dfid::gga_x_pbe, 0.75, 0.25}, {dfid::gga_c_pbe}}}},
        {"pbe", {{{dfid::gga_x_pbe, 1.0}, {dfid::gga_c_pbe, 1.0}}}},
        {"pbepbe", {{{dfid::gga_x_pbe, 1.0}, {dfid::gga_c_pbe, 1.0}}}},
        {"pbe0", {{{dfid::gga_x_pbe, 0.75, 0.25}, {dfid::gga_c_pbe}}}},
        {"svwn", {{{dfid::lda_x}, {dfid::lda_c_vwn_3}}}},
        {"lda", {{{dfid::lda_x}, {dfid::lda_c_vwn_3}}}},
        {"lsda", {{{dfid::lda_x}, {dfid::lda_c_vwn_3}}}},
        {"svwn5", {{{dfid::lda_x}, {dfid::lda_c_vwn}}}},
        {"blyp", {{{dfid::gga_x_b88}, {dfid::gga_c_lyp}}}},
        {"bpbe", {{{dfid::gga_x_b88}, {dfid::gga_c_pbe}}}},
        {"bp86", {{{dfid::gga_x_b88}, {dfid::gga_c_p86}}}},
        {"m062x", {{{dfid::hyb_mgga_x_m06_2x}, {dfid::mgga_c_m06_2x}}}},
        {"tpss", {{{dfid::mgga_x_tpss}, {dfid::mgga_c_tpss}}}},
        {"r2scan", {{{dfid::mgga_x_r2scan}, {dfid::mgga_c_r2scan}}}},
        {"wb97x", {{{dfid::hyb_gga_xc_wb97x}}}},
        {"wb97m", {{{dfid::hyb_mgga_xc_wb97m_v}}, "vv10"}},
        {"wb97m-v", {{{dfid::hyb_mgga_xc_wb97m_v}}, "vv10"}}};

double DFTMethod::exchange_factor() const {
  for (const auto &func : functionals) {
    double ex_factor = func.exact_exchange_factor();
    if (ex_factor > 0.0) {
      return ex_factor;
    }
  }
  return 0.0;
}

RangeSeparatedParameters DFTMethod::range_separated_parameters() const {
  for (const auto &func : functionals) {
    auto rs = func.range_separated_parameters();
    if (rs.omega != 0.0)
      return rs;
  }
  return {};
}

inline std::string data_path() {
  std::string path{"."};
  const char *data_path_env = occ::get_data_directory();

  if (data_path_env) {
    path = data_path_env;
  }
  // Construct the methods directory path
  std::string methods_dir = path + "/methods";
  std::string methods_file = methods_dir + "/dft_methods.json";

  // Check if the methods file exists
  bool file_exists = fs::exists(methods_file);

  if (!file_exists) {
    // Try to look in the current directory
    occ::log::warn("DFT methods file not found at '{}', "
                   "checking current directory",
                   methods_file);

    methods_file = "dft_methods.json";
    file_exists = fs::exists(methods_file);

    if (!file_exists) {
      // Still not found, provide a warning but return the path anyway
      // (the caller might create the file)
      occ::log::warn("DFT methods file not found in current directory either");
      methods_file =
          methods_dir + "/dft_methods.json"; // Return the original path for
                                             // potential creation
    } else {
      occ::log::info("Using DFT methods file from current directory");
    }
  }

  return methods_file;
}

inline ankerl::unordered_dense::map<std::string, MethodDefinition>
get_method_definitions() {
  std::string methods_file = data_path();

  if (fs::exists(methods_file)) {
    try {
      std::ifstream file(methods_file);
      nlohmann::json j;
      file >> j;

      ankerl::unordered_dense::map<std::string, MethodDefinition>
          loaded_methods;

      if (j.contains("methods") && j["methods"].is_object()) {
        const auto &methods_json = j["methods"];

        for (auto it = methods_json.begin(); it != methods_json.end(); ++it) {
          MethodDefinition def;
          from_json(it.value(), def);
          loaded_methods[it.key()] = def;
        }

        occ::log::info("Loaded {} DFT method definitions from {}",
                       loaded_methods.size(), methods_file);
        return loaded_methods;
      } else {
        occ::log::warn(
            "Invalid format in DFT methods file: missing 'methods' object");
      }
    } catch (const std::exception &e) {
      occ::log::error("Error loading DFT methods from file: {}", e.what());
    }
  }

  occ::log::info("Using builtin DFT method definitions");
  return builtin_definitions;
}

DFTMethod create_dft_method_from_definition(const MethodDefinition &def) {

  DFTMethod result;
  result.dispersion = def.dispersion;
  result.gcp = def.gcp;
  result.basis_set = def.basis_set;

  for (const auto &comp : def.components) {
    auto func = DensityFunctional(comp.id, false);
    auto func_pol = DensityFunctional(comp.id, true);

    if (comp.factor != 1.0) {
      func.set_scale_factor(comp.factor);
      func_pol.set_scale_factor(comp.factor);
    }

    if (comp.hfx > 0.0) {
      func_pol.set_exchange_factor(comp.hfx);
    }

    result.functionals.push_back(func);
    result.functionals_polarized.push_back(func_pol);
  }

  return result;
}

DFTMethod get_dft_method(const std::string &method_string) {
  DFTMethod result;
  std::string method = occ::util::trim_copy(method_string);
  occ::util::to_lower(method);
  auto tokens = occ::util::tokenize(method, " ");
  occ::log::info("Functionals:");

  auto definitions = get_method_definitions();

  for (const auto &token : tokens) {
    std::string m = token;
    occ::log::debug("Token: {}", m);

    bool is_polarized = false;
    if (!m.empty() && m[0] == 'u') {
      is_polarized = true;
      m = m.substr(1);
    }

    if (definitions.contains(m)) {
      const auto &def = definitions.at(m);
      occ::log::debug("Found definition: {}", m);

      if (!def.dispersion.empty() && result.dispersion.empty()) {
        result.dispersion = def.dispersion;
        occ::log::info("    Dispersion: {}", def.dispersion);
      }

      if (!def.gcp.empty() && result.gcp.empty()) {
        result.gcp = def.gcp;
        occ::log::info("    GCP: {}", def.gcp);
      }

      if (!def.basis_set.empty() && result.basis_set.empty()) {
        result.basis_set = def.basis_set;
        occ::log::info("    Basis set: {}", def.basis_set);
      }

      for (const auto &comp : def.components) {
        occ::log::debug("Component id: {}", static_cast<int>(comp.id));
        auto func = DensityFunctional(comp.id, false);
        auto func_pol = DensityFunctional(comp.id, true);

        if (comp.factor != 1.0) {
          func.set_scale_factor(comp.factor);
          func_pol.set_scale_factor(comp.factor);
          occ::log::info("    {} x {}", comp.factor, func.name());
        } else {
          occ::log::info("    {}", func.name());
        }

        if (comp.hfx > 0.0) {
          func_pol.set_exchange_factor(comp.hfx);
        }

        result.functionals.push_back(func);
        result.functionals_polarized.push_back(func_pol);
      }
    } else {
      occ::log::info("    {} (custom)", m);
      result.functionals.emplace_back(m, false);
      result.functionals_polarized.emplace_back(m, true);
    }
  }

  return result;
}

} // namespace occ::dft

void to_json(nlohmann::json &j, const occ::dft::FuncComponent &fc) {
  j = nlohmann::json{{"id", occ::dft::dfid_to_string(fc.id)}};
  if (fc.factor != 1.0)
    j["factor"] = fc.factor;
  if (fc.hfx != 0.0)
    j["hfx"] = fc.hfx;
}

void from_json(const nlohmann::json &j, occ::dft::FuncComponent &fc) {
  fc.id = occ::dft::string_to_dfid(j.at("id").get<std::string>());
  fc.factor = j.value("factor", 1.0);
  fc.hfx = j.value("hfx", 0.0);
}

void to_json(nlohmann::json &j, const occ::dft::MethodDefinition &def) {
  j = nlohmann::json();
  j["components"] = nlohmann::json::array();
  for (const auto &comp : def.components) {
    nlohmann::json cj;
    to_json(cj, comp);
    j["components"].push_back(cj);
  }

  if (!def.dispersion.empty())
    j["dispersion"] = def.dispersion;
  if (!def.gcp.empty())
    j["gcp"] = def.gcp;
  if (!def.basis_set.empty())
    j["basis_set"] = def.basis_set;
}

void from_json(const nlohmann::json &j, occ::dft::MethodDefinition &def) {
  def.components.clear();
  for (const auto &comp_json : j["components"]) {
    occ::dft::FuncComponent comp;
    from_json(comp_json, comp);
    def.components.push_back(comp);
  }

  def.dispersion = j.value("dispersion", "");
  def.gcp = j.value("gcp", "");
  def.basis_set = j.value("basis_set", "");
}
