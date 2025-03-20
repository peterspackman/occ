#pragma once
#include <ankerl/unordered_dense.h>
#include <nlohmann/json.hpp>
#include <occ/dft/functional.h>
#include <string>
#include <vector>

namespace occ::dft {

using dfid = DensityFunctional::Identifier;

struct FuncComponent {
  dfid id;
  double factor{1.0};
  double hfx{0.0};
};

struct MethodDefinition {
  std::vector<FuncComponent> components;
  std::string dispersion;
  std::string gcp;
  std::string basis_set;
};

struct DFTMethod {
  std::vector<DensityFunctional> functionals;
  std::vector<DensityFunctional> functionals_polarized;
  std::string dispersion;
  std::string gcp;
  std::string basis_set;

  double exchange_factor() const;

  RangeSeparatedParameters range_separated_parameters() const;
};

DFTMethod get_dft_method(const std::string &method_string);

DFTMethod create_dft_method_from_definition(const MethodDefinition &def,
                                            bool polarized = false);

ankerl::unordered_dense::map<std::string, MethodDefinition>
load_method_definitions(const std::string &filename);

void export_method_definitions(
    const std::string &filename,
    const ankerl::unordered_dense::map<std::string, MethodDefinition>
        &definitions);

} // namespace occ::dft

void to_json(nlohmann::json &j, const occ::dft::FuncComponent &fc);
void from_json(const nlohmann::json &j, occ::dft::FuncComponent &fc);
void to_json(nlohmann::json &j, const occ::dft::MethodDefinition &def);
void from_json(const nlohmann::json &j, occ::dft::MethodDefinition &def);
