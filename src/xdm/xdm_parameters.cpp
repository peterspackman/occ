#include <occ/xdm/xdm_parameters.h>
#include <occ/core/util.h>
#include <unordered_map>

namespace occ::xdm {

std::optional<XDM::Parameters> get_xdm_parameters(const std::string &functional) {
  // XDM damping parameters from aug-cc-pvtz basis set
  // Source: https://github.com/aoterodelaroza/postg/blob/master/dat/xdm.param
  // These are basis-set independent approximations - best with large basis sets
  static const std::unordered_map<std::string, XDM::Parameters> params = {
    // Pure GGA functionals
    {"hf",         {0.3698, 2.1961}},
    {"pw86pbe",    {0.7564, 1.4545}},
    {"b86bpbe",    {0.7839, 1.2544}},
    {"pw86lda",    {0.7266, 1.1188}},
    {"pw86hahlda", {0.2661, 2.6661}},
    {"blyp",       {0.7647, 0.8457}},
    {"pbe",        {0.4492, 2.5517}},
    {"olyp",       {0.6170, 0.8653}},
    {"bpbe",       {0.5023, 1.3961}},
    {"bp86",       {0.8538, 0.6415}},

    // Hybrid functionals
    {"b3lyp",      {0.6356, 1.5119}},
    {"b3pw91",     {0.6002, 1.4043}},
    {"b3p86",      {1.0400, 0.3741}},
    {"pbe0",       {0.4186, 2.6791}},
    {"camb3lyp",   {0.3248, 2.8607}},
    {"b97-1",      {0.1998, 3.5367}},
    {"bhalfandhalf", {0.5610, 1.9894}},
    {"bhahlyp",    {0.5610, 1.9894}},  // Alias
    {"hse06",      {0.3691, 2.8793}},
    {"pw86hse",    {0.7201, 1.5033}},

    // Meta-GGA functionals
    {"tpss",       {0.6612, 1.5111}},

    // Range-separated functionals
    {"lcwpbe",       {1.0149, 0.6755}},
    {"lc-wpbe",      {1.0149, 0.6755}},  // Alias
    {"lcwpbe(w=0.2)", {0.6897, 1.6455}},
    {"lcwpbe(w=0.4)", {1.0381, 0.6136}},

    // Dispersion-corrected (parameters for base functional)
    {"b97d",       {0.6484, 1.1589}},
    {"b97",       {0.6484, 1.1589}},
  };

  auto lc_func = occ::util::to_lower_copy(functional);
  auto it = params.find(lc_func);
  if (it != params.end()) {
    return it->second;
  }
  return std::nullopt;
}

} // namespace occ::xdm
