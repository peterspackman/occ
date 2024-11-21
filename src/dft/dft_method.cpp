#include <ankerl/unordered_dense.h>
#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/dft/dft_method.h>

namespace occ::dft {

using dfid = DensityFunctional::Identifier;

struct FuncComponent {
  dfid id;
  double factor{1.0};
  double hfx{0.0};
};

inline const ankerl::unordered_dense::map<std::string,
                                          std::vector<FuncComponent>>
    builtin_functionals({
        {"b3lyp", {{dfid::hyb_gga_xc_b3lyp, 1.0}}},
        {"b3pw91", {{dfid::hyb_gga_xc_b3pw91, 1.0}}},
        {"b3p86", {{dfid::hyb_gga_xc_b3p86, 1.0}}},
        {"o3lyp", {{dfid::hyb_gga_xc_o3lyp, 1.0}}},
        {"pbeh", {{dfid::hyb_gga_xc_pbeh}}},
        {"b97", {{dfid::hyb_gga_xc_b97}}},
        {"b971", {{dfid::hyb_gga_xc_b97_1}}},
        {"b972", {{dfid::hyb_gga_xc_b97_2}}},
        {"x3lyp", {{dfid::hyb_gga_xc_x3lyp}}},
        {"b97k", {{dfid::hyb_gga_xc_b97_k}}},
        {"b973", {{dfid::hyb_gga_xc_b97_3}}},
        {"mpw3pw", {{dfid::hyb_gga_xc_mpw3pw}}},
        {"mpw3lyp", {{dfid::hyb_gga_xc_mpw3lyp}}},
        {"bhandh", {{dfid::hyb_gga_xc_bhandh}}},
        {"bhandhlyp", {{dfid::hyb_gga_xc_bhandhlyp}}},
        {"b3lyp5", {{dfid::hyb_gga_xc_b3lyp5}}},
        {"pbe1pbe", {{dfid::gga_x_pbe, 0.75, 0.25}, {dfid::gga_c_pbe}}},
        {"pbe", {{dfid::gga_x_pbe, 1.0}, {dfid::gga_c_pbe, 1.0}}},
        {"pbepbe", {{dfid::gga_x_pbe, 1.0}, {dfid::gga_c_pbe, 1.0}}},
        {"pbe0", {{dfid::gga_x_pbe, 0.75, 0.25}, {dfid::gga_c_pbe}}},
        {"svwn", {{dfid::lda_x}, {dfid::lda_c_vwn_3}}},
        {"lda", {{dfid::lda_x}, {dfid::lda_c_vwn_3}}},
        {"lsda", {{dfid::lda_x}, {dfid::lda_c_vwn_3}}},
        {"svwn5", {{dfid::lda_x}, {dfid::lda_c_vwn}}},
        {"blyp", {{dfid::gga_x_b88}, {dfid::gga_c_lyp}}},
        {"bpbe", {{dfid::gga_x_b88}, {dfid::gga_c_pbe}}},
        {"bp86", {{dfid::gga_x_b88}, {dfid::gga_c_p86}}},
        {"m062x", {{dfid::hyb_mgga_x_m06_2x}, {dfid::mgga_c_m06_2x}}},
        {"tpss", {{dfid::mgga_x_tpss}, {dfid::mgga_c_tpss}}},
        {"b86bpbe", {{dfid::gga_x_b86_mgc}, {dfid::gga_c_pbe}}},
        {"b86bpbeh", {{dfid::gga_x_b86_mgc, 0.75, 0.25}, {dfid::gga_c_pbe}}},
        {"r2scan", {{dfid::mgga_x_r2scan}, {dfid::mgga_c_r2scan}}},
        {"wb97x", {{dfid::hyb_gga_xc_wb97x}}},
        {"wb97m", {{dfid::hyb_mgga_xc_wb97m_v}}},
        {"wb97m-v", {{dfid::hyb_mgga_xc_wb97m_v}}},
    });

Functionals parse_dft_method(const std::string &method_string) {

  Functionals result;
  std::string method = occ::util::trim_copy(method_string);
  occ::util::to_lower(method);

  auto tokens = occ::util::tokenize(method, " ");
  occ::log::info("Functionals:");
  for (const auto &token : tokens) {
    std::string m = token;
    occ::log::debug("Token: {}", m);
    if (m[0] == 'u') {
      // TODO handle unrestricted convenience case
      m = m.substr(1);
    }
    if (builtin_functionals.contains(m)) {
      auto combo = builtin_functionals.at(m);
      occ::log::debug("Found builtin functional combination for {}", m);
      for (const auto &func : combo) {
        occ::log::debug("id: {}", static_cast<int>(func.id));

        auto f = DensityFunctional(func.id, false);
        auto fp = DensityFunctional(func.id, true);

        occ::log::debug("scale factor: {}", func.factor);

        f.set_scale_factor(func.factor);
        fp.set_scale_factor(func.factor);

        if (func.factor != 1.0) {
          occ::log::info("    {} x {}", func.factor, f.name());
        } else {
          occ::log::info("    {}", f.name());
        }
        if (func.hfx > 0.0) {
          f.set_exchange_factor(func.hfx);
          fp.set_exchange_factor(func.hfx);
        }
        result.unpolarized.push_back(f);
        result.polarized.push_back(fp);
      }
    } else {
      occ::log::info("    {}", token);
      result.unpolarized.emplace_back(token, false);
      result.polarized.emplace_back(token, true);
    }
  }
  return result;
}

} // namespace occ::dft
