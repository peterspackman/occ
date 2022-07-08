#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/3rdparty/robin_hood.h>
#include <occ/core/atom.h>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>

namespace occ::dft {

using occ::qm::AOBasis;
using occ::qm::SpinorbitalKind;

using dfid = DensityFunctional::Identifier;

struct FuncComponent {
    dfid id;
    double factor{1.0};
    double hfx{0.0};
};

const robin_hood::unordered_map<std::string, std::vector<FuncComponent>>
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
    });

int DFT::density_derivative() const {
    int deriv = 0;
    for (const auto &func : m_funcs) {
        deriv = std::max(deriv, func.derivative_order());
    }
    return deriv;
}

DFT::DFT(const std::string &method, const AOBasis &basis, SpinorbitalKind kind)
    : m_spinorbital_kind(kind), m_hf(basis), m_grid(basis) {
    occ::log::debug("start calculating atom grids... ");
    for (size_t i = 0; i < basis.atoms().size(); i++) {
        m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
    }
    size_t num_grid_points = std::accumulate(
        m_atom_grids.begin(), m_atom_grids.end(), 0.0,
        [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
    occ::log::info("finished calculating atom grids ({} points)",
                   num_grid_points);
    occ::log::debug("Grid initialization took {} seconds",
                    occ::timing::total(occ::timing::grid_init));
    occ::log::debug("Grid point creation took {} seconds",
                    occ::timing::total(occ::timing::grid_points));
    m_funcs = parse_method(method,
                           m_spinorbital_kind == SpinorbitalKind::Unrestricted);
    for (const auto &func : m_funcs) {
        occ::log::debug(
            "Functional: {} {} {}, exact exchange = {}, polarized = {}",
            func.name(), func.kind_string(), func.family_string(),
            func.exact_exchange_factor(), func.polarized());
    }
    double hfx = exact_exchange_factor();
    if (hfx > 0.0)
        fmt::print("    {} x HF exchange\n", hfx);
}

std::vector<DensityFunctional> parse_method(const std::string &method_string,
                                            bool polarized) {
    std::vector<DensityFunctional> funcs;
    std::string method = occ::util::trim_copy(method_string);
    occ::util::to_lower(method);

    auto tokens = occ::util::tokenize(method_string, " ");
    fmt::print("Functionals:\n");
    for (const auto &token : tokens) {
        std::string m = token;
        occ::log::debug("Token: {}", m);
        if (m[0] == 'u')
            m = m.substr(1);
        if (builtin_functionals.contains(m)) {
            auto combo = builtin_functionals.at(m);
            occ::log::debug("Found builtin functional combination for {}", m);
            for (const auto &func : combo) {
                occ::log::debug("id: {}", static_cast<int>(func.id));
                auto f = DensityFunctional(func.id, polarized);
                occ::log::debug("scale factor: {}", func.factor);
                f.set_scale_factor(func.factor);
                fmt::print("    ");
                if (func.factor != 1.0)
                    fmt::print("{} x ", func.factor);
                fmt::print("{}\n", f.name());
                if (func.hfx > 0.0)
                    f.set_exchange_factor(func.hfx);
                funcs.push_back(f);
            }
        } else {
            fmt::print("    {}\n", token);
            funcs.push_back(DensityFunctional(token, polarized));
        }
    }
    return funcs;
}

} // namespace occ::dft
