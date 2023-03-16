#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/3rdparty/parallel_hashmap/phmap.h>
#include <occ/core/atom.h>
#include <occ/core/log.h>
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

const phmap::flat_hash_map<std::string, std::vector<FuncComponent>>
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
        {"wb97x", {{dfid::hyb_gga_xc_wb97x}}},
        {"wb97m", {{dfid::hyb_mgga_xc_wb97m_v}}},
        {"wb97m-v", {{dfid::hyb_mgga_xc_wb97m_v}}},
    });

int DFT::density_derivative() const {
    int deriv = 0;
    for (const auto &func : m_funcs) {
        deriv = std::max(deriv, func.derivative_order());
    }
    return deriv;
}

DFT::DFT(const std::string &method, const AOBasis &basis,
         const AtomGridSettings &grid_settings)
    : m_hf(basis), m_grid(basis, grid_settings) {

    if (method[0] == 'u') {
        set_method(method.substr(1), true);
        fmt::print("Unrestricted\n");
    } else {
        set_method(method, false);
    }
    set_integration_grid(grid_settings);
}

void DFT::set_unrestricted(bool unrestricted) {
    set_method(m_method_string, unrestricted);
}

void DFT::set_method(const std::string &method_string, bool unrestricted) {
    m_method_string = method_string;
    m_funcs = parse_method(method_string, unrestricted);
    for (const auto &func : m_funcs) {
        occ::log::debug(
            "Functional: {} {} {}, exact exchange = {}, polarized = {}",
            func.name(), func.kind_string(), func.family_string(),
            func.exact_exchange_factor(), func.polarized());
    }

    m_rs_params = {};
    for (const auto &func : m_funcs) {
        auto rs = func.range_separated_parameters();
        if (rs.omega != 0.0) {
            m_rs_params = rs;
        }
    }
    if (m_rs_params.omega != 0.0) {
        occ::log::info("    RS omega = {}", m_rs_params.omega);
        occ::log::info("    RS alpha = {}", m_rs_params.alpha);
        occ::log::info("    RS beta  = {}", m_rs_params.beta);
    }

    double hfx = exact_exchange_factor();
    if (hfx > 0.0)
        occ::log::info("    {} x HF exchange", hfx);
}

void DFT::set_integration_grid(const AtomGridSettings &settings) {
    const auto &atoms = m_hf.aobasis().atoms();
    if (settings != m_grid.settings()) {
        m_grid = MolecularGrid(m_hf.aobasis(), settings);
    }
    occ::log::debug("start calculating atom grids... ");
    m_atom_grids.clear();
    for (size_t i = 0; i < atoms.size(); i++) {
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

    for (const auto &func : m_funcs) {
        if (func.needs_nlc_correction()) {
            m_nlc.set_integration_grid(m_hf.aobasis());
            break;
        }
    }
}

std::vector<DensityFunctional> parse_method(const std::string &method_string,
                                            bool polarized) {
    std::vector<DensityFunctional> funcs;
    std::string method = occ::util::trim_copy(method_string);
    occ::util::to_lower(method);

    auto tokens = occ::util::tokenize(method_string, " ");
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
                auto f = DensityFunctional(func.id, polarized);
                occ::log::debug("scale factor: {}", func.factor);
                f.set_scale_factor(func.factor);
                if (func.factor != 1.0) {
                    occ::log::info("    {} x {}", func.factor, f.name());
                } else {
                    occ::log::info("    {}", f.name());
                }
                if (func.hfx > 0.0)
                    f.set_exchange_factor(func.hfx);
                funcs.push_back(f);
            }
        } else {
            occ::log::info("    {}", token);
            funcs.push_back(DensityFunctional(token, polarized));
        }
    }
    return funcs;
}

RangeSeparatedParameters DFT::range_separated_parameters() const {
    return m_rs_params;
}

} // namespace occ::dft
