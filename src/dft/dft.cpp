#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/gto/gto.h>
#include <occ/gto/density.h>
#include <occ/core/atom.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/3rdparty/robin_hood.h>
namespace occ::dft {

using occ::qm::SpinorbitalKind;
using occ::qm::BasisSet;

using dfid = DensityFunctional::Identifier;

struct FuncComponent {
    dfid id;
    double factor{1.0};
    double hfx{0.0};
};

const robin_hood::unordered_map<std::string, std::vector<FuncComponent>> builtin_functionals({
    {"b3lyp", {{dfid::hyb_gga_xc_b3lyp, 1.0}}},
    {"pbe1pbe", {{dfid::gga_x_pbe, 0.75, 0.25}, {dfid::gga_c_pbe}}},
    {"pbe", {{dfid::gga_x_pbe, 1.0}, {dfid::gga_c_pbe, 1.0}}},
    {"pbepbe", {{dfid::gga_x_pbe, 1.0}, {dfid::gga_c_pbe, 1.0}}},
    {"pbe0", {{dfid::gga_x_pbe, 0.75, 0.25}, {dfid::gga_c_pbe}}},
    {"svwn", {{dfid::lda_x}, {dfid::lda_c_vwn_3}}},
    {"svwn5", {{dfid::lda_x}, {dfid::lda_c_vwn}}},
    {"blyp", {{dfid::gga_x_b88}, {dfid::gga_c_lyp}}},
    {"bpbe", {{dfid::gga_x_b88}, {dfid::gga_c_pbe}}},
    {"m062x", {{dfid::hyb_mgga_x_m06_2x}, {dfid::mgga_c_m06_2x}}},
});


int DFT::density_derivative() const {
    int deriv = 0;
    for(const auto& func: m_funcs) {
        deriv = std::max(deriv, func.derivative_order());
    }
    return deriv;
}

DFT::DFT(const std::string& method, const BasisSet& basis, const std::vector<occ::core::Atom>& atoms, SpinorbitalKind kind) :
   m_spinorbital_kind(kind), m_hf(atoms, basis), m_grid(basis, atoms)
{
    occ::log::debug("start calculating atom grids... ");
    m_grid.set_max_angular_points(590);
    m_grid.set_min_angular_points(86);
    m_grid.set_radial_precision(1e-12);
    for(size_t i = 0; i < atoms.size(); i++) {
        m_atom_grids.push_back(m_grid.generate_partitioned_atom_grid(i));
    }
    size_t num_grid_points = std::accumulate(m_atom_grids.begin(), m_atom_grids.end(), 0.0, [&](double tot, const auto& grid) { return tot + grid.points.cols(); });
    occ::log::debug("finished calculating atom grids ({} points)", num_grid_points);
    occ::log::debug("Grid initialization took {} seconds", occ::timing::total(occ::timing::grid_init));
    occ::log::debug("Grid point creation took {} seconds", occ::timing::total(occ::timing::grid_points));
    m_funcs = parse_method(method, m_spinorbital_kind == SpinorbitalKind::Unrestricted);
    for(const auto& func: m_funcs) {
        occ::log::debug("Functional: {} {} {}, exact exchange = {}, polarized = {}",
                          func.name(), func.kind_string(), func.family_string(), func.exact_exchange_factor(), func.polarized());
    }
    double hfx = exact_exchange_factor();
    if(hfx > 0.0) fmt::print("    {} x HF exchange\n", hfx);
}

std::vector<DensityFunctional> parse_method(const std::string& method_string, bool polarized)
{
    std::vector<DensityFunctional> funcs;
    std::string method = occ::util::trim_copy(method_string);
    occ::util::to_lower(method);

    auto tokens = occ::util::tokenize(method_string, " ");
    fmt::print("Functionals:\n");
    for(const auto& token: tokens) {
        std::string m = token;
        occ::log::debug("Token: {}", m);
        if(m[0] == 'u') m = m.substr(1);
        if(builtin_functionals.contains(m)) {
            auto combo = builtin_functionals.at(m);
            occ::log::debug("Found builtin functional combination for {}", m);
            for(const auto& func: combo) {
                occ::log::debug("id: {}", func.id);
                auto f = DensityFunctional(func.id, polarized);
                occ::log::debug("scale factor: {}", func.factor);
                f.set_scale_factor(func.factor);
                fmt::print("    ");
                if(func.factor != 1.0) fmt::print("{} x ", func.factor);
                fmt::print("{}\n", f.name());
                if(func.hfx > 0.0) f.set_exchange_factor(func.hfx);
                funcs.push_back(f);
            }
        }
        else 
        {
            fmt::print("    {}\n", token);
            funcs.push_back(DensityFunctional(token, polarized));
        }
    }
    return funcs;
}

}
