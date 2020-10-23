#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "logger.h"
#include "density.h"
#include "timings.h"
#include "gto.h"
#include "util.h"

namespace tonto::dft {
using tonto::qm::SpinorbitalKind;

int DFT::density_derivative() const {
    int deriv = 0;
    for(const auto& func: m_funcs) {
        deriv = std::max(deriv, func.derivative_order());
    }
    return deriv;
}

DFT::DFT(const std::string& method, const libint2::BasisSet& basis, const std::vector<libint2::Atom>& atoms, SpinorbitalKind kind) :
   m_spinorbital_kind(kind), m_hf(atoms, basis), m_grid(basis, atoms)
{
    tonto::log::debug("start calculating atom grids... ");
    m_grid.set_max_angular_points(530);
    m_grid.set_min_angular_points(80);
    m_grid.set_radial_precision(1e-12);
    for(size_t i = 0; i < atoms.size(); i++) {
        m_atom_grids.push_back(m_grid.grid_points(i));
    }
    size_t num_grid_points = std::accumulate(m_atom_grids.begin(), m_atom_grids.end(), 0.0, [&](double tot, const auto& grid) { return tot + grid.first.cols(); });
    tonto::log::debug("finished calculating atom grids ({} points)", num_grid_points);
    tonto::log::debug("Grid initialization took {} seconds", tonto::timing::total(tonto::timing::grid_init));
    tonto::log::debug("Grid point creation took {} seconds", tonto::timing::total(tonto::timing::grid_points));
    m_funcs = parse_method(method, m_spinorbital_kind == SpinorbitalKind::Unrestricted);
    for(const auto& func: m_funcs) {
        tonto::log::debug("Functional: {} {} {}, exact exchange = {}, polarized = {}",
                          func.name(), func.kind_string(), func.family_string(), func.exact_exchange_factor(), func.polarized());
    }
    tonto::log::debug("Total exchange factor: {}", exact_exchange_factor());
}

std::vector<DensityFunctional> parse_method(const std::string& method_string, bool polarized)
{
    std::vector<DensityFunctional> funcs;
    std::string method = tonto::util::trim_copy(method_string);
    tonto::util::to_lower(method);
    auto tokens = tonto::util::tokenize(method_string, " ");
    for(const auto& token: tokens) {
        if(token == "pbe") {
            funcs.push_back(DensityFunctional("xc_gga_x_pbe", polarized));
            funcs.push_back(DensityFunctional("xc_gga_c_pbe", polarized));
        }
        else if(token == "blyp") {
            funcs.push_back(DensityFunctional("xc_gga_x_b88", polarized));
            funcs.push_back(DensityFunctional("xc_gga_c_lyp", polarized));
        }
        else funcs.push_back(DensityFunctional(token, polarized));
    }
    return funcs;
}

}
