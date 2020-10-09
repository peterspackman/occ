#include "density_functional.h"
#include "logger.h"

namespace tonto::dft {

DensityFunctional::DensityFunctional(const std::string& name, bool polarized) :
    m_func_name(name), m_polarized(polarized)
{
    m_func = std::unique_ptr<xc_func_type>(new xc_func_type);
    int func_id = functional_id(name);
    m_func_id = static_cast<Identifier>(func_id);
    int err = xc_func_init(m_func.get(), func_id, m_polarized ? XC_POLARIZED : XC_UNPOLARIZED);
}

void DensityFunctional::add_energy(const tonto::Vec& rho, tonto::Vec& e) const
{
    int n_pts = rho.rows();
    switch(family()) {
    case Family::LDA: {
        xc_lda_exc(m_func.get(), n_pts, rho.data(), e.data());
        break;
    }
    default: throw std::runtime_error("Unhandled functional family");
    }
}

tonto::Vec DensityFunctional::energy(const tonto::Vec& rho) const {
    tonto::Vec e = tonto::Vec::Zero(rho.rows(), rho.cols());
    add_energy(rho, e);
    return e;
}

void DensityFunctional::add_potential(const tonto::Vec& rho, tonto::Vec& v) const
{
    int n_pts = rho.rows();
    switch(family()) {
    case Family::LDA: {
        xc_lda_vxc(m_func.get(), n_pts, rho.data(), v.data());
        break;
    }
    default: throw std::runtime_error("Unhandled functional family");
    }
}

tonto::Vec DensityFunctional::potential(const tonto::Vec& rho) const {
    tonto::Vec v = tonto::Vec::Zero(rho.rows(), rho.cols());
    add_potential(rho, v);
    return v;
}

void DensityFunctional::add_energy_potential(const tonto::Vec& rho, tonto::Vec& e, tonto::Vec& v) const
{
    int n_pts = rho.rows();
    switch(family()) {
    case Family::LDA: {
        xc_lda_exc_vxc(m_func.get(), n_pts, rho.data(), e.data(), v.data());
        break;
    }
    default: throw std::runtime_error("Unhandled functional family");
    }
}

std::pair<tonto::Vec, tonto::Vec> DensityFunctional::energy_potential(const tonto::Vec& rho) const {
    tonto::Vec e = tonto::Vec::Zero(rho.rows(), rho.cols());
    tonto::Vec v = tonto::Vec::Zero(rho.rows(), rho.cols());
    add_energy_potential(rho, e, v);
    return std::make_pair(e, v);
}

int DensityFunctional::functional_id(const std::string& name) {
    if (name == "lda" || name == "slater" || name == "S") {
        return XC_LDA_X;
    }
    if (name == "VWN5" || name == "vwn5" || name == "vwn") {
        return XC_LDA_C_VWN;
    }
    int func = xc_functional_get_number(name.c_str());
    if(func == 0) throw std::runtime_error(fmt::format("Unknown functional name {}", name));
    return func;
}

}
