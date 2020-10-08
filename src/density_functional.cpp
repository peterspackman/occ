#include "density_functional.h"
#include "logger.h"

namespace tonto::dft {

DensityFunctional::DensityFunctional(const std::string& name)
{
    m_func = std::unique_ptr<xc_func_type>(new xc_func_type);
    int err = xc_func_init(m_func.get(), XC_LDA_X, XC_UNPOLARIZED);
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

}
