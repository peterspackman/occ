#include "density_functional.h"
#include "logger.h"

namespace tonto::dft {

DensityFunctional::DensityFunctional(const std::string& name)
{
    m_func = std::unique_ptr<xc_func_type>(new xc_func_type);
    int err = xc_func_init(m_func.get(), XC_LDA_X, XC_UNPOLARIZED);
}

tonto::Vec DensityFunctional::energy(const tonto::Vec& rho) const
{
    tonto::Vec e(rho.rows(), rho.cols());
    int n_pts = rho.rows();
    switch(family()) {
    case Family::LDA: {
        tonto::log::debug("{} npts", n_pts);
        xc_lda_exc(m_func.get(), n_pts, rho.data(), e.data());
        break;
    }
    default: throw std::runtime_error("Unhandled functional family");
    }
    return e;
}

tonto::Vec DensityFunctional::potential(const tonto::Vec& rho) const
{
    tonto::Vec v(rho.rows(), rho.cols());
    int n_pts = rho.rows();
    switch(family()) {
    case Family::LDA: {
        xc_lda_vxc(m_func.get(), n_pts, rho.data(), v.data());
        break;
    }
    default: throw std::runtime_error("Unhandled functional family");
    }
    return v;
}

}
