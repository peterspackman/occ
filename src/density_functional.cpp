#include "density_functional.h"
#include "logger.h"

namespace tonto::dft {

DensityFunctional::DensityFunctional(const std::string& name)
{
    int err = xc_func_init(&m_func, XC_LDA_X, XC_UNPOLARIZED);
    tonto::log::debug("xc_func_init returned {}", err);
}

tonto::Vec DensityFunctional::energy(const tonto::Vec& rho) const
{
    tonto::Vec e(rho.rows(), rho.cols());
    tonto::log::debug("rho.shape n_pts = {} {}", rho.rows(), rho.cols());
    int n_pts = rho.rows();
    switch(family()) {
    case Family::LDA: {
        xc_lda_exc(&m_func, n_pts, rho.data(), e.data());
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
    tonto::log::debug("rho.shape n_pts = {} {}", rho.rows(), rho.cols());

    switch(family()) {
    case Family::LDA: {
        xc_lda_vxc(&m_func, n_pts, rho.data(), v.data());
        break;
    }
    default: throw std::runtime_error("Unhandled functional family");
    }
    return v;
}

DensityFunctional::~DensityFunctional()
{
    xc_func_end(&m_func);
}

}
