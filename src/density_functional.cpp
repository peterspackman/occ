#include "density_functional.h"

namespace tonto::dft {

DensityFunctional::DensityFunctional(const std::string& name)
{
    xc_func_init(&m_func, XC_LDA_X, XC_UNPOLARIZED);
}

tonto::Vec DensityFunctional::energy(const tonto::Vec& rho) const
{
    tonto::Vec e(rho.rows(), rho.cols());
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
    throw std::runtime_error("Unhandled functional family");
}

DensityFunctional::~DensityFunctional()
{
    xc_func_end(&m_func);
}

}
