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

DensityFunctional::Result DensityFunctional::evaluate(const Params& params) const
{
    int n_pts = params.num_points();
    Family fam = family();
    Result result;
    int nvrho = m_func.get()->dim.vrho;
    int nexc = m_func.get()->dim.zk;
    result.vrho.resize(n_pts * nvrho);
    result.exc.resize(n_pts * nexc);
    switch(fam) {
    case LDA: {
        xc_lda_exc_vxc(m_func.get(), n_pts, params.rho.data(), result.exc.data(), result.vrho.data());
        break;
    }
    case GGA: {
        assert(("Sigma array must be provided for GGA functionals", params.sigma.cols() > 0));
        int nvsigma = m_func.get()->dim.vsigma;
        result.vsigma.resize(n_pts * nvsigma);
        xc_gga_exc_vxc(m_func.get(), n_pts, params.rho.data(), params.sigma.data(), result.exc.data(), result.vrho.data(), result.vsigma.data());
        break;
    }
    default: throw std::runtime_error("Unhandled functional family");
    }
    return result;
}

int DensityFunctional::functional_id(const std::string& name) {
    int func = xc_functional_get_number(name.c_str());
    if(func == 0) throw std::runtime_error(fmt::format("Unknown functional name {}", name));
    return func;
}

}
