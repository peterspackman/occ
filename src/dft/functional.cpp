#include <cstdlib>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/dft/functional.h>

namespace occ::dft {

DensityFunctional::DensityFunctional(DensityFunctional::Identifier id,
                                     bool polarized)
    : m_func_id(id), m_polarized(polarized) {
    char *cstr = xc_functional_get_name(id);
    m_func_name = std::string(cstr);
    free(cstr);
}

DensityFunctional::DensityFunctional(const std::string &name, bool polarized)
    : m_polarized(polarized), m_func_name(name) {
    int func_id = functional_id(name);
    m_func_id = static_cast<Identifier>(func_id);
}

DensityFunctional::Result
DensityFunctional::evaluate(const Params &params) const {
    int n_pts = params.npts;
    Family fam = family();
    Result result(params.npts, family(),
                  m_polarized ? SpinorbitalKind::Unrestricted
                              : SpinorbitalKind::Restricted);
    xc_func_type func;

    // this takes quite a while (can be 1-5% of total DFT XC evaluation time),
    // should probably cache this
    int err = xc_func_init(&func, static_cast<int>(m_func_id),
                           m_polarized ? XC_POLARIZED : XC_UNPOLARIZED);
    switch (fam) {
    case LDA: {
        xc_lda_exc_vxc(&func, n_pts, params.rho.data(), result.exc.data(),
                       result.vrho.data());
        break;
    }
    case HGGA:
    case GGA: {
        assert(params.sigma.cols() > 0);
        xc_gga_exc_vxc(&func, n_pts, params.rho.data(), params.sigma.data(),
                       result.exc.data(), result.vrho.data(),
                       result.vsigma.data());
        break;
    }
    case HMGGA:
    case MGGA: {
        assert(params.sigma.cols() > 0);
        assert(params.laplacian.cols() > 0);
        assert(params.tau.cols() > 0);
        xc_mgga_exc_vxc(&func, n_pts, params.rho.data(), params.sigma.data(),
                        params.laplacian.data(), params.tau.data(),
                        result.exc.data(), result.vrho.data(),
                        result.vsigma.data(), result.vlaplacian.data(),
                        result.vtau.data());
        break;
    }
    default:
        throw std::runtime_error("Unhandled functional family");
    }
    xc_func_end(&func);
    return result;
}

int DensityFunctional::functional_id(const std::string &name) {
    int func = xc_functional_get_number(name.c_str());
    if (func <= 0)
        throw std::runtime_error(
            fmt::format("Unknown functional name {}", name));
    return func;
}

bool DensityFunctional::needs_nlc_correction() const {
    switch (m_func_id) {
    default:
        return false;
    case hyb_mgga_xc_wb97m_v:
        return true;
    }
}

} // namespace occ::dft
