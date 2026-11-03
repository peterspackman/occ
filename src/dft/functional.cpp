#include <fmt/core.h>
#include <cstdlib>
#include <memory>
#include <vector>
#include <occ/dft/functional.h>

namespace occ::dft {

namespace {

// Initialising a libxc functional costs about as much as evaluating it on a
// small grid batch, and the XC build evaluates every batch separately. Keep
// one initialised functional per (id, spin) per thread: an xc_func_type is
// read-only during evaluation, and per-thread copies need no locking.
class FunctionalCache {
public:
  ~FunctionalCache() {
    for (auto &e : m_entries)
      xc_func_end(&e->func);
  }
  const xc_func_type *get(int id, bool polarized) {
    for (const auto &e : m_entries)
      if (e->id == id && e->polarized == polarized)
        return &e->func;
    auto e = std::make_unique<Entry>();
    e->id = id;
    e->polarized = polarized;
    if (xc_func_init(&e->func, id, polarized ? XC_POLARIZED : XC_UNPOLARIZED))
      throw std::runtime_error(
          fmt::format("functional with id = {}, polarized = {} not found", id,
                      polarized));
    m_entries.push_back(std::move(e));
    return &m_entries.back()->func;
  }

private:
  struct Entry {
    int id{0};
    bool polarized{false};
    xc_func_type func;
  };
  std::vector<std::unique_ptr<Entry>> m_entries;
};

const xc_func_type *cached_functional(int id, bool polarized) {
  thread_local FunctionalCache cache;
  return cache.get(id, polarized);
}

} // namespace

DensityFunctional::DensityFunctional(DensityFunctional::Identifier id,
                                     bool polarized)
    : m_func_id(id), m_polarized(polarized) {
  m_func_name = dfid_to_string(id);
}

DensityFunctional::DensityFunctional(const std::string &name, bool polarized)
    : m_polarized(polarized), m_func_name(name) {
  m_func_id = string_to_dfid(name);
}

DensityFunctional::Result
DensityFunctional::evaluate(const Params &params) const {
  int n_pts = params.npts;
  Family fam = family();
  Result result(params.npts, family(),
                m_polarized ? SpinorbitalKind::Unrestricted
                            : SpinorbitalKind::Restricted);
  const xc_func_type *func =
      cached_functional(static_cast<int>(m_func_id), m_polarized);
  switch (fam) {
  case LDA: {
    xc_lda_exc_vxc(func, n_pts, params.rho.data(), result.exc.data(),
                   result.vrho.data());
    break;
  }
  case HGGA:
  case GGA: {
    assert(params.sigma.cols() > 0);
    xc_gga_exc_vxc(func, n_pts, params.rho.data(), params.sigma.data(),
                   result.exc.data(), result.vrho.data(), result.vsigma.data());
    break;
  }
  case HMGGA:
  case MGGA: {
    assert(params.sigma.cols() > 0);
    assert(params.laplacian.cols() > 0);
    assert(params.tau.cols() > 0);
    xc_mgga_exc_vxc(func, n_pts, params.rho.data(), params.sigma.data(),
                    params.laplacian.data(), params.tau.data(),
                    result.exc.data(), result.vrho.data(), result.vsigma.data(),
                    result.vlaplacian.data(), result.vtau.data());
    break;
  }
  default:
    throw std::runtime_error("Unhandled functional family");
  }
  // composite methods (e.g. pbe0 = 0.75 x gga_x_pbe + ...): E and every
  // potential term are linear in the component weight
  if (m_factor != 1.0) {
    result.exc *= m_factor;
    result.vrho *= m_factor;
    result.vsigma *= m_factor;
    result.vlaplacian *= m_factor;
    result.vtau *= m_factor;
  }
  return result;
}

DensityFunctional::SecondDerivatives
DensityFunctional::evaluate_second_derivatives(const Params &params) const {
  const size_t n = params.npts;
  const Family fam = family();
  const int ns = m_polarized ? 2 : 1;
  SecondDerivatives out;
  out.vrho = MatRM::Zero(n, ns);
  out.v2rho2 = MatRM::Zero(n, m_polarized ? 3 : 1);
  const xc_func_type *func =
      cached_functional(static_cast<int>(m_func_id), m_polarized);
  switch (fam) {
  case LDA:
    xc_lda_vxc_fxc(func, n, params.rho.data(), out.vrho.data(),
                   out.v2rho2.data());
    break;
  case GGA:
  case HGGA:
    out.gga = true;
    out.vsigma = MatRM::Zero(n, m_polarized ? 3 : 1);
    out.v2rhosigma = MatRM::Zero(n, m_polarized ? 6 : 1);
    out.v2sigma2 = MatRM::Zero(n, m_polarized ? 6 : 1);
    xc_gga_vxc_fxc(func, n, params.rho.data(), params.sigma.data(),
                   out.vrho.data(), out.vsigma.data(), out.v2rho2.data(),
                   out.v2rhosigma.data(), out.v2sigma2.data());
    break;
  case MGGA:
  case HMGGA:
    if (func->info->flags & XC_FLAGS_NEEDS_LAPLACIAN)
      throw std::runtime_error(fmt::format(
          "Second functional derivatives: {} depends on the density "
          "Laplacian, which is not supported",
          m_func_name));
    out.gga = true;
    out.mgga = true;
    out.vsigma = MatRM::Zero(n, m_polarized ? 3 : 1);
    out.v2rhosigma = MatRM::Zero(n, m_polarized ? 6 : 1);
    out.v2sigma2 = MatRM::Zero(n, m_polarized ? 6 : 1);
    out.vtau = MatRM::Zero(n, ns);
    out.v2rhotau = MatRM::Zero(n, m_polarized ? 4 : 1);
    out.v2sigmatau = MatRM::Zero(n, m_polarized ? 6 : 1);
    out.v2tau2 = MatRM::Zero(n, m_polarized ? 3 : 1);
    xc_mgga_vxc_fxc(func, n, params.rho.data(), params.sigma.data(),
                    params.laplacian.data(), params.tau.data(), out.vrho.data(),
                    out.vsigma.data(), nullptr, out.vtau.data(),
                    out.v2rho2.data(), out.v2rhosigma.data(), nullptr,
                    out.v2rhotau.data(), out.v2sigma2.data(), nullptr,
                    out.v2sigmatau.data(), nullptr, nullptr,
                    out.v2tau2.data());
    break;
  default:
    throw std::runtime_error("Second functional derivatives: unhandled "
                             "functional family");
  }
  if (m_factor != 1.0) {
    out.vrho *= m_factor;
    out.v2rho2 *= m_factor;
    if (out.gga) {
      out.vsigma *= m_factor;
      out.v2rhosigma *= m_factor;
      out.v2sigma2 *= m_factor;
    }
    if (out.mgga) {
      out.vtau *= m_factor;
      out.v2rhotau *= m_factor;
      out.v2sigmatau *= m_factor;
      out.v2tau2 *= m_factor;
    }
  }
  return out;
}

bool DensityFunctional::needs_laplacian() const {
  const xc_func_type *func =
      cached_functional(static_cast<int>(m_func_id), m_polarized);
  return func->info->flags & XC_FLAGS_NEEDS_LAPLACIAN;
}

bool DensityFunctional::needs_nlc_correction() const {
  switch (m_func_id) {
  default:
    return false;
  case hyb_mgga_xc_wb97m_v:
  case hyb_gga_xc_wb97x_v:
    return true;
  }
}

std::string dfid_to_string(DensityFunctional::Identifier id) {
  char *cstr = xc_functional_get_name(id);
  if (!cstr) {
    return "unknown";
  }
  std::string result(cstr);
  free(cstr);
  return result;
}

DensityFunctional::Identifier string_to_dfid(const std::string &name) {
  int func_id = xc_functional_get_number(name.c_str());
  if (func_id <= 0)
    throw std::runtime_error(fmt::format("Unknown functional name {}", name));
  return static_cast<DensityFunctional::Identifier>(func_id);
}

} // namespace occ::dft
