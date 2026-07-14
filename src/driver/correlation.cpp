#include <occ/core/log.h>
#include <occ/core/util.h>
#include <occ/driver/correlation.h>
#include <occ/driver/method_parser.h>
#include <occ/qm/cc/ccsd.h>
#include <occ/qm/cc/integrals.h>
#include <occ/qm/cc/thc_mp2.h>
#include <occ/qm/cc/triples.h>
#include <occ/qm/cc/uccsd.h>
#include <occ/qm/correlation/mp2.h>
#include <occ/qm/fitting_basis.h>

namespace occ::driver {

namespace {

using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
namespace cc = occ::qm::cc;

size_t memory_budget_bytes(double gb) {
  return static_cast<size_t>(gb * 1024.0 * 1024.0 * 1024.0);
}

// Effective backend: an explicit option wins, then a method-name prefix
// ("ri-mp2", "thc-ccsd(t)"), then the method family default.
std::string resolve_backend(const CorrelationOptions &opts,
                            const MethodSpec &mspec, bool is_mp2) {
  std::string backend = occ::util::to_lower_copy(opts.backend);
  const std::string family_default = is_mp2 ? "conventional" : "exact";
  if (backend.empty() || backend == "auto") {
    backend = mspec.backend.empty() ? family_default : mspec.backend;
    // For MP2 an explicit auxiliary basis implies RI (matching the CLI, where
    // --ri-basis selects RI-MP2 without needing --mp2-backend).
    if (is_mp2 && backend == "conventional" && !opts.aux_basis.empty())
      backend = "df";
  }
  if (backend == "ri")
    backend = "df";
  if (backend == "conventional" && !is_mp2)
    backend = "exact";
  if (backend == "exact" && is_mp2)
    backend = "conventional";
  return backend;
}

std::string resolve_aux_name(const CorrelationOptions &opts,
                             const Wavefunction &wfn) {
  if (!opts.aux_basis.empty())
    return opts.aux_basis;
  return occ::qm::resolve_fitting_basis(wfn.basis.name(),
                                        occ::qm::FittingKind::Correlation);
}

occ::gto::AOBasis load_aux_basis(const Wavefunction &wfn,
                                 const std::string &name) {
  auto aux = occ::gto::AOBasis::load(wfn.basis.atoms(), name);
  aux.set_pure(wfn.basis.is_pure());
  return aux;
}

int clamp_frozen(int n_frozen, size_t nocc_min) {
  return std::max(0, std::min(n_frozen, static_cast<int>(nocc_min) - 1));
}

// -1 = auto (chemical core), otherwise the requested count, clamped so at
// least one occupied orbital stays active.
int resolve_frozen(const CorrelationOptions &opts, const Wavefunction &wfn) {
  const bool auto_fc = opts.n_frozen < 0;
  int n = auto_fc ? cc::num_frozen_core(wfn.basis) : opts.n_frozen;
  const size_t nocc_min = std::min(wfn.mo.n_alpha, wfn.mo.n_beta);
  n = clamp_frozen(n, nocc_min);
  occ::log::info("Frozen core: {} orbitals ({} electrons){}", n, 2 * n,
                 auto_fc ? " [auto]" : "");
  return n;
}

cc::ThcOptions thc_options(const CorrelationOptions &opts, size_t budget) {
  cc::ThcOptions thc;
  thc.c_isdf = opts.thc_c_isdf;
  thc.memory_budget = budget;
  thc.grid_max_angular = opts.thc_grid_angular;
  thc.grid_radial_precision = opts.thc_grid_radial_precision;
  const std::string sel = occ::util::to_lower_copy(opts.thc_isdf_method);
  thc.method = (sel == "qr") ? cc::IsdfMethod::QR : cc::IsdfMethod::Cholesky;
  return thc;
}

// SCS/SOS coefficients; returns whether scaling is active.
bool spin_scaling_coeffs(const std::string &scaling, double &c_ss,
                         double &c_os) {
  c_ss = 1.0;
  c_os = 1.0;
  if (scaling == "scs") {
    c_ss = 1.0 / 3.0;
    c_os = 6.0 / 5.0;
    return true;
  }
  if (scaling == "sos") {
    c_ss = 0.0;
    c_os = 1.3;
    return true;
  }
  if (scaling != "none" && !scaling.empty())
    occ::log::warn("Unknown MP2 spin scaling '{}', using unscaled MP2",
                   scaling);
  return false;
}

std::string scaled_label(const std::string &scaling, const std::string &base) {
  if (scaling == "scs")
    return "SCS-" + base;
  if (scaling == "sos")
    return "SOS-" + base;
  return base;
}

CorrelationResult run_mp2(const Wavefunction &wfn,
                          const CorrelationOptions &opts,
                          const MethodSpec &mspec) {
  occ::log::info("{:=^72s}", "  MP2 Calculation  ");
  const std::string backend = resolve_backend(opts, mspec, true);
  const std::string scaling = occ::util::to_lower_copy(opts.spin_scaling);
  const size_t budget = memory_budget_bytes(opts.max_memory_gb);

  double c_ss, c_os;
  const bool scaled = spin_scaling_coeffs(scaling, c_ss, c_os);

  CorrelationResult result;
  result.scf_energy = wfn.energy.total;

  if (backend == "thc") {
    const std::string auxname = resolve_aux_name(opts, wfn);
    occ::log::info("Method: THC-MP2 (auxiliary basis: {})", auxname);
    auto aux = load_aux_basis(wfn, auxname);

    cc::ThcMP2Options thc_opts;
    thc_opts.thc = thc_options(opts, budget);
    thc_opts.n_laplace = opts.laplace_points;
    thc_opts.memory_budget = budget;
    thc_opts.n_frozen = resolve_frozen(opts, wfn);
    // SOS-MP2 needs only the opposite-spin energy: skip the same-spin
    // exchange so the whole calculation is the cubic Coulomb path.
    thc_opts.opposite_spin_only = (scaling == "sos");

    occ::log::info("THC rank c = {}, ISDF selector = {}, Laplace points = {}",
                   thc_opts.thc.c_isdf,
                   thc_opts.thc.method == cc::IsdfMethod::QR ? "qr"
                                                             : "cholesky",
                   thc_opts.n_laplace);

    const auto r = cc::thc_mp2(wfn.basis, aux, wfn.mo, thc_opts);

    result.method = scaled_label(scaling, "THC-MP2");
    result.same_spin = r.same_spin;
    result.opposite_spin = r.opposite_spin;
    result.scaled_correlation = c_ss * r.same_spin + c_os * r.opposite_spin;
    result.correlation_energy = scaled ? result.scaled_correlation : r.total;
    result.n_frozen = thc_opts.n_frozen;

    occ::log::info(
        "THC-MP2: {} interpolation points, {} Laplace points (max rel err "
        "{:.2e})",
        r.n_isdf, r.n_laplace, r.laplace_max_rel_error);
    occ::log::info("SCF energy:                       {: 20.12f}",
                   result.scf_energy);
    if (!thc_opts.opposite_spin_only) {
      occ::log::info("MP2 correlation energy:           {: 20.12f}", r.total);
      occ::log::info("  same-spin:                      {: 20.12f}",
                     r.same_spin);
    }
    occ::log::info("  opposite-spin:                  {: 20.12f}",
                   r.opposite_spin);
  } else {
    occ::qm::MP2 mp2 = [&]() {
      if (backend == "df") {
        const std::string auxname = resolve_aux_name(opts, wfn);
        occ::log::info("Method: RI-MP2 (auxiliary basis: {})", auxname);
        auto aux = load_aux_basis(wfn, auxname);
        return occ::qm::MP2(wfn.basis, aux, wfn.mo, wfn.energy.total);
      }
      if (backend != "conventional")
        throw std::runtime_error("Unknown MP2 backend '" + backend +
                                 "' (expected conventional | ri/df | thc)");
      occ::log::info("Method: Conventional MP2");
      return occ::qm::MP2(wfn.basis, wfn.mo, wfn.energy.total);
    }();

    if (opts.n_frozen < 0)
      mp2.set_frozen_core_auto();
    else
      mp2.set_frozen_core(
          clamp_frozen(opts.n_frozen,
                       std::min(wfn.mo.n_alpha, wfn.mo.n_beta)));
    mp2.set_memory_budget(budget);
    if (scaled)
      mp2.set_scs_parameters(c_ss, c_os);

    const double corr = mp2.compute_correlation_energy();
    const auto &r = mp2.results();

    result.method =
        scaled_label(scaling, backend == "df" ? "RI-MP2" : "MP2");
    result.same_spin = r.same_spin_correlation;
    result.opposite_spin = r.opposite_spin_correlation;
    result.scaled_correlation = r.scs_mp2_correlation;
    result.correlation_energy = scaled ? r.scs_mp2_correlation : corr;
    result.n_frozen = static_cast<int>(r.n_frozen_core);

    occ::log::info("SCF energy:                       {: 20.12f}",
                   result.scf_energy);
    occ::log::info("MP2 correlation energy:           {: 20.12f}", corr);
    occ::log::info("  same-spin:                      {: 20.12f}",
                   r.same_spin_correlation);
    occ::log::info("  opposite-spin:                  {: 20.12f}",
                   r.opposite_spin_correlation);
  }

  if (scaled)
    occ::log::info("{}-MP2 correlation energy:        {: 20.12f}",
                   scaling == "scs" ? "SCS" : "SOS", result.scaled_correlation);
  result.total_energy = result.scf_energy + result.correlation_energy;
  occ::log::info("MP2 total energy:                 {: 20.12f}",
                 result.total_energy);
  return result;
}

CorrelationResult run_ccsd(const Wavefunction &wfn,
                           const CorrelationOptions &opts,
                           const MethodSpec &mspec, bool with_triples) {
  const bool open_shell = wfn.mo.kind != SpinorbitalKind::Restricted;
  const std::string backend = resolve_backend(opts, mspec, false);
  const size_t budget = memory_budget_bytes(opts.max_memory_gb);

  CorrelationResult result;
  result.scf_energy = wfn.energy.total;
  result.method = with_triples ? "CCSD(T)" : "CCSD";

  if (open_shell) {
    occ::log::info("{:=^72s}", with_triples ? "  UHF CCSD(T) Calculation  "
                                            : "  UHF CCSD Calculation  ");
    occ::log::info("Backend: {}", backend);
    cc::UCCSDOptions uopts;
    uopts.backend = backend;
    uopts.n_frozen = resolve_frozen(opts, wfn);
    uopts.with_triples = with_triples;
    uopts.max_cycle = opts.max_cycle;
    uopts.tol = opts.tol;
    uopts.memory_budget = budget;
    uopts.thc = thc_options(opts, budget);

    cc::UCCSDResult r;
    if (backend == "exact") {
      occ::log::info("Note: the exact backend stores the O(V^4) vvvv block; "
                     "use df or thc for larger systems.");
      r = cc::uccsd(wfn.basis, wfn.mo, uopts);
    } else {
      const std::string auxname = resolve_aux_name(opts, wfn);
      occ::log::info("Auxiliary basis: {}", auxname);
      auto aux = load_aux_basis(wfn, auxname);
      r = cc::uccsd(wfn.basis, aux, wfn.mo, uopts);
    }
    if (!r.converged)
      occ::log::warn("CCSD did not converge in {} iterations", r.iterations);

    result.ccsd_correlation = r.e_corr;
    result.triples_correction = r.e_triples;
    result.correlation_energy = r.e_corr + r.e_triples;
    result.iterations = r.iterations;
    result.converged = r.converged;
    result.n_frozen = uopts.n_frozen;
  } else {
    occ::log::info("{:=^72s}", with_triples ? "  CCSD(T) Calculation  "
                                            : "  CCSD Calculation  ");
    occ::log::info("Backend: {}", backend);
    const int n_frozen = resolve_frozen(opts, wfn);

    cc::CCIntegrals eris = [&]() -> cc::CCIntegrals {
      if (backend == "exact") {
        occ::log::info("Note: the exact backend stores the O(V^4) vvvv block; "
                       "use df or thc for larger systems.");
        return cc::exact_eris(wfn.basis, wfn.mo, n_frozen, budget);
      }
      const std::string auxname = resolve_aux_name(opts, wfn);
      occ::log::info("Auxiliary basis: {}", auxname);
      auto aux = load_aux_basis(wfn, auxname);
      if (backend == "df")
        return cc::df_eris(wfn.basis, aux, wfn.mo, n_frozen, budget);
      if (backend == "thc") {
        auto thc = thc_options(opts, budget);
        occ::log::info("THC rank c = {}, ISDF selector = {}", thc.c_isdf,
                       thc.method == cc::IsdfMethod::QR ? "qr" : "cholesky");
        return cc::thc_eris(wfn.basis, aux, wfn.mo, thc, n_frozen, budget);
      }
      throw std::runtime_error("Unknown CCSD backend '" + backend +
                               "' (expected exact | df | thc)");
    }();

    cc::CCSDOptions copts;
    copts.max_cycle = opts.max_cycle;
    copts.tol = opts.tol;
    const cc::CCSDResult res = cc::ccsd(eris, copts);
    if (!res.converged)
      occ::log::warn("CCSD did not converge in {} iterations", res.iterations);

    const double et = with_triples ? cc::ccsd_t(res.t1, res.t2, eris) : 0.0;
    result.ccsd_correlation = res.e_corr;
    result.triples_correction = et;
    result.correlation_energy = res.e_corr + et;
    result.iterations = res.iterations;
    result.converged = res.converged;
    result.n_frozen = n_frozen;
  }

  result.total_energy = result.scf_energy + result.correlation_energy;
  occ::log::info("SCF energy:                       {: 20.12f}",
                 result.scf_energy);
  occ::log::info("CCSD correlation energy:          {: 20.12f}",
                 result.ccsd_correlation);
  if (with_triples) {
    occ::log::info("(T) correction:                   {: 20.12f}",
                   result.triples_correction);
    occ::log::info("CCSD(T) correlation energy:       {: 20.12f}",
                   result.correlation_energy);
  }
  occ::log::info("{:<33s} {: 20.12f}",
                 with_triples ? "CCSD(T) total energy:" : "CCSD total energy:",
                 result.total_energy);
  return result;
}

} // namespace

CorrelationResult run_correlation(const qm::Wavefunction &wfn,
                                  const CorrelationOptions &opts) {
  const auto mspec = parse_method_string(opts.method);
  switch (mspec.kind) {
  case MethodKind::MP2:
    return run_mp2(wfn, opts, mspec);
  case MethodKind::CCSD:
    return run_ccsd(wfn, opts, mspec, false);
  case MethodKind::CCSD_T:
    return run_ccsd(wfn, opts, mspec, true);
  default:
    throw std::runtime_error(
        "run_correlation: method '" + opts.method +
        "' is not a correlation method (expected mp2 | ccsd | ccsd(t))");
  }
}

} // namespace occ::driver
