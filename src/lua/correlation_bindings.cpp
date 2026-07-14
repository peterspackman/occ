#include "correlation_bindings.h"
#include "eigen_conv.h"
#include <fmt/core.h>
#include <occ/driver/correlation.h>
#include <occ/qm/cc/ccsd.h>
#include <occ/qm/cc/integrals.h>
#include <occ/qm/cc/triples.h>
#include <occ/qm/cc/uccsd.h>
#include <occ/qm/correlation/mp2.h>
#include <occ/qm/fitting_basis.h>
#include <occ/qm/wavefunction.h>

namespace occ::lua_bindings {

using occ::driver::CorrelationOptions;
using occ::driver::CorrelationResult;
using occ::gto::AOBasis;
using occ::qm::MolecularOrbitals;
using occ::qm::MP2;
using occ::qm::Wavefunction;
namespace cc = occ::qm::cc;
namespace lb = luabridge;

namespace {

// Accept nil (defaults), a method string, a CorrelationOptions userdata, or a
// plain table of option fields.
CorrelationOptions options_from_ref(const lb::LuaRef &ref) {
  CorrelationOptions o;
  if (ref.isNil())
    return o;
  if (ref.isString()) {
    o.method = ref.unsafe_cast<std::string>();
    return o;
  }
  if (ref.isUserdata())
    return ref.unsafe_cast<CorrelationOptions>();
  if (!ref.isTable())
    throw std::runtime_error("run_correlation: expected a method string, an "
                             "occ.CorrelationOptions or a table of options");
  auto get_str = [&](const char *key, std::string &out) {
    lb::LuaRef v = ref[key];
    if (!v.isNil())
      out = v.unsafe_cast<std::string>();
  };
  auto get_num = [&](const char *key, auto &out) {
    lb::LuaRef v = ref[key];
    if (!v.isNil())
      out = static_cast<std::decay_t<decltype(out)>>(
          v.unsafe_cast<double>());
  };
  get_str("method", o.method);
  get_str("backend", o.backend);
  get_str("aux_basis", o.aux_basis);
  get_str("spin_scaling", o.spin_scaling);
  get_str("thc_isdf_method", o.thc_isdf_method);
  get_num("n_frozen", o.n_frozen);
  get_num("max_memory_gb", o.max_memory_gb);
  get_num("max_cycle", o.max_cycle);
  get_num("tol", o.tol);
  get_num("thc_c_isdf", o.thc_c_isdf);
  get_num("thc_grid_angular", o.thc_grid_angular);
  get_num("thc_grid_radial_precision", o.thc_grid_radial_precision);
  get_num("laplace_points", o.laplace_points);
  return o;
}

occ::qm::FittingKind fitting_kind_from_string(const std::string &s) {
  if (s == "jk" || s == "JK")
    return occ::qm::FittingKind::JK;
  if (s == "correlation" || s == "Correlation")
    return occ::qm::FittingKind::Correlation;
  throw std::runtime_error("resolve_fitting_basis: unknown fitting kind '" +
                           s + "' (expected 'jk' or 'correlation')");
}

} // namespace

void register_correlation_bindings(lua_State *L) {
  lb::getGlobalNamespace(L)
      .beginNamespace("occ")

      // High-level uniform API: converged SCF wavefunction in, correlated
      // energies out (CLI-equivalent backend/aux/frozen-core handling).
      .beginClass<CorrelationOptions>("CorrelationOptions")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("method", &CorrelationOptions::method)
      .addPropertyReadWrite("backend", &CorrelationOptions::backend)
      .addPropertyReadWrite("aux_basis", &CorrelationOptions::aux_basis)
      .addPropertyReadWrite("spin_scaling", &CorrelationOptions::spin_scaling)
      .addPropertyReadWrite("n_frozen", &CorrelationOptions::n_frozen)
      .addPropertyReadWrite("max_memory_gb",
                            &CorrelationOptions::max_memory_gb)
      .addPropertyReadWrite("max_cycle", &CorrelationOptions::max_cycle)
      .addPropertyReadWrite("tol", &CorrelationOptions::tol)
      .addPropertyReadWrite("thc_c_isdf", &CorrelationOptions::thc_c_isdf)
      .addPropertyReadWrite("thc_isdf_method",
                            &CorrelationOptions::thc_isdf_method)
      .addPropertyReadWrite("thc_grid_angular",
                            &CorrelationOptions::thc_grid_angular)
      .addPropertyReadWrite("thc_grid_radial_precision",
                            &CorrelationOptions::thc_grid_radial_precision)
      .addPropertyReadWrite("laplace_points",
                            &CorrelationOptions::laplace_points)
      .endClass()

      .beginClass<CorrelationResult>("CorrelationResult")
      .addProperty("method", &CorrelationResult::method)
      .addProperty("scf_energy", &CorrelationResult::scf_energy)
      .addProperty("correlation_energy",
                   &CorrelationResult::correlation_energy)
      .addProperty("total_energy", &CorrelationResult::total_energy)
      .addProperty("same_spin", &CorrelationResult::same_spin)
      .addProperty("opposite_spin", &CorrelationResult::opposite_spin)
      .addProperty("scaled_correlation",
                   &CorrelationResult::scaled_correlation)
      .addProperty("ccsd_correlation", &CorrelationResult::ccsd_correlation)
      .addProperty("triples_correction",
                   &CorrelationResult::triples_correction)
      .addProperty("iterations", &CorrelationResult::iterations)
      .addProperty("converged", &CorrelationResult::converged)
      .addProperty("n_frozen", &CorrelationResult::n_frozen)
      .addFunction(
          "__tostring", +[](const CorrelationResult *r) {
            return fmt::format("<CorrelationResult {} E={:.10f}>", r->method,
                               r->total_energy);
          })
      .endClass()

      // occ.run_correlation(wfn)                    -> MP2, all defaults
      // occ.run_correlation(wfn, "ccsd(t)")         -> method string
      // occ.run_correlation(wfn, {method="ccsd", backend="df"})
      // occ.run_correlation(wfn, occ.CorrelationOptions())
      .addFunction(
          "run_correlation",
          +[](const Wavefunction &wfn, const lb::LuaRef &options) {
            return occ::driver::run_correlation(wfn,
                                                options_from_ref(options));
          })

      // Low-level surface for building on the internals.
      .beginClass<MP2::Results>("MP2Results")
      .addProperty("same_spin_correlation",
                   &MP2::Results::same_spin_correlation)
      .addProperty("opposite_spin_correlation",
                   &MP2::Results::opposite_spin_correlation)
      .addProperty("total_correlation", &MP2::Results::total_correlation)
      .addProperty("scs_mp2_correlation", &MP2::Results::scs_mp2_correlation)
      .addProperty("n_frozen_core", &MP2::Results::n_frozen_core)
      .addProperty("n_active_occ", &MP2::Results::n_active_occ)
      .addProperty("n_active_virt", &MP2::Results::n_active_virt)
      .endClass()

      .beginClass<MP2>("MP2")
      // LuaBridge3 doesn't auto-overload: canonical conventional ctor, RI
      // via a static factory (matching the HF.with_kind pattern).
      .addConstructor<void (*)(const AOBasis &, const MolecularOrbitals &,
                               double)>()
      .addStaticFunction(
          "with_aux_basis",
          +[](const AOBasis &basis, const AOBasis &aux,
              const MolecularOrbitals &mo, double scf_energy) {
            return new MP2(basis, aux, mo, scf_energy);
          })
      .addFunction("compute_correlation_energy",
                   &MP2::compute_correlation_energy)
      .addFunction("set_frozen_core", &MP2::set_frozen_core)
      .addFunction("set_frozen_core_auto", &MP2::set_frozen_core_auto)
      .addFunction("set_memory_budget", &MP2::set_memory_budget)
      .addFunction("set_scs_parameters", &MP2::set_scs_parameters)
      .addProperty(
          "results", +[](const MP2 *mp2) { return mp2->results(); })
      // These live on the PostHFMethod base: a base-class member-function
      // pointer doesn't satisfy LuaBridge3's Class<MP2> getter shape, so
      // wrap in free-function getters.
      .addProperty(
          "scf_energy", +[](const MP2 *m) { return m->scf_energy(); })
      .addProperty(
          "correlation_energy",
          +[](const MP2 *m) { return m->correlation_energy(); })
      .addProperty(
          "total_energy", +[](const MP2 *m) { return m->total_energy(); })
      .endClass()

      .beginClass<cc::CCIntegrals>("CCIntegrals")
      .addProperty("nocc", &cc::CCIntegrals::nocc)
      .addProperty("nvir", &cc::CCIntegrals::nvir)
      .addFunction(
          "__tostring", +[](const cc::CCIntegrals *e) {
            return fmt::format("<CCIntegrals nocc={} nvir={}>", e->nocc,
                               e->nvir);
          })
      .endClass()

      .beginClass<cc::CCSDOptions>("CCSDOptions")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("max_cycle", &cc::CCSDOptions::max_cycle)
      .addPropertyReadWrite("tol", &cc::CCSDOptions::tol)
      .addPropertyReadWrite("diis", &cc::CCSDOptions::diis)
      .endClass()

      .beginClass<cc::CCSDResult>("CCSDResult")
      .addProperty("e_corr", &cc::CCSDResult::e_corr)
      .addProperty("iterations", &cc::CCSDResult::iterations)
      .addProperty("converged", &cc::CCSDResult::converged)
      .endClass()

      .addFunction("num_frozen_core", &cc::num_frozen_core)
      .addFunction(
          "exact_eris",
          +[](const AOBasis &basis, const MolecularOrbitals &mo,
              int n_frozen) {
            return cc::exact_eris(basis, mo, n_frozen);
          })
      .addFunction(
          "df_eris",
          +[](const AOBasis &basis, const AOBasis &aux,
              const MolecularOrbitals &mo, int n_frozen) {
            return cc::df_eris(basis, aux, mo, n_frozen);
          })
      .addFunction(
          "ccsd",
          +[](const cc::CCIntegrals &eris, const lb::LuaRef &options) {
            cc::CCSDOptions opts;
            if (options.isUserdata())
              opts = options.unsafe_cast<cc::CCSDOptions>();
            return cc::ccsd(eris, opts);
          })
      .addFunction(
          "ccsd_t",
          +[](const cc::CCSDResult &r, const cc::CCIntegrals &eris) {
            return cc::ccsd_t(r.t1, r.t2, eris);
          })

      .beginClass<cc::UCCSDOptions>("UCCSDOptions")
      .addConstructor<void (*)()>()
      .addPropertyReadWrite("backend", &cc::UCCSDOptions::backend)
      .addPropertyReadWrite("n_frozen", &cc::UCCSDOptions::n_frozen)
      .addPropertyReadWrite("with_triples", &cc::UCCSDOptions::with_triples)
      .addPropertyReadWrite("max_cycle", &cc::UCCSDOptions::max_cycle)
      .addPropertyReadWrite("tol", &cc::UCCSDOptions::tol)
      .endClass()

      .beginClass<cc::UCCSDResult>("UCCSDResult")
      .addProperty("e_corr", &cc::UCCSDResult::e_corr)
      .addProperty("e_triples", &cc::UCCSDResult::e_triples)
      .addProperty("iterations", &cc::UCCSDResult::iterations)
      .addProperty("converged", &cc::UCCSDResult::converged)
      .endClass()

      .addFunction(
          "uccsd",
          +[](const AOBasis &basis, const MolecularOrbitals &mo,
              const cc::UCCSDOptions &opts) {
            return cc::uccsd(basis, mo, opts);
          })
      .addFunction(
          "uccsd_with_aux_basis",
          +[](const AOBasis &basis, const AOBasis &aux,
              const MolecularOrbitals &mo, const cc::UCCSDOptions &opts) {
            return cc::uccsd(basis, aux, mo, opts);
          })

      .addFunction(
          "resolve_fitting_basis",
          +[](const std::string &orbital_basis, const std::string &kind) {
            return occ::qm::resolve_fitting_basis(
                orbital_basis, fitting_kind_from_string(kind));
          })

      .endNamespace();
}

} // namespace occ::lua_bindings
