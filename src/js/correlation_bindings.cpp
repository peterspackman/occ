#include "correlation_bindings.h"
#include <emscripten/bind.h>
#include <occ/driver/correlation.h>
#include <occ/qm/cc/ccsd.h>
#include <occ/qm/cc/integrals.h>
#include <occ/qm/cc/triples.h>
#include <occ/qm/cc/uccsd.h>
#include <occ/qm/correlation/mp2.h>
#include <occ/qm/fitting_basis.h>
#include <occ/qm/wavefunction.h>

using namespace emscripten;
using occ::driver::CorrelationOptions;
using occ::driver::CorrelationResult;
using occ::gto::AOBasis;
using occ::qm::MolecularOrbitals;
using occ::qm::MP2;
using occ::qm::Wavefunction;
namespace cc = occ::qm::cc;

void register_correlation_bindings() {
  // High-level uniform API: converged SCF wavefunction in, correlated
  // energies out (CLI-equivalent backend/aux-basis/frozen-core handling).
  class_<CorrelationOptions>("CorrelationOptions")
      .constructor<>()
      .property("method", &CorrelationOptions::method)
      .property("backend", &CorrelationOptions::backend)
      .property("auxBasis", &CorrelationOptions::aux_basis)
      .property("spinScaling", &CorrelationOptions::spin_scaling)
      .property("nFrozen", &CorrelationOptions::n_frozen)
      .property("maxMemoryGb", &CorrelationOptions::max_memory_gb)
      .property("maxCycle", &CorrelationOptions::max_cycle)
      .property("tol", &CorrelationOptions::tol)
      .property("thcCIsdf", &CorrelationOptions::thc_c_isdf)
      .property("thcIsdfMethod", &CorrelationOptions::thc_isdf_method)
      .property("thcGridAngular", &CorrelationOptions::thc_grid_angular)
      .property("thcGridRadialPrecision",
                &CorrelationOptions::thc_grid_radial_precision)
      .property("laplacePoints", &CorrelationOptions::laplace_points);

  class_<CorrelationResult>("CorrelationResult")
      .property("method", &CorrelationResult::method)
      .property("scfEnergy", &CorrelationResult::scf_energy)
      .property("correlationEnergy", &CorrelationResult::correlation_energy)
      .property("totalEnergy", &CorrelationResult::total_energy)
      .property("sameSpin", &CorrelationResult::same_spin)
      .property("oppositeSpin", &CorrelationResult::opposite_spin)
      .property("scaledCorrelation", &CorrelationResult::scaled_correlation)
      .property("ccsdCorrelation", &CorrelationResult::ccsd_correlation)
      .property("triplesCorrection", &CorrelationResult::triples_correction)
      .property("iterations", &CorrelationResult::iterations)
      .property("converged", &CorrelationResult::converged)
      .property("nFrozen", &CorrelationResult::n_frozen)
      .function("toString",
                optional_override([](const CorrelationResult &r) {
                  return std::string("<CorrelationResult ") + r.method +
                         " E=" + std::to_string(r.total_energy) + ">";
                }));

  // runCorrelation(wfn) / runCorrelation(wfn, "ccsd(t)") /
  // runCorrelation(wfn, options)
  function("runCorrelation",
           optional_override([](const Wavefunction &wfn) {
             return occ::driver::run_correlation(wfn, {});
           }));
  function("runCorrelationMethod",
           optional_override(
               [](const Wavefunction &wfn, const std::string &method) {
                 CorrelationOptions opts;
                 opts.method = method;
                 return occ::driver::run_correlation(wfn, opts);
               }));
  function("runCorrelationWithOptions",
           optional_override(
               [](const Wavefunction &wfn, const CorrelationOptions &opts) {
                 return occ::driver::run_correlation(wfn, opts);
               }));

  // Low-level surface for building on the internals.
  class_<MP2>("MP2")
      .constructor<const AOBasis &, const MolecularOrbitals &, double>()
      .constructor<const AOBasis &, const AOBasis &, const MolecularOrbitals &,
                   double>()
      .function("computeCorrelationEnergy", &MP2::compute_correlation_energy)
      .function("setFrozenCore", &MP2::set_frozen_core)
      .function("setFrozenCoreAuto", &MP2::set_frozen_core_auto)
      .function("setMemoryBudget", &MP2::set_memory_budget)
      .function("setScsParameters", &MP2::set_scs_parameters)
      .function("results", optional_override([](const MP2 &mp2) {
                  return mp2.results();
                }))
      .function("scfEnergy", optional_override([](const MP2 &mp2) {
                  return mp2.scf_energy();
                }))
      .function("correlationEnergy", optional_override([](const MP2 &mp2) {
                  return mp2.correlation_energy();
                }))
      .function("totalEnergy", optional_override([](const MP2 &mp2) {
                  return mp2.total_energy();
                }));

  class_<MP2::Results>("MP2Results")
      .property("sameSpinCorrelation", &MP2::Results::same_spin_correlation)
      .property("oppositeSpinCorrelation",
                &MP2::Results::opposite_spin_correlation)
      .property("totalCorrelation", &MP2::Results::total_correlation)
      .property("scsMp2Correlation", &MP2::Results::scs_mp2_correlation)
      .property("nFrozenCore", &MP2::Results::n_frozen_core)
      .property("nActiveOcc", &MP2::Results::n_active_occ)
      .property("nActiveVirt", &MP2::Results::n_active_virt);

  // Restricted coupled cluster: integral backends + solver + (T).
  class_<cc::CCIntegrals>("CCIntegrals")
      .property("nocc", &cc::CCIntegrals::nocc)
      .property("nvir", &cc::CCIntegrals::nvir);

  class_<cc::CCSDOptions>("CCSDOptions")
      .constructor<>()
      .property("maxCycle", &cc::CCSDOptions::max_cycle)
      .property("tol", &cc::CCSDOptions::tol)
      .property("diis", &cc::CCSDOptions::diis);

  class_<cc::CCSDResult>("CCSDResult")
      .property("eCorr", &cc::CCSDResult::e_corr)
      .property("iterations", &cc::CCSDResult::iterations)
      .property("converged", &cc::CCSDResult::converged);

  function("numFrozenCore", &cc::num_frozen_core);
  function("exactEris",
           optional_override([](const AOBasis &basis,
                                const MolecularOrbitals &mo, int n_frozen) {
             return cc::exact_eris(basis, mo, n_frozen);
           }));
  function("dfEris",
           optional_override([](const AOBasis &basis, const AOBasis &aux,
                                const MolecularOrbitals &mo, int n_frozen) {
             return cc::df_eris(basis, aux, mo, n_frozen);
           }));
  function("ccsd", optional_override([](const cc::CCIntegrals &eris) {
             return cc::ccsd(eris);
           }));
  function("ccsdWithOptions",
           optional_override([](const cc::CCIntegrals &eris,
                                const cc::CCSDOptions &opts) {
             return cc::ccsd(eris, opts);
           }));
  function("ccsdT", optional_override([](const cc::CCSDResult &r,
                                         const cc::CCIntegrals &eris) {
             return cc::ccsd_t(r.t1, r.t2, eris);
           }));

  // Unrestricted (spin-adapted) CCSD(T).
  class_<cc::UCCSDOptions>("UCCSDOptions")
      .constructor<>()
      .property("backend", &cc::UCCSDOptions::backend)
      .property("nFrozen", &cc::UCCSDOptions::n_frozen)
      .property("withTriples", &cc::UCCSDOptions::with_triples)
      .property("maxCycle", &cc::UCCSDOptions::max_cycle)
      .property("tol", &cc::UCCSDOptions::tol);

  class_<cc::UCCSDResult>("UCCSDResult")
      .property("eCorr", &cc::UCCSDResult::e_corr)
      .property("eTriples", &cc::UCCSDResult::e_triples)
      .property("iterations", &cc::UCCSDResult::iterations)
      .property("converged", &cc::UCCSDResult::converged);

  function("uccsd",
           optional_override([](const AOBasis &basis,
                                const MolecularOrbitals &mo,
                                const cc::UCCSDOptions &opts) {
             return cc::uccsd(basis, mo, opts);
           }));
  function("uccsdWithAuxBasis",
           optional_override([](const AOBasis &basis, const AOBasis &aux,
                                const MolecularOrbitals &mo,
                                const cc::UCCSDOptions &opts) {
             return cc::uccsd(basis, aux, mo, opts);
           }));

  // Auxiliary (fitting) basis resolution, as used by the CLI.
  function("resolveFittingBasis",
           optional_override(
               [](const std::string &orbital_basis, const std::string &kind) {
                 auto k = (kind == "jk" || kind == "JK")
                              ? occ::qm::FittingKind::JK
                              : occ::qm::FittingKind::Correlation;
                 return occ::qm::resolve_fitting_basis(orbital_basis, k);
               }));
}
