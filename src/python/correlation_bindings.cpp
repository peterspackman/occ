#include "correlation_bindings.h"
#include <fmt/core.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <occ/driver/correlation.h>
#include <occ/qm/cc/ccsd.h>
#include <occ/qm/cc/integrals.h>
#include <occ/qm/cc/thc_mp2.h>
#include <occ/qm/cc/triples.h>
#include <occ/qm/cc/uccsd.h>
#include <occ/qm/correlation/mp2.h>
#include <occ/qm/fitting_basis.h>
#include <occ/qm/wavefunction.h>

using namespace nb::literals;
using occ::driver::CorrelationOptions;
using occ::driver::CorrelationResult;
using occ::gto::AOBasis;
using occ::qm::MolecularOrbitals;
using occ::qm::MP2;
using occ::qm::Wavefunction;
namespace cc = occ::qm::cc;

namespace {

// Copy an Eigen (column-major) Tensor into a numpy array with the same shape.
template <int N>
nb::ndarray<nb::numpy, double> tensor_to_numpy(
    const Eigen::Tensor<double, N> &t) {
  std::array<size_t, N> shape;
  for (int i = 0; i < N; i++)
    shape[i] = static_cast<size_t>(t.dimension(i));
  double *data = new double[t.size()];
  std::copy(t.data(), t.data() + t.size(), data);
  nb::capsule owner(data, [](void *p) noexcept { delete[] (double *)p; });
  return nb::ndarray<nb::numpy, double>(data, N, shape.data(), owner, nullptr,
                                        nb::dtype<double>(),
                                        nb::device::cpu::value, 0,
                                        'F' /* column-major */);
}

} // namespace

nb::module_ register_correlation_bindings(nb::module_ &m) {

  // High-level uniform API: converged SCF wavefunction in, correlated
  // energies out. Backend/aux-basis/frozen-core resolution matches the CLI.
  nb::class_<CorrelationOptions>(m, "CorrelationOptions")
      .def(nb::init<>())
      .def_rw("method", &CorrelationOptions::method,
              "mp2 | ccsd | ccsd(t), with optional ri-/df-/thc- prefix")
      .def_rw("backend", &CorrelationOptions::backend,
              "auto | conventional/exact | ri/df | thc")
      .def_rw("aux_basis", &CorrelationOptions::aux_basis,
              "auxiliary basis name; empty = auto-resolve")
      .def_rw("spin_scaling", &CorrelationOptions::spin_scaling,
              "none | scs | sos (MP2 only)")
      .def_rw("n_frozen", &CorrelationOptions::n_frozen,
              "-1 = auto (chemical core), 0 = all-electron")
      .def_rw("max_memory_gb", &CorrelationOptions::max_memory_gb)
      .def_rw("max_cycle", &CorrelationOptions::max_cycle)
      .def_rw("tol", &CorrelationOptions::tol)
      .def_rw("thc_c_isdf", &CorrelationOptions::thc_c_isdf)
      .def_rw("thc_isdf_method", &CorrelationOptions::thc_isdf_method)
      .def_rw("thc_grid_angular", &CorrelationOptions::thc_grid_angular)
      .def_rw("thc_grid_radial_precision",
              &CorrelationOptions::thc_grid_radial_precision)
      .def_rw("laplace_points", &CorrelationOptions::laplace_points);

  nb::class_<CorrelationResult>(m, "CorrelationResult")
      .def_ro("method", &CorrelationResult::method)
      .def_ro("scf_energy", &CorrelationResult::scf_energy)
      .def_ro("correlation_energy", &CorrelationResult::correlation_energy)
      .def_ro("total_energy", &CorrelationResult::total_energy)
      .def_ro("same_spin", &CorrelationResult::same_spin)
      .def_ro("opposite_spin", &CorrelationResult::opposite_spin)
      .def_ro("scaled_correlation", &CorrelationResult::scaled_correlation)
      .def_ro("ccsd_correlation", &CorrelationResult::ccsd_correlation)
      .def_ro("triples_correction", &CorrelationResult::triples_correction)
      .def_ro("iterations", &CorrelationResult::iterations)
      .def_ro("converged", &CorrelationResult::converged)
      .def_ro("n_frozen", &CorrelationResult::n_frozen)
      .def("__repr__", [](const CorrelationResult &r) {
        return fmt::format("<CorrelationResult {} E={:.10f} Ecorr={:.10f}>",
                           r.method, r.total_energy, r.correlation_energy);
      });

  m.def("run_correlation", &occ::driver::run_correlation, "wavefunction"_a,
        "options"_a,
        "Run MP2/CCSD/CCSD(T) on a converged SCF wavefunction (CLI-equivalent "
        "backend and auxiliary-basis handling)");
  m.def(
      "run_correlation",
      [](const Wavefunction &wfn, const std::string &method,
         const std::string &backend, const std::string &aux_basis,
         const std::string &spin_scaling, int n_frozen, double max_memory_gb,
         int max_cycle, double tol) {
        CorrelationOptions opts;
        opts.method = method;
        opts.backend = backend;
        opts.aux_basis = aux_basis;
        opts.spin_scaling = spin_scaling;
        opts.n_frozen = n_frozen;
        opts.max_memory_gb = max_memory_gb;
        opts.max_cycle = max_cycle;
        opts.tol = tol;
        return occ::driver::run_correlation(wfn, opts);
      },
      "wavefunction"_a, "method"_a = "mp2", nb::kw_only(),
      "backend"_a = "auto", "aux_basis"_a = "", "spin_scaling"_a = "none",
      "n_frozen"_a = -1, "max_memory_gb"_a = 1.0, "max_cycle"_a = 100,
      "tol"_a = 1e-9);

  // Low-level surface for building on the internals.
  nb::class_<MP2> mp2(m, "MP2");

  nb::enum_<MP2::Algorithm>(mp2, "Algorithm")
      .value("Conventional", MP2::Conventional)
      .value("RI", MP2::RI);

  nb::class_<MP2::Results>(mp2, "Results")
      .def_ro("same_spin_correlation", &MP2::Results::same_spin_correlation)
      .def_ro("opposite_spin_correlation",
              &MP2::Results::opposite_spin_correlation)
      .def_ro("total_correlation", &MP2::Results::total_correlation)
      .def_ro("scs_mp2_correlation", &MP2::Results::scs_mp2_correlation)
      .def_ro("pair_energies", &MP2::Results::pair_energies)
      .def_ro("n_frozen_core", &MP2::Results::n_frozen_core)
      .def_ro("n_active_occ", &MP2::Results::n_active_occ)
      .def_ro("n_active_virt", &MP2::Results::n_active_virt)
      .def_ro("n_total_occ", &MP2::Results::n_total_occ)
      .def_ro("n_total_virt", &MP2::Results::n_total_virt);

  // PostHFMethod stores a reference to the MolecularOrbitals: keep the
  // ctor arguments alive as long as the MP2 object exists.
  mp2.def(nb::init<const AOBasis &, const MolecularOrbitals &, double>(),
          "basis"_a, "mo"_a, "scf_energy"_a, nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 3>())
      .def(nb::init<const AOBasis &, const AOBasis &,
                    const MolecularOrbitals &, double>(),
           "basis"_a, "aux_basis"_a, "mo"_a, "scf_energy"_a,
           nb::keep_alive<1, 2>(), nb::keep_alive<1, 3>(),
           nb::keep_alive<1, 4>(), "RI-MP2 with an auxiliary basis")
      .def("compute_correlation_energy", &MP2::compute_correlation_energy)
      .def("set_frozen_core", &MP2::set_frozen_core)
      .def("set_frozen_core_auto", &MP2::set_frozen_core_auto)
      .def("set_virtual_cutoff_energy", &MP2::set_virtual_cutoff_energy)
      .def("set_max_virtuals", &MP2::set_max_virtuals)
      .def("set_orbital_energy_cutoffs", &MP2::set_orbital_energy_cutoffs,
           "e_min"_a = -1.5, "e_max"_a = 1000.0)
      .def("set_memory_budget", &MP2::set_memory_budget)
      .def("set_scs_parameters", &MP2::set_scs_parameters,
           "c_ss"_a = 1.0 / 3.0, "c_os"_a = 1.2)
      .def_prop_ro("algorithm", &MP2::algorithm)
      .def_prop_ro("results", &MP2::results)
      .def_prop_ro("scf_energy", &MP2::scf_energy)
      .def_prop_ro("correlation_energy", &MP2::correlation_energy)
      .def_prop_ro("total_energy", &MP2::total_energy);

  // Restricted coupled cluster: integral backends + solver + (T).
  nb::class_<cc::CCIntegrals>(m, "CCIntegrals")
      .def_ro("nocc", &cc::CCIntegrals::nocc)
      .def_ro("nvir", &cc::CCIntegrals::nvir)
      .def_ro("mo_energy", &cc::CCIntegrals::mo_energy)
      .def("__repr__", [](const cc::CCIntegrals &e) {
        return fmt::format("<CCIntegrals nocc={} nvir={}>", e.nocc, e.nvir);
      });

  // Registered before the eris factories: thc_eris takes a ThcOptions
  // default argument, which requires the type to be bound already.
  nb::enum_<cc::IsdfMethod>(m, "IsdfMethod")
      .value("QR", cc::IsdfMethod::QR)
      .value("Cholesky", cc::IsdfMethod::Cholesky);

  nb::class_<cc::ThcOptions>(m, "ThcOptions")
      .def(nb::init<>())
      .def_rw("method", &cc::ThcOptions::method)
      .def_rw("n_isdf", &cc::ThcOptions::n_isdf)
      .def_rw("c_isdf", &cc::ThcOptions::c_isdf)
      .def_rw("tol", &cc::ThcOptions::tol)
      .def_rw("grid_max_angular", &cc::ThcOptions::grid_max_angular)
      .def_rw("grid_radial_precision", &cc::ThcOptions::grid_radial_precision)
      .def_rw("reg", &cc::ThcOptions::reg)
      .def_rw("memory_budget", &cc::ThcOptions::memory_budget);

  m.def("num_frozen_core", &cc::num_frozen_core, "basis"_a,
        "Chemical-core frozen orbital count for a basis (CCSD(T) default)");
  m.def("exact_eris", &cc::exact_eris, "basis"_a, "mo"_a, "n_frozen"_a = 0,
        "memory_budget"_a = size_t(1) << 30,
        "Exact MO integral blocks (stores the O(V^4) vvvv ladder)");
  m.def("df_eris", &cc::df_eris, "basis"_a, "aux_basis"_a, "mo"_a,
        "n_frozen"_a = 0, "memory_budget"_a = size_t(1) << 30,
        "Density-fitted MO integral blocks (vvvv never formed)");
  m.def("thc_eris", &cc::thc_eris, "basis"_a, "aux_basis"_a, "mo"_a,
        "options"_a = cc::ThcOptions{}, "n_frozen"_a = 0,
        "memory_budget"_a = size_t(1) << 30,
        "THC MO integral blocks (vvvv ladder via THC factors)");

  nb::class_<cc::CCSDOptions>(m, "CCSDOptions")
      .def(nb::init<>())
      .def_rw("max_cycle", &cc::CCSDOptions::max_cycle)
      .def_rw("tol", &cc::CCSDOptions::tol)
      .def_rw("diis", &cc::CCSDOptions::diis);

  nb::class_<cc::CCSDResult>(m, "CCSDResult")
      .def_ro("e_corr", &cc::CCSDResult::e_corr)
      .def_ro("iterations", &cc::CCSDResult::iterations)
      .def_ro("converged", &cc::CCSDResult::converged)
      .def_prop_ro(
          "t1",
          [](const cc::CCSDResult &r) { return tensor_to_numpy<2>(r.t1); },
          nb::rv_policy::automatic,
          "converged singles amplitudes (nocc x nvir)")
      .def_prop_ro(
          "t2",
          [](const cc::CCSDResult &r) { return tensor_to_numpy<4>(r.t2); },
          nb::rv_policy::automatic,
          "converged doubles amplitudes (nocc x nocc x nvir x nvir)")
      .def("__repr__", [](const cc::CCSDResult &r) {
        return fmt::format("<CCSDResult e_corr={:.10f} converged={}>",
                           r.e_corr, r.converged);
      });

  m.def("ccsd", &cc::ccsd, "eris"_a, "options"_a = cc::CCSDOptions{},
        "Restricted CCSD; backend-agnostic over exact/df/thc integrals");
  m.def(
      "ccsd_t",
      [](const cc::CCSDResult &r, const cc::CCIntegrals &eris) {
        return cc::ccsd_t(r.t1, r.t2, eris);
      },
      "ccsd_result"_a, "eris"_a,
      "Perturbative (T) correction from converged CCSD amplitudes");

  // Unrestricted (spin-adapted) CCSD(T).
  nb::class_<cc::UCCSDOptions>(m, "UCCSDOptions")
      .def(nb::init<>())
      .def_rw("backend", &cc::UCCSDOptions::backend, "exact | df | thc")
      .def_rw("n_frozen", &cc::UCCSDOptions::n_frozen)
      .def_rw("with_triples", &cc::UCCSDOptions::with_triples)
      .def_rw("max_cycle", &cc::UCCSDOptions::max_cycle)
      .def_rw("tol", &cc::UCCSDOptions::tol)
      .def_rw("memory_budget", &cc::UCCSDOptions::memory_budget)
      .def_rw("thc", &cc::UCCSDOptions::thc);

  nb::class_<cc::UCCSDResult>(m, "UCCSDResult")
      .def_ro("e_corr", &cc::UCCSDResult::e_corr)
      .def_ro("e_triples", &cc::UCCSDResult::e_triples)
      .def_ro("iterations", &cc::UCCSDResult::iterations)
      .def_ro("converged", &cc::UCCSDResult::converged)
      .def("__repr__", [](const cc::UCCSDResult &r) {
        return fmt::format(
            "<UCCSDResult e_corr={:.10f} e_triples={:.10f} converged={}>",
            r.e_corr, r.e_triples, r.converged);
      });

  m.def("uccsd",
        nb::overload_cast<const AOBasis &, const MolecularOrbitals &,
                          const cc::UCCSDOptions &>(&cc::uccsd),
        "basis"_a, "mo"_a, "options"_a = cc::UCCSDOptions{},
        "Spin-adapted UCCSD(T), exact backend");
  m.def("uccsd",
        nb::overload_cast<const AOBasis &, const AOBasis &,
                          const MolecularOrbitals &, const cc::UCCSDOptions &>(
            &cc::uccsd),
        "basis"_a, "aux_basis"_a, "mo"_a, "options"_a = cc::UCCSDOptions{},
        "Spin-adapted UCCSD(T) with an auxiliary basis (df/thc backends)");

  // THC-MP2 (Laplace + THC factors).
  nb::class_<cc::ThcMP2Options>(m, "ThcMP2Options")
      .def(nb::init<>())
      .def_rw("thc", &cc::ThcMP2Options::thc)
      .def_rw("n_laplace", &cc::ThcMP2Options::n_laplace)
      .def_rw("n_frozen", &cc::ThcMP2Options::n_frozen)
      .def_rw("memory_budget", &cc::ThcMP2Options::memory_budget)
      .def_rw("opposite_spin_only", &cc::ThcMP2Options::opposite_spin_only);

  nb::class_<cc::ThcMP2Result>(m, "ThcMP2Result")
      .def_ro("same_spin", &cc::ThcMP2Result::same_spin)
      .def_ro("opposite_spin", &cc::ThcMP2Result::opposite_spin)
      .def_ro("total", &cc::ThcMP2Result::total)
      .def_ro("n_isdf", &cc::ThcMP2Result::n_isdf)
      .def_ro("n_laplace", &cc::ThcMP2Result::n_laplace);

  m.def("thc_mp2", &cc::thc_mp2, "basis"_a, "aux_basis"_a, "mo"_a,
        "options"_a = cc::ThcMP2Options{}, "LS-THC-MP2 correlation energy");

  // Auxiliary (fitting) basis resolution, as used by the CLI.
  nb::enum_<occ::qm::FittingKind>(m, "FittingKind")
      .value("JK", occ::qm::FittingKind::JK)
      .value("Correlation", occ::qm::FittingKind::Correlation);

  m.def("resolve_fitting_basis", &occ::qm::resolve_fitting_basis,
        "orbital_basis_name"_a, "kind"_a,
        "Recommended auxiliary basis name for an orbital basis");

  return m;
}
