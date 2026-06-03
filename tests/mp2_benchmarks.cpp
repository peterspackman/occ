#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/core.h>
#include <occ/core/atom.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/gto/gto.h>
#include <occ/qm/correlation/mp2.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <thread>

// Baseline timing harness for the MP2 paths. These establish a reference point
// for the performance refactor (see IMPLEMENTATION_PLAN.md). They are tagged
// [.benchmark] so they are hidden by default; run explicitly with, e.g.:
//   ./build/tests/mp2_tests "[mp2-bench]"
// For a per-step breakdown (AO ints / transform / energy) of a single run, set
//   OCC_LOG_LEVEL=DEBUG
// which makes the transform and energy stages log their individual durations.

using occ::qm::HartreeFock;
using occ::qm::MP2;
using occ::qm::SCF;

namespace {

// Run HF to convergence and return the converged wavefunction context.
struct ScfResult {
  occ::qm::MolecularOrbitals mo;
  double energy{0.0};
};

ScfResult converged_scf(const occ::gto::AOBasis &basis) {
  HartreeFock hf(basis);
  SCF<HartreeFock> scf(hf);
  double e = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);
  return {scf.ctx.mo, e};
}

} // namespace

TEST_CASE("Benchmark: MP2 H2O/def2-SVP", "[.benchmark][mp2-bench]") {
  occ::parallel::set_num_threads(std::thread::hardware_concurrency());
  occ::log::set_log_level("warn"); // keep the benchmark output readable

  std::vector<occ::core::Atom> atoms{
      {8, 0.0000000, 0.0000000, 0.1173470 * occ::units::ANGSTROM_TO_BOHR},
      {1, 0.0000000, 0.7572150 * occ::units::ANGSTROM_TO_BOHR,
       -0.4693880 * occ::units::ANGSTROM_TO_BOHR},
      {1, 0.0000000, -0.7572150 * occ::units::ANGSTROM_TO_BOHR,
       -0.4693880 * occ::units::ANGSTROM_TO_BOHR}};

  auto basis = occ::gto::AOBasis::load(atoms, "def2-svp");
  basis.set_pure(false);
  auto aux_basis = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");
  aux_basis.set_kind(basis.kind());

  auto scf = converged_scf(basis);

  fmt::print("\nBenchmark: MP2 H2O/def2-SVP ({} threads)\n",
             occ::parallel::get_num_threads());
  fmt::print("  AO basis: {} functions, Aux basis: {} functions\n", basis.nbf(),
             aux_basis.nbf());

  // Correctness anchor: conventional and RI should agree within DF error.
  double e_conv = [&] {
    MP2 mp2(basis, scf.mo, scf.energy);
    mp2.set_frozen_core_auto();
    return mp2.compute_correlation_energy();
  }();
  double e_ri = [&] {
    MP2 mp2(basis, aux_basis, scf.mo, scf.energy);
    mp2.set_frozen_core_auto();
    return mp2.compute_correlation_energy();
  }();
  fmt::print("  E_corr conventional: {:16.10f} Eh\n", e_conv);
  fmt::print("  E_corr RI:           {:16.10f} Eh\n", e_ri);
  fmt::print("  |conv - RI|:         {:12.2e} Eh\n", std::abs(e_conv - e_ri));
  REQUIRE_THAT(e_ri, Catch::Matchers::WithinAbs(e_conv, 5e-3));

  BENCHMARK("Conventional MP2 (full compute)") {
    MP2 mp2(basis, scf.mo, scf.energy);
    mp2.set_frozen_core_auto();
    return mp2.compute_correlation_energy();
  };

  BENCHMARK("RI-MP2 (full compute)") {
    MP2 mp2(basis, aux_basis, scf.mo, scf.energy);
    mp2.set_frozen_core_auto();
    return mp2.compute_correlation_energy();
  };
}

// Larger case: RI-MP2 only (conventional would materialize a multi-GB N^4
// tensor). Tagged [slow] in addition to [.benchmark].
TEST_CASE("Benchmark: RI-MP2 benzene/def2-SVP", "[.benchmark][mp2-bench][slow]") {
  occ::parallel::set_num_threads(std::thread::hardware_concurrency());
  occ::log::set_log_level("warn"); // keep the benchmark output readable

  std::vector<occ::core::Atom> atoms{
      {6, 0.0000000, 1.3970000, 0.0000000},
      {6, 1.2098079, 0.6985000, 0.0000000},
      {6, 1.2098079, -0.6985000, 0.0000000},
      {6, 0.0000000, -1.3970000, 0.0000000},
      {6, -1.2098079, -0.6985000, 0.0000000},
      {6, -1.2098079, 0.6985000, 0.0000000},
      {1, 0.0000000, 2.4810000, 0.0000000},
      {1, 2.1486254, 1.2405000, 0.0000000},
      {1, 2.1486254, -1.2405000, 0.0000000},
      {1, 0.0000000, -2.4810000, 0.0000000},
      {1, -2.1486254, -1.2405000, 0.0000000},
      {1, -2.1486254, 1.2405000, 0.0000000}};
  for (auto &a : atoms) {
    a.x *= occ::units::ANGSTROM_TO_BOHR;
    a.y *= occ::units::ANGSTROM_TO_BOHR;
    a.z *= occ::units::ANGSTROM_TO_BOHR;
  }

  auto basis = occ::gto::AOBasis::load(atoms, "def2-svp");
  basis.set_pure(false);
  auto aux_basis = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");
  aux_basis.set_kind(basis.kind());

  auto scf = converged_scf(basis);

  fmt::print("\nBenchmark: RI-MP2 benzene/def2-SVP ({} threads)\n",
             occ::parallel::get_num_threads());
  fmt::print("  AO basis: {} functions, Aux basis: {} functions\n", basis.nbf(),
             aux_basis.nbf());

  double e_ri = [&] {
    MP2 mp2(basis, aux_basis, scf.mo, scf.energy);
    mp2.set_frozen_core_auto();
    return mp2.compute_correlation_energy();
  }();
  fmt::print("  E_corr RI: {:16.10f} Eh\n", e_ri);

  BENCHMARK("RI-MP2 (full compute)") {
    MP2 mp2(basis, aux_basis, scf.mo, scf.energy);
    mp2.set_frozen_core_auto();
    return mp2.compute_correlation_energy();
  };
}
