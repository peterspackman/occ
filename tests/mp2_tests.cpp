#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/core/format_matrix.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/correlation/mo_integral_engine.h>
#include <occ/qm/correlation/mp2.h>
#include <occ/core/parallel.h>
#include <occ/qm/mo.h>
#include <occ/qm/scf.h>
#include <thread>
#include <vector>

using Catch::Matchers::WithinAbs;
using occ::format_matrix;
using occ::Mat;
using occ::qm::HartreeFock;
using occ::qm::MolecularOrbitals;
using occ::qm::SpinorbitalKind;
using occ::util::all_close;

// MP2 Tests

TEST_CASE("MO integral transformation validation", "[mp2]") {
  // Simple H2 molecule for easy validation
  std::vector<occ::core::Atom> h2_atoms{
      {1, 0.0, 0.0, 0.0},                               // H
      {1, 0.0, 0.0, 1.4 * occ::units::ANGSTROM_TO_BOHR} // H at 1.4 Angstrom
  };

  auto basis = occ::gto::AOBasis::load(h2_atoms, "sto-3g");
  basis.set_pure(false);

  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);

  double energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  // Create engines
  occ::qm::IntegralEngine ao_engine(basis);
  occ::qm::MOIntegralEngine mo_engine(ao_engine, scf.ctx.mo);

  // Test: Check MO overlap matrix should be identity
  // S_MO = C^T * S_AO * C = I

  // Get AO overlap matrix
  Mat S_AO = ao_engine.one_electron_operator(occ::qm::cint::Operator::overlap);
  const auto &C = scf.ctx.mo.C;

  // Transform to MO basis: S_MO = C^T * S_AO * C
  Mat S_MO = C.transpose() * S_AO * C;

  occ::log::info("AO overlap matrix dimensions: {}x{}", S_AO.rows(),
                 S_AO.cols());
  occ::log::info("MO coefficient matrix dimensions: {}x{}", C.rows(), C.cols());
  occ::log::info("MO overlap matrix dimensions: {}x{}", S_MO.rows(),
                 S_MO.cols());

  // Check that MO overlap matrix is identity
  Mat identity = Mat::Identity(S_MO.rows(), S_MO.cols());
  Mat diff = S_MO - identity;
  double max_error = diff.cwiseAbs().maxCoeff();

  occ::log::info("Maximum deviation from identity: {:.2e}", max_error);

  // Print a few diagonal and off-diagonal elements for debugging
  for (size_t i = 0; i < std::min(3ul, static_cast<size_t>(S_MO.rows())); ++i) {
    occ::log::info("S_MO({},{}) = {:.10f} (should be 1.0)", i, i, S_MO(i, i));
    if (i + 1 < S_MO.rows()) {
      occ::log::info("S_MO({},{}) = {:.10f} (should be 0.0)", i, i + 1,
                     S_MO(i, i + 1));
    }
  }

  // MO overlap should be identity within numerical precision
  REQUIRE(max_error < 1e-10);
}

TEST_CASE("MOIntegralEngine initialization", "[mp2]") {
  // Water molecule coordinates (convert from Angstroms to Bohr)
  std::vector<occ::core::Atom> water_atoms{
      {8, -0.7021961 * occ::units::ANGSTROM_TO_BOHR,
       -0.0560603 * occ::units::ANGSTROM_TO_BOHR,
       0.0099423 * occ::units::ANGSTROM_TO_BOHR}, // O
      {1, -1.0221932 * occ::units::ANGSTROM_TO_BOHR,
       0.8467758 * occ::units::ANGSTROM_TO_BOHR,
       -0.0114887 * occ::units::ANGSTROM_TO_BOHR}, // H
      {1, 0.2575211 * occ::units::ANGSTROM_TO_BOHR,
       0.0421215 * occ::units::ANGSTROM_TO_BOHR,
       0.0052190 * occ::units::ANGSTROM_TO_BOHR} // H
  };

  auto basis = occ::gto::AOBasis::load(water_atoms, "3-21G");
  basis.set_pure(false); // Use Cartesian GTOs like ORCA default

  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);

  double energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  // Test MOIntegralEngine initialization
  occ::qm::IntegralEngine engine(basis);
  occ::qm::MOIntegralEngine mo_engine(engine, scf.ctx.mo);

  REQUIRE(mo_engine.n_ao() == basis.nbf());
  REQUIRE(mo_engine.n_occupied() > 0);
  REQUIRE(mo_engine.n_virtual() > 0);
  REQUIRE(mo_engine.n_occupied() + mo_engine.n_virtual() == basis.nbf());
}

TEST_CASE("MP2 water 3-21G energy", "[mp2]") {
  // Water molecule coordinates (convert from Angstroms to Bohr)
  std::vector<occ::core::Atom> water_atoms{
      {8, -0.7021961 * occ::units::ANGSTROM_TO_BOHR,
       -0.0560603 * occ::units::ANGSTROM_TO_BOHR,
       0.0099423 * occ::units::ANGSTROM_TO_BOHR}, // O
      {1, -1.0221932 * occ::units::ANGSTROM_TO_BOHR,
       0.8467758 * occ::units::ANGSTROM_TO_BOHR,
       -0.0114887 * occ::units::ANGSTROM_TO_BOHR}, // H
      {1, 0.2575211 * occ::units::ANGSTROM_TO_BOHR,
       0.0421215 * occ::units::ANGSTROM_TO_BOHR,
       0.0052190 * occ::units::ANGSTROM_TO_BOHR} // H
  };

  auto basis = occ::gto::AOBasis::load(water_atoms, "3-21G");
  basis.set_pure(false); // Use Cartesian GTOs like ORCA default

  // Run HF calculation
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);

  double hf_energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  // Run MP2 calculation with automatic frozen core (matches CLI behavior)
  occ::qm::MP2 mp2(basis, scf.ctx.mo, hf_energy);
  mp2.set_frozen_core_auto(); // Use automatic frozen core like CLI
  double correlation_energy = mp2.compute_correlation_energy();
  double total_mp2_energy = mp2.total_energy();

  // Reference values with automatic frozen core (1s orbital of O frozen)
  constexpr double ref_hf_energy = -75.58532568528730;    // Eh
  constexpr double ref_correlation_energy = -0.120945919; // Eh
  constexpr double ref_total_energy = -75.706271609; // Eh (with frozen core)

  // Check energies (allowing reasonable numerical tolerance)
  REQUIRE_THAT(hf_energy, WithinAbs(ref_hf_energy, 1e-6));
  REQUIRE_THAT(correlation_energy, WithinAbs(ref_correlation_energy, 1e-4));
  REQUIRE_THAT(total_mp2_energy, WithinAbs(ref_total_energy, 1e-4));
}

TEST_CASE("MP2 H2 STO-3G energy", "[mp2]") {
  // Simple hydrogen molecule
  std::vector<occ::core::Atom> h2_atoms{{1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4}};

  auto basis = occ::gto::AOBasis::load(h2_atoms, "sto-3g");

  // Run HF calculation
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);

  double hf_energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  // For H2 with STO-3G: 1 occupied orbital, 1 virtual orbital
  // Manual calculation: MP2 energy = (00|00)^2 / (eps_0 - eps_1)
  const auto &mo_energies = scf.ctx.mo.energies;
  double eps_0 = mo_energies(0); // occupied
  double eps_1 = mo_energies(1); // virtual

  INFO("Orbital energies: eps_0 = " << eps_0 << ", eps_1 = " << eps_1);
  INFO("Denominator: " << eps_0 - eps_1);

  // Run MP2 calculation
  occ::qm::MP2 mp2(basis, scf.ctx.mo, hf_energy);
  double correlation_energy = mp2.compute_correlation_energy();

  // Reference value from ORCA MP2/STO-3G calculation
  constexpr double ref_correlation_energy = -0.013164456;

  INFO("MP2 correlation energy: " << correlation_energy);

  REQUIRE_THAT(correlation_energy, WithinAbs(ref_correlation_energy, 1e-5));
}

TEST_CASE("MO integral transformation", "[mo_transform]") {
  // H2 molecule for simple test
  std::vector<occ::core::Atom> h2_atoms{{1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4}};

  auto basis = occ::gto::AOBasis::load(h2_atoms, "sto-3g");

  // Run HF calculation
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);

  double hf_energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  const auto &mo = scf.ctx.mo;

  INFO("System: H2, basis: STO-3G");
  INFO("Number of AO functions: " << basis.nbf());
  INFO("Number of MO functions: " << mo.C.cols());

  // Get AO matrices
  auto S_ao = hf.compute_overlap_matrix(); // Overlap matrix
  auto T_ao = hf.compute_kinetic_matrix(); // Kinetic energy matrix
  auto V_ao =
      hf.compute_nuclear_attraction_matrix(); // Nuclear attraction matrix

  INFO("S_AO(0,0) = " << S_ao(0, 0) << ", S_AO(0,1) = " << S_ao(0, 1));
  INFO("T_AO(0,0) = " << T_ao(0, 0) << ", T_AO(0,1) = " << T_ao(0, 1));
  INFO("V_AO(0,0) = " << V_ao(0, 0) << ", V_AO(0,1) = " << V_ao(0, 1));

  // Transform to MO basis: X_MO = C^T * X_AO * C
  auto S_mo = mo.C.transpose() * S_ao * mo.C;
  auto T_mo = mo.C.transpose() * T_ao * mo.C;
  auto V_mo = mo.C.transpose() * V_ao * mo.C;

  INFO("S_MO(0,0) = " << S_mo(0, 0) << ", S_MO(0,1) = " << S_mo(0, 1));
  INFO("T_MO(0,0) = " << T_mo(0, 0) << ", T_MO(0,1) = " << T_mo(0, 1));
  INFO("V_MO(0,0) = " << V_mo(0, 0) << ", V_MO(0,1) = " << V_mo(0, 1));

  // Check MO overlap matrix (should be identity)
  auto I = occ::Mat::Identity(S_mo.rows(), S_mo.cols());
  auto S_deviation = (S_mo - I).array().abs().maxCoeff();
  INFO("Max deviation from identity: " << S_deviation);

  // MO overlap should be identity within numerical precision
  REQUIRE_THAT(S_deviation, WithinAbs(0.0, 1e-12));

  // Check orbital energies via MO Fock matrix
  auto F_ao_computed = hf.compute_fock(mo);
  auto F_mo_computed = mo.C.transpose() * F_ao_computed * mo.C;

  // Also check with the converged Fock matrix from SCF context
  const auto &F_ao_converged = scf.ctx.F;
  auto F_mo_converged = mo.C.transpose() * F_ao_converged * mo.C;

  INFO("F_AO_computed(0,0) = " << F_ao_computed(0, 0)
                               << ", F_AO_converged(0,0) = "
                               << F_ao_converged(0, 0));
  INFO("F_MO_computed(0,0) = " << F_mo_computed(0, 0)
                               << ", eps[0] = " << mo.energies(0));
  INFO("F_MO_converged(0,0) = " << F_mo_converged(0, 0)
                                << ", eps[0] = " << mo.energies(0));
  INFO("F_MO_computed(1,1) = " << F_mo_computed(1, 1)
                               << ", eps[1] = " << mo.energies(1));
  INFO("F_MO_converged(1,1) = " << F_mo_converged(1, 1)
                                << ", eps[1] = " << mo.energies(1));

  // The converged Fock matrix diagonal should match orbital energies
  for (int i = 0; i < F_mo_converged.rows(); ++i) {
    REQUIRE_THAT(F_mo_converged(i, i), WithinAbs(mo.energies(i), 1e-10));
  }

  // Test 2-electron integrals transformation
  // For now, create a new integral engine with the same basis
  occ::qm::IntegralEngine engine(basis);
  occ::qm::MOIntegralEngine mo_engine(engine, mo);

  // Debug MO coefficients
  INFO("MO coefficient matrix:");
  INFO("  C(0,0) = " << mo.C(0, 0) << ", C(0,1) = " << mo.C(0, 1));
  INFO("  C(1,0) = " << mo.C(1, 0) << ", C(1,1) = " << mo.C(1, 1));
  INFO("MOIntegralEngine info:");
  INFO("  n_occ = " << mo_engine.n_occupied()
                    << ", n_virt = " << mo_engine.n_virtual());

  // Check the split coefficient matrices
  auto C_occ_test = mo.C.leftCols(mo_engine.n_occupied());
  auto C_virt_test = mo.C.rightCols(mo_engine.n_virtual());

  INFO("Coefficient matrix splitting check:");
  INFO("  Original C dimensions: " << mo.C.rows() << "x" << mo.C.cols());
  INFO("  C_occ dimensions: " << C_occ_test.rows() << "x" << C_occ_test.cols());
  INFO("  C_virt dimensions: " << C_virt_test.rows() << "x"
                               << C_virt_test.cols());
  INFO("  C_occ(0,0) = " << C_occ_test(0, 0)
                         << " [should match C(0,0) = " << mo.C(0, 0) << "]");
  INFO("  C_occ(1,0) = " << C_occ_test(1, 0)
                         << " [should match C(1,0) = " << mo.C(1, 0) << "]");
  INFO("  C_virt(0,0) = " << C_virt_test(0, 0)
                          << " [should match C(0,1) = " << mo.C(0, 1) << "]");
  INFO("  C_virt(1,0) = " << C_virt_test(1, 0)
                          << " [should match C(1,1) = " << mo.C(1, 1) << "]");

  // Test specific integrals for H2
  double integral_0000 = mo_engine.compute_mo_eri(0, 0, 0, 0); // (00|00)
  double integral_0001 = mo_engine.compute_mo_eri(0, 0, 0, 1); // (00|01)
  double integral_0101 = mo_engine.compute_mo_eri(0, 1, 0, 1); // (01|01)
  double integral_1111 = mo_engine.compute_mo_eri(1, 1, 1, 1); // (11|11)

  INFO("Direct integrals:");
  INFO("  (00|00) = " << integral_0000);
  INFO("  (00|01) = " << integral_0001);
  INFO("  (01|01) = " << integral_0101);
  INFO("  (11|11) = " << integral_1111);

  // compute_mo_eri must respect 8-fold permutational symmetry of (ij|kl).
  REQUIRE_THAT(integral_0001,
               WithinAbs(mo_engine.compute_mo_eri(0, 0, 1, 0), 1e-12));
  REQUIRE_THAT(integral_0001,
               WithinAbs(mo_engine.compute_mo_eri(0, 1, 0, 0), 1e-12));
  REQUIRE_THAT(integral_0001,
               WithinAbs(mo_engine.compute_mo_eri(1, 0, 0, 0), 1e-12));
  // Diagonal Coulomb integrals are positive.
  REQUIRE(integral_0000 > 0.0);
  REQUIRE(integral_0101 > 0.0);
  REQUIRE(integral_1111 > 0.0);
}

TEST_CASE("Conventional MP2 (AO-direct) memory and threads", "[mp2]") {
  std::vector<occ::core::Atom> water_atoms{
      {8, -0.7021961 * occ::units::ANGSTROM_TO_BOHR,
       -0.0560603 * occ::units::ANGSTROM_TO_BOHR,
       0.0099423 * occ::units::ANGSTROM_TO_BOHR},
      {1, -1.0221932 * occ::units::ANGSTROM_TO_BOHR,
       0.8467758 * occ::units::ANGSTROM_TO_BOHR,
       -0.0114887 * occ::units::ANGSTROM_TO_BOHR},
      {1, 0.2575211 * occ::units::ANGSTROM_TO_BOHR,
       0.0421215 * occ::units::ANGSTROM_TO_BOHR,
       0.0052190 * occ::units::ANGSTROM_TO_BOHR}};

  auto basis = occ::gto::AOBasis::load(water_atoms, "def2-svp");
  basis.set_pure(false);

  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  double hf_energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  auto run = [&](auto configure) {
    occ::qm::MP2 mp2(basis, scf.ctx.mo, hf_energy);
    mp2.set_frozen_core_auto();
    configure(mp2);
    return mp2.compute_correlation_energy();
  };

  SECTION("memory-budget blocking is exact") {
    double e_full = run([](occ::qm::MP2 &) {});
    // A 1-byte budget forces the smallest occupied block (fully AO-direct).
    double e_block = run([](occ::qm::MP2 &m) { m.set_memory_budget(1); });
    INFO("full: " << e_full << "  blocked: " << e_block);
    REQUIRE_THAT(e_block, WithinAbs(e_full, 1e-10));
  }

  SECTION("thread-count invariance") {
    occ::parallel::set_num_threads(1);
    double e1 = run([](occ::qm::MP2 &) {});
    unsigned int nthreads = std::max(2u, std::thread::hardware_concurrency());
    occ::parallel::set_num_threads(static_cast<int>(nthreads));
    double e2 = run([](occ::qm::MP2 &) {});
    INFO("1 thread: " << e1 << "  " << nthreads << " threads: " << e2);
    REQUIRE_THAT(e2, WithinAbs(e1, 1e-10));
  }
}

TEST_CASE("MP2 results structure", "[mp2]") {
  // Simple hydrogen molecule for fast test
  std::vector<occ::core::Atom> h2_atoms{{1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4}};

  auto basis = occ::gto::AOBasis::load(h2_atoms, "sto-3g");

  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);

  double hf_energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  occ::qm::MP2 mp2(basis, scf.ctx.mo, hf_energy);
  double correlation_energy = mp2.compute_correlation_energy();

  const auto &results = mp2.results();

  // Check that results structure is populated
  REQUIRE(results.total_correlation != 0.0);
  REQUIRE(results.total_correlation == correlation_energy);

  // For closed-shell molecules, should have both same-spin and opposite-spin
  // components (though for H2 with minimal basis, contributions may be small)
  REQUIRE(std::abs(results.same_spin_correlation +
                   results.opposite_spin_correlation -
                   results.total_correlation) < 1e-10);

  // SCS-MP2 should be different from regular MP2
  REQUIRE(std::abs(results.scs_mp2_correlation - results.total_correlation) >
          1e-12);
}

TEST_CASE("MP2 H2 def2-SVP energy", "[mp2]") {
  // H2 with def2-SVP basis to test with more virtual orbitals and p functions
  std::vector<occ::core::Atom> h2_atoms{
      {1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4} // Bohr
  };

  auto basis = occ::gto::AOBasis::load(h2_atoms, "def2-svp");
  basis.set_pure(false); // Use Cartesian GTOs

  // Run HF calculation
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);

  double hf_energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  // Run MP2 calculation
  occ::qm::MP2 mp2(basis, scf.ctx.mo, hf_energy);
  double correlation_energy = mp2.compute_correlation_energy();
  double total_mp2_energy = mp2.total_energy();

  // Reference values from PySCF (1.4 Bohr H-H distance)
  constexpr double ref_hf_energy = -1.1289016975;
  constexpr double ref_correlation_energy = -0.0262747430;
  constexpr double ref_total_energy = -1.1551764404;

  INFO("H2 def2-SVP test");
  INFO("Number of basis functions: " << basis.nbf());
  INFO("Number of occupied orbitals: " << scf.ctx.mo.n_alpha);
  INFO("Number of virtual orbitals: " << basis.nbf() - scf.ctx.mo.n_alpha);
  INFO("HF energy: " << hf_energy);
  INFO("MP2 correlation energy: " << correlation_energy);
  INFO("Total MP2 energy: " << total_mp2_energy);

  // Check energies
  REQUIRE_THAT(hf_energy, WithinAbs(ref_hf_energy, 1e-6));
  REQUIRE_THAT(correlation_energy, WithinAbs(ref_correlation_energy, 5e-4));
  REQUIRE_THAT(total_mp2_energy, WithinAbs(ref_total_energy, 5e-4));
}

TEST_CASE("RI-MP2 (B-tensor) correctness", "[mp2][ri]") {
  // Water / def2-SVP with a def2 auxiliary basis.
  std::vector<occ::core::Atom> water_atoms{
      {8, -0.7021961 * occ::units::ANGSTROM_TO_BOHR,
       -0.0560603 * occ::units::ANGSTROM_TO_BOHR,
       0.0099423 * occ::units::ANGSTROM_TO_BOHR},
      {1, -1.0221932 * occ::units::ANGSTROM_TO_BOHR,
       0.8467758 * occ::units::ANGSTROM_TO_BOHR,
       -0.0114887 * occ::units::ANGSTROM_TO_BOHR},
      {1, 0.2575211 * occ::units::ANGSTROM_TO_BOHR,
       0.0421215 * occ::units::ANGSTROM_TO_BOHR,
       0.0052190 * occ::units::ANGSTROM_TO_BOHR}};

  auto basis = occ::gto::AOBasis::load(water_atoms, "def2-svp");
  basis.set_pure(false);
  auto aux_basis = occ::gto::AOBasis::load(water_atoms, "def2-universal-jkfit");
  aux_basis.set_kind(basis.kind());

  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  double hf_energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  auto run_ri = [&](auto configure) {
    occ::qm::MP2 ri(basis, aux_basis, scf.ctx.mo, hf_energy);
    ri.set_frozen_core_auto();
    configure(ri);
    return ri.compute_correlation_energy();
  };

  SECTION("matches conventional MP2 within DF error") {
    occ::qm::MP2 conv(basis, scf.ctx.mo, hf_energy);
    conv.set_frozen_core_auto();
    double e_conv = conv.compute_correlation_energy();

    occ::qm::MP2 ri(basis, aux_basis, scf.ctx.mo, hf_energy);
    ri.set_frozen_core_auto();
    double e_ri = ri.compute_correlation_energy();
    INFO("conventional: " << e_conv << "  RI: " << e_ri);
    REQUIRE_THAT(e_ri, WithinAbs(e_conv, 2e-3));

    // same/opposite spin components must sum to the total correlation energy
    const auto &r = ri.results();
    REQUIRE_THAT(r.same_spin_correlation + r.opposite_spin_correlation,
                 WithinAbs(e_ri, 1e-12));
  }

  SECTION("memory-budget blocking is exact") {
    double e_full = run_ri([](occ::qm::MP2 &) {});
    // A 1-byte budget forces the smallest possible occupied block size.
    double e_block = run_ri([](occ::qm::MP2 &m) { m.set_memory_budget(1); });
    INFO("full: " << e_full << "  blocked: " << e_block);
    REQUIRE_THAT(e_block, WithinAbs(e_full, 1e-10));
  }

  SECTION("thread-count invariance") {
    occ::parallel::set_num_threads(1);
    double e1 = run_ri([](occ::qm::MP2 &) {});
    unsigned int nthreads = std::max(2u, std::thread::hardware_concurrency());
    occ::parallel::set_num_threads(static_cast<int>(nthreads));
    double e2 = run_ri([](occ::qm::MP2 &) {});
    INFO("1 thread: " << e1 << "  " << nthreads << " threads: " << e2);
    REQUIRE_THAT(e2, WithinAbs(e1, 1e-10));
  }

  SECTION("integral-direct 3-center matches stored") {
    // Default budget keeps the dense (μν|P) store; a sub-store budget forces
    // the integral-direct B build (no store) but still allows a full block.
    double e_stored = run_ri([](occ::qm::MP2 &) {});
    double e_direct =
        run_ri([](occ::qm::MP2 &m) { m.set_memory_budget(800 * 1024); });
    INFO("stored: " << e_stored << "  direct: " << e_direct);
    REQUIRE_THAT(e_direct, WithinAbs(e_stored, 1e-10));
  }
}

TEST_CASE("RI-MP2 H2 STO-3G energy", "[mp2][ri]") {
  // Simple H2 molecule test - same setup as conventional MP2 test
  std::vector<occ::core::Atom> h2_atoms{
      {1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4} // Bohr
  };

  auto basis = occ::gto::AOBasis::load(h2_atoms, "sto-3g");
  auto aux_basis = occ::gto::AOBasis::load(h2_atoms, "def2-tzvp-rifit");
  basis.set_pure(false);

  // Run SCF calculation
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);

  double hf_energy = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);

  INFO("SCF energy: " << hf_energy);

  // Test RI-MP2
  occ::qm::MP2 ri_mp2(basis, aux_basis, scf.ctx.mo, hf_energy);
  double ri_corr_energy = ri_mp2.compute_correlation_energy();

  INFO("RI-MP2 correlation energy: " << ri_corr_energy);

  // Expected conventional MP2 correlation energy for H2/STO-3G is -0.0131578700
  double expected_corr = -0.0131578700;
  double tolerance = 1e-4; // DF typically accurate to ~1e-4

  REQUIRE_THAT(ri_corr_energy, WithinAbs(expected_corr, tolerance));
}

// Helper: water geometry in Bohr (shared by the UHF tests below).
static std::vector<occ::core::Atom> uhf_water_atoms() {
  return {{8, -0.7021961 * occ::units::ANGSTROM_TO_BOHR,
           -0.0560603 * occ::units::ANGSTROM_TO_BOHR,
           0.0099423 * occ::units::ANGSTROM_TO_BOHR},
          {1, -1.0221932 * occ::units::ANGSTROM_TO_BOHR,
           0.8467758 * occ::units::ANGSTROM_TO_BOHR,
           -0.0114887 * occ::units::ANGSTROM_TO_BOHR},
          {1, 0.2575211 * occ::units::ANGSTROM_TO_BOHR,
           0.0421215 * occ::units::ANGSTROM_TO_BOHR,
           0.0052190 * occ::units::ANGSTROM_TO_BOHR}};
}

TEST_CASE("UHF MP2 closed-shell limit equals RHF", "[mp2][uhf]") {
  // A closed-shell system solved unrestricted must reproduce the restricted
  // MP2 energy exactly (validates the αα/ββ/αβ formulas in the RHF limit).
  auto atoms = uhf_water_atoms();
  auto basis = occ::gto::AOBasis::load(atoms, "def2-svp");
  basis.set_pure(false);
  auto aux = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");
  aux.set_kind(basis.kind());

  occ::qm::HartreeFock hf_r(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf_r(hf_r);
  double ehf_r = scf_r.compute_scf_energy();

  occ::qm::HartreeFock hf_u(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf_u(hf_u,
                                           occ::qm::SpinorbitalKind::Unrestricted);
  double ehf_u = scf_u.compute_scf_energy();
  REQUIRE_THAT(ehf_u, WithinAbs(ehf_r, 1e-6));

  auto corr = [](auto &mp2) {
    mp2.set_frozen_core_auto();
    return mp2.compute_correlation_energy();
  };

  occ::qm::MP2 conv_r(basis, scf_r.ctx.mo, ehf_r);
  occ::qm::MP2 conv_u(basis, scf_u.ctx.mo, ehf_u);
  double ec_r = corr(conv_r), ec_u = corr(conv_u);
  INFO("conventional RHF: " << ec_r << "  UHF: " << ec_u);
  REQUIRE_THAT(ec_u, WithinAbs(ec_r, 1e-7));

  occ::qm::MP2 ri_r(basis, aux, scf_r.ctx.mo, ehf_r);
  occ::qm::MP2 ri_u(basis, aux, scf_u.ctx.mo, ehf_u);
  double er_r = corr(ri_r), er_u = corr(ri_u);
  INFO("RI RHF: " << er_r << "  UHF: " << er_u);
  REQUIRE_THAT(er_u, WithinAbs(er_r, 1e-7));
}

TEST_CASE("UHF MP2 open-shell: conventional vs RI", "[mp2][uhf]") {
  // H2O+ doublet (9 electrons): genuinely open-shell (n_alpha != n_beta).
  auto atoms = uhf_water_atoms();
  auto basis = occ::gto::AOBasis::load(atoms, "def2-svp");
  basis.set_pure(false);
  auto aux = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");
  aux.set_kind(basis.kind());

  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf,
                                         occ::qm::SpinorbitalKind::Unrestricted);
  scf.set_charge_multiplicity(1, 2);
  double ehf = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);
  REQUIRE(scf.ctx.mo.n_alpha != scf.ctx.mo.n_beta);

  occ::qm::MP2 conv(basis, scf.ctx.mo, ehf);
  conv.set_frozen_core_auto();
  double e_conv = conv.compute_correlation_energy();

  occ::qm::MP2 ri(basis, aux, scf.ctx.mo, ehf);
  ri.set_frozen_core_auto();
  double e_ri = ri.compute_correlation_energy();

  INFO("open-shell conventional: " << e_conv << "  RI: " << e_ri);
  REQUIRE(e_conv < 0.0);
  // Two independent UHF implementations agree within DF error.
  REQUIRE_THAT(e_ri, WithinAbs(e_conv, 3e-3));

  const auto &r = conv.results();
  REQUIRE_THAT(r.same_spin_correlation + r.opposite_spin_correlation,
               WithinAbs(e_conv, 1e-10));
  REQUIRE(r.same_spin_correlation < 0.0);
  REQUIRE(r.opposite_spin_correlation < 0.0);
}
