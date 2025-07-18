#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/mo.h>
#include <occ/qm/mo_integral_engine.h>
#include <occ/qm/mp2.h>
#include <occ/qm/scf.h>
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

  auto basis = occ::qm::AOBasis::load(h2_atoms, "sto-3g");
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

  auto basis = occ::qm::AOBasis::load(water_atoms, "3-21G");
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

  auto basis = occ::qm::AOBasis::load(water_atoms, "3-21G");
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

  auto basis = occ::qm::AOBasis::load(h2_atoms, "sto-3g");

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

  auto basis = occ::qm::AOBasis::load(h2_atoms, "sto-3g");

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

  // Also check what we get from the ovov block
  auto ovov_block = mo_engine.compute_ovov_block();
  INFO("ovov_block dimensions: " << ovov_block.rows() << "x"
                                 << ovov_block.cols());

  if (ovov_block.size() > 0) {
    double ovov_00 =
        ovov_block(0, 0); // This should be (01|01) in (ov|ov) notation

    INFO("ovov_block(0,0) = " << ovov_00 << " [should be (01|01)]");

    // The ovov block should match the (01|01) direct computation
    REQUIRE_THAT(integral_0101, WithinAbs(ovov_00, 1e-12));
  }
}

TEST_CASE("MP2 results structure", "[mp2]") {
  // Simple hydrogen molecule for fast test
  std::vector<occ::core::Atom> h2_atoms{{1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4}};

  auto basis = occ::qm::AOBasis::load(h2_atoms, "sto-3g");

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

  auto basis = occ::qm::AOBasis::load(h2_atoms, "def2-svp");
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

TEST_CASE("DF tensor comparison", "[mp2][ri]") {
  // Simple H2 molecule test
  std::vector<occ::core::Atom> h2_atoms{
      {1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4} // Bohr
  };

  auto basis = occ::qm::AOBasis::load(h2_atoms, "sto-3g");
  auto aux_basis = occ::qm::AOBasis::load(h2_atoms, "def2-tzvp-rifit");
  basis.set_pure(false); // Use cartesian for simplicity

  INFO("DF tensor comparison test");
  INFO("Primary basis functions: " << basis.nbf());
  INFO("Auxiliary basis functions: " << aux_basis.nbf());

  // Create conventional integral engine
  occ::qm::IntegralEngine conv_engine(basis);
  auto conv_tensor = conv_engine.four_center_integrals_tensor();

  // Create DF integral engine
  const auto &atoms = basis.atoms();
  const auto &ao_shells = basis.shells();
  const auto &aux_shells = aux_basis.shells();
  occ::qm::IntegralEngineDF df_engine(atoms, ao_shells, aux_shells);
  auto df_tensor = df_engine.four_center_integrals_tensor();

  const size_t nbf = basis.nbf();
  // Calculate number of unique integrals: (N(N+1)/2)*(N(N+1)/2+1)/2
  size_t n_pairs = nbf * (nbf + 1) / 2;
  size_t n_unique = n_pairs * (n_pairs + 1) / 2;
  INFO("Comparing " << n_unique << " symmetry-unique integrals (out of "
                    << nbf * nbf * nbf * nbf << " total)");

  // Compare integrals with strict tolerances appropriate for quantum chemistry
  // Only check symmetry-unique integrals (8-fold symmetry: ijkl = jikl = ijlk =
  // jilk = klij = lkij = klji = lkji)
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  double total_conv = 0.0;
  double total_df = 0.0;
  size_t count = 0;
  size_t failures = 0;

  for (size_t i = 0; i < nbf; ++i) {
    for (size_t j = 0; j <= i; ++j) { // j <= i
      for (size_t k = 0; k < nbf; ++k) {
        for (size_t l = 0; l <= k; ++l) { // l <= k
          // Additional constraint: ij >= kl to avoid double counting
          size_t ij = i * (i + 1) / 2 + j;
          size_t kl = k * (k + 1) / 2 + l;
          if (ij < kl)
            continue;
          double conv_val = conv_tensor(i, j, k, l);
          double df_val = df_tensor(i, j, k, l);
          double abs_diff = std::abs(conv_val - df_val);

          // For significant integrals, check both absolute and relative errors
          if (std::abs(conv_val) > 1e-10) {
            max_abs_diff = std::max(max_abs_diff, abs_diff);
            total_conv += std::abs(conv_val);
            total_df += std::abs(df_val);
            count++;

            // Use relative error for large integrals, absolute error for small
            // ones
            double tolerance;
            bool use_relative = std::abs(conv_val) > 1e-6;

            if (use_relative) {
              double rel_diff = abs_diff / std::abs(conv_val);
              max_rel_diff = std::max(max_rel_diff, rel_diff);
              tolerance = 1e-6; // 0.0001% relative error for large integrals

              if (rel_diff > tolerance) {
                failures++;
                if (failures <= 5) { // Report first few failures
                  INFO("Relative error failure at ("
                       << i << "," << j << "," << k << "," << l << "): "
                       << "conv=" << conv_val << ", df=" << df_val
                       << ", rel_err=" << rel_diff << " (tol=" << tolerance
                       << ")");
                }
              }
            } else {
              tolerance = 1e-8; // Absolute error for small integrals

              if (abs_diff > tolerance) {
                failures++;
                if (failures <= 5) { // Report first few failures
                  INFO("Absolute error failure at ("
                       << i << "," << j << "," << k << "," << l << "): "
                       << "conv=" << conv_val << ", df=" << df_val
                       << ", abs_err=" << abs_diff << " (tol=" << tolerance
                       << ")");
                }
              }
            }
          }
        }
      }
    }
  }

  double overall_rel_error =
      (count > 0) ? std::abs(total_conv - total_df) / total_conv : 0.0;

  INFO("Max absolute difference: " << max_abs_diff);
  INFO("Max relative difference: " << max_rel_diff);
  INFO("Total conventional norm: " << total_conv);
  INFO("Total DF norm: " << total_df);
  INFO("Overall relative error: " << overall_rel_error);
  INFO("Significant integrals compared: " << count);
  INFO("Tolerance failures: " << failures);

  // Reasonable tolerances for DF approximation
  REQUIRE(max_abs_diff < 1e-4); // DF typically accurate to ~1e-4 to 1e-5
  REQUIRE(max_rel_diff < 1e-4); // 0.01% relative tolerance for large integrals
  REQUIRE(overall_rel_error < 1e-4); // 0.01% overall accuracy
}

TEST_CASE("RI-MP2 H2 STO-3G energy", "[mp2][ri]") {
  // Simple H2 molecule test - same setup as conventional MP2 test
  std::vector<occ::core::Atom> h2_atoms{
      {1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4} // Bohr
  };

  auto basis = occ::qm::AOBasis::load(h2_atoms, "sto-3g");
  auto aux_basis = occ::qm::AOBasis::load(h2_atoms, "def2-tzvp-rifit");
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
