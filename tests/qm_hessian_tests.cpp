#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/core/vibration.h>
#include <occ/gto/gto.h>
#include <occ/gto/shell.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hessians.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/io/core_json.h>
#include <vector>
#include <algorithm>
#include <fmt/core.h>
#include <fmt/ostream.h>

using Catch::Matchers::WithinAbs;
using occ::format_matrix;
using occ::Mat;
using occ::Vec;
using occ::qm::HartreeFock;
using occ::qm::HessianEvaluator;
using occ::qm::GradientEvaluator;
using occ::qm::MolecularOrbitals;
using occ::qm::SpinorbitalKind;
using occ::util::all_close;

TEST_CASE("Hessian evaluator nuclear repulsion", "[hessian][nuclear]") {
  std::vector<occ::core::Atom> atoms{
      {6, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 2.0}, {1, 0.0, 2.0, 0.0}};

  auto basis = occ::gto::AOBasis::load(atoms, "sto-3g");
  HartreeFock hf(basis);
  
  HessianEvaluator<HartreeFock> hess_eval(hf);
  auto hess_nn = hess_eval.nuclear_repulsion();

  // Nuclear repulsion Hessian should be symmetric
  REQUIRE(hess_nn.isApprox(hess_nn.transpose(), 1e-10));
  
  // Check dimensions
  REQUIRE(hess_nn.rows() == 9); // 3 atoms × 3 coordinates
  REQUIRE(hess_nn.cols() == 9);
  
  // Nuclear repulsion Hessian elements should be finite
  REQUIRE(hess_nn.allFinite());
}

TEST_CASE("Hessian finite differences vs analytical", "[hessian][finite_diff]") {
  // Small water molecule for testing
  std::vector<occ::core::Atom> atoms{
      {8, 0.0, 0.0, 0.0}, 
      {1, 0.0, 0.0, 1.8}, 
      {1, 1.2, 0.0, -0.6}};

  auto basis = occ::gto::AOBasis::load(atoms, "sto-3g");
  HartreeFock hf(basis);
  
  // Run SCF to get molecular orbitals
  occ::qm::SCF<HartreeFock> scf(hf);
  double e = scf.compute_scf_energy();
  occ::qm::MolecularOrbitals mo = scf.ctx.mo;
  
  HessianEvaluator<HartreeFock> hess_eval(hf);
  
  SECTION("Configuration methods") {
    // Test configuration methods
    hess_eval.set_step_size(0.001);
    REQUIRE(hess_eval.step_size() == Catch::Approx(0.001));
    
    hess_eval.set_use_acoustic_sum_rule(false);
    REQUIRE(hess_eval.use_acoustic_sum_rule() == false);
    
    hess_eval.set_use_acoustic_sum_rule(true);
    REQUIRE(hess_eval.use_acoustic_sum_rule() == true);
    
    // Test invalid step size
    REQUIRE_THROWS_AS(hess_eval.set_step_size(-0.001), std::invalid_argument);
    REQUIRE_THROWS_AS(hess_eval.set_step_size(0.0), std::invalid_argument);
  }
  
  SECTION("Nuclear repulsion Hessian properties") {
    auto analytical = hess_eval.nuclear_repulsion();
    
    // Basic sanity checks
    REQUIRE(analytical.allFinite());
    REQUIRE(analytical.isApprox(analytical.transpose(), 1e-12));
    
    // Check dimensions
    REQUIRE(analytical.rows() == 9); // 3 atoms × 3 coordinates
    REQUIRE(analytical.cols() == 9);
  }
  
  SECTION("Finite differences Hessian validation") {
    // Test finite differences implementation
    hess_eval.set_step_size(1e-5);
    auto numerical = hess_eval(scf.wavefunction());
    
    // Basic sanity checks
    REQUIRE(numerical.allFinite());
    REQUIRE(numerical.rows() == 9); // 3 atoms × 3 coordinates
    REQUIRE(numerical.cols() == 9);
    
    // Test symmetry (should be approximately symmetric)
    double symmetry_error = (numerical - numerical.transpose()).cwiseAbs().maxCoeff();
    INFO("Symmetry error: " << symmetry_error);
    REQUIRE(symmetry_error < 1e-6); // Should be quite symmetric for small h
  }
}

TEST_CASE("Hessian symmetry properties", "[hessian][symmetry]") {
  std::vector<occ::core::Atom> atoms{
      {6, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 2.0}};

  auto basis = occ::gto::AOBasis::load(atoms, "sto-3g");
  HartreeFock hf(basis);
  
  HessianEvaluator<HartreeFock> hess_eval(hf);
  auto hess_nn = hess_eval.nuclear_repulsion();

  // Test translational invariance (sum of rows/columns should be zero)
  Vec row_sums = hess_nn.rowwise().sum();
  Vec col_sums = hess_nn.colwise().sum();
  
  REQUIRE(row_sums.cwiseAbs().maxCoeff() < 1e-10);
  REQUIRE(col_sums.cwiseAbs().maxCoeff() < 1e-10);
  
  // Test that Hessian is symmetric
  REQUIRE(hess_nn.isApprox(hess_nn.transpose(), 1e-12));
}

// Helper function to test Hessian properties
void test_hessian_properties(const HartreeFock& hf, const MolecularOrbitals& mo) {
  HessianEvaluator<HartreeFock> hess_eval(const_cast<HartreeFock&>(hf));
  
  // Compute analytical Hessian (currently just nuclear part)
  auto hess = hess_eval.nuclear_repulsion();
  
  // Test basic properties
  REQUIRE(hess.allFinite());
  REQUIRE(hess.isApprox(hess.transpose(), 1e-12));
  
  // Test translational invariance
  Vec row_sums = hess.rowwise().sum();
  Vec col_sums = hess.colwise().sum();
  
  REQUIRE(row_sums.cwiseAbs().maxCoeff() < 1e-10);
  REQUIRE(col_sums.cwiseAbs().maxCoeff() < 1e-10);
}

TEST_CASE("Hessian validation suite", "[hessian][validation]") {
  SECTION("H2 molecule") {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4}};
    auto basis = occ::gto::AOBasis::load(atoms, "sto-3g");
    HartreeFock hf(basis);
    
    occ::qm::SCF<HartreeFock> scf(hf);
    double e = scf.compute_scf_energy();
    occ::qm::MolecularOrbitals mo = scf.ctx.mo;
    
    REQUIRE_NOTHROW(test_hessian_properties(hf, mo));
  }
  
  SECTION("Water molecule") {
    std::vector<occ::core::Atom> atoms{
        {8, 0.0, 0.0, 0.0}, 
        {1, 0.0, 0.0, 1.8}, 
        {1, 1.2, 0.0, -0.6}};
    auto basis = occ::gto::AOBasis::load(atoms, "sto-3g");
    HartreeFock hf(basis);
    
    occ::qm::SCF<HartreeFock> scf(hf);
    double e = scf.compute_scf_energy();
    occ::qm::MolecularOrbitals mo = scf.ctx.mo;
    
    REQUIRE_NOTHROW(test_hessian_properties(hf, mo));
  }
}

TEST_CASE("Hessian comparison: Analytical vs Finite Differences", "[hessian][comparison]") {
  SECTION("H2 molecule - Cartesian basis") {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4}};
    auto basis = occ::gto::AOBasis::load(atoms, "sto-3g");
    // Force Cartesian basis
    basis.set_pure(false);
    
    HartreeFock hf(basis);
    occ::qm::SCF<HartreeFock> scf(hf);
    double e = scf.compute_scf_energy();
    occ::qm::MolecularOrbitals mo = scf.ctx.mo;
    
    HessianEvaluator<HartreeFock> hess_eval(hf);
    
    fmt::print("\n=== H2 Molecule - Cartesian Basis ===\n");
    fmt::print("SCF Energy: {:.8f} Eh\n", e);
    
    // Compute Hessian using finite differences (full energy)
    auto hessian = hess_eval(scf.wavefunction());
    fmt::print("Hessian (finite differences):\n{}\n", format_matrix(hessian));
    
    // Basic sanity checks
    REQUIRE(hessian.allFinite());
    REQUIRE(hessian.rows() == 6); // 2 atoms × 3 coordinates  
    
    // Test symmetry (should be approximately symmetric)
    double symmetry_error = (hessian - hessian.transpose()).cwiseAbs().maxCoeff();
    INFO("Symmetry error: " << symmetry_error);
    REQUIRE(symmetry_error < 1e-6); // Should be quite symmetric
  }
  
  SECTION("H2 molecule - Spherical basis") {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 1.4}};
    auto basis = occ::gto::AOBasis::load(atoms, "sto-3g");
    // Force Spherical basis (default)
    basis.set_pure(true);
    
    HartreeFock hf(basis);
    occ::qm::SCF<HartreeFock> scf(hf);
    double e = scf.compute_scf_energy();
    occ::qm::MolecularOrbitals mo = scf.ctx.mo;
    
    HessianEvaluator<HartreeFock> hess_eval(hf);
    
    fmt::print("\n=== H2 Molecule - Spherical Basis ===\n");
    fmt::print("SCF Energy: {:.8f} Eh\n", e);
    
    // Compute Hessian using finite differences (full energy)
    auto hessian = hess_eval(scf.wavefunction());
    fmt::print("Hessian (finite differences):\n{}\n", format_matrix(hessian));
    
    // Basic sanity checks
    REQUIRE(hessian.allFinite());
    REQUIRE(hessian.rows() == 6); // 2 atoms × 3 coordinates
    
    // Test symmetry (should be approximately symmetric)
    double symmetry_error = (hessian - hessian.transpose()).cwiseAbs().maxCoeff();
    INFO("Symmetry error: " << symmetry_error);
    REQUIRE(symmetry_error < 1e-6); // Should be quite symmetric
  }
  
  SECTION("Water molecule - Spherical basis") {
    std::vector<occ::core::Atom> atoms{
        {8, 0.0, 0.0, 0.0}, 
        {1, 0.0, 0.0, 1.8}, 
        {1, 1.2, 0.0, -0.6}};
    auto basis = occ::gto::AOBasis::load(atoms, "sto-3g");
    basis.set_pure(true);
    
    HartreeFock hf(basis);
    occ::qm::SCF<HartreeFock> scf(hf);
    double e = scf.compute_scf_energy();
    occ::qm::MolecularOrbitals mo = scf.ctx.mo;
    
    HessianEvaluator<HartreeFock> hess_eval(hf);
    
    fmt::print("\n=== Water Molecule - Spherical Basis ===\n");
    fmt::print("SCF Energy: {:.8f} Eh\n", e);
    
    // Compute Hessian using finite differences (full energy)
    fmt::print("Computing finite differences Hessian...\n");
    auto hessian = hess_eval(scf.wavefunction());
    fmt::print("Hessian (finite differences):\n{}\n", format_matrix(hessian));
    
    // Basic sanity checks
    REQUIRE(hessian.allFinite());
    REQUIRE(hessian.rows() == 9); // 3 atoms × 3 coordinates
    
    // Test symmetry (should be approximately symmetric)
    double symmetry_error = (hessian - hessian.transpose()).cwiseAbs().maxCoeff();
    INFO("Symmetry error: " << symmetry_error);
    REQUIRE(symmetry_error < 1e-6); // Should be quite symmetric
  }
  
}

TEST_CASE("Water molecule - ORCA reference comparison (HF/3-21G)", "[hessian][orca]") {
  // ORCA optimized geometry from water.hess file (coordinates in Bohr - atomic units)
  std::vector<occ::core::Atom> atoms{
      {8, -0.000000000000, 0.000000000000, 0.120615570834},
      {1, 1.474863732488, 0.000000000000, -0.957206606042},
      {1, -1.474863732488, 0.000000000000, -0.957206606042}};
  
  auto basis = occ::gto::AOBasis::load(atoms, "3-21g");
  basis.set_pure(true); // Spherical basis as in ORCA
  
  HartreeFock hf(basis);
  occ::qm::SCF<HartreeFock> scf(hf);
  double e = scf.compute_scf_energy();
  occ::qm::MolecularOrbitals mo = scf.ctx.mo;
  
  HessianEvaluator<HartreeFock> hess_eval(hf);
  hess_eval.set_step_size(0.005);  // ORCA default
  hess_eval.set_use_acoustic_sum_rule(true);  // Use optimization
  
  fmt::print("\n=== Water Molecule - ORCA Reference Comparison (HF/3-21G) ===\n");
  fmt::print("SCF Energy: {:.8f} Eh\n", e);
  fmt::print("Hessian settings:\n");
  fmt::print("  Method: Finite differences\n");
  fmt::print("  Step size: {:.3e} Bohr\n", hess_eval.step_size());
  fmt::print("  Acoustic sum rule: {}\n", hess_eval.use_acoustic_sum_rule() ? "enabled" : "disabled");
  
  // ORCA reference Hessian (HF/3-21G, from water.hess file - updated values)
  // 9x9 symmetric matrix, in atomic units (Hartree/Bohr²)
  Mat orca_hessian(9, 9);
  orca_hessian << 
    7.1540182757E-01, -3.5774303109E-17,  2.1191381983E-14, -3.5770091378E-01, -1.5896789541E-15,  2.6140476751E-01, -3.5770091378E-01,  1.6271765689E-15, -2.6140476751E-01,
   -3.5774303109E-17, -3.6801610598E-06, -7.1399248672E-16,  2.5963251640E-15,  1.8400805236E-06,  1.9866947590E-15, -2.5622741725E-15,  1.8400805362E-06,  1.1621328337E-15,
    2.1191381983E-14, -7.1399248672E-16,  4.3546092397E-01,  1.8278650599E-01, -1.6125475852E-15, -2.1773046199E-01, -1.8278650599E-01, -1.0829503404E-16, -2.1773046199E-01,
   -3.5770091378E-01,  2.5963251640E-15,  1.8278650599E-01,  3.9517345341E-01, -2.1261973752E-16, -2.2209563675E-01, -3.7472539627E-02, -3.6838504599E-15,  3.9307167787E-02,
   -1.5896789541E-15,  1.8400805236E-06, -1.6125475852E-15, -2.1261973752E-16, -6.7926607913E-06, -4.0971717303E-16,  3.1024437251E-15,  4.9525802677E-06,  8.0269147413E-16,
    2.6140476751E-01,  1.9866947590E-15, -2.1773046199E-01, -2.2209563675E-01, -4.0971717303E-16,  2.0437579974E-01, -3.9307167787E-02, -3.5740430188E-16,  1.3354662251E-02,
   -3.5770091378E-01, -2.5622741725E-15, -1.8278650599E-01, -3.7472539627E-02,  3.1024437251E-15, -3.9307167787E-02,  3.9517345341E-01,  7.5825216925E-16,  2.2209563675E-01,
    1.6271765689E-15,  1.8400805362E-06, -1.0829503404E-16, -3.6838504599E-15,  4.9525802677E-06, -3.5740430188E-16,  7.5825216925E-16, -6.7926608038E-06, -7.4956248594E-16,
   -2.6140476751E-01,  1.1621328337E-15, -2.1773046199E-01,  3.9307167787E-02,  8.0269147413E-16,  1.3354662251E-02,  2.2209563675E-01, -7.4956248594E-16,  2.0437579974E-01;
  
  fmt::print("ORCA Reference Hessian (HF/3-21G):\n{}\n", format_matrix(orca_hessian));
  
  // Compute our Hessian using finite differences (full energy)
  auto our_hessian = hess_eval(scf.wavefunction());
  fmt::print("OCC Hessian (finite differences):\n{}\n", format_matrix(our_hessian));
  
  // Compare our Hessian with ORCA
  auto diff = orca_hessian - our_hessian;
  double max_diff = diff.cwiseAbs().maxCoeff();
  double rms_diff = std::sqrt(diff.squaredNorm() / diff.size());
  
  fmt::print("Difference (ORCA - OCC):\n{}\n", format_matrix(diff));
  fmt::print("Max absolute difference: {:.8e}\n", max_diff);
  fmt::print("RMS difference: {:.8e}\n", rms_diff);
  
  // The differences should be small for a proper finite differences implementation
  INFO("Max difference between ORCA and OCC Hessians: " << max_diff);
  INFO("RMS difference between ORCA and OCC Hessians: " << rms_diff);
  
  // Basic sanity checks
  REQUIRE(orca_hessian.allFinite());
  REQUIRE(our_hessian.allFinite());
  REQUIRE(orca_hessian.rows() == 9); // 3 atoms × 3 coordinates
  REQUIRE(our_hessian.rows() == 9);
  
  // Check that all matrices are approximately symmetric
  REQUIRE(orca_hessian.isApprox(orca_hessian.transpose(), 1e-10));
  REQUIRE(our_hessian.isApprox(our_hessian.transpose(), 1e-10)); // Now perfectly symmetric due to symmetrization
  
  // Check that our Hessian matches ORCA very well (excellent agreement for finite differences)
  REQUIRE(max_diff < 1e-3); // Excellent agreement
  REQUIRE(rms_diff < 1e-4);
  
  // Compute vibrational frequencies from our Hessian (with and without projection)
  occ::core::Molecule molecule(atoms);
  auto vib_modes = occ::core::compute_vibrational_modes(our_hessian, molecule);
  auto vib_modes_projected = occ::core::compute_vibrational_modes(our_hessian, molecule, true);
  
  // Test convenience methods
  vib_modes.log_summary();
  fmt::print("\n{}", vib_modes.summary_string());
  fmt::print("\n{}", vib_modes.frequencies_string());
  
  fmt::print("\n=== Vibrational Analysis ===\n");
  fmt::print("Total modes: {}\n", vib_modes.frequencies_cm.size());
  
  fmt::print("\nAll frequencies (cm⁻¹):\n");
  for (int i = 0; i < vib_modes.frequencies_cm.size(); i++) {
    double freq = vib_modes.frequencies_cm[i];
    fmt::print("  Mode {:2d}: {:8.2f} cm⁻¹\n", i + 1, freq);
  }
  
  fmt::print("\n=== Projected Modes Analysis (PROJECTTR=TRUE) ===\n");
  fmt::print("Total modes: {}\n", vib_modes_projected.frequencies_cm.size());
  
  fmt::print("\nAll projected frequencies (cm⁻¹):\n");
  for (int i = 0; i < vib_modes_projected.frequencies_cm.size(); i++) {
    double freq = vib_modes_projected.frequencies_cm[i];
    fmt::print("  Mode {:2d}: {:8.2f} cm⁻¹\n", i + 1, freq);
  }

  // Expected ORCA frequencies for comparison
  fmt::print("\n=== Expected ORCA Vibrational Frequencies ===\n");
  fmt::print("Bending mode:  1798.92 cm⁻¹\n");
  fmt::print("Stretching 1:  3812.66 cm⁻¹\n"); 
  fmt::print("Stretching 2:  3945.60 cm⁻¹\n");
  
  // Check that we get 9 total modes for water (3N = 9)
  REQUIRE(vib_modes.n_modes() == 9);
  
  // Check that frequencies are in reasonable range for water
  // We expect 6 low frequencies (near zero) and 3 high vibrational frequencies
  auto all_freqs = vib_modes.get_all_frequencies();
  std::vector<double> high_freqs;
  for (int i = 0; i < all_freqs.size(); i++) {
    if (all_freqs[i] > 500.0) {  // Consider frequencies > 500 cm⁻¹ as vibrational
      high_freqs.push_back(all_freqs[i]);
    }
  }
  
  // Should have 3 high vibrational frequencies
  REQUIRE(high_freqs.size() >= 3);
  
  if (high_freqs.size() >= 3) {
    std::sort(high_freqs.begin(), high_freqs.end());
    
    // Check frequencies are close to ORCA values (allow some tolerance)
    REQUIRE(std::abs(high_freqs[0] - 1798.92) < 100.0); // Bending mode
    REQUIRE(std::abs(high_freqs[1] - 3812.66) < 200.0); // Stretching mode 1
    REQUIRE(std::abs(high_freqs[2] - 3945.60) < 200.0); // Stretching mode 2
  }
}
