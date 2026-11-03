#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/core/format_matrix.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/core/vibration.h>
#include <occ/gto/gto.h>
#include <occ/gto/shell.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hessians.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/scf.h>
#include <occ/qm/cphf.h>
#include <occ/dft/dft.h>
#include <occ/io/core_json.h>
#include <vector>
#include <algorithm>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "fd_test_utils.h"

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

// ---------------------------------------------------------------------------
// Analytic (CPHF/CPKS) Hessians
// ---------------------------------------------------------------------------
//
// Each analytic Hessian H is checked through H v against the central
// difference of the analytic gradient along one generic direction v (see
// fd_test_utils.h), and for translational invariance and symmetry.

namespace {

using occ::test::Atoms;
using occ::test::distorted_water;
using occ::test::generic_direction;
using occ::test::max_abs;
using occ::test::translational_sum_error;
using occ::qm::Wavefunction;

// The ORCA-optimised water of the HF/3-21G reference above (bohr).
Atoms orca_water() {
  return {{8, -0.000000000000, 0.000000000000, 0.120615570834},
          {1, 1.474863732488, 0.000000000000, -0.957206606042},
          {1, -1.474863732488, 0.000000000000, -0.957206606042}};
}

// HOI, distorted (bohr): iodine carries an ECP in def2 bases.
Atoms hypoiodous_acid() {
  return {{53, 0.03, -0.02, 0.01}, {8, 0.12, 0.05, 3.80}, {1, 1.80, 0.31, 4.25}};
}

// Open-shell test molecules (bohr, no symmetry) and their multiplicity.
struct OpenShell {
  Atoms atoms;
  int charge, multiplicity;
};

OpenShell open_shell(const std::string &name) {
  if (name == "NH2")
    return {{{7, 0.02, -0.01, 0.13}, {1, 1.62, 0.05, -0.88}, {1, -1.55, 0.31, -0.95}},
            0, 2};
  if (name == "CH2")
    return {{{6, 0.01, 0.02, 0.10}, {1, 1.84, -0.03, -0.62}, {1, -1.79, 0.12, -0.70}},
            0, 3};
  if (name == "H2O+")
    return {distorted_water(), 1, 2};
  throw std::runtime_error("unknown open-shell test molecule " + name);
}

occ::gto::AOBasis load_basis(const Atoms &atoms, const std::string &name,
                             bool pure = true) {
  auto basis = occ::gto::AOBasis::load(atoms, name);
  basis.set_pure(pure);
  return basis;
}

// "method[/df]": HF or a functional (coarse grid, unscreened), optionally
// with RI-J and RI-K.
template <typename Proc>
Proc make_proc(const occ::gto::AOBasis &basis, const std::string &method) {
  const auto slash = method.find("/df");
  Proc proc = [&]() {
    if constexpr (std::is_same_v<Proc, HartreeFock>)
      return HartreeFock(basis);
    else
      return Proc(method.substr(0, slash), basis, occ::test::coarse_grid());
  }();
  if (slash != std::string::npos)
    proc.set_density_fitting_basis("def2-universal-jkfit");
  if constexpr (!std::is_same_v<Proc, HartreeFock>)
    proc.set_xc_screening_threshold(0.0);
  return proc;
}

// SCF convergence for the reference and the displaced geometries: gradient
// errors of ~1e-9 over the 2e-3 step stay two orders below the Hessian
// tolerances.
const auto convergence = occ::test::scf_convergence(1e-11, 1e-9);

template <typename Proc>
Wavefunction converged_wavefunction(Proc &proc,
                                    SpinorbitalKind kind = SpinorbitalKind::Restricted,
                                    int charge = 0, int multiplicity = 1) {
  occ::qm::SCF<Proc> scf(proc, kind);
  scf.set_charge_multiplicity(charge, multiplicity);
  scf.convergence_settings = convergence;
  scf.compute_scf_energy();
  return scf.wavefunction();
}

// cphf_tolerance <= 0 keeps the solver's default.
template <typename Proc>
Mat analytic_hessian(Proc &proc, const Wavefunction &wfn,
                     double cphf_tolerance = 0.0) {
  HessianEvaluator<Proc> ev(proc);
  ev.set_method(HessianEvaluator<Proc>::Method::Analytical);
  if (cphf_tolerance > 0.0) {
    occ::qm::cphf::Settings s;
    s.tolerance = cphf_tolerance;
    ev.set_cphf_settings(s);
  }
  return ev(wfn);
}

// H v against the central difference of the analytic gradient along v
// (step 1e-3), translational invariance, and symmetry. `symmetric_part`: H is
// the symmetric part of a non-symmetric gradient Jacobian (COSX), compared
// through its projections on two directions instead.
template <typename Proc>
void require_matches_gradient(const std::string &label, Proc &proc,
                              const Wavefunction &wfn, const Mat &H, double tol,
                              double tsum_tol, bool symmetric_part = false) {
  double err;
  if (symmetric_part) {
    err = occ::test::symmetric_projection_error(proc, wfn, H, 1e-3, convergence);
  } else {
    const Vec v = generic_direction(proc.atoms());
    err = max_abs(H * v - occ::test::gradient_difference_along(proc, wfn, v, 1e-3,
                                                               convergence));
  }
  fmt::print("{}: max |analytic - FD| along {} = {:.2e}, translational sum "
             "{:.1e}\n",
             label, symmetric_part ? "v1, v2" : "v", err,
             translational_sum_error(H));
  REQUIRE(err < tol);
  REQUIRE(translational_sum_error(H) < tsum_tol);
  REQUIRE(occ::test::asymmetry(H) < 1e-10 * std::max(1.0, max_abs(H)));
}

// Runs `f(proc)` on HF or a functional per "method[/df]".
template <typename F>
void with_proc(const occ::gto::AOBasis &basis, const std::string &method, F &&f) {
  if (method == "hf" || method == "hf/df") {
    auto hf = make_proc<HartreeFock>(basis, method);
    f(hf);
  } else {
    auto dft = make_proc<occ::dft::DFT>(basis, method);
    f(dft);
  }
}

template <typename Proc>
void check_vs_gradient(Proc &proc, const Wavefunction &wfn,
                       const std::string &label) {
  REQUIRE(proc.analytic_hessian_unsupported(wfn.mo.kind).empty());
  require_matches_gradient(label, proc, wfn, analytic_hessian(proc, wfn), 1e-5,
                           1e-7);
}

} // namespace

TEST_CASE("Analytic RHF Hessian vs ORCA (water HF/3-21G)",
          "[hessian][analytic][orca]") {
  auto hf = HartreeFock(load_basis(orca_water(), "3-21g"));
  const Mat H = analytic_hessian(hf, converged_wavefunction(hf));
  // Elements of the ORCA reference above.
  Vec ref_diag(9);
  ref_diag << 7.1540182757E-01, 0.0, 4.3546092397E-01, 3.9517345341E-01, 0.0,
      2.0437579974E-01, 3.9517345341E-01, 0.0, 2.0437579974E-01;
  for (int i = 0; i < 9; i++)
    REQUIRE(std::abs(H(i, i) - ref_diag(i)) < 2e-5);
  REQUIRE(std::abs(H(0, 3) - -3.5770091378E-01) < 2e-5);
  REQUIRE(std::abs(H(2, 3) - 1.8278650599E-01) < 2e-5);
  REQUIRE(std::abs(H(3, 5) - -2.2209563675E-01) < 2e-5);
  REQUIRE(std::abs(H(3, 8) - 3.9307167787E-02) < 2e-5);
  REQUIRE(translational_sum_error(H) < 1e-8);
}

// The full RHF Hessian, ECP included; the building blocks below cover the
// Cartesian integrals.
TEST_CASE("Analytic RHF Hessian with ECPs vs finite differences (HOI)",
          "[hessian][analytic][finite_diff][ecp]") {
  HartreeFock hf(load_basis(hypoiodous_acid(), "def2-svp"));
  const auto wfn = converged_wavefunction(hf);
  require_matches_gradient("HOI def2-SVP", hf, wfn, analytic_hessian(hf, wfn),
                           1e-5, 1e-8);
}

// Building-block checks: each analytic second-derivative piece against the
// central difference, along v, of the corresponding first-derivative piece,
// with the AO density held fixed (basis functions move with their atoms).
// HOI brings the ECP into the core Hamiltonian derivatives, and d functions
// for spherical vs Cartesian.
TEST_CASE("Hessian building blocks vs finite differences",
          "[hessian][analytic][blocks]") {
  const bool pure = GENERATE(true, false);
  const Atoms atoms = hypoiodous_acid();
  auto make = [&](const Atoms &a) {
    return HartreeFock(load_basis(a, "def2-svp", pure));
  };
  HartreeFock hf = make(atoms);
  occ::qm::SCF<HartreeFock> scf(hf);
  scf.compute_scf_energy();
  const auto mo = scf.molecular_orbitals();
  const Eigen::Index nocc = mo.n_alpha;
  const Mat Co = mo.C.leftCols(nocc);
  const Mat P = 2.0 * Co * Co.transpose();
  const Mat Q = 2.0 * Co * mo.energies.head(nocc).asDiagonal() * Co.transpose();

  std::vector<Mat> S1, H1;
  hf.compute_one_electron_derivative_matrices(S1, H1);
  const auto G1 = hf.two_electron_derivative_fock(mo);
  const Mat H1e = hf.compute_one_electron_hessian(P, Q);
  const Mat H2e = hf.two_electron_hessian(mo);
  const Vec v = generic_direction(atoms);
  auto along = [&](const std::vector<Mat> &M) {
    Mat sum = Mat::Zero(M[0].rows(), M[0].cols());
    for (size_t k = 0; k < M.size(); k++)
      sum += v(k) * M[k];
    return sum;
  };

  auto g1e = [&](HartreeFock &h) {
    std::vector<Mat> s, hh;
    h.compute_one_electron_derivative_matrices(s, hh);
    Vec g(hh.size());
    for (size_t k = 0; k < hh.size(); k++)
      g(k) = P.cwiseProduct(hh[k]).sum() - Q.cwiseProduct(s[k]).sum();
    return g;
  };
  auto hcore = [](HartreeFock &h) {
    Mat H = h.compute_kinetic_matrix() + h.compute_nuclear_attraction_matrix();
    if (h.have_effective_core_potentials())
      H += h.compute_effective_core_potential_matrix();
    return H;
  };

  const double step = 1e-4;
  HartreeFock hp = make(occ::test::displaced_along(atoms, v, step));
  HartreeFock hm = make(occ::test::displaced_along(atoms, v, -step));
  auto d = [&](const auto &fp, const auto &fm) {
    return Mat((fp - fm) / (2 * step));
  };
  const double err_S = max_abs(
      d(hp.compute_overlap_matrix(), hm.compute_overlap_matrix()) - along(S1));
  const double err_H = max_abs(d(hcore(hp), hcore(hm)) - along(H1));
  const double err_G =
      max_abs(d(hp.compute_fock(mo), hm.compute_fock(mo)) - along(G1));
  const double err_1e = max_abs(d(g1e(hp), g1e(hm)) - H1e * v);
  const double err_2e =
      max_abs(d(occ::test::flat(hp.two_electron_gradient(mo)),
                occ::test::flat(hm.two_electron_gradient(mo))) -
              H2e * v);
  fmt::print("HOI {}: S1 {:.2e}  H1 {:.2e}  G1 {:.2e}  1e-skeleton {:.2e}  "
             "2e-skeleton {:.2e}\n",
             pure ? "spherical" : "Cartesian", err_S, err_H, err_G, err_1e,
             err_2e);
  CHECK(err_S < 1e-7);
  CHECK(err_H < 1e-6);
  CHECK(err_G < 1e-6);
  CHECK(err_1e < 1e-6);
  CHECK(err_2e < 1e-6);
}

// GGA, range-separated hybrid (global and long-range exchange), each with
// RI-J (and RI-K, "/df"), and a meta-GGA hybrid. The XC pieces of the other
// functionals are checked term by term in dft_gradient_tests.
TEST_CASE("Analytic DFT Hessian vs finite differences (coarse grid)",
          "[hessian][analytic][dft]") {
  const std::string method = GENERATE(as<std::string>{}, "pbe", "wb97x",
                                      "pbe/df", "wb97x/df", "hyb_mgga_xc_tpssh");
  auto dft = make_proc<occ::dft::DFT>(
      occ::gto::AOBasis::load(distorted_water(), "def2-svp"), method);
  check_vs_gradient(dft, converged_wavefunction(dft), method);
}

TEST_CASE("Analytic RI-JK HF Hessian vs finite differences",
          "[hessian][analytic][df]") {
  auto hf = make_proc<HartreeFock>(
      occ::gto::AOBasis::load(distorted_water(), "def2-svp"), "hf/df");
  check_vs_gradient(hf, converged_wavefunction(hf), "HF/RI-JK");
}

// occ's V_xc carries no Laplacian term, so its derivatives cannot either:
// such functionals fall back to finite differences.
TEST_CASE("Laplacian-dependent meta-GGAs are refused by the analytic Hessian",
          "[hessian][analytic][dft]") {
  auto basis = occ::gto::AOBasis::load(distorted_water(), "def2-svp");
  occ::dft::DFT tpss("tpss", basis), scanl("mgga_x_scanl", basis);
  REQUIRE(tpss.analytic_hessian_unsupported(SpinorbitalKind::Restricted)
              .empty());
  REQUIRE_FALSE(scanl.analytic_hessian_unsupported(SpinorbitalKind::Restricted)
                    .empty());
}

// The Hylleraas-corrected response term makes the Hessian's error quadratic
// in the CPHF residual, so a loose tolerance loses nothing.
TEST_CASE("Analytic Hessian error is quadratic in the CPHF residual",
          "[hessian][analytic][cphf]") {
  auto dft = make_proc<occ::dft::DFT>(
      occ::gto::AOBasis::load(distorted_water(), "def2-svp"), "pbe");
  const auto wfn = converged_wavefunction(dft);
  const Mat ref = analytic_hessian(dft, wfn, 1e-11);
  const double e5 = max_abs(analytic_hessian(dft, wfn, 1e-5) - ref);
  fmt::print("CPHF tolerance 1e-5: max |H - H(1e-11)| = {:.2e}\n", e5);
  REQUIRE(e5 < 1e-8);
}

// ---------------------------------------------------------------------------
// Unrestricted (UHF / UKS) Hessians
// ---------------------------------------------------------------------------

// Skeleton pieces at fixed density, each against central differences along
// v of the first-derivative quantity: the per-spin two-electron Fock
// derivatives (and their C_o contraction) against the Fock build, the
// two-electron skeleton Hessian against the two-electron gradient.
TEST_CASE("Unrestricted Hessian building blocks vs finite differences",
          "[hessian][analytic][blocks][uhf]") {
  const std::string method = GENERATE(as<std::string>{}, "hf", "hf/df");
  const auto mol = open_shell("H2O+");
  auto make = [&](const Atoms &a) {
    return make_proc<HartreeFock>(occ::gto::AOBasis::load(a, "def2-svp"),
                                  method);
  };
  HartreeFock hf = make(mol.atoms);
  const auto mo = converged_wavefunction(hf, SpinorbitalKind::Unrestricted,
                                         mol.charge, mol.multiplicity)
                      .mo;
  const auto G1 = hf.two_electron_derivative_fock(mo);
  const auto G1o = hf.two_electron_derivative_fock_occ(mo);
  const Mat H2e = hf.two_electron_hessian(mo);
  const Vec v = generic_direction(mol.atoms);
  Mat G1v = Mat::Zero(G1[0].rows(), G1[0].cols());
  Mat G1ov = Mat::Zero(G1o[0].rows(), G1o[0].cols());
  for (int k = 0; k < 9; k++) {
    G1v += v(k) * G1[k];
    G1ov += v(k) * G1o[k];
  }
  const double step = 1e-4;
  HartreeFock hp = make(occ::test::displaced_along(mol.atoms, v, step));
  HartreeFock hm = make(occ::test::displaced_along(mol.atoms, v, -step));
  const Mat dG = (hp.compute_fock(mo) - hm.compute_fock(mo)) / (2 * step);
  const double err_G = max_abs(dG - G1v);
  const double err_Go =
      max_abs(occ::qm::cphf::contract_occupied(dG, mo) - G1ov);
  const Vec d2 = (occ::test::flat(hp.two_electron_gradient(mo)) -
                  occ::test::flat(hm.two_electron_gradient(mo))) /
                 (2 * step);
  const double err_2e = max_abs(d2 - H2e * v);
  fmt::print("U{} H2O+: G1 {:.2e}  G1 C_o {:.2e}  2e-skeleton {:.2e}\n",
             method, err_G, err_Go, err_2e);
  CHECK(err_G < 1e-6);
  CHECK(err_Go < 1e-6);
  CHECK(err_2e < 1e-6);
}

// UHF, and UKS on the coarse grid, for open shells with a non-degenerate
// SOMO: OH's singly occupied pi orbital can turn about the bond at no cost,
// which leaves its orbital Hessian (near-)singular and the comparison
// ill-conditioned.
TEST_CASE("Analytic unrestricted Hessian vs finite differences",
          "[hessian][analytic][uhf]") {
  const auto [system, method] = GENERATE(table<std::string, std::string>(
      {{"NH2", "hf"}, {"CH2", "hf/df"}, {"NH2", "svwn5"}, {"CH2", "b3lyp"},
       {"NH2", "wb97x/df"}}));
  const auto mol = open_shell(system);
  with_proc(occ::gto::AOBasis::load(mol.atoms, "def2-svp"), method,
            [&](auto &proc) {
              const auto wfn = converged_wavefunction(
                  proc, SpinorbitalKind::Unrestricted, mol.charge,
                  mol.multiplicity);
              check_vs_gradient(proc, wfn, fmt::format("U{} {}", method, system));
            });
}

// A closed shell run unrestricted is the restricted calculation, term by
// term.
TEST_CASE("Closed-shell unrestricted Hessian equals the restricted one",
          "[hessian][analytic][uhf]") {
  const std::string method = GENERATE(as<std::string>{}, "hf", "b3lyp/df");
  with_proc(occ::gto::AOBasis::load(distorted_water(), "def2-svp"), method,
            [&](auto &proc) {
              const Mat Hr = analytic_hessian(
                  proc, converged_wavefunction(proc, SpinorbitalKind::Restricted));
              const Mat Hu = analytic_hessian(
                  proc, converged_wavefunction(proc, SpinorbitalKind::Unrestricted));
              const double err = max_abs(Hr - Hu);
              fmt::print("{}: max |H(unrestricted) - H(restricted)| = {:.2e}\n",
                         method, err);
              REQUIRE(err < 1e-7);
            });
}

// ---------------------------------------------------------------------------
// COSX exchange Hessians
// ---------------------------------------------------------------------------

namespace {
// COSX on GRIDX1, where the grid response is large; unscreened, so finite
// differences see a smooth surface.
template <typename Proc> void use_cosx_grid1(Proc &proc) {
  proc.set_cosx_exchange(occ::numint::COSXGridLevel::Grid1);
  occ::qm::cosx::Settings cs;
  cs.f_threshold = 1e-14;
  proc.set_cosx_settings(cs);
}
} // namespace

// Each COSX term at fixed AO density against central differences along v of
// the next-lower analytic quantity: the skeleton Hessian against the
// exchange gradient, the SCF's exchange Fock derivative against K, and the
// energy's density derivative against directional derivatives of E_x in the
// density.
TEST_CASE("COSX exchange Hessian terms vs finite differences",
          "[hessian][cosx][blocks]") {
  const Atoms atoms = distorted_water();
  const double omega = GENERATE(0.0, 0.3);
  auto mk = [&](const Atoms &a) {
    HartreeFock hf(occ::gto::AOBasis::load(a, "def2-svp"));
    use_cosx_grid1(hf);
    return hf;
  };
  auto hf = mk(atoms);
  occ::qm::SCF<HartreeFock> scf(hf);
  scf.compute_scf_energy();
  const auto mo = scf.molecular_orbitals();
  const Mat Co = mo.C.leftCols(mo.n_alpha);
  const HartreeFock::ExchangeOperators K1{{1.0, omega}};

  const Mat H = hf.compute_jk_hessian(mo, 0.0, K1);
  const auto F1 = hf.compute_jk_derivative_fock_occ(mo, 0.0, K1);
  const auto dF = hf.compute_exchange_nonvariational_fock_occ(mo, K1);
  REQUIRE(dF.size() == size_t(9));

  auto K = [&](HartreeFock &h, const Mat &D) {
    MolecularOrbitals m = mo;
    m.D = D;
    return omega == 0.0 ? h.compute_JK(m).K : h.compute_K_long_range(m, omega);
  };
  // E_x = -tr(D K[D]) is quadratic in D, so its directional derivative along
  // Co Y^T + Y Co^T is a central difference of any step.
  const Mat Y = Mat::NullaryExpr(Co.rows(), Co.cols(), [](Eigen::Index i,
                                                          Eigen::Index j) {
    return std::cos(0.7 * i + 1.3 * j);
  });
  const Mat dD = Co * Y.transpose() + Y * Co.transpose();
  auto energy_slope = [&](HartreeFock &h) {
    auto E = [&](const Mat &D) { return -D.cwiseProduct(K(h, D)).sum(); };
    return (E(mo.D + dD) - E(mo.D - dD)) / 2.0;
  };

  const Vec v = generic_direction(atoms);
  Mat F1v = Mat::Zero(F1[0].rows(), F1[0].cols());
  double slope_v = 0.0;
  for (int k = 0; k < 9; k++) {
    F1v += v(k) * F1[k];
    slope_v += v(k) * Y.cwiseProduct(F1[k] + dF[k]).sum();
  }
  const double step = 1e-4;
  auto hp = mk(occ::test::displaced_along(atoms, v, step));
  auto hm = mk(occ::test::displaced_along(atoms, v, -step));
  const Vec dg = (occ::test::flat(hp.compute_exchange_gradient(mo, 1.0, omega)) -
                  occ::test::flat(hm.compute_exchange_gradient(mo, 1.0, omega))) /
                 (2 * step);
  const double err_H = max_abs(dg - H * v);
  const Mat dK = (K(hp, mo.D) - K(hm, mo.D)) / (2 * step);
  const double err_F = max_abs(-dK * Co - F1v);
  // F1 - dF = -d/dR (-1/2 dE_x/dD) C_o, so <Y, .> = 1/4 d/dR slope
  const double ref = (energy_slope(hp) - energy_slope(hm)) / (2 * step) / 4;
  const double err_E = std::abs(slope_v - ref) / (1.0 + std::abs(ref));
  fmt::print("COSX (omega {}): skeleton {:.2e}  Fock derivative {:.2e}  "
             "energy density derivative {:.2e}\n",
             omega, err_H, err_F, err_E);
  CHECK(err_H < 1e-7);
  CHECK(err_F < 1e-7);
  CHECK(err_E < 1e-7);
  CHECK(translational_sum_error(H) < 1e-9);
}

// The full analytic Hessian of a COSX SCF against central differences of its
// analytic gradient (each displaced SCF also COSX). The COSX Fock is not the
// energy's density derivative, so this needs the response term formed with
// the latter (nonvariational_fock_derivative_occ), and the gradient's
// Jacobian is not symmetric: the analytic Hessian is its symmetric part.
TEST_CASE("Analytic COSX Hessian vs finite differences of the COSX gradient",
          "[hessian][analytic][cosx]") {
  const std::string method =
      GENERATE(as<std::string>{}, "hf", "b3lyp", "wb97x");
  with_proc(occ::gto::AOBasis::load(distorted_water(), "def2-svp"), method,
            [&](auto &proc) {
              use_cosx_grid1(proc);
              const auto wfn = converged_wavefunction(proc);
              require_matches_gradient(method + "/COSX", proc, wfn,
                                       analytic_hessian(proc, wfn, 1e-8), 5e-6,
                                       1e-7, true);
            });
}
