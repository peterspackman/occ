#include <fmt/core.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <occ/core/format_matrix.h>
#include <occ/core/util.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/scf.h>
#include <occ/qm/gradients.h>
#include <occ/qm/seminumerical_exchange.h>
#include <functional>
#include "fd_test_utils.h"

using occ::format_matrix;
using occ::Mat;
using occ::Mat3N;
using occ::qm::HartreeFock;
using occ::util::all_close;

// =============================================================================
// RHF Gradient Tests (migrated from qm_tests.cpp)
// =============================================================================

TEST_CASE("RHF integral gradients", "[rhf][gradient][integrals]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};

  auto obs = occ::gto::AOBasis::load(atoms, "STO-3G");

  occ::Vec e(obs.nbf()), occ(obs.nbf());
  e << -20.2434, -1.2673, -0.6143, -0.4545, -0.3916, 0.6029, 0.7350;
  occ::Mat D(obs.nbf(), obs.nbf()), C(obs.nbf(), obs.nbf());
  D << 2.106529, -0.447611, 0.057951, 0.091761, -0.002396, -0.027622, -0.027249,
      -0.447611, 1.974382, -0.328030, -0.521593, 0.013615, -0.038544, -0.037120,
      0.057951, -0.328030, 0.877559, 0.221255, -0.002740, -0.203698, 0.711984,
      0.091761, -0.521593, 0.221255, 1.089979, 0.021845, 0.689851, 0.111189,
      -0.002396, 0.013615, -0.002740, 0.021845, 1.999469, -0.016473, -0.004447,
      -0.027622, -0.038544, -0.203698, 0.689851, -0.016473, 0.603384, -0.189923,
      -0.027249, -0.037120, 0.711984, 0.111189, -0.004447, -0.189923, 0.606432;

  C << 0.99414, -0.23288, -0.00108, 0.10350, 0.00000, -0.13135, 0.00403,
      0.02646, 0.83484, 0.00590, -0.53804, -0.00000, 0.87498, -0.02968, 0.00228,
      0.06746, 0.51167, 0.41522, 0.00241, 0.42062, 0.82123, 0.00368, 0.10998,
      -0.32748, 0.65195, 0.02458, 0.61693, -0.54485, -0.00010, -0.00287,
      0.00682, -0.01703, 0.99969, -0.01618, 0.01142, -0.00596, 0.15957,
      -0.44585, 0.27823, 0.00000, -0.77288, 0.85930, -0.00587, 0.15700, 0.44567,
      0.28269, -0.00000, -0.81270, -0.81126;

  occ << 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0;

  occ::qm::MolecularOrbitals mo;
  mo.kind = occ::qm::SpinorbitalKind::Restricted;
  mo.D = D * 0.5;
  mo.energies = e;
  mo.occupation = occ;
  mo.C = C;

  occ::qm::IntegralEngine engine(obs);
  HartreeFock hf(obs);
  auto [J, K] = hf.compute_JK(mo);
  auto [grad, grad_k] = hf.compute_JK_gradient(mo);

  occ::Mat Xref(obs.nbf(), obs.nbf()), Yref(obs.nbf(), obs.nbf()),
      Zref(obs.nbf(), obs.nbf());

  Xref << -0.03898, -0.01702, 13.27512, 0.01153, -0.00015, -0.27938, 0.80778,
      0.00038, 0.02486, 4.61696, -0.00033, -0.00000, -0.67535, 2.04809,
      -15.50871, -7.76594, 0.06736, 0.00329, -0.00019, -2.04167, 0.14166,
      0.00312, 0.00478, 0.02816, -0.00443, 0.00035, -0.76461, 0.25917, -0.00004,
      -0.00007, -0.00084, 0.00035, 0.01086, 0.01816, -0.01219, 0.31666, 0.94311,
      1.04257, 0.92737, -0.02203, 0.31173, 0.91043, -0.92953, -2.75624,
      -1.52403, -0.30732, 0.01446, -1.08941, -0.86829;

  Yref << -0.06087, -0.02660, 0.01153, 13.28648, 0.00109, 0.77487, 0.07826,
      0.00060, 0.03938, -0.00033, 4.61698, -0.00008, 1.96012, 0.22767, 0.00312,
      0.00478, 0.02816, -0.00443, 0.00035, -0.76461, 0.25917, -15.50563,
      -7.76067, -0.01976, 0.07139, -0.00086, -0.11750, -2.24876, 0.00029,
      0.00037, 0.00035, -0.00150, 0.01734, -0.05199, -0.00127, -0.89387,
      -2.62418, 0.90500, -1.22562, 0.06155, -0.81827, -0.76975, -0.09524,
      -0.26983, -0.28513, 1.31597, 0.00142, 0.48830, -0.06852;

  Zref << 0.00159, 0.00069, -0.00015, 0.00109, 13.33186, -0.01838, -0.00387,
      -0.00002, -0.00103, -0.00000, -0.00008, 4.61356, -0.04657, -0.01053,
      -0.00004, -0.00007, -0.00084, 0.00035, 0.01086, 0.01816, -0.01219,
      0.00029, 0.00037, 0.00035, -0.00150, 0.01734, -0.05199, -0.00127,
      -15.49334, -7.74518, -0.00447, -0.00753, -0.00071, -2.30545, -2.27486,
      0.02122, 0.06226, -0.02148, 0.06149, 1.36441, 0.01937, 0.01674, 0.00458,
      0.01327, 0.01392, 0.00148, 1.34587, -0.00939, 0.00377;

  REQUIRE(all_close(grad.x, Xref, 1e-5, 1e-5));
  REQUIRE(all_close(grad.y, Yref, 1e-5, 1e-5));
  REQUIRE(all_close(grad.z, Zref, 1e-5, 1e-5));

  Xref << 0.02578, -0.06436, 1.28249, 0.09634, -0.00124, -0.09719, 0.30588,
      0.00034, -0.00791, 0.57682, 0.01804, -0.00023, -0.10010, 0.31664,
      -4.04562, -1.38240, 0.15905, 0.07964, -0.00240, -0.45088, -0.20785,
      0.00565, 0.00970, 0.01519, -0.05754, 0.00105, -0.08108, 0.03159, -0.00007,
      -0.00013, -0.00071, 0.00106, -0.01162, 0.00198, -0.00137, 0.08256,
      0.14178, 0.13574, 0.09923, -0.00230, 0.07807, 0.14131, -0.24018, -0.42145,
      -0.12875, -0.02855, 0.00147, -0.18523, -0.22229;
  Yref << 0.04098, -0.10140, 0.09628, 1.37646, 0.00895, 0.29431, 0.03805,
      0.00054, -0.01202, 0.01806, 0.59557, 0.00164, 0.30414, 0.03958, 0.00565,
      0.00970, 0.01519, -0.05754, 0.00105, -0.08108, 0.03159, -4.04016,
      -1.37206, -0.01624, 0.14585, -0.00444, -0.23608, -0.46844, 0.00054,
      0.00083, 0.00110, -0.00268, -0.01815, -0.00526, 0.00032, -0.23054,
      -0.40240, 0.09643, -0.09421, 0.00684, -0.21117, -0.13764, -0.02378,
      -0.04384, -0.02571, 0.16718, 0.00064, 0.06778, -0.01980;
  Zref << -0.00107, 0.00265, -0.00124, 0.00895, 1.74979, -0.00700, -0.00167,
      -0.00001, 0.00031, -0.00023, 0.00164, 0.66388, -0.00724, -0.00174,
      -0.00007, -0.00013, -0.00071, 0.00106, -0.01162, 0.00198, -0.00137,
      0.00054, 0.00083, 0.00110, -0.00268, -0.01815, -0.00526, 0.00032,
      -4.01780, -1.33735, 0.03156, 0.04934, -0.00034, -0.45775, -0.45243,
      0.00547, 0.00955, -0.00223, 0.00683, 0.19326, 0.00500, 0.00304, 0.00116,
      0.00209, 0.00140, 0.00064, 0.19059, -0.00122, 0.00102;

  REQUIRE(all_close(grad_k.x, Xref, 1e-5, 1e-5));
  REQUIRE(all_close(grad_k.y, Yref, 1e-5, 1e-5));
  REQUIRE(all_close(grad_k.z, Zref, 1e-5, 1e-5));

  occ::Mat3N nuc_expected(3, 3);
  nuc_expected << 1.57939, 0.91879, -2.49818, 2.54459, -2.36485, -0.17974,
      -0.06637, 0.05594, 0.01043;

  auto g = occ::qm::GradientEvaluator(hf);
  REQUIRE(all_close(g.nuclear_repulsion(), nuc_expected, 1e-5, 1e-5));

  occ::Mat3N elec_expected(3, 3);
  elec_expected << -1.55750, -0.91352, 2.47102, -2.49540, 2.32632, 0.16908,
      0.06511, -0.05501, -0.01010;
  REQUIRE(all_close(g.electronic(mo), elec_expected, 1e-4, 1e-4));

  occ::Mat3N expected_atom_gradients(3, 3);
  expected_atom_gradients << 0.02189, 0.00527, -0.02716, 0.04919, -0.03853,
      -0.01067, -0.00126, 0.00093, 0.00033;

  auto atom_gradients = g(mo);

  fmt::print("Atom gradients\n{}\n", format_matrix(atom_gradients));
  fmt::print("Difference\n{}\n",
             format_matrix(atom_gradients - expected_atom_gradients));
  REQUIRE(all_close(atom_gradients, expected_atom_gradients, 1e-4, 1e-4));
}

// =============================================================================
// UHF Gradient Tests (NEW)
// =============================================================================

TEST_CASE("UHF gradient water closed-shell STO-3G", "[uhf][gradient]") {
  // Water molecule (closed-shell, multiplicity=1)
  // This should match RHF gradients since it's a closed-shell system
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};

  auto basis = occ::gto::AOBasis::load(atoms, "STO-3G");
  occ::qm::HartreeFock hf(basis);

  // Run UHF SCF
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf, occ::qm::SpinorbitalKind::Unrestricted);
  scf.set_charge_multiplicity(0, 1);  // neutral, singlet
  double energy = scf.compute_scf_energy();
  const auto& mo = scf.molecular_orbitals();

  // Compute gradient
  occ::qm::GradientEvaluator evaluator(hf);
  auto gradient = evaluator(mo);

  // Reference from PySCF 2.7.0 UHF/STO-3G
  // Water UHF closed-shell STO-3G
  // Format: gradient(coord, atom) where coord=0,1,2 for x,y,z
  occ::Mat3N expected(3, 3);
  expected <<     0.021890305681,     0.005271094299,    -0.027161399980,
          0.049191974977,    -0.038526219510,    -0.010665755466,
         -0.001262304502,     0.000934684107,     0.000327620395;

  fmt::print("UHF water closed-shell gradient test:\n");
  fmt::print("Expected:\n{}\n", format_matrix(expected));
  fmt::print("Found:\n{}\n", format_matrix(gradient));
  fmt::print("Diff:\n{}\n", format_matrix(gradient - expected));

  REQUIRE(all_close(gradient, expected, 1e-5, 1e-5));
}

TEST_CASE("UHF gradient OH radical STO-3G", "[uhf][gradient]") {
  // OH radical (doublet, multiplicity=2)
  std::vector<occ::core::Atom> atoms{
      {8, 0.0, 0.0, 0.0},      // O
      {1, 0.0, 0.0, 1.8324}    // H (O-H distance ~0.97 Å)
  };

  auto basis = occ::gto::AOBasis::load(atoms, "STO-3G");
  occ::qm::HartreeFock hf(basis);

  // Run UHF SCF
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf, occ::qm::SpinorbitalKind::Unrestricted);
  scf.set_charge_multiplicity(0, 2);  // neutral, doublet
  double energy = scf.compute_scf_energy();
  const auto& mo = scf.molecular_orbitals();

  // Compute gradient
  occ::qm::GradientEvaluator evaluator(hf);
  auto gradient = evaluator(mo);

  // Reference from PySCF 2.7.0 UHF/STO-3G
  // OH radical UHF STO-3G doublet
  // Format: gradient(coord, atom) where coord=0,1,2 for x,y,z
  occ::Mat3N expected(3, 2);
  expected <<     0.000000000000,    -0.000000000000,     0.000000000000,
        -0.000000000000,     0.056139309345,    -0.056139309345;

  fmt::print("UHF OH radical gradient test:\n");
  fmt::print("Expected:\n{}\n", format_matrix(expected));
  fmt::print("Found:\n{}\n", format_matrix(gradient));
  fmt::print("Diff:\n{}\n", format_matrix(gradient - expected));

  REQUIRE(all_close(gradient, expected, 1e-5, 1e-5));
}

TEST_CASE("UHF gradient CH3 radical 3-21G", "[uhf][gradient]") {
  // CH3 radical (doublet, multiplicity=2)
  std::vector<occ::core::Atom> atoms{
      {6,  0.0000000000,  0.0000000000,  0.0000000000},  // C
      {1,  0.0000000000,  1.0814764143,  0.0000000000},  // H
      {1,  0.9365524973, -0.5407382072,  0.0000000000},  // H
      {1, -0.9365524973, -0.5407382072,  0.0000000000}   // H
  };

  auto basis = occ::gto::AOBasis::load(atoms, "3-21G");
  occ::qm::HartreeFock hf(basis);

  // Run UHF SCF
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf, occ::qm::SpinorbitalKind::Unrestricted);
  scf.set_charge_multiplicity(0, 2);  // neutral, doublet
  double energy = scf.compute_scf_energy();
  const auto& mo = scf.molecular_orbitals();

  // Compute gradient
  occ::qm::GradientEvaluator evaluator(hf);
  auto gradient = evaluator(mo);

  // Reference from PySCF 2.7.0 UHF/3-21G
  // CH3 radical UHF 3-21G doublet
  // Format: gradient(coord, atom) where coord=0,1,2 for x,y,z
  occ::Mat3N expected(3, 4);
  expected <<     0.000000000000,     0.000000000000,    -1.799187323786,
        1.799187323786,    -0.000244878398,    -2.077335928382,
        1.038790403390,     1.038790403390,     0.000000000000,
       -0.000000000000,    -0.000000000000,     0.000000000000;

  fmt::print("UHF CH3 radical gradient test:\n");
  fmt::print("Expected:\n{}\n", format_matrix(expected));
  fmt::print("Found:\n{}\n", format_matrix(gradient));
  fmt::print("Diff:\n{}\n", format_matrix(gradient - expected));

  REQUIRE(all_close(gradient, expected, 1e-5, 1e-5));
}

// =============================================================================
// GHF Gradient Tests
// =============================================================================

TEST_CASE("GHF gradient of a collinear radical matches UHF",
          "[ghf][gradient]") {
  // NH2: GHF converges to the UHF state (non-degenerate singly occupied
  // orbital), so the two gradients must agree. Turning the spin axis of the
  // converged spinors gives another GHF solution with the same gradient, and
  // that one has filled, non-symmetric ab blocks.
  std::vector<occ::core::Atom> atoms{{7, 0.0, 0.0, 0.270610},
                                     {1, 0.0, 1.516320, -0.946941},
                                     {1, 0.1, -1.516320, -0.946941}};
  auto basis = occ::gto::AOBasis::load(atoms, "6-31G");

  auto converged = [&](occ::qm::SpinorbitalKind kind, HartreeFock &hf) {
    occ::qm::SCF<HartreeFock> scf(hf, kind);
    scf.set_charge_multiplicity(0, 2);
    scf.convergence_settings.energy_threshold = 1e-11;
    scf.convergence_settings.commutator_threshold = 1e-9;
    scf.compute_scf_energy();
    return scf.molecular_orbitals();
  };

  HartreeFock uhf(basis), ghf(basis);
  const auto mo_u = converged(occ::qm::SpinorbitalKind::Unrestricted, uhf);
  const auto mo_g = converged(occ::qm::SpinorbitalKind::General, ghf);
  const Mat3N g_u = occ::qm::GradientEvaluator(uhf)(mo_u);
  const Mat3N g_g = occ::qm::GradientEvaluator(ghf)(mo_g);
  fmt::print("UHF gradient\n{}\nGHF gradient\n{}\n", format_matrix(g_u),
             format_matrix(g_g));
  REQUIRE(all_close(g_g, g_u, 1e-6, 1e-6));

  auto rotated = mo_g;
  const size_t nbf = basis.nbf();
  const double c = std::cos(0.9), s = std::sin(0.9);
  rotated.C.topRows(nbf) = c * mo_g.C.topRows(nbf) - s * mo_g.C.bottomRows(nbf);
  rotated.C.bottomRows(nbf) =
      s * mo_g.C.topRows(nbf) + c * mo_g.C.bottomRows(nbf);
  rotated.update_occupied_orbitals();
  rotated.update_density_matrix();
  REQUIRE(occ::qm::block::ab(rotated.D).cwiseAbs().maxCoeff() > 1e-2);
  const Mat3N g_rot = occ::qm::GradientEvaluator(ghf)(rotated);
  fmt::print("GHF gradient, spin axis turned\n{}\n", format_matrix(g_rot));
  REQUIRE(all_close(g_rot, g_u, 1e-6, 1e-6));
}


// =============================================================================
// ECP, COSX and density-fitted gradients
// =============================================================================

namespace {

using occ::qm::MolecularOrbitals;
using occ::qm::SpinorbitalKind;
using occ::test::Atoms;
using occ::test::distorted_water;
using occ::test::directional_difference;
using occ::test::generic_direction;
using occ::test::max_abs;
using occ::Vec;

struct Spin {
  SpinorbitalKind kind{SpinorbitalKind::Restricted};
  int charge{0}, multiplicity{1};
  const char *label() const {
    return kind == SpinorbitalKind::Restricted ? "RHF" : "UHF";
  }
};
// The water cation: a non-degenerate doublet, so the state cannot switch
// between displaced geometries.
constexpr Spin closed_shell{};
constexpr Spin water_cation{SpinorbitalKind::Unrestricted, 1, 2};

using Setup = std::function<void(HartreeFock &)>;
using Converged = std::function<void(HartreeFock &, const MolecularOrbitals &)>;

// Tightly converged HF energy on `atoms`, `setup` choosing the J/K
// approximation; `done` sees the converged procedure and orbitals.
double hf_energy(const Atoms &atoms, const std::string &basis_name, Spin spin,
                 const Setup &setup, const Converged &done = {}) {
  HartreeFock hf(occ::gto::AOBasis::load(atoms, basis_name));
  setup(hf);
  occ::qm::SCF<HartreeFock> scf(hf, spin.kind);
  scf.set_charge_multiplicity(spin.charge, spin.multiplicity);
  scf.convergence_settings.energy_threshold = 1e-12;
  scf.convergence_settings.commutator_threshold = 1e-10;
  const double e = scf.compute_scf_energy();
  if (done)
    done(hf, scf.molecular_orbitals());
  return e;
}

// Orbitals at default convergence: for tests that hold the density fixed.
MolecularOrbitals scf_orbitals(HartreeFock &hf, Spin spin = closed_shell) {
  occ::qm::SCF<HartreeFock> scf(hf, spin.kind);
  scf.set_charge_multiplicity(spin.charge, spin.multiplicity);
  scf.compute_scf_energy();
  return scf.molecular_orbitals();
}

// COSX exchange on a fixed grid level, unscreened so that finite differences
// see a smooth surface.
Setup cosx(occ::numint::COSXGridLevel level) {
  return [level](HartreeFock &hf) {
    hf.set_cosx_exchange(level);
    occ::qm::cosx::Settings cs;
    cs.f_threshold = 1e-14;
    hf.set_cosx_settings(cs);
  };
}

occ::gto::AOBasis load_basis(const Atoms &atoms, const char *name, bool pure) {
  auto b = occ::gto::AOBasis::load(atoms, name);
  b.set_pure(pure);
  return b;
}

// def2-SVP with the def2 universal JK fitting basis.
occ::qm::IntegralEngineDF df_engine(const Atoms &atoms, bool pure,
                                    double omega = 0.0) {
  occ::qm::IntegralEngineDF df(
      atoms, load_basis(atoms, "def2-svp", pure).shells(),
      load_basis(atoms, "def2-universal-jkfit", pure).shells());
  df.set_range_separated_omega(omega);
  return df;
}

// The SCF density alone (RI-J builds from D, not the orbitals).
MolecularOrbitals density_only(const MolecularOrbitals &mo) {
  MolecularOrbitals m;
  m.kind = mo.kind;
  m.D = mo.D;
  return m;
}

const char *basis_kind(bool pure) { return pure ? "spherical" : "Cartesian"; }

// |g . v - fd|: an analytic gradient against the central difference of its
// energy along the unit direction v.
double error_along(const Mat3N &g, const occ::Vec &v, double fd) {
  return std::abs(occ::test::flat(g).dot(v) - fd);
}

// sum_k v_k M_k: per-coordinate derivative matrices along v.
Mat along(const std::vector<Mat> &M, const occ::Vec &v) {
  Mat sum = Mat::Zero(M[0].rows(), M[0].cols());
  for (size_t k = 0; k < M.size(); k++)
    sum += v(k) * M[k];
  return sum;
}

} // namespace

// Effective core potentials: the ECP derivative term, the effective (not bare)
// charge in the Hellmann-Feynman term, and both spin blocks for open shells.
TEST_CASE("ECP gradient vs finite differences (HI def2-SVP)",
          "[gradient][ecp]") {
  const Spin spin{GENERATE(SpinorbitalKind::Restricted,
                           SpinorbitalKind::Unrestricted)};
  // bohr, off-axis so every component is exercised
  const Atoms atoms{{53, 0.05, -0.02, 0.0}, {1, 0.31, 0.22, 3.02}};
  REQUIRE(occ::gto::AOBasis::load(atoms, "def2-svp").have_ecps());
  const Setup none = [](HartreeFock &) {};
  Mat3N g;
  hf_energy(atoms, "def2-svp", spin, none,
            [&](HartreeFock &hf, const MolecularOrbitals &mo) {
              g = occ::qm::GradientEvaluator<HartreeFock>(hf)(mo);
            });
  const Vec v = generic_direction(atoms);
  const double fd = directional_difference(atoms, v, 1e-3, [&](const Atoms &a) {
    return hf_energy(a, "def2-svp", spin, none);
  });
  const double gv = occ::test::flat(g).dot(v);
  fmt::print("ECP gradient ({}): |g.v - FD| = {:.2e}\n", spin.label(),
             std::abs(gv - fd));
  REQUIRE(occ::util::is_close(gv, fd, 1e-6, 1e-6));
}

// The COSX exchange gradient, at fixed AO density (basis functions and grid
// move with the atoms): every term (basis functions, ESP integrals, grid
// points, partition weights, overlap fitting) against central differences.
TEST_CASE("COSX exchange gradient at fixed density vs finite differences",
          "[gradient][cosx]") {
  const Atoms atoms = distorted_water();
  const double omega = GENERATE(0.0, 0.3);
  const Spin spin = GENERATE(closed_shell, water_cation);
  const Setup grid1 = cosx(occ::numint::COSXGridLevel::Grid1);
  auto mk = [&](const Atoms &a) {
    HartreeFock hf(occ::gto::AOBasis::load(a, "def2-svp"));
    grid1(hf);
    return hf;
  };
  auto hf = mk(atoms);
  const auto mo = scf_orbitals(hf, spin);
  // -tr(D K) is the exchange energy for both kinds (occ's unrestricted
  // density blocks are half the spin densities).
  const Mat3N g = hf.compute_exchange_gradient(mo, 1.0, omega);
  const Vec v = generic_direction(atoms);
  const double fd = directional_difference(atoms, v, 1e-4, [&](const Atoms &a) {
    auto h = mk(a);
    const Mat K =
        omega == 0.0 ? h.compute_JK(mo).K : h.compute_K_long_range(mo, omega);
    return -mo.D.cwiseProduct(K).sum();
  });
  const double err = error_along(g, v, fd);
  fmt::print("COSX fixed-density {} exchange gradient (omega {}): |analytic "
             "- FD| = {:.2e}\n",
             spin.label(), omega, err);
  REQUIRE(err < 1e-7);
  REQUIRE(g.rowwise().sum().norm() < 1e-10);
}

// Full SCF gradient with COSX against finite differences of the COSX energy.
// The COSX Fock matrix (K symmetrised with the overlap-fitting projector on
// one side) is not exactly dE_x/dD, so the COSX SCF is not variational and a
// residual of the grid-error size remains; it must still be far below the
// error of using the exact-integral gradient on the COSX surface.
TEST_CASE("COSX SCF gradient vs finite differences of the COSX energy",
          "[gradient][cosx]") {
  using occ::numint::COSXGridLevel;
  // GRIDX1, GRIDX3 (open shell), and the gradient schedule (GRIDX1 ->
  // GRIDX3, restricted). The energy schedule is left out: its GRIDX3 energy
  // at GRIDX2 orbitals is not stationary, see COSXGridSchedule.
  struct Case {
    const char *label;
    Setup setup;
    Spin spin;
  };
  const auto c = GENERATE_COPY(
      Case{"GRIDX1", cosx(COSXGridLevel::Grid1), closed_shell},
      Case{"GRIDX3", cosx(COSXGridLevel::Grid3), water_cation},
      Case{"gradient schedule",
           [](HartreeFock &hf) {
             hf.set_cosx_grid_schedule(
                 occ::qm::COSXGridSchedule::for_gradients());
             occ::qm::cosx::Settings cs;
             cs.f_threshold = 1e-14;
             hf.set_cosx_settings(cs);
           },
           closed_shell});
  const Atoms atoms = distorted_water();
  Mat3N g, g_exact;
  hf_energy(atoms, "def2-svp", c.spin, c.setup,
            [&](HartreeFock &hf, const MolecularOrbitals &mo) {
              g = occ::qm::GradientEvaluator<HartreeFock>(hf)(mo);
              g_exact = g - hf.compute_exchange_gradient(mo, 1.0) +
                        hf.compute_jk_gradient(mo, 0.0, 1.0);
            });
  const Vec v = generic_direction(atoms);
  const double fd = directional_difference(atoms, v, 1e-3, [&](const Atoms &a) {
    return hf_energy(a, "def2-svp", c.spin, c.setup);
  });
  const double err = error_along(g, v, fd),
               err_exact = error_along(g_exact, v, fd);
  fmt::print("COSX {} {}: |COSX gradient - FD| = {:.2e} (exact-integral "
             "gradient {:.2e})\n",
             c.spin.label(), c.label, err, err_exact);
  REQUIRE(err < 0.25 * err_exact);
}

// ORCA's COSX schedule (GRIDX1 -> GRIDX2, final energy on GRIDX3) lands on
// the GRIDX3 energy without iterating on GRIDX3.
TEST_CASE("COSX grid schedule matches a fixed GRIDX3 SCF", "[cosx]") {
  auto basis = occ::gto::AOBasis::load(distorted_water(), "def2-svp");
  auto run = [&](bool schedule) {
    HartreeFock hf(basis);
    if (schedule)
      hf.set_cosx_grid_schedule();
    else
      hf.set_cosx_exchange(occ::numint::COSXGridLevel::Grid3);
    occ::qm::SCF<HartreeFock> scf(hf);
    scf.convergence_settings.energy_threshold = 1e-10;
    scf.convergence_settings.commutator_threshold = 1e-8;
    return scf.compute_scf_energy();
  };
  const double e_sched = run(true), e_fixed = run(false);
  fmt::print("COSX schedule {:.10f}  fixed GRIDX3 {:.10f}  diff {:.2e}\n",
             e_sched, e_fixed, e_sched - e_fixed);
  REQUIRE(std::abs(e_sched - e_fixed) < 5e-6);
}

// tr(D J_DF[D]) at a fixed AO density: its derivative is the RI-J skeleton
// gradient, in the convention of compute_jk_gradient(mo, 1, 0).
TEST_CASE("RI-J Coulomb gradient at fixed density vs finite differences",
          "[gradient][df]") {
  const Atoms atoms = distorted_water();
  const bool pure = GENERATE(true, false);
  HartreeFock hf(load_basis(atoms, "def2-svp", pure));
  const auto mo = scf_orbitals(hf);
  const auto m = density_only(mo);
  const Mat3N g = df_engine(atoms, pure).coulomb_gradient(mo);
  const Vec v = generic_direction(atoms);
  const double fd = directional_difference(atoms, v, 1e-4, [&](const Atoms &a) {
    return mo.D.cwiseProduct(df_engine(a, pure).coulomb(m)).sum();
  });
  fmt::print("RI-J gradient ({}): |analytic - FD| = {:.2e}, |RI-J - exact J| "
             "= {:.2e}\n",
             basis_kind(pure), error_along(g, v, fd),
             max_abs(g - hf.compute_jk_gradient(mo, 1.0, 0.0)));
  REQUIRE(error_along(g, v, fd) < 1e-7);
  REQUIRE(g.rowwise().sum().norm() < 1e-10);
}

// RI-J second-derivative pieces at fixed AO density, against central
// differences of the RI-J gradient and of the RI-J Coulomb matrix, and the
// batched response against the plain RI-J build.
TEST_CASE("RI-J Hessian, derivative Fock and response at fixed density",
          "[hessian][df]") {
  const Atoms atoms = distorted_water();
  const bool pure = GENERATE(true, false);
  HartreeFock hf(load_basis(atoms, "def2-svp", pure));
  const auto mo = scf_orbitals(hf);
  const auto m = density_only(mo);

  auto df = df_engine(atoms, pure);
  const Mat H = df.coulomb_hessian(mo);
  const auto F1 = df.coulomb_derivative_fock(mo);
  const double h = 1e-4;
  const Vec v = generic_direction(atoms);
  auto dp = df_engine(occ::test::displaced_along(atoms, v, h), pure);
  auto dm = df_engine(occ::test::displaced_along(atoms, v, -h), pure);
  const Vec Hv_fd = occ::test::flat(
      (dp.coulomb_gradient(mo) - dm.coulomb_gradient(mo)) / (2 * h));
  const Mat dJ = (dp.coulomb(m) - dm.coulomb(m)) / (2 * h);
  const double fock_err = max_abs(along(F1, v) - dJ);
  // Response: the RI-J build on each density, here two at once. Symmetric,
  // as CPHF densities are (and as the batched build assumes).
  const Mat R0 = Mat::Random(mo.D.rows(), mo.D.cols());
  auto mr = m;
  mr.D = 0.5 * (R0 + R0.transpose());
  const auto J1 = df.coulomb_response({mo.D, mr.D});
  // Relative: the two builds sum the same terms in different orders.
  const Mat J0 = df.coulomb(m), Jr = df.coulomb(mr);
  const double resp_err = std::max(max_abs(J1[0] - J0) / max_abs(J0),
                                   max_abs(J1[1] - Jr) / max_abs(Jr));
  fmt::print("RI-J ({}): |H v - FD| = {:.2e}, |dJ/dv - FD| = {:.2e}, "
             "|response - J build| / |J| = {:.2e}\n",
             basis_kind(pure), max_abs(H * v - Hv_fd), fock_err, resp_err);
  REQUIRE(max_abs(H * v - Hv_fd) < 1e-6);
  REQUIRE(fock_err < 1e-6);
  REQUIRE(resp_err < 1e-12);
}

// RI-K exchange at fixed occupied coefficients: tr(D K_DF) against central
// differences, with the four-centre exchange gradient for scale.
TEST_CASE("RI-K exchange gradient at fixed density vs finite differences",
          "[gradient][df][rik]") {
  const Atoms atoms = distorted_water();
  const bool pure = GENERATE(true, false);
  const double omega = GENERATE(0.0, 0.3);
  HartreeFock hf(load_basis(atoms, "def2-svp", pure));
  const auto mo = scf_orbitals(hf);
  const Mat3N g = df_engine(atoms, pure, omega).exchange_gradient(mo);
  const Vec v = generic_direction(atoms);
  const double fd = directional_difference(atoms, v, 1e-4, [&](const Atoms &a) {
    return mo.D.cwiseProduct(df_engine(a, pure, omega).exchange(mo)).sum();
  });
  // compute_jk_gradient differentiates -tr(D K), the exchange energy as it
  // enters the total.
  const Mat3N g_exact = omega == 0.0
                            ? hf.compute_jk_gradient(mo, 0.0, 1.0)
                            : hf.compute_exchange_gradient(mo, 1.0, omega);
  fmt::print("RI-K gradient ({}, omega {}): |analytic - FD| = {:.2e}, "
             "|RI-K + exact K| = {:.2e}\n",
             basis_kind(pure), omega, error_along(g, v, fd), max_abs(g + g_exact));
  REQUIRE(error_along(g, v, fd) < 1e-7);
  REQUIRE(g.rowwise().sum().norm() < 1e-10);
}

// HF with RI-JK: Coulomb and exchange gradients both differentiate the
// density-fitted energy.
TEST_CASE("RI-JK HF gradient vs finite differences of the SCF energy",
          "[gradient][df][rik][finite_diff]") {
  const Spin spin = GENERATE(closed_shell, water_cation);
  const Atoms atoms = distorted_water();
  const Setup rijk = [](HartreeFock &hf) {
    hf.set_density_fitting_basis("def2-universal-jkfit");
  };
  Mat3N g;
  hf_energy(atoms, "def2-svp", spin, rijk,
            [&](HartreeFock &hf, const MolecularOrbitals &mo) {
              g = occ::qm::GradientEvaluator<HartreeFock>(hf)(mo);
            });
  const Vec v = generic_direction(atoms);
  const double fd = directional_difference(atoms, v, 1e-3, [&](const Atoms &a) {
    return hf_energy(a, "def2-svp", spin, rijk);
  });
  const double err = error_along(g, v, fd);
  fmt::print("HF/RI-JK {}: |analytic - FD| = {:.2e}\n", spin.label(), err);
  REQUIRE(err < 1e-6);
}

// RI-K second-derivative pieces at fixed occupied coefficients.
TEST_CASE("RI-K Hessian, derivative Fock and response at fixed density",
          "[hessian][df][rik]") {
  const Atoms atoms = distorted_water();
  const bool pure = GENERATE(true, false);
  const double omega = GENERATE(0.0, 0.3);
  HartreeFock hf(load_basis(atoms, "def2-svp", pure));
  const auto mo = scf_orbitals(hf);
  auto df = df_engine(atoms, pure, omega);
  const Mat H = df.exchange_hessian(mo);
  const auto F1 = df.exchange_derivative_fock(mo);
  // contracted with C directly, as the Hessian uses it
  const auto F1o = df.exchange_derivative_fock_occ(mo);
  double occ_err = 0.0;
  for (size_t X = 0; X < F1.size(); X++)
    occ_err = std::max(occ_err, max_abs(F1o[X] - F1[X] * mo.Cocc));
  REQUIRE(occ_err < 1e-12);
  // The Cartesian gradients carry more noise (the FD error grows as 1/h),
  // so the step is a little larger than the spherical basis alone needs.
  const double h = 2e-4;
  const Vec v = generic_direction(atoms);
  auto dp = df_engine(occ::test::displaced_along(atoms, v, h), pure, omega);
  auto dm = df_engine(occ::test::displaced_along(atoms, v, -h), pure, omega);
  const Vec Hv_fd = occ::test::flat(
      (dp.exchange_gradient(mo) - dm.exchange_gradient(mo)) / (2 * h));
  const Mat dK = (dp.exchange(mo) - dm.exchange(mo)) / (2 * h);
  const double fock_err = max_abs(along(F1, v) - dK);
  // Response: K is quadratic in C, so the central difference in C along Cx
  // is exactly K[Cx C^T + C Cx^T].
  const Mat Cx = Mat::Random(mo.Cocc.rows(), mo.Cocc.cols());
  const auto K1 = df.exchange_response(mo.Cocc, {Cx, Mat(-2.0 * Cx)});
  auto mp = mo, mm = mo;
  const double e = 1e-3;
  mp.Cocc = mo.Cocc + e * Cx;
  mm.Cocc = mo.Cocc - e * Cx;
  const Mat Kc = (df.exchange(mp) - df.exchange(mm)) / (2 * e);
  const double resp_err = max_abs(K1[0] - Kc) / max_abs(Kc);
  fmt::print("RI-K ({}, omega {}): |H v - FD| = {:.2e}, |dK/dv - FD| = "
             "{:.2e}, |response - dK/dC| / |K| = {:.2e}\n",
             basis_kind(pure), omega, max_abs(H * v - Hv_fd), fock_err, resp_err);
  // Attenuated: the regularised metric is still poorly conditioned, which
  // amplifies the noise in the differenced gradients.
  const double tol = omega == 0.0 ? 1e-6 : 2e-6;
  REQUIRE(max_abs(H * v - Hv_fd) < tol);
  REQUIRE(fock_err < tol);
  REQUIRE(resp_err < 1e-10);
  REQUIRE(max_abs(K1[1] + 2.0 * K1[0]) < 1e-12);
}
