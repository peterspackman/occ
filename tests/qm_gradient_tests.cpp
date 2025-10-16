#include <catch2/catch_test_macros.hpp>
#include <occ/core/util.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/gradients.h>

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

  auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");

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

  auto basis = occ::qm::AOBasis::load(atoms, "STO-3G");
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

  auto basis = occ::qm::AOBasis::load(atoms, "STO-3G");
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

  auto basis = occ::qm::AOBasis::load(atoms, "3-21G");
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
