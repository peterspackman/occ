#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <occ/gto/gto.h>
#include <occ/gto/rotation.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hf.h>
#include <occ/qm/mo.h>
#include <occ/qm/oniom.h>
#include <occ/qm/partitioning.h>
#include <occ/qm/scf.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>
#include <vector>

using Catch::Matchers::WithinAbs;
using occ::format_matrix;
using occ::Mat;
using occ::Mat3;
using occ::qm::HartreeFock;
using occ::qm::SpinorbitalKind;
using occ::util::all_close;

// Basis

TEST_CASE("AOBasis set pure spherical") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};
  auto basis = occ::qm::AOBasis::load(atoms, "6-31G");
  basis.set_pure(true);
  for (const auto &sh : basis.shells()) {
    fmt::print("{: 12.4f}\n", sh);
  }
}

TEST_CASE("AOBasis load") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(atoms, "6-31G");
  for (const auto &sh : basis.shells()) {
    fmt::print("{: 12.4f}\n", sh);
  }
}

// Density Fitting

TEST_CASE("Density Fitting H2O/6-31G J/K matrices") {
  std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                     {1, 0.0, 0.0, 1.39839733}};
  auto basis = occ::qm::AOBasis::load(atoms, "sto-3g");
  basis.set_pure(false);
  auto hf = HartreeFock(basis);
  hf.set_density_fitting_basis("def2-universal-jkfit");

  occ::qm::MolecularOrbitals mo;
  mo.kind = occ::qm::SpinorbitalKind::Restricted;

  mo.C = occ::Mat(2, 2);
  mo.C << 0.54884228, 1.21245192, 0.54884228, -1.21245192;

  mo.Cocc = mo.C.leftCols(1);
  mo.D = mo.Cocc * mo.Cocc.transpose();
  fmt::print("D:\n{}\n", format_matrix(mo.D));

  occ::Mat Fexact(2, 2);
  Fexact << 1.50976125, 0.7301775, 0.7301775, 1.50976125;

  occ::Mat Jexact(2, 2);
  Jexact << 1.34575531, 0.89426314, 0.89426314, 1.34575531;

  occ::Mat Kexact(2, 2);
  Kexact << 1.18164378, 1.05837468, 1.05837468, 1.18164378;

  occ::qm::JKPair jk_approx = hf.compute_JK(mo);
  occ::Mat F = 2 * (jk_approx.J - jk_approx.K);
  fmt::print("Fexact\n{}\n", format_matrix(Fexact));
  fmt::print("Fapprox\n{}\n", format_matrix(F));

  fmt::print("Jexact\n{}\n", format_matrix(Jexact));
  fmt::print("Japprox\n{}\n", format_matrix(jk_approx.J));

  fmt::print("Kexact\n{}\n", format_matrix(Kexact));
  fmt::print("Kapprox\n{}\n", format_matrix(2 * jk_approx.K));
}

TEST_CASE("Electric Field evaluation H2/STO-3G") {
  std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                     {1, 0.0, 0.0, 1.398397}};
  auto basis = occ::qm::AOBasis::load(atoms, "sto-3g");
  Mat D(2, 2);
  D.setConstant(0.301228);
  auto grid_pts = occ::Mat3N(3, 4);
  grid_pts << 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0;
  HartreeFock hf(basis);

  occ::qm::MolecularOrbitals mo;
  mo.D = D;
  mo.kind = occ::qm::SpinorbitalKind::Restricted;
  occ::Vec expected_esp = occ::Vec(4);
  expected_esp << -1.37628, -1.37628, -1.95486, -1.45387;

  auto field_values = hf.nuclear_electric_field_contribution(grid_pts);
  fmt::print("Grid points\n{}\n", format_matrix(grid_pts));
  fmt::print("Nuclear E field values:\n{}\n", format_matrix(field_values));

  auto esp = hf.electronic_electric_potential_contribution(mo, grid_pts);
  fmt::print("ESP:\n{}\n", format_matrix(esp));
  REQUIRE(all_close(esp, expected_esp, 1e-5, 1e-5));
  occ::Mat expected_efield(field_values.rows(), field_values.cols());
  occ::Mat efield;

  expected_efield << -0.592642, 0.0, 0.0, 0.0, 0.0, -0.592642, 0.0, -0.652486,
      0.26967, 0.26967, -0.0880444, -0.116878;

  double delta = 1e-8;
  occ::Mat3N efield_fd(field_values.rows(), field_values.cols());
  for (size_t i = 0; i < 3; i++) {
    auto grid_pts_d = grid_pts;
    grid_pts_d.row(i).array() += delta;
    auto esp_d = hf.electronic_electric_potential_contribution(mo, grid_pts_d);
    efield_fd.row(i) = -(esp_d - esp) / delta;
  }
  REQUIRE(all_close(efield_fd, expected_efield, 1e-5, 1e-5));
  fmt::print("Electric field FD:\n{}\n", format_matrix(efield_fd));
}

// MO Rotation

TEST_CASE("Cartesian gaussian basic rotation matrices", "[mo_rotation]") {
  Mat3 rot = Mat3::Identity(3, 3);
  auto drot = occ::gto::cartesian_gaussian_rotation_matrices(2, rot)[2];
  REQUIRE(all_close(drot, Mat::Identity(6, 6)));

  auto frot = occ::gto::cartesian_gaussian_rotation_matrices(3, rot)[3];
  REQUIRE(all_close(frot, Mat::Identity(10, 10)));
}

TEST_CASE("Water 3-21G basis set rotation energy consistency", "[basis]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};
  auto basis = occ::qm::AOBasis::load(atoms, "3-21G");
  basis.set_pure(false);
  fmt::print("basis.size() {}\n", basis.size());
  Mat3 rotation =
      Eigen::AngleAxisd(M_PI / 2, occ::Vec3{0, 1, 0}).toRotationMatrix();
  fmt::print("Rotation by:\n{}\n", format_matrix(rotation));

  auto hf = HartreeFock(basis);
  occ::qm::SCF<HartreeFock> scf(hf);
  double e = scf.compute_scf_energy();

  occ::qm::AOBasis rot_basis = basis;
  rot_basis.rotate(rotation);
  auto rot_atoms = rot_basis.atoms();
  fmt::print("rot_basis.size() {}\n", rot_basis.size());
  auto hf_rot = HartreeFock(rot_basis);
  occ::qm::SCF<HartreeFock> scf_rot(hf_rot);
  double e_rot = scf_rot.compute_scf_energy();

  REQUIRE(e == Catch::Approx(e_rot));
}

occ::Mat interatomic_distances(const std::vector<occ::core::Atom> &atoms) {
  size_t natoms = atoms.size();
  occ::Mat dists(natoms, natoms);
  for (size_t i = 0; i < natoms; i++) {
    dists(i, i) = 0;
    for (size_t j = i + 1; j < natoms; j++) {
      double dx = atoms[i].x - atoms[j].x;
      double dy = atoms[i].y - atoms[j].y;
      double dz = atoms[i].z - atoms[j].z;
      dists(i, j) = sqrt(dx * dx + dy * dy + dz * dz);
      dists(j, i) = dists(i, j);
    }
  }
  return dists;
}

TEST_CASE("Water def2-tzvp MO rotation energy consistency", "[basis]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};
  auto basis = occ::qm::AOBasis::load(atoms, "def2-tzvp");
  basis.set_pure(true);
  Eigen::Quaterniond r(Eigen::AngleAxisd(
      0.423, Eigen::Vector3d(0.234, -0.642, 0.829).normalized()));
  Mat3 rotation = r.toRotationMatrix();

  fmt::print("Rotation by:\n{}\n", format_matrix(rotation));
  fmt::print("Distances before rotation:\n{}\n",
             format_matrix(interatomic_distances(atoms)));
  auto hf = HartreeFock(basis);

  auto rot_basis = basis;
  rot_basis.rotate(rotation);
  auto rot_atoms = rot_basis.atoms();
  auto shell2atom = rot_basis.shell_to_atom();

  fmt::print("Distances after rotation:\n{}\n",
             format_matrix(interatomic_distances(rot_atoms)));
  auto hf_rot = HartreeFock(rot_basis);
  REQUIRE(hf.nuclear_repulsion_energy() ==
          Catch::Approx(hf_rot.nuclear_repulsion_energy()));
  occ::qm::SCF<HartreeFock> scf(hf);
  double e = scf.compute_scf_energy();
  occ::qm::MolecularOrbitals mos = scf.ctx.mo;
  Mat C_occ = mos.C.leftCols(scf.ctx.n_occ);
  Mat D = C_occ * C_occ.transpose();

  mos.rotate(rot_basis, rotation);
  Mat rot_C_occ = mos.C.leftCols(scf.ctx.n_occ);
  Mat rot_D = rot_C_occ * rot_C_occ.transpose();

  double e_en = occ::qm::expectation<occ::qm::SpinorbitalKind::Restricted>(
      D, hf.compute_nuclear_attraction_matrix());
  double e_en_rot = occ::qm::expectation<occ::qm::SpinorbitalKind::Restricted>(
      rot_D, hf_rot.compute_nuclear_attraction_matrix());
  fmt::print("E_en      {}\n", e_en);
  fmt::print("E_en'     {}\n", e_en_rot);
  REQUIRE(e_en == Catch::Approx(e_en_rot));
}

// SCF

TEST_CASE("Water RHF SCF energy", "[scf]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};

  SECTION("STO-3G") {
    auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");
    HartreeFock hf(obs);
    occ::qm::SCF<HartreeFock> scf(hf);
    scf.convergence_settings.energy_threshold = 1e-8;
    double e = scf.compute_scf_energy();
    REQUIRE(e == Catch::Approx(-74.963706080054).epsilon(1e-8));
  }

  SECTION("3-21G") {
    auto obs = occ::qm::AOBasis::load(atoms, "3-21G");
    HartreeFock hf(obs);
    occ::qm::SCF<HartreeFock> scf(hf);
    scf.convergence_settings.energy_threshold = 1e-8;
    double e = scf.compute_scf_energy();
    REQUIRE(e == Catch::Approx(-75.585325673488).epsilon(1e-8));
  }
}

TEST_CASE("Water UHF SCF energy", "[scf]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};

  SECTION("STO-3G") {
    auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");
    HartreeFock hf(obs);
    occ::qm::SCF<HartreeFock> scf(hf, SpinorbitalKind::Unrestricted);
    scf.convergence_settings.energy_threshold = 1e-8;
    double e = scf.compute_scf_energy();
    REQUIRE(e == Catch::Approx(-74.963706080054).epsilon(1e-8));
  }

  SECTION("3-21G") {
    auto obs = occ::qm::AOBasis::load(atoms, "3-21G");
    HartreeFock hf(obs);
    occ::qm::SCF<HartreeFock> scf(hf, SpinorbitalKind::Unrestricted);
    scf.convergence_settings.energy_threshold = 1e-8;
    double e = scf.compute_scf_energy();
    REQUIRE(e == Catch::Approx(-75.585325673488).epsilon(1e-8));
  }
}

TEST_CASE("Water GHF SCF energy", "[scf]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};

  SECTION("STO-3G") {
    auto obs = occ::qm::AOBasis::load(atoms, "STO-3G");
    HartreeFock hf(obs);
    occ::qm::SCF<HartreeFock> scf(hf, SpinorbitalKind::General);
    scf.convergence_settings.energy_threshold = 1e-8;
    double e = scf.compute_scf_energy();
    REQUIRE(e == Catch::Approx(-74.963706080054).epsilon(1e-8));
  }

  SECTION("3-21G") {
    auto obs = occ::qm::AOBasis::load(atoms, "3-21G");
    HartreeFock hf(obs);
    occ::qm::SCF<HartreeFock> scf(hf, SpinorbitalKind::General);
    scf.convergence_settings.energy_threshold = 1e-8;
    double e = scf.compute_scf_energy();
    REQUIRE(e == Catch::Approx(-75.585325673488).epsilon(1e-8));
  }
}

TEST_CASE("Smearing functions", "[smearing]") {
  occ::qm::MolecularOrbitals mo;
  mo.energies = occ::Vec(43);

  mo.energies << -20.7759836, -1.75714997, -0.6755331, -0.47838917, -0.34397101,
      0.39919219, 0.53017716, 0.84601736, 0.91026033, 0.93422352, 0.93541066,
      1.07197252, 1.1152775, 1.74421062, 1.78568455, 1.87179641, 2.12778396,
      2.22487896, 2.41855156, 2.63997607, 2.69396251, 2.72429482, 2.94010404,
      2.99844011, 3.27233171, 3.29230622, 3.30213655, 3.40824971, 4.35313802,
      4.62532886, 5.78639639, 5.89978906, 6.07407985, 6.11565547, 6.16161542,
      6.80026627, 6.96537841, 7.04215449, 7.10247792, 7.17168709, 7.63716226,
      7.75364544, 45.00621777;

  SECTION("Fermi") {
    occ::qm::OrbitalSmearing smearing;
    smearing.sigma = 0.095;
    smearing.mu = -0.06;
    smearing.kind = occ::qm::OrbitalSmearing::Kind::Fermi;

    occ::Vec res = smearing.calculate_fermi_occupations(mo);

    occ::Vec expected(43);
    expected << 1.00000000e+00, 9.99999983e-01, 9.98467461e-01, 9.87920549e-01,
        9.52082391e-01, 7.89497886e-03, 2.00042906e-03, 7.21259277e-05,
        3.66791050e-05, 2.85019162e-05, 2.81479763e-05, 6.68591892e-06,
        4.23830834e-06, 5.64954547e-09, 3.65102304e-09, 1.47486548e-09,
        9.96552046e-11, 3.58614765e-11, 4.66927912e-12, 4.53944824e-13,
        2.57159713e-13, 1.86869380e-13, 1.92735534e-14, 1.04298300e-14,
        5.83681688e-16, 4.73001131e-16, 4.26503520e-16, 1.39580290e-16,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00;

    REQUIRE(all_close(res, expected, 1e-5, 1e-5));

    // override occupations to be exactly the right thing
    mo.occupation = expected;
    smearing.entropy = smearing.calculate_entropy(mo);
    REQUIRE(smearing.ec_entropy() == Catch::Approx(-0.06301068258742577));
  }
}

TEST_CASE("H2 smearing", "[smearing]") {
  occ::Vec expected_energies(16), expected_correlations(16), separations(16);

  expected_energies << -1.05401645, -1.10204847, -1.10241449, -1.09833917,
      -1.0429699, -0.96723086, -0.89171816, -0.82552549, -0.77490759,
      -0.74204562, -0.72385175, -0.71057038, -0.7076913, -0.70707591,
      -0.70694503, -0.70691935;

  separations << 1.0, 1.3, 1.4, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0,
      7.0, 8.0, 9.0, 10.0;

  expected_correlations << -0.02090322, -0.02916732, -0.03247401, -0.03609615,
      -0.0596489, -0.09272139, -0.13295573, -0.17469005, -0.21056677,
      -0.2356074, -0.25006385, -0.26093526, -0.2633199, -0.2638024, -0.26389139,
      -0.26390591;

  occ::Vec energies(16);

  for (int i = 0; i < separations.rows(); i++) {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                       {1, separations(i), 0.0, 0.0}};
    auto obs = occ::qm::AOBasis::load(atoms, "cc-pvdz");
    obs.set_pure(true);
    HartreeFock hf(obs);
    occ::qm::SCF<HartreeFock> scf(hf);

    scf.ctx.mo.smearing.kind = occ::qm::OrbitalSmearing::Kind::Fermi;
    scf.ctx.mo.smearing.sigma = 0.095;

    energies(i) = scf.compute_scf_energy();

    REQUIRE_THAT(energies(i), WithinAbs(expected_energies(i), 1e-5));
    REQUIRE_THAT(scf.ctx.mo.smearing.ec_entropy(),
                 WithinAbs(expected_correlations(i), 1e-5));
  }
}

TEST_CASE("Integral gradients", "[integrals]") {

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

TEST_CASE("Oniom ethane", "[oniom]") {
  using occ::core::Atom;
  using occ::qm::SCF;
  std::vector<Atom> atoms{
      {1, 2.239513249136882, -0.007369927999015981, 1.8661035638534056},
      {6, 1.4203174061693364, -0.042518815378938354, -0.039495255174213845},
      {1, 2.205120251808141, 1.5741410315846955, -1.075820515343538},
      {1, 2.1079883802313657, -1.7629245718671818, -0.9722635783317236},
      {6, -1.4203174061693364, 0.042518815378938354, 0.039495255174213845},
      {1, -2.205120251808141, -1.5748969216358768, 1.0746866802667663},
      {1, -2.108366325256956, 1.762357654328796, 0.9733974134084954},
      {1, -2.2393242766240866, 0.00831479056299239, -1.8661035638534056}};

  Atom artificial_h1 = atoms[1];
  artificial_h1.atomic_number = 1;

  Atom artificial_h2 = atoms[4];
  artificial_h2.atomic_number = 1;

  std::vector<Atom> methane1{atoms[0], atoms[1], atoms[2], atoms[3],
                             artificial_h2};

  std::vector<Atom> methane2{artificial_h1, atoms[4], atoms[5], atoms[6],
                             atoms[7]};

  HartreeFock system_low(occ::qm::AOBasis::load(atoms, "STO-3G"));
  HartreeFock methane1_low(occ::qm::AOBasis::load(methane1, "STO-3G"));
  HartreeFock methane2_low(occ::qm::AOBasis::load(methane2, "STO-3G"));

  HartreeFock methane1_high(occ::qm::AOBasis::load(methane1, "def2-tzvp"));
  HartreeFock methane2_high(occ::qm::AOBasis::load(methane2, "def2-tzvp"));

  using Proc = SCF<HartreeFock>;
  occ::qm::Oniom<Proc, Proc> oniom{{SCF(methane1_high), SCF(methane2_high)},
                                   {SCF(methane1_low), SCF(methane2_low)},
                                   SCF(system_low)};

  fmt::print("Total energy: {}\n", oniom.compute_scf_energy());
}

TEST_CASE("Mulliken partition", "[partitioning]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};

  auto obs = occ::qm::AOBasis::load(atoms, "3-21G");

  auto hf = HartreeFock(obs);
  occ::qm::SCF<HartreeFock> scf(hf);
  double e = scf.compute_scf_energy();

  auto wfn = scf.wavefunction();

  auto charges = wfn.mulliken_charges();
  fmt::print("Charges:\n{}\n", format_matrix(charges));
  occ::Vec expected(3);
  expected << -0.724463, 0.363043, 0.361419;
  fmt::print("Expected:\n{}\n", format_matrix(expected));
  REQUIRE(all_close(expected, charges, 1e-5, 1e-5));

  auto energies = occ::qm::mulliken_partition(obs, wfn.mo, wfn.V);
  double total = occ::qm::expectation(wfn.mo.kind, wfn.mo.D, wfn.V);
  fmt::print("Partitioned energy\n{}\n", format_matrix(energies));
  REQUIRE(energies.sum() == Catch::Approx(total));
}
