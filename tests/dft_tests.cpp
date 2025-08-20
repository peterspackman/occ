#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/atom.h>
#include <occ/core/molecule.h>
#include <occ/core/multipole.h>
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/dft/grid_types.h>
#include <occ/dft/grid_utils.h>
#include <occ/dft/hirshfeld.h>
#include <occ/dft/voronoi_charges.h>
#include <occ/dft/lebedev.h>
#include <occ/dft/molecular_grid.h>
#include <occ/dft/nonlocal_correlation.h>
#include <occ/dft/seminumerical_exchange.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/shell.h>
#include <occ/qm/wavefunction.h>
#include <vector>
#include <chrono>

// DFT

using Catch::Approx;
using occ::format_matrix;
using occ::Mat;
using occ::Vec;

TEST_CASE("LDA (Slater) exchange energy density", "[lda]") {
  occ::dft::DensityFunctional lda("xc_lda_x");
  occ::dft::DensityFunctional lda_u("xc_lda_x", true);
  occ::dft::DensityFunctional::Params params(
      5, occ::dft::DensityFunctional::Family::LDA,
      occ::qm::SpinorbitalKind::Restricted);
  REQUIRE(params.rho.size() == 5);
  params.rho = occ::Vec::LinSpaced(5, 0, 1);
  fmt::print("Rho:\n{}\n", format_matrix(params.rho));
  auto res = lda.evaluate(params);
  fmt::print("exc:\n{}\nvrho\n{}\n", format_matrix(res.exc),
             format_matrix(res.vrho));

  occ::dft::DensityFunctional::Params params_u(
      5, occ::dft::DensityFunctional::Family::LDA,
      occ::qm::SpinorbitalKind::Unrestricted);
  REQUIRE(params_u.rho.size() == 10);
  for (size_t i = 0; i < params.rho.rows(); i++) {
    params_u.rho(2 * i) = params.rho(i);
    params_u.rho(2 * i + 1) = params.rho(i);
  }
  fmt::print("Rho interleaved:\n{}\n", format_matrix(params_u.rho));
  auto res1 = lda_u.evaluate(params_u);
  fmt::print("exc:\n{}\nvrho\n{}\n", format_matrix(res1.exc),
             format_matrix(res1.vrho));

  params_u.rho = params.rho.replicate(2, 1);
  fmt::print("Rho block:\n{}\n", format_matrix(params_u.rho));
  auto res2 = lda_u.evaluate(params_u);
  fmt::print("exc:\n{}\nvrho\n{}\n", format_matrix(res2.exc),
             format_matrix(res2.vrho));
}

TEST_CASE("GGA (PBE) exchange energy density", "[gga]") {
  using occ::util::all_close;
  namespace block = occ::qm::block;
  occ::dft::DensityFunctional gga("xc_gga_x_pbe");
  occ::dft::DensityFunctional gga_u("xc_gga_x_pbe", true);
  occ::Mat rho(8, 4);
  rho << 3.009700173159170558e-02, -6.373586084157208120e-02,
      -0.000000000000000000e+00, 1.865655330995227498e-02,
      3.009700173159170558e-02, -0.000000000000000000e+00,
      -6.373586084157208120e-02, 1.865655330995227498e-02,
      1.508591677559790178e-01, -0.000000000000000000e+00,
      -0.000000000000000000e+00, 1.303514966003518905e-01,
      1.775122853194434636e-01, -0.000000000000000000e+00,
      -0.000000000000000000e+00, 7.842601108050306635e-02,
      3.009700173159170558e-02, -6.373586084157208120e-02,
      -0.000000000000000000e+00, 1.865655330995227498e-02,
      3.009700173159170558e-02, -0.000000000000000000e+00,
      -6.373586084157208120e-02, 1.865655330995227498e-02,
      1.508591677559790178e-01, -0.000000000000000000e+00,
      -0.000000000000000000e+00, 1.303514966003518905e-01,
      1.775122853194434636e-01, -0.000000000000000000e+00,
      -0.000000000000000000e+00, 7.842601108050306635e-02;
  occ::dft::DensityFunctional::Params params(
      4, occ::dft::DensityFunctional::Family::GGA,
      occ::qm::SpinorbitalKind::Restricted);
  REQUIRE(params.rho.size() == 4);
  params.rho.col(0) = block::a(rho).col(0);
  auto rho_a = block::a(rho), rho_b = block::b(rho);
  fmt::print("Rho_a:\n{}\n", format_matrix(rho_a));
  params.sigma.col(0) = rho_a.col(1).array() * rho_a.col(1).array() +
                        rho_a.col(2).array() * rho_a.col(2).array() +
                        rho_a.col(3).array() * rho_a.col(3).array();
  fmt::print("GGA-----\nRho:\n{}\n\nsigma\n{}\n", format_matrix(params.rho),
             format_matrix(params.sigma));
  auto res = gga.evaluate(params);
  fmt::print("exc:\n{}\nvrho\n{}\nvsigma\n{}\n", format_matrix(res.exc),
             format_matrix(res.vrho), format_matrix(res.vsigma));

  occ::Vec expected_exc(4);
  expected_exc << -0.27851489, -0.27851489, -0.39899553, -0.41654061;
  REQUIRE(all_close(expected_exc, res.exc, 1e-6));

  occ::dft::DensityFunctional::Params params_u(
      4, occ::dft::DensityFunctional::Family::GGA,
      occ::qm::SpinorbitalKind::Unrestricted);
  REQUIRE(params_u.rho.size() == 8);
  params_u.rho.col(0) = rho_a.col(0);
  params_u.rho.col(1) = rho_b.col(0);
  params_u.sigma.col(0) = rho_a.col(1).array() * rho_a.col(1).array() +
                          rho_a.col(2).array() * rho_a.col(2).array() +
                          rho_a.col(3).array() * rho_a.col(3).array();
  params_u.sigma.col(1) = rho_a.col(1).array() * rho_b.col(1).array() +
                          rho_a.col(2).array() * rho_b.col(2).array() +
                          rho_a.col(3).array() * rho_b.col(3).array();
  params_u.sigma.col(2) = rho_b.col(1).array() * rho_b.col(1).array() +
                          rho_b.col(2).array() * rho_b.col(2).array() +
                          rho_b.col(3).array() * rho_b.col(3).array();
  fmt::print("rho_xyz\n{}\n{}\n", format_matrix(rho_a.block(0, 1, 4, 3)),
             format_matrix(rho_b.block(0, 1, 4, 3)));
  fmt::print("\n\nRho interleaved:\n{}\nsigma\n{}\n",
             format_matrix(params_u.rho), format_matrix(params_u.sigma));
  auto res1 = gga_u.evaluate(params_u);
  fmt::print("exc:\n{}\nvrho\n{}\nvsigma\n{}\n", format_matrix(res1.exc),
             format_matrix(res1.vrho), format_matrix(res1.vsigma));
  // assert(all_close(expected_exc, res1.exc, 1e-6));
}

TEST_CASE("MGGA") {
  occ::dft::DensityFunctional mgga("xc_mgga_x_tpss");
  occ::dft::DensityFunctional mgga_c("xc_mgga_c_tpss");
  occ::Mat rho(4, 6);
  rho << 0.06019401627252133, -0.12747174847145987, -0, 0.03731311973522952,
      0.09472176570942804, 0.036634279981580156, 0.06019401627252133, -0,
      -0.12747174847145987, 0.03731311973522952, 0.09472176570942804,
      0.036634279981580156, 0.30171849914696935, -0, -0, 0.26070301592021455,
      -2.594981941107205, 0.02815789498408782, 0.3550246251878335, -0, -0,
      0.15685208520532756, -6.307746108179051, 0.008662278222335566;
  fmt::print("rho =\n{}\n", format_matrix(rho));

  occ::dft::DensityFunctional::Params params(
      4, occ::dft::DensityFunctional::Family::MGGA,
      occ::qm::SpinorbitalKind::Restricted);
  params.rho.col(0) = rho.col(0);
  params.sigma.col(0) = rho.col(1).array() * rho.col(1).array() +
                        rho.col(2).array() * rho.col(2).array() +
                        rho.col(3).array() * rho.col(3).array();
  params.laplacian.col(0) = rho.col(4);
  params.tau.col(0) = rho.col(5);
  occ::Vec expected_exc(4);
  expected_exc << -0.33175882, -0.33175882, -0.56154382, -0.59257418;
  auto res = mgga.evaluate(params);
  fmt::print("Expected exc TPSSx\n{}\n", format_matrix(expected_exc));
  fmt::print("MGGA exc TPSSx\n{}\n", format_matrix(res.exc));
  fmt::print("Difference\n{}\n", format_matrix(res.exc - expected_exc));
  REQUIRE(occ::util::all_close(expected_exc, res.exc, 1e-6));
  occ::Vec expected_exc_c(4);
  expected_exc_c << -0.02276044, -0.02276044, -0.04168478, -0.04268936;
  auto res_c = mgga_c.evaluate(params);
  fmt::print("Expected exc TPSSc\n{}\n", format_matrix(expected_exc_c));
  fmt::print("MGGA exc TPSSc\n{}\n", format_matrix(res_c.exc));
  fmt::print("Difference\n{}\n", format_matrix(res_c.exc - expected_exc_c));
  REQUIRE(occ::util::all_close(expected_exc_c, res_c.exc, 1e-6));
  fmt::print("MGGA vrho TPSSc\n{}\n", format_matrix(res_c.vrho));
  fmt::print("MGGA vsigma TPSSc\n{}\n", format_matrix(res_c.vsigma));
  fmt::print("MGGA vlaplacian TPSSc\n{}\n", format_matrix(res_c.vlaplacian));
  fmt::print("MGGA vtau TPSSc\n{}\n", format_matrix(res_c.vtau));
  occ::Vec expected_vrho_c(4), expected_vsigma_c(4), expected_vtau_c(4),
      expected_vlapl_c(4);
  expected_vrho_c << -5.57565777e-02, -5.57565777e-02, -8.00593613e-02,
      -8.53775165e-02;
  expected_vsigma_c << 3.91761083e-02, 3.91761083e-02, 1.47458233e-01,
      5.51608074e-01;
  expected_vlapl_c << 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      0.00000000e+00;
  expected_vtau_c << 1.10283011e-03, 1.10283011e-03, -3.50002323e-01,
      -1.56487879e+00;
  REQUIRE(occ::util::all_close(expected_vrho_c, res_c.vrho, 1e-6));
  REQUIRE(occ::util::all_close(expected_vsigma_c, res_c.vsigma, 1e-6));
  REQUIRE(occ::util::all_close(expected_vtau_c, res_c.vtau, 1e-6));
  REQUIRE(occ::util::all_close(expected_vlapl_c, res_c.vlaplacian, 1e-6));
}

// Grid
using occ::util::all_close;

TEST_CASE("Lebedev grid construction", "[grid]") {
  auto grid = occ::dft::grid::lebedev(110);
  fmt::print("grid:\n{}\n", format_matrix(grid));
  REQUIRE(grid.rows() == 110);
  REQUIRE(grid.cols() == 4);

  /*
  BENCHMARK("Lebedev 86") {
      return occ::dft::grid::lebedev(86);
  };0

  BENCHMARK("Lebedev 590") {
      return occ::dft::grid::lebedev(590);
  };

  BENCHMARK("Lebedev 5810") {
      return occ::dft::grid::lebedev(5810);
  };
  */
}

TEST_CASE("Becke radial grid points", "[radial]") {
  auto radial = occ::dft::generate_becke_radial_grid(3, 0.6614041435977716);
  occ::Vec3 expected_pts{9.21217133, 0.66140414, 0.04748668};
  occ::Vec3 expected_weights{77.17570606, 1.3852416, 0.39782349};
  fmt::print("Becke radial grid:\n{} == {}\n{} == {}\n",
             format_matrix(radial.points), format_matrix(expected_pts),
             format_matrix(radial.weights), format_matrix(expected_weights));

  REQUIRE(all_close(radial.points, expected_pts, 1e-5));
  REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
  /*
  BENCHMARK("Becke radial 10") {
      return occ::dft::generate_becke_radial_grid(10);
  };

  BENCHMARK("Becke radial 50") {
      return occ::dft::generate_becke_radial_grid(10);
  };

  BENCHMARK("Becke radial 80") {
      return occ::dft::generate_becke_radial_grid(10);
  };
  */
}

TEST_CASE("Gauss-Chebyshev radial grid points", "[radial]") {
  auto radial = occ::dft::generate_gauss_chebyshev_radial_grid(3);
  occ::Vec3 expected_pts{8.66025404e-01, 6.123234e-17, -8.66025404e-01};
  occ::Vec3 expected_weights{1.04719755, 1.04719755, 1.04719755};
  fmt::print("Gauss-Chebyshev radial grid:\n{} == {}\n{} == {}\n",
             format_matrix(radial.points), format_matrix(expected_pts),
             format_matrix(radial.weights), format_matrix(expected_weights));

  REQUIRE(all_close(radial.points, expected_pts, 1e-5));
  REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
}

TEST_CASE("Mura-Knowles radial grid points", "[radial]") {
  auto radial = occ::dft::generate_mura_knowles_radial_grid(3, 1);
  occ::Vec3 expected_pts{0.02412997, 0.69436324, 4.49497829};
  occ::Vec3 expected_weights{0.14511628, 1.48571429, 8.57142857};
  fmt::print("Mura-Knowles radial grid:\n{} == {}\n{} == {}\n",
             format_matrix(radial.points), format_matrix(expected_pts),
             format_matrix(radial.weights), format_matrix(expected_weights));

  REQUIRE(all_close(radial.points, expected_pts, 1e-5));
  REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
  /*
  BENCHMARK("Becke radial 10") {
      return occ::dft::generate_becke_radial_grid(10);
  };

  BENCHMARK("Becke radial 50") {
      return occ::dft::generate_becke_radial_grid(10);
  };

  BENCHMARK("Becke radial 80") {
      return occ::dft::generate_becke_radial_grid(10);
  };
  */
}

TEST_CASE("Treutler-Alrichs radial grid points", "[radial]") {
  auto radial = occ::dft::generate_treutler_alrichs_radial_grid(3);
  occ::Vec3 expected_pts{0.10934791, 1, 3.82014324};
  occ::Vec3 expected_weights{0.34905607, 1.60432893, 4.51614622};
  fmt::print("Treutler-Alrichs radial grid:\n{} == {}\n{} == {}\n",
             format_matrix(radial.points), format_matrix(expected_pts),
             format_matrix(radial.weights), format_matrix(expected_weights));

  REQUIRE(all_close(radial.points, expected_pts, 1e-5));
  REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
}

TEST_CASE("Euler-Maclaurin radial grid points", "[radial]") {
  auto radial = occ::dft::generate_euler_maclaurin_radial_grid(10, 1.2);
  occ::Vec expected_pts(9);
  occ::Vec expected_weights(9);

  expected_pts << 0.0148148, 0.0750000, 0.2204082, 0.5333333, 1.2000000,
      2.7000000, 6.5333333, 19.2000000, 97.2000000;

  expected_weights << 0.7225637E-05, 0.5273438E-03, 0.1019750E-01,
      0.1264198E+00, 0.1382400E+01, 0.1640250E+02, 0.2655921E+03, 0.8847360E+04,
      0.2040733E+07;

  fmt::print("Euler- Maclaurin radial grid found:\n{}\nexpected\n{}\n",
             format_matrix(radial.points), format_matrix(expected_pts));
  fmt::print("weights found:\n{}\nexpected\n{}\n",
             format_matrix(radial.weights), format_matrix(expected_weights));

  REQUIRE(all_close(radial.points, expected_pts, 1e-5));
  REQUIRE(all_close(radial.weights, expected_weights, 1e-5));
}

TEST_CASE("Becke partitioned atom grid H2", "[grid]") {
  std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                     {1, 0.0, 0.0, 1.39839733}};
  auto basis = occ::qm::AOBasis::load(atoms, "sto-3g");
  occ::dft::MolecularGrid mgrid(basis);

  auto grid = mgrid.get_partitioned_atom_grid(0);
}

// Seminumerical Exchange

TEST_CASE("Water seminumerical exchange approximation", "[scf]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};
  auto basis = occ::qm::AOBasis::load(atoms, "6-31G");
  basis.set_pure(false);
  auto hf = occ::qm::HartreeFock(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  double e = scf.compute_scf_energy();

  occ::io::GridSettings settings;
  settings.max_angular_points = 50;
  settings.radial_precision = 1e-3;
  fmt::print("Construct\n");
  occ::dft::cosx::SemiNumericalExchange sgx(basis, settings);
  fmt::print("Construct done\n");

  occ::timing::StopWatch<2> sw;
  sw.start(0);
  fmt::print("Compute K SGX\n");
  occ::Mat result = sgx.compute_K(scf.ctx.mo);
  sw.stop(0);
  fmt::print("Compute K SGX done\n");
  sw.start(1);
  occ::qm::JKPair jk_exact = hf.compute_JK(scf.ctx.mo, occ::Mat());
  sw.stop(1);
  int i, j;
  fmt::print("K - Kexact: {:12.8f}\n",
             (result - jk_exact.K).array().cwiseAbs().maxCoeff(&i, &j));
  /*
  const auto &bf2shell = sgx.engine().aobasis().bf_to_shell();
  int s1 = bf2shell[i];
  const auto &sh1 = sgx.engine().aobasis()[s1];
  int s2 = bf2shell[i];
  const auto &sh2 = sgx.engine().aobasis()[s2];

  fmt::print("Shells:\n");
  std::cout << s1 << sh1 << '\n' << s2 << sh2 << '\n';
  int bf1 = sgx.engine().aobasis().first_bf()[s1];
  int bf2 = sgx.engine().aobasis().first_bf()[s2];

  fmt::print("K SGX\n{}\n", result.block(bf1, bf2, sh1.size(), sh2.size()));
  fmt::print("K exact\n{}\n", Kexact.block(bf1, bf2, sh1.size(), sh2.size()));
  */
  fmt::print("Speedup = ({} vs. {}) {:.3f} times\n", sw.read(0), sw.read(1),
             sw.read(1) / sw.read(0));

  occ::timing::print_timings();
}

TEST_CASE("H2 VV10 from ORCA wavefunction") {

  const char *h2_molden_content = R"(
[Molden Format]
[Title]
 Molden file created by orca_2mkl for BaseName=h2

[Atoms] AU
H    1   1          0.0000000000         0.0000000000         0.0000000000 
H    2   1          0.0000000000         0.0000000000         1.3983973391 
[GTO]
  1 0
s   3 1.0 
        3.4252509100         0.2769343610
        0.6239137300         0.2678388518
        0.1688554000         0.0834736696

  2 0
s   3 1.0 
        3.4252509100         0.2769343610
        0.6239137300         0.2678388518
        0.1688554000         0.0834736696

[5D]
[7F]
[9G]
[MO]
 Sym=     1a
 Ene= -5.15449496312014E-01
 Spin= Alpha
 Occup= 2.000000
  1      -0.548842275549
  2      -0.548842275421
 Sym=     1a
 Ene= 5.11943815961741E-01
 Spin= Alpha
 Occup= 0.000000
  1      -1.212451915878
  2       1.212451915935
)";

  std::istringstream molden(h2_molden_content);
  occ::io::MoldenReader reader(molden);
  occ::qm::Wavefunction wfn(reader);
  occ::dft::NonLocalCorrelationFunctional nlc;
  nlc.set_integration_grid(wfn.basis);
  nlc.set_parameters(
      {occ::dft::NonLocalCorrelationFunctional::Kind::VV10, 6.0, 0.01});

  fmt::print("{} atoms read\n", wfn.atoms.size());
  for (const auto &atom : wfn.atoms) {
    fmt::print("{} {:12.5f} {:12.5f} {:12.5f}\n", atom.atomic_number, atom.x,
               atom.y, atom.z);
  }
  fmt::print("wfn num e = {}\n", wfn.num_electrons);
  fmt::print("Mo energies\n{}\n", format_matrix(wfn.mo.energies));
  fmt::print("MO coefficients\n{}\n", format_matrix(wfn.mo.C));
  fmt::print("MO occupied\n{}\n", format_matrix(wfn.mo.Cocc));
  fmt::print("MO D\n{}\n", format_matrix(wfn.mo.D));
  fmt::print("Basis shells {}\n", wfn.basis.size());
  for (const auto &sh : wfn.basis.shells()) {
    fmt::print("Shell {} primitives {}\n", sh.symbol(), sh.num_primitives());
  }
  auto result = nlc(wfn.basis, wfn.mo);
  double expected = 0.0089406089;
  fmt::print("NLC = {} vs {}\n", result.energy, expected);
  REQUIRE(result.energy == Catch::Approx(expected));
}

// Hirshfeld charge tests

namespace {

occ::core::Molecule make_h2o_for_hirshfeld() {
  occ::Vec3 O{0.0, 0.0, 0.0};
  occ::Vec3 H1{0.0, -0.757, 0.587};
  occ::Vec3 H2{0.0, 0.757, 0.587};
  occ::Mat3N pos(3, 3);
  pos << O(0), H1(0), H2(0), O(1), H1(1), H2(1), O(2), H1(2), H2(2);
  occ::IVec atomic_numbers(3);
  atomic_numbers << 8, 1, 1;
  return occ::core::Molecule(atomic_numbers, pos);
}

occ::core::Molecule make_ch4_for_hirshfeld() {
  occ::Vec3 C{0.0, 0.0, 0.0};
  occ::Vec3 H1{0.626, 0.626, 0.626};
  occ::Vec3 H2{-0.626, -0.626, 0.626};
  occ::Vec3 H3{-0.626, 0.626, -0.626};
  occ::Vec3 H4{0.626, -0.626, -0.626};
  occ::Mat3N pos(3, 5);
  pos << C(0), H1(0), H2(0), H3(0), H4(0), C(1), H1(1), H2(1), H3(1), H4(1),
      C(2), H1(2), H2(2), H3(2), H4(2);
  occ::IVec atomic_numbers(5);
  atomic_numbers << 6, 1, 1, 1, 1;
  return occ::core::Molecule(atomic_numbers, pos);
}

} // namespace

TEST_CASE("Hirshfeld charges for water", "[hirshfeld]") {
  auto mol = make_h2o_for_hirshfeld();
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(mol.atoms(), "STO-3G");
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  double energy = scf.compute_scf_energy();
  auto mo = scf.molecular_orbitals();

  occ::dft::HirshfeldPartition hirshfeld(basis);
  occ::Vec charges = hirshfeld.calculate(mo);

  // Expected Hirshfeld charges for water molecule with STO-3G
  // Oxygen should be negative, hydrogens positive
  REQUIRE(charges.size() == 3);
  REQUIRE(charges(0) < 0);
  REQUIRE(charges(1) > 0);
  REQUIRE(charges(2) > 0);

  auto volumes = hirshfeld.atom_volumes();
  REQUIRE(volumes.size() == 3);
  REQUIRE(volumes(0) > 0);
  REQUIRE(volumes(1) > 0);
  REQUIRE(volumes(2) > 0);

  // Check conservation of charge
  REQUIRE(charges.sum() == Approx(0.0).margin(1e-8));

  fmt::print("Water Hirshfeld charges:\n{}\n", format_matrix(charges));
  fmt::print("Water atom volumes:\n{}\n", format_matrix(volumes));
  fmt::print("Water free atom volumes:\n{}\n",
             format_matrix(hirshfeld.free_atom_volumes()));
}

TEST_CASE("Hirshfeld charges for methane", "[hirshfeld]") {
  auto mol = make_ch4_for_hirshfeld();
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(mol.atoms(), "STO-3G");
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  double energy = scf.compute_scf_energy();
  auto mo = scf.molecular_orbitals();

  occ::dft::HirshfeldPartition hirshfeld(basis);
  occ::Vec charges = hirshfeld.calculate(mo);

  // Expected Hirshfeld charges for methane with STO-3G
  // Carbon should be slightly negative, hydrogens slightly positive
  REQUIRE(charges.size() == 5);

  // Check approximate charge symmetry for hydrogens
  auto h_charge_avg = (charges(1) + charges(2) + charges(3) + charges(4)) / 4.0;
  REQUIRE(charges(1) == Approx(h_charge_avg).margin(1e-3));
  REQUIRE(charges(2) == Approx(h_charge_avg).margin(1e-3));
  REQUIRE(charges(3) == Approx(h_charge_avg).margin(1e-3));
  REQUIRE(charges(4) == Approx(h_charge_avg).margin(1e-3));

  // Check conservation of charge (allowing for slight numerical error)
  REQUIRE(charges.sum() == Approx(0.0).margin(1e-8));

  fmt::print("Methane Hirshfeld charges:\n{}\n", format_matrix(charges));

  // Test the convenience function
  occ::Vec charges2 = occ::dft::calculate_hirshfeld_charges(basis, mo);
  REQUIRE(charges2.size() == 5);
  REQUIRE(all_close(charges, charges2));
}

TEST_CASE("Hirshfeld multipoles for water", "[hirshfeld]") {
  auto mol = make_h2o_for_hirshfeld();
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(mol.atoms(), "STO-3G");
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  double energy = scf.compute_scf_energy();
  auto mo = scf.molecular_orbitals();

  // Calculate multipoles up to hexadecapoles (L=4)
  occ::dft::HirshfeldPartition hirshfeld(basis, 4);
  auto multipoles = hirshfeld.calculate_multipoles(mo);

  // Check the number of atoms
  REQUIRE(multipoles.size() == 3);

  // Check that charges match the monopole component
  auto charges = hirshfeld.charges();
  for (size_t i = 0; i < charges.size(); i++) {
    REQUIRE(charges(i) == Approx(multipoles[i].components[0]));
  }

  // Verify that the dipole components for hydrogen atoms are non-zero
  // (due to the asymmetric electron distribution)
  REQUIRE(std::abs(multipoles[1].components[3]) >
          0.01); // H1 z-component should be significant
  REQUIRE(std::abs(multipoles[2].components[3]) >
          0.01); // H2 z-component should be significant

  // Check that quadrupole components follow expected symmetry for water
  // The oxygen atom should have a negative Qzz component due to the lone pairs
  REQUIRE(multipoles[0].components[9] < 0); // Oxygen Qzz

  // Test the convenience function
  auto multipoles2 = occ::dft::calculate_hirshfeld_multipoles(basis, mo, 4);
  REQUIRE(multipoles2.size() == 3);

  // Check that results match between class and convenience function
  for (size_t i = 0; i < multipoles.size(); i++) {
    for (size_t j = 0; j < multipoles[i].components.size(); j++) {
      REQUIRE(multipoles[i].components[j] ==
              Approx(multipoles2[i].components[j]));
    }
  }

  fmt::print("Water Hirshfeld multipoles (oxygen atom):\n{}\n",
             multipoles[0].to_string());
}

TEST_CASE("Hirshfeld multipoles for methane", "[hirshfeld]") {
  auto mol = make_ch4_for_hirshfeld();
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(mol.atoms(), "STO-3G");
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  double energy = scf.compute_scf_energy();
  auto mo = scf.molecular_orbitals();

  // Calculate multipoles up to hexadecapoles (L=4)
  occ::dft::HirshfeldPartition hirshfeld(basis, 4);
  auto multipoles = hirshfeld.calculate_multipoles(mo);

  // Check the number of atoms
  REQUIRE(multipoles.size() == 5);

  // Print all hydrogen dipole values for inspection
  fmt::print("Hydrogen dipole Z components:\n");
  fmt::print("H1: {}\n", multipoles[1].components[3]);
  fmt::print("H2: {}\n", multipoles[2].components[3]);
  fmt::print("H3: {}\n", multipoles[3].components[3]);
  fmt::print("H4: {}\n", multipoles[4].components[3]);

  // Check the magnitude of dipole components, they should be significant
  REQUIRE(std::abs(multipoles[1].components[3]) > 0.01);
  REQUIRE(std::abs(multipoles[2].components[3]) > 0.01);
  REQUIRE(std::abs(multipoles[3].components[3]) > 0.01);
  REQUIRE(std::abs(multipoles[4].components[3]) > 0.01);

  // Due to the symmetric nature of methane, the carbon atom should have small
  // dipole components Note: These might not be exactly zero due to numerical
  // integration
  REQUIRE(std::abs(multipoles[0].components[1]) < 0.05); // C Dx
  REQUIRE(std::abs(multipoles[0].components[2]) < 0.05); // C Dy
  REQUIRE(std::abs(multipoles[0].components[3]) < 0.05); // C Dz

  fmt::print("Methane Hirshfeld multipoles (carbon atom):\n{}\n",
             multipoles[0].to_string());
}

TEST_CASE("LDA gradient HF", "[dft_gradient]") {
  // Set up a water molecule and basis
  occ::Mat3N pos(3, 2);
  pos.setZero();
  pos(2, 0) = 1.0;
  occ::IVec atomic_numbers(2);
  atomic_numbers << 1, 9;
  occ::core::Molecule mol(atomic_numbers, pos);

  // Create basis and DFT object
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(mol.atoms(), "sto-3g");
  basis.set_pure(true);
  occ::dft::DFT dft("lda_x", basis);

  // Run SCF calculation
  occ::qm::SCF<occ::dft::DFT> scf(dft);
  double energy = scf.compute_scf_energy();
  auto mo = scf.molecular_orbitals();

  // Compute fock gradient
  auto fock_grad = dft.compute_fock_gradient(mo);

  // Basic checks
  REQUIRE(fock_grad.x.rows() == basis.nbf());
  REQUIRE(fock_grad.y.cols() == basis.nbf());
  REQUIRE(fock_grad.z.cols() == basis.nbf());

  // Compute full gradient
  occ::qm::GradientEvaluator<occ::dft::DFT> evaluator(dft);
  auto gradient = evaluator(mo);
  occ::Mat3N expected(3, 2);
  expected << 0.0, 0.0, 0.0, 0.0, -0.0093513386, 0.0093534193;
  fmt::print("Expected:\n{}\n", format_matrix(expected));
  fmt::print("Found:\n{}\n", format_matrix(gradient));
  fmt::print("Diff:\n{}\n", format_matrix(gradient - expected));

  REQUIRE(occ::util::all_close(expected, gradient, 1e-3, 1e-3));
}

TEST_CASE("DFT gradient for water", "[dft_gradient]") {
  // Set up a water molecule and basis
  occ::Vec3 O{0.0, 0.0, 0.0};
  occ::Vec3 H1{0.0, -0.757, 0.587};
  occ::Vec3 H2{0.0, 0.757, 0.587};
  occ::Mat3N pos(3, 3);
  pos << O(0), H1(0), H2(0), O(1), H1(1), H2(1), O(2), H1(2), H2(2);
  occ::IVec atomic_numbers(3);
  atomic_numbers << 8, 1, 1;
  occ::core::Molecule mol(atomic_numbers, pos);

  // Create basis and DFT object
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(mol.atoms(), "6-31G");
  basis.set_pure(true);
  occ::dft::DFT dft("b3lyp", basis);

  // Run SCF calculation
  occ::qm::SCF<occ::dft::DFT> scf(dft);
  double energy = scf.compute_scf_energy();
  auto mo = scf.molecular_orbitals();

  // Compute fock gradient
  auto fock_grad = dft.compute_fock_gradient(mo);

  // Basic checks
  REQUIRE(fock_grad.x.rows() == basis.nbf());
  REQUIRE(fock_grad.y.cols() == basis.nbf());
  REQUIRE(fock_grad.z.cols() == basis.nbf());

  // Compute full gradient
  occ::qm::GradientEvaluator<occ::dft::DFT> evaluator(dft);
  auto gradient = evaluator(mo);
  occ::Mat3N expected(3, 3);
  expected << 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000,
      0.0211113106, -0.0211113106, 0.0121560609, -0.0060818084, -0.0060818084;

  fmt::print("Expected:\n{}\n", format_matrix(expected));
  fmt::print("Found:\n{}\n", format_matrix(gradient));

  fmt::print("Diff:\n{}\n", format_matrix(gradient - expected));

  REQUIRE(occ::util::all_close(expected, gradient, 1e-3, 1e-3));
}

TEST_CASE("wB97X range-separated gradient for water", "[dft_gradient][wb97x]") {
  // Water molecule geometry (same as used in reference calculations)
  occ::Vec3 O{0.000000000, 0.000000000, 0.117176000};
  occ::Vec3 H1{0.000000000, 0.755453000, -0.468704000};
  occ::Vec3 H2{0.000000000, -0.755453000, -0.468704000};
  occ::Mat3N pos(3, 3);
  pos << O(0), H1(0), H2(0), O(1), H1(1), H2(1), O(2), H1(2), H2(2);
  occ::IVec atomic_numbers(3);
  atomic_numbers << 8, 1, 1;
  occ::core::Molecule mol(atomic_numbers, pos);

  // Create basis and DFT object with wB97X functional
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(mol.atoms(), "3-21G");
  basis.set_pure(true);  // Use spherical harmonics to match ORCA
  occ::dft::DFT dft("wb97x", basis);

  // Run SCF calculation
  occ::qm::SCF<occ::dft::DFT> scf(dft);
  double energy = scf.compute_scf_energy();
  auto mo = scf.molecular_orbitals();

  // Verify range-separated parameters are loaded
  auto rs_params = dft.range_separated_parameters();
  REQUIRE(rs_params.omega != 0.0); // wB97X should have omega parameter
  fmt::print("wB97X parameters: ω={:.6f}, α={:.6f}, β={:.6f}\n", 
             rs_params.omega, rs_params.alpha, rs_params.beta);

  // Compute gradients
  occ::qm::GradientEvaluator<occ::dft::DFT> evaluator(dft);
  auto gradient = evaluator(mo);

  // Expected gradients from ORCA 6.1.0 wB97X/3-21G NORI NOCOSX calculation
  // Format: gradient(coord, atom) where coord=0,1,2 for x,y,z and atom=0,1,2 for O,H1,H2
  occ::Mat3N expected(3, 3);
  expected << -0.000000000,  0.000000000,  0.000000000,    // x: O, H1, H2
              -0.000000000, -0.026426576,  0.026426576,    // y: O, H1, H2
              -0.029214261,  0.014607130,  0.014607130;    // z: O, H1, H2

  fmt::print("wB97X gradient calculation:\n");
  fmt::print("Expected:\n{}\n", format_matrix(expected));
  fmt::print("Found:\n{}\n", format_matrix(gradient));
  fmt::print("Diff:\n{}\n", format_matrix(gradient - expected));

  // Check dimensions
  REQUIRE(gradient.rows() == 3);
  REQUIRE(gradient.cols() == 3);
  REQUIRE(std::isfinite(gradient.sum()));
  
  // Compare with ORCA reference values (allow for small differences due to integral implementations)
  REQUIRE(occ::util::all_close(expected, gradient, 1e-3, 1e-3));
}

// Voronoi charge tests

TEST_CASE("Voronoi basic functionality", "[voronoi]") {
  auto mol = make_h2o_for_hirshfeld();
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(mol.atoms(), "STO-3G");
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  scf.compute_scf_energy();
  auto mo = scf.molecular_orbitals();

  // Test class interface
  occ::dft::VoronoiPartition voronoi(basis);
  occ::Vec charges = voronoi.calculate(mo);
  
  // Test convenience function (should give identical results)
  occ::Vec charges2 = occ::dft::calculate_voronoi_charges(basis, mo);

  // Basic validation
  REQUIRE(charges.size() == 3);
  REQUIRE(charges(0) < 0);  // Oxygen negative
  REQUIRE(charges(1) > 0);  // Hydrogens positive  
  REQUIRE(charges(2) > 0);
  REQUIRE(charges.sum() == Approx(0.0).margin(1e-8));  // Charge conservation
  REQUIRE(all_close(charges, charges2));  // Class vs function consistency
  
  auto volumes = voronoi.atom_volumes();
  REQUIRE(volumes.size() == 3);
  REQUIRE((volumes.array() > 0).all());  // All volumes positive
}

TEST_CASE("Voronoi VDW scaling and temperature effects", "[voronoi]") {
  auto mol = make_ch4_for_hirshfeld();
  occ::qm::AOBasis basis = occ::qm::AOBasis::load(mol.atoms(), "STO-3G");
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  scf.compute_scf_energy();
  auto mo = scf.molecular_orbitals();

  // Test pure geometric Voronoi
  occ::dft::VoronoiPartition voronoi_geom(basis, 0, 0.1, false);
  occ::Vec charges_geom = voronoi_geom.calculate(mo);
  
  // Test VDW-scaled Voronoi with optimized temperature
  occ::dft::VoronoiPartition voronoi_vdw(basis, 0, 0.37, true);
  occ::Vec charges_vdw = voronoi_vdw.calculate(mo);

  // Basic validation for both methods
  REQUIRE(charges_geom.size() == 5);
  REQUIRE(charges_vdw.size() == 5);
  REQUIRE(charges_geom.sum() == Approx(0.0).margin(1e-8));
  REQUIRE(charges_vdw.sum() == Approx(0.0).margin(1e-8));
  
  // Check symmetry preservation for methane hydrogens
  auto h_avg_geom = (charges_geom(1) + charges_geom(2) + charges_geom(3) + charges_geom(4)) / 4.0;
  auto h_avg_vdw = (charges_vdw(1) + charges_vdw(2) + charges_vdw(3) + charges_vdw(4)) / 4.0;
  
  for (int i = 1; i <= 4; i++) {
    REQUIRE(charges_geom(i) == Approx(h_avg_geom).margin(1e-2));
    REQUIRE(charges_vdw(i) == Approx(h_avg_vdw).margin(1e-2));
  }
  
  // VDW and geometric should produce different results
  REQUIRE(!all_close(charges_geom, charges_vdw, 1e-3));
}

