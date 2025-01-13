#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/atom.h>
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/dft/grid.h>
#include <occ/dft/lebedev.h>
#include <occ/dft/nonlocal_correlation.h>
#include <occ/dft/seminumerical_exchange.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/shell.h>
#include <vector>

// DFT

using occ::format_matrix;

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

TEST_CASE("Becke partitioned atom grid H2", "[grid]") {
  auto atom = occ::dft::generate_atom_grid(1, 302, 50);
  // fmt::print("Atom grid:\n{}\n", atom.points.transpose());
  fmt::print("Sum weights\n{}\n", atom.weights.array().sum());
  fmt::print("Shape: {} {}\n", atom.points.rows(), atom.points.cols());
  fmt::print("Max weight\n{}\n", atom.weights.maxCoeff());

  /*
  BENCHMARK("Atom Carbon") {
      return occ::dft::generate_atom_grid(6);
  };

  BENCHMARK("Atom Carbon 590") {
      return occ::dft::generate_atom_grid(6, 590);
  };
*/
  std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                     {1, 0.0, 0.0, 1.39839733}};
  auto basis = occ::qm::AOBasis::load(atoms, "sto-3g");
  occ::dft::MolecularGrid mgrid(basis);

  auto grid = mgrid.generate_partitioned_atom_grid(0);
  // fmt::print("{}\n{}\n", grid.points, grid.atomic_number);
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

  occ::io::BeckeGridSettings settings;
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
