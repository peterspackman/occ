#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/atom.h>
#include <occ/core/molecule.h>
#include <occ/core/multipole.h>
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/numint/grid_types.h>
#include <occ/numint/grid_utils.h>
#include <occ/dft/hirshfeld.h>
#include <occ/dft/voronoi_charges.h>
#include <occ/numint/lebedev.h>
#include <occ/numint/molecular_grid.h>
#include <occ/dft/nonlocal_correlation.h>
#include <occ/dft/seminumerical_exchange.h>
#include <occ/dft/spatial_grid_hierarchy.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/gto/shell.h>
#include <occ/qm/wavefunction.h>
#include <vector>
#include <chrono>

// DFT

using Catch::Approx;
using occ::format_matrix;
using occ::Mat;
using occ::Mat3N;
using occ::Vec;
using occ::Vec3;

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
  auto basis = occ::gto::AOBasis::load(atoms, "sto-3g");
  occ::dft::MolecularGrid mgrid(basis);

  auto grid = mgrid.get_partitioned_atom_grid(0);
}

// Seminumerical Exchange

TEST_CASE("Water seminumerical exchange approximation", "[scf]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};
  auto basis = occ::gto::AOBasis::load(atoms, "6-31G");
  basis.set_pure(false);
  auto hf = occ::qm::HartreeFock(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  double e = scf.compute_scf_energy();

  occ::io::GridSettings settings = occ::io::GridSettings::for_sgx(50);
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
  double expected = 0.008940471;
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
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "STO-3G");
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
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "STO-3G");
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
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "STO-3G");
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
              Approx(multipoles2[i].components[j]).epsilon(1e-12).margin(1e-12));
    }
  }

  fmt::print("Water Hirshfeld multipoles (oxygen atom):\n{}\n",
             multipoles[0].to_string());
}

TEST_CASE("Hirshfeld multipoles for methane", "[hirshfeld]") {
  auto mol = make_ch4_for_hirshfeld();
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "STO-3G");
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
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "sto-3g");
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
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "6-31G");
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
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "3-21G");
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
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "STO-3G");
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
  occ::gto::AOBasis basis = occ::gto::AOBasis::load(mol.atoms(), "STO-3G");
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

TEST_CASE("COSX shell extents", "[cosx][screening]") {
  // Create a simple water molecule
  std::vector<occ::core::Atom> atoms{{8, 0.0, 0.0, 0.0},
                                     {1, 0.0, -0.757, 0.587},
                                     {1, 0.0, 0.757, 0.587}};

  // Load def2-svp basis for realistic extents
  auto basis = occ::gto::AOBasis::load(atoms, "def2-svp");

  // Create SemiNumericalExchange instance with COSX-appropriate grid
  occ::io::GridSettings settings = occ::io::GridSettings::for_sgx(50);
  occ::dft::cosx::SemiNumericalExchange sgx(basis, settings);

  // Verify that shell extents are populated in the engine's basis
  // Note: the extents are set on the internal m_basis, accessed via engine().aobasis()
  const auto &shells = sgx.engine().aobasis().shells();
  REQUIRE(shells.size() > 0);

  fmt::print("COSX shell extent verification:\n");
  fmt::print("{:>5} {:>8} {:>12}\n", "Shell", "L", "Extent");

  size_t num_nonzero = 0;
  for (size_t i = 0; i < shells.size(); i++) {
    const auto &sh = shells[i];
    fmt::print("{:5d} {:>8} {:12.6f}\n", i, sh.symbol(), sh.extent);

    // All extents should be positive (non-zero)
    REQUIRE(sh.extent > 0.0);
    if (sh.extent > 0.0) num_nonzero++;

    // For def2-svp, extents should be reasonable (typically 5-30 Bohr)
    // Very diffuse functions might be larger, but should be < 100 Bohr
    REQUIRE(sh.extent < 100.0);

    // Extents should be at least 1 Bohr for any reasonable basis
    REQUIRE(sh.extent > 1.0);
  }

  // All shells should have non-zero extents
  REQUIRE(num_nonzero == shells.size());

  fmt::print("All {} shells have valid extents\n", shells.size());
}

TEST_CASE("COSX batch bounding sphere", "[cosx][screening]") {
    using namespace occ::dft::cosx;

    // Test with known points
    occ::Mat3N pts(3, 4);
    pts.col(0) << 0.0, 0.0, 0.0;
    pts.col(1) << 2.0, 0.0, 0.0;
    pts.col(2) << 0.0, 2.0, 0.0;
    pts.col(3) << 0.0, 0.0, 2.0;

    auto info = compute_batch_info(pts);

    // Center should be at (0.5, 0.5, 0.5)
    REQUIRE(info.center(0) == Approx(0.5));
    REQUIRE(info.center(1) == Approx(0.5));
    REQUIRE(info.center(2) == Approx(0.5));

    // Radius should be distance from center to corners
    double expected_radius = std::sqrt(0.5*0.5 + 0.5*0.5 + 1.5*1.5);  // to (0,0,2)
    REQUIRE(info.radius == Approx(expected_radius).epsilon(1e-10));

    fmt::print("COSX batch bounding sphere test:\n");
    fmt::print("  Center: ({:.6f}, {:.6f}, {:.6f})\n", info.center(0), info.center(1), info.center(2));
    fmt::print("  Radius: {:.6f}\n", info.radius);
    fmt::print("  Expected radius: {:.6f}\n", expected_radius);
}

TEST_CASE("COSX shell pair screening", "[cosx][screening]") {
    using namespace occ::dft::cosx;
    using namespace occ::qm;

    // Create test shells at different positions
    // Shell 1: at origin with extent 5.0
    // Shell 2: at (3, 0, 0) with extent 5.0
    // Shell 3: at (20, 0, 0) with extent 5.0 (far away)

    std::vector<Shell> shells;
    shells.push_back(Shell(0, {1.0}, {{1.0}}, {0.0, 0.0, 0.0}));
    shells.push_back(Shell(0, {1.0}, {{1.0}}, {3.0, 0.0, 0.0}));
    shells.push_back(Shell(0, {1.0}, {{1.0}}, {20.0, 0.0, 0.0}));

    // Set screening extents manually for test (simulating 1e-6 threshold)
    occ::Vec screening_extents(3);
    screening_extents << 5.0, 5.0, 5.0;

    // Create batch centered at (1, 0, 0) with radius 2.0
    GridBatchInfo batch;
    batch.center = occ::Vec3(1.0, 0.0, 0.0);
    batch.radius = 2.0;

    auto screened = screen_shell_pairs(batch, shells, screening_extents);

    // Shell pairs:
    // (0,0) idx=0: shell 0 at origin, near batch
    // (1,0) idx=1: shells 0 and 1, both near
    // (1,1) idx=2: shell 1 at (3,0,0), near batch
    // (2,0) idx=3: shell 2 at (20,0,0) is far, shell 0 near -> list2
    // (2,1) idx=4: shell 2 far, shell 1 near -> list2
    // (2,2) idx=5: shell 2 far -> list3 (skipped)

    // list1 should have pairs where both shells near: 0, 1, 2
    REQUIRE(screened.list1.size() == 3);

    // list2 should have pairs with one near, one far: 3, 4
    REQUIRE(screened.list2.size() == 2);

    // Total evaluated = list1 + list2 = 5, skipped = 1 (pair 5)
    REQUIRE(screened.list1.size() + screened.list2.size() == 5);

    fmt::print("COSX shell pair screening test:\n");
    fmt::print("  Batch center: ({:.1f}, {:.1f}, {:.1f}), radius: {:.1f}\n",
               batch.center(0), batch.center(1), batch.center(2), batch.radius);
    fmt::print("  List1 (both near): {} pairs\n", screened.list1.size());
    fmt::print("  List2 (one near):  {} pairs\n", screened.list2.size());
    fmt::print("  List3 (skipped):   {} pairs\n", 6 - screened.list1.size() - screened.list2.size());
}

TEST_CASE("SHARK-style shell list screening", "[cosx][shark][screening]") {
    using namespace occ::dft::cosx;
    using occ::gto::Shell;

    // Create test shells at different distances
    // Shell 0: at origin with extent 5.0
    // Shell 1: at (3, 0, 0) with extent 5.0
    // Shell 2: at (20, 0, 0) with extent 5.0 (far away)

    std::vector<Shell> shells;
    shells.push_back(Shell(0, {1.0}, {{1.0}}, {0.0, 0.0, 0.0}));
    shells.push_back(Shell(0, {1.0}, {{1.0}}, {3.0, 0.0, 0.0}));
    shells.push_back(Shell(0, {1.0}, {{1.0}}, {20.0, 0.0, 0.0}));

    // Set screening extents manually for test
    occ::Vec screening_extents(3);
    screening_extents << 5.0, 5.0, 5.0;

    // Create batch centered at (1, 0, 0) with radius 2.0
    GridBatchInfo batch;
    batch.center = occ::Vec3(1.0, 0.0, 0.0);
    batch.radius = 2.0;

    // Test geometric screening (list-1)
    auto list1 = screen_shells_geometric(batch, shells, screening_extents);

    // Shells 0 and 1 are near, shell 2 is far
    REQUIRE(list1.size() == 2);
    REQUIRE(std::find(list1.begin(), list1.end(), 0) != list1.end());
    REQUIRE(std::find(list1.begin(), list1.end(), 1) != list1.end());

    // Test density-based screening (list-2)
    // Create fake Fg matrix where shell 0 has large values, shell 1 has small values
    std::vector<int> first_bf = {0, 1, 2};  // Each shell has 1 bf (s-type)
    const int npts = 10;
    occ::Mat Fg = occ::Mat::Zero(npts, 3);
    Fg.col(0).setConstant(1.0);    // Shell 0: significant F
    Fg.col(1).setConstant(1e-12);  // Shell 1: negligible F
    Fg.col(2).setConstant(1e-15);  // Shell 2: negligible F

    auto list2 = screen_shells_density(list1, Fg, shells, first_bf, 1e-10);

    // Only shell 0 should pass density screening
    REQUIRE(list2.size() == 1);
    REQUIRE(list2[0] == 0);

    // Test full build_shell_lists
    occ::Mat Fg2 = occ::Mat::Zero(npts, 3);
    Fg2.col(0).setConstant(1.0);   // Shell 0: significant
    Fg2.col(1).setConstant(0.5);   // Shell 1: significant
    Fg2.col(2).setConstant(1e-15); // Shell 2: negligible (also geometrically far)

    auto lists = build_shell_lists(batch, Fg2, shells, first_bf, screening_extents);

    // list1: geometric - shells 0, 1 near
    REQUIRE(lists.list1.size() == 2);

    // list2: density - shells 0, 1 have significant F
    REQUIRE(lists.list2.size() == 2);

    // list3: overlap - subset of list2 with tighter overlap criterion
    // Both shells 0 and 1 should pass since they're close
    REQUIRE(lists.list3.size() >= 1);

    fmt::print("SHARK shell list screening test:\n");
    fmt::print("  Batch center: ({:.1f}, {:.1f}, {:.1f}), radius: {:.1f}\n",
               batch.center(0), batch.center(1), batch.center(2), batch.radius);
    fmt::print("  List-1 (geometric):  {} shells\n", lists.list1.size());
    fmt::print("  List-2 (density):    {} shells\n", lists.list2.size());
    fmt::print("  List-3 (overlap):    {} shells\n", lists.list3.size());
}

TEST_CASE("SpatialGridHierarchy Morton ordering", "[spatial][hierarchy]") {
    using namespace occ::dft;

    // Create test points in a grid pattern
    const int n = 1000;
    Mat3N points(3, n);
    Vec weights(n);

    // Random-ish points in a cube
    for (int i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / n;
        points(0, i) = std::sin(t * 10.0) * 5.0;
        points(1, i) = std::cos(t * 7.0) * 5.0;
        points(2, i) = std::sin(t * 3.0 + 1.0) * 5.0;
        weights(i) = 1.0;
    }

    SpatialHierarchySettings settings;
    settings.target_leaf_size = 64;

    SpatialGridHierarchy hierarchy(points, weights, settings);

    SECTION("All points preserved") {
        REQUIRE(hierarchy.sorted_points().cols() == n);
        REQUIRE(hierarchy.sorted_weights().size() == n);

        // Check that sorted weights sum equals original
        double orig_sum = weights.sum();
        double sorted_sum = hierarchy.sorted_weights().sum();
        REQUIRE(sorted_sum == Approx(orig_sum).epsilon(1e-12));
    }

    SECTION("Leaves cover all points") {
        size_t total = 0;
        for (const auto& leaf : hierarchy.leaves()) {
            total += leaf.count;
        }
        REQUIRE(total == n);
    }

    SECTION("Leaf sizes within bounds") {
        for (const auto& leaf : hierarchy.leaves()) {
            REQUIRE(leaf.count >= settings.min_leaf_size / 2);  // Allow smaller last leaf
            REQUIRE(leaf.count <= settings.max_leaf_size);
        }
    }

    SECTION("Bounding spheres contain all points") {
        for (size_t i = 0; i < hierarchy.num_leaves(); ++i) {
            auto pts = hierarchy.leaf_points(i);
            const auto& bounds = hierarchy.leaf_bounds(i);

            for (Eigen::Index j = 0; j < pts.cols(); ++j) {
                double dist = (pts.col(j) - bounds.center).norm();
                REQUIRE(dist <= bounds.radius + 1e-10);
            }
        }
    }

    SECTION("Morton ordering improves locality") {
        // Points in same leaf should be spatially close
        // Compare average intra-leaf distance vs random batching
        double avg_leaf_diameter = 0.0;
        for (size_t i = 0; i < hierarchy.num_leaves(); ++i) {
            avg_leaf_diameter += hierarchy.leaf_bounds(i).radius * 2;
        }
        avg_leaf_diameter /= hierarchy.num_leaves();

        // Should be much smaller than the total extent
        Vec3 min_pt = points.rowwise().minCoeff();
        Vec3 max_pt = points.rowwise().maxCoeff();
        double total_diameter = (max_pt - min_pt).norm();

        REQUIRE(avg_leaf_diameter < total_diameter * 0.5);  // Leaves should be compact
    }
}

TEST_CASE("SpatialGridHierarchy with real grid", "[spatial][hierarchy]") {
    using namespace occ::dft;
    using namespace occ::qm;
    using occ::core::Atom;

    // Create water molecule
    std::vector<Atom> atoms = {
        {8, 0.0, 0.0, 0.0},
        {1, 1.43, 1.11, 0.0},
        {1, -1.43, 1.11, 0.0}
    };

    auto basis = AOBasis::load(atoms, "def2-svp");

    occ::io::GridSettings settings;
    settings.max_angular_points = 50;
    settings.radial_precision = 1e-5;

    MolecularGrid grid(basis, settings);
    const auto& grid_pts = grid.get_molecular_grid_points();

    SpatialHierarchySettings hier_settings;
    hier_settings.target_leaf_size = 128;

    SpatialGridHierarchy hierarchy(grid_pts.points(), grid_pts.weights(), hier_settings);

    REQUIRE(hierarchy.num_leaves() > 0);
    REQUIRE(hierarchy.sorted_points().cols() == grid_pts.points().cols());

    // Verify all bounding spheres are valid
    for (size_t i = 0; i < hierarchy.num_leaves(); ++i) {
        const auto& bounds = hierarchy.leaf_bounds(i);
        REQUIRE(bounds.radius >= 0.0);
        REQUIRE(bounds.radius < 100.0);  // Reasonable for water
    }
}

TEST_CASE("MolecularGridPoints hierarchy integration", "[spatial][hierarchy]") {
    using namespace occ::dft;
    using namespace occ::qm;
    using occ::core::Atom;

    // Create water molecule
    std::vector<Atom> atoms = {
        {8, 0.0, 0.0, 0.0},
        {1, 1.43, 1.11, 0.0},
        {1, -1.43, 1.11, 0.0}
    };

    auto basis = AOBasis::load(atoms, "def2-svp");

    occ::io::GridSettings settings;
    settings.max_angular_points = 50;
    settings.radial_precision = 1e-5;

    MolecularGrid grid(basis, settings);
    const auto& grid_pts = grid.get_molecular_grid_points();

    SECTION("Hierarchy is lazily constructed") {
        REQUIRE_FALSE(grid_pts.has_hierarchy());

        const auto& hier = grid_pts.get_hierarchy();

        REQUIRE(grid_pts.has_hierarchy());
        REQUIRE(hier.num_leaves() > 0);
    }

    SECTION("Hierarchy is cached") {
        const auto& hier1 = grid_pts.get_hierarchy();
        const auto& hier2 = grid_pts.get_hierarchy();

        // Should return same object (pointer equality)
        REQUIRE(&hier1 == &hier2);
    }

    SECTION("Hierarchy preserves all points") {
        const auto& hier = grid_pts.get_hierarchy();

        REQUIRE(hier.sorted_points().cols() == grid_pts.points().cols());
        REQUIRE(hier.sorted_weights().sum() == Approx(grid_pts.weights().sum()).epsilon(1e-12));
    }

    SECTION("Custom settings work") {
        SpatialHierarchySettings custom;
        custom.target_leaf_size = 64;

        // Clear any existing hierarchy first
        const_cast<MolecularGridPoints&>(grid_pts).clear_hierarchy();

        const auto& hier = grid_pts.get_hierarchy(custom);

        // With smaller leaf size, should have more leaves
        size_t expected_min_leaves = grid_pts.points().cols() / custom.max_leaf_size;
        REQUIRE(hier.num_leaves() >= expected_min_leaves);
    }
}

TEST_CASE("HartreeFock with COSX exchange", "[hf][cosx]") {
    // Create a simple water molecule
    std::vector<occ::core::Atom> atoms{{8, 0.0, 0.0, 0.0},
                                       {1, 0.0, -0.757, 0.587},
                                       {1, 0.0, 0.757, 0.587}};

    auto basis = occ::gto::AOBasis::load(atoms, "6-31G");

    SECTION("COSX can be enabled") {
        occ::qm::HartreeFock hf(basis);
        REQUIRE_FALSE(hf.using_cosx());

        hf.set_cosx_exchange(occ::io::COSXGridLevel::Grid1);
        REQUIRE(hf.using_cosx());

        fmt::print("COSX enabled successfully\n");
    }

    SECTION("COSX produces reasonable energies") {
        // Exact HF
        occ::qm::HartreeFock hf_exact(basis);
        occ::qm::SCF<occ::qm::HartreeFock> scf_exact(hf_exact);
        double e_exact = scf_exact.compute_scf_energy();

        // COSX HF with Grid3 (finest)
        occ::qm::HartreeFock hf_cosx(basis);
        hf_cosx.set_cosx_exchange(occ::io::COSXGridLevel::Grid3);
        occ::qm::SCF<occ::qm::HartreeFock> scf_cosx(hf_cosx);
        double e_cosx = scf_cosx.compute_scf_energy();

        fmt::print("Energy (exact): {:.10f} Hartree\n", e_exact);
        fmt::print("Energy (COSX Grid3): {:.10f} Hartree\n", e_cosx);
        fmt::print("Difference: {:.6e} Hartree\n", std::abs(e_exact - e_cosx));

        // COSX should be close to exact (within 1 mHartree for Grid3 with 6-31G)
        REQUIRE(std::abs(e_exact - e_cosx) < 1e-3);
    }

    SECTION("Different COSX grid levels") {
        occ::qm::HartreeFock hf_exact(basis);
        occ::qm::SCF<occ::qm::HartreeFock> scf_exact(hf_exact);
        double e_exact = scf_exact.compute_scf_energy();

        // Test all grid levels
        for (auto level : {occ::io::COSXGridLevel::Grid1,
                           occ::io::COSXGridLevel::Grid2,
                           occ::io::COSXGridLevel::Grid3}) {
            occ::qm::HartreeFock hf_cosx(basis);
            hf_cosx.set_cosx_exchange(level);
            occ::qm::SCF<occ::qm::HartreeFock> scf_cosx(hf_cosx);
            double e_cosx = scf_cosx.compute_scf_energy();

            fmt::print("Energy (COSX {}): {:.10f} Hartree (error: {:.6e})\n",
                       occ::io::cosx_grid_level_to_string(level),
                       e_cosx, std::abs(e_exact - e_cosx));

            // All grid levels should give reasonable results (< 10 mHartree)
            REQUIRE(std::abs(e_exact - e_cosx) < 0.01);
        }
    }
}

TEST_CASE("UHF with COSX exchange", "[uhf][cosx]") {
    // Use water for closed-shell UHF test (should match RHF)
    std::vector<occ::core::Atom> atoms{{8, 0.0, 0.0, 0.116868},
                                       {1, 0.0, 0.763239, -0.467473},
                                       {1, 0.0, -0.763239, -0.467473}};

    auto basis = occ::gto::AOBasis::load(atoms, "STO-3G");

    SECTION("Restricted COSX for reference") {
        // Exact RHF
        occ::qm::HartreeFock hf_exact(basis);
        occ::qm::SCF<occ::qm::HartreeFock> scf_exact(hf_exact, occ::qm::SpinorbitalKind::Restricted);
        double e_exact = scf_exact.compute_scf_energy();

        // RHF with COSX Grid3
        occ::qm::HartreeFock hf_cosx(basis);
        hf_cosx.set_cosx_exchange(occ::io::COSXGridLevel::Grid3);
        occ::qm::SCF<occ::qm::HartreeFock> scf_cosx(hf_cosx, occ::qm::SpinorbitalKind::Restricted);
        double e_cosx = scf_cosx.compute_scf_energy();

        fmt::print("RHF water (STO-3G):\n");
        fmt::print("  Energy (exact):       {:.10f} Hartree\n", e_exact);
        fmt::print("  Energy (COSX Grid3):  {:.10f} Hartree\n", e_cosx);
        fmt::print("  Difference: {:.6e} Hartree\n", std::abs(e_exact - e_cosx));

        REQUIRE(std::abs(e_exact - e_cosx) < 1e-3);
    }

    SECTION("UHF COSX closed-shell (should match RHF)") {
        // Exact UHF on closed-shell water
        occ::qm::HartreeFock hf_exact(basis);
        occ::qm::SCF<occ::qm::HartreeFock> scf_exact(hf_exact, occ::qm::SpinorbitalKind::Unrestricted);
        double e_exact = scf_exact.compute_scf_energy();

        // UHF with COSX Grid3
        occ::qm::HartreeFock hf_cosx(basis);
        hf_cosx.set_cosx_exchange(occ::io::COSXGridLevel::Grid3);
        occ::qm::SCF<occ::qm::HartreeFock> scf_cosx(hf_cosx, occ::qm::SpinorbitalKind::Unrestricted);
        double e_cosx = scf_cosx.compute_scf_energy();

        fmt::print("UHF water closed-shell (STO-3G):\n");
        fmt::print("  Energy (exact):       {:.10f} Hartree\n", e_exact);
        fmt::print("  Energy (COSX Grid3):  {:.10f} Hartree\n", e_cosx);
        fmt::print("  Difference: {:.6e} Hartree\n", std::abs(e_exact - e_cosx));

        // COSX should be close to exact (within 1 mHartree)
        REQUIRE(std::abs(e_exact - e_cosx) < 1e-3);
    }
}

TEST_CASE("GHF with COSX exchange", "[ghf][cosx]") {
    // Use water for closed-shell GHF test (should match RHF)
    std::vector<occ::core::Atom> atoms{{8, 0.0, 0.0, 0.116868},
                                       {1, 0.0, 0.763239, -0.467473},
                                       {1, 0.0, -0.763239, -0.467473}};

    auto basis = occ::gto::AOBasis::load(atoms, "STO-3G");

    SECTION("GHF COSX closed-shell (should match RHF)") {
        // Exact GHF on closed-shell water
        occ::qm::HartreeFock hf_exact(basis);
        occ::qm::SCF<occ::qm::HartreeFock> scf_exact(hf_exact, occ::qm::SpinorbitalKind::General);
        double e_exact = scf_exact.compute_scf_energy();

        // GHF with COSX Grid3
        occ::qm::HartreeFock hf_cosx(basis);
        hf_cosx.set_cosx_exchange(occ::io::COSXGridLevel::Grid3);
        occ::qm::SCF<occ::qm::HartreeFock> scf_cosx(hf_cosx, occ::qm::SpinorbitalKind::General);
        double e_cosx = scf_cosx.compute_scf_energy();

        fmt::print("GHF water closed-shell (STO-3G):\n");
        fmt::print("  Energy (exact):       {:.10f} Hartree\n", e_exact);
        fmt::print("  Energy (COSX Grid3):  {:.10f} Hartree\n", e_cosx);
        fmt::print("  Difference: {:.6e} Hartree\n", std::abs(e_exact - e_cosx));

        // COSX should be close to exact (within 1 mHartree)
        REQUIRE(std::abs(e_exact - e_cosx) < 1e-3);
    }
}

