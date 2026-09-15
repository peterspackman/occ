#include "occ/core/data_directory.h"
#include <fmt/core.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <nlohmann/json.hpp>
#include <occ/core/element.h>
#include <occ/core/format_matrix.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/dma/dma.h>
#include <occ/dma/linear_multipole_calculator.h>
#include <occ/driver/dma_driver.h>
#include <occ/mults/dma_force_field.h>
#include <occ/qm/io/fchkreader.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include "test_utils.h"
#include <set>
#include <string>

using namespace occ;
using Catch::Approx;
using namespace occ::dma;
namespace fs = std::filesystem;

const std::string h2_contents = read_file(fs::path(occ::get_data_directory()) / "tests" / "h2.fchk" );

const std::string unrestricted_fchk_contents = read_file(fs::path(occ::get_data_directory()) / "tests" / "water_unrestricted.fchk");

TEST_CASE("DMA linear", "[dma]") {
  using namespace occ::qm;
  using namespace occ::io;
  std::istringstream fchk(h2_contents);
  FchkReader reader(fchk);

  occ::qm::Wavefunction wfn(reader);

  SECTION("Basis normalization") {
    const Vec3 coeffs_dma(0.27693435, 0.26783885, 0.08347367);
    const auto &shell = wfn.basis.shells()[0];
    for (int i = 0; i < 3; i++) {
      REQUIRE(shell.coeff_normalized_dma(0, i) ==
              Approx(coeffs_dma(i)).margin(1e-6));
    }
  }

  // Setup settings for the linear calculator
  LinearDMASettings settings;
  settings.max_rank = 2;
  settings.include_nuclei = true;
  settings.use_slices = false;

  // Create and use the linear multipole calculator
  LinearMultipoleCalculator calculator(wfn, settings);
  auto result = calculator.calculate();

  // Check the result
  REQUIRE(result.size());

  Mat expected(3, 2);
  expected << 0.0, 0.0, 0.19113723, -0.19113723, -0.11109289, -0.11109289;

  for (int site = 0; site < 2; site++) {
    const auto &m = result[site];
    for (int term = 0; term < 3; term++) {
      REQUIRE(m.q(term) == Approx(expected(term, site)).margin(1e-6));
    }
  }
}

const char *h2o_contents =
    R"(fchk produced by OCC                                                    
SP         HF                                                      6-31G
Number of atoms                            I                3
Charge                                     I                0
Multiplicity                               I                1
Number of electrons                        I               10
Number of alpha electrons                  I                5
Number of beta electrons                   I                5
Number of basis functions                  I               13
Number of independent functions            I               13
Number of point charges in /Mol/           I                0
Number of translation vectors              I                0
Atomic numbers                             I   N=           3
           8           1           1
Nuclear charges                            R   N=           3
  8.00000000e+00  1.00000000e+00  1.00000000e+00
Current cartesian coordinates              R   N=           9
 -1.32695831e+00 -1.05938613e-01  1.87882240e-02 -1.93166519e+00  1.60017435e+00
 -2.17104965e-02  4.86644350e-01  7.95980990e-02  9.86248064e-03
Force Field                                I                0
Int Atom Types                             I   N=           3
           0           0           0
MM Charges                                 R   N=           3
  0.00000000e+00  0.00000000e+00  0.00000000e+00
Integer atomic weights                     I   N=           3
          16           1           1
Real atomic weights                        R   N=           3
  1.59994001e+01  1.00794005e+00  1.00794005e+00
Atom residue num                           I   N=           3
           0           0           0
Number of contracted shells                I                9
Number of primitive shells                 I               22
Pure/Cartesian d shells                    I                1
Pure/Cartesian f shells                    I                1
Highest angular momentum                   I                1
Largest degree of contraction              I                6
Shell types                                I   N=           9
           0           0           1           0           1           0
           0           0           0
Number of primitives per shell             I   N=           9
           6           3           3           1           1           3
           1           3           1
Shell to atom map                          I   N=           9
           1           1           1           1           1           2
           2           3           3
Primitive exponents                        R   N=          22
  5.48467166e+03  8.25234946e+02  1.88046958e+02  5.29645000e+01  1.68975704e+01
  5.79963534e+00  1.55396162e+01  3.59993359e+00  1.01376175e+00  1.55396162e+01
  3.59993359e+00  1.01376175e+00  2.70005823e-01  2.70005823e-01  1.87311370e+01
  2.82539437e+00  6.40121692e-01  1.61277759e-01  1.87311370e+01  2.82539437e+00
  6.40121692e-01  1.61277759e-01
Contraction coefficients                   R   N=          22
  1.83107443e-03  1.39501722e-02  6.84450781e-02  2.32714336e-01  4.70192898e-01
  3.58520853e-01 -1.10777550e-01 -1.48026263e-01  1.13076702e+00  7.08742682e-02
  3.39752839e-01  7.27158577e-01  1.00000000e+00  1.00000000e+00  3.34946043e-02
  2.34726954e-01  8.13757326e-01  1.00000000e+00  3.34946043e-02  2.34726954e-01
  8.13757326e-01  1.00000000e+00
P(S=P) Contraction coefficients            R   N=          22
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00
Coordinates of each shell                  R   N=          27
 -1.32695831e+00 -1.05938613e-01  1.87882240e-02 -1.32695831e+00 -1.05938613e-01
  1.87882240e-02 -1.32695831e+00 -1.05938613e-01  1.87882240e-02 -1.32695831e+00
 -1.05938613e-01  1.87882240e-02 -1.32695831e+00 -1.05938613e-01  1.87882240e-02
 -1.93166519e+00  1.60017435e+00 -2.17104965e-02 -1.93166519e+00  1.60017435e+00
 -2.17104965e-02  4.86644350e-01  7.95980990e-02  9.86248064e-03  4.86644350e-01
  7.95980990e-02  9.86248064e-03
Virial ratio                               R      2.000058389893980E+00
SCF Energy                                 R     -7.598355029024600E+01
Alpha Orbital Energies                     R   N=          13
 -2.05618546e+01 -1.35511969e+00 -7.06370837e-01 -5.61652463e-01 -5.01574219e-01
  2.02567800e-01  2.98573567e-01  1.05126722e+00  1.16413457e+00  1.18474182e+00
  1.21796386e+00  1.37742184e+00  1.69877897e+00
Alpha MO coefficients                      R   N=         169
 -9.95787539e-01 -2.19410259e-02 -1.11608735e-03 -1.78176307e-03  4.64990459e-05
  8.03072301e-03  1.04111369e-03  1.66407438e-03 -4.34246823e-05 -1.26445438e-04
 -1.98884936e-03 -1.39696853e-04 -1.94952319e-03 -2.12429144e-01  4.69075062e-01
  5.82352345e-02  9.50382910e-02 -2.47711558e-03  4.80414592e-01  3.25816298e-02
  5.16699150e-02 -1.34895998e-03  1.40692488e-01 -8.31741296e-03  1.37969869e-01
 -7.95569462e-03 -9.16916197e-04  1.70622903e-03  4.25556541e-01 -2.72913369e-01
  5.68713248e-03  4.97412981e-03  2.28694850e-01 -1.45812434e-01  3.03533176e-03
 -2.63396195e-01 -1.25337370e-01  2.61282464e-01  1.25676898e-01 -7.62646401e-02
  1.81880692e-01 -2.95359376e-01 -4.62673853e-01  1.20878325e-02  3.07882441e-01
 -2.15851637e-01 -3.37758164e-01  8.82484714e-03 -1.43970010e-01 -8.04688637e-02
 -1.46313689e-01 -8.31039363e-02 -1.20530847e-17  1.61195673e-16  1.54423195e-03
  1.57831762e-02  6.41849259e-01 -2.76643168e-16  1.22553999e-03  1.25259121e-02
  5.09387167e-01  1.88710519e-16  2.91023702e-16 -1.10295217e-16 -5.00162066e-17
 -8.46293720e-02  1.03659039e-01  1.28468143e-01  1.94813125e-01 -5.09956885e-03
  1.17240977e+00  2.62330041e-01  4.00917980e-01 -1.04897800e-02 -5.66676055e-02
 -9.80750923e-01 -5.89507806e-02 -9.96526011e-01 -8.71489874e-04  6.93667003e-04
 -2.86125263e-01  1.82233784e-01 -3.79276615e-03  1.86505976e-02 -6.91076674e-01
  4.41933808e-01 -9.20455447e-03 -4.33891810e-02 -1.39238005e+00  4.47192302e-02
  1.36660394e+00  9.96173663e-04 -8.22179320e-03 -9.56458408e-02  4.43863972e-02
 -8.61353278e-04  1.78415213e-02 -6.14925531e-01  4.00277632e-01 -8.36343528e-03
 -9.53763272e-01  4.84528187e-01  9.90994930e-01 -5.22284699e-01 -2.02845761e-16
  2.10126580e-15 -2.31175438e-03 -2.36278149e-02 -9.60864617e-01 -2.66539139e-15
  2.49537820e-03  2.55045842e-02  1.03718658e+00 -7.71107609e-16  1.87453762e-15
 -3.22693314e-15  2.16184949e-15  4.78667184e-02 -2.40936751e-01 -3.87801840e-01
 -5.99918696e-01  1.56851135e-02  1.85159783e-01  1.90415340e-01  2.27919165e-01
 -6.06269110e-03  8.30733831e-01 -5.43799726e-01  7.83054291e-01 -5.46959765e-01
  7.29445324e-02 -4.70658837e-01  3.53531142e-01  5.74558991e-01 -1.49790620e-02
  8.93938070e-02 -6.05405609e-01 -9.94905853e-01  2.59214462e-02  5.80329451e-01
 -9.41926182e-02  5.76139068e-01 -1.15784004e-01  1.22734594e-03 -9.95321177e-03
  8.72464979e-01 -5.53363242e-01  1.15082181e-02  1.10484053e-02 -1.30124016e+00
  8.14396828e-01 -1.68954810e-02  1.07402858e-01 -9.12870350e-01 -7.71658925e-02
  8.93168901e-01  5.30643338e-02 -1.66758678e+00 -1.10140502e-01 -1.70349797e-01
  4.45391703e-03  2.75041401e+00  4.68201774e-01  7.45831921e-01 -1.94665781e-02
 -4.79401914e-01 -6.12880590e-01 -4.81651996e-01 -5.97331885e-01
Total SCF Density                          R   N=          91
  2.08507220e+00 -1.83338487e-01  5.07192639e-01  2.17516034e-02 -5.13057726e-02
  5.43461007e-01  3.42424965e-02 -7.99958142e-02  5.21518600e-02  5.95666723e-01
 -8.94360892e-04  2.09054700e-03 -6.06402802e-04  5.50019546e-03  8.24310137e-01
 -2.67072117e-01  5.62361321e-01 -1.21702197e-01 -1.96326378e-01  5.12050170e-03
  6.51358025e-01  1.65882752e-02 -4.72173045e-02  3.25949047e-01  8.11380334e-02
 -1.20521725e-03 -9.93166392e-02  1.99914824e-01  2.65188845e-02 -7.49598415e-02
  8.14702016e-02  4.02342776e-01  5.99962737e-03 -1.59757506e-01  8.25193465e-02
  2.76342582e-01 -6.92013529e-04  1.95687513e-03 -1.21338532e-03  6.00042654e-03
  6.54154102e-01  4.16740964e-03 -1.26083027e-03  5.77502537e-03  5.19128397e-01
 -3.70798774e-02  7.87266653e-02 -1.22747360e-01  3.03733803e-01 -7.17354372e-03
  4.39074076e-02 -4.91546642e-02  1.88605690e-01 -4.51958243e-03  2.19798623e-01
  1.99983702e-02 -3.74148809e-02 -6.01061020e-02  1.41300308e-01 -3.32998732e-03
 -5.88203460e-02 -2.31354812e-02  9.00433860e-02 -2.15851940e-03  8.68570868e-02
  4.45156584e-02 -3.65014519e-02  7.71109252e-02  3.24880890e-01  1.90014210e-02
 -1.24888237e-03  4.50677070e-02  1.91662282e-01  3.68981400e-02 -1.36845350e-03
 -5.66893632e-02 -4.42440700e-02  2.17423851e-01  1.97079778e-02 -3.71792225e-02
  1.55134054e-01  6.79698159e-03 -5.40377534e-04 -5.75975948e-02  9.28370817e-02
  1.86589287e-02 -6.82183768e-04 -4.45148042e-02 -1.79893688e-02  8.77980784e-02
  4.55360812e-02
)";

TEST_CASE("DMA general", "[dma]") {
  using namespace occ::qm;
  using namespace occ::io;
  std::istringstream fchk(h2o_contents);
  FchkReader reader(fchk);

  occ::qm::Wavefunction wfn(reader);

  occ::dma::DMACalculator calc(wfn);

  occ::dma::DMASettings settings;
  settings.max_rank = 2;
  settings.big_exponent = 0.0;

  calc.update_settings(settings);
  calc.set_radius_for_element(1, 0.325);

  // Test DMA calculation (analytical method)
  auto dma_result = calc.compute_multipoles();
  const auto &result = dma_result.multipoles;

  // Check the result
  REQUIRE(result.size() > 0);

  auto expected = Mat(3, 12);

  expected << -0.704577, -0.004420, 0.102103, 0.169772, -0.397276, 0.003845,
      -0.013801, 0.104311, -0.233917, 0.000000, 0.000000, 0.000000, 0.352029,
      -0.000939, -0.014779, 0.039620, -0.013593, 0.000254, -0.000950, -0.016151,
      -0.011038, 0.000000, 0.000000, 0.000000, 0.352548, -0.000192, 0.043710,
      0.003513, -0.013054, -0.000235, -0.000130, 0.017866, 0.005581, 0.000000,
      0.000000, 0.000000;

  for (int site = 0; site < 3; site++) {
    const auto &m = result[site];
    for (int term = 0; term < 12; term++) {
      REQUIRE(m.q(term) == Approx(expected(site, term)).margin(1e-6));
    }
  }
}

TEST_CASE("DMA general4", "[dma]") {
  using namespace occ::qm;
  using namespace occ::io;
  std::istringstream fchk(h2o_contents);
  FchkReader reader(fchk);

  occ::qm::Wavefunction wfn(reader);

  occ::dma::DMACalculator calc(wfn);

  occ::dma::DMASettings settings;
  settings.max_rank = 2;
  settings.big_exponent = 4.0;

  calc.update_settings(settings);
  calc.set_radius_for_element(1, 0.325);

  // Test DMA calculation (analytical method)
  auto dma_result = calc.compute_multipoles();
  const auto &result = dma_result.multipoles;

  // Check the result
  REQUIRE(result.size() > 0);

  auto expected = Mat(3, 12);

  expected << -0.427605, -0.013835, 0.327911, 0.530539, -1.009920, 0.006672,
      -0.036657, 0.214050, -0.463494, 0.000000, 0.000000, 0.000000, 0.212067,
      0.001162, 0.018114, -0.049016, -0.107774, 0.002061, -0.006636, -0.092154,
      -0.093068, 0.000000, 0.000000, 0.000000, 0.215539, 0.000231, -0.051145,
      -0.004406, -0.107397, -0.001061, -0.001414, 0.129761, 0.012256, 0.000000,
      0.000000, 0.000000;

  for (int site = 0; site < 3; site++) {
    const auto &m = result[site];
    for (int term = 0; term < 12; term++) {
      REQUIRE(m.q(term) == Approx(expected(site, term)).margin(1e-3));
    }
  }
}

const std::string c2h4_contents = read_file(fs::path(occ::get_data_directory()) / "tests" / "c2h4.fchk" );
TEST_CASE("C2H4 G03", "[dma]") {
  using namespace occ::qm;
  using namespace occ::io;
  std::istringstream fchk(c2h4_contents);
  FchkReader reader(fchk);

  occ::qm::Wavefunction wfn(reader);

  occ::dma::DMACalculator calc(wfn);

  occ::dma::DMASettings settings;
  settings.max_rank = 4;
  settings.big_exponent = 4.0;

  calc.update_settings(settings);
  calc.set_radius_for_element(1, 0.35);
  calc.set_limit_for_element(1, 1);

  auto t1 = std::chrono::high_resolution_clock::now();
  // Test DMA calculation (analytical method)
  auto dma_result = calc.compute_multipoles();
  const auto &result = dma_result.multipoles;
  auto t2 = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration<double, std::milli>(t2 - t1);
  occ::timing::print_timings();

  // Check the result
  REQUIRE(result.size());

  occ::Mat expected(6, 12);
  expected.setZero();
  expected << -0.035811, 0.095601, 0.000000, -0.000000, 0.731238, -0.000000,
      0.000000, -1.399558, 0.000000, -2.874517, 0.000000, -0.000000, -0.035811,
      -0.095601, 0.000000, -0.000000, 0.731238, -0.000000, -0.000000, -1.399558,
      0.000000, 2.874517, -0.000000, 0.000000, 0.017905, -0.067488, 0.000000,
      -0.070550, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
      0.000000, 0.000000, 0.017905, -0.067488, -0.000000, 0.070550, 0.000000,
      0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
      0.017905, 0.067488, -0.000000, -0.070550, 0.000000, 0.000000, 0.000000,
      0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.017905, 0.067488,
      -0.000000, 0.070550, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
      0.000000, 0.000000, 0.000000;

  fmt::print("expected\n{}\n", format_matrix(expected));

  for (int site = 0; site < 6; site++) {
    int lm = (site < 2) ? 4 : 1;
    fmt::print("Site: {}\n{}\n", site, result[site].to_string(lm));
    const auto &m = result[site];
    for (int term = 0; term < 12; term++) {
      CAPTURE(site, term);
      CHECK(m.q(term) == Approx(expected(site, term)).margin(1e-3));
    }
  }
}

TEST_CASE("Mult accessor functions", "[dma]") {
  // Create a test multipole with rank 4
  Mult mult(4);
  
  // Set some test values using direct access
  mult.q(0) = 1.5;    // Q00
  mult.q(1) = -0.3;   // Q10  
  mult.q(2) = 0.7;    // Q11c
  mult.q(3) = -0.2;   // Q11s
  mult.q(4) = 0.9;    // Q20
  mult.q(5) = 0.4;    // Q21c
  mult.q(6) = -0.6;   // Q21s
  mult.q(7) = 0.1;    // Q22c
  mult.q(8) = 0.8;    // Q22s
  mult.q(16) = 2.1;   // Q40
  mult.q(23) = -1.2;  // Q44c
  mult.q(24) = 0.5;   // Q44s
  
  SECTION("Test component_name_to_lm helper function") {
    // Test various component name formats
    CHECK(Mult::component_name_to_lm("charge") == std::make_pair(0, 0));
    CHECK(Mult::component_name_to_lm("Q00") == std::make_pair(0, 0));
    CHECK(Mult::component_name_to_lm("Q10") == std::make_pair(1, 0));
    CHECK(Mult::component_name_to_lm("Q11c") == std::make_pair(1, 1));
    CHECK(Mult::component_name_to_lm("Q11s") == std::make_pair(1, -1));
    CHECK(Mult::component_name_to_lm("Q20") == std::make_pair(2, 0));
    CHECK(Mult::component_name_to_lm("Q21c") == std::make_pair(2, 1));
    CHECK(Mult::component_name_to_lm("Q21s") == std::make_pair(2, -1));
    CHECK(Mult::component_name_to_lm("Q22c") == std::make_pair(2, 2));
    CHECK(Mult::component_name_to_lm("Q22s") == std::make_pair(2, -2));
    CHECK(Mult::component_name_to_lm("Q40") == std::make_pair(4, 0));
    CHECK(Mult::component_name_to_lm("Q44c") == std::make_pair(4, 4));
    CHECK(Mult::component_name_to_lm("Q44s") == std::make_pair(4, -4));
    
    // Test invalid names
    CHECK(Mult::component_name_to_lm("invalid") == std::make_pair(-1, 0));
    CHECK(Mult::component_name_to_lm("Q") == std::make_pair(-1, 0));
    CHECK(Mult::component_name_to_lm("Q1") == std::make_pair(-1, 0));
  }
  
  SECTION("Test get_multipole vs inline accessors") {
    // Test rank 0 (monopole)
    CHECK(mult.get_multipole(0, 0) == Approx(mult.Q00()));
    CHECK(mult.get_multipole(0, 0) == Approx(mult.charge()));
    
    // Test rank 1 (dipole)
    CHECK(mult.get_multipole(1, 0) == Approx(mult.Q10()));
    CHECK(mult.get_multipole(1, 1) == Approx(mult.Q11c()));
    CHECK(mult.get_multipole(1, -1) == Approx(mult.Q11s()));
    
    // Test rank 2 (quadrupole)
    CHECK(mult.get_multipole(2, 0) == Approx(mult.Q20()));
    CHECK(mult.get_multipole(2, 1) == Approx(mult.Q21c()));
    CHECK(mult.get_multipole(2, -1) == Approx(mult.Q21s()));
    CHECK(mult.get_multipole(2, 2) == Approx(mult.Q22c()));
    CHECK(mult.get_multipole(2, -2) == Approx(mult.Q22s()));
    
    // Test rank 4 (hexadecapole)
    CHECK(mult.get_multipole(4, 0) == Approx(mult.Q40()));
    CHECK(mult.get_multipole(4, 4) == Approx(mult.Q44c()));
    CHECK(mult.get_multipole(4, -4) == Approx(mult.Q44s()));
  }
  
  SECTION("Test get_component vs inline accessors") {
    // Test using component names
    CHECK(mult.get_component("charge") == Approx(mult.charge()));
    CHECK(mult.get_component("Q00") == Approx(mult.Q00()));
    CHECK(mult.get_component("Q10") == Approx(mult.Q10()));
    CHECK(mult.get_component("Q11c") == Approx(mult.Q11c()));
    CHECK(mult.get_component("Q11s") == Approx(mult.Q11s()));
    CHECK(mult.get_component("Q20") == Approx(mult.Q20()));
    CHECK(mult.get_component("Q21c") == Approx(mult.Q21c()));
    CHECK(mult.get_component("Q21s") == Approx(mult.Q21s()));
    CHECK(mult.get_component("Q22c") == Approx(mult.Q22c()));
    CHECK(mult.get_component("Q22s") == Approx(mult.Q22s()));
    CHECK(mult.get_component("Q40") == Approx(mult.Q40()));
    CHECK(mult.get_component("Q44c") == Approx(mult.Q44c()));
    CHECK(mult.get_component("Q44s") == Approx(mult.Q44s()));
  }
  
  SECTION("Test boundary conditions") {
    // Test invalid l,m combinations
    CHECK(mult.get_multipole(-1, 0) == 0.0);  // Negative rank
    CHECK(mult.get_multipole(1, 2) == 0.0);   // |m| > l
    CHECK(mult.get_multipole(1, -2) == 0.0);  // |m| > l
    CHECK(mult.get_multipole(10, 0) == 0.0);  // Rank > max_rank
    
    // Test invalid component names
    CHECK(mult.get_component("invalid") == 0.0);
    CHECK(mult.get_component("Q99c") == 0.0);  // Beyond max rank
    CHECK(mult.get_component("Q1x") == 0.0);   // Invalid format
  }
  
  SECTION("Test modifiable accessors") {
    // Test that we can modify values through the new accessors
    mult.get_multipole(3, 1) = 99.9;
    CHECK(mult.Q31c() == Approx(99.9));
    
    mult.get_component("Q32s") = -88.8;
    CHECK(mult.Q32s() == Approx(-88.8));
    CHECK(mult.get_multipole(3, -2) == Approx(-88.8));
  }
}

// ============================================================================
// Driver level: the frame results are reported in, the effective per-site
// settings, and the force-field output.
// ============================================================================

TEST_CASE("DMADriver total multipoles are in the analysis frame", "[dma][driver]") {
  std::istringstream fchk(h2o_contents);
  occ::qm::FchkReader reader(fchk);
  occ::qm::Wavefunction wfn(reader);

  // Water is neutral, so the magnitude of the total dipole is invariant under
  // both rotation and choice of origin: every axis method must agree.
  const auto total_dipole_norm = [&](const std::string &axis_method,
                                     double big_exponent) {
    occ::driver::DMAConfig config;
    config.write_punch = false;
    config.settings.max_rank = 2;
    config.settings.big_exponent = big_exponent;
    config.axis_method = axis_method;
    occ::driver::DMADriver driver(config);
    auto output = driver.run(wfn);
    return output.total.q.segment(1, 3).norm();
  };

  // Two effects bound how exact this can be, so 1e-6 rather than machine
  // precision: the fchk density above is printed to ~9 significant figures,
  // and with big_exponent > 0 the diffuse part goes through a molecule-
  // oriented numerical grid that a rotation re-samples. Neither is a frame
  // error -- the bug this guards against moved |Q1| by ~30%.
  // test_dma.py checks the same invariance on a live SCF wavefunction, where
  // the analytic path holds to 1e-12.
  for (double big_exponent : {0.0, 4.0}) {
    const double reference = total_dipole_norm("none", big_exponent);
    REQUIRE(reference > 0.1);
    CHECK(total_dipole_norm("moi", big_exponent) ==
          Approx(reference).epsilon(1e-6));
    CHECK(total_dipole_norm("pca", big_exponent) ==
          Approx(reference).epsilon(1e-6));
  }
}

TEST_CASE("DMADriver centres the oriented molecule on its centre of mass",
          "[dma][driver]") {
  std::istringstream fchk(h2o_contents);
  occ::qm::FchkReader reader(fchk);
  occ::qm::Wavefunction wfn(reader);

  occ::driver::DMAConfig config;
  config.write_punch = false;
  config.settings.max_rank = 0;
  config.axis_method = "moi";

  occ::driver::DMADriver driver(config);
  auto output = driver.run(wfn);

  occ::Vec3 com = occ::Vec3::Zero();
  double total_mass = 0.0;
  for (int i = 0; i < output.sites.size(); i++) {
    const int z = output.sites.atoms[output.sites.atom_indices(i)].atomic_number;
    const double mass = occ::core::Element(z).mass();
    com += mass * output.sites.positions.col(i);
    total_mass += mass;
  }
  com /= total_mass;
  CHECK(com.norm() == Approx(0.0).margin(1e-9));
}

TEST_CASE("DMADriver clamps per-element limits to max_rank", "[dma][driver]") {
  std::istringstream fchk(h2o_contents);
  occ::qm::FchkReader reader(fchk);
  occ::qm::Wavefunction wfn(reader);

  occ::driver::DMAConfig config;
  config.write_punch = false;
  config.settings.max_rank = 2;
  // An override above max_rank must be clamped, not taken at face value.
  config.atom_limits["O"] = 6;
  config.atom_limits["H"] = 1;

  occ::driver::DMADriver driver(config);
  auto output = driver.run(wfn);

  for (int i = 0; i < output.sites.size(); i++) {
    const int z = output.sites.atoms[output.sites.atom_indices(i)].atomic_number;
    CHECK(output.sites.limits(i) <= config.settings.max_rank);
    CHECK(output.sites.limits(i) == (z == 1 ? 1 : 2));
  }
}

TEST_CASE("DMADriver applies a supplied wavefunction transform", "[dma][driver]") {
  std::istringstream fchk(h2o_contents);
  occ::qm::FchkReader reader(fchk);
  occ::qm::Wavefunction wfn(reader);

  occ::driver::DMAConfig plain;
  plain.write_punch = false;
  plain.settings.max_rank = 1;
  auto reference = occ::driver::DMADriver(plain).run(wfn);

  occ::driver::DMAConfig moved = plain;
  // 90 degrees about z, then a translation.
  moved.wfn_rotation << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  moved.wfn_translation = occ::Vec3(1.0, 2.0, 3.0);
  auto transformed = occ::driver::DMADriver(moved).run(wfn);

  const occ::Vec3 translation_bohr =
      moved.wfn_translation * occ::units::ANGSTROM_TO_BOHR;
  for (int i = 0; i < reference.sites.size(); i++) {
    const occ::Vec3 expected =
        moved.wfn_rotation * reference.sites.positions.col(i) + translation_bohr;
    CHECK((transformed.sites.positions.col(i) - expected).norm() ==
          Approx(0.0).margin(1e-9));
    // Rank 0 is a scalar: unchanged by any rigid-body motion.
    CHECK(transformed.result.multipoles[i].q(0) ==
          Approx(reference.result.multipoles[i].q(0)).margin(1e-9));
  }
  CHECK(transformed.rotation.isApprox(moved.wfn_rotation, 1e-12));
}

TEST_CASE("DMADriver JSON reports the effective site settings and frame",
          "[dma][driver][json]") {
  std::istringstream fchk(h2o_contents);
  occ::qm::FchkReader reader(fchk);
  occ::qm::Wavefunction wfn(reader);

  occ::driver::DMAConfig config;
  config.write_punch = false;
  config.settings.max_rank = 4;
  config.atom_radii["H"] = 0.50;
  config.atom_limits["H"] = 2;
  // Requesting CSP output promotes "none" to "moi"; the JSON has to say so.
  config.csp_input_filename = "unused-in-this-test.json";

  occ::driver::DMADriver driver(config);
  auto output = driver.run(wfn);
  auto j = nlohmann::json::parse(
      occ::driver::DMADriver::generate_json(config, output));

  CHECK(j["settings"]["axis_method_requested"] == "none");
  CHECK(j["settings"]["axis_method_applied"] == "moi");
  CHECK(j["units"]["position"] == "angstrom");
  CHECK(j["units"]["multipole"] == "atomic");

  REQUIRE(j["sites"].size() == 3);
  for (const auto &site : j["sites"]) {
    const int z = site["atomic_number"].get<int>();
    if (z == 1) {
      CHECK(site["radius"].get<double>() == Approx(0.50));
      CHECK(site["limit"].get<int>() == 2);
      CHECK(site["rank"].get<int>() == 2);
    }
    // Components are (rank + 1)^2 in the punch file's ordering.
    const int rank = site["rank"].get<int>();
    CHECK(site["multipoles"].size() == static_cast<size_t>((rank + 1) * (rank + 1)));
  }

  // The recorded transform must reproduce the reported site positions.
  const auto rotation = j["transform"]["rotation"].get<std::vector<double>>();
  const auto translation = j["transform"]["translation"].get<std::vector<double>>();
  occ::Mat3 R = Eigen::Map<const occ::Mat3RM>(rotation.data());
  occ::Vec3 t = Eigen::Map<const occ::Vec3>(translation.data()) *
                occ::units::ANGSTROM_TO_BOHR;
  const occ::Mat3N input_positions = wfn.positions();
  for (int i = 0; i < output.sites.size(); i++) {
    const occ::Vec3 expected = R * input_positions.col(i) + t;
    CHECK((output.sites.positions.col(i) - expected).norm() ==
          Approx(0.0).margin(1e-9));
  }
}

TEST_CASE("DMA force-field basis labels sites for the chosen set",
          "[dma][forcefield]") {
  std::istringstream fchk(h2o_contents);
  occ::qm::FchkReader reader(fchk);
  occ::qm::Wavefunction wfn(reader);

  occ::driver::DMAConfig config;
  config.write_punch = false;
  config.settings.max_rank = 2;
  auto output = occ::driver::DMADriver(config).run(wfn);

  const auto build = [&](const std::string &force_field) {
    occ::mults::DMAForceFieldOptions options;
    options.force_field = force_field;
    return occ::mults::build_dma_force_field_basis(
        output.sites, output.result.multipoles, options);
  };

  SECTION("w99 uses NEIGHCRYS labels") {
    auto basis = build("w99");
    CHECK(basis.potentials.force_field == "w99");
    CHECK(basis.potentials.atom_typing == "neighcrys");
    CHECK(basis.molecule_types[0].sites[0].type == "O_Wa");
    CHECK(basis.molecule_types[0].sites[1].type == "H_Wa");
  }

  SECTION("fit groups polar hydrogens into H_F2") {
    auto basis = build("fit");
    CHECK(basis.potentials.force_field == "fit");
    CHECK(basis.potentials.atom_typing == "neighcrys-fit");
    CHECK(basis.molecule_types[0].sites[0].type == "O_F1");
    CHECK(basis.molecule_types[0].sites[1].type == "H_F2");
  }

  SECTION("williams-de falls back to element names") {
    auto basis = build("williams-de");
    CHECK(basis.potentials.atom_typing == "none");
    CHECK(basis.molecule_types[0].sites[0].type == "O");
    CHECK(basis.molecule_types[0].sites[1].type == "H");
  }

  SECTION("aliases resolve and unknown names throw") {
    CHECK(build("williams").potentials.force_field == "w99");
    CHECK_THROWS(build("not-a-force-field"));
  }

  SECTION("every emitted pair references a type present on a site") {
    for (const auto &name : {"w99", "fit", "williams-de"}) {
      auto basis = build(name);
      std::set<std::string> present;
      for (const auto &site : basis.molecule_types[0].sites)
        present.insert(site.type);
      REQUIRE(!basis.potentials.buckingham.empty());
      for (const auto &pair : basis.potentials.buckingham) {
        INFO(name << ": " << pair.types[0] << "-" << pair.types[1]);
        CHECK(present.count(pair.types[0]) == 1);
        CHECK(present.count(pair.types[1]) == 1);
        CHECK(pair.rho > 0.0);
      }
    }
  }
}
