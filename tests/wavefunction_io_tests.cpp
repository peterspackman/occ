#include "occ/core/data_directory.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <filesystem>
#include <fmt/ostream.h>
#include <occ/core/format_matrix.h>
#include <occ/core/util.h>
#include <occ/qm/io/fchkreader.h>
#include <occ/qm/io/fchkwriter.h>
#include <occ/qm/io/gaussian_input_file.h>
#include <occ/qm/io/moldenreader.h>
#include <occ/qm/io/wavefunction_json.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <sstream>
#include <iostream>
#include <string>
#include "test_utils.h"


using occ::format_matrix;
using occ::Mat;
using occ::qm::HartreeFock;
using occ::util::all_close;
namespace fs = std::filesystem;


const std::string fchk_contents = read_file(fs::path(occ::get_data_directory()) / "tests" / "h2.fchk" );

const std::string unrestricted_fchk_contents = read_file(fs::path(occ::get_data_directory()) / "tests" / "water_unrestricted.fchk");

TEST_CASE("Read H2 fchk contents", "[read]") {
  std::istringstream fchk(fchk_contents);
  occ::io::FchkReader reader(fchk);
  REQUIRE(reader.num_alpha() == 1);
  REQUIRE(reader.num_basis_functions() == 2);
  fmt::print("Alpha MOs:\n{}\n", format_matrix(reader.alpha_mo_coefficients()));
  fmt::print("Alpha MO energies:\n");
  const auto &energies = reader.alpha_mo_energies();
  for (int i = 0; i < energies.rows(); i++) {
    fmt::print("{:12.5f}\n", energies(i));
  }
  fmt::print("Positions:\n{}\n", format_matrix(reader.atomic_positions()));
  REQUIRE(reader.basis().primitive_exponents[0] == Catch::Approx(3.42525091));

  occ::io::FchkReader::FchkBasis basis = reader.basis();
  basis.print();
  //    fmt::print("Libint2 basis:\n");
  //    for(const auto& shell: reader.basis_set())
  //    {
  //        fmt::print("\n{}\n", shell);
  //    }
  auto density = reader.scf_density_matrix();
  fmt::print("Density matrix\n{}\n", format_matrix(density));

  REQUIRE(density(1, 0) == Catch::Approx(0.301228));
}

TEST_CASE("Write H2 fchk contents", "[write]") {
  occ::io::FchkWriter writer(std::cout);
  writer.set_scalar("Charge", 0);
  writer.set_scalar("Multiplicity", 1);
  writer.set_scalar("Number of electrons", 10);
  writer.set_scalar("SCF Energy", -76.42165911602731);
  writer.set_scalar("Test string", std::string("this string"));
  Mat identity = Mat::Identity(3, 3);
  writer.set_vector("Identity matrix", identity);
  writer.write();
  REQUIRE(true);
}

TEST_CASE("Read water UHF fchk file", "[read]") {
  std::istringstream fchk(unrestricted_fchk_contents);
  occ::io::FchkReader reader(fchk);
  REQUIRE(reader.num_alpha() == 5);
  REQUIRE(reader.num_beta() == 5);
  REQUIRE(reader.num_basis_functions() == 13);
  fmt::print("Alpha MOs:\n{}\n", format_matrix(reader.alpha_mo_coefficients()));
  fmt::print("Beta MOs:\n{}\n", format_matrix(reader.beta_mo_coefficients()));
  fmt::print("Alpha MO energies:\n");
  const auto &energies = reader.alpha_mo_energies();
  for (int i = 0; i < energies.rows(); i++) {
    fmt::print("{:12.5f}\n", energies(i));
  }

  fmt::print("Beta MO energies:\n");
  const auto &energies_beta = reader.beta_mo_energies();
  for (int i = 0; i < energies_beta.rows(); i++) {
    fmt::print("{:12.5f}\n", energies_beta(i));
  }

  fmt::print("Positions:\n{}\n", format_matrix(reader.atomic_positions()));
  CHECK(reader.basis().primitive_exponents[0] == Catch::Approx(3.22037000e02));

  occ::io::FchkReader::FchkBasis basis = reader.basis();
  basis.print();
  //    fmt::print("Libint2 basis:\n");
  //    for(const auto& shell: reader.basis_set())
  //    {
  //        fmt::print("\n{}\n", shell);
  //    }
  auto density = reader.scf_density_matrix();
  int nbf = 13;
  Mat density_expected(nbf, nbf);
  density_expected << 1.027159055811635913e+00, 3.704023489202387664e-02,
      1.260729573645737699e-02, 1.989001246282115845e-02,
      -5.194306171339705324e-04, -2.354159924174853424e-01,
      8.929778175037792615e-03, 1.410491755866299067e-02,
      -3.683264284039888077e-04, -1.248623214013617339e-02,
      1.192270398277695161e-02, -1.222512880781454966e-02,
      1.161990007192056397e-02, 3.704023489202387664e-02,
      6.334771294985715173e-02, -9.750618291678152944e-03,
      -1.510891725501560187e-02, 3.949898091787798038e-04,
      1.837846543064179383e-01, -1.185271014402013157e-02,
      -1.860736245068380507e-02, 4.860745829954922904e-04,
      1.486469825614799428e-02, -3.960908199636133109e-03,
      1.447565770463709982e-02, -3.946469581940933134e-03,
      1.260729573645737699e-02, -9.750618291678152944e-03,
      1.703672774402610057e-01, 1.903238124037974141e-02,
      -2.234370062451817760e-04, -6.394885284870387154e-02,
      1.714101107876532804e-01, 3.747558671183424950e-02,
      -5.410625524466284057e-04, -4.228670877418309881e-02,
      -3.623315486170241434e-02, 1.141641772014494183e-01,
      8.897124031412841083e-02, 1.989001246282115845e-02,
      -1.510891725501560187e-02, 1.903238124037974141e-02,
      1.894877061985975886e-01, 1.983742456148060874e-03,
      -1.020308913292919395e-01, 3.739629697712439083e-02,
      2.071882458996356091e-01, 2.918899383189254605e-03,
      1.073430944241324791e-01, 8.011861553997037810e-02,
      7.642275436028479405e-03, 1.777498323894319537e-03,
      -5.194306171339705324e-04, 3.949898091787798038e-04,
      -2.234370062451817760e-04, 1.983742456148060874e-03,
      2.719738047075293541e-01, 2.662811170929253561e-03,
      -5.391127968595087281e-04, 2.918708620554830071e-03,
      3.294785660005331573e-01, -2.537845853486923875e-03,
      -1.882955832520055890e-03, -4.625935829770838376e-04,
      -2.577658182671186912e-04, -2.354159924174853424e-01,
      1.837846543064179383e-01, -6.394885284870387154e-02,
      -1.020308913292919395e-01, 2.662811170929253561e-03,
      6.665841367832672226e-01, -7.378471882986549490e-02,
      -1.178241994049793046e-01, 3.074835361423340196e-03,
      2.860002713323228640e-02, -3.033674947761145699e-02,
      2.854964855994925055e-02, -2.958604610690653350e-02,
      8.929778175037792615e-03, -1.185271014402013157e-02,
      1.714101107876532804e-01, 3.739629697712439083e-02,
      -5.391127968595087281e-04, -7.378471882986549490e-02,
      1.742711290797470391e-01, 5.752851659054463579e-02,
      -8.733761301656478267e-04, -3.167969125219934762e-02,
      -2.828689868979465671e-02, 1.143103465506893623e-01,
      8.868697115510079665e-02, 1.410491755866299067e-02,
      -1.860736245068380507e-02, 3.747558671183424950e-02,
      2.071882458996356091e-01, 2.918708620554830071e-03,
      -1.178241994049793046e-01, 5.752851659054463579e-02,
      2.282817058714250447e-01, 4.065527461716612195e-03,
      1.119443906633796754e-01, 8.315995481986421245e-02,
      1.941025435977715430e-02, 1.067323079244111329e-02,
      -3.683264284039888077e-04, 4.860745829954922904e-04,
      -5.410625524466284057e-04, 2.918899383189254605e-03,
      3.294785660005331573e-01, 3.074835361423340196e-03,
      -8.733761301656478267e-04, 4.065527461716612195e-03,
      3.991439223409505299e-01, -2.676511993842408017e-03,
      -1.976860822710261884e-03, -7.523217461259853458e-04,
      -4.758293834331794002e-04, -1.248623214013617339e-02,
      1.486469825614799428e-02, -4.228670877418309881e-02,
      1.073430944241324791e-01, -2.537845853486923875e-03,
      2.860002713323228640e-02, -3.167969125219934762e-02,
      1.119443906633796754e-01, -2.676511993842408017e-03,
      8.574442027115855569e-02, 5.926026744033328908e-02,
      -2.330959662430042573e-02, -2.696912988410838166e-02,
      1.192270398277695161e-02, -3.960908199636133109e-03,
      -3.623315486170241434e-02, 8.011861553997037810e-02,
      -1.882955832520055890e-03, -3.033674947761145699e-02,
      -2.828689868979465671e-02, 8.315995481986421245e-02,
      -1.976860822710261884e-03, 5.926026744033328908e-02,
      4.558327548581628946e-02, -2.668742503593509591e-02,
      -2.255196308268581779e-02, -1.222512880781454966e-02,
      1.447565770463709982e-02, 1.141641772014494183e-01,
      7.642275436028479405e-03, -4.625935829770838376e-04,
      2.854964855994925055e-02, 1.143103465506893623e-01,
      1.941025435977715430e-02, -7.523217461259853458e-04,
      -2.330959662430042573e-02, -2.668742503593509591e-02,
      8.463961223714372428e-02, 5.983864384181302593e-02,
      1.161990007192056397e-02, -3.946469581940933134e-03,
      8.897124031412841083e-02, 1.777498323894319754e-03,
      -2.577658182671185827e-04, -2.958604610690653697e-02,
      8.868697115510079665e-02, 1.067323079244111676e-02,
      -4.758293834331793460e-04, -2.696912988410838166e-02,
      -2.255196308268581432e-02, 5.983864384181302593e-02,
      4.685409901721312304e-02;

  for (int i = 0; i < nbf; i++) {
    for (int j = 0; j < nbf; j++) {
      INFO("DENSITY(" << i << ", " << j << ")");
      CHECK(density(i, j) == Catch::Approx(density_expected(i, j)));
    }
  }
}

// Fchk Pure

const std::string pure_fchk_contents = read_file(fs::path(occ::get_data_directory()) / "tests" / "water_pure.fchk" );

TEST_CASE("Read Gaussian09 water fchk spherical basis", "[read]") {
  std::istringstream fchk(pure_fchk_contents);
  occ::io::FchkReader reader(fchk);
  REQUIRE(reader.num_alpha() == 5);
  REQUIRE(reader.num_basis_functions() == 24);
  REQUIRE(reader.basis().primitive_exponents[0] ==
          Catch::Approx(5.48467166e03));
  occ::io::FchkReader::FchkBasis basis = reader.basis();
  auto density = reader.scf_density_matrix();
  REQUIRE(density(0, 0) == Catch::Approx(2.08289728 * 0.5));
  auto obs = reader.basis_set();
}

void check_wavefunctions(const occ::qm::Wavefunction &wfn,
                         const occ::qm::Wavefunction &wfn2) {
  // fmt::print("Original D\n{}\n", wfn.mo.D);
  // fmt::print("After reading D\n{}\n", wfn2.mo.D);
  CHECK(wfn2.mo.D.rows() == wfn.mo.D.rows());
  CHECK(wfn2.mo.D.cols() == wfn.mo.D.cols());
  CHECK(all_close(wfn2.mo.D, wfn.mo.D, 1e-6, 1e-6));
  // fmt::print("Original C\n{}\n", wfn.mo.C);
  // fmt::print("After reading C\n{}\n", wfn2.mo.C);

  CHECK(wfn2.mo.C.rows() == wfn.mo.C.rows());
  CHECK(wfn2.mo.C.cols() == wfn.mo.C.cols());
  CHECK(all_close(wfn2.mo.C, wfn.mo.C, 1e-6, 1e-6));

  auto check_same_position = [](const occ::Vec3 &x1, const occ::Vec3 &x2) {
    for (size_t i = 0; i < 3; i++) {
      CHECK(x1(i) == Catch::Approx(x2(i)));
    }
  };

  auto check_same_coeffs = [](const auto &c1, const auto &c2) {
    for (size_t i = 0; i < c1.rows(); i++) {
      for (size_t j = 0; j < c1.cols(); j++) {
        CHECK(c1(i, j) == Catch::Approx(c2(i, j)));
      }
    }
  };

  for (size_t i = 0; i < wfn.basis.size(); i++) {
    const auto &sh1 = wfn.basis[i];
    const auto &sh2 = wfn2.basis[i];
    check_same_position(sh1.origin, sh2.origin);
    check_same_coeffs(sh1.exponents, sh2.exponents);
    check_same_coeffs(sh1.contraction_coefficients,
                      sh2.contraction_coefficients);
    CHECK(sh1.l == sh2.l);
    CHECK(sh1.kind == sh2.kind);
    CHECK(sh2.is_pure());
  }
}

TEST_CASE("Read/write pure spherical water 6-31G** fchk consistency",
          "[read,write]") {
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};

  auto obs = occ::gto::AOBasis::load(atoms, "6-31G**");
  obs.set_pure(true);
  HartreeFock hf(obs);
  occ::qm::SCF<HartreeFock> scf(hf);
  scf.convergence_settings.energy_threshold = 1e-8;
  double e = scf.compute_scf_energy();

  occ::qm::Wavefunction wfn = scf.wavefunction();

  std::string water_fchk_contents;
  {
    std::ostringstream fchk_os;
    occ::io::FchkWriter fchk_writer(fchk_os);
    fchk_writer.set_title("test water fchk");
    fchk_writer.set_method("hf");
    fchk_writer.set_basis_name("6-31G**");
    wfn.save(fchk_writer);
    fchk_writer.write();
    water_fchk_contents = fchk_os.str();
  }

  occ::qm::Wavefunction wfn2 = [&]() {
    std::istringstream fchk_is(water_fchk_contents);
    occ::io::FchkReader reader(fchk_is);
    return occ::qm::Wavefunction(reader);
  }();

  check_wavefunctions(wfn, wfn2);

  SECTION("Rotate by I") {
    occ::Mat rot = occ::Mat::Identity(3, 3);
    wfn2.apply_rotation(rot);
    fmt::print("orig MOs\n{}\n", format_matrix(wfn.mo.C));
    fmt::print("rot  MOs\n{}\n", format_matrix(wfn2.mo.C));
    check_wavefunctions(wfn, wfn2);
  }
}

// Gaussian Input

const char *g09_input_contents = R""""(%chk=h2o
#p HF/3-21G 6D 10f

this is a comment

0 1
O -0.0148150000  2.2222190000  0.0000000000
H  0.2124040000  2.7681990000 -0.7555250000
H  0.2124040000  2.7681990000  0.7555250000

)"""";

TEST_CASE("Read Gaussian input HF/3-21G water", "[read]") {
  std::istringstream inp(g09_input_contents);
  occ::io::GaussianInputFile reader(inp);
  REQUIRE(reader.charge == 0);
  REQUIRE(reader.method == "hf");
  REQUIRE(reader.basis_name == "3-21g");

  REQUIRE(reader.atomic_positions.size() == 3);
  REQUIRE(reader.elements.size() == 3);
  REQUIRE(reader.elements[2].atomic_number() == 1);
  REQUIRE(reader.spinorbital_kind() == occ::qm::SpinorbitalKind::Restricted);
}

// Molden

const std::string molden_contents = read_file(fs::path(occ::get_data_directory()) / "tests" / "molden_format.molden" );
;

TEST_CASE("Read molden output formamide molecule", "[read]") {
  std::istringstream molden(molden_contents);
  occ::io::MoldenReader reader(molden);
  for (const auto &atom : reader.atoms()) {
    fmt::print("{} {} {} {}\n", atom.atomic_number, atom.x, atom.y, atom.z);
  }
}

// Orca JSON

using occ::Mat;

const char *json_contents = R"(
{
    "Molecule": {
        "Atoms": [
            {
                "BasisFunctions": [
                    {
                        "Coefficients": [
                            0.1543289707029839,
                            0.5353281424384732,
                            0.44463454202535485
                        ],
                        "Exponents": [
                            3.42525091,
                            0.62391373,
                            0.1688554
                        ],
                        "Shell": "s"
                    }
                ],
                "Coords": [
                    0.0,
                    0.0,
                    0.0
                ],
                "ElementLabel": "H",
                "ElementNumber": 1,
                "Idx": 0,
                "NuclearCharge": 1.0
            },
            {
                "BasisFunctions": [
                    {
                        "Coefficients": [
                            0.1543289707029839,
                            0.5353281424384732,
                            0.44463454202535485
                        ],
                        "Exponents": [
                            3.42525091,
                            0.62391373,
                            0.1688554
                        ],
                        "Shell": "s"
                    }
                ],
                "Coords": [
                    0.0,
                    0.0,
                    2.6456165874897524
                ],
                "ElementLabel": "H",
                "ElementNumber": 1,
                "Idx": 1,
                "NuclearCharge": 1.0
            }
        ],
        "BaseName": "h2",
        "Charge": 0,
        "CoordinateUnits": "Bohrs",
        "H-Matrix": [
            [
                -0.8422289108173067,
                -0.3785030189433606
            ],
            [
                -0.3785030189433606,
                -0.8422289108173067
            ]
        ],
        "HFTyp": "RHF",
        "MolecularOrbitals": {
            "EnergyUnit": "Eh",
            "MOs": [
                {
                    "MOCoefficients": [
                        0.621202118970972,
                        0.621202118970972
                    ],
                    "Occupancy": 2.0,
                    "OrbitalEnergy": -0.3773228237944341
                },
                {
                    "MOCoefficients": [
                        -0.8425697665943784,
                        0.8425697665943784
                    ],
                    "Occupancy": 0.0,
                    "OrbitalEnergy": 0.25890197203084375
                }
            ],
            "OrbitalLabels": [
                "0H   1s",
                "1H   1s"
            ]
        },
        "Multiplicity": 1,
        "S-Matrix": [
            [
                1.0,
                0.29569907102006404
            ],
            [
                0.29569907102006404,
                1.0
            ]
        ],
        "T-Matrix": [
            [
                0.7600318835666087,
                0.019745102188373418
            ],
            [
                0.019745102188373418,
                0.7600318835666087
            ]
        ]
    },
    "ORCA Header": {
        "Version": "5.0 - current"
    }
}
)";

TEST_CASE("Read orca output JSON H2", "[read]") {

  std::istringstream json_istream(json_contents);
  occ::io::OrcaJSONReader reader(json_istream);
  occ::IVec nums(2);
  nums << 1, 1;
  REQUIRE(nums == reader.atomic_numbers());

  occ::Mat3N pos = occ::Mat3N::Zero(3, 2);
  pos(2, 1) = 2.6456165874897524;

  REQUIRE(occ::util::all_close(pos, reader.atom_positions()));

  std::vector<occ::core::Atom> atoms = reader.atoms();
  Mat S1 = reader.overlap_matrix();
  fmt::print("ORCA Overlap matrix:\n{}\n", format_matrix(S1));

  HartreeFock hf(reader.basis_set());
  Mat S2 = hf.compute_overlap_matrix();
  fmt::print("OUR Overlap matrix:\n{}\n", format_matrix(S2));
  fmt::print("Difference\n{}\n", format_matrix(S2 - S1));
  REQUIRE(occ::util::all_close(S1, S2));
}

TEST_CASE("Read/Write Wavefunction JSON", "[JSON]") {
  using occ::io::JsonWavefunctionReader;
  using occ::io::JsonWavefunctionWriter;
  std::vector<occ::core::Atom> atoms{{8, -1.32695761, -0.10593856, 0.01878821},
                                     {1, -1.93166418, 1.60017351, -0.02171049},
                                     {1, 0.48664409, 0.07959806, 0.00986248}};

  auto obs = occ::gto::AOBasis::load(atoms, "6-31G**");
  obs.set_pure(true);
  HartreeFock hf(obs);
  occ::qm::SCF<HartreeFock> scf(hf);
  scf.convergence_settings.energy_threshold = 1e-8;
  double e = scf.compute_scf_energy();

  occ::qm::Wavefunction wfn = scf.wavefunction();

  JsonWavefunctionWriter writer;

  std::string json_contents = writer.to_string(wfn);
  std::istringstream json_stream(json_contents);

  JsonWavefunctionReader reader(json_stream);

  auto wfn2 = reader.wavefunction();

  REQUIRE(wfn.basis == wfn2.basis);
  REQUIRE(wfn.atoms.size() == wfn2.atoms.size());
}

TEST_CASE("Read/Write Wavefunction JSON with ECP", "[JSON]") {
  using occ::io::JsonWavefunctionReader;
  using occ::io::JsonWavefunctionWriter;
  std::vector<occ::core::Atom> atoms{{86, 0.0, 0.0, 0.0}};

  auto obs = occ::gto::AOBasis::load(atoms, "def2-svp");
  obs.set_pure(true);
  HartreeFock hf(obs);
  occ::qm::SCF<HartreeFock> scf(hf);
  scf.convergence_settings.energy_threshold = 1e-8;
  double e = scf.compute_scf_energy();

  occ::qm::Wavefunction wfn = scf.wavefunction();

  JsonWavefunctionWriter writer;

  std::string json_contents = writer.to_string(wfn);
  std::istringstream json_stream(json_contents);

  JsonWavefunctionReader reader(json_stream);

  auto wfn2 = reader.wavefunction();

  REQUIRE(wfn.basis == wfn2.basis);
  REQUIRE(wfn.atoms.size() == wfn2.atoms.size());
}
