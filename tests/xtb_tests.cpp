#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/molecule.h>
#include <occ/core/util.h>
#include <occ/xtb/xtb_wrapper.h>

#ifdef OCC_HAVE_TBLITE
#include <occ/xtb/tblite_wrapper.h>

using occ::Mat;
using occ::Mat3N;
using occ::Vec;

TEST_CASE("Water energy", "[xtb]") {

  std::vector<occ::core::Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  occ::core::Molecule water(atoms);

  occ::xtb::TbliteCalculator calc(water);

  double energy = calc.single_point_energy();

  double expected = -5.0702559583;
  REQUIRE(energy == Catch::Approx(expected).margin(1e-8));
}

TEST_CASE("Water solvation", "[xtb]") {

  std::vector<occ::core::Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  occ::core::Molecule water(atoms);

  occ::xtb::TbliteCalculator calc(water);
  double energy = calc.single_point_energy();

  bool success = calc.set_solvent("water");
  fmt::print("Success: {}\n", success);
  double energy_solvated = calc.single_point_energy();

  double expected = -5.0702559583;
  REQUIRE(energy == Catch::Approx(expected).margin(1e-8));

  double expected_solv = -5.0683301007;
  REQUIRE(energy_solvated == Catch::Approx(expected_solv).margin(1e-8));

  double diff = energy_solvated - energy;
  fmt::print("Difference = {} ({:.2f} kJ/mol)\n", diff,
             diff * occ::units::AU_TO_KJ_PER_MOL);
}
#endif
