#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/ostream.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/disp/dftd4.h>

/* Dimer tests */
using occ::core::Molecule;

using Catch::Approx;
using Catch::Matchers::WithinAbs;

inline Molecule water_molecule() {
  std::vector<occ::core::Atom> atoms{
      {8, -1.326958314777, -0.105938613464, 0.018788224049},
      {1, -1.931665194449, 1.600174350956, -0.021710496528},
      {1, 0.486644350311, 0.079598098958, 0.009862480644}};
  return Molecule(atoms);
}

inline Molecule benzene_molecule() {
  std::vector<occ::core::Atom> atoms{
      {6, 13.117401613224, 2.479717517988, 12.780236678069},
      {6, 12.042090756531, 0.795714538198, 14.490533307157},
      {6, 15.040386917638, 1.705356884998, 11.163878334636},
      {1, 12.504336663875, 4.453857708595, 12.710751448467},
      {1, 10.532464144617, 1.390275289336, 15.752983741971},
      {1, 15.879689878627, 2.992381318336, 9.831942670220},
      {6, 12.889765204252, -1.705356884998, 14.574172585433},
      {6, 15.888061365359, -0.795714538198, 11.247536510173},
      {6, 14.812750508666, -2.479717517988, 12.957833139261},
      {1, 12.050462243263, -2.992381318336, 15.906127147110},
      {1, 17.397687977273, -1.390275289336, 9.985086075359},
      {1, 15.425815458015, -4.453857708595, 13.027318368863}};
  return Molecule(atoms);
}

TEST_CASE("dftd4 dispersion", "[disp]") {

  SECTION("water pbe") {
    Molecule m = water_molecule();
    occ::disp::D4Dispersion disp(m);
    double e = disp.energy();
    REQUIRE(e == Approx(-1.960327305609419e-04));
  }

  SECTION("water blyp") {
    Molecule m = water_molecule();

    occ::disp::D4Dispersion disp(m);
    disp.set_functional("blyp");
    double e = disp.energy();
    REQUIRE(e == Approx(-4.734890496748087e-04));
  }

  SECTION("benzene wb97x") {
    Molecule m = benzene_molecule();

    occ::disp::D4Dispersion disp(m);
    disp.set_functional("wb97x");
    double e = disp.energy();
    REQUIRE(e == Approx(-1.355940054241274e-03));
  }

  SECTION("benzene b3lyp +1") {
    Molecule m = benzene_molecule();
    m.set_charge(1);

    occ::disp::D4Dispersion disp(m);
    disp.set_functional("b3lyp");
    double e = disp.energy();
    REQUIRE(e == Approx(-1.522647086484191e-02));
  }

}
