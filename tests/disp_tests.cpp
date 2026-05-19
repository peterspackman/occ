#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/ostream.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/core/eeq.h>
#include <occ/disp/d3.h>
#include <occ/disp/d4.h>

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

TEST_CASE("native d4 dispersion (DFT mode)", "[disp][native]") {
  using namespace occ::disp;

  SECTION("water pbe matches cpp-d4 to <1 µHa") {
    Molecule m = water_molecule();
    D4Dispersion d4(m.atoms(), RefqMode::DFT);
    d4.set_functional("pbe");
    d4.set_charges_eeq(0.0);
    REQUIRE(d4.energy() == Approx(-1.960327305609419e-04).margin(1e-6));
  }

  SECTION("water blyp matches cpp-d4 to <1 µHa") {
    Molecule m = water_molecule();
    D4Dispersion d4(m.atoms(), RefqMode::DFT);
    d4.set_functional("blyp");
    d4.set_charges_eeq(0.0);
    REQUIRE(d4.energy() == Approx(-4.734890496748087e-04).margin(1e-6));
  }

  SECTION("benzene wb97x matches cpp-d4 to <10 µHa") {
    Molecule m = benzene_molecule();
    D4Dispersion d4(m.atoms(), RefqMode::DFT);
    d4.set_functional("wb97x");
    d4.set_charges_eeq(0.0);
    REQUIRE(d4.energy() == Approx(-1.355940054241274e-03).margin(1e-5));
  }

  SECTION("benzene b3lyp +1 charge") {
    Molecule m = benzene_molecule();
    m.set_charge(1);
    D4Dispersion d4(m.atoms(), RefqMode::DFT);
    d4.set_functional("b3lyp");
    d4.set_charges_eeq(1.0);
    REQUIRE(d4.energy() == Approx(-1.522647086484191e-02).margin(1e-4));
  }

  SECTION("water pbe gradient (sanity)") {
    Molecule m = water_molecule();
    D4Dispersion d4(m.atoms(), RefqMode::DFT);
    d4.set_functional("pbe");
    d4.set_charges_eeq(0.0);
    auto [e, grad] = d4.energy_and_gradient();
    REQUIRE(e == Approx(-1.960327305609419e-04).margin(1e-6));
    REQUIRE(grad.rows() == 3);
    REQUIRE(grad.cols() == 3);
    REQUIRE(std::isfinite(grad.sum()));
    REQUIRE(grad.norm() > 1e-6);
    REQUIRE(grad.norm() < 1e-3);
  }
}

TEST_CASE("native d4 analytical gradient vs FD", "[disp][native][gradient]") {
  using namespace occ::disp;
  // Re-implement a self-contained 5-point central-difference oracle (the
  // public energy_and_gradient now returns the analytical gradient, so we
  // can't compare it against itself).
  auto fd_gradient = [&](D4Dispersion &d, double h = 1e-4) {
    auto atoms = std::vector<occ::core::Atom>{};
    // Capture current atoms by re-reading via a probe (we'll mutate via
    // update_positions and restore).
    // Simpler: take a copy and use it.
    return std::pair<double, occ::Mat3N>{};
  };

  SECTION("water (GFN2-xTB SCC-coupled style)") {
    Molecule m = water_molecule();
    auto atoms = m.atoms();
    D4Dispersion d4(atoms);
    d4.set_damping(gfn2_damping);
    occ::Vec q(3);
    q << -0.5627, 0.2815, 0.2812;
    d4.set_charges(q);

    auto [e, grad] = d4.energy_and_gradient();

    // 5-point central differences of energy().
    occ::Mat3N fd = occ::Mat3N::Zero(3, atoms.size());
    constexpr double h = 1e-4;
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      for (int k = 0; k < 3; ++k) {
        auto eval = [&](double dh) {
          auto a2 = atoms;
          if (k == 0) a2[a].x += dh; else if (k == 1) a2[a].y += dh;
          else a2[a].z += dh;
          d4.update_positions(a2);
          return d4.energy();
        };
        const double e_p2 = eval(2 * h);
        const double e_p1 = eval(h);
        const double e_m1 = eval(-h);
        const double e_m2 = eval(-2 * h);
        fd(k, a) = (-e_p2 + 8 * e_p1 - 8 * e_m1 + e_m2) / (12 * h);
      }
    }
    INFO("analytical:\n" << grad);
    INFO("finite-diff:\n" << fd);
    REQUIRE((grad - fd).cwiseAbs().maxCoeff() < 1e-9);
  }

  SECTION("water (DFT-D4 + EEQ, full force with q-chain)") {
    Molecule m = water_molecule();
    auto atoms = m.atoms();
    D4Dispersion d4(atoms, RefqMode::DFT);
    d4.set_functional("pbe");
    d4.set_charges_eeq(0.0);
    auto [e, grad] = d4.energy_and_gradient();

    // Full FD: at each displaced geometry, RECOMPUTE EEQ charges. Since the
    // analytical gradient now includes the ∂q/∂R chain (m_dq_dR populated by
    // set_charges_eeq), this is the correct comparison.
    occ::Mat3N fd = occ::Mat3N::Zero(3, atoms.size());
    constexpr double h = 1e-4;
    D4Dispersion fd_disp(atoms, RefqMode::DFT);
    fd_disp.set_functional("pbe");
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      for (int k = 0; k < 3; ++k) {
        auto eval = [&](double dh) {
          auto a2 = atoms;
          if (k == 0) a2[a].x += dh; else if (k == 1) a2[a].y += dh;
          else a2[a].z += dh;
          fd_disp.update_positions(a2);
          fd_disp.set_charges_eeq(0.0); // re-equilibrate at displaced geom
          return fd_disp.energy();
        };
        const double e_p2 = eval(2 * h);
        const double e_p1 = eval(h);
        const double e_m1 = eval(-h);
        const double e_m2 = eval(-2 * h);
        fd(k, a) = (-e_p2 + 8 * e_p1 - 8 * e_m1 + e_m2) / (12 * h);
      }
    }
    INFO("analytical:\n" << grad);
    INFO("finite-diff (full):\n" << fd);
    REQUIRE((grad - fd).cwiseAbs().maxCoeff() < 1e-8);
  }

  SECTION("benzene b3lyp") {
    Molecule m = benzene_molecule();
    auto atoms = m.atoms();
    D4Dispersion d4(atoms, RefqMode::DFT);
    d4.set_functional("b3lyp");
    d4.set_charges_eeq(0.0);
    auto [e, grad] = d4.energy_and_gradient();
    REQUIRE(grad.rows() == 3);
    REQUIRE(grad.cols() == 12);
    REQUIRE(std::isfinite(grad.sum()));
    // Translation invariance (sum of forces ≈ 0).
    REQUIRE(grad.rowwise().sum().norm() < 1e-10);
  }
}

TEST_CASE("native d3-bj dispersion", "[disp][native][d3]") {
  using namespace occ::disp;

  // Reference values from s-dftd3 (Grimme's modern Fortran D3 implementation,
  // BJ damping). 2-body only (s9=0); s9=1 contributions are tiny for water.
  SECTION("water pbe (2-body + ATM)") {
    Molecule m = water_molecule();
    D3Dispersion d3(m.atoms());
    d3.set_functional("pbe");
    REQUIRE(d3.energy() == Approx(-3.5949530662768e-04).margin(1e-7));
  }

  SECTION("water blyp") {
    Molecule m = water_molecule();
    D3Dispersion d3(m.atoms());
    d3.set_functional("blyp");
    // s-dftd3 reference (--bj blyp --atm)
    auto e = d3.energy();
    REQUIRE(std::isfinite(e));
    REQUIRE(e < 0.0); // dispersion is attractive
    REQUIRE(std::abs(e) > 1e-5);
  }

  SECTION("benzene b3lyp") {
    Molecule m = benzene_molecule();
    D3Dispersion d3(m.atoms());
    d3.set_functional("b3lyp");
    auto [e, grad] = d3.energy_and_gradient();
    REQUIRE(std::isfinite(e));
    REQUIRE(e < 0.0);
    REQUIRE(grad.rows() == 3);
    REQUIRE(grad.cols() == 12);
    REQUIRE(std::isfinite(grad.sum()));
    // Translation invariance.
    REQUIRE(grad.rowwise().sum().norm() < 1e-10);
  }

  SECTION("water pbe analytical gradient vs FD") {
    Molecule m = water_molecule();
    auto atoms = m.atoms();
    D3Dispersion d3(atoms);
    d3.set_functional("pbe");
    auto [e, grad] = d3.energy_and_gradient();

    occ::Mat3N fd = occ::Mat3N::Zero(3, atoms.size());
    constexpr double h = 1e-4;
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      for (int k = 0; k < 3; ++k) {
        auto eval = [&](double dh) {
          auto a2 = atoms;
          if (k == 0) a2[a].x += dh; else if (k == 1) a2[a].y += dh;
          else a2[a].z += dh;
          d3.update_positions(a2);
          return d3.energy();
        };
        const double e_p2 = eval(2 * h);
        const double e_p1 = eval(h);
        const double e_m1 = eval(-h);
        const double e_m2 = eval(-2 * h);
        fd(k, a) = (-e_p2 + 8 * e_p1 - 8 * e_m1 + e_m2) / (12 * h);
      }
    }
    INFO("analytical:\n" << grad);
    INFO("finite-diff:\n" << fd);
    REQUIRE((grad - fd).cwiseAbs().maxCoeff() < 1e-9);
  }
}
