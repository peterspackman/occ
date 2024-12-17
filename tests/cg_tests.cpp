#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/cg/cg_json.h>
#include <occ/cg/result_types.h>
#include <occ/cg/solvation_types.h>
#include <occ/cg/solvent_surface.h>

using namespace occ::cg;
using Catch::Approx;

TEST_CASE("DimerResult energy components", "[results]") {
  DimerResult result;

  SECTION("Default values are zero") {
    CHECK(result.total_energy() == Approx(0.0));
    CHECK(result.energy_component(components::total) == Approx(0.0));
    CHECK(result.energy_component(components::crystal_total) == Approx(0.0));
    CHECK(result.energy_component(components::solvation_total) == Approx(0.0));
  }

  SECTION("Setting and getting components") {
    result.set_energy_component(components::total, 1.23);
    result.set_energy_component(components::crystal_total, 2.34);
    result.set_energy_component(components::coulomb, 3.45);

    CHECK(result.energy_component(components::total) == Approx(1.23));
    CHECK(result.energy_component(components::crystal_total) == Approx(2.34));
    CHECK(result.energy_component(components::coulomb) == Approx(3.45));
  }

  SECTION("Non-existent components return zero") {
    CHECK(result.energy_component("non_existent") == Approx(0.0));
  }

  SECTION("Has component checks") {
    result.set_energy_component(components::total, 1.23);
    CHECK(result.has_energy_component(components::total));
    CHECK_FALSE(result.has_energy_component("non_existent"));
  }

  SECTION("Component overwriting") {
    result.set_energy_component(components::total, 1.23);
    CHECK(result.energy_component(components::total) == Approx(1.23));

    result.set_energy_component(components::total, 2.34);
    CHECK(result.energy_component(components::total) == Approx(2.34));
  }
}

TEST_CASE("MoleculeResult energy components", "[results]") {
  MoleculeResult result;

  SECTION("Default values are zero") {
    CHECK(result.total_energy() == Approx(0.0));
    CHECK(result.energy_component(components::total) == Approx(0.0));
    CHECK(result.energy_component(components::crystal_total) == Approx(0.0));
    CHECK(result.energy_component(components::solvation_total) == Approx(0.0));
  }

  SECTION("Setting and getting components") {
    result.set_energy_component(components::total, 1.23);
    result.set_energy_component(components::crystal_total, 2.34);
    result.set_energy_component(components::coulomb, 3.45);

    CHECK(result.energy_component(components::total) == Approx(1.23));
    CHECK(result.energy_component(components::crystal_total) == Approx(2.34));
    CHECK(result.energy_component(components::coulomb) == Approx(3.45));
  }

  SECTION("Dimer results management") {
    DimerResult dimer1;
    dimer1.set_energy_component(components::total, 1.23);
    dimer1.is_nearest_neighbor = true;
    dimer1.unique_idx = 1;

    DimerResult dimer2;
    dimer2.set_energy_component(components::total, 2.34);
    dimer2.is_nearest_neighbor = false;
    dimer2.unique_idx = 2;

    result.dimer_results.push_back(dimer1);
    result.dimer_results.push_back(dimer2);

    CHECK(result.dimer_results.size() == 2);
    CHECK(result.dimer_results[0].total_energy() == Approx(1.23));
    CHECK(result.dimer_results[0].is_nearest_neighbor);
    CHECK(result.dimer_results[0].unique_idx == 1);
    CHECK(result.dimer_results[1].total_energy() == Approx(2.34));
    CHECK_FALSE(result.dimer_results[1].is_nearest_neighbor);
    CHECK(result.dimer_results[1].unique_idx == 2);
  }

  SECTION("Missing components return zero") {
    CHECK(result.energy_component("non_existent") == Approx(0.0));
  }

  SECTION("Has component checks") {
    result.set_energy_component(components::total, 1.23);
    CHECK(result.has_energy_component(components::total));
    CHECK_FALSE(result.has_energy_component("non_existent"));
  }
}

TEST_CASE("Complete calculation workflow", "[results]") {
  MoleculeResult mol_result;

  // Set molecule-level energies
  mol_result.set_energy_component(components::total, 10.0);
  mol_result.set_energy_component(components::crystal_total, 5.0);
  mol_result.set_energy_component(components::solvation_total, 3.0);

  // Create and add dimer results
  DimerResult dimer1;
  dimer1.set_energy_component(components::total, 2.0);
  dimer1.set_energy_component(components::crystal_total, 1.5);
  dimer1.set_energy_component(components::coulomb, 0.5);
  dimer1.is_nearest_neighbor = true;
  dimer1.unique_idx = 1;

  DimerResult dimer2;
  dimer2.set_energy_component(components::total, 3.0);
  dimer2.set_energy_component(components::crystal_total, 2.0);
  dimer2.set_energy_component(components::polarization, 0.7);
  dimer2.is_nearest_neighbor = true;
  dimer2.unique_idx = 2;

  mol_result.dimer_results.push_back(dimer1);
  mol_result.dimer_results.push_back(dimer2);

  // Verify everything
  CHECK(mol_result.total_energy() == Approx(10.0));
  CHECK(mol_result.energy_component(components::crystal_total) == Approx(5.0));
  CHECK(mol_result.energy_component(components::solvation_total) ==
        Approx(3.0));

  CHECK(mol_result.dimer_results.size() == 2);

  const auto &d1 = mol_result.dimer_results[0];
  CHECK(d1.total_energy() == Approx(2.0));
  CHECK(d1.energy_component(components::crystal_total) == Approx(1.5));
  CHECK(d1.energy_component(components::coulomb) == Approx(0.5));
  CHECK(d1.is_nearest_neighbor);
  CHECK(d1.unique_idx == 1);

  const auto &d2 = mol_result.dimer_results[1];
  CHECK(d2.total_energy() == Approx(3.0));
  CHECK(d2.energy_component(components::crystal_total) == Approx(2.0));
  CHECK(d2.energy_component(components::polarization) == Approx(0.7));
  CHECK(d2.is_nearest_neighbor);
  CHECK(d2.unique_idx == 2);
}

TEST_CASE("MoleculeResult add_dimer_result", "[results]") {
  MoleculeResult result;

  SECTION("Adding nearest neighbor dimers updates totals") {
    DimerResult dimer1;
    dimer1.set_energy_component(components::total, 2.0);
    dimer1.set_energy_component(components::crystal_total, 1.5);
    dimer1.is_nearest_neighbor = true;

    DimerResult dimer2;
    dimer2.set_energy_component(components::total, 3.0);
    dimer2.set_energy_component(components::crystal_total, 2.0);
    dimer2.is_nearest_neighbor = true;

    result.add_dimer_result(dimer1);
    CHECK(result.total_energy() == Approx(2.0));
    CHECK(result.energy_component(components::crystal_total) == Approx(1.5));

    result.add_dimer_result(dimer2);
    CHECK(result.total_energy() == Approx(5.0)); // 2.0 + 3.0
    CHECK(result.energy_component(components::crystal_total) ==
          Approx(3.5)); // 1.5 + 2.0
  }

  SECTION("Non-nearest neighbor dimers don't affect totals") {
    DimerResult dimer;
    dimer.set_energy_component(components::total, 2.0);
    dimer.set_energy_component(components::crystal_total, 1.5);
    dimer.is_nearest_neighbor = false;

    result.add_dimer_result(dimer);
    CHECK(result.total_energy() == Approx(0.0));
    CHECK(result.energy_component(components::crystal_total) == Approx(0.0));
    CHECK(result.dimer_results.size() == 1); // Still added to vector
  }
}

TEST_CASE("CG: ContributionPair basic operations", "[cg]") {
  using namespace occ::cg;

  SECTION("Default construction") {
    ContributionPair pair;
    CHECK(pair.forward == 0.0);
    CHECK(pair.reverse == 0.0);
    CHECK(pair.total() == 0.0);
  }

  SECTION("Total calculation") {
    ContributionPair pair{1.5, 2.5};
    CHECK(pair.total() == 4.0);
  }

  SECTION("Exchange operation") {
    ContributionPair pair1{1.0, 2.0};
    ContributionPair pair2{3.0, 4.0};

    pair1.exchange_with(pair2);

    CHECK(pair1.forward == 1.0);
    CHECK(pair1.reverse == 3.0);
    CHECK(pair2.forward == 3.0);
    CHECK(pair2.reverse == 1.0);
  }
}

TEST_CASE("CG: SolvationContribution functionality", "[cg]") {
  using namespace occ::cg;

  SECTION("Default construction") {
    SolvationContribution contrib;
    CHECK(contrib.total_energy() == 0.0);
  }

  SECTION("Adding contributions") {
    SolvationContribution contrib;

    // Add forward contributions
    contrib.add_coulomb(1.0);
    contrib.add_cds(2.0);
    contrib.add_coulomb_area(3.0);
    contrib.add_cds_area(4.0);

    // Add reverse contributions
    contrib.add_coulomb(0.5, false);
    contrib.add_cds(1.5, false);
    contrib.add_coulomb_area(2.5, false);
    contrib.add_cds_area(3.5, false);

    const auto &coulomb = contrib.coulomb();
    const auto &cds = contrib.cds();

    CHECK(coulomb.forward == 1.0);
    CHECK(coulomb.reverse == 0.5);
    CHECK(cds.forward == 2.0);
    CHECK(cds.reverse == 1.5);
  }

  SECTION("Total energy calculation") {
    SolvationContribution contrib;

    // Add symmetric contributions
    contrib.add_coulomb(1.0);
    contrib.add_coulomb(1.0, false);
    contrib.add_cds(2.0);
    contrib.add_cds(2.0, false);

    CHECK(contrib.total_energy() == 6.0);

    SolvationContribution asymmetric;
    asymmetric.add_coulomb(2.0);        // forward
    asymmetric.add_coulomb(1.0, false); // reverse
    asymmetric.add_cds(4.0);            // forward
    asymmetric.add_cds(2.0, false);     // reverse

    CHECK(asymmetric.total_energy() == 10.5);
  }

  SECTION("Exchange between contributions") {
    SolvationContribution contrib1;
    contrib1.add_coulomb(1.0);
    contrib1.add_cds(2.0);

    SolvationContribution contrib2;
    contrib2.add_coulomb(3.0);
    contrib2.add_cds(4.0);

    contrib1.exchange_with(contrib2);

    CHECK(contrib1.coulomb().reverse == 3.0);
    CHECK(contrib1.cds().reverse == 4.0);
    CHECK(contrib2.coulomb().reverse == 1.0);
    CHECK(contrib2.cds().reverse == 2.0);
  }
}

TEST_CASE("CG: SolventSurface operations", "[cg]") {
  using namespace occ::cg;

  SECTION("Default construction") {
    SolventSurface surface;
    CHECK(surface.size() == 0);
    CHECK(surface.total_energy() == 0.0);
    CHECK(surface.total_area() == 0.0);
  }

  SECTION("Surface with data") {
    SolventSurface surface;

    // Create test data
    surface.positions = occ::Mat3N::Random(3, 5);
    surface.energies = occ::Vec::Ones(5) * 2.0;
    surface.areas = occ::Vec::Ones(5) * 3.0;

    CHECK(surface.size() == 5);
    CHECK(surface.total_energy() == Approx(10.0)); // 5 points * 2.0 energy each
    CHECK(surface.total_area() == Approx(15.0)); // 5 points * 3.0 area each
  }

  SECTION("JSON serialization") {
    SolventSurface surface;
    surface.positions = occ::Mat3N::Random(3, 3);
    surface.energies = occ::Vec::Ones(3) * 2.0;
    surface.areas = occ::Vec::Ones(3) * 1.5;

    nlohmann::json j = surface;
    auto deserialized = j.get<SolventSurface>();

    CHECK(deserialized.size() == surface.size());
    CHECK(deserialized.total_energy() == surface.total_energy());
    CHECK(deserialized.total_area() == surface.total_area());
    CHECK(deserialized.positions.isApprox(surface.positions));
    CHECK(deserialized.energies.isApprox(surface.energies));
    CHECK(deserialized.areas.isApprox(surface.areas));
  }
}

TEST_CASE("CG: SMDSolventSurfaces functionality", "[cg]") {
  using namespace occ::cg;

  SECTION("Default construction") {
    SMDSolventSurfaces surfaces;
    CHECK(surfaces.total_energy() == 0.0);
    CHECK(surfaces.total_solvation_energy == 0.0);
    CHECK(surfaces.electronic_contribution == 0.0);
    CHECK(surfaces.gas_phase_contribution == 0.0);
    CHECK(surfaces.free_energy_correction == 0.0);
  }

  SECTION("Energy calculations") {
    SMDSolventSurfaces surfaces;

    // Set up coulomb surface
    surfaces.coulomb.positions = occ::Mat3N::Random(3, 3);
    surfaces.coulomb.energies = occ::Vec::Ones(3) * 2.0;
    surfaces.coulomb.areas = occ::Vec::Ones(3);

    // Set up CDS surface
    surfaces.cds.positions = occ::Mat3N::Random(3, 4);
    surfaces.cds.energies = occ::Vec::Ones(4) * 3.0;
    surfaces.cds.areas = occ::Vec::Ones(4);

    surfaces.electronic_energies = occ::Vec::Ones(3);

    double expected_total =
        surfaces.coulomb.total_energy() +   // 3 points * 2.0 = 6.0
        surfaces.cds.total_energy() +       // 4 points * 3.0 = 12.0
        surfaces.electronic_energies.sum(); // 3 points * 1.0 = 3.0

    CHECK(surfaces.total_energy() == Approx(21.0));
  }

  SECTION("JSON serialization") {
    SMDSolventSurfaces surfaces;

    surfaces.coulomb.positions = occ::Mat3N::Random(3, 2);
    surfaces.coulomb.energies = occ::Vec::Ones(2) * 1.5;
    surfaces.coulomb.areas = occ::Vec::Ones(2) * 1.0;

    surfaces.cds.positions = occ::Mat3N::Random(3, 3);
    surfaces.cds.energies = occ::Vec::Ones(3) * 2.0;
    surfaces.cds.areas = occ::Vec::Ones(3) * 1.2;

    surfaces.electronic_energies = occ::Vec::Ones(2) * 0.5;

    surfaces.total_solvation_energy = 1.0;
    surfaces.electronic_contribution = 2.0;
    surfaces.gas_phase_contribution = 3.0;
    surfaces.free_energy_correction = 4.0;

    nlohmann::json j = surfaces;
    auto deserialized = j.get<SMDSolventSurfaces>();

    // Check all components are properly serialized/deserialized
    CHECK(deserialized.coulomb.size() == surfaces.coulomb.size());
    CHECK(deserialized.cds.size() == surfaces.cds.size());
    CHECK(deserialized.electronic_energies.isApprox(
        surfaces.electronic_energies));
    CHECK(deserialized.total_solvation_energy ==
          surfaces.total_solvation_energy);
    CHECK(deserialized.electronic_contribution ==
          surfaces.electronic_contribution);
    CHECK(deserialized.gas_phase_contribution ==
          surfaces.gas_phase_contribution);
    CHECK(deserialized.free_energy_correction ==
          surfaces.free_energy_correction);
    CHECK(deserialized.total_energy() == surfaces.total_energy());
  }
}
