#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/cg/cg_json.h>
#include <occ/cg/distance_partition.h>
#include <occ/cg/result_types.h>
#include <occ/cg/solvation_types.h>
#include <occ/cg/solvation_data.h>
#include <occ/core/data_directory.h>
#include <occ/core/units.h>
#include <occ/solvent/surface.h>
#include <occ/xtb/smd_xtb.h>
#include <occ/xtb/xtb_calculator.h>

#ifndef OCC_GFN2_DATA_DIR
#define OCC_GFN2_DATA_DIR "share"
#endif

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

TEST_CASE("CG: SolvationContribution carries arbitrary channels", "[cg]") {
  using namespace occ::cg;

  SECTION("Named energy channels sum; descriptors do not") {
    SolvationContribution contrib;
    contrib.set_antisymmetrize(false);
    contrib.add_energy("dielectric", -3.0);
    contrib.add_energy("residual", 1.0);
    contrib.add_descriptor("hbond_area", 12.0);
    contrib.add_descriptor("reorganisation", 0.5);

    REQUIRE(contrib.energy("dielectric").forward == -3.0);
    REQUIRE(contrib.energy("residual").forward == 1.0);
    REQUIRE(contrib.total_energy() == -2.0);
    // Descriptors are carried but never enter the energy.
    REQUIRE(contrib.descriptor("hbond_area").forward == 12.0);
    REQUIRE(contrib.energy_channels().size() == 2);
    REQUIRE(contrib.descriptor_channels().size() == 2);
  }

  SECTION("Unknown channels read as zero") {
    SolvationContribution contrib;
    REQUIRE(contrib.energy("not-a-channel").forward == 0.0);
    REQUIRE(contrib.descriptor("not-a-channel").total() == 0.0);
  }

  SECTION("Exchange moves every channel, including one-sided ones") {
    SolvationContribution a, b;
    a.add_energy("dielectric", -3.0);
    a.add_descriptor("hbond_area", 12.0);
    b.add_energy("dielectric", -5.0);
    // Only b has this one; it must still reach a.
    b.add_energy("residual", 2.0);

    a.exchange_with(b);
    REQUIRE(a.energy("dielectric").reverse == -5.0);
    REQUIRE(b.energy("dielectric").reverse == -3.0);
    REQUIRE(a.energy("residual").reverse == 2.0);
    REQUIRE(a.descriptor("hbond_area").forward == 12.0);
    REQUIRE(b.descriptor("hbond_area").reverse == 12.0);
    REQUIRE(a.has_been_exchanged());
  }

  SECTION("SMD accessors map onto the named channels") {
    SolvationContribution contrib;
    contrib.add_coulomb(-1.5);
    contrib.add_cds(-0.5);
    contrib.add_coulomb_area(20.0);
    REQUIRE(contrib.coulomb().forward == contrib.energy("coulomb").forward);
    REQUIRE(contrib.cds().forward == contrib.energy("cds").forward);
    REQUIRE(contrib.coulomb_area().forward ==
            contrib.descriptor("coulomb_area").forward);
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

TEST_CASE("CG: CavitySurface operations", "[cg]") {
  using namespace occ::cg;

  SECTION("Default construction") {
    CavitySurface cavity;
    CHECK(cavity.size() == 0);
    CHECK(cavity.total_energy() == 0.0);
    CHECK(cavity.total_area() == 0.0);
  }

  SECTION("Energies sum across channels; descriptors do not") {
    CavitySurface cavity;
    cavity.name = "conductor";
    cavity.positions = occ::Mat3N::Random(3, 5);
    cavity.areas = occ::Vec::Ones(5) * 3.0;
    cavity.energies.push_back({"dielectric", occ::Vec::Ones(5) * 2.0});
    cavity.energies.push_back({"residual", occ::Vec::Ones(5) * -0.5});
    cavity.descriptors.push_back({"hbond_area", occ::Vec::Ones(5) * 99.0});

    CHECK(cavity.size() == 5);
    CHECK(cavity.total_area() == Approx(15.0));
    CHECK(cavity.total_energy() == Approx(7.5)); // 5*(2.0 - 0.5)
  }
}

TEST_CASE("CG: SolvationData round-trips through JSON", "[cg]") {
  using namespace occ::cg;

  SolvationData data;
  auto &coulomb = add_cavity(data, "coulomb", occ::Mat3N::Random(3, 4),
                             occ::Vec::Ones(4) * 1.5, occ::Vec::Ones(4) * -0.25);
  coulomb.descriptors.push_back({"hbond_area", occ::Vec::Ones(4) * 7.0});
  add_cavity(data, "cds", occ::Mat3N::Random(3, 3), occ::Vec::Ones(3) * 2.0,
             occ::Vec::Ones(3) * -0.1);
  data.total_solvation_energy = -1.3;
  data.gas_phase_contribution = 0.4;

  nlohmann::json j = data;
  auto restored = j.get<SolvationData>();

  REQUIRE(restored.cavities.size() == 2);
  CHECK(restored.total_energy() == Approx(data.total_energy()));
  CHECK(restored.total_solvation_energy == data.total_solvation_energy);
  CHECK(restored.gas_phase_contribution == data.gas_phase_contribution);

  const auto *c = restored.find("coulomb");
  REQUIRE(c != nullptr);
  CHECK(c->positions.isApprox(coulomb.positions));
  CHECK(c->areas.isApprox(coulomb.areas));
  REQUIRE(c->energies.size() == 1);
  CHECK(c->energies[0].name == "coulomb");
  REQUIRE(c->descriptors.size() == 1);
  CHECK(c->descriptors[0].name == "hbond_area");
  CHECK(restored.find("not-a-cavity") == nullptr);
}

namespace {
// Test fixture
auto acetic_acid_crystal() {
  const std::vector<std::string> labels = {"C1", "C2", "H1", "H2",
                                           "H3", "H4", "O1", "O2"};
  occ::IVec nums(labels.size());
  occ::Mat positions(labels.size(), 3);
  for (size_t i = 0; i < labels.size(); i++) {
    nums(i) = occ::core::Element(labels[i]).atomic_number();
  }
  positions << 0.16510, 0.28580, 0.17090, 0.08940, 0.37620, 0.34810, 0.18200,
      0.05100, -0.11600, 0.12800, 0.51000, 0.49100, 0.03300, 0.54000, 0.27900,
      0.05300, 0.16800, 0.42100, 0.12870, 0.10750, 0.00000, 0.25290, 0.37030,
      0.17690;
  occ::crystal::AsymmetricUnit asym =
      occ::crystal::AsymmetricUnit(positions.transpose(), nums, labels);
  occ::crystal::SpaceGroup sg(33);
  occ::crystal::UnitCell cell =
      occ::crystal::orthorhombic_cell(13.31, 4.1, 5.75);
  return occ::crystal::Crystal(asym, sg, cell);
}
} // namespace

TEST_CASE("CG: SolventSurfacePartitioner with acetic acid crystal",
          "[cg][partition]") {
  using namespace occ::cg;
  using namespace occ::crystal;
  using namespace occ::core;

  auto crystal = acetic_acid_crystal();
  double radius = 7.0; // 7 Angstrom cutoff

  // Generate dimers using symmetry unique dimers
  auto dimers = crystal.symmetry_unique_dimers(radius);
  auto neighbors = dimers.molecule_neighbors[0];

  SECTION("Test surface partitioning with real surface") {
    auto mol = crystal.symmetry_unique_molecules()[0];
    // Get molecule geometry for surface generation
    const auto &asym = crystal.asymmetric_unit();
    const auto &nums = mol.atomic_numbers();
    const auto &positions = mol.positions();

    // Generate solvent surface
    Vec coulomb_radii = Vec::Ones(nums.size()) * 1.2; // Simple test radii
    auto solvent_surface = occ::solvent::surface::solvent_surface(
        coulomb_radii, nums, positions, 0.0);

    // Two cavities on the same points, as SMD would produce.
    const Eigen::Index n = solvent_surface.vertices.cols();
    SolvationData surface;
    auto &coulomb =
        add_cavity(surface, "coulomb", solvent_surface.vertices,
                   solvent_surface.areas, occ::Vec::Ones(n) * -0.5);
    coulomb.energies.push_back({"electronic", occ::Vec::Ones(n) * -0.2});
    add_cavity(surface, "cds", solvent_surface.vertices, solvent_surface.areas,
               occ::Vec::Ones(n) * -0.3);

    // Create partitioner
    SolventSurfacePartitioner partitioner(crystal, neighbors);
    partitioner.set_should_write_surface_files(false);

    SECTION("Standard distances") {
      partitioner.set_use_normalized_distance(false);
      auto contributions = partitioner.partition(neighbors, surface);

      REQUIRE(contributions.size() == neighbors.size());

      // Track total energies and which points are assigned
      double total_coulomb = 0.0;
      double total_cds = 0.0;
      int assigned_points = 0;

      for (const auto &contrib : contributions) {
        total_coulomb += contrib.coulomb().forward;
        total_cds += contrib.cds().forward;
        // Count how many points got assigned
        if (contrib.coulomb().forward != 0.0) {
          assigned_points++;
        }
      }

      CAPTURE(solvent_surface.vertices.cols()); // Number of surface points
      CAPTURE(neighbors.size());                // Number of dimers
      CAPTURE(total_coulomb);                   // Total coulomb energy
      CAPTURE(total_cds);                       // Total CDS energy
      CAPTURE(assigned_points);                 // Number of assignments made

      CHECK(total_coulomb != 0.0);
      CHECK(total_cds != 0.0);

      // Each channel is now partitioned separately, so `coulomb()` carries
      // only the electrostatic term and `electronic` sits alongside it.
      const double n_points = solvent_surface.vertices.cols();
      CHECK(std::abs(total_coulomb) ==
            Approx(std::abs(-0.5 * n_points)).margin(1e-10));

      double total_electronic = 0.0, total_forward = 0.0;
      for (const auto &contrib : contributions) {
        total_electronic += contrib.energy("electronic").forward;
        total_forward += contrib.forward_energy();
      }
      CHECK(std::abs(total_electronic) ==
            Approx(std::abs(-0.2 * n_points)).margin(1e-10));
      // Summed over channels this is every point's energy, once.
      CHECK(std::abs(total_forward) ==
            Approx(std::abs(-1.0 * n_points)).margin(1e-10));
    }

    SECTION("Normalized distances") {
      partitioner.set_use_normalized_distance(true);
      auto contributions = partitioner.partition(neighbors, surface);

      // Track distribution of points
      std::map<int, int> point_assignments;
      for (size_t i = 0; i < neighbors.size(); i++) {
        if (contributions[i].coulomb().forward != 0.0) {
          point_assignments[i]++;
        }
      }

      CAPTURE(point_assignments.size()); // How many dimers got points
      CHECK(point_assignments.size() > 0);

      // Check exchange pairs
      int exchange_count = 0;
      for (size_t i = 0; i < neighbors.size(); i++) {
        for (size_t j = i + 1; j < neighbors.size(); j++) {
          const auto &d1 = neighbors[i].dimer;
          const auto &d2 = neighbors[j].dimer;
          if (d1.equivalent_in_opposite_frame(d2)) {
            exchange_count++;
            CHECK(contributions[i].has_been_exchanged());
            CHECK(contributions[j].has_been_exchanged());

            // Verify energy exchange
            CHECK(contributions[i].coulomb().reverse ==
                  contributions[j].coulomb().forward);
            CHECK(contributions[i].cds().reverse ==
                  contributions[j].cds().forward);
          }
        }
      }

      CAPTURE(exchange_count);
      CHECK(exchange_count > 0);
    }
  }
}

// ============================================================================
// Phase 7E — xtb backend → cg partitioner end-to-end
// ============================================================================

namespace {
struct DataDirGuard {
  DataDirGuard() { occ::set_data_directory(OCC_GFN2_DATA_DIR); }
};
DataDirGuard _xtb_data_guard;
} // namespace

TEST_CASE("CG: xtb SMD surfaces partition through acetic-acid crystal",
          "[cg][partition][xtb][solvation]") {
  using namespace occ::crystal;
  auto crystal = acetic_acid_crystal();

  // Use the cg molecules + neighbours just like the existing test.
  auto dimers = crystal.symmetry_unique_dimers(7.0);
  auto neighbors = dimers.molecule_neighbors[0];

  // ---------------------------------------------------------------
  // Drive the in-tree xtb backend with SMD on a single monomer.
  // ---------------------------------------------------------------
  auto mol = crystal.symmetry_unique_molecules()[0];
  occ::xtb::XtbCalculator calc(mol);
  auto model = std::make_shared<occ::xtb::SmdSolvationModel>("water");
  calc.set_solvation_model(model);
  (void)calc.single_point_energy();

  const auto &res = calc.last_result();
  REQUIRE(res.converged);
  REQUIRE(res.solvation_surfaces.has_value());
  const auto &xtb_surfs = res.solvation_surfaces.value();
  REQUIRE(xtb_surfs.coulomb.has_value());
  REQUIRE(xtb_surfs.cds.has_value());

  // ---------------------------------------------------------------
  // Convert and partition.
  // ---------------------------------------------------------------
  SolvationData cg_surfs = from_xtb_surfaces(xtb_surfs);
  const auto *coulomb = cg_surfs.find("coulomb");
  const auto *cds = cg_surfs.find("cds");
  REQUIRE(coulomb != nullptr);
  REQUIRE(cds != nullptr);
  REQUIRE(coulomb->size() == xtb_surfs.coulomb->size());
  REQUIRE(cds->size() == xtb_surfs.cds->size());

  // Round-trip identity: per-element sums should equal the underlying model.
  CHECK(coulomb->total_energy() == Approx(model->e_es()).margin(1e-12));
  CHECK(cds->total_energy() == Approx(model->e_cds()).margin(1e-12));
  CHECK(cg_surfs.total_energy() == Approx(model->energy()).margin(1e-12));

  // ---------------------------------------------------------------
  // SolventSurfacePartitioner over the crystal's neighbour list.
  // ---------------------------------------------------------------
  SolventSurfacePartitioner partitioner(crystal, neighbors);
  partitioner.set_should_write_surface_files(false);
  partitioner.set_use_normalized_distance(false);
  auto contributions = partitioner.partition(neighbors, cg_surfs);

  REQUIRE(contributions.size() == neighbors.size());

  // Sum the partitioned forward Coulomb + CDS contributions. The partitioner
  // assigns each surface element's energy to a single neighbour's *forward*
  // slot before `exchange_matching_forward_reverse_pairs` copies forwards
  // into the partner's reverse, so the forward sum is the original
  // per-element total (no double-counting).
  double sum_coulomb_forward = 0.0;
  double sum_cds_forward = 0.0;
  int neighbours_with_assignment = 0;
  for (const auto &c : contributions) {
    sum_coulomb_forward += c.coulomb().forward;
    sum_cds_forward += c.cds().forward;
    if (c.coulomb().forward != 0.0 || c.coulomb().reverse != 0.0)
      neighbours_with_assignment++;
  }

  CAPTURE(sum_coulomb_forward);
  CAPTURE(model->e_es());
  CAPTURE(sum_cds_forward);
  CAPTURE(model->e_cds());
  CAPTURE(neighbours_with_assignment);
  CHECK(sum_coulomb_forward == Approx(model->e_es()).margin(1e-9));
  CHECK(sum_cds_forward == Approx(model->e_cds()).margin(1e-9));
  CHECK(neighbours_with_assignment > 0);
}

TEST_CASE("CG: from_xtb_surfaces handles CPCM-X (no cds)",
          "[cg][xtb][solvation]") {
  // Synthesise an xtb SolvationSurfaces with only the coulomb branch and
  // confirm the adapter produces a single cavity with a sensible total.
  occ::xtb::SolvationSurfaces s;
  occ::xtb::SolvationSurface c;
  c.positions = occ::Mat3N::Random(3, 5);
  c.areas = occ::Vec::Ones(5);
  c.atom_index = occ::IVec::Zero(5);
  c.energies = occ::Vec::Constant(5, -0.01);
  s.coulomb = std::move(c);

  auto cg_s = from_xtb_surfaces(s);
  REQUIRE(cg_s.cavities.size() == 1);
  const auto *coulomb = cg_s.find("coulomb");
  REQUIRE(coulomb != nullptr);
  CHECK(coulomb->size() == 5);
  CHECK(cg_s.find("cds") == nullptr);
  CHECK(coulomb->total_energy() == Approx(-0.05).margin(1e-12));
  CHECK(cg_s.total_solvation_energy == Approx(-0.05).margin(1e-12));
}
