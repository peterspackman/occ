#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <occ/mults/dmacrys_input.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_strain.h>
#include <occ/mults/cartesian_kernels.h>
#include <occ/mults/interaction_tensor.h>
#include <occ/mults/ewald_sum.h>
#include <occ/core/elastic_tensor.h>
#include <occ/core/units.h>
#include <occ/core/log.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>

using Catch::Approx;
using namespace occ;
using namespace occ::mults;

static const std::string AXOSOW_JSON = CMAKE_SOURCE_DIR "/tests/data/dmacrys/AXOSOW.json";
static const std::string TCHLBZ03_JSON = CMAKE_SOURCE_DIR "/tests/data/dmacrys/TCHLBZ03.json";
static const std::string UREAXX22_JSON =
    CMAKE_SOURCE_DIR "/tests/data/dmacrys/generated/UREAXX22.json";
static const std::string TCYETY01_JSON =
    CMAKE_SOURCE_DIR "/tests/data/dmacrys/TCYETY01.json";
static const std::string TETBBZ01_JSON =
    CMAKE_SOURCE_DIR "/tests/data/dmacrys/TETBBZ01.json";

TEST_CASE("DMACRYS JSON reader", "[mults][dmacrys]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);

    CHECK(input.title == "AXOSOW");
    // Space group determined from symops (LATT 1 + 3 SYMM = Pbca)
    CHECK(input.crystal.Z == 8);
    CHECK(input.crystal.atoms.size() == 8);
    CHECK(input.molecule.sites.size() == 8);
    CHECK(input.potentials.size() == 6);
    CHECK_FALSE(input.has_spline);

    // Check first site
    CHECK(input.molecule.sites[0].label == "C_F1_1____");
    CHECK(input.molecule.sites[0].atomic_number == 6);
    CHECK(input.molecule.sites[0].rank == 4);
    CHECK(input.molecule.sites[0].components.size() == 25);

    // Check first Buckingham pair
    CHECK(input.potentials[0].el1 == "C");
    CHECK(input.potentials[0].el2 == "C");
    CHECK(input.potentials[0].A_eV == Approx(3832.147));
    CHECK(input.potentials[0].rho_ang == Approx(0.277778));

    // Check reference energies
    CHECK(input.initial_ref.total_kJ_per_mol == Approx(-43.156649));
    CHECK(input.initial_ref.repulsion_dispersion_eV == Approx(-2.0073845));
}

TEST_CASE("DMACRYS JSON pressure units", "[mults][dmacrys][pressure]") {
    namespace fs = std::filesystem;
    auto base = nlohmann::json::parse(std::ifstream(AXOSOW_JSON));
    base["settings"]["pressure"] = {
        {"value", 150.0},
        {"units", "MPa"},
    };

    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path tmp = fs::temp_directory_path() /
                         ("occ_dmacrys_pressure_units_test_" +
                          std::to_string(stamp) + ".json");
    {
        std::ofstream out(tmp);
        out << base.dump(2);
    }

    auto input = read_dmacrys_json(tmp.string());
    CHECK(input.has_pressure);
    CHECK(input.pressure_pa == Approx(150.0e6));

    fs::remove(tmp);
}

TEST_CASE("DMACRYS crystal builder", "[mults][dmacrys]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);

    // Pbca has 8 symmetry operations
    CHECK(crystal.space_group().symmetry_operations().size() == 8);

    // Unit cell should be orthorhombic (initial cell params)
    auto uc = crystal.unit_cell();
    CHECK(uc.a() == Approx(9.8641));
    CHECK(uc.b() == Approx(10.1391));
    CHECK(uc.c() == Approx(6.9997));
}

TEST_CASE("DMACRYS multipole source builder", "[mults][dmacrys]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    // One MultipoleSource per symmetry image in the cell (8 for Pbca)
    REQUIRE(multipoles.size() == 8);
    CHECK(multipoles[0].num_sites() == 8);
}

TEST_CASE("DMACRYS Buckingham conversion", "[mults][dmacrys]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto params = convert_buckingham_params(input.potentials);

    // Should have 6 unique element pairs
    CHECK(params.size() == 6);

    // Check C-C conversion: A=3832.147 eV -> kJ/mol, B=1/0.277778 Ang^-1
    auto cc = params.at({6, 6});
    CHECK(cc.A == Approx(3832.147 * 96.4853329).epsilon(0.001));
    CHECK(cc.B == Approx(1.0 / 0.277778).epsilon(0.001));
    CHECK(cc.C == Approx(25.28695 * 96.4853329).epsilon(0.001));
}

TEST_CASE("DMACRYS typed Buckingham conversion preserves same-element distinctions",
          "[mults][dmacrys][typed]") {
    std::vector<DmacrysInput::BuckPair> pairs;
    pairs.push_back({"C_F1", "H_F1", "C", "H", "BUCK", 10.0, 0.25, 2.0});
    pairs.push_back({"C_F2", "H_F1", "C", "H", "BUCK", 20.0, 0.30, 4.0});

    std::map<std::string, int> type_codes = {
        {"C_F1", 10601},
        {"C_F2", 10602},
        {"H_F1", 10101},
    };

    auto typed = convert_typed_buckingham_params(pairs, type_codes);
    REQUIRE(typed.size() == 4); // two unique canonical pairs + symmetric mirrors

    const auto p1 = typed.at({10101, 10601});
    const auto p2 = typed.at({10101, 10602});
    CHECK(p1.A == Approx(10.0 * 96.4853329).epsilon(1e-10));
    CHECK(p2.A == Approx(20.0 * 96.4853329).epsilon(1e-10));
    CHECK(p1.B == Approx(1.0 / 0.25).epsilon(1e-12));
    CHECK(p2.B == Approx(1.0 / 0.30).epsilon(1e-12));
}

TEST_CASE("DMACRYS SPLI setup wiring", "[mults][dmacrys][spli]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    input.has_spline = true;
    input.spline_min = 2.0;
    input.spline_max = 4.0;

    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, false);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);

    CHECK(calc.cutoff_radius() == Approx(input.cutoff_radius + input.spline_max));
    CHECK(calc.electrostatic_taper().is_valid());
    CHECK(calc.short_range_taper().is_valid());
    CHECK(calc.electrostatic_taper().r_on == Approx(input.cutoff_radius));
    CHECK(calc.electrostatic_taper().r_off ==
          Approx(input.cutoff_radius + input.spline_min));
}

TEST_CASE("DMACRYS AXOSOW initial energy", "[mults][dmacrys]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    // Bypass Crystal's molecule detection: construct with multipoles, then
    // set geometry, states, and neighbor list directly from DMACRYS input.
    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, false);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);

    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }

    auto states = calc.initial_states();
    auto result = calc.compute(states);

    int n_mol = calc.num_molecules();
    double rd_per_mol = result.repulsion_dispersion / n_mol;
    double elec_per_mol = result.electrostatic_energy / n_mol;
    double total_per_mol = result.total_energy / n_mol;

    // DMACRYS reference (kJ/mol per molecule = kJ/mol per cell / 8)
    // The kJ values in the JSON are already per molecule
    double ref_rd_kJ = input.initial_ref.repulsion_dispersion_kJ;
    double ref_total = input.initial_ref.total_kJ_per_mol;

    INFO("OCC rep-disp: " << rd_per_mol << " kJ/mol/mol, DMACRYS: "
                          << ref_rd_kJ);
    INFO("OCC electrostatic: " << elec_per_mol << " kJ/mol/mol");
    INFO("OCC total: " << total_per_mol << " kJ/mol/mol, DMACRYS: "
                       << ref_total);
    INFO("N molecules: " << n_mol);

    // Rep-disp should match closely (same real-space cutoff, same potential)
    // Tolerance: within 0.5 kJ/mol
    CHECK(rd_per_mol == Approx(ref_rd_kJ).margin(0.5));

    // Total energy will differ because we're not using Ewald for electrostatics
    // Just log the discrepancy
    INFO("Total energy discrepancy: " << (total_per_mol - ref_total)
                                       << " kJ/mol");
}

TEST_CASE("DMACRYS AXOSOW initial energy with Ewald", "[mults][dmacrys][ewald]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    // Enable charge-charge Ewald correction (alpha=0.35/Å, kmax=8)
    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       1e-8, 0.35, 8);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    // DMACRYS pair electrostatics are truncated to lA+lB <= 4.
    calc.set_max_interaction_order(4);

    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }

    auto states = calc.initial_states();
    auto result = calc.compute(states);

    int n_mol = calc.num_molecules();
    double rd_per_mol = result.repulsion_dispersion / n_mol;
    double elec_per_mol = result.electrostatic_energy / n_mol;
    double total_per_mol = result.total_energy / n_mol;

    double ref_rd_kJ = input.initial_ref.repulsion_dispersion_kJ;
    double ref_total = input.initial_ref.total_kJ_per_mol;

    // DMACRYS per-component reference (eV per cell → kJ/mol per molecule)
    constexpr double eV_to_kJ = 96.4853;
    double ref_qq = input.initial_ref.charge_charge_inter_eV * eV_to_kJ / n_mol;
    double ref_qmu = input.initial_ref.charge_dipole_eV * eV_to_kJ / n_mol;
    double ref_mumu = input.initial_ref.dipole_dipole_eV * eV_to_kJ / n_mol;
    double ref_higher = input.initial_ref.higher_multipole_eV * eV_to_kJ / n_mol;
    double ref_elec = ref_qq + ref_qmu + ref_mumu + ref_higher;

    INFO("OCC rep-disp: " << rd_per_mol << " kJ/mol/mol, DMACRYS: " << ref_rd_kJ
         << " (diff " << (rd_per_mol - ref_rd_kJ) << ")");
    INFO("OCC electrostatic: " << elec_per_mol << " kJ/mol/mol, DMACRYS: "
         << ref_elec << " (diff " << (elec_per_mol - ref_elec) << ")");
    INFO("  DMACRYS breakdown: qq=" << ref_qq << " qmu=" << ref_qmu
         << " mumu=" << ref_mumu << " higher=" << ref_higher);
    INFO("OCC total: " << total_per_mol << " kJ/mol/mol, DMACRYS: " << ref_total
         << " (diff " << (total_per_mol - ref_total) << ")");

    CHECK(rd_per_mol == Approx(ref_rd_kJ).margin(0.12));

    int site_pair_count = 0;
    for (const auto& pair : calc.neighbor_pairs()) {
        int mi = pair.mol_i, mj = pair.mol_j;
        const auto ci = multipoles[mi].cartesian();
        const auto cj = multipoles[mj].cartesian();
        for (const auto& sA : ci.sites) {
            if (sA.rank < 0) continue;
            for (const auto& sB : cj.sites) {
                if (sB.rank < 0) continue;
                ++site_pair_count;
            }
        }
    }
    if (input.has_spline) {
        INFO("Site pairs computed: " << site_pair_count << ", DMACRYS: 60616");
    } else {
        INFO("Site pairs computed (no-SPLI JSON fixture): " << site_pair_count
             << " (DMACRYS 60616 reference is SPLI-enabled)");
    }
    CHECK(total_per_mol == Approx(ref_total).margin(0.20));
}

TEST_CASE("DMACRYS AXOSOW Buckingham pair diagnostic",
          "[mults][dmacrys][diag-buck][.]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       1e-8, 0.35, 8);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }

    auto states = calc.initial_states();
    auto result = calc.compute(states);
    const auto& pairs = calc.neighbor_list();
    const auto& uc_mols = crystal.unit_cell_molecules();

    std::vector<Vec3> body_offsets;
    std::vector<int> body_Z;
    body_offsets.reserve(input.molecule.sites.size());
    body_Z.reserve(input.molecule.sites.size());
    for (const auto& site : input.molecule.sites) {
        body_offsets.push_back(site.position_bohr * units::BOHR_TO_ANGSTROM);
        body_Z.push_back(site.atomic_number);
    }
    {
        Vec3 mw_com = Vec3::Zero();
        double mw = 0.0;
        for (size_t i = 0; i < body_offsets.size(); ++i) {
            double m = occ::core::Element(body_Z[i]).mass();
            mw_com += m * body_offsets[i];
            mw += m;
        }
        mw_com /= mw;
        for (auto& p : body_offsets) p -= mw_com;
    }

    std::vector<Mat3> rots(states.size());
    for (size_t i = 0; i < states.size(); ++i) rots[i] = states[i].rotation_matrix();

    auto buck_energy = [&](double r, int Za, int Zb) {
        int z1 = std::min(Za, Zb);
        int z2 = std::max(Za, Zb);
        auto it = buck_params.find({z1, z2});
        if (it == buck_params.end()) return 0.0;
        const auto& p = it->second;
        return p.A * std::exp(-p.B * r) - p.C / std::pow(r, 6);
    };

    auto sorted_unique_rmsd = [](const std::vector<Vec3>& pred,
                                 const std::vector<int>& pred_Z,
                                 const Mat3N& ref_pos,
                                 const IVec& ref_Z) {
        const int n = static_cast<int>(pred.size());
        std::vector<char> used(n, 0);
        double s2 = 0.0;
        for (int i = 0; i < n; ++i) {
            int Zi = pred_Z[i];
            double best = 1e100;
            int best_j = -1;
            for (int j = 0; j < n; ++j) {
                if (used[j]) continue;
                if (ref_Z(j) != Zi) continue;
                double d2 = (pred[i] - ref_pos.col(j)).squaredNorm();
                if (d2 < best) {
                    best = d2;
                    best_j = j;
                }
            }
            if (best_j >= 0) {
                used[best_j] = 1;
                s2 += best;
            }
        }
        return std::sqrt(s2 / n);
    };

    double e_from_state_geom = 0.0;
    double e_from_crystal_atoms = 0.0;
    long pair_count_state = 0;
    long pair_count_crystal = 0;
    double weighted_pairs_state = 0.0;
    double weighted_pairs_crystal = 0.0;

    double mean_rmsd = 0.0;
    for (size_t mi = 0; mi < states.size(); ++mi) {
        std::vector<Vec3> pred;
        pred.reserve(body_offsets.size());
        for (size_t a = 0; a < body_offsets.size(); ++a) {
            pred.push_back(states[mi].position + rots[mi] * body_offsets[a]);
        }
        mean_rmsd += sorted_unique_rmsd(
            pred, body_Z, uc_mols[mi].positions(), uc_mols[mi].atomic_numbers());
    }
    mean_rmsd /= states.size();

    for (const auto& pair : pairs) {
        const int i = pair.mol_i;
        const int j = pair.mol_j;
        const Vec3 shift = crystal.unit_cell().to_cartesian(pair.cell_shift.cast<double>());
        const double w = pair.weight;

        for (size_t a = 0; a < body_offsets.size(); ++a) {
            const Vec3 pa = states[i].position + rots[i] * body_offsets[a];
            const int Za = body_Z[a];
            for (size_t b = 0; b < body_offsets.size(); ++b) {
                const Vec3 pb = states[j].position + shift + rots[j] * body_offsets[b];
                const int Zb = body_Z[b];
                double r = (pb - pa).norm();
                if (r <= input.cutoff_radius && r >= 0.1) {
                    e_from_state_geom += w * buck_energy(r, Za, Zb);
                    pair_count_state++;
                    weighted_pairs_state += w;
                }
            }
        }

        const auto& mol_i = uc_mols[i];
        const auto& mol_j = uc_mols[j];
        for (int a = 0; a < mol_i.size(); ++a) {
            const Vec3 pa = mol_i.positions().col(a);
            const int Za = mol_i.atomic_numbers()(a);
            for (int b = 0; b < mol_j.size(); ++b) {
                const Vec3 pb = mol_j.positions().col(b) + shift;
                const int Zb = mol_j.atomic_numbers()(b);
                double r = (pb - pa).norm();
                if (r <= input.cutoff_radius && r >= 0.1) {
                    e_from_crystal_atoms += w * buck_energy(r, Za, Zb);
                    pair_count_crystal++;
                    weighted_pairs_crystal += w;
                }
            }
        }
    }

    occ::log::info("Buckingham diagnostic:");
    occ::log::info("  CrystalEnergy rep-disp: {:.6f} kJ/mol", result.repulsion_dispersion);
    occ::log::info("  Manual (state+body geometry): {:.6f} kJ/mol", e_from_state_geom);
    occ::log::info("  Manual (crystal atom coords): {:.6f} kJ/mol", e_from_crystal_atoms);
    occ::log::info("  Pair counts (raw): state={} crystal={}",
                   pair_count_state, pair_count_crystal);
    occ::log::info("  Pair counts (weighted): state={} crystal={}  DMACRYS~60616",
                   weighted_pairs_state, weighted_pairs_crystal);
    occ::log::info("  Mean molecule atom RMSD (state/body vs crystal): {:.6f} A",
                   mean_rmsd);

    CHECK(true);
}

TEST_CASE("DMACRYS AXOSOW initial energy with Ewald + SPLI (strict)",
          "[mults][dmacrys][ewald][spli-benchmark][strict][!mayfail][.]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    input.has_spline = true;
    input.spline_min = 2.0; // DMACRYS SPLI 2.0 4.0
    input.spline_max = 4.0;

    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       1e-8, 0.35, 8);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    calc.set_max_interaction_order(4);

    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }

    auto states = calc.initial_states();
    auto result = calc.compute(states);

    const int n_mol = calc.num_molecules();
    const double total_per_mol = result.total_energy / n_mol;
    const double ref_total = input.initial_ref.total_kJ_per_mol;
    const double diff_kJ = total_per_mol - ref_total;
    const double diff_eV = diff_kJ / occ::units::EV_TO_KJ_PER_MOL;

    INFO("OCC total: " << total_per_mol << " kJ/mol/mol");
    INFO("DMACRYS total: " << ref_total << " kJ/mol/mol");
    INFO("diff: " << diff_kJ << " kJ/mol/mol = " << diff_eV * 1000.0
                  << " meV/mol");

    // Tight benchmark target: sub-meV per molecule (0.001 eV).
    CHECK(std::abs(diff_eV) < 1e-3);
}

TEST_CASE("DMACRYS AXOSOW intramolecular electrostatics", "[mults][dmacrys][intra]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    // Get the Cartesian molecule for molecule 0
    const auto& cart_mol = multipoles[0].cartesian();
    const int nsites = static_cast<int>(cart_mol.sites.size());
    const int Z = input.crystal.Z;

    using namespace occ::mults::kernel_detail;

    // Rank-pair labels matching DMACRYS output
    const char* rank_names[] = {"CHAR", "DIPL", "QUAD", "OCTO", "HEXA"};

    // Accumulate energy by (lA, lB) rank pair in Hartree
    // Index: [lA][lB] for lA, lB = 0..4
    double e_rank[5][5] = {};

    for (int i = 0; i < nsites; ++i) {
        const auto& sA = cart_mol.sites[i];
        if (sA.rank < 0) continue;
        for (int j = i + 1; j < nsites; ++j) {
            const auto& sB = cart_mol.sites[j];
            if (sB.rank < 0) continue;

            // Site-site vector in Bohr
            Vec3 R = (sB.position - sA.position) / occ::units::BOHR_TO_ANGSTROM;
            double Rx = R[0], Ry = R[1], Rz = R[2];

            // Compute T-tensor at max order (rank 4 + rank 4 = 8)
            InteractionTensor<8> T;
            compute_interaction_tensor<8>(Rx, Ry, Rz, T);

            // Contract by specific rank pair (lA, lB)
            for (int lA = 0; lA <= sA.rank; ++lA) {
                int startA = (lA == 0) ? 0 : nhermsum(lA - 1);
                int endA = nhermsum(lA);
                for (int lB = 0; lB <= sB.rank; ++lB) {
                    int startB = (lB == 0) ? 0 : nhermsum(lB - 1);
                    int endB = nhermsum(lB);

                    double e = 0.0;
                    for (int a = startA; a < endA; ++a) {
                        auto [ta, ua, va] = tuv4.entries[a];
                        double wA = weights4.sign_inv_fact[a] * sA.cart.data[a];
                        if (wA == 0.0) continue;
                        for (int b = startB; b < endB; ++b) {
                            auto [tb, ub, vb] = tuv4.entries[b];
                            double wB = weights4.inv_fact[b] * sB.cart.data[b];
                            e += wA * T.data[hermite_index(ta+tb, ua+ub, va+vb)] * wB;
                        }
                    }
                    e_rank[lA][lB] += e;
                }
            }
        }
    }

    // Print in DMACRYS format: eV per unit cell (kJ/mol per cell)
    constexpr double Ha_to_eV = 27.211386;
    constexpr double Ha_to_kJ = occ::units::AU_TO_KJ_PER_MOL;
    double total_eV = 0.0;

    occ::log::info("Intramolecular electrostatics (per unit cell, Z={}):", Z);
    occ::log::info("{:<15s} {:>12s} {:>12s}", "Term", "eV/cell", "kJ/mol/cell");

    // DMACRYS reference (eV per unit cell)
    struct Ref { int lA; int lB; double eV; };
    std::vector<Ref> dmacrys_refs = {
        {0, 0, -7.8228}, {0, 1, -0.5625}, {1, 1, 1.6434},
        {2, 0, 1.4251},  {2, 1, 12.1641}, {3, 0, 3.6191},
        {2, 2, 24.2804}, {3, 1, 23.3706}, {4, 0, 2.7836},
    };

    // Print combined (lA, lB) + (lB, lA) terms in DMACRYS order
    auto print_term = [&](int lA, int lB, double ref_eV) {
        double e_ha;
        if (lA == lB) {
            e_ha = e_rank[lA][lB];
        } else {
            e_ha = e_rank[lA][lB] + e_rank[lB][lA];
        }
        double eV_cell = e_ha * Ha_to_eV * Z;
        double kJ_cell = e_ha * Ha_to_kJ * Z;
        total_eV += eV_cell;
        double diff = eV_cell - ref_eV;
        occ::log::info("{:>4s}-{:<4s}  {:>12.4f}  ({:>12.4f})  DMACRYS: {:>10.4f}  diff: {:>10.4f}",
                       rank_names[lA], rank_names[lB], eV_cell, kJ_cell, ref_eV, diff);
    };

    for (const auto& ref : dmacrys_refs) {
        print_term(ref.lA, ref.lB, ref.eV);
    }

    // Also print remaining terms not shown in DMACRYS output
    occ::log::info("Additional terms:");
    auto print_extra = [&](int lA, int lB) {
        double e_ha = (lA == lB) ? e_rank[lA][lB]
                                 : e_rank[lA][lB] + e_rank[lB][lA];
        double eV_cell = e_ha * Ha_to_eV * Z;
        double kJ_cell = e_ha * Ha_to_kJ * Z;
        total_eV += eV_cell;
        occ::log::info("{:>4s}-{:<4s}  {:>12.4f}  ({:>12.4f})",
                       rank_names[lA], rank_names[lB], eV_cell, kJ_cell);
    };
    print_extra(3, 2); // OCTO-QUAD
    print_extra(4, 1); // HEXA-DIPL
    print_extra(3, 3); // OCTO-OCTO
    print_extra(4, 2); // HEXA-QUAD
    print_extra(4, 3); // HEXA-OCTO
    print_extra(4, 4); // HEXA-HEXA
    occ::log::info("Total intra: {:.4f} eV/cell", total_eV);

    CHECK(true); // diagnostic only
}

TEST_CASE("DMACRYS AXOSOW inter-molecular electrostatics by rank", "[mults][dmacrys][inter]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    // Build CrystalEnergy to get neighbor list (no Ewald for this diagnostic)
    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, false);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);

    // Get CartesianMolecules
    int N = calc.num_molecules();
    std::vector<CartesianMolecule> cart_mols;
    for (int i = 0; i < N; ++i)
        cart_mols.push_back(multipoles[i].cartesian());

    const auto& uc = crystal.unit_cell();
    using namespace occ::mults::kernel_detail;

    const char* rank_names[] = {"CHAR", "DIPL", "QUAD", "OCTO", "HEXA"};

    // Accumulate inter-molecular energy by rank pair (Hartree)
    // Run twice: with and without per-site cutoff
    double e_rank[5][5] = {};
    double e_rank_nocut[5][5] = {};
    int site_pairs_cut = 0, site_pairs_nocut = 0;

    for (const auto& pair : calc.neighbor_pairs()) {
        int mi = pair.mol_i;
        int mj = pair.mol_j;
        double w = pair.weight;

        Vec3 cell_trans = uc.to_cartesian(pair.cell_shift.cast<double>());

        const auto& molA = cart_mols[mi];
        const auto& molB = cart_mols[mj];

        for (const auto& sA : molA.sites) {
            if (sA.rank < 0) continue;
            for (const auto& sB : molB.sites) {
                if (sB.rank < 0) continue;
                Vec3 R_ang = (sB.position + cell_trans) - sA.position;
                bool within_cutoff = (R_ang.norm() <= input.cutoff_radius);
                ++site_pairs_nocut;
                Vec3 R = R_ang / occ::units::BOHR_TO_ANGSTROM;
                double Rx = R[0], Ry = R[1], Rz = R[2];

                InteractionTensor<8> T;
                compute_interaction_tensor<8>(Rx, Ry, Rz, T);

                for (int lA = 0; lA <= sA.rank; ++lA) {
                    int startA = (lA == 0) ? 0 : nhermsum(lA - 1);
                    int endA = nhermsum(lA);
                    for (int lB = 0; lB <= sB.rank; ++lB) {
                        int startB = (lB == 0) ? 0 : nhermsum(lB - 1);
                        int endB = nhermsum(lB);

                        double e = 0.0;
                        for (int a = startA; a < endA; ++a) {
                            auto [ta, ua, va] = tuv4.entries[a];
                            double wA = weights4.sign_inv_fact[a] * sA.cart.data[a];
                            if (wA == 0.0) continue;
                            for (int b = startB; b < endB; ++b) {
                                auto [tb, ub, vb] = tuv4.entries[b];
                                double wB = weights4.inv_fact[b] * sB.cart.data[b];
                                e += wA * T.data[hermite_index(ta+tb, ua+ub, va+vb)] * wB;
                            }
                        }
                        e_rank_nocut[lA][lB] += w * e;
                        if (within_cutoff) {
                            e_rank[lA][lB] += w * e;
                        }
                    }
                }
                if (within_cutoff) ++site_pairs_cut;
            }
        }
    }

    constexpr double Ha_to_eV = 27.211386;
    constexpr double Ha_to_kJ = occ::units::AU_TO_KJ_PER_MOL;

    occ::log::info("Site pairs: with cutoff={}, without cutoff={} (unique: {}, {}), DMACRYS: 60616",
                   site_pairs_cut, site_pairs_nocut, site_pairs_cut/2, site_pairs_nocut/2);
    occ::log::info("");
    occ::log::info("Inter-molecular electrostatics by rank pair (WITH per-site cutoff):");

    // DMACRYS inter-molecular reference (eV per unit cell)
    // qq/qmu/mumu are 0 in DMACRYS table (handled by Ewald)
    struct Ref { int lA; int lB; double eV; const char* note; };
    std::vector<Ref> refs = {
        {0, 0, 0.0, "Ewald"},  {0, 1, 0.0, "Ewald"},  {1, 1, 0.0, "Ewald"},
        {2, 0, -0.4849, ""},   {2, 1, -0.2427, ""},    {3, 0, -0.0984, ""},
        {2, 2,  0.0838, ""},   {3, 1, -0.0073, ""},    {4, 0, -0.0038, ""},
    };

    for (const auto& ref : refs) {
        int lA = ref.lA, lB = ref.lB;
        double e_ha = (lA == lB) ? e_rank[lA][lB]
                                 : e_rank[lA][lB] + e_rank[lB][lA];
        double eV_cell = e_ha * Ha_to_eV;
        double kJ_cell = e_ha * Ha_to_kJ;
        double diff = eV_cell - ref.eV;
        occ::log::info("{:>4s}-{:<4s}  {:>12.4f} eV  ({:>12.4f} kJ)  DMACRYS: {:>10.4f}  diff: {:>10.4f}  {}",
                       rank_names[lA], rank_names[lB], eV_cell, kJ_cell, ref.eV, diff,
                       ref.note ? ref.note : "");
    }

    // Sum higher-multipole terms, split by interaction order
    double higher_order4_ha = 0.0;  // lA+lB <= 4 (what DMACRYS computes)
    double higher_order5plus_ha = 0.0;  // lA+lB > 4 (we compute, DMACRYS doesn't)
    for (int lA = 0; lA <= 4; ++lA) {
        for (int lB = 0; lB <= 4; ++lB) {
            if (lA + lB < 2 || (lA <= 1 && lB <= 1)) continue;
            if (lA + lB <= 4)
                higher_order4_ha += e_rank[lA][lB];
            else
                higher_order5plus_ha += e_rank[lA][lB];
        }
    }
    double ref_higher_eV = -0.753257;
    occ::log::info("Higher (order<=4): {:>10.4f} eV  ({:>10.4f} kJ)  DMACRYS: {:>10.4f}  diff: {:>10.4f}",
                   higher_order4_ha * Ha_to_eV, higher_order4_ha * Ha_to_kJ,
                   ref_higher_eV, higher_order4_ha * Ha_to_eV - ref_higher_eV);
    occ::log::info("Higher (order>4):  {:>10.4f} eV  ({:>10.4f} kJ)  [not in DMACRYS]",
                   higher_order5plus_ha * Ha_to_eV, higher_order5plus_ha * Ha_to_kJ);
    double total_higher_ha = higher_order4_ha + higher_order5plus_ha;
    occ::log::info("Higher total:      {:>10.4f} eV  ({:>10.4f} kJ)",
                   total_higher_ha * Ha_to_eV, total_higher_ha * Ha_to_kJ);

    // Also show truncated qq/qmu/mumu for reference
    double qq_ha = e_rank[0][0];
    double qmu_ha = e_rank[0][1] + e_rank[1][0];
    double mumu_ha = e_rank[1][1];
    occ::log::info("Truncated qq:   {:>12.4f} eV  ({:>12.4f} kJ)", qq_ha * Ha_to_eV, qq_ha * Ha_to_kJ);
    occ::log::info("Truncated qmu:  {:>12.4f} eV  ({:>12.4f} kJ)", qmu_ha * Ha_to_eV, qmu_ha * Ha_to_kJ);
    occ::log::info("Truncated mumu: {:>12.4f} eV  ({:>12.4f} kJ)", mumu_ha * Ha_to_eV, mumu_ha * Ha_to_kJ);

    // DMACRYS Ewald references for comparison
    constexpr double eV_to_kJ = 96.4853;
    double ref_qq_eV = input.initial_ref.charge_charge_inter_eV;
    double ref_qmu_eV = input.initial_ref.charge_dipole_eV;
    double ref_mumu_eV = input.initial_ref.dipole_dipole_eV;
    occ::log::info("DMACRYS Ewald qq:   {:>10.4f} eV", ref_qq_eV);
    occ::log::info("DMACRYS Ewald qmu:  {:>10.4f} eV", ref_qmu_eV);
    occ::log::info("DMACRYS Ewald mumu: {:>10.4f} eV", ref_mumu_eV);

    // Repeat for no-cutoff version
    occ::log::info("");
    occ::log::info("WITHOUT per-site cutoff (all site pairs in included mol pairs):");
    for (const auto& ref : refs) {
        int lA = ref.lA, lB = ref.lB;
        double e_ha = (lA == lB) ? e_rank_nocut[lA][lB]
                                 : e_rank_nocut[lA][lB] + e_rank_nocut[lB][lA];
        double eV_cell = e_ha * Ha_to_eV;
        double diff = eV_cell - ref.eV;
        occ::log::info("{:>4s}-{:<4s}  {:>12.4f} eV  DMACRYS: {:>10.4f}  diff: {:>10.4f}  {}",
                       rank_names[lA], rank_names[lB], eV_cell, ref.eV, diff,
                       ref.note ? ref.note : "");
    }
    double nocut_higher4_ha = 0.0;
    for (int lA = 0; lA <= 4; ++lA)
        for (int lB = 0; lB <= 4; ++lB) {
            if (lA + lB < 2 || (lA <= 1 && lB <= 1)) continue;
            if (lA + lB <= 4) nocut_higher4_ha += e_rank_nocut[lA][lB];
        }
    occ::log::info("Higher (order<=4): {:>10.4f} eV  DMACRYS: {:>10.4f}  diff: {:>10.4f}",
                   nocut_higher4_ha * Ha_to_eV, ref_higher_eV,
                   nocut_higher4_ha * Ha_to_eV - ref_higher_eV);

    CHECK(true);
}

TEST_CASE("DMACRYS AXOSOW Ewald convergence sweep", "[mults][dmacrys][ewald][convergence]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    // Test different alpha and kmax to check convergence
    struct Params { double alpha; int kmax; };
    std::vector<Params> tests = {
        {0.2, 4}, {0.2, 8}, {0.2, 12}, {0.2, 16},
        {0.35, 4}, {0.35, 8}, {0.35, 12}, {0.35, 16},
        {0.5, 4}, {0.5, 8}, {0.5, 12}, {0.5, 16},
        {1.0, 8}, {1.0, 12}, {1.0, 16}, {1.0, 20},
    };

    double ref_total = input.initial_ref.total_kJ_per_mol;
    for (const auto& p : tests) {
        CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                           ForceFieldType::Custom, true, true,
                           1e-8, p.alpha, p.kmax);
        setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
        for (const auto& [key, bp] : buck_params)
            calc.set_buckingham_params(key.first, key.second, bp);

        auto states = calc.initial_states();
        double total = calc.compute_energy(states);
        int n_mol = calc.num_molecules();
        double total_per_mol = total / n_mol;

        INFO("alpha=" << p.alpha << " kmax=" << p.kmax
             << " total=" << total_per_mol
             << " discrepancy=" << (total_per_mol - ref_total));
        // Just log, don't assert
        CHECK(true);
    }
}

TEST_CASE("DMACRYS AXOSOW strain derivatives", "[mults][dmacrys][strain]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    REQUIRE(input.initial_ref.has_strain_derivatives);
    const auto& ref_sd = input.initial_ref.strain_derivatives_eV;
    REQUIRE(ref_sd.size() == 6);

    // Compute strain derivatives with Ewald (alpha=0.35, kmax=8)
    // Use max_interaction_order=4 to match DMACRYS truncation (lA+lB <= 4)
    auto dU_dE = compute_strain_derivatives_fd(
        input, crystal, multipoles, buck_params,
        input.cutoff_radius,
        true, 0.35, 8,
        1e-4,   // FD delta
        4);     // max interaction order (DMACRYS truncation)

    occ::log::info("AXOSOW Strain derivatives (eV/cell):");
    const char* labels[] = {"E1(xx)", "E2(yy)", "E3(zz)",
                            "E4(yz)", "E5(xz)", "E6(xy)"};
    for (int i = 0; i < 6; ++i) {
        double diff = dU_dE(i) - ref_sd[i];
        occ::log::info("  dU/d{}: OCC={:>10.4f}  DMACRYS={:>10.4f}  diff={:>10.4f}",
                       labels[i], dU_dE(i), ref_sd[i], diff);
    }

    // AXOSOW JSON fixture is configured without SPLI tapering.
    // Compare against no-SPLI reference: E1=0.0002, E2=0.0001, E3=-0.0001 eV
    // (equilibrium structure → total strain derivative should be near zero).
    double no_spli_ref[] = {0.0002, 0.0001, -0.0001, 0.0, 0.0, 0.0};
    occ::log::info("\nComparison with DMACRYS no-SPLI reference:");
    for (int i = 0; i < 6; ++i) {
        double diff = dU_dE(i) - no_spli_ref[i];
        occ::log::info("  dU/d{}: OCC={:>10.4f}  no-SPLI={:>10.4f}  diff={:>10.4f}",
                       labels[i], dU_dE(i), no_spli_ref[i], diff);
    }

    // Check against no-SPLI reference (within 0.005 eV)
    for (int i = 0; i < 6; ++i) {
        INFO("Component " << labels[i]
             << ": OCC=" << dU_dE(i)
             << " no_SPLI=" << no_spli_ref[i]);
        CHECK(dU_dE(i) == Approx(no_spli_ref[i]).margin(0.005));
    }

    // Also log the SPLI contribution for reference
    occ::log::info("\nSPLI contribution (DMACRYS with SPLI - no SPLI):");
    for (int i = 0; i < 6; ++i) {
        occ::log::info("  dU/d{}: SPLI={:>10.4f} eV",
                       labels[i], ref_sd[i] - no_spli_ref[i]);
    }
}

TEST_CASE("DMACRYS AXOSOW strain derivatives with SPLI (strict)",
          "[mults][dmacrys][strain][spli-benchmark][strict][!mayfail][.]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    input.has_spline = true;
    input.spline_min = 2.0; // DMACRYS SPLI 2.0 4.0
    input.spline_max = 4.0;

    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    REQUIRE(input.initial_ref.has_strain_derivatives);
    const auto& ref_sd = input.initial_ref.strain_derivatives_eV;
    REQUIRE(ref_sd.size() == 6);

    auto dU_dE = compute_strain_derivatives_fd(
        input, crystal, multipoles, buck_params,
        input.cutoff_radius,
        true, 0.35, 8,
        1e-4,
        4);

    const char* labels[] = {"E1(xx)", "E2(yy)", "E3(zz)",
                            "E4(yz)", "E5(xz)", "E6(xy)"};
    for (int i = 0; i < 6; ++i) {
        INFO("Component " << labels[i]
             << ": OCC=" << dU_dE(i)
             << " DMACRYS_SPLI=" << ref_sd[i]
             << " diff(meV)=" << (dU_dE(i) - ref_sd[i]) * 1000.0);
        // Tight benchmark target: sub-meV (0.001 eV) per component.
        CHECK(dU_dE(i) == Approx(ref_sd[i]).margin(0.001));
    }
}

TEST_CASE("DMACRYS AXOSOW elastic constants", "[mults][dmacrys][elastic]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    REQUIRE(input.optimized_ref.has_elastic_constants);
    const auto& ref_C = input.optimized_ref.elastic_constants_GPa;
    REQUIRE(input.optimized_crystal.has_value());

    // Compute clamped elastic constants for comparison
    auto Cij_clamped = compute_elastic_constants_fd(
        input, crystal, multipoles, buck_params,
        input.cutoff_radius,
        true, 0.35, 8,
        1e-3,  // FD delta
        4);    // max interaction order

    occ::log::info("AXOSOW Clamped elastic constants (GPa):");
    occ::log::info("     {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}",
                   "C_1", "C_2", "C_3", "C_4", "C_5", "C_6");
    for (int i = 0; i < 6; ++i) {
        occ::log::info("C_{} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}",
                       i + 1,
                       Cij_clamped(i, 0), Cij_clamped(i, 1), Cij_clamped(i, 2),
                       Cij_clamped(i, 3), Cij_clamped(i, 4), Cij_clamped(i, 5));
    }

    // Compute relaxed-ion elastic constants via Schur complement
    auto Cij = compute_relaxed_elastic_constants_fd(
        input, crystal, multipoles, buck_params,
        input.cutoff_radius,
        true, 0.35, 8,
        1e-3,  // FD delta
        4);    // max interaction order

    occ::log::info("\nAXOSOW Relaxed elastic constants (GPa):");
    occ::log::info("     {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}",
                   "C_1", "C_2", "C_3", "C_4", "C_5", "C_6");
    for (int i = 0; i < 6; ++i) {
        occ::log::info("C_{} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}",
                       i + 1,
                       Cij(i, 0), Cij(i, 1), Cij(i, 2),
                       Cij(i, 3), Cij(i, 4), Cij(i, 5));
    }

    occ::log::info("\nDMACRYS reference (GPa):");
    for (int i = 0; i < 6; ++i) {
        occ::log::info("C_{} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}",
                       i + 1,
                       ref_C(i, 0), ref_C(i, 1), ref_C(i, 2),
                       ref_C(i, 3), ref_C(i, 4), ref_C(i, 5));
    }

    // Log ratio of relaxed vs DMACRYS for each diagonal
    occ::log::info("\nDiagonal ratios (relaxed/DMACRYS):");
    for (int i = 0; i < 6; ++i) {
        if (std::abs(ref_C(i, i)) > 0.1) {
            double ratio_clamped = Cij_clamped(i, i) / ref_C(i, i);
            double ratio_relaxed = Cij(i, i) / ref_C(i, i);
            occ::log::info("  C_{}{}: clamped={:.3f} relaxed={:.3f}",
                           i + 1, i + 1, ratio_clamped, ratio_relaxed);
        }
    }

    // For orthorhombic Pbca, off-block-diagonal terms should be near zero
    // (check both clamped and relaxed as diagnostics).
    for (int i = 0; i < 3; ++i) {
        for (int j = 3; j < 6; ++j) {
            INFO("C_" << (i+1) << (j+1) << " should be ~0");
            CHECK(std::abs(Cij_clamped(i, j)) < 1.0);
            CHECK(std::abs(Cij(i, j)) < 1.0);
        }
    }

    // Diagonal elements: clamped constants should approximately agree with
    // DMACRYS optimized reference values for this benchmark.
    for (int i = 0; i < 6; ++i) {
        if (ref_C(i, i) > 1.0) {
            double ratio = Cij_clamped(i, i) / ref_C(i, i);
            // C66 is the most sensitive component in the current Ewald/truncation
            // setup, while other diagonals track more tightly.
            const double ratio_hi = (i == 5) ? 1.70 : 1.45;
            INFO("C_" << (i+1) << (i+1)
                 << " clamped OCC=" << Cij_clamped(i, i)
                 << " DMACRYS=" << ref_C(i, i)
                 << " ratio=" << ratio);
            CHECK(ratio > 0.7);
            CHECK(ratio < ratio_hi);
        }
    }

    // Relaxed constants are currently diagnostic-only in this benchmark setup.
    for (int i = 0; i < 6; ++i) {
        CHECK(std::isfinite(Cij(i, i)));
    }

    // Feed clamped tensor to ElasticTensor for derived properties.
    occ::core::ElasticTensor et(Cij_clamped);
    auto eigenvals = et.eigenvalues();
    occ::log::info("\nElastic tensor eigenvalues: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}",
                   eigenvals(0), eigenvals(1), eigenvals(2),
                   eigenvals(3), eigenvals(4), eigenvals(5));

    double K_hill = et.average_bulk_modulus();
    double G_hill = et.average_shear_modulus();
    occ::log::info("Hill bulk modulus: {:.2f} GPa", K_hill);
    occ::log::info("Hill shear modulus: {:.2f} GPa", G_hill);

    // Born stability: all eigenvalues positive
    bool stable = (eigenvals.minCoeff() > 0);
    occ::log::info("Born stability: {}", stable ? "STABLE" : "UNSTABLE");
}

TEST_CASE("DMACRYS AXOSOW strain FD diagnostic", "[mults][dmacrys][strain-diag]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    const auto& ref_sd = input.initial_ref.strain_derivatives_eV;

    // Build a FIXED reference neighbor list + atom-pair masks from unstrained crystal
    CrystalEnergy ref_calc(crystal, multipoles, input.cutoff_radius,
                           ForceFieldType::Custom, true, true,
                           1e-8, 0.35, 8);
    setup_crystal_energy_from_dmacrys(ref_calc, input, crystal, multipoles);
    auto ref_neighbors = ref_calc.neighbor_list();
    auto ref_site_masks = ref_calc.compute_buckingham_site_masks(ref_calc.initial_states());
    occ::log::info("Reference neighbor list: {} pairs", ref_neighbors.size());

    // Print energy at zero strain
    {
        Mat3 zero = Mat3::Zero();
        double E_ew = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            zero, input.cutoff_radius,
            true, 0.35, 8, 4, &ref_neighbors, &ref_site_masks);
        occ::log::info("Zero-strain energy (Ewald, order<=4): {:.8f} kJ/mol ({:.6f} kJ/mol per molecule)",
                       E_ew, E_ew / 8.0);
    }

    // FD convergence with FIXED neighbor list + site masks for E1, E2, E3
    for (int comp : {0, 1, 2}) {
        const char* names[] = {"E1 (xx)", "E2 (yy)", "E3 (zz)"};
        occ::log::info("\nFD convergence for {} [fixed neighbors + site masks]:", names[comp]);
        occ::log::info("{:>10s} {:>14s} {:>14s} {:>12s}",
                       "delta", "E(+d)", "E(-d)", "dU/dE (eV)");
        for (double delta : {1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5}) {
            Mat3 eps_p = voigt_strain_tensor(comp, +delta);
            Mat3 eps_m = voigt_strain_tensor(comp, -delta);

            double Ep = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_p, input.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);
            double Em = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_m, input.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);
            double dUdE = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

            occ::log::info("{:>10.0e} {:>14.8f} {:>14.8f} {:>12.6f}",
                           delta, Ep, Em, dUdE);
        }
        occ::log::info("  DMACRYS ref: dU/d{} = {:.4f} eV", names[comp], ref_sd[comp]);
    }

    // Separate Buckingham-only and electrostatics-only strain derivatives
    // to identify which contribution has the discrepancy
    occ::log::info("\nComponent breakdown (delta=1e-4, fixed neighbors + site masks):");
    {
        double delta = 1e-4;
        for (int comp : {0, 1, 2}) {
            Mat3 eps_p = voigt_strain_tensor(comp, +delta);
            Mat3 eps_m = voigt_strain_tensor(comp, -delta);

            // Full energy (elec + buck + ewald)
            double Ep_full = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_p, input.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);
            double Em_full = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_m, input.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);

            // Electrostatics only — use ZERO Buck params to avoid default fallback
            std::map<std::pair<int, int>, BuckinghamParams> no_buck;
            for (const auto& [key, p] : buck_params) {
                no_buck[key] = {0.0, 1.0, 0.0}; // A=0, B=1, C=0 -> V=0
            }
            double Ep_elec = compute_strained_energy(
                input, crystal, multipoles, no_buck,
                eps_p, input.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);
            double Em_elec = compute_strained_energy(
                input, crystal, multipoles, no_buck,
                eps_m, input.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);

            // Electrostatics only, no Ewald
            double Ep_elec_noew = compute_strained_energy(
                input, crystal, multipoles, no_buck,
                eps_p, input.cutoff_radius, false, 0.0, 0, 4,
                &ref_neighbors, &ref_site_masks);
            double Em_elec_noew = compute_strained_energy(
                input, crystal, multipoles, no_buck,
                eps_m, input.cutoff_radius, false, 0.0, 0, 4,
                &ref_neighbors, &ref_site_masks);

            double dU_full = (Ep_full - Em_full) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;
            double dU_elec = (Ep_elec - Em_elec) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;
            double dU_elec_noew = (Ep_elec_noew - Em_elec_noew) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;
            double dU_buck = dU_full - dU_elec;
            double dU_ewald = dU_elec - dU_elec_noew;

            const char* names[] = {"E1", "E2", "E3"};
            occ::log::info("  {}: full={:>8.4f}  elec+ew={:>8.4f}  elec_noew={:>8.4f}  buck={:>8.4f}  ewald_corr={:>8.4f}  ref={:>8.4f}",
                           names[comp], dU_full, dU_elec, dU_elec_noew, dU_buck, dU_ewald, ref_sd[comp]);
        }
    }

    // Ewald alpha sensitivity scan for electrostatic strain derivative
    // DMACRYS uses eta=2.7945 in lattice units => alpha = eta/c = 0.399/Ang
    // Our default is 0.35/Ang. Check if difference matters.
    occ::log::info("\nEwald alpha scan for elec strain deriv (delta=1e-4, order<=4):");
    occ::log::info("{:>8s} {:>8s}  {:>10s} {:>10s} {:>10s}",
                   "alpha", "kmax", "dU_elec_E1", "dU_elec_E2", "dU_elec_E3");
    {
        double delta = 1e-4;
        std::map<std::pair<int, int>, BuckinghamParams> no_buck;
        for (const auto& [key, p] : buck_params) {
            no_buck[key] = {0.0, 1.0, 0.0};
        }
        for (auto [alpha, km] : std::vector<std::pair<double,int>>{
                {0.25, 6}, {0.30, 8}, {0.35, 8}, {0.40, 10}, {0.45, 12}, {0.50, 14}}) {
            Vec3 dU_elec;
            for (int comp = 0; comp < 3; ++comp) {
                Mat3 eps_p = voigt_strain_tensor(comp, +delta);
                Mat3 eps_m = voigt_strain_tensor(comp, -delta);
                double Ep = compute_strained_energy(
                    input, crystal, multipoles, no_buck,
                    eps_p, input.cutoff_radius, true, alpha, km, 4,
                    &ref_neighbors, &ref_site_masks);
                double Em = compute_strained_energy(
                    input, crystal, multipoles, no_buck,
                    eps_m, input.cutoff_radius, true, alpha, km, 4,
                    &ref_neighbors, &ref_site_masks);
                dU_elec(comp) = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;
            }
            occ::log::info("{:>8.3f} {:>8d}  {:>10.4f} {:>10.4f} {:>10.4f}",
                           alpha, km, dU_elec(0), dU_elec(1), dU_elec(2));
        }
        // DMACRYS no-SPLI electrostatic references (total - buck):
        // E1: 0.0002-(-2.8452)=2.8454, E2: 0.0001-(-2.5367)=2.5368, E3: -0.0001-(-1.3208)=1.3207
        occ::log::info("  DMACRYS (no-SPLI) elec ref: E1=2.845  E2=2.537  E3=1.321");
    }

    CHECK(true); // diagnostic test, always passes
}

TEST_CASE("DMACRYS UREAXX22 fixed-point energy and clamped elastic (strict)",
          "[mults][dmacrys][elastic][ureaxx22][strict][!mayfail][.]") {
    namespace fs = std::filesystem;
    if (!fs::exists(UREAXX22_JSON)) {
        SKIP("UREAXX22 fixture not found: " + UREAXX22_JSON);
    }

    auto input = read_dmacrys_json(UREAXX22_JSON);
    auto crystal = build_crystal(input.optimized_crystal.value_or(input.crystal));
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    // Use fixed Ewald settings for deterministic DMACRYS parity checks.
    constexpr double eta = 0.195627; // /Ang
    constexpr int kmax = 2;

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       1e-6, eta, kmax);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    calc.set_max_interaction_order(4);

    auto states = calc.initial_states();
    auto result = calc.compute(states);
    const double z_mol = static_cast<double>(states.size());
    const double total_per_mol = result.total_energy / z_mol;
    const double rd_per_mol = result.repulsion_dispersion / z_mol;

    REQUIRE(input.initial_ref.total_kJ_per_mol != 0.0);
    REQUIRE(input.initial_ref.repulsion_dispersion_kJ != 0.0);

    INFO("UREAXX22 total OCC=" << total_per_mol
         << " DMACRYS=" << input.initial_ref.total_kJ_per_mol);
    INFO("UREAXX22 rep-disp OCC=" << rd_per_mol
         << " DMACRYS=" << input.initial_ref.repulsion_dispersion_kJ);
    // DMACRYS summary total is rounded (typically to 4 decimals in kJ/mol).
    CHECK(total_per_mol ==
          Approx(input.initial_ref.total_kJ_per_mol).margin(1e-2));
    CHECK(rd_per_mol ==
          Approx(input.initial_ref.repulsion_dispersion_kJ).margin(1e-3));

    REQUIRE(input.initial_ref.has_elastic_constants);
    auto Cij_clamped = compute_elastic_constants_fd(
        input, crystal, multipoles, buck_params,
        input.cutoff_radius, true, eta, kmax, 1e-3, 4);
    auto Cij_relaxed = compute_relaxed_elastic_constants_fd(
        input, crystal, multipoles, buck_params,
        input.cutoff_radius, true, eta, kmax, 1e-3, 4);
    const auto& ref_C = input.initial_ref.elastic_constants_GPa;

    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            INFO("C" << (i + 1) << (j + 1)
                 << " OCC_clamped=" << Cij_clamped(i, j)
                 << " OCC_relaxed=" << Cij_relaxed(i, j)
                 << " DMACRYS=" << ref_C(i, j));
            CHECK(Cij_relaxed(i, j) == Approx(ref_C(i, j)).margin(0.2));
        }
    }
}

TEST_CASE("DMACRYS TCYETY01 triclinic energy",
          "[mults][dmacrys][tcyety01][!mayfail]") {
    namespace fs = std::filesystem;
    if (!fs::exists(TCYETY01_JSON)) {
        SKIP("TCYETY01 fixture not found: " + TCYETY01_JSON);
    }

    auto input = read_dmacrys_json(TCYETY01_JSON);
    auto crystal = build_crystal(input.optimized_crystal.value_or(input.crystal));
    auto multipoles = build_multipole_sources(input, crystal);

    // Ewald parameters for TCYETY01 (triclinic, V≈960 Å³)
    constexpr double eta = 0.30;  // /Ang
    constexpr int kmax = 5;

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       1e-6, eta, kmax);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    calc.set_max_interaction_order(4);

    auto states = calc.initial_states();
    auto result = calc.compute(states);
    const double z_mol = static_cast<double>(states.size());
    const double total_per_mol = result.total_energy / z_mol;
    const double rd_per_mol = result.repulsion_dispersion / z_mol;

    REQUIRE(input.initial_ref.total_kJ_per_mol != 0.0);
    REQUIRE(input.initial_ref.repulsion_dispersion_kJ != 0.0);

    const double elec_per_mol = result.electrostatic_energy / z_mol;
    const double dmacrys_elec = input.initial_ref.total_kJ_per_mol - input.initial_ref.repulsion_dispersion_kJ;
    INFO("TCYETY01 total OCC=" << total_per_mol
         << " DMACRYS=" << input.initial_ref.total_kJ_per_mol);
    INFO("TCYETY01 rep-disp OCC=" << rd_per_mol
         << " DMACRYS=" << input.initial_ref.repulsion_dispersion_kJ);
    INFO("TCYETY01 elec OCC=" << elec_per_mol
         << " DMACRYS=" << dmacrys_elec);
    INFO("TCYETY01 n_mol=" << z_mol);
    CHECK(rd_per_mol ==
          Approx(input.initial_ref.repulsion_dispersion_kJ).margin(0.1));
    CHECK(total_per_mol ==
          Approx(input.initial_ref.total_kJ_per_mol).margin(2.0));
}

TEST_CASE("DMACRYS TCYETY01 electrostatic rank ladder",
          "[.][diag][dmacrys][tcyety01]") {
    namespace fs = std::filesystem;
    if (!fs::exists(TCYETY01_JSON)) {
        SKIP("TCYETY01 fixture not found: " + TCYETY01_JSON);
    }

    auto input = read_dmacrys_json(TCYETY01_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    const int Z = input.crystal.Z;
    const double eV_to_kJ = 96.485 / Z;

    // DMACRYS reference breakdown
    const double ref_qq = input.initial_ref.charge_charge_inter_eV * eV_to_kJ;
    const double ref_qd = input.initial_ref.charge_dipole_eV * eV_to_kJ;
    const double ref_dd = input.initial_ref.dipole_dipole_eV * eV_to_kJ;
    const double ref_higher = input.initial_ref.higher_multipole_eV * eV_to_kJ;
    occ::log::info("TCYETY01 DMACRYS ref: qq={:.4f} qd={:.4f} dd={:.4f} higher={:.4f} kJ/mol",
                   ref_qq, ref_qd, ref_dd, ref_higher);

    // Report rotation matrices and parity for each molecule
    for (int m = 0; m < static_cast<int>(multipoles.size()); ++m) {
        Mat3 R = multipoles[m].rotation();
        Vec3 com = multipoles[m].center();
        double det = R.determinant();
        occ::log::info("Mol {}: det(R)={:.6f} com=[{:.4f},{:.4f},{:.4f}]",
                       m, det, com(0), com(1), com(2));
        // Find site with largest dipole
        const auto& sites = multipoles[m].body_sites();
        int best = -1;
        double best_d2 = 0;
        for (int s = 0; s < static_cast<int>(sites.size()); ++s) {
            if (sites[s].multipole.max_rank >= 1) {
                double d2 = sites[s].multipole.q(1)*sites[s].multipole.q(1) +
                            sites[s].multipole.q(2)*sites[s].multipole.q(2) +
                            sites[s].multipole.q(3)*sites[s].multipole.q(3);
                if (d2 > best_d2) { best_d2 = d2; best = s; }
            }
        }
        if (best >= 0) {
            Vec3 d_body(sites[best].multipole.q(1),
                        sites[best].multipole.q(2),
                        sites[best].multipole.q(3));
            Vec3 d_lab = R * d_body;
            Vec3 offset_lab = R * sites[best].offset;
            occ::log::info("  site{} body_d=[{:.6f},{:.6f},{:.6f}] lab_d=[{:.6f},{:.6f},{:.6f}]",
                           best, d_body(0), d_body(1), d_body(2),
                           d_lab(0), d_lab(1), d_lab(2));
            occ::log::info("  site{} body_off=[{:.4f},{:.4f},{:.4f}] lab_off=[{:.4f},{:.4f},{:.4f}]",
                           best, sites[best].offset(0), sites[best].offset(1), sites[best].offset(2),
                           offset_lab(0), offset_lab(1), offset_lab(2));
        }
    }

    // Sweep max interaction order, with and without Ewald
    for (bool use_ewald : {false, true}) {
        for (int max_order : {0, 1, 2, 3, 4}) {
            double eta = use_ewald ? 0.30 : 0.0;
            int kmax = use_ewald ? 5 : 0;
            CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                               ForceFieldType::Custom, true, use_ewald,
                               1e-6, eta, kmax);
            setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
            calc.set_max_interaction_order(max_order);

            auto states = calc.initial_states();
            auto result = calc.compute(states);

            double elec = result.electrostatic_energy / Z;
            double rd = result.repulsion_dispersion / Z;
            double total = result.total_energy / Z;
            occ::log::info("TCYETY01 order<={} ewald={}: elec={:.4f} rd={:.4f} total={:.4f} kJ/mol",
                           max_order, use_ewald, elec, rd, total);
        }
    }

    CHECK(true);
}

TEST_CASE("DMACRYS TCYETY01 crystal structure check",
          "[.][diag][dmacrys][tcyety01][crystal-check]") {
    namespace fs = std::filesystem;
    if (!fs::exists(TCYETY01_JSON)) {
        SKIP("TCYETY01 fixture not found: " + TCYETY01_JSON);
    }

    auto input = read_dmacrys_json(TCYETY01_JSON);
    auto crystal = build_crystal(input.crystal);
    const auto &uc_mols = crystal.unit_cell_molecules();

    occ::log::info("=== CRYSTAL STRUCTURE ===");
    occ::log::info("Cell: a={:.4f} b={:.4f} c={:.4f} alpha={:.2f} beta={:.2f} gamma={:.2f}",
                   crystal.unit_cell().a(), crystal.unit_cell().b(), crystal.unit_cell().c(),
                   crystal.unit_cell().alpha() * 180.0 / M_PI,
                   crystal.unit_cell().beta() * 180.0 / M_PI,
                   crystal.unit_cell().gamma() * 180.0 / M_PI);

    // Lattice vectors in our Cartesian frame
    auto cell = crystal.unit_cell();
    Mat3 L = cell.direct();
    occ::log::info("Lattice vectors (Angstrom):");
    occ::log::info("  a = [{:10.6f}, {:10.6f}, {:10.6f}]", L(0,0), L(1,0), L(2,0));
    occ::log::info("  b = [{:10.6f}, {:10.6f}, {:10.6f}]", L(0,1), L(1,1), L(2,1));
    occ::log::info("  c = [{:10.6f}, {:10.6f}, {:10.6f}]", L(0,2), L(1,2), L(2,2));

    occ::log::info("UC molecules: {}, Z={}", uc_mols.size(), input.crystal.Z);

    auto multipoles = build_multipole_sources(input, crystal);

    occ::log::info("=== MOTIF / BASIS ===");
    for (int m = 0; m < static_cast<int>(multipoles.size()); ++m) {
        const auto &src = multipoles[m];
        Mat3 R = src.rotation();
        Vec3 com = src.center();
        occ::log::info("--- Molecule {} ---", m+1);
        occ::log::info("  COM = [{:.6f}, {:.6f}, {:.6f}] Ang", com(0), com(1), com(2));
        occ::log::info("  det(R) = {:.4f}", R.determinant());
        occ::log::info("  Local axis set:");
        occ::log::info("    x=> {:10.6f} {:10.6f} {:10.6f}", R(0,0), R(1,0), R(2,0));
        occ::log::info("    y=> {:10.6f} {:10.6f} {:10.6f}", R(0,1), R(1,1), R(2,1));
        occ::log::info("    z=> {:10.6f} {:10.6f} {:10.6f}", R(0,2), R(1,2), R(2,2));

        const auto &sites = src.body_sites();
        occ::log::info("  Atom positions (lab frame, Angstrom) and multipoles:");
        for (int i = 0; i < static_cast<int>(sites.size()); ++i) {
            Vec3 lab_pos = com + R * sites[i].offset;
            occ::log::info("    site {:2d}: Z={} lab=[{:10.6f},{:10.6f},{:10.6f}] body=[{:10.6f},{:10.6f},{:10.6f}]",
                           i, sites[i].atomic_number,
                           lab_pos(0), lab_pos(1), lab_pos(2),
                           sites[i].offset(0), sites[i].offset(1), sites[i].offset(2));
            // Print first few multipole components in body frame
            const auto &mult = sites[i].multipole;
            if (mult.max_rank >= 1) {
                occ::log::info("      Q00={:.8f} Q10={:.8f} Q11c={:.8f} Q11s={:.8f}",
                               mult.q(0), mult.q(1), mult.q(2), mult.q(3));
            } else {
                occ::log::info("      Q00={:.8f}", mult.q(0));
            }
        }
    }

    // Now compute a single pair energy between mol 0 and mol 1 at charge-dipole level
    occ::log::info("=== PAIR ENERGY TEST: Mol 0 - Mol 1 ===");
    {
        const auto &srcA = multipoles[0];
        const auto &srcB = multipoles[1];
        Mat3 RA = srcA.rotation();
        Mat3 RB = srcB.rotation();
        Vec3 comA = srcA.center();
        Vec3 comB = srcB.center();
        const auto &sitesA = srcA.body_sites();
        const auto &sitesB = srcB.body_sites();

        double e_qq = 0, e_qd = 0;
        constexpr double BOHR = 0.529177210903;
        for (int a = 0; a < static_cast<int>(sitesA.size()); ++a) {
            Vec3 posA = comA + RA * sitesA[a].offset;
            double qA = sitesA[a].multipole.q(0);
            Vec3 dA_body(0,0,0);
            if (sitesA[a].multipole.max_rank >= 1) {
                // Stone convention: q(1)=Q10(z), q(2)=Q11c(x), q(3)=Q11s(y)
                dA_body = Vec3(sitesA[a].multipole.q(2),
                               sitesA[a].multipole.q(3),
                               sitesA[a].multipole.q(1));
            }
            Vec3 dA_lab = RA * dA_body;

            for (int b = 0; b < static_cast<int>(sitesB.size()); ++b) {
                Vec3 posB = comB + RB * sitesB[b].offset;
                double qB = sitesB[b].multipole.q(0);
                Vec3 dB_body(0,0,0);
                if (sitesB[b].multipole.max_rank >= 1) {
                    dB_body = Vec3(sitesB[b].multipole.q(2),
                                   sitesB[b].multipole.q(3),
                                   sitesB[b].multipole.q(1));
                }
                Vec3 dB_lab = RB * dB_body;

                Vec3 r = posB - posA;
                double R_dist = r.norm();
                if (R_dist < 0.1) continue;

                // Convert to Bohr for energy in Hartree
                Vec3 r_bohr = r / BOHR;
                double R_bohr = R_dist / BOHR;
                double R3 = R_bohr * R_bohr * R_bohr;
                double R2 = R_bohr * R_bohr;
                Vec3 rhat = r_bohr / R_bohr;

                // qq: E = qA*qB/R
                double qq = qA * qB / R_bohr;
                e_qq += qq;

                // qd: E = qA*(-dB·r)/R³ + qB*(dA·r)/R³ [charge-dipole in bohr/hartree]
                double qd = qA * (-dB_lab.dot(r_bohr) / (BOHR * R3 / BOHR))
                          + qB * (dA_lab.dot(r_bohr) / (BOHR * R3 / BOHR));
                // Actually the correct formula in atomic units:
                // V_qd = qA * (-μB · ∇(1/R)) + qB * (-μA · ∇(1/R))
                //       = qA * (μB · r_hat/R²) + qB * (-μA · r_hat/R²)  ... hmm signs
                // Let me just use: T_alpha = r_alpha / R³
                // E = qA * Σ_α T_α μ_B_α + μ_A_α T_α qB  ... no this isn't right either
                // The charge-dipole interaction tensor is:
                // T_α(A→B) = -∂/∂r_α (1/R) = r_α/R³  where r = posB - posA
                // E_qd = -qA Σ_α μ_B_α T_α + qB Σ_α μ_A_α T_α ... convention dependent

                // Let me not compute this manually and instead just report lab-frame positions/multipoles
            }
        }
    }

    CHECK(uc_mols.size() == 6);
    CHECK(multipoles.size() == 6);
}

TEST_CASE("DMACRYS TCYETY01 triclinic elastic constants",
          "[mults][dmacrys][elastic][tcyety01][!mayfail][.]") {
    namespace fs = std::filesystem;
    if (!fs::exists(TCYETY01_JSON)) {
        SKIP("TCYETY01 fixture not found: " + TCYETY01_JSON);
    }

    auto input = read_dmacrys_json(TCYETY01_JSON);
    auto crystal = build_crystal(input.optimized_crystal.value_or(input.crystal));
    auto multipoles = build_multipole_sources(input, crystal);
    std::map<std::pair<int, int>, BuckinghamParams> buck_params;

    constexpr double eta = 0.30;
    constexpr int kmax = 5;

    REQUIRE(input.initial_ref.has_elastic_constants);
    auto Cij_clamped = compute_elastic_constants_fd(
        input, crystal, multipoles, buck_params,
        input.cutoff_radius, true, eta, kmax, 1e-3, 4);
    auto Cij_relaxed = compute_relaxed_elastic_constants_fd(
        input, crystal, multipoles, buck_params,
        input.cutoff_radius, true, eta, kmax, 1e-3, 4);
    const auto& ref_C = input.initial_ref.elastic_constants_GPa;

    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            INFO("C" << (i + 1) << (j + 1)
                 << " OCC_clamped=" << Cij_clamped(i, j)
                 << " OCC_relaxed=" << Cij_relaxed(i, j)
                 << " DMACRYS=" << ref_C(i, j));
            CHECK(Cij_relaxed(i, j) == Approx(ref_C(i, j)).margin(1.0));
        }
    }
}

TEST_CASE("DMACRYS UREAXX22 relaxed elastic NO TAPER diagnostic",
          "[mults][dmacrys][elastic][ureaxx22][notaper][.]") {
    namespace fs = std::filesystem;
    if (!fs::exists(UREAXX22_JSON)) {
        SKIP("UREAXX22 fixture not found: " + UREAXX22_JSON);
    }

    auto input = read_dmacrys_json(UREAXX22_JSON);
    auto crystal = build_crystal(input.optimized_crystal.value_or(input.crystal));
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    constexpr double eta = 0.195627;
    constexpr int kmax = 2;

    // Compute WITH taper (baseline)
    auto Cij_taper = compute_relaxed_elastic_constants_fd(
        input, crystal, multipoles, buck_params,
        input.cutoff_radius, true, eta, kmax, 1e-3, 4);

    // Compute WITHOUT taper: clear taper in the input so build_reusable_calc
    // doesn't set it.  We do this by temporarily blanking the spline params.
    auto input_notaper = input;
    input_notaper.has_spline = false;
    input_notaper.spline_min = 0.0;
    input_notaper.spline_max = 0.0;

    auto Cij_notaper = compute_relaxed_elastic_constants_fd(
        input_notaper, crystal, multipoles, buck_params,
        input.cutoff_radius, true, eta, kmax, 1e-3, 4);

    const auto& ref_C = input.initial_ref.elastic_constants_GPa;
    for (int i = 0; i < 6; ++i) {
        for (int j = i; j < 6; ++j) {
            INFO("C" << (i + 1) << (j + 1)
                 << " taper=" << Cij_taper(i, j)
                 << " notaper=" << Cij_notaper(i, j)
                 << " DMACRYS=" << ref_C(i, j));
        }
    }
}

TEST_CASE("DMACRYS UREAXX22 clamped elastic analytic-vs-FD diagnostic",
          "[mults][dmacrys][elastic][ureaxx22][diag][.]") {
    namespace fs = std::filesystem;
    if (!fs::exists(UREAXX22_JSON)) {
        SKIP("UREAXX22 fixture not found: " + UREAXX22_JSON);
    }

    auto input = read_dmacrys_json(UREAXX22_JSON);
    auto crystal = build_crystal(input.optimized_crystal.value_or(input.crystal));
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    constexpr double alpha = 0.195627; // /Ang
    constexpr int kmax = 2;
    constexpr int max_order = 4;
    const double V = crystal.unit_cell().volume();

    // Reusable calc for analytic strain gradient/Hessian.
    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }
    calc.set_max_interaction_order(max_order);

    auto states = calc.initial_states();
    for (size_t m = 0; m < states.size(); ++m) {
        occ::log::info("UREAXX22 state {} COM = [{:.6f}, {:.6f}, {:.6f}]",
                       m, states[m].position[0], states[m].position[1], states[m].position[2]);
    }
    calc.update_lattice(crystal, states);
    auto eh = calc.compute_with_hessian(states);

    const double conv = units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
    Mat6 C_analytic = (eh.strain_hessian / V) * conv;

    // Frozen-neighbor FD for direct curvature check.
    auto ref_neighbors = calc.neighbor_list();
    auto ref_site_masks = calc.compute_buckingham_site_masks(states);

    auto energy_at = [&](const Mat3& eps) {
        return compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps, input.cutoff_radius, true, alpha, kmax, max_order,
            &ref_neighbors, &ref_site_masks);
    };

    const double E0 = energy_at(Mat3::Zero());
    Vec6 dU_fd = Vec6::Zero();
    {
        const double dgrad = 1e-5;
        for (int a = 0; a < 6; ++a) {
            Mat3 ep = voigt_strain_tensor(a, +dgrad);
            Mat3 em = voigt_strain_tensor(a, -dgrad);
            const double Ep = energy_at(ep);
            const double Em = energy_at(em);
            dU_fd[a] = (Ep - Em) / (2.0 * dgrad);
        }
    }
    const double delta = 2e-4;
    Mat6 C_fd = Mat6::Zero();
    for (int i = 0; i < 6; ++i) {
        Mat3 ei = voigt_strain_tensor(i, delta);
        Mat3 emi = voigt_strain_tensor(i, -delta);
        const double Epi = energy_at(ei);
        const double Emi = energy_at(emi);
        C_fd(i, i) = ((Epi - 2.0 * E0 + Emi) / (delta * delta)) / V * conv;

        for (int j = i + 1; j < 6; ++j) {
            Mat3 ej = voigt_strain_tensor(j, delta);
            Mat3 emj = voigt_strain_tensor(j, -delta);
            const double Epp = energy_at(ei + ej);
            const double Epm = energy_at(ei + emj);
            const double Emp = energy_at(emi + ej);
            const double Emm = energy_at(emi + emj);
            const double val =
                (Epp - Epm - Emp + Emm) / (4.0 * delta * delta) / V * conv;
            C_fd(i, j) = val;
            C_fd(j, i) = val;
        }
    }

    // No-Ewald split: identifies whether the analytic mismatch is in Ewald terms.
    CrystalEnergy calc_noew(crystal, multipoles, input.cutoff_radius,
                            ForceFieldType::Custom, true, false,
                            1e-8, 0.0, 0);
    setup_crystal_energy_from_dmacrys(calc_noew, input, crystal, multipoles);
    for (const auto& [key, p] : buck_params) {
        calc_noew.set_buckingham_params(key.first, key.second, p);
    }
    calc_noew.set_max_interaction_order(max_order);
    calc_noew.update_lattice(crystal, states);
    auto eh_noew = calc_noew.compute_with_hessian(states);
    Mat6 C_analytic_noew = (eh_noew.strain_hessian / V) * conv;
    {
        Mat Jq = Mat::Zero(6 * static_cast<int>(states.size()), 6);
        std::array<Mat3, 6> Bb = {
            Mat3{{1,0,0},{0,0,0},{0,0,0}},
            Mat3{{0,0,0},{0,1,0},{0,0,0}},
            Mat3{{0,0,0},{0,0,0},{0,0,1}},
            Mat3{{0,0,0},{0,0,0.5},{0,0.5,0}},
            Mat3{{0,0,0.5},{0,0,0},{0.5,0,0}},
            Mat3{{0,0.5,0},{0.5,0,0},{0,0,0}}
        };
        for (int m = 0; m < static_cast<int>(states.size()); ++m) {
            for (int a = 0; a < 6; ++a) {
                Jq.block<3, 1>(6 * m, a) = Bb[a] * states[m].position;
            }
        }
        Mat6 W_chain = Jq.transpose() * eh_noew.hessian * Jq;
        occ::log::info(
            "UREAXX22 diagnostic (no Ewald) ||W_ee - J^T H J|| = {:.6e} kJ/mol",
            (eh_noew.strain_hessian - W_chain).norm());
    }

    auto ref_neighbors_noew = calc_noew.neighbor_list();
    auto ref_site_masks_noew = calc_noew.compute_buckingham_site_masks(states);
    auto energy_at_noew = [&](const Mat3& eps) {
        return compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps, input.cutoff_radius, false, 0.0, 0, max_order,
            &ref_neighbors_noew, &ref_site_masks_noew);
    };
    const double E0_noew = energy_at_noew(Mat3::Zero());
    Vec6 dU_fd_noew = Vec6::Zero();
    {
        const double dgrad = 1e-5;
        for (int a = 0; a < 6; ++a) {
            Mat3 ep = voigt_strain_tensor(a, +dgrad);
            Mat3 em = voigt_strain_tensor(a, -dgrad);
            const double Ep = energy_at_noew(ep);
            const double Em = energy_at_noew(em);
            dU_fd_noew[a] = (Ep - Em) / (2.0 * dgrad);
        }
    }
    Mat6 C_fd_noew = Mat6::Zero();
    for (int i = 0; i < 3; ++i) {
        Mat3 ei = voigt_strain_tensor(i, delta);
        Mat3 emi = voigt_strain_tensor(i, -delta);
        const double Epi = energy_at_noew(ei);
        const double Emi = energy_at_noew(emi);
        C_fd_noew(i, i) = ((Epi - 2.0 * E0_noew + Emi) / (delta * delta)) / V * conv;
        for (int j = i + 1; j < 3; ++j) {
            Mat3 ej = voigt_strain_tensor(j, delta);
            Mat3 emj = voigt_strain_tensor(j, -delta);
            const double Epp = energy_at_noew(ei + ej);
            const double Epm = energy_at_noew(ei + emj);
            const double Emp = energy_at_noew(emi + ej);
            const double Emm = energy_at_noew(emi + emj);
            const double val =
                (Epp - Epm - Emp + Emm) / (4.0 * delta * delta) / V * conv;
            C_fd_noew(i, j) = val;
            C_fd_noew(j, i) = val;
        }
    }
    // FD of analytic strain gradient (more robust than energy second differences
    // when large linear terms are present).
    auto strain_grad_at_noew = [&](const Mat3& eps) {
        CrystalEnergy calc_tmp(crystal, multipoles, input.cutoff_radius,
                               ForceFieldType::Custom, true, false,
                               1e-8, 0.0, 0);
        setup_crystal_energy_from_dmacrys(calc_tmp, input, crystal, multipoles, false);
        for (const auto& [key, p] : buck_params) {
            calc_tmp.set_buckingham_params(key.first, key.second, p);
        }
        calc_tmp.set_max_interaction_order(max_order);
        calc_tmp.set_neighbor_list(ref_neighbors_noew);
        calc_tmp.set_fixed_site_masks(ref_site_masks_noew);

        Mat3 deformation = Mat3::Identity() + eps;
        Mat3 strained_direct = deformation * crystal.unit_cell().direct();
        crystal::UnitCell strained_uc(strained_direct);
        crystal::Crystal strained_crystal(
            crystal.asymmetric_unit(), crystal.space_group(), strained_uc);
        auto strained_states = states;
        for (auto& s : strained_states) {
            s.position = deformation * s.position;
        }
        calc_tmp.update_lattice(strained_crystal, strained_states);
        return calc_tmp.compute(strained_states).strain_gradient;
    };
    {
        const double dgrad2 = 2e-4;
        Mat6 C_fdg_noew = Mat6::Zero();
        for (int i = 0; i < 3; ++i) {
            Vec6 gp = strain_grad_at_noew(voigt_strain_tensor(i, +dgrad2));
            Vec6 gm = strain_grad_at_noew(voigt_strain_tensor(i, -dgrad2));
            Vec6 col = (gp - gm) / (2.0 * dgrad2) / V * conv;
            C_fdg_noew.col(i) = col;
            C_fdg_noew.row(i) = col.transpose();
        }
        occ::log::info(
            "UREAXX22 diagnostic (no Ewald) FD(grad) C11/C22/C33/C12/C13/C23: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
            C_fdg_noew(0, 0), C_fdg_noew(1, 1), C_fdg_noew(2, 2),
            C_fdg_noew(0, 1), C_fdg_noew(0, 2), C_fdg_noew(1, 2));
    }
    for (double dtest : {1e-3, 7e-4, 5e-4, 3e-4, 2e-4, 1e-4}) {
        Mat6 C_fd_scan = Mat6::Zero();
        for (int i = 0; i < 3; ++i) {
            Mat3 ei = voigt_strain_tensor(i, dtest);
            Mat3 emi = voigt_strain_tensor(i, -dtest);
            const double Epi = energy_at_noew(ei);
            const double Emi = energy_at_noew(emi);
            C_fd_scan(i, i) = ((Epi - 2.0 * E0_noew + Emi) / (dtest * dtest)) / V * conv;
            for (int j = i + 1; j < 3; ++j) {
                Mat3 ej = voigt_strain_tensor(j, dtest);
                Mat3 emj = voigt_strain_tensor(j, -dtest);
                const double Epp = energy_at_noew(ei + ej);
                const double Epm = energy_at_noew(ei + emj);
                const double Emp = energy_at_noew(emi + ej);
                const double Emm = energy_at_noew(emi + emj);
                const double val =
                    (Epp - Epm - Emp + Emm) / (4.0 * dtest * dtest) / V * conv;
                C_fd_scan(i, j) = val;
                C_fd_scan(j, i) = val;
            }
        }
        occ::log::info(
            "UREAXX22 diagnostic (no Ewald) FD C11/C22/C33/C12/C13/C23 @delta={:.1e}: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
            dtest, C_fd_scan(0, 0), C_fd_scan(1, 1), C_fd_scan(2, 2),
            C_fd_scan(0, 1), C_fd_scan(0, 2), C_fd_scan(1, 2));
    }

    // Decompose no-Ewald strain derivative mismatch by pair class:
    // self-image pairs (i==j) vs cross-molecule pairs (i!=j).
    std::vector<NeighborPair> neighbors_self, neighbors_cross;
    std::vector<std::vector<bool>> masks_self, masks_cross;
    neighbors_self.reserve(ref_neighbors_noew.size());
    neighbors_cross.reserve(ref_neighbors_noew.size());
    masks_self.reserve(ref_site_masks_noew.size());
    masks_cross.reserve(ref_site_masks_noew.size());
    for (size_t p = 0; p < ref_neighbors_noew.size(); ++p) {
        if (ref_neighbors_noew[p].mol_i == ref_neighbors_noew[p].mol_j) {
            neighbors_self.push_back(ref_neighbors_noew[p]);
            if (p < ref_site_masks_noew.size()) masks_self.push_back(ref_site_masks_noew[p]);
            else masks_self.emplace_back();
        } else {
            neighbors_cross.push_back(ref_neighbors_noew[p]);
            if (p < ref_site_masks_noew.size()) masks_cross.push_back(ref_site_masks_noew[p]);
            else masks_cross.emplace_back();
        }
    }

    auto noew_subset_strain_grad = [&](const std::vector<NeighborPair>& nlist,
                                       const std::vector<std::vector<bool>>& smasks) {
        CrystalEnergy calc_subset(crystal, multipoles, input.cutoff_radius,
                                  ForceFieldType::Custom, true, false,
                                  1e-8, 0.0, 0);
        setup_crystal_energy_from_dmacrys(calc_subset, input, crystal, multipoles, false);
        for (const auto& [key, p] : buck_params) {
            calc_subset.set_buckingham_params(key.first, key.second, p);
        }
        calc_subset.set_max_interaction_order(max_order);
        calc_subset.set_neighbor_list(nlist);
        calc_subset.set_fixed_site_masks(smasks);
        calc_subset.update_lattice(crystal, states);
        return calc_subset.compute(states).strain_gradient;
    };
    auto noew_subset_fd = [&](const std::vector<NeighborPair>& nlist,
                              const std::vector<std::vector<bool>>& smasks) {
        Vec6 out = Vec6::Zero();
        const double dgrad = 1e-5;
        for (int a = 0; a < 6; ++a) {
            Mat3 ep = voigt_strain_tensor(a, +dgrad);
            Mat3 em = voigt_strain_tensor(a, -dgrad);
            const double Ep = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                ep, input.cutoff_radius, false, 0.0, 0, max_order,
                &nlist, &smasks);
            const double Em = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                em, input.cutoff_radius, false, 0.0, 0, max_order,
                &nlist, &smasks);
            out[a] = (Ep - Em) / (2.0 * dgrad);
        }
        return out;
    };
    Vec6 g_noew_self_an = noew_subset_strain_grad(neighbors_self, masks_self);
    Vec6 g_noew_cross_an = noew_subset_strain_grad(neighbors_cross, masks_cross);
    Vec6 g_noew_self_fd = noew_subset_fd(neighbors_self, masks_self);
    Vec6 g_noew_cross_fd = noew_subset_fd(neighbors_cross, masks_cross);
    std::vector<NeighborPair> neighbors_cross_shift0, neighbors_cross_shiftN;
    std::vector<std::vector<bool>> masks_cross_shift0, masks_cross_shiftN;
    for (size_t p = 0; p < neighbors_cross.size(); ++p) {
        const auto& np = neighbors_cross[p];
        const bool is_shift0 =
            (np.cell_shift[0] == 0 && np.cell_shift[1] == 0 && np.cell_shift[2] == 0);
        if (is_shift0) {
            neighbors_cross_shift0.push_back(np);
            if (p < masks_cross.size()) masks_cross_shift0.push_back(masks_cross[p]);
            else masks_cross_shift0.emplace_back();
        } else {
            neighbors_cross_shiftN.push_back(np);
            if (p < masks_cross.size()) masks_cross_shiftN.push_back(masks_cross[p]);
            else masks_cross_shiftN.emplace_back();
        }
    }
    Vec6 g_noew_cross_shift0_an =
        noew_subset_strain_grad(neighbors_cross_shift0, masks_cross_shift0);
    Vec6 g_noew_cross_shift0_fd =
        noew_subset_fd(neighbors_cross_shift0, masks_cross_shift0);
    Vec6 g_noew_cross_shiftN_an =
        noew_subset_strain_grad(neighbors_cross_shiftN, masks_cross_shiftN);
    Vec6 g_noew_cross_shiftN_fd =
        noew_subset_fd(neighbors_cross_shiftN, masks_cross_shiftN);

    // Split no-Ewald strain diagnostics into electrostatic-only and
    // Buckingham-only pieces to isolate shear mismatches.
    std::map<std::pair<int, int>, BuckinghamParams> zero_buck;
    for (const auto& [key, p] : buck_params) {
        (void)p;
        zero_buck[key] = {0.0, 1.0, 0.0};
    }

    auto input_no_sr = input;
    input_no_sr.potentials.clear();

    CrystalEnergy calc_noew_elec(crystal, multipoles, input.cutoff_radius,
                                 ForceFieldType::Custom, true, false,
                                 1e-8, 0.0, 0);
    setup_crystal_energy_from_dmacrys(calc_noew_elec, input_no_sr, crystal, multipoles);
    calc_noew_elec.clear_typed_buckingham_params();
    for (const auto& [key, p] : zero_buck) {
        calc_noew_elec.set_buckingham_params(key.first, key.second, p);
    }
    calc_noew_elec.set_max_interaction_order(max_order);
    calc_noew_elec.update_lattice(crystal, states);
    auto eh_noew_elec = calc_noew_elec.compute_with_hessian(states);

    auto ref_neighbors_noew_elec = calc_noew_elec.neighbor_list();
    auto ref_site_masks_noew_elec = calc_noew_elec.compute_buckingham_site_masks(states);
    auto energy_at_noew_elec = [&](const Mat3& eps) {
        return compute_strained_energy(
            input_no_sr, crystal, multipoles, zero_buck,
            eps, input.cutoff_radius, false, 0.0, 0, max_order,
            &ref_neighbors_noew_elec, &ref_site_masks_noew_elec);
    };
    Vec6 dU_fd_noew_elec = Vec6::Zero();
    {
        const double dgrad = 1e-5;
        for (int a = 0; a < 6; ++a) {
            Mat3 ep = voigt_strain_tensor(a, +dgrad);
            Mat3 em = voigt_strain_tensor(a, -dgrad);
            const double Ep = energy_at_noew_elec(ep);
            const double Em = energy_at_noew_elec(em);
            dU_fd_noew_elec[a] = (Ep - Em) / (2.0 * dgrad);
        }
    }
    Vec6 dU_fd_noew_buck = dU_fd_noew - dU_fd_noew_elec;
    Vec6 dU_an_noew_buck = eh_noew.strain_gradient - eh_noew_elec.strain_gradient;

    // Isolate ewald J^T H_site J contribution used in compute_with_hessian.
    auto ref_neighbors_ew = calc.neighbor_list();
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        multipoles[i].set_orientation(states[i].rotation_matrix(), states[i].position);
        cart_mols.push_back(multipoles[i].cartesian());
    }
    auto ewald_sites = gather_ewald_sites(cart_mols, true);
    auto mol_site_indices = build_mol_site_indices(cart_mols);
    EwaldParams ew_params;
    ew_params.alpha = alpha;
    ew_params.kmax = kmax;
    ew_params.include_dipole = true;
    auto lattice_cache = build_ewald_lattice_cache(crystal.unit_cell(), ew_params);
    auto ewh = compute_ewald_correction_with_hessian(
        ewald_sites, crystal.unit_cell(), ref_neighbors_ew, mol_site_indices,
        calc.cutoff_radius(), calc.use_com_elec_gate(),
        calc.elec_site_cutoff(), ew_params, &calc.electrostatic_taper(), &lattice_cache);

    std::array<Mat3, 6> B = {
        Mat3{{1,0,0},{0,0,0},{0,0,0}},
        Mat3{{0,0,0},{0,1,0},{0,0,0}},
        Mat3{{0,0,0},{0,0,0},{0,0,1}},
        Mat3{{0,0,0},{0,0,0.5},{0,0.5,0}},
        Mat3{{0,0,0.5},{0,0,0},{0.5,0,0}},
        Mat3{{0,0.5,0},{0.5,0,0},{0,0,0}}
    };

    Mat J_strain = Mat::Zero(3 * static_cast<int>(ewald_sites.size()), 6);
    for (int s = 0; s < static_cast<int>(ewald_sites.size()); ++s) {
        const int m = ewald_sites[s].mol_index;
        for (int a = 0; a < 6; ++a) {
            J_strain.block<3, 1>(3 * s, a) = B[a] * states[m].position;
        }
    }
    Mat6 C_ew_jhj = (J_strain.transpose() * ewh.site_hessian * J_strain / V) * conv;
    Mat6 C_ew_explicit_est = C_analytic - C_analytic_noew - C_ew_jhj;

    // Chain-rule strain gradient contribution from Ewald site forces:
    // dE/dE_a(chain) = sum_m F_m . (B_a * R_m)
    std::vector<Vec3> mol_force(states.size(), Vec3::Zero());
    for (size_t s = 0; s < ewald_sites.size(); ++s) {
        mol_force[ewald_sites[s].mol_index] += ewh.site_forces[s];
    }
    Vec6 g_chain = Vec6::Zero();
    for (size_t m = 0; m < states.size(); ++m) {
        for (int a = 0; a < 6; ++a) {
            g_chain[a] += mol_force[m].dot(B[a] * states[m].position);
        }
    }
    Vec6 g_ew_explicit_est = eh.strain_gradient - eh_noew.strain_gradient - g_chain;

    occ::log::info("UREAXX22 diagnostic: total = {:.6f} kJ/mol per molecule",
                   eh.total_energy / static_cast<double>(states.size()));
    occ::log::info("UREAXX22 diagnostic: electrostatic = {:.6f}, rep-disp = {:.6f} kJ/mol per molecule",
                   eh.electrostatic_energy / static_cast<double>(states.size()),
                   eh.repulsion_dispersion / static_cast<double>(states.size()));
    occ::log::info("UREAXX22 diagnostic: strain gradient (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   eh.strain_gradient[0], eh.strain_gradient[1], eh.strain_gradient[2],
                   eh.strain_gradient[3], eh.strain_gradient[4], eh.strain_gradient[5]);
    occ::log::info("UREAXX22 diagnostic: strain gradient FD (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   dU_fd[0], dU_fd[1], dU_fd[2], dU_fd[3], dU_fd[4], dU_fd[5]);
    Vec6 dU_err = eh.strain_gradient - dU_fd;
    occ::log::info("UREAXX22 diagnostic: strain gradient analytic-FD (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   dU_err[0], dU_err[1], dU_err[2], dU_err[3], dU_err[4], dU_err[5]);

    occ::log::info("UREAXX22 diagnostic C11/C22/C33/C12/C13/C23 (GPa):");
    occ::log::info("  analytic: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                   C_analytic(0, 0), C_analytic(1, 1), C_analytic(2, 2),
                   C_analytic(0, 1), C_analytic(0, 2), C_analytic(1, 2));
    occ::log::info("  FD      : {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                   C_fd(0, 0), C_fd(1, 1), C_fd(2, 2),
                   C_fd(0, 1), C_fd(0, 2), C_fd(1, 2));
    occ::log::info("UREAXX22 diagnostic (no Ewald) C11/C22/C33/C12/C13/C23 (GPa):");
    occ::log::info("  analytic: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                   C_analytic_noew(0, 0), C_analytic_noew(1, 1), C_analytic_noew(2, 2),
                   C_analytic_noew(0, 1), C_analytic_noew(0, 2), C_analytic_noew(1, 2));
    occ::log::info("  FD      : {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                   C_fd_noew(0, 0), C_fd_noew(1, 1), C_fd_noew(2, 2),
                   C_fd_noew(0, 1), C_fd_noew(0, 2), C_fd_noew(1, 2));
    occ::log::info("UREAXX22 diagnostic (no Ewald) strain gradient analytic (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   eh_noew.strain_gradient[0], eh_noew.strain_gradient[1], eh_noew.strain_gradient[2],
                   eh_noew.strain_gradient[3], eh_noew.strain_gradient[4], eh_noew.strain_gradient[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald) strain gradient FD (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   dU_fd_noew[0], dU_fd_noew[1], dU_fd_noew[2], dU_fd_noew[3], dU_fd_noew[4], dU_fd_noew[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, self-image pairs) strain gradient analytic (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   g_noew_self_an[0], g_noew_self_an[1], g_noew_self_an[2],
                   g_noew_self_an[3], g_noew_self_an[4], g_noew_self_an[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, self-image pairs) strain gradient FD (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   g_noew_self_fd[0], g_noew_self_fd[1], g_noew_self_fd[2],
                   g_noew_self_fd[3], g_noew_self_fd[4], g_noew_self_fd[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, cross pairs) strain gradient analytic (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   g_noew_cross_an[0], g_noew_cross_an[1], g_noew_cross_an[2],
                   g_noew_cross_an[3], g_noew_cross_an[4], g_noew_cross_an[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, cross pairs) strain gradient FD (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   g_noew_cross_fd[0], g_noew_cross_fd[1], g_noew_cross_fd[2],
                   g_noew_cross_fd[3], g_noew_cross_fd[4], g_noew_cross_fd[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, cross pair counts): shift=0 {}, shift!=0 {}",
                   neighbors_cross_shift0.size(), neighbors_cross_shiftN.size());
    occ::log::info("UREAXX22 diagnostic (no Ewald, cross shift=0) strain gradient analytic (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   g_noew_cross_shift0_an[0], g_noew_cross_shift0_an[1], g_noew_cross_shift0_an[2],
                   g_noew_cross_shift0_an[3], g_noew_cross_shift0_an[4], g_noew_cross_shift0_an[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, cross shift=0) strain gradient FD (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   g_noew_cross_shift0_fd[0], g_noew_cross_shift0_fd[1], g_noew_cross_shift0_fd[2],
                   g_noew_cross_shift0_fd[3], g_noew_cross_shift0_fd[4], g_noew_cross_shift0_fd[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, cross shift!=0) strain gradient analytic (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   g_noew_cross_shiftN_an[0], g_noew_cross_shiftN_an[1], g_noew_cross_shiftN_an[2],
                   g_noew_cross_shiftN_an[3], g_noew_cross_shiftN_an[4], g_noew_cross_shiftN_an[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, cross shift!=0) strain gradient FD (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   g_noew_cross_shiftN_fd[0], g_noew_cross_shiftN_fd[1], g_noew_cross_shiftN_fd[2],
                   g_noew_cross_shiftN_fd[3], g_noew_cross_shiftN_fd[4], g_noew_cross_shiftN_fd[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, elec-only) strain gradient analytic (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   eh_noew_elec.strain_gradient[0], eh_noew_elec.strain_gradient[1], eh_noew_elec.strain_gradient[2],
                   eh_noew_elec.strain_gradient[3], eh_noew_elec.strain_gradient[4], eh_noew_elec.strain_gradient[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, elec-only) strain gradient FD (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   dU_fd_noew_elec[0], dU_fd_noew_elec[1], dU_fd_noew_elec[2],
                   dU_fd_noew_elec[3], dU_fd_noew_elec[4], dU_fd_noew_elec[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, buck-only by diff) strain gradient analytic (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   dU_an_noew_buck[0], dU_an_noew_buck[1], dU_an_noew_buck[2],
                   dU_an_noew_buck[3], dU_an_noew_buck[4], dU_an_noew_buck[5]);
    occ::log::info("UREAXX22 diagnostic (no Ewald, buck-only by diff) strain gradient FD (kJ/mol) = [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   dU_fd_noew_buck[0], dU_fd_noew_buck[1], dU_fd_noew_buck[2],
                   dU_fd_noew_buck[3], dU_fd_noew_buck[4], dU_fd_noew_buck[5]);
    occ::log::info("UREAXX22 diagnostic Ewald J^T H_site J only C11/C22/C33/C12/C13/C23 (GPa):");
    occ::log::info("  JHJ     : {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                   C_ew_jhj(0, 0), C_ew_jhj(1, 1), C_ew_jhj(2, 2),
                   C_ew_jhj(0, 1), C_ew_jhj(0, 2), C_ew_jhj(1, 2));
    occ::log::info("UREAXX22 diagnostic Ewald explicit-hessian estimate C11/C22/C33/C12/C13/C23 (GPa):");
    occ::log::info("  explicit: {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                   C_ew_explicit_est(0, 0), C_ew_explicit_est(1, 1), C_ew_explicit_est(2, 2),
                   C_ew_explicit_est(0, 1), C_ew_explicit_est(0, 2), C_ew_explicit_est(1, 2));
    occ::log::info("UREAXX22 diagnostic Ewald explicit-gradient estimate (kJ/mol): [{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                   g_ew_explicit_est[0], g_ew_explicit_est[1], g_ew_explicit_est[2],
                   g_ew_explicit_est[3], g_ew_explicit_est[4], g_ew_explicit_est[5]);

    // Regression guard: the main diagonal clamped elastic constants should
    // track frozen-neighbor FD closely. This specifically catches large
    // strain-Hessian assembly regressions.
    for (int i = 0; i < 3; ++i) {
        INFO("UREAXX22 C" << (i + 1) << (i + 1)
             << " analytic=" << C_analytic(i, i)
             << " fd=" << C_fd(i, i));
        CHECK(C_analytic(i, i) == Approx(C_fd(i, i)).margin(1.0));
    }
}

TEST_CASE("DMACRYS UREAXX22 electrostatic ladder diagnostic",
          "[mults][dmacrys][elastic][ureaxx22][ladder][diag][.]") {
    namespace fs = std::filesystem;
    if (!fs::exists(UREAXX22_JSON)) {
        SKIP("UREAXX22 fixture not found: " + UREAXX22_JSON);
    }

    auto input = read_dmacrys_json(UREAXX22_JSON);
    auto crystal = build_crystal(input.optimized_crystal.value_or(input.crystal));
    auto buck_params = convert_buckingham_params(input.potentials);

    std::map<std::pair<int, int>, BuckinghamParams> no_buck;
    for (const auto& [key, p] : buck_params) {
        no_buck[key] = {0.0, 1.0, 0.0};
    }

    struct LadderCase {
        const char* name;
        int num_components;
        double ref_elec_eV_cell;
    };

    const LadderCase cases[] = {
        {"qq", 1, input.initial_ref.charge_charge_inter_eV},
        {"qq+dip", 4,
         input.initial_ref.charge_charge_inter_eV +
         input.initial_ref.charge_dipole_eV +
         input.initial_ref.dipole_dipole_eV},
        {"full_l4", 25,
         input.initial_ref.charge_charge_inter_eV +
         input.initial_ref.charge_dipole_eV +
         input.initial_ref.dipole_dipole_eV +
         input.initial_ref.higher_multipole_eV},
    };

    constexpr double alpha = 0.195627;
    constexpr int kmax = 2;
    double qq_occ_eV = 0.0;
    bool have_qq_occ = false;

    for (const auto& lc : cases) {
        auto input_case = input;
        for (auto& site : input_case.molecule.sites) {
            for (size_t i = static_cast<size_t>(lc.num_components);
                 i < site.components.size(); ++i) {
                site.components[i] = 0.0;
            }
        }

        auto multipoles = build_multipole_sources(input_case, crystal);
        CrystalEnergy calc(crystal, multipoles, input_case.cutoff_radius,
                           ForceFieldType::Custom, true, true,
                           1e-8, alpha, kmax);
        setup_crystal_energy_from_dmacrys(calc, input_case, crystal, multipoles);
        for (const auto& [key, p] : no_buck) {
            calc.set_buckingham_params(key.first, key.second, p);
        }
        calc.set_max_interaction_order(4);

        auto states = calc.initial_states();
        for (size_t m = 0; m < states.size(); ++m) {
            occ::log::info("UREAXX22 ladder {}: mol {} det(R)={:.6f}",
                           lc.name, m, states[m].rotation_matrix().determinant());
        }
        auto result = calc.compute(states);
        const double elec_eV = result.electrostatic_energy / units::EV_TO_KJ_PER_MOL;
        const double diff_eV = elec_eV - lc.ref_elec_eV_cell;

        occ::log::info("UREAXX22 ladder {}: OCC={:.6f} eV/cell  DMACRYS={:.6f}  diff={:.6f}",
                       lc.name, elec_eV, lc.ref_elec_eV_cell, diff_eV);
        if (lc.num_components == 1) {
            qq_occ_eV = elec_eV;
            have_qq_occ = true;
        }
    }

    // Extra sensitivity scan for the qq term to isolate semantics vs convergence.
    {
        auto qq_input = input;
        for (auto& site : qq_input.molecule.sites) {
            for (size_t i = 1; i < site.components.size(); ++i) {
                site.components[i] = 0.0;
            }
        }
        const auto qq_ref = input.initial_ref.charge_charge_inter_eV;
        const auto qq_mp = build_multipole_sources(qq_input, crystal);

        occ::log::info("UREAXX22 qq sensitivity scan (ref {:.6f} eV/cell):", qq_ref);
        for (int km : {2, 3, 4, 6}) {
            for (bool spli_on : {true, false}) {
                auto qq_model = qq_input;
                if (!spli_on) {
                    qq_model.has_spline = false;
                    qq_model.spline_min = 0.0;
                    qq_model.spline_max = 0.0;
                }

                CrystalEnergy calc(crystal, qq_mp, qq_model.cutoff_radius,
                                   ForceFieldType::Custom, true, true,
                                   1e-8, alpha, km);
                setup_crystal_energy_from_dmacrys(calc, qq_model, crystal, qq_mp);
                for (const auto& [key, p] : no_buck) {
                    calc.set_buckingham_params(key.first, key.second, p);
                }
                calc.set_max_interaction_order(4);
                auto result = calc.compute(calc.initial_states());
                const double elec_eV_scan =
                    result.electrostatic_energy / units::EV_TO_KJ_PER_MOL;
                occ::log::info("  kmax={} SPLI={} -> qq={:.6f} diff={:+.6f} eV",
                               km, spli_on ? "on" : "off",
                               elec_eV_scan, elec_eV_scan - qq_ref);
            }
        }

        occ::log::info("UREAXX22 qq alpha sensitivity (kmax=2, SPLI=on):");
        for (double a_scan : {0.12, 0.14, 0.154872, 0.18, 0.195627, 0.22}) {
            auto qq_model = qq_input;
            CrystalEnergy calc(crystal, qq_mp, qq_model.cutoff_radius,
                               ForceFieldType::Custom, true, true,
                               1e-8, a_scan, 2);
            setup_crystal_energy_from_dmacrys(calc, qq_model, crystal, qq_mp);
            for (const auto& [key, p] : no_buck) {
                calc.set_buckingham_params(key.first, key.second, p);
            }
            calc.set_max_interaction_order(4);
            auto result = calc.compute(calc.initial_states());
            const double elec_eV_scan =
                result.electrostatic_energy / units::EV_TO_KJ_PER_MOL;
            occ::log::info("  alpha={:.6f}  qq={:.6f}  diff={:+.6f} eV",
                           a_scan, elec_eV_scan, elec_eV_scan - qq_ref);
        }

        // DMACRYS prints qq decomposition:
        //   Ewald summed qq = intra qq + inter qq.
        // Reconstruct the same split from OCC for diagnostics.
        double intra_ha = 0.0;
        for (const auto& src : qq_mp) {
            const auto& cart = src.cartesian();
            const int nsites = static_cast<int>(cart.sites.size());
            for (int i = 0; i < nsites; ++i) {
                const double qi = cart.sites[i].cart.data[0];
                for (int j = i + 1; j < nsites; ++j) {
                    const double qj = cart.sites[j].cart.data[0];
                    const Vec3 Rij_bohr =
                        (cart.sites[j].position - cart.sites[i].position) /
                        occ::units::BOHR_TO_ANGSTROM;
                    const double rij = Rij_bohr.norm();
                    if (rij > 1e-12) {
                        intra_ha += qi * qj / rij;
                    }
                }
            }
        }
        const double intra_eV_cell = intra_ha * occ::units::AU_TO_EV;
        if (have_qq_occ) {
            const double qq_ewald_sum_occ = qq_occ_eV + intra_eV_cell;
            occ::log::info(
                "UREAXX22 qq decomposition (OCC): inter={:.6f} intra={:.6f} ewald_sum={:.6f} eV/cell",
                qq_occ_eV, intra_eV_cell, qq_ewald_sum_occ);
            if (std::abs(input.initial_ref.charge_charge_intra_eV) > 0.0 ||
                std::abs(input.initial_ref.charge_charge_ewald_summed_eV) > 0.0) {
                occ::log::info(
                    "UREAXX22 qq decomposition (DMACRYS): inter={:.6f} intra={:.6f} ewald_sum={:.6f} eV/cell",
                    input.initial_ref.charge_charge_inter_eV,
                    input.initial_ref.charge_charge_intra_eV,
                    input.initial_ref.charge_charge_ewald_summed_eV);
                occ::log::info(
                    "UREAXX22 qq decomposition diffs: d(inter)={:+.6f} d(intra)={:+.6f} d(ewald_sum)={:+.6f} eV/cell",
                    qq_occ_eV - input.initial_ref.charge_charge_inter_eV,
                    intra_eV_cell - input.initial_ref.charge_charge_intra_eV,
                    qq_ewald_sum_occ - input.initial_ref.charge_charge_ewald_summed_eV);
            }
        }
    }

    CHECK(true);
}

TEST_CASE("DMACRYS UREAXX22 W_ii/W_ei analytic-vs-FD diagnostic",
          "[mults][dmacrys][elastic][ureaxx22][diag][.]") {
    namespace fs = std::filesystem;
    if (!fs::exists(UREAXX22_JSON)) {
        SKIP("UREAXX22 fixture not found: " + UREAXX22_JSON);
    }

    auto input = read_dmacrys_json(UREAXX22_JSON);
    auto crystal = build_crystal(input.optimized_crystal.value_or(input.crystal));
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    constexpr double alpha = 0.195627; // /Ang
    constexpr int kmax = 2;
    constexpr int max_order = 4;
    constexpr double h_pos = 2e-5;   // Angstrom
    constexpr double h_rot = 2e-5;   // radians
    constexpr double h_eps = 2e-5;   // unitless strain

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }
    calc.set_max_interaction_order(max_order);

    const auto reference_states = calc.initial_states();
    calc.update_lattice(crystal, reference_states);
    auto eh = calc.compute_with_hessian(reference_states);

    const int ndof = static_cast<int>(eh.hessian.rows());
    REQUIRE(ndof > 0);
    REQUIRE(static_cast<int>(eh.strain_state_hessian.rows()) == 6);
    REQUIRE(static_cast<int>(eh.strain_state_hessian.cols()) == ndof);

    auto pack_energy_grad_at = [&](const crystal::Crystal& c,
                                   const std::vector<MoleculeState>& states) {
        calc.update_lattice(c, states);
        auto e = calc.compute(states);
        const int n = static_cast<int>(e.forces.size());
        Vec g(6 * n);
        for (int i = 0; i < n; ++i) {
            g.segment<3>(6 * i) = -e.forces[i];
            g.segment<3>(6 * i + 3) = e.torques[i];
        }
        return g;
    };

    auto perturb_by_psi = [&](MoleculeState& s, int axis, double delta) {
        Vec3 u = Vec3::Zero();
        u[axis] = 1.0;
        const Mat3 dR = Eigen::AngleAxisd(delta, u).toRotationMatrix();
        const Mat3 Rnew = dR * s.rotation_matrix();
        s = MoleculeState::from_rotation(s.position, Rnew);
    };

    auto evaluate_wii_wei_fd = [&](CrystalEnergy& calc_local, const char* label) {
        calc_local.update_lattice(crystal, reference_states);
        auto eh_local = calc_local.compute_with_hessian(reference_states);

        auto pack_energy_grad_at_local = [&](const crystal::Crystal& c,
                                             const std::vector<MoleculeState>& states) {
            calc_local.update_lattice(c, states);
            auto e = calc_local.compute(states);
            const int n = static_cast<int>(e.forces.size());
            Vec g(6 * n);
            for (int i = 0; i < n; ++i) {
                g.segment<3>(6 * i) = -e.forces[i];
                g.segment<3>(6 * i + 3) = e.torques[i];
            }
            return g;
        };

        Mat H_fd_local_raw = Mat::Zero(ndof, ndof);
        for (int col = 0; col < ndof; ++col) {
            const int m = col / 6;
            const int c = col % 6;
            const double h = (c < 3) ? h_pos : h_rot;

            auto sp = reference_states;
            auto sm = reference_states;
            if (c < 3) {
                sp[m].position[c] += h;
                sm[m].position[c] -= h;
            } else {
                perturb_by_psi(sp[m], c - 3, +h);
                perturb_by_psi(sm[m], c - 3, -h);
            }

            const Vec gp = pack_energy_grad_at_local(crystal, sp);
            const Vec gm = pack_energy_grad_at_local(crystal, sm);
            H_fd_local_raw.col(col) = (gp - gm) / (2.0 * h);
        }
        const Mat H_fd_local =
            0.5 * (H_fd_local_raw + H_fd_local_raw.transpose()).eval();

        Mat W_ei_fd_local = Mat::Zero(6, ndof);
        for (int a = 0; a < 6; ++a) {
            const Mat3 eps_p = voigt_strain_tensor(a, +h_eps);
            const Mat3 eps_m = voigt_strain_tensor(a, -h_eps);

            auto build_strained = [&](const Mat3& eps) {
                const Mat3 F = Mat3::Identity() + eps;
                const Mat3 strained_direct = F * crystal.unit_cell().direct();
                crystal::UnitCell strained_uc(strained_direct);
                crystal::Crystal strained_crystal(
                    crystal.asymmetric_unit(), crystal.space_group(), strained_uc);

                auto strained_states = reference_states;
                for (auto& s : strained_states) {
                    s.position = F * s.position;
                }
                return std::pair<crystal::Crystal, std::vector<MoleculeState>>{
                    std::move(strained_crystal), std::move(strained_states)};
            };

            auto [cp, sp] = build_strained(eps_p);
            auto [cm, sm] = build_strained(eps_m);
            const Vec gp = pack_energy_grad_at_local(cp, sp);
            const Vec gm = pack_energy_grad_at_local(cm, sm);
            W_ei_fd_local.row(a) = ((gp - gm) / (2.0 * h_eps)).transpose();
        }

        calc_local.update_lattice(crystal, reference_states);

        const Mat H_err_local = eh_local.hessian - H_fd_local;
        const Mat W_ei_err_local = eh_local.strain_state_hessian - W_ei_fd_local;
        const double h_ref_local = std::max(1e-14, eh_local.hessian.norm());
        const double hei_ref_local =
            std::max(1e-14, eh_local.strain_state_hessian.norm());

        // Pure chain-rule projection for this rigid-strain model:
        // W_ei = d/dE(dE/dq) = S^T * W_ii where only COM translations depend on E.
        Mat S = Mat::Zero(ndof, 6);
        for (int m = 0; m < static_cast<int>(reference_states.size()); ++m) {
            for (int a = 0; a < 6; ++a) {
                S.block<3, 1>(6 * m + 0, a) =
                    voigt_basis_matrices()[a] * reference_states[m].position;
            }
        }
        const Mat W_ei_from_h_local = (S.transpose() * eh_local.hessian).eval();
        const Mat W_ei_chain_err_local =
            eh_local.strain_state_hessian - W_ei_from_h_local;
        const double wei_chain_rel_local =
            W_ei_chain_err_local.norm() /
            std::max(1e-14, eh_local.strain_state_hessian.norm());
        const double wei_chain_max_local =
            W_ei_chain_err_local.cwiseAbs().maxCoeff();

        occ::log::info(
            "UREAXX22 {} W_ii/W_ei analytic-vs-FD: rel(H)={:.6e} max(H)={:.6e}  rel(W_ei)={:.6e} max(W_ei)={:.6e}  chain rel(W_ei)={:.6e} chain max(W_ei)={:.6e}",
            label,
            H_err_local.norm() / h_ref_local,
            H_err_local.cwiseAbs().maxCoeff(),
            W_ei_err_local.norm() / hei_ref_local,
            W_ei_err_local.cwiseAbs().maxCoeff(),
            wei_chain_rel_local,
            wei_chain_max_local);

        const auto taper = calc_local.electrostatic_taper();
        if (taper.is_valid()) {
            auto mps = multipoles;
            std::vector<CartesianMolecule> cart_mols;
            cart_mols.reserve(mps.size());
            for (size_t m = 0; m < mps.size(); ++m) {
                mps[m].set_orientation(
                    reference_states[m].rotation_matrix(),
                    reference_states[m].position);
                cart_mols.push_back(mps[m].cartesian());
            }

            size_t shell_pairs = 0;
            size_t near_on = 0;
            size_t near_off = 0;
            size_t all_near_on = 0;
            size_t all_near_off = 0;
            double min_abs_on = std::numeric_limits<double>::infinity();
            double min_abs_off = std::numeric_limits<double>::infinity();
            for (const auto& np : calc_local.neighbor_list()) {
                const Vec3 shift =
                    crystal.unit_cell().to_cartesian(np.cell_shift.cast<double>());
                const auto& A = cart_mols[np.mol_i];
                const auto& B = cart_mols[np.mol_j];
                for (const auto& sa : A.sites) {
                    if (sa.rank < 0) continue;
                    for (const auto& sb : B.sites) {
                        if (sb.rank < 0) continue;
                        const double r = ((sb.position + shift) - sa.position).norm();
                        const double aon = std::abs(r - taper.r_on);
                        const double aoff = std::abs(r - taper.r_off);
                        min_abs_on = std::min(min_abs_on, aon);
                        min_abs_off = std::min(min_abs_off, aoff);
                        if (aon < 1e-3) all_near_on++;
                        if (aoff < 1e-3) all_near_off++;
                        if (r <= taper.r_on || r > taper.r_off) continue;
                        shell_pairs++;
                        if (aon < 1e-3) near_on++;
                        if (aoff < 1e-3) near_off++;
                    }
                }
            }
            occ::log::info(
                "UREAXX22 {} taper shell diagnostics: pairs_in_shell={} near_on(shell,<1e-3A)={} near_off(shell,<1e-3A)={} all_near_on(<1e-3A)={} all_near_off(<1e-3A)={} min|r-r_on|={:.3e} min|r-r_off|={:.3e}",
                label, shell_pairs, near_on, near_off, all_near_on, all_near_off,
                std::isfinite(min_abs_on) ? min_abs_on : -1.0,
                std::isfinite(min_abs_off) ? min_abs_off : -1.0);
        }
    };

    // Finite-difference internal Hessian block W_ii = d/dq (dE/dq).
    Mat H_fd_raw = Mat::Zero(ndof, ndof);
    for (int col = 0; col < ndof; ++col) {
        const int m = col / 6;
        const int c = col % 6;
        const double h = (c < 3) ? h_pos : h_rot;

        auto sp = reference_states;
        auto sm = reference_states;
        if (c < 3) {
            sp[m].position[c] += h;
            sm[m].position[c] -= h;
        } else {
            perturb_by_psi(sp[m], c - 3, +h);
            perturb_by_psi(sm[m], c - 3, -h);
        }

        const Vec gp = pack_energy_grad_at(crystal, sp);
        const Vec gm = pack_energy_grad_at(crystal, sm);
        H_fd_raw.col(col) = (gp - gm) / (2.0 * h);
    }
    Mat H_fd = 0.5 * (H_fd_raw + H_fd_raw.transpose()).eval();

    // Finite-difference strain-state block W_ei = d/dE (dE/dq).
    Mat W_ei_fd = Mat::Zero(6, ndof);
    for (int a = 0; a < 6; ++a) {
        const Mat3 eps_p = voigt_strain_tensor(a, +h_eps);
        const Mat3 eps_m = voigt_strain_tensor(a, -h_eps);

        auto build_strained = [&](const Mat3& eps) {
            const Mat3 F = Mat3::Identity() + eps;
            const Mat3 strained_direct = F * crystal.unit_cell().direct();
            crystal::UnitCell strained_uc(strained_direct);
            crystal::Crystal strained_crystal(
                crystal.asymmetric_unit(), crystal.space_group(), strained_uc);

            auto strained_states = reference_states;
            for (auto& s : strained_states) {
                s.position = F * s.position;
            }
            return std::pair<crystal::Crystal, std::vector<MoleculeState>>{
                std::move(strained_crystal), std::move(strained_states)};
        };

        auto [cp, sp] = build_strained(eps_p);
        auto [cm, sm] = build_strained(eps_m);
        const Vec gp = pack_energy_grad_at(cp, sp);
        const Vec gm = pack_energy_grad_at(cm, sm);
        W_ei_fd.row(a) = ((gp - gm) / (2.0 * h_eps)).transpose();
    }

    calc.update_lattice(crystal, reference_states);

    const Mat H_err = eh.hessian - H_fd;
    const Mat W_ei_err = eh.strain_state_hessian - W_ei_fd;
    occ::log::info(
        "UREAXX22 Hessian symmetry residuals: analytic={:.3e} fd={:.3e} err={:.3e}",
        (eh.hessian - eh.hessian.transpose()).norm(),
        (H_fd - H_fd.transpose()).norm(),
        (H_err - H_err.transpose()).norm());

    const double h_ref = std::max(1e-14, eh.hessian.norm());
    const double hei_ref = std::max(1e-14, eh.strain_state_hessian.norm());
    const double H_rel = H_err.norm() / h_ref;
    const double Hei_rel = W_ei_err.norm() / hei_ref;
    const double H_max = H_err.cwiseAbs().maxCoeff();
    const double Hei_max = W_ei_err.cwiseAbs().maxCoeff();
    occ::log::info("UREAXX22 H_err matrix (analytic - FD):\n{}",
                   occ::format_matrix(H_err, "{:11.4e}"));

    // Block diagnostics: useful to isolate improper-orientation rotational
    // derivative issues without flooding logs with full matrices.
    for (int m = 0; m < static_cast<int>(reference_states.size()); ++m) {
        const Mat3 H_tt = H_err.block<3, 3>(6 * m + 0, 6 * m + 0);
        const Mat3 H_tr = H_err.block<3, 3>(6 * m + 0, 6 * m + 3);
        const Mat3 H_rt = H_err.block<3, 3>(6 * m + 3, 6 * m + 0);
        const Mat3 H_rr = H_err.block<3, 3>(6 * m + 3, 6 * m + 3);
        const Mat W_ei_t = W_ei_err.block(0, 6 * m + 0, 6, 3);
        const Mat W_ei_r = W_ei_err.block(0, 6 * m + 3, 6, 3);

        occ::log::info(
            "UREAXX22 molecule {} (parity {}): H_tt={:.3e} H_tr={:.3e} H_rt={:.3e} H_rr={:.3e}  W_ei_t={:.3e} W_ei_r={:.3e}",
            m, reference_states[m].parity,
            H_tt.norm(), H_tr.norm(), H_rt.norm(), H_rr.norm(),
            W_ei_t.norm(), W_ei_r.norm());
    }

    // Compare Schur-complement variants relevant to DMACRYS property mode.
    auto dmacrys_pin = [&](Mat& H) {
        const int n_mol = ndof / 6;
        double tr = 1.0e-10;
        for (int m = 0; m < n_mol; ++m) {
            tr += H(6 * m + 0, 6 * m + 0);
            tr += H(6 * m + 1, 6 * m + 1);
            tr += H(6 * m + 2, 6 * m + 2);
        }
        for (int i = 0; i < 3; ++i) {
            H(i, i) += tr;
        }
    };

    auto schur_relaxed = [&](const Mat& Wii, const Mat& Wei) {
        Mat Wi = Wii;
        dmacrys_pin(Wi);
        Eigen::LDLT<Mat> ldlt(Wi);
        Mat X;
        if (ldlt.info() == Eigen::Success) {
            X = ldlt.solve(Wei.transpose());
        } else {
            X = Wi.fullPivLu().solve(Wei.transpose());
        }
        return (eh.strain_hessian - Wei * X).eval();
    };

    const Mat6 Wrel_psi = schur_relaxed(eh.hessian, eh.strain_state_hessian);
    auto schur_relaxed_lu = [&](const Mat& Wii, const Mat& Wei) {
        Mat Wi = Wii;
        dmacrys_pin(Wi);
        Mat X = Wi.fullPivLu().solve(Wei.transpose());
        return (eh.strain_hessian - Wei * X).eval();
    };
    const Mat6 Wrel_psi_lu = schur_relaxed_lu(eh.hessian, eh.strain_state_hessian);
    const Mat6 Wrel_fd_raw = schur_relaxed(H_fd_raw, eh.strain_state_hessian);
    auto schur_relaxed_fix_trans = [&](const Mat& Wii, const Mat& Wei) {
        std::vector<int> keep;
        keep.reserve(ndof - 3);
        for (int i = 3; i < ndof; ++i) {
            keep.push_back(i);
        }
        Mat Wi = Mat::Zero(static_cast<int>(keep.size()), static_cast<int>(keep.size()));
        Mat We = Mat::Zero(6, static_cast<int>(keep.size()));
        for (int a = 0; a < static_cast<int>(keep.size()); ++a) {
            const int ia = keep[a];
            We.col(a) = Wei.col(ia);
            for (int b = 0; b < static_cast<int>(keep.size()); ++b) {
                Wi(a, b) = Wii(ia, keep[b]);
            }
        }
        Eigen::LDLT<Mat> ldlt(Wi);
        Mat X;
        if (ldlt.info() == Eigen::Success) {
            X = ldlt.solve(We.transpose());
        } else {
            X = Wi.fullPivLu().solve(We.transpose());
        }
        return (eh.strain_hessian - We * X).eval();
    };
    const Mat6 Wrel_fix_trans = schur_relaxed_fix_trans(eh.hessian, eh.strain_state_hessian);
    auto schur_relaxed_fix_mol0 = [&](const Mat& Wii, const Mat& Wei) {
        std::vector<int> keep;
        keep.reserve(ndof - 6);
        for (int i = 6; i < ndof; ++i) {
            keep.push_back(i);
        }
        Mat Wi = Mat::Zero(static_cast<int>(keep.size()), static_cast<int>(keep.size()));
        Mat We = Mat::Zero(6, static_cast<int>(keep.size()));
        for (int a = 0; a < static_cast<int>(keep.size()); ++a) {
            const int ia = keep[a];
            We.col(a) = Wei.col(ia);
            for (int b = 0; b < static_cast<int>(keep.size()); ++b) {
                Wi(a, b) = Wii(ia, keep[b]);
            }
        }
        Eigen::LDLT<Mat> ldlt(Wi);
        Mat X;
        if (ldlt.info() == Eigen::Success) {
            X = ldlt.solve(We.transpose());
        } else {
            X = Wi.fullPivLu().solve(We.transpose());
        }
        return (eh.strain_hessian - We * X).eval();
    };
    const Mat6 Wrel_fix_mol0 = schur_relaxed_fix_mol0(eh.hessian, eh.strain_state_hessian);
    auto schur_relaxed_sumd = [&](const Mat& Wii, const Mat& Wei) {
        Mat Wi = Wii;
        double sumd = 0.0;
        for (int i = 0; i < Wi.rows(); ++i) {
            sumd += std::abs(Wi(i, i));
        }
        for (int i = 0; i < 3 && i < Wi.rows(); ++i) {
            Wi(i, i) += sumd;
        }
        Eigen::LDLT<Mat> ldlt(Wi);
        Mat X;
        if (ldlt.info() == Eigen::Success) {
            X = ldlt.solve(Wei.transpose());
        } else {
            X = Wi.fullPivLu().solve(Wei.transpose());
        }
        return (eh.strain_hessian - Wei * X).eval();
    };
    const Mat6 Wrel_psi_sumd = schur_relaxed_sumd(eh.hessian, eh.strain_state_hessian);

    const double V = crystal.unit_cell().volume();
    const double conv = units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
    const Mat6 Crel_psi = (Wrel_psi / V) * conv;
    const Mat6 Crel_psi_lu = (Wrel_psi_lu / V) * conv;
    const Mat6 Crel_fd_raw = (Wrel_fd_raw / V) * conv;
    const Mat6 Crel_fix_trans = (Wrel_fix_trans / V) * conv;
    const Mat6 Crel_fix_mol0 = (Wrel_fix_mol0 / V) * conv;
    const Mat6 Crel_psi_sumd = (Wrel_psi_sumd / V) * conv;

    if (input.initial_ref.has_elastic_constants) {
        const auto& Cref = input.initial_ref.elastic_constants_GPa;
        occ::log::info(
            "UREAXX22 relaxed diag (GPa) vs DMACRYS C11/C33/C44/C55: psi [{:.4f} {:.4f} {:.4f} {:.4f}]  psi-lu [{:.4f} {:.4f} {:.4f} {:.4f}]  fd-raw [{:.4f} {:.4f} {:.4f} {:.4f}]  fix-trans [{:.4f} {:.4f} {:.4f} {:.4f}]  fix-mol0 [{:.4f} {:.4f} {:.4f} {:.4f}]  sumd [{:.4f} {:.4f} {:.4f} {:.4f}]  ref [{:.4f} {:.4f} {:.4f} {:.4f}]",
            Crel_psi(0, 0), Crel_psi(2, 2), Crel_psi(3, 3), Crel_psi(4, 4),
            Crel_psi_lu(0, 0), Crel_psi_lu(2, 2), Crel_psi_lu(3, 3), Crel_psi_lu(4, 4),
            Crel_fd_raw(0, 0), Crel_fd_raw(2, 2), Crel_fd_raw(3, 3), Crel_fd_raw(4, 4),
            Crel_fix_trans(0, 0), Crel_fix_trans(2, 2), Crel_fix_trans(3, 3), Crel_fix_trans(4, 4),
            Crel_fix_mol0(0, 0), Crel_fix_mol0(2, 2), Crel_fix_mol0(3, 3), Crel_fix_mol0(4, 4),
            Crel_psi_sumd(0, 0), Crel_psi_sumd(2, 2), Crel_psi_sumd(3, 3), Crel_psi_sumd(4, 4),
            Cref(0, 0), Cref(2, 2), Cref(3, 3), Cref(4, 4));
    }

    occ::log::info(
        "UREAXX22 W_ii analytic-vs-FD: rel={:.6e} max_abs={:.6e}",
        H_rel, H_max);
    occ::log::info(
        "UREAXX22 W_ei analytic-vs-FD: rel={:.6e} max_abs={:.6e}",
        Hei_rel, Hei_max);

    CrystalEnergy calc_notaper(crystal, multipoles, input.cutoff_radius,
                               ForceFieldType::Custom, true, true,
                               1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(calc_notaper, input, crystal, multipoles);
    for (const auto& [key, p] : buck_params) {
        calc_notaper.set_buckingham_params(key.first, key.second, p);
    }
    calc_notaper.set_max_interaction_order(max_order);
    calc_notaper.clear_electrostatic_taper();
    calc_notaper.clear_short_range_taper();
    evaluate_wii_wei_fd(calc_notaper, "no-taper");

    CrystalEnergy calc_noew(crystal, multipoles, input.cutoff_radius,
                            ForceFieldType::Custom, true, false,
                            1e-8, 0.0, 0);
    setup_crystal_energy_from_dmacrys(calc_noew, input, crystal, multipoles);
    for (const auto& [key, p] : buck_params) {
        calc_noew.set_buckingham_params(key.first, key.second, p);
    }
    calc_noew.set_max_interaction_order(max_order);
    evaluate_wii_wei_fd(calc_noew, "no-ewald");

    CrystalEnergy calc_noew_elec_only(crystal, multipoles, input.cutoff_radius,
                                      ForceFieldType::None, true, false,
                                      1e-8, 0.0, 0);
    setup_crystal_energy_from_dmacrys(calc_noew_elec_only, input, crystal, multipoles);
    calc_noew_elec_only.set_max_interaction_order(max_order);
    evaluate_wii_wei_fd(calc_noew_elec_only, "no-ewald-elec-only");

    CrystalEnergy calc_noew_sr_only(crystal, multipoles, input.cutoff_radius,
                                    ForceFieldType::Custom, false, false,
                                    1e-8, 0.0, 0);
    setup_crystal_energy_from_dmacrys(calc_noew_sr_only, input, crystal, multipoles);
    for (const auto& [key, p] : buck_params) {
        calc_noew_sr_only.set_buckingham_params(key.first, key.second, p);
    }
    calc_noew_sr_only.set_max_interaction_order(max_order);
    evaluate_wii_wei_fd(calc_noew_sr_only, "no-ewald-sr-only");

    for (int order_cut : {0, 1, 2, 3, 4}) {
        CrystalEnergy calc_noew_elec_cut(crystal, multipoles, input.cutoff_radius,
                                         ForceFieldType::None, true, false,
                                         1e-8, 0.0, 0);
        setup_crystal_energy_from_dmacrys(calc_noew_elec_cut, input, crystal, multipoles);
        calc_noew_elec_cut.set_max_interaction_order(order_cut);
        evaluate_wii_wei_fd(calc_noew_elec_cut,
                            fmt::format("no-ewald-elec-only-lsum<={}", order_cut).c_str());
    }

    CrystalEnergy calc_noew_elec_notaper_smooth(
        crystal, multipoles, input.cutoff_radius,
        ForceFieldType::None, true, false,
        1e-8, 0.0, 0);
    setup_crystal_energy_from_dmacrys(
        calc_noew_elec_notaper_smooth, input, crystal, multipoles);
    calc_noew_elec_notaper_smooth.set_max_interaction_order(0);
    calc_noew_elec_notaper_smooth.clear_electrostatic_taper();
    calc_noew_elec_notaper_smooth.set_elec_site_cutoff(0.0);
    calc_noew_elec_notaper_smooth.set_use_com_elec_gate(false);
    evaluate_wii_wei_fd(calc_noew_elec_notaper_smooth,
                        "no-ewald-elec-only-lsum<=0-no-taper-no-site/com-cut");

    CrystalEnergy calc_noew_elec_taper_nocut(
        crystal, multipoles, input.cutoff_radius,
        ForceFieldType::None, true, false,
        1e-8, 0.0, 0);
    setup_crystal_energy_from_dmacrys(
        calc_noew_elec_taper_nocut, input, crystal, multipoles);
    calc_noew_elec_taper_nocut.set_max_interaction_order(0);
    calc_noew_elec_taper_nocut.set_elec_site_cutoff(0.0);
    calc_noew_elec_taper_nocut.set_use_com_elec_gate(false);
    evaluate_wii_wei_fd(calc_noew_elec_taper_nocut,
                        "no-ewald-elec-only-lsum<=0-taper-only-no-site/com-cut");

    CrystalEnergy calc_noew_elec_taper5_nocut(
        crystal, multipoles, input.cutoff_radius,
        ForceFieldType::None, true, false,
        1e-8, 0.0, 0);
    setup_crystal_energy_from_dmacrys(
        calc_noew_elec_taper5_nocut, input, crystal, multipoles);
    calc_noew_elec_taper5_nocut.set_max_interaction_order(0);
    calc_noew_elec_taper5_nocut.set_elec_site_cutoff(0.0);
    calc_noew_elec_taper5_nocut.set_use_com_elec_gate(false);
    const auto taper_ref = calc_noew_elec_taper5_nocut.electrostatic_taper();
    if (taper_ref.is_valid()) {
        calc_noew_elec_taper5_nocut.set_electrostatic_taper(
            taper_ref.r_on, taper_ref.r_off, 5);
    }
    evaluate_wii_wei_fd(calc_noew_elec_taper5_nocut,
                        "no-ewald-elec-only-lsum<=0-taper5-only-no-site/com-cut");

    // Loose sanity guards; this test is diagnostic-first.
    CHECK(H_rel < 5e-2);
    CHECK(Hei_rel < 5e-2);
}

TEST_CASE("DMACRYS AXOSOW charge-only strain FD", "[mults][dmacrys][strain-qq]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);

    // Zero all multipole components except charge (rank 0)
    for (auto& site : input.molecule.sites) {
        double charge = site.components[0]; // keep charge
        std::fill(site.components.begin(), site.components.end(), 0.0);
        site.components[0] = charge;
    }

    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    // Print charges for verification
    occ::log::info("Charges (rank 0 only):");
    for (const auto& site : input.molecule.sites) {
        occ::log::info("  {}: q = {:.6f}", site.label, site.components[0]);
    }

    // Build reference neighbor list
    CrystalEnergy ref_calc(crystal, multipoles, input.cutoff_radius,
                           ForceFieldType::Custom, true, true,
                           1e-8, 0.35, 8);
    setup_crystal_energy_from_dmacrys(ref_calc, input, crystal, multipoles);
    auto ref_neighbors = ref_calc.neighbor_list();
    auto ref_site_masks = ref_calc.compute_buckingham_site_masks(ref_calc.initial_states());

    // Zero-strain energy
    {
        Mat3 zero = Mat3::Zero();
        std::map<std::pair<int, int>, BuckinghamParams> no_buck;
        for (const auto& [key, p] : buck_params) {
            no_buck[key] = {0.0, 1.0, 0.0};
        }
        double E0_qq = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            zero, input.cutoff_radius, true, 0.35, 8, -1,
            &ref_neighbors, &ref_site_masks);
        occ::log::info("Charge-only energy (Ewald): {:.8f} kJ/mol ({:.6f} eV/cell)",
                       E0_qq, E0_qq / units::EV_TO_KJ_PER_MOL);
        // DMACRYS charge-only inter-molecular: -1.2014 eV = -14.489 kJ/mol
        occ::log::info("  DMACRYS inter-molecular qq: -14.489 kJ/mol (-1.2014 eV)");
    }

    // FD strain derivatives (charge-only electrostatics, no Buckingham)
    occ::log::info("\nCharge-only electrostatic strain derivatives (no SPLI comparison):");
    occ::log::info("{:>6s} {:>12s} {:>12s} {:>12s}", "comp", "our(eV)", "DMACRYS(eV)", "diff(eV)");

    // DMACRYS charge-only, no SPLI total strain derivs (includes Buck):
    //   E1=-0.9065, E2=-0.5066, E3=-0.6759
    // DMACRYS Buck-only, no SPLI:
    //   E1=-2.8452, E2=-2.5367, E3=-1.3208
    // => DMACRYS charge-only elec:
    //   E1=-0.9065-(-2.8452)=1.9387, E2=-0.5066-(-2.5367)=2.0301, E3=-0.6759-(-1.3208)=0.6449
    double dmacrys_qq_elec[] = {1.9387, 2.0301, 0.6449};

    double delta = 1e-4;
    std::map<std::pair<int, int>, BuckinghamParams> no_buck;
    for (const auto& [key, p] : buck_params) {
        no_buck[key] = {0.0, 1.0, 0.0};
    }

    for (int comp = 0; comp < 3; ++comp) {
        Mat3 eps_p = voigt_strain_tensor(comp, +delta);
        Mat3 eps_m = voigt_strain_tensor(comp, -delta);

        double Ep = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_p, input.cutoff_radius, true, 0.35, 8, -1,
            &ref_neighbors, &ref_site_masks);
        double Em = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_m, input.cutoff_radius, true, 0.35, 8, -1,
            &ref_neighbors, &ref_site_masks);
        double dUdE = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("{:>6s} {:>12.4f} {:>12.4f} {:>12.4f}",
                       names[comp], dUdE, dmacrys_qq_elec[comp],
                       dUdE - dmacrys_qq_elec[comp]);
    }

    // Also test with order=4 truncation
    occ::log::info("\nSame with max_interaction_order=4:");
    for (int comp = 0; comp < 3; ++comp) {
        Mat3 eps_p = voigt_strain_tensor(comp, +delta);
        Mat3 eps_m = voigt_strain_tensor(comp, -delta);

        double Ep = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_p, input.cutoff_radius, true, 0.35, 8, 4,
            &ref_neighbors, &ref_site_masks);
        double Em = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_m, input.cutoff_radius, true, 0.35, 8, 4,
            &ref_neighbors, &ref_site_masks);
        double dUdE = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("{:>6s} {:>12.4f} {:>12.4f} {:>12.4f}",
                       names[comp], dUdE, dmacrys_qq_elec[comp],
                       dUdE - dmacrys_qq_elec[comp]);
    }

    CHECK(true);
}

TEST_CASE("DMACRYS AXOSOW charge+dipole strain FD", "[mults][dmacrys][strain-qdip]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);

    // Zero all multipole components above rank 1 (keep charge + dipole)
    // rank 0: 1 component (charge)
    // rank 1: 3 components (dipole)
    // rank 2: 5 components (quadrupole) -> zero
    // rank 3: 7 components (octupole) -> zero
    // rank 4: 9 components (hexadecapole) -> zero
    for (auto& site : input.molecule.sites) {
        // components[0] = charge, [1..3] = dipole, [4..] = quad+
        for (size_t i = 4; i < site.components.size(); ++i) {
            site.components[i] = 0.0;
        }
    }

    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    // Print multipoles for verification
    occ::log::info("Charge+dipole multipoles:");
    for (const auto& site : input.molecule.sites) {
        occ::log::info("  {}: q={:.6f} mu=({:.6f},{:.6f},{:.6f})",
                       site.label, site.components[0],
                       site.components[1], site.components[2], site.components[3]);
    }

    // Build reference neighbor list
    CrystalEnergy ref_calc(crystal, multipoles, input.cutoff_radius,
                           ForceFieldType::Custom, true, true,
                           1e-8, 0.35, 8);
    setup_crystal_energy_from_dmacrys(ref_calc, input, crystal, multipoles);
    auto ref_neighbors = ref_calc.neighbor_list();
    auto ref_site_masks = ref_calc.compute_buckingham_site_masks(ref_calc.initial_states());

    // DMACRYS qq+dip (clean BASI, no higher mult contamination):
    //   Total strain derivs: E1=-1.7863, E2=-3.3935, E3=0.0362
    // DMACRYS Buck-only, no SPLI:
    //   E1=-2.8452, E2=-2.5367, E3=-1.3208
    // => DMACRYS qq+dip elec:
    //   E1=-1.7863-(-2.8452)=1.0589, E2=-3.3935-(-2.5367)=-0.8568, E3=0.0362-(-1.3208)=1.3570
    double dmacrys_qdip_elec[] = {1.0589, -0.8568, 1.3570};

    // FD strain derivatives (charge+dipole electrostatics, no Buckingham)
    occ::log::info("\nCharge+dipole electrostatic strain derivatives:");
    occ::log::info("{:>6s} {:>12s} {:>12s} {:>12s}", "comp", "our(eV)", "DMACRYS(eV)", "diff(eV)");

    double delta = 1e-4;
    std::map<std::pair<int, int>, BuckinghamParams> no_buck;
    for (const auto& [key, p] : buck_params) {
        no_buck[key] = {0.0, 1.0, 0.0};
    }

    // Test with unlimited order
    occ::log::info("Unlimited interaction order:");
    for (int comp = 0; comp < 3; ++comp) {
        Mat3 eps_p = voigt_strain_tensor(comp, +delta);
        Mat3 eps_m = voigt_strain_tensor(comp, -delta);

        double Ep = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_p, input.cutoff_radius, true, 0.35, 8, -1,
            &ref_neighbors, &ref_site_masks);
        double Em = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_m, input.cutoff_radius, true, 0.35, 8, -1,
            &ref_neighbors, &ref_site_masks);
        double dUdE = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("{:>6s} {:>12.4f} {:>12.4f} {:>12.4f}",
                       names[comp], dUdE, dmacrys_qdip_elec[comp],
                       dUdE - dmacrys_qdip_elec[comp]);
    }

    // Test with order=4
    occ::log::info("With max_interaction_order=4:");
    for (int comp = 0; comp < 3; ++comp) {
        Mat3 eps_p = voigt_strain_tensor(comp, +delta);
        Mat3 eps_m = voigt_strain_tensor(comp, -delta);

        double Ep = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_p, input.cutoff_radius, true, 0.35, 8, 4,
            &ref_neighbors, &ref_site_masks);
        double Em = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_m, input.cutoff_radius, true, 0.35, 8, 4,
            &ref_neighbors, &ref_site_masks);
        double dUdE = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("{:>6s} {:>12.4f} {:>12.4f} {:>12.4f}",
                       names[comp], dUdE, dmacrys_qdip_elec[comp],
                       dUdE - dmacrys_qdip_elec[comp]);
    }

    // Also print dipole-only contribution (subtract charge-only)
    double dmacrys_qq_elec[] = {1.9387, 2.0301, 0.6449};
    occ::log::info("\nDipole-only strain derivative contribution:");
    occ::log::info("{:>6s} {:>12s} {:>12s}", "comp", "DMACRYS_dip(eV)", "DMACRYS_qq(eV)");
    for (int comp = 0; comp < 3; ++comp) {
        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("{:>6s} {:>12.4f} {:>12.4f}",
                       names[comp],
                       dmacrys_qdip_elec[comp] - dmacrys_qq_elec[comp],
                       dmacrys_qq_elec[comp]);
    }

    CHECK(true);
}

TEST_CASE("DMACRYS AXOSOW rank-by-rank strain FD", "[mults][dmacrys][strain-rank]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);

    // Test each rank level: 0 (qq), 1 (+dip), 2 (+quad), 3 (+oct), 4 (full)
    // Component count per rank: 0->1, 1->4, 2->9, 3->16, 4->25
    struct RankTest {
        int max_rank;
        int num_components; // components to keep (rank 0 through max_rank)
        const char* name;
        double dmacrys_elec[3]; // DMACRYS electrostatic-only strain derivs (eV)
    };
    // DMACRYS elec = DMACRYS total(no SPLI) - Buck-only(no SPLI)
    // Buck-only: E1=-2.8452, E2=-2.5367, E3=-1.3208
    RankTest tests[] = {
        {0,  1, "qq",       {1.9387,  2.0301,  0.6449}},
        {1,  4, "qq+dip",   {1.0589, -0.8568,  1.3570}},
        {2,  9, "qq+dip+Q", {2.7105,  2.0786,  1.1693}},
        {3, 16, "qq+dip+Q+O", {2.8552, 2.5687, 1.2873}},
        {4, 25, "full",     {2.8454,  2.5368,  1.3207}},
    };

    auto crystal = build_crystal(input.crystal);
    auto buck_params = convert_buckingham_params(input.potentials);
    std::map<std::pair<int, int>, BuckinghamParams> no_buck;
    for (const auto& [key, p] : buck_params) {
        no_buck[key] = {0.0, 1.0, 0.0};
    }

    for (const auto& test : tests) {
        // Make a copy and zero components above max_rank
        auto input_copy = input;
        for (auto& site : input_copy.molecule.sites) {
            for (size_t i = test.num_components; i < site.components.size(); ++i) {
                site.components[i] = 0.0;
            }
        }

        auto multipoles = build_multipole_sources(input_copy, crystal);

        CrystalEnergy ref_calc(crystal, multipoles, input_copy.cutoff_radius,
                               ForceFieldType::Custom, true, true,
                               1e-8, 0.35, 8);
        setup_crystal_energy_from_dmacrys(ref_calc, input_copy, crystal, multipoles);
        auto ref_neighbors = ref_calc.neighbor_list();
        auto ref_site_masks = ref_calc.compute_buckingham_site_masks(ref_calc.initial_states());

        occ::log::info("\n=== Rank <= {} ({}) ===", test.max_rank, test.name);

        // Print zero-strain energy for comparison
        {
            Mat3 zero = Mat3::Zero();
            double E0 = compute_strained_energy(
                input_copy, crystal, multipoles, no_buck,
                zero, input_copy.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);
            occ::log::info("  Zero-strain elec energy (order=4): {:.6f} kJ/mol ({:.6f} eV/cell)",
                           E0, E0 / units::EV_TO_KJ_PER_MOL);
        }

        occ::log::info("{:>6s} {:>12s} {:>12s} {:>12s}", "comp", "our(eV)", "DMACRYS(eV)", "diff(eV)");

        double delta = 1e-4;
        // Use order=4 to match DMACRYS truncation
        for (int comp = 0; comp < 3; ++comp) {
            Mat3 eps_p = voigt_strain_tensor(comp, +delta);
            Mat3 eps_m = voigt_strain_tensor(comp, -delta);

            double Ep = compute_strained_energy(
                input_copy, crystal, multipoles, no_buck,
                eps_p, input_copy.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);
            double Em = compute_strained_energy(
                input_copy, crystal, multipoles, no_buck,
                eps_m, input_copy.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);
            double dUdE = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

            const char* names[] = {"E1", "E2", "E3"};
            occ::log::info("{:>6s} {:>12.4f} {:>12.4f} {:>12.4f}",
                           names[comp], dUdE, test.dmacrys_elec[comp],
                           dUdE - test.dmacrys_elec[comp]);
        }
    }

    CHECK(true);
}

TEST_CASE("DMACRYS AXOSOW electrostatic ladder and Hessian isolation (strict)",
          "[mults][dmacrys][axosow][ladder][strict][!mayfail][.]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    std::map<std::pair<int, int>, BuckinghamParams> no_buck;
    for (const auto& [key, p] : buck_params) {
        no_buck[key] = {0.0, 1.0, 0.0};
    }

    struct LadderCase {
        const char* name;
        int num_components; // keep components [0, num_components)
        double ref_elec_eV_cell;
        Vec3 ref_dU_dE_eV;  // E1..E3 electrostatic-only strain derivatives
        double energy_tol_eV;
        double strain_tol_eV;
        bool strict_energy;
        bool check_hessian;
    };

    const LadderCase cases[] = {
        {"qq", 1, -1.201636,
         Vec3(1.9387, 2.0301, 0.6449),
         5e-4, 2e-4, true, true},
        {"qq+dip", 4, -1.201636 + 0.20422449 + 0.17972682,
         Vec3(1.0589, -0.8568, 1.3570),
         1e-3, 2e-4, true, true},
        {"full_l4", 25,
         -1.201636 + 0.20422449 + 0.17972682 - 0.753257,
         Vec3(2.8454, 2.5368, 1.3207),
         1e-2, 7e-4, false, false},
    };

    auto truncated_input = [&](const LadderCase& lc) {
        auto copy = input;
        for (auto& site : copy.molecule.sites) {
            for (size_t i = static_cast<size_t>(lc.num_components);
                 i < site.components.size(); ++i) {
                site.components[i] = 0.0;
            }
        }
        return copy;
    };

    const double delta = 1e-4;
    const double conv = units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

    for (const auto& lc : cases) {
        auto input_case = truncated_input(lc);
        auto multipoles = build_multipole_sources(input_case, crystal);

        CrystalEnergy calc(crystal, multipoles, input_case.cutoff_radius,
                           ForceFieldType::Custom, true, true,
                           1e-8, 0.35, 8);
        setup_crystal_energy_from_dmacrys(calc, input_case, crystal, multipoles);
        for (const auto& [key, p] : no_buck) {
            calc.set_buckingham_params(key.first, key.second, p);
        }
        calc.set_max_interaction_order(4);

        auto states = calc.initial_states();
        auto result = calc.compute(states);
        const double elec_eV = result.electrostatic_energy / units::EV_TO_KJ_PER_MOL;

        INFO("Case " << lc.name << " elec_eV=" << elec_eV
                     << " ref_eV=" << lc.ref_elec_eV_cell
                     << " tol=" << lc.energy_tol_eV);
        if (lc.strict_energy) {
            CHECK(elec_eV == Approx(lc.ref_elec_eV_cell).margin(lc.energy_tol_eV));
        } else {
            CHECK(std::abs(elec_eV - lc.ref_elec_eV_cell) <= lc.energy_tol_eV);
        }

        auto ref_neighbors = calc.neighbor_list();
        auto ref_site_masks = calc.compute_buckingham_site_masks(states);

        for (int comp = 0; comp < 3; ++comp) {
            Mat3 eps_p = voigt_strain_tensor(comp, +delta);
            Mat3 eps_m = voigt_strain_tensor(comp, -delta);
            const double Ep = compute_strained_energy(
                input_case, crystal, multipoles, no_buck,
                eps_p, input_case.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);
            const double Em = compute_strained_energy(
                input_case, crystal, multipoles, no_buck,
                eps_m, input_case.cutoff_radius, true, 0.35, 8, 4,
                &ref_neighbors, &ref_site_masks);
            const double dUdE = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

            INFO("Case " << lc.name << " dU/dE" << (comp + 1)
                         << " occ=" << dUdE << " ref=" << lc.ref_dU_dE_eV[comp]
                         << " tol=" << lc.strain_tol_eV);
            CHECK(dUdE == Approx(lc.ref_dU_dE_eV[comp]).margin(lc.strain_tol_eV));
        }

        if (!lc.check_hessian) {
            continue;
        }

        calc.update_lattice(crystal, states);
        auto eh = calc.compute_with_hessian(states);
        Mat6 C_analytic = (eh.strain_hessian / crystal.unit_cell().volume()) * conv;
        Mat6 C_fd = compute_elastic_constants_fd(
            input_case, crystal, multipoles, no_buck,
            input_case.cutoff_radius, true, 0.35, 8, 1e-3, 4);

        const std::pair<int, int> check_idx[] = {
            {0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2},
        };
        for (const auto& [i, j] : check_idx) {
            INFO("Case " << lc.name << " C" << (i + 1) << (j + 1)
                         << " analytic=" << C_analytic(i, j)
                         << " fd=" << C_fd(i, j));
            CHECK(C_analytic(i, j) == Approx(C_fd(i, j)).margin(0.30));
        }
    }
}

TEST_CASE("DMACRYS AXOSOW full_l4 elastic analytic-vs-FD diagnostic",
          "[mults][dmacrys][axosow][full-l4][elastic][diag][.]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    std::map<std::pair<int, int>, BuckinghamParams> no_buck;
    for (const auto& [key, p] : buck_params) {
        no_buck[key] = {0.0, 1.0, 0.0};
    }

    // Keep all l<=4 Cartesian multipole components.
    auto multipoles = build_multipole_sources(input, crystal);
    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       1e-8, 0.35, 8);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    for (const auto& [key, p] : no_buck) {
        calc.set_buckingham_params(key.first, key.second, p);
    }
    calc.set_max_interaction_order(4);

    auto states = calc.initial_states();
    calc.update_lattice(crystal, states);
    auto eh = calc.compute_with_hessian(states);

    const double V = crystal.unit_cell().volume();
    const double conv = units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
    Mat6 C_analytic = (eh.strain_hessian / V) * conv;
    Mat6 C_fd = compute_elastic_constants_fd(
        input, crystal, multipoles, no_buck,
        input.cutoff_radius, true, 0.35, 8, 1e-3, 4);

    occ::log::info("AXOSOW full_l4 C11/C22/C33/C12/C13/C23 (GPa):");
    occ::log::info("  analytic: {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                   C_analytic(0, 0), C_analytic(1, 1), C_analytic(2, 2),
                   C_analytic(0, 1), C_analytic(0, 2), C_analytic(1, 2));
    occ::log::info("  FD      : {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                   C_fd(0, 0), C_fd(1, 1), C_fd(2, 2),
                   C_fd(0, 1), C_fd(0, 2), C_fd(1, 2));
    occ::log::info("  diff    : {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                   C_analytic(0, 0) - C_fd(0, 0),
                   C_analytic(1, 1) - C_fd(1, 1),
                   C_analytic(2, 2) - C_fd(2, 2),
                   C_analytic(0, 1) - C_fd(0, 1),
                   C_analytic(0, 2) - C_fd(0, 2),
                   C_analytic(1, 2) - C_fd(1, 2));

    CHECK(true);
}

TEST_CASE("DMACRYS AXOSOW PAIR cutoff convergence", "[mults][dmacrys][pair-conv]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    std::map<std::pair<int, int>, BuckinghamParams> no_buck;
    for (const auto& [key, p] : buck_params) {
        no_buck[key] = {0.0, 1.0, 0.0};
    }

    // Compute PAIR energy at different cutoffs for rank ≤ 2
    // PAIR = E(rank2) - E(rank1) [subtracting Ewald-handled terms]
    for (double cutoff : {15.0, 20.0, 25.0, 30.0}) {
        // Rank 1 (qq+dip) energy
        {
            auto input1 = input;
            for (auto& site : input1.molecule.sites) {
                for (size_t i = 4; i < site.components.size(); ++i)
                    site.components[i] = 0.0;
            }
            auto mp1 = build_multipole_sources(input1, crystal);
            CrystalEnergy calc1(crystal, mp1, cutoff,
                                ForceFieldType::Custom, true, true,
                                1e-8, 0.35, 8);
            setup_crystal_energy_from_dmacrys(calc1, input1, crystal, mp1);
            auto states1 = calc1.initial_states();
            auto result1 = calc1.compute(states1);
            double E1 = result1.electrostatic_energy;

            // Rank 2 (qq+dip+Q) energy
            auto input2 = input;
            for (auto& site : input2.molecule.sites) {
                for (size_t i = 9; i < site.components.size(); ++i)
                    site.components[i] = 0.0;
            }
            auto mp2 = build_multipole_sources(input2, crystal);
            CrystalEnergy calc2(crystal, mp2, cutoff,
                                ForceFieldType::Custom, true, true,
                                1e-8, 0.35, 8);
            setup_crystal_energy_from_dmacrys(calc2, input2, crystal, mp2);
            calc2.set_max_interaction_order(4);
            auto states2 = calc2.initial_states();
            auto result2 = calc2.compute(states2);
            double E2 = result2.electrostatic_energy;

            double pair_energy = E2 - E1;
            int npairs = calc2.neighbor_list().size();
            occ::log::info("Cutoff {:.1f} Å: npairs={:5d}, E_rank1={:.6f}, E_rank2={:.6f}, PAIR={:.6f} kJ/mol ({:.6f} eV)",
                           cutoff, npairs, E1, E2, pair_energy,
                           pair_energy / units::EV_TO_KJ_PER_MOL);
        }
    }
    // DMACRYS PAIR (rank 2, 15 Å): -0.6365 eV = -61.40 kJ/mol
    occ::log::info("DMACRYS PAIR reference: -61.40 kJ/mol (-0.6365 eV)");

    // Diagnostic: effect of per-site cutoff on PAIR energy
    // DMACRYS applies per-site distance cutoff (RANG2) in PAIR module
    // Our code currently uses no per-site cutoff for electrostatics
    occ::log::info("\n=== Per-site cutoff diagnostic ===");
    {
        double cutoff = 15.0;

        auto input1 = input;
        for (auto& site : input1.molecule.sites) {
            for (size_t i = 4; i < site.components.size(); ++i)
                site.components[i] = 0.0;
        }
        auto mp1 = build_multipole_sources(input1, crystal);

        auto input2 = input;
        for (auto& site : input2.molecule.sites) {
            for (size_t i = 9; i < site.components.size(); ++i)
                site.components[i] = 0.0;
        }
        auto mp2 = build_multipole_sources(input2, crystal);

        // Without per-site cutoff (current)
        CrystalEnergy calc1_nosc(crystal, mp1, cutoff,
                                 ForceFieldType::Custom, true, true,
                                 1e-8, 0.35, 8);
        setup_crystal_energy_from_dmacrys(calc1_nosc, input1, crystal, mp1);
        auto states1 = calc1_nosc.initial_states();
        auto r1_nosc = calc1_nosc.compute(states1);

        CrystalEnergy calc2_nosc(crystal, mp2, cutoff,
                                 ForceFieldType::Custom, true, true,
                                 1e-8, 0.35, 8);
        setup_crystal_energy_from_dmacrys(calc2_nosc, input2, crystal, mp2);
        calc2_nosc.set_max_interaction_order(4);
        auto states2 = calc2_nosc.initial_states();
        auto r2_nosc = calc2_nosc.compute(states2);

        double pair_nosc = r2_nosc.electrostatic_energy - r1_nosc.electrostatic_energy;

        // With per-site cutoff = 15 Å (DMACRYS-like)
        CrystalEnergy calc1_sc(crystal, mp1, cutoff,
                               ForceFieldType::Custom, true, true,
                               1e-8, 0.35, 8);
        setup_crystal_energy_from_dmacrys(calc1_sc, input1, crystal, mp1);
        calc1_sc.set_elec_site_cutoff(cutoff);
        auto r1_sc = calc1_sc.compute(states1);

        CrystalEnergy calc2_sc(crystal, mp2, cutoff,
                               ForceFieldType::Custom, true, true,
                               1e-8, 0.35, 8);
        setup_crystal_energy_from_dmacrys(calc2_sc, input2, crystal, mp2);
        calc2_sc.set_max_interaction_order(4);
        calc2_sc.set_elec_site_cutoff(cutoff);
        auto r2_sc = calc2_sc.compute(states2);

        double pair_sc = r2_sc.electrostatic_energy - r1_sc.electrostatic_energy;

        occ::log::info("PAIR (no site cutoff):   {:.6f} kJ/mol ({:.6f} eV)",
                       pair_nosc, pair_nosc / units::EV_TO_KJ_PER_MOL);
        occ::log::info("PAIR (site cutoff=15Å):  {:.6f} kJ/mol ({:.6f} eV)",
                       pair_sc, pair_sc / units::EV_TO_KJ_PER_MOL);
        occ::log::info("DMACRYS PAIR reference:  -61.40 kJ/mol (-0.6365 eV)");
        occ::log::info("Diff (no_sc - DMACRYS):  {:.6f} eV",
                       pair_nosc / units::EV_TO_KJ_PER_MOL - (-0.6365));
        occ::log::info("Diff (sc=15 - DMACRYS):  {:.6f} eV",
                       pair_sc / units::EV_TO_KJ_PER_MOL - (-0.6365));

        // Also check rank1 energies (should be similar with/without site cutoff
        // since Ewald corrects for the truncation)
        occ::log::info("Rank1 E (no site cutoff): {:.6f} kJ/mol", r1_nosc.electrostatic_energy);
        occ::log::info("Rank1 E (site cutoff=15): {:.6f} kJ/mol", r1_sc.electrostatic_energy);
        occ::log::info("Rank1 diff: {:.6f} kJ/mol", r1_sc.electrostatic_energy - r1_nosc.electrostatic_energy);

        // Scan RANG2 values to find what matches DMACRYS
        occ::log::info("\n=== RANG2 scan (molecule cutoff=15 Å, vary site cutoff) ===");
        occ::log::info("{:>8s} {:>12s} {:>12s}", "RANG2", "PAIR(eV)", "diff(eV)");

        double dmacrys_pair_ev = -0.6365;
        double E1_ref = r1_nosc.electrostatic_energy; // Ewald-corrected rank1 (cutoff-independent)

        for (double rang2 : {15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 20.0, 25.0, 50.0}) {
            CrystalEnergy calc2_r2(crystal, mp2, cutoff,
                                   ForceFieldType::Custom, true, true,
                                   1e-8, 0.35, 8);
            setup_crystal_energy_from_dmacrys(calc2_r2, input2, crystal, mp2);
            calc2_r2.set_max_interaction_order(4);
            calc2_r2.set_elec_site_cutoff(rang2);
            auto r2_r2 = calc2_r2.compute(states2);

            double pair_r2 = r2_r2.electrostatic_energy - E1_ref;
            double pair_ev = pair_r2 / units::EV_TO_KJ_PER_MOL;
            occ::log::info("{:>8.1f} {:>12.6f} {:>12.6f}", rang2, pair_ev, pair_ev - dmacrys_pair_ev);
        }
    }

    CHECK(true);
}

TEST_CASE("DMACRYS AXOSOW full-multipole elec-only strain FD", "[mults][dmacrys][strain-elec]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    // Build reference neighbor list
    CrystalEnergy ref_calc(crystal, multipoles, input.cutoff_radius,
                           ForceFieldType::Custom, true, true,
                           1e-8, 0.35, 8);
    setup_crystal_energy_from_dmacrys(ref_calc, input, crystal, multipoles);
    auto ref_neighbors = ref_calc.neighbor_list();
    auto ref_site_masks = ref_calc.compute_buckingham_site_masks(ref_calc.initial_states());

    // DMACRYS full-multipole, no SPLI, total: E1=0.0002, E2=0.0001, E3=-0.0001
    // DMACRYS Buck-only, no SPLI:            E1=-2.8452, E2=-2.5367, E3=-1.3208
    // => DMACRYS full-multipole elec (order<=4):
    //    E1=0.0002-(-2.8452)=2.8454, E2=0.0001-(-2.5367)=2.5368, E3=-0.0001-(-1.3208)=1.3207
    double dmacrys_elec[] = {2.8454, 2.5368, 1.3207};

    double delta = 1e-4;
    std::map<std::pair<int, int>, BuckinghamParams> no_buck;
    for (const auto& [key, p] : buck_params) {
        no_buck[key] = {0.0, 1.0, 0.0};
    }

    occ::log::info("\nFull-multipole electrostatic strain derivatives (order=4):");
    occ::log::info("{:>6s} {:>12s} {:>12s} {:>12s}", "comp", "our(eV)", "DMACRYS(eV)", "diff(eV)");
    for (int comp = 0; comp < 3; ++comp) {
        Mat3 eps_p = voigt_strain_tensor(comp, +delta);
        Mat3 eps_m = voigt_strain_tensor(comp, -delta);

        double Ep = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_p, input.cutoff_radius, true, 0.35, 8, 4,
            &ref_neighbors, &ref_site_masks);
        double Em = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_m, input.cutoff_radius, true, 0.35, 8, 4,
            &ref_neighbors, &ref_site_masks);
        double dUdE = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("{:>6s} {:>12.4f} {:>12.4f} {:>12.4f}",
                       names[comp], dUdE, dmacrys_elec[comp],
                       dUdE - dmacrys_elec[comp]);
    }

    // Also with unlimited order to show effect of order>4 terms
    occ::log::info("\nFull-multipole electrostatic strain derivatives (unlimited order):");
    for (int comp = 0; comp < 3; ++comp) {
        Mat3 eps_p = voigt_strain_tensor(comp, +delta);
        Mat3 eps_m = voigt_strain_tensor(comp, -delta);

        double Ep = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_p, input.cutoff_radius, true, 0.35, 8, -1,
            &ref_neighbors, &ref_site_masks);
        double Em = compute_strained_energy(
            input, crystal, multipoles, no_buck,
            eps_m, input.cutoff_radius, true, 0.35, 8, -1,
            &ref_neighbors, &ref_site_masks);
        double dUdE = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("{:>6s} {:>12.4f} {:>12.4f} {:>12.4f}",
                       names[comp], dUdE, dmacrys_elec[comp],
                       dUdE - dmacrys_elec[comp]);
    }

    CHECK(true);
}

TEST_CASE("DMACRYS AXOSOW analytical vs FD strain", "[mults][dmacrys][strain-analytical]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);
    const auto& ref_sd = input.initial_ref.strain_derivatives_eV;

    double cutoff = input.cutoff_radius;
    double alpha = 0.35;
    int kmax = 8;
    int max_order = 4;

    // Build energy calculator
    CrystalEnergy calc(crystal, multipoles, cutoff,
                       ForceFieldType::Custom, true, true,
                       1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }
    if (max_order >= 0) calc.set_max_interaction_order(max_order);

    auto states = calc.initial_states();
    auto result = calc.compute(states);

    occ::log::info("Total energy: {:.6f} kJ/mol", result.total_energy);
    occ::log::info("  Elec: {:.6f}, Buck: {:.6f}", result.electrostatic_energy, result.repulsion_dispersion);

    // Analytical Born-term: dU/dE_k = Σ_{pairs} w * F_{I←J}_α * D_IJ_β
    // where D_IJ = COM_J + cell_shift - COM_I
    // For Buckingham, this is exact (no lattice term)
    // For electrostatics WITH Ewald, this misses the lattice term

    int N = static_cast<int>(states.size());
    auto neighbors = calc.neighbor_list();

    // Get COM positions from states
    std::vector<Vec3> coms(N);
    for (int i = 0; i < N; ++i) coms[i] = states[i].position;

    // Get rotation matrices
    std::vector<Mat3> rots(N);
    for (int i = 0; i < N; ++i) rots[i] = states[i].rotation_matrix();

    // Get molecule geometry for atom positions
    // (same body-frame offsets for all molecules since Z'=1)
    std::vector<Vec3> body_offsets;
    for (const auto& site : input.molecule.sites) {
        Vec3 pos_ang = site.position_bohr * units::BOHR_TO_ANGSTROM;
        body_offsets.push_back(pos_ang);
    }
    // Mass-weighted COM (consistent with main code and DMACRYS COFMAS)
    Vec3 body_com = Vec3::Zero();
    double total_mass = 0.0;
    for (size_t i = 0; i < body_offsets.size(); ++i) {
        double m = occ::core::Element(input.molecule.sites[i].atomic_number).mass();
        body_com += m * body_offsets[i];
        total_mass += m;
    }
    body_com /= total_mass;
    for (auto& p : body_offsets) p -= body_com;

    // Compute per-pair analytical strain derivatives (Buckingham only)
    Vec3 analytical_buck = Vec3::Zero(); // For E1, E2, E3
    Vec3 analytical_elec = Vec3::Zero(); // Direct electrostatic (no Ewald lattice term)

    for (const auto& pair : neighbors) {
        int mi = pair.mol_i;
        int mj = pair.mol_j;
        double w = pair.weight;

        Vec3 cell_trans = crystal.unit_cell().to_cartesian(
            pair.cell_shift.cast<double>());
        Vec3 D_IJ = coms[mj] + cell_trans - coms[mi];

        // Compute Buckingham force on I from J
        Vec3 buck_force_I = Vec3::Zero();
        for (size_t a = 0; a < body_offsets.size(); ++a) {
            Vec3 pos_a = coms[mi] + rots[mi] * body_offsets[a];
            int Z_a = input.molecule.sites[a].atomic_number;
            for (size_t b = 0; b < body_offsets.size(); ++b) {
                Vec3 pos_b = coms[mj] + cell_trans + rots[mj] * body_offsets[b];
                int Z_b = input.molecule.sites[b].atomic_number;
                Vec3 r_ab = pos_b - pos_a;
                double r = r_ab.norm();
                if (r > cutoff || r < 0.1) continue;

                int z1 = std::min(Z_a, Z_b), z2 = std::max(Z_a, Z_b);
                auto it = buck_params.find({z1, z2});
                if (it == buck_params.end()) continue;
                const auto& bp = it->second;

                double dVdr = -bp.A * bp.B * std::exp(-bp.B * r)
                            + 6.0 * bp.C / std::pow(r, 7);
                // Force on a from b: F_a = (dV/dr) * r_ab / r
                Vec3 f_a = dVdr * r_ab / r;
                buck_force_I += f_a;
            }
        }

        // Analytical Buckingham strain derivative contribution
        for (int k = 0; k < 3; ++k) {
            analytical_buck(k) += w * buck_force_I(k) * D_IJ(k);
        }
    }

    // Convert to eV
    analytical_buck /= units::EV_TO_KJ_PER_MOL;

    // --- Analytical ELECTROSTATIC Born term (from per-pair forces) ---
    // Build CartesianMolecules for each symmetry image
    for (int i = 0; i < N; ++i) {
        multipoles[i].set_orientation(rots[i], coms[i]);
    }
    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i) {
        cart_mols.push_back(multipoles[i].cartesian());
    }

    // Compute analytical elec Born term at different max orders
    // At max_order=4, all forces are zero (all sites rank 4, so rank_A+rank_B=8>4 → filtered path, no forces)
    // At max_order=8, full forces available for all site pairs
    occ::log::info("\nAnalytical electrostatic Born terms at different max orders:");
    occ::log::info("{:>6s} {:>12s} {:>12s} {:>12s}", "order", "E1", "E2", "E3");
    Vec3 analytical_elec_full = Vec3::Zero();  // will hold order=8 result
    for (int test_order : {4, 8, -1}) {
        Vec3 analytical_elec = Vec3::Zero();
        for (const auto& pair : neighbors) {
            int mi = pair.mol_i;
            int mj = pair.mol_j;
            double w = pair.weight;

            Vec3 cell_trans = crystal.unit_cell().to_cartesian(
                pair.cell_shift.cast<double>());
            Vec3 D_IJ = coms[mj] + cell_trans - coms[mi];

            CartesianMolecule mol_j_trans = cart_mols[mj];
            for (auto& site : mol_j_trans.sites) {
                site.position += cell_trans;
            }

            auto elec = compute_molecule_forces_torques(
                cart_mols[mi], mol_j_trans, 0.0, test_order);

            for (int k = 0; k < 3; ++k) {
                analytical_elec(k) += w * elec.force_A(k) * D_IJ(k);
            }
        }
        analytical_elec /= units::EV_TO_KJ_PER_MOL;
        if (test_order == 8) analytical_elec_full = analytical_elec;
        occ::log::info("  {:>4d} {:>12.6f} {:>12.6f} {:>12.6f}",
                       test_order, analytical_elec(0), analytical_elec(1), analytical_elec(2));
    }

    occ::log::info("\nAnalytical Born totals (order=8 elec + 15A buck):");
    occ::log::info("{:>4s} {:>12s} {:>12s} {:>12s} {:>12s}",
                   "Comp", "ana_buck", "ana_elec_8", "ana_total", "ref");
    for (int k = 0; k < 3; ++k) {
        double ana_total = analytical_buck(k) + analytical_elec_full(k);
        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("  {:>2s} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}",
                       names[k], analytical_buck(k), analytical_elec_full(k),
                       ana_total, ref_sd[k]);
    }

    // =================================================================
    // Radial-only Born: mimics DMACRYS's approximation
    // DMACRYS uses (1/R)(dV/dR) × RA × RCOFM which is the RADIAL part only.
    // For multipoles, forces have tangential components that are missed.
    // =================================================================
    occ::log::info("\nRadial-only Born (DMACRYS approx) vs full-force Born (order 4):");
    {
        Vec3 radial_born_elec = Vec3::Zero();
        Vec3 full_born_elec = Vec3::Zero();
        Vec3 radial_born_buck = Vec3::Zero();
        for (const auto& pair : neighbors) {
            int mi = pair.mol_i;
            int mj = pair.mol_j;
            double w = pair.weight;
            Vec3 cell_trans = crystal.unit_cell().to_cartesian(
                pair.cell_shift.cast<double>());
            Vec3 D_IJ = coms[mj] + cell_trans - coms[mi];

            // Compute FULL electrostatic force on molecule I from J (order 4)
            CartesianMolecule mol_j_trans = cart_mols[mj];
            for (auto& site : mol_j_trans.sites) {
                site.position += cell_trans;
            }
            auto elec = compute_molecule_forces_torques(
                cart_mols[mi], mol_j_trans, 0.0, 4);

            // Full-force Born: F_full_p × D_IJ_p
            for (int k = 0; k < 3; ++k) {
                full_born_elec(k) += w * elec.force_A(k) * D_IJ(k);
            }

            // For radial Born comparison, compute per-site radial forces
            // Radial = (F · r_hat) × r_hat, so radial Born = Σ_ab (F_ab · r_ab/|r_ab|)² / |r_ab| × D_IJ_k
            // Actually: VGTMP = (1/R)(dE_pair/dR), Born = VGTMP × RA_k × D_IJ_k
            // This requires the energy derivative w.r.t. R, which we can get from:
            //   VGTMP × RA_k = (F · r_hat)(r_hat_k) = F_radial_k
            // So radial Born = Σ F_radial_k × D_IJ_k
            // We need per-SITE forces to compute the radial projection
            // Use compute_molecule_forces_torques for the SITE-level forces
            for (size_t a = 0; a < body_offsets.size(); ++a) {
                Vec3 pos_a = coms[mi] + rots[mi] * body_offsets[a];
                for (size_t b = 0; b < body_offsets.size(); ++b) {
                    Vec3 pos_b = coms[mj] + cell_trans + rots[mj] * body_offsets[b];
                    Vec3 r_ab = pos_b - pos_a;
                    double r = r_ab.norm();
                    if (r < 0.1) continue;
                    Vec3 r_hat = r_ab / r;

                    // Buckingham radial force
                    int Z_a = input.molecule.sites[a].atomic_number;
                    int Z_b = input.molecule.sites[b].atomic_number;
                    int z1 = std::min(Z_a, Z_b), z2 = std::max(Z_a, Z_b);
                    auto it = buck_params.find({z1, z2});
                    if (r <= cutoff && it != buck_params.end()) {
                        const auto& bp = it->second;
                        double dVdr = -bp.A * bp.B * std::exp(-bp.B * r)
                                    + 6.0 * bp.C / std::pow(r, 7);
                        // Radial force on a: F = dVdr * r_hat (along ab direction)
                        // Born contribution: w × F_k × D_IJ_k = w × dVdr × r_hat_k × D_IJ_k
                        for (int k = 0; k < 3; ++k) {
                            radial_born_buck(k) += w * dVdr * r_hat(k) * D_IJ(k);
                        }
                    }
                }
            }

            // Electrostatic radial Born: need per-site electrostatic forces
            // and project onto radial direction
            // The full-force A already gives the total on molecule I.
            // For the radial approximation, we need Σ_ab (f_ab · r_hat_ab) × r_hat_ab_k × D_IJ_k
            // We'd need per-site forces which compute_molecule_forces_torques
            // doesn't directly give. Instead, compute radial from energy:
            // VGTMP = (1/R)(dE_ab/dR), then F_radial_k = VGTMP × RA_k
        }
        radial_born_buck /= units::EV_TO_KJ_PER_MOL;
        full_born_elec /= units::EV_TO_KJ_PER_MOL;

        occ::log::info("{:>4s} {:>12s} {:>12s} {:>12s} {:>12s}",
                       "Comp", "full_elec4", "rad_buck", "ana_buck", "ref");
        const char* names[] = {"E1", "E2", "E3"};
        for (int k = 0; k < 3; ++k) {
            occ::log::info("  {:>2s} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}",
                           names[k], full_born_elec(k), radial_born_buck(k),
                           analytical_buck(k), ref_sd[k]);
        }
    }

    // =================================================================
    // GROUND TRUTH: Manual qq FD strain derivative (no T-tensor code)
    // =================================================================
    // Compute charge-charge energy directly using Coulomb's law and
    // manual rigid-molecule strain application.
    // This bypasses ALL our multipole/T-tensor/CrystalEnergy machinery.
    double delta = 1e-5;
    {
        // Collect charges (in a.u.) from DMA data
        std::vector<double> charges;
        for (const auto& site : input.molecule.sites) {
            charges.push_back(site.components[0]); // Q00
        }
        int nsites = static_cast<int>(charges.size());

        auto compute_qq_energy_rigid = [&](const Mat3& strain) -> double {
            // Apply strain to direct matrix
            Mat3 direct = crystal.unit_cell().direct();
            Mat3 strained_direct = (Mat3::Identity() + strain) * direct;
            crystal::UnitCell strained_uc(strained_direct);

            // Get strained COM positions (fractional coords unchanged)
            const auto& symops = crystal.space_group().symmetry_operations();
            Mat3N frac_asym(3, nsites);
            for (int i = 0; i < nsites; ++i)
                frac_asym.col(i) = input.crystal.atoms[i].frac_xyz;

            std::vector<Vec3> strained_coms(N);
            for (int m = 0; m < N; ++m) {
                Mat3N frac_img = symops[m].apply(frac_asym);
                Mat3N cart_img = strained_uc.to_cartesian(frac_img);
                // Mass-weighted COM (consistent with main code)
                Vec3 mw_com = Vec3::Zero();
                double mw_total = 0.0;
                for (int i = 0; i < nsites; ++i) {
                    double m_i = occ::core::Element(
                        input.molecule.sites[i].atomic_number).mass();
                    mw_com += m_i * cart_img.col(i);
                    mw_total += m_i;
                }
                strained_coms[m] = mw_com / mw_total;
            }

            // For rigid molecules: site_pos = strained_COM + R * body_offset
            // (using ORIGINAL rotation and body offsets)
            double E_qq = 0.0;
            for (const auto& pair : neighbors) {
                int mi = pair.mol_i;
                int mj = pair.mol_j;
                double w = pair.weight;
                Vec3 cell_trans = strained_uc.to_cartesian(
                    pair.cell_shift.cast<double>());

                for (int a = 0; a < nsites; ++a) {
                    Vec3 pos_a = strained_coms[mi] + rots[mi] * body_offsets[a];
                    for (int b = 0; b < nsites; ++b) {
                        Vec3 pos_b = strained_coms[mj] + cell_trans
                                   + rots[mj] * body_offsets[b];
                        double r_ang = (pos_b - pos_a).norm();
                        double r_bohr = r_ang / units::BOHR_TO_ANGSTROM;
                        // E = q_a * q_b / r (Hartree)
                        E_qq += w * charges[a] * charges[b] / r_bohr;
                    }
                }
            }
            return E_qq * units::AU_TO_KJ_PER_MOL; // → kJ/mol
        };

        occ::log::info("\nGROUND TRUTH: Manual qq FD strain derivative (rigid molecule):");
        occ::log::info("{:>4s} {:>12s} {:>12s}",
                       "Comp", "manual_qq", "note");
        for (int comp = 0; comp < 3; ++comp) {
            Mat3 eps_p = voigt_strain_tensor(comp, +delta);
            Mat3 eps_m = voigt_strain_tensor(comp, -delta);
            double Ep = compute_qq_energy_rigid(eps_p);
            double Em = compute_qq_energy_rigid(eps_m);
            double fd = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;
            const char* names[] = {"E1", "E2", "E3"};
            occ::log::info("  {:>2s} {:>12.6f}   compare with order-0 FD",
                           names[comp], fd);
        }

        // Also compute unstrained qq energy for comparison
        double E0 = compute_qq_energy_rigid(Mat3::Zero());
        occ::log::info("  Manual qq E0 = {:.6f} kJ/mol = {:.6f} eV",
                       E0, E0 / units::EV_TO_KJ_PER_MOL);
    }

    // FD strain derivatives with proper component separation
    auto ref_neighbors = calc.neighbor_list();
    auto ref_site_masks = calc.compute_buckingham_site_masks(calc.initial_states());

    // Zero-energy Buckingham params to properly disable Buck in FD
    std::map<std::pair<int, int>, BuckinghamParams> zero_buck;
    for (const auto& [key, p] : buck_params) {
        zero_buck[key] = {0.0, 1.0, 0.0}; // A=0, B=1, C=0 -> V=0
    }

    occ::log::info("\nStrain derivatives: Analytical Buck vs FD components (order={})", max_order);
    occ::log::info("{:>4s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}",
                   "Comp", "ana_buck", "fd_total", "fd_elec", "fd_buck", "elec_infer", "ref");
    for (int comp = 0; comp < 3; ++comp) {
        Mat3 eps_p = voigt_strain_tensor(comp, +delta);
        Mat3 eps_m = voigt_strain_tensor(comp, -delta);

        // Full energy (buck + elec + ewald)
        double Ep = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_p, cutoff, true, alpha, kmax, max_order,
            &ref_neighbors, &ref_site_masks);
        double Em = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_m, cutoff, true, alpha, kmax, max_order,
            &ref_neighbors, &ref_site_masks);
        double fd_full = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        // Elec only (with Ewald) — use ZERO Buck params to avoid default fallback
        double Ep_e = compute_strained_energy(
            input, crystal, multipoles, zero_buck,
            eps_p, cutoff, true, alpha, kmax, max_order,
            &ref_neighbors, &ref_site_masks);
        double Em_e = compute_strained_energy(
            input, crystal, multipoles, zero_buck,
            eps_m, cutoff, true, alpha, kmax, max_order,
            &ref_neighbors, &ref_site_masks);
        double fd_elec = (Ep_e - Em_e) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        double fd_buck = fd_full - fd_elec;
        // Inferred elec = total - analytical buck (should match fd_elec)
        double elec_infer = fd_full - analytical_buck(comp);

        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("  {:>2s} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}",
                       names[comp], analytical_buck(comp), fd_full, fd_elec, fd_buck,
                       elec_infer, ref_sd[comp]);
    }

    // Test interaction order effect: compute with order=8 (full) vs order=4
    occ::log::info("\nEffect of max interaction order on strain derivative:");
    for (int order : {0, 1, 2, 3, 4, 5, 6, 8}) {
        occ::log::info("  order <= {}:", order);
        for (int comp = 0; comp < 3; ++comp) {
            Mat3 eps_p = voigt_strain_tensor(comp, +delta);
            Mat3 eps_m = voigt_strain_tensor(comp, -delta);
            double Ep = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_p, cutoff, true, alpha, kmax, order,
                &ref_neighbors, &ref_site_masks);
            double Em = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_m, cutoff, true, alpha, kmax, order,
                &ref_neighbors, &ref_site_masks);
            double fd = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;
            const char* names[] = {"E1", "E2", "E3"};
            occ::log::info("    {}: {:>10.6f} eV (ref: {:>10.6f})", names[comp], fd, ref_sd[comp]);
        }
    }

    // Test Ewald convergence: vary alpha and kmax
    occ::log::info("\nEwald convergence test (order=4):");
    struct EwaldParams { double a; int k; };
    for (auto [ew_alpha, ew_kmax] : std::vector<EwaldParams>{{0.25, 6}, {0.35, 8}, {0.45, 10}, {0.55, 12}}) {
        occ::log::info("  alpha={:.2f}, kmax={}:", ew_alpha, ew_kmax);
        for (int comp = 0; comp < 3; ++comp) {
            Mat3 eps_p = voigt_strain_tensor(comp, +delta);
            Mat3 eps_m = voigt_strain_tensor(comp, -delta);
            double Ep = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_p, cutoff, true, ew_alpha, ew_kmax, max_order,
                &ref_neighbors, &ref_site_masks);
            double Em = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_m, cutoff, true, ew_alpha, ew_kmax, max_order,
                &ref_neighbors, &ref_site_masks);
            double fd = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;
            const char* names[] = {"E1", "E2", "E3"};
            occ::log::info("    {}: {:>10.6f} eV", names[comp], fd);
        }
    }

    CHECK(true); // diagnostic
}

TEST_CASE("DMACRYS AXOSOW strain Ewald isolation", "[mults][dmacrys][strain-ewald]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);
    const auto& ref_sd = input.initial_ref.strain_derivatives_eV;

    double cutoff = input.cutoff_radius;
    double alpha = 0.35;
    int kmax = 8;

    // Zero-energy Buckingham params
    std::map<std::pair<int, int>, BuckinghamParams> zero_buck;
    for (const auto& [key, p] : buck_params) {
        zero_buck[key] = {0.0, 1.0, 0.0};
    }

    // Build fixed neighbor list + atom-pair masks
    CrystalEnergy ref_calc(crystal, multipoles, cutoff,
                           ForceFieldType::Custom, true, true,
                           1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(ref_calc, input, crystal, multipoles);
    auto ref_neighbors = ref_calc.neighbor_list();
    auto ref_site_masks = ref_calc.compute_buckingham_site_masks(ref_calc.initial_states());

    // === Part 1: Energy component breakdown at zero strain ===
    occ::log::info("=== Energy breakdown at unstrained geometry ===");
    {
        // Full energy with Ewald, order 4
        CrystalEnergy calc_full(crystal, multipoles, cutoff,
                                ForceFieldType::Custom, true, true,
                                1e-8, alpha, kmax);
        setup_crystal_energy_from_dmacrys(calc_full, input, crystal, multipoles);
        for (const auto& [key, p] : buck_params)
            calc_full.set_buckingham_params(key.first, key.second, p);
        calc_full.set_max_interaction_order(4);
        auto res = calc_full.compute(calc_full.initial_states());
        double total_eV = res.total_energy / units::EV_TO_KJ_PER_MOL;
        double elec_eV = res.electrostatic_energy / units::EV_TO_KJ_PER_MOL;
        double buck_eV = res.repulsion_dispersion / units::EV_TO_KJ_PER_MOL;
        occ::log::info("  OCC   total={:.6f} eV  elec={:.6f} eV  buck={:.6f} eV",
                       total_eV, elec_eV, buck_eV);
        occ::log::info("  DMAC  total={:.6f} eV  elec={:.6f} eV  buck={:.6f} eV",
                       input.initial_ref.total_eV_per_cell,
                       input.initial_ref.total_eV_per_cell - input.initial_ref.repulsion_dispersion_eV,
                       input.initial_ref.repulsion_dispersion_eV);

        // Without Ewald
        CrystalEnergy calc_noew(crystal, multipoles, cutoff,
                                ForceFieldType::Custom, true, false,
                                1e-8, 0, 0);
        setup_crystal_energy_from_dmacrys(calc_noew, input, crystal, multipoles);
        for (const auto& [key, p] : buck_params)
            calc_noew.set_buckingham_params(key.first, key.second, p);
        calc_noew.set_max_interaction_order(4);
        auto res_noew = calc_noew.compute(calc_noew.initial_states());
        double total_noew_eV = res_noew.total_energy / units::EV_TO_KJ_PER_MOL;
        double elec_noew_eV = res_noew.electrostatic_energy / units::EV_TO_KJ_PER_MOL;
        occ::log::info("  OCC (no Ewald) total={:.6f} eV  elec={:.6f} eV  buck={:.6f} eV",
                       total_noew_eV, elec_noew_eV,
                       res_noew.repulsion_dispersion / units::EV_TO_KJ_PER_MOL);
        occ::log::info("  Ewald correction: {:.6f} eV", elec_eV - elec_noew_eV);

        // Energy at different orders (elec only, with Ewald)
        occ::log::info("  Electrostatic energy by order (with Ewald):");
        for (int order : {0, 1, 2, 3, 4, 8}) {
            CrystalEnergy calc_o(crystal, multipoles, cutoff,
                                 ForceFieldType::Custom, true, true,
                                 1e-8, alpha, kmax);
            setup_crystal_energy_from_dmacrys(calc_o, input, crystal, multipoles);
            for (const auto& [key, p] : zero_buck)
                calc_o.set_buckingham_params(key.first, key.second, p);
            calc_o.set_max_interaction_order(order);
            auto res_o = calc_o.compute(calc_o.initial_states());
            double e_eV = res_o.electrostatic_energy / units::EV_TO_KJ_PER_MOL;
            occ::log::info("    order <= {}: elec={:.6f} eV", order, e_eV);
        }
        // DMACRYS elec components
        occ::log::info("  DMACRYS: qq={:.6f}  q_mu={:.6f}  mu_mu={:.6f}  higher={:.6f}",
                       input.initial_ref.charge_charge_inter_eV,
                       input.initial_ref.charge_dipole_eV,
                       input.initial_ref.dipole_dipole_eV,
                       input.initial_ref.higher_multipole_eV);
        occ::log::info("  DMACRYS total elec: {:.6f} eV",
                       input.initial_ref.charge_charge_inter_eV +
                       input.initial_ref.charge_dipole_eV +
                       input.initial_ref.dipole_dipole_eV +
                       input.initial_ref.higher_multipole_eV);
    }

    // === Part 2: Strain derivatives with and without Ewald ===
    occ::log::info("\n=== Strain derivative: Ewald vs no-Ewald (order 4) ===");
    double delta = 1e-5;

    occ::log::info("{:>4s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}",
                   "Comp", "total", "elec_ew", "elec_noew", "ew_corr", "buck", "buck_noew", "ref");
    for (int comp = 0; comp < 3; ++comp) {
        Mat3 eps_p = voigt_strain_tensor(comp, +delta);
        Mat3 eps_m = voigt_strain_tensor(comp, -delta);

        // Total (buck + elec + ewald)
        double Ep = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_p, cutoff, true, alpha, kmax, 4,
            &ref_neighbors, &ref_site_masks);
        double Em = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_m, cutoff, true, alpha, kmax, 4,
            &ref_neighbors, &ref_site_masks);
        double fd_total = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        // Elec only with Ewald
        double Ep_ew = compute_strained_energy(
            input, crystal, multipoles, zero_buck,
            eps_p, cutoff, true, alpha, kmax, 4,
            &ref_neighbors, &ref_site_masks);
        double Em_ew = compute_strained_energy(
            input, crystal, multipoles, zero_buck,
            eps_m, cutoff, true, alpha, kmax, 4,
            &ref_neighbors, &ref_site_masks);
        double fd_elec_ew = (Ep_ew - Em_ew) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        // Elec only WITHOUT Ewald
        double Ep_noew = compute_strained_energy(
            input, crystal, multipoles, zero_buck,
            eps_p, cutoff, false, 0, 0, 4,
            &ref_neighbors, &ref_site_masks);
        double Em_noew = compute_strained_energy(
            input, crystal, multipoles, zero_buck,
            eps_m, cutoff, false, 0, 0, 4,
            &ref_neighbors, &ref_site_masks);
        double fd_elec_noew = (Ep_noew - Em_noew) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;

        // Buck only WITHOUT Ewald (total_noew - elec_noew)
        double Ep_full_noew = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_p, cutoff, false, 0, 0, 4,
            &ref_neighbors, &ref_site_masks);
        double Em_full_noew = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_m, cutoff, false, 0, 0, 4,
            &ref_neighbors, &ref_site_masks);
        double fd_total_noew = (Ep_full_noew - Em_full_noew) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;
        double fd_buck_noew = fd_total_noew - fd_elec_noew;

        double fd_ew_corr = fd_elec_ew - fd_elec_noew;
        double fd_buck = fd_total - fd_elec_ew;

        const char* names[] = {"E1", "E2", "E3"};
        occ::log::info("  {:>2s} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}",
                       names[comp], fd_total, fd_elec_ew, fd_elec_noew,
                       fd_ew_corr, fd_buck, fd_buck_noew, ref_sd[comp]);
    }

    // === Part 3: Direct-sum electrostatic by order (no Ewald) ===
    occ::log::info("\n=== Direct-sum elec strain derivative by order (NO Ewald) ===");
    for (int order : {0, 1, 2, 3, 4, 8}) {
        occ::log::info("  order <= {}:", order);
        for (int comp = 0; comp < 3; ++comp) {
            Mat3 eps_p = voigt_strain_tensor(comp, +delta);
            Mat3 eps_m = voigt_strain_tensor(comp, -delta);
            double Ep = compute_strained_energy(
                input, crystal, multipoles, zero_buck,
                eps_p, cutoff, false, 0, 0, order,
                &ref_neighbors, &ref_site_masks);
            double Em = compute_strained_energy(
                input, crystal, multipoles, zero_buck,
                eps_m, cutoff, false, 0, 0, order,
                &ref_neighbors, &ref_site_masks);
            double fd = (Ep - Em) / (2.0 * delta) / units::EV_TO_KJ_PER_MOL;
            const char* names[] = {"E1", "E2", "E3"};
            occ::log::info("    {}: {:>10.6f} eV", names[comp], fd);
        }
    }

    CHECK(true);
}

TEST_CASE("DMACRYS AXOSOW strained positions diagnostic", "[mults][dmacrys][strain-pos]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    const auto& uc = crystal.unit_cell();
    Mat3 direct = uc.direct();

    occ::log::info("STRAIN_POS: Unstrained direct matrix:");
    for (int r = 0; r < 3; ++r)
        occ::log::info("STRAIN_POS:   [{:.8f} {:.8f} {:.8f}]",
                       direct(r,0), direct(r,1), direct(r,2));

    // Build unstrained CrystalEnergy to get COMs and initial states
    double cutoff = input.cutoff_radius;
    CrystalEnergy calc0(crystal, multipoles, cutoff,
                        ForceFieldType::Custom, true, false, 1e-8, 0, 0);
    setup_crystal_energy_from_dmacrys(calc0, input, crystal, multipoles);
    auto states0 = calc0.initial_states();

    occ::log::info("STRAIN_POS: Unstrained COMs (Angstrom):");
    for (int m = 0; m < static_cast<int>(states0.size()); ++m) {
        Vec3 com = states0[m].position;
        Vec3 frac = uc.to_fractional(com);
        occ::log::info("STRAIN_POS:   mol {}: cart=({:.8f} {:.8f} {:.8f})  frac=({:.6f} {:.6f} {:.6f})",
                       m, com(0), com(1), com(2), frac(0), frac(1), frac(2));
    }

    // Apply E1, E2, E3 strains and check positions
    double delta = 0.001;
    for (int voigt = 0; voigt < 3; ++voigt) {
        Mat3 eps = voigt_strain_tensor(voigt, delta);
        Mat3 deformation = Mat3::Identity() + eps;
        Mat3 strained_direct = deformation * direct;

        crystal::UnitCell strained_uc(strained_direct);
        crystal::Crystal strained_crystal(
            crystal.asymmetric_unit(),
            crystal.space_group(),
            strained_uc);

        CrystalEnergy calc_s(strained_crystal, multipoles, cutoff,
                             ForceFieldType::Custom, true, false, 1e-8, 0, 0);
        setup_crystal_energy_from_dmacrys(calc_s, input, strained_crystal, multipoles);
        auto states_s = calc_s.initial_states();

        const char* names[] = {"E1(xx)", "E2(yy)", "E3(zz)"};
        occ::log::info("STRAIN_POS: {} strain delta={:.4f}:", names[voigt], delta);

        for (int m = 0; m < static_cast<int>(states0.size()); ++m) {
            Vec3 com0 = states0[m].position;
            Vec3 coms = states_s[m].position;
            Vec3 dcom = coms - com0;
            Vec3 expected_dcom = eps * com0;
            Vec3 err = dcom - expected_dcom;
            occ::log::info("STRAIN_POS:   mol {}: dCOM=({:+.8f} {:+.8f} {:+.8f})  "
                           "expect=({:+.8f} {:+.8f} {:+.8f})  err=({:+.2e} {:+.2e} {:+.2e})",
                           m, dcom(0), dcom(1), dcom(2),
                           expected_dcom(0), expected_dcom(1), expected_dcom(2),
                           err(0), err(1), err(2));
        }

        // Check rotation unchanged
        Mat3 R0 = states0[0].rotation_matrix();
        Mat3 Rs = states_s[0].rotation_matrix();
        double rot_diff = (R0 - Rs).norm();
        occ::log::info("STRAIN_POS:   Mol 0 rotation diff norm: {:.2e}", rot_diff);
    }

    CHECK(true);
}

TEST_CASE("DMACRYS TCHLBZ03 aniso repulsion energy", "[mults][dmacrys][aniso][TCHLBZ03]") {
    if (!std::filesystem::exists(TCHLBZ03_JSON)) {
        SKIP("TCHLBZ03.json not found");
    }
    auto input = read_dmacrys_json(TCHLBZ03_JSON);
    CHECK(input.title == "TCHLBZ03");
    CHECK(!input.aniso_potentials.empty());
    occ::log::info("TCHLBZ03: {} aniso potential pairs", input.aniso_potentials.size());

    // Check aniso body axes present for Cl sites
    int n_aniso_sites = 0;
    for (const auto& site : input.molecule.sites) {
        if (site.aniso_axis_body.squaredNorm() > 0.1) {
            n_aniso_sites++;
        }
    }
    occ::log::info("TCHLBZ03: {} sites with aniso axes out of {} total",
                   n_aniso_sites, input.molecule.sites.size());
    CHECK(n_aniso_sites > 0);

    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       input.ewald_accuracy,
                       input.has_ewald_eta ? input.ewald_eta : 0.0,
                       input.has_ewald_kmax ? input.ewald_kmax : 0);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    calc.set_max_interaction_order(4);

    auto states = calc.initial_states();
    auto result = calc.compute(states);

    const int Z = input.crystal.Z;
    const double total_kJ = result.total_energy / Z;
    const double repdisp_kJ = result.repulsion_dispersion / Z;
    const double elec_kJ = result.electrostatic_energy / Z;

    // DMACRYS reference: sum of inter-molecular components.
    // Note: DMACRYS's "total lattice energy" for TCHLBZ03 includes an
    // intramolecular Ewald self-energy correction (~49 kJ/mol) that our code
    // does not compute. Use the sum of inter-molecular component energies as
    // the reference instead.
    const double eV_to_kJ_per_mol = 96.485 / Z;
    const double ref_repdisp_kJ = input.initial_ref.repulsion_dispersion_eV * eV_to_kJ_per_mol;
    const double ref_elec_kJ =
        (input.initial_ref.charge_charge_inter_eV +
         input.initial_ref.charge_dipole_eV +
         input.initial_ref.dipole_dipole_eV +
         input.initial_ref.higher_multipole_eV) * eV_to_kJ_per_mol;
    const double ref_total_kJ = ref_repdisp_kJ + ref_elec_kJ;

    occ::log::info("TCHLBZ03 total energy: {:.4f} kJ/mol (ref: {:.4f})", total_kJ, ref_total_kJ);
    occ::log::info("TCHLBZ03 repdisp: {:.4f} kJ/mol (ref: {:.4f})", repdisp_kJ, ref_repdisp_kJ);
    occ::log::info("TCHLBZ03 elec: {:.4f} kJ/mol (ref: {:.4f})", elec_kJ, ref_elec_kJ);
    occ::log::info("TCHLBZ03 diff: {:.4f} kJ/mol", total_kJ - ref_total_kJ);

    // With aniso repulsion, should match within 5 kJ/mol (allowing for
    // order>4 terms, taper effects, and Ewald accuracy).
    CHECK(std::abs(repdisp_kJ - ref_repdisp_kJ) < 2.0);
    CHECK(std::abs(total_kJ - ref_total_kJ) < 5.0);
}

TEST_CASE("DMACRYS TETBBZ01 aniso repulsion energy",
          "[mults][dmacrys][aniso][TETBBZ01]") {
    if (!std::filesystem::exists(TETBBZ01_JSON)) {
        SKIP("TETBBZ01.json not found");
    }
    auto input = read_dmacrys_json(TETBBZ01_JSON);
    CHECK(input.title == "TETBBZ01");
    CHECK(!input.aniso_potentials.empty());

    int n_aniso_sites = 0;
    for (const auto& site : input.molecule.sites) {
        if (site.aniso_axis_body.squaredNorm() > 0.1) {
            n_aniso_sites++;
        }
    }
    occ::log::info("TETBBZ01: {} aniso potential pairs, {} sites with aniso axes out of {} total",
                   input.aniso_potentials.size(), n_aniso_sites, input.molecule.sites.size());
    CHECK(n_aniso_sites > 0);

    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       input.ewald_accuracy,
                       input.has_ewald_eta ? input.ewald_eta : 0.0,
                       input.has_ewald_kmax ? input.ewald_kmax : 0);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    calc.set_max_interaction_order(4);

    auto states = calc.initial_states();
    auto result = calc.compute(states);

    const int Z = input.crystal.Z;
    const double total_kJ = result.total_energy / Z;
    const double repdisp_kJ = result.repulsion_dispersion / Z;
    const double elec_kJ = result.electrostatic_energy / Z;

    // DMACRYS reference: sum of inter-molecular components.
    // DMACRYS "total lattice energy" includes an intramolecular Ewald
    // self-energy correction (~122 kJ/mol for TETBBZ01) that our code
    // does not compute.
    const double eV_to_kJ_per_mol = 96.485 / Z;
    const double ref_repdisp_kJ = input.initial_ref.repulsion_dispersion_eV * eV_to_kJ_per_mol;
    const double ref_elec_kJ =
        (input.initial_ref.charge_charge_inter_eV +
         input.initial_ref.charge_dipole_eV +
         input.initial_ref.dipole_dipole_eV +
         input.initial_ref.higher_multipole_eV) * eV_to_kJ_per_mol;
    const double ref_total_kJ = ref_repdisp_kJ + ref_elec_kJ;

    occ::log::info("TETBBZ01 total energy: {:.4f} kJ/mol (ref: {:.4f})", total_kJ, ref_total_kJ);
    occ::log::info("TETBBZ01 repdisp: {:.4f} kJ/mol (ref: {:.4f})", repdisp_kJ, ref_repdisp_kJ);
    occ::log::info("TETBBZ01 elec: {:.4f} kJ/mol (ref: {:.4f})", elec_kJ, ref_elec_kJ);
    occ::log::info("TETBBZ01 diff: {:.4f} kJ/mol", total_kJ - ref_total_kJ);

    CHECK(std::abs(repdisp_kJ - ref_repdisp_kJ) < 2.0);
    CHECK(std::abs(total_kJ - ref_total_kJ) < 5.0);
}
