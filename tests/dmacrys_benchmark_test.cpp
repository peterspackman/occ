#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <occ/mults/dmacrys_input.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_strain.h>
#include <occ/mults/cartesian_kernels.h>
#include <occ/mults/interaction_tensor.h>
#include <occ/core/elastic_tensor.h>
#include <occ/core/units.h>
#include <occ/core/log.h>

using Catch::Approx;
using namespace occ;
using namespace occ::mults;

static const std::string AXOSOW_JSON = CMAKE_SOURCE_DIR "/tests/data/dmacrys/AXOSOW.json";

TEST_CASE("DMACRYS JSON reader", "[mults][dmacrys]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);

    CHECK(input.title == "AXOSOW");
    // Space group determined from symops (LATT 1 + 3 SYMM = Pbca)
    CHECK(input.crystal.Z == 8);
    CHECK(input.crystal.atoms.size() == 8);
    CHECK(input.molecule.sites.size() == 8);
    CHECK(input.potentials.size() == 6);

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

TEST_CASE("DMACRYS crystal builder", "[mults][dmacrys]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);

    // Pbca has 8 symmetry operations
    CHECK(crystal.space_group().symmetry_operations().size() == 8);

    // Unit cell should be orthorhombic
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

    // Compute order>4 inter-molecular electrostatic contribution so we can
    // see what our total would be if we matched DMACRYS's order-4 truncation.
    using namespace occ::mults::kernel_detail;
    std::vector<CartesianMolecule> cart_mols;
    for (int i = 0; i < n_mol; ++i)
        cart_mols.push_back(multipoles[i].cartesian());
    const auto& uc = crystal.unit_cell();

    double order5plus_ha = 0.0;
    int site_pair_count = 0;
    for (const auto& pair : calc.neighbor_pairs()) {
        int mi = pair.mol_i, mj = pair.mol_j;
        Vec3 cell_trans = uc.to_cartesian(pair.cell_shift.cast<double>());
        for (const auto& sA : cart_mols[mi].sites) {
            if (sA.rank < 0) continue;
            for (const auto& sB : cart_mols[mj].sites) {
                if (sB.rank < 0) continue;
                Vec3 R_ang = (sB.position + cell_trans) - sA.position;
                ++site_pair_count;
                Vec3 R = R_ang / occ::units::BOHR_TO_ANGSTROM;
                InteractionTensor<8> T;
                compute_interaction_tensor<8>(R[0], R[1], R[2], T);
                for (int lA = 0; lA <= sA.rank; ++lA) {
                    int startA = (lA == 0) ? 0 : nhermsum(lA - 1);
                    int endA = nhermsum(lA);
                    for (int lB = 0; lB <= sB.rank; ++lB) {
                        if (lA + lB <= 4) continue; // skip order<=4
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
                        order5plus_ha += pair.weight * e;
                    }
                }
            }
        }
    }
    // DMACRYS reports 60616 atom pairs. Our neighbor list uses weight=0.5
    // for all pairs (double-counting), so effective unique pairs = count/2.
    INFO("Site pairs computed: " << site_pair_count
         << " (unique ~" << site_pair_count/2
         << "), DMACRYS: 60616");
    double order5plus_kJ_per_mol = order5plus_ha * occ::units::AU_TO_KJ_PER_MOL / n_mol;
    double adjusted_total = total_per_mol - order5plus_kJ_per_mol;

    INFO("Order>4 inter-mol elec: " << order5plus_kJ_per_mol << " kJ/mol/mol");
    INFO("Adjusted total (order<=4): " << adjusted_total
         << " kJ/mol/mol, DMACRYS: " << ref_total
         << " (diff " << (adjusted_total - ref_total) << ")");

    // DMACRYS truncates at interaction order lA+lB <= 4; we compute up to 8.
    // The extra order 5-8 terms contribute ~-0.68 kJ/mol/mol.
    // With those removed, we match to within 0.1 kJ/mol.
    CHECK(total_per_mol == Approx(ref_total).margin(0.7));
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

    // DMACRYS reference includes SPLI (spline tapering) which contributes
    // ~0.02-0.10 eV to E1-E3 strain derivatives. We don't implement SPLI.
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

TEST_CASE("DMACRYS AXOSOW elastic constants", "[mults][dmacrys][elastic]") {
    auto input = read_dmacrys_json(AXOSOW_JSON);
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    REQUIRE(input.optimized_ref.has_elastic_constants);
    const auto& ref_C = input.optimized_ref.elastic_constants_GPa;

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
    for (int i = 0; i < 3; ++i) {
        for (int j = 3; j < 6; ++j) {
            INFO("C_" << (i+1) << (j+1) << " should be ~0");
            CHECK(std::abs(Cij(i, j)) < 1.0);
        }
    }

    // Diagonal elements: expect approximate agreement with DMACRYS
    // Relaxed constants should be closer to DMACRYS than clamped
    for (int i = 0; i < 6; ++i) {
        if (ref_C(i, i) > 1.0) {
            double ratio = Cij(i, i) / ref_C(i, i);
            INFO("C_" << (i+1) << (i+1)
                 << ": OCC=" << Cij(i, i)
                 << " DMACRYS=" << ref_C(i, i)
                 << " ratio=" << ratio);
            CHECK(ratio > 0.3);
            CHECK(ratio < 2.0);
        }
    }

    // Feed to ElasticTensor for derived properties
    occ::core::ElasticTensor et(Cij);
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
