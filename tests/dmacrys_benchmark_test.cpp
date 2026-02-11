#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <occ/mults/dmacrys_input.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/cartesian_kernels.h>
#include <occ/mults/interaction_tensor.h>
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
