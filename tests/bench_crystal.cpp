#include <occ/mults/dmacrys_input.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_strain.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cartesian_force.h>
#include <occ/mults/multipole_coarsening.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <chrono>
#include <cstdio>
#include <map>

using namespace occ::mults;
using hrclock = std::chrono::high_resolution_clock;

double elapsed_ms(hrclock::time_point a, hrclock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main() {
    occ::log::set_log_level(occ::log::level::info);
    auto input = read_dmacrys_json(CMAKE_SOURCE_DIR "/tests/data/dmacrys/AXOSOW.json");
    auto crystal = build_crystal(input.crystal);
    auto multipoles = build_multipole_sources(input, crystal);
    auto buck_params = convert_buckingham_params(input.potentials);

    CrystalEnergy calc(crystal, multipoles, input.cutoff_radius,
                       ForceFieldType::Custom, true, true,
                       1e-8, 0.35, 8);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);
    for (const auto& [key, p] : buck_params)
        calc.set_buckingham_params(key.first, key.second, p);

    auto states = calc.initial_states();

    // ================================================================
    // 1. Full compute() with internal timing breakdown
    // ================================================================
    printf("=== Full compute() breakdown (1 call) ===\n");
    occ::log::set_log_level(occ::log::level::debug);
    auto r = calc.compute(states);
    occ::log::set_log_level(occ::log::level::info);
    printf("Energy: %.6f kJ/mol (elec=%.4f, SR=%.4f)\n\n",
           r.total_energy, r.electrostatic_energy, r.repulsion_dispersion);

    // ================================================================
    // 2. Prepare CartesianMolecules for isolated benchmarks
    // ================================================================
    const int N = static_cast<int>(states.size());
    for (int i = 0; i < N; ++i)
        multipoles[i].set_orientation(states[i].rotation_matrix(), states[i].position);

    std::vector<CartesianMolecule> cart_mols;
    cart_mols.reserve(N);
    for (int i = 0; i < N; ++i)
        cart_mols.push_back(multipoles[i].cartesian());

    printf("=== Site info ===\n");
    printf("Sites per molecule: %zu\n", cart_mols[0].sites.size());
    for (size_t s = 0; s < cart_mols[0].sites.size(); ++s)
        printf("  site %zu: rank=%d\n", s, cart_mols[0].sites[s].rank);

    const auto& neighbors = calc.neighbor_pairs();
    int n_elec = 0;
    for (const auto& pair : neighbors) {
        if (!calc.use_com_elec_gate() || pair.com_distance <= input.cutoff_radius)
            n_elec++;
    }
    printf("Neighbor pairs: %zu total, %d electrostatic\n", neighbors.size(), n_elec);

    // Order distribution
    std::map<int, int> order_hist;
    for (const auto& pair : neighbors) {
        if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
        for (const auto& sA : cart_mols[pair.mol_i].sites)
            for (const auto& sB : cart_mols[pair.mol_j].sites)
                order_hist[sA.rank + sB.rank]++;
    }
    printf("Site-pair order distribution:\n");
    int total_sp = 0;
    for (const auto& [order, count] : order_hist) {
        printf("  order %d: %d pairs\n", order, count);
        total_sp += count;
    }
    printf("  total: %d site-pairs\n\n", total_sp);

    // ================================================================
    // 3. Isolated electrostatic benchmarks
    // ================================================================
    printf("=== Electrostatic benchmarks (5 reps each) ===\n");
    constexpr int N_REP = 5;

    // Helper to get cell translation
    auto get_trans = [&](const NeighborPair& pair) -> occ::Vec3 {
        return crystal.unit_cell().to_cartesian(pair.cell_shift.cast<double>());
    };

    // 3a. T-tensor energy+force+torque (full order, no truncation)
    {
        auto t0 = hrclock::now();
        double e = 0.0;
        for (int rep = 0; rep < N_REP; ++rep) {
            for (const auto& pair : neighbors) {
                if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
                auto res = compute_molecule_forces_torques(
                    cart_mols[pair.mol_i], cart_mols[pair.mol_j], 0.0, -1, get_trans(pair));
                e += pair.weight * res.energy;
            }
        }
        auto t1 = hrclock::now();
        printf("force+torque (full order):   %7.2f ms/eval  E=%.4f kJ/mol\n",
               elapsed_ms(t0, t1) / N_REP, e / N_REP);
    }

    // 3b. T-tensor energy+force+torque (order<=4, DMACRYS truncation)
    {
        auto t0 = hrclock::now();
        double e = 0.0;
        for (int rep = 0; rep < N_REP; ++rep) {
            for (const auto& pair : neighbors) {
                if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
                auto res = compute_molecule_forces_torques(
                    cart_mols[pair.mol_i], cart_mols[pair.mol_j], 0.0, 4, get_trans(pair));
                e += pair.weight * res.energy;
            }
        }
        auto t1 = hrclock::now();
        printf("force+torque (order<=4):     %7.2f ms/eval  E=%.4f kJ/mol\n",
               elapsed_ms(t0, t1) / N_REP, e / N_REP);
    }

    // 3c. T-tensor energy-only (scalar, full order) via compute_molecule_interaction
    {
        auto t0 = hrclock::now();
        double e = 0.0;
        for (int rep = 0; rep < N_REP; ++rep) {
            for (const auto& pair : neighbors) {
                if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
                occ::Vec3 trans = get_trans(pair);
                CartesianMolecule molB = cart_mols[pair.mol_j];
                for (auto& s : molB.sites) s.position += trans;
                e += pair.weight * compute_molecule_interaction(cart_mols[pair.mol_i], molB);
            }
        }
        auto t1 = hrclock::now();
        printf("energy-only  (full order):   %7.2f ms/eval  E=%.4f kJ/mol\n",
               elapsed_ms(t0, t1) / N_REP, e / N_REP);
    }

    // 3d. T-tensor energy-only SIMD
    {
        auto t0 = hrclock::now();
        double e = 0.0;
        for (int rep = 0; rep < N_REP; ++rep) {
            for (const auto& pair : neighbors) {
                if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
                occ::Vec3 trans = get_trans(pair);
                CartesianMolecule molB = cart_mols[pair.mol_j];
                for (auto& s : molB.sites) s.position += trans;
                e += pair.weight * compute_molecule_interaction_simd(cart_mols[pair.mol_i], molB);
            }
        }
        auto t1 = hrclock::now();
        printf("energy-only  (SIMD batch):   %7.2f ms/eval  E=%.4f kJ/mol\n",
               elapsed_ms(t0, t1) / N_REP, e / N_REP);
    }

    // 3e. Pure T-tensor<9> construction cost (no contraction)
    {
        auto t0 = hrclock::now();
        volatile double sink = 0.0;
        for (int rep = 0; rep < N_REP; ++rep) {
            for (const auto& pair : neighbors) {
                if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
                occ::Vec3 trans = get_trans(pair);
                for (const auto& sA : cart_mols[pair.mol_i].sites) {
                    if (sA.rank < 0) continue;
                    for (const auto& sB : cart_mols[pair.mol_j].sites) {
                        if (sB.rank < 0) continue;
                        occ::Vec3 R = ((sB.position + trans) - sA.position) / occ::units::BOHR_TO_ANGSTROM;
                        InteractionTensor<9> T;
                        compute_interaction_tensor<9>(R[0], R[1], R[2], T);
                        sink += T.data[0];
                    }
                }
            }
        }
        auto t1 = hrclock::now();
        printf("T-tensor<9>  build only:     %7.2f ms/eval\n", elapsed_ms(t0, t1) / N_REP);
    }

    // 3f. Pure T-tensor<5> construction cost (order 4)
    {
        auto t0 = hrclock::now();
        volatile double sink = 0.0;
        for (int rep = 0; rep < N_REP; ++rep) {
            for (const auto& pair : neighbors) {
                if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
                occ::Vec3 trans = get_trans(pair);
                for (const auto& sA : cart_mols[pair.mol_i].sites) {
                    if (sA.rank < 0) continue;
                    for (const auto& sB : cart_mols[pair.mol_j].sites) {
                        if (sB.rank < 0) continue;
                        occ::Vec3 R = ((sB.position + trans) - sA.position) / occ::units::BOHR_TO_ANGSTROM;
                        InteractionTensor<5> T;
                        compute_interaction_tensor<5>(R[0], R[1], R[2], T);
                        sink += T.data[0];
                    }
                }
            }
        }
        auto t1 = hrclock::now();
        printf("T-tensor<5>  build only:     %7.2f ms/eval\n", elapsed_ms(t0, t1) / N_REP);
    }

    // 3g. Cartesian build cost (MultipoleSource::cartesian)
    {
        auto t0 = hrclock::now();
        for (int rep = 0; rep < N_REP; ++rep) {
            for (int i = 0; i < N; ++i) {
                multipoles[i].set_orientation(states[i].rotation_matrix(), states[i].position);
            }
            for (int i = 0; i < N; ++i) {
                auto cm = multipoles[i].cartesian();
            }
        }
        auto t1 = hrclock::now();
        printf("CartesianMol build (%dx%d):   %7.2f ms/eval\n",
               N_REP, N, elapsed_ms(t0, t1) / N_REP);
    }

    // ================================================================
    // 4. Coarsening accuracy & performance benchmark
    // ================================================================
    printf("\n=== Coarsening benchmark ===\n");

    // Pre-merge all molecules
    std::vector<CartesianMolecule> merged_mols;
    merged_mols.reserve(N);
    {
        auto t0 = hrclock::now();
        for (int i = 0; i < N; ++i)
            merged_mols.push_back(merge_to_single_site(cart_mols[i]));
        auto t1 = hrclock::now();
        printf("Merge %d molecules:           %7.3f ms\n", N, elapsed_ms(t0, t1));
    }

    for (int i = 0; i < N; ++i) {
        printf("  mol %d: %zu sites -> 1 site, merged rank=%d\n",
               i, cart_mols[i].sites.size(), merged_mols[i].sites[0].rank);
    }

    // COM distance histogram
    printf("\nCOM distance distribution:\n");
    int bins[16] = {};
    for (const auto& pair : neighbors) {
        int b = static_cast<int>(pair.com_distance);
        if (b < 16) bins[b]++;
    }
    for (int b = 0; b < 16; ++b) {
        if (bins[b] > 0)
            printf("  %2d-%2d Ang: %d pairs\n", b, b+1, bins[b]);
    }

    // Accuracy at different COM distance thresholds
    printf("\nAccuracy: exact vs merged by COM distance range\n");
    printf("%8s  %8s  %6s  %12s  %12s  %8s\n",
           "d_min", "d_max", "pairs", "E_exact", "E_merged", "rel_err");

    double distance_bins[] = {0, 4, 6, 8, 10, 12, 15};
    int n_bins = 6;

    for (int b = 0; b < n_bins; ++b) {
        double d_min = distance_bins[b];
        double d_max = distance_bins[b + 1];
        double e_exact = 0.0, e_merged = 0.0;
        int count = 0;

        for (const auto& pair : neighbors) {
            if (pair.com_distance < d_min || pair.com_distance >= d_max) continue;
            occ::Vec3 trans = get_trans(pair);

            // Exact (8x8 site pairs)
            CartesianMolecule molB_exact = cart_mols[pair.mol_j];
            for (auto& s : molB_exact.sites) s.position += trans;
            e_exact += pair.weight * compute_molecule_interaction(cart_mols[pair.mol_i], molB_exact);

            // Merged (1x1 site pair)
            CartesianMolecule molB_merged = merged_mols[pair.mol_j];
            for (auto& s : molB_merged.sites) s.position += trans;
            e_merged += pair.weight * compute_molecule_interaction(merged_mols[pair.mol_i], molB_merged);

            count++;
        }

        double rel = (std::abs(e_exact) > 1e-15) ?
            std::abs(e_merged - e_exact) / std::abs(e_exact) : 0.0;

        printf("%6.1f  %6.1f  %6d  %12.4f  %12.4f  %8.4f%%\n",
               d_min, d_max, count, e_exact, e_merged, rel * 100.0);
    }

    // Total energy comparison
    {
        double e_exact_total = 0.0, e_merged_total = 0.0;
        for (const auto& pair : neighbors) {
            if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
            occ::Vec3 trans = get_trans(pair);

            CartesianMolecule molB_exact = cart_mols[pair.mol_j];
            for (auto& s : molB_exact.sites) s.position += trans;
            e_exact_total += pair.weight * compute_molecule_interaction(cart_mols[pair.mol_i], molB_exact);

            CartesianMolecule molB_merged = merged_mols[pair.mol_j];
            for (auto& s : molB_merged.sites) s.position += trans;
            e_merged_total += pair.weight * compute_molecule_interaction(merged_mols[pair.mol_i], molB_merged);
        }

        double rel = std::abs(e_merged_total - e_exact_total) / std::abs(e_exact_total);
        printf("\nTotal: exact=%.4f  merged=%.4f  rel_err=%.4f%%\n",
               e_exact_total, e_merged_total, rel * 100.0);
    }

    // Hybrid: exact for close pairs, merged for distant
    printf("\nHybrid accuracy (exact for d<threshold, merged for d>=threshold):\n");
    printf("%8s  %12s  %8s  %8s  %8s\n",
           "thresh", "E_hybrid", "rel_err", "exact_pr", "merge_pr");

    for (double thresh : {6.0, 8.0, 10.0, 12.0}) {
        double e_hybrid = 0.0;
        double e_exact_ref = 0.0;
        int n_exact_pairs = 0, n_merged_pairs = 0;

        for (const auto& pair : neighbors) {
            if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
            occ::Vec3 trans = get_trans(pair);

            // Always compute exact for reference
            CartesianMolecule molB_exact = cart_mols[pair.mol_j];
            for (auto& s : molB_exact.sites) s.position += trans;
            e_exact_ref += pair.weight * compute_molecule_interaction(cart_mols[pair.mol_i], molB_exact);

            if (pair.com_distance < thresh) {
                // Close: use exact
                e_hybrid += pair.weight * compute_molecule_interaction(cart_mols[pair.mol_i], molB_exact);
                n_exact_pairs++;
            } else {
                // Far: use merged
                CartesianMolecule molB_merged = merged_mols[pair.mol_j];
                for (auto& s : molB_merged.sites) s.position += trans;
                e_hybrid += pair.weight * compute_molecule_interaction(merged_mols[pair.mol_i], molB_merged);
                n_merged_pairs++;
            }
        }

        double rel = std::abs(e_hybrid - e_exact_ref) / std::abs(e_exact_ref);
        printf("%6.1f  %12.4f  %7.4f%%  %6d   %6d\n",
               thresh, e_hybrid, rel * 100.0, n_exact_pairs, n_merged_pairs);
    }

    // Performance: exact vs all-merged vs hybrid
    printf("\nPerformance (10 reps):\n");
    constexpr int N_PERF = 10;

    {
        auto t0 = hrclock::now();
        volatile double sink = 0.0;
        for (int rep = 0; rep < N_PERF; ++rep) {
            double e = 0.0;
            for (const auto& pair : neighbors) {
                if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
                occ::Vec3 trans = get_trans(pair);
                CartesianMolecule molB = cart_mols[pair.mol_j];
                for (auto& s : molB.sites) s.position += trans;
                e += pair.weight * compute_molecule_interaction(cart_mols[pair.mol_i], molB);
            }
            sink = e;
        }
        auto t1 = hrclock::now();
        printf("  All exact (8x8):           %7.2f ms/eval\n", elapsed_ms(t0, t1) / N_PERF);
    }

    {
        auto t0 = hrclock::now();
        volatile double sink = 0.0;
        for (int rep = 0; rep < N_PERF; ++rep) {
            double e = 0.0;
            for (const auto& pair : neighbors) {
                if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
                occ::Vec3 trans = get_trans(pair);
                CartesianMolecule molB = merged_mols[pair.mol_j];
                for (auto& s : molB.sites) s.position += trans;
                e += pair.weight * compute_molecule_interaction(merged_mols[pair.mol_i], molB);
            }
            sink = e;
        }
        auto t1 = hrclock::now();
        printf("  All merged (1x1):          %7.2f ms/eval\n", elapsed_ms(t0, t1) / N_PERF);
    }

    for (double thresh : {8.0, 10.0, 12.0}) {
        auto t0 = hrclock::now();
        volatile double sink = 0.0;
        for (int rep = 0; rep < N_PERF; ++rep) {
            double e = 0.0;
            for (const auto& pair : neighbors) {
                if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
                occ::Vec3 trans = get_trans(pair);
                if (pair.com_distance < thresh) {
                    CartesianMolecule molB = cart_mols[pair.mol_j];
                    for (auto& s : molB.sites) s.position += trans;
                    e += pair.weight * compute_molecule_interaction(cart_mols[pair.mol_i], molB);
                } else {
                    CartesianMolecule molB = merged_mols[pair.mol_j];
                    for (auto& s : molB.sites) s.position += trans;
                    e += pair.weight * compute_molecule_interaction(merged_mols[pair.mol_i], molB);
                }
            }
            sink = e;
        }
        auto t1 = hrclock::now();
        printf("  Hybrid (exact<%.0f, merge>=%.0f): %7.2f ms/eval\n",
               thresh, thresh, elapsed_ms(t0, t1) / N_PERF);
    }

    // ================================================================
    // 5. Force accuracy: exact vs merged translational forces
    // ================================================================
    printf("\n=== Force accuracy (translational) ===\n");

    // Accumulate total translational force on each molecule from all neighbors.
    // "Exact" uses the full 8-site molecules; "merged" uses 1-site.
    std::vector<occ::Vec3> force_exact(N, occ::Vec3::Zero());
    std::vector<occ::Vec3> force_merged(N, occ::Vec3::Zero());

    for (const auto& pair : neighbors) {
        if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
        occ::Vec3 trans = get_trans(pair);

        // Exact forces via compute_molecule_forces
        {
            CartesianMolecule molB = cart_mols[pair.mol_j];
            for (auto& s : molB.sites) s.position += trans;
            auto res = compute_molecule_forces(cart_mols[pair.mol_i], molB);
            occ::Vec3 fA_total = occ::Vec3::Zero();
            occ::Vec3 fB_total = occ::Vec3::Zero();
            for (const auto& f : res.forces_A) fA_total += f;
            for (const auto& f : res.forces_B) fB_total += f;
            force_exact[pair.mol_i] += pair.weight * fA_total;
            force_exact[pair.mol_j] += pair.weight * fB_total;
        }

        // Merged forces
        {
            CartesianMolecule molB = merged_mols[pair.mol_j];
            for (auto& s : molB.sites) s.position += trans;
            auto res = compute_molecule_forces(merged_mols[pair.mol_i], molB);
            force_merged[pair.mol_i] += pair.weight * res.forces_A[0];
            force_merged[pair.mol_j] += pair.weight * res.forces_B[0];
        }
    }

    printf("\nPer-molecule total force comparison (kJ/mol/Ang):\n");
    printf("%4s  %12s  %12s  %12s  %12s  %12s  %12s  %8s\n",
           "mol", "Fx_exact", "Fy_exact", "Fz_exact",
           "Fx_merged", "Fy_merged", "Fz_merged", "rel_err");

    double max_force_err = 0.0;
    double rms_force_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double f_exact_norm = force_exact[i].norm();
        double diff_norm = (force_merged[i] - force_exact[i]).norm();
        double rel = (f_exact_norm > 1e-10) ? diff_norm / f_exact_norm : diff_norm;
        max_force_err = std::max(max_force_err, rel);
        rms_force_err += rel * rel;
        printf("%4d  %12.4f  %12.4f  %12.4f  %12.4f  %12.4f  %12.4f  %7.3f%%\n",
               i,
               force_exact[i][0], force_exact[i][1], force_exact[i][2],
               force_merged[i][0], force_merged[i][1], force_merged[i][2],
               rel * 100.0);
    }
    rms_force_err = std::sqrt(rms_force_err / N);
    printf("Max relative error: %.4f%%  RMS relative error: %.4f%%\n\n",
           max_force_err * 100.0, rms_force_err * 100.0);

    // Hybrid force accuracy at different thresholds
    printf("Hybrid force accuracy (exact for d<threshold, merged for d>=threshold):\n");
    printf("%8s  %8s  %8s\n", "thresh", "max_err", "rms_err");

    for (double thresh : {6.0, 8.0, 10.0, 12.0}) {
        std::vector<occ::Vec3> force_hybrid(N, occ::Vec3::Zero());

        for (const auto& pair : neighbors) {
            if (calc.use_com_elec_gate() && pair.com_distance > input.cutoff_radius) continue;
            occ::Vec3 trans = get_trans(pair);

            if (pair.com_distance < thresh) {
                CartesianMolecule molB = cart_mols[pair.mol_j];
                for (auto& s : molB.sites) s.position += trans;
                auto res = compute_molecule_forces(cart_mols[pair.mol_i], molB);
                occ::Vec3 fA = occ::Vec3::Zero(), fB = occ::Vec3::Zero();
                for (const auto& f : res.forces_A) fA += f;
                for (const auto& f : res.forces_B) fB += f;
                force_hybrid[pair.mol_i] += pair.weight * fA;
                force_hybrid[pair.mol_j] += pair.weight * fB;
            } else {
                CartesianMolecule molB = merged_mols[pair.mol_j];
                for (auto& s : molB.sites) s.position += trans;
                auto res = compute_molecule_forces(merged_mols[pair.mol_i], molB);
                force_hybrid[pair.mol_i] += pair.weight * res.forces_A[0];
                force_hybrid[pair.mol_j] += pair.weight * res.forces_B[0];
            }
        }

        double max_err = 0.0, sum_sq = 0.0;
        for (int i = 0; i < N; ++i) {
            double f_norm = force_exact[i].norm();
            double diff = (force_hybrid[i] - force_exact[i]).norm();
            double rel = (f_norm > 1e-10) ? diff / f_norm : diff;
            max_err = std::max(max_err, rel);
            sum_sq += rel * rel;
        }
        double rms = std::sqrt(sum_sq / N);
        printf("%6.1f   %7.3f%%  %7.3f%%\n", thresh, max_err * 100.0, rms * 100.0);
    }

    // Force by distance bin
    printf("\nForce accuracy by COM distance bin:\n");
    printf("%8s  %8s  %6s  %12s  %12s  %8s\n",
           "d_min", "d_max", "pairs", "|F_exact|", "|F_err|", "rel_err");

    for (int b = 0; b < n_bins; ++b) {
        double d_min = distance_bins[b];
        double d_max = distance_bins[b + 1];
        double sum_f_exact = 0.0, sum_f_err = 0.0;
        int count = 0;

        for (const auto& pair : neighbors) {
            if (pair.com_distance < d_min || pair.com_distance >= d_max) continue;
            occ::Vec3 trans = get_trans(pair);

            // Exact
            CartesianMolecule molB_exact = cart_mols[pair.mol_j];
            for (auto& s : molB_exact.sites) s.position += trans;
            auto res_exact = compute_molecule_forces(cart_mols[pair.mol_i], molB_exact);
            occ::Vec3 fA_exact = occ::Vec3::Zero();
            for (const auto& f : res_exact.forces_A) fA_exact += f;

            // Merged
            CartesianMolecule molB_merged = merged_mols[pair.mol_j];
            for (auto& s : molB_merged.sites) s.position += trans;
            auto res_merged = compute_molecule_forces(merged_mols[pair.mol_i], molB_merged);
            occ::Vec3 fA_merged = res_merged.forces_A[0];

            sum_f_exact += pair.weight * fA_exact.norm();
            sum_f_err += pair.weight * (fA_merged - fA_exact).norm();
            count++;
        }

        double rel = (sum_f_exact > 1e-15) ? sum_f_err / sum_f_exact : 0.0;
        printf("%6.1f  %6.1f  %6d  %12.4f  %12.4f  %7.3f%%\n",
               d_min, d_max, count, sum_f_exact, sum_f_err, rel * 100.0);
    }

    return 0;
}
