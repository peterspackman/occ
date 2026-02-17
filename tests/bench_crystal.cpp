#include <occ/mults/dmacrys_input.h>
#include <occ/mults/crystal_energy.h>
#include <occ/mults/crystal_strain.h>
#include <occ/mults/cartesian_molecule.h>
#include <occ/mults/cartesian_force.h>
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

    return 0;
}
