#include <occ/mults/crystal_strain.h>
#include <occ/core/log.h>
#include <occ/core/units.h>

namespace occ::mults {

Mat3 voigt_strain_tensor(int voigt_index, double magnitude) {
    Mat3 eps = Mat3::Zero();
    switch (voigt_index) {
    case 0: // E_1 = eps_11
        eps(0, 0) = magnitude;
        break;
    case 1: // E_2 = eps_22
        eps(1, 1) = magnitude;
        break;
    case 2: // E_3 = eps_33
        eps(2, 2) = magnitude;
        break;
    case 3: // E_4 = 2*eps_23
        eps(1, 2) = magnitude / 2.0;
        eps(2, 1) = magnitude / 2.0;
        break;
    case 4: // E_5 = 2*eps_13
        eps(0, 2) = magnitude / 2.0;
        eps(2, 0) = magnitude / 2.0;
        break;
    case 5: // E_6 = 2*eps_12
        eps(0, 1) = magnitude / 2.0;
        eps(1, 0) = magnitude / 2.0;
        break;
    default:
        throw std::runtime_error("Invalid Voigt index: " +
                                 std::to_string(voigt_index));
    }
    return eps;
}

/// Reference data from unstrained crystal: neighbor list + atom-pair masks.
struct ReferenceNeighborData {
    std::vector<NeighborPair> neighbors;
    std::vector<std::vector<bool>> site_masks;
};

// Helper: build a CrystalEnergy for the unstrained crystal and return its
// neighbor list AND frozen Buckingham site-pair masks.
static ReferenceNeighborData build_reference_neighbor_data(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    double cutoff,
    bool use_ewald, double alpha, int kmax) {

    CrystalEnergy calc(crystal, multipoles, cutoff,
                       ForceFieldType::Custom, true, use_ewald,
                       1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);

    auto states = calc.initial_states();
    ReferenceNeighborData ref;
    ref.neighbors = calc.neighbor_list();
    ref.site_masks = calc.compute_buckingham_site_masks(states);
    return ref;
}

// Helper: build a reusable CrystalEnergy from reference data.
static CrystalEnergy build_reusable_calc(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    int max_interaction_order,
    const std::vector<NeighborPair>& neighbors,
    const std::vector<std::vector<bool>>& site_masks) {

    CrystalEnergy calc(crystal, multipoles, cutoff,
                       ForceFieldType::Custom, true, use_ewald,
                       1e-8, alpha, kmax);

    // Skip building neighbor list since we set it directly
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles, false);
    calc.set_neighbor_list(neighbors);
    calc.set_fixed_site_masks(site_masks);

    if (max_interaction_order >= 0) {
        calc.set_max_interaction_order(max_interaction_order);
    }

    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }

    return std::move(calc);
}

/// Evaluate a strained crystal by updating a reusable CrystalEnergy in place.
/// Returns the CrystalEnergyResult directly.
static CrystalEnergyResult evaluate_strained(
    CrystalEnergy& calc,
    const DmacrysInput& input,
    const crystal::Crystal& ref_crystal,
    const std::vector<MultipoleSource>& multipoles,
    const Mat3& strain) {

    const auto& uc = ref_crystal.unit_cell();
    Mat3 direct = uc.direct();

    // Deform: direct' = (I + eps) * direct
    Mat3 deformation = Mat3::Identity() + strain;
    Mat3 strained_direct = deformation * direct;

    crystal::UnitCell strained_uc(strained_direct);
    crystal::Crystal strained_crystal(
        ref_crystal.asymmetric_unit(),
        ref_crystal.space_group(),
        strained_uc);

    auto states = compute_molecule_states(input, strained_crystal, multipoles);
    calc.update_lattice(strained_crystal, states);
    return calc.compute(states);
}

StrainedResult compute_strained(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    const Mat3& strain,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    int max_interaction_order,
    const std::vector<NeighborPair>* fixed_neighbors,
    const std::vector<std::vector<bool>>* fixed_site_masks) {

    // When called with fixed neighbors, build a reusable calc
    if (fixed_neighbors && fixed_site_masks) {
        auto calc = build_reusable_calc(
            input, crystal, multipoles, buck_params,
            cutoff, use_ewald, alpha, kmax,
            max_interaction_order, *fixed_neighbors, *fixed_site_masks);

        auto result = evaluate_strained(calc, input, crystal, multipoles, strain);

        StrainedResult out;
        out.energy = result.total_energy;
        const int N = static_cast<int>(result.forces.size());
        out.gradient.resize(6 * N);
        for (int i = 0; i < N; ++i) {
            out.gradient.segment<3>(6 * i) = -result.forces[i];
            out.gradient.segment<3>(6 * i + 3) = result.torques[i];
        }
        return out;
    }

    // Fallback: build fresh CrystalEnergy for the strained geometry
    const auto& uc = crystal.unit_cell();
    Mat3 direct = uc.direct();
    Mat3 deformation = Mat3::Identity() + strain;
    Mat3 strained_direct = deformation * direct;

    crystal::UnitCell strained_uc(strained_direct);
    crystal::Crystal strained_crystal(
        crystal.asymmetric_unit(),
        crystal.space_group(),
        strained_uc);

    CrystalEnergy calc(strained_crystal, multipoles, cutoff,
                       ForceFieldType::Custom, true, use_ewald,
                       1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(calc, input, strained_crystal, multipoles);

    if (max_interaction_order >= 0) {
        calc.set_max_interaction_order(max_interaction_order);
    }
    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }

    auto states = calc.initial_states();
    auto result = calc.compute(states);

    StrainedResult out;
    out.energy = result.total_energy;
    const int N = static_cast<int>(result.forces.size());
    out.gradient.resize(6 * N);
    for (int i = 0; i < N; ++i) {
        out.gradient.segment<3>(6 * i) = -result.forces[i];
        out.gradient.segment<3>(6 * i + 3) = result.torques[i];
    }
    return out;
}

Vec6 compute_strain_derivatives_fd(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    double delta,
    int max_interaction_order) {

    auto ref = build_reference_neighbor_data(
        input, crystal, multipoles, cutoff, use_ewald, alpha, kmax);

    // Build one CrystalEnergy and reuse for all evaluations
    auto calc = build_reusable_calc(
        input, crystal, multipoles, buck_params,
        cutoff, use_ewald, alpha, kmax,
        max_interaction_order, ref.neighbors, ref.site_masks);

    Vec6 dU_dE;

    for (int i = 0; i < 6; ++i) {
        Mat3 eps_plus = voigt_strain_tensor(i, +delta);
        auto res_plus = evaluate_strained(calc, input, crystal, multipoles, eps_plus);

        Mat3 eps_minus = voigt_strain_tensor(i, -delta);
        auto res_minus = evaluate_strained(calc, input, crystal, multipoles, eps_minus);

        double dU_kJ = (res_plus.total_energy - res_minus.total_energy) / (2.0 * delta);
        dU_dE(i) = dU_kJ / units::EV_TO_KJ_PER_MOL;

        occ::log::info("Strain derivative E_{}: E(+d)={:.10f} E(-d)={:.10f} "
                        "dU/dE={:.6f} eV",
                        i + 1, res_plus.total_energy, res_minus.total_energy, dU_dE(i));
    }

    return dU_dE;
}

Mat6 compute_elastic_constants_fd(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    double delta,
    int max_interaction_order) {

    auto ref = build_reference_neighbor_data(
        input, crystal, multipoles, cutoff, use_ewald, alpha, kmax);

    auto calc = build_reusable_calc(
        input, crystal, multipoles, buck_params,
        cutoff, use_ewald, alpha, kmax,
        max_interaction_order, ref.neighbors, ref.site_masks);

    Mat3 zero_strain = Mat3::Zero();
    auto res0 = evaluate_strained(calc, input, crystal, multipoles, zero_strain);
    double E0 = res0.total_energy;

    double V = crystal.unit_cell().volume();

    Mat6 Cij = Mat6::Zero();

    std::array<double, 6> E_plus{}, E_minus{};

    for (int i = 0; i < 6; ++i) {
        Mat3 eps_p = voigt_strain_tensor(i, +delta);
        Mat3 eps_m = voigt_strain_tensor(i, -delta);

        E_plus[i] = evaluate_strained(calc, input, crystal, multipoles, eps_p).total_energy;
        E_minus[i] = evaluate_strained(calc, input, crystal, multipoles, eps_m).total_energy;

        double d2U = (E_plus[i] - 2.0 * E0 + E_minus[i]) / (delta * delta);
        Cij(i, i) = (d2U / V) * units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

        occ::log::debug("C_{}{}: E(+d)={:.8f} E0={:.8f} E(-d)={:.8f} "
                         "d2U={:.6f} C={:.2f} GPa",
                         i + 1, i + 1,
                         E_plus[i], E0, E_minus[i], d2U, Cij(i, i));
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            Mat3 eps_pp = voigt_strain_tensor(i, +delta) +
                          voigt_strain_tensor(j, +delta);
            Mat3 eps_pm = voigt_strain_tensor(i, +delta) +
                          voigt_strain_tensor(j, -delta);
            Mat3 eps_mp = voigt_strain_tensor(i, -delta) +
                          voigt_strain_tensor(j, +delta);
            Mat3 eps_mm = voigt_strain_tensor(i, -delta) +
                          voigt_strain_tensor(j, -delta);

            double E_pp = evaluate_strained(calc, input, crystal, multipoles, eps_pp).total_energy;
            double E_pm = evaluate_strained(calc, input, crystal, multipoles, eps_pm).total_energy;
            double E_mp = evaluate_strained(calc, input, crystal, multipoles, eps_mp).total_energy;
            double E_mm = evaluate_strained(calc, input, crystal, multipoles, eps_mm).total_energy;

            double d2U = (E_pp - E_pm - E_mp + E_mm) / (4.0 * delta * delta);
            double c_val = (d2U / V) * units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

            Cij(i, j) = c_val;
            Cij(j, i) = c_val;

            occ::log::debug("C_{}{}: E(++)={:.8f} E(+-)={:.8f} "
                             "E(-+)={:.8f} E(--)={:.8f} C={:.2f} GPa",
                             i + 1, j + 1, E_pp, E_pm, E_mp, E_mm, c_val);
        }
    }

    return Cij;
}

Mat6 compute_relaxed_elastic_constants_fd(
    const DmacrysInput& input,
    const crystal::Crystal& crystal,
    const std::vector<MultipoleSource>& multipoles,
    const std::map<std::pair<int, int>, BuckinghamParams>& buck_params,
    double cutoff,
    bool use_ewald, double alpha, int kmax,
    double delta,
    int max_interaction_order) {

    // ========================================================================
    // Step 1: Build reference data and reusable CrystalEnergy
    // ========================================================================
    auto ref = build_reference_neighbor_data(
        input, crystal, multipoles, cutoff, use_ewald, alpha, kmax);

    auto calc = build_reusable_calc(
        input, crystal, multipoles, buck_params,
        cutoff, use_ewald, alpha, kmax,
        max_interaction_order, ref.neighbors, ref.site_masks);

    auto states = calc.initial_states();
    const int N = static_cast<int>(states.size());

    // ========================================================================
    // Step 2: Compute W_ee (6x6) -- strain-strain second derivatives in kJ/mol
    // ========================================================================
    Mat3 zero_strain = Mat3::Zero();
    double E0 = evaluate_strained(calc, input, crystal, multipoles, zero_strain).total_energy;

    Mat6 W_ee = Mat6::Zero();

    std::array<double, 6> E_plus{}, E_minus{};
    for (int i = 0; i < 6; ++i) {
        Mat3 eps_p = voigt_strain_tensor(i, +delta);
        Mat3 eps_m = voigt_strain_tensor(i, -delta);

        E_plus[i] = evaluate_strained(calc, input, crystal, multipoles, eps_p).total_energy;
        E_minus[i] = evaluate_strained(calc, input, crystal, multipoles, eps_m).total_energy;

        W_ee(i, i) = (E_plus[i] - 2.0 * E0 + E_minus[i]) / (delta * delta);
    }

    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            Mat3 eps_pp = voigt_strain_tensor(i, +delta) +
                          voigt_strain_tensor(j, +delta);
            Mat3 eps_pm = voigt_strain_tensor(i, +delta) +
                          voigt_strain_tensor(j, -delta);
            Mat3 eps_mp = voigt_strain_tensor(i, -delta) +
                          voigt_strain_tensor(j, +delta);
            Mat3 eps_mm = voigt_strain_tensor(i, -delta) +
                          voigt_strain_tensor(j, -delta);

            double E_pp = evaluate_strained(calc, input, crystal, multipoles, eps_pp).total_energy;
            double E_pm = evaluate_strained(calc, input, crystal, multipoles, eps_pm).total_energy;
            double E_mp = evaluate_strained(calc, input, crystal, multipoles, eps_mp).total_energy;
            double E_mm = evaluate_strained(calc, input, crystal, multipoles, eps_mm).total_energy;

            double d2U = (E_pp - E_pm - E_mp + E_mm) / (4.0 * delta * delta);
            W_ee(i, j) = d2U;
            W_ee(j, i) = d2U;
        }
    }

    occ::log::info("W_ee (strain-strain, kJ/mol):");
    for (int i = 0; i < 6; ++i) {
        occ::log::info("  {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}",
                       W_ee(i, 0), W_ee(i, 1), W_ee(i, 2),
                       W_ee(i, 3), W_ee(i, 4), W_ee(i, 5));
    }

    // ========================================================================
    // Step 3: Compute W_ii -- internal DOF Hessian
    // ========================================================================
    // Restore unstrained lattice for Hessian computation
    auto unstrained_states = compute_molecule_states(input, crystal, multipoles);
    calc.update_lattice(crystal, unstrained_states);
    auto hess_result = calc.compute_with_hessian(unstrained_states);

    // Remove 3 translational DOF of molecule 0 (zero modes)
    Mat W_ii = hess_result.pack_hessian(true, false);
    int ndof_reduced = W_ii.rows();

    occ::log::info("W_ii: {}x{} (removed 3 translational DOF of mol 0)",
                   ndof_reduced, ndof_reduced);

    // Check reference gradient magnitude
    {
        auto ref_grad = hess_result.pack_gradient();
        occ::log::info("Reference gradient norm: {:.6f} (should be ~0 at minimum)",
                       ref_grad.norm());
    }

    // ========================================================================
    // Step 4: Compute W_ei (6 x ndof_reduced) -- strain-internal coupling
    // ========================================================================
    std::vector<int> reduced_to_full;
    reduced_to_full.reserve(ndof_reduced);
    // Molecule 0: skip translations (0,1,2), keep rotations (3,4,5)
    reduced_to_full.push_back(3);
    reduced_to_full.push_back(4);
    reduced_to_full.push_back(5);
    // Molecules 1..N-1: all 6 DOF
    for (int i = 1; i < N; ++i) {
        for (int c = 0; c < 6; ++c) {
            reduced_to_full.push_back(6 * i + c);
        }
    }

    Mat W_ei = Mat::Zero(6, ndof_reduced);

    for (int i = 0; i < 6; ++i) {
        Mat3 eps_plus = voigt_strain_tensor(i, +delta);
        Mat3 eps_minus = voigt_strain_tensor(i, -delta);

        auto res_plus = evaluate_strained(calc, input, crystal, multipoles, eps_plus);
        auto res_minus = evaluate_strained(calc, input, crystal, multipoles, eps_minus);

        // Pack as gradient: [-force, +torque]
        int nf = static_cast<int>(res_plus.forces.size());
        Vec g_plus(6 * nf), g_minus(6 * nf);
        for (int k = 0; k < nf; ++k) {
            g_plus.segment<3>(6 * k) = -res_plus.forces[k];
            g_plus.segment<3>(6 * k + 3) = res_plus.torques[k];
            g_minus.segment<3>(6 * k) = -res_minus.forces[k];
            g_minus.segment<3>(6 * k + 3) = res_minus.torques[k];
        }

        Vec dg = (g_plus - g_minus) / (2.0 * delta);

        for (int k = 0; k < ndof_reduced; ++k) {
            W_ei(i, k) = dg(reduced_to_full[k]);
        }
    }

    occ::log::info("W_ei norm: {:.6f}", W_ei.norm());

    // ========================================================================
    // Step 5: Schur complement
    // ========================================================================
    Mat X = W_ii.ldlt().solve(W_ei.transpose());
    Mat6 correction = W_ei * X;

    Mat6 W_relaxed = W_ee - correction;

    occ::log::info("Correction (W_ei * W_ii^-1 * W_ie) diagonal:");
    occ::log::info("  {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                   correction(0, 0), correction(1, 1), correction(2, 2),
                   correction(3, 3), correction(4, 4), correction(5, 5));

    // ========================================================================
    // Step 6: Convert to GPa
    // ========================================================================
    double V = crystal.unit_cell().volume();
    Mat6 Cij = (W_relaxed / V) * units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

    Mat6 Cij_clamped = (W_ee / V) * units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

    occ::log::info("\nClamped elastic constants (GPa):");
    for (int i = 0; i < 6; ++i) {
        occ::log::info("  {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}",
                       Cij_clamped(i, 0), Cij_clamped(i, 1), Cij_clamped(i, 2),
                       Cij_clamped(i, 3), Cij_clamped(i, 4), Cij_clamped(i, 5));
    }

    occ::log::info("\nRelaxed elastic constants (GPa):");
    for (int i = 0; i < 6; ++i) {
        occ::log::info("  {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.2f}",
                       Cij(i, 0), Cij(i, 1), Cij(i, 2),
                       Cij(i, 3), Cij(i, 4), Cij(i, 5));
    }

    return Cij;
}

} // namespace occ::mults
