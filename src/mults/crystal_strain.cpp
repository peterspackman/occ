#include <occ/mults/crystal_strain.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <Eigen/Eigenvalues>

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

double compute_strained_energy(
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

    const auto& uc = crystal.unit_cell();
    Mat3 direct = uc.direct();

    // Deform: direct' = (I + eps) * direct
    Mat3 deformation = Mat3::Identity() + strain;
    Mat3 strained_direct = deformation * direct;

    // Build new UnitCell from deformed lattice vectors
    crystal::UnitCell strained_uc(strained_direct);

    // Build new Crystal with same asymmetric unit and space group
    crystal::Crystal strained_crystal(
        crystal.asymmetric_unit(),
        crystal.space_group(),
        strained_uc);

    // Use ORIGINAL multipoles (unstrained rotations) but pass the
    // STRAINED crystal for COM position computation.
    //
    // Under strain, molecules translate affinely with the lattice
    // (COMs move) but their orientations are FIXED (rigid molecules).
    // DMACRYS holds orientations fixed when computing dU/dE.
    // setup_crystal_energy_from_dmacrys() gets:
    //   - COM positions from the crystal argument (strained -> correct)
    //   - Rotations from the multipoles argument (original -> correct)
    CrystalEnergy calc(strained_crystal, multipoles, cutoff,
                       ForceFieldType::Custom, true, use_ewald,
                       1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(calc, input, strained_crystal,
                                      multipoles);

    // Use fixed neighbor list if provided (avoids hard-cutoff discontinuities
    // from molecular pairs entering/exiting the cutoff under strain).
    if (fixed_neighbors) {
        calc.set_neighbor_list(*fixed_neighbors);
    }

    // Use frozen atom-pair masks for Buckingham if provided.
    // This ensures exactly the same set of atom-atom interactions are
    // computed at every strain step, giving a smooth energy surface for FD.
    if (fixed_site_masks) {
        calc.set_fixed_site_masks(*fixed_site_masks);
    }

    if (max_interaction_order >= 0) {
        calc.set_max_interaction_order(max_interaction_order);
    }

    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }

    auto states = calc.initial_states();
    return calc.compute_energy(states);
}

Vec compute_strained_gradient(
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

    const auto& uc = crystal.unit_cell();
    Mat3 direct = uc.direct();

    // Deform: direct' = (I + eps) * direct
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
    setup_crystal_energy_from_dmacrys(calc, input, strained_crystal,
                                      multipoles);

    if (fixed_neighbors) {
        calc.set_neighbor_list(*fixed_neighbors);
    }
    if (fixed_site_masks) {
        calc.set_fixed_site_masks(*fixed_site_masks);
    }
    if (max_interaction_order >= 0) {
        calc.set_max_interaction_order(max_interaction_order);
    }

    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }

    auto states = calc.initial_states();
    auto result = calc.compute(states);

    // Pack as energy gradient: [-force, +torque]
    const int N = static_cast<int>(result.forces.size());
    Vec grad(6 * N);
    for (int i = 0; i < N; ++i) {
        grad.segment<3>(6 * i) = -result.forces[i];
        grad.segment<3>(6 * i + 3) = result.torques[i];
    }
    return grad;
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

    // Build neighbor list + atom-pair masks from the UNSTRAINED crystal.
    // Both are frozen to avoid discontinuities from molecule pairs or
    // atom pairs entering/exiting the cutoff under tiny strain perturbations.
    auto ref = build_reference_neighbor_data(
        input, crystal, multipoles, cutoff, use_ewald, alpha, kmax);

    Vec6 dU_dE;

    for (int i = 0; i < 6; ++i) {
        Mat3 eps_plus = voigt_strain_tensor(i, +delta);
        double E_plus = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_plus, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);

        Mat3 eps_minus = voigt_strain_tensor(i, -delta);
        double E_minus = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_minus, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);

        // Central difference: dU/dE_i in kJ/mol per cell
        double dU_kJ = (E_plus - E_minus) / (2.0 * delta);

        // Convert to eV per unit cell
        dU_dE(i) = dU_kJ / units::EV_TO_KJ_PER_MOL;

        occ::log::info("Strain derivative E_{}: E(+d)={:.10f} E(-d)={:.10f} "
                        "dU/dE={:.6f} eV",
                        i + 1, E_plus, E_minus, dU_dE(i));
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

    // Build neighbor list + atom-pair masks from the UNSTRAINED crystal.
    auto ref = build_reference_neighbor_data(
        input, crystal, multipoles, cutoff, use_ewald, alpha, kmax);

    // Energy at zero strain
    Mat3 zero_strain = Mat3::Zero();
    double E0 = compute_strained_energy(
        input, crystal, multipoles, buck_params,
        zero_strain, cutoff, use_ewald, alpha, kmax,
        max_interaction_order, &ref.neighbors, &ref.site_masks);

    double V = crystal.unit_cell().volume(); // Angstrom^3

    Mat6 Cij = Mat6::Zero();

    // Cache single-strain energies for diagonal computation
    std::array<double, 6> E_plus{}, E_minus{};

    for (int i = 0; i < 6; ++i) {
        Mat3 eps_p = voigt_strain_tensor(i, +delta);
        Mat3 eps_m = voigt_strain_tensor(i, -delta);

        E_plus[i] = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_p, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);
        E_minus[i] = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_m, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);

        // Diagonal: C_ii = (1/V) * [U(+d) - 2*U(0) + U(-d)] / d^2
        double d2U = (E_plus[i] - 2.0 * E0 + E_minus[i]) / (delta * delta);
        Cij(i, i) = (d2U / V) * units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

        occ::log::debug("C_{}{}: E(+d)={:.8f} E0={:.8f} E(-d)={:.8f} "
                         "d2U={:.6f} C={:.2f} GPa",
                         i + 1, i + 1,
                         E_plus[i], E0, E_minus[i], d2U, Cij(i, i));
    }

    // Off-diagonal: C_ij = (1/V) * [U(++)-U(+-)-U(-+)+U(--)] / (4*d^2)
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

            double E_pp = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_pp, cutoff, use_ewald, alpha, kmax,
                max_interaction_order, &ref.neighbors, &ref.site_masks);
            double E_pm = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_pm, cutoff, use_ewald, alpha, kmax,
                max_interaction_order, &ref.neighbors, &ref.site_masks);
            double E_mp = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_mp, cutoff, use_ewald, alpha, kmax,
                max_interaction_order, &ref.neighbors, &ref.site_masks);
            double E_mm = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_mm, cutoff, use_ewald, alpha, kmax,
                max_interaction_order, &ref.neighbors, &ref.site_masks);

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
    // Step 1: Build reference data
    // ========================================================================
    auto ref = build_reference_neighbor_data(
        input, crystal, multipoles, cutoff, use_ewald, alpha, kmax);

    // Build reference CrystalEnergy for Hessian computation
    CrystalEnergy ref_calc(crystal, multipoles, cutoff,
                           ForceFieldType::Custom, true, use_ewald,
                           1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(ref_calc, input, crystal, multipoles);
    ref_calc.set_neighbor_list(ref.neighbors);
    ref_calc.set_fixed_site_masks(ref.site_masks);
    if (max_interaction_order >= 0) {
        ref_calc.set_max_interaction_order(max_interaction_order);
    }
    for (const auto& [key, p] : buck_params) {
        ref_calc.set_buckingham_params(key.first, key.second, p);
    }

    auto states = ref_calc.initial_states();
    const int N = static_cast<int>(states.size());
    const int ndof_full = 6 * N;

    // ========================================================================
    // Step 2: Compute W_εε (6x6) — strain-strain second derivatives in kJ/mol
    // ========================================================================
    Mat3 zero_strain = Mat3::Zero();
    double E0 = compute_strained_energy(
        input, crystal, multipoles, buck_params,
        zero_strain, cutoff, use_ewald, alpha, kmax,
        max_interaction_order, &ref.neighbors, &ref.site_masks);

    Mat6 W_ee = Mat6::Zero();

    std::array<double, 6> E_plus{}, E_minus{};
    for (int i = 0; i < 6; ++i) {
        Mat3 eps_p = voigt_strain_tensor(i, +delta);
        Mat3 eps_m = voigt_strain_tensor(i, -delta);

        E_plus[i] = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_p, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);
        E_minus[i] = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_m, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);

        // Diagonal: d²U/dE_i² = [U(+d) - 2*U(0) + U(-d)] / d²
        W_ee(i, i) = (E_plus[i] - 2.0 * E0 + E_minus[i]) / (delta * delta);
    }

    // Off-diagonal
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

            double E_pp = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_pp, cutoff, use_ewald, alpha, kmax,
                max_interaction_order, &ref.neighbors, &ref.site_masks);
            double E_pm = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_pm, cutoff, use_ewald, alpha, kmax,
                max_interaction_order, &ref.neighbors, &ref.site_masks);
            double E_mp = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_mp, cutoff, use_ewald, alpha, kmax,
                max_interaction_order, &ref.neighbors, &ref.site_masks);
            double E_mm = compute_strained_energy(
                input, crystal, multipoles, buck_params,
                eps_mm, cutoff, use_ewald, alpha, kmax,
                max_interaction_order, &ref.neighbors, &ref.site_masks);

            double d2U = (E_pp - E_pm - E_mp + E_mm) / (4.0 * delta * delta);
            W_ee(i, j) = d2U;
            W_ee(j, i) = d2U;
        }
    }

    occ::log::info("W_εε (strain-strain, kJ/mol):");
    for (int i = 0; i < 6; ++i) {
        occ::log::info("  {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}",
                       W_ee(i, 0), W_ee(i, 1), W_ee(i, 2),
                       W_ee(i, 3), W_ee(i, 4), W_ee(i, 5));
    }

    // ========================================================================
    // Step 3: Compute W_ii — internal DOF Hessian
    // ========================================================================
    auto hess_result = ref_calc.compute_with_hessian(states);

    // Remove 3 translational DOF of molecule 0 (zero modes)
    Mat W_ii = hess_result.pack_hessian(true, false);
    int ndof_reduced = W_ii.rows();

    occ::log::info("W_ii: {}x{} (removed 3 translational DOF of mol 0)",
                   ndof_reduced, ndof_reduced);

    // Eigenvalue diagnostics for W_ii
    {
        Eigen::SelfAdjointEigenSolver<Mat> eigsolver(W_ii);
        auto eigenvalues = eigsolver.eigenvalues();
        occ::log::info("W_ii eigenvalue range: min={:.6f} max={:.6f}",
                       eigenvalues.minCoeff(), eigenvalues.maxCoeff());
        int n_negative = (eigenvalues.array() < 0).count();
        int n_small = (eigenvalues.array().abs() < 1.0).count();
        occ::log::info("W_ii: {} negative, {} near-zero (<1 kJ/mol) eigenvalues",
                       n_negative, n_small);
        // Print first few eigenvalues
        int n_print = std::min(10, static_cast<int>(eigenvalues.size()));
        for (int i = 0; i < n_print; ++i) {
            occ::log::info("  eigenvalue[{}] = {:.6f}", i, eigenvalues(i));
        }
    }

    // Check reference gradient magnitude
    {
        auto ref_grad = hess_result.pack_gradient();
        occ::log::info("Reference gradient norm: {:.6f} (should be ~0 at minimum)",
                       ref_grad.norm());
    }

    // ========================================================================
    // Step 4: Compute W_εi (6 × ndof_reduced) — strain-internal coupling
    // ========================================================================

    // Build the same DOF index mapping as pack_hessian(true, false)
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

        Vec g_plus = compute_strained_gradient(
            input, crystal, multipoles, buck_params,
            eps_plus, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);

        Vec g_minus = compute_strained_gradient(
            input, crystal, multipoles, buck_params,
            eps_minus, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);

        // Full gradient difference (6N)
        Vec dg = (g_plus - g_minus) / (2.0 * delta);

        // Extract reduced DOF
        for (int k = 0; k < ndof_reduced; ++k) {
            W_ei(i, k) = dg(reduced_to_full[k]);
        }
    }

    occ::log::info("W_εi norm: {:.6f}", W_ei.norm());

    // ========================================================================
    // Step 5: Schur complement
    // ========================================================================
    // correction = W_εi * W_ii^{-1} * W_εi^T
    // Solve W_ii * X = W_εi^T  =>  X = W_ii^{-1} * W_εi^T  (ndof_reduced × 6)
    Mat X = W_ii.ldlt().solve(W_ei.transpose());
    Mat6 correction = W_ei * X;  // 6×6

    Mat6 W_relaxed = W_ee - correction;

    occ::log::info("Correction (W_εi * W_ii^-1 * W_iε) diagonal:");
    occ::log::info("  {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                   correction(0, 0), correction(1, 1), correction(2, 2),
                   correction(3, 3), correction(4, 4), correction(5, 5));

    // ========================================================================
    // Step 6: Convert to GPa
    // ========================================================================
    double V = crystal.unit_cell().volume();
    Mat6 Cij = (W_relaxed / V) * units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

    // Also log clamped for comparison
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
