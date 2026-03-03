#include <occ/mults/crystal_strain.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <array>
#include <cmath>
#include <stdexcept>

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

std::vector<int> reduced_internal_dof_indices(int n_mol) {
    std::vector<int> idx;
    idx.reserve(6 * n_mol - 3);
    // Molecule 0: remove global translations, keep rotations.
    idx.push_back(3);
    idx.push_back(4);
    idx.push_back(5);
    for (int i = 1; i < n_mol; ++i) {
        for (int c = 0; c < 6; ++c) {
            idx.push_back(6 * i + c);
        }
    }
    return idx;
}

Mat3 skew_symmetric(const Vec3& v) {
    Mat3 S;
    S <<      0, -v[2],  v[1],
           v[2],     0, -v[0],
          -v[1],  v[0],     0;
    return S;
}

Mat3 so3_left_jacobian_transpose(const Vec3& theta) {
    const double phi2 = theta.squaredNorm();
    const Mat3 I = Mat3::Identity();

    if (phi2 < 1e-16) {
        const Mat3 K = skew_symmetric(theta);
        return I - 0.5 * K + (1.0 / 6.0) * (K * K);
    }

    const double phi = std::sqrt(phi2);
    const double a = (1.0 - std::cos(phi)) / phi2;
    const double b = (phi - std::sin(phi)) / (phi2 * phi);

    const Mat3 K = skew_symmetric(theta);
    return I - a * K + b * (K * K);
}

std::array<Mat3, 3> so3_left_jacobian_transpose_derivatives(const Vec3& theta) {
    std::array<Mat3, 3> dJ{};
    const Mat3 K = skew_symmetric(theta);
    const Mat3 K2 = K * K;

    std::array<Mat3, 3> E;
    E[0] = skew_symmetric(Vec3(1.0, 0.0, 0.0));
    E[1] = skew_symmetric(Vec3(0.0, 1.0, 0.0));
    E[2] = skew_symmetric(Vec3(0.0, 0.0, 1.0));

    const double phi2 = theta.squaredNorm();
    const double phi = std::sqrt(phi2);

    double a = 0.0;
    double b = 0.0;
    double da_dphi = 0.0;
    double db_dphi = 0.0;

    if (phi < 1e-6) {
        const double phi4 = phi2 * phi2;
        a = 0.5 - phi2 / 24.0 + phi4 / 720.0;
        b = 1.0 / 6.0 - phi2 / 120.0 + phi4 / 5040.0;
        da_dphi = -phi / 12.0 + (phi * phi2) / 180.0;
        db_dphi = -phi / 60.0 + (phi * phi2) / 1260.0;
    } else {
        a = (1.0 - std::cos(phi)) / phi2;
        b = (phi - std::sin(phi)) / (phi2 * phi);
        da_dphi = std::sin(phi) / phi2 -
                  2.0 * (1.0 - std::cos(phi)) / (phi2 * phi);
        db_dphi = (1.0 - std::cos(phi)) / (phi2 * phi) -
                  3.0 * (phi - std::sin(phi)) / (phi2 * phi2);
    }

    for (int l = 0; l < 3; ++l) {
        const double dphi_dtheta = (phi > 1e-14) ? theta[l] / phi : 0.0;
        const double da = da_dphi * dphi_dtheta;
        const double db = db_dphi * dphi_dtheta;
        dJ[l] = -da * K - a * E[l] + db * K2 + b * (E[l] * K + K * E[l]);
    }

    return dJ;
}

Mat solve_symmetric_filtered(const Mat& A, const Mat& B,
                             double rel_tol, double abs_tol,
                             int* dropped_modes = nullptr) {
    Eigen::SelfAdjointEigenSolver<Mat> es(A);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error(
            "Failed eigen-decomposition of internal Hessian in relaxed elastic solve");
    }

    const Vec evals = es.eigenvalues();
    const Mat evecs = es.eigenvectors();
    const double max_abs_eval =
        (evals.size() > 0) ? evals.cwiseAbs().maxCoeff() : 0.0;
    const double thresh = std::max(abs_tol, rel_tol * max_abs_eval);

    Vec inv = Vec::Zero(evals.size());
    int dropped = 0;
    for (int i = 0; i < evals.size(); ++i) {
        if (std::abs(evals(i)) > thresh) {
            inv(i) = 1.0 / evals(i);
        } else {
            dropped++;
        }
    }

    if (dropped_modes) {
        *dropped_modes = dropped;
    }

    return (evecs * inv.asDiagonal() * evecs.transpose() * B).eval();
}

Mat solve_symmetric_dense(const Mat& A, const Mat& B) {
    Eigen::LDLT<Mat> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
        return ldlt.solve(B).eval();
    }
    // Fallback for indefinite/near-singular cases.
    return A.fullPivLu().solve(B).eval();
}

void dmacrys_fix_translation_singularity(Mat& W_ii, int n_mol) {
    if (n_mol <= 0 || W_ii.rows() < 6 * n_mol || W_ii.cols() < 6 * n_mol) {
        return;
    }

    double trace = 1.0e-10;
    // DMACRYS trace sum is over translational d2U/drdr terms for all molecules.
    for (int m = 0; m < n_mol; ++m) {
        trace += W_ii(6 * m + 0, 6 * m + 0);
        trace += W_ii(6 * m + 1, 6 * m + 1);
        trace += W_ii(6 * m + 2, 6 * m + 2);
    }
    for (int i = 0; i < 3; ++i) {
        W_ii(i, i) += trace;
    }
}

static Mat3 stress_tensor_from_strain_gradient(
    const Vec6& dE_dE,
    double volume_ang3) {

    Mat3 sigma = Mat3::Zero();
    if (volume_ang3 <= 0.0) {
        return sigma;
    }

    Vec6 sigma_voigt = Vec6::Zero();
    sigma_voigt[0] = dE_dE[0] / volume_ang3;
    sigma_voigt[1] = dE_dE[1] / volume_ang3;
    sigma_voigt[2] = dE_dE[2] / volume_ang3;
    sigma_voigt[3] = 2.0 * dE_dE[3] / volume_ang3;
    sigma_voigt[4] = 2.0 * dE_dE[4] / volume_ang3;
    sigma_voigt[5] = 2.0 * dE_dE[5] / volume_ang3;
    sigma_voigt *= units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;

    sigma(0, 0) = sigma_voigt[0];
    sigma(1, 1) = sigma_voigt[1];
    sigma(2, 2) = sigma_voigt[2];
    sigma(1, 2) = sigma_voigt[3];
    sigma(2, 1) = sigma_voigt[3];
    sigma(0, 2) = sigma_voigt[4];
    sigma(2, 0) = sigma_voigt[4];
    sigma(0, 1) = sigma_voigt[5];
    sigma(1, 0) = sigma_voigt[5];
    return sigma;
}

static Mat6 finite_stress_correction_voigt_gpa(const Mat3& sigma_gpa) {
    static const std::array<std::pair<int, int>, 6> voigt_pairs{{
        {0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}
    }};
    static const std::array<double, 6> voigt_strain_scale{{1.0, 1.0, 1.0, 2.0, 2.0, 2.0}};

    auto delta = [](int a, int b) { return (a == b) ? 1.0 : 0.0; };

    Mat6 corr = Mat6::Zero();
    for (int a = 0; a < 6; ++a) {
        const auto [i, j] = voigt_pairs[a];
        for (int b = 0; b < 6; ++b) {
            const auto [k, l] = voigt_pairs[b];

            const double term =
                delta(i, k) * sigma_gpa(j, l) +
                delta(j, l) * sigma_gpa(i, k) -
                delta(i, l) * sigma_gpa(j, k) -
                delta(j, k) * sigma_gpa(i, l);

            // E_b uses engineering shear strain for b=4..6.
            corr(a, b) = term / voigt_strain_scale[b];
        }
    }
    return 0.5 * (corr + corr.transpose());
}

/// Evaluate a strained crystal by updating a reusable CrystalEnergy in place.
/// Returns the CrystalEnergyResult directly.
static CrystalEnergyResult evaluate_strained(
    CrystalEnergy& calc,
    const crystal::Crystal& ref_crystal,
    const std::vector<MoleculeState>& reference_states,
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

    // Keep rigid-molecule orientations fixed under affine cell strain.
    // This matches the optimizer/cell-DOF model used for analytical derivatives.
    std::vector<MoleculeState> strained_states = reference_states;
    for (auto& s : strained_states) {
        s.position = deformation * s.position;
    }

    calc.update_lattice(strained_crystal, strained_states);
    return calc.compute(strained_states);
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
        const auto reference_states = calc.initial_states();

        auto result = evaluate_strained(calc, crystal, reference_states, strain);

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

    // Fallback: build CrystalEnergy on reference crystal and apply strain via
    // update_lattice with rigid-body states (same model as optimizer).
    CrystalEnergy calc(crystal, multipoles, cutoff,
                       ForceFieldType::Custom, true, use_ewald,
                       1e-8, alpha, kmax);
    setup_crystal_energy_from_dmacrys(calc, input, crystal, multipoles);

    if (max_interaction_order >= 0) {
        calc.set_max_interaction_order(max_interaction_order);
    }
    for (const auto& [key, p] : buck_params) {
        calc.set_buckingham_params(key.first, key.second, p);
    }

    const auto reference_states = calc.initial_states();
    auto result = evaluate_strained(calc, crystal, reference_states, strain);

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

    Vec6 dU_dE = Vec6::Zero();
    for (int i = 0; i < 6; ++i) {
        Mat3 eps_plus = voigt_strain_tensor(i, +delta);
        Mat3 eps_minus = voigt_strain_tensor(i, -delta);

        double E_plus = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_plus, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);
        double E_minus = compute_strained_energy(
            input, crystal, multipoles, buck_params,
            eps_minus, cutoff, use_ewald, alpha, kmax,
            max_interaction_order, &ref.neighbors, &ref.site_masks);

        dU_dE(i) = (E_plus - E_minus) / (2.0 * delta) /
                   units::EV_TO_KJ_PER_MOL;
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
    (void)delta; // Legacy API parameter retained for compatibility.

    auto ref = build_reference_neighbor_data(
        input, crystal, multipoles, cutoff, use_ewald, alpha, kmax);

    auto calc = build_reusable_calc(
        input, crystal, multipoles, buck_params,
        cutoff, use_ewald, alpha, kmax,
        max_interaction_order, ref.neighbors, ref.site_masks);

    // Use the exact states prepared inside setup_crystal_energy_from_dmacrys().
    // Reconstructing states independently can introduce small O(3)/COM mapping
    // differences that pollute strain-coupling terms in elastic constants.
    auto states = calc.initial_states();
    calc.update_lattice(crystal, states);
    auto eh = calc.compute_with_hessian(states);

    const double V = crystal.unit_cell().volume();
    // DMACRYS STAR/PROP elastic constants are compared against the raw
    // strain-strain Hessian block (no finite-stress correction).
    return (eh.strain_hessian / V) * units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
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
    (void)delta; // Legacy API parameter retained for compatibility.

    // ========================================================================
    // Step 1: Build reference data and reusable CrystalEnergy
    // ========================================================================
    auto ref = build_reference_neighbor_data(
        input, crystal, multipoles, cutoff, use_ewald, alpha, kmax);

    auto calc = build_reusable_calc(
        input, crystal, multipoles, buck_params,
        cutoff, use_ewald, alpha, kmax,
        max_interaction_order, ref.neighbors, ref.site_masks);

    // Keep state/geometry mapping identical to the reusable calculator setup.
    auto states = calc.initial_states();
    const int N = static_cast<int>(states.size());
    calc.update_lattice(crystal, states);

    auto hess_result = calc.compute_with_hessian(states);
    Mat6 W_ee = hess_result.strain_hessian;

    occ::log::info("W_ee (strain-strain, kJ/mol):");
    for (int i = 0; i < 6; ++i) {
        occ::log::info("  {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}",
                       W_ee(i, 0), W_ee(i, 1), W_ee(i, 2),
                       W_ee(i, 3), W_ee(i, 4), W_ee(i, 5));
    }

    // DMACRYS LTPROP path:
    // 1) keep full internal block
    // 2) lift translational singularity by pinning molecule-0 translations
    // 3) direct dense solve for Schur complement
    Mat W_ii = hess_result.pack_hessian(false, false);
    dmacrys_fix_translation_singularity(W_ii, N);
    int ndof_reduced = W_ii.rows();

    occ::log::info("W_ii: {}x{} (full internal block, DMACRYS-style gauge fix)",
                   ndof_reduced, ndof_reduced);

    // Print W_ii eigenvalues
    {
        Eigen::SelfAdjointEigenSolver<Mat> es(W_ii);
        const Vec evals = es.eigenvalues();
        occ::log::info("W_ii eigenvalues (kJ/mol):");
        for (int i = 0; i < evals.size(); ++i) {
            occ::log::info("  [{:2d}] {:.6f}", i, evals[i]);
        }
    }

    // Check reference gradient magnitude
    {
        auto ref_grad = hess_result.pack_gradient();
        occ::log::info("Reference gradient norm: {:.6f} (should be ~0 at minimum)",
                       ref_grad.norm());
        for (int m = 0; m < N; ++m) {
            occ::log::info("  mol {} force=[{:.6f},{:.6f},{:.6f}] torque=[{:.6f},{:.6f},{:.6f}]",
                           m,
                           ref_grad[6*m], ref_grad[6*m+1], ref_grad[6*m+2],
                           ref_grad[6*m+3], ref_grad[6*m+4], ref_grad[6*m+5]);
        }
    }

    // ========================================================================
    // Step 3: Compute W_ei (6 x ndof_reduced) -- strain-internal coupling
    // ========================================================================
    Mat W_ei_analytic = hess_result.strain_state_hessian;
    Mat W_ei = W_ei_analytic;

    // Always compute FD W_ei for comparison / use when taper is active
    {
        occ::log::info("Computing W_ei via finite differences");
        const double h_eps = 2e-5;
        const Mat3 direct0 = crystal.unit_cell().direct();

        auto pack_grad_at_strain = [&](const Mat3& eps) {
            const Mat3 F = Mat3::Identity() + eps;
            const Mat3 strained_direct = F * direct0;
            crystal::UnitCell strained_uc(strained_direct);
            crystal::Crystal strained_crystal(
                crystal.asymmetric_unit(), crystal.space_group(), strained_uc);

            auto strained_states = states;
            for (auto& s : strained_states) {
                s.position = F * s.position;
            }
            calc.update_lattice(strained_crystal, strained_states);
            auto e = calc.compute(strained_states);

            const int n = static_cast<int>(e.forces.size());
            Vec g(6 * n);
            for (int i = 0; i < n; ++i) {
                g.segment<3>(6 * i) = -e.forces[i];     // gradient = -force
                g.segment<3>(6 * i + 3) = e.torques[i];  // gradient = torque
            }
            return g;
        };

        Mat W_ei_fd = Mat::Zero(6, 6 * N);
        for (int a = 0; a < 6; ++a) {
            const Mat3 eps_p = voigt_strain_tensor(a, +h_eps);
            const Mat3 eps_m = voigt_strain_tensor(a, -h_eps);
            const Vec gp = pack_grad_at_strain(eps_p);
            const Vec gm = pack_grad_at_strain(eps_m);
            W_ei_fd.row(a) = ((gp - gm) / (2.0 * h_eps)).transpose();
        }

        // Restore reference lattice
        calc.update_lattice(crystal, states);

        occ::log::info("W_ei analytic norm: {:.6f}", W_ei_analytic.norm());
        occ::log::info("W_ei FD norm: {:.6f}", W_ei_fd.norm());
        occ::log::info("W_ei diff norm: {:.6f}", (W_ei_analytic - W_ei_fd).norm());

        // Use FD if taper is active (analytic W_ei misses taper-strain terms)
        if (!hess_result.exact_for_model) {
            occ::log::info("Using FD W_ei (taper active)");
            W_ei = W_ei_fd;
        }
    }

    occ::log::info("W_ei norm: {:.6f}", W_ei.norm());

    // ========================================================================
    // Step 4: Schur complement
    // ========================================================================
    Mat X = solve_symmetric_dense(W_ii, W_ei.transpose());
    Mat6 correction = W_ei * X;

    Mat6 W_relaxed = W_ee - correction;

    occ::log::info("Correction (W_ei * W_ii^-1 * W_ie) diagonal:");
    occ::log::info("  {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                   correction(0, 0), correction(1, 1), correction(2, 2),
                   correction(3, 3), correction(4, 4), correction(5, 5));

    // ========================================================================
    // Step 5: Convert to GPa (raw, DMACRYS-comparable)
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

    // DMACRYS STAR/PROP compares against raw relaxed elastic constants.
    return Cij;
}

} // namespace occ::mults
