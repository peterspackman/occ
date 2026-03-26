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
    case 0: eps(0, 0) = magnitude; break;
    case 1: eps(1, 1) = magnitude; break;
    case 2: eps(2, 2) = magnitude; break;
    case 3:
        eps(1, 2) = magnitude / 2.0;
        eps(2, 1) = magnitude / 2.0;
        break;
    case 4:
        eps(0, 2) = magnitude / 2.0;
        eps(2, 0) = magnitude / 2.0;
        break;
    case 5:
        eps(0, 1) = magnitude / 2.0;
        eps(1, 0) = magnitude / 2.0;
        break;
    default:
        throw std::runtime_error("Invalid Voigt index: " +
                                 std::to_string(voigt_index));
    }
    return eps;
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Reference data from unstrained crystal: neighbor list + atom-pair masks.
struct ReferenceNeighborData {
    std::vector<NeighborPair> neighbors;
    std::vector<std::vector<bool>> site_masks;
};

/// Build reference neighbor data from a setup.
static ReferenceNeighborData
build_reference_neighbor_data(const CrystalEnergySetup &setup) {
    CrystalEnergy calc(setup);
    auto states = calc.initial_states();
    return {calc.neighbor_list(), calc.compute_buckingham_site_masks(states)};
}

/// Build a CrystalEnergy with a fixed neighbor list and site masks.
static CrystalEnergy
build_calc_with_fixed_neighbors(CrystalEnergySetup setup,
                                const std::vector<NeighborPair> &neighbors,
                                const std::vector<std::vector<bool>> &masks) {
    CrystalEnergy calc(std::move(setup));
    calc.set_neighbor_list(neighbors);
    calc.set_fixed_site_masks(masks);
    return calc;
}

/// Apply strain to a setup, creating a new setup with deformed unit cell
/// and deformed molecule positions. Orientations are kept fixed.
static CrystalEnergySetup
apply_strain(const CrystalEnergySetup &setup, const Mat3 &strain) {
    Mat3 deformation = Mat3::Identity() + strain;
    Mat3 strained_direct = deformation * setup.unit_cell.direct();

    CrystalEnergySetup strained = setup;
    strained.unit_cell = crystal::UnitCell(strained_direct);
    for (auto &mol : strained.molecules) {
        mol.com = deformation * mol.com;
    }
    return strained;
}

/// Evaluate energy at a strained geometry using a reusable CrystalEnergy.
/// Updates the calc in-place with strained lattice and positions.
static CrystalEnergyResult
evaluate_strained(CrystalEnergy &calc,
                  const CrystalEnergySetup &ref_setup,
                  const std::vector<MoleculeState> &ref_states,
                  const Mat3 &strain) {
    Mat3 deformation = Mat3::Identity() + strain;
    Mat3 strained_direct = deformation * ref_setup.unit_cell.direct();

    crystal::UnitCell strained_uc(strained_direct);
    // Build a minimal Crystal for update_lattice (only UnitCell is used)
    crystal::Crystal strained_crystal(crystal::AsymmetricUnit{},
                                      crystal::SpaceGroup("P1"),
                                      strained_uc);

    std::vector<MoleculeState> strained_states = ref_states;
    for (auto &s : strained_states) {
        s.position = deformation * s.position;
    }

    calc.update_lattice(strained_crystal, strained_states);
    return calc.compute(strained_states);
}

/// Pack forces/torques into a 6N gradient vector.
static Vec pack_gradient(const CrystalEnergyResult &result) {
    const int N = static_cast<int>(result.forces.size());
    Vec g(6 * N);
    for (int i = 0; i < N; ++i) {
        g.segment<3>(6 * i) = -result.forces[i];
        g.segment<3>(6 * i + 3) = result.torques[i];
    }
    return g;
}

/// Fix translational singularity in internal Hessian (DMACRYS convention).
static void fix_translation_singularity(Mat &W_ii, int n_mol) {
    if (n_mol <= 0 || W_ii.rows() < 6 * n_mol)
        return;
    double trace = 1.0e-10;
    for (int m = 0; m < n_mol; ++m) {
        trace += W_ii(6 * m + 0, 6 * m + 0);
        trace += W_ii(6 * m + 1, 6 * m + 1);
        trace += W_ii(6 * m + 2, 6 * m + 2);
    }
    for (int i = 0; i < 3; ++i) {
        W_ii(i, i) += trace;
    }
}

static Mat solve_symmetric_dense(const Mat &A, const Mat &B) {
    Eigen::LDLT<Mat> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
        return ldlt.solve(B).eval();
    }
    return A.fullPivLu().solve(B).eval();
}

// ============================================================================
// Public API
// ============================================================================

StrainedResult compute_strained(
    const CrystalEnergySetup &setup, const Mat3 &strain,
    const std::vector<NeighborPair> *fixed_neighbors,
    const std::vector<std::vector<bool>> *fixed_site_masks) {

    CrystalEnergy calc = (fixed_neighbors && fixed_site_masks)
        ? build_calc_with_fixed_neighbors(setup, *fixed_neighbors, *fixed_site_masks)
        : CrystalEnergy(setup);

    auto ref_states = calc.initial_states();
    auto result = evaluate_strained(calc, setup, ref_states, strain);

    StrainedResult out;
    out.energy = result.total_energy;
    out.gradient = pack_gradient(result);
    return out;
}

Vec6 compute_strain_derivatives_fd(const CrystalEnergySetup &setup,
                                   double delta) {
    auto ref = build_reference_neighbor_data(setup);

    Vec6 dU_dE = Vec6::Zero();
    for (int i = 0; i < 6; ++i) {
        Mat3 eps_plus = voigt_strain_tensor(i, +delta);
        Mat3 eps_minus = voigt_strain_tensor(i, -delta);

        double E_plus = compute_strained_energy(setup, eps_plus,
                                                &ref.neighbors, &ref.site_masks);
        double E_minus = compute_strained_energy(setup, eps_minus,
                                                 &ref.neighbors, &ref.site_masks);

        dU_dE(i) = (E_plus - E_minus) / (2.0 * delta) /
                   units::EV_TO_KJ_PER_MOL;
    }
    return dU_dE;
}

Mat6 compute_elastic_constants_fd(const CrystalEnergySetup &setup,
                                  double delta) {
    (void)delta;

    auto ref = build_reference_neighbor_data(setup);
    auto calc = build_calc_with_fixed_neighbors(setup, ref.neighbors, ref.site_masks);

    auto states = calc.initial_states();
    // Build minimal Crystal for update_lattice
    crystal::Crystal ref_crystal(crystal::AsymmetricUnit{},
                                 crystal::SpaceGroup("P1"),
                                 setup.unit_cell);
    calc.update_lattice(ref_crystal, states);
    auto eh = calc.compute_with_hessian(states);

    const double V = setup.unit_cell.volume();
    return (eh.strain_hessian / V) * units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
}

Mat6 compute_relaxed_elastic_constants_fd(const CrystalEnergySetup &setup,
                                          double delta) {
    (void)delta;

    auto ref = build_reference_neighbor_data(setup);
    auto calc = build_calc_with_fixed_neighbors(setup, ref.neighbors, ref.site_masks);

    auto states = calc.initial_states();
    const int N = static_cast<int>(states.size());

    crystal::Crystal ref_crystal(crystal::AsymmetricUnit{},
                                 crystal::SpaceGroup("P1"),
                                 setup.unit_cell);
    calc.update_lattice(ref_crystal, states);

    auto hess_result = calc.compute_with_hessian(states);
    Mat6 W_ee = hess_result.strain_hessian;

    // Internal Hessian with translational singularity fix
    Mat W_ii = hess_result.pack_hessian(false, false);
    fix_translation_singularity(W_ii, N);

    // Strain-internal coupling
    Mat W_ei = hess_result.strain_state_hessian;

    // Use FD W_ei when taper is active (analytic misses taper-strain terms)
    if (!hess_result.exact_for_model) {
        const double h_eps = 2e-5;
        const Mat3 direct0 = setup.unit_cell.direct();

        Mat W_ei_fd = Mat::Zero(6, 6 * N);
        for (int a = 0; a < 6; ++a) {
            const Mat3 eps_p = voigt_strain_tensor(a, +h_eps);
            const Mat3 eps_m = voigt_strain_tensor(a, -h_eps);

            auto eval = [&](const Mat3 &eps) {
                const Mat3 F = Mat3::Identity() + eps;
                crystal::UnitCell strained_uc(F * direct0);
                crystal::Crystal sc(crystal::AsymmetricUnit{},
                                    crystal::SpaceGroup("P1"), strained_uc);
                auto ss = states;
                for (auto &s : ss) s.position = F * s.position;
                calc.update_lattice(sc, ss);
                auto e = calc.compute(ss);
                return pack_gradient(e);
            };

            W_ei_fd.row(a) =
                ((eval(eps_p) - eval(eps_m)) / (2.0 * h_eps)).transpose();
        }
        // Restore reference lattice
        calc.update_lattice(ref_crystal, states);
        W_ei = W_ei_fd;
    }

    // Schur complement: C_relaxed = W_ee - W_ei * W_ii^-1 * W_ie
    Mat X = solve_symmetric_dense(W_ii, W_ei.transpose());
    Mat6 W_relaxed = W_ee - W_ei * X;

    const double V = setup.unit_cell.volume();
    return (W_relaxed / V) * units::KJ_PER_MOL_PER_ANGSTROM3_TO_GPA;
}

} // namespace occ::mults
