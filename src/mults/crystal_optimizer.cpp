#include <occ/mults/crystal_optimizer.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/element.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace occ::mults {

// ============================================================================
// Constructor
// ============================================================================

CrystalOptimizer::CrystalOptimizer(const crystal::Crystal& crystal,
                                   std::vector<MultipoleSource> multipoles,
                                   const CrystalOptimizerSettings& settings)
    : m_settings(settings)
    , m_energy(crystal, std::move(multipoles),
               settings.neighbor_radius,
               settings.force_field,
               settings.use_cartesian_engine) {

    m_num_molecules = m_energy.num_molecules();

    if (m_num_molecules == 0) {
        throw std::invalid_argument("CrystalOptimizer: no molecules in crystal");
    }

    // Calculate number of parameters:
    // - Molecule 0: 3 position (if not fixed) + 3 rotation (if not fixed)
    // - Molecules 1..N-1: 6 each (position + rotation)
    int mol0_dof = 0;
    if (!m_settings.fix_first_translation) mol0_dof += 3;
    if (!m_settings.fix_first_rotation) mol0_dof += 3;

    m_num_parameters = mol0_dof + 6 * (m_num_molecules - 1);

    occ::log::debug("CrystalOptimizer: {} molecules, {} parameters",
                   m_num_molecules, m_num_parameters);

    // Log multipole info
    int lmax = m_energy.max_multipole_rank();
    size_t num_sites = m_energy.num_sites();
    size_t num_pairs = m_energy.num_neighbor_pairs();
    occ::log::info("  Multipoles: {} sites, lmax = {} ({})",
                   num_sites, lmax,
                   lmax == 0 ? "charge" :
                   lmax == 1 ? "dipole" :
                   lmax == 2 ? "quadrupole" :
                   lmax == 3 ? "octopole" :
                   lmax == 4 ? "hexadecapole" : "unknown");
    occ::log::info("  Neighbor pairs: {}", num_pairs);

    // Initialize states from crystal geometry
    m_states = m_energy.initial_states();
    m_initial_states = m_states;
}

// ============================================================================
// Parameter Packing/Unpacking
// ============================================================================

Vec CrystalOptimizer::pack_parameters(const std::vector<MoleculeState>& states) const {
    Vec params(m_num_parameters);
    int offset = 0;

    // Molecule 0: partial DOF based on settings
    if (!m_settings.fix_first_translation) {
        params.segment<3>(offset) = states[0].position;
        offset += 3;
    }
    if (!m_settings.fix_first_rotation) {
        params.segment<3>(offset) = states[0].angle_axis;
        offset += 3;
    }

    // Molecules 1..N-1: full 6 DOF
    for (int i = 1; i < m_num_molecules; ++i) {
        params.segment<3>(offset) = states[i].position;
        params.segment<3>(offset + 3) = states[i].angle_axis;
        offset += 6;
    }

    return params;
}

std::vector<MoleculeState> CrystalOptimizer::unpack_parameters(const Vec& params) const {
    std::vector<MoleculeState> states = m_states;  // Start with current states
    int offset = 0;

    // Molecule 0: partial DOF based on settings
    if (!m_settings.fix_first_translation) {
        states[0].position = params.segment<3>(offset);
        offset += 3;
    }
    if (!m_settings.fix_first_rotation) {
        states[0].angle_axis = params.segment<3>(offset);
        offset += 3;
    }

    // Molecules 1..N-1: full 6 DOF
    for (int i = 1; i < m_num_molecules; ++i) {
        states[i].position = params.segment<3>(offset);
        states[i].angle_axis = params.segment<3>(offset + 3);
        offset += 6;
    }

    return states;
}

Vec CrystalOptimizer::get_parameters() const {
    return pack_parameters(m_states);
}

void CrystalOptimizer::set_parameters(const Vec& params) {
    m_states = unpack_parameters(params);
}

// ============================================================================
// Energy and Gradient
// ============================================================================

CrystalEnergyResult CrystalOptimizer::compute_energy_gradient() {
    return m_energy.compute(m_states);
}

double CrystalOptimizer::objective(const Vec& params, Vec& gradient) {
    auto states = unpack_parameters(params);
    auto result = m_energy.compute(states);

    // Pack gradient with same layout as parameters
    // Note: forces = -∇E, but L-BFGS expects +∇E, so we negate
    gradient.resize(m_num_parameters);
    int offset = 0;

    // Molecule 0: partial DOF based on settings
    if (!m_settings.fix_first_translation) {
        gradient.segment<3>(offset) = -result.forces[0];
        offset += 3;
    }
    if (!m_settings.fix_first_rotation) {
        gradient.segment<3>(offset) = -result.torques[0];
        offset += 3;
    }

    // Molecules 1..N-1: full 6 DOF
    for (int i = 1; i < m_num_molecules; ++i) {
        gradient.segment<3>(offset) = -result.forces[i];
        gradient.segment<3>(offset + 3) = -result.torques[i];
        offset += 6;
    }

    return result.total_energy;
}

// ============================================================================
// Optimization
// ============================================================================

std::pair<double, Vec> CrystalOptimizer::objective_pair(const Vec& params) {
    Vec grad;
    double energy = objective(params, grad);
    return {energy, grad};
}

Mat CrystalOptimizer::compute_hessian(const Vec& params) {
    auto states = unpack_parameters(params);
    auto result = m_energy.compute_with_hessian(states);

    // Match gradient scaling: gradients are unscaled, so Hessian should stay unscaled
    // (If scaling is reintroduced, it must also be applied consistently to Hessian here.)

    // Pack Hessian with proper DOF reduction
    return result.pack_hessian(m_settings.fix_first_translation,
                               m_settings.fix_first_rotation);
}

CrystalOptimizerResult CrystalOptimizer::optimize() {
    return optimize(nullptr);
}

CrystalOptimizerResult CrystalOptimizer::optimize(IterationCallback callback) {
    if (m_settings.method == OptimizationMethod::TrustRegion) {
        return optimize_trust_region(callback);
    } else {
        return optimize_lbfgs(callback);
    }
}

CrystalOptimizerResult CrystalOptimizer::optimize_trust_region(IterationCallback callback) {
    CrystalOptimizerResult result;

    // Handle edge case: no parameters to optimize
    if (m_num_parameters == 0) {
        occ::log::warn("No parameters to optimize (all DOF fixed)");
        auto energy_result = m_energy.compute(m_states);
        result.initial_energy = energy_result.total_energy / m_num_molecules;
        result.final_energy = result.initial_energy;
        result.electrostatic_energy = energy_result.electrostatic_energy / m_num_molecules;
        result.repulsion_dispersion_energy = energy_result.repulsion_dispersion / m_num_molecules;
        result.iterations = 0;
        result.function_evaluations = 1;
        result.converged = true;
        result.termination_reason = "No parameters to optimize";
        result.final_states = m_states;
        result.optimized_crystal = build_optimized_crystal();
        return result;
    }

    // Compute initial energy
    Vec params = get_parameters();
    Vec grad(m_num_parameters);

    occ::timing::StopWatch<> sw;
    sw.start();
    result.initial_energy = objective(params, grad);
    double init_time = sw.stop().count();

    occ::log::info("Initial:    E = {:14.6f} kJ/mol   |g| = {:.3e}  ({:.3f} s)",
                   result.initial_energy / m_num_molecules, grad.norm(), init_time);
    std::fflush(stdout);

    // Open trajectory file if requested (callback will write frames)
    bool write_trajectory = !m_settings.trajectory_file.empty();
    if (write_trajectory) {
        m_trajectory_stream.open(m_settings.trajectory_file);
        if (m_trajectory_stream) {
            occ::log::info("Writing trajectory to {}", m_settings.trajectory_file);
        } else {
            occ::log::warn("Could not open trajectory file: {}", m_settings.trajectory_file);
            write_trajectory = false;
        }
    }

    // Setup Trust Region Newton
    TrustRegionSettings tr_settings;
    tr_settings.initial_radius = m_settings.trust_region_radius;
    // Scale tolerances by total DOF so user-provided values are per-DOF/per-molecule
    double grad_tol = m_settings.gradient_tolerance * std::sqrt(static_cast<double>(m_num_parameters));
    tr_settings.gradient_tol = grad_tol;
    // Ignore energy-based convergence to avoid premature stops with high gradients
    tr_settings.energy_tol = 0.0;
    tr_settings.max_iterations = m_settings.max_iterations;
    tr_settings.hessian_update_interval = m_settings.hessian_update_interval;
    tr_settings.verbose = false;  // We use our own callback for logging

    TrustRegion optimizer(tr_settings);

    int eval_count = 1;  // Already did one evaluation
    int hess_count = 0;

    // Objective function wrapper
    auto tr_objective = [this, &eval_count](const Vec& x) -> std::pair<double, Vec> {
        eval_count++;
        return objective_pair(x);
    };

    // Hessian function wrapper
    auto tr_hessian = [this, &hess_count](const Vec& x) -> Mat {
        hess_count++;
        return compute_hessian(x);
    };

    // Iteration callback for trajectory writing
    occ::timing::StopWatch<> iter_sw;
    iter_sw.start();

    auto tr_callback = [this, &callback, &eval_count, &iter_sw, write_trajectory](
        int iter, const Vec& x, double f, const Vec& g) -> bool {

        double iter_time = iter_sw.stop().count();
        iter_sw.start();

        // Update states for trajectory writing
        const_cast<CrystalOptimizer*>(this)->set_parameters(x);
        if (iter % 5 == 0) {
            m_energy.update_neighbors();
        }

        double gnorm = g.norm();
        double e_per_mol = f / m_num_molecules;
        occ::log::info("Iter {:4d}:  E = {:14.6f} kJ/mol   |g| = {:.3e}  ({:.3f} s, {} evals)",
                       iter, e_per_mol, gnorm, iter_time, eval_count);
        std::fflush(stdout);

        // Write trajectory frame
        if (write_trajectory && m_trajectory_stream) {
            write_trajectory_frame(m_trajectory_stream, iter, f);
        }

        if (callback) {
            return callback(iter, f, gnorm);
        }
        return true;
    };

    // Run optimization
    TrustRegionResult tr_result = optimizer.minimize(tr_objective, tr_hessian, params, tr_callback);

    double opt_time = iter_sw.stop().count();

    // Update states with final parameters
    set_parameters(tr_result.x);

    // Final energy evaluation for components
    auto final_result = m_energy.compute(m_states);

    // Build result
    result.final_energy = final_result.total_energy / m_num_molecules;
    result.electrostatic_energy = final_result.electrostatic_energy / m_num_molecules;
    result.repulsion_dispersion_energy = final_result.repulsion_dispersion / m_num_molecules;
    result.iterations = tr_result.iterations;
    result.function_evaluations = eval_count;
    result.converged = tr_result.converged;
    result.termination_reason = tr_result.termination_reason;
    result.final_states = m_states;
    result.optimized_crystal = build_optimized_crystal();
    result.initial_energy /= m_num_molecules;

    occ::log::info("\nTrust Region: {} iterations, {} function evals, {} Hessian evals ({:.1f} s)",
                   tr_result.iterations, eval_count, hess_count, opt_time);

    // Close trajectory file
    if (m_trajectory_stream.is_open()) {
        m_trajectory_stream.close();
    }

    // Refresh neighbor list if geometry moved significantly during optimization
    // (simple policy: rebuild at the end to keep subsequent calls consistent)
    m_energy.update_neighbors();

    return result;
}

CrystalOptimizerResult CrystalOptimizer::optimize_lbfgs(IterationCallback callback) {
    CrystalOptimizerResult result;

    // Handle edge case: no parameters to optimize
    if (m_num_parameters == 0) {
        occ::log::warn("No parameters to optimize (all DOF fixed)");
        auto energy_result = m_energy.compute(m_states);
        result.initial_energy = energy_result.total_energy / m_num_molecules;
        result.final_energy = result.initial_energy;
        result.electrostatic_energy = energy_result.electrostatic_energy / m_num_molecules;
        result.repulsion_dispersion_energy = energy_result.repulsion_dispersion / m_num_molecules;
        result.iterations = 0;
        result.function_evaluations = 1;
        result.converged = true;
        result.termination_reason = "No parameters to optimize";
        result.final_states = m_states;
        result.optimized_crystal = build_optimized_crystal();
        return result;
    }

    // Compute initial energy
    Vec params = get_parameters();
    Vec grad(m_num_parameters);

    occ::timing::StopWatch<> sw;
    sw.start();
    result.initial_energy = objective(params, grad);
    double init_time = sw.stop().count();

    // Scale gradient tolerance by DOF to interpret user tol as per-DOF RMS
    double grad_tol = m_settings.gradient_tolerance * std::sqrt(static_cast<double>(m_num_parameters));

    occ::log::info("Initial:    E = {:14.6f} kJ/mol   |g| = {:.3e}  ({:.3f} s)",
                   result.initial_energy / m_num_molecules, grad.norm(), init_time);
    std::fflush(stdout);

    // Open trajectory file if requested (callback will write frames)
    bool write_trajectory = !m_settings.trajectory_file.empty();
    if (write_trajectory) {
        m_trajectory_stream.open(m_settings.trajectory_file);
        if (m_trajectory_stream) {
            occ::log::info("Writing trajectory to {}", m_settings.trajectory_file);
        } else {
            occ::log::warn("Could not open trajectory file: {}", m_settings.trajectory_file);
            write_trajectory = false;
        }
    }

    // Setup L-BFGS with simple backtracking line search
    LBFGSSettings lbfgs_settings;
    lbfgs_settings.memory = m_settings.lbfgs_memory;
    lbfgs_settings.gradient_tol = grad_tol;
    lbfgs_settings.energy_tol = m_settings.energy_tolerance * static_cast<double>(m_num_molecules);
    // Use simple backtracking (Armijo only) for fewer evaluations
    lbfgs_settings.backtracking_only = true;
    lbfgs_settings.backtrack_factor = 0.5;
    lbfgs_settings.max_linesearch = 20;
    // Initial step scaled by inverse gradient norm
    lbfgs_settings.initial_step = std::min(1.0, 1.0 / grad.norm());

    LBFGS optimizer(lbfgs_settings);

    // Run optimization
    int eval_count = 1;  // Already did one evaluation
    occ::timing::StopWatch<> iter_sw;
    iter_sw.start();

    auto lbfgs_callback = [this, &callback, &eval_count, &iter_sw, write_trajectory](
        int iter, const Vec& x, double f, const Vec& g) -> bool {

        double iter_time = iter_sw.stop().count();
        iter_sw.start();  // Reset for next iteration

        // Update states for trajectory writing
        const_cast<CrystalOptimizer*>(this)->set_parameters(x);
        if (iter % 5 == 0) {
            m_energy.update_neighbors();
        }

        double gnorm = g.norm();
        double e_per_mol = f / m_num_molecules;
        occ::log::info("Iter {:4d}:  E = {:14.6f} kJ/mol   |g| = {:.3e}  ({:.3f} s, {} evals)",
                       iter, e_per_mol, gnorm, iter_time, eval_count);
        std::fflush(stdout);

        // Write trajectory frame
        if (write_trajectory && m_trajectory_stream) {
            write_trajectory_frame(m_trajectory_stream, iter, f);
        }

        if (callback) {
            return callback(iter, f, gnorm);
        }
        return true;
    };

    auto lbfgs_objective = [this, &eval_count](const Vec& x, Vec& g) -> double {
        eval_count++;
        return objective(x, g);
    };

    LBFGSResult lbfgs_result = optimizer.minimize(
        lbfgs_objective, params, lbfgs_callback, m_settings.max_iterations);

    // Update states with final parameters
    set_parameters(lbfgs_result.x);

    // Final energy evaluation for components
    auto final_result = m_energy.compute(m_states);

    // Build result
    result.final_energy = final_result.total_energy / m_num_molecules;
    result.electrostatic_energy = final_result.electrostatic_energy / m_num_molecules;
    result.repulsion_dispersion_energy = final_result.repulsion_dispersion / m_num_molecules;
    result.iterations = lbfgs_result.iterations;
    result.function_evaluations = eval_count;
    result.converged = lbfgs_result.converged;
    result.termination_reason = lbfgs_result.termination_reason;
    result.final_states = m_states;
    result.optimized_crystal = build_optimized_crystal();
    result.initial_energy /= m_num_molecules;

    // Close trajectory file
    if (m_trajectory_stream.is_open()) {
        m_trajectory_stream.close();
    }

    return result;
}

// ============================================================================
// Crystal Reconstruction
// ============================================================================

crystal::Crystal CrystalOptimizer::build_optimized_crystal() const {
    const auto& orig = m_energy.crystal();
    const auto& unique_mols = orig.symmetry_unique_molecules();

    // Copy the asymmetric unit and modify positions
    crystal::AsymmetricUnit asym = orig.asymmetric_unit();

    // For each unique molecule, compute new atom positions
    for (size_t mol_idx = 0; mol_idx < m_states.size() && mol_idx < unique_mols.size(); ++mol_idx) {
        const auto& mol = unique_mols[mol_idx];
        const auto& state = m_states[mol_idx];
        const auto& asym_indices = mol.asymmetric_unit_idx();

        // Get the transformation from asymmetric unit to this molecule
        auto [asym_rot, asym_trans] = mol.asymmetric_unit_transformation();

        Vec3 original_com = mol.center_of_mass();
        Mat3 R = state.rotation_matrix();

        // Transform each atom
        for (int atom_idx = 0; atom_idx < mol.size(); ++atom_idx) {
            Vec3 original_pos = mol.positions().col(atom_idx);
            Vec3 body_pos = original_pos - original_com;
            Vec3 new_cart_pos = state.position + R * body_pos;

            // Convert new position to fractional
            Vec3 new_frac_pos = orig.to_fractional(new_cart_pos);

            // Get asymmetric unit index for this atom
            int asym_idx = asym_indices(atom_idx);
            if (asym_idx >= 0 && asym_idx < asym.positions.cols()) {
                // Apply inverse of molecule's symop to get asymmetric unit position
                Vec3 asym_frac = asym_rot.transpose() * (new_frac_pos - asym_trans);
                asym.positions.col(asym_idx) = asym_frac;
            }
        }
    }

    // Construct new Crystal with modified asymmetric unit (fresh caches)
    return crystal::Crystal(asym, orig.space_group(), orig.unit_cell());
}

// ============================================================================
// Gradient Check (for debugging)
// ============================================================================

bool CrystalOptimizer::check_gradient(const Vec& params, double tol) {
    if (m_num_parameters == 0) {
        occ::log::info("No parameters to check");
        return true;
    }

    const double h = 1e-5;
    Vec grad(m_num_parameters);
    double f0 = objective(params, grad);

    Vec fd_grad(m_num_parameters);
    Vec params_p = params;
    Vec params_m = params;

    for (int i = 0; i < m_num_parameters; ++i) {
        params_p[i] = params[i] + h;
        params_m[i] = params[i] - h;

        Vec dummy(m_num_parameters);
        double fp = objective(params_p, dummy);
        double fm = objective(params_m, dummy);

        fd_grad[i] = (fp - fm) / (2.0 * h);

        params_p[i] = params[i];
        params_m[i] = params[i];
    }

    double max_error = 0.0;
    for (int i = 0; i < m_num_parameters; ++i) {
        double error = std::abs(grad[i] - fd_grad[i]);
        double denom = std::max(1.0, std::abs(grad[i]));
        double rel_error = error / denom;
        max_error = std::max(max_error, rel_error);

        if (rel_error > tol) {
            occ::log::warn("Gradient check failed at index {}: analytical={:.6e}, fd={:.6e}, error={:.6e}",
                          i, grad[i], fd_grad[i], rel_error);
        }
    }

    occ::log::info("Gradient check: max relative error = {:.6e}", max_error);
    return max_error < tol;
}

// ============================================================================
// Trajectory Output
// ============================================================================

void CrystalOptimizer::write_trajectory_frame(std::ofstream& file, int iter, double energy) const {
    const auto& crystal = m_energy.crystal();
    const auto& uc_mols = crystal.unit_cell_molecules();
    const auto& unique_mols = crystal.symmetry_unique_molecules();
    const auto& unit_cell = crystal.unit_cell();

    // Count total atoms
    int total_atoms = 0;
    for (const auto& mol : uc_mols) {
        total_atoms += mol.size();
    }

    Mat3 lattice = unit_cell.direct();

    // Write extended XYZ header
    file << total_atoms << "\n";
    file << fmt::format(
        "Lattice=\"{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\" "
        "Properties=species:S:1:pos:R:3 "
        "energy={:.10f} "
        "iter={} "
        "pbc=\"T T T\"\n",
        lattice(0, 0), lattice(1, 0), lattice(2, 0),
        lattice(0, 1), lattice(1, 1), lattice(2, 1),
        lattice(0, 2), lattice(1, 2), lattice(2, 2),
        energy / m_num_molecules, iter);

    // For each UC molecule, compute transformed positions
    // UC mol is related to unique mol by symop: uc_frac = symop_rot * unique_frac + symop_trans
    for (const auto& uc_mol : uc_mols) {
        int asym_idx = uc_mol.asymmetric_molecule_idx();

        if (asym_idx < 0 || asym_idx >= static_cast<int>(m_states.size())) {
            // Fallback: write original positions
            for (int atom_idx = 0; atom_idx < uc_mol.size(); ++atom_idx) {
                Vec3 pos = uc_mol.positions().col(atom_idx);
                std::string symbol = occ::core::Element(uc_mol.atomic_numbers()(atom_idx)).symbol();
                file << fmt::format("{:2s} {:16.10f} {:16.10f} {:16.10f}\n",
                                   symbol, pos.x(), pos.y(), pos.z());
            }
            continue;
        }

        const auto& state = m_states[asym_idx];
        const auto& unique_mol = unique_mols[asym_idx];

        // Use the UC molecule's OWN original positions and apply relative transformation
        // This is simpler and avoids symop issues
        Vec3 unique_com_orig = unique_mol.center_of_mass();
        Vec3 uc_com_orig = uc_mol.center_of_mass();
        Mat3 R = state.rotation_matrix();

        // Delta: how much did the unique molecule's COM move?
        Vec3 delta_com = state.position - unique_com_orig;

        // Transform each atom using UC molecule's own body-frame positions
        for (int atom_idx = 0; atom_idx < uc_mol.size(); ++atom_idx) {
            Vec3 uc_orig_pos = uc_mol.positions().col(atom_idx);
            Vec3 uc_body = uc_orig_pos - uc_com_orig;

            // Apply rotation around UC mol's own COM, then translate by delta
            Vec3 new_pos = (uc_com_orig + delta_com) + R * uc_body;

            std::string symbol = occ::core::Element(uc_mol.atomic_numbers()(atom_idx)).symbol();
            file << fmt::format("{:2s} {:16.10f} {:16.10f} {:16.10f}\n",
                               symbol, new_pos.x(), new_pos.y(), new_pos.z());
        }
    }
}

} // namespace occ::mults
