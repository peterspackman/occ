#pragma once
#include <occ/mults/crystal_energy.h>
#include <occ/mults/lbfgs.h>
#include <occ/mults/trust_region.h>
#include <occ/crystal/crystal.h>
#include <functional>
#include <optional>
#include <string>
#include <fstream>

namespace occ::mults {

/**
 * @brief Optimization method selection.
 */
enum class OptimizationMethod {
    LBFGS,          ///< L-BFGS quasi-Newton (first-order)
    TrustRegion     ///< Trust Region Newton (second-order with Hessian)
};

/**
 * @brief Settings for crystal structure optimization.
 */
struct CrystalOptimizerSettings {
    OptimizationMethod method = OptimizationMethod::TrustRegion;  ///< Optimization method
    double gradient_tolerance = 1e-4;  ///< Convergence: ||g|| < tol
    double energy_tolerance = 1e-7;    ///< Convergence: |ΔE| < tol
    int max_iterations = 200;          ///< Maximum iterations
    double neighbor_radius = 20.0;     ///< Cutoff for neighbor interactions (Angstrom)
    ForceFieldType force_field = ForceFieldType::BuckinghamDE;
    bool use_cartesian_engine = true;  ///< Use Cartesian T-tensor (vs S-functions)
    bool fix_first_translation = true; ///< Fix translation of molecule 0 (removes 3 global DOF)
    bool fix_first_rotation = false;   ///< Fix rotation of molecule 0 (only for testing)
    int lbfgs_memory = 10;             ///< L-BFGS history size
    double trust_region_radius = 1.0;  ///< Initial trust region radius
    int hessian_update_interval = 5;   ///< Recompute Hessian every N iterations (1=always, >1 uses SR1)
    std::string trajectory_file;       ///< If non-empty, write XYZ trajectory to this file

    // Electrostatics
    int max_interaction_order = 4;     ///< Max multipole interaction order (lA+lB); -1 = no truncation

    // Ewald electrostatics
    bool use_ewald = true;             ///< Enable Ewald-split electrostatics
    double ewald_accuracy = 1e-6;      ///< Target accuracy for eta/kmax auto choice
    double ewald_eta = 0.0;            ///< Override Gaussian split (Ang^-1); 0 => auto
    int ewald_kmax = 0;                ///< Override reciprocal cutoff integer extent; 0 => auto
};

/**
 * @brief Result from crystal structure optimization.
 */
struct CrystalOptimizerResult {
    std::optional<crystal::Crystal> optimized_crystal;  ///< Optimized structure
    double final_energy = 0.0;           ///< Final total energy (kJ/mol per molecule)
    double electrostatic_energy = 0.0;   ///< Electrostatic component
    double repulsion_dispersion_energy = 0.0;  ///< Short-range component
    int iterations = 0;                  ///< L-BFGS iterations
    int function_evaluations = 0;        ///< Total energy/gradient evaluations
    bool converged = false;              ///< Whether optimization converged
    std::string termination_reason;      ///< Reason for stopping

    std::vector<MoleculeState> final_states;  ///< Final molecular states
    double initial_energy = 0.0;         ///< Starting energy for comparison
};

/**
 * @brief Rigid molecule crystal structure optimizer.
 *
 * Optimizes molecular positions and orientations to minimize the crystal
 * lattice energy. Uses L-BFGS with analytical gradients.
 *
 * Degrees of freedom:
 * - 6 per molecule (3 translation + 3 rotation via angle-axis)
 * - First molecule optionally fixed (removes 6 global DOF)
 *
 * Usage:
 * @code
 * Crystal crystal = io::load_cif("structure.cif");
 * std::vector<MultipoleSource> multipoles = compute_dma(crystal);
 *
 * CrystalOptimizerSettings settings;
 * settings.force_field = ForceFieldType::BuckinghamDE;
 *
 * CrystalOptimizer optimizer(crystal, multipoles, settings);
 * auto result = optimizer.optimize();
 *
 * std::cout << "Final energy: " << result.final_energy << " kJ/mol\n";
 * @endcode
 */
class CrystalOptimizer {
public:
    /// Callback for iteration monitoring: callback(iter, energy, gradient_norm) -> continue?
    using IterationCallback = std::function<bool(int, double, double)>;

    /**
     * @brief Construct optimizer for a crystal structure.
     *
     * @param crystal Input crystal structure
     * @param multipoles Body-frame multipoles for each unique molecule
     * @param settings Optimization settings
     */
    CrystalOptimizer(const crystal::Crystal& crystal,
                     std::vector<MultipoleSource> multipoles,
                     const CrystalOptimizerSettings& settings = {});

    /**
     * @brief Run optimization.
     *
     * @return Optimization result including optimized structure
     */
    CrystalOptimizerResult optimize();

    /**
     * @brief Run optimization with iteration callback.
     *
     * @param callback Called after each iteration
     * @return Optimization result
     */
    CrystalOptimizerResult optimize(IterationCallback callback);

    /// Get current parameters as flat vector.
    Vec get_parameters() const;

    /// Set parameters from flat vector.
    void set_parameters(const Vec& params);

    /// Get current molecular states.
    const std::vector<MoleculeState>& states() const { return m_states; }

    /// Build optimized crystal from current states.
    crystal::Crystal build_optimized_crystal() const;

    /// Compute energy and gradient at current state.
    CrystalEnergyResult compute_energy_gradient();

    /// Access underlying energy calculator.
    CrystalEnergy& energy_calculator() { return m_energy; }
    const CrystalEnergy& energy_calculator() const { return m_energy; }

    /// Access settings.
    const CrystalOptimizerSettings& settings() const { return m_settings; }

private:
    CrystalOptimizerSettings m_settings;
    CrystalEnergy m_energy;
    std::vector<MoleculeState> m_states;
    std::vector<MoleculeState> m_initial_states;

    int m_num_molecules;
    int m_num_free_molecules;  // N or N-1 depending on fix_first_molecule
    int m_num_parameters;      // 6 * num_free_molecules

    /// Pack molecular states into parameter vector.
    Vec pack_parameters(const std::vector<MoleculeState>& states) const;

    /// Unpack parameter vector into molecular states.
    std::vector<MoleculeState> unpack_parameters(const Vec& params) const;

    /// Objective function for L-BFGS (returns energy, fills gradient).
    double objective(const Vec& params, Vec& gradient);

    /// Objective function for Trust Region (returns energy and gradient).
    std::pair<double, Vec> objective_pair(const Vec& params);

    /// Hessian function for Trust Region.
    Mat compute_hessian(const Vec& params);

    /// Run L-BFGS optimization.
    CrystalOptimizerResult optimize_lbfgs(IterationCallback callback);

    /// Run Trust Region Newton optimization.
    CrystalOptimizerResult optimize_trust_region(IterationCallback callback);

    /// Finite difference gradient check (for debugging).
    bool check_gradient(const Vec& params, double tol = 1e-4);

    /// Write XYZ trajectory frame.
    void write_trajectory_frame(std::ofstream& file, int iter, double energy) const;

    /// Trajectory output stream (opened if trajectory_file is set).
    mutable std::ofstream m_trajectory_stream;
};

} // namespace occ::mults
