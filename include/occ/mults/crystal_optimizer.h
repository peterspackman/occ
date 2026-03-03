#pragma once
#include <occ/mults/crystal_energy.h>
#include <occ/mults/lbfgs.h>
#include <occ/mults/mstmin.h>
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
    MSTMIN,         ///< DMACRYS-style quasi-Newton (full inverse-Hessian updates)
    LBFGS,          ///< L-BFGS quasi-Newton (first-order)
    TrustRegion     ///< Trust Region Newton (second-order with Hessian)
};

/**
 * @brief Maps unit cell molecules to their symmetry-independent representatives.
 *
 * For Z'=1 crystals, there is 1 independent molecule and Z images.
 * For Z'<1 (special positions), the independent molecule has reduced DOF.
 */
struct SymmetryMapping {
    int num_independent = 0;  ///< Z' (number of symmetry-independent molecules)
    int num_uc_molecules = 0; ///< Z (total molecules in unit cell)

    /// Information about how a UC molecule relates to its independent representative.
    struct UCMoleculeInfo {
        int independent_idx = -1;  ///< Which independent molecule this derives from
        Mat3 R_cart = Mat3::Identity();  ///< Cartesian rotation from symop
        Vec3 t_cart = Vec3::Zero();      ///< Cartesian translation from symop
    };
    std::vector<UCMoleculeInfo> uc_molecules;  ///< Size = Z

    /// DOF and projection info for each independent molecule.
    struct IndependentMoleculeInfo {
        std::vector<int> uc_indices;   ///< Which UC molecules derive from this one
        Mat3 trans_projector = Mat3::Identity();  ///< Projects translation gradient
        Mat3 rot_projector = Mat3::Identity();    ///< Projects rotation gradient
        int trans_dof = 3;  ///< Number of free translational DOF (0-3)
        int rot_dof = 3;    ///< Number of free rotational DOF (0-3)
        /// Basis vectors for allowed translation directions (cols = basis)
        Mat trans_basis;  ///< 3 x trans_dof
        /// Basis vectors for allowed rotation directions (cols = basis)
        Mat rot_basis;    ///< 3 x rot_dof
    };
    std::vector<IndependentMoleculeInfo> independent;  ///< Size = Z'

    /// Total optimizable DOF across all independent molecules.
    int total_dof() const;
};

/**
 * @brief Build symmetry mapping from crystal structure.
 *
 * Uses Crystal's symmetry_unique_molecules() and unit_cell_molecules() to
 * determine which UC molecules map to which independent molecules, and
 * detects site symmetry constraints to reduce DOF for Z'<1.
 */
SymmetryMapping build_symmetry_mapping(const crystal::Crystal& crystal);

/**
 * @brief Generate all Z unit cell states from Z' independent states.
 *
 * Applies symmetry operations to produce the full set of UC molecule states.
 */
std::vector<MoleculeState> generate_uc_states(
    const std::vector<MoleculeState>& independent_states,
    const SymmetryMapping& mapping);

/**
 * @brief Accumulate UC-level forces/torques back to independent molecules.
 *
 * For each independent molecule, sums the (back-rotated) forces and torques
 * from all its symmetry images.
 *
 * Forces are projected onto allowed translational subspace. Torques are left
 * unprojected so the SO(3) chain rule can be applied first (dE/dtheta from
 * dE/dpsi), then projected in parameter space.
 */
void accumulate_gradients(
    const std::vector<Vec3>& uc_forces,
    const std::vector<Vec3>& uc_torques,
    const SymmetryMapping& mapping,
    std::vector<Vec3>& indep_forces,
    std::vector<Vec3>& indep_torques);

/**
 * @brief Settings for crystal structure optimization.
 */
struct CrystalOptimizerSettings {
    OptimizationMethod method = OptimizationMethod::MSTMIN; ///< Optimization method
    double gradient_tolerance = 1e-4;  ///< Convergence: ||g|| < tol
    double energy_tolerance = 1e-7;    ///< Convergence: |ΔE| < tol
    int max_iterations = 200;          ///< Maximum iterations
    double neighbor_radius = 20.0;     ///< Cutoff for neighbor interactions (Angstrom)
    ForceFieldType force_field = ForceFieldType::BuckinghamDE;
    bool use_cartesian_engine = true;  ///< Use Cartesian T-tensor (vs S-functions)
    bool fix_first_translation = true; ///< Legacy mode only (use_symmetry=false): fix translation of molecule 0
    bool fix_first_rotation = false;   ///< Legacy mode only (use_symmetry=false): fix rotation of molecule 0
    int lbfgs_memory = 10;             ///< L-BFGS history size
    double max_displacement = 0.05;    ///< MSTMIN max component displacement per cycle
    double mst_step_tolerance = 1e-6;  ///< MSTMIN convergence on max |step component|
    double mst_rotation_scale = 0.2;   ///< Internal MSTMIN scale factor for rotational DOF
    double mst_cell_scale = 0.1;       ///< Internal MSTMIN scale factor for cell strain DOF
    int max_hessian_updates = 1000;    ///< MSTMIN updates before inverse-Hessian reset
    int mst_max_line_search = 40;      ///< MSTMIN max line-search trial steps per cycle
    int mst_max_line_search_restarts = 2; ///< MSTMIN auto-restarts after line-search failure
    int mst_max_function_evaluations = 4000; ///< MSTMIN global objective-evaluation cap
    int mst_line_search_report_interval = 100; ///< Report line-search stalls every N trial evals
    double trust_region_radius = 1.0;  ///< Initial trust region radius
    int hessian_update_interval = 5;   ///< Recompute Hessian every N iterations (1=always, >1 uses SR1)
    std::string trajectory_file;       ///< If non-empty, write XYZ trajectory to this file
    bool require_exact_hessian = true; ///< If true, disable TrustRegion when exact Hessian is unavailable

    /// Use crystallographic symmetry to reduce DOF (Z' molecules instead of Z).
    /// When true, only independent molecule DOF are optimized.
    bool use_symmetry = true;

    /// Optimize the unit cell with 6 Voigt strain DOF (E1..E6).
    bool optimize_cell = false;
    /// Constrain cell strain components to match lattice system (default true).
    /// Example: monoclinic -> only one shear component (single angle) is active.
    bool constrain_cell_strain_by_lattice = true;

    /// Adaptive neighbor-list rebuild controls (for explicit neighbor mode).
    /// Default false for deterministic DMACRYS-style rebuild behavior.
    bool adaptive_neighbor_rebuild = false;
    double neighbor_rebuild_displacement = 0.25;  ///< Angstrom COM threshold
    double neighbor_rebuild_rotation = 0.15;      ///< Radians per-molecule threshold
    double neighbor_rebuild_cell_strain = 0.01;   ///< Max Voigt strain change threshold
    int neighbor_rebuild_interval = 10;           ///< Force rebuild every N objective calls
    /// Keep neighbor list fixed for trial points within line search / trust steps,
    /// and only rebuild after accepted iterations (in optimizer callbacks).
    bool freeze_neighbors_during_linesearch = true;

    // Electrostatics
    int max_interaction_order = 4;     ///< Max multipole interaction order (lA+lB); -1 = no truncation

    // Ewald electrostatics
    bool use_ewald = true;             ///< Enable Ewald-split electrostatics
    double ewald_accuracy = 1e-6;      ///< Target accuracy for eta/kmax auto choice
    double ewald_eta = 0.0;            ///< Override Gaussian split (Ang^-1); 0 => auto
    int ewald_kmax = 0;                ///< Override reciprocal cutoff integer extent; 0 => auto

    // External pressure (PRES) for enthalpy minimization.
    // Value is in GPa, with pV added to the objective.
    double external_pressure_gpa = 0.0;
};

/**
 * @brief Result from crystal structure optimization.
 */
struct CrystalOptimizerResult {
    std::optional<crystal::Crystal> optimized_crystal;  ///< Optimized structure
    double final_energy = 0.0;           ///< Final total energy (kJ/mol per molecule)
    double electrostatic_energy = 0.0;   ///< Electrostatic component
    double repulsion_dispersion_energy = 0.0;  ///< Short-range component
    double pressure_volume_energy = 0.0; ///< pV term in objective (kJ/mol per molecule)
    int iterations = 0;                  ///< Optimizer iterations
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
 * lattice energy. Can operate in two modes:
 *
 * 1. Symmetry-aware (use_symmetry=true, default): Optimizes only Z'
 *    independent molecule DOF and generates the remaining Z molecules
 *    via symmetry operations. For Z'<1, site symmetry constraints
 *    further reduce DOF.
 *
 * 2. All-molecule (use_symmetry=false): Treats all Z molecules as
 *    independent (legacy behavior). Useful for debugging.
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

    /// Get current molecular states (all Z UC molecules).
    const std::vector<MoleculeState>& states() const { return m_states; }

    /// Get initial molecular states (before optimization).
    const std::vector<MoleculeState>& initial_states() const { return m_initial_states; }

    /// Build optimized crystal from current states.
    crystal::Crystal build_optimized_crystal() const;

    /// Compute energy and gradient at current state.
    CrystalEnergyResult compute_energy_gradient();

    /// Re-sync optimizer's states from the energy calculator.
    /// Call this after modifying the energy calculator externally
    /// (e.g., via setup_crystal_energy_from_dmacrys).
    void reinitialize_states();

    /// Access underlying energy calculator.
    CrystalEnergy& energy_calculator() { return m_energy; }
    const CrystalEnergy& energy_calculator() const { return m_energy; }

    /// Access settings.
    const CrystalOptimizerSettings& settings() const { return m_settings; }

    /// Access symmetry mapping (valid when use_symmetry=true).
    const SymmetryMapping& symmetry_mapping() const { return m_symmetry_mapping; }

    /// Number of optimizable parameters.
    int num_parameters() const { return m_num_parameters; }

private:
    CrystalOptimizerSettings m_settings;
    CrystalEnergy m_energy;
    crystal::Crystal m_reference_crystal;      ///< Reference crystal for cell strain DOF
    std::vector<MoleculeState> m_states;        ///< All Z UC molecule states
    std::vector<MoleculeState> m_initial_states;
    std::vector<MoleculeState> m_initial_independent_states;

    int m_num_molecules;       ///< Z (total UC molecules)
    int m_num_free_molecules;  ///< N or N-1 depending on fix_first_molecule (legacy mode)
    int m_num_parameters;      ///< Total optimizable DOF
    int m_num_molecular_parameters = 0; ///< Molecular subset of parameters
    int m_num_cell_parameters = 0;      ///< Active cell-strain DOF subset (0..6)
    Vec6 m_cell_strain = Vec6::Zero();  ///< Current cell variables (Voigt strain parameters)
    Vec6 m_cell_strain_mask = Vec6::Ones(); ///< Active Voigt components (1=active, 0=fixed)
    std::vector<int> m_active_cell_components; ///< Active Voigt indices in parameter order
    int m_objective_eval_count = 0;
    double m_last_eval_max_force = 0.0;   ///< Max |force| across UC molecules (kJ/mol/Ang)
    double m_last_eval_max_torque = 0.0;  ///< Max |torque| across UC molecules (kJ/mol/rad)
    double m_last_eval_max_stress = 0.0;  ///< Max |stress Voigt component| (GPa)
    double m_last_eval_pv_energy = 0.0;   ///< pV term at last evaluation (kJ/mol per unit cell)
    bool m_have_neighbor_reference = false;
    std::vector<MoleculeState> m_neighbor_reference_states;
    Vec6 m_neighbor_reference_cell = Vec6::Zero();

    /// Symmetry mapping (populated when use_symmetry=true).
    SymmetryMapping m_symmetry_mapping;

    /// Independent molecule states (used in symmetry mode).
    std::vector<MoleculeState> m_independent_states;

    /// Pack molecular states into parameter vector.
    Vec pack_parameters(const std::vector<MoleculeState>& states) const;

    /// Unpack parameter vector into molecular states.
    std::vector<MoleculeState> unpack_parameters(const Vec& params) const;

    /// Objective function for L-BFGS (returns energy, fills gradient).
    double objective(const Vec& params, Vec& gradient,
                     bool allow_neighbor_rebuild = true);

    /// Objective function for Trust Region (returns energy and gradient).
    std::pair<double, Vec> objective_pair(const Vec& params,
                                          bool allow_neighbor_rebuild = true);

    /// Hessian function for Trust Region.
    Mat compute_hessian(const Vec& params);

    /// Run L-BFGS optimization.
    CrystalOptimizerResult optimize_lbfgs(IterationCallback callback);

    /// Run DMACRYS-style MSTMIN quasi-Newton optimization.
    CrystalOptimizerResult optimize_mstmin(IterationCallback callback);

    /// Run Trust Region Newton optimization.
    CrystalOptimizerResult optimize_trust_region(IterationCallback callback);

    /// Finite difference gradient check (for debugging).
    bool check_gradient(const Vec& params, double tol = 1e-4);

    /// Write XYZ trajectory frame.
    void write_trajectory_frame(std::ofstream& file, int iter, double energy) const;

    /// Trajectory output stream (opened if trajectory_file is set).
    mutable std::ofstream m_trajectory_stream;

    // --- Symmetry-mode helpers ---

    /// Pack independent molecule states into reduced parameter vector.
    Vec pack_symmetric_parameters(const std::vector<MoleculeState>& indep_states) const;

    /// Unpack reduced parameters into independent states, then generate all Z UC states.
    std::vector<MoleculeState> unpack_symmetric_parameters(const Vec& params) const;

    /// Unpack reduced parameters into independent molecule states (Z' only).
    std::vector<MoleculeState> unpack_symmetric_independent_parameters(
        const Vec& params) const;

    /// Pack gradient from independent molecule forces/torques.
    Vec pack_symmetric_gradient(const std::vector<Vec3>& forces,
                                const std::vector<Vec3>& torques) const;

    /// Build strain tensor from Voigt components [E1..E6].
    static Mat3 voigt_to_strain(const Vec6& voigt);

    /// Map optimizer cell variables to Voigt strain parameters.
    /// In constrained mode, fixed components are projected to zero.
    Vec6 bounded_cell_strain(const Vec6& vars) const;

    /// Expand active cell tail parameters into full 6-component Voigt vector.
    Vec6 unpack_active_cell_parameters(const Vec& params) const;

    /// Pack full 6-component Voigt vector into active cell tail parameters.
    Vec pack_active_cell_parameters(const Vec6& full_cell) const;

    /// Project full 6-component strain gradient into active cell subspace.
    Vec project_active_cell_gradient(const Vec6& full_grad) const;

    /// Project full 6x6 cell Hessian into active cell subspace.
    Mat project_active_cell_hessian(const Mat6& full_hessian) const;

    /// Project full molecular-cell coupling (nm x 6) into active cell subspace.
    Mat project_active_cell_coupling(const Mat& full_coupling) const;

    /// Initialize cell strain component mask from the reference unit cell.
    void initialize_cell_strain_mask();

    /// Apply strain to a crystal and molecular COM positions.
    std::pair<crystal::Crystal, std::vector<MoleculeState>>
    apply_cell_strain(const std::vector<MoleculeState>& base_states,
                      const Vec6& cell_strain) const;

    /// Evaluate energy/gradient at molecular states with optional cell strain.
    CrystalEnergyResult evaluate_model(const std::vector<MoleculeState>& base_states,
                                       const Vec6& cell_strain,
                                       bool rebuild_neighbors = true);

    bool should_rebuild_neighbors(const std::vector<MoleculeState>& strained_states,
                                  const Vec6& cell_strain,
                                  bool force) const;
    void update_neighbor_reference(const std::vector<MoleculeState>& strained_states,
                                   const Vec6& cell_strain);
};

} // namespace occ::mults
